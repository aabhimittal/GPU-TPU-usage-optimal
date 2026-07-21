"""Pluggable accelerator back-ends for TPU and NPU sampling.

Neither TPUs nor NPUs expose a single universal Python metrics API the way
NVML does for NVIDIA GPUs, so each back-end tries a couple of best-effort
sources and degrades to *unavailable* (returning ``None``) when none are
present. Back-ends are injectable, which keeps the collector testable on
machines without the hardware.

A back-end implements :class:`AcceleratorBackend`: an ``available`` flag and a
``read(index)`` returning an :class:`AcceleratorReading` (or ``None``).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class AcceleratorReading:
    """One accelerator's memory/utilisation snapshot (MB, percentages)."""

    used_mb: Optional[float] = None
    total_mb: Optional[float] = None
    util: Optional[float] = None  # compute utilisation 0-100

    @property
    def percent(self) -> Optional[float]:
        if self.used_mb is None or not self.total_mb:
            return None
        return 100.0 * self.used_mb / self.total_mb


@runtime_checkable
class AcceleratorBackend(Protocol):
    """Interface every accelerator back-end satisfies."""

    name: str

    @property
    def available(self) -> bool:  # pragma: no cover - trivial
        ...

    def read(self, index: int = 0) -> Optional[AcceleratorReading]:
        ...


# ---------------------------------------------------------------------------
# TPU
# ---------------------------------------------------------------------------
class TpuBackend:
    """Google Cloud TPU metrics.

    Sources tried, in order:

    1. ``tpu_info`` (the ``tpu-info`` pip package), which reads HBM usage and
       duty-cycle from the libtpu runtime metrics endpoint.
    2. JAX device ``memory_stats()`` when running inside a JAX/TPU program.

    Falls back to *unavailable* if neither import succeeds.
    """

    name = "tpu"

    def __init__(self) -> None:
        self._mode: Optional[str] = None
        self._probe()

    def _probe(self) -> None:
        try:  # preferred: dedicated metrics package
            import tpu_info  # type: ignore  # noqa: F401

            self._mode = "tpu_info"
            return
        except Exception:
            pass
        try:  # fallback: JAX runtime
            import jax  # type: ignore

            if jax.local_devices() and jax.local_devices()[0].platform == "tpu":
                self._mode = "jax"
        except Exception:
            self._mode = None

    @property
    def available(self) -> bool:
        return self._mode is not None

    def read(self, index: int = 0) -> Optional[AcceleratorReading]:
        if self._mode == "tpu_info":
            return self._read_tpu_info(index)
        if self._mode == "jax":
            return self._read_jax(index)
        return None

    def _read_tpu_info(self, index: int) -> Optional[AcceleratorReading]:
        try:  # pragma: no cover - requires TPU hardware
            from tpu_info import device, metrics  # type: ignore

            chip_type, count = device.get_local_chips()
            usages = metrics.get_chip_usage(chip_type)
            if index >= len(usages):
                return None
            u = usages[index]
            return AcceleratorReading(
                used_mb=u.memory_usage / 1024 / 1024,
                total_mb=u.total_memory / 1024 / 1024,
                util=float(getattr(u, "duty_cycle_pct", None) or 0.0),
            )
        except Exception:
            return None

    def _read_jax(self, index: int) -> Optional[AcceleratorReading]:
        try:  # pragma: no cover - requires TPU hardware
            import jax  # type: ignore

            devs = jax.local_devices()
            if index >= len(devs):
                return None
            stats = devs[index].memory_stats() or {}
            used = stats.get("bytes_in_use")
            total = stats.get("bytes_limit")
            return AcceleratorReading(
                used_mb=used / 1024 / 1024 if used is not None else None,
                total_mb=total / 1024 / 1024 if total is not None else None,
                util=None,
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# NPU
# ---------------------------------------------------------------------------
class NpuBackend:
    """Neural-processing-unit metrics.

    Sources tried, in order:

    1. Huawei Ascend ``npu-smi info`` CLI, parsed for HBM/DDR usage and AI-Core
       utilisation (the most common NPU tooling on Linux).
    2. Intel NPU exposed via ``openvino``: total (and, when the plugin reports
       it, allocated) device memory read through OpenVINO device properties.

    Falls back to *unavailable* if neither is found.
    """

    name = "npu"
    _OV_DEVICE = "NPU"
    # OpenVINO property keys carrying device memory in bytes. Names have varied
    # across releases, so several candidates are tried in order.
    _OV_TOTAL_KEYS = ("NPU_DEVICE_TOTAL_MEM_SIZE", "DEVICE_TOTAL_MEM_SIZE",
                      "GPU_DEVICE_TOTAL_MEM_SIZE")
    _OV_USED_KEYS = ("NPU_DEVICE_ALLOC_MEM_SIZE", "DEVICE_ALLOCATED_MEM_SIZE",
                     "DEVICE_USED_MEM_SIZE")

    def __init__(self) -> None:
        self._mode: Optional[str] = None
        self._ov_core: Optional[Any] = None
        self._probe()

    def _probe(self) -> None:
        if shutil.which("npu-smi"):
            self._mode = "ascend"
            return
        core = self._make_openvino_core()
        if core is not None and self._OV_DEVICE in getattr(
            core, "available_devices", []
        ):
            self._ov_core = core
            self._mode = "openvino"

    @staticmethod
    def _make_openvino_core() -> Optional[Any]:
        try:  # pragma: no cover - requires openvino installed
            import openvino as ov  # type: ignore

            return ov.Core()
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self._mode is not None

    def read(self, index: int = 0) -> Optional[AcceleratorReading]:
        if self._mode == "ascend":
            return self._read_ascend(index)
        if self._mode == "openvino":
            return self._read_openvino(index)
        return None

    def _read_ascend(self, index: int) -> Optional[AcceleratorReading]:
        try:  # pragma: no cover - requires NPU hardware
            out = subprocess.run(
                ["npu-smi", "info", "-i", str(index)],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            return self._parse_ascend(out)
        except Exception:
            return None

    @staticmethod
    def _parse_ascend(text: str) -> Optional[AcceleratorReading]:
        """Parse ``npu-smi info`` output.

        Recognises ``HBM-Usage(MB) : 1234 / 32768`` (or ``Memory-Usage``) and an
        ``AICore(%)`` / ``AI Core`` utilisation figure. Split out for testing.
        """
        used = total = util = None
        mem = re.search(
            r"(?:HBM|Memory|DDR)[- ]?Usage\(MB\)\s*[:|]\s*([\d.]+)\s*/\s*([\d.]+)",
            text,
            re.IGNORECASE,
        )
        if mem:
            used = float(mem.group(1))
            total = float(mem.group(2))
        u = re.search(r"AI[ -]?Core(?:\(%\))?\s*[:|]\s*([\d.]+)", text, re.IGNORECASE)
        if u:
            util = float(u.group(1))
        if used is None and util is None:
            return None
        return AcceleratorReading(used_mb=used, total_mb=total, util=util)

    def _read_openvino(self, index: int) -> Optional[AcceleratorReading]:
        if self._ov_core is None:
            return None
        return self._query_openvino(self._ov_core, self._OV_DEVICE)

    @classmethod
    def _query_openvino(cls, core: Any, device: str = "NPU"
                        ) -> Optional[AcceleratorReading]:
        """Build a reading from an OpenVINO ``Core``'s device properties.

        ``core`` only needs a ``get_property(device, key)`` method, which keeps
        this unit-testable without OpenVINO or NPU hardware. Total memory is
        always attempted; ``used`` is filled only when the plugin exposes an
        allocated-memory property (many builds do not), leaving ``percent`` at
        ``None`` in that case.
        """
        total = cls._first_prop_bytes(core, device, cls._OV_TOTAL_KEYS)
        used = cls._first_prop_bytes(core, device, cls._OV_USED_KEYS)
        if total is None and used is None:
            return None
        to_mb = lambda b: b / 1024 / 1024 if b is not None else None
        return AcceleratorReading(used_mb=to_mb(used), total_mb=to_mb(total),
                                  util=None)

    @staticmethod
    def _first_prop_bytes(core: Any, device: str, keys) -> Optional[float]:
        """Return the first property in ``keys`` that yields a positive number."""
        for key in keys:
            try:
                val = core.get_property(device, key)
            except Exception:
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if num > 0:
                return num
        return None
