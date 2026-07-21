"""Sampling of GPU and system-memory usage.

GPU metrics are read through NVML (``pynvml``) when available. Both GPU and
CPU/RAM back-ends degrade gracefully: if a dependency or device is missing the
collector still returns a :class:`Sample` with ``None`` for the unavailable
fields, so callers never have to guard against import errors.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

from .backends import AcceleratorBackend, NpuBackend, TpuBackend

# ---- optional back-ends -------------------------------------------------
try:  # GPU metrics
    import pynvml  # type: ignore

    _NVML_OK = True
except Exception:  # pragma: no cover - environment dependent
    pynvml = None  # type: ignore
    _NVML_OK = False

try:  # system RAM / CPU metrics
    import psutil  # type: ignore

    _PSUTIL_OK = True
except Exception:  # pragma: no cover - environment dependent
    psutil = None  # type: ignore
    _PSUTIL_OK = False


@dataclass
class Sample:
    """A single point-in-time snapshot of resource usage.

    Memory values are megabytes; utilisation values are percentages (0-100).
    Fields are ``None`` when the corresponding back-end is unavailable.
    """

    ts: float
    ram_used_mb: Optional[float] = None
    ram_total_mb: Optional[float] = None
    ram_percent: Optional[float] = None
    cpu_percent: Optional[float] = None
    gpu_used_mb: Optional[float] = None
    gpu_total_mb: Optional[float] = None
    gpu_percent: Optional[float] = None
    gpu_util: Optional[float] = None
    tpu_used_mb: Optional[float] = None
    tpu_total_mb: Optional[float] = None
    tpu_percent: Optional[float] = None
    tpu_util: Optional[float] = None
    npu_used_mb: Optional[float] = None
    npu_total_mb: Optional[float] = None
    npu_percent: Optional[float] = None
    npu_util: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceCollector:
    """Collects :class:`Sample` snapshots from the local machine.

    Parameters
    ----------
    gpu_index, tpu_index, npu_index:
        Device indices to monitor when several of a kind are present.
    tpu_backend, npu_backend:
        Optional back-end overrides (mainly for testing / custom hardware).
        When omitted the default best-effort back-ends are probed.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        tpu_index: int = 0,
        npu_index: int = 0,
        tpu_backend: Optional[AcceleratorBackend] = None,
        npu_backend: Optional[AcceleratorBackend] = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.tpu_index = tpu_index
        self.npu_index = npu_index
        self._nvml_ready = False
        self._handle = None
        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self._nvml_ready = True
            except Exception:  # pragma: no cover - no GPU present
                self._nvml_ready = False
        self.tpu = tpu_backend if tpu_backend is not None else TpuBackend()
        self.npu = npu_backend if npu_backend is not None else NpuBackend()

    # -- capability flags -------------------------------------------------
    @property
    def gpu_available(self) -> bool:
        return self._nvml_ready

    @property
    def ram_available(self) -> bool:
        return _PSUTIL_OK

    @property
    def tpu_available(self) -> bool:
        return bool(self.tpu and self.tpu.available)

    @property
    def npu_available(self) -> bool:
        return bool(self.npu and self.npu.available)

    # -- sampling ---------------------------------------------------------
    def _sample_ram(self, s: Sample) -> None:
        if not _PSUTIL_OK:
            return
        vm = psutil.virtual_memory()
        s.ram_total_mb = vm.total / 1024 / 1024
        s.ram_used_mb = (vm.total - vm.available) / 1024 / 1024
        s.ram_percent = float(vm.percent)
        # non-blocking read; first call returns 0.0 by design
        s.cpu_percent = psutil.cpu_percent(interval=None)

    def _sample_gpu(self, s: Sample) -> None:
        if not self._nvml_ready:
            return
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            s.gpu_total_mb = mem.total / 1024 / 1024
            s.gpu_used_mb = mem.used / 1024 / 1024
            s.gpu_percent = 100.0 * mem.used / mem.total if mem.total else None
            s.gpu_util = float(util.gpu)
        except Exception:  # pragma: no cover - transient NVML errors
            pass

    def _sample_accelerator(self, s: Sample, backend: Optional[AcceleratorBackend],
                            index: int, prefix: str) -> None:
        if not backend or not backend.available:
            return
        reading = backend.read(index)
        if reading is None:
            return
        setattr(s, f"{prefix}_used_mb", reading.used_mb)
        setattr(s, f"{prefix}_total_mb", reading.total_mb)
        setattr(s, f"{prefix}_percent", reading.percent)
        setattr(s, f"{prefix}_util", reading.util)

    def sample(self) -> Sample:
        """Return one snapshot of current usage."""
        s = Sample(ts=time.time())
        self._sample_ram(s)
        self._sample_gpu(s)
        self._sample_accelerator(s, self.tpu, self.tpu_index, "tpu")
        self._sample_accelerator(s, self.npu, self.npu_index, "npu")
        return s

    def collect(self, n: int, interval: float = 1.0) -> List[Sample]:
        """Collect ``n`` samples spaced ``interval`` seconds apart."""
        out: List[Sample] = []
        for i in range(n):
            out.append(self.sample())
            if i < n - 1:
                time.sleep(interval)
        return out

    def close(self) -> None:
        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                pass
            self._nvml_ready = False

    def __enter__(self) -> "ResourceCollector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
