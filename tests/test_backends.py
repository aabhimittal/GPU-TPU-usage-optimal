"""Tests for TPU/NPU back-ends and their integration through the collector."""
from resource_optimal.backends import AcceleratorReading, NpuBackend, TpuBackend
from resource_optimal.collector import ResourceCollector


class FakeBackend:
    """Injectable accelerator back-end returning a fixed reading."""

    def __init__(self, name, reading, available=True):
        self.name = name
        self._reading = reading
        self._available = available

    @property
    def available(self):
        return self._available

    def read(self, index=0):
        return self._reading


# ---- AcceleratorReading -------------------------------------------------
def test_reading_percent_computed():
    r = AcceleratorReading(used_mb=4096, total_mb=16384, util=50.0)
    assert r.percent == 25.0


def test_reading_percent_none_without_total():
    assert AcceleratorReading(used_mb=100).percent is None
    assert AcceleratorReading(used_mb=100, total_mb=0).percent is None


# ---- default back-ends probe safely -------------------------------------
def test_default_backends_do_not_raise_when_hardware_absent():
    # On CI without TPU/NPU these must simply report unavailable.
    assert TpuBackend().available in (True, False)
    assert NpuBackend().available in (True, False)


# ---- npu-smi parser -----------------------------------------------------
def test_ascend_parser_extracts_memory_and_util():
    text = (
        "NPU  Name  | Health | AI-Core(%) : 42\n"
        "HBM-Usage(MB) : 1234 / 32768\n"
    )
    r = NpuBackend._parse_ascend(text)
    assert r is not None
    assert r.used_mb == 1234 and r.total_mb == 32768
    assert r.util == 42.0
    assert round(r.percent, 2) == round(100 * 1234 / 32768, 2)


def test_ascend_parser_returns_none_on_garbage():
    assert NpuBackend._parse_ascend("no useful metrics here") is None


# ---- OpenVINO NPU memory path -------------------------------------------
class FakeCore:
    """Minimal stand-in for openvino.Core exposing get_property."""

    def __init__(self, props, devices=("NPU",)):
        self._props = props
        self.available_devices = list(devices)

    def get_property(self, device, key):
        if device == "NPU" and key in self._props:
            return self._props[key]
        raise RuntimeError(f"unknown property {key}")


def test_openvino_reports_total_and_used():
    core = FakeCore({
        "NPU_DEVICE_TOTAL_MEM_SIZE": 8 * 1024 * 1024 * 1024,   # 8 GiB
        "NPU_DEVICE_ALLOC_MEM_SIZE": 2 * 1024 * 1024 * 1024,   # 2 GiB
    })
    r = NpuBackend._query_openvino(core)
    assert r is not None
    assert round(r.total_mb) == 8192 and round(r.used_mb) == 2048
    assert round(r.percent) == 25


def test_openvino_total_only_leaves_percent_none():
    core = FakeCore({"DEVICE_TOTAL_MEM_SIZE": 4 * 1024 * 1024 * 1024})
    r = NpuBackend._query_openvino(core)
    assert r is not None
    assert round(r.total_mb) == 4096
    assert r.used_mb is None and r.percent is None


def test_openvino_no_memory_props_returns_none():
    assert NpuBackend._query_openvino(FakeCore({})) is None


def test_openvino_first_prop_skips_bad_values():
    core = FakeCore({
        "NPU_DEVICE_TOTAL_MEM_SIZE": "not-a-number",
        "DEVICE_TOTAL_MEM_SIZE": 0,               # non-positive, skipped
        "GPU_DEVICE_TOTAL_MEM_SIZE": 1024 * 1024,  # 1 MiB, first valid
    })
    got = NpuBackend._first_prop_bytes(core, "NPU", NpuBackend._OV_TOTAL_KEYS)
    assert got == 1024 * 1024


# ---- collector integration with injected back-ends ----------------------
def test_collector_samples_injected_tpu_and_npu():
    tpu = FakeBackend("tpu", AcceleratorReading(used_mb=8000, total_mb=16000, util=70))
    npu = FakeBackend("npu", AcceleratorReading(used_mb=500, total_mb=4000, util=30))
    with ResourceCollector(tpu_backend=tpu, npu_backend=npu) as c:
        assert c.tpu_available and c.npu_available
        s = c.sample()
    assert s.tpu_used_mb == 8000 and s.tpu_total_mb == 16000
    assert s.tpu_percent == 50.0 and s.tpu_util == 70
    assert s.npu_used_mb == 500 and s.npu_percent == 12.5


def test_collector_ignores_unavailable_backend():
    tpu = FakeBackend("tpu", AcceleratorReading(used_mb=1), available=False)
    with ResourceCollector(tpu_backend=tpu) as c:
        assert not c.tpu_available
        s = c.sample()
    assert s.tpu_used_mb is None


def test_collector_handles_none_reading():
    npu = FakeBackend("npu", None)
    with ResourceCollector(npu_backend=npu) as c:
        s = c.sample()
    assert s.npu_used_mb is None
