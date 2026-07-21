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
