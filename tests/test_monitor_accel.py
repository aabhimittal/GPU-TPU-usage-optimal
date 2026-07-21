"""End-to-end monitor flow with injected TPU/NPU back-ends."""
from resource_optimal.advisor import OptimalAdvisor
from resource_optimal.backends import AcceleratorReading
from resource_optimal.collector import ResourceCollector
from helpers import FakeBackend

from resource_optimal.monitor import ResourceMonitor


def _monitor_with_accels():
    tpu = FakeBackend("tpu", AcceleratorReading(used_mb=8000, total_mb=16000, util=70))
    npu = FakeBackend("npu", AcceleratorReading(used_mb=3900, total_mb=4000, util=95))
    collector = ResourceCollector(tpu_backend=tpu, npu_backend=npu)
    return ResourceMonitor(collector=collector, advisor=OptimalAdvisor())


def test_monitor_recommends_for_tpu_and_npu():
    mon = _monitor_with_accels()
    try:
        mon.run(samples=6, interval=0)
        recs = {r.resource: r for r in mon.recommendations(horizon=3)}
        assert "TPU" in recs and "NPU" in recs
        # NPU is nearly full -> should not be flagged idle
        assert recs["NPU"].status in {"tight", "over_capacity", "ok"}
    finally:
        mon.close()


def test_dashboard_includes_accelerator_rows():
    mon = _monitor_with_accels()
    try:
        mon.run(samples=4, interval=0)
        out = mon.dashboard()
        assert "TPU" in out and "NPU" in out
    finally:
        mon.close()
