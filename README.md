# GPU / RAM usage-optimal

A lightweight Python resource **tracker + advisor**: it samples GPU, TPU, NPU
and RAM usage, stores the time series, forecasts near-future demand with a small
online model, and recommends an *optimal* resource budget for running intensive
tasks — all with an efficient, zero-dependency terminal visualisation.

## The novel idea

Most monitors show you the *past*. This one sizes the *future*. Instead of
predicting a single number, the forecaster projects a **demand band** — the
expected usage level plus an empirical headroom derived from recent volatility —
and the advisor budgets resources against the *upper* edge of that band plus a
safety margin, clamped to the device's real capacity. That turns raw history
into a concrete answer: *reserve N MB, you have M MB of headroom* — or *this
task will not fit, shard it.*

```
history ─▶ TimeSeriesStore ─▶ UsageForecaster ─▶ OptimalAdvisor ─▶ Recommendation
                                     │                                    │
                                     └────────── Visualizer ──────────────┘
```

## Install

```bash
pip install -e .            # core (pure Python, no deps)
pip install -e ".[all]"     # real metrics + plots + ML model
```

Optional extras: `metrics` (psutil + NVML), `viz` (matplotlib), `ml`
(scikit-learn). The core runs and is fully tested without any of them.

### Accelerator back-ends

| Accelerator | Source tried | Install |
| --- | --- | --- |
| **GPU** (NVIDIA) | NVML via `nvidia-ml-py` | `pip install nvidia-ml-py` |
| **TPU** (Google) | `tpu-info` package, else JAX `memory_stats()` | `pip install tpu-info` |
| **NPU** | Huawei Ascend `npu-smi info`, else OpenVINO presence | vendor toolkit on `PATH` |

Back-ends probe safely and report *unavailable* when the hardware or tooling is
missing. You can also inject your own — anything satisfying `AcceleratorBackend`
(an `available` flag + `read(index) -> AcceleratorReading`):

```python
from resource_optimal import ResourceCollector
collector = ResourceCollector(tpu_backend=MyTpuBackend(), npu_backend=MyNpuBackend())
```

## Quick start

Command line:

```bash
python -m resource_optimal -n 20 -i 1 --horizon 5 --live
python -m resource_optimal -n 60 --db usage.db --report report.png
```

Library:

```python
from resource_optimal import ResourceMonitor

with ResourceMonitor(db_path="usage.db") as mon:
    recs = mon.run(samples=30, interval=1.0, live=True)
    for r in recs:
        print(r.status, r.resource, r.message)
```

No GPU handy? Run the offline demo on a synthetic trend:

```bash
python examples/demo.py
```

## Components

| Module | Role |
| --- | --- |
| `collector` | Samples RAM (psutil), GPU (NVML), TPU and NPU; degrades to `None` when a back-end is absent |
| `backends` | Pluggable TPU (`tpu-info`/JAX) and NPU (Ascend `npu-smi`/OpenVINO) readers; injectable for custom hardware |
| `storage` | SQLite time-series store of samples |
| `forecaster` | Holt's linear smoother (pure Python) + optional ridge AR model |
| `advisor` | Demand-band → optimal reserve + status (`ok`/`tight`/`over_capacity`/`over_provisioned`) |
| `visualizer` | Unicode sparklines/dashboard + optional matplotlib PNG report |
| `monitor` | Orchestrates collect → store → forecast → recommend |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite covers the core (forecaster, advisor, storage, visualiser) and
requires no optional dependencies.

## License

MIT — see [LICENSE](LICENSE).
