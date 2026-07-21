"""Command-line interface: ``python -m resource_optimal``."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .monitor import ResourceMonitor


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="resource-optimal",
        description="Track, visualise and optimise GPU/RAM usage.",
    )
    p.add_argument("-n", "--samples", type=int, default=10,
                   help="number of samples to collect (default: 10)")
    p.add_argument("-i", "--interval", type=float, default=1.0,
                   help="seconds between samples (default: 1.0)")
    p.add_argument("--db", default=":memory:",
                   help="SQLite path for history (default: in-memory)")
    p.add_argument("--gpu-index", type=int, default=0,
                   help="GPU index to monitor (default: 0)")
    p.add_argument("--tpu-index", type=int, default=0,
                   help="TPU index to monitor (default: 0)")
    p.add_argument("--npu-index", type=int, default=0,
                   help="NPU index to monitor (default: 0)")
    p.add_argument("--horizon", type=int, default=5,
                   help="forecast horizon in steps (default: 5)")
    p.add_argument("--live", action="store_true",
                   help="redraw the dashboard after every sample")
    p.add_argument("--report", metavar="PATH",
                   help="write a matplotlib PNG report to PATH")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    with ResourceMonitor(
        db_path=args.db,
        gpu_index=args.gpu_index,
        tpu_index=args.tpu_index,
        npu_index=args.npu_index,
    ) as mon:
        c = mon.collector
        if not (c.ram_available or c.gpu_available or c.tpu_available
                or c.npu_available):
            print("warning: no back-end available; install 'psutil' (RAM), "
                  "'nvidia-ml-py' (GPU), 'tpu-info'/JAX (TPU) or Ascend "
                  "'npu-smi' (NPU) for real metrics.",
                  file=sys.stderr)
        recs = mon.run(args.samples, interval=args.interval, live=args.live)
        print("\n" + mon.dashboard())
        print("\nOptimal-usage recommendations (horizon="
              f"{args.horizon}):")
        recs = mon.recommendations(horizon=args.horizon)
        if not recs:
            print("  (not enough data — collect more samples)")
        for r in recs:
            print(f"  [{r.status}] {r.resource}: {r.message}")

        if args.report:
            _write_report(mon, args)
    return 0


def _write_report(mon: ResourceMonitor, args: argparse.Namespace) -> None:
    from .forecaster import UsageForecaster
    from .visualizer import save_report

    ram = mon.store.series("ram_used_mb", limit=240)
    cap = mon._last_nonnull("ram_total_mb")
    if not ram or not cap:
        print("  (no RAM history to report)", file=sys.stderr)
        return
    fc = UsageForecaster().forecast(ram, horizon=args.horizon)
    rec = mon.advisor.recommend("RAM", ram, cap, horizon=args.horizon)
    try:
        out = save_report(args.report, ram, fc, rec, title="RAM usage forecast")
        print(f"  report written to {out}")
    except RuntimeError as exc:
        print(f"  report skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
