"""Run common.py bottleneck benchmarks and compare to a saved baseline."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / ".asv" / "baseline-pre-common-opt.json"


def _median_ms(
    fn: Callable[[], None], warmup: int = 2, repeats: int = 7
) -> float:
    times: list[float] = []
    for i in range(warmup + repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(ms)
    return statistics.median(times)


def _run_bottleneck() -> dict[str, float]:
    from benchmarks.bench_common import ResultQuantize, TimeframeCompression

    out: dict[str, float] = {}

    quantize_bench = ResultQuantize()
    quantize_bench.setup()
    out["ResultQuantize.time_result_quantize"] = _median_ms(
        quantize_bench.time_result_quantize
    )

    tf_bench = TimeframeCompression()
    tf_bench.setup()
    out["TimeframeCompression.time_compress_5m_large"] = _median_ms(
        tf_bench.time_compress_5m_large
    )
    out["TimeframeCompression.time_compress_timeframes_multi"] = _median_ms(
        tf_bench.time_compress_timeframes_multi
    )

    return out


def _run_macro() -> dict[str, float]:
    from benchmarks.bench_backtest import (
        StoreSliceKernels,
        WalkforwardLarge,
        WalkforwardTimeframes,
    )

    out: dict[str, float] = {}

    store = StoreSliceKernels()
    store.setup(10, 252 * 5)
    out[
        "StoreSliceKernels.n_symbols=10.n_days=1260."
        "time_slice_symbol_array_store_by_dates"
    ] = _median_ms(
        lambda: store.time_slice_symbol_array_store_by_dates(10, 252 * 5)
    )

    wf_tf = WalkforwardTimeframes()
    wf_tf.setup()
    out["WalkforwardTimeframes.time_walkforward_timeframes"] = _median_ms(
        wf_tf.time_walkforward_timeframes
    )

    wf_large = WalkforwardLarge()
    wf_large.setup()
    out["WalkforwardLarge.time_walkforward_large"] = _median_ms(
        wf_large.time_walkforward_large
    )

    return out


def run_all() -> dict[str, Any]:
    bottleneck = _run_bottleneck()
    macro = _run_macro()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    return {"commit": commit, "bottleneck": bottleneck, "macro": macro}


def save_baseline(path: Path) -> None:
    results = run_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "commit": results["commit"],
        "bottleneck": results["bottleneck"],
        "macro": results["macro"],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"Saved common baseline to {path}")


def _pct(base: float, cur: float) -> str:
    if base == 0:
        return "—"
    return f"{((cur - base) / base) * 100:+.1f}%"


def _print_section(
    title: str,
    base_metrics: dict[str, float],
    cur_metrics: dict[str, float],
) -> None:
    print(title)
    print(f"{'Benchmark':<72} {'Before':>10} {'After':>10} {'Change':>8}")
    print("-" * 104)
    for name in sorted(set(base_metrics) | set(cur_metrics)):
        b, c = base_metrics.get(name), cur_metrics.get(name)
        if b is None:
            print(f"{name:<72} {'—':>10} {c:>10.2f} {'new':>8}")
        elif c is None:
            print(f"{name:<72} {b:>10.2f} {'—':>10} {'removed':>8}")
        else:
            print(f"{name:<72} {b:>10.2f} {c:>10.2f} {_pct(b, c):>8}")
    print()


def compare(baseline_path: Path) -> int:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    cur = run_all()
    _print_section(
        "Bottleneck benchmarks:",
        base.get("bottleneck", {}),
        cur["bottleneck"],
    )
    _print_section(
        "Macro regression benchmarks:",
        base.get("macro", {}),
        cur["macro"],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-baseline",
        type=Path,
        nargs="?",
        const=DEFAULT_BASELINE,
        default=None,
    )
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="?",
        const=DEFAULT_BASELINE,
        default=None,
    )
    args = parser.parse_args(argv)

    if args.save_baseline is not None:
        save_baseline(args.save_baseline)
        return 0
    if args.compare is not None:
        return compare(args.compare)

    results = run_all()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
