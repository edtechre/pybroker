"""Micro-benchmark for timeframe compression hot path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pybroker.common import DataCol
from pybroker.timeframe import (
    compress,
    compress_symbol_df,
    compress_timeframes_from_frame,
)

DAILY_SECONDS = 86400.0
LARGE_MINUTE_BARS = 252 * 390


def _minute_bars(n: int = 390) -> tuple[np.ndarray, ...]:
    dates = pd.date_range("2020-01-06 09:30", periods=n, freq="1min").to_numpy(
        dtype="datetime64[ns]"
    )
    close = np.linspace(100.0, 110.0, n)
    return (
        dates,
        close,
        close + 0.5,
        close - 0.5,
        close,
        np.ones(n),
    )


def _daily_bars(n: int = 252) -> tuple[np.ndarray, ...]:
    dates = pd.date_range("2020-01-02", periods=n, freq="B").to_numpy(
        dtype="datetime64[ns]"
    )
    close = np.linspace(100.0, 150.0, n)
    return (
        dates,
        close,
        close + 1.0,
        close - 1.0,
        close,
        np.ones(n) * 1_000_000,
    )


def _multi_symbol_daily_df(
    n_symbols: int = 10, n_days: int = 504
) -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    frames: list[pd.DataFrame] = []
    for i in range(n_symbols):
        sym = f"SYM{i:02d}"
        close = np.linspace(100.0 + i, 150.0 + i, n_days)
        frames.append(
            pd.DataFrame(
                {
                    DataCol.SYMBOL.value: [sym] * n_days,
                    DataCol.DATE.value: dates,
                    DataCol.OPEN.value: close,
                    DataCol.HIGH.value: close + 1.0,
                    DataCol.LOW.value: close - 1.0,
                    DataCol.CLOSE.value: close,
                    DataCol.VOLUME.value: np.ones(n_days) * 1_000_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _time_case(fn: Callable[[], object], iterations: int = 200) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1e6


def _run_benchmarks() -> dict[str, float]:
    results: dict[str, float] = {}

    dates, o, h, low, c, v = _minute_bars()
    results["compress_5m_390"] = _time_case(
        lambda: compress(dates, o, h, low, c, v, "5m")
    )

    dates, o, h, low, c, v = _daily_bars()
    results["compress_weekly_252"] = _time_case(
        lambda: compress(dates, o, h, low, c, v, "weekly")
    )

    sym_df = pd.DataFrame(
        {
            DataCol.DATE.value: dates,
            DataCol.OPEN.value: o,
            DataCol.HIGH.value: h,
            DataCol.LOW.value: low,
            DataCol.CLOSE.value: c,
            DataCol.VOLUME.value: v,
        }
    )
    results["compress_symbol_df_weekly_252"] = _time_case(
        lambda: compress_symbol_df(
            sym_df, "weekly", frozenset(), DAILY_SECONDS
        ),
        iterations=100,
    )

    dates, o, h, low, c, v = _minute_bars(LARGE_MINUTE_BARS)
    results["compress_5m_large"] = _time_case(
        lambda: compress(dates, o, h, low, c, v, "5m"),
        iterations=50,
    )
    results["compress_every5_large"] = _time_case(
        lambda: compress(dates, o, h, low, c, v, 5),
        iterations=50,
    )

    multi_df = _multi_symbol_daily_df()
    symbols = set(multi_df[DataCol.SYMBOL.value].unique())
    intervals = frozenset({"weekly", 5, "monthly"})
    custom_cols: frozenset[str] = frozenset()
    results["compress_timeframes_multi"] = _time_case(
        lambda: compress_timeframes_from_frame(
            multi_df,
            symbols,
            intervals,
            custom_cols,
            DAILY_SECONDS,
        ),
        iterations=20,
    )

    return results


def _print_results(results: dict[str, float]) -> None:
    for name, us_per_op in results.items():
        print(f"{name}: {us_per_op:.1f} us/op")


def _pct_change(base: float, cur: float) -> str:
    if base == 0:
        return "—"
    return f"{((cur - base) / base) * 100:+.1f}%"


def _compare(baseline_path: Path, current: dict[str, float]) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_metrics: dict[str, float] = baseline.get("metrics", baseline)
    names = sorted(set(base_metrics) | set(current))
    print(f"{'Benchmark':<32} {'Baseline':>12} {'Current':>12} {'Change':>10}")
    print("-" * 68)
    for name in names:
        b = base_metrics.get(name)
        c = current.get(name)
        b_str = f"{b:.1f}us" if b is not None else "—"
        c_str = f"{c:.1f}us" if c is not None else "—"
        change = _pct_change(b, c) if b is not None and c is not None else "—"
        print(f"{name:<32} {b_str:>12} {c_str:>12} {change:>10}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-baseline",
        type=Path,
        metavar="PATH",
        help="Save benchmark results as baseline JSON",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        metavar="PATH",
        help="Compare current results against baseline JSON",
    )
    args = parser.parse_args(argv)

    results = _run_benchmarks()

    if args.save_baseline:
        args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metrics": results}
        args.save_baseline.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            f"Saved baseline ({len(results)} benchmarks) to {args.save_baseline}"
        )
        _print_results(results)
        return 0

    if args.compare:
        _compare(args.compare, results)
        return 0

    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
