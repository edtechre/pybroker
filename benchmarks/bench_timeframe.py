"""Micro-benchmark for timeframe compression hot path."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pybroker.common import DataCol
from pybroker.timeframe import compress, compress_symbol_df

ONE_MINUTE_SECONDS = 60.0
DAILY_SECONDS = 86400.0


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


def _time_compress(label: str, fn, iterations: int = 200) -> None:
    # Warm up Numba JIT.
    fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    print(
        f"{label}: {elapsed / iterations * 1e6:.1f} us/op ({iterations} iters)"
    )


def main() -> None:
    dates, o, h, low, c, v = _minute_bars()
    _time_compress(
        "compress 5m (390 1-min bars)",
        lambda: compress(dates, o, h, low, c, v, "5m"),
    )

    dates, o, h, low, c, v = _daily_bars()
    _time_compress(
        "compress weekly (252 daily bars)",
        lambda: compress(dates, o, h, low, c, v, "weekly"),
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
    _time_compress(
        "compress_symbol_df weekly (252 daily bars)",
        lambda: compress_symbol_df(
            sym_df, "weekly", frozenset(), DAILY_SECONDS
        ),
        iterations=100,
    )


if __name__ == "__main__":
    main()
