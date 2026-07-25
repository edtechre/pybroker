"""DataSource micro-benchmarks for cache I/O and YFinance reshape.

Pinned fixtures only — no live API calls. Cache benches use a 50-symbol
dataset synthesized from ``tests/testdata/daily_1.pkl`` to expose O(n²)
concat cost on cache reads.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd
import yfinance

import pybroker
from pybroker.cache import (
    clear_data_source_cache,
    disable_data_source_cache,
    enable_data_source_cache,
)
from pybroker.data import DataSourceCacheMixin, YFinance

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "tests" / "testdata" / "daily_1.pkl"
YFINANCE_PATH = REPO_ROOT / "tests" / "testdata" / "yfinance.pkl"

NUM_CACHE_SYMBOLS = 50
TIMEFRAME = "1d"
START_DATE = datetime.strptime("2021-02-02", "%Y-%m-%d")
END_DATE = datetime.strptime("2022-02-02", "%Y-%m-%d")
ADJUST = "all"


def _load_base_bars() -> pd.DataFrame:
    df = pd.read_pickle(DATA_PATH)  # noqa: S301 - trusted test fixture
    df["date"] = pd.to_datetime(df["date"])
    return df


def _build_multi_symbol_bars(
    num_symbols: int,
) -> tuple[pd.DataFrame, list[str]]:
    base = _load_base_bars()
    template_sym = base["symbol"].iloc[0]
    template = base[base["symbol"] == template_sym].copy()
    symbols = [f"SYM{i:03d}" for i in range(num_symbols)]
    frames = []
    for sym in symbols:
        sym_df = template.copy()
        sym_df["symbol"] = sym
        frames.append(sym_df)
    return pd.concat(frames, ignore_index=True), symbols


def _load_yfinance_fixture() -> pd.DataFrame:
    return pd.read_pickle(YFINANCE_PATH)  # noqa: S301 - trusted test fixture


class DataSourceCacheRead:
    """Full cache hit: get_cached for 50 symbols."""

    timeout = 60

    def setup(self) -> None:
        pybroker.disable_logging()
        self._cache_dir = tempfile.TemporaryDirectory()
        enable_data_source_cache("bench-data-read", self._cache_dir.name)
        self._mixin = DataSourceCacheMixin()
        self._df, self._symbols = _build_multi_symbol_bars(NUM_CACHE_SYMBOLS)
        self._mixin.set_cached(
            TIMEFRAME, START_DATE, END_DATE, ADJUST, self._df
        )
        self._mixin.get_cached(
            self._symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST
        )

    def teardown(self) -> None:
        clear_data_source_cache()
        disable_data_source_cache()
        self._cache_dir.cleanup()

    def time_cache_read(self) -> None:
        self._mixin.get_cached(
            self._symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST
        )


class DataSourceCacheWrite:
    """Write 50 symbols to disk cache via set_cached."""

    timeout = 60

    def setup(self) -> None:
        pybroker.disable_logging()
        self._cache_dir = tempfile.TemporaryDirectory()
        enable_data_source_cache("bench-data-write", self._cache_dir.name)
        self._mixin = DataSourceCacheMixin()
        self._df, _ = _build_multi_symbol_bars(NUM_CACHE_SYMBOLS)
        clear_data_source_cache()
        self._mixin.set_cached(
            TIMEFRAME, START_DATE, END_DATE, ADJUST, self._df
        )

    def teardown(self) -> None:
        clear_data_source_cache()
        disable_data_source_cache()
        self._cache_dir.cleanup()

    def time_cache_write(self) -> None:
        clear_data_source_cache()
        self._mixin.set_cached(
            TIMEFRAME, START_DATE, END_DATE, ADJUST, self._df
        )


class YFinanceMultiSymbol:
    """Multi-symbol yfinance download reshape (_fetch_data only)."""

    timeout = 60

    def setup(self) -> None:
        pybroker.disable_logging()
        disable_data_source_cache()
        self._fixture = _load_yfinance_fixture()
        self._symbols = frozenset(
            self._fixture.columns.get_level_values(1).unique()
        )
        self._yf = YFinance(auto_adjust=False)
        self._patch = mock.patch.object(
            yfinance, "download", return_value=self._fixture
        )
        self._patch.start()
        self._yf._fetch_data(self._symbols, START_DATE, END_DATE, "1d", None)

    def teardown(self) -> None:
        self._patch.stop()

    def time_yfinance_reshape(self) -> None:
        self._yf._fetch_data(self._symbols, START_DATE, END_DATE, "1d", None)
