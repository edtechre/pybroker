"""Unit tests for data.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import akshare
import os
import pandas as pd
import pytest
import re
import sys
import yfinance
from .fixtures import *  # noqa: F401
from datetime import datetime
from pybroker.cache import DataSourceCacheKey
from pybroker.common import to_seconds
from pybroker.data import (
    Alpaca,
    AlpacaCrypto,
    DataSource,
    DataSourceCacheMixin,
    YFinance,
)
from pybroker.ext.data import AKShare
from pybroker.ext.data import YQuery
from unittest import mock
from yahooquery import Ticker

API_KEY = "api_key"
API_SECRET = "api_secret"
API_VERSION = "v2"
TIMEFRAME = "1m"
START_DATE = datetime.strptime("2021-02-02", "%Y-%m-%d")
END_DATE = datetime.strptime("2022-02-02", "%Y-%m-%d")
ADJUST = "all"
ALPACA_COLS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
]
ALPACA_CRYPTO_COLS = ALPACA_COLS + ["trade_count"]


@pytest.fixture()
def alpaca_df():
    df = pd.read_pickle(
        os.path.join(os.path.dirname(__file__), "testdata/daily_1.pkl")
    )
    df["date"] = df["date"].dt.tz_localize("US/Eastern")
    return df.assign(vwap=1)[ALPACA_COLS]


@pytest.fixture()
def alpaca_crypto_df():
    df = pd.read_pickle(
        os.path.join(os.path.dirname(__file__), "testdata/daily_1.pkl")
    )
    df["date"] = df["date"].dt.tz_localize("US/Eastern")
    return df.assign(vwap=1, trade_count=1)[ALPACA_CRYPTO_COLS]


@pytest.fixture()
def bars_df(alpaca_df):
    return alpaca_df.rename(columns={"date": "timestamp"})


@pytest.fixture()
def crypto_bars_df(alpaca_crypto_df):
    return alpaca_crypto_df.rename(columns={"date": "timestamp"})


@pytest.fixture()
def yfinance_df():
    return pd.read_pickle(
        os.path.join(os.path.dirname(__file__), "testdata/yfinance.pkl")
    )


@pytest.fixture()
def yfinance_single_df():
    return pd.read_pickle(
        os.path.join(os.path.dirname(__file__), "testdata/yfinance_single.pkl")
    )


@pytest.fixture()
def symbols(alpaca_df):
    return list(alpaca_df["symbol"].unique())


@pytest.fixture()
def mock_cache(scope):
    with (
        mock.patch.object(scope, "data_source_cache") as cache,
        mock.patch.object(cache, "get", return_value=None),
    ):
        yield cache


@pytest.fixture()
def mock_alpaca():
    with mock.patch(
        "alpaca.data.historical.stock.StockHistoricalDataClient"
    ) as client:
        yield client


@pytest.fixture()
def mock_alpaca_crypto():
    with mock.patch(
        "alpaca.data.historical.crypto.CryptoHistoricalDataClient"
    ) as client:
        yield client


class TestDataSourceCacheMixin:
    def test_get_cached_when_different_source_then_miss(
        self, scope, alpaca_df, symbols
    ):
        """Two DataSources sharing one cache namespace must not cross-serve
        each other's bars: the cache key carries the source identity."""

        class SourceA(DataSourceCacheMixin):
            pass

        class SourceB(DataSourceCacheMixin):
            pass

        class DictCache:
            def __init__(self):
                self.store = {}

            def get(self, key, default=None):
                return self.store.get(key, default)

            def set(self, key, value):
                self.store[key] = value
                return True

        with mock.patch.object(scope, "data_source_cache", DictCache()):
            SourceA().set_cached(
                TIMEFRAME, START_DATE, END_DATE, ADJUST, alpaca_df
            )
            df_a, uncached_a = SourceA().get_cached(
                symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST
            )
            assert not df_a.empty
            assert not uncached_a
            df_b, uncached_b = SourceB().get_cached(
                symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST
            )
            assert df_b.empty
            assert list(uncached_b) == list(symbols)

    def test_query_when_full_cache_hit_then_deterministic_order(
        self, scope, alpaca_df, symbols
    ):
        """A full cache hit must return the same date-major row order and
        RangeIndex as the cold fetch path, independent of hash seed."""

        class FakeSource(DataSource):
            def _fetch_data(
                self, symbols, start_date, end_date, timeframe, adjust
            ):
                return alpaca_df[alpaca_df["symbol"].isin(symbols)]

        class DictCache:
            def __init__(self):
                self.store = {}

            def get(self, key, default=None):
                return self.store.get(key, default)

            def set(self, key, value):
                self.store[key] = value
                return True

        with mock.patch.object(scope, "data_source_cache", DictCache()):
            source = FakeSource()
            cold = source.query(
                symbols, START_DATE, END_DATE, TIMEFRAME, ADJUST
            )
            warm = source.query(
                symbols, START_DATE, END_DATE, TIMEFRAME, ADJUST
            )
        pd.testing.assert_frame_equal(warm, cold)

    def test_query_when_symbol_has_no_data_then_cache_preserved(
        self, scope, alpaca_df, symbols
    ):
        """A symbol whose fetch returns a column-less empty frame must not
        wipe the cache and re-fetch every symbol on each query."""
        fetch_calls = []

        class FakeSource(DataSource):
            def _fetch_data(
                self, symbols, start_date, end_date, timeframe, adjust
            ):
                fetch_calls.append(set(symbols))
                rows = alpaca_df[alpaca_df["symbol"].isin(symbols)]
                if rows.empty:
                    return pd.DataFrame([])
                return rows

        class DictCache:
            def __init__(self):
                self.store = {}

            def get(self, key, default=None):
                return self.store.get(key, default)

            def set(self, key, value):
                self.store[key] = value
                return True

            def clear(self):
                self.store.clear()

        cache = DictCache()
        query_symbols = list(symbols) + ["ZZZNODATA"]
        with mock.patch.object(scope, "data_source_cache", cache):
            source = FakeSource()
            first = source.query(
                query_symbols, START_DATE, END_DATE, TIMEFRAME, ADJUST
            )
            cached_keys = set(cache.store)
            assert cached_keys
            second = source.query(
                query_symbols, START_DATE, END_DATE, TIMEFRAME, ADJUST
            )
        assert set(cache.store) == cached_keys
        # Only the no-data symbol is re-fetched on the second query.
        assert fetch_calls[-1] == {"ZZZNODATA"}
        pd.testing.assert_frame_equal(second, first)

    @pytest.mark.usefixtures("scope")
    def test_set_cached(self, alpaca_df, symbols, mock_cache):
        cache_mixin = DataSourceCacheMixin()
        cache_mixin.set_cached(
            TIMEFRAME, START_DATE, END_DATE, ADJUST, alpaca_df
        )
        assert len(mock_cache.set.call_args_list) == len(symbols)
        for i, sym in enumerate(symbols):
            expected_cache_key = DataSourceCacheKey(
                symbol=sym,
                tf_seconds=to_seconds(TIMEFRAME),
                start_date=START_DATE,
                end_date=END_DATE,
                adjust=ADJUST,
                source="pybroker.data.DataSourceCacheMixin",
            )
            cache_key, sym_df = mock_cache.set.call_args_list[i].args
            assert cache_key == expected_cache_key
            assert sym_df.equals(alpaca_df[alpaca_df["symbol"] == sym])

    @pytest.mark.usefixtures("scope")
    @pytest.mark.parametrize("query_symbols", [[], LazyFixture("symbols")])
    def test_get_cached_when_empty(self, mock_cache, query_symbols, request):
        query_symbols = get_fixture(request, query_symbols)
        cache_mixin = DataSourceCacheMixin()
        df, uncached_syms = cache_mixin.get_cached(
            query_symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST
        )
        assert df.empty
        assert uncached_syms == query_symbols
        assert len(mock_cache.get.call_args_list) == len(query_symbols)
        for i, sym in enumerate(query_symbols):
            expected_cache_key = DataSourceCacheKey(
                symbol=sym,
                tf_seconds=to_seconds(TIMEFRAME),
                start_date=START_DATE,
                end_date=END_DATE,
                adjust=ADJUST,
                source="pybroker.data.DataSourceCacheMixin",
            )
            cache_key = mock_cache.get.call_args_list[i].args[0]
            assert cache_key == expected_cache_key

    @pytest.mark.usefixtures("setup_enabled_ds_cache")
    def test_set_and_get_cached(self, alpaca_df, symbols):
        cache_mixin = DataSourceCacheMixin()
        cache_mixin.set_cached(
            TIMEFRAME, START_DATE, END_DATE, ADJUST, alpaca_df
        )
        df, uncached_syms = cache_mixin.get_cached(
            symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST
        )
        assert df.equals(alpaca_df)
        assert not len(uncached_syms)

    @pytest.mark.usefixtures("setup_enabled_ds_cache")
    def test_set_and_get_cached_when_partial(self, alpaca_df, symbols):
        cache_mixin = DataSourceCacheMixin()
        cached_df = alpaca_df[alpaca_df["symbol"].isin(symbols[:2])]
        cache_mixin.set_cached(
            TIMEFRAME, START_DATE, END_DATE, ADJUST, cached_df
        )
        df, uncached_syms = cache_mixin.get_cached(
            symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST
        )
        assert df.equals(cached_df)
        assert uncached_syms == symbols[2:]

    @pytest.mark.usefixtures("mock_cache")
    @pytest.mark.parametrize(
        "timeframe, start_date, end_date, error",
        [
            (
                "dffdfdf",
                datetime.strptime("2022-02-02", "%Y-%m-%d"),
                datetime.strptime("2021-02-02", "%Y-%m-%d"),
                ValueError,
            ),
            (
                "1m",
                "sdfdfdfg",
                datetime.strptime("2022-02-02", "%Y-%m-%d"),
                Exception,
            ),
            (
                "1m",
                datetime.strptime("2021-02-02", "%Y-%m-%d"),
                "sdfsdf",
                Exception,
            ),
        ],
    )
    def test_set_and_get_cached_when_invalid_times_then_error(
        self, alpaca_df, symbols, timeframe, start_date, end_date, error
    ):
        cache_mixin = DataSourceCacheMixin()
        with pytest.raises(error):
            cache_mixin.set_cached(
                timeframe, start_date, end_date, ADJUST, alpaca_df
            )
        with pytest.raises(error):
            cache_mixin.get_cached(
                symbols, timeframe, start_date, end_date, ADJUST
            )

    def test_set_and_get_cached_when_cache_disabled(self, alpaca_df, symbols):
        cache_mixin = DataSourceCacheMixin()
        cache_mixin.set_cached(
            TIMEFRAME, START_DATE, END_DATE, ADJUST, alpaca_df
        )
        df, uncached_syms = cache_mixin.get_cached(
            symbols, TIMEFRAME, START_DATE, END_DATE, ADJUST
        )
        assert df.empty
        assert uncached_syms == symbols


class TestAlpaca:
    def test_init(self, mock_alpaca):
        Alpaca(API_KEY, API_SECRET)
        mock_alpaca.assert_called_once_with(API_KEY, API_SECRET)

    @pytest.mark.usefixtures("setup_ds_cache", "mock_alpaca")
    def test_query_when_empty_cache(self, alpaca_df, bars_df, symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        mock_bars = mock.Mock()
        mock_bars.df = bars_df
        with mock.patch.object(
            alpaca._api, "get_stock_bars", return_value=mock_bars
        ):
            df = alpaca.query(
                symbols, START_DATE, END_DATE, TIMEFRAME, adjust="all"
            )
            df = (
                df.sort_values(["symbol", "date"])
                .reset_index(drop=True)
                .sort_index(axis=1)
            )
            expected = (
                alpaca_df.sort_values(["symbol", "date"])
                .reset_index(drop=True)
                .sort_index(axis=1)
            )
            assert df.equals(expected)

    def test_query_when_invalid_adj_then_error(self, symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        with pytest.raises(
            ValueError,
            match=re.escape("Unknown adjustment: foo"),
        ):
            alpaca.query(
                symbols, START_DATE, END_DATE, TIMEFRAME, adjust="foo"
            )

    @pytest.mark.usefixtures(
        "setup_enabled_ds_cache", "mock_alpaca", "tmp_path"
    )
    def test_query_when_partial_cache(self, alpaca_df, bars_df, symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        cached_df = alpaca_df[alpaca_df["symbol"].isin(symbols[-1:])]
        alpaca.set_cached(TIMEFRAME, START_DATE, END_DATE, ADJUST, cached_df)
        mock_bars = mock.Mock()
        mock_bars.df = bars_df[bars_df["symbol"].isin(symbols[:-1])]
        with mock.patch.object(
            alpaca._api, "get_stock_bars", return_value=mock_bars
        ):
            df = alpaca.query(symbols, START_DATE, END_DATE, TIMEFRAME, ADJUST)
            df = (
                df.sort_values(["symbol", "date"])
                .reset_index(drop=True)
                .sort_index(axis=1)
            )
            expected = (
                alpaca_df.sort_values(["symbol", "date"])
                .reset_index(drop=True)
                .sort_index(axis=1)
            )
            assert df.equals(expected)

    @pytest.mark.usefixtures(
        "setup_enabled_ds_cache", "mock_alpaca", "tmp_path"
    )
    def test_query_when_cache_mismatch(self, alpaca_df, bars_df, symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        cached_df = alpaca_df[alpaca_df["symbol"].isin(symbols[-1:])]
        cached_df = cached_df.drop(columns=["vwap"])
        alpaca.set_cached(TIMEFRAME, START_DATE, END_DATE, ADJUST, cached_df)
        mock_bars = mock.Mock()
        mock_bars.df = bars_df[bars_df["symbol"].isin(symbols[:-1])]
        with mock.patch.object(
            alpaca._api, "get_stock_bars", return_value=mock_bars
        ):
            df = alpaca.query(symbols, START_DATE, END_DATE, TIMEFRAME, ADJUST)
            assert not df.empty
            assert set(df.columns) == set(
                (
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "symbol",
                    "vwap",
                )
            )

    @pytest.mark.usefixtures("setup_ds_cache", "mock_alpaca")
    def test_query_when_cached(self, alpaca_df, bars_df, symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        mock_bars = mock.Mock()
        mock_bars.df = bars_df
        with mock.patch.object(
            alpaca._api, "get_stock_bars", return_value=mock_bars
        ):
            alpaca.query(symbols, START_DATE, END_DATE, TIMEFRAME)
            df = alpaca.query(symbols, START_DATE, END_DATE, TIMEFRAME)
            df = (
                df.sort_values(["symbol", "date"])
                .reset_index(drop=True)
                .sort_index(axis=1)
            )
            expected = (
                alpaca_df.sort_values(["symbol", "date"])
                .reset_index(drop=True)
                .sort_index(axis=1)
            )
            assert df.equals(expected)

    @pytest.mark.parametrize(
        "columns",
        [
            [],
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "vwap",
            ],
        ],
    )
    @pytest.mark.usefixtures("setup_ds_cache", "mock_alpaca")
    def test_query_when_empty_result(self, symbols, columns):
        alpaca = Alpaca(API_KEY, API_SECRET)
        mock_bars = mock.Mock()
        mock_bars.df = pd.DataFrame(columns=columns)
        with mock.patch.object(
            alpaca._api, "get_stock_bars", return_value=mock_bars
        ):
            df = alpaca.query(symbols, START_DATE, END_DATE, TIMEFRAME)
            assert df.empty
            assert set(df.columns) == set(
                (
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "symbol",
                    "vwap",
                )
            )

    @pytest.mark.parametrize("empty_symbols", ["", []])
    @pytest.mark.usefixtures("setup_ds_cache", "mock_alpaca")
    def test_query_when_symbols_empty(self, empty_symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        with pytest.raises(
            ValueError, match=re.escape("Symbols cannot be empty.")
        ):
            alpaca.query(empty_symbols, START_DATE, END_DATE, TIMEFRAME)

    @pytest.mark.parametrize("timeframe", ["1w 2d", "30s"])
    def test_query_when_invalid_timeframe_then_error(self, symbols, timeframe):
        alpaca = Alpaca(API_KEY, API_SECRET)
        with pytest.raises(
            ValueError,
            match=re.escape(f"Invalid Alpaca timeframe: {timeframe}"),
        ):
            alpaca.query(symbols, START_DATE, END_DATE, timeframe)

    def test_query_when_null_timeframe_then_error(self, symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        with pytest.raises(
            ValueError,
            match=re.escape("Timeframe needs to be specified for Alpaca."),
        ):
            alpaca.query(symbols, START_DATE, END_DATE, timeframe=None)


class TestAlpacaCrypto:
    def test_init(self, mock_alpaca_crypto):
        AlpacaCrypto(API_KEY, API_SECRET)
        mock_alpaca_crypto.assert_called_once_with(API_KEY, API_SECRET)

    @pytest.mark.usefixtures("setup_ds_cache", "mock_alpaca")
    def test_query(self, alpaca_crypto_df, crypto_bars_df, symbols):
        crypto = AlpacaCrypto(API_KEY, API_SECRET)
        mock_bars = mock.Mock()
        mock_bars.df = crypto_bars_df
        with mock.patch.object(
            crypto._api, "get_crypto_bars", return_value=mock_bars
        ):
            df = crypto.query(symbols, START_DATE, END_DATE, TIMEFRAME)
            df = (
                df.sort_values(["symbol", "date"])
                .reset_index(drop=True)
                .sort_index(axis=1)
            )
            expected = (
                alpaca_crypto_df.sort_values(["symbol", "date"])
                .reset_index(drop=True)
                .sort_index(axis=1)
            )
            assert df.equals(expected)

    @pytest.mark.parametrize(
        "columns",
        [
            [],
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "vwap",
                "trade_count",
            ],
        ],
    )
    @pytest.mark.usefixtures("setup_ds_cache", "mock_alpaca")
    def test_query_when_empty_result(self, symbols, columns):
        crypto = AlpacaCrypto(API_KEY, API_SECRET)
        mock_bars = mock.Mock()
        mock_bars.df = pd.DataFrame(columns=columns)
        with mock.patch.object(
            crypto._api, "get_crypto_bars", return_value=mock_bars
        ):
            df = crypto.query(symbols, START_DATE, END_DATE, TIMEFRAME)
            assert df.empty
            assert set(df.columns) == set(
                (
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "symbol",
                    "vwap",
                    "trade_count",
                )
            )

    @pytest.mark.parametrize("timeframe", ["1w 2d", "30s"])
    def test_query_when_invalid_timeframe_then_error(self, symbols, timeframe):
        crypto = AlpacaCrypto(API_KEY, API_SECRET)
        with pytest.raises(
            ValueError,
            match=re.escape(f"Invalid Alpaca timeframe: {timeframe}"),
        ):
            crypto.query(symbols, START_DATE, END_DATE, timeframe)

    def test_query_when_null_timeframe_then_error(self, symbols):
        crypto = Alpaca(API_KEY, API_SECRET)
        with pytest.raises(
            ValueError,
            match=re.escape("Timeframe needs to be specified for Alpaca."),
        ):
            crypto.query(symbols, START_DATE, END_DATE, timeframe=None)


class TestYFinance:
    @pytest.mark.parametrize(
        "param_symbols, expected_df, expected_rows",
        [
            (
                LazyFixture("symbols"),
                LazyFixture("yfinance_df"),
                2020,
            ),
            (["SPY"], LazyFixture("yfinance_single_df"), 505),
        ],
    )
    @pytest.mark.usefixtures("setup_ds_cache")
    @pytest.mark.parametrize("auto_adjust", [True, False])
    def test_query(
        self, param_symbols, expected_df, expected_rows, request, auto_adjust
    ):
        param_symbols = get_fixture(request, param_symbols)
        expected_df = get_fixture(request, expected_df)
        if auto_adjust:
            expected_df = expected_df.drop(columns=["Adj Close"])
        yf = YFinance(auto_adjust=auto_adjust)
        with mock.patch.object(yfinance, "download", return_value=expected_df):
            df = yf.query(param_symbols, START_DATE, END_DATE)
        expected_columns = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        }
        if not auto_adjust:
            expected_columns.add("adj_close")
        assert set(df.columns) == expected_columns
        assert df.shape[0] == expected_rows
        assert set(df["symbol"].unique()) == set(param_symbols)
        assert (df["date"].unique() == expected_df.index.unique()).all()

    @pytest.mark.usefixtures("setup_ds_cache")
    @pytest.mark.parametrize("auto_adjust", [True, False])
    def test_query_when_single_symbol_multiindex_columns(
        self, yfinance_single_df, auto_adjust
    ):
        # yfinance returns symbol-keyed MultiIndex columns even when a single
        # symbol is downloaded.
        expected_df = yfinance_single_df.copy()
        if auto_adjust:
            expected_df = expected_df.drop(columns=["Adj Close"])
        expected_df.columns = pd.MultiIndex.from_product(
            [expected_df.columns, ["SPY"]]
        )
        yf = YFinance(auto_adjust=auto_adjust)
        with mock.patch.object(yfinance, "download", return_value=expected_df):
            df = yf.query(["SPY"], START_DATE, END_DATE)
        expected_columns = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        }
        if not auto_adjust:
            expected_columns.add("adj_close")
        assert set(df.columns) == expected_columns
        assert df.shape[0] == 505
        assert set(df["symbol"].unique()) == {"SPY"}
        assert (df["date"].unique() == expected_df.index.unique()).all()

    @pytest.mark.usefixtures("setup_ds_cache")
    @pytest.mark.parametrize(
        "columns",
        [
            [],
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "adj_close",
            ],
        ],
    )
    @pytest.mark.parametrize("auto_adjust", [True, False])
    def test_query_when_empty_result(self, symbols, columns, auto_adjust):
        yf = YFinance(auto_adjust=auto_adjust)
        if auto_adjust and "adj_close" in columns:
            columns = [col for col in columns if col != "adj_close"]
        with mock.patch.object(
            yfinance, "download", return_value=pd.DataFrame(columns=columns)
        ):
            df = yf.query(symbols, START_DATE, END_DATE)
        assert df.empty
        expected_columns = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        }
        if not auto_adjust:
            expected_columns.add("adj_close")
        assert set(df.columns) == expected_columns


class TestAKShare:
    @pytest.mark.usefixtures("setup_ds_cache")
    @pytest.mark.parametrize("timeframe", [None, "", "1d", "1w"])
    def test_query(self, timeframe):
        symbols = ["A"]
        ak = AKShare()
        expected_df = pd.DataFrame(
            {
                "日期": [END_DATE],
                "开盘": [1],
                "收盘": [2],
                "最高": [3],
                "最低": [4],
                "成交量": [5],
                "symbol": symbols,
            }
        )
        with mock.patch("akshare.stock_zh_a_hist", return_value=expected_df):
            df = ak.query(symbols, START_DATE, END_DATE, timeframe)
        assert set(df.columns) == {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        }
        assert df.shape[0] == expected_df.shape[0]
        assert set(df["symbol"].unique()) == set(symbols)
        assert (df["date"].unique() == expected_df["日期"].unique()).all()

    @pytest.mark.parametrize(
        "columns",
        [
            [],
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
            ],
        ],
    )
    @pytest.mark.usefixtures("setup_ds_cache")
    def test_query_when_empty_result(self, columns):
        ak = AKShare()
        with mock.patch(
            "akshare.stock_zh_a_hist",
            return_value=pd.DataFrame(columns=columns),
        ):
            df = ak.query(["A"], START_DATE, END_DATE)
        assert df.empty
        assert set(df.columns) == set(
            (
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
            )
        )

    @pytest.mark.usefixtures("setup_ds_cache")
    def test_query_when_em_unavailable_then_uses_tx_fallback_legacy_schema(
        self,
    ):
        # akshare < 1.18.74: stock_zh_a_hist_tx reports volume under
        # "amount" and has no dedicated "volume" column.
        symbols = ["000001.SZ"]
        ak = AKShare()
        expected_df = pd.DataFrame(
            {
                "date": [END_DATE.date()],
                "open": [1.0],
                "close": [2.0],
                "high": [3.0],
                "low": [4.0],
                "amount": [5.0],
            }
        )
        with (
            mock.patch(
                "akshare.stock_zh_a_hist",
                side_effect=ConnectionError("failed"),
            ),
            mock.patch(
                "akshare.stock_zh_a_hist_tx", return_value=expected_df
            ) as mock_tx,
        ):
            df = ak.query(symbols, START_DATE, END_DATE, "1d")
        mock_tx.assert_called_once_with(
            symbol="sz000001",
            start_date=START_DATE.strftime("%Y%m%d"),
            end_date=END_DATE.strftime("%Y%m%d"),
            adjust="",
        )
        assert df.shape[0] == expected_df.shape[0]
        assert list(df.columns).count("volume") == 1
        assert set(df.columns) == {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        }
        assert df["volume"].iloc[0] == 5.0

    @pytest.mark.usefixtures("setup_ds_cache")
    def test_query_when_em_unavailable_then_uses_tx_fallback_current_schema(
        self,
    ):
        # akshare >= 1.18.74: stock_zh_a_hist_tx added a real "volume"
        # column and repurposed "amount"/"turnover" as distinct RMB
        # figures, no longer standing in for volume. Regression guard for
        # the rename map producing two "volume" columns.
        symbols = ["000001.SZ"]
        ak = AKShare()
        expected_df = pd.DataFrame(
            {
                "date": [END_DATE.date()],
                "open": [1.0],
                "close": [2.0],
                "high": [3.0],
                "low": [4.0],
                "volume": [6.0],
                "turnover": [7.0],
                "amount": [80000.0],
            }
        )
        with (
            mock.patch.object(
                akshare,
                "stock_zh_a_hist",
                side_effect=ConnectionError("failed"),
            ),
            mock.patch.object(
                akshare, "stock_zh_a_hist_tx", return_value=expected_df
            ),
        ):
            df = ak.query(symbols, START_DATE, END_DATE, "1d")
        assert df.shape[0] == expected_df.shape[0]
        assert list(df.columns).count("volume") == 1
        assert set(df.columns) == {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        }
        assert df["volume"].iloc[0] == 6.0

    @pytest.mark.usefixtures("setup_ds_cache")
    def test_query_when_unsupported_timeframe_then_raises(self):
        """An hourly backtest against a daily-only source is a
        misconfiguration, not an empty market -- silently returning zero bars
        produced a clean empty backtest with no diagnostic, where YQuery
        raises for the identical condition."""
        ak = AKShare()
        with mock.patch("akshare.stock_zh_a_hist") as fetch:
            with pytest.raises(ValueError, match="Unsupported timeframe"):
                ak.query(["A"], START_DATE, END_DATE, "2d")
        assert not fetch.called

    @pytest.mark.usefixtures("setup_ds_cache")
    def test_query_when_akshare_not_installed_then_raises(self):
        ak = AKShare()
        with mock.patch.dict(sys.modules, {"akshare": None}):
            with pytest.raises(ImportError, match="akshare>=1.17.50"):
                ak.query(["A"], START_DATE, END_DATE)


class TestYQuery:
    @pytest.mark.usefixtures("setup_ds_cache")
    @pytest.mark.parametrize("timeframe", [None, "", "1h", "1d", "5d", "1w"])
    def test_query(self, timeframe):
        yq = YQuery()
        symbols = ["A"]
        expected_df = pd.DataFrame(
            {
                "date": [END_DATE],
                "open": [1],
                "high": [2],
                "low": [3],
                "close": [4],
                "volume": [5],
                "symbol": symbols,
            }
        )
        with mock.patch.object(Ticker, "history", return_value=expected_df):
            df = yq.query(symbols, START_DATE, END_DATE, timeframe)
        assert set(df.columns) == {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        }
        assert df.shape[0] == expected_df.shape[0]
        assert set(df["symbol"].unique()) == set(symbols)
        assert (df["date"].unique() == expected_df["date"].unique()).all()

    @pytest.mark.parametrize(
        "columns",
        [
            [],
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
            ],
        ],
    )
    @pytest.mark.usefixtures("setup_ds_cache")
    def test_query_when_empty_result(self, columns):
        yq = YQuery()
        with mock.patch.object(
            Ticker, "history", return_value=pd.DataFrame(columns=columns)
        ):
            df = yq.query(["A"], START_DATE, END_DATE)
        assert df.empty
        assert set(df.columns) == set(
            (
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
            )
        )

    @pytest.mark.usefixtures("setup_ds_cache")
    def test_query_when_unsupported_timeframe_then_error(self):
        yq = YQuery()
        symbols = ["A"]
        expected_df = pd.DataFrame(
            {
                "date": [END_DATE],
                "open": [1],
                "high": [2],
                "low": [3],
                "close": [4],
                "volume": [5],
                "symbol": symbols,
            }
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unsupported timeframe: '90min'.\n"
                "Supported timeframes: ['', '1hour', '1day', '5day', '1week']."
            ),
        ):
            with mock.patch.object(
                Ticker, "history", return_value=expected_df
            ):
                yq.query(symbols, START_DATE, END_DATE, "90m")


class TestExtDataEdgeCases:
    """Empty results and mixed schemas from the extension data sources."""

    @pytest.mark.usefixtures("setup_enabled_ds_cache")
    def test_akshare_empty_result_keeps_canonical_columns(self):
        """A legitimately empty response must not leak un-renamed columns.

        The early empty-return fired before the rename, so an empty EM frame
        escaped with 日期/开盘/... still attached and failed the
        required-column check downstream.
        """
        empty_em = pd.DataFrame(
            columns=["日期", "开盘", "收盘", "最高", "最低", "成交量"]
        )
        ak = AKShare()
        with mock.patch("akshare.stock_zh_a_hist", return_value=empty_em):
            df = ak.query(["A"], START_DATE, END_DATE, "1d")
        assert df.empty
        assert {"date", "symbol", "open", "high", "low", "close"} <= set(
            df.columns
        )

    @pytest.mark.usefixtures("setup_enabled_ds_cache")
    def test_akshare_mixed_em_and_tx_schemas_across_symbols(self):
        """The EM and TX fallback schemas must mix cleanly in one query.

        Renaming the concatenated union mapped 日期 and ``date`` onto the
        same label, producing duplicate columns and "cannot assemble with
        duplicate keys" -- on exactly the partial-outage path the TX fallback
        exists to serve.
        """
        em_frame = pd.DataFrame(
            {
                "日期": [END_DATE],
                "开盘": [1.0],
                "收盘": [2.0],
                "最高": [3.0],
                "最低": [0.5],
                "成交量": [100.0],
            }
        )
        tx_frame = pd.DataFrame(
            {
                "date": [END_DATE],
                "open": [10.0],
                "close": [20.0],
                "high": [30.0],
                "low": [5.0],
                "amount": [200.0],
            }
        )

        def em_fetch(symbol, **_kwargs):
            if symbol == "000002":
                raise ConnectionError("EM down for this symbol")
            return em_frame.copy()

        ak = AKShare()
        with mock.patch("akshare.stock_zh_a_hist", side_effect=em_fetch):
            with mock.patch(
                "akshare.stock_zh_a_hist_tx", return_value=tx_frame.copy()
            ):
                df = ak.query(["000001", "000002"], START_DATE, END_DATE)
        assert len(df) == 2
        assert set(df["symbol"]) == {"000001", "000002"}
        # One column per name -- no duplicate labels from the double rename.
        assert not df.columns.duplicated().any()

    @pytest.mark.usefixtures("setup_enabled_ds_cache")
    def test_yquery_dict_failure_raises_clear_error(self):
        """yahooquery returns a dict of error strings when every request
        fails; reaching for .columns on it raised a bare AttributeError."""
        yq = YQuery()
        failure = {"SPY": "Data doesn't exist for startDate=..."}
        ticker = mock.MagicMock()
        ticker.history.return_value = failure
        with mock.patch("yahooquery.Ticker", return_value=ticker):
            with pytest.raises(ValueError, match="yahooquery returned"):
                yq.query(["SPY"], START_DATE, END_DATE, "1d")

    @pytest.mark.usefixtures("setup_enabled_ds_cache")
    def test_yquery_empty_result_keeps_canonical_columns(self):
        """An empty result must not keep symbol/date as MultiIndex levels."""
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.MultiIndex.from_arrays(
                [[], []], names=["symbol", "date"]
            ),
        )
        yq = YQuery()
        ticker = mock.MagicMock()
        ticker.history.return_value = empty
        with mock.patch("yahooquery.Ticker", return_value=ticker):
            df = yq.query(["SPY"], START_DATE, END_DATE, "1d")
        assert df.empty
        assert {"symbol", "date"} <= set(df.columns)


@pytest.mark.usefixtures("setup_enabled_ds_cache")
def test_cache_invalidation_retry_keeps_adjust():
    """The clear-and-retry must re-fetch with the caller's ``adjust``.

    Dropping it re-fetched UNADJUSTED data for a request that asked for
    adjusted prices and cached it under the adjust=None key -- silently,
    since the frame is otherwise well-formed.
    """
    fetch_calls = []

    class StubSource(DataSource):
        def __init__(self):
            super().__init__()
            self.extra_col = False

        def _fetch_data(
            self, symbols, start_date, end_date, timeframe, adjust
        ):
            fetch_calls.append((sorted(symbols), adjust))
            dates = pd.date_range(start_date, periods=3)
            frames = []
            for sym in sorted(symbols):
                df = pd.DataFrame(
                    {
                        "symbol": sym,
                        "date": dates,
                        "open": 1.0,
                        "high": 2.0,
                        "low": 0.5,
                        # An adjust-honoring source returns different prices.
                        "close": 100.0 if adjust is not None else 999.0,
                        "volume": 10.0,
                    }
                )
                if self.extra_col:
                    df["vwap"] = 1.0
                frames.append(df)
            return pd.concat(frames, ignore_index=True)

    source = StubSource()
    df1 = source.query(["X"], START_DATE, END_DATE, "1d", adjust="all")
    assert set(df1["close"]) == {100.0}
    # A column-set change between cached and fresh frames triggers the
    # clear-and-retry path.
    source.extra_col = True
    df2 = source.query(["X", "Z"], START_DATE, END_DATE, "1d", adjust="all")
    # The retry re-fetched with the caller's adjust, not the default.
    assert all(adj == "all" for _, adj in fetch_calls), fetch_calls
    assert set(df2["close"]) == {100.0}
