"""Unit tests for data.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import akshare
import json
import os
import pandas as pd
import pytest
import re
import yfinance
from .fixtures import *  # noqa: F401
from datetime import datetime
from urllib.error import HTTPError
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
from pybroker.ext.data import AdanosSentiment
from pybroker.ext.data import YQuery
from pybroker.ext import data as ext_data
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


class StaticDataSource(DataSource):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def _fetch_data(self, symbols, start_date, end_date, _timeframe, _adjust):
        df = self.df[self.df["symbol"].isin(symbols)].copy()
        return df[(df["date"] >= start_date) & (df["date"] <= end_date)]


class JsonResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.data).encode("utf-8")


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
            )
            cache_key, sym_df = mock_cache.set.call_args_list[i].args
            assert cache_key == repr(expected_cache_key)
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
            )
            cache_key = mock_cache.get.call_args_list[i].args[0]
            assert cache_key == repr(expected_cache_key)

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
        with mock.patch.object(
            akshare, "stock_zh_a_hist", return_value=expected_df
        ):
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
        with mock.patch.object(
            akshare,
            "stock_zh_a_hist",
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
    def test_query_when_unsupported_timeframe_then_empty(self):
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
        with mock.patch.object(
            akshare, "stock_zh_a_hist", return_value=expected_df
        ):
            df = ak.query(symbols, START_DATE, END_DATE, "2d")
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


class TestAdanosSentiment:
    @pytest.mark.usefixtures("scope", "setup_ds_cache")
    def test_query_enriches_wrapped_data_source(self):
        prices = pd.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "open": [1, 2, 3, 4],
                "high": [2, 3, 4, 5],
                "low": [0, 1, 2, 3],
                "close": [1.5, 2.5, 3.5, 4.5],
                "volume": [10, 20, 30, 40],
            }
        )

        def fetcher(source, symbol, days, **_kwargs):
            assert source in {"reddit", "news"}
            assert days == 2
            sentiment = 0.2 if symbol == "AAPL" else -0.1
            if source == "news":
                sentiment += 0.2
            return {
                "sentiment_score": 0.99,
                "buzz_score": 99,
                "mentions": 99,
                "daily_trend": [
                    {
                        "date": "2024-01-02",
                        "sentiment_score": sentiment,
                        "buzz_score": 10,
                        "mentions": 2,
                    },
                    {
                        "date": "2024-01-03",
                        "sentiment_score": sentiment + 0.1,
                        "buzz_score": 12,
                        "mentions": 3,
                    },
                ],
            }

        with mock.patch.object(ext_data, "_fetch_adanos_json", fetcher):
            adanos = AdanosSentiment(
                StaticDataSource(prices),
                api_key="test-key",
                sources=("reddit", "news"),
            )
            df = adanos.query(["AAPL", "MSFT"], "2024-01-02", "2024-01-03")

        assert {
            "adanos_sentiment",
            "adanos_buzz",
            "adanos_mentions",
            "adanos_reddit_sentiment",
            "adanos_reddit_buzz",
            "adanos_reddit_mentions",
            "adanos_news_sentiment",
            "adanos_news_buzz",
            "adanos_news_mentions",
        }.issubset(df.columns)
        aapl = df[
            (df["symbol"] == "AAPL")
            & (df["date"] == pd.Timestamp("2024-01-02"))
        ].iloc[0]
        assert aapl["adanos_reddit_sentiment"] == pytest.approx(0.2)
        assert aapl["adanos_news_sentiment"] == pytest.approx(0.4)
        assert aapl["adanos_sentiment"] == pytest.approx(0.3)
        assert aapl["adanos_buzz"] == pytest.approx(10)
        assert aapl["adanos_mentions"] == pytest.approx(4)
        aapl_end = df[
            (df["symbol"] == "AAPL")
            & (df["date"] == pd.Timestamp("2024-01-03"))
        ].iloc[0]
        assert aapl_end["adanos_reddit_sentiment"] == pytest.approx(0.3)
        assert aapl_end["adanos_news_sentiment"] == pytest.approx(0.5)
        assert aapl_end["adanos_sentiment"] == pytest.approx(0.4)
        assert aapl_end["adanos_buzz"] == pytest.approx(12)
        assert aapl_end["adanos_mentions"] == pytest.approx(6)

    @pytest.mark.usefixtures("scope")
    def test_query_joins_timezone_aware_dates(self):
        prices = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02", tz="US/Eastern")],
                "symbol": ["AAPL"],
                "open": [1],
                "high": [2],
                "low": [0],
                "close": [1.5],
                "volume": [10],
            }
        )

        def fetcher(_source, _symbol, _days, **_kwargs):
            return {
                "daily_trend": [
                    {
                        "date": "2024-01-02",
                        "sentiment_score": 0.25,
                        "buzz_score": 11,
                        "mentions": 2,
                    }
                ]
            }

        with mock.patch.object(ext_data, "_fetch_adanos_json", fetcher):
            adanos = AdanosSentiment(
                StaticDataSource(prices),
                api_key="test-key",
                sources=("reddit",),
            )
            df = adanos.query(
                "AAPL",
                "2024-01-02T00:00:00-05:00",
                "2024-01-02T00:00:00-05:00",
            )

        assert df["adanos_reddit_sentiment"].iloc[0] == pytest.approx(0.25)

    @pytest.mark.usefixtures("scope")
    def test_query_uses_top_level_payload_without_daily_trend(self):
        prices = pd.DataFrame(
            {
                "date": [datetime(2024, 1, 3)],
                "symbol": ["AAPL"],
                "open": [1],
                "high": [2],
                "low": [0],
                "close": [1.5],
                "volume": [10],
            }
        )

        def fetcher(_source, _symbol, _days, **_kwargs):
            return {
                "sentiment_score": 0.45,
                "buzz_score": 22,
                "mentions": 7,
            }

        with mock.patch.object(ext_data, "_fetch_adanos_json", fetcher):
            adanos = AdanosSentiment(
                StaticDataSource(prices),
                api_key="test-key",
                sources=("reddit",),
            )
            df = adanos.query("AAPL", "2024-01-03", "2024-01-03")

        assert df["adanos_reddit_sentiment"].iloc[0] == pytest.approx(0.45)
        assert df["adanos_reddit_buzz"].iloc[0] == pytest.approx(22)
        assert df["adanos_reddit_mentions"].iloc[0] == pytest.approx(7)

    @pytest.mark.usefixtures("scope")
    def test_query_when_aggregate_disabled(self):
        prices = pd.DataFrame(
            {
                "date": [datetime(2024, 1, 2)],
                "symbol": ["AAPL"],
                "open": [1],
                "high": [2],
                "low": [0],
                "close": [1.5],
                "volume": [10],
            }
        )

        def fetcher(_source, _symbol, _days, **_kwargs):
            return {
                "daily_trend": [
                    {
                        "date": "2024-01-02",
                        "sentiment_score": 0.25,
                        "buzz_score": 11,
                        "mentions": 2,
                    }
                ]
            }

        with mock.patch.object(ext_data, "_fetch_adanos_json", fetcher):
            adanos = AdanosSentiment(
                StaticDataSource(prices),
                api_key="test-key",
                sources=("reddit",),
                include_aggregate=False,
            )
            df = adanos.query("AAPL", "2024-01-02", "2024-01-02")

        assert "adanos_sentiment" not in df.columns
        assert df["adanos_reddit_sentiment"].iloc[0] == pytest.approx(0.25)

    @pytest.mark.usefixtures("scope")
    def test_query_when_source_has_no_data(self):
        prices = pd.DataFrame(
            {
                "date": [datetime(2024, 1, 2)],
                "symbol": ["AAPL"],
                "open": [1],
                "high": [2],
                "low": [0],
                "close": [1.5],
                "volume": [10],
            }
        )
        with mock.patch.object(
            ext_data, "_fetch_adanos_json", lambda *_a, **_k: {}
        ):
            adanos = AdanosSentiment(
                StaticDataSource(prices),
                api_key="test-key",
                sources=("reddit",),
            )
            df = adanos.query("AAPL", "2024-01-02", "2024-01-02")

        assert pd.isna(df["adanos_sentiment"].iloc[0])
        assert pd.isna(df["adanos_reddit_sentiment"].iloc[0])

    @pytest.mark.usefixtures("scope", "setup_enabled_ds_cache")
    def test_query_uses_separate_cache_key_from_wrapped_source(self):
        prices = pd.DataFrame(
            {
                "date": [datetime(2024, 1, 2)],
                "symbol": ["AAPL"],
                "open": [1],
                "high": [2],
                "low": [0],
                "close": [1.5],
                "volume": [10],
            }
        )
        data_source = StaticDataSource(prices)
        data_source.query("AAPL", "2024-01-02", "2024-01-02")

        def fetcher(_source, _symbol, _days, **_kwargs):
            return {
                "daily_trend": [
                    {
                        "date": "2024-01-02",
                        "sentiment_score": 0.4,
                        "buzz_score": 20,
                        "mentions": 5,
                    }
                ]
            }

        with mock.patch.object(ext_data, "_fetch_adanos_json", fetcher):
            adanos = AdanosSentiment(
                data_source,
                api_key="test-key",
                sources=("reddit",),
            )
            df = adanos.query("AAPL", "2024-01-02", "2024-01-02")

        assert df["adanos_reddit_sentiment"].iloc[0] == pytest.approx(0.4)

    @pytest.mark.usefixtures("scope")
    def test_init_when_unsupported_source_then_error(self):
        with pytest.raises(
            ValueError,
            match=re.escape("Unsupported Adanos source: 'crypto'."),
        ):
            AdanosSentiment(
                StaticDataSource(pd.DataFrame()),
                api_key="test-key",
                sources=("crypto",),
            )

    def test_fetch_adanos_json_uses_source_endpoint_and_api_key(self):
        def urlopen(request, timeout):
            assert timeout == 3
            assert request.full_url == (
                "https://example.com/reddit/stocks/v1/stock/AAPL?days=7"
            )
            assert request.get_header("X-api-key") == "test-key"
            return JsonResponse({"sentiment_score": 0.5})

        with mock.patch.object(ext_data, "urlopen", urlopen):
            payload = ext_data._fetch_adanos_json(
                "reddit",
                "AAPL",
                7,
                api_key="test-key",
                base_url="https://example.com/",
                timeout=3,
            )

        assert payload == {"sentiment_score": 0.5}

    def test_fetch_adanos_json_when_404_returns_empty_payload(self):
        def urlopen(_request, timeout):
            assert timeout == 3
            raise HTTPError("url", 404, "Not Found", {}, None)

        with mock.patch.object(ext_data, "urlopen", urlopen):
            payload = ext_data._fetch_adanos_json(
                "reddit",
                "MISSING",
                7,
                api_key="test-key",
                base_url="https://example.com",
                timeout=3,
            )

        assert payload == {}
