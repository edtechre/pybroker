"""Unit tests for data.py module."""

"""Copyright (C) 2023 Edward West

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .fixtures import *  # noqa: F401
from datetime import datetime
from pybroker.cache import DataSourceCacheKey
from pybroker.common import to_seconds
from pybroker.data import Alpaca, AlpacaCrypto, DataSourceCacheMixin, YFinance
from unittest import mock
import joblib
import os
import pandas as pd
import pytest
import re
import yfinance

API_KEY = "api_key"
API_SECRET = "api_secret"
API_VERSION = "v2"
TIMEFRAME = "1m"
START_DATE = datetime.strptime("2021-02-02", "%Y-%m-%d")
END_DATE = datetime.strptime("2022-02-02", "%Y-%m-%d")
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
ALPACA_CRYPTO_COLS = ALPACA_COLS + ["exchange", "trade_count"]
EXCHANGE = "CBSE"


@pytest.fixture()
def alpaca_df():
    df = joblib.load(
        os.path.join(os.path.dirname(__file__), "testdata/daily_1.joblib")
    )
    df["date"] = df["date"].dt.tz_localize("US/Eastern")
    return df.assign(vwap=1)[ALPACA_COLS]


@pytest.fixture()
def alpaca_crypto_df():
    df = joblib.load(
        os.path.join(os.path.dirname(__file__), "testdata/daily_1.joblib")
    )
    df["date"] = df["date"].dt.tz_localize("US/Eastern")
    return df.assign(vwap=1, trade_count=1, exchange=EXCHANGE)[
        ALPACA_CRYPTO_COLS
    ]


@pytest.fixture()
def bars_df(alpaca_df):
    return alpaca_df.rename(columns={"date": "timestamp"})


@pytest.fixture()
def crypto_bars_df(alpaca_crypto_df):
    return alpaca_crypto_df.rename(columns={"date": "timestamp"})


@pytest.fixture()
def yfinance_df():
    return joblib.load(
        os.path.join(os.path.dirname(__file__), "testdata/yfinance.joblib")
    )


@pytest.fixture()
def yfinance_single_df():
    return joblib.load(
        os.path.join(
            os.path.dirname(__file__), "testdata/yfinance_single.joblib"
        )
    )


@pytest.fixture()
def symbols(alpaca_df):
    return list(alpaca_df["symbol"].unique())


@pytest.fixture()
def mock_cache(scope):
    with mock.patch.object(
        scope, "data_source_cache"
    ) as cache, mock.patch.object(cache, "get", return_value=None):
        yield cache


@pytest.fixture()
def mock_alpaca():
    with mock.patch("alpaca_trade_api.REST") as rest:
        yield rest


class TestDataSourceCacheMixin:
    @pytest.mark.usefixtures("scope")
    def test_set_cached(self, alpaca_df, symbols, mock_cache):
        cache_mixin = DataSourceCacheMixin()
        cache_mixin.set_cached(TIMEFRAME, START_DATE, END_DATE, alpaca_df)
        assert len(mock_cache.set.call_args_list) == len(symbols)
        for i, sym in enumerate(symbols):
            expected_cache_key = DataSourceCacheKey(
                symbol=sym,
                tf_seconds=to_seconds(TIMEFRAME),
                start_date=START_DATE,
                end_date=END_DATE,
            )
            cache_key, sym_df = mock_cache.set.call_args_list[i].args
            assert cache_key == repr(expected_cache_key)
            assert sym_df.equals(alpaca_df[alpaca_df["symbol"] == sym])

    @pytest.mark.usefixtures("scope")
    @pytest.mark.parametrize(
        "query_symbols", [[], pytest.lazy_fixture("symbols")]
    )
    def test_get_cached_when_empty(self, mock_cache, query_symbols):
        cache_mixin = DataSourceCacheMixin()
        df, uncached_syms = cache_mixin.get_cached(
            query_symbols, TIMEFRAME, START_DATE, END_DATE
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
            )
            cache_key = mock_cache.get.call_args_list[i].args[0]
            assert cache_key == repr(expected_cache_key)

    @pytest.mark.usefixtures("setup_enabled_ds_cache")
    def test_set_and_get_cached(self, alpaca_df, symbols):
        cache_mixin = DataSourceCacheMixin()
        cache_mixin.set_cached(TIMEFRAME, START_DATE, END_DATE, alpaca_df)
        df, uncached_syms = cache_mixin.get_cached(
            symbols, TIMEFRAME, START_DATE, END_DATE
        )
        assert df.equals(alpaca_df)
        assert not len(uncached_syms)

    @pytest.mark.usefixtures("setup_enabled_ds_cache")
    def test_set_and_get_cached_when_partial(self, alpaca_df, symbols):
        cache_mixin = DataSourceCacheMixin()
        cached_df = alpaca_df[alpaca_df["symbol"].isin(symbols[:2])]
        cache_mixin.set_cached(TIMEFRAME, START_DATE, END_DATE, cached_df)
        df, uncached_syms = cache_mixin.get_cached(
            symbols, TIMEFRAME, START_DATE, END_DATE
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
            cache_mixin.set_cached(timeframe, start_date, end_date, alpaca_df)
        with pytest.raises(error):
            cache_mixin.get_cached(symbols, timeframe, start_date, end_date)

    def test_set_and_get_cached_when_cache_disabled(self, alpaca_df, symbols):
        cache_mixin = DataSourceCacheMixin()
        cache_mixin.set_cached(TIMEFRAME, START_DATE, END_DATE, alpaca_df)
        df, uncached_syms = cache_mixin.get_cached(
            symbols, TIMEFRAME, START_DATE, END_DATE
        )
        assert df.empty
        assert uncached_syms == symbols


class TestAlpaca:
    def test_init(self, mock_alpaca):
        Alpaca(API_KEY, API_SECRET)
        mock_alpaca.assert_called_once_with(
            API_KEY, API_SECRET, api_version=API_VERSION
        )

    @pytest.mark.usefixtures("setup_ds_cache", "mock_alpaca")
    def test_query_when_empty_cache(self, alpaca_df, bars_df, symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        mock_bars = mock.Mock()
        mock_bars.df = bars_df
        with mock.patch.object(
            alpaca._api, "get_bars", return_value=mock_bars
        ):
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

    @pytest.mark.usefixtures(
        "setup_enabled_ds_cache", "mock_alpaca", "tmp_path"
    )
    def test_query_when_partial_cache(self, alpaca_df, bars_df, symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        cached_df = alpaca_df[alpaca_df["symbol"].isin(symbols[-1:])]
        alpaca.set_cached(TIMEFRAME, START_DATE, END_DATE, cached_df)
        mock_bars = mock.Mock()
        mock_bars.df = bars_df[bars_df["symbol"].isin(symbols[:-1])]
        with mock.patch.object(
            alpaca._api, "get_bars", return_value=mock_bars
        ):
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

    @pytest.mark.usefixtures(
        "setup_enabled_ds_cache", "mock_alpaca", "tmp_path"
    )
    def test_query_when_cache_mismatch(self, alpaca_df, bars_df, symbols):
        alpaca = Alpaca(API_KEY, API_SECRET)
        cached_df = alpaca_df[alpaca_df["symbol"].isin(symbols[-1:])]
        cached_df = cached_df.drop(columns=["vwap"])
        alpaca.set_cached(TIMEFRAME, START_DATE, END_DATE, cached_df)
        mock_bars = mock.Mock()
        mock_bars.df = bars_df[bars_df["symbol"].isin(symbols[:-1])]
        with mock.patch.object(
            alpaca._api, "get_bars", return_value=mock_bars
        ):
            df = alpaca.query(symbols, START_DATE, END_DATE, TIMEFRAME)
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
            alpaca._api, "get_bars", return_value=mock_bars
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
            alpaca._api, "get_bars", return_value=mock_bars
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


class TestAlpacaCrypto:
    def test_init(self, mock_alpaca):
        AlpacaCrypto(API_KEY, API_SECRET, EXCHANGE)
        mock_alpaca.assert_called_once_with(
            API_KEY, API_SECRET, api_version=API_VERSION
        )

    @pytest.mark.usefixtures("setup_ds_cache", "mock_alpaca")
    def test_query(self, alpaca_crypto_df, crypto_bars_df, symbols):
        crypto = AlpacaCrypto(API_KEY, API_SECRET, EXCHANGE)
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
                alpaca_crypto_df.drop(columns=["exchange"])
                .sort_values(["symbol", "date"])
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
        crypto = AlpacaCrypto(API_KEY, API_SECRET, EXCHANGE)
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


class TestYFinance:
    @pytest.mark.parametrize(
        "param_symbols, expected_df, expected_rows",
        [
            (
                pytest.lazy_fixture("symbols"),
                pytest.lazy_fixture("yfinance_df"),
                2020,
            ),
            (["SPY"], pytest.lazy_fixture("yfinance_single_df"), 505),
        ],
    )
    @pytest.mark.usefixtures("setup_ds_cache")
    def test_query(self, param_symbols, expected_df, expected_rows):
        yf = YFinance()
        with mock.patch.object(yfinance, "download", return_value=expected_df):
            df = yf.query(param_symbols, START_DATE, END_DATE)
        assert set(df.columns) == {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
            "symbol",
        }
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
    def test_query_when_empty_result(self, symbols, columns):
        yf = YFinance()
        with mock.patch.object(
            yfinance, "download", return_value=pd.DataFrame(columns=columns)
        ):
            df = yf.query(symbols, START_DATE, END_DATE)
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
                "adj_close",
            )
        )
