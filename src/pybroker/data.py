r"""Contains :class:`.DataSource`\ s used to fetch external data."""

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

from .cache import DataSourceCacheKey
from .common import (
    DataCol,
    parse_timeframe,
    to_datetime,
    to_seconds,
    verify_data_source_columns,
)
from .scope import StaticScope
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Final, Iterable, Optional, Union
import alpaca_trade_api as tradeapi
import itertools
import numpy as np
import pandas as pd
import yfinance


class DataSourceCacheMixin:
    """Mixin that implements fetching and storing cached :class:`.DataSource`
    data.
    """

    def get_cached(
        self,
        symbols: Iterable[str],
        timeframe: str,
        start_date: Union[str, datetime, pd.Timestamp, np.datetime64],
        end_date: Union[str, datetime, pd.Timestamp, np.datetime64],
    ) -> tuple[pd.DataFrame, Iterable[str]]:
        """Retrieves cached data from disk when caching is enabled with
        :meth:`pybroker.cache.enable_data_source_cache`.

        Args:
            symbols: :class:`Iterable` of symbols for fetching cached data.
            timeframe: Formatted string that specifies the timeframe
                resolution of the cached data. The timeframe string supports
                the following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks

                An example timeframe string is ``1h 30m``.
            start_date: Starting date of the cached data (inclusive).
            end_date: Ending date of the cached data (inclusive).

        Returns:
            ``tuple[pandas.DataFrame, Iterable[str]]`` containing a
            :class:`pandas.DataFrame` with the cached data, and an
            ``Iterable[str]`` of symbols for which no cached data was
            found.
        """
        df = pd.DataFrame()
        scope = StaticScope.instance()
        cache = scope.data_source_cache
        if cache is None:
            return df, symbols
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        tf_seconds = to_seconds(timeframe)
        uncached_syms = []
        cached_syms = []
        for sym in symbols:
            cache_key = DataSourceCacheKey(
                symbol=sym,
                tf_seconds=tf_seconds,
                start_date=start_date,
                end_date=end_date,
            )
            cached = cache.get(repr(cache_key))
            scope.logger.debug_get_data_source_cache(cache_key)
            if cached is None:
                uncached_syms.append(sym)
            else:
                cached_syms.append(sym)
                df = pd.concat([df, cached])
        if not uncached_syms:
            scope.logger.loaded_bar_data()
        scope.logger.info_loaded_bar_data(
            symbols=cached_syms,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        return df, uncached_syms

    def set_cached(
        self,
        timeframe: str,
        start_date: Union[str, datetime, pd.Timestamp, np.datetime64],
        end_date: Union[str, datetime, pd.Timestamp, np.datetime64],
        data: pd.DataFrame,
    ):
        """Stores data to disk cache when caching is enabled with
        :meth:`pybroker.cache.enable_data_source_cache`.

        Args:
            timeframe: Formatted string that specifies the timeframe
                resolution of the data to cache. The timeframe string supports
                the following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks

                An example timeframe string would be ``1h 30m``.
            start_date: Starting date of the data to cache (inclusive).
            end_date: Ending date of the data to cache (inclusive).
            data: :class:`pandas.DataFrame` containing the data to cache.
        """
        if data.empty:
            return
        scope = StaticScope.instance()
        cache = scope.data_source_cache
        if cache is None:
            return
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        tf_seconds = to_seconds(timeframe)
        for sym in data[DataCol.SYMBOL.value].unique():
            df = data[data[DataCol.SYMBOL.value] == sym]
            cache_key = DataSourceCacheKey(
                symbol=sym,
                tf_seconds=tf_seconds,
                start_date=start_date,
                end_date=end_date,
            )
            cache.set(repr(cache_key), df)
            scope.logger.debug_set_data_source_cache(cache_key)


class DataSource(ABC, DataSourceCacheMixin):
    """Base class for querying data from an external source. Extend this class
    and override :meth:`._fetch_data` to implement a custom
    :class:`.DataSource` that can be used with
    :class:`pybroker.strategy.Strategy`.
    """

    def __init__(self):
        self._scope = StaticScope.instance()
        self._logger = self._scope.logger

    def query(
        self,
        symbols: Union[str, Iterable[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: Optional[str] = "",
    ) -> pd.DataFrame:
        """Queries data. Cached data is returned if caching is enabled by
        calling :meth:`pybroker.cache.enable_data_source_cache`.

        Args:
            symbols: Symbols of the data to query.
            start_date: Start date of the data to query (inclusive).
            end_date: End date of the data to query (inclusive).
            timeframe: Formatted string that specifies the timeframe
                resolution to query. The timeframe string supports the
                following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks

                An example timeframe string is ``1h 30m``.

        Returns:
            :class:`pandas.DataFrame` containing the queried data.
        """
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        if type(symbols) == str and not symbols:
            raise ValueError("Symbols cannot be empty.")
        unique_syms = (
            frozenset((symbols,))
            if type(symbols) == str
            else frozenset(symbols)
        )
        if not unique_syms:
            raise ValueError("Symbols cannot be empty.")
        timeframe = self._format_timeframe(timeframe)
        cached_df, uncached_syms = self.get_cached(
            symbols=unique_syms,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        if not uncached_syms:
            return cached_df
        self._logger.download_bar_data_start()
        self._logger.info_download_bar_data_start(
            symbols=uncached_syms,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        df = self._fetch_data(unique_syms, start_date, end_date, timeframe)
        if (
            self._scope.data_source_cache is not None
            and not cached_df.columns.empty
            and set(cached_df.columns) != set(df.columns)
        ):
            self._logger.info_invalidate_data_source_cache()
            self._scope.data_source_cache.clear()
            return self.query(symbols, start_date, end_date, timeframe)
        verify_data_source_columns(df)
        self.set_cached(timeframe, start_date, end_date, df)
        df = pd.concat((cached_df, df))
        if not df.empty:
            df = df.sort_values(by=[DataCol.DATE.value, DataCol.SYMBOL.value])
        self._logger.download_bar_data_completed()
        return df.reset_index(drop=True)

    @abstractmethod
    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str],
    ) -> pd.DataFrame:
        """:meta public:
        Override this method to return data from a custom
        source. The returned :class:`pandas.DataFrame` must contain the
        following columns: ``symbol``, ``date``, ``open``, ``high``, ``low``,
        and ``close``.

        Args:
            symbols: Ticker symbols of the data to query.
            start_date: Start date of the data to query (inclusive).
            end_date: End date of the data to query (inclusive).
            timeframe: Formatted string that specifies the timeframe
                resolution to query. The timeframe string supports the
                following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks

                An example timeframe string is ``1h 30m``.

        Returns:
            :class:`pandas.DataFrame` containing the queried data.
        """

    def _format_timeframe(self, timeframe: Optional[str]) -> str:
        if not timeframe:
            return ""
        return " ".join(
            f"{part[0]}{part[1]}" for part in parse_timeframe(timeframe)
        )


class Alpaca(DataSource):
    """Retrieves stock data from `Alpaca <https://alpaca.markets/>`_."""

    __NY: Final = "America/New_York"
    __EST: Final = "US/Eastern"
    __API_VERSION: Final = "v2"

    def __init__(self, api_key: str, api_secret: str):
        super().__init__()
        self._api = tradeapi.REST(
            api_key, api_secret, api_version=self.__API_VERSION
        )

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: Optional[str],
    ) -> pd.DataFrame:
        """:meta private:"""
        start = pd.Timestamp(start_date, tz=self.__NY).isoformat()
        end = pd.Timestamp(end_date, tz=self.__NY).isoformat()
        df = self._api.get_bars(symbols, timeframe, start, end).df
        if df.columns.empty:
            return pd.DataFrame(
                columns=[
                    DataCol.SYMBOL.value,
                    DataCol.DATE.value,
                    DataCol.OPEN.value,
                    DataCol.HIGH.value,
                    DataCol.LOW.value,
                    DataCol.CLOSE.value,
                    DataCol.VOLUME.value,
                    DataCol.VWAP.value,
                ]
            )
        if df.empty:
            return df
        df = df.reset_index()
        df.rename(columns={"timestamp": DataCol.DATE.value}, inplace=True)
        df = df[[col.value for col in DataCol]]
        df[DataCol.DATE.value] = pd.to_datetime(df[DataCol.DATE.value])
        df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(
            self.__EST
        )
        return df


class AlpacaCrypto(DataSource):
    """Retrieves crypto data from `Alpaca <https://alpaca.markets/>`_.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        exchange: Cryptocurrency exchange to use for data.
    """

    TRADE_COUNT: Final = "trade_count"
    COLUMNS: Final = (
        DataCol.SYMBOL.value,
        DataCol.DATE.value,
        DataCol.OPEN.value,
        DataCol.HIGH.value,
        DataCol.LOW.value,
        DataCol.CLOSE.value,
        DataCol.VOLUME.value,
        DataCol.VWAP.value,
        TRADE_COUNT,
    )

    __NY: Final = "America/New_York"
    __EST: Final = "US/Eastern"
    __API_VERSION: Final = "v2"

    def __init__(self, api_key: str, api_secret: str, exchange: str):
        super().__init__()
        self._scope.register_custom_cols(self.TRADE_COUNT)
        self._api = tradeapi.REST(
            api_key, api_secret, api_version=self.__API_VERSION
        )
        self.exchange = exchange

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: Optional[str],
    ) -> pd.DataFrame:
        """:meta private:"""
        start = pd.Timestamp(start_date, tz=self.__NY).isoformat()
        end = pd.Timestamp(end_date, tz=self.__NY).isoformat()
        df = self._api.get_crypto_bars(symbols, timeframe, start, end).df
        if df.columns.empty:
            return pd.DataFrame(columns=self.COLUMNS)
        if df.empty:
            return df
        df = df[df["exchange"] == self.exchange]
        df = df.reset_index()
        df.rename(columns={"timestamp": DataCol.DATE.value}, inplace=True)
        df = df[[col for col in self.COLUMNS]]
        df[DataCol.DATE.value] = pd.to_datetime(df[DataCol.DATE.value])
        df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(
            self.__EST
        )
        return df


class YFinance(DataSource):
    r"""Retrieves data from `Yahoo Finance <https://finance.yahoo.com/>`_\ .

    Attributes:
        ADJ_CLOSE: Column name of adjusted close prices.
    """

    ADJ_CLOSE: Final = "adj_close"
    __TIMEFRAME: Final = "1d"

    def __init__(self):
        super().__init__()
        self._scope.register_custom_cols(self.ADJ_CLOSE)

    def query(
        self,
        symbols: Union[str, Iterable[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        _: Optional[str] = "",
    ) -> pd.DataFrame:
        r"""Queries data from `Yahoo Finance <https://finance.yahoo.com/>`_\ .
        The timeframe of the data is limited to per day only.

        Args:
            symbols: Ticker symbols of the data to query.
            start_date: Start date of the data to query (inclusive).
            end_date: End date of the data to query (inclusive).

        Returns:
            :class:`pandas.DataFrame` containing the queried data.
        """
        return super().query(symbols, start_date, end_date, self.__TIMEFRAME)

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        _: Optional[str],
    ) -> pd.DataFrame:
        """:meta private:"""
        df = yfinance.download(list(symbols), start=start_date, end=end_date)
        if df.columns.empty:
            return pd.DataFrame(
                columns=[
                    DataCol.SYMBOL.value,
                    DataCol.DATE.value,
                    DataCol.OPEN.value,
                    DataCol.HIGH.value,
                    DataCol.LOW.value,
                    DataCol.CLOSE.value,
                    DataCol.VOLUME.value,
                    self.ADJ_CLOSE,
                ]
            )
        if df.empty:
            return df
        df = df.reset_index()
        result = pd.DataFrame()
        if len(symbols) == 1:
            result[DataCol.DATE.value] = df["Date"].values
            result[DataCol.SYMBOL.value] = tuple(
                itertools.repeat(next(iter(symbols)), len(df["Close"].values))
            )
            result[DataCol.OPEN.value] = df["Open"].values
            result[DataCol.HIGH.value] = df["High"].values
            result[DataCol.LOW.value] = df["Low"].values
            result[DataCol.CLOSE.value] = df["Close"].values
            result[DataCol.VOLUME.value] = df["Volume"].values
            result[self.ADJ_CLOSE] = df["Adj Close"].values
        else:
            df.columns = df.columns.to_flat_index()
            for sym in symbols:
                sym_df = pd.DataFrame()
                sym_df[DataCol.DATE.value] = df[("Date", "")].values
                sym_df[DataCol.SYMBOL.value] = tuple(
                    itertools.repeat(sym, len(df[("Close", sym)].values))
                )
                sym_df[DataCol.OPEN.value] = df[("Open", sym)].values
                sym_df[DataCol.HIGH.value] = df[("High", sym)].values
                sym_df[DataCol.LOW.value] = df[("Low", sym)].values
                sym_df[DataCol.CLOSE.value] = df[("Close", sym)].values
                sym_df[DataCol.VOLUME.value] = df[("Volume", sym)].values
                sym_df[self.ADJ_CLOSE] = df[("Adj Close", sym)].values
                result = pd.concat((result, sym_df))
        return result
