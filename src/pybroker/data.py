r"""Contains :class:`.DataSource`\ s used to fetch external data."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Final, Iterable, Optional, Union

import alpaca.data.historical.crypto as alpaca_crypto
import alpaca.data.historical.stock as alpaca_stock
import numpy as np
import pandas as pd
import yfinance
from alpaca.data.enums import Adjustment
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from pybroker.cache import DataSourceCacheKey
from pybroker.common import (
    DataCol,
    parse_timeframe,
    to_datetime,
    to_seconds,
    verify_data_source_columns,
    verify_date_range,
)
from pybroker.scope import StaticScope


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
        adjust: Optional[Any],
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
            adjust: The type of adjustment to make.

        Returns:
            ``tuple[pandas.DataFrame, Iterable[str]]`` containing a
            :class:`pandas.DataFrame` with the cached data, and an
            ``Iterable[str]`` of symbols for which no cached data was
            found.
        """
        scope = StaticScope.instance()
        cache = scope.data_source_cache
        if cache is None:
            return pd.DataFrame(), symbols
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        tf_seconds = to_seconds(timeframe)
        cached_frames: list[pd.DataFrame] = []
        uncached_syms = []
        cached_syms = []
        for sym in symbols:
            cache_key = DataSourceCacheKey(
                symbol=sym,
                tf_seconds=tf_seconds,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                source=f"{type(self).__module__}.{type(self).__qualname__}",
            )
            cached = cache.get(cache_key)
            scope.logger.debug_get_data_source_cache(cache_key)
            if cached is None:
                uncached_syms.append(sym)
            else:
                cached_syms.append(sym)
                cached_frames.append(cached)
        df = pd.concat(cached_frames) if cached_frames else pd.DataFrame()
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
        adjust: Optional[Any],
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
            adjust: The type of adjustment to make.
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
        for sym, sym_df in data.groupby(DataCol.SYMBOL.value, sort=False):
            cache_key = DataSourceCacheKey(
                symbol=sym,
                tf_seconds=tf_seconds,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                source=f"{type(self).__module__}.{type(self).__qualname__}",
            )
            cache.set(cache_key, sym_df)
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
        adjust: Optional[Any] = None,
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
            adjust: The type of adjustment to make.

        Returns:
            :class:`pandas.DataFrame` containing the queried data.
        """
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        verify_date_range(start_date, end_date)
        if isinstance(symbols, str) and not symbols:
            raise ValueError("Symbols cannot be empty.")
        unique_syms = (
            frozenset((symbols,))
            if isinstance(symbols, str)
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
            adjust=adjust,
        )
        if not uncached_syms:
            # Mirror the fetch path's normalization below: the cached
            # frames were concatenated iterating an unordered set, so
            # without the sort the row order (and index) would vary with
            # PYTHONHASHSEED.
            if not cached_df.empty:
                cached_df = cached_df.sort_values(
                    by=[DataCol.DATE.value, DataCol.SYMBOL.value]
                )
            return cached_df.reset_index(drop=True)
        self._logger.download_bar_data_start()
        self._logger.info_download_bar_data_start(
            symbols=uncached_syms,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        df = self._fetch_data(
            frozenset(uncached_syms), start_date, end_date, timeframe, adjust
        )
        if (
            self._scope.data_source_cache is not None
            # An empty fetch carries no schema evidence: without this
            # guard, a symbol with no data (a column-less empty frame)
            # wiped the entire cache on every query.
            and not df.empty
            and not cached_df.columns.empty
            and set(cached_df.columns) != set(df.columns)
        ):
            self._logger.info_invalidate_data_source_cache()
            self._scope.data_source_cache.clear()
            # ``adjust`` rides along: dropping it here re-fetched UNADJUSTED
            # data for a request that asked for adjusted prices, and cached
            # it under the adjust=None key -- silently, since the frame is
            # otherwise well-formed.
            return self.query(symbols, start_date, end_date, timeframe, adjust)
        if df.empty and df.columns.empty:
            # Normalize a no-data fetch to the canonical columns so
            # validation passes and the symbol simply contributes no rows.
            df = pd.DataFrame(
                columns=[
                    DataCol.SYMBOL.value,
                    DataCol.DATE.value,
                    DataCol.OPEN.value,
                    DataCol.HIGH.value,
                    DataCol.LOW.value,
                    DataCol.CLOSE.value,
                ]
            )
        verify_data_source_columns(df)
        self.set_cached(timeframe, start_date, end_date, adjust, df)
        # Concatenating an all-empty fetch would degrade the cached
        # frame's dtypes (datetime columns fall back to object).
        df = (
            cached_df
            if df.empty and not cached_df.empty
            else pd.concat((cached_df, df), ignore_index=True)
        )
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
        adjust: Optional[Any],
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
            adjust: The type of adjustment to make.

        Returns:
            :class:`pandas.DataFrame` containing the queried data.
        """

    def _format_timeframe(self, timeframe: Optional[str]) -> str:
        if not timeframe:
            return ""
        return " ".join(
            f"{part[0]}{part[1]}" for part in parse_timeframe(timeframe)
        )


def _parse_alpaca_timeframe(
    timeframe: Optional[str],
) -> tuple[int, TimeFrameUnit]:
    if timeframe is None:
        raise ValueError("Timeframe needs to be specified for Alpaca.")
    parts = parse_timeframe(timeframe)
    if len(parts) != 1:
        raise ValueError(f"Invalid Alpaca timeframe: {timeframe}")
    tf = parts[0]
    if tf[1] == "min":
        unit = TimeFrameUnit.Minute
    elif tf[1] == "hour":
        unit = TimeFrameUnit.Hour
    elif tf[1] == "day":
        unit = TimeFrameUnit.Day
    elif tf[1] == "week":
        unit = TimeFrameUnit.Week
    else:
        raise ValueError(f"Invalid Alpaca timeframe: {timeframe}")
    return tf[0], unit


def _get_alpaca_crypto_bars(
    api: alpaca_crypto.CryptoHistoricalDataClient,
    request: CryptoBarsRequest,
):
    get_crypto_bars = api.get_crypto_bars
    try:
        from alpaca.data.enums import CryptoFeed
    except ImportError:
        try:
            return get_crypto_bars(request)
        except TypeError as exc:
            raise ImportError(
                "AlpacaCrypto requires alpaca-py>=0.10.0 in the same Python "
                "environment as your notebook kernel. Upgrade with: "
                "python -m pip install 'alpaca-py>=0.10.0'"
            ) from exc

    try:
        return get_crypto_bars(request, feed=CryptoFeed.US)
    except TypeError:
        return get_crypto_bars(request)


class Alpaca(DataSource):
    """Retrieves stock data from `Alpaca <https://alpaca.markets/>`_."""

    __EST: Final = "US/Eastern"

    def __init__(self, api_key: str, api_secret: str):
        super().__init__()
        self._api = alpaca_stock.StockHistoricalDataClient(api_key, api_secret)

    def query(
        self,
        symbols: Union[str, Iterable[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: Optional[str] = "1d",
        adjust: Optional[Any] = None,
    ) -> pd.DataFrame:
        _parse_alpaca_timeframe(timeframe)
        return super().query(symbols, start_date, end_date, timeframe, adjust)

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str],
        adjust: Optional[Any],
    ) -> pd.DataFrame:
        """:meta private:"""
        amount, unit = _parse_alpaca_timeframe(timeframe)
        adj_enum = None
        if adjust is not None:
            for member in Adjustment:
                if member.value == adjust:
                    adj_enum = member
                    break
            if adj_enum is None:
                raise ValueError(f"Unknown adjustment: {adjust}.")
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            start=start_date,
            end=end_date,
            timeframe=TimeFrame(amount, unit),
            limit=None,
            adjustment=adj_enum,
            feed=None,
        )
        df = self._api.get_stock_bars(request).df  # type: ignore[union-attr]
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

    __EST: Final = "US/Eastern"

    def __init__(self, api_key: str, api_secret: str):
        super().__init__()
        self._scope.register_custom_cols(self.TRADE_COUNT)
        self._api = alpaca_crypto.CryptoHistoricalDataClient(
            api_key, api_secret
        )

    def query(
        self,
        symbols: Union[str, Iterable[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: Optional[str] = "1d",
        _adjust: Optional[str] = None,
    ) -> pd.DataFrame:
        _parse_alpaca_timeframe(timeframe)
        return super().query(symbols, start_date, end_date, timeframe, _adjust)

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str],
        _adjust: Optional[str],
    ) -> pd.DataFrame:
        """:meta private:"""
        amount, unit = _parse_alpaca_timeframe(timeframe)
        request = CryptoBarsRequest(
            symbol_or_symbols=list(symbols),
            start=start_date,
            end=end_date,
            timeframe=TimeFrame(amount, unit),
            limit=None,
        )
        df = _get_alpaca_crypto_bars(self._api, request).df
        if df.columns.empty:
            return pd.DataFrame(columns=self.COLUMNS)
        if df.empty:
            return df
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

    Args:
        auto_adjust: Whether to auto adjust close prices. If ``True``, then
            adjusted close prices are stored in the ``close`` column. Defaults
            to ``False``.

    Attributes:
        ADJ_CLOSE: Column name of adjusted close prices.
    """

    ADJ_CLOSE: Final = "adj_close"
    __TIMEFRAME: Final = "1d"

    def __init__(self, auto_adjust: bool = False):
        super().__init__()
        self.auto_adjust = auto_adjust
        self._scope.register_custom_cols(self.ADJ_CLOSE)

    def query(
        self,
        symbols: Union[str, Iterable[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        _timeframe: Optional[str] = "",
        _adjust: Optional[Any] = None,
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
        return super().query(
            symbols, start_date, end_date, self.__TIMEFRAME, _adjust
        )

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        _timeframe: Optional[str],
        _adjust: Optional[Any],
    ) -> pd.DataFrame:
        """:meta private:"""
        show_yf_progress_bar = (
            not self._logger._disabled
            and not self._logger._progress_bar_disabled
        )
        df = yfinance.download(
            list(symbols),
            start=start_date,
            end=end_date,
            progress=show_yf_progress_bar,
            auto_adjust=self.auto_adjust,
        )
        if show_yf_progress_bar:
            # yfinance's progress bar leaves its final newline unflushed on
            # stderr; flush before the caller logs to stdout, or the newline
            # surfaces as a stray stderr block in notebook output.
            sys.stderr.flush()
        if df.columns.empty:
            columns = [
                DataCol.SYMBOL.value,
                DataCol.DATE.value,
                DataCol.OPEN.value,
                DataCol.HIGH.value,
                DataCol.LOW.value,
                DataCol.CLOSE.value,
                DataCol.VOLUME.value,
            ]
            if not self.auto_adjust:
                columns.append(self.ADJ_CLOSE)
            return pd.DataFrame(columns=columns)
        if df.empty:
            return df
        df = df.reset_index()
        if len(symbols) == 1:
            sym = next(iter(symbols))
            if isinstance(df.columns, pd.MultiIndex):
                # yfinance returns symbol-keyed MultiIndex columns even for a
                # single symbol, which would make each df[col] a DataFrame.
                df.columns = df.columns.get_level_values(0)
            result = pd.DataFrame(
                {
                    DataCol.DATE.value: df["Date"].values,
                    DataCol.SYMBOL.value: sym,
                    DataCol.OPEN.value: df["Open"].values,
                    DataCol.HIGH.value: df["High"].values,
                    DataCol.LOW.value: df["Low"].values,
                    DataCol.CLOSE.value: df["Close"].values,
                    DataCol.VOLUME.value: df["Volume"].values,
                }
            )
            if not self.auto_adjust:
                result[self.ADJ_CLOSE] = df["Adj Close"].values
        else:
            df.columns = df.columns.to_flat_index()
            sym_list = list(symbols)
            n = len(df)
            result_data: dict[str, Any] = {
                DataCol.DATE.value: np.tile(
                    df[("Date", "")].values, len(sym_list)
                ),
                DataCol.SYMBOL.value: np.repeat(sym_list, n),
                DataCol.OPEN.value: np.concatenate(
                    [df[("Open", sym)].values for sym in sym_list]
                ),
                DataCol.HIGH.value: np.concatenate(
                    [df[("High", sym)].values for sym in sym_list]
                ),
                DataCol.LOW.value: np.concatenate(
                    [df[("Low", sym)].values for sym in sym_list]
                ),
                DataCol.CLOSE.value: np.concatenate(
                    [df[("Close", sym)].values for sym in sym_list]
                ),
                DataCol.VOLUME.value: np.concatenate(
                    [df[("Volume", sym)].values for sym in sym_list]
                ),
            }
            if not self.auto_adjust:
                result_data[self.ADJ_CLOSE] = np.concatenate(
                    [df[("Adj Close", sym)].values for sym in sym_list]
                )
            result = pd.DataFrame(result_data)
        return result
