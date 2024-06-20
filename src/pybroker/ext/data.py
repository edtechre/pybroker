r"""Contains extension classes."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

from datetime import datetime
from typing import Final, Iterable, Optional, Union

import akshare
import pandas as pd
from yahooquery import Ticker

from pybroker.common import DataCol, to_datetime
from pybroker.data import DataSource


class AKShare(DataSource):
    r"""Retrieves data from `AKShare <https://akshare.akfamily.xyz/>`_."""

    _tf_to_period = {
        "": "daily",
        "1day": "daily",
        "1week": "weekly",
    }

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str],
        adjust: Optional[str],
    ) -> pd.DataFrame:
        """:meta private:"""
        start_date_str = to_datetime(start_date).strftime("%Y%m%d")
        end_date_str = to_datetime(end_date).strftime("%Y%m%d")
        symbols_list = list(symbols)
        symbols_simple = [item.split(".")[0] for item in symbols_list]
        result = pd.DataFrame()
        formatted_tf = self._format_timeframe(timeframe)
        if formatted_tf in AKShare._tf_to_period:
            period = AKShare._tf_to_period[formatted_tf]
            for i in range(len(symbols_list)):
                temp_df = akshare.stock_zh_a_hist(
                    symbol=symbols_simple[i],
                    start_date=start_date_str,
                    end_date=end_date_str,
                    period=period,
                    adjust=adjust if adjust is not None else "",
                )
                if not temp_df.columns.empty:
                    temp_df["symbol"] = symbols_list[i]
                result = pd.concat([result, temp_df], ignore_index=True)
        if result.columns.empty:
            return pd.DataFrame(
                columns=[
                    DataCol.SYMBOL.value,
                    DataCol.DATE.value,
                    DataCol.OPEN.value,
                    DataCol.HIGH.value,
                    DataCol.LOW.value,
                    DataCol.CLOSE.value,
                    DataCol.VOLUME.value,
                ]
            )
        if result.empty:
            return result
        result.rename(
            columns={
                "日期": DataCol.DATE.value,
                "开盘": DataCol.OPEN.value,
                "收盘": DataCol.CLOSE.value,
                "最高": DataCol.HIGH.value,
                "最低": DataCol.LOW.value,
                "成交量": DataCol.VOLUME.value,
            },
            inplace=True,
        )
        result["date"] = pd.to_datetime(result["date"])
        result = result[
            [
                DataCol.DATE.value,
                DataCol.SYMBOL.value,
                DataCol.OPEN.value,
                DataCol.HIGH.value,
                DataCol.LOW.value,
                DataCol.CLOSE.value,
                DataCol.VOLUME.value,
            ]
        ]
        return result


class YQDataSource(DataSource):
    r"""Retrieves data from `Yahoo Finance <https://finance.yahoo.com/>`_\ .

    Attributes:
        ADJ_CLOSE: Column name of adjusted close prices.
    """

    ADJ_CLOSE: Final = "adj_close"
    __TIMEFRAME: Final = "1d"
    _tf_to_period = {
        "": "1d",
        "1min": "1m",
        "2min": "2m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "60min": "60m",
        "90min": "90m",
        "1hour": "1h",
        "1day": "1d",
        "5day": "5d",
        "1week": "1wk",
    }

    def __init__(self, proxies: dict = None):
        super().__init__()
        self._scope.register_custom_cols(self.ADJ_CLOSE)
        self.proxies = proxies

    def query(
            self,
            symbols: Union[str, Iterable[str]],
            start_date: Union[str, datetime],
            end_date: Union[str, datetime],
            _timeframe: Optional[str] = "",
            _adjust: Optional[str] = None,
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
        return super().query(symbols, start_date, end_date, self.__TIMEFRAME, _adjust)

    def _fetch_data(
            self,
            symbols: frozenset[str],
            start_date: datetime,
            end_date: datetime,
            _timeframe: Optional[str],
            _adjust: Optional[str],
    ) -> pd.DataFrame:
        """:meta private:"""
        show_yf_progress_bar = (
                not self._logger._disabled and not self._logger._progress_bar_disabled
        )
        tickers = Ticker(
            symbols,
            asynchronous=True,
            progress=show_yf_progress_bar,
            proxies=self.proxies,
        )
        df = tickers.history(
            start=start_date,
            end=end_date,
            interval=self._tf_to_period[_timeframe],
            adj_ohlc=_adjust,
        )
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
        df["date"] = pd.to_datetime(df["date"])
        df.rename(columns={"adjclose": self.ADJ_CLOSE}, inplace=True)
        df = df[
            ["date", "symbol", "open", "high", "low", "close", "volume", self.ADJ_CLOSE]
        ]
        return df
