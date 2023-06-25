r"""Contains extension classes."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import akshare
import pandas as pd

from pybroker.common import DataCol, to_datetime
from pybroker.data import DataSource
from datetime import datetime
from typing import Final, Iterable, Optional, Union


class AKShare(DataSource):
    r"""Retrieves data from `AKShare <https://akshare.akfamily.xyz/>`_\ .

    Attributes:
        ADJ_CLOSE: Column name of adjusted close prices.
    """

    __TIMEFRAME: Final = "1d"

    def query(
        self,
        symbols: Union[str, Iterable[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        _: Optional[str] = "",
        adjust: Optional[str] = "hfq",
    ) -> pd.DataFrame:
        r"""Queries data from `AKShare <https://akshare.akfamily.xyz/>`_\ .
        The timeframe of the data is limited to per day only.

        Args:
            symbols: Ticker symbols of the data to query.
            start_date: Start date of the data to query (inclusive).
            end_date: End date of the data to query (inclusive).
            adjust: The type of adjustment to make.

        Returns:
            :class:`pandas.DataFrame` containing the queried data.
        """
        return super().query(
            symbols, start_date, end_date, self.__TIMEFRAME, adjust
        )

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        _: Optional[str],
        adjust: Optional[str],
    ) -> pd.DataFrame:
        """:meta private:"""
        start_date_str = to_datetime(start_date).strftime("%Y%m%d")
        end_date_str = to_datetime(end_date).strftime("%Y%m%d")
        symbols_list = list(symbols)
        symbols_simple = [item.split(".")[0] for item in symbols_list]
        result = pd.DataFrame()
        for i in range(len(symbols_list)):
            try:
                temp_df = akshare.stock_zh_a_hist(
                    symbols_simple[i],
                    start_date=start_date_str,
                    end_date=end_date_str,
                    period="daily",
                    adjust=adjust if adjust is not None else "",
                )
                if not temp_df.columns.empty:
                    temp_df["symbol"] = symbols_list[i]
            except KeyError:
                temp_df = pd.DataFrame()
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
