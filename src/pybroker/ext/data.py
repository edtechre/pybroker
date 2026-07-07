r"""Contains extension classes."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

from datetime import datetime
import json
import os
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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


class YQuery(DataSource):
    r"""Retrieves data from Yahoo Finance using
    `Yahooquery <https://github.com/dpguthrie/yahooquery>`_\ ."""

    _tf_to_period = {
        "": "1d",
        "1hour": "1h",
        "1day": "1d",
        "5day": "5d",
        "1week": "1wk",
    }

    def __init__(self, proxies: Optional[dict] = None):
        super().__init__()
        self.proxies = proxies

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str],
        adjust: Optional[bool],
    ) -> pd.DataFrame:
        """:meta private:"""
        show_yf_progress_bar = (
            not self._logger._disabled
            and not self._logger._progress_bar_disabled
        )
        ticker = Ticker(
            symbols,
            asynchronous=True,
            progress=show_yf_progress_bar,
            proxies=self.proxies,
        )
        timeframe = self._format_timeframe(timeframe)
        if timeframe not in self._tf_to_period:
            raise ValueError(
                f"Unsupported timeframe: '{timeframe}'.\n"
                f"Supported timeframes: {list(self._tf_to_period.keys())}."
            )
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=self._tf_to_period[timeframe],
            adj_ohlc=adjust,
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
                ]
            )
        if df.empty:
            return df
        df = df.reset_index()
        df[DataCol.DATE.value] = pd.to_datetime(df[DataCol.DATE.value])
        df = df[
            [
                DataCol.SYMBOL.value,
                DataCol.DATE.value,
                DataCol.OPEN.value,
                DataCol.HIGH.value,
                DataCol.LOW.value,
                DataCol.CLOSE.value,
                DataCol.VOLUME.value,
            ]
        ]
        return df


class FXMacroData(DataSource):
    r"""Retrieves daily FX spot rates from
    `FXMacroData <https://fxmacrodata.com/>`_."""

    _columns = [
        DataCol.DATE.value,
        DataCol.SYMBOL.value,
        DataCol.OPEN.value,
        DataCol.HIGH.value,
        DataCol.LOW.value,
        DataCol.CLOSE.value,
        DataCol.VOLUME.value,
    ]
    _env_api_keys = ("FXMACRODATA_API_KEY", "FXMD_API_KEY")
    _supported_timeframes = {"", "1day"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.fxmacrodata.com/v1",
        timeout: float = 30,
    ):
        super().__init__()
        self.api_key = api_key or self._get_env_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str],
        adjust: Optional[bool],
    ) -> pd.DataFrame:
        """:meta private:"""
        if timeframe not in self._supported_timeframes:
            raise ValueError(
                "FXMacroData only supports daily data; use timeframe='1d' "
                "or leave it empty."
            )
        result = pd.DataFrame()
        for symbol in symbols:
            base, quote = self._parse_symbol(symbol)
            payload = self._get_json(base, quote, start_date, end_date)
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            rows_df = self._rows_to_df(symbol, rows)
            result = pd.concat([result, rows_df], ignore_index=True)
        if result.empty:
            return self._empty_df()
        result = result.sort_values(
            by=[DataCol.SYMBOL.value, DataCol.DATE.value]
        )
        return result.reset_index(drop=True)

    @classmethod
    def _get_env_api_key(cls) -> Optional[str]:
        for name in cls._env_api_keys:
            api_key = os.getenv(name)
            if api_key:
                return api_key
        return None

    @staticmethod
    def _parse_symbol(symbol: str) -> tuple[str, str]:
        clean_symbol = symbol.strip().upper()
        if clean_symbol.endswith("=X"):
            clean_symbol = clean_symbol[:-2]
        clean_symbol = clean_symbol.replace("/", "")
        clean_symbol = clean_symbol.replace("-", "")
        clean_symbol = clean_symbol.replace("_", "")
        if len(clean_symbol) != 6 or not clean_symbol.isalpha():
            raise ValueError(
                "FXMacroData symbols must be currency pairs such as "
                "'EURUSD' or 'EUR/USD'."
            )
        return clean_symbol[:3].lower(), clean_symbol[3:].lower()

    @classmethod
    def _empty_df(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._columns)

    def _get_json(
        self,
        base: str,
        quote: str,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        params = urlencode(
            {
                "start_date": to_datetime(start_date).strftime("%Y-%m-%d"),
                "end_date": to_datetime(end_date).strftime("%Y-%m-%d"),
            }
        )
        url = f"{self.base_url}/forex/{base}/{quote}?{params}"
        request = Request(url)
        if self.api_key:
            request.add_header("X-API-Key", self.api_key)
        with urlopen(request, timeout=self.timeout) as response:
            content = response.read().decode("utf-8")
        return json.loads(content)

    @classmethod
    def _rows_to_df(cls, symbol: str, rows: list[dict]) -> pd.DataFrame:
        result = []
        for row in rows:
            date = row.get("date")
            rate = cls._get_rate(row)
            if date is None or rate is None:
                continue
            result.append(
                {
                    DataCol.DATE.value: pd.to_datetime(date),
                    DataCol.SYMBOL.value: symbol,
                    DataCol.OPEN.value: rate,
                    DataCol.HIGH.value: rate,
                    DataCol.LOW.value: rate,
                    DataCol.CLOSE.value: rate,
                    DataCol.VOLUME.value: 0,
                }
            )
        if not result:
            return cls._empty_df()
        return pd.DataFrame(result, columns=cls._columns)

    @staticmethod
    def _get_rate(row: dict) -> Optional[float]:
        for key in ("val", "value", "close", "rate"):
            value = row.get(key)
            if value is not None:
                return float(value)
        return None
