r"""Contains extension classes."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Union
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import akshare
import numpy as np
import pandas as pd
from yahooquery import Ticker

from pybroker.common import DataCol, to_datetime
from pybroker.data import DataSource

_ADANOS_DEFAULT_BASE_URL = "https://api.adanos.org"
_ADANOS_SOURCE_PATHS = {
    "reddit": "/reddit/stocks/v1",
    "x": "/x/stocks/v1",
    "news": "/news/stocks/v1",
    "polymarket": "/polymarket/stocks/v1",
}
_ADANOS_METRICS = ("sentiment", "buzz", "mentions")


def _clean_adanos_source(source: str) -> str:
    source = source.lower().strip()
    if source not in _ADANOS_SOURCE_PATHS:
        raise ValueError(
            f"Unsupported Adanos source: {source!r}.\n"
            f"Supported sources: {list(_ADANOS_SOURCE_PATHS.keys())}."
        )
    return source


def _empty_adanos_features(
    columns: Iterable[str],
) -> dict[str, np.float64]:
    return {col: np.float64(np.nan) for col in columns}


def _metric_value(data: Mapping[str, Any], metric: str) -> Optional[float]:
    keys = {
        "sentiment": ("sentiment_score", "sentiment"),
        "buzz": ("buzz_score", "buzz"),
        "mentions": ("mentions", "trade_count"),
    }[metric]
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None


def _adanos_columns(
    sources: Iterable[str], include_aggregate: bool
) -> tuple[str, ...]:
    columns = []
    if include_aggregate:
        columns.extend(f"adanos_{metric}" for metric in _ADANOS_METRICS)
    for source in sources:
        columns.extend(
            f"adanos_{source}_{metric}" for metric in _ADANOS_METRICS
        )
    return tuple(columns)


def _fetch_adanos_json(
    source: str,
    symbol: str,
    days: int,
    *,
    api_key: str,
    base_url: str,
    timeout: float,
) -> Mapping[str, Any]:
    path = _ADANOS_SOURCE_PATHS[source]
    query = urlencode({"days": days})
    url = f"{base_url.rstrip('/')}{path}/stock/{symbol}?{query}"
    request = Request(url, headers={"X-API-Key": api_key})
    try:
        with urlopen(request, timeout=timeout) as response:
            import json

            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return {}
        raise


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


class AdanosSentiment(DataSource):
    r"""Enriches another :class:`pybroker.data.DataSource` with Adanos Market
    Sentiment API features.

    The wrapped data source still provides the required OHLCV bars. Adanos
    sentiment is left-joined onto those bars as custom PyBroker columns, making
    the features available from :class:`pybroker.context.ExecContext` and model
    training data.

    Args:
        data_source: Data source used to query price bars.
        api_key: Adanos API key.
        sources: Adanos sources to query. Supported values are ``"reddit"``,
            ``"x"``, ``"news"``, and ``"polymarket"``.
        include_aggregate: Whether to include aggregate ``adanos_sentiment``,
            ``adanos_buzz``, and ``adanos_mentions`` columns across sources.
        base_url: Adanos API base URL.
        timeout: HTTP timeout in seconds.
    """

    _CACHE_KEY = "adanos_sentiment"

    def __init__(
        self,
        data_source: DataSource,
        api_key: str,
        sources: Iterable[str] = ("reddit", "x", "news", "polymarket"),
        include_aggregate: bool = True,
        base_url: str = _ADANOS_DEFAULT_BASE_URL,
        timeout: float = 10.0,
    ):
        super().__init__()
        self.data_source = data_source
        self.api_key = api_key
        self.sources = tuple(
            _clean_adanos_source(source) for source in sources
        )
        if not self.sources:
            raise ValueError("At least one Adanos source is required.")
        self.include_aggregate = include_aggregate
        self.base_url = base_url
        self.timeout = timeout
        self.columns = _adanos_columns(self.sources, include_aggregate)
        self._scope.register_custom_cols(self.columns)

    def query(
        self,
        symbols: Union[str, Iterable[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: Optional[str] = "",
        adjust: Optional[Any] = None,
    ) -> pd.DataFrame:
        cache_adjust = (
            self._CACHE_KEY,
            adjust,
            self.sources,
            self.include_aggregate,
            self.base_url,
        )
        return super().query(
            symbols, start_date, end_date, timeframe, cache_adjust
        )

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str],
        adjust: Optional[Any],
    ) -> pd.DataFrame:
        """:meta private:"""
        data_source_adjust = (
            adjust[1]
            if isinstance(adjust, tuple) and adjust[0] == self._CACHE_KEY
            else adjust
        )
        price_df = self.data_source.query(
            symbols, start_date, end_date, timeframe, data_source_adjust
        )
        if price_df.empty:
            for col in self.columns:
                price_df[col] = np.nan
            return price_df

        features = self._fetch_sentiment_features(
            symbols, start_date, end_date
        )
        if features.empty:
            for col in self.columns:
                price_df[col] = np.nan
            return price_df

        price_df = price_df.copy()
        price_df[DataCol.DATE.value] = pd.to_datetime(
            price_df[DataCol.DATE.value]
        )
        price_df["_adanos_date"] = price_df[DataCol.DATE.value].dt.date
        features = features.copy()
        features["_adanos_date"] = pd.to_datetime(
            features[DataCol.DATE.value]
        ).dt.date
        enriched = price_df.merge(
            features,
            how="left",
            left_on=[DataCol.SYMBOL.value, "_adanos_date"],
            right_on=[DataCol.SYMBOL.value, "_adanos_date"],
            suffixes=("", "_adanos_feature"),
        )
        enriched.drop(
            columns=["_adanos_date", f"{DataCol.DATE.value}_adanos_feature"],
            inplace=True,
        )
        for col in self.columns:
            if col not in enriched.columns:
                enriched[col] = np.nan
        return enriched

    def _fetch_sentiment_features(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        rows = []
        days = max(1, (end_date.date() - start_date.date()).days + 1)
        for symbol in symbols:
            daily_rows: dict[pd.Timestamp, dict[str, Any]] = {}
            for source in self.sources:
                payload = self._get_sentiment_payload(source, symbol, days)
                self._add_source_payload(daily_rows, source, end_date, payload)
            for date, values in daily_rows.items():
                values[DataCol.SYMBOL.value] = symbol
                values[DataCol.DATE.value] = date
                self._set_aggregate_values(values)
                rows.append(values)
        if not rows:
            return pd.DataFrame(
                columns=(
                    DataCol.SYMBOL.value,
                    DataCol.DATE.value,
                    *self.columns,
                )
            )
        features = pd.DataFrame(rows)
        for col in self.columns:
            if col not in features.columns:
                features[col] = np.nan
        return features[
            [DataCol.SYMBOL.value, DataCol.DATE.value, *self.columns]
        ]

    def _get_sentiment_payload(
        self,
        source: str,
        symbol: str,
        days: int,
    ) -> Mapping[str, Any]:
        return _fetch_adanos_json(
            source,
            symbol,
            days,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _add_source_payload(
        self,
        daily_rows: dict[pd.Timestamp, dict[str, Any]],
        source: str,
        end_date: datetime,
        payload: Mapping[str, Any],
    ):
        trend = (
            payload.get("daily_trend") or payload.get("trend_history") or []
        )
        added_daily_trend = False
        if isinstance(trend, list):
            for item in trend:
                if not isinstance(item, Mapping):
                    continue
                date = item.get("date")
                if date is None:
                    continue
                self._set_source_values(
                    daily_rows,
                    pd.to_datetime(date).normalize(),
                    source,
                    item,
                )
                added_daily_trend = True
        if payload and not added_daily_trend:
            self._set_source_values(
                daily_rows,
                pd.Timestamp(end_date).normalize(),
                source,
                payload,
            )

    def _set_source_values(
        self,
        daily_rows: dict[pd.Timestamp, dict[str, Any]],
        date: pd.Timestamp,
        source: str,
        data: Mapping[str, Any],
    ):
        row = daily_rows.setdefault(date, _empty_adanos_features(self.columns))
        for metric in _ADANOS_METRICS:
            value = _metric_value(data, metric)
            if value is not None:
                row[f"adanos_{source}_{metric}"] = value

    def _set_aggregate_values(self, values: dict[str, Any]):
        if not self.include_aggregate:
            return
        for metric in _ADANOS_METRICS:
            source_values = [
                values.get(f"adanos_{source}_{metric}")
                for source in self.sources
                if not pd.isna(values.get(f"adanos_{source}_{metric}"))
            ]
            if not source_values:
                continue
            if metric == "mentions":
                values[f"adanos_{metric}"] = float(sum(source_values))
            else:
                values[f"adanos_{metric}"] = float(
                    sum(source_values) / len(source_values)
                )
