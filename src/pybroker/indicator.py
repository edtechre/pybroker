"""Contains indicator related functionality."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import functools
import itertools
import numpy as np
import operator as op
import pandas as pd
import pybroker.vect as vect
from pybroker.cache import CacheDateFields, IndicatorCacheKey
from pybroker.common import BarData, DataCol, IndicatorSymbol, default_parallel
from pybroker.eval import iqr, relative_entropy
from pybroker.scope import StaticScope
from pybroker.vect import highv, lowv, returnv
from collections import defaultdict
from dataclasses import asdict
from joblib import delayed
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Optional,
    Union,
)


def _to_bar_data(df: pd.DataFrame) -> BarData:
    df = df.reset_index()
    required_cols = (
        DataCol.DATE,
        DataCol.OPEN,
        DataCol.HIGH,
        DataCol.LOW,
        DataCol.CLOSE,
    )
    for col in required_cols:
        if col.value not in df.columns:
            raise ValueError(
                f"DataFrame is missing required column: {col.value}"
            )
    return BarData(
        **{col.value: df[col.value].to_numpy() for col in required_cols},
        **{
            col.value: (
                df[col.value].to_numpy() if col.value in df.columns else None
            )
            for col in (DataCol.VOLUME, DataCol.VWAP)
        },  # type: ignore[arg-type]
        **{
            col: df[col].to_numpy() if col in df.columns else None
            for col in StaticScope.instance().custom_data_cols
        },  # type: ignore[arg-type]
    )


class Indicator:
    """Class representing an indicator.

    Args:
        name: Name of indicator.
        fn: :class:`Callable` used to compute the series of indicator values.
        kwargs: ``dict`` of kwargs to pass to ``fn``.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., NDArray[np.float64]],
        kwargs: dict[str, Any],
    ):
        self.name = name
        self._fn = functools.partial(fn, **kwargs)
        self._kwargs = kwargs

    def relative_entropy(self, data: Union[BarData, pd.DataFrame]) -> float:
        """Generates indicator data with ``data`` and computes its relative
        `entropy
        <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.
        """
        return relative_entropy(self(data).values)

    def iqr(self, data: Union[BarData, pd.DataFrame]) -> float:
        """Generates indicator data with ``data`` and computes its
        `interquartile range (IQR)
        <https://en.wikipedia.org/wiki/Interquartile_range>`_.
        """
        return iqr(self(data).values)

    def __call__(self, data: Union[BarData, pd.DataFrame]) -> pd.Series:
        """Computes indicator values."""
        if isinstance(data, pd.DataFrame):
            data = _to_bar_data(data)
        values = self._fn(data)
        if isinstance(values, pd.Series):
            values = values.to_numpy()
        if len(values.shape) != 1:
            raise ValueError(
                f"Indicator {self.name} must return a one-dimensional array."
            )
        return pd.Series(values, index=data.date)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Indicator({self.name!r}, {self._kwargs})"


def indicator(
    name: str, fn: Callable[..., NDArray[np.float64]], **kwargs
) -> Indicator:
    r"""Creates an :class:`.Indicator` instance and registers it globally with
    ``name``.

    Args:
        name: Name for referencing the indicator globally.
        fn: ``Callable[[BarData, ...], NDArray[float]]`` used to compute the
            series of indicator values.
        \**kwargs: Additional arguments to pass to ``fn``.

    Returns:
        :class:`.Indicator` instance.
    """
    scope = StaticScope.instance()
    indicator = Indicator(name, fn, kwargs)
    scope.set_indicator(indicator)
    return indicator


def _decorate_indicator_fn(ind_name: str):
    fn = StaticScope.instance().get_indicator(ind_name).__call__

    def decorated_indicator_fn(
        symbol: str,
        ind_name: str,
        date: NDArray[np.datetime64],
        open: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        volume: Optional[NDArray[np.float64]],
        vwap: Optional[NDArray[np.float64]],
        custom_col_data: Mapping[str, Optional[NDArray]],
    ) -> tuple[IndicatorSymbol, pd.Series]:
        bar_data = BarData(
            date=date,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap,
            **custom_col_data,
        )
        series = fn(bar_data)
        return IndicatorSymbol(ind_name, symbol), series

    return decorated_indicator_fn


class IndicatorsMixin:
    """Mixin implementing indicator related functionality."""

    def compute_indicators(
        self,
        df: pd.DataFrame,
        indicator_syms: Iterable[IndicatorSymbol],
        cache_date_fields: Optional[CacheDateFields],
        disable_parallel: bool,
    ) -> dict[IndicatorSymbol, pd.Series]:
        """Computes indicator data for the provided
        :class:`pybroker.common.IndicatorSymbol` pairs.

        Args:
            df: :class:`pandas.DataFrame` used to compute the indicator values.
            indicator_syms: ``Iterable`` of
                :class:`pybroker.common.IndicatorSymbol` pairs of indicators
                to compute.
            cache_date_fields: Date fields used to key cache data. Pass
                ``None`` to disable caching.
            disable_parallel: If ``True``, indicator data is computed
                serially for all :class:`pybroker.common.IndicatorSymbol`
                pairs. If ``False``, indicator data is computed in parallel
                using multiple processes.

        Returns:
            ``dict`` mapping each :class:`pybroker.common.IndicatorSymbol` pair
            to a computed :class:`pandas.Series` of indicator values.
        """
        if not indicator_syms or df.empty:
            return {}
        scope = StaticScope.instance()
        indicator_data, uncached_ind_syms = self._get_cached_indicators(
            indicator_syms, cache_date_fields
        )
        if not uncached_ind_syms:
            scope.logger.loaded_indicator_data()
            scope.logger.info_loaded_indicator_data(indicator_syms)
            return indicator_data
        if indicator_data:
            scope.logger.info_loaded_indicator_data(indicator_data.keys())
        scope.logger.indicator_data_start(uncached_ind_syms)
        scope.logger.info_indicator_data_start(uncached_ind_syms)
        sym_data: dict[str, dict[str, Optional[NDArray]]] = defaultdict(dict)
        for _, sym in uncached_ind_syms:
            if sym in sym_data:
                continue
            data = df[df[DataCol.SYMBOL.value] == sym]
            for col in scope.all_data_cols:
                if col not in data.columns:
                    sym_data[sym][col] = None
                    continue
                sym_data[sym][col] = data[col].to_numpy()
        for i, (ind_sym, series) in enumerate(
            self._run_indicators(sym_data, uncached_ind_syms, disable_parallel)
        ):
            indicator_data[ind_sym] = series
            self._set_cached_indicator(series, ind_sym, cache_date_fields)
            scope.logger.indicator_data_loading(i + 1)
        return indicator_data

    def _get_cached_indicators(
        self,
        indicator_syms: Iterable[IndicatorSymbol],
        cache_date_fields: Optional[CacheDateFields],
    ) -> tuple[dict[IndicatorSymbol, pd.Series], list[IndicatorSymbol]]:
        indicator_syms = sorted(indicator_syms)
        indicator_data: dict[IndicatorSymbol, pd.Series] = {}
        if cache_date_fields is None:
            return indicator_data, indicator_syms
        scope = StaticScope.instance()
        if scope.indicator_cache is None:
            return indicator_data, indicator_syms
        uncached_ind_syms = []
        for ind_sym in indicator_syms:
            cache_key = IndicatorCacheKey(
                symbol=ind_sym.symbol,
                ind_name=ind_sym.ind_name,
                **asdict(cache_date_fields),
            )
            scope.logger.debug_get_indicator_cache(cache_key)
            data = scope.indicator_cache.get(repr(cache_key))
            if data is not None:
                indicator_data[ind_sym] = data
            else:
                uncached_ind_syms.append(ind_sym)
        return indicator_data, uncached_ind_syms

    def _set_cached_indicator(
        self,
        series: pd.Series,
        ind_sym: IndicatorSymbol,
        cache_date_fields: Optional[CacheDateFields],
    ):
        if cache_date_fields is None:
            return
        scope = StaticScope.instance()
        if scope.indicator_cache is None:
            return
        cache_key = IndicatorCacheKey(
            symbol=ind_sym.symbol,
            ind_name=ind_sym.ind_name,
            **asdict(cache_date_fields),
        )
        scope.logger.debug_set_indicator_cache(cache_key)
        scope.indicator_cache.set(repr(cache_key), series)

    def _run_indicators(
        self,
        sym_data: Mapping[str, Mapping[str, Optional[NDArray]]],
        ind_syms: Collection[IndicatorSymbol],
        disable_parallel: bool,
    ) -> Iterable[tuple[IndicatorSymbol, pd.Series]]:
        fns = {}
        for ind_name, _ in ind_syms:
            if ind_name in fns:
                continue
            fns[ind_name] = _decorate_indicator_fn(ind_name)
        scope = StaticScope.instance()

        def args_fn(ind_name, sym):
            return {
                "symbol": sym,
                "ind_name": ind_name,
                "custom_col_data": {
                    col: sym_data[sym][col] for col in scope.custom_data_cols
                },
                **{col: sym_data[sym][col] for col in scope.default_data_cols},
            }

        if disable_parallel or len(ind_syms) == 1:
            scope.logger.debug_compute_indicators(is_parallel=False)
            return tuple(
                fns[ind_name](**args_fn(ind_name, sym))
                for ind_name, sym in ind_syms
            )
        else:
            scope.logger.debug_compute_indicators(is_parallel=True)

            with default_parallel() as parallel:
                return parallel(
                    delayed(fns[ind_name])(**args_fn(ind_name, sym))
                    for ind_name, sym in ind_syms
                )


class IndicatorSet(IndicatorsMixin):
    """Computes data for multiple indicators."""

    def __init__(self):
        self._ind_names: set[str] = set()

    def add(self, indicators: Union[Indicator, Iterable[Indicator]], *args):
        """Adds indicators."""
        if isinstance(indicators, Indicator):
            indicators = (indicators, *args)
        else:
            indicators = (*indicators, *args)
        self._ind_names.update(map(op.attrgetter("name"), indicators))

    def remove(self, indicators: Union[Indicator, Iterable[Indicator]], *args):
        """Removes indicators."""
        if isinstance(indicators, Indicator):
            indicators = (indicators, *args)
        else:
            indicators = (*indicators, *args)
        self._ind_names.difference_update(
            map(op.attrgetter("name"), indicators)
        )

    def clear(self):
        """Removes all indicators."""
        self._ind_names.clear()

    def __call__(
        self, df: pd.DataFrame, disable_parallel: bool = False
    ) -> pd.DataFrame:
        """Computes indicator data.

        Args:
            df: :class:`pandas.DataFrame` of input data.
            disable_parallel: If ``True``, indicator data is computed serially.
                If ``False``, indicator data is computed in parallel using
                multiple processes. Defaults to ``False``.

        Returns:
            :class:`pandas.DataFrame` containing the computed indicator data.
        """
        if not self._ind_names:
            raise ValueError("No indicators were added.")
        if df.empty:
            return pd.DataFrame(
                columns=[DataCol.DATE.value, DataCol.SYMBOL.value]
                + list(self._ind_names)
            )
        syms = df[DataCol.SYMBOL.value].unique()
        ind_syms = tuple(
            itertools.starmap(
                IndicatorSymbol, itertools.product(self._ind_names, syms)
            )
        )
        ind_dict = self.compute_indicators(
            df=df,
            indicator_syms=ind_syms,
            cache_date_fields=None,
            disable_parallel=disable_parallel,
        )
        sym_dict: dict[str, dict[str, pd.Series]] = defaultdict(dict)
        for ind_sym, series in ind_dict.items():
            sym_dict[ind_sym.symbol][ind_sym.ind_name] = series
        data: dict[str, list] = defaultdict(list)
        for sym, ind_series in sym_dict.items():
            dates = df[df[DataCol.SYMBOL.value] == sym][DataCol.DATE.value]
            data[DataCol.SYMBOL.value].extend(
                itertools.repeat(sym, len(dates))
            )
            data[DataCol.DATE.value].extend(dates)
            for ind_name, series in ind_series.items():
                data[ind_name].extend(series.values)
        return pd.DataFrame.from_dict(data)


def highest(name: str, field: str, period: int) -> Indicator:
    """Creates a rolling high :class:`.Indicator`.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field for computing the rolling
            high.
        period: Lookback period.

    Returns:
        Rolling high :class:`.Indicator`.
    """

    def _highest(data: BarData):
        values = getattr(data, field)
        return highv(values, period)

    return indicator(name, _highest)


def lowest(name: str, field: str, period: int) -> Indicator:
    """Creates a rolling low :class:`.Indicator`.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field for computing the rolling
            low.
        period: Lookback period.

    Returns:
        Rolling low :class:`.Indicator`.
    """

    def _lowest(data: BarData):
        values = getattr(data, field)
        return lowv(values, period)

    return indicator(name, _lowest)


def returns(name: str, field: str, period: int = 1) -> Indicator:
    """Creates a rolling returns :class:`.Indicator`.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field for computing the rolling
            returns.
        period: Returns period. Defaults to 1.

    Returns:
        Rolling returns :class:`.Indicator`.
    """

    def _returns(data: BarData):
        values = getattr(data, field)
        return returnv(values, period)

    return indicator(name, _returns)


def detrended_rsi(
    name: str, field: str, short_length: int, long_length: int, reg_length: int
) -> Indicator:
    """Detrended Relative Strength Index (RSI).

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        short_length: Lookback for the short-term RSI.
        long_length: Lookback for the long-term RSI.
        reg_length: Number of bars used for linear regressions.

    Returns:
        Detrended RSI :class:`.Indicator`.
    """

    def _detrended_rsi(data: BarData):
        values = getattr(data, field)
        return vect.detrended_rsi(
            values,
            short_length=short_length,
            long_length=long_length,
            reg_length=reg_length,
        )

    return indicator(name, _detrended_rsi)


def macd(
    name: str,
    short_length: int,
    long_length: int,
    smoothing: float = 0.0,
    scale: float = 1.0,
) -> Indicator:
    """Moving Average Convergence Divergence.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        short_length: Short-term lookback.
        long_length: Long-term lookback.
        smoothing: Compute MACD minus smoothed if >= 2.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Moving Average Convergence Divergence :class:`.Indicator`.
    """

    def _macd(data: BarData):
        return vect.macd(
            high=data.high,
            low=data.low,
            close=data.close,
            short_length=short_length,
            long_length=long_length,
            smoothing=smoothing,
            scale=scale,
        )

    return indicator(name, _macd)


def stochastic(name: str, lookback: int, smoothing: int = 0) -> Indicator:
    """Stochastic.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        smoothing: Number of times the raw stochastic is smoothed, either 0,
            1, or 2 times. Defaults to ``0``.

    Returns:
        Stochastic :class:`.Indicator`.
    """

    def _stochastic(data: BarData):
        return vect.stochastic(
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            smoothing=smoothing,
        )

    return indicator(name, _stochastic)


def stochastic_rsi(
    name: str,
    field: str,
    rsi_lookback: int,
    sto_lookback: int,
    smoothing: float = 0.0,
) -> Indicator:
    """Stochastic Relative Strength Index (RSI).

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        rsi_lookback: Lookback length for RSI calculation.
        sto_lookback: Lookback length for Stochastic calculation.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        Stochastic RSI :class:`.Indicator`.
    """

    def _stochastic_rsi(data: BarData):
        values = getattr(data, field)
        return vect.stochastic_rsi(
            values,
            rsi_lookback=rsi_lookback,
            sto_lookback=sto_lookback,
            smoothing=smoothing,
        )

    return indicator(name, _stochastic_rsi)


def linear_trend(
    name: str, field: str, lookback: int, atr_length: int, scale: float = 1.0
) -> Indicator:
    """Linear Trend Strength.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Linear Trend Strength :class:`.Indicator`.
    """

    def _linear_trend(data: BarData):
        values = getattr(data, field)
        return vect.linear_trend(
            values,
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            atr_length=atr_length,
            scale=scale,
        )

    return indicator(name, _linear_trend)


def quadratic_trend(
    name: str, field: str, lookback: int, atr_length: int, scale: float = 1.0
) -> Indicator:
    """Quadratic Trend Strength.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Quadratic Trend Strength :class:`.Indicator`.
    """

    def _quadratic_trend(data: BarData):
        values = getattr(data, field)
        return vect.quadratic_trend(
            values,
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            atr_length=atr_length,
            scale=scale,
        )

    return indicator(name, _quadratic_trend)


def cubic_trend(
    name: str, field: str, lookback: int, atr_length: int, scale: float = 1.0
) -> Indicator:
    """Cubic Trend Strength.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Cubic Trend Strength :class:`.Indicator`.
    """

    def _cubic_trend(data: BarData):
        values = getattr(data, field)
        return vect.cubic_trend(
            values,
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            atr_length=atr_length,
            scale=scale,
        )

    return indicator(name, _cubic_trend)


def adx(name: str, lookback: int) -> Indicator:
    """Average Directional Movement Index.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.

    Returns:
        Average Directional Movement Index :class:`.Indicator`.
    """

    def _adx(data: BarData):
        return vect.adx(
            high=data.high, low=data.low, close=data.close, lookback=lookback
        )

    return indicator(name, _adx)


def aroon_up(name: str, lookback: int) -> Indicator:
    """Aroon Upward Trend.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.

    Returns:
        Aroon Upward Trend :class:`.Indicator`.
    """

    def _aroon_up(data: BarData):
        return vect.aroon_up(high=data.high, low=data.low, lookback=lookback)

    return indicator(name, _aroon_up)


def aroon_down(name: str, lookback: int) -> Indicator:
    """Aroon Downward Trend.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.

    Returns:
        Aroon Downward Trend :class:`.Indicator`.
    """

    def _aroon_down(data: BarData):
        return vect.aroon_down(high=data.high, low=data.low, lookback=lookback)

    return indicator(name, _aroon_down)


def aroon_diff(name: str, lookback: int) -> Indicator:
    """Aroon Upward Trend minus Aroon Downward Trend.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.

    Returns:
        Aroon Upward Trend minus Aroon Downward Trend :class:`.Indicator`.
    """

    def _aroon_diff(data: BarData):
        return vect.aroon_diff(high=data.high, low=data.low, lookback=lookback)

    return indicator(name, _aroon_diff)


def close_minus_ma(
    name: str, lookback: int, atr_length: int, scale: float = 1.0
) -> Indicator:
    """Close Minus Moving Average.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Close Minus Moving Average :class:`.Indicator`.
    """

    def _close_minus_ma(data: BarData):
        return vect.close_minus_ma(
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            atr_length=atr_length,
            scale=scale,
        )

    return indicator(name, _close_minus_ma)


def linear_deviation(
    name: str, field: str, lookback: int, scale: float = 0.6
) -> Indicator:
    """Deviation from Linear Trend.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Deviation from Linear Trend :class:`.Indicator`.
    """

    def _linear_deviation(data: BarData):
        values = getattr(data, field)
        return vect.linear_deviation(values, lookback=lookback, scale=scale)

    return indicator(name, _linear_deviation)


def quadratic_deviation(
    name: str, field: str, lookback: int, scale: float = 0.6
) -> Indicator:
    """Deviation from Quadratic Trend.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Deviation from Quadratic Trend :class:`.Indicator`.
    """

    def _quadratic_deviation(data: BarData):
        values = getattr(data, field)
        return vect.quadratic_deviation(values, lookback=lookback, scale=scale)

    return indicator(name, _quadratic_deviation)


def cubic_deviation(
    name: str, field: str, lookback: int, scale: float = 0.6
) -> Indicator:
    """Deviation from Cubic Trend.

    Args:
        name: Indicator name.
        field: :class:`pybroker.common.BarData` field name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Deviation from Cubic Trend :class:`.Indicator`.
    """

    def _cubic_deviation(data: BarData):
        values = getattr(data, field)
        return vect.cubic_deviation(values, lookback=lookback, scale=scale)

    return indicator(name, _cubic_deviation)


def price_intensity(
    name: str, smoothing: float = 0.0, scale: float = 0.8
) -> Indicator:
    """Price Intensity.

    Args:
        name: Indicator name.
        smoothing: Amount of smoothing. Defaults to ``0``.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.8``.

    Returns:
        Price Intensity :class:`.Indicator`.
    """

    def _price_intensity(data: BarData):
        return vect.price_intensity(
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            smoothing=smoothing,
            scale=scale,
        )

    return indicator(name, _price_intensity)


def price_change_oscillator(
    name: str, short_length: int, multiplier: int, scale: float = 4.0
) -> Indicator:
    """Price Change Oscillator.

    Args:
        name: Indicator name.
        short_length: Number of short lookback bars.
        multiplier: Multiplier used to compute number of long lookback bars =
            ``multiplier * short_length``.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``4.0``.

    Returns:
        Price Change Oscillator :class:`.Indicator`.
    """

    def _price_change_oscillator(data: BarData):
        return vect.price_change_oscillator(
            high=data.high,
            low=data.low,
            close=data.close,
            short_length=short_length,
            multiplier=multiplier,
            scale=scale,
        )

    return indicator(name, _price_change_oscillator)


def intraday_intensity(
    name: str, lookback: int, smoothing: float = 0.0
) -> Indicator:
    """Intraday Intensity.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        Intraday Intensity :class:`.Indicator`.
    """

    def _intraday_intensity(data: BarData):
        return vect.intraday_intensity(
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            smoothing=smoothing,
        )

    return indicator(name, _intraday_intensity)


def money_flow(name: str, lookback: int, smoothing: float = 0.0) -> Indicator:
    """Chaikin's Money Flow.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        Chaikin's Money Flow :class:`.Indicator`.
    """

    def _money_flow(data: BarData):
        return vect.money_flow(
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            smoothing=smoothing,
        )

    return indicator(name, _money_flow)


def reactivity(
    name: str, lookback: int, smoothing: float = 0.0, scale: float = 0.6
) -> Indicator:
    """Reactivity.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        smoothing: Smoothing multiplier.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Reactivity :class:`.Indicator`.
    """

    def _reactivity(data: BarData):
        return vect.reactivity(
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            smoothing=smoothing,
            scale=scale,
        )

    return indicator(name, _reactivity)


def price_volume_fit(
    name: str, lookback: int, scale: float = 9.0
) -> Indicator:
    """Price Volume Fit.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``9.0``.

    Returns:
        Price Volume Fit :class:`.Indicator`.
    """

    def _price_volume_fit(data: BarData):
        return vect.price_volume_fit(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _price_volume_fit)


def volume_weighted_ma_ratio(
    name: str, lookback: int, scale: float = 1.0
) -> Indicator:
    """Volume-Weighted Moving Average Ratio.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        Volume-Weighted Moving Average Ratio :class:`.Indicator`.
    """

    def _volume_weighted_ma_ratio(data: BarData):
        return vect.volume_weighted_ma_ratio(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _volume_weighted_ma_ratio)


def normalized_on_balance_volume(
    name: str, lookback: int, scale: float = 0.6
) -> Indicator:
    """Normalized On-Balance Volume.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Normalized On-Balance Volume :class:`.Indicator`.
    """

    def _normalized_on_balance_volume(data: BarData):
        return vect.normalized_on_balance_volume(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _normalized_on_balance_volume)


def delta_on_balance_volume(
    name: str, lookback: int, delta_length: int = 0, scale: float = 0.6
) -> Indicator:
    """Delta On-Balance Volume.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        delta_length: Lag for differencing.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        Delta On-Balance Volume :class:`.Indicator`.
    """

    def _delta_on_balance_volume(data: BarData):
        return vect.delta_on_balance_volume(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            delta_length=delta_length,
            scale=scale,
        )

    return indicator(name, _delta_on_balance_volume)


def normalized_positive_volume_index(
    name: str, lookback: int, scale: float = 0.5
) -> Indicator:
    """Normalized Positive Volume Index.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.5``.

    Returns:
        Normalized Positive Volume Index :class:`.Indicator`.
    """

    def _normalized_positive_volume_index(data: BarData):
        return vect.normalized_positive_volume_index(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _normalized_positive_volume_index)


def normalized_negative_volume_index(
    name: str, lookback: int, scale: float = 0.5
) -> Indicator:
    """Normalized Negative Volume Index.

    Args:
        name: Indicator name.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.5``.

    Returns:
        Normalized Negative Volume Index :class:`.Indicator`.
    """

    def _normalized_negative_volume_index(data: BarData):
        return vect.normalized_negative_volume_index(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _normalized_negative_volume_index)


def volume_momentum(
    name: str, short_length: int, multiplier: int = 2, scale: float = 3.0
) -> Indicator:
    """Volume Momentum.

    Args:
        name: Indicator name.
        short_length: Number of short lookback bars.
        multiplier: Lookback multiplier. Defaults to ``2``.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``3.0``.

    Returns:
        Volume Momentum :class:`.Indicator`.
    """

    def _volume_momentum(data: BarData):
        return vect.volume_momentum(
            volume=data.volume,
            short_length=short_length,
            multiplier=multiplier,
            scale=scale,
        )

    return indicator(name, _volume_momentum)


def laguerre_rsi(name: str, fe_length: int = 13) -> Indicator:
    """Laguerre Relative Strength Index (RSI).

    Args:
        name: Indicator name.
        fe_length: Fractal Energy length. Defaults to ``13``.

    Returns:
        Laguerre RSI :class:`.Indicator`.
    """

    def _laguerre_rsi(data: BarData):
        return vect.laguerre_rsi(
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            fe_length=fe_length,
        )

    return indicator(name, _laguerre_rsi)
