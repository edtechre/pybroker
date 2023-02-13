"""Contains indicator related functionality."""

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

from .cache import CacheDateFields, IndicatorCacheKey
from .common import BarData, DataCol, IndicatorSymbol, default_parallel
from .eval import iqr, relative_entropy
from .scope import StaticScope
from .vect import highv, lowv, returnv
from collections import defaultdict
from dataclasses import asdict
import itertools
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
import functools
import numpy as np
import operator as op
import pandas as pd


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
        },
        **{
            col: df[col].to_numpy() if col in df.columns else None
            for col in StaticScope.instance().custom_data_cols
        },
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
        fn: Callable[..., NDArray[np.float_]],
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
        return pd.Series(values, index=data.date)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Indicator({self.name!r}, {self._kwargs})"


def indicator(
    name: str, fn: Callable[..., NDArray[np.float_]], **kwargs
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
        open: NDArray[np.float_],
        high: NDArray[np.float_],
        low: NDArray[np.float_],
        close: NDArray[np.float_],
        volume: Optional[NDArray[np.float_]],
        vwap: Optional[NDArray[np.float_]],
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
        indicator_syms: Collection[IndicatorSymbol],
        cache_date_fields: Optional[CacheDateFields],
        disable_parallel: bool,
    ) -> dict[IndicatorSymbol, pd.Series]:
        """Computes indicator data for the provided
        :class:`pybroker.common.IndicatorSymbol` pairs.

        Args:
            df: :class:`pandas.DataFrame` used to compute the indicator values.
            indicator_syms: ``Collection`` of
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
        indicator_syms: Collection[IndicatorSymbol],
        cache_date_fields: Optional[CacheDateFields],
    ) -> tuple[dict[IndicatorSymbol, pd.Series], Collection[IndicatorSymbol]]:
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
            data = scope.indicator_cache.get(repr(cache_key))
            scope.logger.debug_get_indicator_cache(cache_key)
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
