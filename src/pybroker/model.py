"""Contains model related functionality."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import functools
import numpy as np
import pandas as pd
from numba import njit
from pybroker.cache import CacheDateFields, ModelCacheKey
from pybroker.common import (
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    TrainedModel,
    get_unique_sorted_dates,
    to_datetime,
)
from pybroker.indicator import Indicator
from pybroker.timeseries import (
    LagSeriesCache,
    ModelInput,
    apply_lags_to_model_input,
    apply_lags_to_model_input_pooled,
    merge_lag_series_cache_from_store,
    merge_timeframe_lag_series_cache,
    model_input_from_arrays,
    model_input_to_dataframe,
)
from pybroker.parallel import parallel
from pybroker.scope import (
    StaticScope,
    SymbolArrayStore,
    merge_symbol_array_stores,
    symbol_array_store_from_frame,
)
from pybroker.timeframe import (
    TimeframeData,
    TimeframeInterval,
    build_compressed_symbol_arrays,
    parse_model_timeframe_name,
    format_timeframe_interval,
    slice_arrays_by_dates,
)
from dataclasses import asdict
from datetime import datetime
from joblib import delayed
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Union,
    cast,
)


@njit(cache=True)
def _indicator_values_for_dates_njit(
    ind_dates: NDArray,
    values: NDArray[np.float64],
    dates: NDArray,
) -> NDArray[np.float64]:
    """Aligns indicator values to ``dates`` via batched sorted datetime search."""
    n = len(dates)
    result = np.full(n, np.nan, dtype=np.float64)
    pos = np.searchsorted(ind_dates, dates)
    m = len(ind_dates)
    for i in range(n):
        p = pos[i]
        if p < m and ind_dates[p] == dates[i]:
            result[i] = values[p]
    return result


def _indicator_values_for_dates(
    ind_series: pd.Series, dates: np.ndarray
) -> NDArray[np.float64]:
    """Aligns indicator values to ``dates`` without pandas indexing."""
    if len(dates) == 0:
        return np.array([], dtype=np.float64)
    values = ind_series.to_numpy(dtype=np.float64, copy=False)
    index = ind_series.index
    if getattr(index, "is_monotonic_increasing", False):
        ind_dates = index.to_numpy(dtype="datetime64[ns]", copy=False)
        pos = np.searchsorted(ind_dates, dates)
        valid = pos < len(ind_dates)
        matched = np.zeros(len(dates), dtype=bool)
        matched[valid] = ind_dates[pos[valid]] == dates[valid]
        result = np.full(len(dates), np.nan, dtype=np.float64)
        result[matched] = values[pos[matched]]
        return result
    positions = ind_series.index.get_indexer(dates)
    result = np.full(len(dates), np.nan, dtype=np.float64)
    valid = positions >= 0
    result[valid] = values[positions[valid]]
    return result


def _model_input_columns(
    indicators: tuple[str, ...],
    available: frozenset[str],
    *,
    pooled: bool = False,
) -> tuple[str, ...]:
    """Returns ordered model input columns present in ``available``."""
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    scope = StaticScope.instance()
    columns: list[str] = [sym_col, date_col] if pooled else [date_col]
    for col in scope.all_data_cols:
        if col in (sym_col, date_col) or col not in available:
            continue
        columns.append(col)
    for ind_name in indicators:
        columns.append(ind_name)
    return tuple(dict.fromkeys(columns))


def _empty_model_input(
    columns: tuple[str, ...], *, pooled: bool
) -> ModelInput:
    sym_col = DataCol.SYMBOL.value
    arrays = {col: np.array([], dtype=np.float64) for col in columns}
    if pooled and sym_col in arrays:
        arrays[sym_col] = np.array([], dtype=object)
    return model_input_from_arrays(
        columns, arrays, np.array([], dtype="datetime64[ns]")
    )


def _symbol_model_input_from_store(
    store: SymbolArrayStore,
    symbol: str,
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicators: tuple[str, ...],
) -> ModelInput:
    """Builds per-symbol :class:`ModelInput` from a :class:`SymbolArrayStore`."""
    date_col = DataCol.DATE.value
    if symbol not in store.sym_arrays:
        available: frozenset[str] = frozenset()
        for sym_data in store.sym_arrays.values():
            available |= frozenset(sym_data.keys())
        columns = _model_input_columns(indicators, available, pooled=False)
        return _empty_model_input(columns, pooled=False)
    sym_data = store.sym_arrays[symbol]
    available = frozenset(sym_data.keys())
    columns_tuple = _model_input_columns(indicators, available, pooled=False)
    dates_arr = sym_data.get(date_col)
    if dates_arr is None or len(dates_arr) == 0:
        return _empty_model_input(columns_tuple, pooled=False)
    dates = np.asarray(dates_arr, dtype="datetime64[ns]")
    arrays: dict[str, NDArray] = {date_col: dates}
    for col in columns_tuple:
        if col == date_col:
            continue
        if col in sym_data:
            arrays[col] = sym_data[col]
        elif col in indicators:
            arrays[col] = _indicator_values_for_dates(
                indicator_data[IndicatorSymbol(col, symbol)], dates
            )
    return model_input_from_arrays(columns_tuple, arrays, dates)


def _pooled_model_input_from_store(
    store: SymbolArrayStore,
    symbols: frozenset[str],
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicators: tuple[str, ...],
) -> ModelInput:
    """Builds pooled multi-symbol :class:`ModelInput` from a store."""
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    available: set[str] = set()
    for sym in symbols:
        if sym in store.sym_arrays:
            available.update(store.sym_arrays[sym].keys())
    columns_tuple = _model_input_columns(
        indicators, frozenset(available), pooled=True
    )
    sym_parts: list[NDArray] = []
    date_parts: list[NDArray] = []
    col_parts: dict[str, list[NDArray]] = {
        col: [] for col in columns_tuple if col not in (sym_col, date_col)
    }
    has_rows = False
    for sym in symbols:
        if sym not in store.sym_arrays:
            continue
        sym_data = store.sym_arrays[sym]
        dates_arr = sym_data.get(date_col)
        if dates_arr is None or len(dates_arr) == 0:
            continue
        has_rows = True
        dates = np.asarray(dates_arr, dtype="datetime64[ns]")
        n = len(dates)
        sym_parts.append(np.full(n, sym, dtype=object))
        date_parts.append(dates)
        for col in col_parts:
            if col in sym_data:
                col_parts[col].append(sym_data[col])
            elif col in indicators:
                col_parts[col].append(
                    _indicator_values_for_dates(
                        indicator_data[IndicatorSymbol(col, sym)], dates
                    )
                )
    if not has_rows:
        return _empty_model_input(columns_tuple, pooled=True)
    sym_vals = np.concatenate(sym_parts)
    dates = np.concatenate(date_parts)
    order = np.lexsort((dates, sym_vals))
    sym_vals = sym_vals[order]
    dates = dates[order]
    arrays: dict[str, NDArray] = {sym_col: sym_vals, date_col: dates}
    for col, parts in col_parts.items():
        arrays[col] = np.concatenate(parts)[order]
    return model_input_from_arrays(columns_tuple, arrays, dates)


def _symbol_model_input(
    symbol: str,
    df: pd.DataFrame,
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicators: tuple[str, ...],
) -> ModelInput:
    """Builds per-symbol :class:`ModelInput` via boolean masks."""
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    if df.empty:
        return ModelInput((), {}, np.array([], dtype="datetime64[ns]"))
    available = frozenset(df.columns)
    columns_tuple = _model_input_columns(indicators, available, pooled=False)
    sym_arr = df[sym_col].to_numpy()
    mask = sym_arr == symbol
    if not mask.any():
        return _empty_model_input(columns_tuple, pooled=False)
    rows = np.flatnonzero(mask)
    dates = df[date_col].to_numpy(copy=False)[rows]
    order = np.argsort(dates)
    rows = rows[order]
    dates = dates[order]
    arrays: dict[str, NDArray] = {date_col: dates}
    for col in columns_tuple:
        if col == date_col:
            continue
        if col in df.columns:
            arrays[col] = df[col].to_numpy(copy=False)[rows]
        elif col in indicators:
            arrays[col] = _indicator_values_for_dates(
                indicator_data[IndicatorSymbol(col, symbol)], dates
            )
    return model_input_from_arrays(columns_tuple, arrays, dates)


def _pooled_model_input(
    df: pd.DataFrame,
    symbols: frozenset[str],
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicators: tuple[str, ...],
) -> ModelInput:
    """Builds pooled multi-symbol :class:`ModelInput` without frame copies."""
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    if df.empty:
        available: frozenset[str] = frozenset()
        columns = _model_input_columns(indicators, available, pooled=True)
        return _empty_model_input(columns, pooled=True)
    available = frozenset(df.columns)
    columns_tuple = _model_input_columns(indicators, available, pooled=True)
    sym_arr = df[sym_col].to_numpy()
    mask = np.isin(sym_arr, tuple(symbols))
    if not mask.any():
        return _empty_model_input(columns_tuple, pooled=True)
    rows = np.flatnonzero(mask)
    dates = df[date_col].to_numpy(copy=False)[rows]
    sym_vals = sym_arr[rows]
    order = np.lexsort((dates, sym_vals))
    rows = rows[order]
    dates = dates[order]
    sym_vals = sym_vals[order]
    arrays: dict[str, NDArray] = {sym_col: sym_vals, date_col: dates}
    for col in columns_tuple:
        if col in (sym_col, date_col):
            continue
        if col in df.columns:
            arrays[col] = df[col].to_numpy(copy=False)[rows]
        elif col in indicators:
            ind_values = np.empty(len(rows), dtype=np.float64)
            for sym in symbols:
                sym_mask = sym_vals == sym
                if not sym_mask.any():
                    continue
                ind_values[sym_mask] = _indicator_values_for_dates(
                    indicator_data[IndicatorSymbol(col, sym)],
                    dates[sym_mask],
                )
            arrays[col] = ind_values
    return model_input_from_arrays(columns_tuple, arrays, dates)


def _history_store(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *,
    train_store: Optional[SymbolArrayStore] = None,
    test_store: Optional[SymbolArrayStore] = None,
) -> SymbolArrayStore:
    if train_store is not None and test_store is not None:
        if train_data.empty:
            return test_store
        if test_data.empty:
            return train_store
        return merge_symbol_array_stores(train_store, test_store)
    if train_store is not None:
        if test_data.empty:
            return train_store
        return merge_symbol_array_stores(
            train_store, symbol_array_store_from_frame(test_data)
        )
    if test_store is not None:
        if train_data.empty:
            return test_store
        return merge_symbol_array_stores(
            symbol_array_store_from_frame(train_data), test_store
        )
    if train_data.empty:
        return symbol_array_store_from_frame(test_data)
    if test_data.empty:
        return symbol_array_store_from_frame(train_data)
    return merge_symbol_array_stores(
        symbol_array_store_from_frame(train_data),
        symbol_array_store_from_frame(test_data),
    )


class ModelSource:
    r"""Base class of a model source. A model source provides a model instance
    either by training one or by loading a pre-trained model.

    Args:
        name: Name of model.
        indicator_names: :class:`Iterable` of names of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: :class:`Callable[[DataFrame], DataFrame]` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data.
        predict_fn: :class:`Callable[[Model, DataFrame], ndarray]` that
            overrides calling the model's default ``predict`` function. If set,
            ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data.
        lags: Number of lag steps to include for each input column inferred
            from training data or returned by ``fn``. Stored as transform
            metadata on model input (see :mod:`pybroker.timeseries`).
        per_bar: If ``True``, ``predict_fn`` is called once per bar with input
            truncated to rows up to and including the current bar.
        pooled: If ``True``, the model is trained once per execution using
            combined multi-symbol data. Defaults to ``False``.
        kwargs: ``dict`` of additional kwargs.
    """

    def __init__(
        self,
        name: str,
        indicator_names: Iterable[str],
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]],
        pooled: bool,
        kwargs: dict[str, Any],
        lags: Optional[int] = None,
        per_bar: bool = False,
    ):
        self.name = name
        self.indicators = tuple(indicator_names)
        self._input_data_fn = input_data_fn
        self._predict_fn = predict_fn
        self.lags = lags
        self.per_bar = per_bar
        self.pooled = pooled
        self._kwargs = kwargs

    def prepare_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares a :class:`pandas.DataFrame` of input data for passing to a
        model when making predictions. If set, the ``input_data_fn``
        is used to preprocess the input data. If ``False``, then indicator
        columns in ``df`` are used as input features.
        """
        if df.empty:
            return df
        if self._input_data_fn is None:
            df_cols = frozenset(df.columns)
            for ind_name in self.indicators:
                if ind_name not in df_cols:
                    raise ValueError(
                        f"Indicator {ind_name!r} not found in DataFrame."
                    )
            return df[[*self.indicators]]
        return self._input_data_fn(df)


class ModelLoader(ModelSource):
    r"""Loads a pre-trained model.

    Args:
        name: Name of model.
        load_fn: ``Callable[[symbol: str, train_start_date: datetime,
            train_end_date: datetime, ...], DataFrame]`` used to load and
            return a pre-trained model. This is expected to
            return either a trained model instance, or a tuple containing a
            trained model instance and a :class:`Iterable` of column names to
            to be used as input for the model when making predictions.
        indicator_names: :class:`Iterable` of names of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: :class:`Callable[[DataFrame], DataFrame]` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data.
        predict_fn: :class:`Callable[[Model, DataFrame], ndarray]` that
            overrides calling the model's default ``predict`` function. If set,
            ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data.
        pooled: If ``True``, the model is trained once per execution using
            combined multi-symbol data. Defaults to ``False``.
        kwargs: ``dict`` of kwargs to pass to ``load_fn``.
    """

    def __init__(
        self,
        name: str,
        load_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]],
        indicator_names: Iterable[str],
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]],
        pooled: bool,
        kwargs: dict[str, Any],
        lags: Optional[int] = None,
        per_bar: bool = False,
    ):
        super().__init__(
            name,
            indicator_names,
            input_data_fn,
            predict_fn,
            pooled,
            kwargs,
            lags=lags,
            per_bar=per_bar,
        )
        self._load_fn = functools.partial(load_fn, **kwargs)

    def __call__(
        self, symbol: str, train_start_date: datetime, train_end_date: datetime
    ) -> Union[Any, tuple[Any, Iterable[str]]]:
        """Loads pre-trained model.

        Args:
            symbol: Ticker symbol for loading the pre-trained model.
            train_start_date: Start date of training window.
            train_end_date: End date of training window.

        Returns:
            Pre-trained model.
        """
        return self._load_fn(symbol, train_start_date, train_end_date)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelLoader({self.name!r}, {self._kwargs})"


class ModelTrainer(ModelSource):
    r"""Trains a model.

    Args:
        name: Name of model.
        train_fn: When ``pooled`` is ``False``, ``Callable[[symbol: str,
            train_data: DataFrame, test_data: DataFrame, ...], DataFrame]``.
            When ``pooled`` is ``True``, ``Callable[[train_data: DataFrame,
            test_data: DataFrame, ...], DataFrame]``. This is expected to
            return either a trained model instance, or a tuple containing a
            trained model instance and a :class:`Iterable` of column names to
            to be used as input for the model when making predictions.
        indicator_names: :class:`Iterable` of names of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: :class:`Callable[[DataFrame], DataFrame]` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data.
        predict_fn: :class:`Callable[[Model, DataFrame], ndarray]` that
            overrides calling the model's default ``predict`` function. If set,
            ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data.
        pooled: If ``True``, the model is trained once per execution using
            combined multi-symbol data. Defaults to ``False``.
        kwargs: ``dict`` of kwargs to pass to ``train_fn``.
    """

    def __init__(
        self,
        name: str,
        train_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]],
        indicator_names: Iterable[str],
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]],
        pooled: bool,
        kwargs: dict[str, Any],
        lags: Optional[int] = None,
        per_bar: bool = False,
    ):
        super().__init__(
            name,
            indicator_names,
            input_data_fn,
            predict_fn,
            pooled,
            kwargs,
            lags=lags,
            per_bar=per_bar,
        )
        self._train_fn = functools.partial(train_fn, **kwargs)

    def __call__(
        self, symbol: str, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> Union[Any, tuple[Any, Iterable[str]]]:
        """Trains model per symbol.

        Args:
            symbol: Ticker symbol of model (models are trained per symbol).
            train_data: Train data.
            test_data: Test data.

        Returns:
            Trained model.
        """
        return self._train_fn(symbol, train_data, test_data)

    def train_pooled(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> Union[Any, tuple[Any, Iterable[str]]]:
        """Trains model using combined multi-symbol data.

        Args:
            train_data: Train data containing a ``symbol`` column.
            test_data: Test data containing a ``symbol`` column.

        Returns:
            Trained model.
        """
        return self._train_fn(train_data, test_data)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelTrainer({self.name!r}, {self._kwargs})"


def model(
    name: str,
    fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]],
    indicators: Optional[Iterable[Indicator]] = None,
    lags: Optional[int] = None,
    per_bar: bool = False,
    input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]] = None,
    pretrained: bool = False,
    pooled: bool = False,
    **kwargs,
) -> ModelSource:
    r"""Creates a :class:`.ModelSource` instance and registers it globally with
    ``name``.

    Args:
        name: Name for referencing the model globally.
        fn: :class:`Callable` used to either train or load a model instance. If
            for training with ``pooled=False``, then ``fn`` has signature
            ``Callable[[symbol: str, train_data: DataFrame, test_data:
            DataFrame, ...], DataFrame]``. If for training with
            ``pooled=True``, then ``fn`` has signature ``Callable[[train_data:
            DataFrame, test_data: DataFrame, ...], DataFrame]`` where both
            frames contain a ``symbol`` column with data for all symbols in the
            execution. If for loading, then ``fn`` has signature
            ``Callable[[symbol: str, train_start_date: datetime,
            train_end_date: datetime, ...], DataFrame]``. This is expected to
            return either a trained model instance, or a tuple containing a
            trained model instance and a :class:`Iterable` of column names to
            to be used as input for the model when making predictions. When
            only a model instance is returned, columns from the training
            DataFrame are used for prediction. For pooled models, the
            ``symbol`` column is omitted from inferred prediction columns.
        indicators: :class:`Iterable` of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        lags: Number of lag steps to include for each input column inferred
            from training data or returned by ``fn``. Stored as transform
            metadata on model input (see :mod:`pybroker.timeseries`).
        per_bar: If ``True``, ``predict_fn`` is called once per bar with input
            truncated to rows up to and including the current bar. Requires
            ``predict_fn``.
        input_data_fn: :class:`Callable[[DataFrame], DataFrame]` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data.
        predict_fn: :class:`Callable[[Model, DataFrame], ndarray]` that
            overrides calling the model's default ``predict`` function. If set,
            ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data. When
            ``per_bar=True``, ``predict_fn`` receives input rows up to and
            including the current bar and must return a scalar prediction.
        pretrained: If ``True``, then ``fn`` is used to load and return a
            pre-trained model. If ``False``, ``fn`` is used to train and return
            a new model. Defaults to ``False``.
        pooled: If ``True``, the model is trained once per execution using
            combined multi-symbol data. Defaults to ``False``.
        \**kwargs: Additional arguments to pass to ``fn``.

    Returns:
        :class:`.ModelSource` instance.
    """
    if lags is not None:
        if not isinstance(lags, int) or lags <= 0:
            raise ValueError("lags must be a positive integer.")
    if per_bar and pooled:
        raise ValueError("per_bar=True is not supported with pooled=True.")
    if per_bar and predict_fn is None:
        raise ValueError("per_bar=True requires predict_fn to be set.")
    scope = StaticScope.instance()
    indicator_names = (
        tuple(sorted(set(ind.name for ind in indicators)))
        if indicators is not None
        else tuple()
    )
    if pretrained:
        loader = ModelLoader(
            name=name,
            load_fn=fn,
            indicator_names=indicator_names,
            input_data_fn=input_data_fn,
            predict_fn=predict_fn,
            pooled=pooled,
            kwargs=kwargs,
            lags=lags,
            per_bar=per_bar,
        )
        scope.set_model_source(loader)
        return loader
    else:
        trainer = ModelTrainer(
            name=name,
            train_fn=fn,
            indicator_names=indicator_names,
            input_data_fn=input_data_fn,
            predict_fn=predict_fn,
            pooled=pooled,
            kwargs=kwargs,
            lags=lags,
            per_bar=per_bar,
        )
        scope.set_model_source(trainer)
        return trainer


class CachedModel(NamedTuple):
    """Stores cached model data.

    Attributes:
        model: Trained model instance.
        input_cols: Names of the columns to be used as input for the model when
            making predictions.
    """

    model: Any
    input_cols: Optional[tuple[str]]


class _TrainerTask(NamedTuple):
    pooled: bool
    source: ModelTrainer
    model_name: str
    symbols: frozenset[str]
    model_sym: Optional[ModelSymbol]
    train_data: ModelInput
    test_data: ModelInput


PooledTrainResult = tuple[str, frozenset[str], Any, Optional[tuple[str]]]
SymTrainResult = tuple[ModelSymbol, Any, Optional[tuple[str]]]
PooledTrainerReturn = tuple[Literal["pooled"], PooledTrainResult]
SymTrainerReturn = tuple[Literal["sym"], SymTrainResult]
TrainerReturn = Union[PooledTrainerReturn, SymTrainerReturn]


def _infer_input_cols(
    train_data: ModelInput, pooled: bool, indicators: tuple[str, ...]
) -> tuple[str, ...]:
    data_cols = {col.value for col in DataCol}
    cols = [
        col
        for col in train_data.columns
        if col in data_cols or col in indicators
    ]
    if pooled:
        symbol_col = DataCol.SYMBOL.value
        cols = [col for col in cols if col != symbol_col]
    return tuple(cols)


def _lag_feature_cols(
    train_data: ModelInput, pooled: bool, indicators: tuple[str, ...]
) -> tuple[str, ...]:
    date_col = DataCol.DATE.value
    return tuple(
        col
        for col in _infer_input_cols(train_data, pooled, indicators)
        if col != date_col
    )


def _parse_model_result(
    model_result: Union[Any, tuple[Any, Iterable[str]]],
    train_data: ModelInput,
    pooled: bool,
    indicators: tuple[str, ...],
) -> tuple[Any, Optional[tuple[str]]]:
    if isinstance(model_result, tuple):
        model = model_result[0]
        input_cols = cast(tuple[str], tuple(model_result[1]))
    else:
        model = model_result
        input_cols = cast(
            tuple[str], _infer_input_cols(train_data, pooled, indicators)
        )
    return model, input_cols


def _train_model_sym(
    source: ModelTrainer,
    model_sym: ModelSymbol,
    sym_train_data: ModelInput,
    sym_test_data: ModelInput,
) -> SymTrainResult:
    model_name, sym = model_sym
    model_result = source(
        sym,
        model_input_to_dataframe(sym_train_data),
        model_input_to_dataframe(sym_test_data),
    )
    model, input_cols = _parse_model_result(
        model_result,
        sym_train_data,
        pooled=False,
        indicators=source.indicators,
    )
    return model_sym, model, input_cols


def _train_model_pooled(
    source: ModelTrainer,
    model_name: str,
    symbols: frozenset[str],
    pooled_train_data: ModelInput,
    pooled_test_data: ModelInput,
) -> PooledTrainResult:
    model_result = source.train_pooled(
        model_input_to_dataframe(pooled_train_data),
        model_input_to_dataframe(pooled_test_data),
    )
    model, input_cols = _parse_model_result(
        model_result,
        pooled_train_data,
        pooled=True,
        indicators=source.indicators,
    )
    return model_name, symbols, model, input_cols


def _run_trainer_task(task: _TrainerTask) -> TrainerReturn:
    if task.pooled:
        return (
            "pooled",
            _train_model_pooled(
                task.source,
                task.model_name,
                task.symbols,
                task.train_data,
                task.test_data,
            ),
        )
    assert task.model_sym is not None
    return (
        "sym",
        _train_model_sym(
            task.source,
            task.model_sym,
            task.train_data,
            task.test_data,
        ),
    )


class ModelsMixin:
    """Mixin implementing model related functionality."""

    def train_models(
        self,
        model_syms: Iterable[ModelSymbol],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        cache_date_fields: CacheDateFields,
        enable_parallel_models: bool = False,
        pooled_model_groups: Optional[
            Mapping[tuple[str, int], frozenset[str]]
        ] = None,
        timeframe_data: Optional[TimeframeData] = None,
        *,
        history_store: Optional[SymbolArrayStore] = None,
        train_store: Optional[SymbolArrayStore] = None,
        test_store: Optional[SymbolArrayStore] = None,
    ) -> dict[ModelSymbol, TrainedModel]:
        """Trains models for the provided :class:`pybroker.common.ModelSymbol`
        pairs.

        Args:
            model_syms: ``Iterable`` of
                :class:`pybroker.common.ModelSymbol` pairs of models to train.
            train_data: :class:`pandas.DataFrame` of training data.
            test_data: :class:`pandas.DataFrame` of test data.
            indicator_data: ``Mapping`` of
                :class:`pybroker.common.IndicatorSymbol` pairs to
                ``pandas.Series`` of :class:`pybroker.indicator.Indicator`
                values.
            cache_date_fields: Date fields used to key cache data.
            enable_parallel_models: If ``True``, :class:`.ModelTrainer` models
                are trained in parallel using multiple processes. Defaults to
                ``False``.
            pooled_model_groups: ``Mapping`` of ``(model_name, execution_id)``
                pairs to ``frozenset[str]`` of symbols for pooled training.
                Defaults to ``None``.

        Returns:
            ``dict`` mapping each :class:`pybroker.common.ModelSymbol` pair
            to a :class:`pybroker.common.TrainedModel`.
        """
        if train_data.empty or not model_syms:
            return {}
        if train_store is None and not train_data.empty:
            train_store = symbol_array_store_from_frame(train_data)
        if test_store is None and not test_data.empty:
            test_store = symbol_array_store_from_frame(test_data)
        history_store = _history_store(
            train_data,
            test_data,
            train_store=train_store,
            test_store=test_store,
        )
        lag_series_cache: LagSeriesCache = {}
        history_dates: dict[str, np.ndarray] = {}
        if pooled_model_groups is None:
            pooled_model_groups = {}
        scope = StaticScope.instance()
        train_dates = get_unique_sorted_dates(train_data[DataCol.DATE.value])
        test_dates = get_unique_sorted_dates(test_data[DataCol.DATE.value])
        scope.logger.train_split_start(train_dates)
        scope.logger.info_train_split_start(model_syms)
        models, uncached_model_syms = self._get_cached_models(
            model_syms, cache_date_fields, pooled_model_groups
        )
        if not uncached_model_syms and not self._has_uncached_pooled_groups(
            model_syms, models, pooled_model_groups
        ):
            scope.logger.loaded_models()
            scope.logger.info_loaded_models(model_syms)
            return models
        if models:
            scope.logger.info_loaded_models(models.keys())
        start_date = to_datetime(train_dates[0])
        end_date = to_datetime(train_dates[-1])
        uncached_model_sym_set = set(uncached_model_syms)
        trainer_tasks: list[_TrainerTask] = []
        loader_syms: list[tuple[ModelLoader, ModelSymbol]] = []
        covered_pooled_model_syms: set[ModelSymbol] = set()

        for (model_name, _), symbols in pooled_model_groups.items():
            group_model_syms = {
                ModelSymbol(model_name, sym) for sym in symbols
            }
            if group_model_syms.issubset(models.keys()):
                continue
            if not group_model_syms & uncached_model_sym_set:
                continue
            base_name, token = parse_model_timeframe_name(model_name)
            source = scope.get_model_source(base_name)
            if not isinstance(source, ModelTrainer) or not source.pooled:
                raise TypeError(
                    f"ModelSource {model_name!r} is not a pooled ModelTrainer."
                )
            if token is not None:
                if timeframe_data is None:
                    raise ValueError(
                        f"Timeframe data required to train model {model_name!r}."
                    )
                pooled_train_data, pooled_test_data = (
                    self._prepare_pooled_timeframe_data(
                        symbols,
                        token,
                        train_dates,
                        test_dates,
                        indicator_data,
                        source,
                        timeframe_data,
                        lag_series_cache,
                    )
                )
            else:
                pooled_train_data, pooled_test_data = (
                    self._prepare_pooled_data(
                        symbols,
                        train_data,
                        test_data,
                        indicator_data,
                        source,
                        train_dates,
                        test_dates,
                        history_store,
                        lag_series_cache,
                        history_dates,
                        train_store=train_store,
                        test_store=test_store,
                    )
                )
            trainer_tasks.append(
                _TrainerTask(
                    pooled=True,
                    source=source,
                    model_name=model_name,
                    symbols=symbols,
                    model_sym=None,
                    train_data=pooled_train_data,
                    test_data=pooled_test_data,
                )
            )
            covered_pooled_model_syms.update(group_model_syms)

        for model_sym in uncached_model_syms:
            if model_sym in models or model_sym in covered_pooled_model_syms:
                continue
            model_name, sym = model_sym
            base_name, token = parse_model_timeframe_name(model_name)
            source = scope.get_model_source(base_name)
            if token is not None:
                if isinstance(source, ModelLoader):
                    raise ValueError(
                        f"Pretrained model {base_name!r} does not support "
                        f"multi-timeframe training on {token!r}."
                    )
                if timeframe_data is None:
                    raise ValueError(
                        f"Timeframe data required to train model {model_name!r}."
                    )
                sym_train_data, sym_test_data = (
                    self._prepare_timeframe_symbol_data(
                        sym,
                        token,
                        train_dates,
                        test_dates,
                        indicator_data,
                        source,
                        timeframe_data,
                        lag_series_cache,
                    )
                )
            elif isinstance(source, ModelTrainer):
                if source.pooled:
                    continue
                if train_store is not None:
                    sym_train_data = _symbol_model_input_from_store(
                        train_store, sym, indicator_data, source.indicators
                    )
                else:
                    sym_train_data = _symbol_model_input(
                        sym, train_data, indicator_data, source.indicators
                    )
                if test_store is not None:
                    sym_test_data = _symbol_model_input_from_store(
                        test_store, sym, indicator_data, source.indicators
                    )
                else:
                    sym_test_data = _symbol_model_input(
                        sym, test_data, indicator_data, source.indicators
                    )
                if source.lags is not None:
                    lag_cols = _lag_feature_cols(
                        sym_train_data,
                        pooled=False,
                        indicators=source.indicators,
                    )
                    merge_lag_series_cache_from_store(
                        lag_series_cache,
                        history_store,
                        (sym,),
                        lag_cols,
                        source.lags,
                        history_dates,
                    )
                    apply_lags_to_model_input(
                        sym_train_data,
                        lag_cols,
                        source.lags,
                        lag_series_cache,
                        sym,
                        history_dates[sym],
                    )
                    apply_lags_to_model_input(
                        sym_test_data,
                        lag_cols,
                        source.lags,
                        lag_series_cache,
                        sym,
                        history_dates[sym],
                    )
                    sym_train_data = sym_train_data.drop_lag_warmup()
            else:
                sym_train_data = ModelInput(
                    (), {}, np.array([], dtype="datetime64[ns]")
                )
                sym_test_data = ModelInput(
                    (), {}, np.array([], dtype="datetime64[ns]")
                )
            if isinstance(source, ModelTrainer):
                trainer_tasks.append(
                    _TrainerTask(
                        pooled=False,
                        source=source,
                        model_name=model_name,
                        symbols=frozenset(),
                        model_sym=model_sym,
                        train_data=sym_train_data,
                        test_data=sym_test_data,
                    )
                )
            elif isinstance(source, ModelLoader):
                if token is not None:
                    raise ValueError(
                        f"Pretrained model {base_name!r} does not support "
                        f"multi-timeframe training on {token!r}."
                    )
                loader_syms.append((source, model_sym))
            else:
                raise TypeError(f"Invalid ModelSource type: {type(source)}")

        trainer_results = self._run_model_trainers(
            trainer_tasks, enable_parallel_models
        )
        for task, trainer_result in zip(trainer_tasks, trainer_results):
            if trainer_result[0] == "pooled":
                _, pooled_result = cast(PooledTrainerReturn, trainer_result)
                model_name, symbols, model, input_cols = pooled_result
                for sym in symbols:
                    model_sym = ModelSymbol(model_name, sym)
                    scope.logger.info_train_model_start(model_sym)
                    models[model_sym] = TrainedModel(
                        name=model_name,
                        instance=model,
                        predict_fn=task.source._predict_fn,
                        input_cols=input_cols,
                        per_bar=task.source.per_bar,
                    )
                    self._set_cached_model(
                        model, input_cols, model_sym, cache_date_fields
                    )
                    scope.logger.info_train_model_completed(model_sym)
            else:
                _, sym_result = cast(SymTrainerReturn, trainer_result)
                model_sym, model, input_cols = sym_result
                model_name, _ = model_sym
                scope.logger.info_train_model_start(model_sym)
                models[model_sym] = TrainedModel(
                    name=model_name,
                    instance=model,
                    predict_fn=task.source._predict_fn,
                    input_cols=input_cols,
                    per_bar=task.source.per_bar,
                )
                self._set_cached_model(
                    model, input_cols, model_sym, cache_date_fields
                )
                scope.logger.info_train_model_completed(model_sym)
        for source, model_sym in loader_syms:
            model_name, sym = model_sym
            scope.logger.info_loaded_model(model_sym)
            model_result = source(sym, start_date, end_date)
            input_cols = None
            if isinstance(model_result, tuple):
                model = model_result[0]
                input_cols = tuple(model_result[1])  # type: ignore[assignment]
            else:
                model = model_result
            models[model_sym] = TrainedModel(
                name=model_name,
                instance=model,
                predict_fn=source._predict_fn,
                input_cols=input_cols,
                per_bar=source.per_bar,
            )
            self._set_cached_model(
                model, input_cols, model_sym, cache_date_fields
            )
        scope.logger.train_split_completed()
        return models

    def _has_uncached_pooled_groups(
        self,
        model_syms: Iterable[ModelSymbol],
        models: Mapping[ModelSymbol, TrainedModel],
        pooled_model_groups: Mapping[tuple[str, int], frozenset[str]],
    ) -> bool:
        uncached_model_sym_set = set(model_syms) - set(models.keys())
        for (model_name, _), symbols in pooled_model_groups.items():
            group_model_syms = {
                ModelSymbol(model_name, sym) for sym in symbols
            }
            if group_model_syms & uncached_model_sym_set:
                return True
        return False

    def _run_model_trainers(
        self,
        trainer_tasks: Collection[_TrainerTask],
        enable_parallel_models: bool,
    ) -> list[TrainerReturn]:
        if enable_parallel_models and len(trainer_tasks) > 1:
            with parallel() as pool:
                return pool(
                    delayed(_run_trainer_task)(task) for task in trainer_tasks
                )
        return [_run_trainer_task(task) for task in trainer_tasks]

    def _prepare_pooled_data(
        self,
        symbols: frozenset[str],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        source: ModelTrainer,
        train_dates: Collection,
        test_dates: Collection,
        history_store: SymbolArrayStore,
        lag_series_cache: LagSeriesCache,
        history_dates: dict[str, np.ndarray],
        *,
        train_store: Optional[SymbolArrayStore] = None,
        test_store: Optional[SymbolArrayStore] = None,
    ) -> tuple[ModelInput, ModelInput]:
        del train_dates, test_dates
        if train_store is not None:
            pooled_train_input = _pooled_model_input_from_store(
                train_store, symbols, indicator_data, source.indicators
            )
        else:
            pooled_train_input = _pooled_model_input(
                train_data, symbols, indicator_data, source.indicators
            )
        if test_store is not None:
            pooled_test_input = _pooled_model_input_from_store(
                test_store, symbols, indicator_data, source.indicators
            )
        else:
            pooled_test_input = _pooled_model_input(
                test_data, symbols, indicator_data, source.indicators
            )
        if source.lags is not None:
            lag_cols = _lag_feature_cols(
                pooled_train_input,
                pooled=True,
                indicators=source.indicators,
            )
            merge_lag_series_cache_from_store(
                lag_series_cache,
                history_store,
                symbols,
                lag_cols,
                source.lags,
                history_dates,
            )
            apply_lags_to_model_input_pooled(
                pooled_train_input,
                lag_cols,
                source.lags,
                lag_series_cache,
                history_dates,
                symbols,
            )
            apply_lags_to_model_input_pooled(
                pooled_test_input,
                lag_cols,
                source.lags,
                lag_series_cache,
                history_dates,
                symbols,
            )
            pooled_train_input = pooled_train_input.drop_lag_warmup()
        return pooled_train_input, pooled_test_input

    def _prepare_pooled_timeframe_data(
        self,
        symbols: frozenset[str],
        token: TimeframeInterval,
        train_dates: Collection,
        test_dates: Collection,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        source: ModelTrainer,
        timeframe_data: TimeframeData,
        lag_series_cache: LagSeriesCache,
    ) -> tuple[ModelInput, ModelInput]:
        sym_col = DataCol.SYMBOL.value
        scope = StaticScope.instance()
        train_parts: dict[str, list[NDArray]] = {}
        test_parts: dict[str, list[NDArray]] = {}
        columns: tuple[str, ...] = ()
        history_dates: dict[str, np.ndarray] = {}
        for sym in symbols:
            key = (sym, token)
            if key not in timeframe_data.compressed:
                raise ValueError(
                    f"Timeframe {token!r} data not found for {sym!r}."
                )
            compressed = timeframe_data.compressed[key]
            sym_columns, arrays, bar_dates = build_compressed_symbol_arrays(
                sym,
                token,
                compressed,
                indicator_data,
                source.indicators,
                scope.custom_data_cols,
            )
            columns = sym_columns + (sym_col,)
            history_dates[sym] = np.asarray(bar_dates, dtype="datetime64[ns]")
            _, train_arrays, train_dates_arr = slice_arrays_by_dates(
                sym_columns,
                arrays,
                bar_dates,
                train_dates,
            )
            _, test_arrays, test_dates_arr = slice_arrays_by_dates(
                sym_columns,
                arrays,
                bar_dates,
                test_dates,
            )
            if train_dates_arr.size:
                for col in sym_columns:
                    train_parts.setdefault(col, []).append(train_arrays[col])
                train_parts.setdefault(sym_col, []).append(
                    np.full(train_dates_arr.size, sym)
                )
            if test_dates_arr.size:
                for col in sym_columns:
                    test_parts.setdefault(col, []).append(test_arrays[col])
                test_parts.setdefault(sym_col, []).append(
                    np.full(test_dates_arr.size, sym)
                )

        def _concat_pooled(
            parts: dict[str, list[NDArray]],
        ) -> ModelInput:
            if not parts or sym_col not in parts:
                return ModelInput((), {}, np.array([], dtype="datetime64[ns]"))
            date_col = DataCol.DATE.value
            pooled: dict[str, NDArray] = {
                col: np.concatenate(arrs) for col, arrs in parts.items()
            }
            order = np.lexsort((pooled[date_col], pooled[sym_col]))
            for col in pooled:
                pooled[col] = pooled[col][order]
            return model_input_from_arrays(columns, pooled, pooled[date_col])

        pooled_train_input = _concat_pooled(train_parts)
        pooled_test_input = _concat_pooled(test_parts)
        if source.lags is not None:
            lag_cols = _lag_feature_cols(
                pooled_train_input,
                pooled=True,
                indicators=source.indicators,
            )
            interval = format_timeframe_interval(token)

            def bars_by_symbol(sym, token=token):
                key = (sym, token)
                if key not in timeframe_data.compressed:
                    return None
                return timeframe_data.compressed[key].bars

            merge_timeframe_lag_series_cache(
                lag_series_cache,
                tuple(symbols),
                lag_cols,
                source.lags,
                interval,
                bars_by_symbol,
            )
            apply_lags_to_model_input_pooled(
                pooled_train_input,
                lag_cols,
                source.lags,
                lag_series_cache,
                history_dates,
                symbols,
                interval=interval,
            )
            apply_lags_to_model_input_pooled(
                pooled_test_input,
                lag_cols,
                source.lags,
                lag_series_cache,
                history_dates,
                symbols,
                interval=interval,
            )
            pooled_train_input = pooled_train_input.drop_lag_warmup()
        return pooled_train_input, pooled_test_input

    def _prepare_timeframe_symbol_data(
        self,
        symbol: str,
        token: TimeframeInterval,
        train_dates: Collection,
        test_dates: Collection,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        source: ModelSource,
        timeframe_data: TimeframeData,
        lag_series_cache: LagSeriesCache,
    ) -> tuple[ModelInput, ModelInput]:
        scope = StaticScope.instance()
        key = (symbol, token)
        if key not in timeframe_data.compressed:
            raise ValueError(
                f"Timeframe {token!r} data not found for {symbol!r}."
            )
        compressed = timeframe_data.compressed[key]
        columns, arrays, bar_dates = build_compressed_symbol_arrays(
            symbol,
            token,
            compressed,
            indicator_data,
            source.indicators,
            scope.custom_data_cols,
        )
        _, train_arrays, train_dates = slice_arrays_by_dates(
            columns,
            arrays,
            bar_dates,
            train_dates,
        )
        _, test_arrays, test_dates = slice_arrays_by_dates(
            columns,
            arrays,
            bar_dates,
            test_dates,
        )
        sym_train_data = model_input_from_arrays(
            columns, train_arrays, train_dates
        )
        sym_test_data = model_input_from_arrays(
            columns, test_arrays, test_dates
        )
        if source.lags is not None:
            lag_cols = _lag_feature_cols(
                sym_train_data,
                pooled=False,
                indicators=source.indicators,
            )
            interval = format_timeframe_interval(token)

            def bars_by_symbol(sym, interval=interval, token=token):
                key = (sym, token)
                if key not in timeframe_data.compressed:
                    return None
                return timeframe_data.compressed[key].bars

            merge_timeframe_lag_series_cache(
                lag_series_cache,
                (symbol,),
                lag_cols,
                source.lags,
                interval,
                bars_by_symbol,
            )
            history_dates = np.asarray(compressed.bars.dates)
            apply_lags_to_model_input(
                sym_train_data,
                lag_cols,
                source.lags,
                lag_series_cache,
                symbol,
                history_dates,
                interval,
            )
            apply_lags_to_model_input(
                sym_test_data,
                lag_cols,
                source.lags,
                lag_series_cache,
                symbol,
                history_dates,
                interval,
            )
            sym_train_data = sym_train_data.drop_lag_warmup()
        return sym_train_data, sym_test_data

    def _load_pooled_group_cache(
        self,
        model_name: str,
        symbols: frozenset[str],
        cache_date_fields: CacheDateFields,
    ) -> tuple[bool, dict[ModelSymbol, TrainedModel]]:
        """Loads a fully cached pooled group in a single cache pass."""
        scope = StaticScope.instance()
        model_cache = scope.model_cache
        if model_cache is None:
            return False, {}
        cached_by_sym: dict[ModelSymbol, Union[CachedModel, Any]] = {}
        for sym in symbols:
            group_model_sym = ModelSymbol(model_name, sym)
            cache_key = ModelCacheKey(
                symbol=group_model_sym.symbol,
                model_name=group_model_sym.model_name,
                **asdict(cache_date_fields),
            )
            scope.logger.debug_get_model_cache(cache_key)
            cached_data = model_cache.get(repr(cache_key))
            if cached_data is None:
                return False, {}
            cached_by_sym[group_model_sym] = cached_data
        loaded: dict[ModelSymbol, TrainedModel] = {}
        for group_model_sym, cached_data in cached_by_sym.items():
            input_cols = None
            if isinstance(cached_data, CachedModel):
                model = cached_data.model
                input_cols = cached_data.input_cols
            else:
                model = cached_data
            source = scope.get_model_source(group_model_sym.model_name)
            loaded[group_model_sym] = TrainedModel(
                name=group_model_sym.model_name,
                instance=model,
                predict_fn=source._predict_fn,
                input_cols=input_cols,
                per_bar=source.per_bar,
            )
        return True, loaded

    def _get_cached_models(
        self,
        model_syms: Iterable[ModelSymbol],
        cache_date_fields: CacheDateFields,
        pooled_model_groups: Mapping[tuple[str, int], frozenset[str]],
    ) -> tuple[dict[ModelSymbol, TrainedModel], list[ModelSymbol]]:
        model_syms = sorted(model_syms)
        models: dict[ModelSymbol, TrainedModel] = {}
        scope = StaticScope.instance()
        if scope.model_cache is None:
            return models, model_syms
        uncached_model_syms: list[ModelSymbol] = []
        pooled_groups_by_model_sym: dict[ModelSymbol, frozenset[str]] = {}
        for (model_name, _), symbols in pooled_model_groups.items():
            for sym in symbols:
                pooled_groups_by_model_sym[ModelSymbol(model_name, sym)] = (
                    symbols
                )
        processed_pooled_groups: set[tuple[str, frozenset[str]]] = set()
        for model_sym in model_syms:
            if model_sym in pooled_groups_by_model_sym:
                symbols = pooled_groups_by_model_sym[model_sym]
                group_key = (model_sym.model_name, symbols)
                if group_key in processed_pooled_groups:
                    continue
                processed_pooled_groups.add(group_key)
                group_cached, loaded = self._load_pooled_group_cache(
                    model_sym.model_name, symbols, cache_date_fields
                )
                if group_cached:
                    models.update(loaded)
                else:
                    uncached_model_syms.append(model_sym)
                continue
            cache_key = ModelCacheKey(
                symbol=model_sym.symbol,
                model_name=model_sym.model_name,
                **asdict(cache_date_fields),
            )
            scope.logger.debug_get_model_cache(cache_key)
            cached_data = scope.model_cache.get(repr(cache_key))
            if cached_data is not None:
                input_cols = None
                if isinstance(cached_data, CachedModel):
                    model = cached_data.model
                    input_cols = cached_data.input_cols
                else:
                    model = cached_data
                source = scope.get_model_source(model_sym.model_name)
                models[model_sym] = TrainedModel(
                    name=model_sym.model_name,
                    instance=model,
                    predict_fn=source._predict_fn,
                    input_cols=input_cols,
                    per_bar=source.per_bar,
                )
            else:
                uncached_model_syms.append(model_sym)
        return models, uncached_model_syms

    def _set_cached_model(
        self,
        model: Any,
        input_cols: Optional[tuple[str]],
        model_sym: ModelSymbol,
        cache_date_fields: CacheDateFields,
    ):
        scope = StaticScope.instance()
        if scope.model_cache is None:
            return
        cache_key = ModelCacheKey(
            symbol=model_sym.symbol,
            model_name=model_sym.model_name,
            **asdict(cache_date_fields),
        )
        cached_model = CachedModel(model, input_cols)
        scope.logger.debug_set_model_cache(cache_key)
        scope.model_cache.set(repr(cache_key), cached_model)
