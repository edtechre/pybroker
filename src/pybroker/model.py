"""Contains model related functionality."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import functools
import pandas as pd
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
from pybroker.parallel import parallel
from pybroker.scope import StaticScope
from pybroker.timeframe import (
    TimeframeData,
    TimeframeInterval,
    build_compressed_symbol_df,
    model_timeframe_name,
    normalize_timeframe_interval,
    parse_model_timeframe_name,
    slice_compressed_df_by_dates,
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
    ):
        self.name = name
        self.indicators = tuple(indicator_names)
        self._input_data_fn = input_data_fn
        self._predict_fn = predict_fn
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

    def timeframe(self, interval: TimeframeInterval) -> "TimeframeModelSource":
        """Returns a timeframe-bound variant trained on compressed bars."""
        return TimeframeModelSource(self, interval)


class TimeframeModelSource:
    """Lightweight wrapper binding a model source to a compression timeframe."""

    def __init__(self, base: ModelSource, interval: TimeframeInterval):
        self.base = base
        self.interval = normalize_timeframe_interval(interval)
        self.name = model_timeframe_name(base.name, self.interval)
        self.indicators = base.indicators
        self.pooled = base.pooled

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"TimeframeModelSource({self.base.name!r}, {self.interval!r})"


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
    ):
        super().__init__(
            name, indicator_names, input_data_fn, predict_fn, pooled, kwargs
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
    ):
        super().__init__(
            name, indicator_names, input_data_fn, predict_fn, pooled, kwargs
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
        input_data_fn: :class:`Callable[[DataFrame], DataFrame]` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data.
        predict_fn: :class:`Callable[[Model, DataFrame], ndarray]` that
            overrides calling the model's default ``predict`` function. If set,
            ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data.
        pretrained: If ``True``, then ``fn`` is used to load and return a
            pre-trained model. If ``False``, ``fn`` is used to train and return
            a new model. Defaults to ``False``.
        pooled: If ``True``, the model is trained once per execution using
            combined multi-symbol data. Defaults to ``False``.
        \**kwargs: Additional arguments to pass to ``fn``.

    Returns:
        :class:`.ModelSource` instance.
    """
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
    train_data: pd.DataFrame
    test_data: pd.DataFrame


PooledTrainResult = tuple[str, frozenset[str], Any, Optional[tuple[str]]]
SymTrainResult = tuple[ModelSymbol, Any, Optional[tuple[str]]]
PooledTrainerReturn = tuple[Literal["pooled"], PooledTrainResult]
SymTrainerReturn = tuple[Literal["sym"], SymTrainResult]
TrainerReturn = Union[PooledTrainerReturn, SymTrainerReturn]


def _infer_input_cols(
    train_data: pd.DataFrame, pooled: bool, indicators: tuple[str, ...]
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


def _parse_model_result(
    model_result: Union[Any, tuple[Any, Iterable[str]]],
    train_data: pd.DataFrame,
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
    sym_train_data: pd.DataFrame,
    sym_test_data: pd.DataFrame,
) -> SymTrainResult:
    model_name, sym = model_sym
    model_result = source(sym, sym_train_data, sym_test_data)
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
    pooled_train_data: pd.DataFrame,
    pooled_test_data: pd.DataFrame,
) -> PooledTrainResult:
    model_result = source.train_pooled(pooled_train_data, pooled_test_data)
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
            if token is not None:
                continue
            source = scope.get_model_source(base_name)
            if not isinstance(source, ModelTrainer) or not source.pooled:
                raise TypeError(
                    f"ModelSource {model_name!r} is not a pooled ModelTrainer."
                )
            pooled_train_data, pooled_test_data = self._prepare_pooled_data(
                symbols,
                train_data,
                test_data,
                indicator_data,
                source,
                train_dates,
                test_dates,
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
                        f".timeframe({token!r})."
                    )
                if source.pooled:
                    raise ValueError(
                        f"Pooled model {base_name!r} does not support "
                        f".timeframe({token!r})."
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
                    )
                )
            elif isinstance(source, ModelTrainer):
                if source.pooled:
                    continue
                sym_train_data = self._slice_by_symbol(sym, train_data)
                sym_test_data = self._slice_by_symbol(sym, test_data)
                for ind_name in source.indicators:
                    ind_series = indicator_data[IndicatorSymbol(ind_name, sym)]
                    if not sym_train_data.empty:
                        sym_train_data[ind_name] = ind_series[
                            ind_series.index.isin(train_dates)
                        ].values
                    if not sym_test_data.empty:
                        sym_test_data[ind_name] = ind_series[
                            ind_series.index.isin(test_dates)
                        ].values
            else:
                sym_train_data = pd.DataFrame()
                sym_test_data = pd.DataFrame()
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
                        f".timeframe({token!r})."
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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sym_col = DataCol.SYMBOL.value
        date_col = DataCol.DATE.value
        pooled_train = train_data[train_data[sym_col].isin(symbols)].copy()
        pooled_test = test_data[test_data[sym_col].isin(symbols)].copy()
        if not pooled_train.empty:
            pooled_train = pooled_train.sort_values([sym_col, date_col])
        if not pooled_test.empty:
            pooled_test = pooled_test.sort_values([sym_col, date_col])
        for sym in symbols:
            for ind_name in source.indicators:
                ind_series = indicator_data[IndicatorSymbol(ind_name, sym)]
                train_mask = pooled_train[sym_col] == sym
                if train_mask.any():
                    sym_train_dates = pooled_train.loc[train_mask, date_col]
                    pooled_train.loc[train_mask, ind_name] = ind_series[
                        ind_series.index.isin(sym_train_dates)
                    ].values
                test_mask = pooled_test[sym_col] == sym
                if test_mask.any():
                    sym_test_dates = pooled_test.loc[test_mask, date_col]
                    pooled_test.loc[test_mask, ind_name] = ind_series[
                        ind_series.index.isin(sym_test_dates)
                    ].values
        return pooled_train, pooled_test

    def _slice_by_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.loc[df[DataCol.SYMBOL.value] == symbol]
            .drop(columns=DataCol.SYMBOL.value)
            .sort_values(DataCol.DATE.value)
        )

    def _prepare_timeframe_symbol_data(
        self,
        symbol: str,
        token: TimeframeInterval,
        train_dates: Collection,
        test_dates: Collection,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        source: ModelSource,
        timeframe_data: TimeframeData,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        scope = StaticScope.instance()
        key = (symbol, token)
        if key not in timeframe_data.compressed:
            raise ValueError(
                f"Timeframe {token!r} data not found for {symbol!r}."
            )
        compressed = timeframe_data.compressed[key]
        full_df = build_compressed_symbol_df(
            symbol,
            token,
            compressed,
            indicator_data,
            source.indicators,
            scope.custom_data_cols,
        )
        sym_train_data = slice_compressed_df_by_dates(full_df, train_dates)
        sym_test_data = slice_compressed_df_by_dates(full_df, test_dates)
        return sym_train_data, sym_test_data

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
        uncached_model_syms = []
        pooled_groups_by_model_sym: dict[ModelSymbol, frozenset[str]] = {}
        for (model_name, _), symbols in pooled_model_groups.items():
            for sym in symbols:
                pooled_groups_by_model_sym[ModelSymbol(model_name, sym)] = (
                    symbols
                )
        for model_sym in model_syms:
            if model_sym in pooled_groups_by_model_sym:
                symbols = pooled_groups_by_model_sym[model_sym]
                group_cached = True
                for sym in symbols:
                    group_model_sym = ModelSymbol(model_sym.model_name, sym)
                    cache_key = ModelCacheKey(
                        symbol=group_model_sym.symbol,
                        model_name=group_model_sym.model_name,
                        **asdict(cache_date_fields),
                    )
                    scope.logger.debug_get_model_cache(cache_key)
                    cached_data = scope.model_cache.get(repr(cache_key))
                    if cached_data is None:
                        group_cached = False
                        break
                if group_cached:
                    for sym in symbols:
                        group_model_sym = ModelSymbol(
                            model_sym.model_name, sym
                        )
                        if group_model_sym in models:
                            continue
                        cache_key = ModelCacheKey(
                            symbol=group_model_sym.symbol,
                            model_name=group_model_sym.model_name,
                            **asdict(cache_date_fields),
                        )
                        cached_data = scope.model_cache.get(repr(cache_key))
                        input_cols = None
                        if isinstance(cached_data, CachedModel):
                            model = cached_data.model
                            input_cols = cached_data.input_cols
                        else:
                            model = cached_data
                        source = scope.get_model_source(
                            group_model_sym.model_name
                        )
                        models[group_model_sym] = TrainedModel(
                            name=group_model_sym.model_name,
                            instance=model,
                            predict_fn=source._predict_fn,
                            input_cols=input_cols,
                        )
                    continue
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
