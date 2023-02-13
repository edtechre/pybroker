"""Contains model related functionality."""

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

from .cache import CacheDateFields, ModelCacheKey
from .common import (
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    TrainedModel,
)
from .indicator import Indicator
from .scope import StaticScope
from dataclasses import asdict
from numpy.typing import NDArray
from typing import Any, Callable, Collection, Iterable, Mapping, Optional
import functools
import pandas as pd


class ModelSource:
    r"""Base class of a model source. A model source provides a model instance
    either by training one or by loading a pre-trained model.

    Args:
        name: Name of model.
        indicator_names: :class:`Iterable` of names of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: :class:`Callable` for preprocessing input data passed to
            the model when making predictions. If set, ```input_data_fn``` will
            be called with a :class:`pandas.DataFrame` containing all test
            data.
        predict_fn: :class:`Callable` that overrides calling the model's
            default ``predict`` function.
        kwargs: ``dict`` of additional kwargs.
    """

    def __init__(
        self,
        name: str,
        indicator_names: Iterable[str],
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]],
        kwargs: dict[str, Any],
    ):
        self.name = name
        self.indicators = tuple(indicator_names)
        self._input_data_fn = input_data_fn
        self._predict_fn = predict_fn
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
        load_fn: ``Callable[[symbol: str, ...], DataFrame]`` used to load and
            return a pre-trained model.
        indicator_names: :class:`Iterable` of names of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: :class:`Callable` for preprocessing input data passed to
            the model when making predictions. If set, ```input_data_fn``` will
            be called with a :class:`pandas.DataFrame` containing all test
            data.
        predict_fn: :class:`Callable` that overrides calling the model's
            default ``predict`` function.
        kwargs: ``dict`` of kwargs to pass to ``load_fn``.
    """

    def __init__(
        self,
        name: str,
        load_fn: Callable[..., Any],
        indicator_names: Iterable[str],
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]],
        kwargs: dict[str, Any],
    ):
        super().__init__(
            name, indicator_names, input_data_fn, predict_fn, kwargs
        )
        self._load_fn = functools.partial(load_fn, **kwargs)

    def __call__(self, symbol: str) -> Any:
        """Loads pre-trained model.

        Args:
            symbol: Ticker symbol for loading the pre-trained model.

        Returns:
            Pre-trained model.
        """
        return self._load_fn(symbol)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelLoader({self.name!r}, {self._kwargs})"


class ModelTrainer(ModelSource):
    r"""Trains a model.

    Args:
        name: Name of model.
        train_fn: ``Callable[[symbol: str, train_data: DataFrame,
            test_data: DataFrame, ...], DataFrame]`` used to train and return a
            model.
        indicator_names: :class:`Iterable` of names of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: :class:`Callable` for preprocessing input data passed to
            the model when making predictions. If set, ```input_data_fn``` will
            be called with a :class:`pandas.DataFrame` containing all test
            data.
        predict_fn: :class:`Callable` that overrides calling the model's
            default ``predict`` function.
        kwargs: ``dict`` of kwargs to pass to ``train_fn``.
    """

    def __init__(
        self,
        name: str,
        train_fn: Callable[..., Any],
        indicator_names: Iterable[str],
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]],
        kwargs: dict[str, Any],
    ):
        super().__init__(
            name, indicator_names, input_data_fn, predict_fn, kwargs
        )
        self._train_fn = functools.partial(train_fn, **kwargs)

    def __call__(
        self, symbol: str, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> Any:
        """Trains model.

        Args:
            symbol: Ticker symbol of model (models are trained per symbol).
            train_data: Train data.
            test_data: Test data.

        Returns:
            Trained model.
        """
        return self._train_fn(symbol, train_data, test_data)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelTrainer({self.name!r}, {self._kwargs})"


def model(
    name: str,
    fn: Callable[..., Any],
    indicators: Optional[Iterable[Indicator]] = None,
    input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]] = None,
    pretrained: bool = False,
    **kwargs,
) -> ModelSource:
    r"""Creates a :class:`.ModelSource` instance and registers it globally with
    ``name``.

    Args:
        name: Name for referencing the model globally.
        fn: :class:`Callable` used to either train or load a model instance. If
            for training, then ``fn`` has signature ``Callable[[symbol: str,
            train_data: DataFrame, test_data: DataFrame, ...], DataFrame]``.
            If for loading, then ``fn`` has signature
            ``Callable[[symbol: str, ...], DataFrame]``.
        indicators: :class:`Iterable` of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: :class:`Callable` for preprocessing input data passed to
            the model when making predictions. If set, ```input_data_fn``` will
            be called with a :class:`pandas.DataFrame` containing all test
            data.
        predict_fn: :class:`Callable` that overrides calling the model's
            default ``predict`` function.
        pretrained: If ``True``, then ``fn`` is used to load and return a
            pre-trained model. If ``False``, ``fn`` is used to train and return
            a new model. Defaults to ``False``.
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
            kwargs=kwargs,
        )
        scope.set_model_source(trainer)
        return trainer


class ModelsMixin:
    """Mixin implementing model related functionality."""

    def train_models(
        self,
        model_syms: Collection[ModelSymbol],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        cache_date_fields: CacheDateFields,
    ) -> dict[ModelSymbol, TrainedModel]:
        """Trains models for the provided :class:`pybroker.common.ModelSymbol`
        pairs.

        Args:
            model_syms: ``Collection`` of
                :class:`pybroker.common.ModelSymbol` pairs of models to train.
            train_data: :class:`pandas.DataFrame` of training data.
            test_data: :class:`pandas.DataFrame` of test data.
            indicator_data: ``Mapping`` of
                :class:`pybroker.common.IndicatorSymbol` pairs to
                ``pandas.Series`` of :class:`pybroker.indicator.Indicator`
                values.
            cache_date_fields: Date fields used to key cache data.

        Returns:
            ``dict`` mapping each :class:`pybroker.common.ModelSymbol` pair
            to a :class:`pybroker.common.TrainedModel`.
        """
        if train_data.empty or not model_syms:
            return {}
        scope = StaticScope.instance()
        train_dates = train_data[DataCol.DATE.value].unique()
        train_dates.sort()
        test_dates = test_data[DataCol.DATE.value].unique()
        test_dates.sort()
        scope.logger.train_split_start(train_dates)
        scope.logger.info_train_split_start(model_syms)
        models, uncached_model_syms = self._get_cached_models(
            model_syms, cache_date_fields
        )
        if not uncached_model_syms:
            scope.logger.loaded_models()
            scope.logger.info_loaded_models(model_syms)
            return models
        if models:
            scope.logger.info_loaded_models(models.keys())
        for model_sym in uncached_model_syms:
            if model_sym in models:
                continue
            model_name, sym = model_sym
            source = scope.get_model_source(model_name)
            source_type = type(source)
            if source_type == ModelTrainer:
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
                scope.logger.info_train_model_start(model_sym)
                model = source(sym, sym_train_data, sym_test_data)
                scope.logger.info_train_model_completed(model_sym)
            elif source_type == ModelLoader:
                model = source(sym)
                scope.logger.info_loaded_model(model_sym)
            else:
                raise TypeError(f"Invalid ModelSource type: {source_type}")
            models[model_sym] = TrainedModel(
                model_name, model, source._predict_fn
            )
            self._set_cached_model(model, model_sym, cache_date_fields)
        scope.logger.train_split_completed()
        return models

    def _slice_by_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[df[DataCol.SYMBOL.value] == symbol]
        df = df.drop(columns=DataCol.SYMBOL.value)
        return df.sort_values(DataCol.DATE.value)

    def _get_cached_models(
        self,
        model_syms: Collection[ModelSymbol],
        cache_date_fields: CacheDateFields,
    ) -> tuple[dict[ModelSymbol, TrainedModel], Collection[ModelSymbol]]:
        model_syms = sorted(model_syms)
        models: dict[ModelSymbol, TrainedModel] = {}
        scope = StaticScope.instance()
        if scope.model_cache is None:
            return models, model_syms
        uncached_model_syms = []
        for model_sym in model_syms:
            cache_key = ModelCacheKey(
                symbol=model_sym.symbol,
                model_name=model_sym.model_name,
                **asdict(cache_date_fields),
            )
            model = scope.model_cache.get(repr(cache_key))
            scope.logger.debug_get_model_cache(cache_key)
            if model is not None:
                source = scope.get_model_source(model_sym.model_name)
                models[model_sym] = TrainedModel(
                    model_sym.model_name, model, source._predict_fn
                )
            else:
                uncached_model_syms.append(model_sym)
        return models, uncached_model_syms

    def _set_cached_model(
        self,
        model: Any,
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
        scope.model_cache.set(repr(cache_key), model)
