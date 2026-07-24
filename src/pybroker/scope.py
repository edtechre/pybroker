"""Contains scopes that store data and object references used to execute a
:class:`pybroker.strategy.Strategy`.
"""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
from pybroker.common import (
    BarData,
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    PriceType,
    TrainedModel,
    to_decimal,
)
from pybroker.timeseries import (
    LagSeriesCache,
    ModelInput,
    apply_lags_to_model_input,
    apply_prepare_input_data,
    merge_lag_series_cache_from_arrays,
    merge_timeframe_lag_series_cache,
    model_input_to_dataframe,
)
from pybroker.log import Logger
from pybroker.timeframe import (
    TimeframeData,
    TimeframeInterval,
    format_timeframe_interval,
    indicator_timeframe_name,
    model_timeframe_name,
    normalize_timeframe_interval,
    parse_indicator_timeframe_name,
)
from collections import defaultdict
from decimal import Decimal
from diskcache import Cache
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Final,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    cast,
)

_EMPTY_PARAM: Final = object()

# Cached enum accesses. Hot paths (ColumnScope.bar_data_from_data_columns,
# PriceScope.fetch) hit these 10_000+ times per backtest; binding once here
# skips the enum descriptor call per access.
_COL_DATE: Final = DataCol.DATE.value
_COL_OPEN: Final = DataCol.OPEN.value
_COL_HIGH: Final = DataCol.HIGH.value
_COL_LOW: Final = DataCol.LOW.value
_COL_CLOSE: Final = DataCol.CLOSE.value
_COL_VOLUME: Final = DataCol.VOLUME.value
_COL_VWAP: Final = DataCol.VWAP.value
_PRICE_OPEN: Final = PriceType.OPEN
_PRICE_HIGH: Final = PriceType.HIGH
_PRICE_LOW: Final = PriceType.LOW
_PRICE_CLOSE: Final = PriceType.CLOSE
_PRICE_MIDDLE: Final = PriceType.MIDDLE
_PRICE_AVERAGE: Final = PriceType.AVERAGE


class StaticScope:
    """A static registry of data and object references.

    Attributes:
        logger: :class:`pybroker.log.Logger`
        data_source_cache: :class:`diskcache.Cache` that stores data retrieved
            from :class:`pybroker.data.DataSource`.
        data_source_cache_ns: Namespace set for  :attr:`.data_source_cache`.
        indicator_cache: :class:`diskcache.Cache` that stores
            :class:`pybroker.indicator.Indicator` data.
        indicator_cache_ns: Namespace set for :attr:`.indicator_cache`.
        model_cache: :class:`diskcache.Cache` that stores trained models.
        model_cache_ns: Namespace set for :attr:`.model_cache`.
        default_data_cols: Default data columns in :class:`pandas.DataFrame`
            retrieved from a :class:`pybroker.data.DataSource`.
        custom_data_cols: User-defined data columns in
            :class:`pandas.DataFrame` retrieved from a
            :class:`pybroker.data.DataSource`.
    """

    __instance = None

    def __init__(self):
        self.logger = Logger(self)
        self.data_source_cache: Optional[Cache] = None
        self.data_source_cache_ns: str = ""
        self.indicator_cache: Optional[Cache] = None
        self.indicator_cache_ns: str = ""
        self.model_cache: Optional[Cache] = None
        self.model_cache_ns: str = ""
        self._indicators = {}
        self._model_sources = {}
        self.default_data_cols = frozenset(
            (
                DataCol.DATE.value,
                DataCol.OPEN.value,
                DataCol.HIGH.value,
                DataCol.LOW.value,
                DataCol.CLOSE.value,
                DataCol.VOLUME.value,
                DataCol.VWAP.value,
            )
        )
        self.custom_data_cols = set()
        self._cols_frozen: bool = False
        self._params: dict[str, Any] = {}

    def set_indicator(self, indicator):
        """Stores :class:`pybroker.indicator.Indicator` in static scope."""
        self._indicators[indicator.name] = indicator

    def has_indicator(self, name: str) -> bool:
        """Whether :class:`pybroker.indicator.Indicator` is stored in static
        scope.
        """
        return name in self._indicators

    def get_indicator(self, name: str):
        """Retrieves a :class:`pybroker.indicator.Indicator` from static
        scope."""
        if not self.has_indicator(name):
            raise ValueError(f"Indicator {name!r} does not exist.")
        return self._indicators[name]

    def get_indicator_names(self, model_name: str) -> tuple[str]:
        """Returns a ``tuple[str]`` of all
        :class:`pybroker.indicator.Indicator` names that are registered with
        :class:`pybroker.model.ModelSource` having ``model_name``.
        """
        return self._model_sources[model_name].indicators

    def set_model_source(self, source):
        """Stores :class:`pybroker.model.ModelSource` in static scope."""
        self._model_sources[source.name] = source

    def has_model_source(self, name: str) -> bool:
        """Whether :class:`pybroker.model.ModelSource` is stored in static
        scope.
        """
        return name in self._model_sources

    def get_model_source(self, name: str):
        """Retrieves a :class:`pybroker.model.ModelSource` from static
        scope.
        """
        if not self.has_model_source(name):
            raise ValueError(f"ModelSource {name!r} does not exist.")
        return self._model_sources[name]

    def register_custom_cols(self, names: Union[str, Iterable[str]], *args):
        """Registers user-defined column names."""
        self._verify_unfrozen_cols()
        if isinstance(names, str):
            names = (names, *args)
        else:
            names = (*names, *args)
        names = filter(lambda col: col not in self.default_data_cols, names)
        self.custom_data_cols.update(names)

    def unregister_custom_cols(self, names: Union[str, Iterable[str]], *args):
        """Unregisters user-defined column names."""
        self._verify_unfrozen_cols()
        if isinstance(names, str):
            names = (names, *args)
        else:
            names = (*names, *args)
        self.custom_data_cols.difference_update(names)

    @property
    def all_data_cols(self) -> frozenset[str]:
        """All registered data column names."""
        return self.default_data_cols | self.custom_data_cols

    def _verify_unfrozen_cols(self):
        if self._cols_frozen:
            raise ValueError("Cannot modify columns when strategy is running.")

    def freeze_data_cols(self):
        """Prevents additional data columns from being registered."""
        self._cols_frozen = True

    def unfreeze_data_cols(self):
        """Allows additional data columns to be registered if
        :func:`pybroker.scope.StaticScope.freeze_data_cols` was called.
        """
        self._cols_frozen = False

    def param(
        self, name: str, value: Optional[Any] = _EMPTY_PARAM
    ) -> Optional[Any]:
        """Get or set a global parameter."""
        if value is _EMPTY_PARAM:
            return self._params.get(name, None)
        self._params[name] = value
        return value

    def clear_params(self):
        """Clears all global parameters."""
        self._params.clear()

    @classmethod
    def instance(cls) -> "StaticScope":
        """Returns singleton instance."""
        if cls.__instance is None:
            cls.__instance = StaticScope()
        return cls.__instance


def disable_logging():
    """Disables event logging."""
    StaticScope.instance().logger.disable()


def enable_logging():
    """Enables event logging."""
    StaticScope.instance().logger.enable()


def disable_progress_bar():
    """Disables logging a progress bar."""
    StaticScope.instance().logger.disable_progress_bar()


def enable_progress_bar():
    """Enables logging a progress bar."""
    StaticScope.instance().logger.enable_progress_bar()


def register_columns(names: Union[str, Iterable[str]], *args):
    """Registers ``names`` of user-defined data columns."""
    StaticScope.instance().register_custom_cols(names, *args)


def unregister_columns(names: Union[str, Iterable[str]], *args):
    """Unregisters ``names`` of user-defined data columns."""
    StaticScope.instance().unregister_custom_cols(names, *args)


def param(name: str, value: Optional[Any] = _EMPTY_PARAM) -> Optional[Any]:
    """Get or set a global parameter."""
    return StaticScope.instance().param(name, value)


def clear_params():
    """Clears all global parameters."""
    StaticScope.instance().clear_params()


class ColumnScope:
    """Caches and retrieves column data queried from :class:`pandas.DataFrame`.

    Args:
        df: :class:`pandas.DataFrame` containing the column data.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.sort_index()
        self._symbols = frozenset(df.index.get_level_values(0).unique())
        self._sym_cols: dict[str, dict[str, Optional[NDArray]]] = defaultdict(
            dict
        )

    def fetch_dict(
        self,
        symbol: str,
        names: Iterable[str],
        end_index: Optional[int] = None,
    ) -> dict[str, Optional[NDArray]]:
        r"""Fetches a ``dict`` of column data for ``symbol``.

        Args:
            symbol: Ticker symbol to query.
            names: Names of columns to query.
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.

        Returns:
            ``dict`` mapping column names to :class:`numpy.ndarray`\ s of
            column values.
        """
        result: dict[str, Optional[NDArray]] = {}
        if not names:
            return result
        sym_dfs: dict[str, pd.DataFrame] = {}
        for name in names:
            if symbol in self._sym_cols and name in self._sym_cols[symbol]:
                result[name] = self._sym_cols[symbol][name]
                if result[name] is not None:
                    result[name] = result[name][:end_index]  # type: ignore[index]
                continue
            if symbol in sym_dfs:
                sym_df = sym_dfs[symbol]
            else:
                if symbol not in self._symbols:
                    raise ValueError(f"Symbol not found: {symbol}.")
                sym_df = self._df.loc[pd.IndexSlice[symbol, :]].reset_index()
                sym_dfs[symbol] = sym_df
            if name not in sym_df.columns:
                self._sym_cols[symbol][name] = None
                result[name] = None
                continue
            array = sym_df[name].to_numpy(copy=True)
            self._sym_cols[symbol][name] = array
            result[name] = array[:end_index]
        return result

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> Optional[NDArray]:
        """Fetches a :class:`numpy.ndarray` of column data for ``symbol``.

        Args:
            symbol: Ticker symbol to query.
            name: Name of column to query.
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.

        Returns:
            :class:`numpy.ndarray` of column data for every bar until
            ``end_index`` (when specified).
        """
        result = self.fetch_dict(symbol, (name,), end_index)
        return result.get(name, None)

    def bar_data_from_data_columns(
        self, symbol: str, end_index: int
    ) -> BarData:
        """Returns a new :class:`pybroker.common.BarData` instance containing
        column data of default and custom data columns registered with
        :class:`.StaticScope`.

        Args:
            symbol: Ticker symbol to query.
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.
        """
        static_scope = StaticScope.instance()
        default_col_data = self.fetch_dict(
            symbol, static_scope.default_data_cols, end_index
        )
        custom_col_data = self.fetch_dict(
            symbol, static_scope.custom_data_cols, end_index
        )
        return BarData(
            **default_col_data,  # type: ignore[arg-type]
            **custom_col_data,  # type: ignore[arg-type]
        )


class IndicatorScope:
    """Caches and retrieves :class:`pybroker.indicator.Indicator` data.

    Args:
        indicator_data: :class:`Mapping` of
            :class:`pybroker.common.IndicatorSymbol` pairs to ``pandas.Series``
            of :class:`pybroker.indicator.Indicator` values.
        filter_dates: Filters :class:`pybroker.indicator.Indicator` data on
            :class:`Sequence` of dates.
    """

    def __init__(
        self,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        filter_dates: Sequence[np.datetime64],
    ):
        self._indicator_data = indicator_data
        self._filter_dates = filter_dates
        self._sym_inds: dict[IndicatorSymbol, NDArray[np.float64]] = {}

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Fetches :class:`pybroker.indicator.Indicator` data.

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.indicator.Indicator` to query.
            end_index: Truncates the array of
                :class:`pybroker.indicator.Indicator` data returned
                (exclusive). If ``None``, then indicator data is not truncated.

        Returns:
            :class:`numpy.ndarray` of :class:`pybroker.indicator.Indicator`
            data for every bar until ``end_index`` (when specified).
        """
        ind_sym = IndicatorSymbol(name, symbol)
        if ind_sym in self._sym_inds:
            cached = self._sym_inds[ind_sym]
            return cached if end_index is None else cached[:end_index]
        if ind_sym not in self._indicator_data:
            raise ValueError(f"Indicator {name!r} not found for {symbol}.")
        ind_series = self._indicator_data[ind_sym]
        _, token = parse_indicator_timeframe_name(name)
        if token is not None:
            ind_data = ind_series.to_numpy(copy=True)
        else:
            ind_data = ind_series[
                ind_series.index.isin(self._filter_dates)
            ].to_numpy(copy=True)
        self._sym_inds[ind_sym] = ind_data
        return ind_data if end_index is None else ind_data[:end_index]

    def fetch_full(self, symbol: str, name: str) -> NDArray[np.float64]:
        """Fetches the full indicator array without truncation."""
        return self.fetch(symbol, name, end_index=None)


class TimeframeScope:
    """Serves compressed bar and indicator data through alignment maps."""

    def __init__(
        self,
        timeframe_data: TimeframeData,
        ind_scope: IndicatorScope,
        declared_timeframes: frozenset[TimeframeInterval],
        models: Optional[Mapping[ModelSymbol, TrainedModel]] = None,
        test_dates: Optional[Sequence[np.datetime64]] = None,
    ):
        self._timeframe_data = timeframe_data
        self._ind_scope = ind_scope
        self._declared_timeframes = declared_timeframes
        self._models = models or {}
        self._lag_series_cache: LagSeriesCache = {}
        self._test_dates = [] if test_dates is None else test_dates
        self._scope = StaticScope.instance()
        self._bar_cache: dict[
            tuple[str, TimeframeInterval, str], NDArray[Any]
        ] = {}
        self._sym_inputs: dict[ModelSymbol, ModelInput] = {}
        self._sym_preds: dict[ModelSymbol, NDArray] = {}

    def _lag_cols(self, input_cols: tuple[str, ...]) -> tuple[str, ...]:
        date_col = DataCol.DATE.value
        return tuple(col for col in input_cols if col != date_col)

    def _ensure_lag_cache(
        self,
        symbol: str,
        interval: TimeframeInterval,
        lag_cols: tuple[str, ...],
        lags: int,
    ) -> None:
        interval = normalize_timeframe_interval(interval)
        interval_str = format_timeframe_interval(interval)

        def bars_by_symbol(sym, interval_str=interval_str, interval=interval):
            key = (sym, interval)
            if key not in self._timeframe_data.compressed:
                return None
            return self._timeframe_data.compressed[key].bars

        merge_timeframe_lag_series_cache(
            self._lag_series_cache,
            (symbol,),
            lag_cols,
            lags,
            interval_str,
            bars_by_symbol,
        )

    def is_declared(self, interval: TimeframeInterval) -> bool:
        interval = normalize_timeframe_interval(interval)
        return interval in self._declared_timeframes

    def completed_index(
        self, symbol: str, interval: TimeframeInterval, end_index: int
    ) -> int:
        interval = normalize_timeframe_interval(interval)
        key = (symbol, interval)
        if key not in self._timeframe_data.compressed:
            raise ValueError(
                f"Timeframe {interval!r} data not found for {symbol!r}."
            )
        completed = self._timeframe_data.compressed[key].completed
        return int(completed[end_index - 1])

    def fetch_bar(
        self,
        symbol: str,
        interval: TimeframeInterval,
        col: str,
        end_index: int,
    ) -> NDArray[Any]:
        interval = normalize_timeframe_interval(interval)
        cache_key = (symbol, interval, col)
        data: NDArray[Any]
        if cache_key not in self._bar_cache:
            key = (symbol, interval)
            if key not in self._timeframe_data.compressed:
                raise ValueError(
                    f"Timeframe {interval!r} data not found for {symbol!r}."
                )
            bars = self._timeframe_data.compressed[key].bars
            if col == DataCol.DATE.value:
                data = bars.dates
            elif col == DataCol.OPEN.value:
                data = bars.open
            elif col == DataCol.HIGH.value:
                data = bars.high
            elif col == DataCol.LOW.value:
                data = bars.low
            elif col == DataCol.CLOSE.value:
                data = bars.close
            elif col == DataCol.VOLUME.value:
                data = bars.volume
            elif col in bars.custom:
                data = bars.custom[col]
            else:
                raise ValueError(
                    f"Column {col!r} not found for timeframe {interval!r}."
                )
            self._bar_cache[cache_key] = data
        data = self._bar_cache[cache_key]
        idx = self.completed_index(symbol, interval, end_index)
        if idx < 0:
            return np.array([], dtype=data.dtype)
        return data[: idx + 1]

    def fetch_indicator(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_name: str,
        end_index: int,
    ) -> NDArray[np.float64]:
        interval = normalize_timeframe_interval(interval)
        name = indicator_timeframe_name(base_name, interval)
        values = self._ind_scope.fetch_full(symbol, name)
        idx = self.completed_index(symbol, interval, end_index)
        if idx < 0:
            return np.array([], dtype=np.float64)
        return values[: idx + 1]

    def fetch_input(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_model_name: str,
        end_index: int,
    ) -> pd.DataFrame:
        interval = normalize_timeframe_interval(interval)
        model_input = self._prepare_full_input(
            symbol, interval, base_model_name
        )
        idx = self.completed_index(symbol, interval, end_index)
        if idx < 0:
            return model_input_to_dataframe(model_input.slice(0))
        return model_input_to_dataframe(model_input.slice(idx + 1))

    def fetch_preds(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_model_name: str,
        end_index: int,
    ) -> NDArray:
        interval = normalize_timeframe_interval(interval)
        model_sym = ModelSymbol(
            model_timeframe_name(base_model_name, interval), symbol
        )
        trained_model = self._models.get(model_sym)
        if trained_model is None:
            raise ValueError(
                f"Model {base_model_name!r} not found for {symbol}."
            )
        if trained_model.per_bar:
            return self._fetch_preds_per_bar(
                symbol,
                interval,
                base_model_name,
                model_sym,
                trained_model,
                end_index,
            )
        if model_sym not in self._sym_preds:
            input_ = self._prepare_full_input(
                symbol, interval, base_model_name
            )
            if input_.empty() or not input_.columns:
                raise ValueError(
                    f"No input data found for model {base_model_name!r}. "
                    "Consider passing input_data_fn to pybroker#model() if "
                    "custom columns were registered."
                )
            pred = self._run_predict(
                trained_model, model_input_to_dataframe(input_)
            )
            self._sym_preds[model_sym] = pred
        pred = self._sym_preds[model_sym]
        idx = self.completed_index(symbol, interval, end_index)
        if idx < 0:
            return np.array([], dtype=pred.dtype)
        return pred[: idx + 1]

    def _fetch_preds_per_bar(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_model_name: str,
        model_sym: ModelSymbol,
        trained_model: TrainedModel,
        end_index: int,
    ) -> NDArray:
        if model_sym not in self._sym_preds:
            self._sym_preds[model_sym] = np.array([], dtype=np.float64)
        pred = self._sym_preds[model_sym]
        target_len = self.completed_index(symbol, interval, end_index) + 1
        if target_len <= 0:
            return np.array([], dtype=pred.dtype)
        while len(pred) < target_len:
            bar_end_index = len(pred) + 1
            input_ = self.fetch_input(
                symbol, interval, base_model_name, bar_end_index
            )
            scalar = self._run_predict_scalar(trained_model, input_)
            pred = np.append(pred, scalar)
            self._sym_preds[model_sym] = pred
        return pred[:target_len]

    @staticmethod
    def _run_predict(
        trained_model: TrainedModel, input_: pd.DataFrame
    ) -> NDArray:
        if trained_model.predict_fn is not None:
            pred = trained_model.predict_fn(trained_model.instance, input_)
        else:
            predict_fn = getattr(trained_model.instance, "predict", None)
            if predict_fn is not None and callable(predict_fn):
                pred = trained_model.instance.predict(input_)
            else:
                raise ValueError(
                    f"Model instance trained for {trained_model.name!r} "
                    "does not define a predict function. Please pass a "
                    "predict_fn to pybroker.model()."
                )
        pred_arr = np.asarray(pred)
        if pred_arr.ndim == 0:
            return np.array([pred_arr.item()])
        if len(pred_arr.shape) > 1:
            pred_arr = np.squeeze(pred_arr)
        return pred_arr

    @staticmethod
    def _run_predict_scalar(
        trained_model: TrainedModel, input_: pd.DataFrame
    ) -> float:
        pred = TimeframeScope._run_predict(trained_model, input_)
        return float(np.asarray(pred).reshape(-1)[0])

    def _prepare_full_input(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_model_name: str,
    ) -> ModelInput:
        interval = normalize_timeframe_interval(interval)
        model_sym = ModelSymbol(
            model_timeframe_name(base_model_name, interval), symbol
        )
        if model_sym in self._sym_inputs:
            return self._sym_inputs[model_sym]
        if not self._scope.has_model_source(base_model_name):
            raise ValueError(f"Model {base_model_name!r} not found.")
        source = self._scope.get_model_source(base_model_name)
        model_input = self._build_compressed_model_input(
            symbol, interval, source
        )
        if model_sym not in self._models:
            raise ValueError(
                f"Model {base_model_name!r} not found for {symbol}."
            )
        trained_model = self._models[model_sym]
        if trained_model.input_cols is not None:
            for input_col in trained_model.input_cols:
                if input_col not in model_input:
                    raise ValueError(
                        f"Missing column {input_col!r} for input data to "
                        f"model {model_sym.model_name!r}."
                    )
            model_input = model_input.select_columns(trained_model.input_cols)
        if source.lags is not None:
            if trained_model.input_cols is None:
                raise ValueError(
                    f"Model {model_sym.model_name!r} requires input columns "
                    f"from training before applying lags."
                )
            lag_cols = self._lag_cols(trained_model.input_cols)
            self._ensure_lag_cache(symbol, interval, lag_cols, source.lags)
            apply_lags_to_model_input(
                model_input,
                lag_cols,
                source.lags,
                self._lag_series_cache,
                symbol,
                np.asarray(
                    self._timeframe_data.compressed[
                        (symbol, interval)
                    ].bars.dates
                ),
                format_timeframe_interval(interval),
            )
        if not trained_model.input_cols or source._input_data_fn:
            model_input = apply_prepare_input_data(
                model_input, source.prepare_input_data
            )
        self._sym_inputs[model_sym] = model_input
        return model_input

    def _build_compressed_model_input(
        self,
        symbol: str,
        interval: TimeframeInterval,
        source,
    ) -> ModelInput:
        interval = normalize_timeframe_interval(interval)
        key = (symbol, interval)
        if key not in self._timeframe_data.compressed:
            raise ValueError(
                f"Timeframe {interval!r} data not found for {symbol!r}."
            )
        bars = self._timeframe_data.compressed[key].bars
        arrays: dict[str, NDArray[Any]] = {
            DataCol.DATE.value: bars.dates,
            DataCol.OPEN.value: bars.open,
            DataCol.HIGH.value: bars.high,
            DataCol.LOW.value: bars.low,
            DataCol.CLOSE.value: bars.close,
            DataCol.VOLUME.value: bars.volume,
        }
        for col in self._scope.custom_data_cols:
            if col in bars.custom:
                arrays[col] = bars.custom[col]
        for ind_name in source.indicators:
            arrays[ind_name] = self._ind_scope.fetch_full(
                symbol, indicator_timeframe_name(ind_name, interval)
            )
        columns = tuple(arrays.keys())
        return ModelInput(columns, arrays, bars.dates)

    def clear_cache(self):
        self._bar_cache.clear()
        self._sym_inputs.clear()
        self._sym_preds.clear()


class ModelInputScope:
    r"""Caches and retrieves model input data.

    Args:
        col_scope: :class:`.ColumnScope`.
        ind_scope: :class:`.IndicatorScope`.
        models: :class:`Mapping` of
            :class:`pybroker.common.ModelSymbol` pairs to
            :class:`pybroker.common.TrainedModel`\ s.
    """

    def __init__(
        self,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        models: Mapping[ModelSymbol, TrainedModel],
        history_col_scope: Optional["ColumnScope"] = None,
        test_dates: Optional[Sequence[np.datetime64]] = None,
    ):
        self._col_scope = col_scope
        self._ind_scope = ind_scope
        self._models = models
        self._history_col_scope = history_col_scope
        self._lag_series_cache: LagSeriesCache = {}
        self._history_dates: dict[str, np.ndarray] = {}
        self._test_dates = [] if test_dates is None else test_dates
        self._sym_inputs: dict[ModelSymbol, ModelInput] = {}
        self._scope = StaticScope.instance()

    def _lag_cols(self, input_cols: tuple[str, ...]) -> tuple[str, ...]:
        date_col = DataCol.DATE.value
        return tuple(col for col in input_cols if col != date_col)

    def _ensure_lag_cache(
        self,
        symbol: str,
        lag_cols: tuple[str, ...],
        lags: int,
    ) -> None:
        if symbol in self._history_dates:
            return
        if self._history_col_scope is None:
            raise ValueError(
                f"History data required to compute lags for {symbol!r}."
            )
        dates = self._history_col_scope.fetch(symbol, DataCol.DATE.value)
        if dates is None:
            raise ValueError(f"History dates not found for {symbol!r}.")
        self._history_dates[symbol] = dates
        column_arrays: dict[str, NDArray[Any]] = {}
        for col in lag_cols:
            col_data = self._history_col_scope.fetch(symbol, col)
            if col_data is None:
                raise ValueError(
                    f"History column {col!r} not found for {symbol!r}."
                )
            column_arrays[col] = col_data
        merge_lag_series_cache_from_arrays(
            self._lag_series_cache,
            symbol,
            lag_cols,
            lags,
            dates,
            column_arrays,
        )

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetches model input data.

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.model.ModelSource` to query input
                data.
            end_index: Truncates the array of model input data returned
                (exclusive). If ``None``, then model input data is not
                truncated.

        Returns:
            :class:`pandas.DataFrame` of model input data for every bar until
            ``end_index`` (when specified).
        """
        model_input = self._fetch_model_input(symbol, name, end_index)
        return model_input_to_dataframe(model_input)

    def _fetch_model_input(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> ModelInput:
        model_sym = ModelSymbol(name, symbol)
        if model_sym in self._sym_inputs:
            model_input = self._sym_inputs[model_sym]
            return (
                model_input
                if end_index is None
                else model_input.slice(end_index)
            )
        input_: dict[str, NDArray[Any]] = {}
        for col in self._scope.all_data_cols:
            data = self._col_scope.fetch(symbol, col)
            if data is not None:
                input_[col] = data
        if not self._scope.has_model_source(name):
            raise ValueError(f"Model {name!r} not found.")
        for ind_name in self._scope.get_indicator_names(name):
            input_[ind_name] = self._ind_scope.fetch(symbol, ind_name)
        if model_sym not in self._models:
            raise ValueError(f"Model {name!r} not found for {symbol}.")
        trained_model = self._models[model_sym]
        model_source = self._scope.get_model_source(name)
        date_col = DataCol.DATE.value
        row_dates = input_.get(date_col)
        if row_dates is None:
            row_dates = self._col_scope.fetch(symbol, date_col)
        assert row_dates is not None
        columns = tuple(input_.keys())
        model_input = ModelInput(columns, input_, row_dates)
        if trained_model.input_cols is not None:
            for input_col in trained_model.input_cols:
                if input_col not in model_input:
                    raise ValueError(
                        f"Missing column {input_col!r} for input data to "
                        f"model {model_sym.model_name!r}."
                    )
            model_input = model_input.select_columns(trained_model.input_cols)
        if model_source.lags is not None:
            if trained_model.input_cols is None:
                raise ValueError(
                    f"Model {model_sym.model_name!r} requires input columns "
                    f"from training before applying lags."
                )
            lag_cols = self._lag_cols(trained_model.input_cols)
            self._ensure_lag_cache(symbol, lag_cols, model_source.lags)
            apply_lags_to_model_input(
                model_input,
                lag_cols,
                model_source.lags,
                self._lag_series_cache,
                symbol,
                self._history_dates[symbol],
            )
        if not trained_model.input_cols or model_source._input_data_fn:
            model_input = apply_prepare_input_data(
                model_input, model_source.prepare_input_data
            )
        self._sym_inputs[model_sym] = model_input
        return (
            model_input if end_index is None else model_input.slice(end_index)
        )


class PredictionScope:
    r"""Caches and retrieves model predictions.

    Args:
        models: :class:`Mapping` of
            :class:`pybroker.common.ModelSymbol` pairs to
            :class:`pybroker.common.TrainedModel`\ s.
        input_scope: :class:`.ModelInputScope`.
    """

    def __init__(
        self,
        models: Mapping[ModelSymbol, TrainedModel],
        input_scope: ModelInputScope,
    ):
        self._models = models
        self._input_scope = input_scope
        self._sym_preds: dict[ModelSymbol, NDArray] = {}

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> NDArray:
        """Fetches model predictions.

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.model.ModelSource` that made the
                predictions.
            end_index: Truncates the array of predictions returned (exclusive).
                If ``None``, then predictions are not truncated.

        Returns:
            :class:`numpy.ndarray` of model predictions for every bar until
            ``end_index`` (when specified).
        """
        model_sym = ModelSymbol(name, symbol)
        trained_model = self._models.get(model_sym)
        if trained_model is not None and trained_model.per_bar:
            return self._fetch_per_bar(
                symbol, name, model_sym, trained_model, end_index
            )
        if model_sym in self._sym_preds:
            return self._sym_preds[model_sym][:end_index]
        input_ = self._input_scope.fetch(symbol, name)
        if input_.empty or len(input_.columns) == 0:
            raise ValueError(
                f"No input data found for model {name!r}. Consider "
                "passing input_data_fn to pybroker#model() if custom columns "
                "were registered."
            )
        if model_sym not in self._models:
            raise ValueError(f"Model {name!r} not found for {symbol}.")
        trained_model = self._models[model_sym]
        pred = self._run_predict(trained_model, input_)
        self._sym_preds[model_sym] = pred
        return pred[:end_index]

    def _fetch_per_bar(
        self,
        symbol: str,
        name: str,
        model_sym: ModelSymbol,
        trained_model: TrainedModel,
        end_index: Optional[int],
    ) -> NDArray:
        if model_sym not in self._sym_preds:
            self._sym_preds[model_sym] = np.array([], dtype=np.float64)
        pred = self._sym_preds[model_sym]
        if end_index is None:
            input_full = self._input_scope.fetch(symbol, name)
            target_len = len(input_full)
        else:
            target_len = end_index
        while len(pred) < target_len:
            bar_end_index = len(pred) + 1
            input_ = self._input_scope.fetch(symbol, name, bar_end_index)
            scalar = self._run_predict_scalar(trained_model, input_)
            pred = np.append(pred, scalar)
            self._sym_preds[model_sym] = pred
        return pred if end_index is None else pred[:end_index]

    @staticmethod
    def _run_predict(
        trained_model: TrainedModel, input_: pd.DataFrame
    ) -> NDArray:
        if trained_model.predict_fn is not None:
            pred = trained_model.predict_fn(trained_model.instance, input_)
        else:
            predict_fn = getattr(trained_model.instance, "predict", None)
            if predict_fn is not None and callable(predict_fn):
                pred = trained_model.instance.predict(input_)
            else:
                raise ValueError(
                    f"Model instance trained for {trained_model.name!r} "
                    "does not define a predict function. Please pass a "
                    "predict_fn to pybroker.model()."
                )
        pred_arr = np.asarray(pred)
        if pred_arr.ndim == 0:
            return np.array([pred_arr.item()])
        if len(pred_arr.shape) > 1:
            pred_arr = np.squeeze(pred_arr)
        return pred_arr

    @staticmethod
    def _run_predict_scalar(
        trained_model: TrainedModel, input_: pd.DataFrame
    ) -> float:
        pred = PredictionScope._run_predict(trained_model, input_)
        return float(np.asarray(pred).reshape(-1)[0])


class PriceScope:
    """Retrieves most recent prices."""

    def __init__(
        self,
        col_scope: ColumnScope,
        sym_end_index: Mapping[str, int],
        round_fill_price: bool,
    ):
        self._col_scope = col_scope
        self._sym_end_index = sym_end_index
        self._round_fill_price = round_fill_price

    def fetch(
        self,
        symbol: str,
        price: Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ],
    ) -> Decimal:
        end_index = self._sym_end_index[symbol]
        price_type = type(price)
        fill_price = None
        if price_type is PriceType:
            if price is _PRICE_OPEN:
                open_ = self._col_scope.fetch(symbol, _COL_OPEN, end_index)
                if open_ is None:
                    raise ValueError("Open price not found.")
                fill_price = open_[-1]
            elif price is _PRICE_HIGH:
                high = self._col_scope.fetch(symbol, _COL_HIGH, end_index)
                if high is None:
                    raise ValueError("High price not found.")
                fill_price = high[-1]
            elif price is _PRICE_LOW:
                low = self._col_scope.fetch(symbol, _COL_LOW, end_index)
                if low is None:
                    raise ValueError("Low price not found.")
                fill_price = low[-1]
            elif price is _PRICE_CLOSE:
                close = self._col_scope.fetch(symbol, _COL_CLOSE, end_index)
                if close is None:
                    raise ValueError("Close price not found.")
                fill_price = close[-1]
            elif price is _PRICE_MIDDLE:
                low = self._col_scope.fetch(symbol, _COL_LOW, end_index)
                if low is None:
                    raise ValueError("Low price not found.")
                high = self._col_scope.fetch(symbol, _COL_HIGH, end_index)
                if high is None:
                    raise ValueError("High price not found.")
                fill_price = low[-1] + (high[-1] - low[-1]) / 2.0
            elif price is _PRICE_AVERAGE:
                open_ = self._col_scope.fetch(symbol, _COL_OPEN, end_index)
                if open_ is None:
                    raise ValueError("Open price not found.")
                high = self._col_scope.fetch(symbol, _COL_HIGH, end_index)
                if high is None:
                    raise ValueError("High price not found.")
                low = self._col_scope.fetch(symbol, _COL_LOW, end_index)
                if low is None:
                    raise ValueError("Low price not found.")
                close = self._col_scope.fetch(symbol, _COL_CLOSE, end_index)
                if close is None:
                    raise ValueError("Close price not found.")
                fill_price = (open_[-1] + low[-1] + high[-1] + close[-1]) / 4.0
            else:
                raise ValueError(f"Unknown price: {price_type}")
        elif (
            price_type is float
            or price_type is int
            or isinstance(price, np.floating)
            or isinstance(price, Decimal)
        ):
            fill_price = price
        elif callable(price):
            bar_data = self._col_scope.bar_data_from_data_columns(
                symbol, self._sym_end_index[symbol]
            )
            fill_price = price(symbol, bar_data)
        else:
            raise ValueError(f"Unknown price: {price_type}")
        if self._round_fill_price:
            # Fast 2-decimal rounding. builtins.round(float, 2) costs
            # ~7 us/call; the integer divmod-100 form is ~0.5 us/call and
            # yields the same clean decimal string when the result is then
            # wrapped via Decimal(str(x)). Must divide by 100.0 (integer
            # ratio), NOT multiply by 0.01 (binary float 0.01 re-introduces
            # the rounding artifact we just removed).
            fp = float(fill_price)  # type: ignore[arg-type]
            if fp >= 0.0:
                fill_price = int(fp * 100.0 + 0.5) / 100.0
            else:
                fill_price = -int(-fp * 100.0 + 0.5) / 100.0
        return to_decimal(cast(float, fill_price))


class PendingOrder(NamedTuple):
    """Holds data for a pending order.

    Attributes:
        id: Unique ID.
        type: Type of order, either ``buy`` or ``sell``.
        symbol: Ticker symbol of the order.
        created: Date the order was created.
        exec_date: Date the order will be executed.
        shares: Number of shares to be bought or sold.
        limit_price: Limit price to use for the order.
        fill_price: Price that the order will be filled at.
    """

    id: int
    type: Literal["buy", "sell"]
    symbol: str
    created: np.datetime64
    exec_date: np.datetime64
    shares: Decimal
    limit_price: Optional[Decimal]
    fill_price: Union[
        int,
        float,
        np.floating,
        Decimal,
        PriceType,
        Callable[[str, BarData], Union[int, float, Decimal]],
    ]


class PendingOrderScope:
    r"""Stores :class:`.PendingOrder`\ s"""

    _order_id: int = 0

    def __init__(self):
        self._orders: dict[int, PendingOrder] = {}
        self._sym_orders: dict[str, set[PendingOrder]] = defaultdict(set)

    def contains(self, order_id: int) -> bool:
        """Returns whether a :class:`.PendingOrder` exists with
        ``order_id``.
        """
        return order_id in self._orders

    def add(
        self,
        type: Literal["buy", "sell"],
        symbol: str,
        created: np.datetime64,
        exec_date: np.datetime64,
        shares: Decimal,
        limit_price: Optional[Decimal],
        fill_price: Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ],
    ) -> int:
        """Creates a :class:`.PendingOrder`.

        Args:
            type: Type of order, either ``buy`` or ``sell``.
            symbol: Ticker symbol of the order.
            created: Date the order was created.
            exec_date: Date the order will be executed.
            shares: Number of shares to be bought or sold.
            limit_price: Limit price to use for the order.
            fill_price: Price that the order will be filled at.

        Returns:
            ID of the :class:`.PendingOrder`.
        """
        self._order_id += 1
        order = PendingOrder(
            id=self._order_id,
            type=type,
            symbol=symbol,
            created=created,
            exec_date=exec_date,
            shares=shares,
            limit_price=limit_price,
            fill_price=fill_price,
        )
        self._orders[self._order_id] = order
        self._sym_orders[symbol].add(order)
        return order.id

    def remove(self, order_id: int) -> bool:
        """Removes a :class:`.PendingOrder` with ``order_id```."""
        if order_id in self._orders:
            order = self._orders[order_id]
            del self._orders[order_id]
            if (
                order.symbol in self._sym_orders
                and order in self._sym_orders[order.symbol]
            ):
                self._sym_orders[order.symbol].remove(order)
            return True
        return False

    def remove_all(self, symbol: Optional[str] = None):
        r"""Removes all :class:`.PendingOrder`\ s."""
        if symbol is None:
            cancel_ids = tuple(self._orders.keys())
            for order_id in cancel_ids:
                self.remove(order_id)
        elif symbol in self._sym_orders:
            cancel_ids = tuple(order.id for order in self._sym_orders[symbol])
            for order_id in cancel_ids:
                self.remove(order_id)

    def orders(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[int] = None,
    ) -> Iterable[PendingOrder]:
        r"""Returns an :class:`Iterable` of :class:`.PendingOrder`\ s.

        Args:
            symbol: Filter by ticker symbol.
            order_id: Filter by order ID.
        """
        if order_id is not None and symbol is not None:
            order = self._orders.get(order_id)
            if order is not None and order.symbol == symbol:
                return [order]
            return []
        elif order_id is not None:
            order = self._orders.get(order_id)
            if order is not None:
                return [order]
            return []
        elif symbol is not None:
            if symbol not in self._sym_orders:
                return []
            return self._sym_orders[symbol]
        else:
            return self._orders.values()


def get_signals(
    symbols: Iterable[str],
    col_scope: ColumnScope,
    ind_scope: IndicatorScope,
    pred_scope: PredictionScope,
) -> dict[str, pd.DataFrame]:
    r"""Retrieves dictionary of :class:`pandas.DataFrame`\ s
    containing bar data, indicator data, and model predictions for each symbol.
    """
    static_scope = StaticScope.instance()
    cols = static_scope.all_data_cols
    inds = static_scope._indicators.keys()
    models = static_scope._model_sources.keys()
    dates = col_scope._df.index.get_level_values(1)
    dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        data = {DataCol.DATE.value: dates}
        for col in cols:
            data[col] = col_scope.fetch(sym, col)
        for ind in inds:
            try:
                data[ind] = ind_scope.fetch(sym, ind)
            except ValueError:
                continue
        for model in models:
            try:
                data[f"{model}_pred"] = pred_scope.fetch(sym, model)
            except ValueError:
                continue
        dfs[sym] = pd.DataFrame(data)
    return dfs
