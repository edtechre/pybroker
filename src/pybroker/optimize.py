"""Hyperparameter optimization with Optuna."""

from __future__ import annotations

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import warnings
import functools
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Union,
)

import numpy as np
import optuna
import pandas as pd
from joblib import delayed
from optuna.samplers import BaseSampler, GridSampler, RandomSampler, TPESampler

from pybroker.cache import CacheDateFields
from pybroker.common import (
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    get_unique_sorted_dates,
    to_datetime,
    to_seconds,
    verify_date_range,
)
from pybroker.model import ModelTrainer
from pybroker.hyperparam import (
    Hyperparam,
    SearchSpace,
    _find_hyperparam_names,
    build_run_hyperparams,
)
from pybroker.indicator import DEFAULT_INDICATOR_MEMO_MAX
from pybroker.parallel import parallel
from pybroker.portfolio import Portfolio
from pybroker.scope import (
    ColumnScope,
    StaticScope,
    merge_symbol_array_stores,
    slice_symbol_array_store_by_dates,
    symbol_array_store_from_frame,
)
from pybroker.timeframe import (
    model_timeframe_name,
    parse_indicator_timeframe_name,
    parse_model_timeframe_name,
    symbol_dates_from_frame,
)

if TYPE_CHECKING:
    from pybroker.common import TrainedModel
    from pybroker.config import StrategyConfig
    from pybroker.context import ExecContext, RotationContext
    from pybroker.slippage import SlippageModel
    from pybroker.strategy import Execution, TestResult, WalkforwardWindow
    from pybroker.timeframe import TimeframeData, TimeframeInterval

    class _ExecutionsHost(Protocol):
        _executions: set[Execution]
        _max_long_positions: Union[int, Hyperparam, None]
        _max_short_positions: Union[int, Hyperparam, None]
        _worst_rank_held: Union[int, Hyperparam, None]

    class _OptimizeTrialHost(Protocol):
        def _run_optimize_trial(
            self,
            df: pd.DataFrame,
            train_rows: np.ndarray,
            run_hyperparams: dict[str, Any],
            invariant_indicator_data: dict[IndicatorSymbol, pd.Series],
            window_executions: set[Execution],
            master_store: Any,
            timeframe_data: Any,
            disable_parallel_indicators: bool,
            warmup: Optional[int],
            pretrained_models: Mapping[ModelSymbol, TrainedModel],
        ) -> TestResult: ...


_MODEL_OPTIMIZE_ERROR = (
    "optimize() does not support trainable model sources "
    "({trainable}). Pretrained models (pretrained=True) are supported. "
    "Tune trainable model hyperparameters inside train_fn with a validation "
    "split, or use walkforward() for exec params that depend on model "
    "outputs."
)

_GRID_EXPLOSION_THRESHOLD = 1000


@functools.lru_cache(maxsize=1)
def _strategy_module():
    import pybroker.strategy as strategy

    return strategy


def collect_hyperparams(strategy: _ExecutionsHost) -> dict[str, Hyperparam]:
    """Collects all hyperparams reachable from ``strategy``."""
    scope = StaticScope.instance()
    names: set[str] = set()
    specs: dict[str, Hyperparam] = {}

    for value in (
        strategy._max_long_positions,
        strategy._max_short_positions,
        strategy._worst_rank_held,
    ):
        if isinstance(value, Hyperparam):
            if not scope.has_hyperparam(value.name):
                raise ValueError(
                    f"Hyperparam {value.name!r} was not registered."
                )
            names.add(value.name)
            specs[value.name] = scope.get_hyperparam(value.name)

    for execution in strategy._executions:
        for ind_name in execution.indicator_names:
            ind = scope.get_indicator(ind_name)
            for hp_name in ind.hyperparam_names:
                if not scope.has_hyperparam(hp_name):
                    raise ValueError(
                        f"Hyperparam {hp_name!r} in indicator "
                        f"{ind_name!r} is not registered."
                    )
                names.add(hp_name)
                specs[hp_name] = scope.get_hyperparam(hp_name)

        for hp_name in execution.hyperparam_names:
            if not scope.has_hyperparam(hp_name):
                raise ValueError(f"Hyperparam {hp_name!r} was not registered.")
            names.add(hp_name)
            specs[hp_name] = scope.get_hyperparam(hp_name)

        for model_name in execution.model_names:
            base_name, _ = parse_model_timeframe_name(model_name)
            source = scope.get_model_source(base_name)
            if _find_hyperparam_names(source._kwargs):
                raise ValueError(
                    f"Model {model_name!r} has hyperparams in kwargs; "
                    "models are excluded from optimize()."
                )

    for hp in scope.iter_hyperparams():
        if hp.name not in names:
            warnings.warn(
                f"Hyperparam {hp.name!r} is registered but not reachable "
                "from any execution; it inflates the grid if searchable.",
                stacklevel=2,
            )

    if len(names) != len(specs):
        raise ValueError("Duplicate hyperparam names in search space.")

    return specs


def collect_search_space(strategy: _ExecutionsHost) -> SearchSpace:
    """Collects searchable hyperparams reachable from ``strategy``."""
    specs = collect_hyperparams(strategy)
    searchable = {name: hp for name, hp in specs.items() if hp.low < hp.high}
    return SearchSpace(frozenset(searchable), searchable)


def _validate_optimize_models(strategy: _ExecutionsHost) -> None:
    scope = StaticScope.instance()
    trainable: list[str] = []
    for execution in strategy._executions:
        for model_name in execution.model_names:
            base_name, _ = parse_model_timeframe_name(model_name)
            source = scope.get_model_source(base_name)
            if isinstance(source, ModelTrainer):
                trainable.append(model_name)
    if trainable:
        raise ValueError(
            _MODEL_OPTIMIZE_ERROR.format(
                trainable=", ".join(sorted(set(trainable)))
            )
        )


def _suggest_from_spec(
    trial: optuna.Trial, hp: Hyperparam
) -> Union[int, float]:
    if hp._is_float():
        return trial.suggest_float(
            hp.name, float(hp.low), float(hp.high), step=float(hp.step)
        )
    return trial.suggest_int(
        hp.name, int(hp.low), int(hp.high), step=int(hp.step)
    )


def _trial_params(
    trial: optuna.Trial, search_space: SearchSpace
) -> dict[str, Any]:
    return {
        name: _suggest_from_spec(trial, search_space.specs[name])
        for name in search_space.hyperparams
    }


def _build_sampler(
    sampler: Union[str, BaseSampler],
    search_space: SearchSpace,
    seed: Optional[int],
) -> BaseSampler:
    if isinstance(sampler, BaseSampler):
        return sampler
    if sampler == "grid":
        grid = {
            name: list(search_space.specs[name])
            for name in sorted(search_space.hyperparams)
        }
        return GridSampler(grid) if grid else GridSampler({})
    if sampler == "tpe":
        return TPESampler(seed=seed)
    if sampler == "random":
        return RandomSampler(seed=seed)
    raise ValueError(
        f"Unknown sampler {sampler!r}; use 'grid', 'tpe', 'random', or a "
        "optuna.samplers.BaseSampler instance."
    )


def _validate_grid_sampler(
    sampler: BaseSampler, search_space: SearchSpace
) -> None:
    if not isinstance(sampler, GridSampler):
        return
    declared = set(search_space.hyperparams)
    grid_space = getattr(sampler, "search_space", None) or getattr(
        sampler, "_search_space", {}
    )
    for name in grid_space.keys():
        if name not in declared:
            raise ValueError(
                f"GridSampler param {name!r} is not in the declared search "
                f"space: {sorted(declared)}."
            )


def _grid_trial_count(search_space: SearchSpace) -> int:
    return search_space.grid_size()


def _sampler_name(sampler: BaseSampler) -> str:
    if isinstance(sampler, GridSampler):
        return "grid"
    if isinstance(sampler, TPESampler):
        return "tpe"
    if isinstance(sampler, RandomSampler):
        return "random"
    return type(sampler).__name__.removesuffix("Sampler").lower()


def _resolve_n_trials(
    n_trials: Optional[int],
    sampler: BaseSampler,
    search_space: SearchSpace,
) -> int:
    if n_trials is not None:
        return n_trials
    if isinstance(sampler, GridSampler):
        return search_space.grid_size()
    raise ValueError("n_trials is required for non-grid samplers.")


def _log_optimize_trials(
    n_trials: int,
    sampler: BaseSampler,
    search_space: SearchSpace,
    *,
    windows: int = 1,
) -> None:
    grid_size = (
        search_space.grid_size() if isinstance(sampler, GridSampler) else None
    )
    StaticScope.instance().logger.optimize_start(
        n_trials=n_trials,
        sampler=_sampler_name(sampler),
        grid_size=grid_size,
        windows=windows,
    )


def _log_search_space(search_space: SearchSpace) -> None:
    searched = sorted(search_space.hyperparams)
    if searched:
        warnings.warn(f"Searched hyperparams: {searched}", stacklevel=3)


@dataclass(frozen=True)
class WindowOptimizeResult:
    """Per-window walk-forward optimization result."""

    params: dict[str, Any]
    study: optuna.Study
    test_result: TestResult
    train_score: float
    train_pnl: float


@dataclass(frozen=True)
class OptimizeResult:
    """Result of :meth:`Strategy.optimize`."""

    best_params: dict[str, Any]
    best_score: float
    result: TestResult
    study: optuna.Study
    windows: Optional[tuple[WindowOptimizeResult, ...]] = None
    walkforward_efficiency: Optional[float] = None


@dataclass(frozen=True)
class ObjectiveBundle:
    """Return value of :func:`make_objective`."""

    objective: Callable[[optuna.Trial], float]
    search_space: SearchSpace


def make_objective(
    strategy: _OptimizeTrialHost,
    score_fn: Callable[[TestResult], float],
    *,
    train_rows: np.ndarray,
    df: pd.DataFrame,
    hyperparams: Mapping[str, Hyperparam],
    search_space: SearchSpace,
    invariant_indicator_data: dict[IndicatorSymbol, pd.Series],
    window_executions: set[Execution],
    master_store: Any,
    timeframe_data: Any,
    disable_parallel_indicators: bool,
    warmup: Optional[int],
    pretrained_models: Mapping[ModelSymbol, TrainedModel],
) -> ObjectiveBundle:
    """Builds an Optuna objective for train-window scoring."""

    def objective(trial: optuna.Trial) -> float:
        overrides = _trial_params(trial, search_space)
        run_hp = build_run_hyperparams(hyperparams, overrides)
        result = strategy._run_optimize_trial(
            df=df,
            train_rows=train_rows,
            run_hyperparams=run_hp,
            invariant_indicator_data=invariant_indicator_data,
            window_executions=window_executions,
            master_store=master_store,
            timeframe_data=timeframe_data,
            disable_parallel_indicators=disable_parallel_indicators,
            warmup=warmup,
            pretrained_models=pretrained_models,
        )
        return score_fn(result)

    return ObjectiveBundle(objective=objective, search_space=search_space)


class OptimizeMixin:
    """Mixin implementing hyperparameter optimization."""

    if TYPE_CHECKING:
        _config: StrategyConfig
        _executions: set[Execution]
        _before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]]
        _after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]]
        _max_long_positions: Any
        _max_short_positions: Any
        _worst_rank_held: Any
        _rotation_sizer: Optional[Callable[[RotationContext], None]]
        _slippage_model: Optional[SlippageModel]
        _timeframes: frozenset[TimeframeInterval]
        _start_date: datetime
        _end_date: datetime
        _base_bar_seconds: Optional[float]
        _indicator_memo_max: int

        def _fractional_shares_enabled(self) -> bool: ...

        def train_models(
            self, *args: Any, **kwargs: Any
        ) -> dict[ModelSymbol, TrainedModel]: ...

        def _fetch_indicators(
            self, *args: Any, **kwargs: Any
        ) -> dict[IndicatorSymbol, pd.Series]: ...

        def _resolve_backtest_settings(
            self, run_hyperparams: Optional[dict[str, Any]] = None
        ) -> Any: ...

        def _effective_config(self, settings: Any) -> StrategyConfig: ...

        def backtest_executions(
            self, *args: Any, **kwargs: Any
        ) -> dict[str, pd.DataFrame]: ...

        def _to_test_result(self, *args: Any, **kwargs: Any) -> TestResult: ...

        def compute_indicators(
            self, *args: Any, **kwargs: Any
        ) -> dict[IndicatorSymbol, pd.Series]: ...

        def _fetch_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame: ...

        def _to_day_ids(self, *args: Any, **kwargs: Any) -> Any: ...

        def _filter_dates(self, *args: Any, **kwargs: Any) -> pd.DataFrame: ...

        def _validate_timeframes_for_base(
            self, *args: Any, **kwargs: Any
        ) -> None: ...

        def _has_symbol_selector(self) -> bool: ...

        def _compress_timeframes(
            self, *args: Any, **kwargs: Any
        ) -> TimeframeData: ...

        def walkforward_split(
            self, *args: Any, **kwargs: Any
        ) -> Iterator[WalkforwardWindow]: ...

        def _liquidate_dropped_symbols(
            self, *args: Any, **kwargs: Any
        ) -> None: ...

    def _collect_hyperparams(self) -> dict[str, Hyperparam]:
        return collect_hyperparams(self)

    def _collect_search_space(self) -> SearchSpace:
        return collect_search_space(self)

    def _load_pretrained_models(
        self,
        df: pd.DataFrame,
        train_rows: np.ndarray,
        test_rows: np.ndarray,
        window_executions: set[Execution],
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        master_store: Any,
        timeframe_data: Any,
        tf_seconds: int,
        between_time: Optional[tuple[str, str]],
        days: Optional[Any],
    ) -> dict[ModelSymbol, TrainedModel]:
        if not any(execution.model_names for execution in window_executions):
            return {}
        sym_col = DataCol.SYMBOL.value
        date_col = DataCol.DATE.value
        train_data = df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
        test_data = df.iloc[test_rows] if len(test_rows) else df.iloc[:0]
        if train_data.empty:
            return {}
        master_dates_arr = df[date_col].to_numpy(
            dtype="datetime64[ns]", copy=False
        )
        train_store = None
        test_store = None
        history_store = None
        if not test_data.empty:
            test_dates_arr = np.unique(master_dates_arr[test_rows])
            test_store = slice_symbol_array_store_by_dates(
                master_store, test_dates_arr
            )
        train_dates_arr = np.unique(master_dates_arr[train_rows])
        train_store = slice_symbol_array_store_by_dates(
            master_store, train_dates_arr
        )
        if test_store is not None:
            history_store = merge_symbol_array_stores(train_store, test_store)
        else:
            history_store = train_store
        train_symbols = set(train_data[sym_col].unique())
        model_syms: set[ModelSymbol] = set()
        for sym in train_symbols:
            for execution in window_executions:
                if sym not in _strategy_module()._static_symbols(
                    execution.symbols
                ):
                    continue
                for model_name in execution.model_names:
                    base_name, token = parse_model_timeframe_name(model_name)
                    if token is not None:
                        model_syms.add(ModelSymbol(model_name, sym))
                        continue
                    model_syms.add(ModelSymbol(model_name, sym))
                    for tf in self._timeframes:
                        model_syms.add(
                            ModelSymbol(
                                model_timeframe_name(base_name, tf), sym
                            )
                        )
        pooled_model_groups: dict[tuple[str, int], frozenset[str]] = {}
        for execution in window_executions:
            exec_syms = frozenset(
                sym
                for sym in _strategy_module()._static_symbols(
                    execution.symbols
                )
                if sym in train_symbols
            )
            if not exec_syms:
                continue
            for model_name in execution.model_names:
                base_name, token = parse_model_timeframe_name(model_name)
                if token is not None:
                    continue
                source = StaticScope.instance().get_model_source(base_name)
                if isinstance(source, ModelTrainer) and source.pooled:
                    pooled_model_groups[(model_name, execution.id)] = exec_syms
                    for tf in self._timeframes:
                        pooled_model_groups[
                            (model_timeframe_name(base_name, tf), execution.id)
                        ] = exec_syms
        train_dates = get_unique_sorted_dates(train_data[date_col])
        return self.train_models(
            model_syms=model_syms,
            train_data=train_data,
            test_data=test_data,
            indicator_data=indicator_data,
            cache_date_fields=CacheDateFields(
                start_date=to_datetime(train_dates[0]),
                end_date=to_datetime(train_dates[-1]),
                tf_seconds=tf_seconds,
                between_time=between_time,
                days=days,
            ),
            pooled_model_groups=pooled_model_groups,
            timeframe_data=timeframe_data,
            history_store=history_store,
            train_store=train_store,
            test_store=test_store,
        )

    def _run_optimize_trial(
        self,
        df: pd.DataFrame,
        train_rows: np.ndarray,
        run_hyperparams: dict[str, Any],
        invariant_indicator_data: dict[IndicatorSymbol, pd.Series],
        window_executions: set[Execution],
        master_store: Any,
        timeframe_data: Any,
        disable_parallel_indicators: bool,
        warmup: Optional[int],
        pretrained_models: Mapping[ModelSymbol, TrainedModel],
    ) -> TestResult:
        train_data = df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
        trial_indicators = self._fetch_indicators(
            df=train_data,
            cache_date_fields=None,
            disable_parallel_indicators=disable_parallel_indicators,
            timeframe_data=timeframe_data,
            executions=window_executions,
            symbol_store=master_store,
            hyperparams=run_hyperparams,
        )
        indicator_data = {**invariant_indicator_data, **trial_indicators}
        backtest_settings = self._resolve_backtest_settings(run_hyperparams)
        effective_config = self._effective_config(backtest_settings)
        portfolio = Portfolio(
            effective_config.initial_cash,
            effective_config.fee_mode,
            effective_config.fee_amount,
            self._fractional_shares_enabled(),
            effective_config.position_mode,
            backtest_settings.max_long_positions,
            backtest_settings.max_short_positions,
            effective_config.return_stops,
            effective_config.leverage,
            effective_config.interest_rate,
            effective_config.bars_per_year,
            record_portfolio_bars=effective_config.record_portfolio_bars,
            record_position_bars=effective_config.record_position_bars,
        )
        date_col = DataCol.DATE.value
        train_store = None
        if not train_data.empty:
            master_dates_arr = df[date_col].to_numpy(
                dtype="datetime64[ns]", copy=False
            )
            train_dates_arr = np.unique(master_dates_arr[train_rows])
            train_store = slice_symbol_array_store_by_dates(
                master_store, train_dates_arr
            )
        sessions: dict[str, dict] = defaultdict(dict)
        for sym in _static_symbols_from_executions(
            window_executions, train_data
        ):
            sessions[sym] = {}
        self.backtest_executions(
            config=effective_config,
            executions=window_executions,
            before_exec_fn=self._before_exec_fn,
            after_exec_fn=self._after_exec_fn,
            sessions=sessions,
            models=pretrained_models,
            indicator_data=indicator_data,
            timeframe_data=timeframe_data,
            declared_timeframes=self._timeframes,
            test_data=train_data,
            portfolio=portfolio,
            exit_dates={},
            backtest_settings=backtest_settings,
            rotation_sizer=self._rotation_sizer,
            slippage_model=self._slippage_model,
            enable_fractional_shares=self._fractional_shares_enabled(),
            round_fill_price=effective_config.round_fill_price,
            warmup=warmup,
            history_col_scope=ColumnScope(train_store)
            if train_store
            else None,
            test_col_scope=ColumnScope(train_store) if train_store else None,
            run_hyperparams=run_hyperparams,
        )
        if train_data.empty:
            start_dt = self._start_date
            end_dt = self._end_date
        else:
            dates = train_data[date_col]
            start_dt = pd.Timestamp(dates.min()).to_pydatetime()
            end_dt = pd.Timestamp(dates.max()).to_pydatetime()
        return self._to_test_result(
            start_dt,
            end_dt,
            portfolio,
            calc_bootstrap=False,
            train_only=False,
            signals=None,
            seed=None,
        )

    def _compute_invariant_indicators(
        self,
        df: pd.DataFrame,
        cache_date_fields: CacheDateFields,
        disable_parallel_indicators: bool,
        timeframe_data: Any,
        master_store: Any,
        executions: set[Execution],
    ) -> dict[IndicatorSymbol, pd.Series]:
        scope = StaticScope.instance()
        invariant_names: set[str] = set()
        for execution in executions:
            for ind_name in execution.indicator_names:
                base, _ = parse_indicator_timeframe_name(ind_name)
                ind = scope.get_indicator(base)
                if not ind.hyperparam_names:
                    invariant_names.add(ind_name)
        ind_syms: set[IndicatorSymbol] = set()
        for execution in executions:
            for sym in _strategy_module()._static_symbols(execution.symbols):
                for ind_name in execution.indicator_names:
                    if ind_name not in invariant_names:
                        continue
                    ind_syms.add(IndicatorSymbol(ind_name, sym))
        if not ind_syms:
            return {}
        return self.compute_indicators(
            df=df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel_indicators=disable_parallel_indicators,
            timeframe_data=timeframe_data,
            symbol_store=master_store,
            hyperparams=None,
        )

    def optimize(
        self,
        score_fn: Callable[[TestResult], float],
        *,
        sampler: Union[str, BaseSampler] = "grid",
        n_trials: Optional[int] = None,
        direction: str = "maximize",
        seed: Optional[int] = None,
        windows: Optional[int] = None,
        study: Optional[optuna.Study] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        train_size: float = 0.5,
        lookahead: int = 1,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = "",
        between_time: Optional[tuple[str, str]] = None,
        days: Optional[Any] = None,
        warmup: Optional[int] = None,
        disable_parallel_indicators: bool = False,
        adjust: Optional[Any] = None,
        indicator_memo_max: int = DEFAULT_INDICATOR_MEMO_MAX,
    ) -> OptimizeResult:
        """Optimizes hyperparameters on the train window, then evaluates on test.

        Pretrained models (``model(..., pretrained=True)``) are loaded once per
        train window and reused across trials. Trainable models are not
        supported.

        Args:
            indicator_memo_max: Maximum in-memory hyperparameter indicator
                results to retain while optimizing. Hyperparameterized
                indicators bypass the disk cache; this memo avoids recomputing
                identical ``(indicator, symbol, hyperparams)`` combinations
                across trials. When full, the oldest entry is evicted. Set to
                ``0`` to disable. Defaults to
                :data:`pybroker.indicator.DEFAULT_INDICATOR_MEMO_MAX`.
        """
        if indicator_memo_max < 0:
            raise ValueError(
                "indicator_memo_max must be greater than or equal to 0."
            )
        _validate_optimize_models(self)
        if not self._executions:
            raise ValueError("No executions were added.")
        search_space = collect_search_space(self)
        hyperparams = collect_hyperparams(self)
        _log_search_space(search_space)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        scope = StaticScope.instance()
        scope.freeze_data_cols()
        self._indicator_memo_max = indicator_memo_max
        try:
            start_dt = (
                self._start_date
                if start_date is None
                else to_datetime(start_date)
            )
            end_dt = (
                self._end_date if end_date is None else to_datetime(end_date)
            )
            verify_date_range(start_dt, end_dt)
            df = self._fetch_data(timeframe, adjust)
            day_ids = self._to_day_ids(days)
            df = self._filter_dates(
                df=df,
                start_date=start_dt,
                end_date=end_dt,
                between_time=between_time,
                days=day_ids,
            )
            self._validate_timeframes_for_base(df, timeframe)
            has_selector = self._has_symbol_selector()
            timeframe_data = self._compress_timeframes(
                df,
                set(df[DataCol.SYMBOL.value].unique())
                if has_selector
                else {
                    sym
                    for execution in self._executions
                    for sym in _strategy_module()._static_symbols(
                        execution.symbols
                    )
                },
            )
            tf_seconds = (
                int(self._base_bar_seconds)
                if self._base_bar_seconds is not None
                else to_seconds(timeframe)
            )
            cache_date_fields = CacheDateFields(
                start_date=start_dt,
                end_date=end_dt,
                tf_seconds=tf_seconds,
                between_time=between_time,
                days=day_ids,
            )
            master_store = symbol_array_store_from_frame(
                _strategy_module()._ensure_range_index(df)
            )

            wf_windows = windows if windows is not None else 1
            total_grid = _grid_trial_count(search_space) * wf_windows
            is_grid = sampler == "grid" or isinstance(sampler, GridSampler)
            if (
                is_grid
                and total_grid > _GRID_EXPLOSION_THRESHOLD
                and n_trials is None
            ):
                warnings.warn(
                    f"Grid size {total_grid} exceeds threshold "
                    f"{_GRID_EXPLOSION_THRESHOLD}. Set n_trials= to limit trials, "
                    "or use sampler='tpe'.",
                    stacklevel=2,
                )

            if windows is not None and windows > 1:
                return self._optimize_walkforward(
                    score_fn=score_fn,
                    sampler=sampler,
                    n_trials=n_trials,
                    direction=direction,
                    seed=seed,
                    pruner=pruner,
                    train_size=train_size,
                    lookahead=lookahead,
                    df=df,
                    master_store=master_store,
                    timeframe_data=timeframe_data,
                    cache_date_fields=cache_date_fields,
                    disable_parallel_indicators=disable_parallel_indicators,
                    warmup=warmup,
                    has_selector=has_selector,
                    hyperparams=hyperparams,
                    search_space=search_space,
                    windows=windows,
                    start_dt=start_dt,
                    end_dt=end_dt,
                )

            splits = list(
                self.walkforward_split(
                    df=df,
                    windows=1,
                    lookahead=lookahead,
                    train_size=train_size,
                )
            )
            train_rows, test_rows = splits[0]
            train_data = (
                df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
            )
            test_data = df.iloc[test_rows] if len(test_rows) else df.iloc[:0]
            selection_data = _strategy_module()._selection_df(
                train_data, test_data
            )
            window_executions = (
                _strategy_module()._resolve_executions(
                    self._executions, selection_data
                )
                if has_selector
                else self._executions
            )
            invariant_data = self._compute_invariant_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                disable_parallel_indicators=disable_parallel_indicators,
                timeframe_data=timeframe_data,
                master_store=master_store,
                executions=window_executions,
            )
            default_run_hp = build_run_hyperparams(hyperparams)
            load_indicators = self._fetch_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                disable_parallel_indicators=disable_parallel_indicators,
                timeframe_data=timeframe_data,
                executions=window_executions,
                symbol_store=master_store,
                hyperparams=default_run_hp,
            )
            pretrained_models = self._load_pretrained_models(
                df=df,
                train_rows=train_rows,
                test_rows=test_rows,
                window_executions=window_executions,
                indicator_data={**invariant_data, **load_indicators},
                master_store=master_store,
                timeframe_data=timeframe_data,
                tf_seconds=tf_seconds,
                between_time=between_time,
                days=day_ids,
            )
            bundle = make_objective(
                self,
                score_fn,
                train_rows=train_rows,
                df=df,
                hyperparams=hyperparams,
                search_space=search_space,
                invariant_indicator_data=invariant_data,
                window_executions=window_executions,
                master_store=master_store,
                timeframe_data=timeframe_data,
                disable_parallel_indicators=disable_parallel_indicators,
                warmup=warmup,
                pretrained_models=pretrained_models,
            )
            built_sampler = _build_sampler(sampler, search_space, seed)
            _validate_grid_sampler(built_sampler, search_space)
            n_trials = _resolve_n_trials(n_trials, built_sampler, search_space)
            _log_optimize_trials(n_trials, built_sampler, search_space)
            if study is None:
                study = optuna.create_study(
                    direction=direction, sampler=built_sampler, pruner=pruner
                )
            study.optimize(bundle.objective, n_trials=n_trials)
            best_params = build_run_hyperparams(hyperparams, study.best_params)
            test_result = self._run_optimize_test(
                df=df,
                test_rows=test_rows,
                run_hyperparams=best_params,
                invariant_indicator_data=invariant_data,
                window_executions=window_executions,
                master_store=master_store,
                timeframe_data=timeframe_data,
                cache_date_fields=cache_date_fields,
                disable_parallel_indicators=disable_parallel_indicators,
                warmup=warmup,
                start_dt=start_dt,
                end_dt=end_dt,
                pretrained_models=pretrained_models,
            )
            return OptimizeResult(
                best_params=best_params,
                best_score=study.best_value,
                result=test_result,
                study=study,
            )
        finally:
            scope.unfreeze_data_cols()
            if hasattr(self, "_indicator_memo_max"):
                del self._indicator_memo_max

    def _run_optimize_test(
        self,
        df: pd.DataFrame,
        test_rows: np.ndarray,
        run_hyperparams: dict[str, Any],
        invariant_indicator_data: dict[IndicatorSymbol, pd.Series],
        window_executions: set[Execution],
        master_store: Any,
        timeframe_data: Any,
        cache_date_fields: CacheDateFields,
        disable_parallel_indicators: bool,
        warmup: Optional[int],
        start_dt: datetime,
        end_dt: datetime,
        portfolio: Optional[Portfolio] = None,
        calc_bootstrap: bool = True,
        pretrained_models: Optional[Mapping[ModelSymbol, TrainedModel]] = None,
    ) -> TestResult:
        if pretrained_models is None:
            pretrained_models = {}
        test_data = df.iloc[test_rows] if len(test_rows) else df.iloc[:0]
        trial_indicators = self._fetch_indicators(
            df=df,
            cache_date_fields=cache_date_fields,
            disable_parallel_indicators=disable_parallel_indicators,
            timeframe_data=timeframe_data,
            executions=window_executions,
            symbol_store=master_store,
            hyperparams=run_hyperparams,
        )
        indicator_data = {**invariant_indicator_data, **trial_indicators}
        backtest_settings = self._resolve_backtest_settings(run_hyperparams)
        effective_config = self._effective_config(backtest_settings)
        if portfolio is None:
            portfolio = Portfolio(
                effective_config.initial_cash,
                effective_config.fee_mode,
                effective_config.fee_amount,
                self._fractional_shares_enabled(),
                effective_config.position_mode,
                backtest_settings.max_long_positions,
                backtest_settings.max_short_positions,
                effective_config.return_stops,
                effective_config.leverage,
                effective_config.interest_rate,
                effective_config.bars_per_year,
                record_portfolio_bars=effective_config.record_portfolio_bars,
                record_position_bars=effective_config.record_position_bars,
            )
        date_col = DataCol.DATE.value
        master_dates_arr = df[date_col].to_numpy(
            dtype="datetime64[ns]", copy=False
        )
        test_store = None
        if not test_data.empty:
            test_dates_arr = np.unique(master_dates_arr[test_rows])
            test_store = slice_symbol_array_store_by_dates(
                master_store, test_dates_arr
            )
        sessions: dict[str, dict] = defaultdict(dict)
        self.backtest_executions(
            config=effective_config,
            executions=window_executions,
            before_exec_fn=self._before_exec_fn,
            after_exec_fn=self._after_exec_fn,
            sessions=sessions,
            models=pretrained_models,
            indicator_data=indicator_data,
            timeframe_data=timeframe_data.slice_for_test(
                symbol_dates_from_frame(test_data)
            ),
            declared_timeframes=self._timeframes,
            test_data=test_data,
            portfolio=portfolio,
            exit_dates={},
            backtest_settings=backtest_settings,
            rotation_sizer=self._rotation_sizer,
            slippage_model=self._slippage_model,
            enable_fractional_shares=self._fractional_shares_enabled(),
            round_fill_price=effective_config.round_fill_price,
            warmup=warmup,
            test_col_scope=ColumnScope(test_store) if test_store else None,
            run_hyperparams=run_hyperparams,
        )
        return self._to_test_result(
            start_dt,
            end_dt,
            portfolio,
            calc_bootstrap=calc_bootstrap,
            train_only=False,
            signals=None,
            seed=None,
        )

    def _optimize_walkforward(
        self,
        *,
        score_fn: Callable[[TestResult], float],
        sampler: Union[str, BaseSampler],
        n_trials: Optional[int],
        direction: str,
        seed: Optional[int],
        pruner: Optional[optuna.pruners.BasePruner],
        train_size: float,
        lookahead: int,
        df: pd.DataFrame,
        master_store: Any,
        timeframe_data: Any,
        cache_date_fields: CacheDateFields,
        disable_parallel_indicators: bool,
        warmup: Optional[int],
        has_selector: bool,
        hyperparams: Mapping[str, Hyperparam],
        search_space: SearchSpace,
        windows: int,
        start_dt: datetime,
        end_dt: datetime,
    ) -> OptimizeResult:
        splits = list(
            self.walkforward_split(
                df=df,
                windows=windows,
                lookahead=lookahead,
                train_size=train_size,
            )
        )
        built_sampler = _build_sampler(sampler, search_space, seed)
        _validate_grid_sampler(built_sampler, search_space)
        window_n_trials = _resolve_n_trials(
            n_trials, built_sampler, search_space
        )
        _log_optimize_trials(
            window_n_trials,
            built_sampler,
            search_space,
            windows=windows,
        )

        def run_window_study(
            train_rows: np.ndarray,
            test_rows: np.ndarray,
        ) -> WindowOptimizeResult:
            train_data = (
                df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
            )
            test_data = df.iloc[test_rows] if len(test_rows) else df.iloc[:0]
            selection_data = _strategy_module()._selection_df(
                train_data, test_data
            )
            window_executions = (
                _strategy_module()._resolve_executions(
                    self._executions, selection_data
                )
                if has_selector
                else self._executions
            )
            invariant_data = self._compute_invariant_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                disable_parallel_indicators=disable_parallel_indicators,
                timeframe_data=timeframe_data,
                master_store=master_store,
                executions=window_executions,
            )
            default_run_hp = build_run_hyperparams(hyperparams)
            load_indicators = self._fetch_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                disable_parallel_indicators=disable_parallel_indicators,
                timeframe_data=timeframe_data,
                executions=window_executions,
                symbol_store=master_store,
                hyperparams=default_run_hp,
            )
            pretrained_models = self._load_pretrained_models(
                df=df,
                train_rows=train_rows,
                test_rows=test_rows,
                window_executions=window_executions,
                indicator_data={**invariant_data, **load_indicators},
                master_store=master_store,
                timeframe_data=timeframe_data,
                tf_seconds=cache_date_fields.tf_seconds,
                between_time=cache_date_fields.between_time,
                days=cache_date_fields.days,
            )
            bundle = make_objective(
                self,
                score_fn,
                train_rows=train_rows,
                df=df,
                hyperparams=hyperparams,
                search_space=search_space,
                invariant_indicator_data=invariant_data,
                window_executions=window_executions,
                master_store=master_store,
                timeframe_data=timeframe_data,
                disable_parallel_indicators=disable_parallel_indicators,
                warmup=warmup,
                pretrained_models=pretrained_models,
            )
            window_study = optuna.create_study(
                direction=direction, sampler=built_sampler, pruner=pruner
            )
            window_study.optimize(bundle.objective, n_trials=window_n_trials)
            best_params = build_run_hyperparams(
                hyperparams, window_study.best_params
            )
            is_result = self._run_optimize_trial(
                df=df,
                train_rows=train_rows,
                run_hyperparams=best_params,
                invariant_indicator_data=invariant_data,
                window_executions=window_executions,
                master_store=master_store,
                timeframe_data=timeframe_data,
                disable_parallel_indicators=disable_parallel_indicators,
                warmup=warmup,
                pretrained_models=pretrained_models,
            )
            test_result = self._run_optimize_test(
                df=df,
                test_rows=test_rows,
                run_hyperparams=best_params,
                invariant_indicator_data=invariant_data,
                window_executions=window_executions,
                master_store=master_store,
                timeframe_data=timeframe_data,
                cache_date_fields=cache_date_fields,
                disable_parallel_indicators=disable_parallel_indicators,
                warmup=warmup,
                start_dt=start_dt,
                end_dt=end_dt,
                pretrained_models=pretrained_models,
            )
            return WindowOptimizeResult(
                params=best_params,
                study=window_study,
                test_result=test_result,
                train_score=window_study.best_value,
                train_pnl=is_result.metrics.total_pnl,
            )

        with parallel() as pool:
            window_results = pool(
                delayed(run_window_study)(train_rows, test_rows)
                for train_rows, test_rows in splits
            )

        default_settings = self._resolve_backtest_settings(None)
        portfolio = Portfolio(
            self._config.initial_cash,
            self._config.fee_mode,
            self._config.fee_amount,
            self._fractional_shares_enabled(),
            self._config.position_mode,
            default_settings.max_long_positions,
            default_settings.max_short_positions,
            self._config.return_stops,
            self._config.leverage,
            self._config.interest_rate,
            self._config.bars_per_year,
            record_portfolio_bars=self._config.record_portfolio_bars,
            record_position_bars=self._config.record_position_bars,
        )
        date_col = DataCol.DATE.value
        master_dates_arr = df[date_col].to_numpy(
            dtype="datetime64[ns]", copy=False
        )
        for i, (train_rows, test_rows) in enumerate(splits):
            wr = window_results[i]
            train_data = (
                df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
            )
            test_data = df.iloc[test_rows] if len(test_rows) else df.iloc[:0]
            selection_data = _strategy_module()._selection_df(
                train_data, test_data
            )
            window_executions = (
                _strategy_module()._resolve_executions(
                    self._executions, selection_data
                )
                if has_selector
                else self._executions
            )
            invariant_data = self._compute_invariant_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                disable_parallel_indicators=disable_parallel_indicators,
                timeframe_data=timeframe_data,
                master_store=master_store,
                executions=window_executions,
            )
            trial_indicators = self._fetch_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                disable_parallel_indicators=disable_parallel_indicators,
                timeframe_data=timeframe_data,
                executions=window_executions,
                symbol_store=master_store,
                hyperparams=wr.params,
            )
            indicator_data = {**invariant_data, **trial_indicators}
            pretrained_models = self._load_pretrained_models(
                df=df,
                train_rows=train_rows,
                test_rows=test_rows,
                window_executions=window_executions,
                indicator_data=indicator_data,
                master_store=master_store,
                timeframe_data=timeframe_data,
                tf_seconds=cache_date_fields.tf_seconds,
                between_time=cache_date_fields.between_time,
                days=cache_date_fields.days,
            )
            test_store = None
            history_store = None
            if not test_data.empty:
                test_dates_arr = np.unique(master_dates_arr[test_rows])
                test_store = slice_symbol_array_store_by_dates(
                    master_store, test_dates_arr
                )
            if not train_data.empty:
                train_dates_arr = np.unique(master_dates_arr[train_rows])
                train_store = slice_symbol_array_store_by_dates(
                    master_store, train_dates_arr
                )
                history_store = (
                    merge_symbol_array_stores(train_store, test_store)
                    if test_store is not None
                    else train_store
                )
            elif test_store is not None:
                history_store = test_store
            selected_syms = {
                sym
                for execution in window_executions
                for sym in _strategy_module()._static_symbols(
                    execution.symbols
                )
            }
            self._liquidate_dropped_symbols(
                portfolio, selected_syms, test_data
            )
            sessions: dict[str, dict] = defaultdict(dict)
            window_settings = self._resolve_backtest_settings(wr.params)
            window_config = self._effective_config(window_settings)
            self.backtest_executions(
                config=window_config,
                executions=window_executions,
                before_exec_fn=self._before_exec_fn,
                after_exec_fn=self._after_exec_fn,
                sessions=sessions,
                models=pretrained_models,
                indicator_data=indicator_data,
                timeframe_data=timeframe_data.slice_for_test(
                    symbol_dates_from_frame(test_data)
                ),
                declared_timeframes=self._timeframes,
                test_data=test_data,
                portfolio=portfolio,
                exit_dates={},
                backtest_settings=window_settings,
                rotation_sizer=self._rotation_sizer,
                slippage_model=self._slippage_model,
                enable_fractional_shares=self._fractional_shares_enabled(),
                round_fill_price=window_config.round_fill_price,
                warmup=warmup,
                history_col_scope=ColumnScope(history_store)
                if history_store
                else None,
                test_col_scope=ColumnScope(test_store) if test_store else None,
                run_hyperparams=wr.params,
            )

        oos_pnl = sum(
            wr.test_result.metrics.total_pnl for wr in window_results
        )
        is_pnl = sum(wr.train_pnl for wr in window_results)
        wfe = oos_pnl / is_pnl if is_pnl != 0 else float("nan")
        stitched = self._to_test_result(
            start_dt,
            end_dt,
            portfolio,
            calc_bootstrap=True,
            train_only=False,
            signals=None,
            seed=None,
        )
        last = window_results[-1]
        return OptimizeResult(
            best_params=last.params,
            best_score=last.train_score,
            result=stitched,
            study=last.study,
            windows=tuple(window_results),
            walkforward_efficiency=wfe,
        )


def _static_symbols_from_executions(
    executions: set[Execution], df: pd.DataFrame
) -> set[str]:
    syms: set[str] = set()
    sym_col = DataCol.SYMBOL.value
    if sym_col in df.columns:
        loaded = set(df[sym_col].unique())
    else:
        loaded = set()
    for execution in executions:
        for sym in _strategy_module()._static_symbols(execution.symbols):
            if not loaded or sym in loaded:
                syms.add(sym)
    return syms
