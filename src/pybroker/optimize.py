"""Hyperparameter declaration and optimization with Optuna.

Hyperparams declare tunable values for indicators and executions. Each
hyperparam is registered globally by name via :func:`hyperparam` and
resolved to a concrete int or float at backtest or optimization time.

Pass hyperparams as keyword arguments to
:func:`pybroker.indicator.indicator`, or list them on
:meth:`pybroker.strategy.Strategy.add_execution` to read them inside an
execution with ``ctx.hyperparam(name)``.
"""

from __future__ import annotations

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import copy
import json
import math
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Union,
    cast,
)

import numpy as np
import optuna
import pandas as pd
from joblib import delayed
from optuna.distributions import BaseDistribution, CategoricalDistribution
from optuna.samplers import BaseSampler, GridSampler, RandomSampler, TPESampler

from pybroker.scope import StaticScope


@dataclass(frozen=True)
class Hyperparam:
    """Declares a named hyperparameter with bounds and step size.

    Created with :func:`hyperparam` and registered globally by ``name``.

    Attributes:
        name: Unique identifier used in indicator kwargs, execution
            hyperparam lists, and optimization results.
        default: Value for backtests and the baseline during optimization.
            Should lie within ``[low, high]``.
        low: Minimum candidate value searched during optimize (inclusive).
        high: Maximum candidate value searched during optimize (inclusive).
            Candidate values are ``low``, ``low + step``, ... up to the
            largest value not exceeding ``high``.
        step: Spacing between candidate values. Must be positive. Integer
            hyperparams use integer steps; float hyperparams use float
            steps with values rounded to match Optuna stepped suggestions.

    Examples:
        Indicator period from 5 to 50 in steps of 5::

            period = hyperparam("period", default=14, low=5, high=50, step=5)
    """

    name: str
    default: Union[int, float]
    low: Union[int, float]
    high: Union[int, float]
    step: Union[int, float]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("default", self.default),
            ("low", self.low),
            ("high", self.high),
            ("step", self.step),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Hyperparam {self.name!r}: {field_name} must be int or "
                    f"float, got {type(value).__name__}."
                )
        value_types = {
            type(self.default),
            type(self.low),
            type(self.high),
            type(self.step),
        }
        if value_types != {int} and value_types != {float}:
            raise TypeError(
                f"Hyperparam {self.name!r}: default, low, high, and step must "
                "all be int or all be float."
            )
        if self.step <= 0:
            raise ValueError(
                f"Hyperparam {self.name!r}: step must be positive."
            )
        if self.low > self.high:
            raise ValueError(
                f"Hyperparam {self.name!r}: low cannot exceed high."
            )
        if self.low != self.high:
            span = Decimal(str(self.high)) - Decimal(str(self.low))
            if span % Decimal(str(self.step)) != 0:
                raise ValueError(
                    f"Hyperparam {self.name!r}: high - low "
                    f"({self.high} - {self.low}) must be a multiple of step "
                    f"({self.step}); otherwise the largest candidate value "
                    "falls short of high."
                )

    def _is_float(self) -> bool:
        return isinstance(self.default, float)

    def _decimals(self) -> tuple[Decimal, Decimal, Decimal]:
        """Returns ``(low, high, step)`` as exact decimals.

        Candidates are generated as ``low + i * step`` in :class:`Decimal`
        rather than binary floats so that the lattice is exactly the one the
        user declared. Rounding the running value to a precision guessed from
        ``step`` would snap candidates onto a coarser decimal grid: a ``step``
        of ``0.25`` would yield ``0.2`` and ``0.8``, values that Optuna's own
        :class:`optuna.distributions.FloatDistribution` rejects.
        """
        return (
            Decimal(str(self.low)),
            Decimal(str(self.high)),
            Decimal(str(self.step)),
        )

    def _within_high(self, val: Union[int, float]) -> bool:
        if self._is_float():
            _, high, _ = self._decimals()
            return Decimal(str(val)) <= high
        return int(val) <= int(self.high)

    def _lattice_count(self) -> int:
        if self.low == self.high:
            raise TypeError(
                f"Hyperparam {self.name!r} is fixed; lattice is undefined."
            )
        if self._is_float():
            low, high, step = self._decimals()
            return int((high - low) // step) + 1
        low_i = int(self.low)
        high_i = int(self.high)
        step_i = int(self.step)
        return (high_i - low_i) // step_i + 1

    def _lattice_values(self) -> Iterator[Union[int, float]]:
        if self.low == self.high:
            raise TypeError(
                f"Hyperparam {self.name!r} is fixed; lattice is undefined."
            )
        if self._lattice_count() == 0:
            raise ValueError(
                f"Hyperparam {self.name!r}: empty lattice for "
                f"low={self.low}, high={self.high}, step={self.step}."
            )
        if self._is_float():
            low, high, step = self._decimals()
            val = low
            while val <= high:
                yield float(val)
                val += step
        else:
            low_i = int(self.low)
            high_i = int(self.high)
            step_i = int(self.step)
            val_i = low_i
            while val_i <= high_i:
                yield val_i
                val_i += step_i

    def __iter__(self) -> Iterator[Union[int, float]]:
        return self._lattice_values()

    def __len__(self) -> int:
        return self._lattice_count()


def hyperparam(
    name: str,
    *,
    default: Union[int, float],
    low: Union[int, float],
    high: Union[int, float],
    step: Union[int, float],
) -> Hyperparam:
    """Creates and registers a :class:`Hyperparam`.

    Args:
        name: Unique identifier for the hyperparam. Referenced in indicator
            kwargs, ``add_execution(..., hyperparams=[...])``, and
            ``ctx.hyperparam(name)``.
        default: Value used for backtests.
        low: Minimum candidate value searched during optimize (inclusive).
        high: Maximum candidate value searched during optimize (inclusive).
        step: Spacing between candidate values. Must be positive.

    Returns:
        The registered :class:`Hyperparam` instance.
    """
    hp = Hyperparam(name=name, default=default, low=low, high=high, step=step)
    StaticScope.instance().set_hyperparam(hp)
    return hp


def _is_hyperparam(value: Any) -> bool:
    return isinstance(value, Hyperparam)


def _find_hyperparam_names(mapping: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        value.name for value in mapping.values() if _is_hyperparam(value)
    )


def _resolve_hyperparams(
    mapping: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    """Replaces :class:`Hyperparam` values in ``mapping`` with run values.

    Args:
        mapping: Keyword arguments that may contain :class:`Hyperparam`
            instances (for example, indicator ``_kwargs``).
        params: Dict of ``name -> value`` for the current run.

    Returns:
        A new dict with hyperparams replaced by their resolved values.
    """
    resolved: dict[str, Any] = {}
    for key, value in mapping.items():
        if _is_hyperparam(value):
            if value.name not in params:
                raise KeyError(
                    f"Hyperparam {value.name!r} is not in the run hyperparams "
                    "dict."
                )
            resolved[key] = params[value.name]
        else:
            resolved[key] = value
    return resolved


def _hyperparam_specs_from_kwargs(
    mapping: Mapping[str, Any],
) -> dict[str, Hyperparam]:
    return {
        value.name: value
        for value in mapping.values()
        if _is_hyperparam(value)
    }


@dataclass(frozen=True)
class SearchSpace:
    """Searchable hyperparameters collected from a strategy.

    Only includes hyperparams with ``low < high`` that are passed to Optuna
    during :meth:`pybroker.strategy.Strategy.optimize`.

    Attributes:
        hyperparams: Names of hyperparams searched during optimize.
        specs: Mapping of hyperparam name to :class:`Hyperparam` spec.
    """

    hyperparams: frozenset[str]
    specs: Mapping[str, Hyperparam]

    def grid_size(self) -> int:
        """Total number of grid combinations."""
        size = 1
        for name in self.hyperparams:
            size *= len(self.specs[name])
        return size


def build_run_hyperparams(
    specs: Mapping[str, Hyperparam],
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Builds the hyperparam dict for a single backtest or trial run.

    Args:
        specs: All hyperparams reachable from the strategy.
        overrides: Trial or user-supplied values to merge over defaults.

    Returns:
        Dict of ``name -> value`` for every hyperparam in ``specs``.
    """
    result: dict[str, Any] = {name: specs[name].default for name in specs}
    if overrides:
        for name, value in overrides.items():
            if name not in specs:
                raise ValueError(
                    f"Unknown hyperparam override {name!r}. "
                    f"Declared: {sorted(specs)}."
                )
            result[name] = value
    return result


from pybroker.cache import CacheDateFields
from pybroker.common import (
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    _ensure_range_index,
    _json_safe,
    _resolve_executions,
    _selected_symbols,
    _selection_df,
    _static_symbols,
    get_unique_sorted_dates,
    to_datetime,
    to_seconds,
    verify_date_range,
)
from pybroker.parallel import _effective_n_jobs, parallel
from pybroker.portfolio import Portfolio
from pybroker.scope import (
    ColumnScope,
    PendingOrderScope,
    run_with_scope,
    symbol_array_store_from_frame,
)
from pybroker.interval import (
    model_interval_name,
    parse_indicator_interval_name,
    parse_model_interval_name,
    symbol_dates_from_frame,
)

if TYPE_CHECKING:
    from pybroker.common import TrainedModel
    from pybroker.config import StrategyConfig
    from pybroker.context import ExecContext, RotationContext
    from pybroker.slippage import SlippageModel
    from pybroker.strategy import (
        Execution,
        Strategy,
        TestResult,
        WalkforwardWindow,
    )
    from pybroker.interval import IntervalData

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
            interval_data: Any,
            enable_parallel_indicators: bool,
            warmup: Optional[int],
            pretrained_models: Mapping[ModelSymbol, TrainedModel],
            exit_dates: Mapping[str, np.datetime64],
        ) -> TestResult: ...


_MODEL_OPTIMIZE_ERROR = (
    "optimize() does not support trainable model sources "
    "({trainable}). Pretrained models (pretrained=True) are supported. "
    "Tune trainable model hyperparameters inside train_fn with a validation "
    "split, or use walkforward() for exec params that depend on model "
    "outputs."
)

_GRID_EXPLOSION_THRESHOLD = 1000
_DEFAULT_INDICATOR_MEMO_MAX = 256


def _is_trainable_model_source(source: object) -> bool:
    return hasattr(source, "_train_fn")


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

    def add_indicator_hyperparams(ind_name: str) -> None:
        base, _ = parse_indicator_interval_name(ind_name)
        ind = scope.get_indicator(base)
        for hp_name in ind.hyperparam_names:
            if not scope.has_hyperparam(hp_name):
                raise ValueError(
                    f"Hyperparam {hp_name!r} in indicator "
                    f"{ind_name!r} is not registered."
                )
            names.add(hp_name)
            specs[hp_name] = scope.get_hyperparam(hp_name)

    for execution in strategy._executions:
        for ind_name in execution.indicator_names:
            add_indicator_hyperparams(ind_name)

        # Indicators registered on a model source are expanded by
        # Strategy._fetch_indicators even when they are not also listed on the
        # execution, so their hyperparams have to be collected here or the run
        # hyperparams dict is missing a name the indicator asks for.
        for model_name in execution.model_names:
            base_name, _ = parse_model_interval_name(model_name)
            for ind_name in scope.get_indicator_names(base_name):
                add_indicator_hyperparams(ind_name)

        for hp_name in execution.hyperparam_names:
            if not scope.has_hyperparam(hp_name):
                raise ValueError(f"Hyperparam {hp_name!r} was not registered.")
            names.add(hp_name)
            specs[hp_name] = scope.get_hyperparam(hp_name)

        for model_name in execution.model_names:
            base_name, _ = parse_model_interval_name(model_name)
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


def _search_space_from_specs(
    specs: Mapping[str, Hyperparam],
) -> SearchSpace:
    """Builds the searchable subset of already collected ``specs``."""
    searchable = {name: hp for name, hp in specs.items() if hp.low < hp.high}
    return SearchSpace(frozenset(searchable), searchable)


def collect_search_space(strategy: _ExecutionsHost) -> SearchSpace:
    """Collects searchable hyperparams reachable from ``strategy``."""
    return _search_space_from_specs(collect_hyperparams(strategy))


def _validate_optimize_models(strategy: _ExecutionsHost) -> None:
    scope = StaticScope.instance()
    trainable: list[str] = []
    for execution in strategy._executions:
        for model_name in execution.model_names:
            base_name, _ = parse_model_interval_name(model_name)
            source = scope.get_model_source(base_name)
            if _is_trainable_model_source(source):
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
    # Sorted, not frozenset order: samplers draw from one sequential RNG, so
    # the order of suggest_* calls decides which draw lands on which
    # hyperparam. Iterating the frozenset would vary with string hash
    # randomization and make ``seed`` fail to reproduce across processes.
    return {
        name: _suggest_from_spec(trial, search_space.specs[name])
        for name in sorted(search_space.hyperparams)
    }


def _reseed_sampler(sampler: BaseSampler, seed: int, _depth: int = 0) -> None:
    """Re-seeds ``sampler``'s RNGs in place, deterministically.

    Optuna's samplers hold their generator on a private ``_rng`` wrapped in a
    ``LazyRandomState``, and there is no public *deterministic* re-seed hook
    (``reseed_rng`` draws fresh entropy). This reaches for the private state
    defensively and leaves a sampler that does not expose it -- a third-party
    subclass, say -- untouched rather than raising.

    Recurses into delegated inner samplers: ``TPESampler`` draws its first
    ``n_startup_trials`` from an inner ``RandomSampler``, so with the small
    per-window trial counts a walkforward uses, re-seeding only the outer
    ``_rng`` changes none of the draws that matter and every window still
    explores one candidate set. Each level gets a distinct derived seed so
    inner and outer streams do not mirror each other.
    """
    if _depth > 3:
        return
    rng = getattr(sampler, "_rng", None)
    if rng is not None:
        state = np.random.RandomState(seed)
        try:
            rng.rng = state
        except AttributeError:
            if hasattr(rng, "_rng"):
                rng._rng = state
    for offset, attr in enumerate(
        ("_random_sampler", "_independent_sampler", "_base_sampler")
    ):
        inner = getattr(sampler, attr, None)
        if isinstance(inner, BaseSampler):
            _reseed_sampler(inner, seed + 1000003 * (offset + 1), _depth + 1)


def _build_sampler(
    sampler: Union[str, BaseSampler],
    search_space: SearchSpace,
    seed: Optional[int],
) -> BaseSampler:
    if isinstance(sampler, BaseSampler):
        # Copied, and re-seeded when it carries a seedable RNG. Returning the
        # instance itself hands every walkforward window the same sampler:
        # sequentially they share one advancing RNG, and under n_jobs > 1 each
        # worker gets a pickled copy at the same state, so every window
        # explores an identical candidate set. Either way the per-window seed
        # is discarded and ``seed=`` does not make the run reproducible.
        try:
            sampler = copy.deepcopy(sampler)
        except Exception:
            # A sampler holding an unpicklable handle (a lock, a DB
            # connection) cannot be copied. Re-seeding the instance itself
            # still gives each sequential window distinct draws; under
            # n_jobs > 1 the pickle to the worker fails with its own error.
            pass
        if seed is not None:
            _reseed_sampler(sampler, seed)
        else:
            # No seed means no reproducibility contract, but a deepcopy of an
            # already-materialized RNG replays one candidate set in every
            # window -- worse than the shared advancing state it replaced.
            # reseed_rng is optuna's public hook; TPESampler's implementation
            # recurses into its startup RandomSampler itself.
            sampler.reseed_rng()
        return sampler
    if sampler == "grid":
        grid = {
            name: list(search_space.specs[name])
            for name in sorted(search_space.hyperparams)
        }
        # GridSampler shuffles the grid, so seed it too or a truncated grid
        # search (n_trials < grid_size) is not reproducible.
        return GridSampler(grid, seed=seed)
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


def _validate_study_direction(study: optuna.Study, direction: str) -> None:
    """Rejects a supplied ``study`` whose direction contradicts ``direction``.

    ``optuna.create_study`` defaults to minimizing, so silently deferring to
    the study would return the worst trial as the best one.
    """
    expected = direction.strip().lower()
    if expected not in ("minimize", "maximize"):
        raise ValueError(
            f"Unknown direction {direction!r}; use 'minimize' or 'maximize'."
        )
    actual = study.direction.name.lower()
    if actual != expected:
        raise ValueError(
            f"study= has direction {actual!r} but direction={direction!r} was "
            "requested. Create the study with the matching direction, or omit "
            "direction."
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
        StaticScope.instance().logger.info_optimize_search_space(searched)


def _study_summary(study: optuna.Study) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_trials": len(study.trials)}
    # Tested by state, not by ``study.best_trial is not None``: optuna raises
    # "No trials are completed yet" from that property rather than returning
    # None, so summarizing a study whose every trial failed -- a score_fn that
    # returned NaN throughout, say -- aborted with an error about optuna
    # internals instead of reporting an empty result.
    if study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    ):
        summary["best_value"] = study.best_value
        summary["best_params"] = study.best_params
    else:
        summary["best_value"] = None
        summary["best_params"] = {}
    if study.user_attrs:
        summary["user_attrs"] = dict(study.user_attrs)
    return _json_safe(summary)


@dataclass(frozen=True)
class WindowOptimizeResult:
    """Per-window walk-forward optimization result."""

    params: dict[str, Any]
    study: optuna.Study
    test_result: TestResult
    train_score: float
    train_pnl: float
    # Symbols each execution id resolved to for this window. Carried back from
    # the study so the final replay reuses the study's selection instead of
    # running the SymbolSelector a second time, which a stateful selector would
    # answer differently. ``None`` when no execution uses a selector.
    execution_symbols: Optional[dict[int, frozenset[str]]] = None

    def to_json(
        self,
        *,
        include: Optional[frozenset[str]] = None,
        max_rows: Optional[int] = 100,
        symbols: Optional[frozenset[str]] = None,
    ) -> dict[str, Any]:
        """Returns JSON-serializable walk-forward optimization window results."""
        from pybroker.strategy import _DEFAULT_JSON_INCLUDE

        if include is None:
            include = _DEFAULT_JSON_INCLUDE
        return _json_safe(
            {
                "params": self.params,
                "train_score": self.train_score,
                "train_pnl": self.train_pnl,
                "study": _study_summary(self.study),
                "test_result": self.test_result.to_json(
                    include=include,
                    max_rows=max_rows,
                    symbols=symbols,
                ),
            }
        )

    def to_json_str(
        self,
        *,
        include: Optional[frozenset[str]] = None,
        max_rows: Optional[int] = 100,
        symbols: Optional[frozenset[str]] = None,
    ) -> str:
        """Returns strict JSON text from :meth:`to_json`."""
        return json.dumps(
            self.to_json(
                include=include,
                max_rows=max_rows,
                symbols=symbols,
            ),
            allow_nan=False,
        )


@dataclass(frozen=True)
class OptimizeResult:
    r"""Result of :meth:`Strategy.optimize`.

    Attributes:
        best_params: Winning hyperparameter values, including the ones that were
            fixed rather than searched. When ``windows`` is set, these are the
            **last** window's values, since each window is tuned separately.
        best_score: ``score_fn`` value that ``best_params`` earned on the train
            window. When ``windows`` is set, this is the last window's score.
        result: :class:`pybroker.strategy.TestResult` for the test window. When
            ``windows`` is set, this is a single continuous result stitched from
            every window's test data, with positions and cash carried across
            window boundaries -- so it is *not* the same as
            ``windows[-1].test_result``.
        study: :class:`optuna.Study` holding the trials. When ``windows`` is set,
            this is the last window's study; see ``windows`` for the rest.
        windows: Per-window results, or ``None`` for a single train/test split.
        walkforward_efficiency: Sum of the windows' out-of-sample profit divided
            by the sum of their in-sample profit. Above ``1`` means the tuned
            values held up out of sample. ``None`` for a single split, or when
            in-sample profit was not positive, which leaves the ratio undefined.
    """

    best_params: dict[str, Any]
    best_score: float
    result: TestResult
    study: optuna.Study
    windows: Optional[tuple[WindowOptimizeResult, ...]] = None
    walkforward_efficiency: Optional[float] = None

    def to_json(
        self,
        *,
        include: Optional[frozenset[str]] = None,
        max_rows: Optional[int] = 100,
        symbols: Optional[frozenset[str]] = None,
    ) -> dict[str, Any]:
        """Returns JSON-serializable optimization results."""
        from pybroker.strategy import _DEFAULT_JSON_INCLUDE

        if include is None:
            include = _DEFAULT_JSON_INCLUDE
        payload: dict[str, Any] = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "walkforward_efficiency": self.walkforward_efficiency,
            "study": _study_summary(self.study),
            "result": self.result.to_json(
                include=include,
                max_rows=max_rows,
                symbols=symbols,
            ),
        }
        if self.windows is not None:
            payload["windows"] = [
                window.to_json(
                    include=include,
                    max_rows=max_rows,
                    symbols=symbols,
                )
                for window in self.windows
            ]
        return _json_safe(payload)

    def to_json_str(
        self,
        *,
        include: Optional[frozenset[str]] = None,
        max_rows: Optional[int] = 100,
        symbols: Optional[frozenset[str]] = None,
    ) -> str:
        """Returns strict JSON text from :meth:`to_json`."""
        return json.dumps(
            self.to_json(
                include=include,
                max_rows=max_rows,
                symbols=symbols,
            ),
            allow_nan=False,
        )


@dataclass(frozen=True)
class ObjectiveBundle:
    """Return value of :func:`make_objective`."""

    objective: Callable[[optuna.Trial], float]
    search_space: SearchSpace
    score_overrides: Callable[[dict[str, Any]], float]


def _sampler_grid(
    sampler: GridSampler,
) -> tuple[list[dict[str, Any]], dict[str, BaseDistribution]]:
    """Returns ``sampler``'s grid points and their matching distributions.

    Points come back in the order the sampler will visit them:
    :class:`optuna.samplers.GridSampler` shuffles its grid at construction
    (seeded by ``seed or 0``) so that a truncated search does not just take the
    lexicographic prefix. Reading that order back is what keeps the parallel
    path evaluating the same points as the sequential one.

    Distributions are derived from the same grid so their names always match the
    points', including when a caller supplied a sampler covering only some of
    the declared hyperparams.
    """
    names = sampler._param_names
    combos = [dict(zip(names, grid)) for grid in sampler._all_grids]
    dists: dict[str, BaseDistribution] = {
        name: CategoricalDistribution(list(sampler._search_space[name]))
        for name in names
    }
    return combos, dists


def _run_study(
    study: optuna.Study,
    bundle: ObjectiveBundle,
    n_trials: int,
    sampler: BaseSampler,
) -> None:
    """Runs ``n_trials`` of ``study``, in parallel when configured.

    Grid search needs no feedback between trials, so every point is enumerated
    up front and evaluated at once, then registered as a completed trial. This
    also sidesteps ``GridSampler.after_trial`` calling ``Study.stop()``, which
    raises outside of :meth:`optuna.Study.optimize`.

    Other samplers are adaptive, so trials are evaluated in batches: each batch
    is asked, scored concurrently, then told back before the next is drawn.
    Batching costs the sampler some information (a batch cannot observe its own
    results), which matters for :class:`optuna.samplers.TPESampler` but not for
    :class:`optuna.samplers.RandomSampler`.
    """
    if isinstance(sampler, GridSampler):
        # A supplied sampler may cover only some of the declared hyperparams.
        # Sequentially that raises when the objective asks for a missing name;
        # in parallel the point simply lacks it and the default is silently
        # used, so the two paths would search different spaces. Reject it up
        # front on both.
        missing = sorted(
            set(bundle.search_space.hyperparams) - set(sampler._param_names)
        )
        if missing:
            raise ValueError(
                "GridSampler search space is missing declared "
                f"hyperparameter(s): {missing}. Include them in the grid "
                "passed to GridSampler, or let optimize() build the sampler."
            )
    workers = _effective_n_jobs()
    if workers <= 1:
        study.optimize(bundle.objective, n_trials=n_trials)
        return
    scope = StaticScope.instance()
    if isinstance(sampler, GridSampler):
        # Taken from the sampler's own already-shuffled grid rather than
        # re-enumerating it. itertools.product order would pin every hyperparam
        # but the last to its lowest values whenever n_trials < grid_size, and
        # reshuffling here with a different RNG would still not agree with the
        # sequential path, where GridSampler picks the points itself.
        combos, dists = _sampler_grid(sampler)
        # Offset by the trials already in the study. The sequential path lets
        # GridSampler.before_trial map grid_id from trial.number, so a resumed
        # study continues where it left off; slicing from 0 re-evaluates the
        # same opening points and never reaches the rest of the grid.
        start = len(study.trials)
        combos = combos[start : start + n_trials]
        with parallel() as pool:
            scores = pool(
                delayed(_run_scoped_task)(
                    scope, bundle.score_overrides, params
                )
                for params in combos
            )
        for params, score in zip(combos, scores):
            if _is_failed_score(score):
                # create_trial rejects NaN just as study.tell does. Record the
                # trial as failed, matching what study.optimize does on the
                # sequential path, instead of aborting the whole study.
                study.add_trial(
                    optuna.trial.create_trial(
                        params=params,
                        distributions=dists,
                        state=optuna.trial.TrialState.FAIL,
                    )
                )
                continue
            study.add_trial(
                optuna.trial.create_trial(
                    params=params, distributions=dists, value=score
                )
            )
        return
    remaining = n_trials
    stopped = False
    while remaining > 0 and not stopped:
        size = min(workers, remaining)
        trials = [study.ask() for _ in range(size)]
        overrides = [
            _trial_params(trial, bundle.search_space) for trial in trials
        ]
        with parallel() as pool:
            scores = pool(
                delayed(_run_scoped_task)(
                    scope, bundle.score_overrides, params
                )
                for params in overrides
            )
        # A sampler may call Study.stop() from after_trial (BruteForceSampler
        # does once the space is exhausted). Outside Study.optimize that
        # raises instead of setting the stop flag, so declare the loop: this
        # batch is an optimize loop, just not optuna's own.
        with _optimize_loop(study):
            for trial, score in zip(trials, scores):
                if _is_failed_score(score):
                    # study.tell rejects NaN, while study.optimize records the
                    # trial as failed and carries on. Match the sequential
                    # path rather than abort the study on one bad trial.
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    continue
                # Every score in this batch has already been paid for, so all
                # of them are recorded even once a stop is requested. Breaking
                # here would abandon them in RUNNING and report a best trial
                # that ignores results already in hand.
                study.tell(trial, score)
            stopped = bool(study._stop_flag)
        remaining -= size


@contextmanager
def _optimize_loop(study: optuna.Study) -> Iterator[None]:
    """Marks ``study`` as being inside an optimization loop.

    :meth:`optuna.Study.stop` raises unless it is called from within
    ``Study.optimize``; a sampler that stops from ``after_trial`` therefore
    blows up when trials are driven by ``ask``/``tell`` instead. Declaring the
    loop lets ``stop`` set its flag as it normally would, which is then read
    back to end the batch loop.
    """
    thread_local = getattr(study, "_thread_local", None)
    if thread_local is None:  # pragma: no cover - optuna internals changed
        yield
        return
    previous = getattr(thread_local, "in_optimize_loop", False)
    thread_local.in_optimize_loop = True
    try:
        yield
    finally:
        thread_local.in_optimize_loop = previous


def _is_failed_score(score: Any) -> bool:
    """Whether ``score`` must be recorded as a failed trial.

    Optuna rejects a non-numeric or NaN objective value outright, both in
    ``Study.tell`` and in ``create_trial``, while ``Study.optimize`` marks the
    trial failed and continues. Covers ``numpy`` floats, which are not
    ``float`` instances.
    """
    if score is None:
        return True
    try:
        return math.isnan(float(score))
    except (TypeError, ValueError):
        return True


def _run_scoped_task(scope: StaticScope, fn: Callable[..., Any], *args) -> Any:
    """Installs ``scope`` as this process' scope, then runs ``fn``.

    :class:`pybroker.scope.StaticScope` is a per-process singleton, so a worker
    process starts with an empty one and would not see the caller's registered
    indicators, model sources, params or custom columns. Running sequentially,
    ``scope`` is already the installed instance and this is a no-op.
    """
    return run_with_scope(scope, fn, *args)


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
    interval_data: Any,
    enable_parallel_indicators: bool,
    warmup: Optional[int],
    pretrained_models: Mapping[ModelSymbol, TrainedModel],
    exit_dates: Mapping[str, np.datetime64],
) -> ObjectiveBundle:
    """Builds an Optuna objective for train-window scoring."""

    def score_overrides(overrides: dict[str, Any]) -> float:
        """Scores one already-resolved set of hyperparam values.

        Kept separate from ``objective`` so trials can be evaluated in worker
        processes: an :class:`optuna.Trial` holds a reference to its study and
        must not cross a process boundary, but a plain params ``dict`` can.
        """
        run_hp = build_run_hyperparams(hyperparams, overrides)
        result = strategy._run_optimize_trial(
            df=df,
            train_rows=train_rows,
            run_hyperparams=run_hp,
            invariant_indicator_data=invariant_indicator_data,
            window_executions=window_executions,
            master_store=master_store,
            interval_data=interval_data,
            enable_parallel_indicators=enable_parallel_indicators,
            warmup=warmup,
            pretrained_models=pretrained_models,
            exit_dates=exit_dates,
        )
        return score_fn(result)

    def objective(trial: optuna.Trial) -> float:
        return score_overrides(_trial_params(trial, search_space))

    return ObjectiveBundle(
        objective=objective,
        search_space=search_space,
        score_overrides=score_overrides,
    )


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
        _start_date: datetime
        _end_date: datetime
        _indicator_memo_max: int

        def _fractional_shares_enabled(self) -> bool: ...

        def train_models(
            self, *args: Any, **kwargs: Any
        ) -> dict[ModelSymbol, TrainedModel]: ...

        def _fetch_indicators(
            self, *args: Any, **kwargs: Any
        ) -> dict[IndicatorSymbol, pd.Series]: ...

        def _indicator_syms(
            self, executions: Optional[set[Execution]] = None
        ) -> set[IndicatorSymbol]: ...

        def _build_window_stores(
            self, *args: Any, **kwargs: Any
        ) -> tuple[Any, Any, Any]: ...

        def _build_exit_dates(
            self, df: pd.DataFrame, has_selector: bool
        ) -> dict[str, np.datetime64]: ...

        def _indicator_memo_store(self) -> dict[Any, pd.Series]: ...

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

        def _has_symbol_selector(self) -> bool: ...

        def _build_interval_data(
            self, *args: Any, **kwargs: Any
        ) -> IntervalData: ...

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
        interval_data: Any,
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
        train_store, test_store, history_store = self._build_window_stores(
            master_store=master_store,
            master_dates_arr=master_dates_arr,
            train_rows=train_rows,
            test_rows=test_rows,
            train_empty=train_data.empty,
            test_empty=test_data.empty,
        )
        train_symbols = set(train_data[sym_col].unique())
        model_syms: set[ModelSymbol] = set()
        for sym in train_symbols:
            for execution in window_executions:
                if sym not in _static_symbols(execution.symbols):
                    continue
                for model_name in execution.model_names:
                    base_name, token = parse_model_interval_name(model_name)
                    if token is not None:
                        model_syms.add(ModelSymbol(model_name, sym))
                        continue
                    model_syms.add(ModelSymbol(model_name, sym))
                    if not _is_trainable_model_source(
                        StaticScope.instance().get_model_source(base_name)
                    ):
                        # Pretrained models stay bound to the base timeframe.
                        continue
                    for tf in execution.intervals:
                        model_syms.add(
                            ModelSymbol(
                                model_interval_name(base_name, tf), sym
                            )
                        )
        pooled_model_groups: dict[tuple[str, int], frozenset[str]] = {}
        for execution in window_executions:
            exec_syms = frozenset(
                sym
                for sym in _static_symbols(execution.symbols)
                if sym in train_symbols
            )
            if not exec_syms:
                continue
            for model_name in execution.model_names:
                base_name, token = parse_model_interval_name(model_name)
                if token is not None:
                    continue
                source = StaticScope.instance().get_model_source(base_name)
                if _is_trainable_model_source(source) and source.pooled:
                    pooled_model_groups[(model_name, execution.id)] = exec_syms
                    for tf in execution.intervals:
                        pooled_model_groups[
                            (model_interval_name(base_name, tf), execution.id)
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
            interval_data=interval_data,
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
        interval_data: Any,
        enable_parallel_indicators: bool,
        warmup: Optional[int],
        pretrained_models: Mapping[ModelSymbol, TrainedModel],
        exit_dates: Mapping[str, np.datetime64],
    ) -> TestResult:
        train_data = df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
        # Only the hyperparameterized indicators vary per trial. Fetching the
        # whole set would recompute the invariant ones on every trial and
        # discard ``invariant_indicator_data`` by overwriting it in the merge
        # below, since hyperparam-free indicators are neither disk cached here
        # (cache_date_fields=None) nor memoized (Indicator._memo_key is None
        # without hyperparams).
        _, tuned_syms = self._partition_indicator_syms(window_executions)
        trial_indicators = (
            self._fetch_indicators(
                df=df,
                cache_date_fields=None,
                enable_parallel_indicators=enable_parallel_indicators,
                interval_data=interval_data,
                executions=window_executions,
                symbol_store=master_store,
                hyperparams=run_hyperparams,
                indicator_syms=tuned_syms,
            )
            if tuned_syms
            else {}
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
        train_store, _, _ = self._build_window_stores(
            master_store=master_store,
            master_dates_arr=df[date_col].to_numpy(
                dtype="datetime64[ns]", copy=False
            ),
            train_rows=train_rows,
            test_rows=train_rows[:0],
            train_empty=train_data.empty,
            test_empty=True,
        )
        sessions: dict[str, dict] = defaultdict(dict)
        for sym in _static_symbols_from_executions(
            window_executions, train_data
        ):
            sessions[sym] = {}
        signals = self.backtest_executions(
            config=effective_config,
            executions=window_executions,
            before_exec_fn=self._before_exec_fn,
            after_exec_fn=self._after_exec_fn,
            sessions=sessions,
            models=pretrained_models,
            indicator_data=indicator_data,
            interval_data=interval_data.slice_for_test(
                symbol_dates_from_frame(train_data)
            ),
            test_data=train_data,
            portfolio=portfolio,
            exit_dates=dict(exit_dates),
            backtest_settings=backtest_settings,
            rotation_sizer=self._rotation_sizer,
            slippage_model=self._slippage_model,
            enable_fractional_shares=self._fractional_shares_enabled(),
            round_fill_price=effective_config.round_fill_price,
            warmup=warmup,
            history_col_scope=ColumnScope(train_store)
            if train_store is not None
            else None,
            test_col_scope=ColumnScope(train_store)
            if train_store is not None
            else None,
            run_hyperparams=run_hyperparams,
            master_col_scope=ColumnScope(master_store),
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
            signals=signals if self._config.return_signals else None,
            seed=None,
        )

    def _partition_indicator_syms(
        self, executions: set[Execution]
    ) -> tuple[set[IndicatorSymbol], set[IndicatorSymbol]]:
        """Splits ``executions``' indicator symbols into invariant and tuned.

        Returns ``(invariant, tuned)``, where ``invariant`` names indicators
        that declare no hyperparams and so produce the same series for every
        trial, and ``tuned`` names the rest. Partitioning the set that
        :meth:`pybroker.strategy.Strategy._indicator_syms` builds keeps model
        registered indicators and per-interval variants on the invariant side
        instead of leaving them to be recomputed each trial.
        """
        scope = StaticScope.instance()
        invariant: set[IndicatorSymbol] = set()
        tuned: set[IndicatorSymbol] = set()
        for ind_sym in self._indicator_syms(executions):
            base, _ = parse_indicator_interval_name(ind_sym.ind_name)
            if scope.get_indicator(base).hyperparam_names:
                tuned.add(ind_sym)
            else:
                invariant.add(ind_sym)
        return invariant, tuned

    def _compute_invariant_indicators(
        self,
        df: pd.DataFrame,
        cache_date_fields: CacheDateFields,
        enable_parallel_indicators: bool,
        interval_data: Any,
        master_store: Any,
        executions: set[Execution],
    ) -> dict[IndicatorSymbol, pd.Series]:
        ind_syms, _ = self._partition_indicator_syms(executions)
        if not ind_syms:
            return {}
        return self.compute_indicators(
            df=df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            enable_parallel_indicators=enable_parallel_indicators,
            interval_data=interval_data,
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
        enable_parallel_indicators: bool = False,
        adjust: Optional[Any] = None,
        calc_bootstrap: bool = False,
        indicator_memo_max: int = _DEFAULT_INDICATOR_MEMO_MAX,
    ) -> OptimizeResult:
        r"""Searches :func:`pybroker.optimize.hyperparam` values on a training
        window, then evaluates the best values on the held out test window.

        Data supplied by the :class:`pybroker.data.DataSource` is split into
        train and test as specified by ``train_size``. Every trial backtests the
        train window with one combination of hyperparameter values and scores it
        with ``score_fn``. The winning combination is then replayed on the test
        window, which ``score_fn`` never sees.

        Pretrained models (``model(..., pretrained=True)``) are loaded per train
        window and reused across that window's trials. Trainable models are not
        supported; tune them inside ``train_fn`` with a validation split, or use
        :meth:`pybroker.strategy.Strategy.walkforward`.

        Args:
            score_fn: ``Callable[[TestResult], float]`` that scores one trial's
                train window backtest. Maximized by default; see ``direction``.
            sampler: How candidate values are chosen. ``"grid"`` (the default)
                exhaustively enumerates every combination, ``"tpe"`` uses
                :class:`optuna.samplers.TPESampler`, ``"random"`` uses
                :class:`optuna.samplers.RandomSampler`. An
                :class:`optuna.samplers.BaseSampler` instance is also accepted.
            n_trials: Number of trials to run. Required for every sampler except
                ``"grid"``, where it defaults to the full grid size and a
                smaller value samples that many combinations at random.
            direction: ``"maximize"`` (default) or ``"minimize"`` ``score_fn``.
            seed: Random seed for the sampler and for bootstrap metrics.
                Defaults to ``None``, which does not reproduce.
            windows: When greater than ``1``, hyperparameters are optimized
                separately in each of ``windows`` walkforward windows and the
                test windows are stitched into one continuous result. Defaults
                to ``None``, a single train/test split.
            study: Existing :class:`optuna.Study` to record trials in, for
                example one backed by persistent storage. The study's own
                sampler and pruner are used, and its direction must match
                ``direction``. Not supported when ``windows`` is greater
                than ``1``.
            pruner: :class:`optuna.pruners.BasePruner` attached to the created
                study. Each trial is one complete backtest with no intermediate
                values to report, so pruning never actually triggers.
            train_size: Fraction of each window used for training, exclusive of
                ``0`` and ``1``. Defaults to ``0.5``.
            lookahead: Number of bars in the future of the target prediction.
                Held out between train and test to prevent training data from
                leaking across the boundary. Defaults to ``1``.
            start_date: Starting date of the optimization (inclusive). Must be
                within the range passed to :meth:`.__init__`.
            end_date: Ending date of the optimization (inclusive). Must be
                within the range passed to :meth:`.__init__`.
            timeframe: Formatted string specifying the timeframe resolution of
                the data, as in :meth:`pybroker.strategy.Strategy.walkforward`.
            between_time: ``tuple[str, str]`` of times of day e.g.
                ``('9:30', '16:00')`` used to filter the data (inclusive).
            days: Days (e.g. ``"mon"``, ``"tues"``) used to filter the data.
            warmup: Number of bars that need to pass before running the
                executions. Must be greater than ``0`` when set.
            enable_parallel_indicators: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed in
                parallel using multiple processes. Defaults to ``False``.
            adjust: The type of adjustment to make to the
                :class:`pybroker.data.DataSource`.
            calc_bootstrap: Whether to compute randomized bootstrap evaluation
                metrics for the test result. Defaults to ``False``.
            indicator_memo_max: Maximum in-memory hyperparameter indicator
                results to retain while optimizing. Hyperparameterized
                indicators bypass the disk cache; this memo avoids recomputing
                identical ``(indicator, symbol, hyperparams)`` combinations
                across trials. When full, the oldest entry is evicted. Set to
                ``0`` to disable. Defaults to
                :data:`pybroker.indicator.DEFAULT_INDICATOR_MEMO_MAX`. The memo
                is discarded when ``optimize`` returns.

        Returns:
            :class:`.OptimizeResult` with the winning hyperparameter values, the
            train window score they earned, and the test window
            :class:`pybroker.strategy.TestResult` they produced. When ``windows``
            is greater than ``1``, ``best_params``, ``best_score``, and ``study``
            describe the **last** window while ``result`` is stitched across all
            of them; see :attr:`.OptimizeResult.windows` for the per-window
            results.

        Raises:
            ValueError: If no executions were added, if any model source is
                trainable, if ``train_size`` is not between ``0`` and ``1``
                exclusive, if ``warmup`` is not greater than ``0``, if
                ``windows`` is not greater than ``0``, if ``study`` is combined
                with ``windows`` greater than ``1`` or has a conflicting
                direction, or if the dates fall outside the range passed to
                :meth:`.__init__`.
        """
        if indicator_memo_max < 0:
            raise ValueError(
                "indicator_memo_max must be greater than or equal to 0."
            )
        if not 0 < train_size < 1:
            raise ValueError(
                f"optimize requires 0 < train_size < 1, got {train_size}. "
                "train_size=0 leaves no data to score trials on, and "
                "train_size=1 leaves no test window to evaluate on."
            )
        if warmup is not None and warmup < 1:
            raise ValueError("warmup must be > 0.")
        if windows is not None and windows < 1:
            raise ValueError("windows must be > 0.")
        if study is not None and windows is not None and windows > 1:
            raise ValueError(
                "study= is not supported with windows > 1, which runs one "
                "study per window. Inspect OptimizeResult.windows instead."
            )
        _validate_optimize_models(self)
        if not self._executions:
            raise ValueError("No executions were added.")
        if self._slippage_model is not None:
            self._slippage_model.validate(cast("Strategy", self))
        # Collected once: collect_hyperparams warns about registered but
        # unreachable hyperparams, and calling it twice would warn twice.
        hyperparams = collect_hyperparams(self)
        search_space = _search_space_from_specs(hyperparams)
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
            if start_dt < self._start_date or start_dt > self._end_date:
                raise ValueError(
                    f"start_date must be between {self._start_date} and "
                    f"{self._end_date}."
                )
            end_dt = (
                self._end_date if end_date is None else to_datetime(end_date)
            )
            if end_dt < self._start_date or end_dt > self._end_date:
                raise ValueError(
                    f"end_date must be between {self._start_date} and "
                    f"{self._end_date}."
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
            interval_data = self._build_interval_data(df, timeframe)
            has_selector = self._has_symbol_selector()
            tf_seconds = to_seconds(timeframe)
            cache_date_fields = CacheDateFields(
                start_date=start_dt,
                end_date=end_dt,
                tf_seconds=tf_seconds,
                between_time=between_time,
                days=day_ids,
            )
            master_store = symbol_array_store_from_frame(
                _ensure_range_index(df)
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
                    interval_data=interval_data,
                    cache_date_fields=cache_date_fields,
                    enable_parallel_indicators=enable_parallel_indicators,
                    warmup=warmup,
                    has_selector=has_selector,
                    hyperparams=hyperparams,
                    search_space=search_space,
                    windows=windows,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    calc_bootstrap=calc_bootstrap,
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
            selection_data = _selection_df(
                self._executions, train_data, test_data
            )
            window_executions = (
                _resolve_executions(self._executions, selection_data)
                if has_selector
                else self._executions
            )
            invariant_data = self._compute_invariant_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                enable_parallel_indicators=enable_parallel_indicators,
                interval_data=interval_data,
                master_store=master_store,
                executions=window_executions,
            )
            # Model features only: _load_pretrained_models is a no-op without
            # models, and the invariant indicators are already computed above,
            # so only the tuned ones are fetched here.
            _, tuned_syms = self._partition_indicator_syms(window_executions)
            needs_models = any(
                execution.model_names for execution in window_executions
            )
            load_indicators = (
                self._fetch_indicators(
                    df=df,
                    cache_date_fields=cache_date_fields,
                    enable_parallel_indicators=enable_parallel_indicators,
                    interval_data=interval_data,
                    executions=window_executions,
                    symbol_store=master_store,
                    hyperparams=build_run_hyperparams(hyperparams),
                    indicator_syms=tuned_syms,
                )
                if needs_models and tuned_syms
                else {}
            )
            pretrained_models = self._load_pretrained_models(
                df=df,
                train_rows=train_rows,
                test_rows=test_rows,
                window_executions=window_executions,
                indicator_data={**invariant_data, **load_indicators},
                master_store=master_store,
                interval_data=interval_data,
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
                interval_data=interval_data,
                enable_parallel_indicators=enable_parallel_indicators,
                warmup=warmup,
                pretrained_models=pretrained_models,
                exit_dates=self._build_exit_dates(train_data, has_selector),
            )
            if study is None:
                built_sampler = _build_sampler(sampler, search_space, seed)
                _validate_grid_sampler(built_sampler, search_space)
                study = optuna.create_study(
                    direction=direction, sampler=built_sampler, pruner=pruner
                )
            else:
                # A supplied study owns its own sampler, direction and pruner.
                # Deriving the trial budget or the _run_study branch from the
                # arguments instead would search a grid the study never samples.
                _validate_study_direction(study, direction)
                built_sampler = study.sampler
                _validate_grid_sampler(built_sampler, search_space)
            n_trials = _resolve_n_trials(n_trials, built_sampler, search_space)
            _log_optimize_trials(n_trials, built_sampler, search_space)
            _run_study(study, bundle, n_trials, built_sampler)
            best_params = build_run_hyperparams(hyperparams, study.best_params)
            test_result = self._run_optimize_test(
                df=df,
                test_rows=test_rows,
                train_rows=train_rows,
                run_hyperparams=best_params,
                invariant_indicator_data=invariant_data,
                window_executions=window_executions,
                master_store=master_store,
                interval_data=interval_data,
                cache_date_fields=cache_date_fields,
                enable_parallel_indicators=enable_parallel_indicators,
                warmup=warmup,
                start_dt=start_dt,
                end_dt=end_dt,
                exit_dates=self._build_exit_dates(test_data, has_selector),
                seed=seed,
                calc_bootstrap=calc_bootstrap,
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
            # The memo is keyed by (indicator, symbol, hyperparams) with no
            # notion of the data window, so keeping it past this call would
            # serve this run's series to a later optimize() over other dates.
            self._indicator_memo_store().clear()

    def _run_optimize_test(
        self,
        df: pd.DataFrame,
        test_rows: np.ndarray,
        run_hyperparams: dict[str, Any],
        invariant_indicator_data: dict[IndicatorSymbol, pd.Series],
        window_executions: set[Execution],
        master_store: Any,
        interval_data: Any,
        cache_date_fields: CacheDateFields,
        enable_parallel_indicators: bool,
        warmup: Optional[int],
        start_dt: datetime,
        end_dt: datetime,
        exit_dates: Mapping[str, np.datetime64],
        train_rows: np.ndarray,
        seed: Optional[int],
        portfolio: Optional[Portfolio] = None,
        calc_bootstrap: bool = False,
        pretrained_models: Optional[Mapping[ModelSymbol, TrainedModel]] = None,
    ) -> TestResult:
        if pretrained_models is None:
            pretrained_models = {}
        train_data = df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
        test_data = df.iloc[test_rows] if len(test_rows) else df.iloc[:0]
        # Only the tuned indicators are refetched; the invariant ones were
        # already computed once for this window and do not depend on the params.
        _, tuned_syms = self._partition_indicator_syms(window_executions)
        trial_indicators = (
            self._fetch_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                enable_parallel_indicators=enable_parallel_indicators,
                interval_data=interval_data,
                executions=window_executions,
                symbol_store=master_store,
                hyperparams=run_hyperparams,
                indicator_syms=tuned_syms,
            )
            if tuned_syms
            else {}
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
        _, test_store, history_store = self._build_window_stores(
            master_store=master_store,
            master_dates_arr=master_dates_arr,
            train_rows=train_rows,
            test_rows=test_rows,
            train_empty=train_data.empty,
            test_empty=test_data.empty,
        )
        sessions: dict[str, dict] = defaultdict(dict)
        signals = self.backtest_executions(
            config=effective_config,
            executions=window_executions,
            before_exec_fn=self._before_exec_fn,
            after_exec_fn=self._after_exec_fn,
            sessions=sessions,
            models=pretrained_models,
            indicator_data=indicator_data,
            interval_data=interval_data.slice_for_test(
                symbol_dates_from_frame(test_data)
            ),
            test_data=test_data,
            portfolio=portfolio,
            exit_dates=dict(exit_dates),
            backtest_settings=backtest_settings,
            rotation_sizer=self._rotation_sizer,
            slippage_model=self._slippage_model,
            enable_fractional_shares=self._fractional_shares_enabled(),
            round_fill_price=effective_config.round_fill_price,
            warmup=warmup,
            history_col_scope=ColumnScope(history_store)
            if history_store is not None
            else None,
            test_col_scope=ColumnScope(test_store)
            if test_store is not None
            else None,
            run_hyperparams=run_hyperparams,
            master_col_scope=ColumnScope(master_store),
        )
        return self._to_test_result(
            start_dt,
            end_dt,
            portfolio,
            calc_bootstrap=calc_bootstrap,
            train_only=False,
            signals=signals if self._config.return_signals else None,
            seed=seed,
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
        interval_data: Any,
        cache_date_fields: CacheDateFields,
        enable_parallel_indicators: bool,
        warmup: Optional[int],
        has_selector: bool,
        hyperparams: Mapping[str, Hyperparam],
        search_space: SearchSpace,
        windows: int,
        start_dt: datetime,
        end_dt: datetime,
        calc_bootstrap: bool,
    ) -> OptimizeResult:
        splits = list(
            self.walkforward_split(
                df=df,
                windows=windows,
                lookahead=lookahead,
                train_size=train_size,
            )
        )
        # Sized from a throwaway sampler so the budget can be logged up front.
        # Each window builds its own below: samplers carry RNG state, and one
        # shared instance pickled into every worker process would make every
        # window replay the same draws.
        budget_sampler = _build_sampler(sampler, search_space, seed)
        _validate_grid_sampler(budget_sampler, search_space)
        window_n_trials = _resolve_n_trials(
            n_trials, budget_sampler, search_space
        )
        _log_optimize_trials(
            window_n_trials,
            budget_sampler,
            search_space,
            windows=windows,
        )

        def window_seed(index: int) -> Optional[int]:
            return None if seed is None else seed + index

        def run_window_study(
            index: int,
            train_rows: np.ndarray,
            test_rows: np.ndarray,
        ) -> WindowOptimizeResult:
            train_data = (
                df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
            )
            test_data = df.iloc[test_rows] if len(test_rows) else df.iloc[:0]
            train_exit_dates = self._build_exit_dates(train_data, has_selector)
            selection_data = _selection_df(
                self._executions, train_data, test_data
            )
            window_executions = (
                _resolve_executions(self._executions, selection_data)
                if has_selector
                else self._executions
            )
            invariant_data = self._compute_invariant_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                enable_parallel_indicators=enable_parallel_indicators,
                interval_data=interval_data,
                master_store=master_store,
                executions=window_executions,
            )
            # Model features only: _load_pretrained_models is a no-op without
            # models, and the invariant indicators are already computed above,
            # so only the tuned ones are fetched here.
            _, tuned_syms = self._partition_indicator_syms(window_executions)
            needs_models = any(
                execution.model_names for execution in window_executions
            )
            load_indicators = (
                self._fetch_indicators(
                    df=df,
                    cache_date_fields=cache_date_fields,
                    enable_parallel_indicators=enable_parallel_indicators,
                    interval_data=interval_data,
                    executions=window_executions,
                    symbol_store=master_store,
                    hyperparams=build_run_hyperparams(hyperparams),
                    indicator_syms=tuned_syms,
                )
                if needs_models and tuned_syms
                else {}
            )
            pretrained_models = self._load_pretrained_models(
                df=df,
                train_rows=train_rows,
                test_rows=test_rows,
                window_executions=window_executions,
                indicator_data={**invariant_data, **load_indicators},
                master_store=master_store,
                interval_data=interval_data,
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
                interval_data=interval_data,
                enable_parallel_indicators=enable_parallel_indicators,
                warmup=warmup,
                pretrained_models=pretrained_models,
                exit_dates=train_exit_dates,
            )
            this_seed = window_seed(index)
            window_sampler = _build_sampler(sampler, search_space, this_seed)
            window_study = optuna.create_study(
                direction=direction, sampler=window_sampler, pruner=pruner
            )
            _run_study(window_study, bundle, window_n_trials, window_sampler)
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
                interval_data=interval_data,
                enable_parallel_indicators=enable_parallel_indicators,
                warmup=warmup,
                pretrained_models=pretrained_models,
                exit_dates=train_exit_dates,
            )
            test_result = self._run_optimize_test(
                df=df,
                test_rows=test_rows,
                train_rows=train_rows,
                run_hyperparams=best_params,
                invariant_indicator_data=invariant_data,
                window_executions=window_executions,
                master_store=master_store,
                interval_data=interval_data,
                cache_date_fields=cache_date_fields,
                enable_parallel_indicators=enable_parallel_indicators,
                warmup=warmup,
                start_dt=start_dt,
                end_dt=end_dt,
                exit_dates=self._build_exit_dates(test_data, has_selector),
                seed=this_seed,
                calc_bootstrap=calc_bootstrap,
                pretrained_models=pretrained_models,
            )
            return WindowOptimizeResult(
                params=best_params,
                study=window_study,
                test_result=test_result,
                train_score=window_study.best_value,
                train_pnl=is_result.metrics.total_pnl,
                execution_symbols=(
                    {
                        e.id: _static_symbols(e.symbols)
                        for e in window_executions
                    }
                    if has_selector
                    else None
                ),
            )

        scope = StaticScope.instance()
        with parallel() as pool:
            window_results = pool(
                delayed(_run_scoped_task)(
                    scope, run_window_study, i, train_rows, test_rows
                )
                for i, (train_rows, test_rows) in enumerate(splits)
            )

        # Sized from the first window's winning values rather than the
        # hyperparam defaults: _resolve_backtest_settings(None) raises when a
        # position limit is a Hyperparam. The caps are re-applied per window
        # below, since the Portfolio enforces the ones it was constructed with.
        first_settings = self._resolve_backtest_settings(
            window_results[0].params
        )
        portfolio = Portfolio(
            self._config.initial_cash,
            self._config.fee_mode,
            self._config.fee_amount,
            self._fractional_shares_enabled(),
            self._config.position_mode,
            first_settings.max_long_positions,
            first_settings.max_short_positions,
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
        # Allocated once, like Strategy._run_walkforward does: the stitched
        # replay carries positions and cash across window boundaries, so
        # ctx.session has to persist across them too.
        sessions: dict[str, dict] = defaultdict(dict)
        # Allocated once, like Strategy._run_walkforward does: a fresh scope
        # per window drops every persistent limit order at every boundary, so
        # the stitched result -- the one a caller acts on -- silently loses
        # fills the per-window results kept.
        pending_order_scope = PendingOrderScope()
        signal_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
        # Accumulated across windows, mirroring Strategy._run_walkforward. A
        # dropped symbol's indicators were only ever computed for the window
        # that still selected it, so passing this window's dict alone leaves
        # the boundary liquidation with an empty IndicatorScope and an
        # ATR-scaled slippage model silently skips the fill.
        replay_indicator_data: dict[IndicatorSymbol, pd.Series] = {}
        # Computed over the whole df so only the true final bar liquidates.
        stitched_exit_dates = self._build_exit_dates(df, has_selector)
        for i, (train_rows, test_rows) in enumerate(splits):
            wr = window_results[i]
            train_data = (
                df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
            )
            test_data = df.iloc[test_rows] if len(test_rows) else df.iloc[:0]
            # Reuse the study's selection rather than re-running the selector,
            # which a stateful one would answer differently and leave the
            # replayed result describing a different universe than was tuned.
            window_executions = (
                {
                    e._replace(symbols=wr.execution_symbols[e.id])
                    for e in self._executions
                }
                if wr.execution_symbols is not None
                else self._executions
            )
            invariant_data = self._compute_invariant_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                enable_parallel_indicators=enable_parallel_indicators,
                interval_data=interval_data,
                master_store=master_store,
                executions=window_executions,
            )
            trial_indicators = self._fetch_indicators(
                df=df,
                cache_date_fields=cache_date_fields,
                enable_parallel_indicators=enable_parallel_indicators,
                interval_data=interval_data,
                executions=window_executions,
                symbol_store=master_store,
                hyperparams=wr.params,
            )
            indicator_data = {**invariant_data, **trial_indicators}
            replay_indicator_data.update(indicator_data)
            pretrained_models = self._load_pretrained_models(
                df=df,
                train_rows=train_rows,
                test_rows=test_rows,
                window_executions=window_executions,
                indicator_data=indicator_data,
                master_store=master_store,
                interval_data=interval_data,
                tf_seconds=cache_date_fields.tf_seconds,
                between_time=cache_date_fields.between_time,
                days=cache_date_fields.days,
            )
            _, test_store, history_store = self._build_window_stores(
                master_store=master_store,
                master_dates_arr=master_dates_arr,
                train_rows=train_rows,
                test_rows=test_rows,
                train_empty=train_data.empty,
                test_empty=test_data.empty,
            )
            selected_syms = _selected_symbols(
                window_executions, test_data, has_selector
            )
            self._liquidate_dropped_symbols(
                portfolio,
                selected_syms,
                test_data,
                master_store=master_store,
                slippage_model=self._slippage_model,
                indicator_data=replay_indicator_data,
                pending_order_scope=pending_order_scope,
            )
            window_settings = self._resolve_backtest_settings(wr.params)
            window_config = self._effective_config(window_settings)
            # The Portfolio gates position counts with the caps it was built
            # with, so a tuned limit has to be pushed onto it per window or the
            # stitched result silently trades the first window's caps.
            portfolio._max_long_positions = window_settings.max_long_positions
            portfolio._max_short_positions = (
                window_settings.max_short_positions
            )
            split_signals = self.backtest_executions(
                config=window_config,
                executions=window_executions,
                before_exec_fn=self._before_exec_fn,
                after_exec_fn=self._after_exec_fn,
                sessions=sessions,
                models=pretrained_models,
                indicator_data=indicator_data,
                interval_data=interval_data.slice_for_test(
                    symbol_dates_from_frame(test_data)
                ),
                test_data=test_data,
                portfolio=portfolio,
                exit_dates=dict(stitched_exit_dates),
                backtest_settings=window_settings,
                rotation_sizer=self._rotation_sizer,
                slippage_model=self._slippage_model,
                enable_fractional_shares=self._fractional_shares_enabled(),
                round_fill_price=window_config.round_fill_price,
                warmup=warmup,
                history_col_scope=ColumnScope(history_store)
                if history_store is not None
                else None,
                test_col_scope=ColumnScope(test_store)
                if test_store is not None
                else None,
                run_hyperparams=wr.params,
                pending_order_scope=pending_order_scope,
                master_col_scope=ColumnScope(master_store),
            )
            for sym, signals_df in split_signals.items():
                signal_frames[sym].append(signals_df)

        oos_pnl = sum(
            wr.test_result.metrics.total_pnl for wr in window_results
        )
        is_pnl = sum(wr.train_pnl for wr in window_results)
        # Only defined against a profitable in-sample leg: dividing by a
        # negative denominator flips the sign, so a losing out-of-sample run
        # would report an efficiency above 1 and a winning one below 0.
        wfe = oos_pnl / is_pnl if is_pnl > 0 else None
        stitched_signals: Optional[dict[str, pd.DataFrame]] = None
        if self._config.return_signals:
            stitched_signals = {
                sym: (
                    frames[0]
                    if len(frames) == 1
                    else pd.concat(frames, ignore_index=True)
                )
                for sym, frames in signal_frames.items()
            }
        stitched = self._to_test_result(
            start_dt,
            end_dt,
            portfolio,
            calc_bootstrap=calc_bootstrap,
            train_only=False,
            signals=stitched_signals,
            seed=seed,
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
        for sym in _static_symbols(execution.symbols):
            if not loaded or sym in loaded:
                syms.add(sym)
    return syms
