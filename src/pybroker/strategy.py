"""Contains implementation for backtesting trading strategies."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import dataclasses
import json
import math
import warnings
import numpy as np
import pandas as pd
from pybroker.cache import CacheDateFields
from pybroker.common import (
    BarData,
    DataCol,
    Day,
    IndicatorSymbol,
    ModelSymbol,
    OrderType,
    PriceType,
    SymbolSelector,
    _dataframe_records,
    _ensure_range_index,
    _is_symbol_selector,
    _json_safe,
    _resolve_executions,
    _selected_symbols,
    _selection_df,
    _static_symbols,
    get_unique_sorted_dates,
    get_unique_sorted_dates_array,
    quantize,
    to_datetime,
    to_decimal,
    to_seconds,
    verify_data_source_columns,
    verify_date_range,
)
from pybroker.config import StrategyConfig
from pybroker.context import (
    ExecContext,
    ExecResult,
    RotationContext,
    set_exec_ctx_data,
)
from pybroker.data import AlpacaCrypto, DataSource
from pybroker.eval import BootstrapResult, EvalMetrics, EvaluateMixin
from pybroker.optimize import Hyperparam, OptimizeMixin, build_run_hyperparams
from pybroker.indicator import Indicator, IndicatorsMixin
from pybroker.model import (
    ModelSource,
    ModelTrainer,
    ModelsMixin,
    TrainedModel,
)
from pybroker.portfolio import (
    Order,
    Portfolio,
    PortfolioBar,
    PositionBar,
    StopRecord,
    Trade,
)
from pybroker.scope import (
    ColumnScope,
    IndicatorScope,
    ModelInputScope,
    PendingOrder,
    PendingOrderScope,
    PredictionScope,
    PriceScope,
    StaticScope,
    SymbolArrayStore,
    IntervalScope,
    column_scope_from_frame,
    get_signals,
    slice_symbol_array_store_by_dates,
    sym_exec_dates_from_store,
    symbol_array_store_from_frame,
)
from pybroker.interval import (
    IntervalData,
    TimeframeInterval,
    _iter_symbol_date_groups,
    base_timeframe_to_seconds,
    compress_intervals_from_frame,
    indicator_interval_name,
    model_interval_name,
    normalize_interval,
    parse_indicator_interval_name,
    parse_model_interval_name,
    symbol_dates_from_frame,
    validate_interval,
)
from pybroker.slippage import (
    FillSlippageContext,
    FixedSlippageModel,
    SlippageModel,
)
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeGuard,
    Union,
)
from typing_extensions import Concatenate, ParamSpec


P = ParamSpec("P")

_EMPTY_KWARGS: dict[str, Any] = {}


def _unique_dates_from_rows(
    dates_arr: NDArray[np.datetime64],
    row_indices: NDArray[np.int_],
) -> NDArray[np.datetime64]:
    if len(row_indices) == 0:
        return np.array([], dtype="datetime64[ns]")
    return get_unique_sorted_dates_array(dates_arr[row_indices])


def _date_to_active_syms(
    sym_exec_dates: Mapping[str, frozenset[np.datetime64]],
    test_dates: Sequence[np.datetime64],
) -> tuple[dict[np.datetime64, tuple[str, ...]], bool]:
    test_dates_set = frozenset(test_dates)
    if not sym_exec_dates:
        return {}, True
    aligned = all(dates == test_dates_set for dates in sym_exec_dates.values())
    if aligned:
        return {}, True
    date_to_syms: dict[np.datetime64, list[str]] = defaultdict(list)
    for sym, dates in sym_exec_dates.items():
        for date in dates:
            date_to_syms[date].append(sym)
    return {date: tuple(syms) for date, syms in date_to_syms.items()}, False


def _between(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    if df.empty:
        return df
    date_col = DataCol.DATE.value
    col = df[date_col]
    if col.dt.tz is not None:
        col = col.dt.tz_convert(None)
    dates = col.to_numpy(dtype="datetime64[ns]", copy=False)
    start = np.datetime64(start_date)
    end = np.datetime64(end_date)
    mask = (dates >= start) & (dates <= end)
    rows = np.flatnonzero(mask)
    if len(rows) == len(df):
        return df
    return df.iloc[rows]


def _sort_by_buy_score(result: ExecResult) -> float:
    if result.long_score is not None:
        return result.long_score
    return 0.0 if result.score is None else result.score


def _sort_by_sell_score(result: ExecResult) -> float:
    if result.short_score is not None:
        return -result.short_score
    return 0.0 if result.score is None else result.score


def _is_persistent_limit(pending: PendingOrder) -> bool:
    return pending.limit_price is not None and pending.timeout_bars is not None


def _is_rankable(score: Optional[float]) -> TypeGuard[float]:
    if score is None:
        return False
    if isinstance(score, float) and math.isnan(score):
        return False
    if pd.isna(score):
        return False
    return True


def _rank_by_score(scores: Mapping[str, float]) -> dict[str, int]:
    sorted_scores = sorted(
        scores.items(), key=lambda item: (-item[1], item[0])
    )
    return {symbol: rank + 1 for rank, (symbol, _) in enumerate(sorted_scores)}


def _rank_by_short_score(scores: Mapping[str, float]) -> dict[str, int]:
    sorted_scores = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    return {symbol: rank + 1 for rank, (symbol, _) in enumerate(sorted_scores)}


def _long_rotation_score(ctx: ExecContext) -> Optional[float]:
    if _is_rankable(ctx.long_score):
        return ctx.long_score
    return None


def _short_rotation_score(ctx: ExecContext) -> Optional[float]:
    if _is_rankable(ctx.short_score):
        return ctx.short_score
    return None


StrategySetting = Union[int, Hyperparam, None]


@dataclass(frozen=True)
class BacktestSettings:
    max_long_positions: Optional[int] = None
    max_short_positions: Optional[int] = None
    worst_rank_held: Optional[int] = None


def _resolve_strategy_setting(
    value: StrategySetting,
    run_hyperparams: Optional[dict[str, Any]],
) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, Hyperparam):
        if run_hyperparams is None:
            raise ValueError(
                f"Hyperparam {value.name!r} requires run_hyperparams."
            )
        if value.name not in run_hyperparams:
            raise ValueError(f"Hyperparam {value.name!r} was not resolved.")
        return int(run_hyperparams[value.name])
    return int(value)


def _validate_worst_rank_held(
    worst_rank_held: int,
    max_long_positions: Optional[int],
    max_short_positions: Optional[int],
) -> None:
    if max_long_positions is None and max_short_positions is None:
        raise ValueError(
            "worst_rank_held requires max_long_positions or "
            "max_short_positions to be set."
        )
    if max_long_positions is not None and worst_rank_held < max_long_positions:
        raise ValueError(
            "worst_rank_held must be greater than or equal to "
            "max_long_positions."
        )
    if (
        max_short_positions is not None
        and worst_rank_held < max_short_positions
    ):
        raise ValueError(
            "worst_rank_held must be greater than or equal to "
            "max_short_positions."
        )


def _rotation_target_size(settings: BacktestSettings) -> float:
    slots = 0
    if settings.max_long_positions is not None:
        slots += settings.max_long_positions
    if settings.max_short_positions is not None:
        slots += settings.max_short_positions
    return 1 / slots


def _reset_rotation_orders(active_ctxs: Mapping[str, ExecContext]) -> None:
    """Discards orders placed by execution functions.

    Rotation drives trading entirely from :attr:`.ExecContext.long_score` and
    :attr:`.ExecContext.short_score`, so orders set during an execution are
    ignored. Fill prices and stops are kept, since those still shape the orders
    that rotation goes on to place.
    """
    for ctx in active_ctxs.values():
        ctx.buy_shares = None
        ctx.buy_limit_price = None
        ctx.buy_timeout_bars = None
        ctx.sell_shares = None
        ctx.sell_limit_price = None
        ctx.sell_timeout_bars = None
        ctx.hold_bars = None
        ctx._cover = False
        ctx._exiting_pos = False


def _clear_unused_rotation_signals(
    active_ctxs: Mapping[str, ExecContext],
) -> None:
    """Drops fill prices and stops that rotation left without an order.

    :meth:`pybroker.context.ExecContext.to_result` rejects a fill price or stop
    that has no accompanying order, and an execution function has no way to
    know in advance which symbols rotation will trade, or in which direction.
    """
    for ctx in active_ctxs.values():
        if ctx.buy_shares is None:
            ctx.buy_fill_price = None
        if ctx.sell_shares is None:
            ctx.sell_fill_price = None
        if ctx.buy_shares is not None or ctx.sell_shares is not None:
            continue
        ctx.stop_loss = None
        ctx.stop_loss_pct = None
        ctx.stop_loss_limit = None
        ctx.stop_loss_exit_price = None
        ctx.stop_profit = None
        ctx.stop_profit_pct = None
        ctx.stop_profit_limit = None
        ctx.stop_profit_exit_price = None
        ctx.stop_trailing = None
        ctx.stop_trailing_pct = None
        ctx.stop_trailing_limit = None
        ctx.stop_trailing_exit_price = None


def _rotation_ranks(
    active_ctxs: Mapping[str, ExecContext],
    dir: Literal["long", "short"],
) -> dict[str, int]:
    scores: dict[str, float] = {}
    for sym, ctx in active_ctxs.items():
        score = (
            _long_rotation_score(ctx)
            if dir == "long"
            else _short_rotation_score(ctx)
        )
        if score is not None:
            scores[sym] = score
    if dir == "long":
        return _rank_by_score(scores)
    return _rank_by_short_score(scores)


def _has_pending_order(
    ctx: ExecContext, order_type: Literal["buy", "sell"]
) -> bool:
    return any(
        order.type == order_type for order in ctx.pending_orders(ctx.symbol)
    )


def _rotation_exits(
    active_ctxs: Mapping[str, ExecContext],
    portfolio: Portfolio,
    ranks: Mapping[str, int],
    worst_rank_held: int,
    dir: Literal["long", "short"],
) -> int:
    """Liquidates held positions ranked outside the hold band.

    Returns the number of positions leaving the portfolio, which frees the same
    number of slots for entries on this bar.
    """
    if dir == "long":
        positions: Mapping[str, Any] = portfolio.long_positions
        exit_order_type: Literal["buy", "sell"] = "sell"
    else:
        positions = portfolio.short_positions
        exit_order_type = "buy"
    exiting = 0
    for sym in positions:
        ctx = active_ctxs.get(sym)
        if ctx is None:
            # Without a bar for this symbol there is nothing to trade against,
            # so the position keeps its slot.
            continue
        rank = ranks.get(sym)
        if rank is not None and rank <= worst_rank_held:
            continue
        exiting += 1
        if _has_pending_order(ctx, exit_order_type):
            # The liquidation is already in flight. Re-issuing it would queue a
            # redundant order against a position that is already spoken for.
            continue
        if dir == "long":
            ctx.sell_all_shares()
        else:
            ctx.cover_all_shares()
    return exiting


def _rotation_candidates(
    active_ctxs: Mapping[str, ExecContext],
    portfolio: Portfolio,
    ranks: Mapping[str, int],
    worst_rank_held: int,
    max_positions: int,
    exiting: int,
    dir: Literal["long", "short"],
) -> list[str]:
    """Returns the top-ranked symbols to enter, limited to free position slots.

    Candidates ranked worse than ``worst_rank_held`` are excluded, since such a
    position would be liquidated on the following bar.
    """
    if dir == "long":
        held: Mapping[str, Any] = portfolio.long_positions
        entry_order_type: Literal["buy", "sell"] = "buy"
    else:
        held = portfolio.short_positions
        entry_order_type = "sell"
    pending = 0
    eligible: list[str] = []
    for sym, ctx in active_ctxs.items():
        if sym in portfolio.long_positions or sym in portfolio.short_positions:
            continue
        rank = ranks.get(sym)
        if rank is None or rank > worst_rank_held:
            continue
        if _has_pending_order(ctx, entry_order_type):
            # The entry is already in flight and holds its slot. Re-issuing it
            # would stack a second order and overshoot the target allocation.
            pending += 1
            continue
        eligible.append(sym)
    free = max_positions - (len(held) - exiting) - pending
    if free <= 0:
        return []
    eligible.sort(key=lambda sym: ranks[sym])
    return eligible[:free]


def _resolve_rotation_overlap(
    long_cands: list[str],
    short_cands: list[str],
    long_ranks: Mapping[str, int],
    short_ranks: Mapping[str, int],
) -> tuple[list[str], list[str]]:
    """Assigns a symbol picked by both legs to the side it ranks better on.

    Ties go long. Overlap is only reachable when the long and short position
    limits together exceed the number of rankable symbols.
    """
    overlap = set(long_cands) & set(short_cands)
    if not overlap:
        return long_cands, short_cands
    drop_from_long = {
        sym for sym in overlap if short_ranks[sym] < long_ranks[sym]
    }
    drop_from_short = overlap - drop_from_long
    return (
        [sym for sym in long_cands if sym not in drop_from_long],
        [sym for sym in short_cands if sym not in drop_from_short],
    )


def _apply_worst_rank_held(
    active_ctxs: Mapping[str, ExecContext],
    portfolio: Portfolio,
    settings: BacktestSettings,
) -> tuple[dict[str, int], dict[str, int]]:
    worst_rank_held = settings.worst_rank_held
    if worst_rank_held is None:
        return {}, {}
    target_size = _rotation_target_size(settings)
    long_ranks: dict[str, int] = {}
    short_ranks: dict[str, int] = {}
    long_cands: list[str] = []
    short_cands: list[str] = []
    # Both legs decide before either places an order, so a symbol that ranks on
    # both sides cannot end up with a buy and a sell on the same bar.
    if settings.max_long_positions is not None:
        long_ranks = _rotation_ranks(active_ctxs, "long")
        long_cands = _rotation_candidates(
            active_ctxs,
            portfolio,
            long_ranks,
            worst_rank_held,
            settings.max_long_positions,
            _rotation_exits(
                active_ctxs, portfolio, long_ranks, worst_rank_held, "long"
            ),
            "long",
        )
    if settings.max_short_positions is not None:
        short_ranks = _rotation_ranks(active_ctxs, "short")
        short_cands = _rotation_candidates(
            active_ctxs,
            portfolio,
            short_ranks,
            worst_rank_held,
            settings.max_short_positions,
            _rotation_exits(
                active_ctxs, portfolio, short_ranks, worst_rank_held, "short"
            ),
            "short",
        )
    long_cands, short_cands = _resolve_rotation_overlap(
        long_cands, short_cands, long_ranks, short_ranks
    )
    for sym in long_cands:
        active_ctxs[sym].set_target_shares(target_size, dir="long")
    for sym in short_cands:
        active_ctxs[sym].set_target_shares(target_size, dir="short")
    return long_ranks, short_ranks


class Execution(NamedTuple):
    r"""Represents an execution of a :class:`.Strategy`. Holds a reference to
    a :class:`Callable` that implements trading logic.

    Attributes:
        id: Unique ID.
        symbols: Ticker symbols used for execution of ``fn``.
        fn: Implements trading logic.
        model_names: Names of :class:`pybroker.model.ModelSource`\ s used for
            execution of ``fn``.
        indicator_names: Names of :class:`pybroker.indicator.Indicator`\ s
            used for execution of ``fn``.
        intervals: Compression intervals available to ``fn`` through
            :meth:`pybroker.context.ExecContext.interval`.
        args: Additional positional arguments for ``fn``.
        kwargs: Additional keyword arguments for ``fn``.
    """

    id: int
    symbols: Union[frozenset[str], SymbolSelector]
    fn: Optional[Callable[[ExecContext], None]]
    model_names: frozenset[str]
    indicator_names: frozenset[str]
    # Construct with keyword arguments only: inserting a field here shifts the
    # positional index of every field below it.
    intervals: frozenset[TimeframeInterval] = frozenset()
    hyperparam_names: frozenset[str] = frozenset()
    args: tuple[Any, ...] = tuple()
    kwargs: tuple[tuple[str, Any], ...] = tuple()


def _all_intervals(
    executions: Iterable[Execution],
) -> frozenset[TimeframeInterval]:
    """Returns the union of intervals declared across ``executions``."""
    intervals: set[TimeframeInterval] = set()
    for execution in executions:
        intervals.update(execution.intervals)
    return frozenset(intervals)


def _symbol_intervals(
    executions: Iterable[Execution],
    df: pd.DataFrame,
) -> dict[str, frozenset[TimeframeInterval]]:
    r"""Maps each symbol to the intervals that must be compressed for it.

    A :class:`pybroker.common.SymbolSelector` execution does not resolve its
    symbols until a walkforward window is split, but compression runs once up
    front over the whole frame. A selector may return any symbol the frame
    holds, so its intervals are attached to every symbol in ``df``: narrowing
    would leave a later window without compressed data for a symbol the
    selector picked.
    """
    result: dict[str, set[TimeframeInterval]] = defaultdict(set)
    selector_intervals: set[TimeframeInterval] = set()
    for execution in executions:
        if not execution.intervals:
            continue
        if _is_symbol_selector(execution.symbols):
            selector_intervals.update(execution.intervals)
        else:
            for sym in _static_symbols(execution.symbols):
                result[sym].update(execution.intervals)
    if selector_intervals:
        for sym in df[DataCol.SYMBOL.value].unique():
            result[str(sym)].update(selector_intervals)
    return {sym: frozenset(intervals) for sym, intervals in result.items()}


class BacktestMixin:
    """Mixin implementing backtesting functionality."""

    def backtest_executions(
        self,
        config: StrategyConfig,
        executions: set[Execution],
        before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],
        after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],
        sessions: Mapping[str, MutableMapping],
        models: Mapping[ModelSymbol, TrainedModel],
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        test_data: pd.DataFrame,
        portfolio: Portfolio,
        exit_dates: Mapping[str, np.datetime64],
        backtest_settings: BacktestSettings = BacktestSettings(),
        rotation_sizer: Optional[Callable[[RotationContext], None]] = None,
        train_only: bool = False,
        slippage_model: Optional[SlippageModel] = None,
        enable_fractional_shares: bool = False,
        round_fill_price: bool = True,
        warmup: Optional[int] = None,
        interval_data: IntervalData = IntervalData(),
        history_col_scope: Optional[ColumnScope] = None,
        test_col_scope: Optional[ColumnScope] = None,
        run_hyperparams: Optional[dict[str, Any]] = None,
    ) -> dict[str, pd.DataFrame]:
        r"""Backtests a ``set`` of :class:`.Execution`\ s that implement
        trading logic.

        Args:
            config: :class:`pybroker.config.StrategyConfig`.
            executions: :class:`.Execution`\ s to run.
            sessions: :class:`Mapping` of symbols to :class:`Mapping` of custom
                data that persists for every bar during the
                :class:`.Execution`.
            models: :class:`Mapping` of :class:`pybroker.common.ModelSymbol`
                pairs to :class:`pybroker.common.TrainedModel`\ s.
            indicator_data: :class:`Mapping` of
                :class:`pybroker.common.IndicatorSymbol` pairs to
                :class:`pandas.Series` of :class:`pybroker.indicator.Indicator`
                values.
            test_data: :class:`pandas.DataFrame` of test data.
            portfolio: :class:`pybroker.portfolio.Portfolio`.
            exit_dates: :class:`Mapping` of symbols to exit dates.
            train_only: Whether the backtest is run with trading rules or
                only trains models.
            slippage_model: ``Optional``
                :class:`pybroker.slippage.SlippageModel` applied to order
                fills, stop exits, and position exits.
            enable_fractional_shares: Whether to enable trading fractional
                shares.
            round_fill_price: Whether to round fill prices to the nearest cent.
            warmup: Number of bars that need to pass before running the
                executions.

        Returns:
            Dictionary of :class:`pandas.DataFrame`\ s containing bar data,
            indicator data, and model predictions for each symbol when
            :attr:`pybroker.config.StrategyConfig.return_signals` is ``True``.
        """
        if (
            rotation_sizer is not None
            and backtest_settings.worst_rank_held is None
        ):
            raise ValueError(
                "Rotation sizer is set but rotation is not enabled; call "
                "enable_rotation(worst_rank_held=...) first."
            )
        test_dates: Sequence[np.datetime64]
        if test_col_scope is not None:
            # Derive from the store so callers need not materialize (or ship
            # to a worker) a DataFrame purely for its dates and symbols.
            col_scope = test_col_scope
            test_dates = list(test_col_scope.unique_dates())
            test_syms = sorted(test_col_scope.symbols)
        else:
            test_dates = get_unique_sorted_dates(test_data[DataCol.DATE.value])
            test_syms = sorted(test_data[DataCol.SYMBOL.value].unique())
            col_scope = column_scope_from_frame(_ensure_range_index(test_data))
        # A SymbolSelector leaves the whole candidate universe in the store,
        # so signals are scoped to what these executions actually traded.
        exec_syms = {
            sym for exec in executions for sym in _static_symbols(exec.symbols)
        }
        signal_syms = [sym for sym in test_syms if sym in exec_syms]
        ind_scope = IndicatorScope(indicator_data, test_dates)
        input_scope = ModelInputScope(
            col_scope,
            ind_scope,
            models,
            history_col_scope,
            test_dates,
        )
        interval_scope = IntervalScope(
            interval_data,
            ind_scope,
            models,
            test_dates,
        )
        pred_scope = PredictionScope(models, input_scope)
        if train_only:
            if config.return_signals:
                return get_signals(
                    signal_syms, col_scope, ind_scope, pred_scope
                )
            return {}
        sym_end_index: dict[str, int] = defaultdict(int)
        price_scope = PriceScope(col_scope, sym_end_index, round_fill_price)
        pending_order_scope = PendingOrderScope()
        exec_ctxs: dict[str, ExecContext] = {}
        exec_fns: dict[str, Callable[[ExecContext], None]] = {}
        exec_args: dict[str, tuple[Any, ...]] = {}
        exec_kwargs: dict[str, dict[str, Any]] = {}
        rotation_enabled = backtest_settings.worst_rank_held is not None
        for sym in test_syms:
            for exec in executions:
                if sym not in _static_symbols(exec.symbols):
                    continue
                exec_ctxs[sym] = ExecContext(
                    symbol=sym,
                    config=config,
                    portfolio=portfolio,
                    col_scope=col_scope,
                    ind_scope=ind_scope,
                    interval_scope=interval_scope,
                    declared_intervals=exec.intervals,
                    input_scope=input_scope,
                    pred_scope=pred_scope,
                    pending_order_scope=pending_order_scope,
                    models=models,
                    sym_end_index=sym_end_index,
                    session=sessions[sym],
                    run_hyperparams=run_hyperparams,
                    allowed_hyperparam_names=exec.hyperparam_names,
                    rotation_enabled=rotation_enabled,
                )
                exec_args[sym] = exec.args
                exec_kwargs[sym] = dict(exec.kwargs)
                if exec.fn is not None:
                    exec_fns[sym] = exec.fn
                # Executions hold disjoint symbols, so the first match owns
                # this symbol. Stopping here keeps a latent overlap from
                # silently swapping the context's declared intervals.
                break
        sym_exec_dates = {
            sym: dates
            for sym, dates in sym_exec_dates_from_store(
                col_scope.store
            ).items()
            if sym in exec_ctxs
        }
        date_to_syms, calendar_aligned = _date_to_active_syms(
            sym_exec_dates, test_dates
        )
        cover_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        buy_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        sell_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        logger = StaticScope.instance().logger
        logger.backtest_executions_start(test_dates)
        cover_results: deque[ExecResult] = deque()
        buy_results: deque[ExecResult] = deque()
        sell_results: deque[ExecResult] = deque()
        exit_ctxs: deque[ExecContext] = deque()
        active_ctxs: dict[str, ExecContext] = {}
        for i, date in enumerate(test_dates):
            active_ctxs.clear()
            price_scope.reset_bar()
            if calendar_aligned:
                active_iter: Iterable[tuple[str, ExecContext]] = (
                    exec_ctxs.items()
                )
            else:
                active_iter = (
                    (sym, exec_ctxs[sym])
                    for sym in date_to_syms.get(date, ())
                    if sym in exec_ctxs
                )
            for sym, ctx in active_iter:
                sym_end_index[sym] += 1
                if warmup and sym_end_index[sym] <= warmup:
                    continue
                active_ctxs[sym] = ctx
                set_exec_ctx_data(ctx, date)
                if (
                    exit_dates
                    and sym in exit_dates
                    and date == exit_dates[sym]
                ):
                    exit_ctxs.append(ctx)
            is_cover_sched = date in cover_sched
            is_buy_sched = date in buy_sched
            is_sell_sched = date in sell_sched
            if config.max_long_positions is not None:
                # Covers and buys are placed from separate schedules, so both
                # need sorting; ranking only one leaves the other filling in
                # scheduling order once the position limit binds.
                if is_cover_sched:
                    cover_sched[date].sort(
                        key=_sort_by_buy_score, reverse=True
                    )
                if is_buy_sched:
                    buy_sched[date].sort(key=_sort_by_buy_score, reverse=True)
            if is_sell_sched and config.max_short_positions is not None:
                sell_sched[date].sort(key=_sort_by_sell_score, reverse=True)
            portfolio.check_stops(
                date,
                price_scope,
                col_scope,
                sym_end_index,
                ind_scope=ind_scope,
                slippage_model=slippage_model,
            )
            if is_cover_sched:
                self._place_buy_orders(
                    date=date,
                    price_scope=price_scope,
                    col_scope=col_scope,
                    ind_scope=ind_scope,
                    sym_end_index=sym_end_index,
                    slippage_model=slippage_model,
                    pending_order_scope=pending_order_scope,
                    buy_sched=cover_sched,
                    portfolio=portfolio,
                    enable_fractional_shares=enable_fractional_shares,
                )
            if is_sell_sched:
                self._place_sell_orders(
                    date=date,
                    price_scope=price_scope,
                    col_scope=col_scope,
                    ind_scope=ind_scope,
                    sym_end_index=sym_end_index,
                    slippage_model=slippage_model,
                    pending_order_scope=pending_order_scope,
                    sell_sched=sell_sched,
                    portfolio=portfolio,
                    enable_fractional_shares=enable_fractional_shares,
                )
            if is_buy_sched:
                self._place_buy_orders(
                    date=date,
                    price_scope=price_scope,
                    col_scope=col_scope,
                    ind_scope=ind_scope,
                    sym_end_index=sym_end_index,
                    slippage_model=slippage_model,
                    pending_order_scope=pending_order_scope,
                    buy_sched=buy_sched,
                    portfolio=portfolio,
                    enable_fractional_shares=enable_fractional_shares,
                )
            self._process_persistent_orders(
                date=date,
                sym_end_index=sym_end_index,
                price_scope=price_scope,
                col_scope=col_scope,
                ind_scope=ind_scope,
                slippage_model=slippage_model,
                pending_order_scope=pending_order_scope,
                portfolio=portfolio,
                enable_fractional_shares=enable_fractional_shares,
            )
            portfolio.capture_bar(
                date, col_scope, sym_end_index, price_scope=price_scope
            )
            if before_exec_fn is not None and active_ctxs:
                before_exec_fn(active_ctxs)
            for sym, ctx in active_ctxs.items():
                if sym in exec_fns:
                    exec_fns[sym](
                        ctx,
                        *exec_args.get(sym, ()),
                        **exec_kwargs.get(sym, _EMPTY_KWARGS),
                    )
            if after_exec_fn is not None and active_ctxs:
                after_exec_fn(active_ctxs)
            if backtest_settings.worst_rank_held is not None:
                _reset_rotation_orders(active_ctxs)
                long_ranks, short_ranks = _apply_worst_rank_held(
                    active_ctxs, portfolio, backtest_settings
                )
                if rotation_sizer is not None and active_ctxs:
                    rotation_sizer(
                        RotationContext(
                            ctxs=active_ctxs,
                            portfolio=portfolio,
                            long_ranks=long_ranks,
                            short_ranks=short_ranks,
                            config=config,
                        )
                    )
                _clear_unused_rotation_signals(active_ctxs)
            for ctx in active_ctxs.values():
                if (
                    slippage_model
                    and slippage_model.uses_signal_slippage
                    and (ctx.buy_shares or ctx.sell_shares)
                ):
                    self._apply_slippage(slippage_model, ctx)
                result = ctx.to_result()
                if result is None:
                    continue
                if result.buy_shares is not None:
                    if result.cover:
                        cover_results.append(result)
                    else:
                        buy_results.append(result)
                if result.sell_shares is not None:
                    sell_results.append(result)
            while cover_results:
                self._schedule_order(
                    result=cover_results.popleft(),
                    created=date,
                    sym_end_index=sym_end_index,
                    delay=config.buy_delay,
                    sched=cover_sched,
                    col_scope=col_scope,
                    pending_order_scope=pending_order_scope,
                )
            while buy_results:
                self._schedule_order(
                    result=buy_results.popleft(),
                    created=date,
                    sym_end_index=sym_end_index,
                    delay=config.buy_delay,
                    sched=buy_sched,
                    col_scope=col_scope,
                    pending_order_scope=pending_order_scope,
                )
            while sell_results:
                self._schedule_order(
                    result=sell_results.popleft(),
                    created=date,
                    sym_end_index=sym_end_index,
                    delay=config.sell_delay,
                    sched=sell_sched,
                    col_scope=col_scope,
                    pending_order_scope=pending_order_scope,
                )
            while exit_ctxs:
                self._exit_position(
                    portfolio=portfolio,
                    date=date,
                    ctx=exit_ctxs.popleft(),
                    exit_cover_fill_price=config.exit_cover_fill_price,
                    exit_sell_fill_price=config.exit_sell_fill_price,
                    price_scope=price_scope,
                    col_scope=col_scope,
                    ind_scope=ind_scope,
                    sym_end_index=sym_end_index,
                    slippage_model=slippage_model,
                )
            portfolio.incr_bars()
            if i % 10 == 0 or i == len(test_dates) - 1:
                logger.backtest_executions_loading(i + 1)
        return (
            get_signals(signal_syms, col_scope, ind_scope, pred_scope)
            if config.return_signals
            else {}
        )

    def _apply_slippage(
        self,
        slippage_model: SlippageModel,
        ctx: ExecContext,
    ):
        buy_shares = to_decimal(ctx.buy_shares) if ctx.buy_shares else None
        sell_shares = to_decimal(ctx.sell_shares) if ctx.sell_shares else None
        slippage_model.apply_slippage(
            ctx, buy_shares=buy_shares, sell_shares=sell_shares
        )

    def _exit_position(
        self,
        portfolio: Portfolio,
        date: np.datetime64,
        ctx: ExecContext,
        exit_cover_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ],
        exit_sell_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ],
        price_scope: PriceScope,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        sym_end_index: Mapping[str, int],
        slippage_model: Optional[SlippageModel],
    ):
        buy_fill_price = price_scope.fetch(ctx.symbol, exit_cover_fill_price)
        sell_fill_price = price_scope.fetch(ctx.symbol, exit_sell_fill_price)
        portfolio.exit_position(
            date,
            ctx.symbol,
            buy_fill_price=buy_fill_price,
            sell_fill_price=sell_fill_price,
            col_scope=col_scope,
            ind_scope=ind_scope,
            sym_end_index=sym_end_index,
            slippage_model=slippage_model,
        )

    def _schedule_order(
        self,
        result: ExecResult,
        created: np.datetime64,
        sym_end_index: Mapping[str, int],
        delay: int,
        sched: Mapping[np.datetime64, list[ExecResult]],
        col_scope: ColumnScope,
        pending_order_scope: PendingOrderScope,
    ):
        date_loc = sym_end_index[result.symbol] - 1
        dates = col_scope.fetch(result.symbol, DataCol.DATE.value)
        if dates is None:
            raise ValueError("Dates not found.")
        logger = StaticScope.instance().logger
        if date_loc + delay < len(dates):
            date = dates[date_loc + delay]
            order_type: Literal["buy", "sell"]
            if result.buy_shares is not None:
                order_type = "buy"
                shares = result.buy_shares
                limit_price = result.buy_limit_price
                fill_price = result.buy_fill_price
            elif result.sell_shares is not None:
                order_type = "sell"
                shares = result.sell_shares
                limit_price = result.sell_limit_price
                fill_price = result.sell_fill_price
            else:
                raise ValueError("buy_shares or sell_shares needs to be set.")
            if order_type == "buy":
                timeout_bars = result.buy_timeout_bars
                stops = result.long_stops
            else:
                timeout_bars = result.sell_timeout_bars
                stops = result.short_stops
            if stops is not None and not stops:
                stops = None
            exec_bar = sym_end_index[result.symbol] + delay
            result.pending_order_id = pending_order_scope.add(
                type=order_type,
                symbol=result.symbol,
                created=created,
                exec_date=date,
                shares=shares,
                limit_price=limit_price,
                fill_price=fill_price,
                exec_bar=exec_bar,
                timeout_bars=timeout_bars,
                stops=stops,
            )
            sched[date].append(result)
            logger.debug_schedule_order(date, result)
        else:
            logger.debug_unscheduled_order(result)

    def _place_buy_orders(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        sym_end_index: Mapping[str, int],
        slippage_model: Optional[SlippageModel],
        pending_order_scope: PendingOrderScope,
        buy_sched: dict[np.datetime64, list[ExecResult]],
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ):
        buy_results = buy_sched[date]
        logger = StaticScope.instance().logger
        for result in buy_results:
            if result.buy_shares is None:
                continue
            if (
                result.pending_order_id is None
                or not pending_order_scope.contains(result.pending_order_id)
            ):
                continue
            pending = pending_order_scope.get(result.pending_order_id)
            if pending is None:
                continue
            order, shares, fill_price = self._attempt_pending_order(
                pending=pending,
                date=date,
                price_scope=price_scope,
                col_scope=col_scope,
                ind_scope=ind_scope,
                sym_end_index=sym_end_index,
                slippage_model=slippage_model,
                portfolio=portfolio,
                enable_fractional_shares=enable_fractional_shares,
            )
            if order is None:
                logger.debug_unfilled_buy_order(
                    date=date,
                    symbol=pending.symbol,
                    shares=shares,
                    fill_price=fill_price,
                    limit_price=pending.limit_price,
                )
            else:
                logger.debug_filled_buy_order(
                    date=date,
                    symbol=pending.symbol,
                    shares=shares,
                    fill_price=fill_price,
                    limit_price=pending.limit_price,
                )
            if order is not None or not _is_persistent_limit(pending):
                pending_order_scope.remove(pending.id)
        del buy_sched[date]

    def _place_sell_orders(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        sym_end_index: Mapping[str, int],
        slippage_model: Optional[SlippageModel],
        pending_order_scope: PendingOrderScope,
        sell_sched: dict[np.datetime64, list[ExecResult]],
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ):
        sell_results = sell_sched[date]
        logger = StaticScope.instance().logger
        for result in sell_results:
            if result.sell_shares is None:
                continue
            if (
                result.pending_order_id is None
                or not pending_order_scope.contains(result.pending_order_id)
            ):
                continue
            pending = pending_order_scope.get(result.pending_order_id)
            if pending is None:
                continue
            order, shares, fill_price = self._attempt_pending_order(
                pending=pending,
                date=date,
                price_scope=price_scope,
                col_scope=col_scope,
                ind_scope=ind_scope,
                sym_end_index=sym_end_index,
                slippage_model=slippage_model,
                portfolio=portfolio,
                enable_fractional_shares=enable_fractional_shares,
            )
            if order is None:
                logger.debug_unfilled_sell_order(
                    date=date,
                    symbol=pending.symbol,
                    shares=shares,
                    fill_price=fill_price,
                    limit_price=pending.limit_price,
                )
            else:
                logger.debug_filled_sell_order(
                    date=date,
                    symbol=pending.symbol,
                    shares=shares,
                    fill_price=fill_price,
                    limit_price=pending.limit_price,
                )
            if order is not None or not _is_persistent_limit(pending):
                pending_order_scope.remove(pending.id)
        del sell_sched[date]

    def _process_persistent_orders(
        self,
        date: np.datetime64,
        sym_end_index: Mapping[str, int],
        price_scope: PriceScope,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        slippage_model: Optional[SlippageModel],
        pending_order_scope: PendingOrderScope,
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ):
        if not pending_order_scope.has_orders():
            return
        logger = StaticScope.instance().logger
        for pending in list(pending_order_scope.orders()):
            if pending.exec_date >= date:
                continue
            if pending.timeout_bars is None:
                continue
            bars_since_attempt = (
                sym_end_index[pending.symbol] - pending.exec_bar
            )
            if (
                pending.timeout_bars >= 0
                and bars_since_attempt > pending.timeout_bars
            ):
                pending_order_scope.remove(pending.id)
                logger.debug_timeout_order(date=date, pending_order=pending)
                continue
            order, shares, fill_price = self._attempt_pending_order(
                pending=pending,
                date=date,
                price_scope=price_scope,
                col_scope=col_scope,
                ind_scope=ind_scope,
                sym_end_index=sym_end_index,
                slippage_model=slippage_model,
                portfolio=portfolio,
                enable_fractional_shares=enable_fractional_shares,
            )
            if order is None:
                if pending.type == "buy":
                    logger.debug_unfilled_buy_order(
                        date=date,
                        symbol=pending.symbol,
                        shares=shares,
                        fill_price=fill_price,
                        limit_price=pending.limit_price,
                    )
                else:
                    logger.debug_unfilled_sell_order(
                        date=date,
                        symbol=pending.symbol,
                        shares=shares,
                        fill_price=fill_price,
                        limit_price=pending.limit_price,
                    )
            else:
                if pending.type == "buy":
                    logger.debug_filled_buy_order(
                        date=date,
                        symbol=pending.symbol,
                        shares=shares,
                        fill_price=fill_price,
                        limit_price=pending.limit_price,
                    )
                else:
                    logger.debug_filled_sell_order(
                        date=date,
                        symbol=pending.symbol,
                        shares=shares,
                        fill_price=fill_price,
                        limit_price=pending.limit_price,
                    )
            if order is not None:
                pending_order_scope.remove(pending.id)

    def _attempt_pending_order(
        self,
        pending: PendingOrder,
        date: np.datetime64,
        price_scope: PriceScope,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        sym_end_index: Mapping[str, int],
        slippage_model: Optional[SlippageModel],
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ) -> tuple[Optional[Order], Decimal, Decimal]:
        shares = self._get_shares(pending.shares, enable_fractional_shares)
        fill_price = price_scope.fetch(pending.symbol, pending.fill_price)
        if slippage_model is not None:
            # Exact type check, not isinstance: a subclass may override
            # apply_at_fill and must not be routed to the fast path.
            if type(slippage_model) is FixedSlippageModel:
                fill_price = slippage_model.adjust_fill_price(
                    pending.type, fill_price
                )
            elif not slippage_model.is_fill_noop:
                fill_ctx = FillSlippageContext(
                    side=pending.type,
                    symbol=pending.symbol,
                    shares=shares,
                    fill_price=fill_price,
                    col_scope=col_scope,
                    ind_scope=ind_scope,
                    sym_end_index=sym_end_index,
                    enable_fractional_shares=enable_fractional_shares,
                )
                shares, fill_price = slippage_model.apply_at_fill(fill_ctx)
        if pending.type == "buy":
            order_type = (
                OrderType.LIMIT
                if pending.limit_price is not None
                else OrderType.MARKET
            )
            return (
                portfolio.buy(
                    date=date,
                    symbol=pending.symbol,
                    shares=shares,
                    fill_price=fill_price,
                    limit_price=pending.limit_price,
                    stops=pending.stops,
                    created=pending.created,
                    order_type=order_type,
                ),
                shares,
                fill_price,
            )
        order_type = (
            OrderType.LIMIT
            if pending.limit_price is not None
            else OrderType.MARKET
        )
        return (
            portfolio.sell(
                date=date,
                symbol=pending.symbol,
                shares=shares,
                fill_price=fill_price,
                limit_price=pending.limit_price,
                stops=pending.stops,
                created=pending.created,
                order_type=order_type,
            ),
            shares,
            fill_price,
        )

    def _get_shares(
        self,
        shares: Union[int, float, Decimal],
        enable_fractional_shares: bool,
    ) -> Decimal:
        if enable_fractional_shares:
            return to_decimal(shares)
        else:
            return to_decimal(int(shares))


class WalkforwardWindow(NamedTuple):
    """Contains train/test row indices for a walkforward window.

    Attributes:
        train_data: Integer row indices into the master frame for training.
        test_data: Integer row indices into the master frame for testing.
    """

    train_data: NDArray[np.int_]
    test_data: NDArray[np.int_]


def _walkforward_row_mask(
    dates: NDArray[np.datetime64],
    low: np.datetime64,
    high: np.datetime64,
    *,
    below_low: bool = False,
    above_high: bool = False,
) -> NDArray[np.int_]:
    if below_low:
        lo = dates > low
    else:
        lo = dates >= low
    if above_high:
        hi = dates < high
    else:
        hi = dates <= high
    return np.flatnonzero(lo & hi).astype(np.int_)


class WalkforwardMixin:
    """Mixin implementing logic for `Walkforward Analysis
    <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.
    """

    def walkforward_split(
        self,
        df: pd.DataFrame,
        windows: int,
        lookahead: int,
        train_size: float = 0.9,
        shuffle: bool = False,
    ) -> Iterator[WalkforwardWindow]:
        r"""Splits a :class:`pandas.DataFrame` containing data for multiple
        ticker symbols into an :class:`Iterator` of train/test time windows for
        `Walkforward Analysis
        <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

        Args:
            df: :class:`pandas.DataFrame` of data to split into train/test
                windows for Walkforward Analysis.
            windows: Number of walkforward time windows.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of data in ``df`` to use for training, where
                the max ``train_size`` is ``1``. For example, a ``train_size``
                of ``0.9`` would result in 90% of data in ``df`` being used for
                training and the remaining 10% of data being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``.

        Returns:
            :class:`Iterator` of :class:`.WalkforwardWindow`\ s containing
            train and test data.
        """
        if windows <= 0:
            raise ValueError("windows needs to be > 0.")
        if lookahead <= 0:
            raise ValueError("lookahead needs to be > 0.")
        if train_size < 0:
            raise ValueError("train_size cannot be negative.")
        if df.empty:
            raise ValueError("DataFrame is empty.")
        date_col = DataCol.DATE.value
        dates_arr = df[date_col].to_numpy(copy=False, dtype="datetime64[ns]")
        window_dates = get_unique_sorted_dates_array(dates_arr)
        error_msg = f"""
        Invalid params for {len(window_dates)} dates:
        windows: {windows}
        lookahead: {lookahead}
        train_size: {train_size}
        """
        if train_size == 0 or train_size == 1:
            window_length = int(len(window_dates) / windows)
            offset = len(window_dates) - window_length * windows
            for i in range(windows):
                start = offset + i * window_length
                end = start + window_length
                if train_size == 0:
                    test_rows = _walkforward_row_mask(
                        dates_arr,
                        window_dates[start],
                        window_dates[end - 1],
                    )
                    yield WalkforwardWindow(
                        np.array((), dtype=np.int_), test_rows
                    )
                else:
                    train_rows = _walkforward_row_mask(
                        dates_arr,
                        window_dates[start],
                        window_dates[end - 1],
                    )
                    if shuffle:
                        np.random.shuffle(train_rows)
                    yield WalkforwardWindow(
                        train_rows, np.array((), dtype=np.int_)
                    )
        elif windows == 1:
            res = len(window_dates) - 1 - lookahead
            if res <= 0:
                raise ValueError(error_msg)
            train_length = int(res * train_size)
            test_length = int(res * (1 - train_size))
            train_start = (
                len(window_dates) - lookahead - train_length - test_length - 1
            )
            train_end = train_start + train_length
            test_start = train_end + lookahead
            if test_start >= len(window_dates):
                raise ValueError(error_msg)
            test_end = len(window_dates) - 1
            train_rows = _walkforward_row_mask(
                dates_arr,
                window_dates[train_start],
                window_dates[train_end],
            )
            test_rows = _walkforward_row_mask(
                dates_arr,
                window_dates[test_start],
                window_dates[test_end],
            )
            if shuffle:
                np.random.shuffle(train_rows)
            yield WalkforwardWindow(train_rows, test_rows)
        else:
            res = len(window_dates) - (lookahead - 1) * windows
            window_length = res / windows  # type: ignore[assignment]
            train_length = int(window_length * train_size)
            test_length = int(window_length * (1 - train_size))
            if train_length < 0 or test_length < 0:
                raise ValueError(error_msg)
            while True:
                rem = (res - (train_length + test_length * windows)) / windows
                train_incr = int(rem * train_size)
                test_incr = int(rem * (1 - train_size))
                if train_incr == 0 or test_incr == 0:
                    break
                train_length += train_incr
                test_length += test_incr
            if train_length == 0 and test_length == 0:
                raise ValueError(error_msg)
            window_idx = []
            for i in range(windows):
                test_end = i * test_length
                test_start = test_end + test_length
                train_end = test_start + lookahead - 1
                train_start = train_end + train_length
                window_idx.append(
                    (train_start, train_end, test_start, test_end)
                )
            window_idx.reverse()
            reversed_dates = window_dates[::-1]
            for train_start, train_end, test_start, test_end in window_idx:
                train_rows = _walkforward_row_mask(
                    dates_arr,
                    reversed_dates[train_start],
                    reversed_dates[train_end],
                    below_low=True,
                )
                test_rows = _walkforward_row_mask(
                    dates_arr,
                    reversed_dates[test_start],
                    reversed_dates[test_end],
                    below_low=True,
                )
                if shuffle:
                    np.random.shuffle(train_rows)
                yield WalkforwardWindow(train_rows, test_rows)


_DEFAULT_JSON_INCLUDE = frozenset({"metrics", "trades", "orders", "bootstrap"})


def _filter_df_symbols(
    df: pd.DataFrame, symbols: Optional[frozenset[str]]
) -> pd.DataFrame:
    if symbols is None or df.empty:
        return df
    if "symbol" in df.columns:
        return df[df["symbol"].isin(symbols)]
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        mask = df.index.get_level_values("symbol").isin(symbols)
        return df[mask]
    return df


@dataclass(frozen=True)
class TestResult:
    r"""Contains the results of backtesting a :class:`.Strategy`.

    Attributes:
        start_date: Starting date of backtest.
        end_date: Ending date of backtest.
        portfolio: :class:`pandas.DataFrame` of
            :class:`pybroker.portfolio.Portfolio` balances for every bar.
        positions: :class:`pandas.DataFrame` of
            :class:`pybroker.portfolio.Position` balances for every bar.
        orders: :class:`pandas.DataFrame` of all orders that were placed.
        trades: :class:`pandas.DataFrame` of all trades that were made.
        metrics: Evaluation metrics.
        metrics_df: :class:`pandas.DataFrame` of evaluation metrics.
        bootstrap: Randomized bootstrap evaluation metrics.
        signals: Dictionary of :class:`pandas.DataFrame`\ s containing bar
            data, indicator data, and model predictions for each symbol when
            :attr:`pybroker.config.StrategyConfig.return_signals` is ``True``.
        stops: :class:`pandas.DataFrame` containing stop data per-bar when
            :attr:`pybroker.config.StrategyConfig.return_stops` is ``True``.
    """

    start_date: datetime
    end_date: datetime
    portfolio: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    metrics: EvalMetrics
    metrics_df: pd.DataFrame
    bootstrap: Optional[BootstrapResult]
    signals: Optional[dict[str, pd.DataFrame]]
    stops: Optional[pd.DataFrame]

    def to_json(
        self,
        *,
        include: frozenset[str] = _DEFAULT_JSON_INCLUDE,
        max_rows: Optional[int] = 100,
        symbols: Optional[frozenset[str]] = None,
    ) -> dict[str, Any]:
        """Returns JSON-serializable backtest results.

        By default includes ``start_date``, ``end_date``, ``metrics``,
        ``trades``, ``orders``, and ``bootstrap`` (when present). Large
        time series such as ``portfolio``, ``positions``, ``signals``, and
        ``stops`` are opt-in via ``include``.

        Args:
            include: Names of optional result sections to include.
            max_rows: Maximum rows per tabular section. ``None`` for no limit.
            symbols: When set, filter symbol-specific sections to these tickers.
        """
        payload: dict[str, Any] = {
            "start_date": _json_safe(self.start_date),
            "end_date": _json_safe(self.end_date),
        }
        if "metrics" in include:
            payload["metrics"] = self.metrics.to_json()
        if "metrics_df" in include:
            payload["metrics_df"] = _dataframe_records(
                self.metrics_df, max_rows=max_rows
            )
        if "trades" in include:
            payload["trades"] = _dataframe_records(
                _filter_df_symbols(self.trades, symbols),
                max_rows=max_rows,
            )
        if "orders" in include:
            payload["orders"] = _dataframe_records(
                _filter_df_symbols(self.orders, symbols),
                max_rows=max_rows,
            )
        if "portfolio" in include:
            payload["portfolio"] = _dataframe_records(
                self.portfolio, max_rows=max_rows
            )
        if "positions" in include:
            payload["positions"] = _dataframe_records(
                _filter_df_symbols(self.positions, symbols),
                max_rows=max_rows,
            )
        if "bootstrap" in include and self.bootstrap is not None:
            payload["bootstrap"] = self.bootstrap.to_json()
        if "signals" in include and self.signals is not None:
            payload["signals"] = {
                sym: _dataframe_records(df, max_rows=max_rows)
                for sym, df in self.signals.items()
                if symbols is None or sym in symbols
            }
        if "stops" in include and self.stops is not None:
            stops = self.stops
            if symbols is not None and "symbol" in stops.columns:
                stops = stops[stops["symbol"].isin(symbols)]
            payload["stops"] = _dataframe_records(stops, max_rows=max_rows)
        return payload

    def to_json_str(
        self,
        *,
        include: frozenset[str] = _DEFAULT_JSON_INCLUDE,
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


class Strategy(
    BacktestMixin,
    EvaluateMixin,
    IndicatorsMixin,
    ModelsMixin,
    WalkforwardMixin,
    OptimizeMixin,
):
    """Class representing a trading strategy to backtest.

    Args:
        data_source: :class:`pybroker.data.DataSource` or
            :class:`pandas.DataFrame` of backtesting data.
        start_date: Starting date of the data to fetch from ``data_source``
            (inclusive).
        end_date: Ending date of the data to fetch from ``data_source``
            (inclusive).
        config: ``Optional`` :class:`pybroker.config.StrategyConfig`.
    """

    _execution_id: int = 0

    def __init__(
        self,
        data_source: Union[DataSource, pd.DataFrame],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        config: Optional[StrategyConfig] = None,
    ):
        self._verify_data_source(data_source)
        self._data_source = data_source
        self._start_date = to_datetime(start_date)
        self._end_date = to_datetime(end_date)
        verify_date_range(self._start_date, self._end_date)
        if config is not None:
            self._verify_config(config)
            self._config = config
        else:
            self._config = StrategyConfig()
        self._executions: set[Execution] = set()
        self._before_exec_fn: Optional[
            Callable[[Mapping[str, ExecContext]], None]
        ] = None
        self._after_exec_fn: Optional[
            Callable[[Mapping[str, ExecContext]], None]
        ] = None
        self._max_long_positions: StrategySetting = None
        self._max_short_positions: StrategySetting = None
        self._worst_rank_held: StrategySetting = None
        self._rotation_sizer: Optional[Callable[[RotationContext], None]] = (
            None
        )
        self._slippage_model: Optional[SlippageModel] = None
        self._scope = StaticScope.instance()
        self._logger = self._scope.logger

    def _verify_config(self, config: StrategyConfig):
        if config.initial_cash <= 0:
            raise ValueError("initial_cash must be greater than 0.")
        if (
            config.max_long_positions is not None
            and config.max_long_positions <= 0
        ):
            raise ValueError("max_long_positions must be greater than 0.")
        if (
            config.max_short_positions is not None
            and config.max_short_positions <= 0
        ):
            raise ValueError("max_short_positions must be greater than 0.")
        if config.max_long_positions is not None:
            warnings.warn(
                "StrategyConfig.max_long_positions is deprecated; use "
                "Strategy.set_max_long_positions().",
                DeprecationWarning,
                stacklevel=2,
            )
        if config.max_short_positions is not None:
            warnings.warn(
                "StrategyConfig.max_short_positions is deprecated; use "
                "Strategy.set_max_short_positions().",
                DeprecationWarning,
                stacklevel=2,
            )
        if config.buy_delay <= 0:
            raise ValueError("buy_delay must be greater than 0.")
        if config.sell_delay <= 0:
            raise ValueError("sell_delay must be greater than 0.")
        if config.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be greater than 0.")
        if config.leverage < 1:
            raise ValueError("leverage must be greater than or equal to 1.")
        if config.interest_rate < 0:
            raise ValueError(
                "interest_rate must be greater than or equal to 0."
            )
        if config.interest_rate > 0 and config.bars_per_year is None:
            raise ValueError(
                "bars_per_year is required when interest_rate is set, since "
                "it sets the interest accrual period. For example, use 252 "
                "for daily bars or 98280 for 1-minute US equity bars."
            )

    def _resolve_backtest_settings(
        self, run_hyperparams: Optional[dict[str, Any]] = None
    ) -> BacktestSettings:
        max_long = _resolve_strategy_setting(
            self._max_long_positions, run_hyperparams
        )
        max_short = _resolve_strategy_setting(
            self._max_short_positions, run_hyperparams
        )
        worst = _resolve_strategy_setting(
            self._worst_rank_held, run_hyperparams
        )

        if self._config.max_long_positions is not None:
            if max_long is not None:
                warnings.warn(
                    "Strategy.set_max_long_positions takes precedence over "
                    "StrategyConfig.max_long_positions.",
                    stacklevel=2,
                )
            else:
                max_long = self._config.max_long_positions
        if self._config.max_short_positions is not None:
            if max_short is not None:
                warnings.warn(
                    "Strategy.set_max_short_positions takes precedence over "
                    "StrategyConfig.max_short_positions.",
                    stacklevel=2,
                )
            else:
                max_short = self._config.max_short_positions

        if max_long is not None and max_long <= 0:
            raise ValueError("max_long_positions must be greater than 0.")
        if max_short is not None and max_short <= 0:
            raise ValueError("max_short_positions must be greater than 0.")
        if self._rotation_sizer is not None and worst is None:
            raise ValueError(
                "Rotation sizer is set but rotation is not enabled; call "
                "enable_rotation(worst_rank_held=...) first."
            )
        if worst is not None:
            _validate_worst_rank_held(worst, max_long, max_short)

        return BacktestSettings(
            max_long_positions=max_long,
            max_short_positions=max_short,
            worst_rank_held=worst,
        )

    def _effective_config(self, settings: BacktestSettings) -> StrategyConfig:
        return dataclasses.replace(
            self._config,
            max_long_positions=settings.max_long_positions,
            max_short_positions=settings.max_short_positions,
        )

    def set_max_long_positions(self, max_long: StrategySetting) -> None:
        r"""Sets the maximum number of long positions held at any time.

        Args:
            max_long: Maximum long positions, a searchable
                :class:`pybroker.optimize.Hyperparam`, or ``None`` for
                unlimited.
        """
        if isinstance(max_long, int) and max_long <= 0:
            raise ValueError("max_long_positions must be greater than 0.")
        self._max_long_positions = max_long

    def set_max_short_positions(self, max_short: StrategySetting) -> None:
        r"""Sets the maximum number of short positions held at any time.

        Args:
            max_short: Maximum short positions, a searchable
                :class:`pybroker.optimize.Hyperparam`, or ``None`` for
                unlimited.
        """
        if isinstance(max_short, int) and max_short <= 0:
            raise ValueError("max_short_positions must be greater than 0.")
        self._max_short_positions = max_short

    def enable_rotation(
        self,
        worst_rank_held: StrategySetting,
        sizer: Optional[Callable[[RotationContext], None]] = None,
    ) -> None:
        r"""Enables rotational hold-band logic and optional custom sizing.

        Each bar, held positions ranked worse than ``worst_rank_held`` are
        liquidated, and the top-ranked symbols are entered to fill the position
        slots that remain free. Without a ``sizer``, entries are equal-weighted
        across :meth:`.set_max_long_positions` plus
        :meth:`.set_max_short_positions` slots.

        Rotation is exclusive: trading is driven entirely by
        :attr:`pybroker.context.ExecContext.long_score` and
        :attr:`pybroker.context.ExecContext.short_score`, and orders placed by
        an :class:`.Execution` are ignored. Fill prices and stops set during an
        execution are kept and applied to the orders rotation places.

        Ranking spans the whole portfolio, so a held position without a
        rankable score is liquidated even when another :class:`.Execution`
        opened it.

        Args:
            worst_rank_held: Worst score rank at which a held position is
                kept, a searchable :class:`pybroker.optimize.Hyperparam`, or
                ``None`` to disable rotation. Must be greater than or equal to
                the maximum long and short position counts.
            sizer: Optional :class:`Callable` that takes a
                :class:`pybroker.context.RotationContext` to override
                equal-weight entry sizing after rotation decisions are made.
                Do not override sell or cover signals set by rotation.
        """
        if worst_rank_held is None:
            self._worst_rank_held = None
            self._rotation_sizer = None
            return
        self._worst_rank_held = worst_rank_held
        self._rotation_sizer = sizer

    def _verify_data_source(
        self, data_source: Union[DataSource, pd.DataFrame]
    ):
        if isinstance(data_source, pd.DataFrame):
            verify_data_source_columns(data_source)
        elif not isinstance(data_source, DataSource):
            raise TypeError(f"Invalid data_source type: {type(data_source)}")

    def set_slippage_model(self, slippage_model: Optional[SlippageModel]):
        """Sets :class:`pybroker.slippage.SlippageModel`.

        Built-in models are
        :class:`pybroker.slippage.FixedSlippageModel` (fixed basis points),
        :class:`pybroker.slippage.VolatilitySlippageModel` (ATR-scaled), and
        :class:`pybroker.slippage.VolumeSlippageModel` (participation cap and
        square-law price impact). Pass ``None`` to disable slippage.

        Fill-time slippage applies to scheduled orders, stop exits, and
        position exits. Stop and position exits use the adjusted fill price
        only; share adjustments are ignored on those paths because they exit
        an entry in full.
        """
        self._slippage_model = slippage_model

    def _supports_interval_training(self, base_model_name: str) -> bool:
        """Returns whether a model source can be trained per interval.

        Pretrained models (:class:`pybroker.model.ModelLoader`) are loaded
        rather than trained, so they stay bound to the base timeframe and are
        accessed with :meth:`pybroker.context.ExecContext.preds` instead of
        ``ctx.interval(interval).preds()``.
        """
        return isinstance(
            self._scope.get_model_source(base_model_name), ModelTrainer
        )

    def _build_interval_data(
        self, df: pd.DataFrame, timeframe: str
    ) -> IntervalData:
        r"""Validates and compresses the intervals declared by executions.

        Compression narrows to the ``(symbol, interval)`` pairs some execution
        actually declared, so an execution that asks for no intervals costs
        nothing and one that asks for ``'weekly'`` does not force ``'weekly'``
        onto every other symbol in the frame.
        """
        intervals = _all_intervals(self._executions)
        if not intervals:
            return IntervalData()
        if not timeframe.strip():
            raise ValueError(
                "add_execution(intervals=...) needs the base bar spacing of "
                "the data: pass timeframe= to backtest() or walkforward() "
                "(e.g. walkforward(windows=1, timeframe='1d'))."
            )
        base_bar_seconds = base_timeframe_to_seconds(timeframe)
        # Validate the union rather than the per-symbol map so an interval
        # declared by an execution whose symbols have no rows still raises.
        for interval in intervals:
            validate_interval(interval, base_bar_seconds)
        return compress_intervals_from_frame(
            df,
            _symbol_intervals(self._executions, df),
            sorted(self._scope.custom_data_cols),
            base_bar_seconds,
        )

    def add_execution(
        self,
        fn: Optional[Callable[Concatenate[ExecContext, P], None]],
        symbols: Union[str, Iterable[str], SymbolSelector],
        models: Optional[Union[ModelSource, Iterable[ModelSource]]] = None,
        indicators: Optional[Union[Indicator, Iterable[Indicator]]] = None,
        hyperparams: Optional[Iterable[Hyperparam]] = None,
        intervals: Optional[
            Union[TimeframeInterval, Iterable[TimeframeInterval]]
        ] = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        r"""Adds an execution to backtest.

        A :class:`~pybroker.TimeframeInterval` passed to ``intervals`` is one
        of the following:

        - **Every-n-bars** (``int``): Compress every ``n`` base bars into one
          bar, where ``n > 1``. On 1-minute data, ``5`` yields 5-bar bins
          (approximately 5-minute bars).

        - **Duration** (``str``): Fixed time span as digits plus one unit
          letter — ``"1m"``, ``"5m"``, ``"1h"``, ``"30s"``, ``"1d"``, or
          ``"1w"``.

        - **Calendar** (``str``): Calendar buckets — ``"daily"``,
          ``"weekly"``, ``"monthly"``, ``"quarterly"``, or ``"yearly"``.

        For example, to use weekly bars, 5-bar bins, and 1-hour duration bars
        on a 1-minute feed::

            strategy.add_execution(
                fn,
                "SPY",
                indicators=[sma],
                intervals=["weekly", 5, "1h"],
            )
            strategy.walkforward(windows=1, timeframe="1m")

        Args:
            fn: :class:`Callable` invoked on every bar of data during the
                backtest and passed an :class:`pybroker.context.ExecContext`
                for each ticker symbol in ``symbols``.
            symbols: Ticker symbols used to run ``fn``, where ``fn`` is called
                separately for each symbol. Can also be a
                :class:`pybroker.common.SymbolSelector` — a :class:`Callable`
                ``(df) -> Sequence[str]`` that picks the symbols to trade once
                per walkforward window, so the universe changes over the
                backtest. It receives the window's **training** data, never test
                data, and therefore requires a training window:
                :meth:`.backtest` and ``train_size=0`` raise ``ValueError``.
                The candidate universe must be supplied as a
                :class:`pandas.DataFrame` rather than a
                :class:`pybroker.data.DataSource`, since the symbols to query
                are unknown until a window is split. A position in a symbol that
                a later window drops is closed at the first bar of that window;
                if the symbol has no bars left, it is closed at its final bar.
                Note that ``shuffle=True`` randomizes the training frame's row
                order, so avoid it with a selector that depends on bar order.
            models: :class:`Iterable` of :class:`pybroker.model.ModelSource`\ s
                to train/load for backtesting.
            indicators: :class:`Iterable` of
                :class:`pybroker.indicator.Indicator`\ s to compute for
                backtesting.
            hyperparams: :class:`Iterable` of
                :class:`pybroker.scope.Hyperparam`\ s that ``fn`` can read with
                :meth:`pybroker.context.ExecContext.hyperparam`.
            intervals: One or more compression intervals made available to
                ``fn`` through :meth:`pybroker.context.ExecContext.interval`.
                Each must be strictly coarser than the base bar spacing passed
                as ``timeframe`` to :meth:`.backtest` or :meth:`.walkforward`;
                invalid combinations raise ``ValueError`` when the backtest
                runs. Intervals are scoped to this execution, so another
                execution's :class:`pybroker.context.ExecContext` cannot read
                them — including inside a :meth:`.set_before_exec` or
                :meth:`.set_after_exec` callback, which receives contexts from
                every execution.
            args: Positional arguments passed to ``fn``.
            kwargs: Keyword arguments passed to ``fn``.
        """
        if callable(symbols) and not isinstance(symbols, (str, bytes)):
            stored_symbols: Union[frozenset[str], SymbolSelector] = symbols
        elif isinstance(symbols, str):
            stored_symbols = frozenset((symbols,))
        else:
            stored_symbols = frozenset(symbols)
        if isinstance(stored_symbols, frozenset):
            if not stored_symbols:
                raise ValueError("symbols cannot be empty.")
            for sym in stored_symbols:
                for exec in self._executions:
                    exec_syms = _static_symbols(exec.symbols)
                    if not exec_syms:
                        continue
                    if sym in exec_syms:
                        raise ValueError(
                            f"{sym} was already added to an execution."
                        )
        if models is not None:
            model_name_set: set[str] = set()
            for model in (
                (models,) if isinstance(models, ModelSource) else models
            ):
                if not isinstance(model, ModelSource):
                    raise TypeError(f"Invalid model type: {type(model)!r}.")
                if not self._scope.has_model_source(model.name):
                    raise ValueError(
                        f"ModelSource {model.name!r} was not registered."
                    )
                if model is not self._scope.get_model_source(model.name):
                    raise ValueError(
                        f"ModelSource {model.name!r} does not match "
                        "registered ModelSource."
                    )
                model_name_set.add(model.name)
        model_names = (
            frozenset(model_name_set) if models is not None else frozenset()
        )
        if indicators is not None:
            ind_name_set: set[str] = set()
            for ind in (
                (indicators,)
                if isinstance(indicators, Indicator)
                else indicators
            ):
                if not isinstance(ind, Indicator):
                    raise TypeError(f"Invalid indicator type: {type(ind)!r}.")
                if not self._scope.has_indicator(ind.name):
                    raise ValueError(
                        f"Indicator {ind.name!r} was not registered."
                    )
                if ind is not self._scope.get_indicator(ind.name):
                    raise ValueError(
                        f"Indicator {ind.name!r} does not match registered "
                        "Indicator."
                    )
                ind_name_set.add(ind.name)
            ind_names = frozenset(ind_name_set)
        else:
            ind_names = frozenset()
        hyperparam_name_set: set[str] = set()
        if hyperparams is not None:
            for hp in hyperparams:
                if not isinstance(hp, Hyperparam):
                    raise TypeError(f"Invalid hyperparam type: {type(hp)!r}.")
                if not self._scope.has_hyperparam(hp.name):
                    raise ValueError(
                        f"Hyperparam {hp.name!r} was not registered."
                    )
                if hp is not self._scope.get_hyperparam(hp.name):
                    raise ValueError(
                        f"Hyperparam {hp.name!r} does not match registered "
                        "Hyperparam."
                    )
                hyperparam_name_set.add(hp.name)
        if intervals is None:
            interval_set: frozenset[TimeframeInterval] = frozenset()
        else:
            # str is Iterable, so 'weekly' must not split into characters.
            declared = (
                (intervals,)
                if isinstance(intervals, (int, str))
                else tuple(intervals)
            )
            if not declared:
                raise ValueError("intervals cannot be empty.")
            seen_intervals: set[TimeframeInterval] = set()
            for interval in declared:
                norm = normalize_interval(interval)
                if norm in seen_intervals:
                    raise ValueError(f"Duplicate interval: {interval!r}.")
                seen_intervals.add(norm)
            interval_set = frozenset(seen_intervals)
        self._execution_id += 1
        self._executions.add(
            Execution(
                id=self._execution_id,
                symbols=stored_symbols,
                fn=fn,
                model_names=model_names,
                indicator_names=ind_names,
                intervals=interval_set,
                hyperparam_names=frozenset(hyperparam_name_set),
                args=args,
                kwargs=tuple(sorted(kwargs.items())),
            )
        )

    def set_before_exec(
        self, fn: Optional[Callable[[Mapping[str, ExecContext]], None]]
    ):
        r""":class:`Callable[[Mapping[str, ExecContext]]` that runs before all
        execution functions.

        Args:
            fn: :class:`Callable` that takes a :class:`Mapping` of all ticker
                symbols to :class:`ExecContext`\ s.
        """
        self._before_exec_fn = fn

    def set_after_exec(
        self, fn: Optional[Callable[[Mapping[str, ExecContext]], None]]
    ):
        r""":class:`Callable[[Mapping[str, ExecContext]]` that runs after all
        execution functions.

        Args:
            fn: :class:`Callable` that takes a :class:`Mapping` of all ticker
                symbols to :class:`ExecContext`\ s.
        """
        self._after_exec_fn = fn

    def clear_executions(self):
        """Clears executions that were added with :meth:`.add_execution`."""
        self._executions.clear()

    def backtest(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = "",
        between_time: Optional[tuple[str, str]] = None,
        days: Optional[Union[str, Day, Iterable[Union[str, Day]]]] = None,
        lookahead: int = 1,
        train_size: float = 0,
        shuffle: bool = False,
        calc_bootstrap: bool = False,
        disable_parallel_indicators: bool = False,
        enable_parallel_models: bool = False,
        warmup: Optional[int] = None,
        portfolio: Optional[Portfolio] = None,
        adjust: Optional[Any] = None,
        seed: Optional[int] = 42,
        params: Optional[dict[str, Any]] = None,
    ) -> TestResult:
        """Backtests the trading strategy by running executions that were added
        with :meth:`.add_execution`.

        Args:
            start_date: Starting date of the backtest (inclusive). Must be
                within ``start_date`` and ``end_date`` range that was passed to
                :meth:`.__init__`.
            end_date: Ending date of the backtest (inclusive). Must be
                within ``start_date`` and ``end_date`` range that was passed to
                :meth:`.__init__`.
            timeframe: Formatted string that specifies the timeframe
                resolution of the backtesting data. The timeframe string
                supports the following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks

                An example timeframe string is ``1h 30m``. Required when any
                execution declares ``intervals``, since it defines the base
                bar spacing that compression intervals are validated and
                aligned against.
            between_time: ``tuple[str, str]`` of times of day e.g.
                ('9:30', '16:00') used to filter the backtesting data
                (inclusive).
            days: Days (e.g. ``"mon"``, ``"tues"`` etc.) used to filter the
                backtesting data.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of :class:`pybroker.data.DataSource` data to use
                for training, where the max ``train_size`` is ``1``. For
                example, a ``train_size`` of ``0.9`` would result in 90% of
                data being used for training and the remaining 10% of data
                being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``. Disabled when model caching is enabled
                via :meth:`pybroker.cache.enable_model_cache`.
            calc_bootstrap: Whether to compute randomized bootstrap evaluation
                metrics. Defaults to ``False``.
            disable_parallel_indicators: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed
                serially. If ``False``, :class:`pybroker.indicator.Indicator`
                data is computed in parallel using multiple processes.
                Defaults to ``False``.
            enable_parallel_models: If ``True``, :class:`pybroker.model.ModelTrainer`
                models are trained in parallel using multiple processes.
                Defaults to ``False``.
            warmup: Number of bars that need to pass before running the
                executions.
            portfolio: Custom :class:`pybroker.portfolio.Portfolio` to use for
                backtests.
            adjust: The type of adjustment to make to the
                :class:`pybroker.data.DataSource`.
            seed: Random seed used for reproducibility. Defaults to ``42``.

        Returns:
            :class:`.TestResult` containing portfolio balances, order
            history, and evaluation metrics.
        """
        return self.walkforward(
            windows=1,
            lookahead=lookahead,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            between_time=between_time,
            days=days,
            train_size=train_size,
            shuffle=shuffle,
            calc_bootstrap=calc_bootstrap,
            disable_parallel_indicators=disable_parallel_indicators,
            enable_parallel_models=enable_parallel_models,
            warmup=warmup,
            portfolio=portfolio,
            adjust=adjust,
            seed=seed,
            params=params,
        )

    def walkforward(
        self,
        windows: int,
        lookahead: int = 1,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = "",
        between_time: Optional[tuple[str, str]] = None,
        days: Optional[Union[str, Day, Iterable[Union[str, Day]]]] = None,
        train_size: float = 0.5,
        shuffle: bool = False,
        calc_bootstrap: bool = False,
        disable_parallel_indicators: bool = False,
        enable_parallel_models: bool = False,
        warmup: Optional[int] = None,
        portfolio: Optional[Portfolio] = None,
        adjust: Optional[Any] = None,
        seed: Optional[int] = 42,
        params: Optional[dict[str, Any]] = None,
    ) -> TestResult:
        """Backtests the trading strategy using `Walkforward Analysis
        <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.
        Backtesting data supplied by the :class:`pybroker.data.DataSource` is
        divided into ``windows`` number of equal sized time windows, with each
        window split into train and test data as specified by ``train_size``.
        The backtest "walks forward" in time through each window, running
        executions that were added with :meth:`.add_execution`.

        Args:
            windows: Number of walkforward time windows.
            start_date: Starting date of the Walkforward Analysis (inclusive).
                Must be within ``start_date`` and ``end_date`` range that was
                passed to :meth:`.__init__`.
            end_date: Ending date of the Walkforward Analysis (inclusive). Must
                be within ``start_date`` and ``end_date`` range that was passed
                to :meth:`.__init__`.
            timeframe: Formatted string that specifies the timeframe
                resolution of the backtesting data. The timeframe string
                supports the following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks

                An example timeframe string is ``1h 30m``. Required when any
                execution declares ``intervals``, since it defines the base
                bar spacing that compression intervals are validated and
                aligned against.
            between_time: ``tuple[str, str]`` of times of day e.g.
                ('9:30', '16:00') used to filter the backtesting data
                (inclusive).
            days: Days (e.g. ``"mon"``, ``"tues"`` etc.) used to filter the
                backtesting data.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of :class:`pybroker.data.DataSource` data to use
                for training, where the max ``train_size`` is ``1``. For
                example, a ``train_size`` of ``0.9`` would result in 90% of
                data being used for training and the remaining 10% of data
                being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``. Disabled when model caching is enabled
                via :meth:`pybroker.cache.enable_model_cache`.
            calc_bootstrap: Whether to compute randomized bootstrap evaluation
                metrics. Defaults to ``False``.
            disable_parallel_indicators: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed
                serially. If ``False``, :class:`pybroker.indicator.Indicator`
                data is computed in parallel using multiple processes.
                Defaults to ``False``.
            enable_parallel_models: If ``True``, :class:`pybroker.model.ModelTrainer`
                models are trained in parallel using multiple processes.
                Defaults to ``False``.
            warmup: Number of bars that need to pass before running the
                executions.
            portfolio: Custom :class:`pybroker.portfolio.Portfolio` to use for
                backtests.
            adjust: The type of adjustment to make to the
                :class:`pybroker.data.DataSource`.
            seed: Random seed used for reproducibility. Defaults to ``42``.

        Returns:
            :class:`.TestResult` containing portfolio balances, order
            history, and evaluation metrics.
        """
        if warmup is not None and warmup < 1:
            raise ValueError("warmup must be > 0.")
        scope = StaticScope.instance()
        try:
            scope.freeze_data_cols()
            if not self._executions:
                raise ValueError("No executions were added.")
            if self._slippage_model is not None:
                self._slippage_model.validate(self)
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
            if start_dt is not None and end_dt is not None:
                verify_date_range(start_dt, end_dt)
            self._logger.walkforward_start(start_dt, end_dt)
            hyperparams = self._collect_hyperparams()
            run_hyperparams = build_run_hyperparams(hyperparams, params)
            backtest_settings = self._resolve_backtest_settings(
                run_hyperparams
            )
            effective_config = self._effective_config(backtest_settings)
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
            date_col = DataCol.DATE.value
            master_store = symbol_array_store_from_frame(
                _ensure_range_index(df)
            )
            master_dates_arr = df[date_col].to_numpy(
                dtype="datetime64[ns]", copy=False
            )
            if has_selector:
                indicator_data: dict[IndicatorSymbol, pd.Series] = {}
            else:
                indicator_data = self._fetch_indicators(
                    df=df,
                    cache_date_fields=cache_date_fields,
                    disable_parallel_indicators=disable_parallel_indicators,
                    interval_data=interval_data,
                    symbol_store=master_store,
                    hyperparams=run_hyperparams or None,
                )
            train_only = (
                self._before_exec_fn is None
                and self._after_exec_fn is None
                and self._rotation_sizer is None
                and backtest_settings.worst_rank_held is None
                and all(map(lambda e: e.fn is None, self._executions))
            )
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
            signals = self._run_walkforward(
                portfolio=portfolio,
                df=df,
                master_store=master_store,
                master_dates_arr=master_dates_arr,
                indicator_data=indicator_data,
                interval_data=interval_data,
                tf_seconds=tf_seconds,
                between_time=between_time,
                days=day_ids,
                windows=windows,
                lookahead=lookahead,
                train_size=train_size,
                shuffle=shuffle,
                train_only=train_only,
                warmup=warmup,
                enable_parallel_models=enable_parallel_models,
                has_selector=has_selector,
                disable_parallel_indicators=disable_parallel_indicators,
                global_cache_date_fields=cache_date_fields,
                run_hyperparams=run_hyperparams,
                backtest_settings=backtest_settings,
                effective_config=effective_config,
                rotation_sizer=self._rotation_sizer,
            )
            if train_only:
                self._logger.walkforward_completed()
            return self._to_test_result(
                start_dt,
                end_dt,
                portfolio,
                calc_bootstrap,
                train_only,
                signals if self._config.return_signals else None,
                seed,
            )
        finally:
            scope.unfreeze_data_cols()

    def _to_day_ids(
        self, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]
    ) -> Optional[tuple[int]]:
        if days is None:
            return None
        days = (
            (days,) if isinstance(days, str) or isinstance(days, Day) else days
        )
        return tuple(
            sorted(
                (day.value if isinstance(day, Day) else Day[day.upper()].value)  # type: ignore[union-attr]
                for day in set(days)  # type: ignore[arg-type]
            )
        )  # type: ignore[return-value]

    def _has_symbol_selector(self) -> bool:
        return any(_is_symbol_selector(e.symbols) for e in self._executions)

    def _liquidate_dropped_symbols(
        self,
        portfolio: Portfolio,
        selected_syms: set[str],
        test_data: pd.DataFrame,
        master_store: Optional[SymbolArrayStore] = None,
        slippage_model: Optional[SlippageModel] = None,
    ) -> None:
        """Closes positions in symbols the new window no longer selects.

        A dropped symbol exits on the first bar of the new test window. One with
        no bars left there -- delisted, or simply absent from the window -- exits
        at its final bar in ``master_store`` instead: leaving it open would
        strand the capital for the rest of the run and record no
        :class:`pybroker.portfolio.Trade`.
        """
        held = set(portfolio.long_positions) | set(portfolio.short_positions)
        dropped = frozenset(held - selected_syms)
        if not dropped:
            return
        exited: set[str] = set()
        if not test_data.empty:
            exited = self._exit_dropped_at_bar(
                portfolio=portfolio,
                store=symbol_array_store_from_frame(
                    _ensure_range_index(test_data), symbols=dropped
                ),
                symbols=dropped,
                first_bar=True,
                slippage_model=slippage_model,
            )
        remaining = dropped - exited
        if remaining and master_store is not None:
            self._exit_dropped_at_bar(
                portfolio=portfolio,
                store=master_store,
                symbols=remaining,
                first_bar=False,
                slippage_model=slippage_model,
            )

    def _exit_dropped_at_bar(
        self,
        portfolio: Portfolio,
        store: SymbolArrayStore,
        symbols: Iterable[str],
        first_bar: bool,
        slippage_model: Optional[SlippageModel],
    ) -> set[str]:
        """Exits ``symbols`` at their first or last bar in ``store``.

        Returns the symbols that had a bar to exit on.
        """
        date_col = DataCol.DATE.value
        sym_end_index: dict[str, int] = {}
        exit_dates: dict[str, np.datetime64] = {}
        for sym in symbols:
            arrays = store.sym_arrays.get(sym)
            if arrays is None:
                continue
            date_arr = arrays.get(date_col)
            if date_arr is None or not len(date_arr):
                continue
            # Locate the bar rather than indexing the ends, since a store is
            # only as ordered as the frame it was built from.
            loc = int(
                np.argmin(date_arr) if first_bar else np.argmax(date_arr)
            )
            sym_end_index[sym] = loc + 1
            exit_dates[sym] = date_arr[loc]
        if not sym_end_index:
            return set()
        col_scope = ColumnScope(store)
        price_scope = PriceScope(
            col_scope, sym_end_index, self._config.round_fill_price
        )
        ind_scope = IndicatorScope({}, sorted(set(exit_dates.values())))
        for sym, date in exit_dates.items():
            portfolio.exit_position(
                date,
                sym,
                buy_fill_price=price_scope.fetch(
                    sym, self._config.exit_cover_fill_price
                ),
                sell_fill_price=price_scope.fetch(
                    sym, self._config.exit_sell_fill_price
                ),
                col_scope=col_scope,
                ind_scope=ind_scope,
                sym_end_index=sym_end_index,
                slippage_model=slippage_model,
            )
        return set(sym_end_index)

    def _fractional_shares_enabled(self):
        return self._config.enable_fractional_shares or isinstance(
            self._data_source, AlpacaCrypto
        )

    def _run_walkforward(
        self,
        portfolio: Portfolio,
        df: pd.DataFrame,
        master_store: SymbolArrayStore,
        master_dates_arr: NDArray[np.datetime64],
        indicator_data: dict[IndicatorSymbol, pd.Series],
        interval_data: IntervalData,
        tf_seconds: int,
        between_time: Optional[tuple[str, str]],
        days: Optional[tuple[int]],
        windows: int,
        lookahead: int,
        train_size: float,
        shuffle: bool,
        train_only: bool,
        warmup: Optional[int],
        enable_parallel_models: bool = False,
        has_selector: bool = False,
        disable_parallel_indicators: bool = False,
        global_cache_date_fields: Optional[CacheDateFields] = None,
        run_hyperparams: Optional[dict[str, Any]] = None,
        backtest_settings: Optional[BacktestSettings] = None,
        effective_config: Optional[StrategyConfig] = None,
        rotation_sizer: Optional[Callable[[RotationContext], None]] = None,
    ) -> dict[str, pd.DataFrame]:
        if backtest_settings is None:
            backtest_settings = self._resolve_backtest_settings(
                run_hyperparams
            )
        if effective_config is None:
            effective_config = self._effective_config(backtest_settings)
        if rotation_sizer is None:
            rotation_sizer = self._rotation_sizer
        sessions: dict[str, dict] = defaultdict(dict)
        exit_dates: dict[str, np.datetime64] = {}
        sym_col = DataCol.SYMBOL.value
        date_col = DataCol.DATE.value
        if self._config.exit_on_last_bar:
            if has_selector:
                exit_symbols = set(df[sym_col].unique())
            else:
                exit_symbols = {
                    sym
                    for exec in self._executions
                    for sym in _static_symbols(exec.symbols)
                }
            if exit_symbols and not df.empty:
                mask = df[sym_col].isin(exit_symbols)
                masked = df.loc[mask]
                for sym, sym_dates in _iter_symbol_date_groups(masked):
                    if sym in exit_symbols:
                        exit_dates[sym] = np.datetime64(sym_dates.max())
        signals: dict[str, pd.DataFrame] = {}
        signal_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
        for train_rows, test_rows in self.walkforward_split(
            df=df,
            windows=windows,
            lookahead=lookahead,
            train_size=train_size,
            shuffle=shuffle,
        ):
            models: dict[ModelSymbol, TrainedModel] = {}
            train_data = (
                df.iloc[train_rows] if len(train_rows) else df.iloc[:0]
            )
            test_data = df.iloc[test_rows] if len(test_rows) else df.iloc[:0]
            selection_data = _selection_df(
                self._executions, train_data, test_data
            )
            if has_selector:
                if global_cache_date_fields is None:
                    raise ValueError("global_cache_date_fields is required.")
                window_executions = _resolve_executions(
                    self._executions, selection_data
                )
                window_indicator_data = self._fetch_indicators(
                    df=df,
                    cache_date_fields=global_cache_date_fields,
                    disable_parallel_indicators=disable_parallel_indicators,
                    interval_data=interval_data,
                    executions=window_executions,
                    symbol_store=master_store,
                    hyperparams=run_hyperparams,
                )
                indicator_data.update(window_indicator_data)
            else:
                window_executions = self._executions
            selected_syms = _selected_symbols(
                window_executions, test_data, has_selector
            )
            self._liquidate_dropped_symbols(
                portfolio,
                selected_syms,
                test_data,
                master_store=master_store,
                slippage_model=self._slippage_model,
            )
            train_store = None
            test_store = None
            history_store = None
            if not test_data.empty:
                test_dates_arr = _unique_dates_from_rows(
                    master_dates_arr, test_rows
                )
                test_store = slice_symbol_array_store_by_dates(
                    master_store, test_dates_arr
                )
            if not train_data.empty:
                train_dates_arr = _unique_dates_from_rows(
                    master_dates_arr, train_rows
                )
                train_store = slice_symbol_array_store_by_dates(
                    master_store, train_dates_arr
                )
                if test_store is not None:
                    # Slice the contiguous span rather than merging the two
                    # windows: with lookahead > 1 the skipped bars fall
                    # between them, and omitting those would make "lag 1" at
                    # the train/test boundary reach lookahead bars back.
                    span_mask = (master_dates_arr >= train_dates_arr[0]) & (
                        master_dates_arr <= test_dates_arr[-1]
                    )
                    history_store = slice_symbol_array_store_by_dates(
                        master_store,
                        np.unique(master_dates_arr[span_mask]),
                    )
                else:
                    history_store = train_store
                train_symbols = set(train_data[sym_col].unique())
                model_syms: set[ModelSymbol] = set()
                for sym in train_symbols:
                    for execution in window_executions:
                        if sym not in _static_symbols(execution.symbols):
                            continue
                        for model_name in execution.model_names:
                            base_name, token = parse_model_interval_name(
                                model_name
                            )
                            if token is not None:
                                model_syms.add(ModelSymbol(model_name, sym))
                                continue
                            model_syms.add(ModelSymbol(model_name, sym))
                            if not self._supports_interval_training(base_name):
                                continue
                            for tf in execution.intervals:
                                model_syms.add(
                                    ModelSymbol(
                                        model_interval_name(base_name, tf),
                                        sym,
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
                        base_name, token = parse_model_interval_name(
                            model_name
                        )
                        if token is not None:
                            continue
                        source = self._scope.get_model_source(base_name)
                        if isinstance(source, ModelTrainer) and source.pooled:
                            pooled_model_groups[(model_name, execution.id)] = (
                                exec_syms
                            )
                            for tf in execution.intervals:
                                pooled_model_groups[
                                    (
                                        model_interval_name(base_name, tf),
                                        execution.id,
                                    )
                                ] = exec_syms
                train_dates = get_unique_sorted_dates(train_data[date_col])
                models = self.train_models(
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
                    enable_parallel_models=enable_parallel_models,
                    pooled_model_groups=pooled_model_groups,
                    interval_data=interval_data,
                    history_store=history_store,
                    train_store=train_store,
                    test_store=test_store,
                )
            if test_data.empty:
                return signals
            if history_store is None:
                history_store = test_store
            history_col_scope = ColumnScope(history_store)
            test_col_scope = ColumnScope(test_store)
            split_signals = self.backtest_executions(
                config=effective_config,
                executions=window_executions,
                before_exec_fn=self._before_exec_fn,
                after_exec_fn=self._after_exec_fn,
                sessions=sessions,
                models=models,
                indicator_data=indicator_data,
                interval_data=interval_data.slice_for_test(
                    symbol_dates_from_frame(test_data)
                ),
                test_data=test_data,
                portfolio=portfolio,
                exit_dates=exit_dates,
                backtest_settings=backtest_settings,
                rotation_sizer=rotation_sizer,
                train_only=train_only,
                slippage_model=self._slippage_model,
                enable_fractional_shares=self._fractional_shares_enabled(),
                round_fill_price=effective_config.round_fill_price,
                warmup=warmup,
                history_col_scope=history_col_scope,
                test_col_scope=test_col_scope,
                run_hyperparams=run_hyperparams,
            )
            for sym, signals_df in split_signals.items():
                signal_frames[sym].append(signals_df)
        for sym, frames in signal_frames.items():
            signals[sym] = (
                frames[0]
                if len(frames) == 1
                else pd.concat(frames, ignore_index=True)
            )
        return signals

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        between_time: Optional[tuple[str, str]],
        days: Optional[tuple[int]],
    ) -> pd.DataFrame:
        if start_date != self._start_date or end_date != self._end_date:
            df = _between(df, start_date, end_date)
            df = _ensure_range_index(df)
        if df[DataCol.DATE.value].dt.tz is not None:
            # Fixes bug on Windows.
            # https://stackoverflow.com/questions/51827582/message-exception-ignored-when-dealing-pandas-datetime-type
            df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(None)
        is_time_range = between_time is not None or days is not None
        if is_time_range:
            df = df.reset_index(drop=True).set_index(DataCol.DATE.value)
        if days is not None:
            self._logger.info_walkforward_on_days(days)
            df = df[df.index.weekday.isin(frozenset(days))]
        if between_time is not None:
            if len(between_time) != 2:
                raise ValueError(
                    "between_time must be a tuple[str, str] of start time and"
                    f" end time, received {between_time!r}."
                )
            self._logger.info_walkforward_between_time(between_time)
            df = df.between_time(*between_time)
        if is_time_range:
            df = df.reset_index()
        return df

    def _fetch_indicators(
        self,
        df: pd.DataFrame,
        cache_date_fields: CacheDateFields,
        disable_parallel_indicators: bool,
        interval_data: Optional[IntervalData] = None,
        executions: Optional[set[Execution]] = None,
        symbol_store: Optional[SymbolArrayStore] = None,
        hyperparams: Optional[dict[str, Any]] = None,
    ) -> dict[IndicatorSymbol, pd.Series]:
        exec_set = executions if executions is not None else self._executions
        indicator_syms: set[IndicatorSymbol] = set()
        for execution in exec_set:
            for sym in _static_symbols(execution.symbols):
                for model_name in execution.model_names:
                    base_name, token = parse_model_interval_name(model_name)
                    ind_names = self._scope.get_indicator_names(base_name)
                    for ind_name in ind_names:
                        indicator_syms.add(IndicatorSymbol(ind_name, sym))
                        if token is not None:
                            indicator_syms.add(
                                IndicatorSymbol(
                                    indicator_interval_name(ind_name, token),
                                    sym,
                                )
                            )
                        elif execution.intervals and (
                            self._supports_interval_training(base_name)
                        ):
                            for tf in execution.intervals:
                                indicator_syms.add(
                                    IndicatorSymbol(
                                        indicator_interval_name(ind_name, tf),
                                        sym,
                                    )
                                )
                for ind_name in execution.indicator_names:
                    base_name, token = parse_indicator_interval_name(ind_name)
                    indicator_syms.add(IndicatorSymbol(ind_name, sym))
                    if token is None and execution.intervals:
                        for tf in execution.intervals:
                            indicator_syms.add(
                                IndicatorSymbol(
                                    indicator_interval_name(base_name, tf),
                                    sym,
                                )
                            )
        return self.compute_indicators(
            df=df,
            indicator_syms=indicator_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel_indicators=disable_parallel_indicators,
            interval_data=interval_data,
            symbol_store=symbol_store,
            hyperparams=hyperparams,
        )

    def _fetch_data(
        self, timeframe: str, adjust: Optional[Any]
    ) -> pd.DataFrame:
        has_selector = self._has_symbol_selector()
        if has_selector and isinstance(self._data_source, DataSource):
            raise ValueError(
                "Dynamic symbol selection requires a pandas DataFrame data "
                "source containing the candidate universe."
            )
        if isinstance(self._data_source, DataSource):
            unique_syms = frozenset(
                sym
                for execution in self._executions
                for sym in _static_symbols(execution.symbols)
            )
            df = self._data_source.query(
                unique_syms,
                self._start_date,
                self._end_date,
                timeframe,
                adjust,
            )
        else:
            df = _between(self._data_source, self._start_date, self._end_date)
            if not has_selector:
                unique_syms = frozenset(
                    sym
                    for execution in self._executions
                    for sym in _static_symbols(execution.symbols)
                )
                df = df[df[DataCol.SYMBOL.value].isin(unique_syms)]
        if df.empty:
            raise ValueError("DataSource is empty.")
        return _ensure_range_index(df)

    def _to_test_result(
        self,
        start_date: datetime,
        end_date: datetime,
        portfolio: Portfolio,
        calc_bootstrap: bool,
        train_only: bool,
        signals: Optional[dict[str, pd.DataFrame]],
        seed: Optional[int],
    ) -> TestResult:
        if train_only:
            return TestResult(
                start_date=start_date,
                end_date=end_date,
                portfolio=pd.DataFrame(),
                positions=pd.DataFrame(),
                orders=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics=EvalMetrics(),
                metrics_df=pd.DataFrame(),
                bootstrap=None,
                signals=signals,
                stops=None,
            )
        pos_df = pd.DataFrame.from_records(
            portfolio.position_bars, columns=PositionBar._fields
        )
        for col in (
            "close",
            "equity",
            "market_value",
            "margin",
            "unrealized_pnl",
        ):
            if not pos_df.empty:
                pos_df[col] = quantize(
                    pos_df, col, self._config.round_test_result
                )
        if not pos_df.empty:
            pos_df.set_index(["symbol", "date"], inplace=True)
        bar_records = (
            portfolio.bars if portfolio.bars else portfolio._metrics_bars
        )
        portfolio_df = pd.DataFrame.from_records(
            bar_records, columns=PortfolioBar._fields, index="date"
        )
        for col in (
            "cash",
            "equity",
            "margin",
            "margin_loan",
            "net_cash_balance",
            "market_value",
            "pnl",
            "unrealized_pnl",
            "fees",
        ):
            portfolio_df[col] = quantize(
                portfolio_df, col, self._config.round_test_result
            )
        orders_df = pd.DataFrame.from_records(
            portfolio.orders, columns=Order._fields, index="id"
        )
        for col in ("limit_price", "fill_price", "fees"):
            orders_df[col] = quantize(
                orders_df, col, self._config.round_test_result
            )
        trades_df = pd.DataFrame.from_records(
            portfolio.trades, columns=Trade._fields, index="id"
        )
        trades_df["bars"] = trades_df["bars"].astype(int)
        for col in (
            "entry",
            "exit",
            "pnl",
            "return_pct",
            "agg_pnl",
            "pnl_per_bar",
            "mae",
            "mfe",
        ):
            trades_df[col] = quantize(
                trades_df, col, self._config.round_test_result
            )
        shares_type = float if self._fractional_shares_enabled() else int
        pos_df["long_shares"] = pos_df["long_shares"].astype(shares_type)
        pos_df["short_shares"] = pos_df["short_shares"].astype(shares_type)
        orders_df["shares"] = orders_df["shares"].astype(shares_type)
        trades_df["shares"] = trades_df["shares"].astype(shares_type)
        eval_result = self.evaluate(
            portfolio_df=portfolio_df,
            trades_df=trades_df,
            calc_bootstrap=calc_bootstrap,
            bootstrap_samples=self._config.bootstrap_samples,
            bars_per_year=self._config.bars_per_year,
            seed=seed,
        )
        metrics = [
            (k, v)
            for k, v in dataclasses.asdict(eval_result.metrics).items()
            if v is not None
        ]
        metrics_df = pd.DataFrame(metrics, columns=["name", "value"])
        stops_df = None
        if self._config.return_stops:
            stops_df = pd.DataFrame.from_records(
                portfolio._stop_records, columns=StopRecord._fields
            )
        self._logger.walkforward_completed()
        return TestResult(
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio_df,
            positions=pos_df,
            orders=orders_df,
            trades=trades_df,
            metrics=eval_result.metrics,
            metrics_df=metrics_df,
            bootstrap=eval_result.bootstrap,
            signals=signals,
            stops=stops_df,
        )
