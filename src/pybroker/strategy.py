"""Contains implementation for backtesting trading strategies."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import dataclasses
import math
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
    get_unique_sorted_dates,
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
    set_exec_ctx_data,
)
from pybroker.data import AlpacaCrypto, DataSource
from pybroker.eval import BootstrapResult, EvalMetrics, EvaluateMixin
from pybroker.indicator import Indicator, IndicatorsMixin, TimeframeIndicator
from pybroker.model import (
    ModelLoader,
    ModelSource,
    ModelTrainer,
    ModelsMixin,
    TimeframeModelSource,
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
    PendingOrderScope,
    PredictionScope,
    PriceScope,
    StaticScope,
    TimeframeScope,
    get_signals,
)
from pybroker.timeframe import (
    TimeframeData,
    TimeframeInterval,
    compress_symbol_df,
    indicator_timeframe_name,
    infer_base_bar_seconds,
    normalize_timeframe_interval,
    parse_model_timeframe_name,
    validate_timeframe_interval,
)
from pybroker.slippage import SlippageModel
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
    TypeGuard,
    Union,
)
from typing_extensions import Concatenate, ParamSpec


P = ParamSpec("P")


def _between(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    if df.empty:
        return df
    return df[
        (df[DataCol.DATE.value].dt.tz_localize(None) >= start_date)
        & (df[DataCol.DATE.value].dt.tz_localize(None) <= end_date)
    ]


def _sort_by_buy_score(result: ExecResult) -> float:
    if result.long_score is not None:
        return result.long_score
    return 0.0 if result.score is None else result.score


def _sort_by_sell_score(result: ExecResult) -> float:
    if result.short_score is not None:
        return -result.short_score
    return 0.0 if result.score is None else result.score


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


def _set_short_target_shares(ctx: ExecContext, target: float):
    target_shares = ctx.calc_target_shares(target)
    pos = ctx.short_pos()
    if pos is None:
        ctx.sell_shares = target_shares
    elif pos.shares < target_shares:
        ctx.sell_shares = target_shares - pos.shares
    elif pos.shares > target_shares:
        ctx.cover_shares = pos.shares - target_shares


def _apply_worst_rank_held_long(
    active_ctxs: Mapping[str, ExecContext],
    portfolio: Portfolio,
    max_long_positions: int,
    worst_rank_held: int,
):
    rankable_scores: dict[str, float] = {}
    for sym, ctx in active_ctxs.items():
        score = _long_rotation_score(ctx)
        if score is not None:
            rankable_scores[sym] = score
    ranks = _rank_by_score(rankable_scores)
    target_size = 1 / max_long_positions
    for sym in portfolio.long_positions:
        if sym not in active_ctxs:
            continue
        ctx = active_ctxs[sym]
        if ctx.sell_shares is not None:
            continue
        rank = ranks.get(sym)
        if rank is None or rank > worst_rank_held:
            ctx.sell_all_shares()
    for sym, ctx in active_ctxs.items():
        if sym in portfolio.long_positions or sym in portfolio.short_positions:
            continue
        if ctx.buy_shares is not None or _long_rotation_score(ctx) is None:
            continue
        ctx.set_target_shares(target_size)


def _apply_worst_rank_held_short(
    active_ctxs: Mapping[str, ExecContext],
    portfolio: Portfolio,
    max_short_positions: int,
    worst_rank_held: int,
):
    rankable_scores: dict[str, float] = {}
    for sym, ctx in active_ctxs.items():
        score = _short_rotation_score(ctx)
        if score is not None:
            rankable_scores[sym] = score
    ranks = _rank_by_short_score(rankable_scores)
    target_size = 1 / max_short_positions
    for sym in portfolio.short_positions:
        if sym not in active_ctxs:
            continue
        ctx = active_ctxs[sym]
        if ctx.buy_shares is not None:
            continue
        rank = ranks.get(sym)
        if rank is None or rank > worst_rank_held:
            ctx.cover_all_shares()
    for sym, ctx in active_ctxs.items():
        if sym in portfolio.short_positions or sym in portfolio.long_positions:
            continue
        if ctx.sell_shares is not None or _short_rotation_score(ctx) is None:
            continue
        _set_short_target_shares(ctx, target_size)


def _apply_worst_rank_held(
    active_ctxs: Mapping[str, ExecContext],
    portfolio: Portfolio,
    config: StrategyConfig,
):
    worst_rank_held = config.worst_rank_held
    if worst_rank_held is None:
        return
    max_long_positions = config.max_long_positions
    max_short_positions = config.max_short_positions
    if max_long_positions is not None:
        _apply_worst_rank_held_long(
            active_ctxs,
            portfolio,
            max_long_positions,
            worst_rank_held,
        )
    if max_short_positions is not None:
        _apply_worst_rank_held_short(
            active_ctxs,
            portfolio,
            max_short_positions,
            worst_rank_held,
        )


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
        args: Additional positional arguments for ``fn``.
        kwargs: Additional keyword arguments for ``fn``.
    """

    id: int
    symbols: frozenset[str]
    fn: Optional[Callable[[ExecContext], None]]
    model_names: frozenset[str]
    indicator_names: frozenset[str]
    args: tuple[Any, ...] = tuple()
    kwargs: tuple[tuple[str, Any], ...] = tuple()


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
        train_only: bool = False,
        slippage_model: Optional[SlippageModel] = None,
        enable_fractional_shares: bool = False,
        round_fill_price: bool = True,
        warmup: Optional[int] = None,
        timeframe_data: TimeframeData = TimeframeData(),
        declared_timeframes: frozenset[TimeframeInterval] = frozenset(),
        history_col_scope: Optional[ColumnScope] = None,
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
        test_dates = get_unique_sorted_dates(test_data[DataCol.DATE.value])
        test_syms = sorted(test_data[DataCol.SYMBOL.value].unique())
        test_data = (
            test_data.reset_index(drop=True)
            .set_index([DataCol.SYMBOL.value, DataCol.DATE.value])
            .sort_index()
        )
        col_scope = ColumnScope(test_data)
        ind_scope = IndicatorScope(indicator_data, test_dates)
        input_scope = ModelInputScope(
            col_scope,
            ind_scope,
            models,
            history_col_scope,
            test_dates,
        )
        timeframe_scope = TimeframeScope(
            timeframe_data,
            ind_scope,
            declared_timeframes,
            models,
            test_dates,
        )
        pred_scope = PredictionScope(models, input_scope)
        if train_only:
            if config.return_signals:
                return get_signals(test_syms, col_scope, ind_scope, pred_scope)
            return {}
        sym_end_index: dict[str, int] = defaultdict(int)
        price_scope = PriceScope(col_scope, sym_end_index, round_fill_price)
        pending_order_scope = PendingOrderScope()
        exec_ctxs: dict[str, ExecContext] = {}
        exec_fns: dict[str, Callable[[ExecContext], None]] = {}
        exec_args: dict[str, tuple[Any, ...]] = {}
        exec_kwargs: dict[str, tuple[tuple[str, Any], ...]] = {}
        for sym in test_syms:
            for exec in executions:
                if sym not in exec.symbols:
                    continue
                exec_ctxs[sym] = ExecContext(
                    symbol=sym,
                    config=config,
                    portfolio=portfolio,
                    col_scope=col_scope,
                    ind_scope=ind_scope,
                    timeframe_scope=timeframe_scope,
                    declared_timeframes=declared_timeframes,
                    input_scope=input_scope,
                    pred_scope=pred_scope,
                    pending_order_scope=pending_order_scope,
                    models=models,
                    sym_end_index=sym_end_index,
                    session=sessions[sym],
                )
                exec_args[sym] = exec.args
                exec_kwargs[sym] = exec.kwargs
                if exec.fn is not None:
                    exec_fns[sym] = exec.fn
        sym_exec_dates = {
            sym: frozenset(test_data.loc[pd.IndexSlice[sym, :]].index.values)
            for sym in exec_ctxs.keys()
        }
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
            for sym, ctx in exec_ctxs.items():
                if date not in sym_exec_dates[sym]:
                    continue
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
                if is_cover_sched:
                    cover_sched[date].sort(
                        key=_sort_by_buy_score, reverse=True
                    )
                elif is_buy_sched:
                    buy_sched[date].sort(key=_sort_by_buy_score, reverse=True)
            if is_sell_sched and config.max_short_positions is not None:
                sell_sched[date].sort(key=_sort_by_sell_score, reverse=True)
            portfolio.check_stops(date, price_scope)
            if is_cover_sched:
                self._place_buy_orders(
                    date=date,
                    price_scope=price_scope,
                    pending_order_scope=pending_order_scope,
                    buy_sched=cover_sched,
                    portfolio=portfolio,
                    enable_fractional_shares=enable_fractional_shares,
                )
            if is_sell_sched:
                self._place_sell_orders(
                    date=date,
                    price_scope=price_scope,
                    pending_order_scope=pending_order_scope,
                    sell_sched=sell_sched,
                    portfolio=portfolio,
                    enable_fractional_shares=enable_fractional_shares,
                )
            if is_buy_sched:
                self._place_buy_orders(
                    date=date,
                    price_scope=price_scope,
                    pending_order_scope=pending_order_scope,
                    buy_sched=buy_sched,
                    portfolio=portfolio,
                    enable_fractional_shares=enable_fractional_shares,
                )
            portfolio.capture_bar(date, col_scope, sym_end_index)
            if before_exec_fn is not None and active_ctxs:
                before_exec_fn(active_ctxs)
            for sym, ctx in active_ctxs.items():
                if sym in exec_fns:
                    exec_fns[sym](
                        ctx,
                        *exec_args.get(sym, ()),
                        **dict(exec_kwargs.get(sym, ())),
                    )
            if after_exec_fn is not None and active_ctxs:
                after_exec_fn(active_ctxs)
            if config.worst_rank_held is not None:
                _apply_worst_rank_held(active_ctxs, portfolio, config)
            for ctx in active_ctxs.values():
                if (
                    slippage_model
                    and not ctx._exiting_pos
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
            if config.worst_rank_held is not None:
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
            else:
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
                )
            portfolio.incr_bars()
            if i % 10 == 0 or i == len(test_dates) - 1:
                logger.backtest_executions_loading(i + 1)
        return (
            get_signals(test_syms, col_scope, ind_scope, pred_scope)
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
    ):
        buy_fill_price = price_scope.fetch(ctx.symbol, exit_cover_fill_price)
        sell_fill_price = price_scope.fetch(ctx.symbol, exit_sell_fill_price)
        portfolio.exit_position(
            date,
            ctx.symbol,
            buy_fill_price=buy_fill_price,
            sell_fill_price=sell_fill_price,
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
            result.pending_order_id = pending_order_scope.add(
                type=order_type,
                symbol=result.symbol,
                created=created,
                exec_date=date,
                shares=shares,
                limit_price=limit_price,
                fill_price=fill_price,
            )
            sched[date].append(result)
            logger.debug_schedule_order(date, result)
        else:
            logger.debug_unscheduled_order(result)

    def _place_buy_orders(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        pending_order_scope: PendingOrderScope,
        buy_sched: dict[np.datetime64, list[ExecResult]],
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ):
        buy_results = buy_sched[date]
        for result in buy_results:
            if result.buy_shares is None:
                continue
            if (
                result.pending_order_id is None
                or not pending_order_scope.contains(result.pending_order_id)
            ):
                continue
            pending = tuple(
                pending_order_scope.orders(
                    order_id=result.pending_order_id,
                )
            )
            pending_order_scope.remove(result.pending_order_id)
            buy_shares = self._get_shares(
                result.buy_shares, enable_fractional_shares
            )
            fill_price = price_scope.fetch(
                result.symbol, result.buy_fill_price
            )
            created = pending[0].created if pending else None
            order_type = (
                OrderType.LIMIT
                if result.buy_limit_price is not None
                else OrderType.MARKET
            )
            order = portfolio.buy(
                date=date,
                symbol=result.symbol,
                shares=buy_shares,
                fill_price=fill_price,
                limit_price=result.buy_limit_price,
                stops=result.long_stops,
                created=created,
                order_type=order_type,
            )
            logger = StaticScope.instance().logger
            if order is None:
                logger.debug_unfilled_buy_order(
                    date=date,
                    symbol=result.symbol,
                    shares=buy_shares,
                    fill_price=fill_price,
                    limit_price=result.buy_limit_price,
                )
            else:
                logger.debug_filled_buy_order(
                    date=date,
                    symbol=result.symbol,
                    shares=buy_shares,
                    fill_price=fill_price,
                    limit_price=result.buy_limit_price,
                )
        del buy_sched[date]

    def _place_sell_orders(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        pending_order_scope: PendingOrderScope,
        sell_sched: dict[np.datetime64, list[ExecResult]],
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ):
        sell_results = sell_sched[date]
        for result in sell_results:
            if result.sell_shares is None:
                continue
            if (
                result.pending_order_id is None
                or not pending_order_scope.contains(result.pending_order_id)
            ):
                continue
            pending = tuple(
                pending_order_scope.orders(
                    order_id=result.pending_order_id,
                )
            )
            pending_order_scope.remove(result.pending_order_id)
            sell_shares = self._get_shares(
                result.sell_shares, enable_fractional_shares
            )
            fill_price = price_scope.fetch(
                result.symbol, result.sell_fill_price
            )
            created = pending[0].created if pending else None
            order_type = (
                OrderType.LIMIT
                if result.sell_limit_price is not None
                else OrderType.MARKET
            )
            order = portfolio.sell(
                date=date,
                symbol=result.symbol,
                shares=sell_shares,
                fill_price=fill_price,
                limit_price=result.sell_limit_price,
                stops=result.short_stops,
                created=created,
                order_type=order_type,
            )
            logger = StaticScope.instance().logger
            if order is None:
                logger.debug_unfilled_sell_order(
                    date=date,
                    symbol=result.symbol,
                    shares=sell_shares,
                    fill_price=fill_price,
                    limit_price=result.sell_limit_price,
                )
            else:
                logger.debug_filled_sell_order(
                    date=date,
                    symbol=result.symbol,
                    shares=sell_shares,
                    fill_price=fill_price,
                    limit_price=result.sell_limit_price,
                )
        del sell_sched[date]

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
    """Contains ``train_data`` and ``test_data`` of a time window used for
    `Walkforward Analysis
    <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

    Attributes:
        train_data: Train data.
        test_data: Test data.
    """

    train_data: NDArray[np.int_]
    test_data: NDArray[np.int_]


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
        dates = df[[date_col]]
        window_dates = get_unique_sorted_dates(df[date_col])
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
                    test_idx = dates[
                        (dates[date_col] >= window_dates[start])
                        & (dates[date_col] <= window_dates[end - 1])
                    ]
                    test_idx = test_idx.index.to_numpy(copy=True)
                    yield WalkforwardWindow(np.array(tuple()), test_idx)
                else:
                    train_idx = dates[
                        (dates[date_col] >= window_dates[start])
                        & (dates[date_col] <= window_dates[end - 1])
                    ]
                    train_idx = train_idx.index.to_numpy(copy=True)
                    if shuffle:
                        np.random.shuffle(train_idx)
                    yield WalkforwardWindow(train_idx, np.array(tuple()))
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
            train_idx = dates[
                (dates[date_col] >= window_dates[train_start])
                & (dates[date_col] <= window_dates[train_end])
            ]
            test_idx = dates[
                (dates[date_col] >= window_dates[test_start])
                & (dates[date_col] <= window_dates[test_end])
            ]
            train_idx = train_idx.index.to_numpy(copy=True)
            test_idx = test_idx.index.to_numpy(copy=True)
            if shuffle:
                np.random.shuffle(train_idx)
            yield WalkforwardWindow(train_idx, test_idx)
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
            window_dates = window_dates[::-1]
            for train_start, train_end, test_start, test_end in window_idx:
                train_idx = dates[
                    (dates[date_col] > window_dates[train_start])
                    & (dates[date_col] <= window_dates[train_end])
                ]
                test_idx = dates[
                    (dates[date_col] > window_dates[test_start])
                    & (dates[date_col] <= window_dates[test_end])
                ]
                train_idx = train_idx.index.to_numpy(copy=True)
                test_idx = test_idx.index.to_numpy(copy=True)
                if shuffle:
                    np.random.shuffle(train_idx)
                yield WalkforwardWindow(train_idx, test_idx)


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


class Strategy(
    BacktestMixin,
    EvaluateMixin,
    IndicatorsMixin,
    ModelsMixin,
    WalkforwardMixin,
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
        self._slippage_model: Optional[SlippageModel] = None
        self._timeframes: frozenset[TimeframeInterval] = frozenset()
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
        if config.buy_delay <= 0:
            raise ValueError("buy_delay must be greater than 0.")
        if config.sell_delay <= 0:
            raise ValueError("sell_delay must be greater than 0.")
        if config.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be greater than 0.")
        if config.bootstrap_sample_size <= 0:
            raise ValueError("bootstrap_sample_size must be greater than 0.")
        if config.worst_rank_held is not None:
            if (
                config.max_long_positions is None
                and config.max_short_positions is None
            ):
                raise ValueError(
                    "worst_rank_held requires max_long_positions or "
                    "max_short_positions to be set."
                )
            if (
                config.max_long_positions is not None
                and config.worst_rank_held < config.max_long_positions
            ):
                raise ValueError(
                    "worst_rank_held must be greater than or equal to "
                    "max_long_positions."
                )
            if (
                config.max_short_positions is not None
                and config.worst_rank_held < config.max_short_positions
            ):
                raise ValueError(
                    "worst_rank_held must be greater than or equal to "
                    "max_short_positions."
                )

    def _verify_data_source(
        self, data_source: Union[DataSource, pd.DataFrame]
    ):
        if isinstance(data_source, pd.DataFrame):
            verify_data_source_columns(data_source)
        elif not isinstance(data_source, DataSource):
            raise TypeError(f"Invalid data_source type: {type(data_source)}")

    def set_slippage_model(self, slippage_model: Optional[SlippageModel]):
        """Sets :class:`pybroker.slippage.SlippageModel`."""
        self._slippage_model = slippage_model

    def set_timeframes(self, *intervals: TimeframeInterval) -> None:
        r"""Declares compression timeframes for multi-timeframe strategies.

        Each interval must be strictly coarser than the base bar feed; invalid
        combinations raise ``ValueError`` when :meth:`.backtest` or
        :meth:`.walkforward` runs. Declared intervals can be accessed in
        execution functions with
        :meth:`pybroker.context.ExecContext.timeframe` and bound to indicators
        and models with :meth:`pybroker.indicator.Indicator.timeframe` and
        :meth:`pybroker.model.ModelSource.timeframe`.

        A :class:`~pybroker.TimeframeInterval` is one of the following:

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

            strategy.set_timeframes("weekly", 5, "1h")
            strategy.add_execution(
                fn,
                "SPY",
                indicators=[sma.timeframe("weekly"), sma.timeframe(5)],
            )

        Args:
            *intervals: One or more compression intervals to make available
                for the strategy.
        """
        normalized = []
        seen = set()
        for interval in intervals:
            norm = normalize_timeframe_interval(interval)
            if norm in seen:
                raise ValueError(
                    f"Duplicate timeframe interval: {interval!r}."
                )
            seen.add(norm)
            normalized.append(norm)
        self._timeframes = frozenset(normalized)

    def _validate_timeframes_for_base(
        self, df: pd.DataFrame, timeframe: str
    ) -> None:
        if not self._timeframes:
            return
        base_bar_seconds = infer_base_bar_seconds(df, timeframe)
        for interval in self._timeframes:
            validate_timeframe_interval(interval, base_bar_seconds)

    def _compress_timeframes(
        self, df: pd.DataFrame, symbols: Iterable[str]
    ) -> TimeframeData:
        if not self._timeframes:
            return TimeframeData()
        timeframe_data = TimeframeData()
        custom_cols = self._scope.custom_data_cols
        sym_col = DataCol.SYMBOL.value
        symbol_set = set(symbols)
        for sym, group in df.groupby(sym_col, sort=False, observed=True):
            if sym not in symbol_set:
                continue
            sym_df = group.reset_index(drop=True)
            for interval in self._timeframes:
                data = compress_symbol_df(sym_df, interval, custom_cols)
                timeframe_data.compressed[(sym, interval)] = data
        return timeframe_data

    def add_execution(
        self,
        fn: Optional[Callable[Concatenate[ExecContext, P], None]],
        symbols: Union[str, Iterable[str]],
        models: Optional[Union[ModelSource, Iterable[ModelSource]]] = None,
        indicators: Optional[Union[Indicator, Iterable[Indicator]]] = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        r"""Adds an execution to backtest.

        Args:
            fn: :class:`Callable` invoked on every bar of data during the
                backtest and passed an :class:`pybroker.context.ExecContext`
                for each ticker symbol in ``symbols``.
            symbols: Ticker symbols used to run ``fn``, where ``fn`` is called
                separately for each symbol.
            models: :class:`Iterable` of :class:`pybroker.model.ModelSource`\ s
                to train/load for backtesting.
            indicators: :class:`Iterable` of
                :class:`pybroker.indicator.Indicator`\ s to compute for
                backtesting.
            args: Positional arguments passed to ``fn``.
            kwargs: Keyword arguments passed to ``fn``.
        """
        symbols = (
            frozenset((symbols,))
            if isinstance(symbols, str)
            else frozenset(symbols)
        )
        if not symbols:
            raise ValueError("symbols cannot be empty.")
        for sym in symbols:
            for exec in self._executions:
                if sym in exec.symbols:
                    raise ValueError(
                        f"{sym} was already added to an execution."
                    )
        if models is not None:
            model_name_set: set[str] = set()
            for model in (
                (models,) if isinstance(models, ModelSource) else models
            ):
                if isinstance(model, TimeframeModelSource):
                    if not self._scope.has_model_source(model.base.name):
                        raise ValueError(
                            f"ModelSource {model.base.name!r} was not "
                            "registered."
                        )
                    base_source = self._scope.get_model_source(model.base.name)
                    if model.base is not base_source:
                        raise ValueError(
                            f"ModelSource {model.base.name!r} does not match "
                            "registered ModelSource."
                        )
                    if model.interval not in self._timeframes:
                        raise ValueError(
                            f"Timeframe {model.interval!r} was not declared with "
                            "set_timeframes()."
                        )
                    if model.base.pooled:
                        raise ValueError(
                            f"Pooled model {model.base.name!r} does not support "
                            ".timeframe()."
                        )
                    if isinstance(model.base, ModelLoader):
                        raise ValueError(
                            f"Pretrained model {model.base.name!r} does not "
                            "support .timeframe()."
                        )
                    model_name_set.add(model.name)
                elif isinstance(model, ModelSource):
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
                else:
                    raise TypeError(f"Invalid model type: {type(model)!r}.")
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
                if isinstance(ind, TimeframeIndicator):
                    if not self._scope.has_indicator(ind.base.name):
                        raise ValueError(
                            f"Indicator {ind.base.name!r} was not registered."
                        )
                    if ind.interval not in self._timeframes:
                        raise ValueError(
                            f"Timeframe {ind.interval!r} was not declared with "
                            "set_timeframes()."
                        )
                    ind_name_set.add(ind.name)
                elif isinstance(ind, Indicator):
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
                else:
                    raise TypeError(f"Invalid indicator type: {type(ind)!r}.")
            ind_names = frozenset(ind_name_set)
        else:
            ind_names = frozenset()
        self._execution_id += 1
        self._executions.add(
            Execution(
                id=self._execution_id,
                symbols=symbols,
                fn=fn,
                model_names=model_names,
                indicator_names=ind_names,
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

                An example timeframe string is ``1h 30m``.
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

                An example timeframe string is ``1h 30m``.
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
            unique_syms = {
                sym
                for execution in self._executions
                for sym in execution.symbols
            }
            timeframe_data = self._compress_timeframes(df, unique_syms)
            tf_seconds = to_seconds(timeframe)
            indicator_data = self._fetch_indicators(
                df=df,
                cache_date_fields=CacheDateFields(
                    start_date=start_dt,
                    end_date=end_dt,
                    tf_seconds=tf_seconds,
                    between_time=between_time,
                    days=day_ids,
                ),
                disable_parallel_indicators=disable_parallel_indicators,
                timeframe_data=timeframe_data,
            )
            train_only = (
                self._before_exec_fn is None
                and self._after_exec_fn is None
                and all(map(lambda e: e.fn is None, self._executions))
            )
            if portfolio is None:
                portfolio = Portfolio(
                    self._config.initial_cash,
                    self._config.fee_mode,
                    self._config.fee_amount,
                    self._fractional_shares_enabled(),
                    self._config.position_mode,
                    self._config.max_long_positions,
                    self._config.max_short_positions,
                    self._config.return_stops,
                )
            signals = self._run_walkforward(
                portfolio=portfolio,
                df=df,
                indicator_data=indicator_data,
                timeframe_data=timeframe_data,
                declared_timeframes=self._timeframes,
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

    def _fractional_shares_enabled(self):
        return self._config.enable_fractional_shares or isinstance(
            self._data_source, AlpacaCrypto
        )

    def _run_walkforward(
        self,
        portfolio: Portfolio,
        df: pd.DataFrame,
        indicator_data: dict[IndicatorSymbol, pd.Series],
        timeframe_data: TimeframeData,
        declared_timeframes: frozenset[TimeframeInterval],
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
    ) -> dict[str, pd.DataFrame]:
        sessions: dict[str, dict] = defaultdict(dict)
        exit_dates: dict[str, np.datetime64] = {}
        if self._config.exit_on_last_bar:
            exit_symbols: set[str] = set()
            for exec in self._executions:
                exit_symbols.update(exec.symbols)
            if exit_symbols and not df.empty:
                sym_col = DataCol.SYMBOL.value
                date_col = DataCol.DATE.value
                mask = df[sym_col].isin(exit_symbols)
                grouped = (
                    df.loc[mask].groupby(sym_col, sort=False)[date_col].max()
                )
                exit_dates = {
                    sym: np.datetime64(date) for sym, date in grouped.items()
                }
        signals: dict[str, pd.DataFrame] = {}
        for train_idx, test_idx in self.walkforward_split(
            df=df,
            windows=windows,
            lookahead=lookahead,
            train_size=train_size,
            shuffle=shuffle,
        ):
            models: dict[ModelSymbol, TrainedModel] = {}
            train_data = df.loc[train_idx]
            test_data = df.loc[test_idx]
            if not train_data.empty:
                train_symbols = set(train_data[DataCol.SYMBOL.value].unique())
                model_syms = {
                    ModelSymbol(model_name, sym)
                    for sym in train_symbols
                    for execution in self._executions
                    for model_name in execution.model_names
                    if sym in execution.symbols
                }
                pooled_model_groups: dict[tuple[str, int], frozenset[str]] = {}
                for execution in self._executions:
                    exec_syms = frozenset(
                        sym
                        for sym in execution.symbols
                        if sym in train_symbols
                    )
                    if not exec_syms:
                        continue
                    for model_name in execution.model_names:
                        base_name, token = parse_model_timeframe_name(
                            model_name
                        )
                        if token is not None:
                            continue
                        source = self._scope.get_model_source(base_name)
                        if isinstance(source, ModelTrainer) and source.pooled:
                            pooled_model_groups[(model_name, execution.id)] = (
                                exec_syms
                            )
                train_dates = get_unique_sorted_dates(
                    train_data[DataCol.DATE.value]
                )
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
                    timeframe_data=timeframe_data,
                )
            if test_data.empty:
                return signals
            window_history = (
                pd.concat([train_data, test_data], ignore_index=True)
                if not train_data.empty
                else test_data
            )
            sym_col = DataCol.SYMBOL.value
            date_col = DataCol.DATE.value
            history_col_scope = ColumnScope(
                window_history.set_index([sym_col, date_col]).sort_index()
            )
            split_signals = self.backtest_executions(
                config=self._config,
                executions=self._executions,
                before_exec_fn=self._before_exec_fn,
                after_exec_fn=self._after_exec_fn,
                sessions=sessions,
                models=models,
                indicator_data=indicator_data,
                timeframe_data=timeframe_data.slice_for_test(test_data),
                declared_timeframes=declared_timeframes,
                test_data=test_data,
                portfolio=portfolio,
                exit_dates=exit_dates,
                train_only=train_only,
                slippage_model=self._slippage_model,
                enable_fractional_shares=self._fractional_shares_enabled(),
                round_fill_price=self._config.round_fill_price,
                warmup=warmup,
                history_col_scope=history_col_scope,
            )
            for sym, signals_df in split_signals.items():
                if sym in signals:
                    signals[sym] = pd.concat([signals[sym], signals_df])
                else:
                    signals[sym] = signals_df
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
            df = _between(df, start_date, end_date).reset_index(drop=True)
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
        timeframe_data: Optional[TimeframeData] = None,
    ) -> dict[IndicatorSymbol, pd.Series]:
        indicator_syms = set()
        for execution in self._executions:
            for sym in execution.symbols:
                for model_name in execution.model_names:
                    base_name, token = parse_model_timeframe_name(model_name)
                    ind_names = self._scope.get_indicator_names(base_name)
                    for ind_name in ind_names:
                        if token is not None:
                            ind_name = indicator_timeframe_name(
                                ind_name, token
                            )
                        indicator_syms.add(IndicatorSymbol(ind_name, sym))
                for ind_name in execution.indicator_names:
                    indicator_syms.add(IndicatorSymbol(ind_name, sym))
        return self.compute_indicators(
            df=df,
            indicator_syms=indicator_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel_indicators=disable_parallel_indicators,
            timeframe_data=timeframe_data,
        )

    def _fetch_data(
        self, timeframe: str, adjust: Optional[Any]
    ) -> pd.DataFrame:
        unique_syms = {
            sym for execution in self._executions for sym in execution.symbols
        }
        if isinstance(self._data_source, DataSource):
            df = self._data_source.query(
                unique_syms,
                self._start_date,
                self._end_date,
                timeframe,
                adjust,
            )
        else:
            df = _between(self._data_source, self._start_date, self._end_date)
            df = df[df[DataCol.SYMBOL.value].isin(unique_syms)]
        if df.empty:
            raise ValueError("DataSource is empty.")
        return df.reset_index(drop=True)

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
            pos_df[col] = quantize(pos_df, col, self._config.round_test_result)
        pos_df.set_index(["symbol", "date"], inplace=True)
        portfolio_df = pd.DataFrame.from_records(
            portfolio.bars, columns=PortfolioBar._fields, index="date"
        )
        for col in (
            "cash",
            "equity",
            "margin",
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
            bootstrap_sample_size=self._config.bootstrap_sample_size,
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
