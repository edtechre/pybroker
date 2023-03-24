"""Contains implementation for backtesting trading strategies."""

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

import dataclasses
import numpy as np
import pandas as pd
from .cache import CacheDateFields
from .common import (
    BarData,
    DataCol,
    Day,
    ExecSymbol,
    IndicatorSymbol,
    ModelSymbol,
    PriceType,
    quantize,
    to_datetime,
    to_decimal,
    to_seconds,
    verify_data_source_columns,
    verify_date_range,
)
from .config import StrategyConfig
from .context import (
    ExecContext,
    ExecResult,
    PosSizeContext,
    set_exec_ctx_data,
    set_pos_size_ctx_data,
)
from .data import AlpacaCrypto, DataSource
from .eval import BootstrapResult, EvalMetrics, EvaluateMixin
from .indicator import Indicator, IndicatorsMixin
from .model import ModelSource, ModelsMixin, TrainedModel
from .portfolio import Order, Portfolio, PortfolioBar, PositionBar, Trade
from .scope import (
    ColumnScope,
    IndicatorScope,
    ModelInputScope,
    PendingOrderScope,
    PredictionScope,
    PriceScope,
    StaticScope,
)
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from numpy.typing import NDArray
from typing import (
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Union,
)


def _decorate_execution_fn(fn: Callable[[ExecContext], None]) -> Callable:
    def decorated_execution_fn(
        ctx: ExecContext, session: dict, symbol: str, date: np.datetime64
    ):
        set_exec_ctx_data(ctx, session, symbol, date)
        fn(ctx)
        if ctx.buy_shares is not None and ctx.sell_shares is not None:
            raise ValueError(
                "For each symbol, only one of buy_shares or sell_shares can be"
                " set per bar."
            )
        if ctx.buy_limit_price is not None and ctx.buy_shares is None:
            raise ValueError(
                "buy_shares must be set when buy_limit_price is set."
            )
        if ctx.sell_limit_price is not None and ctx.sell_shares is None:
            raise ValueError(
                "sell_shares must be set when sell_limit_price is set."
            )
        if not ctx.buy_shares and not ctx.sell_shares:
            return
        result = ctx.to_result()
        return result

    return decorated_execution_fn


def _between(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    if df.empty:
        return df
    return df[
        (df[DataCol.DATE.value].dt.tz_localize(None) >= start_date)
        & (df[DataCol.DATE.value].dt.tz_localize(None) <= end_date)
    ]


def _sort_by_score(result: ExecResult) -> float:
    return 0.0 if result.score is None else result.score


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
    """

    id: int
    symbols: frozenset[str]
    fn: Optional[Callable[[ExecContext], None]]
    model_names: frozenset[str]
    indicator_names: frozenset[str]


class BacktestMixin:
    """Mixin implementing backtesting functionality."""

    def backtest_executions(
        self,
        executions: set[Execution],
        sessions: Mapping[ExecSymbol, Mapping],
        models: Mapping[ModelSymbol, TrainedModel],
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        test_data: pd.DataFrame,
        portfolio: Portfolio,
        buy_delay: int,
        sell_delay: int,
        max_long_positions: Optional[int],
        max_short_positions: Optional[int],
        pos_size_handler: Optional[Callable[[PosSizeContext], None]],
        exit_dates: Mapping[str, np.datetime64],
        exit_buy_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ] = PriceType.MIDDLE,
        exit_sell_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ] = PriceType.MIDDLE,
        enable_fractional_shares: bool = False,
    ):
        r"""Backtests a ``set`` of :class:`.Execution`\ s that implement
        trading logic.

        Args:
            executions: :class:`.Execution`\ s to run.
            sessions: :class:`Mapping` of :class:`pybroker.common.ExecSymbol`
                pairs to :class:`Mapping` of custom data that persists for
                every bar during the :class:`.Execution`.
            models: :class:`Mapping` of :class:`pybroker.common.ModelSymbol`
                pairs to :class:`pybroker.common.TrainedModel`\ s.
            indicator_data: :class:`Mapping` of
                :class:`pybroker.common.IndicatorSymbol` pairs to
                :class:`pandas.Series` of :class:`pybroker.indicator.Indicator`
                values.
            test_data: :class:`pandas.DataFrame` of test data.
            portfolio: :class:`pybroker.portfolio.Portfolio`.
            buy_delay: Number of bars before placing an order for a buy signal.
            sell_delay: Number of bars before placing an order for a sell
                signal.
            max_long_positions: Maximum number of long positions that can be
                held at a time. If ``None``, then unlimited.
            max_short_positions: Maximum number of short positions that can be
                held at a time. If ``None``, then unlimited.
            pos_size_handler: :class:`Callable` that sets position sizes when
                placing orders for buy and sell signals.
            exit_dates: :class:`Mapping` of symbols to exit dates.
            exit_buy_fill_price: Fill price for covering an open short position
                for ``exit_dates``.
            exit_sell_fill_price: Fill price for selling an open long position
                for ``exit_dates``.
            enable_fractional_shares: Whether to enable trading fractional
                shares.

        Returns:
            :class:`.TestResult` of the backtest.
        """
        test_dates = test_data[DataCol.DATE.value].unique()
        test_dates.sort()
        exec_symbols = tuple(
            ExecSymbol(exec.id, sym)
            for sym in test_data[DataCol.SYMBOL.value].unique()
            for exec in executions
            if exec.fn is not None and sym in exec.symbols
        )
        test_data = (
            test_data.reset_index(drop=True)
            .set_index([DataCol.SYMBOL.value, DataCol.DATE.value])
            .sort_index()
        )
        test_symbols = frozenset(
            exec_symbol.symbol for exec_symbol in exec_symbols
        )
        sym_exec_dates = {
            sym: frozenset(test_data.loc[pd.IndexSlice[sym, :]].index.values)
            for sym in test_symbols
        }
        buy_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        sell_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        sym_end_index: dict[str, int] = defaultdict(int)
        col_scope = ColumnScope(test_data)
        price_scope = PriceScope(col_scope, sym_end_index)
        ind_scope = IndicatorScope(indicator_data, test_dates)
        input_scope = ModelInputScope(col_scope, ind_scope)
        pred_scope = PredictionScope(models, input_scope)
        pending_order_scope = PendingOrderScope()
        exec_ctx = ExecContext(
            portfolio=portfolio,
            col_scope=col_scope,
            ind_scope=ind_scope,
            input_scope=input_scope,
            pred_scope=pred_scope,
            pending_order_scope=pending_order_scope,
            models=models,
            sym_end_index=sym_end_index,
        )
        exec_fns = {
            exec.id: _decorate_execution_fn(exec.fn)
            for exec in executions
            if exec.fn is not None
        }
        if pos_size_handler is not None:
            pos_ctx = PosSizeContext(
                portfolio=portfolio,
                col_scope=col_scope,
                ind_scope=ind_scope,
                input_scope=input_scope,
                pred_scope=pred_scope,
                pending_order_scope=pending_order_scope,
                models=models,
                sym_end_index=sym_end_index,
                max_long_positions=max_long_positions,
                max_short_positions=max_short_positions,
            )
        logger = StaticScope.instance().logger
        logger.backtest_executions_start(test_dates)
        buy_results: deque[ExecResult] = deque()
        sell_results: deque[ExecResult] = deque()
        exit_syms: deque[str] = deque()
        for i, date in enumerate(test_dates):
            for sym in test_symbols:
                if date not in sym_exec_dates[sym]:
                    continue
                sym_end_index[sym] += 1
            is_buy_sched = date in buy_sched
            is_sell_sched = date in sell_sched
            if is_buy_sched and (
                max_long_positions is not None or pos_size_handler is not None
            ):
                buy_sched[date].sort(key=_sort_by_score, reverse=True)
            if is_sell_sched and (
                max_short_positions is not None or pos_size_handler is not None
            ):
                sell_sched[date].sort(key=_sort_by_score, reverse=True)
            if pos_size_handler is not None and (
                is_buy_sched or is_sell_sched
            ):
                self._set_pos_sizes(
                    pos_size_handler=pos_size_handler,
                    pos_ctx=pos_ctx,
                    buy_results=buy_sched[date] if is_buy_sched else None,
                    sell_results=sell_sched[date] if is_sell_sched else None,
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
            portfolio.check_stops(date, price_scope)
            portfolio.capture_bar(date, test_data)
            for exec_symbol in exec_symbols:
                if date not in sym_exec_dates[exec_symbol.symbol]:
                    continue
                if exit_dates and date == exit_dates[exec_symbol.symbol]:
                    exit_syms.append(exec_symbol.symbol)
                result = exec_fns[exec_symbol.exec_id](
                    exec_ctx, sessions[exec_symbol], exec_symbol.symbol, date
                )
                if result is None:
                    continue
                if result.buy_shares is not None:
                    buy_results.append(result)
                if result.sell_shares is not None:
                    sell_results.append(result)
            while buy_results:
                self._schedule_order(
                    result=buy_results.popleft(),
                    created=date,
                    sym_end_index=sym_end_index,
                    delay=buy_delay,
                    sched=buy_sched,
                    col_scope=col_scope,
                    pending_order_scope=pending_order_scope,
                )
            while sell_results:
                self._schedule_order(
                    result=sell_results.popleft(),
                    created=date,
                    sym_end_index=sym_end_index,
                    delay=sell_delay,
                    sched=sell_sched,
                    col_scope=col_scope,
                    pending_order_scope=pending_order_scope,
                )
            while exit_syms:
                self._exit_position(
                    portfolio=portfolio,
                    date=date,
                    symbol=exit_syms.popleft(),
                    exit_buy_fill_price=exit_buy_fill_price,
                    exit_sell_fill_price=exit_sell_fill_price,
                    price_scope=price_scope,
                )
            portfolio.incr_bars()
            if i % 10 == 0 or i == len(test_dates) - 1:
                logger.backtest_executions_loading(i + 1)

    def _exit_position(
        self,
        portfolio: Portfolio,
        date: np.datetime64,
        symbol: str,
        exit_buy_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ],
        exit_sell_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ],
        price_scope: PriceScope,
    ):
        buy_fill_price = price_scope.fetch(symbol, exit_buy_fill_price)
        sell_fill_price = price_scope.fetch(symbol, exit_sell_fill_price)
        portfolio.exit_position(
            date,
            symbol,
            buy_fill_price=buy_fill_price,
            sell_fill_price=sell_fill_price,
        )

    def _set_pos_sizes(
        self,
        pos_size_handler: Callable[[PosSizeContext], None],
        pos_ctx: PosSizeContext,
        buy_results: Optional[list[ExecResult]],
        sell_results: Optional[list[ExecResult]],
    ):
        set_pos_size_ctx_data(
            ctx=pos_ctx, buy_results=buy_results, sell_results=sell_results
        )
        pos_size_handler(pos_ctx)
        for id, shares in pos_ctx._signal_shares.items():
            if id < 0:
                raise ValueError(f"Invalid ExecSignal id: {id}")
            if buy_results is not None and sell_results is not None:
                if id >= (len(buy_results) + len(sell_results)):
                    raise ValueError(f"Invalid ExecSignal id: {id}")
                if id < len(buy_results):
                    buy_results[id].buy_shares = to_decimal(shares)
                else:
                    sell_results[
                        id - len(buy_results)
                    ].sell_shares = to_decimal(shares)
            elif buy_results is not None:
                if id >= len(buy_results):
                    raise ValueError(f"Invalid ExecSignal id: {id}")
                buy_results[id].buy_shares = to_decimal(shares)
            elif sell_results is not None:
                if id >= len(sell_results):
                    raise ValueError(f"Invalid ExecSignal id: {id}")
                sell_results[id].sell_shares = to_decimal(shares)
            else:
                raise ValueError(
                    "buy_results and sell_results cannot both be None."
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
            pending_order_scope.remove(result.pending_order_id)
            buy_shares = self._get_shares(
                result.buy_shares, enable_fractional_shares
            )
            fill_price = price_scope.fetch(
                result.symbol, result.buy_fill_price
            )
            order = portfolio.buy(
                date=date,
                symbol=result.symbol,
                shares=buy_shares,
                fill_price=fill_price,
                limit_price=result.buy_limit_price,
                stops=result.long_stops,
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
            pending_order_scope.remove(result.pending_order_id)
            sell_shares = self._get_shares(
                result.sell_shares, enable_fractional_shares
            )
            fill_price = price_scope.fetch(
                result.symbol, result.sell_fill_price
            )
            order = portfolio.sell(
                date=date,
                symbol=result.symbol,
                shares=sell_shares,
                fill_price=fill_price,
                limit_price=result.sell_limit_price,
                stops=result.short_stops,
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
        window_dates = df[date_col].unique()
        window_dates.sort()
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
                    test_idx = test_idx.index.to_numpy()
                    yield WalkforwardWindow(np.array(tuple()), test_idx)
                else:
                    train_idx = dates[
                        (dates[date_col] >= window_dates[start])
                        & (dates[date_col] <= window_dates[end - 1])
                    ]
                    train_idx = train_idx.index.to_numpy()
                    if shuffle:
                        np.random.shuffle(train_idx)
                    yield WalkforwardWindow(train_idx, np.array(tuple()))
        elif windows == 1:
            res = len(window_dates) - 1 - lookahead
            if res <= 0:
                raise ValueError(error_msg)
            train_length = int(res * train_size)
            test_length = int(res * (1 - train_size))
            train_start = 0
            train_end = train_length
            test_start = train_end + lookahead
            if test_start >= len(window_dates):
                raise ValueError(error_msg)
            test_end = test_start + test_length
            train_idx = dates[
                (dates[date_col] >= window_dates[train_start])
                & (dates[date_col] <= window_dates[train_end])
            ]
            test_idx = dates[
                (dates[date_col] >= window_dates[test_start])
                & (dates[date_col] <= window_dates[test_end])
            ]
            train_idx = train_idx.index.to_numpy()
            test_idx = test_idx.index.to_numpy()
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
                train_idx = train_idx.index.to_numpy()
                test_idx = test_idx.index.to_numpy()
                if shuffle:
                    np.random.shuffle(train_idx)
                yield WalkforwardWindow(train_idx, test_idx)


@dataclass(frozen=True)
class TestResult:
    """Contains the results of backtesting a :class:`.Strategy`.

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
        self._pos_size_handler: Optional[
            Callable[[PosSizeContext], None]
        ] = None
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

    def _verify_data_source(
        self, data_source: Union[DataSource, pd.DataFrame]
    ):
        if isinstance(data_source, pd.DataFrame):
            verify_data_source_columns(data_source)
        elif not isinstance(data_source, DataSource):
            raise TypeError(f"Invalid data_source type: {type(data_source)}")

    def add_execution(
        self,
        fn: Optional[Callable[[ExecContext], None]],
        symbols: Union[str, Iterable[str]],
        models: Optional[Union[ModelSource, Iterable[ModelSource]]] = None,
        indicators: Optional[Union[Indicator, Iterable[Indicator]]] = None,
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
        """
        symbols = (
            frozenset((symbols,))
            if type(symbols) == str
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
            for model in (
                (models,) if isinstance(models, ModelSource) else models
            ):
                if not self._scope.has_model_source(model.name):
                    raise ValueError(
                        f"ModelSource {model.name!r} was not registered."
                    )
                if model is not self._scope.get_model_source(model.name):
                    raise ValueError(
                        f"ModelSource {model.name!r} does not match "
                        "registered ModelSource."
                    )
        model_names = (
            (
                frozenset((models.name,))
                if isinstance(models, ModelSource)
                else frozenset(model.name for model in models)
            )
            if models is not None
            else frozenset()
        )
        if indicators is not None:
            for ind in (
                (indicators,)
                if isinstance(indicators, Indicator)
                else indicators
            ):
                if not self._scope.has_indicator(ind.name):
                    raise ValueError(
                        f"Indicator {ind.name!r} was not registered."
                    )
                if ind is not self._scope.get_indicator(ind.name):
                    raise ValueError(
                        f"Indicator {ind.name!r} does not match registered "
                        "Indicator."
                    )
        ind_names = (
            (
                frozenset((indicators.name,))
                if isinstance(indicators, Indicator)
                else frozenset(ind.name for ind in indicators)
            )
            if indicators is not None
            else frozenset()
        )
        self._execution_id += 1
        self._executions.add(
            Execution(
                id=self._execution_id,
                symbols=symbols,
                fn=fn,
                model_names=model_names,
                indicator_names=ind_names,
            )
        )

    def clear_executions(self):
        """Clears executions that were added with :meth:`.add_execution`."""
        self._executions.clear()

    def set_pos_size_handler(
        self, fn: Optional[Callable[[PosSizeContext], None]]
    ):
        r"""Sets a :class:`Callable` that determines position sizes to use for
        buy and sell signals.

        Args:
            fn: :class:`Callable` invoked before placing orders for buy and
                sell signals, and is passed a
                :class:`pybroker.context.PosSizeContext`.
        """
        self._pos_size_handler = fn

    def backtest(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = "",
        between_time: Optional[tuple[str, str]] = None,
        days: Optional[Union[str, Day, Iterable[Union[str, Day]]]] = None,
        lookahead: int = 1,
        train_size: int = 0,
        shuffle: bool = False,
        calc_bootstrap: bool = True,
        disable_parallel: bool = False,
    ) -> Optional[TestResult]:
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
                - ``"mo"``/``"month"``: months

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
                metrics. Defaults to ``True``.
            disable_parallel: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed
                serially. If ``False``, :class:`pybroker.indicator.Indicator`
                data is computed in parallel using multiple processes.
                Defaults to ``False``.

        Returns:
            :class:`.BacktestResult` containing portfolio balances, order
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
            disable_parallel=disable_parallel,
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
        calc_bootstrap: bool = True,
        disable_parallel: bool = False,
    ) -> Optional[TestResult]:
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
                - ``"mo"``/``"month"``: months

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
                metrics. Defaults to ``True``.
            disable_parallel: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed
                serially. If ``False``, :class:`pybroker.indicator.Indicator`
                data is computed in parallel using multiple processes.
                Defaults to ``False``.

        Returns:
            :class:`.BacktestResult` containing portfolio balances, order
            history, and evaluation metrics.
        """
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
            df = self._fetch_data(timeframe)
            day_ids = self._to_day_ids(days)
            df = self._filter_dates(
                df=df,
                start_date=start_dt,
                end_date=end_dt,
                between_time=between_time,
                days=day_ids,
            )
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
                disable_parallel=disable_parallel,
            )
            train_only = all(map(lambda e: e.fn is None, self._executions))
            portfolio = Portfolio(
                self._config.initial_cash,
                self._config.fee_mode,
                self._config.fee_amount,
                self._fractional_shares_enabled(),
                self._config.max_long_positions,
                self._config.max_short_positions,
            )
            self._run_walkforward(
                portfolio=portfolio,
                df=df,
                indicator_data=indicator_data,
                tf_seconds=tf_seconds,
                between_time=between_time,
                days=day_ids,
                windows=windows,
                lookahead=lookahead,
                train_size=train_size,
                shuffle=shuffle,
                train_only=train_only,
            )
            if train_only:
                self._logger.walkforward_completed()
                return None
            return self._to_test_result(
                start_dt, end_dt, portfolio, calc_bootstrap
            )
        finally:
            scope.unfreeze_data_cols()

    def _to_day_ids(
        self, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]
    ) -> Optional[tuple[int]]:
        if days is None:
            return None
        days = (days,) if type(days) == str or type(days) == Day else days
        return tuple(
            sorted(
                day.value
                if type(day) == Day
                else Day[day.upper()].value  # type: ignore[union-attr]
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
        tf_seconds: int,
        between_time: Optional[tuple[str, str]],
        days: Optional[tuple[int]],
        windows: int,
        lookahead: int,
        train_size: float,
        shuffle: bool,
        train_only: bool,
    ):
        sessions: dict[ExecSymbol, dict] = {}
        exit_dates: dict[str, np.datetime64] = {}
        for exec in self._executions:
            if exec.fn is None:
                continue
            for sym in exec.symbols:
                sessions[ExecSymbol(exec.id, sym)] = {}
                if self._config.exit_on_last_bar:
                    sym_dates = df[df[DataCol.SYMBOL.value] == sym][
                        DataCol.DATE.value
                    ].values
                    if len(sym_dates):
                        sym_dates.sort()
                        exit_dates[sym] = sym_dates[-1]
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
                model_syms = {
                    ModelSymbol(model_name, sym)
                    for sym in train_data[DataCol.SYMBOL.value].unique()
                    for execution in self._executions
                    for model_name in execution.model_names
                    if sym in execution.symbols
                }
                train_dates = train_data[DataCol.DATE.value].unique()
                train_dates.sort()
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
                )
            if not train_only and not test_data.empty:
                self.backtest_executions(
                    executions=self._executions,
                    sessions=sessions,
                    models=models,
                    indicator_data=indicator_data,
                    test_data=test_data,
                    portfolio=portfolio,
                    buy_delay=self._config.buy_delay,
                    sell_delay=self._config.sell_delay,
                    max_long_positions=self._config.max_long_positions,
                    max_short_positions=self._config.max_short_positions,
                    pos_size_handler=self._pos_size_handler,
                    exit_dates=exit_dates,
                    enable_fractional_shares=self._fractional_shares_enabled(),
                    exit_buy_fill_price=self._config.exit_cover_fill_price,
                    exit_sell_fill_price=self._config.exit_sell_fill_price,
                )

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
        disable_parallel: bool,
    ) -> dict[IndicatorSymbol, pd.Series]:
        indicator_syms = set()
        for execution in self._executions:
            for sym in execution.symbols:
                for model_name in execution.model_names:
                    ind_names = self._scope.get_indicator_names(model_name)
                    for ind_name in ind_names:
                        indicator_syms.add(IndicatorSymbol(ind_name, sym))
                for ind_name in execution.indicator_names:
                    indicator_syms.add(IndicatorSymbol(ind_name, sym))
        return self.compute_indicators(
            df=df,
            indicator_syms=indicator_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel=disable_parallel,
        )

    def _fetch_data(self, timeframe: str) -> pd.DataFrame:
        unique_syms = {
            sym for execution in self._executions for sym in execution.symbols
        }
        if isinstance(self._data_source, DataSource):
            df = self._data_source.query(
                unique_syms, self._start_date, self._end_date, timeframe
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
    ) -> TestResult:
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
            pos_df[col] = quantize(pos_df, col)
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
            "fees",
        ):
            portfolio_df[col] = quantize(portfolio_df, col)
        orders_df = pd.DataFrame.from_records(
            portfolio.orders, columns=Order._fields, index="id"
        )
        for col in ("limit_price", "fill_price", "fees"):
            orders_df[col] = quantize(orders_df, col)
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
        ):
            trades_df[col] = quantize(trades_df, col)
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
        )
        metrics_df = pd.DataFrame(
            dataclasses.asdict(eval_result.metrics).items(),
            columns=["name", "value"],
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
        )
