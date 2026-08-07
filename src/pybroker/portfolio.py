"""Contains portfolio related functionality, such as portfolio metrics and
placing orders.
"""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import itertools
import math
import numpy as np
from pybroker.common import (
    BarData,
    DataCol,
    FeeInfo,
    FeeMode,
    OrderType,
    PositionIntent,
    PositionMode,
    PriceType,
    StopType,
    to_decimal,
)
from pybroker.scope import (
    ColumnScope,
    IndicatorScope,
    PriceScope,
    StaticScope,
)
from collections import deque
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import (
    TYPE_CHECKING,
    Callable,
    Final,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Union,
    cast,
)

if TYPE_CHECKING:
    from pybroker.slippage import SlippageModel

_BarStopFillPrice = Union[
    int,
    float,
    np.floating,
    Decimal,
    PriceType,
    Callable[[str, BarData], Union[int, float, Decimal]],
]

_DECIMAL_100: Final = Decimal(100)

# Scale that clamped fractional share counts are rounded down to. Without this,
# _clamp_shares returns a full-precision quotient and ``pos.shares += shares``
# rounds at the default Decimal context, leaving pos.shares out of step with
# sum(entry.shares) and stranding dust positions on a full exit.
_SHARES_EPSILON: Final = Decimal("1E-9")

# Cached column name strings. DataCol.X.value goes through the enum descriptor
# which shows up in profiling when hit per-bar-per-symbol; binding once at
# module load keeps the hot loop free of enum __get__ calls.
_COL_DATE: Final = DataCol.DATE.value
_COL_CLOSE: Final = DataCol.CLOSE.value
_COL_LOW: Final = DataCol.LOW.value
_COL_HIGH: Final = DataCol.HIGH.value
_CAPTURE_BAR_COLS: Final = (_COL_DATE, _COL_CLOSE, _COL_LOW, _COL_HIGH)


class Stop(NamedTuple):
    """Contains information about a stop set on :class:`.Entry`.

    Attributes:
        id: Unique identifier.
        symbol: Symbol of the stop.
        stop_type: :class:`.StopType`.
        pos_type: Type of  :class:`.Position`, either ``long`` or ``short``.
        percent: Percent from entry price.
        points: Cash amount from entry price.
        bars: Number of bars after which to trigger the stop.
        fill_price: Price that the stop will be filled at.
        limit_price: Limit price to use for the stop.
        exit_price: Exit :class:`pybroker.common.PriceType` to use for the
            stop exit. If set, the stop is checked against the ``exit_price``
            and exits at the ``exit_price`` when triggered.
    """

    id: int
    symbol: str
    stop_type: StopType
    pos_type: Literal["long", "short"]
    percent: Optional[Decimal]
    points: Optional[Decimal]
    bars: Optional[int]
    fill_price: Optional[
        Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ]
    ]
    limit_price: Optional[Decimal]
    exit_price: Optional[PriceType]


class StopRecord(NamedTuple):
    """Records per-bar data about a stop.

    Attributes:
        date: Date of the bar.
        symbol: Symbol of the stop.
        stop_id: Unique identifier.
        stop_type: :class:`.StopType`.
        pos_type: Type of  :class:`.Position`, either ``long`` or ``short``.
        curr_value: Current value of the stop.
        curr_bars: Current bars of the stop.
        percent: Percent from entry price.
        points: Cash amount from entry price.
        bars: Number of bars after which to trigger the stop.
        fill_price: Price that the stop will be filled at.
        limit_price: Limit price to use for the stop.
        exit_price: Exit :class:`pybroker.common.PriceType` to use for the
            stop exit. If set, the stop is checked against the ``exit_price``
            and exits at the ``exit_price`` when triggered.
    """

    date: np.datetime64
    symbol: str
    stop_id: int
    stop_type: str
    pos_type: Literal["long", "short"]
    curr_value: Optional[Decimal]
    curr_bars: Optional[int]
    percent: Optional[Decimal]
    points: Optional[Decimal]
    bars: Optional[int]
    fill_price: Optional[Decimal]
    limit_price: Optional[Decimal]
    exit_price: Optional[PriceType]


@dataclass
class Entry:
    """Contains information about an entry into a :class:`.Position`.

    Attributes:
        id: Unique identifier.
        date: Date of the entry.
        symbol: Symbol of the entry.
        shares: Number of shares.
        price: Share price of the entry.
        type: Type of  :class:`.Position`, either ``long`` or ``short``.
        bars: Current number of bars since entry, counted on the portfolio's
            clock: every bar of the backtest, whether or not ``symbol`` traded
            on it. Feeds :attr:`Trade.bars` and ``pnl_per_bar``, which measure
            elapsed holding time.
        sym_bars: Current number of ``symbol``'s own bars since entry. Advances
            only on dates ``symbol`` has a bar, so it matches what
            :attr:`pybroker.context.ExecContext.bars` reports. This is what
            :attr:`pybroker.common.StopType.BAR` stops count against: when
            calendars are ragged, ``bars`` would expire a ``hold_bars`` stop
            after fewer of the symbol's own bars than the caller asked for.
        stops: Stops set on the entry.
        mae: Maximum adverse excursion (MAE).
        mfe: Maximum favorable excursion (MFE).
    """

    id: int
    date: np.datetime64
    symbol: str
    shares: Decimal
    price: Decimal
    type: Literal["long", "short"]
    bars: int = field(default=0)
    sym_bars: int = field(default=0)
    stops: list[Stop] = field(default_factory=list)
    mae: Decimal = field(default_factory=Decimal)
    mfe: Decimal = field(default_factory=Decimal)


@dataclass
class _StopData:
    value: float
    stop: Stop
    entry: Entry


@dataclass
class Position:
    r"""Contains information about an open position in ``symbol``.

    Attributes:
        symbol: Ticker symbol of the position.
        shares: Number of shares.
        type: Type of position, either ``long`` or ``short``.
        close: Last close price of ``symbol``.
        equity: Equity in the position.
        market_value: Market value of position.
        margin: Amount of margin in position.
        pnl: Unrealized profit and loss (PnL).
        entries: ``deque`` of position :class:`.Entry`\ s sorted in ascending
            chronological order.
        bars: Current number of bars since entry, on the portfolio's clock.
        entry_notional: Total cost basis of the position's open entries.
        unmarked_shares: Shares acquired since the last mark. Valuing these at
            :attr:`.close` would price them at a mark taken before they were
            bought, so they are held at cost until the next mark.
        unmarked_notional: Cost of :attr:`.unmarked_shares`, at fill price.
    """

    symbol: str
    shares: Decimal
    type: Literal["long", "short"]
    close: Decimal = field(default_factory=Decimal)
    equity: Decimal = field(default_factory=Decimal)
    market_value: Decimal = field(default_factory=Decimal)
    margin: Decimal = field(default_factory=Decimal)
    pnl: Decimal = field(default_factory=Decimal)
    entries: deque[Entry] = field(default_factory=deque)
    bars: int = field(default=0)
    entry_notional: Decimal = field(default_factory=Decimal)
    unmarked_shares: Decimal = field(default_factory=Decimal)
    unmarked_notional: Decimal = field(default_factory=Decimal)

    def _marked_value(self) -> Decimal:
        """Returns the position's value at the last mark.

        Shares bought since that mark are held at cost, so a fill moves this
        by its own notional. Valuing them at :attr:`.close` instead would move
        it by the stale mark, which is what buying power is charged against.
        """
        # is_finite() first: Decimal, unlike float, raises InvalidOperation on
        # any ordered comparison against NaN, so a single NaN close aborts the
        # whole backtest from deep inside buying-power arithmetic, naming
        # neither the symbol nor the column.
        if not self.close.is_finite() or self.close <= 0:
            return self.entry_notional
        marked_shares = self.shares - self.unmarked_shares
        if marked_shares <= 0:
            return self.unmarked_notional
        return marked_shares * self.close + self.unmarked_notional

    def _add_unmarked(self, shares: Decimal, fill_price: Decimal):
        self.unmarked_shares += shares
        self.unmarked_notional += shares * fill_price

    def _clamp_unmarked(self):
        """Drops unmarked shares that have since been exited.

        Entries exit oldest-first, and the unmarked ones are the newest, so
        they only shrink once every marked share is gone. The surviving
        notional is then read back off the newest entries rather than scaled
        proportionally: a proportional rescale prices the survivors at the
        average cost of every unmarked fill, which is not what FIFO left
        behind when those fills had different prices.
        """
        if self.unmarked_shares <= 0:
            return
        if self.shares <= 0:
            self.unmarked_shares = Decimal()
            self.unmarked_notional = Decimal()
        elif self.unmarked_shares > self.shares:
            self.unmarked_shares = self.shares
            remaining = self.shares
            notional = Decimal()
            for entry in reversed(self.entries):
                if remaining <= 0:
                    break
                taken = min(remaining, entry.shares)
                notional += taken * entry.price
                remaining -= taken
            self.unmarked_notional = notional

    def _clear_unmarked(self):
        self.unmarked_shares = Decimal()
        self.unmarked_notional = Decimal()


class Trade(NamedTuple):
    """Holds information about a completed trade (entry and exit).

    Attributes:
        id: Unique identifier.
        type: Type of trade, either ``long`` or ``short``.
        symbol: Ticker symbol of the trade.
        entry_date: Entry date.
        exit_date: Exit date.
        entry: Entry price.
        exit: Exit price.
        shares: Number of shares.
        pnl: Profit and loss (PnL).
        return_pct: Return measured in percentage.
        agg_pnl: Aggregate profit and loss (PnL) of the strategy after
            the trade.
        bars: Number of bars the trade was held.
        pnl_per_bar: Profit and loss (PnL) per bar held.
        stop: Type of stop that was triggered, if any.
        mae: Maximum adverse excursion (MAE).
        mfe: Maximum favorable excursion (MFE).
    """

    id: int
    type: Literal["long", "short"]
    symbol: str
    entry_date: np.datetime64
    exit_date: np.datetime64
    entry: Decimal
    exit: Decimal
    shares: Decimal
    pnl: Decimal
    return_pct: Decimal
    agg_pnl: Decimal
    bars: int
    pnl_per_bar: Decimal
    stop: Optional[Literal["bar", "loss", "profit", "trailing"]]
    mae: Decimal
    mfe: Decimal


class Order(NamedTuple):
    """Holds information about a filled order.

    Attributes:
        id: Unique identifier.
        type: Type of order, either ``buy`` or ``sell``.
        symbol: Ticker symbol of the order.
        date: Date the order was filled.
        created: Date the order signal was created, or ``None``
            for stop-triggered orders.
        order_type: How the order originated, either ``market``,
            ``limit``, ``stop_bar``, ``stop_loss``,
            ``stop_profit``, or ``stop_trailing``.
        intent: Position intent, either ``buy_to_open``,
            ``buy_to_close``, ``sell_to_open``, or
            ``sell_to_close``.
        shares: Number of shares bought or sold.
        limit_price: Limit price that was used for the order.
        market_price: Market price prevailing when the order was filled,
            before any fill-time slippage adjustment. Equal to
            ``fill_price`` when no slippage model is set.
        fill_price: Price that the order was filled at.
        fees: Brokerage fees for order.
    """

    id: int
    type: Literal["buy", "sell"]
    symbol: str
    date: np.datetime64
    created: Optional[np.datetime64]
    order_type: Literal[
        "market",
        "limit",
        "stop_bar",
        "stop_loss",
        "stop_profit",
        "stop_trailing",
    ]
    intent: Literal[
        "buy_to_open",
        "buy_to_close",
        "sell_to_open",
        "sell_to_close",
    ]
    shares: Decimal
    limit_price: Optional[Decimal]
    market_price: Decimal
    fill_price: Decimal
    fees: Decimal


class PortfolioBar(NamedTuple):
    """Snapshot of :class:`.Portfolio` state, captured per bar.

    Attributes:
        date: Date of bar.
        cash: Available cash in :class:`.Portfolio`.
        equity: Amount of equity in :class:`.Portfolio`. Open short positions
            are held at cost, so their unrealized PnL is excluded.
        notional: Notional exposure of all open positions at mark: the market
            value of open long positions plus the ``margin`` of open short
            positions.
        margin: Notional exposure of open short positions at mark.
        margin_loan: Borrowed funds used for leveraged long and short
            positions.
        net_cash_balance: ``cash - margin_loan``.
        market_value: Market value of :class:`.Portfolio`, equal to ``equity``
            plus the unrealized PnL of all open short positions.
        pnl: Realized profit and loss (PnL) of :class:`.Portfolio`.
        unrealized_pnl: Unrealized profit and loss (PnL) of
            :class:`.Portfolio`, equal to ``market_value - equity``.
        fees: Brokerage fees.
    """

    date: np.datetime64
    cash: Decimal
    equity: Decimal
    notional: Decimal
    margin: Decimal
    margin_loan: Decimal
    net_cash_balance: Decimal
    market_value: Decimal
    pnl: Decimal
    unrealized_pnl: Decimal
    fees: Decimal


class PositionBar(NamedTuple):
    r"""Snapshot of an open :class:`.Position`\ 's state, captured per bar.

    Attributes:
        symbol: Ticker symbol of :class:`.Position`.
        date: Date of bar.
        long_shares: Number of shares long in :class:`.Position`.
        short_shares: Number of shares short in :class:`.Position`.
        close: Last close price of ``symbol``.
        equity: Amount of equity in :class:`.Position`.
        market_value: Market value of :class:`.Position`.
        margin: Amount of margin in :class:`.Position`.
        unrealized_pnl: Unrealized profit and loss (PnL) of :class:`.Position`.
    """

    symbol: str
    date: np.datetime64
    long_shares: Decimal
    short_shares: Decimal
    close: Decimal
    equity: Decimal
    market_value: Decimal
    margin: Decimal
    unrealized_pnl: Decimal


class _OrderResult(NamedTuple):
    filled_shares: Decimal
    rem_shares: Decimal


def _calculate_pnl_mae_mfe(
    pos: Position,
    close_d: Decimal,
    low: Optional[float],
    high: Optional[float],
):
    if pos.type != "long" and pos.type != "short":
        raise ValueError(f"Unknown position type: {pos.type}")
    low_d = to_decimal(low) if low is not None else None
    high_d = to_decimal(high) if high is not None else None
    pnl = Decimal()
    for entry in pos.entries:
        if pos.type == "long":
            pnl += (close_d - entry.price) * entry.shares
            loss_d = low_d - entry.price if low_d is not None else None
            profit_d = high_d - entry.price if high_d is not None else None
        else:
            pnl += (entry.price - close_d) * entry.shares
            loss_d = entry.price - high_d if high_d is not None else None
            profit_d = entry.price - low_d if low_d is not None else None
        # The update decisions compare float(exact Decimal difference),
        # never raw float subtraction: the two can differ in the last ulp,
        # which would flip a knife-edge update and change recorded MAE/MFE.
        if loss_d is not None:
            loss = float(loss_d)
            if loss < 0 and loss < float(entry.mae):
                entry.mae = loss_d
        if profit_d is not None:
            profit = float(profit_d)
            if profit > 0 and profit > float(entry.mfe):
                entry.mfe = profit_d
    pos.pnl = pnl


class Portfolio:
    r"""Class representing a portfolio of holdings. The portfolio contains
    information about open positions and balances, and is also used to place
    buy and sell orders.

    Args:
        cash: Starting cash balance.
        fee_mode: Brokerage fee mode.
        fee_amount: Brokerage fee amount.
        enable_fractional_shares: Whether to enable trading fractional shares.
        position_mode: Position mode for :class:`.Portfolio`.
        max_long_positions: Maximum number of long :class:`.Position`\ s that
            can be held at a time. If ``None``, then unlimited.
        max_short_positions: Maximum number of short :class:`.Position`\ s that
            can be held at a time. If ``None``, then unlimited.
        record_stops: Whether to record stop data per-bar.

    Attributes:
        cash: Current cash balance.
        equity: Current amount of equity, defined as the net cash balance plus
            the market value of all open long positions plus the collateral
            posted for all open short positions. Short positions are held at
            cost, so their unrealized PnL is excluded.
        market_value: Current market value. The market value is defined as
            :attr:`.equity` added together with the unrealized PnL of all open
            short positions.
        fees: Current brokerage fees.
        fee_amount: Brokerage fee amount.
        enable_fractional_shares: Whether to enable trading fractional shares.
        orders: ``deque`` of all filled orders, sorted in ascending
            chronological order.
        margin: Notional exposure of open short positions at mark.
        margin_loan: Borrowed funds used for leveraged long and short
            positions.
        pnl: Realized profit and loss (PnL).
        long_positions: ``dict`` mapping ticker symbols to open long
            :class:`.Position`\ s.
        short_positions: ``dict`` mapping ticker symbols to open short
            :class:`.Position`\ s.
        symbols: Ticker symbols of all currently open positions.
        bars: ``deque`` of snapshots of :class:`.Portfolio` state on every bar,
            sorted in ascending chronological order.
        position_bars: ``deque`` of snapshots of :class:`.Position` states on
            every bar, sorted in ascending chronological order.
        win_rate: Running win rate of trades.
        loss_rate: Running loss rate of trades.
    """

    def __init__(
        self,
        cash: float,
        fee_mode: Optional[
            Union[FeeMode, Callable[[FeeInfo], Decimal], None]
        ] = None,
        fee_amount: Optional[float] = None,
        enable_fractional_shares: bool = False,
        position_mode: PositionMode = PositionMode.DEFAULT,
        max_long_positions: Optional[int] = None,
        max_short_positions: Optional[int] = None,
        record_stops: Optional[bool] = False,
        leverage: float = 1.0,
        interest_rate: float = 0.0,
        bars_per_year: Optional[int] = None,
        record_portfolio_bars: bool = False,
        record_position_bars: bool = False,
    ):
        self.cash: Decimal = to_decimal(cash)
        self._initial_market_value = self.cash
        self._fee_mode = fee_mode
        self._fee_amount: Optional[Decimal] = (
            None if fee_amount is None else to_decimal(fee_amount)
        )
        self._enable_fractional_shares = enable_fractional_shares
        self._position_mode = position_mode
        self.equity: Decimal = self.cash
        self.market_value: Decimal = self.cash
        self.fees = Decimal()
        self._max_long_positions = max_long_positions
        self._max_short_positions = max_short_positions
        self._record_stops = record_stops
        self._record_portfolio_bars = record_portfolio_bars
        self._record_position_bars = record_position_bars
        self._leverage = leverage
        self._interest_rate = interest_rate
        self._bars_per_year = bars_per_year
        self.orders: deque[Order] = deque()
        self.trades: deque[Trade] = deque()
        self.margin: Decimal = Decimal()
        self._accrued_interest: Decimal = Decimal()
        self.pnl: Decimal = Decimal()
        self.long_positions: dict[str, Position] = {}
        self.short_positions: dict[str, Position] = {}
        self.symbols: set[str] = set()
        self.bars: deque[PortfolioBar] = deque()
        self.position_bars: deque[PositionBar] = deque()
        self._metrics_bars: list[PortfolioBar] = []
        self._wins: Decimal = Decimal()
        self._cached_long_mv: Optional[Decimal] = None
        self._cached_short_mv: Optional[Decimal] = None
        self._logger = StaticScope.instance().logger
        self._stop_data: dict[int, _StopData] = {}
        self._active_stops: dict[str, list[_StopData]] = {}
        self._order_id: int = 0
        self._entry_id: int = 0
        self._trade_id: int = 0
        self._stop_records: list[StopRecord] = []
        # Running sums of Position.entry_notional, kept in step with the four
        # sites that mutate it. They keep margin_loan and _live_market_value
        # O(1); both are hit on every fill via _clamp_shares, where an O(n)
        # pass over open positions is measurable.
        self._long_entry_notional: Decimal = Decimal()
        self._short_entry_notional: Decimal = Decimal()

    @property
    def margin_loan(self) -> Decimal:
        """Borrowed funds used for leveraged long and short positions.

        Derived from the entry notional of all open positions, so it is always
        consistent with the open book and can never go negative.
        """
        if self._leverage <= 1:
            return self._accrued_interest
        leverage = to_decimal(self._leverage)
        notional = self._long_entry_notional + self._short_entry_notional
        return notional - notional / leverage + self._accrued_interest

    @property
    def win_rate(self) -> Decimal:
        if not self.trades:
            return Decimal()
        return self._wins / len(self.trades)

    @property
    def loss_rate(self) -> Decimal:
        if not self.trades:
            return Decimal()
        return Decimal(1) - self.win_rate

    def _calculate_fees(
        self,
        symbol: str,
        fill_price: Decimal,
        shares: Decimal,
        order_type: Literal["buy", "sell"],
    ) -> Decimal:
        fees = Decimal()
        if self._fee_mode is None or self._fee_amount is None:
            return fees
        if callable(self._fee_mode):
            fees = to_decimal(
                self._fee_mode(
                    FeeInfo(
                        symbol=symbol,
                        shares=shares,
                        fill_price=fill_price,
                        order_type=order_type,
                    )
                )
            )
        elif self._fee_mode == FeeMode.ORDER_PERCENT:
            fees = self._fee_amount / _DECIMAL_100 * fill_price * shares
        elif self._fee_mode == FeeMode.PER_ORDER:
            fees = self._fee_amount
        elif self._fee_mode == FeeMode.PER_SHARE:
            fees = self._fee_amount * shares
        else:
            raise ValueError(f"Unknown FeeMode: {self._fee_mode!r}")
        return fees

    def _verify_input(
        self,
        shares: Union[int, float, Decimal],
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ):
        if shares < 0:
            raise ValueError(f"Shares cannot be negative: {shares}")
        if fill_price <= 0:
            raise ValueError(f"Fill price must be > 0: {fill_price}")
        if limit_price is not None and limit_price <= 0:
            raise ValueError(f"Limit price must be > 0: {limit_price}")

    def _add_entry(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        price: Decimal,
        type: Literal["long", "short"],
        pos: Position,
    ) -> Entry:
        self._entry_id += 1
        entry = Entry(
            id=self._entry_id,
            symbol=symbol,
            shares=shares,
            price=price,
            date=date,
            type=type,
        )
        pos.entries.append(entry)
        return entry

    def _add_order(
        self,
        date: np.datetime64,
        symbol: str,
        type: Literal["buy", "sell"],
        created: Optional[np.datetime64],
        order_type: OrderType,
        intent: PositionIntent,
        shares: Decimal,
        limit_price: Optional[Decimal],
        market_price: Optional[Decimal],
        fill_price: Decimal,
    ) -> Order:
        self._order_id += 1
        fees = self._calculate_fees(symbol, fill_price, shares, type)
        order = Order(
            id=self._order_id,
            type=type,
            symbol=symbol,
            date=date,
            created=created,
            order_type=order_type.value,
            intent=intent.value,
            shares=shares,
            limit_price=limit_price,
            market_price=market_price
            if market_price is not None
            else fill_price,
            fill_price=fill_price,
            fees=fees,
        )
        self.orders.append(order)
        self.fees += fees
        if self._fee_mode is not None:
            self.cash -= fees
        return order

    def _add_trade(
        self,
        type: Literal["long", "short"],
        symbol: str,
        entry_date: np.datetime64,
        exit_date: np.datetime64,
        entry_price: Decimal,
        exit_price: Decimal,
        shares: Decimal,
        pnl: Decimal,
        return_pct: Decimal,
        agg_pnl: Decimal,
        bars: int,
        pnl_per_bar: Decimal,
        stop_type: Optional[StopType],
        mae: Decimal,
        mfe: Decimal,
    ):
        self._trade_id += 1
        trade = Trade(
            id=self._trade_id,
            type=type,
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date,
            entry=entry_price,
            exit=exit_price,
            shares=shares,
            pnl=pnl,
            return_pct=return_pct,
            agg_pnl=agg_pnl,
            bars=bars,
            pnl_per_bar=pnl_per_bar,
            stop=None if stop_type is None else stop_type.value,
            mae=mae,
            mfe=mfe,
        )
        self.trades.append(trade)
        if pnl > 0:
            self._wins += 1

    def _invalidate_mv_cache(self):
        self._cached_long_mv = None
        self._cached_short_mv = None

    def _get_stop_amount_f(self, stop: Stop, price: float) -> float:
        if stop.percent is not None:
            return price * float(stop.percent) / 100.0
        if stop.points is not None:
            return float(stop.points)
        raise ValueError("Stop amount not set.")

    def _get_stop_amount(self, stop: Stop, price: Decimal) -> Decimal:
        return to_decimal(self._get_stop_amount_f(stop, float(price)))

    def _add_stops(self, entry: Entry, stops: Iterable[Stop]):
        # Stops arrive as a frozenset, whose iteration order depends on the
        # hash of every member. Stop always carries None fields, and hash(None)
        # is address-derived, so the order varies run to run even under a fixed
        # PYTHONHASHSEED and no seed can pin it. Sorting by the monotonic stop
        # id makes evaluation order, and therefore which of two stops that hit
        # on the same bar wins, reproducible.
        for stop in sorted(stops, key=lambda s: s.id):
            if stop.id in self._stop_data:
                raise ValueError(f"Duplicate stop ID: {stop.id}")
            entry.stops.append(stop)
            if stop.stop_type == StopType.BAR:
                continue
            amount = self._get_stop_amount_f(stop, float(entry.price))
            entry_price = float(entry.price)
            if (
                stop.pos_type == "long" and stop.stop_type == StopType.PROFIT
            ) or (
                stop.pos_type == "short"
                and (
                    stop.stop_type == StopType.LOSS
                    or stop.stop_type == StopType.TRAILING
                )
            ):
                stop_value = entry_price + amount
            else:
                stop_value = entry_price - amount
            stop_data = _StopData(value=stop_value, stop=stop, entry=entry)
            self._stop_data[stop.id] = stop_data
            self._active_stops.setdefault(stop.symbol, []).append(stop_data)

    def _remove_stop_data(self, entry: Entry):
        for stop in entry.stops:
            if stop.id not in self._stop_data:
                continue
            stop_data = self._stop_data.pop(stop.id)
            sym_stops = self._active_stops.get(stop.symbol)
            if sym_stops is not None:
                try:
                    sym_stops.remove(stop_data)
                except ValueError:
                    pass
                if not sym_stops:
                    del self._active_stops[stop.symbol]

    def _long_market_value(self) -> Decimal:
        if self._cached_long_mv is not None:
            return self._cached_long_mv
        total = Decimal()
        for pos in self.long_positions.values():
            total += pos._marked_value()
        self._cached_long_mv = total
        return total

    def _short_market_value(self) -> Decimal:
        if self._cached_short_mv is not None:
            return self._cached_short_mv
        total = Decimal()
        for pos in self.short_positions.values():
            total += pos._marked_value()
        self._cached_short_mv = total
        return total

    def _post_collateral(self, notional: Decimal):
        """Debits the collateral required to open ``notional`` of exposure.

        The borrowed remainder is not tracked here: :attr:`.margin_loan` is
        derived from the open positions' entry notional.
        """
        if self._leverage > 1:
            self.cash -= notional / to_decimal(self._leverage)
        else:
            self.cash -= notional
        self._invalidate_mv_cache()

    def _release_collateral(self, entry_notional: Decimal, pnl: Decimal):
        """Credits back the collateral posted for ``entry_notional``, plus
        realized ``pnl``.

        Must be called *before* the position's ``entry_notional`` is
        decremented, since :attr:`.margin_loan` is derived from it.
        """
        if self._leverage > 1:
            self.cash += entry_notional / to_decimal(self._leverage) + pnl
        else:
            self.cash += entry_notional + pnl

    def _net_cash_balance(self) -> Decimal:
        return self.cash - self.margin_loan

    def _live_market_value(self) -> Decimal:
        """Market value of the account computed from current state.

        :attr:`.market_value` is only refreshed in :meth:`.capture_bar`, which
        runs *after* every fill in the bar, so it cannot be used to size an
        order: it would value the account as it stood before the exits that
        happened earlier in the same bar.

        Shorts contribute ``entry_notional + pnl``, expressed here as
        ``2 * entry_notional - marked`` because ``pnl`` is only recomputed in
        :meth:`.capture_bar` and so is stale after a partial cover, while
        :meth:`._short_market_value` tracks live share counts.

        Positions open since the last mark are still valued at that mark, so
        this trails the true value by at most one bar's price move.
        """
        return (
            self.cash
            - self.margin_loan
            + self._long_market_value()
            + 2 * self._short_entry_notional
            - self._short_market_value()
        )

    def _available_buying_power(self) -> Decimal:
        # A cash account is deliberately capped at settled cash and is blind
        # to the mark on open positions, matching StrategyConfig.leverage's
        # documented "no borrowing" model: an adverse mark cannot be enforced
        # without a margin call, which is not modeled. Note this makes the
        # function discontinuous at leverage exactly 1 -- with a short open,
        # leverage 1.0 and 1.0000001 can report very different numbers.
        if self._leverage <= 1:
            return max(self.cash, Decimal())
        leverage = to_decimal(self._leverage)
        committed = self._long_market_value() + self._short_market_value()
        return max(self._live_market_value() * leverage - committed, Decimal())

    def _sweep_accrued_interest(self):
        """Repays accrued margin interest out of any available cash.

        Leaves the net cash balance unchanged; this only keeps ``cash`` and
        :attr:`.margin_loan` from both carrying a balance that nets out.
        """
        if self._accrued_interest <= 0 or self.cash <= 0:
            return
        payment = min(self.cash, self._accrued_interest)
        self.cash -= payment
        self._accrued_interest -= payment

    def _apply_interest(self):
        if self._interest_rate <= 0 or self._bars_per_year is None:
            return
        self._sweep_accrued_interest()
        net_cash = self._net_cash_balance()
        if net_cash == 0:
            return
        per_bar_rate = (
            to_decimal(self._interest_rate)
            / _DECIMAL_100
            / Decimal(self._bars_per_year)
        )
        interest = abs(net_cash) * per_bar_rate
        if net_cash < 0:
            self._accrued_interest += interest
        else:
            self.cash += interest

    def _clamp_shares(
        self,
        symbol: str,
        fill_price: Decimal,
        shares: Decimal,
        order_type: Literal["buy", "sell"],
    ) -> Decimal:
        buying_power = self._available_buying_power()
        if self._leverage <= 1 and self.cash < 0:
            return Decimal()
        if self._leverage > 1:
            # Fees are paid out of cash, so they reduce equity and therefore
            # cost ``leverage`` times their amount in buying power. Reserving
            # that here keeps the filled order inside the configured leverage.
            #
            # Reserve for the order actually being placed, not for the largest
            # order buying power could support. Under a per-share fee the
            # latter grows with the share count, so a cheap symbol makes the
            # reservation exceed buying power outright and drops an order that
            # costs a few dollars. Fees are non-decreasing in shares, so a
            # reservation taken at the requested count still covers the fill.
            #
            # The reservation is taken once and never re-derived. Recomputing
            # it at the reduced count would yield a *smaller* fee, and so a
            # larger budget, undoing the reservation and letting the fill spill
            # past the leverage cap.
            leverage = to_decimal(self._leverage)
            affordable = min(shares, buying_power / fill_price)
            fees = self._calculate_fees(
                symbol, fill_price, affordable, order_type
            )
            if fees > 0:
                buying_power -= fees * leverage
                if buying_power <= 0:
                    return Decimal()
        max_shares = (
            Decimal(buying_power / fill_price).quantize(
                _SHARES_EPSILON, rounding=ROUND_DOWN
            )
            if self._enable_fractional_shares
            else Decimal(buying_power // fill_price)
        )
        return min(shares, max_shares)

    def buy(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal] = None,
        stops: Optional[Iterable[Stop]] = None,
        created: Optional[np.datetime64] = None,
        order_type: OrderType = OrderType.MARKET,
        market_price: Optional[Decimal] = None,
    ) -> Optional[Order]:
        r"""Places a buy order.

        Args:
            date: Date when the :class:`.Order` is placed.
            symbol: Ticker symbol to buy.
            shares: Number of shares to buy.
            fill_price: If filled, the price used to fill the :class:`.Order`.
            limit_price: Limit price of the :class:`.Order`.
            stops: :class:`.Stop`\ s to set on the :class:`.Entry` created from
                the :class:`.Order`, if filled.
            created: Date the order signal was created.
            order_type: How the order originated.
            market_price: Market price at fill time, before fill-time
                slippage. Defaults to ``fill_price``.

        Returns:
            :class:`.Order` if the order was filled, otherwise ``None``.
        """
        self._verify_input(shares, fill_price, limit_price)
        self._logger.debug_place_buy_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        if limit_price is not None and limit_price < fill_price:
            return None
        if shares == 0:
            return None
        covered = self._cover(date, symbol, shares, fill_price)
        bought_shares = self._long(
            date, symbol, covered.rem_shares, fill_price, limit_price, stops
        )
        if not covered.filled_shares and not bought_shares:
            return None
        if bought_shares:
            intent = PositionIntent.BUY_TO_OPEN
        else:
            intent = PositionIntent.BUY_TO_CLOSE
        order = self._add_order(
            date=date,
            symbol=symbol,
            type="buy",
            created=created,
            order_type=order_type,
            intent=intent,
            shares=covered.filled_shares + bought_shares,
            limit_price=limit_price,
            market_price=market_price,
            fill_price=fill_price,
        )
        return order

    def _cover(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
    ) -> _OrderResult:
        if symbol not in self.short_positions:
            return _OrderResult(Decimal(), shares)
        rem_shares = shares
        if rem_shares <= 0:
            return _OrderResult(Decimal(), shares)
        pos = self.short_positions[symbol]
        # See _sell_existing: stop at zero so no phantom Trade is booked.
        while pos.entries and rem_shares > 0:
            entry = pos.entries[0]
            if rem_shares >= entry.shares:
                rem_shares -= entry.shares
                self._exit_short(
                    date, pos, entry, entry.shares, fill_price, stop_type=None
                )
                self._remove_stop_data(entry)
                pos.entries.popleft()
            else:
                self._exit_short(
                    date, pos, entry, rem_shares, fill_price, stop_type=None
                )
                rem_shares = Decimal()
                break
        self._update_position(pos)
        return _OrderResult(shares - rem_shares, rem_shares)

    def _exit_short(
        self,
        date: np.datetime64,
        pos: Position,
        entry: Entry,
        shares: Decimal,
        fill_price: Decimal,
        stop_type: Optional[StopType],
    ):
        order_amount = shares * fill_price
        entry_amount = shares * entry.price
        entry_pnl = entry_amount - order_amount
        self.pnl += entry_pnl
        self._release_collateral(entry_amount, entry_pnl)
        pos.shares -= shares
        entry.shares -= shares
        pos.entry_notional -= entry_amount
        self._short_entry_notional -= entry_amount
        pos._clamp_unmarked()
        pnl_per_bar = entry_pnl if not entry.bars else entry_pnl / entry.bars
        return_pct = ((entry.price / fill_price) - 1) * 100
        pnl = entry.price - fill_price
        mae = pnl if pnl < 0 and pnl < entry.mae else entry.mae
        mfe = pnl if pnl > 0 and pnl > entry.mfe else entry.mfe
        self._add_trade(
            type=entry.type,
            symbol=entry.symbol,
            entry_date=entry.date,
            exit_date=date,
            entry_price=entry.price,
            exit_price=fill_price,
            shares=shares,
            pnl=entry_pnl,
            return_pct=return_pct,
            agg_pnl=self.pnl,
            bars=entry.bars,
            pnl_per_bar=pnl_per_bar,
            stop_type=stop_type,
            mae=mae,
            mfe=mfe,
        )
        self._invalidate_mv_cache()

    def _long(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal],
        stops: Optional[Iterable[Stop]],
    ) -> Decimal:
        if self._position_mode == PositionMode.SHORT_ONLY:
            return Decimal()
        clamped_shares = self._clamp_shares(symbol, fill_price, shares, "buy")
        if clamped_shares < shares:
            self._logger.debug_buy_shares_exceed_cash(
                date=date,
                symbol=symbol,
                shares=shares,
                fill_price=fill_price,
                limit_price=limit_price,
                cash=self.cash,
                clamped_shares=clamped_shares,
            )
            shares = clamped_shares
        if shares <= 0:
            return Decimal()
        if (
            self._max_long_positions is not None
            and symbol not in self.long_positions
            # Not ``==``: the cap can be lowered mid-run (a walkforward
            # optimization retunes it per window) while more positions are
            # already held, and an equality test would stop binding entirely.
            and len(self.long_positions) >= self._max_long_positions
        ):
            # Logged because the order simply vanishes: no Order row, no
            # warning, and nothing in the result to show the cap bound.
            self._logger.debug_position_limit_reached(
                symbol,
                "long",
                len(self.long_positions),
                self._max_long_positions,
            )
            return Decimal()
        order_amount = shares * fill_price
        self._post_collateral(order_amount)
        if symbol not in self.long_positions:
            self.symbols.add(symbol)
            pos = Position(symbol=symbol, shares=shares, type="long")
            self.long_positions[symbol] = pos
        else:
            pos = self.long_positions[symbol]
            pos.shares += shares
        pos._add_unmarked(shares, fill_price)
        entry = self._add_entry(
            date=date,
            symbol=symbol,
            shares=shares,
            price=fill_price,
            type="long",
            pos=pos,
        )
        if stops is not None:
            self._add_stops(entry, stops)
        pos.entry_notional += shares * fill_price
        self._long_entry_notional += shares * fill_price
        self._invalidate_mv_cache()
        return shares

    def sell(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal] = None,
        stops: Optional[Iterable[Stop]] = None,
        created: Optional[np.datetime64] = None,
        order_type: OrderType = OrderType.MARKET,
        market_price: Optional[Decimal] = None,
    ) -> Optional[Order]:
        r"""Places a sell order.

        Args:
            date: Date when the :class:`.Order` is placed.
            symbol: Ticker symbol to sell.
            shares: Number of shares to sell.
            fill_price: If filled, the price used to fill the :class:`.Order`.
            limit_price: Limit price of the :class:`.Order`.
            stops: :class:`.Stop`\ s to set on the :class:`.Entry` created from
                the :class:`.Order`, if filled.
            created: Date the order signal was created.
            order_type: How the order originated.
            market_price: Market price at fill time, before fill-time
                slippage. Defaults to ``fill_price``.

        Returns:
            :class:`.Order` if the order was filled, otherwise ``None``.
        """
        self._verify_input(shares, fill_price, limit_price)
        self._logger.debug_place_sell_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        if limit_price is not None and limit_price > fill_price:
            return None
        if shares == 0:
            return None
        sold = self._sell_existing(date, symbol, shares, fill_price)
        short_shares = self._short(
            date, symbol, sold.rem_shares, fill_price, stops
        )
        if not sold.filled_shares and not short_shares:
            return None
        if short_shares:
            intent = PositionIntent.SELL_TO_OPEN
        else:
            intent = PositionIntent.SELL_TO_CLOSE
        order = self._add_order(
            date=date,
            symbol=symbol,
            type="sell",
            created=created,
            order_type=order_type,
            intent=intent,
            shares=sold.filled_shares + short_shares,
            limit_price=limit_price,
            market_price=market_price,
            fill_price=fill_price,
        )
        return order

    def _sell_existing(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
    ) -> _OrderResult:
        if symbol not in self.long_positions:
            return _OrderResult(Decimal(), shares)
        rem_shares = shares
        pos = self.long_positions[symbol]
        # Stop once the order is filled. Re-entering with rem_shares == 0 would
        # book a zero-share Trade against the next entry, inflating trade_count
        # and skewing avg_pnl.
        while pos.entries and rem_shares > 0:
            entry = pos.entries[0]
            if rem_shares >= entry.shares:
                rem_shares -= entry.shares
                self._exit_long(
                    date, pos, entry, entry.shares, fill_price, stop_type=None
                )
                self._remove_stop_data(entry)
                pos.entries.popleft()
            else:
                self._exit_long(
                    date, pos, entry, rem_shares, fill_price, stop_type=None
                )
                rem_shares = Decimal()
                break
        self._update_position(pos)
        return _OrderResult(shares - rem_shares, rem_shares)

    def _exit_long(
        self,
        date: np.datetime64,
        pos: Position,
        entry: Entry,
        shares: Decimal,
        fill_price: Decimal,
        stop_type: Optional[StopType],
    ):
        order_amount = shares * fill_price
        entry_amount = shares * entry.price
        entry_pnl = order_amount - entry_amount
        self.pnl += entry_pnl
        self._release_collateral(entry_amount, entry_pnl)
        pos.shares -= shares
        entry.shares -= shares
        pos.entry_notional -= entry_amount
        self._long_entry_notional -= entry_amount
        pos._clamp_unmarked()
        pnl_per_bar = entry_pnl if not entry.bars else entry_pnl / entry.bars
        return_pct = ((fill_price / entry.price) - 1) * 100
        pnl = fill_price - entry.price
        mae = pnl if pnl < 0 and pnl < entry.mae else entry.mae
        mfe = pnl if pnl > 0 and pnl > entry.mfe else entry.mfe
        self._add_trade(
            type=entry.type,
            symbol=entry.symbol,
            entry_date=entry.date,
            exit_date=date,
            entry_price=entry.price,
            exit_price=fill_price,
            shares=shares,
            pnl=entry_pnl,
            return_pct=return_pct,
            agg_pnl=self.pnl,
            bars=entry.bars,
            pnl_per_bar=pnl_per_bar,
            stop_type=stop_type,
            mae=mae,
            mfe=mfe,
        )
        self._invalidate_mv_cache()

    def _update_position(self, pos: Position):
        if pos.entries:
            return
        if pos.type == "long":
            if pos.symbol in self.long_positions:
                del self.long_positions[pos.symbol]
        else:
            if pos.symbol in self.short_positions:
                del self.short_positions[pos.symbol]
        if (
            pos.symbol in self.symbols
            and pos.symbol not in self.long_positions
            and pos.symbol not in self.short_positions
        ):
            self.symbols.remove(pos.symbol)

    def _short(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        stops: Optional[Iterable[Stop]],
    ) -> Decimal:
        if shares <= 0:
            return Decimal()
        if (
            self._max_short_positions is not None
            and symbol not in self.short_positions
            # See _long: the cap can be lowered while positions are held.
            and len(self.short_positions) >= self._max_short_positions
        ):
            # See _long: a cap-discarded order leaves no trace otherwise.
            self._logger.debug_position_limit_reached(
                symbol,
                "short",
                len(self.short_positions),
                self._max_short_positions,
            )
            return Decimal()
        if self._position_mode == PositionMode.LONG_ONLY:
            return Decimal()
        clamped_shares = self._clamp_shares(symbol, fill_price, shares, "sell")
        if clamped_shares < shares:
            self._logger.debug_buy_shares_exceed_cash(
                date=date,
                symbol=symbol,
                shares=shares,
                fill_price=fill_price,
                limit_price=None,
                cash=self.cash,
                clamped_shares=clamped_shares,
            )
            shares = clamped_shares
        if shares <= 0:
            return Decimal()
        order_amount = shares * fill_price
        self._post_collateral(order_amount)
        if symbol not in self.short_positions:
            self.symbols.add(symbol)
            pos = Position(symbol=symbol, shares=shares, type="short")
            self.short_positions[symbol] = pos
        else:
            pos = self.short_positions[symbol]
            pos.shares += shares
        pos._add_unmarked(shares, fill_price)
        entry = self._add_entry(
            date=date,
            symbol=symbol,
            shares=shares,
            price=fill_price,
            type="short",
            pos=pos,
        )
        if stops is not None:
            self._add_stops(entry, stops)
        pos.entry_notional += shares * fill_price
        self._short_entry_notional += shares * fill_price
        self._invalidate_mv_cache()
        return shares

    def exit_position(
        self,
        date: np.datetime64,
        symbol: str,
        buy_fill_price: Decimal,
        sell_fill_price: Decimal,
        col_scope: Optional[ColumnScope] = None,
        ind_scope: Optional[IndicatorScope] = None,
        sym_end_index: Optional[Mapping[str, int]] = None,
        slippage_model: Optional["SlippageModel"] = None,
    ):
        """Exits any long and short positions for ``symbol`` at
        ``buy_fill_price`` and ``sell_fill_price``.

        When ``slippage_model`` is set, both fills are adjusted by it. Share
        adjustments are ignored because the positions are exited in full. The
        unadjusted prices are recorded as ``market_price`` on the resulting
        :class:`.Order`\\ s.
        """
        if symbol in self.long_positions:
            long_shares = self.long_positions[symbol].shares
            fill_price = sell_fill_price
            if slippage_model is not None:
                _, fill_price = slippage_model.adjust_fill(
                    side="sell",
                    symbol=symbol,
                    shares=long_shares,
                    fill_price=fill_price,
                    col_scope=col_scope,
                    ind_scope=ind_scope,
                    sym_end_index=sym_end_index,
                    enable_fractional_shares=self._enable_fractional_shares,
                )
            self.sell(
                date=date,
                symbol=symbol,
                shares=long_shares,
                fill_price=fill_price,
                market_price=sell_fill_price,
            )
        if symbol in self.short_positions:
            short_shares = self.short_positions[symbol].shares
            fill_price = buy_fill_price
            if slippage_model is not None:
                _, fill_price = slippage_model.adjust_fill(
                    side="buy",
                    symbol=symbol,
                    shares=short_shares,
                    fill_price=fill_price,
                    col_scope=col_scope,
                    ind_scope=ind_scope,
                    sym_end_index=sym_end_index,
                    enable_fractional_shares=self._enable_fractional_shares,
                )
            self.buy(
                date=date,
                symbol=symbol,
                shares=short_shares,
                fill_price=fill_price,
                market_price=buy_fill_price,
            )

    def capture_bar(
        self,
        date: np.datetime64,
        col_scope: ColumnScope,
        sym_end_index: Mapping[str, int],
        price_scope: Optional[PriceScope] = None,
    ):
        """Captures portfolio state of the current bar."""
        self._apply_interest()
        cash_f = float(self.cash)
        margin_loan_f = float(self.margin_loan)
        net_cash = cash_f - margin_loan_f
        # Accumulated into lists and summed with math.fsum rather than with
        # ``+=``. Float addition is not associative, and self.symbols is a set
        # whose iteration order varies with PYTHONHASHSEED, so a running total
        # made equity, market_value and every metric derived from them differ
        # between otherwise identical runs -- and ExecContext.calc_target_shares
        # sizes orders off equity, so share counts diverged too. fsum is
        # correctly rounded, hence identical for any permutation.
        equity_parts = [net_cash]
        market_value_parts = [net_cash]
        margin_parts: list[float] = []
        long_value_parts: list[float] = []
        # Sorted so position_bars -- and therefore TestResult.positions row
        # order -- does not depend on set iteration order either.
        for sym in sorted(self.symbols):
            close_f = low_f = high_f = None
            if price_scope is not None:
                close_f, low_f, high_f = price_scope.fetch_bar_ohlc(sym, date)
            else:
                idx = sym_end_index.get(sym, 0) - 1
                if idx >= 0:
                    cols = col_scope.fetch_dict(sym, _CAPTURE_BAR_COLS)
                    date_arr = cols[_COL_DATE]
                    if (
                        date_arr is not None
                        and idx < len(date_arr)
                        and date_arr[idx] == date
                    ):
                        close_arr = cols[_COL_CLOSE]
                        low_arr = cols[_COL_LOW]
                        high_arr = cols[_COL_HIGH]
                        if close_arr is not None:
                            close_f = float(close_arr[idx])
                        if low_arr is not None:
                            low_f = float(low_arr[idx])
                        if high_arr is not None:
                            high_f = float(high_arr[idx])
            if close_f is not None and not math.isfinite(close_f):
                # A halted or vendor-gapped bar. Treated as no bar at all:
                # marking against NaN poisons this bar's equity and
                # market_value, and every metric derived from them, while
                # reporting a total_return of 0.
                close_f = low_f = high_f = None
            close_d = to_decimal(close_f) if close_f is not None else None
            pos_long_shares = Decimal()
            pos_short_shares = Decimal()
            pos_equity = Decimal()
            pos_market_value = Decimal()
            pos_margin = Decimal()
            pos_pnl = Decimal()
            if sym in self.long_positions:
                pos = self.long_positions[sym]
                if close_d is not None:
                    _calculate_pnl_mae_mfe(
                        pos, close_d, low=low_f, high=high_f
                    )
                    pos.equity = pos.shares * close_d
                    pos.market_value = pos.equity
                    pos.close = close_d
                    # Every share is now marked at this close.
                    pos._clear_unmarked()
                    pos_long_shares += pos.shares
                    pos_equity += pos.equity
                    pos_market_value += pos.market_value
                    pos_pnl += pos.pnl
                    equity_parts.append(float(pos.equity))
                    market_value_parts.append(float(pos.equity))
                    long_value_parts.append(float(pos.equity))
                else:
                    # No bar for this symbol on this date, so hold the
                    # position at its last known mark, falling back to cost if
                    # it has never been marked. This matches
                    # _long_market_value, so the snapshot taken here and
                    # _live_market_value agree by construction.
                    held = pos._marked_value()
                    equity_parts.append(float(held))
                    market_value_parts.append(float(held))
                    long_value_parts.append(float(held))
            if sym in self.short_positions:
                pos = self.short_positions[sym]
                entry_notional = pos.entry_notional
                # Shorting posts collateral out of cash, so the collateral is
                # added back here. Equity holds it at cost and market value
                # marks it to market with the position's unrealized PnL.
                equity_parts.append(float(entry_notional))
                if close_d is not None:
                    _calculate_pnl_mae_mfe(
                        pos, close_d, low=low_f, high=high_f
                    )
                    pos.close = close_d
                    pos.margin = close_d * pos.shares
                    pos.market_value = pos.margin + pos.pnl
                    # Every share is now marked at this close.
                    pos._clear_unmarked()
                    pos_margin += pos.margin
                    pos_short_shares += pos.shares
                    pos_equity += entry_notional + pos.pnl
                    pos_market_value += pos.market_value
                    pos_pnl += pos.pnl
                    market_value_parts.append(float(entry_notional + pos.pnl))
                else:
                    # As above: hold at the last known mark. A short's value
                    # is ``entry_notional + pnl``, which equals
                    # ``2 * entry_notional - marked``.
                    marked = pos._marked_value()
                    market_value_parts.append(
                        float(2 * entry_notional - marked)
                    )
                margin_parts.append(float(pos.margin))
            if close_d is not None and self._record_position_bars:
                self.position_bars.append(
                    PositionBar(
                        symbol=sym,
                        date=date,
                        long_shares=pos_long_shares,
                        short_shares=pos_short_shares,
                        close=close_d,
                        equity=pos_equity,
                        market_value=pos_market_value,
                        margin=pos_margin,
                        unrealized_pnl=pos_pnl,
                    )
                )

        self.equity = to_decimal(math.fsum(equity_parts))
        self.market_value = to_decimal(math.fsum(market_value_parts))
        self.margin = to_decimal(math.fsum(margin_parts))
        notional = to_decimal(math.fsum(long_value_parts)) + self.margin
        self._cached_long_mv = None
        self._cached_short_mv = None

        net_cash_balance = self._net_cash_balance()
        bar = PortfolioBar(
            date=date,
            cash=self.cash,
            equity=self.equity,
            notional=notional,
            market_value=self.market_value,
            margin=self.margin,
            margin_loan=self.margin_loan,
            net_cash_balance=net_cash_balance,
            pnl=self.equity - self._initial_market_value,
            unrealized_pnl=self.market_value - self.equity,
            fees=self.fees,
        )
        self._metrics_bars.append(bar)
        if self._record_portfolio_bars:
            self.bars.append(bar)

    def incr_bars(
        self,
        date: Optional[np.datetime64] = None,
        price_scope: Optional[PriceScope] = None,
    ):
        """Increments the number of bars held by every trade entry.

        :attr:`Entry.bars` advances on every bar of the backtest. When ``date``
        and ``price_scope`` are given, :attr:`Entry.sym_bars` advances only on
        dates the entry's symbol has a bar of its own, which is what
        :attr:`pybroker.common.StopType.BAR` stops count against. Without them
        both counters advance together, matching a calendar where every symbol
        trades on every bar.
        """
        for pos in itertools.chain(
            self.long_positions.values(), self.short_positions.values()
        ):
            pos.bars += 1
            has_bar = (
                True
                if price_scope is None or date is None
                else price_scope.has_bar_on(pos.symbol, date)
            )
            for entry in pos.entries:
                entry.bars += 1
                if has_bar:
                    entry.sym_bars += 1

    def remove_stop(self, stop_id: int) -> bool:
        """Removes a :class:`.Stop` with ``stop_id``."""
        if stop_id in self._stop_data:
            stop_data = self._stop_data.pop(stop_id)
            sym_stops = self._active_stops.get(stop_data.stop.symbol)
            if sym_stops is not None:
                try:
                    sym_stops.remove(stop_data)
                except ValueError:
                    pass
                if not sym_stops:
                    del self._active_stops[stop_data.stop.symbol]
            if stop_data.stop in stop_data.entry.stops:
                stop_data.entry.stops.remove(stop_data.stop)
            return True
        for pos in itertools.chain(
            self.long_positions.values(), self.short_positions.values()
        ):
            for entry in pos.entries:
                for stop in entry.stops:
                    if stop.id == stop_id:
                        entry.stops.remove(stop)
                        return True
        return False

    def remove_stops(
        self,
        val: Union[str, Position, Entry],
        stop_type: Optional[StopType] = None,
    ):
        r"""Removes :class:`.Stop`\ s.

        Args:
            val: Ticker symbol, :class:`.Position`, or :class:`.Entry` for
                which to cancel stops.
            stop_type: :class:`pybroker.common.StopType`.
        """
        if isinstance(val, str):
            if val in self.long_positions:
                self._remove_position_stops(
                    self.long_positions[val], stop_type
                )
            if val in self.short_positions:
                self._remove_position_stops(
                    self.short_positions[val], stop_type
                )
        elif isinstance(val, Position):
            self._remove_position_stops(val, stop_type)
        elif isinstance(val, Entry):
            self._remove_entry_stops(val, stop_type)

    def _remove_position_stops(
        self, pos: Position, stop_type: Optional[StopType]
    ):
        for entry in pos.entries:
            self._remove_entry_stops(entry, stop_type)

    def _remove_entry_stops(self, entry: Entry, stop_type: Optional[StopType]):
        if stop_type is None:
            self._remove_stop_data(entry)
            entry.stops.clear()
        else:
            stop_id = None
            for stop in entry.stops:
                if stop.stop_type == stop_type:
                    stop_id = stop.id
                    break
            if stop_id is not None:
                self.remove_stop(stop_id)

    def check_stops(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        col_scope: Optional[ColumnScope] = None,
        sym_end_index: Optional[Mapping[str, int]] = None,
        ind_scope: Optional[IndicatorScope] = None,
        slippage_model: Optional["SlippageModel"] = None,
    ):
        """Checks whether stops are triggered.

        Price stops on one entry are evaluated in ascending :attr:`Stop.id`
        order and the first one to trigger exits the entry. Ids are assigned
        when :meth:`pybroker.context.ExecContext.to_result` builds the stops,
        in a fixed order -- loss, then profit, then trailing -- not in the
        order the attributes were assigned on the context, so precedence is
        deterministic but not caller-controlled.

        :attr:`pybroker.common.StopType.BAR` stops are evaluated after the
        price stops, whatever their id, because a bar-count exit and a price
        exit landing on the same bar have no true ordering: the bar's price
        path decides, and it is not modeled here.

        When ``slippage_model`` is set, triggered stops fill at the adjusted
        price. Slippage never affects whether a stop triggers, and share
        adjustments are ignored because a stop exits its entry in full.
        """
        price_scope.reset_bar()
        executed: deque[tuple[Position, Entry]] = deque()
        triggered_entry_ids: set[int] = set()
        # One verdict per symbol per bar, shared by both loops below: the
        # check itself is memoized reads, but the call volume -- two loops
        # over every symbol holding a stop, every bar -- is hot-path.
        priceable: dict[str, bool] = {}
        for sym, sym_stops in self._active_stops.items():
            can_price = priceable.get(sym)
            if can_price is None:
                can_price = self._can_price_stops(price_scope, sym, date)
                priceable[sym] = can_price
            if not can_price:
                # The symbol has no bar on this date, so its stops cannot be
                # evaluated. Gated on the date rather than on has_bar(): a
                # symbol whose calendar skips this date keeps its cursor on an
                # earlier bar, so has_bar() is True and the stop would trigger
                # and fill against that stale bar's OHLC -- including the
                # entry bar's own extremes, which check_stops never sees
                # because it runs before the fill. has_bar_on() is False for
                # end_index <= 0, so this still covers the symbol that stopped
                # trading and would otherwise raise mid-backtest.
                continue
            for stop_data in sym_stops:
                stop = stop_data.stop
                entry = stop_data.entry
                if entry.id in triggered_entry_ids:
                    continue
                if stop.pos_type == "long":
                    pos = self.long_positions.get(sym)
                else:
                    pos = self.short_positions.get(sym)
                if pos is None:
                    continue
                triggered, fill_price = self._trigger_stop(
                    date,
                    price_scope,
                    pos,
                    entry,
                    stop,
                    col_scope,
                    sym_end_index,
                    ind_scope,
                    slippage_model,
                )
                if self._record_stops:
                    self._capture_stop(date, entry, stop, fill_price)
                if triggered:
                    executed.append((pos, entry))
                    triggered_entry_ids.add(entry.id)

        for pos in itertools.chain(
            self.long_positions.values(), self.short_positions.values()
        ):
            can_price = priceable.get(pos.symbol)
            if can_price is None:
                can_price = self._can_price_stops(
                    price_scope, pos.symbol, date
                )
                priceable[pos.symbol] = can_price
            if not can_price:
                continue
            for entry in pos.entries:
                # A preempted stop never executes, so it is not recorded --
                # matching the loop above, which skips triggered entries
                # outright. Recording it would emit a real, slippage-adjusted
                # fill price for a stop that did not fire.
                if entry.id in triggered_entry_ids:
                    continue
                for stop in entry.stops:
                    if stop.stop_type != StopType.BAR:
                        continue
                    triggered, fill_price = self._trigger_stop(
                        date,
                        price_scope,
                        pos,
                        entry,
                        stop,
                        col_scope,
                        sym_end_index,
                        ind_scope,
                        slippage_model,
                    )
                    if self._record_stops:
                        self._capture_stop(date, entry, stop, fill_price)
                    if triggered:
                        executed.append((pos, entry))
                        triggered_entry_ids.add(entry.id)
                        break

        for pos, entry in executed:
            if pos.entries and pos.entries[0] is entry:
                pos.entries.popleft()
            else:
                pos.entries.remove(entry)
            self._remove_stop_data(entry)
            self._update_position(pos)

    @staticmethod
    def _can_price_stops(
        price_scope: PriceScope, symbol: str, date: np.datetime64
    ) -> bool:
        """Returns whether ``symbol``'s bar on ``date`` can price a stop.

        False when the symbol has no bar on ``date`` -- a ragged calendar --
        and also when the bar's close is not finite, mirroring how
        :meth:`.capture_bar` treats a halted or vendor-gapped bar as no bar
        at all. Price stops are NaN-safe by comparison semantics, but a BAR
        stop resolves a fill price from the bar and aborted the run, and a
        trailing stop's ratchet would poison its stop value with NaN --
        ``max(nan, x)`` is ``nan`` -- disarming it for the rest of the run.
        """
        if not price_scope.has_bar_on(symbol, date):
            return False
        close_f, _, _ = price_scope.fetch_bar_ohlc(symbol, date)
        # None means the frame has no close column at all -- minimal frames
        # are legitimate here -- and only a close that exists but is not
        # finite marks the bar unpriceable.
        return close_f is None or math.isfinite(close_f)

    def _capture_stop(
        self,
        date: np.datetime64,
        entry: Entry,
        stop: Stop,
        fill_price: Optional[Decimal],
    ):
        stop_record = StopRecord(
            date=date,
            stop_id=stop.id,
            symbol=stop.symbol,
            stop_type=stop.stop_type.value,
            pos_type=stop.pos_type,
            curr_value=(
                to_decimal(self._stop_data[stop.id].value)
                if stop.id in self._stop_data
                else None
            ),
            curr_bars=(
                entry.sym_bars if stop.stop_type == StopType.BAR else None
            ),
            bars=stop.bars,
            percent=stop.percent,
            points=stop.points,
            limit_price=stop.limit_price,
            exit_price=stop.exit_price,
            fill_price=fill_price,
        )
        self._stop_records.append(stop_record)

    def _slip_stop_fill_price(
        self,
        stop: Stop,
        entry: Entry,
        fill_price: Optional[Decimal],
        col_scope: Optional[ColumnScope],
        ind_scope: Optional[IndicatorScope],
        sym_end_index: Optional[Mapping[str, int]],
        slippage_model: Optional["SlippageModel"],
    ) -> Optional[Decimal]:
        """Returns ``fill_price`` adjusted by ``slippage_model``.

        Share adjustments are discarded: a stop exits its entry in full.

        A model that returns a non-positive price is ignored rather than
        trusted. Slippage may only ever worsen a fill, and a price at or below
        zero would pay the account to sell, silently corrupting cash.
        """
        if slippage_model is None or fill_price is None:
            return fill_price
        _, slipped_price = slippage_model.adjust_fill(
            side="sell" if stop.pos_type == "long" else "buy",
            symbol=stop.symbol,
            shares=entry.shares,
            fill_price=fill_price,
            col_scope=col_scope,
            ind_scope=ind_scope,
            sym_end_index=sym_end_index,
            enable_fractional_shares=self._enable_fractional_shares,
        )
        if slipped_price is None or slipped_price <= 0:
            return fill_price
        return slipped_price

    def _trigger_stop(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        pos: Position,
        entry: Entry,
        stop: Stop,
        col_scope: Optional[ColumnScope] = None,
        sym_end_index: Optional[Mapping[str, int]] = None,
        ind_scope: Optional[IndicatorScope] = None,
        slippage_model: Optional["SlippageModel"] = None,
    ) -> tuple[bool, Optional[Decimal]]:
        fill_price = None
        if stop.pos_type == "long" and stop.symbol not in self.long_positions:
            return False, fill_price
        if (
            stop.pos_type == "short"
            and stop.symbol not in self.short_positions
        ):
            return False, fill_price
        if stop.stop_type == StopType.BAR:
            fill_price = self._trigger_bar_stop(stop, price_scope, entry)
        elif (
            stop.stop_type == StopType.LOSS
            or stop.stop_type == StopType.PROFIT
        ):
            fill_price = self._trigger_profit_or_loss_stop(stop, price_scope)
        elif stop.stop_type == StopType.TRAILING:
            fill_price = self._trigger_trailing_stop(stop, price_scope)
        else:
            raise ValueError(f"Unknown stop type: {stop.stop_type}")
        if fill_price is None:
            return False, fill_price
        market_price = fill_price
        # Slippage is applied after the trigger decision (which must use the
        # unslipped price) and before the limit check, matching the order in
        # which pending orders are filled.
        fill_price = self._slip_stop_fill_price(
            stop,
            entry,
            fill_price,
            col_scope,
            ind_scope,
            sym_end_index,
            slippage_model,
        )
        assert fill_price is not None
        order_type: Literal["buy", "sell"]
        stop_shares = entry.shares
        if stop.pos_type == "long":
            if stop.limit_price is not None and fill_price < stop.limit_price:
                return False, fill_price
            self._exit_long(
                date, pos, entry, entry.shares, fill_price, stop.stop_type
            )
            order_type = "sell"
        elif stop.pos_type == "short":
            if stop.limit_price is not None and fill_price > stop.limit_price:
                return False, fill_price
            self._exit_short(
                date, pos, entry, entry.shares, fill_price, stop.stop_type
            )
            order_type = "buy"
        else:
            raise ValueError(f"Unknown pos_type: {stop.pos_type}")
        if stop.pos_type == "long":
            intent = PositionIntent.SELL_TO_CLOSE
        else:
            intent = PositionIntent.BUY_TO_CLOSE
        if stop.stop_type == StopType.BAR:
            stop_order_type = OrderType.STOP_BAR
        elif stop.stop_type == StopType.LOSS:
            stop_order_type = OrderType.STOP_LOSS
        elif stop.stop_type == StopType.PROFIT:
            stop_order_type = OrderType.STOP_PROFIT
        elif stop.stop_type == StopType.TRAILING:
            stop_order_type = OrderType.STOP_TRAILING
        else:
            raise ValueError(f"Unknown stop type: {stop.stop_type}")
        self._add_order(
            date=date,
            symbol=pos.symbol,
            type=order_type,
            created=None,
            order_type=stop_order_type,
            intent=intent,
            shares=stop_shares,
            limit_price=stop.limit_price,
            market_price=market_price,
            fill_price=fill_price,
        )
        return True, fill_price

    def _trigger_bar_stop(
        self, stop: Stop, price_scope: PriceScope, entry: Entry
    ) -> Optional[Decimal]:
        if stop.bars is None:
            raise ValueError("Bars not set on bar stop.")
        # Counted against the symbol's own bars, not the portfolio clock: a
        # hold_bars of 3 on a symbol that trades every third date must exit
        # after three of its bars, not after one.
        if entry.sym_bars >= stop.bars:
            fill_price: _BarStopFillPrice = (
                PriceType.MIDDLE
                if stop.fill_price is None
                else cast(_BarStopFillPrice, stop.fill_price)
            )
            resolved = price_scope.fetch(stop.symbol, fill_price)
            self._verify_stop_fill_price(stop, resolved)
            return resolved
        return None

    def _trigger_profit_or_loss_stop(
        self, stop: Stop, price_scope: PriceScope
    ) -> Optional[Decimal]:
        stop_value = self._stop_data[stop.id].value
        if (
            stop.pos_type == "long"
            and (
                stop.stop_type == StopType.LOSS
                or stop.stop_type == StopType.TRAILING
            )
        ) or (stop.pos_type == "short" and stop.stop_type == StopType.PROFIT):
            if stop.exit_price is not None:
                exit_price = price_scope.fetch_float(
                    stop.symbol, stop.exit_price
                )
                if exit_price <= stop_value:
                    return to_decimal(exit_price)
            else:
                low = price_scope.fetch_float(stop.symbol, PriceType.LOW)
                if low <= stop_value:
                    high = price_scope.fetch_float(stop.symbol, PriceType.HIGH)
                    return to_decimal(min(stop_value, high))
        elif (
            stop.pos_type == "long" and stop.stop_type == StopType.PROFIT
        ) or (
            stop.pos_type == "short"
            and (
                stop.stop_type == StopType.LOSS
                or stop.stop_type == StopType.TRAILING
            )
        ):
            if stop.exit_price is not None:
                exit_price = price_scope.fetch_float(
                    stop.symbol, stop.exit_price
                )
                if exit_price >= stop_value:
                    return to_decimal(exit_price)
            else:
                high = price_scope.fetch_float(stop.symbol, PriceType.HIGH)
                if high >= stop_value:
                    low = price_scope.fetch_float(stop.symbol, PriceType.LOW)
                    return to_decimal(max(stop_value, low))
        return None

    def _verify_stop_fill_price(self, stop: Stop, fill_price: Decimal):
        """Rejects a stop fill price that would book a bogus exit.

        Stop exits bypass :meth:`_verify_input`, so without this a fill price
        of zero, a negative, or a NaN silently becomes a completed trade. NaN
        is especially quiet: every limit comparison against it is ``False``.
        """
        if fill_price.is_nan() or not fill_price.is_finite():
            raise ValueError(
                f"Stop {stop.id} for {stop.symbol} resolved to a non-finite "
                f"fill price: {fill_price}."
            )
        if fill_price <= 0:
            raise ValueError(
                f"Stop {stop.id} for {stop.symbol} resolved to a fill price "
                f"of {fill_price}. Stop fill price must be > 0."
            )

    def _trigger_trailing_stop(
        self, stop: Stop, price_scope: PriceScope
    ) -> Optional[Decimal]:
        fill_price = self._trigger_profit_or_loss_stop(stop, price_scope)
        if fill_price is not None:
            return fill_price
        stop_data = self._stop_data[stop.id]
        if stop.pos_type == "long":
            high = price_scope.fetch_float(stop.symbol, PriceType.HIGH)
            amount = self._get_stop_amount_f(stop, high)
            stop_data.value = max(high - amount, stop_data.value)
        else:
            low = price_scope.fetch_float(stop.symbol, PriceType.LOW)
            amount = self._get_stop_amount_f(stop, low)
            stop_data.value = min(low + amount, stop_data.value)
        return None
