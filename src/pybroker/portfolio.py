"""Contains portfolio related functionality, such as portfolio metrics and
placing orders.
"""

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

import itertools
import numpy as np
import pandas as pd
from .common import BarData, DataCol, FeeMode, PriceType, StopType, to_decimal
from .scope import PriceScope, StaticScope
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import (
    Callable,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    Union,
)


class Stop(NamedTuple):
    """Contains information about a stop set on :class:`.Entry`.

    Attributes:
        id: Unique identifier.
        symbol: Symbol of the stop.
        pos_type: Type of  :class:`.Position`, either ``long`` or ``short``.
        percent: Percent from entry price.
        points: Cash amount from entry price.
        bars: Number of bars after which to trigger the stop.
        fill_price: Price that the stop will be filled at.
        limit_price: Limit price to use for the stop.
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
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ]
    ]
    limit_price: Optional[Decimal]


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
        bars: Current number of bars since entry.
        stops: Stops set on the entry.
    """

    id: int
    date: np.datetime64
    symbol: str
    shares: Decimal
    price: Decimal
    type: Literal["long", "short"]
    bars: int = field(default=0)
    stops: set[Stop] = field(default_factory=set)


@dataclass
class _StopData:
    value: Decimal
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
        bars: Current number of bars since entry.
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


class Order(NamedTuple):
    """Holds information about a filled order.

    Attributes:
        id: Unique identifier.
        type: Type of order, either ``buy`` or ``sell``.
        symbol: Ticker symbol of the order.
        date: Date the order was filled.
        shares: Number of shares bought or sold.
        limit_price: Limit price that was used for the order.
        fill_price: Price that the order was filled at.
        fees: Brokerage fees for order.
    """

    id: int
    type: Literal["buy", "sell"]
    symbol: str
    date: np.datetime64
    shares: Decimal
    limit_price: Optional[Decimal]
    fill_price: Decimal
    fees: Decimal


class PortfolioBar(NamedTuple):
    """Snapshot of :class:`.Portfolio` state, captured per bar.

    Attributes:
        date: Date of bar.
        cash: Amount of cash in :class:`.Portfolio`.
        equity: Amount of equity in :class:`.Portfolio`.
        margin: Amount of margin in :class:`.Portfolio`.
        market_value: Market value of :class:`.Portfolio`.
        pnl: Realized + unrealized profit and loss (PnL) of
            :class:`.Portfolio`.
        fees: Brokerage fees.
    """

    date: np.datetime64
    cash: Decimal
    equity: Decimal
    margin: Decimal
    market_value: Decimal
    pnl: Decimal
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


def _calculate_pnl(
    price: Decimal,
    entries: Iterable[Entry],
    entry_type: Literal["short", "long"],
) -> Decimal:
    if entry_type == "long":
        return Decimal(
            sum((price - entry.price) * entry.shares for entry in entries)
        )
    elif entry_type == "short":
        return Decimal(
            sum((entry.price - price) * entry.shares for entry in entries)
        )
    else:
        raise ValueError(f"Unknown entry_type: {entry_type}")


class Portfolio:
    r"""Class representing a portfolio of holdings. The portfolio contains
    information about open positions and balances, and is also used to place
    buy and sell orders.

    Args:
        cash: Starting cash balance.
        fee_mode: Brokerage fee mode.
        fee_amount: Brokerage fee amount.
        max_long_positions: Maximum number of long :class:`.Position`\ s that
            can be held at a time. If ``None``, then unlimited.
        max_short_positions: Maximum number of short :class:`.Position`\ s that
            can be held at a time. If ``None``, then unlimited.

    Attributes:
        cash: Current cash balance.
        equity: Current amount of equity.
        market_value: Current market value. The market value is defined as
            the amount of equity held in cash and long positions added together
            with the unrealized PnL of all open short positions.
        fees: Current brokerage fees.
        enable_fractional_shares: Whether to enable trading fractional shares.
        orders: ``deque`` of all filled orders, sorted in ascending
            chronological order.
        margin: Current amount of margin held in open positions.
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
    """

    _order_id: int = 0
    _entry_id: int = 0
    _trade_id: int = 0

    def __init__(
        self,
        cash: float,
        fee_mode: Optional[FeeMode] = None,
        fee_amount: Optional[float] = None,
        enable_fractional_shares: bool = False,
        max_long_positions: Optional[int] = None,
        max_short_positions: Optional[int] = None,
    ):
        self.cash: Decimal = to_decimal(cash)
        self._initial_market_value = self.cash
        self._fee_mode = fee_mode
        self._fee_amount: Optional[Decimal] = (
            None if fee_amount is None else to_decimal(fee_amount)
        )
        self._enable_fractional_shares = enable_fractional_shares
        self.equity: Decimal = self.cash
        self.market_value: Decimal = self.cash
        self.fees = Decimal()
        self._max_long_positions = max_long_positions
        self._max_short_positions = max_short_positions
        self.orders: deque[Order] = deque()
        self.trades: deque[Trade] = deque()
        self.margin: Decimal = Decimal()
        self.pnl: Decimal = Decimal()
        self.long_positions: dict[str, Position] = {}
        self.short_positions: dict[str, Position] = {}
        self.symbols: set[str] = set()
        self.bars: deque[PortfolioBar] = deque()
        self.position_bars: deque[PositionBar] = deque()
        self._logger = StaticScope.instance().logger
        self._stop_data: dict[int, _StopData] = {}

    def _calculate_fees(self, fill_price: Decimal, shares: Decimal) -> Decimal:
        fees = Decimal()
        if self._fee_mode is None or self._fee_amount is None:
            return fees
        if self._fee_mode == FeeMode.ORDER_PERCENT:
            fees = self._fee_amount / Decimal(100) * fill_price * shares
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
        limit_price: Optional[Decimal],
        fill_price: Decimal,
        shares: Decimal,
    ) -> Order:
        self._order_id += 1
        fees = self._calculate_fees(fill_price, shares)
        order = Order(
            id=self._order_id,
            date=date,
            symbol=symbol,
            type=type,
            limit_price=limit_price,
            fill_price=fill_price,
            shares=shares,
            fees=fees,
        )
        self.orders.append(order)
        self.fees += fees
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
        )
        self.trades.append(trade)

    def _get_stop_amount(self, stop: Stop, price: Decimal) -> Decimal:
        if stop.percent is not None:
            return price * stop.percent / 100
        elif stop.points is not None:
            return stop.points
        else:
            raise ValueError("Stop amount not set.")

    def _add_stops(self, entry: Entry, stops: Iterable[Stop]):
        for stop in stops:
            if stop.id in self._stop_data:
                raise ValueError(f"Duplicate stop ID: {stop.id}")
            entry.stops.add(stop)
            if stop.stop_type == StopType.BAR:
                continue
            amount = self._get_stop_amount(stop, entry.price)
            if (
                stop.pos_type == "long" and stop.stop_type == StopType.PROFIT
            ) or (
                stop.pos_type == "short"
                and (
                    stop.stop_type == StopType.LOSS
                    or stop.stop_type == StopType.TRAILING
                )
            ):
                stop_value = entry.price + amount
            else:
                stop_value = entry.price - amount
            self._stop_data[stop.id] = _StopData(
                value=stop_value, stop=stop, entry=entry
            )

    def _remove_stop_data(self, entry: Entry):
        for stop in entry.stops:
            if stop.id in self._stop_data:
                del self._stop_data[stop.id]

    def _clamp_shares(self, fill_price: Decimal, shares: Decimal) -> Decimal:
        max_shares = (
            Decimal(self.cash / fill_price)
            if self._enable_fractional_shares
            else Decimal(self.cash // fill_price)
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
        covered = self._cover(date, symbol, shares, fill_price, limit_price)
        bought_shares = self._buy(
            date, symbol, covered.rem_shares, fill_price, limit_price, stops
        )
        if not covered.filled_shares and not bought_shares:
            return None
        order = self._add_order(
            date=date,
            symbol=symbol,
            type="buy",
            limit_price=limit_price,
            fill_price=fill_price,
            shares=covered.filled_shares + bought_shares,
        )
        return order

    def _cover(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ) -> _OrderResult:
        if symbol not in self.short_positions:
            return _OrderResult(Decimal(), shares)
        clamped_shares = self._clamp_shares(fill_price, shares)
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
        rem_shares = shares
        if rem_shares <= 0:
            return _OrderResult(Decimal(), shares)
        pos = self.short_positions[symbol]
        while pos.entries:
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
        self.cash -= order_amount
        pos.shares -= shares
        entry.shares -= shares
        pnl_per_bar = entry_pnl if not entry.bars else entry_pnl / entry.bars
        return_pct = ((entry.price / fill_price) - 1) * 100
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
        )

    def _buy(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal],
        stops: Optional[Iterable[Stop]],
    ) -> Decimal:
        clamped_shares = self._clamp_shares(fill_price, shares)
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
            and len(self.long_positions) == self._max_long_positions
        ):
            return Decimal()
        order_amount = shares * fill_price
        self.cash -= order_amount
        if symbol not in self.long_positions:
            self.symbols.add(symbol)
            pos = Position(symbol=symbol, shares=shares, type="long")
            self.long_positions[symbol] = pos
        else:
            pos = self.long_positions[symbol]
            pos.shares += shares
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
        return shares

    def sell(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal] = None,
        stops: Optional[Iterable[Stop]] = None,
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
        order = self._add_order(
            date=date,
            symbol=symbol,
            type="sell",
            limit_price=limit_price,
            fill_price=fill_price,
            shares=sold.filled_shares + short_shares,
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
        while pos.entries:
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
        self.cash += order_amount
        pos.shares -= shares
        entry.shares -= shares
        pnl_per_bar = entry_pnl if not entry.bars else entry_pnl / entry.bars
        return_pct = ((fill_price / entry.price) - 1) * 100
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
        )

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
            and len(self.short_positions) == self._max_short_positions
        ):
            return Decimal()
        self.cash += shares * fill_price
        if symbol not in self.short_positions:
            self.symbols.add(symbol)
            pos = Position(symbol=symbol, shares=shares, type="short")
            self.short_positions[symbol] = pos
        else:
            pos = self.short_positions[symbol]
            pos.shares += shares
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
        return shares

    def exit_position(
        self,
        date: np.datetime64,
        symbol: str,
        buy_fill_price: Decimal,
        sell_fill_price: Decimal,
    ):
        """Exits any long and short positions for ``symbol`` at
        ``buy_fill_price`` and ``sell_fill_price``.
        """
        if symbol in self.long_positions:
            self.sell(
                date=date,
                symbol=symbol,
                shares=self.long_positions[symbol].shares,
                fill_price=sell_fill_price,
            )
        if symbol in self.short_positions:
            self.buy(
                date=date,
                symbol=symbol,
                shares=self.short_positions[symbol].shares,
                fill_price=buy_fill_price,
            )

    def capture_bar(self, date: np.datetime64, df: pd.DataFrame):
        """Captures portfolio state of the current bar.

        Args:
            date: Date of current bar.
            df: :class:`pandas.DataFrame` containing close prices.
        """
        total_equity = self.cash
        total_market_value = total_equity
        total_margin = Decimal()
        for sym in self.symbols:
            index = (sym, date)
            close = None
            if index in df.index:
                close = to_decimal(df.loc[index][DataCol.CLOSE.value])
            pos_long_shares = Decimal()
            pos_short_shares = Decimal()
            pos_equity = Decimal()
            pos_market_value = Decimal()
            pos_margin = Decimal()
            pos_pnl = Decimal()
            if sym in self.long_positions:
                pos = self.long_positions[sym]
                if close is not None:
                    pos.equity = pos.shares * close
                    pos.market_value = pos.equity
                    pos.close = close
                    pos.pnl = _calculate_pnl(close, pos.entries, "long")
                    pos_long_shares += pos.shares
                    pos_equity += pos.equity
                    pos_market_value += pos.market_value
                    pos_pnl += pos.pnl
                total_equity += pos.equity
                total_market_value += pos.equity
            if sym in self.short_positions:
                pos = self.short_positions[sym]
                if close is not None:
                    pos.close = close
                    pos.pnl = _calculate_pnl(close, pos.entries, "short")
                    pos.market_value = pos.pnl
                    pos.margin = close * pos.shares
                    pos_short_shares += pos.shares
                    pos_market_value += pos.pnl
                    pos_pnl += pos.pnl
                    pos_margin += pos.margin
                total_market_value += pos.market_value
                total_margin += pos.margin
            if close is not None:
                self.position_bars.append(
                    PositionBar(
                        symbol=sym,
                        date=date,
                        long_shares=pos_long_shares,
                        short_shares=pos_short_shares,
                        close=close,
                        equity=pos_equity,
                        market_value=pos_market_value,
                        margin=pos_margin,
                        unrealized_pnl=pos_pnl,
                    )
                )

        self.equity = total_equity
        self.market_value = total_market_value
        self.margin = total_margin

        self.bars.append(
            PortfolioBar(
                date=date,
                cash=self.cash,
                equity=self.equity,
                market_value=self.market_value,
                margin=self.margin,
                pnl=self.market_value - self._initial_market_value,
                fees=self.fees,
            )
        )

    def incr_bars(self):
        """Increments the number of bars held by every trade entry."""
        for pos in itertools.chain(
            self.long_positions.values(), self.short_positions.values()
        ):
            pos.bars += 1
            for entry in pos.entries:
                entry.bars += 1

    def remove_stop(self, stop_id: int) -> bool:
        """Removes a :class:`.Stop` with ``stop_id``."""
        if stop_id in self._stop_data:
            stop_data = self._stop_data[stop_id]
            del self._stop_data[stop_id]
            if stop_data.stop in stop_data.entry.stops:
                stop_data.entry.stops.remove(stop_data.stop)
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
        if type(val) == str:
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

    def check_stops(self, date: np.datetime64, price_scope: PriceScope):
        """Checks whether stops are triggered."""
        executed: deque[tuple[Position, Entry]] = deque()
        for pos in itertools.chain(
            self.long_positions.values(), self.short_positions.values()
        ):
            for entry in pos.entries:
                for stop in entry.stops:
                    if self._trigger_stop(date, price_scope, pos, entry, stop):
                        executed.append((pos, entry))
                        break
        for pos, entry in executed:
            pos.entries.remove(entry)
            self._remove_stop_data(entry)
            self._update_position(pos)

    def _trigger_stop(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        pos: Position,
        entry: Entry,
        stop: Stop,
    ) -> bool:
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
            return False
        order_type: Literal["buy", "sell"]
        stop_shares = entry.shares
        if stop.pos_type == "long":
            if stop.limit_price is not None and fill_price < stop.limit_price:
                return False
            self._exit_long(
                date, pos, entry, entry.shares, fill_price, stop.stop_type
            )
            order_type = "sell"
        elif stop.pos_type == "short":
            if stop.limit_price is not None and fill_price > stop.limit_price:
                return False
            self._exit_short(
                date, pos, entry, entry.shares, fill_price, stop.stop_type
            )
            order_type = "buy"
        else:
            raise ValueError(f"Unknown pos_type: {stop.pos_type}")
        self._add_order(
            date=date,
            symbol=pos.symbol,
            type=order_type,
            limit_price=stop.limit_price,
            fill_price=fill_price,
            shares=stop_shares,
        )
        return True

    def _trigger_bar_stop(
        self, stop: Stop, price_scope: PriceScope, entry: Entry
    ) -> Optional[Decimal]:
        if stop.bars is None:
            raise ValueError("Bars not set on bar stop.")
        if entry.bars >= stop.bars:
            return price_scope.fetch(
                stop.symbol,
                PriceType.MIDDLE
                if stop.fill_price is None
                else stop.fill_price,
            )
        return None

    def _trigger_profit_or_loss_stop(
        self, stop: Stop, price_scope: PriceScope
    ) -> Optional[Decimal]:
        if (
            stop.pos_type == "long"
            and (
                stop.stop_type == StopType.LOSS
                or stop.stop_type == StopType.TRAILING
            )
        ) or (stop.pos_type == "short" and stop.stop_type == StopType.PROFIT):
            low = price_scope.fetch(stop.symbol, PriceType.LOW)
            if low <= self._stop_data[stop.id].value:
                return self._stop_data[stop.id].value
        elif (
            stop.pos_type == "long" and stop.stop_type == StopType.PROFIT
        ) or (
            stop.pos_type == "short"
            and (
                stop.stop_type == StopType.LOSS
                or stop.stop_type == StopType.TRAILING
            )
        ):
            high = price_scope.fetch(stop.symbol, PriceType.HIGH)
            if high >= self._stop_data[stop.id].value:
                return self._stop_data[stop.id].value
        return None

    def _trigger_trailing_stop(
        self, stop: Stop, price_scope: PriceScope
    ) -> Optional[Decimal]:
        fill_price = self._trigger_profit_or_loss_stop(stop, price_scope)
        if fill_price is not None:
            return fill_price
        if stop.pos_type == "long":
            high = price_scope.fetch(stop.symbol, PriceType.HIGH)
            amount = self._get_stop_amount(stop, high)
            self._stop_data[stop.id].value = max(
                high - amount, self._stop_data[stop.id].value
            )
        else:
            low = price_scope.fetch(stop.symbol, PriceType.LOW)
            amount = self._get_stop_amount(stop, low)
            self._stop_data[stop.id].value = min(
                low + amount, self._stop_data[stop.id].value
            )
        return None
