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

from .common import DataCol, FeeMode, to_decimal
from .scope import StaticScope
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Literal, NamedTuple, Optional, Union
import math
import numpy as np
import pandas as pd


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
    """

    id: int
    date: np.datetime64
    symbol: str
    shares: Decimal
    price: Decimal
    type: Literal["long", "short"]
    bars: int = field(default=0)


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
        cash: Amount of cash in :class:`.Portfolio` .
        equity: Amount of equity in :class:`.Portfolio`.
        margin: Amount of margin in :class:`.Portfolio`.
        market_value: Market value of :class:`.Portfolio`.
        pnl: Realized profit and loss (PnL) of :class:`.Portfolio`.
        unrealized_pnl: Unrealized profit and loss (PnL) of
            :class:`.Portfolio`.
        fees: Brokerage fees.
    """

    date: np.datetime64
    cash: Decimal
    equity: Decimal
    margin: Decimal
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

    def _calculate_fees(self, fill_price: Decimal, shares: Decimal) -> Decimal:
        fees = Decimal()
        if self._fee_mode is None or self._fee_amount is None:
            return fees
        if self._fee_mode == FeeMode.ORDER_PERCENT:
            fees = self._fee_amount * fill_price * shares
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
        )
        self.trades.append(trade)

    def _clamp_shares(self, fill_price: Decimal, shares: Decimal) -> Decimal:
        max_shares = (
            Decimal(self.cash / fill_price)
            if self._enable_fractional_shares
            else Decimal(math.floor(self.cash / fill_price))
        )
        return min(shares, max_shares)

    def buy(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal] = None,
    ) -> Optional[Order]:
        """Places a buy order.

        Args:
            date: Date when the order is placed.
            symbol: Ticker symbol to buy.
            shares: Number of shares to buy.
            fill_price: If filled, the price used to fill the order.
            limit_price: Limit price of the order.

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
        bought_shares = self._buy(date, symbol, covered.rem_shares, fill_price)
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
    ) -> _OrderResult:
        pnl = Decimal()
        if symbol not in self.short_positions:
            return _OrderResult(Decimal(), shares)
        rem_shares = self._clamp_shares(fill_price, shares)
        if rem_shares <= 0:
            return _OrderResult(Decimal(), shares)
        pos = self.short_positions[symbol]
        while pos.entries:
            entry = pos.entries[0]
            return_pct = ((entry.price / fill_price) - 1) * 100
            if rem_shares >= entry.shares:
                order_amount = entry.shares * fill_price
                entry_amount = entry.shares * entry.price
                entry_pnl = entry_amount - order_amount
                pnl += entry_pnl
                self.cash -= order_amount
                rem_shares -= entry.shares
                pos.shares -= entry.shares
                pos.entries.popleft()
                pnl_per_bar = (
                    entry_pnl if not entry.bars else entry_pnl / entry.bars
                )
                self._add_trade(
                    type=entry.type,
                    symbol=symbol,
                    entry_date=entry.date,
                    exit_date=date,
                    entry_price=entry.price,
                    exit_price=fill_price,
                    shares=entry.shares,
                    pnl=entry_pnl,
                    return_pct=return_pct,
                    agg_pnl=self.pnl + pnl,
                    bars=entry.bars,
                    pnl_per_bar=pnl_per_bar,
                )
            else:
                order_amount = rem_shares * fill_price
                entry_pnl = (rem_shares * entry.price) - order_amount
                pnl += entry_pnl
                self.cash -= order_amount
                entry.shares -= rem_shares
                pos.shares -= rem_shares
                pnl_per_bar = (
                    entry_pnl if not entry.bars else entry_pnl / entry.bars
                )
                self._add_trade(
                    type=entry.type,
                    symbol=symbol,
                    entry_date=entry.date,
                    exit_date=date,
                    entry_price=entry.price,
                    exit_price=fill_price,
                    shares=rem_shares,
                    pnl=entry_pnl,
                    return_pct=return_pct,
                    agg_pnl=self.pnl + pnl,
                    bars=entry.bars,
                    pnl_per_bar=pnl_per_bar,
                )
                rem_shares = Decimal()
                break
        self.pnl += pnl
        if not pos.entries:
            del self.short_positions[symbol]
            if symbol not in self.long_positions:
                self.symbols.remove(symbol)
        return _OrderResult(shares - rem_shares, rem_shares)

    def _buy(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
    ) -> Decimal:
        shares = self._clamp_shares(fill_price, shares)
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
        self._add_entry(
            date=date,
            symbol=symbol,
            shares=shares,
            price=fill_price,
            type="long",
            pos=pos,
        )
        return shares

    def sell(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal] = None,
    ) -> Optional[Order]:
        """Places a sell order.

        Args:
            date: Date when the order is placed.
            symbol: Ticker symbol to sell.
            shares: Number of shares to sell.
            fill_price: If filled, the price used to fill the order.
            limit_price: Limit price of the order.

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
        short_shares = self._short(date, symbol, sold.rem_shares, fill_price)
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
        pnl = Decimal()
        if symbol not in self.long_positions:
            return _OrderResult(Decimal(), shares)
        rem_shares = shares
        pos = self.long_positions[symbol]
        while pos.entries:
            entry = pos.entries[0]
            return_pct = ((fill_price / entry.price) - 1) * 100
            if rem_shares >= entry.shares:
                order_amount = entry.shares * fill_price
                entry_amount = entry.shares * entry.price
                entry_pnl = order_amount - entry_amount
                pnl += entry_pnl
                self.cash += order_amount
                rem_shares -= entry.shares
                pos.shares -= entry.shares
                pos.entries.popleft()
                pnl_per_bar = (
                    entry_pnl if not entry.bars else entry_pnl / entry.bars
                )
                self._add_trade(
                    type=entry.type,
                    symbol=symbol,
                    entry_date=entry.date,
                    exit_date=date,
                    entry_price=entry.price,
                    exit_price=fill_price,
                    shares=entry.shares,
                    pnl=entry_pnl,
                    return_pct=return_pct,
                    agg_pnl=self.pnl + pnl,
                    bars=entry.bars,
                    pnl_per_bar=pnl_per_bar,
                )
            else:
                order_amount = rem_shares * fill_price
                entry_pnl = order_amount - (rem_shares * entry.price)
                pnl += entry_pnl
                self.cash += order_amount
                entry.shares -= rem_shares
                pos.shares -= rem_shares
                pnl_per_bar = (
                    entry_pnl if not entry.bars else entry_pnl / entry.bars
                )
                self._add_trade(
                    type=entry.type,
                    symbol=symbol,
                    entry_date=entry.date,
                    exit_date=date,
                    entry_price=entry.price,
                    exit_price=fill_price,
                    shares=rem_shares,
                    pnl=entry_pnl,
                    return_pct=return_pct,
                    agg_pnl=self.pnl + pnl,
                    bars=entry.bars,
                    pnl_per_bar=pnl_per_bar,
                )
                rem_shares = Decimal()
                break
        self.pnl += pnl
        if not pos.entries:
            del self.long_positions[symbol]
            if symbol not in self.short_positions:
                self.symbols.remove(symbol)
        return _OrderResult(shares - rem_shares, rem_shares)

    def _short(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
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
        self._add_entry(
            date=date,
            symbol=symbol,
            shares=shares,
            price=fill_price,
            type="short",
            pos=pos,
        )
        return shares

    def capture_bar(self, date: np.datetime64, df: pd.DataFrame):
        """Captures portfolio state of the current bar.

        Args:
            date: Date of current bar.
            df: :class:`pandas.DataFrame` containing close prices.
        """
        total_equity = self.cash
        total_market_value = total_equity
        total_pnl = Decimal()
        total_margin = Decimal()
        for sym in self.symbols:
            index = (sym, date)
            if index not in df.index:
                continue
            close = to_decimal(df.loc[index][DataCol.CLOSE.value])
            pos_long_shares = Decimal()
            pos_short_shares = Decimal()
            pos_equity = Decimal()
            pos_market_value = Decimal()
            pos_margin = Decimal()
            pos_pnl = Decimal()
            if sym in self.long_positions:
                pos = self.long_positions[sym]
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
                total_pnl += pos.pnl
            if sym in self.short_positions:
                pos = self.short_positions[sym]
                pos.close = close
                pos.pnl = _calculate_pnl(close, pos.entries, "short")
                pos.market_value = pos.pnl
                pos.margin = close * pos.shares
                pos_short_shares += pos.shares
                pos_market_value += pos.pnl
                pos_pnl += pos.pnl
                pos_margin += pos.margin
                total_market_value += pos.market_value
                total_pnl += pos.pnl
                total_margin += pos.margin
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
                pnl=self.pnl,
                unrealized_pnl=total_pnl,
                fees=self.fees,
            )
        )

    def incr_bars(self):
        """Increments the number of bars held of each trade entry."""
        for pos in self.long_positions.values():
            for entry in pos.entries:
                entry.bars += 1
        for pos in self.short_positions.values():
            for entry in pos.entries:
                entry.bars += 1
