"""Unit tests for portfolio.py module."""

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

from collections import deque
from decimal import Decimal
from pybroker.portfolio import Portfolio
import numpy as np
import pandas as pd
import pytest

SYMBOL_1 = "SPY"
SYMBOL_2 = "AAPL"
CASH = 100_000
FILL_PRICE_1 = Decimal("99.99")
FILL_PRICE_2 = Decimal(100)
FILL_PRICE_3 = Decimal("102.20")
FILL_PRICE_4 = Decimal("103.30")
LIMIT_PRICE_1 = Decimal(100)
LIMIT_PRICE_2 = Decimal(101)
LIMIT_PRICE_3 = Decimal(102)
SHARES_1 = 120
SHARES_2 = 200
DATE_1 = np.datetime64("2020-02-02")
DATE_2 = DATE_1 = np.datetime64("2020-02-03")


def assert_order(
    order, date, symbol, order_type, limit_price, fill_price, shares, pnl
):
    assert order.date == date
    assert order.symbol == symbol
    assert order.order_type == order_type
    assert order.limit_price == limit_price
    assert order.fill_price == fill_price
    assert order.shares == shares
    assert order.pnl == pnl


def assert_portfolio(
    portfolio,
    cash,
    pnl,
    symbols,
    orders,
    short_positions_len,
    long_positions_len,
):
    assert portfolio.cash == cash
    assert portfolio.pnl == pnl
    assert portfolio.symbols == symbols
    assert portfolio.orders == deque(orders)
    assert len(portfolio.short_positions) == short_positions_len
    assert len(portfolio.long_positions) == long_positions_len


def assert_position(pos, symbol, shares, type, entries_len):
    assert pos.symbol == symbol
    assert pos.shares == shares
    assert pos.type == type
    assert len(pos.entries) == entries_len


def assert_entry(entry, date, symbol, shares, price, type):
    assert entry.date == date
    assert entry.symbol == symbol
    assert entry.shares == shares
    assert entry.price == price
    assert entry.type == type


@pytest.mark.parametrize(
    "fill_price, limit_price", [(100, 101), (100, 100), (100, None)]
)
def test_buy(fill_price, limit_price):
    portfolio = Portfolio(CASH)
    order = portfolio.buy(DATE_1, SYMBOL_1, SHARES_1, fill_price, limit_price)
    assert_order(
        order=order,
        date=DATE_1,
        symbol=SYMBOL_1,
        order_type="buy",
        limit_price=limit_price,
        fill_price=fill_price,
        shares=SHARES_1,
        pnl=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - SHARES_1 * fill_price,
        pnl=0,
        symbols={SYMBOL_1},
        orders=[order],
        short_positions_len=0,
        long_positions_len=1,
    )
    pos = portfolio.long_positions[SYMBOL_1]
    assert_position(
        pos=pos, symbol=SYMBOL_1, shares=SHARES_1, type="long", entries_len=1
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=SHARES_1,
        price=fill_price,
        type="long",
    )


def test_buy_when_partial_filled():
    shares = SHARES_1 - 100
    cash = 50 + FILL_PRICE_1 * shares
    portfolio = Portfolio(cash)
    order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    assert_order(
        order=order,
        date=DATE_1,
        symbol=SYMBOL_1,
        order_type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=shares,
        pnl=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=cash - shares * FILL_PRICE_1,
        pnl=0,
        symbols={SYMBOL_1},
        orders=[order],
        short_positions_len=0,
        long_positions_len=1,
    )
    pos = portfolio.long_positions[SYMBOL_1]
    assert_position(
        pos=pos, symbol=SYMBOL_1, shares=shares, type="long", entries_len=1
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=shares,
        price=FILL_PRICE_1,
        type="long",
    )


def test_buy_when_existing_long_position():
    portfolio = Portfolio(CASH)
    order_1 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    order_2 = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_2, LIMIT_PRICE_1
    )
    assert order_1 is not None
    assert_order(
        order=order_2,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_2,
        shares=SHARES_2,
        pnl=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - (SHARES_1 * FILL_PRICE_1 + SHARES_2 * FILL_PRICE_2),
        pnl=0,
        symbols={SYMBOL_1},
        orders=[order_1, order_2],
        short_positions_len=0,
        long_positions_len=1,
    )
    pos = portfolio.long_positions[SYMBOL_1]
    assert_position(
        pos=pos,
        symbol=SYMBOL_1,
        shares=SHARES_1 + SHARES_2,
        type="long",
        entries_len=2,
    )
    entry_1 = pos.entries[0]
    assert_entry(
        entry=entry_1,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=SHARES_1,
        price=FILL_PRICE_1,
        type="long",
    )
    entry_2 = pos.entries[1]
    assert_entry(
        entry=entry_2,
        date=DATE_2,
        symbol=SYMBOL_1,
        shares=SHARES_2,
        price=FILL_PRICE_2,
        type="long",
    )


def test_buy_when_multiple_positions():
    portfolio = Portfolio(CASH)
    order_1 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    order_2 = portfolio.buy(
        DATE_2, SYMBOL_2, SHARES_2, FILL_PRICE_2, LIMIT_PRICE_2
    )
    assert order_1 is not None
    assert_order(
        order=order_2,
        date=DATE_2,
        symbol=SYMBOL_2,
        order_type="buy",
        limit_price=LIMIT_PRICE_2,
        fill_price=FILL_PRICE_2,
        shares=SHARES_2,
        pnl=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - (SHARES_1 * FILL_PRICE_1 + SHARES_2 * FILL_PRICE_2),
        pnl=0,
        symbols={SYMBOL_1, SYMBOL_2},
        orders=[order_1, order_2],
        short_positions_len=0,
        long_positions_len=2,
    )
    pos_1 = portfolio.long_positions[SYMBOL_1]
    assert_position(
        pos=pos_1, symbol=SYMBOL_1, shares=SHARES_1, type="long", entries_len=1
    )
    entry_1 = pos_1.entries[0]
    assert_entry(
        entry=entry_1,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=SHARES_1,
        price=FILL_PRICE_1,
        type="long",
    )
    pos_2 = portfolio.long_positions[SYMBOL_2]
    assert_position(
        pos=pos_2, symbol=SYMBOL_2, shares=SHARES_2, type="long", entries_len=1
    )
    entry_2 = pos_2.entries[0]
    assert_entry(
        entry=entry_2,
        date=DATE_1,
        symbol=SYMBOL_2,
        shares=SHARES_2,
        price=FILL_PRICE_2,
        type="long",
    )


def test_buy_when_existing_short_position():
    portfolio = Portfolio(CASH)
    short_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_1, LIMIT_PRICE_1
    )
    expected_pnl = SHARES_1 * (FILL_PRICE_3 - FILL_PRICE_1)
    expected_shares = SHARES_2 - SHARES_1
    assert short_order is not None
    assert buy_order is not None
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=SHARES_2,
        pnl=expected_pnl,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - (SHARES_2 - SHARES_1) * FILL_PRICE_1 + expected_pnl,
        pnl=expected_pnl,
        symbols={SYMBOL_1},
        orders=[short_order, buy_order],
        short_positions_len=0,
        long_positions_len=1,
    )
    pos = portfolio.long_positions[SYMBOL_1]
    assert_position(
        pos=pos,
        symbol=SYMBOL_1,
        shares=expected_shares,
        type="long",
        entries_len=1,
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_2,
        symbol=SYMBOL_1,
        shares=expected_shares,
        price=FILL_PRICE_1,
        type="long",
    )


def test_buy_when_existing_short_and_not_enough_cash():
    portfolio = Portfolio(100)
    short_order = portfolio.sell(DATE_1, SYMBOL_1, SHARES_1, 5, Decimal("4.9"))
    buy_order = portfolio.buy(DATE_2, SYMBOL_1, SHARES_1, 200, 201)
    expected_shares = SHARES_1 / (200 / 5)
    expected_pnl = (5 - 200) * expected_shares
    assert_order(
        order=short_order,
        date=DATE_1,
        symbol=SYMBOL_1,
        order_type="sell",
        limit_price=Decimal("4.9"),
        fill_price=5,
        shares=SHARES_1,
        pnl=0,
    )
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="buy",
        limit_price=201,
        fill_price=200,
        shares=SHARES_1,
        pnl=expected_pnl,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=100,
        pnl=expected_pnl,
        symbols={SYMBOL_1},
        orders=[short_order, buy_order],
        short_positions_len=1,
        long_positions_len=0,
    )
    pos = portfolio.short_positions[SYMBOL_1]
    assert_position(
        pos=pos,
        symbol=SYMBOL_1,
        shares=SHARES_1 - expected_shares,
        type="short",
        entries_len=1,
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_2,
        symbol=SYMBOL_1,
        shares=pos.shares,
        price=5,
        type="short",
    )


def test_buy_when_negative_cash():
    portfolio = Portfolio(-1000)
    order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    assert order is None
    assert portfolio.cash == -1000
    assert not len(portfolio.long_positions)
    assert not len(portfolio.short_positions)
    assert not len(portfolio.orders)


def test_buy_when_not_filled_max_positions():
    portfolio = Portfolio(CASH, max_long_positions=1)
    order_1 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    assert order_1 is not None
    order_2 = portfolio.buy(
        DATE_2, SYMBOL_2, SHARES_2, FILL_PRICE_2, LIMIT_PRICE_2
    )
    assert order_1 is not None
    assert order_2 is None
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - SHARES_1 * FILL_PRICE_1,
        pnl=0,
        symbols={SYMBOL_1},
        orders=[order_1],
        short_positions_len=0,
        long_positions_len=1,
    )
    pos = portfolio.long_positions[SYMBOL_1]
    assert_position(
        pos=pos, symbol=SYMBOL_1, shares=SHARES_1, type="long", entries_len=1
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=SHARES_1,
        price=FILL_PRICE_1,
        type="long",
    )


def test_buy_when_not_filled_limit():
    portfolio = Portfolio(CASH)
    order = portfolio.buy(DATE_1, SYMBOL_1, SHARES_1, 100, 99)
    assert order is None
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH,
        pnl=0,
        symbols=set(),
        orders=[],
        short_positions_len=0,
        long_positions_len=0,
    )


def test_buy_when_not_filled_cash():
    portfolio = Portfolio(1)
    order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    assert order is None
    assert portfolio.cash == 1
    assert not len(portfolio.long_positions)
    assert not len(portfolio.short_positions)
    assert not len(portfolio.orders)


def test_buy_when_zero_shares():
    portfolio = Portfolio(CASH)
    order = portfolio.buy(DATE_1, SYMBOL_1, 0, FILL_PRICE_1, LIMIT_PRICE_1)
    assert order is None
    assert portfolio.cash == CASH
    assert not len(portfolio.short_positions)
    assert not len(portfolio.long_positions)
    assert not len(portfolio.orders)


@pytest.mark.parametrize(
    "shares, fill_price, limit_price, expected_msg",
    [
        (-1, FILL_PRICE_1, LIMIT_PRICE_1, "Shares cannot be negative: -1"),
        (SHARES_1, -1, LIMIT_PRICE_1, "Fill price must be > 0: -1"),
        (SHARES_1, FILL_PRICE_1, -1, "Limit price must be > 0: -1"),
    ],
)
def test_buy_when_negative_then_error(
    shares, fill_price, limit_price, expected_msg
):
    portfolio = Portfolio(CASH)
    with pytest.raises(ValueError, match=expected_msg):
        portfolio.buy(DATE_1, SYMBOL_1, shares, fill_price, limit_price)


@pytest.mark.parametrize(
    "fill_price, limit_price", [(101, 100), (101, 101), (101, None)]
)
def test_sell_when_all_shares(fill_price, limit_price):
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_1, fill_price, limit_price
    )
    expected_pnl = (fill_price * SHARES_1) - (FILL_PRICE_1 * SHARES_1)
    assert buy_order is not None
    assert_order(
        order=sell_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="sell",
        limit_price=limit_price,
        fill_price=fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        orders=[buy_order, sell_order],
        short_positions_len=0,
        long_positions_len=0,
    )


def test_sell_when_partial_shares():
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_2, FILL_PRICE_1, LIMIT_PRICE_1
    )
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    expected_pnl = (FILL_PRICE_3 * SHARES_1) - (FILL_PRICE_1 * SHARES_1)
    expected_shares = SHARES_2 - SHARES_1
    assert buy_order is not None
    assert_order(
        order=sell_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_3,
        shares=SHARES_1,
        pnl=expected_pnl,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - expected_shares * FILL_PRICE_1 + expected_pnl,
        pnl=expected_pnl,
        symbols={SYMBOL_1},
        orders=[buy_order, sell_order],
        short_positions_len=0,
        long_positions_len=1,
    )
    pos = portfolio.long_positions[SYMBOL_1]
    assert_position(
        pos=pos,
        symbol=SYMBOL_1,
        shares=expected_shares,
        type="long",
        entries_len=1,
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=expected_shares,
        price=FILL_PRICE_1,
        type="long",
    )


def test_sell_when_multiple_entries():
    portfolio = Portfolio(CASH)
    buy_order_1 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    buy_order_2 = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_3, LIMIT_PRICE_3
    )
    expected_pnl = (FILL_PRICE_3 * SHARES_2) - (FILL_PRICE_1 * SHARES_2)
    expected_shares = SHARES_1 * 2 - SHARES_2
    assert buy_order_1 is not None
    assert buy_order_2 is not None
    assert_order(
        order=sell_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_3,
        shares=SHARES_2,
        pnl=expected_pnl,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - expected_shares * FILL_PRICE_1 + expected_pnl,
        pnl=expected_pnl,
        symbols={SYMBOL_1},
        orders=[buy_order_1, buy_order_2, sell_order],
        short_positions_len=0,
        long_positions_len=1,
    )
    pos = portfolio.long_positions[SYMBOL_1]
    assert_position(
        pos=pos,
        symbol=SYMBOL_1,
        shares=expected_shares,
        type="long",
        entries_len=1,
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_2,
        symbol=SYMBOL_1,
        shares=expected_shares,
        price=FILL_PRICE_1,
        type="long",
    )


def test_sell_when_not_filled_limit():
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    sell_order = portfolio.sell(DATE_2, SYMBOL_1, SHARES_1, 99, 100)
    assert buy_order is not None
    assert sell_order is None
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - FILL_PRICE_1 * SHARES_1,
        pnl=0,
        symbols={SYMBOL_1},
        orders=[buy_order],
        short_positions_len=0,
        long_positions_len=1,
    )


def test_sell_when_zero_shares():
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, 0, FILL_PRICE_3, LIMIT_PRICE_3
    )
    assert buy_order is not None
    assert sell_order is None
    assert portfolio.cash == CASH - FILL_PRICE_1 * SHARES_1
    assert len(portfolio.long_positions) == 1
    assert not len(portfolio.short_positions)
    assert portfolio.orders == deque([buy_order])


@pytest.mark.parametrize(
    "shares, fill_price, limit_price, expected_msg",
    [
        (-1, FILL_PRICE_3, LIMIT_PRICE_3, "Shares cannot be negative: -1"),
        (SHARES_1, -1, LIMIT_PRICE_3, "Fill price must be > 0: -1"),
        (SHARES_1, FILL_PRICE_3, -1, "Limit price must be > 0: -1"),
    ],
)
def test_sell_when_negative_shares_then_error(
    shares, fill_price, limit_price, expected_msg
):
    portfolio = Portfolio(CASH)
    with pytest.raises(ValueError, match=expected_msg):
        portfolio.sell(DATE_1, SYMBOL_1, shares, fill_price, limit_price)


@pytest.mark.parametrize(
    "fill_price, limit_price", [(100, 99), (100, 100), (100, None)]
)
def test_short(fill_price, limit_price):
    portfolio = Portfolio(CASH)
    order = portfolio.sell(DATE_1, SYMBOL_1, SHARES_1, fill_price, limit_price)
    assert_order(
        order=order,
        date=DATE_1,
        symbol=SYMBOL_1,
        order_type="sell",
        limit_price=limit_price,
        fill_price=fill_price,
        shares=SHARES_1,
        pnl=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + SHARES_1 * fill_price,
        pnl=0,
        symbols={SYMBOL_1},
        orders=[order],
        long_positions_len=0,
        short_positions_len=1,
    )
    pos = portfolio.short_positions[SYMBOL_1]
    assert_position(
        pos=pos, symbol=SYMBOL_1, shares=SHARES_1, type="short", entries_len=1
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=SHARES_1,
        price=fill_price,
        type="short",
    )


def test_short_when_existing_short_position():
    portfolio = Portfolio(CASH)
    order_1 = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    order_2 = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_4, LIMIT_PRICE_3
    )
    assert order_1 is not None
    assert_order(
        order=order_2,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_4,
        shares=SHARES_2,
        pnl=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + SHARES_1 * FILL_PRICE_3 + SHARES_2 * FILL_PRICE_4,
        pnl=0,
        symbols={SYMBOL_1},
        orders=[order_1, order_2],
        short_positions_len=1,
        long_positions_len=0,
    )
    pos = portfolio.short_positions[SYMBOL_1]
    assert_position(
        pos=pos,
        symbol=SYMBOL_1,
        shares=SHARES_1 + SHARES_2,
        type="short",
        entries_len=2,
    )
    entry_1 = pos.entries[0]
    assert_entry(
        entry=entry_1,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=SHARES_1,
        price=FILL_PRICE_3,
        type="short",
    )
    entry_2 = pos.entries[1]
    assert_entry(
        entry=entry_2,
        date=DATE_2,
        symbol=SYMBOL_1,
        shares=SHARES_2,
        price=FILL_PRICE_4,
        type="short",
    )


def test_short_when_multiple_positions():
    portfolio = Portfolio(CASH)
    order_1 = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_1
    )
    order_2 = portfolio.sell(
        DATE_2, SYMBOL_2, SHARES_2, FILL_PRICE_4, LIMIT_PRICE_3
    )
    assert order_1 is not None
    assert_order(
        order=order_2,
        date=DATE_2,
        symbol=SYMBOL_2,
        order_type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_4,
        shares=SHARES_2,
        pnl=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + SHARES_1 * FILL_PRICE_3 + SHARES_2 * FILL_PRICE_4,
        pnl=0,
        symbols={SYMBOL_1, SYMBOL_2},
        orders=[order_1, order_2],
        short_positions_len=2,
        long_positions_len=0,
    )
    pos_1 = portfolio.short_positions[SYMBOL_1]
    assert_position(
        pos=pos_1,
        symbol=SYMBOL_1,
        shares=SHARES_1,
        type="short",
        entries_len=1,
    )
    entry_1 = pos_1.entries[0]
    assert_entry(
        entry=entry_1,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=SHARES_1,
        price=FILL_PRICE_3,
        type="short",
    )
    pos_2 = portfolio.short_positions[SYMBOL_2]
    assert_position(
        pos=pos_2,
        symbol=SYMBOL_2,
        shares=SHARES_2,
        type="short",
        entries_len=1,
    )
    entry_2 = pos_2.entries[0]
    assert_entry(
        entry=entry_2,
        date=DATE_1,
        symbol=SYMBOL_2,
        shares=SHARES_2,
        price=FILL_PRICE_4,
        type="short",
    )


def test_short_when_existing_long_position():
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    short_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_3, LIMIT_PRICE_3
    )
    expected_pnl = SHARES_1 * (FILL_PRICE_3 - FILL_PRICE_1)
    expected_shares = SHARES_2 - SHARES_1
    assert buy_order is not None
    assert short_order is not None
    assert_order(
        order=short_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_3,
        shares=SHARES_2,
        pnl=expected_pnl,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - SHARES_1 * FILL_PRICE_1 + SHARES_2 * FILL_PRICE_3,
        pnl=expected_pnl,
        symbols={SYMBOL_1},
        orders=[buy_order, short_order],
        short_positions_len=1,
        long_positions_len=0,
    )
    pos = portfolio.short_positions[SYMBOL_1]
    assert_position(
        pos=pos,
        symbol=SYMBOL_1,
        shares=expected_shares,
        type="short",
        entries_len=1,
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_2,
        symbol=SYMBOL_1,
        shares=expected_shares,
        price=FILL_PRICE_3,
        type="short",
    )


def test_short_when_not_filled_max_positions():
    portfolio = Portfolio(CASH, max_short_positions=1)
    order_1 = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    order_2 = portfolio.sell(
        DATE_2, SYMBOL_2, SHARES_2, FILL_PRICE_4, LIMIT_PRICE_3
    )
    assert order_1 is not None
    assert order_2 is None
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + SHARES_1 * FILL_PRICE_3,
        pnl=0,
        symbols={SYMBOL_1},
        orders=[order_1],
        short_positions_len=1,
        long_positions_len=0,
    )
    pos = portfolio.short_positions[SYMBOL_1]
    assert_position(
        pos=pos, symbol=SYMBOL_1, shares=SHARES_1, type="short", entries_len=1
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=SHARES_1,
        price=FILL_PRICE_3,
        type="short",
    )


def test_short_when_not_filled_limit():
    portfolio = Portfolio(CASH)
    sell_order = portfolio.sell(DATE_1, SYMBOL_1, SHARES_1, 99, 100)
    assert sell_order is None
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH,
        pnl=0,
        symbols=set(),
        orders=[],
        short_positions_len=0,
        long_positions_len=0,
    )


def test_short_when_zero_shares():
    portfolio = Portfolio(CASH)
    order = portfolio.sell(DATE_1, SYMBOL_1, 0, FILL_PRICE_3, LIMIT_PRICE_3)
    assert order is None
    assert portfolio.cash == CASH
    assert not len(portfolio.short_positions)
    assert not len(portfolio.long_positions)
    assert not len(portfolio.orders)


@pytest.mark.parametrize(
    "fill_price, limit_price", [(100, 101), (100, 100), (100, None)]
)
def test_cover_when_all_shares(fill_price, limit_price):
    portfolio = Portfolio(CASH)
    sell_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_1, fill_price, limit_price
    )
    expected_pnl = (FILL_PRICE_3 * SHARES_1) - (fill_price * SHARES_1)
    assert sell_order is not None
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="buy",
        limit_price=limit_price,
        fill_price=fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        orders=[sell_order, buy_order],
        short_positions_len=0,
        long_positions_len=0,
    )


def test_cover_when_partial_shares():
    portfolio = Portfolio(CASH)
    sell_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_2, FILL_PRICE_3, LIMIT_PRICE_3
    )
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    expected_pnl = (FILL_PRICE_3 * SHARES_1) - (FILL_PRICE_1 * SHARES_1)
    expected_shares = SHARES_2 - SHARES_1
    assert sell_order is not None
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=SHARES_1,
        pnl=expected_pnl,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + SHARES_2 * FILL_PRICE_3 - SHARES_1 * FILL_PRICE_1,
        pnl=expected_pnl,
        symbols={SYMBOL_1},
        orders=[sell_order, buy_order],
        short_positions_len=1,
        long_positions_len=0,
    )
    pos = portfolio.short_positions[SYMBOL_1]
    assert_position(
        pos=pos,
        symbol=SYMBOL_1,
        shares=expected_shares,
        type="short",
        entries_len=1,
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=expected_shares,
        price=FILL_PRICE_3,
        type="short",
    )


def test_cover_when_multiple_entries():
    portfolio = Portfolio(CASH)
    sell_order_1 = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    sell_order_2 = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_1, LIMIT_PRICE_1
    )
    expected_pnl = (FILL_PRICE_3 * SHARES_2) - (FILL_PRICE_1 * SHARES_2)
    expected_shares = SHARES_1 * 2 - SHARES_2
    assert sell_order_1 is not None
    assert sell_order_2 is not None
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        order_type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=SHARES_2,
        pnl=expected_pnl,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + SHARES_1 * 2 * FILL_PRICE_3 - SHARES_2 * FILL_PRICE_1,
        pnl=expected_pnl,
        symbols={SYMBOL_1},
        orders=[sell_order_1, sell_order_2, buy_order],
        short_positions_len=1,
        long_positions_len=0,
    )
    pos = portfolio.short_positions[SYMBOL_1]
    assert_position(
        pos=pos,
        symbol=SYMBOL_1,
        shares=expected_shares,
        type="short",
        entries_len=1,
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_2,
        symbol=SYMBOL_1,
        shares=expected_shares,
        price=FILL_PRICE_3,
        type="short",
    )


def test_cover_when_not_enough_cash():
    portfolio = Portfolio(100)
    short_order = portfolio.sell(DATE_1, SYMBOL_1, SHARES_1, 5, Decimal("4.9"))
    buy_order = portfolio.buy(DATE_2, SYMBOL_1, SHARES_1, 1000, 1001)
    assert buy_order is None
    assert_order(
        order=short_order,
        date=DATE_1,
        symbol=SYMBOL_1,
        order_type="sell",
        limit_price=Decimal("4.9"),
        fill_price=5,
        shares=SHARES_1,
        pnl=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=100 + SHARES_1 * 5,
        pnl=0,
        symbols={SYMBOL_1},
        orders=[short_order],
        short_positions_len=1,
        long_positions_len=0,
    )
    pos = portfolio.short_positions[SYMBOL_1]
    assert_position(
        pos=pos, symbol=SYMBOL_1, shares=SHARES_1, type="short", entries_len=1
    )
    entry = pos.entries[0]
    assert_entry(
        entry=entry,
        date=DATE_2,
        symbol=SYMBOL_1,
        shares=pos.shares,
        price=5,
        type="short",
    )


def test_cover_when_not_filled_limit():
    portfolio = Portfolio(CASH)
    sell_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    buy_order = portfolio.buy(DATE_2, SYMBOL_1, SHARES_1, 100, 99)
    assert sell_order is not None
    assert buy_order is None
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + FILL_PRICE_3 * SHARES_1,
        pnl=0,
        symbols={SYMBOL_1},
        orders=[sell_order],
        short_positions_len=1,
        long_positions_len=0,
    )


def test_cover_when_zero_shares():
    portfolio = Portfolio(CASH)
    sell_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    buy_order = portfolio.buy(DATE_2, SYMBOL_1, 0, FILL_PRICE_1, LIMIT_PRICE_1)
    assert sell_order is not None
    assert buy_order is None
    assert portfolio.cash == CASH + SHARES_1 * FILL_PRICE_3
    assert len(portfolio.short_positions) == 1
    assert not len(portfolio.long_positions)
    assert portfolio.orders == deque([sell_order])


def test_capture_bar():
    portfolio = Portfolio(CASH)
    buy_order_1 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    buy_order_2 = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_2, LIMIT_PRICE_2
    )
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_2, SHARES_2, FILL_PRICE_3, LIMIT_PRICE_3
    )
    close_1 = 102
    close_2 = 105
    df = pd.DataFrame(
        [[SYMBOL_1, DATE_2, close_1], [SYMBOL_2, DATE_2, close_2]],
        columns=["symbol", "date", "close"],
    )
    df = df.set_index(["symbol", "date"])
    portfolio.capture_bar(DATE_2, df)
    assert buy_order_1 is not None
    assert buy_order_2 is not None
    assert sell_order is not None
    assert (
        portfolio.equity
        == CASH
        - SHARES_1 * FILL_PRICE_1
        - SHARES_2 * FILL_PRICE_2
        + (SHARES_1 + SHARES_2) * close_1
        + SHARES_2 * FILL_PRICE_3
    )
    assert (
        portfolio.market_value
        == portfolio.equity + (FILL_PRICE_3 - close_2) * SHARES_2
    )
    assert portfolio.margin == close_2 * SHARES_2
    assert len(portfolio.position_bars) == 2
    pos_bars = sorted(portfolio.position_bars, key=lambda x: x.symbol)
    pos_bar_1, pos_bar_2 = pos_bars[0], pos_bars[1]
    assert pos_bar_1.symbol == SYMBOL_2
    assert pos_bar_1.date == DATE_2
    assert pos_bar_1.long_shares == 0
    assert pos_bar_1.short_shares == SHARES_2
    assert pos_bar_1.close == close_2
    assert pos_bar_1.equity == 0
    assert pos_bar_1.market_value == (FILL_PRICE_3 - close_2) * SHARES_2
    assert pos_bar_1.margin == close_2 * SHARES_2
    assert pos_bar_1.unrealized_pnl == (FILL_PRICE_3 - close_2) * SHARES_2
    assert pos_bar_2.symbol == SYMBOL_1
    assert pos_bar_2.date == DATE_2
    assert pos_bar_2.long_shares == SHARES_1 + SHARES_2
    assert pos_bar_2.short_shares == 0
    assert pos_bar_2.close == close_1
    assert pos_bar_2.equity == (SHARES_1 + SHARES_2) * close_1
    assert pos_bar_2.market_value == pos_bar_2.equity
    assert pos_bar_2.margin == 0
    assert pos_bar_2.unrealized_pnl == close_1 * (SHARES_1 + SHARES_2) - (
        FILL_PRICE_1 * SHARES_1
    ) - (FILL_PRICE_2 * SHARES_2)
    assert len(portfolio.bars) == 1
    bar = portfolio.bars[0]
    assert bar.date == DATE_2
    assert bar.cash == portfolio.cash
    assert bar.equity == portfolio.equity
    assert bar.margin == portfolio.margin
    assert bar.market_value == portfolio.market_value
    assert bar.pnl == 0
    assert (
        bar.unrealized_pnl
        == pos_bar_1.unrealized_pnl + pos_bar_2.unrealized_pnl
    )
