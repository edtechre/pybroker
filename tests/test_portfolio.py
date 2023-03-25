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

import numpy as np
import pandas as pd
import pytest
from collections import deque
from decimal import Decimal
from pybroker.common import FeeMode, PriceType, StopType
from pybroker.portfolio import Portfolio, Stop
from pybroker.scope import ColumnScope, PriceScope

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
SHARES_1 = Decimal(120)
SHARES_2 = Decimal(200)
DATE_1 = np.datetime64("2020-02-02")
DATE_2 = np.datetime64("2020-02-03")
DATE_3 = np.datetime64("2020-02-04")
DATE_4 = np.datetime64("2020-02-05")


def assert_order(
    order,
    date,
    symbol,
    type,
    limit_price,
    fill_price,
    shares,
    fees,
):
    assert order.date == date
    assert order.symbol == symbol
    assert order.type == type
    assert order.limit_price == limit_price
    assert order.fill_price == fill_price
    assert order.shares == shares
    assert order.fees == fees


def assert_trade(
    trade,
    type,
    symbol,
    entry_date,
    exit_date,
    entry,
    exit,
    shares,
    pnl,
    return_pct,
    agg_pnl,
    bars,
    pnl_per_bar,
    stop_type,
):
    assert trade.type == type
    assert trade.symbol == symbol
    assert trade.entry_date == entry_date
    assert trade.exit_date == exit_date
    assert trade.entry == entry
    assert trade.exit == exit
    assert trade.shares == shares
    assert trade.pnl == pnl
    assert trade.return_pct == return_pct
    assert trade.agg_pnl == agg_pnl
    assert trade.bars == bars
    assert trade.pnl_per_bar == pnl_per_bar
    if stop_type is None:
        assert trade.stop is None
    else:
        assert trade.stop == stop_type.value


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
        type="buy",
        limit_price=limit_price,
        fill_price=fill_price,
        shares=SHARES_1,
        fees=0,
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
    assert not portfolio.trades


def test_buy_when_partial_filled():
    shares = Decimal(SHARES_1 - 100)
    cash = 50 + FILL_PRICE_1 * shares
    portfolio = Portfolio(cash)
    order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    assert_order(
        order=order,
        date=DATE_1,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=shares,
        fees=0,
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
    assert not portfolio.trades


def test_buy_when_existing_long_position():
    portfolio = Portfolio(CASH)
    order_1 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    portfolio.incr_bars()
    order_2 = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_2, LIMIT_PRICE_1
    )
    assert order_1 is not None
    assert_order(
        order=order_2,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_2,
        shares=SHARES_2,
        fees=0,
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
    assert not portfolio.trades


def test_buy_when_multiple_positions():
    portfolio = Portfolio(CASH)
    order_1 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    portfolio.incr_bars()
    order_2 = portfolio.buy(
        DATE_2, SYMBOL_2, SHARES_2, FILL_PRICE_2, LIMIT_PRICE_2
    )
    assert order_1 is not None
    assert_order(
        order=order_2,
        date=DATE_2,
        symbol=SYMBOL_2,
        type="buy",
        limit_price=LIMIT_PRICE_2,
        fill_price=FILL_PRICE_2,
        shares=SHARES_2,
        fees=0,
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
        date=DATE_2,
        symbol=SYMBOL_2,
        shares=SHARES_2,
        price=FILL_PRICE_2,
        type="long",
    )
    assert not portfolio.trades


def test_buy_when_existing_short_position():
    portfolio = Portfolio(CASH)
    short_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    portfolio.incr_bars()
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
        type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=SHARES_2,
        fees=0,
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
    assert len(portfolio.trades) == 1
    expected_return_pct = (FILL_PRICE_3 / FILL_PRICE_1 - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_3,
        exit=FILL_PRICE_1,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
    )


def test_buy_when_existing_short_and_not_enough_cash():
    portfolio = Portfolio(100)
    entry_price = Decimal(5)
    entry_limit = Decimal("4.9")
    exit_price = Decimal(200)
    exit_limit = Decimal(201)
    short_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, entry_price, entry_limit
    )
    portfolio.incr_bars()
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_1, exit_price, exit_limit
    )
    expected_shares = SHARES_1 / (exit_price / entry_price)
    expected_pnl = (entry_price - exit_price) * expected_shares
    assert_order(
        order=short_order,
        date=DATE_1,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=entry_limit,
        fill_price=5,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=exit_limit,
        fill_price=exit_price,
        shares=expected_shares,
        fees=0,
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
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=pos.shares,
        price=5,
        type="short",
    )
    assert len(portfolio.trades) == 1
    expected_return_pct = (entry_price / exit_price - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=entry_price,
        exit=exit_price,
        shares=expected_shares,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
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
    assert not len(portfolio.trades)


def test_buy_when_not_filled_max_positions():
    portfolio = Portfolio(CASH, max_long_positions=1)
    order_1 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    assert order_1 is not None
    portfolio.incr_bars()
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
    assert not portfolio.trades


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
    assert not portfolio.trades


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
    assert not len(portfolio.trades)


def test_buy_when_zero_shares():
    portfolio = Portfolio(CASH)
    order = portfolio.buy(DATE_1, SYMBOL_1, 0, FILL_PRICE_1, LIMIT_PRICE_1)
    assert order is None
    assert portfolio.cash == CASH
    assert not len(portfolio.short_positions)
    assert not len(portfolio.long_positions)
    assert not len(portfolio.orders)
    assert not len(portfolio.trades)


@pytest.mark.parametrize(
    "shares, fill_price, limit_price, expected_msg",
    [
        (-1, FILL_PRICE_1, LIMIT_PRICE_1, "Shares cannot be negative: -1"),
        (SHARES_1, -1, LIMIT_PRICE_1, "Fill price must be > 0: -1"),
        (SHARES_1, FILL_PRICE_1, -1, "Limit price must be > 0: -1"),
    ],
)
def test_buy_when_invalid_input_then_error(
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
    portfolio.incr_bars()
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_1, fill_price, limit_price
    )
    expected_pnl = (fill_price - FILL_PRICE_1) * SHARES_1
    assert buy_order is not None
    assert_order(
        order=sell_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=limit_price,
        fill_price=fill_price,
        shares=SHARES_1,
        fees=0,
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
    assert len(portfolio.trades) == 1
    expected_return_pct = (fill_price / FILL_PRICE_1 - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_1,
        exit=fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
    )


def test_sell_when_all_shares_and_multiple_bars():
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    portfolio.incr_bars()
    portfolio.incr_bars()
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    expected_pnl = (FILL_PRICE_3 - FILL_PRICE_1) * SHARES_1
    assert buy_order is not None
    assert_order(
        order=sell_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_3,
        shares=SHARES_1,
        fees=0,
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
    assert len(portfolio.trades) == 1
    expected_return_pct = (FILL_PRICE_3 / FILL_PRICE_1 - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_1,
        exit=FILL_PRICE_3,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=2,
        pnl_per_bar=expected_pnl / 2,
        stop_type=None,
    )


def test_sell_when_all_shares_and_fractional():
    shares = Decimal("0.34")
    portfolio = Portfolio(CASH, enable_fractional_shares=True)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, shares, FILL_PRICE_1, LIMIT_PRICE_1
    )
    assert_order(
        buy_order,
        date=DATE_1,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=shares,
        fees=0,
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
    portfolio.incr_bars()
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, shares, FILL_PRICE_3, LIMIT_PRICE_3
    )
    expected_pnl = (FILL_PRICE_3 - FILL_PRICE_1) * shares
    assert_order(
        order=sell_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_3,
        shares=shares,
        fees=0,
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
    assert len(portfolio.trades) == 1
    expected_return_pct = (FILL_PRICE_3 / FILL_PRICE_1 - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_1,
        exit=FILL_PRICE_3,
        shares=shares,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
    )


@pytest.mark.parametrize(
    "fee_mode, expected_buy_fees, expected_sell_fees",
    [
        (
            FeeMode.ORDER_PERCENT,
            FILL_PRICE_1 * SHARES_1 * Decimal("0.01"),
            FILL_PRICE_3 * SHARES_1 * Decimal("0.01"),
        ),
        (
            FeeMode.PER_SHARE,
            SHARES_1,
            SHARES_1,
        ),
        (FeeMode.PER_ORDER, Decimal("1"), Decimal("1")),
    ],
)
def test_buy_and_sell_when_fees(
    fee_mode, expected_buy_fees, expected_sell_fees
):
    portfolio = Portfolio(CASH, fee_mode, 1)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    portfolio.incr_bars()
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    assert_order(
        order=buy_order,
        date=DATE_1,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=SHARES_1,
        fees=expected_buy_fees,
    )
    assert_order(
        order=sell_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_3,
        shares=SHARES_1,
        fees=expected_sell_fees,
    )
    assert portfolio.fees == expected_buy_fees + expected_sell_fees


def test_sell_when_partial_shares():
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_2, FILL_PRICE_1, LIMIT_PRICE_1
    )
    portfolio.incr_bars()
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    expected_pnl = (FILL_PRICE_3 - FILL_PRICE_1) * SHARES_1
    expected_shares = SHARES_2 - SHARES_1
    assert buy_order is not None
    assert_order(
        order=sell_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_3,
        shares=SHARES_1,
        fees=0,
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
    assert len(portfolio.trades) == 1
    expected_return_pct = (FILL_PRICE_3 / FILL_PRICE_1 - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_1,
        exit=FILL_PRICE_3,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
    )


def test_sell_when_multiple_entries():
    portfolio = Portfolio(CASH)
    buy_order_1 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    buy_order_2 = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    portfolio.incr_bars()
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_3, LIMIT_PRICE_3
    )
    expected_order_pnl = (FILL_PRICE_3 - FILL_PRICE_1) * SHARES_2
    expected_shares = SHARES_1 * 2 - SHARES_2
    assert buy_order_1 is not None
    assert buy_order_2 is not None
    assert_order(
        order=sell_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_3,
        shares=SHARES_2,
        fees=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - expected_shares * FILL_PRICE_1 + expected_order_pnl,
        pnl=expected_order_pnl,
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
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=expected_shares,
        price=FILL_PRICE_1,
        type="long",
    )
    assert len(portfolio.trades) == 2
    expected_trade_pnl_1 = (FILL_PRICE_3 - FILL_PRICE_1) * SHARES_1
    expected_return_pct = (FILL_PRICE_3 / FILL_PRICE_1 - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_1,
        exit=FILL_PRICE_3,
        shares=SHARES_1,
        pnl=expected_trade_pnl_1,
        return_pct=expected_return_pct,
        agg_pnl=expected_trade_pnl_1,
        bars=1,
        pnl_per_bar=expected_trade_pnl_1,
        stop_type=None,
    )
    expected_trade_pnl_2 = (FILL_PRICE_3 - FILL_PRICE_1) * (
        SHARES_2 - SHARES_1
    )
    assert_trade(
        trade=portfolio.trades[1],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_1,
        exit=FILL_PRICE_3,
        shares=SHARES_2 - SHARES_1,
        pnl=expected_trade_pnl_2,
        return_pct=expected_return_pct,
        agg_pnl=expected_trade_pnl_1 + expected_trade_pnl_2,
        bars=1,
        pnl_per_bar=expected_trade_pnl_2,
        stop_type=None,
    )


def test_sell_when_not_filled_limit():
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    portfolio.incr_bars()
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
    assert not portfolio.trades


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
    assert not portfolio.trades


@pytest.mark.parametrize(
    "shares, fill_price, limit_price, expected_msg",
    [
        (-1, FILL_PRICE_3, LIMIT_PRICE_3, "Shares cannot be negative: -1"),
        (SHARES_1, -1, LIMIT_PRICE_3, "Fill price must be > 0: -1"),
        (SHARES_1, FILL_PRICE_3, -1, "Limit price must be > 0: -1"),
    ],
)
def test_sell_when_invalid_input_then_error(
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
        type="sell",
        limit_price=limit_price,
        fill_price=fill_price,
        shares=SHARES_1,
        fees=0,
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
    assert not portfolio.trades


def test_short_when_existing_short_position():
    portfolio = Portfolio(CASH)
    order_1 = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    portfolio.incr_bars()
    order_2 = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_4, LIMIT_PRICE_3
    )
    assert order_1 is not None
    assert_order(
        order=order_2,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_4,
        shares=SHARES_2,
        fees=0,
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
    assert not portfolio.trades


def test_short_when_multiple_positions():
    portfolio = Portfolio(CASH)
    order_1 = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_1
    )
    portfolio.incr_bars()
    order_2 = portfolio.sell(
        DATE_2, SYMBOL_2, SHARES_2, FILL_PRICE_4, LIMIT_PRICE_3
    )
    assert order_1 is not None
    assert_order(
        order=order_2,
        date=DATE_2,
        symbol=SYMBOL_2,
        type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_4,
        shares=SHARES_2,
        fees=0,
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
        date=DATE_2,
        symbol=SYMBOL_2,
        shares=SHARES_2,
        price=FILL_PRICE_4,
        type="short",
    )
    assert not portfolio.trades


def test_short_when_existing_long_position():
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    portfolio.incr_bars()
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
        type="sell",
        limit_price=LIMIT_PRICE_3,
        fill_price=FILL_PRICE_3,
        shares=SHARES_2,
        fees=0,
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
    assert len(portfolio.trades) == 1
    expected_return_pct = (FILL_PRICE_3 / FILL_PRICE_1 - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_1,
        exit=FILL_PRICE_3,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
    )


def test_short_when_not_filled_max_positions():
    portfolio = Portfolio(CASH, max_short_positions=1)
    order_1 = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    portfolio.incr_bars()
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
    assert not portfolio.trades


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
    assert not portfolio.trades


def test_short_when_zero_shares():
    portfolio = Portfolio(CASH)
    order = portfolio.sell(DATE_1, SYMBOL_1, 0, FILL_PRICE_3, LIMIT_PRICE_3)
    assert order is None
    assert portfolio.cash == CASH
    assert not len(portfolio.short_positions)
    assert not len(portfolio.long_positions)
    assert not len(portfolio.orders)
    assert not len(portfolio.trades)


@pytest.mark.parametrize(
    "fill_price, limit_price", [(100, 101), (100, 100), (100, None)]
)
def test_cover_when_all_shares(fill_price, limit_price):
    portfolio = Portfolio(CASH)
    sell_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    portfolio.incr_bars()
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_1, fill_price, limit_price
    )
    expected_pnl = (FILL_PRICE_3 - fill_price) * SHARES_1
    assert sell_order is not None
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=limit_price,
        fill_price=fill_price,
        shares=SHARES_1,
        fees=0,
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
    assert len(portfolio.trades) == 1
    expected_return_pct = (FILL_PRICE_3 / fill_price - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_3,
        exit=fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
    )


def test_cover_when_partial_shares():
    portfolio = Portfolio(CASH)
    sell_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_2, FILL_PRICE_3, LIMIT_PRICE_3
    )
    portfolio.incr_bars()
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    expected_pnl = (FILL_PRICE_3 - FILL_PRICE_1) * SHARES_1
    expected_shares = SHARES_2 - SHARES_1
    assert sell_order is not None
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=SHARES_1,
        fees=0,
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
    assert len(portfolio.trades) == 1
    expected_return_pct = (FILL_PRICE_3 / FILL_PRICE_1 - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_3,
        exit=FILL_PRICE_1,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
    )


def test_cover_when_multiple_entries():
    portfolio = Portfolio(CASH)
    sell_order_1 = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    sell_order_2 = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    portfolio.incr_bars()
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_1, LIMIT_PRICE_1
    )
    expected_order_pnl = (FILL_PRICE_3 - FILL_PRICE_1) * SHARES_2
    expected_shares = SHARES_1 * 2 - SHARES_2
    assert sell_order_1 is not None
    assert sell_order_2 is not None
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=LIMIT_PRICE_1,
        fill_price=FILL_PRICE_1,
        shares=SHARES_2,
        fees=0,
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + SHARES_1 * 2 * FILL_PRICE_3 - SHARES_2 * FILL_PRICE_1,
        pnl=expected_order_pnl,
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
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=expected_shares,
        price=FILL_PRICE_3,
        type="short",
    )
    assert len(portfolio.trades) == 2
    expected_trade_pnl_1 = (FILL_PRICE_3 - FILL_PRICE_1) * SHARES_1
    expected_return_pct = (FILL_PRICE_3 / FILL_PRICE_1 - 1) * 100
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_3,
        exit=FILL_PRICE_1,
        shares=SHARES_1,
        pnl=expected_trade_pnl_1,
        return_pct=expected_return_pct,
        agg_pnl=expected_trade_pnl_1,
        bars=1,
        pnl_per_bar=expected_trade_pnl_1,
        stop_type=None,
    )
    expected_trade_pnl_2 = (FILL_PRICE_3 - FILL_PRICE_1) * (
        SHARES_2 - SHARES_1
    )
    assert_trade(
        trade=portfolio.trades[1],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_3,
        exit=FILL_PRICE_1,
        shares=SHARES_2 - SHARES_1,
        pnl=expected_trade_pnl_2,
        return_pct=expected_return_pct,
        agg_pnl=expected_trade_pnl_1 + expected_trade_pnl_2,
        bars=1,
        pnl_per_bar=expected_trade_pnl_2,
        stop_type=None,
    )


def test_cover_when_not_enough_cash():
    portfolio = Portfolio(100)
    short_order = portfolio.sell(DATE_1, SYMBOL_1, SHARES_1, 5, Decimal("4.9"))
    portfolio.incr_bars()
    buy_order = portfolio.buy(DATE_2, SYMBOL_1, SHARES_1, 1000, 1001)
    assert buy_order is None
    assert_order(
        order=short_order,
        date=DATE_1,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=Decimal("4.9"),
        fill_price=5,
        shares=SHARES_1,
        fees=0,
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
        date=DATE_1,
        symbol=SYMBOL_1,
        shares=pos.shares,
        price=5,
        type="short",
    )
    assert not portfolio.trades


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
    assert not portfolio.trades


def test_cover_when_zero_shares():
    portfolio = Portfolio(CASH)
    sell_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    portfolio.incr_bars()
    buy_order = portfolio.buy(DATE_2, SYMBOL_1, 0, FILL_PRICE_1, LIMIT_PRICE_1)
    assert sell_order is not None
    assert buy_order is None
    assert portfolio.cash == CASH + SHARES_1 * FILL_PRICE_3
    assert len(portfolio.short_positions) == 1
    assert not len(portfolio.long_positions)
    assert portfolio.orders == deque([sell_order])
    assert not len(portfolio.trades)


def test_exit_position():
    portfolio = Portfolio(CASH)
    portfolio.buy(DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1)
    portfolio.sell(DATE_1, SYMBOL_2, SHARES_2, FILL_PRICE_3, LIMIT_PRICE_3)
    assert len(portfolio.long_positions) == 1
    assert SYMBOL_1 in portfolio.long_positions
    assert len(portfolio.short_positions) == 1
    assert SYMBOL_2 in portfolio.short_positions
    portfolio.incr_bars()
    portfolio.exit_position(
        DATE_2, SYMBOL_1, buy_fill_price=0, sell_fill_price=FILL_PRICE_2
    )
    assert not portfolio.long_positions
    assert len(portfolio.short_positions) == 1
    assert SYMBOL_2 in portfolio.short_positions
    assert len(portfolio.trades) == 1
    long_pnl = (FILL_PRICE_2 - FILL_PRICE_1) * SHARES_1
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_1,
        exit=FILL_PRICE_2,
        shares=SHARES_1,
        pnl=long_pnl,
        return_pct=(FILL_PRICE_2 / FILL_PRICE_1 - 1) * 100,
        bars=1,
        pnl_per_bar=long_pnl,
        agg_pnl=long_pnl,
        stop_type=None,
    )
    assert len(portfolio.orders) == 3
    assert_order(
        order=portfolio.orders[-1],
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=FILL_PRICE_2,
        shares=SHARES_1,
        fees=0,
    )
    portfolio.exit_position(
        DATE_2, SYMBOL_2, buy_fill_price=FILL_PRICE_4, sell_fill_price=0
    )
    assert not portfolio.long_positions
    assert not portfolio.short_positions
    assert len(portfolio.trades) == 2
    short_pnl = (FILL_PRICE_3 - FILL_PRICE_4) * SHARES_2
    assert_trade(
        trade=portfolio.trades[-1],
        type="short",
        symbol=SYMBOL_2,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=FILL_PRICE_3,
        exit=FILL_PRICE_4,
        shares=SHARES_2,
        pnl=short_pnl,
        return_pct=(FILL_PRICE_3 / FILL_PRICE_4 - 1) * 100,
        bars=1,
        pnl_per_bar=short_pnl,
        agg_pnl=short_pnl + long_pnl,
        stop_type=None,
    )
    assert len(portfolio.orders) == 4
    assert_order(
        order=portfolio.orders[-1],
        date=DATE_2,
        symbol=SYMBOL_2,
        type="buy",
        limit_price=None,
        fill_price=FILL_PRICE_4,
        shares=SHARES_2,
        fees=0,
    )


def test_trigger_long_bar_stop():
    expected_fill_price = Decimal(200)
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 100],
            [SYMBOL_1, DATE_2, expected_fill_price],
        ],
        columns=["symbol", "date", "close"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.BAR,
            pos_type="long",
            percent=None,
            points=None,
            bars=1,
            fill_price=PriceType.CLOSE,
            limit_price=None,
        ),
    )
    entry_price = Decimal(100)
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    expected_pnl = (expected_fill_price - entry_price) * SHARES_1
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        short_positions_len=0,
        long_positions_len=0,
        orders=portfolio.orders,
    )
    assert len(portfolio.trades) == 1
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=entry_price,
        exit=expected_fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=(expected_fill_price / entry_price - 1) * 100,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=StopType.BAR,
    )
    assert len(portfolio.orders) == 2
    assert_order(
        order=portfolio.orders[0],
        date=DATE_1,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=entry_price,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[1],
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=expected_fill_price,
        shares=SHARES_1,
        fees=0,
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(160)), (None, Decimal(10), Decimal(190))],
)
def test_trigger_long_loss_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 100],
        ],
        columns=["symbol", "date", "low"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.LOSS,
            pos_type="long",
            percent=percent,
            points=points,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    entry_price = Decimal(200)
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    expected_pnl = (expected_fill_price - entry_price) * SHARES_1
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        short_positions_len=0,
        long_positions_len=0,
        orders=portfolio.orders,
    )
    assert len(portfolio.trades) == 1
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=entry_price,
        exit=expected_fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=(expected_fill_price / entry_price - 1) * 100,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=StopType.LOSS,
    )
    assert len(portfolio.orders) == 2
    assert_order(
        order=portfolio.orders[0],
        date=DATE_1,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=entry_price,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[1],
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=expected_fill_price,
        shares=SHARES_1,
        fees=0,
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(240)), (None, Decimal(10), Decimal(210))],
)
def test_trigger_long_profit_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 300],
        ],
        columns=["symbol", "date", "high"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.PROFIT,
            pos_type="long",
            percent=percent,
            points=points,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    entry_price = Decimal(200)
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    expected_pnl = (expected_fill_price - entry_price) * SHARES_1
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        short_positions_len=0,
        long_positions_len=0,
        orders=portfolio.orders,
    )
    assert len(portfolio.trades) == 1
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=entry_price,
        exit=expected_fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=(expected_fill_price / entry_price - 1) * 100,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=StopType.PROFIT,
    )
    assert len(portfolio.orders) == 2
    assert_order(
        order=portfolio.orders[0],
        date=DATE_1,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=entry_price,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[1],
        date=DATE_2,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=expected_fill_price,
        shares=SHARES_1,
        fees=0,
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(240)), (None, Decimal(20), Decimal(280))],
)
def test_trigger_long_trailing_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 75, 100],
            [SYMBOL_1, DATE_2, 250, 300],
            [SYMBOL_1, DATE_3, 290, 295],
            [SYMBOL_1, DATE_4, 200, 200],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index)
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.TRAILING,
            pos_type="long",
            percent=percent,
            points=points,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    entry_price = Decimal(100)
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_3, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_4, price_scope)
    expected_pnl = (expected_fill_price - entry_price) * SHARES_1
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        short_positions_len=0,
        long_positions_len=0,
        orders=portfolio.orders,
    )
    assert len(portfolio.trades) == 1
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_4,
        entry=entry_price,
        exit=expected_fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=(expected_fill_price / entry_price - 1) * 100,
        agg_pnl=expected_pnl,
        bars=3,
        pnl_per_bar=expected_pnl / 3,
        stop_type=StopType.TRAILING,
    )
    assert len(portfolio.orders) == 2
    assert_order(
        order=portfolio.orders[0],
        date=DATE_1,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=entry_price,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[1],
        date=DATE_4,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=expected_fill_price,
        shares=SHARES_1,
        fees=0,
    )


def test_trigger_short_bar_stop():
    expected_fill_price = Decimal(200)
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 100],
            [SYMBOL_1, DATE_2, expected_fill_price],
        ],
        columns=["symbol", "date", "close"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.BAR,
            pos_type="short",
            percent=None,
            points=None,
            bars=1,
            fill_price=PriceType.CLOSE,
            limit_price=None,
        ),
    )
    entry_price = Decimal(100)
    portfolio = Portfolio(CASH)
    portfolio.sell(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    expected_pnl = (entry_price - expected_fill_price) * SHARES_1
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        short_positions_len=0,
        long_positions_len=0,
        orders=portfolio.orders,
    )
    assert len(portfolio.trades) == 1
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=entry_price,
        exit=expected_fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=(entry_price / expected_fill_price - 1) * 100,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=StopType.BAR,
    )
    assert len(portfolio.orders) == 2
    assert_order(
        order=portfolio.orders[0],
        date=DATE_1,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=entry_price,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[1],
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=expected_fill_price,
        shares=SHARES_1,
        fees=0,
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(240)), (None, Decimal(10), Decimal(210))],
)
def test_trigger_short_loss_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 300],
        ],
        columns=["symbol", "date", "high"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.LOSS,
            pos_type="short",
            percent=percent,
            points=points,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    entry_price = Decimal(200)
    portfolio = Portfolio(CASH)
    portfolio.sell(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    expected_pnl = (entry_price - expected_fill_price) * SHARES_1
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        short_positions_len=0,
        long_positions_len=0,
        orders=portfolio.orders,
    )
    assert len(portfolio.trades) == 1
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=entry_price,
        exit=expected_fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=(entry_price / expected_fill_price - 1) * 100,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=StopType.LOSS,
    )
    assert len(portfolio.orders) == 2
    assert_order(
        order=portfolio.orders[0],
        date=DATE_1,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=entry_price,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[1],
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=expected_fill_price,
        shares=SHARES_1,
        fees=0,
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(160)), (None, Decimal(10), Decimal(190))],
)
def test_trigger_short_profit_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 100],
        ],
        columns=["symbol", "date", "low"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.PROFIT,
            pos_type="short",
            percent=percent,
            points=points,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    entry_price = Decimal(200)
    portfolio = Portfolio(CASH)
    portfolio.sell(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    expected_pnl = (entry_price - expected_fill_price) * SHARES_1
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        short_positions_len=0,
        long_positions_len=0,
        orders=portfolio.orders,
    )
    assert len(portfolio.trades) == 1
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=entry_price,
        exit=expected_fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=(entry_price / expected_fill_price - 1) * 100,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=StopType.PROFIT,
    )
    assert len(portfolio.orders) == 2
    assert_order(
        order=portfolio.orders[0],
        date=DATE_1,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=entry_price,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[1],
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=expected_fill_price,
        shares=SHARES_1,
        fees=0,
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(240)), (None, Decimal(20), Decimal(220))],
)
def test_trigger_short_trailing_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 350, 300],
            [SYMBOL_1, DATE_2, 230, 200],
            [SYMBOL_1, DATE_3, 215, 210],
            [SYMBOL_1, DATE_4, 400, 400],
        ],
        columns=["symbol", "date", "high", "low"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index)
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.TRAILING,
            pos_type="short",
            percent=percent,
            points=points,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    entry_price = Decimal(300)
    portfolio = Portfolio(CASH)
    portfolio.sell(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_3, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_4, price_scope)
    expected_pnl = (entry_price - expected_fill_price) * SHARES_1
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        short_positions_len=0,
        long_positions_len=0,
        orders=portfolio.orders,
    )
    assert len(portfolio.trades) == 1
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_4,
        entry=entry_price,
        exit=expected_fill_price,
        shares=SHARES_1,
        pnl=expected_pnl,
        return_pct=(entry_price / expected_fill_price - 1) * 100,
        agg_pnl=expected_pnl,
        bars=3,
        pnl_per_bar=expected_pnl / 3,
        stop_type=StopType.TRAILING,
    )
    assert len(portfolio.orders) == 2
    assert_order(
        order=portfolio.orders[0],
        date=DATE_1,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=entry_price,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[1],
        date=DATE_4,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=expected_fill_price,
        shares=SHARES_1,
        fees=0,
    )


@pytest.mark.parametrize(
    "stop_type", [StopType.LOSS, StopType.PROFIT, StopType.TRAILING]
)
def test_long_stop_limit_price(stop_type):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 75, 200, 100],
            [SYMBOL_1, DATE_2, 250, 400, 300],
            [SYMBOL_1, DATE_3, 290, 395, 295],
            [SYMBOL_1, DATE_4, 200, 300, 200],
        ],
        columns=["symbol", "date", "low", "high", "close"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index)
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=stop_type,
            pos_type="long",
            percent=10,
            points=20,
            bars=None,
            fill_price=None,
            limit_price=500,
        ),
    )
    entry_price = Decimal(100)
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_3, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_4, price_scope)
    assert portfolio.symbols == set(["SPY"])
    assert len(portfolio.long_positions) == 1
    assert not portfolio.short_positions
    assert not portfolio.trades
    assert len(portfolio.orders) == 1


@pytest.mark.parametrize(
    "stop_type", [StopType.LOSS, StopType.PROFIT, StopType.TRAILING]
)
def test_short_stop_limit_price(stop_type):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200, 350, 300],
            [SYMBOL_1, DATE_2, 100, 230, 200],
            [SYMBOL_1, DATE_3, 110, 215, 210],
            [SYMBOL_1, DATE_4, 300, 400, 400],
        ],
        columns=["symbol", "date", "low", "high", "close"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index)
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=stop_type,
            pos_type="short",
            percent=20,
            points=None,
            bars=None,
            fill_price=None,
            limit_price=50,
        ),
    )
    entry_price = Decimal(300)
    portfolio = Portfolio(CASH)
    portfolio.sell(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price,
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_3, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_4, price_scope)
    assert portfolio.symbols == set(["SPY"])
    assert not portfolio.long_positions
    assert len(portfolio.short_positions) == 1
    assert not portfolio.trades
    assert len(portfolio.orders) == 1


def test_check_stops_when_multiple_entries():
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 300],
            [SYMBOL_1, DATE_3, 200],
            [SYMBOL_1, DATE_4, 100],
        ],
        columns=["symbol", "date", "low"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index)
    entry_price_1 = Decimal(200)
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        entry_price_1,
        limit_price=None,
        stops=(
            Stop(
                id=1,
                symbol=SYMBOL_1,
                stop_type=StopType.LOSS,
                pos_type="long",
                percent=None,
                points=100,
                bars=None,
                fill_price=None,
                limit_price=None,
            ),
        ),
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    entry_price_2 = Decimal(300)
    portfolio.buy(
        DATE_2,
        SYMBOL_1,
        SHARES_2,
        entry_price_2,
        limit_price=None,
        stops=(
            Stop(
                id=2,
                symbol=SYMBOL_1,
                stop_type=StopType.LOSS,
                pos_type="long",
                percent=None,
                points=100,
                bars=None,
                fill_price=None,
                limit_price=None,
            ),
        ),
    )
    portfolio.incr_bars()
    sym_end_index[SYMBOL_1] += 1
    portfolio.check_stops(DATE_3, price_scope)
    portfolio.incr_bars()
    sym_end_index[SYMBOL_1] += 1
    portfolio.check_stops(DATE_4, price_scope)
    expected_fill_price_1 = Decimal(100)
    expected_fill_price_2 = Decimal(200)
    expected_pnl_1 = (expected_fill_price_1 - entry_price_1) * SHARES_1
    expected_pnl_2 = (expected_fill_price_2 - entry_price_2) * SHARES_2
    expected_pnl = expected_pnl_1 + expected_pnl_2
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        short_positions_len=0,
        long_positions_len=0,
        orders=portfolio.orders,
    )
    assert len(portfolio.trades) == 2
    assert_trade(
        trade=portfolio.trades[0],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_2,
        exit_date=DATE_3,
        entry=entry_price_2,
        exit=expected_fill_price_2,
        shares=SHARES_2,
        pnl=expected_pnl_2,
        return_pct=(expected_fill_price_2 / entry_price_2 - 1) * 100,
        agg_pnl=expected_pnl_2,
        bars=1,
        pnl_per_bar=expected_pnl_2,
        stop_type=StopType.LOSS,
    )
    assert_trade(
        trade=portfolio.trades[1],
        type="long",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_4,
        entry=entry_price_1,
        exit=expected_fill_price_1,
        shares=SHARES_1,
        pnl=expected_pnl_1,
        return_pct=(expected_fill_price_1 / entry_price_1 - 1) * 100,
        agg_pnl=expected_pnl,
        bars=3,
        pnl_per_bar=expected_pnl_1 / 3,
        stop_type=StopType.LOSS,
    )
    assert len(portfolio.orders) == 4
    assert_order(
        order=portfolio.orders[0],
        date=DATE_1,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=entry_price_1,
        shares=SHARES_1,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[1],
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=entry_price_2,
        shares=SHARES_2,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[2],
        date=DATE_3,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=expected_fill_price_2,
        shares=SHARES_2,
        fees=0,
    )
    assert_order(
        order=portfolio.orders[3],
        date=DATE_4,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=None,
        fill_price=expected_fill_price_1,
        shares=SHARES_1,
        fees=0,
    )


def test_check_stops_when_multiple_stops_hit():
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 100],
        ],
        columns=["symbol", "date", "low"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 3}
    price_scope = PriceScope(ColumnScope(df), sym_end_index)
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        Decimal(200),
        limit_price=None,
        stops=(
            Stop(
                id=1,
                symbol=SYMBOL_1,
                stop_type=StopType.LOSS,
                pos_type="long",
                percent=None,
                points=10,
                bars=None,
                fill_price=None,
                limit_price=None,
            ),
            Stop(
                id=2,
                symbol=SYMBOL_1,
                stop_type=StopType.TRAILING,
                pos_type="long",
                percent=None,
                points=20,
                bars=None,
                fill_price=None,
                limit_price=None,
            ),
        ),
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    assert not portfolio.symbols
    assert not portfolio.long_positions
    assert not portfolio.short_positions
    assert len(portfolio.trades) == 1
    assert len(portfolio.orders) == 2


def test_remove_stop():
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 100],
        ],
        columns=["symbol", "date", "low"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.LOSS,
            pos_type="long",
            percent=None,
            points=10,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        Decimal(200),
        limit_price=None,
        stops=stops,
    )
    assert portfolio.remove_stop(1)
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    assert len(portfolio.long_positions) == 1
    assert portfolio.symbols == set([SYMBOL_1])
    assert not len(portfolio.trades)


@pytest.mark.parametrize("stop_type", [StopType.LOSS, None])
def test_remove_stops_when_symbol(stop_type):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 100],
        ],
        columns=["symbol", "date", "low"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.LOSS,
            pos_type="long",
            percent=None,
            points=10,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        Decimal(200),
        limit_price=None,
        stops=stops,
    )
    portfolio.remove_stops("SPY", stop_type)
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    assert len(portfolio.long_positions) == 1
    pos = portfolio.long_positions[SYMBOL_1]
    assert len(pos.entries) == 1
    assert not pos.entries[0].stops
    assert portfolio.symbols == set([SYMBOL_1])
    assert not len(portfolio.trades)


@pytest.mark.parametrize("stop_type", [StopType.LOSS, None])
def test_remove_stops_when_position(stop_type):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 100],
        ],
        columns=["symbol", "date", "low"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.LOSS,
            pos_type="long",
            percent=None,
            points=10,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        Decimal(200),
        limit_price=None,
        stops=stops,
    )
    portfolio.remove_stops(portfolio.long_positions["SPY"], stop_type)
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    assert len(portfolio.long_positions) == 1
    pos = portfolio.long_positions[SYMBOL_1]
    assert len(pos.entries) == 1
    assert not pos.entries[0].stops
    assert portfolio.symbols == set([SYMBOL_1])
    assert not len(portfolio.trades)


@pytest.mark.parametrize("stop_type", [StopType.LOSS, None])
def test_remove_stops_when_entry(stop_type):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200],
            [SYMBOL_1, DATE_2, 100],
        ],
        columns=["symbol", "date", "low"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)})
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.LOSS,
            pos_type="long",
            percent=None,
            points=10,
            bars=None,
            fill_price=None,
            limit_price=None,
        ),
    )
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        Decimal(200),
        limit_price=None,
        stops=stops,
    )
    portfolio.remove_stops(
        portfolio.long_positions["SPY"].entries[0], stop_type
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    assert len(portfolio.long_positions) == 1
    pos = portfolio.long_positions[SYMBOL_1]
    assert len(pos.entries) == 1
    assert not pos.entries[0].stops
    assert portfolio.symbols == set([SYMBOL_1])
    assert not len(portfolio.trades)


def test_incr_ids():
    portfolio = Portfolio(CASH)
    buy_order = portfolio.buy(
        DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1
    )
    assert buy_order.id == 1
    assert list(portfolio.long_positions.values())[0].entries[0].id == 1
    sell_order = portfolio.sell(
        DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_3, LIMIT_PRICE_3
    )
    assert sell_order.id == 2
    assert portfolio.trades[0].id == 1


def test_incr_bars():
    portfolio = Portfolio(CASH)
    portfolio.buy(DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1)
    portfolio.incr_bars()
    portfolio.buy(DATE_2, SYMBOL_1, SHARES_2, FILL_PRICE_2)
    portfolio.sell(DATE_2, SYMBOL_2, SHARES_1, FILL_PRICE_3)
    portfolio.incr_bars()
    portfolio.incr_bars()
    assert len(portfolio.long_positions) == 1
    assert len(portfolio.short_positions) == 1
    long_pos = portfolio.long_positions[SYMBOL_1]
    assert long_pos.bars == 3
    assert len(long_pos.entries) == 2
    assert long_pos.entries[0].bars == 3
    assert long_pos.entries[1].bars == 2
    short_pos = portfolio.short_positions[SYMBOL_2]
    assert short_pos.bars == 2
    assert len(short_pos.entries) == 1
    assert short_pos.entries[0].bars == 2
