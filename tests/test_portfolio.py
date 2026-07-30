"""Unit tests for portfolio.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import itertools
import numpy as np
import pandas as pd
import pytest
import re
from collections import deque
from decimal import Decimal
from pybroker.common import (
    FeeMode,
    PositionMode,
    PriceType,
    StopType,
    to_decimal,
)
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
    type,
    symbol,
    date,
    created=None,
    order_type="market",
    intent=None,
    shares=None,
    limit_price=None,
    fill_price=None,
    fees=None,
):
    assert order.type == type
    assert order.symbol == symbol
    assert order.date == date
    assert order.created == created
    assert order.order_type == order_type
    assert order.intent == intent
    assert order.shares == shares
    assert order.limit_price == limit_price
    assert order.fill_price == fill_price
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
    mae,
    mfe,
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
    assert trade.mae == mae
    assert trade.mfe == mfe


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
        intent="buy_to_open",
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
        intent="buy_to_open",
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
        intent="buy_to_open",
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
        intent="buy_to_open",
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
        intent="buy_to_open",
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
    )


def test_buy_when_existing_short_and_not_enough_cash():
    portfolio = Portfolio(100)
    entry_price = Decimal(5)
    entry_limit = Decimal("4.9")
    exit_price = Decimal(200)
    exit_limit = Decimal(201)
    short_shares = Decimal(20)
    short_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, entry_price, entry_limit
    )
    portfolio.incr_bars()
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_1, exit_price, exit_limit
    )
    expected_pnl = (entry_price - exit_price) * short_shares
    assert_order(
        order=short_order,
        date=DATE_1,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=entry_limit,
        fill_price=5,
        shares=short_shares,
        fees=0,
        intent="sell_to_open",
    )
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=exit_limit,
        fill_price=exit_price,
        shares=short_shares,
        fees=0,
        intent="buy_to_close",
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=100 + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        orders=[short_order, buy_order],
        short_positions_len=0,
        long_positions_len=0,
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
        shares=short_shares,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
        mae=entry_price - exit_price,
        mfe=0,
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
        intent="sell_to_close",
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
        mae=0,
        mfe=fill_price - FILL_PRICE_1,
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
        intent="sell_to_close",
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
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
        intent="buy_to_open",
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
        intent="sell_to_close",
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
    )


def calc_fees(fee_info):
    assert fee_info.symbol == SYMBOL_1
    assert fee_info.shares == SHARES_1
    if fee_info.order_type == "buy":
        assert fee_info.fill_price == FILL_PRICE_1
    else:
        assert fee_info.fill_price == FILL_PRICE_3
    return Decimal("9.99")


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
        (calc_fees, Decimal("9.99"), Decimal("9.99")),
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
        intent="buy_to_open",
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
        intent="sell_to_close",
    )
    assert portfolio.fees == expected_buy_fees + expected_sell_fees


def test_subtract_fees():
    portfolio = Portfolio(3, FeeMode.PER_ORDER, fee_amount=1)
    order = portfolio.buy(DATE_1, SYMBOL_1, shares=1, fill_price=1)
    assert_order(
        order=order,
        date=DATE_1,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=1,
        shares=1,
        fees=1,
        intent="buy_to_open",
    )
    assert portfolio.cash == 1
    order = portfolio.buy(DATE_2, SYMBOL_1, shares=1, fill_price=1)
    assert_order(
        order=order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=None,
        fill_price=1,
        shares=1,
        fees=1,
        intent="buy_to_open",
    )
    assert portfolio.cash == -1
    order = portfolio.buy(DATE_2, SYMBOL_1, shares=1, fill_price=1)
    assert order is None
    assert portfolio.cash == -1


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
        intent="sell_to_close",
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
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
        intent="sell_to_close",
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
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
        intent="sell_to_open",
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - SHARES_1 * fill_price,
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
        intent="sell_to_open",
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - SHARES_1 * FILL_PRICE_3 - SHARES_2 * FILL_PRICE_4,
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
        intent="sell_to_open",
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH - SHARES_1 * FILL_PRICE_3 - SHARES_2 * FILL_PRICE_4,
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
        intent="sell_to_open",
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl - expected_shares * FILL_PRICE_3,
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
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
        cash=CASH - SHARES_1 * FILL_PRICE_3,
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
        intent="buy_to_close",
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
        mae=0,
        mfe=FILL_PRICE_3 - fill_price,
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
        intent="buy_to_close",
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_pnl - expected_shares * FILL_PRICE_3,
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
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
        intent="buy_to_close",
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=CASH + expected_order_pnl - expected_shares * FILL_PRICE_3,
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
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
        mae=0,
        mfe=FILL_PRICE_3 - FILL_PRICE_1,
    )


def test_cover_when_not_enough_cash():
    portfolio = Portfolio(100)
    sell_fill_price = 5
    sell_limit_price = Decimal("4.9")
    buy_fill_price = Decimal(1000)
    buy_limit_price = 1001
    short_shares = Decimal(20)
    short_order = portfolio.sell(
        DATE_1, SYMBOL_1, SHARES_1, sell_fill_price, sell_limit_price
    )
    portfolio.incr_bars()
    buy_order = portfolio.buy(
        DATE_2, SYMBOL_1, SHARES_1, buy_fill_price, buy_limit_price
    )
    expected_pnl = (sell_fill_price - buy_fill_price) * short_shares
    expected_return_pct = (sell_fill_price / buy_fill_price - 1) * 100
    assert_order(
        order=short_order,
        date=DATE_1,
        symbol=SYMBOL_1,
        type="sell",
        limit_price=sell_limit_price,
        fill_price=sell_fill_price,
        shares=short_shares,
        fees=0,
        intent="sell_to_open",
    )
    assert_order(
        order=buy_order,
        date=DATE_2,
        symbol=SYMBOL_1,
        type="buy",
        limit_price=buy_limit_price,
        fill_price=buy_fill_price,
        shares=short_shares,
        fees=0,
        intent="buy_to_close",
    )
    assert_portfolio(
        portfolio=portfolio,
        cash=100 + expected_pnl,
        pnl=expected_pnl,
        symbols=set(),
        orders=[short_order, buy_order],
        short_positions_len=0,
        long_positions_len=0,
    )
    assert_trade(
        trade=portfolio.trades[0],
        type="short",
        symbol=SYMBOL_1,
        entry_date=DATE_1,
        exit_date=DATE_2,
        entry=sell_fill_price,
        exit=buy_fill_price,
        shares=short_shares,
        pnl=expected_pnl,
        return_pct=expected_return_pct,
        agg_pnl=expected_pnl,
        bars=1,
        pnl_per_bar=expected_pnl,
        stop_type=None,
        mae=sell_fill_price - buy_fill_price,
        mfe=0,
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
        cash=CASH - SHARES_1 * FILL_PRICE_3,
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
    assert portfolio.cash == CASH - SHARES_1 * FILL_PRICE_3
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
        mae=0,
        mfe=FILL_PRICE_2 - FILL_PRICE_1,
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
        intent="sell_to_close",
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
        mae=FILL_PRICE_3 - FILL_PRICE_4,
        mfe=0,
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
        intent="buy_to_close",
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
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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
        mae=0,
        mfe=expected_fill_price - entry_price,
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
        intent="buy_to_open",
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
        order_type="stop_bar",
        intent="sell_to_close",
    )


@pytest.mark.parametrize("insertion_order", [(1, 2), (2, 1)])
def test_check_stops_evaluates_in_stop_id_order(insertion_order):
    # Stops reach the portfolio as a frozenset. Stop always carries None
    # fields and hash(None) is address-derived, so iteration order varies per
    # process and no seed can pin it. The lower stop id must win regardless of
    # the order the stops were inserted in.
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 100, 100],
            [SYMBOL_1, DATE_2, 90, 110],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    col_scope = ColumnScope(df)
    sym_end_index = {SYMBOL_1: len(df)}
    price_scope = PriceScope(col_scope, sym_end_index, True)
    loss_stop = Stop(
        id=1,
        symbol=SYMBOL_1,
        stop_type=StopType.LOSS,
        pos_type="long",
        percent=Decimal(5),
        points=None,
        bars=None,
        fill_price=None,
        limit_price=None,
        exit_price=None,
    )
    profit_stop = Stop(
        id=2,
        symbol=SYMBOL_1,
        stop_type=StopType.PROFIT,
        pos_type="long",
        percent=Decimal(5),
        points=None,
        bars=None,
        fill_price=None,
        limit_price=None,
        exit_price=None,
    )
    by_id = {1: loss_stop, 2: profit_stop}
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        Decimal(100),
        limit_price=None,
        stops=frozenset(by_id[i] for i in insertion_order),
    )
    portfolio.check_stops(DATE_2, price_scope)
    assert len(portfolio.trades) == 1
    # The bar straddles both levels, so whichever is evaluated first wins.
    # Stop 1 is the stop loss, exiting at 95.
    assert portfolio.trades[0].stop == StopType.LOSS.value
    assert portfolio.trades[0].exit == 95


@pytest.mark.parametrize("fill_price", [Decimal(0), Decimal(-5)])
def test_trigger_bar_stop_when_invalid_fill_price_then_error(fill_price):
    # A bar stop's fill price comes from the user's sell_fill_price and never
    # reaches _verify_input, so without a guard a zero or negative silently
    # books a completed trade.
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 100, 100],
            [SYMBOL_1, DATE_2, 90, 110],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    col_scope = ColumnScope(df)
    sym_end_index = {SYMBOL_1: len(df)}
    price_scope = PriceScope(col_scope, sym_end_index, True)
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.BAR,
            pos_type="long",
            percent=None,
            points=None,
            bars=1,
            fill_price=fill_price,
            limit_price=None,
            exit_price=None,
        ),
    )
    portfolio = Portfolio(CASH)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        Decimal(100),
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    with pytest.raises(ValueError, match=re.escape("must be > 0")):
        portfolio.check_stops(DATE_2, price_scope)


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(160)), (None, Decimal(10), Decimal(190))],
)
def test_trigger_long_loss_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200, 300],
            [SYMBOL_1, DATE_2, 100, 200],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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
        mae=expected_fill_price - entry_price,
        mfe=0,
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
        intent="buy_to_open",
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
        order_type="stop_loss",
        intent="sell_to_close",
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(240)), (None, Decimal(10), Decimal(210))],
)
def test_trigger_long_profit_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 100, 200],
            [SYMBOL_1, DATE_2, 200, 300],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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
        mae=0,
        mfe=expected_fill_price - entry_price,
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
        intent="buy_to_open",
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
        order_type="stop_profit",
        intent="sell_to_close",
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(200)), (None, Decimal(20), Decimal(200))],
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
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
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
            exit_price=None,
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
        mae=0,
        mfe=expected_fill_price - entry_price,
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
        intent="buy_to_open",
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
        order_type="stop_trailing",
        intent="sell_to_close",
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
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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
        mae=entry_price - expected_fill_price,
        mfe=0,
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
        intent="sell_to_open",
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
        order_type="stop_bar",
        intent="buy_to_close",
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(240)), (None, Decimal(10), Decimal(210))],
)
def test_trigger_short_loss_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 100, 200],
            [SYMBOL_1, DATE_2, 200, 300],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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
        mae=entry_price - expected_fill_price,
        mfe=0,
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
        intent="sell_to_open",
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
        order_type="stop_loss",
        intent="buy_to_close",
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(160)), (None, Decimal(10), Decimal(190))],
)
def test_trigger_short_profit_stop(percent, points, expected_fill_price):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200, 300],
            [SYMBOL_1, DATE_2, 100, 200],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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
        mae=0,
        mfe=entry_price - expected_fill_price,
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
        intent="sell_to_open",
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
        order_type="stop_profit",
        intent="buy_to_close",
    )


@pytest.mark.parametrize(
    "percent, points, expected_fill_price",
    [(Decimal(20), None, Decimal(400)), (None, Decimal(20), Decimal(400))],
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
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
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
            exit_price=None,
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
        mae=entry_price - expected_fill_price,
        mfe=0,
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
        intent="sell_to_open",
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
        order_type="stop_trailing",
        intent="buy_to_close",
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
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
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
            exit_price=None,
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
def test_long_stop_exit_price(stop_type):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 75, 200, 100],
            [SYMBOL_1, DATE_2, 250, 400, 300],
            [SYMBOL_1, DATE_3, 290, 395, 295],
            [SYMBOL_1, DATE_4, 200, 300, 200],
        ],
        columns=["symbol", "date", "open", "high", "close"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
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
            exit_price=PriceType.OPEN,
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
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
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
            exit_price=None,
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


@pytest.mark.parametrize(
    "stop_type", [StopType.LOSS, StopType.PROFIT, StopType.TRAILING]
)
def test_short_stop_exit_price(stop_type):
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200, 350, 300],
            [SYMBOL_1, DATE_2, 100, 230, 200],
            [SYMBOL_1, DATE_3, 110, 215, 210],
            [SYMBOL_1, DATE_4, 300, 400, 400],
        ],
        columns=["symbol", "date", "low", "open", "close"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
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
            exit_price=PriceType.OPEN,
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
            [SYMBOL_1, DATE_1, 200, 300],
            [SYMBOL_1, DATE_2, 300, 400],
            [SYMBOL_1, DATE_3, 200, 300],
            [SYMBOL_1, DATE_4, 100, 200],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
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
                exit_price=None,
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
                exit_price=None,
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
        mae=expected_fill_price_1 - entry_price_1,
        mfe=0,
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
        mae=expected_fill_price_2 - entry_price_2,
        mfe=0,
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
        intent="buy_to_open",
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
        intent="buy_to_open",
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
        order_type="stop_loss",
        intent="sell_to_close",
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
        order_type="stop_loss",
        intent="sell_to_close",
    )


def test_check_stops_when_multiple_stops_hit():
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200, 300],
            [SYMBOL_1, DATE_2, 100, 200],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 3}
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
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
                exit_price=None,
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
                exit_price=None,
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
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
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
            exit_price=None,
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


def test_long_stop_when_no_pos():
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 75, 200, 100],
            [SYMBOL_1, DATE_2, 250, 400, 300],
            [SYMBOL_1, DATE_3, 290, 395, 295],
            [SYMBOL_1, DATE_4, 200, 300, 200],
        ],
        columns=["symbol", "date", "open", "high", "close"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.LOSS,
            pos_type="long",
            percent=10,
            points=20,
            bars=None,
            fill_price=None,
            limit_price=500,
            exit_price=PriceType.OPEN,
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
    portfolio.sell(DATE_1, SYMBOL_1, SHARES_1, entry_price)
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_3, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_4, price_scope)
    assert len(portfolio.orders) == 2


def test_short_stop_when_no_pos():
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200, 350, 300],
            [SYMBOL_1, DATE_2, 100, 230, 200],
            [SYMBOL_1, DATE_3, 110, 215, 210],
            [SYMBOL_1, DATE_4, 300, 400, 400],
        ],
        columns=["symbol", "date", "low", "open", "close"],
    )
    df = df.set_index(["symbol", "date"])
    sym_end_index = {SYMBOL_1: 2}
    price_scope = PriceScope(ColumnScope(df), sym_end_index, True)
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.LOSS,
            pos_type="short",
            percent=20,
            points=None,
            bars=None,
            fill_price=None,
            limit_price=50,
            exit_price=PriceType.OPEN,
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
    portfolio.buy(DATE_1, SYMBOL_1, SHARES_1, entry_price)
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_3, price_scope)
    sym_end_index[SYMBOL_1] += 1
    portfolio.incr_bars()
    portfolio.check_stops(DATE_4, price_scope)
    assert len(portfolio.orders) == 2


def test_capture_stops():
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, 200, 300],
            [SYMBOL_1, DATE_2, 200, 300],
        ],
        columns=["symbol", "date", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    price_scope = PriceScope(ColumnScope(df), {SYMBOL_1: len(df)}, True)
    stops = (
        Stop(
            id=1,
            symbol=SYMBOL_1,
            stop_type=StopType.LOSS,
            pos_type="long",
            percent=5,
            points=None,
            bars=None,
            fill_price=None,
            limit_price=None,
            exit_price=None,
        ),
        Stop(
            id=2,
            symbol=SYMBOL_1,
            stop_type=StopType.TRAILING,
            pos_type="long",
            percent=None,
            points=5,
            bars=None,
            fill_price=None,
            limit_price=Decimal(200),
            exit_price=None,
        ),
        Stop(
            id=3,
            symbol=SYMBOL_1,
            stop_type=StopType.BAR,
            pos_type="long",
            percent=None,
            points=None,
            bars=5,
            fill_price=None,
            limit_price=None,
            exit_price=Decimal(200),
        ),
        Stop(
            id=4,
            symbol=SYMBOL_1,
            stop_type=StopType.PROFIT,
            pos_type="long",
            percent=10,
            points=None,
            bars=None,
            fill_price=Decimal(210),
            limit_price=None,
            exit_price=None,
        ),
    )
    portfolio = Portfolio(CASH, record_stops=True)
    portfolio.buy(
        DATE_1,
        SYMBOL_1,
        SHARES_1,
        fill_price=Decimal(200),
        limit_price=None,
        stops=stops,
    )
    portfolio.incr_bars()
    portfolio.check_stops(DATE_2, price_scope)
    stops = {stop.stop_id: stop for stop in portfolio._stop_records}

    # Stop 4 exits the entry, so stop 3 is never reached and is not recorded.
    # Stops that a sibling preempts produce no row, in either check_stops loop.
    assert len(stops) == 3
    assert 3 not in stops
    assert stops[1].date == DATE_2
    assert stops[1].symbol == SYMBOL_1
    assert stops[1].stop_type == StopType.LOSS.value
    assert stops[1].pos_type == "long"
    assert stops[1].curr_value == 190
    assert stops[1].bars is None
    assert stops[1].limit_price is None
    assert stops[1].percent == 5
    assert stops[1].points is None
    assert stops[1].curr_bars is None
    assert stops[1].exit_price is None
    assert stops[1].fill_price is None

    assert stops[2].date == DATE_2
    assert stops[2].symbol == SYMBOL_1
    assert stops[2].stop_type == StopType.TRAILING.value
    assert stops[2].pos_type == "long"
    assert stops[2].curr_value == 295
    assert stops[2].bars is None
    assert stops[2].limit_price == 200
    assert stops[2].percent is None
    assert stops[2].points == 5
    assert stops[2].curr_bars is None
    assert stops[2].exit_price is None
    assert stops[2].fill_price is None

    assert stops[4].date == DATE_2
    assert stops[4].symbol == SYMBOL_1
    assert stops[4].stop_type == StopType.PROFIT.value
    assert stops[4].pos_type == "long"
    assert stops[4].curr_value == 220
    assert stops[4].bars is None
    assert stops[4].limit_price is None
    assert stops[4].percent == 10
    assert stops[4].points is None
    assert stops[4].curr_bars is None
    assert stops[4].exit_price is None
    assert stops[4].fill_price == 220


def test_win_loss_rate():
    portfolio = Portfolio(CASH)
    portfolio.buy(DATE_1, SYMBOL_1, SHARES_1, FILL_PRICE_1, LIMIT_PRICE_1)
    portfolio.buy(DATE_1, SYMBOL_2, SHARES_1, FILL_PRICE_3, limit_price=None)
    portfolio.incr_bars()
    portfolio.sell(DATE_2, SYMBOL_1, SHARES_1, FILL_PRICE_2, limit_price=None)
    portfolio.sell(DATE_2, SYMBOL_2, SHARES_1, FILL_PRICE_2, LIMIT_PRICE_1)
    assert len(portfolio.trades) == 2
    assert portfolio.win_rate == Decimal("0.5")
    assert portfolio.loss_rate == Decimal("0.5")


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


def test_capture_bar_when_short_position():
    cash = 100_000
    fill_price = Decimal("16.72")
    shares = 100
    close_price = Decimal("16.7")
    low_price = Decimal("15.00")
    high_price = Decimal("18.00")
    portfolio = Portfolio(
        cash, record_portfolio_bars=True, record_position_bars=True
    )
    portfolio.sell(DATE_1, SYMBOL_1, shares, fill_price)
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, close_price, low_price, high_price],
        ],
        columns=["symbol", "date", "close", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    portfolio.capture_bar(DATE_1, ColumnScope(df), {SYMBOL_1: 1})
    pos = portfolio.short_positions[SYMBOL_1]
    assert pos.pnl == (fill_price - close_price) * shares
    assert pos.equity == 0
    assert pos.margin == close_price * shares
    assert pos.market_value == pos.margin + pos.pnl
    assert pos.close == close_price
    assert pos.entries[0].mae == fill_price - high_price
    assert pos.entries[0].mfe == fill_price - low_price
    assert len(portfolio.bars) == 1
    bar = portfolio.bars[0]
    assert bar.date == DATE_1
    assert bar.cash == cash - shares * fill_price
    assert bar.equity == cash
    assert bar.margin == close_price * shares
    assert bar.pnl == bar.equity - cash
    assert bar.unrealized_pnl == bar.market_value - bar.equity
    assert bar.market_value == cash + (fill_price - close_price) * shares
    assert bar.fees == 0
    assert len(portfolio.position_bars) == 1
    pos_bar = portfolio.position_bars[0]
    assert pos_bar.symbol == SYMBOL_1
    assert pos_bar.date == DATE_1
    assert pos_bar.long_shares == 0
    assert pos_bar.short_shares == shares
    assert pos_bar.close == close_price
    assert (
        pos_bar.equity
        == shares * fill_price + (fill_price - close_price) * shares
    )
    assert pos_bar.margin == close_price * shares
    assert pos_bar.unrealized_pnl == (fill_price - close_price) * shares
    assert pos_bar.market_value == pos_bar.margin + pos_bar.unrealized_pnl


def test_capture_bar_when_short_position_and_no_price_data():
    cash = 100_000
    fill_price = Decimal("16.72")
    shares = 100
    portfolio = Portfolio(cash, record_portfolio_bars=True)
    portfolio.sell(DATE_1, SYMBOL_1, shares, fill_price)
    df = pd.DataFrame(
        [], columns=["symbol", "date", "close", "low", "high"]
    ).set_index(["symbol", "date"])
    portfolio.capture_bar(DATE_1, ColumnScope(df), {SYMBOL_1: 0})
    bar = portfolio.bars[0]
    assert bar.cash == cash - shares * fill_price
    assert bar.equity == cash
    assert bar.market_value == cash
    assert bar.unrealized_pnl == 0
    assert not portfolio.position_bars


def test_capture_bar_when_short_position_and_leverage():
    fill_price = Decimal("16.72")
    shares = 100
    close_price = Decimal("16.7")
    portfolio = Portfolio(
        CASH, leverage=2.0, record_portfolio_bars=True, record_stops=False
    )
    portfolio.sell(DATE_1, SYMBOL_1, shares, fill_price)
    df = pd.DataFrame(
        [[SYMBOL_1, DATE_1, close_price, Decimal("15.00"), Decimal("18.00")]],
        columns=["symbol", "date", "close", "low", "high"],
    ).set_index(["symbol", "date"])
    portfolio.capture_bar(DATE_1, ColumnScope(df), {SYMBOL_1: 1})
    entry_notional = shares * fill_price
    bar = portfolio.bars[0]
    assert bar.cash == CASH - entry_notional / 2
    assert bar.margin_loan == entry_notional / 2
    assert bar.net_cash_balance == CASH - entry_notional
    assert bar.equity == CASH
    assert bar.market_value == CASH + (fill_price - close_price) * shares


def test_capture_bar_when_long_position():
    cash = 100_000
    fill_price = Decimal("16.72")
    shares = 100
    close_price = Decimal("16.7")
    low_price = Decimal("15.00")
    high_price = Decimal("18.00")
    portfolio = Portfolio(
        cash, record_portfolio_bars=True, record_position_bars=True
    )
    portfolio.buy(DATE_1, SYMBOL_1, shares, fill_price)
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, close_price, low_price, high_price],
        ],
        columns=["symbol", "date", "close", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    portfolio.capture_bar(DATE_1, ColumnScope(df), {SYMBOL_1: 1})
    pos = portfolio.long_positions[SYMBOL_1]
    assert pos.pnl == (close_price - fill_price) * shares
    assert pos.equity == close_price * shares
    assert pos.margin == 0
    assert pos.market_value == pos.equity
    assert pos.close == close_price
    assert pos.entries[0].mae == low_price - fill_price
    assert pos.entries[0].mfe == high_price - fill_price
    assert len(portfolio.bars) == 1
    bar = portfolio.bars[0]
    assert bar.date == DATE_1
    assert bar.cash == cash - shares * fill_price
    assert bar.equity == cash + bar.pnl
    assert bar.margin == 0
    assert bar.pnl == (close_price - fill_price) * shares
    assert bar.market_value == bar.equity
    assert bar.fees == 0
    assert len(portfolio.position_bars) == 1
    pos_bar = portfolio.position_bars[0]
    assert pos_bar.symbol == SYMBOL_1
    assert pos_bar.date == DATE_1
    assert pos_bar.long_shares == shares
    assert pos_bar.short_shares == 0
    assert pos_bar.close == close_price
    assert pos_bar.equity == shares * close_price
    assert pos_bar.margin == 0
    assert pos_bar.unrealized_pnl == (close_price - fill_price) * shares
    assert pos_bar.market_value == pos_bar.equity


def test_mae_mfe_when_short_position():
    cash = 100_000
    fill_price = Decimal("16.72")
    shares = 100
    close_price = Decimal("16.7")
    low_price = Decimal("15.00")
    high_price = Decimal("18.00")
    portfolio = Portfolio(cash)
    portfolio.sell(DATE_1, SYMBOL_1, shares, fill_price)
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, close_price, low_price, high_price],
        ],
        columns=["symbol", "date", "close", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    portfolio.capture_bar(DATE_1, ColumnScope(df), {SYMBOL_1: 1})
    portfolio.buy(DATE_1, SYMBOL_1, shares, fill_price)
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].mae == fill_price - high_price
    assert portfolio.trades[0].mfe == fill_price - low_price


def test_mae_mfe_when_long_position():
    cash = 100_000
    fill_price = Decimal("16.72")
    shares = 100
    close_price = Decimal("16.7")
    low_price = Decimal("15.00")
    high_price = Decimal("18.00")
    portfolio = Portfolio(cash)
    portfolio.buy(DATE_1, SYMBOL_1, shares, fill_price)
    df = pd.DataFrame(
        [
            [SYMBOL_1, DATE_1, close_price, low_price, high_price],
        ],
        columns=["symbol", "date", "close", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    portfolio.capture_bar(DATE_1, ColumnScope(df), {SYMBOL_1: 1})
    portfolio.sell(DATE_1, SYMBOL_1, shares, fill_price)
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].mae == low_price - fill_price
    assert portfolio.trades[0].mfe == high_price - fill_price


def test_long_only_mode():
    cash = 100_000
    portfolio = Portfolio(cash, position_mode=PositionMode.LONG_ONLY)
    portfolio.buy(DATE_1, SYMBOL_1, 100, FILL_PRICE_1)
    portfolio.sell(DATE_2, SYMBOL_1, 200, FILL_PRICE_1)
    assert not portfolio.long_positions
    assert not portfolio.short_positions
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].shares == 100


def test_short_only_mode():
    cash = 100_000
    portfolio = Portfolio(cash, position_mode=PositionMode.SHORT_ONLY)
    portfolio.sell(DATE_1, SYMBOL_1, 100, FILL_PRICE_1)
    portfolio.buy(DATE_2, SYMBOL_1, 200, FILL_PRICE_1)
    assert not portfolio.long_positions
    assert not portfolio.short_positions
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].shares == 100


def test_buy_when_leverage_allows_borrowed_funds():
    portfolio = Portfolio(CASH, leverage=2.0)
    fill_price = Decimal(100)
    shares = Decimal(2000)
    order = portfolio.buy(DATE_1, SYMBOL_1, shares, fill_price)
    assert order is not None
    assert order.shares == shares
    assert portfolio.cash == CASH - shares * fill_price / 2
    assert portfolio.margin_loan == shares * fill_price / 2


def test_buy_when_leverage_clamps_to_buying_power():
    portfolio = Portfolio(CASH, leverage=2.0)
    fill_price = Decimal(100)
    order = portfolio.buy(DATE_1, SYMBOL_1, Decimal(2001), fill_price)
    assert order is not None
    assert order.shares == Decimal(2000)


def test_buy_when_leverage_partial_deployment():
    portfolio = Portfolio(CASH, leverage=2.0)
    fill_price = Decimal(100)
    order_1 = portfolio.buy(DATE_1, SYMBOL_1, Decimal(1000), fill_price)
    assert order_1 is not None
    assert order_1.shares == Decimal(1000)
    assert portfolio.cash == CASH - Decimal(50_000)
    assert portfolio.margin_loan == Decimal(50_000)
    order_2 = portfolio.buy(DATE_2, SYMBOL_1, Decimal(1000), fill_price)
    assert order_2 is not None
    assert order_2.shares == Decimal(1000)
    assert portfolio.cash == 0
    assert portfolio.margin_loan == CASH
    order_3 = portfolio.buy(DATE_3, SYMBOL_1, Decimal(1), fill_price)
    assert order_3 is None


def test_sell_when_leverage_pays_down_margin_loan():
    portfolio = Portfolio(CASH, leverage=2.0)
    fill_price = Decimal(100)
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(2000), fill_price)
    portfolio.incr_bars()
    portfolio.sell(DATE_2, SYMBOL_1, Decimal(2000), fill_price)
    assert portfolio.cash == CASH
    assert portfolio.margin_loan == 0


def test_capture_bar_when_leverage_includes_margin_columns():
    portfolio = Portfolio(
        CASH,
        leverage=2.0,
        record_portfolio_bars=True,
        record_position_bars=True,
    )
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(2000), Decimal(100))
    df = pd.DataFrame(
        [[SYMBOL_1, DATE_1, Decimal(100), Decimal(99), Decimal(101)]],
        columns=["symbol", "date", "close", "low", "high"],
    )
    df = df.set_index(["symbol", "date"])
    portfolio.capture_bar(DATE_1, ColumnScope(df), {SYMBOL_1: 1})
    bar = portfolio.bars[0]
    assert bar.cash == 0
    assert bar.margin_loan == CASH
    assert bar.net_cash_balance == -CASH
    assert bar.equity == CASH


def test_apply_interest_when_net_cash_negative():
    portfolio = Portfolio(
        CASH, leverage=2.0, interest_rate=7.0, bars_per_year=252
    )
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(2000), Decimal(100))
    portfolio._apply_interest()
    expected_interest = CASH * Decimal(7) / Decimal(100) / Decimal(252)
    assert portfolio.margin_loan == CASH + expected_interest


def test_apply_interest_when_net_cash_positive():
    portfolio = Portfolio(CASH, interest_rate=7.0, bars_per_year=252)
    portfolio._apply_interest()
    expected_interest = CASH * Decimal(7) / Decimal(100) / Decimal(252)
    assert portfolio.cash == CASH + expected_interest


def test_available_buying_power_when_leverage_default():
    portfolio = Portfolio(CASH)
    assert portfolio._available_buying_power() == CASH


def test_short_when_leverage_posts_collateral():
    portfolio = Portfolio(CASH, leverage=2.0)
    fill_price = Decimal(100)
    shares = Decimal(2000)
    order = portfolio.sell(DATE_1, SYMBOL_1, shares, fill_price)
    assert order is not None
    assert order.shares == shares
    assert portfolio.cash == CASH - shares * fill_price / 2
    assert portfolio.margin_loan == shares * fill_price / 2


def test_short_when_leverage_clamps_to_buying_power():
    portfolio = Portfolio(CASH, leverage=2.0)
    fill_price = Decimal(100)
    order = portfolio.sell(DATE_1, SYMBOL_1, Decimal(2001), fill_price)
    assert order is not None
    assert order.shares == Decimal(2000)


def test_cover_when_leverage_releases_margin():
    portfolio = Portfolio(CASH, leverage=2.0)
    fill_price = Decimal(100)
    portfolio.sell(DATE_1, SYMBOL_1, Decimal(2000), fill_price)
    portfolio.incr_bars()
    portfolio.buy(DATE_2, SYMBOL_1, Decimal(2000), fill_price)
    assert portfolio.cash == CASH
    assert portfolio.margin_loan == 0


def test_mixed_long_short_shared_buying_power():
    portfolio = Portfolio(CASH, leverage=2.0)
    fill_price = Decimal(100)
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(1000), fill_price)
    order = portfolio.sell(DATE_2, SYMBOL_2, Decimal(1000), fill_price)
    assert order is not None
    assert order.shares == Decimal(1000)
    assert portfolio.cash == 0
    assert portfolio.margin_loan == CASH


def test_short_when_leverage_1_requires_full_cash():
    portfolio = Portfolio(CASH, leverage=1.0)
    fill_price = Decimal(100)
    order = portfolio.sell(DATE_1, SYMBOL_1, Decimal(1001), fill_price)
    assert order is not None
    assert order.shares == Decimal(1000)
    assert portfolio.cash == 0


def _mark(portfolio, date, prices):
    """Runs ``capture_bar`` with ``prices`` mapping symbol to close."""
    df = pd.DataFrame(
        [
            [sym, date, to_decimal(px), to_decimal(px), to_decimal(px)]
            for sym, px in prices.items()
        ],
        columns=["symbol", "date", "close", "low", "high"],
    ).set_index(["symbol", "date"])
    portfolio.capture_bar(date, ColumnScope(df), {sym: 1 for sym in prices})


def _expected_margin_loan(portfolio):
    leverage = to_decimal(portfolio._leverage)
    notional = sum(
        (
            pos.entry_notional
            for pos in itertools.chain(
                portfolio.long_positions.values(),
                portfolio.short_positions.values(),
            )
        ),
        Decimal(),
    )
    if portfolio._leverage <= 1:
        return portfolio._accrued_interest
    return notional - notional / leverage + portfolio._accrued_interest


@pytest.mark.parametrize("leverage", [1.0, 1.5, 2.0, 4.0])
@pytest.mark.parametrize("fractional", [True, False])
def test_margin_loan_invariants_when_random_sequences(leverage, fractional):
    """margin_loan stays non-negative and matches its defining identity."""
    rng = np.random.default_rng(42)
    dates = [DATE_1, DATE_2, DATE_3, DATE_4]
    for _ in range(300):
        portfolio = Portfolio(
            CASH, leverage=leverage, enable_fractional_shares=fractional
        )
        for i in range(12):
            date = dates[i % len(dates)]
            symbol = SYMBOL_1 if rng.random() < 0.5 else SYMBOL_2
            price = to_decimal(round(float(rng.uniform(20, 200)), 2))
            shares = to_decimal(round(float(rng.uniform(1, 900)), 2))
            if rng.random() < 0.5:
                portfolio.buy(date, symbol, shares, price)
            else:
                portfolio.sell(date, symbol, shares, price)
            assert portfolio.margin_loan >= 0
            assert portfolio.margin_loan == _expected_margin_loan(portfolio)


@pytest.mark.parametrize("leverage", [1.0, 2.0, 4.0])
def test_flat_portfolio_is_independent_of_exit_order(leverage):
    """Closing a long and a short in either order lands in the same state."""
    fill_price = Decimal(100)
    shares = Decimal(200)
    states = []
    for long_first in (True, False):
        portfolio = Portfolio(CASH, leverage=leverage)
        portfolio.buy(DATE_1, SYMBOL_1, shares, fill_price)
        portfolio.sell(DATE_1, SYMBOL_2, shares, fill_price)
        if long_first:
            portfolio.sell(DATE_2, SYMBOL_1, shares, fill_price)
            portfolio.buy(DATE_3, SYMBOL_2, shares, fill_price)
        else:
            portfolio.buy(DATE_2, SYMBOL_2, shares, fill_price)
            portfolio.sell(DATE_3, SYMBOL_1, shares, fill_price)
        assert not portfolio.long_positions
        assert not portfolio.short_positions
        states.append((portfolio.cash, portfolio.margin_loan))
        # Flat and zero PnL, so the account is back to its starting cash.
        assert portfolio.cash == CASH
        assert portfolio.margin_loan == 0
    assert states[0] == states[1]


def test_buying_power_when_flat_is_equity_times_leverage():
    portfolio = Portfolio(CASH, leverage=2.0)
    assert portfolio._available_buying_power() == CASH * 2
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(500), Decimal(100))
    _mark(portfolio, DATE_1, {SYMBOL_1: 100})
    portfolio.sell(DATE_2, SYMBOL_1, Decimal(500), Decimal(100))
    _mark(portfolio, DATE_2, {SYMBOL_1: 100})
    assert portfolio.market_value == CASH
    assert portfolio._available_buying_power() == CASH * 2


def test_buying_power_when_partial_exit_redeploys():
    """A partial exit deleverages, so the freed capacity must be reusable."""
    portfolio = Portfolio(CASH, leverage=2.0)
    fill_price = Decimal(100)
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(2000), fill_price)
    _mark(portfolio, DATE_1, {SYMBOL_1: 100})
    assert portfolio._available_buying_power() == 0
    portfolio.sell(DATE_2, SYMBOL_1, Decimal(500), fill_price)
    _mark(portfolio, DATE_2, {SYMBOL_1: 100})
    # 150k of exposure on 100k of equity is 1.5x, so 50k is still available.
    assert portfolio._available_buying_power() == Decimal(50_000)
    order = portfolio.buy(DATE_3, SYMBOL_2, Decimal(500), fill_price)
    assert order is not None
    assert order.shares == Decimal(500)


@pytest.mark.parametrize(
    "close, type", [(120, "short"), (190, "short"), (80, "long")]
)
def test_buying_power_when_adverse_mark_respects_leverage(close, type):
    """An unrealized loss must consume buying power for shorts and longs.

    An adverse mark can push an *already open* book past the cap on its own;
    nothing can be done about that without a margin call, which is not
    modeled. What must hold is that no *new* exposure breaches the cap.
    """
    leverage = Decimal(2)
    portfolio = Portfolio(CASH, leverage=float(leverage))
    fill_price = Decimal(100)
    shares = Decimal(1000)
    if type == "short":
        portfolio.sell(DATE_1, SYMBOL_1, shares, fill_price)
    else:
        portfolio.buy(DATE_1, SYMBOL_1, shares, fill_price)
    _mark(portfolio, DATE_1, {SYMBOL_1: close})
    market_value = portfolio.market_value
    assert market_value > 0
    gross_before = shares * to_decimal(close)
    order = portfolio.buy(DATE_2, SYMBOL_2, Decimal(5000), fill_price)
    gross_after = gross_before + (
        order.shares * fill_price if order is not None else Decimal()
    )
    if gross_before <= leverage * market_value:
        assert gross_after <= leverage * market_value
    else:
        # Already beyond the cap, so nothing more may be added.
        assert order is None


def _capture_bar_without_data(portfolio, date):
    """Runs ``capture_bar`` for a date that has no bar for any symbol."""
    rows = [
        [sym, DATE_4, Decimal(1), Decimal(1), Decimal(1)]
        for sym in portfolio.symbols
    ]
    df = pd.DataFrame(
        rows, columns=["symbol", "date", "close", "low", "high"]
    ).set_index(["symbol", "date"])
    portfolio.capture_bar(
        date, ColumnScope(df), {sym: 1 for sym in portfolio.symbols}
    )


def test_capture_bar_when_no_data_and_never_marked_holds_at_cost():
    """A never-marked long with no bar must not drop out of equity."""
    portfolio = Portfolio(CASH, leverage=2.0)
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(1000), Decimal(100))
    _capture_bar_without_data(portfolio, DATE_1)
    assert portfolio.equity == CASH
    assert portfolio.market_value == CASH
    assert portfolio._long_market_value() == Decimal(100_000)


@pytest.mark.parametrize("type", ["long", "short"])
def test_capture_bar_when_no_data_holds_at_last_mark(type):
    """A marked position with no bar keeps its last mark, not its cost.

    capture_bar and _long_market_value/_short_market_value must agree, since
    _available_buying_power subtracts one from the other.
    """
    portfolio = Portfolio(CASH, leverage=2.0)
    shares = Decimal(500)
    if type == "long":
        portfolio.buy(DATE_1, SYMBOL_1, shares, Decimal(100))
    else:
        portfolio.sell(DATE_1, SYMBOL_1, shares, Decimal(100))
    _mark(portfolio, DATE_1, {SYMBOL_1: 120})
    marked = portfolio.market_value
    _capture_bar_without_data(portfolio, DATE_2)
    assert portfolio.market_value == marked
    assert portfolio.market_value == portfolio._live_market_value()


@pytest.mark.parametrize(
    "fee_mode, fee_amount",
    [
        (FeeMode.PER_SHARE, 1.0),
        (FeeMode.ORDER_PERCENT, 1.0),
        (FeeMode.PER_ORDER, 500.0),
    ],
)
def test_buy_when_fees_do_not_breach_leverage(fee_mode, fee_amount):
    leverage = Decimal(2)
    portfolio = Portfolio(
        CASH,
        leverage=float(leverage),
        fee_mode=fee_mode,
        fee_amount=fee_amount,
    )
    fill_price = Decimal(100)
    order = portfolio.buy(DATE_1, SYMBOL_1, Decimal(5000), fill_price)
    assert order is not None
    _mark(portfolio, DATE_1, {SYMBOL_1: 100})
    gross = order.shares * fill_price
    assert gross / portfolio.market_value <= leverage


def test_apply_interest_when_no_bars_per_year_then_noop():
    portfolio = Portfolio(CASH, interest_rate=7.0)
    portfolio._apply_interest()
    assert portfolio.cash == CASH
    assert portfolio.margin_loan == 0


def test_apply_interest_matches_across_timeframes():
    """The same elapsed year must cost the same regardless of bar size."""
    daily = Portfolio(CASH, leverage=2.0, interest_rate=7.0, bars_per_year=252)
    daily.buy(DATE_1, SYMBOL_1, Decimal(2000), Decimal(100))
    for _ in range(252):
        daily._apply_interest()
    intraday = Portfolio(
        CASH, leverage=2.0, interest_rate=7.0, bars_per_year=252 * 390
    )
    intraday.buy(DATE_1, SYMBOL_1, Decimal(2000), Decimal(100))
    for _ in range(252 * 390):
        intraday._apply_interest()
    assert float(daily.margin_loan) == pytest.approx(
        float(intraday.margin_loan), rel=1e-3
    )


def test_apply_interest_sweeps_accrued_interest_from_cash():
    portfolio = Portfolio(
        CASH, leverage=2.0, interest_rate=7.0, bars_per_year=252
    )
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(2000), Decimal(100))
    portfolio._apply_interest()
    accrued = portfolio._accrued_interest
    assert accrued > 0
    portfolio.sell(DATE_2, SYMBOL_1, Decimal(2000), Decimal(100))
    assert portfolio.cash == CASH
    assert portfolio.margin_loan == accrued
    net_cash = portfolio._net_cash_balance()
    portfolio._sweep_accrued_interest()
    # Repaying out of cash leaves the net balance unchanged.
    assert portfolio._accrued_interest == 0
    assert portfolio.margin_loan == 0
    assert portfolio.cash == CASH - accrued
    assert portfolio._net_cash_balance() == net_cash


def test_cover_when_fractional_shares_leaves_no_dust():
    portfolio = Portfolio(CASH, enable_fractional_shares=True)
    portfolio.sell(DATE_1, SYMBOL_1, Decimal("722.76"), Decimal("132.33"))
    portfolio.sell(DATE_2, SYMBOL_1, Decimal("116.49"), Decimal("105.43"))
    pos = portfolio.short_positions[SYMBOL_1]
    assert pos.shares == sum(entry.shares for entry in pos.entries)
    portfolio.buy(DATE_3, SYMBOL_1, pos.shares, Decimal(100))
    assert not portfolio.short_positions
    assert not portfolio.long_positions
    assert SYMBOL_1 not in portfolio.symbols


def _exit_before_mark(type, exit_shares, leverage=Decimal(2)):
    """Exits, then fills a second order on the same bar, before any mark."""
    portfolio = Portfolio(CASH, leverage=float(leverage))
    shares = Decimal(2000)
    # Adverse move for whichever side is held.
    exit_price = Decimal(85) if type == "long" else Decimal(115)
    if type == "long":
        portfolio.buy(DATE_1, SYMBOL_1, shares, Decimal(100))
        _mark(portfolio, DATE_1, {SYMBOL_1: 100})
        portfolio.sell(DATE_2, SYMBOL_1, exit_shares, exit_price)
    else:
        portfolio.sell(DATE_1, SYMBOL_1, shares, Decimal(100))
        _mark(portfolio, DATE_1, {SYMBOL_1: 100})
        portfolio.buy(DATE_2, SYMBOL_1, exit_shares, exit_price)
    # No mark in between: this fill is still on the same bar as the exit.
    order = portfolio.buy(DATE_2, SYMBOL_2, Decimal(5000), Decimal(100))
    _mark(portfolio, DATE_2, {SYMBOL_1: exit_price, SYMBOL_2: 100})
    gross = (shares - exit_shares) * exit_price
    if order is not None:
        gross += order.shares * Decimal(100)
    return portfolio, gross


@pytest.mark.parametrize("type", ["long", "short"])
def test_buying_power_when_full_exit_before_mark_respects_leverage(type):
    """An order filling after a full exit, before the mark, must not over-lever.

    Every fill in the bar loop happens before ``capture_bar``, so sizing off
    the last snapshot would value the account as it stood before this bar's
    exits. With the position fully closed there is nothing left carrying a
    stale mark, so the cap must bind exactly.
    """
    leverage = Decimal(2)
    portfolio, gross = _exit_before_mark(type, Decimal(2000), leverage)
    assert portfolio.market_value > 0
    assert gross / portfolio.market_value <= leverage


@pytest.mark.parametrize(
    "type, excess", [("long", Decimal(22_500)), ("short", Decimal(67_500))]
)
def test_buying_power_when_partial_exit_before_mark_trails_by_one_bar(
    type, excess
):
    """Pins the known residual: the *unexited* remainder keeps its last mark.

    Everything realized is exact, but a position still open since the last
    ``capture_bar`` is valued at that mark, so buying power trails by
    ``(leverage - 1) * delta`` for longs and ``(leverage + 1) * delta`` for
    shorts, where ``delta`` is the remainder's one-bar move (1500 shares
    moving 15 here). Inherent to the bar loop, not to this formula; fixing it
    means marking at the fill price before pre-``capture_bar`` fills.
    """
    leverage = Decimal(2)
    portfolio, gross = _exit_before_mark(type, Decimal(500), leverage)
    assert gross - leverage * portfolio.market_value == excess


@pytest.mark.parametrize("close", [80, 100, 120])
@pytest.mark.parametrize(
    "ops",
    [
        [("buy", 1000)],
        [("sell", 1000)],
        [("buy", 1000), ("sell", 400)],
        [("sell", 1000), ("buy", 400)],
        [("buy", 1000), ("sell", 1000)],
        [("sell", 1000), ("buy", 1000)],
    ],
)
def test_live_market_value_matches_capture_bar(ops, close):
    """The live mark must equal the snapshot when taken right after one.

    This is what catches expressing a short as ``entry_notional + pnl``:
    ``pnl`` is only recomputed in ``capture_bar``, so it is stale after a
    partial cover, while the realized slice has already been paid to cash.
    """
    portfolio = Portfolio(CASH, leverage=2.0)
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(200), Decimal(100))
    portfolio.sell(DATE_1, SYMBOL_2, Decimal(300), Decimal(100))
    _mark(portfolio, DATE_1, {SYMBOL_1: close, SYMBOL_2: close})
    for op, qty in ops:
        if op == "buy":
            portfolio.buy(DATE_2, SYMBOL_2, Decimal(qty), to_decimal(close))
        else:
            portfolio.sell(DATE_2, SYMBOL_1, Decimal(qty), to_decimal(close))
    _mark(portfolio, DATE_2, {SYMBOL_1: close, SYMBOL_2: close})
    assert portfolio._live_market_value() == portfolio.market_value


def test_buying_power_when_unrealized_gain_is_not_capped_by_cash():
    """Guards against reintroducing a ``cash * leverage`` term.

    Cash is depleted by design under leverage, so capping on it would strand
    the capacity an unrealized gain creates.
    """
    portfolio = Portfolio(CASH, leverage=2.0)
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(2000), Decimal(100))
    _mark(portfolio, DATE_1, {SYMBOL_1: 100})
    assert portfolio.cash == 0
    _mark(portfolio, DATE_2, {SYMBOL_1: 120})
    # Equity 140k, exposure 240k, so 2 * 140k - 240k is still deployable.
    assert portfolio.market_value == Decimal(140_000)
    assert portfolio._available_buying_power() == Decimal(40_000)


@pytest.mark.parametrize("leverage", [1.0, 2.0])
@pytest.mark.parametrize("fractional", [True, False])
def test_entry_notional_totals_track_positions(leverage, fractional):
    """The running totals must never drift from the open book."""
    rng = np.random.default_rng(7)
    dates = [DATE_1, DATE_2, DATE_3, DATE_4]
    for _ in range(200):
        portfolio = Portfolio(
            CASH, leverage=leverage, enable_fractional_shares=fractional
        )
        for i in range(12):
            date = dates[i % len(dates)]
            symbol = SYMBOL_1 if rng.random() < 0.5 else SYMBOL_2
            price = to_decimal(round(float(rng.uniform(20, 200)), 2))
            shares = to_decimal(round(float(rng.uniform(1, 900)), 2))
            if rng.random() < 0.5:
                portfolio.buy(date, symbol, shares, price)
            else:
                portfolio.sell(date, symbol, shares, price)
            for total, positions in (
                (portfolio._long_entry_notional, portfolio.long_positions),
                (portfolio._short_entry_notional, portfolio.short_positions),
            ):
                assert total == sum(
                    (pos.entry_notional for pos in positions.values()),
                    Decimal(),
                )


@pytest.mark.parametrize("type", ["long", "short"])
def test_buying_power_when_add_on_before_mark_charges_fill_price(type):
    """An add-on must consume its own notional of buying power.

    Valuing the whole position at the previous mark prices the new shares at
    a price they were not bought at, so the fill can cost less buying power
    than it consumes -- or none at all, letting exposure compound without
    bound across same-bar add-ons.
    """
    portfolio = Portfolio(CASH, leverage=2.0)
    fill = Decimal(100)
    if type == "long":
        portfolio.buy(DATE_1, SYMBOL_1, Decimal(500), fill)
    else:
        portfolio.sell(DATE_1, SYMBOL_1, Decimal(500), fill)
    _mark(portfolio, DATE_1, {SYMBOL_1: 100})
    # Add on at half the mark, on the next bar but before it is marked.
    add_price = Decimal(50)
    before = portfolio._available_buying_power()
    if type == "long":
        portfolio.buy(DATE_2, SYMBOL_1, Decimal(100), add_price)
    else:
        portfolio.sell(DATE_2, SYMBOL_1, Decimal(100), add_price)
    after = portfolio._available_buying_power()
    assert before - after == Decimal(100) * add_price


@pytest.mark.parametrize("type", ["long", "short"])
def test_buying_power_when_repeated_add_ons_then_exhausted(type):
    """Repeated same-bar add-ons must drain buying power, not regenerate it."""
    portfolio = Portfolio(CASH, leverage=2.0)
    order_fn = portfolio.buy if type == "long" else portfolio.sell
    order_fn(DATE_1, SYMBOL_1, Decimal(500), Decimal(100))
    _mark(portfolio, DATE_1, {SYMBOL_1: 100})
    filled = [
        Decimal() if order is None else order.shares
        for order in (
            order_fn(DATE_2, SYMBOL_1, Decimal(10_000), Decimal(50))
            for _ in range(8)
        )
    ]
    assert portfolio._available_buying_power() == 0
    assert sum(filled[1:]) == 0


@pytest.mark.parametrize("type", ["long", "short"])
def test_exit_when_shares_match_entry_prefix_then_no_empty_trade(type):
    """An exit consuming whole entries must not book a zero-share trade."""
    portfolio = Portfolio(CASH)
    open_fn = portfolio.buy if type == "long" else portfolio.sell
    close_fn = portfolio.sell if type == "long" else portfolio.buy
    open_fn(DATE_1, SYMBOL_1, Decimal(100), Decimal(100))
    open_fn(DATE_2, SYMBOL_1, Decimal(100), Decimal(100))
    close_fn(DATE_3, SYMBOL_1, Decimal(100), Decimal(110))
    assert len(portfolio.trades) == 1
    assert all(trade.shares > 0 for trade in portfolio.trades)


def test_clamp_shares_when_leveraged_per_share_fee_then_order_fills():
    """The fee reserve must be sized to the order, not to the largest one.

    A per-share fee grows with the share count, so reserving for the biggest
    order buying power could support drops an order costing a few dollars.
    """
    portfolio = Portfolio(
        10_000, fee_mode=FeeMode.PER_SHARE, fee_amount=0.01, leverage=2.0
    )
    order = portfolio.buy(DATE_1, SYMBOL_1, Decimal(100), Decimal("0.01"))
    assert order is not None
    assert order.shares == Decimal(100)


@pytest.mark.parametrize("type", ["long", "short"])
def test_position_cap_when_lowered_below_held_count_then_still_binds(type):
    """The cap is retuned per window by walkforward optimization, so it can
    drop below the number of positions already held. An equality test would
    stop binding entirely and let the book keep growing."""
    if type == "long":
        portfolio = Portfolio(1_000_000, max_long_positions=3)
        order_fn = portfolio.buy
        positions = portfolio.long_positions
    else:
        portfolio = Portfolio(1_000_000, max_short_positions=3)
        order_fn = portfolio.sell
        positions = portfolio.short_positions
    for symbol in ("A", "B", "C"):
        order_fn(DATE_1, symbol, Decimal(10), Decimal(100))
    assert len(positions) == 3
    if type == "long":
        portfolio._max_long_positions = 1
    else:
        portfolio._max_short_positions = 1
    assert order_fn(DATE_2, "D", Decimal(10), Decimal(100)) is None
    assert len(positions) == 3


@pytest.mark.parametrize(
    "fee_mode, fee_amount, price",
    [
        (FeeMode.PER_SHARE, 0.05, Decimal(1)),
        (FeeMode.PER_SHARE, 0.005, Decimal(5)),
        (FeeMode.ORDER_PERCENT, 0.5, Decimal(100)),
        (FeeMode.PER_ORDER, 5.0, Decimal(10)),
    ],
)
@pytest.mark.parametrize("leverage", [1.5, 2.0, 4.0])
def test_clamp_shares_fee_reserve_never_exceeds_buying_power(
    fee_mode, fee_amount, price, leverage
):
    """The fill plus the leverage cost of its fees must fit the budget.

    Re-deriving the reserve at a reduced share count yields a *smaller* fee
    and so a larger budget, which undoes the reservation and lets the fill
    spill past the configured leverage.
    """
    portfolio = Portfolio(
        100_000, fee_mode=fee_mode, fee_amount=fee_amount, leverage=leverage
    )
    budget = portfolio._available_buying_power()
    order = portfolio.buy(DATE_1, SYMBOL_1, Decimal(500_000), price)
    if order is None:
        return
    used = order.shares * price + portfolio.fees * to_decimal(leverage)
    assert used <= budget


def test_clamp_unmarked_uses_fifo_cost_of_surviving_entries():
    """Exits are FIFO, so the unmarked survivors are the newest entries.

    Rescaling the notional proportionally prices them at the average cost of
    every unmarked fill, which is not what FIFO actually left behind when
    those fills had different prices.
    """
    portfolio = Portfolio(1_000_000)
    portfolio.buy(DATE_1, SYMBOL_1, Decimal(100), Decimal(10))
    _mark(portfolio, DATE_1, {SYMBOL_1: 10})
    portfolio.buy(DATE_2, SYMBOL_1, Decimal(50), Decimal(12))
    portfolio.buy(DATE_2, SYMBOL_1, Decimal(50), Decimal(14))
    portfolio.sell(DATE_2, SYMBOL_1, Decimal(150), Decimal(20))
    pos = portfolio.long_positions[SYMBOL_1]
    expected = sum(
        (entry.shares * entry.price for entry in pos.entries), Decimal()
    )
    assert pos.unmarked_shares == pos.shares
    assert pos.unmarked_notional == expected
