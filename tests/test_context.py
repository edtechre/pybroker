"""Unit tests for context.py module."""

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

from .fixtures import *
from collections import deque
from decimal import Decimal
from pybroker.common import PriceType, to_datetime
from pybroker.context import (
    ExecContext,
    ExecResult,
    PosSizeContext,
    set_exec_ctx_data,
    set_pos_size_ctx_data,
)
from pybroker.portfolio import Order, Portfolio, Position, Trade
import numpy as np
import pytest
import re


@pytest.fixture()
def portfolio():
    return Portfolio(100_000)


@pytest.fixture()
def end_index():
    return 100


@pytest.fixture()
def sym_end_index(symbols, end_index):
    return {sym: end_index for sym in symbols}


@pytest.fixture()
def session():
    return {"foo": 1, "bar": 2}


@pytest.fixture()
def foreign(symbols):
    return list(symbols)[-1]


@pytest.fixture()
def date(dates, end_index):
    return list(dates)[end_index - 1]


@pytest.fixture()
def orders(dates, symbols):
    return (
        Order(
            id=1,
            date=dates[0],
            symbol=symbols[1],
            type="buy",
            limit_price=None,
            fill_price=10,
            shares=200,
            fees=0,
        ),
        Order(
            id=2,
            date=dates[1],
            symbol=symbols[2],
            type="sell",
            limit_price=100,
            fill_price=101.1,
            shares=100,
            fees=0,
        ),
    )


@pytest.fixture()
def trades(dates, symbols):
    return Trade(
        id=1,
        type="long",
        symbol=symbols[-1],
        entry_date=dates[0],
        exit_date=dates[1],
        entry=100,
        exit=101,
        shares=100,
        pnl=Decimal(100),
        return_pct=Decimal(5),
        agg_pnl=Decimal(100),
        bars=1,
        pnl_per_bar=Decimal(100),
    )


@pytest.fixture()
def ctx(
    col_scope,
    ind_scope,
    input_scope,
    pred_scope,
    portfolio,
    trained_models,
    sym_end_index,
    session,
    symbol,
    date,
):
    ctx = ExecContext(
        portfolio=portfolio,
        col_scope=col_scope,
        ind_scope=ind_scope,
        input_scope=input_scope,
        pred_scope=pred_scope,
        models=trained_models,
        sym_end_index=sym_end_index,
    )
    set_exec_ctx_data(ctx, session, symbol, date)
    return ctx


@pytest.fixture()
def ctx_with_pos(
    col_scope,
    ind_scope,
    input_scope,
    pred_scope,
    portfolio,
    trained_models,
    sym_end_index,
    session,
    symbol,
    symbols,
    date,
):
    portfolio.long_positions = {
        sym: Position(sym, 200, "long") for sym in symbols
    }
    portfolio.short_positions = {
        sym: Position(sym, 100, "short") for sym in symbols
    }
    ctx = ExecContext(
        portfolio=portfolio,
        col_scope=col_scope,
        ind_scope=ind_scope,
        input_scope=input_scope,
        pred_scope=pred_scope,
        models=trained_models,
        sym_end_index=sym_end_index,
    )
    set_exec_ctx_data(ctx, session, symbol, date)
    return ctx


@pytest.fixture()
def ctx_with_orders(
    col_scope,
    ind_scope,
    input_scope,
    pred_scope,
    portfolio,
    trained_models,
    sym_end_index,
    session,
    symbol,
    date,
    orders,
    trades,
):
    portfolio.orders = deque(orders)
    portfolio.trades = deque(trades)
    ctx = ExecContext(
        portfolio=portfolio,
        col_scope=col_scope,
        ind_scope=ind_scope,
        input_scope=input_scope,
        pred_scope=pred_scope,
        models=trained_models,
        sym_end_index=sym_end_index,
    )
    set_exec_ctx_data(ctx, session, symbol, date)
    return ctx


def test_dt(ctx, date):
    assert ctx.dt == to_datetime(date)


@pytest.mark.parametrize(
    "field", ["date", "open", "high", "low", "close", "volume", "adj_close"]
)
def test_fields(scope, ctx, field, end_index):
    scope.register_custom_cols("adj_close")
    assert len(getattr(ctx, field)) == end_index


def test_field_not_found_then_error(ctx):
    with pytest.raises(
        AttributeError, match=re.escape("Attribute 'foo' not found.")
    ):
        ctx.foo


def test_empty_field(ctx):
    assert ctx.vwap is None


@pytest.mark.parametrize(
    "field, port_field",
    [
        ("total_equity", "equity"),
        ("cash", "cash"),
        ("total_margin", "margin"),
        ("total_market_value", "market_value"),
    ],
)
def test_portfolio_field(ctx, portfolio, field, port_field):
    assert getattr(ctx, field) == getattr(portfolio, port_field)


def test_sell_all_shares(ctx_with_pos):
    ctx_with_pos.sell_all_shares()
    assert ctx_with_pos.sell_shares == 200


def test_sell_all_shares_when_no_position(ctx):
    ctx.sell_all_shares()
    assert ctx.sell_shares is None


def test_cover_all_shares(ctx_with_pos):
    ctx_with_pos.cover_all_shares()
    assert ctx_with_pos.buy_shares == 100


def test_cover_all_shares_when_no_position(ctx):
    ctx.cover_all_shares()
    assert ctx.buy_shares is None


def test_model(ctx, trained_models, symbol):
    assert (
        ctx.model(MODEL_NAME) == trained_models[(MODEL_NAME, symbol)].instance
    )


def test_model_when_not_found_then_error(ctx, symbol):
    with pytest.raises(
        ValueError,
        match=re.escape(f"Model 'undefined_model' not found for {symbol}."),
    ):
        ctx.model("undefined_model")


def test_indicator(ctx, end_index):
    assert len(ctx.indicator("hhv")) == end_index


def test_indicator_when_not_found_then_error(ctx, symbol):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Indicator 'undefined_indicator' not found for {symbol}."
        ),
    ):
        ctx.indicator("undefined_indicator")


def test_input(ctx, end_index):
    assert len(ctx.input(MODEL_NAME)["hhv"]) == end_index


def test_input_when_not_found_then_error(ctx):
    with pytest.raises(
        ValueError, match=re.escape("Model 'undefined_model' not found.")
    ):
        ctx.input("undefined_model")


def test_preds(ctx, end_index):
    assert len(ctx.preds(MODEL_NAME)) == end_index


def test_preds_when_not_found_then_error(ctx):
    with pytest.raises(
        ValueError, match=re.escape("Model 'undefined_model' not found.")
    ):
        ctx.preds("undefined_model")


@pytest.mark.parametrize("pos_type", ["long", "short"])
def test_position(ctx_with_pos, pos_type, portfolio, symbol):
    assert (
        getattr(ctx_with_pos, f"{pos_type}_pos")()
        == getattr(portfolio, f"{pos_type}_positions")[symbol]
    )


@pytest.mark.parametrize("pos_fn", ["long_pos", "short_pos"])
def test_position_when_empty(ctx, pos_fn):
    assert getattr(ctx, pos_fn)() is None


def test_position_when_invalid_pos_type_then_error(ctx, symbol):
    with pytest.raises(
        ValueError, match=re.escape("Unknown pos_type: 'invalid'.")
    ):
        ctx.pos(symbol, "invalid")


@pytest.mark.parametrize("pos_type", ["long", "short"])
def test_position_with_foreign_when_empty(ctx, pos_type, foreign):
    assert getattr(ctx, f"{pos_type}_pos")(foreign) is None


@pytest.mark.parametrize("pos_type", ["long", "short", None])
def test_positions(ctx_with_pos, pos_type, portfolio):
    positions = ctx_with_pos.positions(None, pos_type)
    if pos_type is None:
        expected_positions = set(portfolio.long_positions.keys()) | set(
            portfolio.short_positions.keys()
        )
    else:
        expected_positions = set(
            getattr(portfolio, f"{pos_type}_positions").keys()
        )
    assert set(map(lambda p: p.symbol, positions)) == expected_positions


@pytest.mark.parametrize("pos_type", ["long", "short", None])
def test_positions_when_empty(ctx, pos_type):
    assert not len(list(ctx.positions(None, pos_type)))


@pytest.mark.parametrize("pos_type", ["long", "short", None])
def test_positions_with_symbol(ctx_with_pos, pos_type, foreign):
    positions = list(ctx_with_pos.positions(foreign, pos_type))
    if pos_type is None:
        assert len(positions) == 2
        assert positions[0].symbol == foreign
        assert positions[1].symbol == foreign
    else:
        assert len(positions) == 1
        assert positions[0].symbol == foreign


@pytest.mark.parametrize("pos_type", ["long", "short", None])
def test_positions_with_symbol_when_empty(ctx, pos_type, foreign):
    assert not len(list(ctx.positions(foreign, pos_type)))


@pytest.mark.parametrize(
    "col", ["date", "open", "high", "low", "close", "volume", "adj_close"]
)
def test_foreign(ctx, col, foreign, data_source_df, end_index):
    df = data_source_df[data_source_df["symbol"] == foreign]
    assert (ctx.foreign(foreign, col) == df[col].values[:end_index]).all()
    df = data_source_df[data_source_df["symbol"] == foreign]
    assert (ctx.foreign(foreign, col) == df[col].values[:end_index]).all()


def test_foreign_with_custom_col(
    scope, ctx, foreign, data_source_df, end_index
):
    scope.register_custom_cols("adj_close")
    df = data_source_df[data_source_df["symbol"] == foreign]
    assert (
        ctx.foreign(foreign, "adj_close") == df["adj_close"].values[:end_index]
    ).all()


def test_foreign_when_empty(ctx, foreign):
    assert ctx.foreign(foreign, "foo") is None


def test_foreign_when_symbol_not_found_then_error(ctx):
    with pytest.raises(ValueError, match=re.escape("Symbol 'FOO' not found.")):
        ctx.foreign("FOO", "close")


def test_foreign_with_empty_col(
    scope, ctx, foreign, data_source_df, end_index
):
    scope.register_custom_cols("adj_close")

    df = data_source_df[data_source_df["symbol"] == foreign]

    def verify_bar_data(bar_data):
        for col in (
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
        ):
            assert (getattr(bar_data, col) == df[col].values[:end_index]).all()

    verify_bar_data(ctx.foreign(foreign))
    verify_bar_data(ctx.foreign(foreign))


def test_calc_target_shares(ctx):
    assert ctx.calc_target_shares(0.5, 33.50) == 50_000 // 33.5


def test_to_result(ctx, symbol, date):
    ctx.buy_fill_price = PriceType.AVERAGE
    ctx.buy_shares = 20
    ctx.buy_limit_price = 99.99
    ctx.sell_fill_price: PriceType = PriceType.HIGH
    ctx.sell_shares = 20
    ctx.sell_limit_price = 110.11
    ctx.hold_bars = 2
    ctx.score = 7
    result = ctx.to_result()
    assert result == ExecResult(
        symbol=symbol,
        date=date,
        buy_fill_price=PriceType.AVERAGE,
        buy_shares=20,
        buy_limit_price=Decimal("99.99"),
        sell_fill_price=PriceType.HIGH,
        sell_shares=20,
        sell_limit_price=Decimal("110.11"),
        hold_bars=2,
        score=7,
    )


def test_orders(ctx_with_orders, orders):
    assert tuple(ctx_with_orders.orders()) == orders


def test_orders_when_empty(ctx):
    assert not len(list(ctx.orders()))


def test_trades(ctx_with_orders, trades):
    assert tuple(ctx_with_orders.trades()) == trades


def test_trades_when_empty(ctx):
    assert not len(list(ctx.trades()))


def test_set_exec_ctx_data(ctx, symbols, sym_end_index):
    sym = symbols[-1]
    session = {"a": 1, "b": 2}
    date = np.datetime64("2020-01-01")
    set_exec_ctx_data(ctx, session, sym, date)
    assert ctx.session == session
    assert ctx.symbol == sym
    assert ctx.dt == to_datetime(date)
    assert ctx.bars == sym_end_index[sym]


def test_set_pos_ctx_data(
    date,
    portfolio,
    col_scope,
    ind_scope,
    input_scope,
    pred_scope,
    trained_models,
    sym_end_index,
):
    buy_results = [
        ExecResult(
            symbol="SPY",
            date=date,
            buy_shares=100,
            buy_fill_price=99,
            buy_limit_price=99,
            sell_shares=None,
            sell_fill_price=None,
            sell_limit_price=None,
            hold_bars=None,
            score=1,
        ),
        ExecResult(
            symbol="AAPL",
            date=date,
            buy_shares=200,
            buy_fill_price=90,
            buy_limit_price=90,
            sell_shares=None,
            sell_fill_price=None,
            sell_limit_price=None,
            hold_bars=None,
            score=2,
        ),
    ]
    sell_results = [
        ExecResult(
            symbol="TSLA",
            date=date,
            buy_shares=None,
            buy_fill_price=None,
            buy_limit_price=None,
            sell_shares=100,
            sell_fill_price=80,
            sell_limit_price=80,
            hold_bars=None,
            score=1,
        ),
    ]
    ctx = PosSizeContext(
        portfolio,
        col_scope,
        ind_scope,
        input_scope,
        pred_scope,
        trained_models,
        sym_end_index,
        max_long_positions=1,
        max_short_positions=None,
    )
    set_pos_size_ctx_data(ctx, buy_results, sell_results)
    buy_signals = list(ctx.signals("buy"))
    assert len(buy_signals) == 1
    assert buy_signals[0].id == 0
    assert buy_signals[0].symbol == "SPY"
    assert buy_signals[0].shares == 100
    assert buy_signals[0].score == 1
    assert buy_signals[0].type == "buy"
    assert buy_signals[0].bar_data is not None
    sell_signals = list(ctx.signals("sell"))
    assert len(sell_signals) == 1
    assert sell_signals[0].id == 2
    assert sell_signals[0].symbol == "TSLA"
    assert sell_signals[0].shares == 100
    assert sell_signals[0].score == 1
    assert sell_signals[0].type == "sell"
    assert sell_signals[0].bar_data is not None
    all_signals = list(ctx.signals())
    assert len(all_signals) == 2
    assert all_signals[0].id == buy_signals[0].id
    assert all_signals[0].symbol == buy_signals[0].symbol
    assert all_signals[0].shares == buy_signals[0].shares
    assert all_signals[0].score == buy_signals[0].score
    assert all_signals[0].type == buy_signals[0].type
    assert all_signals[0].bar_data is not None
    assert all_signals[1].id == sell_signals[0].id
    assert all_signals[1].symbol == sell_signals[0].symbol
    assert all_signals[1].shares == sell_signals[0].shares
    assert all_signals[1].score == sell_signals[0].score
    assert all_signals[1].type == sell_signals[0].type
    assert all_signals[1].bar_data is not None
