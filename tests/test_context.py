"""Unit tests for context.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pytest
import re
from .fixtures import *
from collections import deque
from decimal import Decimal
from pybroker.common import PriceType, StopType, to_datetime
from pybroker.config import StrategyConfig
from pybroker.context import (
    ExecContext,
    ExecResult,
    PosSizeContext,
    set_exec_ctx_data,
    set_pos_size_ctx_data,
)
from pybroker.portfolio import Order, Portfolio, Position, Trade


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
        stop=None,
        mae=Decimal(-10),
        mfe=Decimal(10),
    )


@pytest.fixture()
def ctx(
    col_scope,
    ind_scope,
    input_scope,
    pred_scope,
    pending_order_scope,
    portfolio,
    trained_models,
    sym_end_index,
    session,
    symbol,
    date,
):
    ctx = ExecContext(
        symbol=symbol,
        config=StrategyConfig(max_long_positions=5),
        portfolio=portfolio,
        col_scope=col_scope,
        ind_scope=ind_scope,
        input_scope=input_scope,
        pred_scope=pred_scope,
        pending_order_scope=pending_order_scope,
        models=trained_models,
        sym_end_index=sym_end_index,
        session=session,
    )
    set_exec_ctx_data(ctx, date)
    return ctx


@pytest.fixture()
def ctx_with_pos(
    col_scope,
    ind_scope,
    input_scope,
    pred_scope,
    pending_order_scope,
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
        symbol=symbol,
        config=StrategyConfig(max_long_positions=5),
        portfolio=portfolio,
        col_scope=col_scope,
        ind_scope=ind_scope,
        input_scope=input_scope,
        pred_scope=pred_scope,
        pending_order_scope=pending_order_scope,
        models=trained_models,
        sym_end_index=sym_end_index,
        session=session,
    )
    set_exec_ctx_data(ctx, date)
    return ctx


@pytest.fixture()
def ctx_with_orders(
    col_scope,
    ind_scope,
    input_scope,
    pred_scope,
    pending_order_scope,
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
    portfolio.win_rate = 1
    portfolio.lose_rate = 0
    ctx = ExecContext(
        symbol=symbol,
        config=StrategyConfig(max_long_positions=5),
        portfolio=portfolio,
        col_scope=col_scope,
        ind_scope=ind_scope,
        input_scope=input_scope,
        pred_scope=pred_scope,
        pending_order_scope=pending_order_scope,
        models=trained_models,
        sym_end_index=sym_end_index,
        session=session,
    )
    set_exec_ctx_data(ctx, date)
    return ctx


def test_config(ctx):
    assert ctx.config.max_long_positions == 5


def test_dt(ctx, date):
    assert ctx.dt == to_datetime(date)


def test_win_rate(ctx_with_orders):
    assert ctx_with_orders.win_rate == 1


def test_loss_rate(ctx_with_orders):
    assert ctx_with_orders.loss_rate == 0


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


def test_long_positions(ctx_with_pos, symbol):
    positions = tuple(ctx_with_pos.long_positions(symbol))
    assert len(positions) == 1
    assert positions[0] == Position(symbol, 200, "long")


def test_long_positions_when_empty(ctx, symbol):
    assert not len(tuple(ctx.long_positions(symbol)))


def test_short_positions(ctx_with_pos, symbol):
    positions = tuple(ctx_with_pos.short_positions(symbol))
    assert len(positions) == 1
    assert positions[0] == Position(symbol, 100, "short")


def test_short_positions_when_empty(ctx, symbol):
    assert not len(tuple(ctx.short_positions(symbol)))


def test_calc_target_shares(ctx):
    assert ctx.calc_target_shares(0.5, 33.50) == 50_000 // 33.5


def test_calc_target_shares_when_enable_fractional_shares(
    col_scope,
    ind_scope,
    input_scope,
    pred_scope,
    pending_order_scope,
    portfolio,
    trained_models,
    sym_end_index,
    session,
    symbol,
):
    ctx = ExecContext(
        symbol=symbol,
        config=StrategyConfig(enable_fractional_shares=True),
        portfolio=portfolio,
        col_scope=col_scope,
        ind_scope=ind_scope,
        input_scope=input_scope,
        pred_scope=pred_scope,
        pending_order_scope=pending_order_scope,
        models=trained_models,
        sym_end_index=sym_end_index,
        session=session,
    )
    assert ctx.calc_target_shares(0.5, 33.50) == Decimal("50_000") / Decimal(
        "33.5"
    )


def test_calc_target_shares_with_cash(ctx):
    assert ctx.calc_target_shares(1 / 3, 20, 10_000) == 166


def test_calc_target_shares_when_negative_then_zero(ctx):
    assert ctx.calc_target_shares(1, 20, -2_000) == 0


def test_to_result_when_buy(ctx, symbol, date):
    ctx.buy_fill_price = PriceType.AVERAGE
    ctx.sell_fill_price = PriceType.HIGH
    ctx.buy_shares = 20
    ctx.buy_limit_price = 99.99
    ctx.hold_bars = 2
    ctx.score = 7
    result = ctx.to_result()
    assert result.symbol == symbol
    assert result.date == date
    assert result.buy_fill_price == PriceType.AVERAGE
    assert result.buy_shares == 20
    assert result.buy_limit_price == Decimal("99.99")
    assert result.hold_bars == 2
    assert result.score == 7
    assert len(result.long_stops) == 1
    assert result.short_stops is None
    stop = next(iter(result.long_stops))
    assert stop.symbol == symbol
    assert stop.stop_type == StopType.BAR
    assert stop.pos_type == "long"
    assert stop.percent is None
    assert stop.points is None
    assert stop.bars == 2
    assert stop.fill_price == PriceType.HIGH
    assert stop.limit_price is None


def test_to_result_when_sell(ctx, symbol, date):
    ctx.buy_fill_price = PriceType.AVERAGE
    ctx.sell_fill_price = PriceType.HIGH
    ctx.sell_shares = 20
    ctx.sell_limit_price = 110.11
    ctx.hold_bars = 2
    ctx.score = 7
    result = ctx.to_result()
    assert result.symbol == symbol
    assert result.date == date
    assert result.sell_fill_price == PriceType.HIGH
    assert result.sell_shares == 20
    assert result.sell_limit_price == Decimal("110.11")
    assert result.hold_bars == 2
    assert result.score == 7
    assert len(result.short_stops) == 1
    assert result.long_stops is None
    stop = next(iter(result.short_stops))
    assert stop.symbol == symbol
    assert stop.stop_type == StopType.BAR
    assert stop.pos_type == "short"
    assert stop.percent is None
    assert stop.points is None
    assert stop.bars == 2
    assert stop.fill_price == PriceType.AVERAGE
    assert stop.limit_price is None


def test_to_result_when_buy_shares_and_sell_shares_then_error(ctx):
    ctx.buy_shares = 100
    ctx.sell_shares = 100
    with pytest.raises(
        ValueError,
        match=re.escape(
            "For each symbol, only one of buy_shares or sell_shares can be "
            "set per bar."
        ),
    ):
        ctx.to_result()


@pytest.mark.parametrize(
    "attr, value, error",
    [
        (
            "buy_limit_price",
            100,
            "buy_shares must be set when buy_limit_price is set.",
        ),
        (
            "buy_fill_price",
            PriceType.CLOSE,
            "buy_shares or hold_bars must be set when buy_fill_price is set.",
        ),
    ],
)
def test_to_result_when_not_buy_shares_then_error(ctx, attr, value, error):
    ctx.sell_shares = 100
    setattr(ctx, attr, value)
    with pytest.raises(ValueError, match=re.escape(error)):
        ctx.to_result()


@pytest.mark.parametrize(
    "attr, value, error",
    [
        (
            "sell_limit_price",
            100,
            "sell_shares must be set when sell_limit_price is set.",
        ),
        (
            "sell_fill_price",
            PriceType.CLOSE,
            "sell_shares or hold_bars must be set when sell_fill_price is "
            "set.",
        ),
    ],
)
def test_to_result_when_not_sell_shares_then_error(ctx, attr, value, error):
    ctx.buy_shares = 100
    setattr(ctx, attr, value)
    with pytest.raises(ValueError, match=re.escape(error)):
        ctx.to_result()


@pytest.mark.parametrize(
    "attr, value, error",
    [
        (
            "hold_bars",
            2,
            "Either buy_shares or sell_shares must be set when hold_bars is "
            "set.",
        ),
    ],
)
def test_to_result_not_buy_shares_and_not_sell_shares_then_error(
    ctx, attr, value, error
):
    setattr(ctx, attr, value)
    with pytest.raises(ValueError, match=re.escape(error)):
        ctx.to_result()


@pytest.mark.parametrize(
    "attr",
    [
        "stop_loss",
        "stop_loss_pct",
        "stop_loss_limit",
        "stop_profit",
        "stop_profit_pct",
        "stop_profit_limit",
        "stop_trailing",
        "stop_trailing_pct",
        "stop_trailing_limit",
    ],
)
def test_to_result_not_buy_shares_and_not_sell_shares_and_stop_then_error(
    ctx, attr
):
    setattr(ctx, attr, 10)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Either buy_shares or sell_shares must be set when a stop is set."
        ),
    ):
        ctx.to_result()


def test_result_when_not_buy_shares_and_not_sell_shares_then_return_none(ctx):
    assert ctx.to_result() is None


def test_result_when_default_buy_fill_price(ctx):
    ctx.buy_shares = 100
    result = ctx.to_result()
    assert result.buy_fill_price == PriceType.MIDDLE


def test_result_when_default_sell_fill_price(ctx):
    ctx.sell_shares = 100
    result = ctx.to_result()
    assert result.sell_fill_price == PriceType.MIDDLE


@pytest.mark.parametrize("pos_type", ["long", "short"])
@pytest.mark.parametrize(
    "stop_attr, expected_stop_type",
    [
        ("stop_loss", StopType.LOSS),
        ("stop_loss_pct", StopType.LOSS),
        ("stop_profit", StopType.PROFIT),
        ("stop_profit_pct", StopType.PROFIT),
        ("stop_trailing", StopType.TRAILING),
        ("stop_trailing_pct", StopType.TRAILING),
    ],
)
def test_to_result_when_stop(
    ctx, symbol, date, pos_type, stop_attr, expected_stop_type
):
    stop_limit = 200
    stop_amount = 20
    exit_price = PriceType.OPEN
    percent = None
    points = None
    if stop_attr.endswith("_pct"):
        percent = stop_amount
    else:
        points = stop_amount
    buy_shares = None
    sell_shares = None
    if pos_type == "long":
        buy_shares = 100
    else:
        sell_shares = 100
    ctx.buy_shares = buy_shares
    ctx.sell_shares = sell_shares
    setattr(ctx, stop_attr, stop_amount)
    setattr(ctx, f"{stop_attr.replace('_pct', '')}_limit", stop_limit)
    setattr(ctx, f"{stop_attr.replace('_pct', '')}_exit_price", exit_price)
    result = ctx.to_result()
    assert result.symbol == symbol
    assert result.date == date
    assert result.buy_fill_price == PriceType.MIDDLE
    assert result.buy_limit_price is None
    assert result.sell_fill_price == PriceType.MIDDLE
    assert result.sell_limit_price is None
    assert result.hold_bars is None
    assert result.score is None
    if pos_type == "long":
        assert result.buy_shares == 100
        assert result.sell_shares is None
        assert len(result.long_stops) == 1
        assert result.short_stops is None
        stop = next(iter(result.long_stops))
    else:
        assert result.sell_shares == 100
        assert result.buy_shares is None
        assert len(result.short_stops) == 1
        assert result.long_stops is None
        stop = next(iter(result.short_stops))
    assert stop.symbol == symbol
    assert stop.stop_type == expected_stop_type
    assert stop.pos_type == pos_type
    assert stop.percent == percent
    assert stop.points == points
    assert stop.bars is None
    assert stop.fill_price is None
    assert stop.limit_price == stop_limit
    assert stop.exit_price == exit_price


@pytest.mark.parametrize(
    "stop_attr", ["stop_loss", "stop_profit", "stop_trailing"]
)
def test_to_result_when_stop_pct_and_points_then_error(ctx, stop_attr):
    ctx.buy_shares = 100
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Only one of {stop_attr} or {stop_attr}_pct can be set."
        ),
    ):
        setattr(ctx, stop_attr, 20)
        setattr(ctx, f"{stop_attr}_pct", 20)
        ctx.to_result()


@pytest.mark.parametrize(
    "stop_attr", ["stop_loss", "stop_profit", "stop_trailing"]
)
def test_to_result_when_stop_limit_and_not_stop_then_error(ctx, stop_attr):
    ctx.buy_shares = 100
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Either {stop_attr} or {stop_attr}_pct must be set when "
            f"{stop_attr}_limit is set."
        ),
    ):
        setattr(ctx, f"{stop_attr}_limit", 20)
        ctx.to_result()


@pytest.mark.parametrize(
    "stop_attr", ["stop_loss", "stop_profit", "stop_trailing"]
)
def test_to_result_when_stop_exit_price_and_not_stop_then_error(
    ctx, stop_attr
):
    ctx.buy_shares = 100
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Either {stop_attr} or {stop_attr}_pct must be set when "
            f"{stop_attr}_exit_price is set."
        ),
    ):
        setattr(ctx, f"{stop_attr}_exit_price", PriceType.CLOSE)
        ctx.to_result()


@pytest.mark.parametrize(
    "stop_attr", ["stop_loss", "stop_profit", "stop_trailing"]
)
def test_to_result_when_stop_exit_not_valid_then_error(ctx, stop_attr):
    ctx.buy_shares = 100
    with pytest.raises(
        ValueError,
        match=re.escape("Stop exit price must be a PriceType."),
    ):
        setattr(ctx, f"{stop_attr}_pct", 10)
        setattr(ctx, f"{stop_attr}_exit_price", 20)
        ctx.to_result()


def test_orders(ctx_with_orders, orders):
    assert tuple(ctx_with_orders.orders()) == orders


def test_orders_when_empty(ctx):
    assert not len(list(ctx.orders()))


def test_trades(ctx_with_orders, trades):
    assert tuple(ctx_with_orders.trades()) == trades


def test_trades_when_empty(ctx):
    assert not len(list(ctx.trades()))


@pytest.mark.parametrize(
    "cover_attr, buy_attr",
    [
        ("cover_fill_price", "buy_fill_price"),
        ("cover_shares", "buy_shares"),
        ("cover_limit_price", "buy_limit_price"),
    ],
)
def test_cover(ctx, cover_attr, buy_attr):
    setattr(ctx, cover_attr, 99)
    assert getattr(ctx, cover_attr) == 99
    assert getattr(ctx, cover_attr) == getattr(ctx, buy_attr)
    assert ctx._cover is True


def test_set_exec_ctx_data(ctx, sym_end_index):
    date = np.datetime64("2020-01-01")
    ctx._foreign = {"SPY": np.random.rand(100)}
    ctx._cover = True
    ctx.buy_fill_price = PriceType.AVERAGE
    ctx.buy_shares = 100
    ctx.buy_limit_price = 99
    ctx.sell_fill_price = PriceType.CLOSE
    ctx.sell_shares = 200
    ctx.sell_limit_price = 80
    ctx.hold_bars = 5
    ctx.score = 45.5
    ctx.stop_loss = 10
    ctx.stop_loss_pct = 20
    ctx.stop_loss_limit = 99
    ctx.stop_profit = 20
    ctx.stop_profit_pct = 30
    ctx.stop_profit_limit = 99.99
    ctx.stop_trailing = 100
    ctx.stop_trailing_pct = 15
    ctx.stop_trailing_limit = 80.8
    set_exec_ctx_data(ctx, date)
    assert ctx.dt == to_datetime(date)
    assert ctx.bars == sym_end_index[ctx.symbol]
    assert not ctx._foreign
    assert ctx._cover is False
    assert ctx.buy_fill_price is None
    assert ctx.buy_shares is None
    assert ctx.buy_limit_price is None
    assert ctx.sell_fill_price is None
    assert ctx.sell_shares is None
    assert ctx.sell_limit_price is None
    assert ctx.hold_bars is None
    assert ctx.score is None
    assert ctx.stop_loss is None
    assert ctx.stop_loss_pct is None
    assert ctx.stop_loss_limit is None
    assert ctx.stop_profit is None
    assert ctx.stop_profit_pct is None
    assert ctx.stop_profit_limit is None
    assert ctx.stop_trailing is None
    assert ctx.stop_trailing_pct is None
    assert ctx.stop_trailing_limit is None


def test_set_pos_ctx_data(
    date,
    portfolio,
    col_scope,
    ind_scope,
    input_scope,
    pred_scope,
    pending_order_scope,
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
            long_stops=None,
            short_stops=None,
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
            long_stops=None,
            short_stops=None,
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
            long_stops=None,
            short_stops=None,
        ),
    ]
    sessions = {"SPY": {}, "AAPL": {}, "TSLA": {"foo": 1}}
    ctx = PosSizeContext(
        StrategyConfig(max_long_positions=1),
        portfolio,
        col_scope,
        ind_scope,
        input_scope,
        pred_scope,
        pending_order_scope,
        trained_models,
        sessions,
        sym_end_index,
    )
    set_pos_size_ctx_data(ctx, buy_results, sell_results)
    assert ctx.sessions == sessions
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


def test_cancel_pending_order(ctx, pending_orders):
    assert ctx.cancel_pending_order(pending_orders[0].id)
    orders = tuple(ctx.pending_orders())
    assert orders == tuple([pending_orders[1]])


def test_cancel_all_pending_orders(ctx):
    ctx.cancel_all_pending_orders()
    assert not tuple(ctx.pending_orders())


def test_cancel_all_pending_orders_when_symbol(ctx, pending_orders):
    ctx.cancel_all_pending_orders("SPY")
    assert tuple(ctx.pending_orders()) == tuple([pending_orders[1]])


def test_pending_orders(ctx, pending_orders):
    assert tuple(ctx.pending_orders()) == pending_orders
    assert tuple(ctx.pending_orders("AAPL")) == tuple([pending_orders[1]])
