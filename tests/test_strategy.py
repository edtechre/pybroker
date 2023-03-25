"""Unit tests for strategy.py module."""

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

import joblib
import numpy as np
import os
import pandas as pd
import pytest
import re
from .fixtures import *
from collections import deque
from datetime import datetime
from decimal import Decimal
from pybroker.common import DataCol, PriceType, to_datetime
from pybroker.config import StrategyConfig
from pybroker.data import DataSource
from pybroker.portfolio import (
    Order,
    Portfolio,
    PortfolioBar,
    PositionBar,
    Trade,
)
from pybroker.scope import PendingOrder
from pybroker.strategy import (
    BacktestMixin,
    Execution,
    ExecSymbol,
    Strategy,
    TestResult,
    WalkforwardMixin,
)
from unittest.mock import Mock


@pytest.fixture(params=[200, 202])
def dates_length(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def lookahead(request):
    return request.param


@pytest.fixture()
def dates():
    dates = pd.date_range(start="1/1/2018", end="1/1/2019").tolist()
    return sorted(dates + dates.copy())


@pytest.fixture(params=list(range(1, 6)))
def windows(request):
    return request.param


@pytest.fixture(params=np.arange(0, 1.05, 0.05).tolist())
def train_size(request):
    return request.param


@pytest.fixture(params=[True, False])
def shuffle(request):
    return request.param


class TestWalkforwardMixin:
    def test_walkforward_split_1(
        self, dates, dates_length, windows, lookahead, train_size, shuffle
    ):
        self._verify_windows(
            dates, dates_length, windows, lookahead, train_size, shuffle
        )

    @pytest.mark.parametrize(
        "dates_length, windows, lookahead",
        [(22, 5, 1), (20, 5, 1), (22, 2, 2), (20, 2, 2)],
    )
    def test_walkforward_split_2(
        self, dates, dates_length, windows, lookahead, train_size, shuffle
    ):
        self._verify_windows(
            dates, dates_length, windows, lookahead, train_size, shuffle
        )

    def _verify_windows(
        self, dates, dates_length, windows, lookahead, train_size, shuffle
    ):
        df = self._data_frame(dates, dates_length)
        mixin = WalkforwardMixin()
        results = list(
            mixin.walkforward_split(
                df, windows, lookahead, train_size, shuffle
            )
        )
        dates = sorted(dates)
        assert len(results) == windows
        for train_idx, test_idx in results:
            assert len(dates) - (len(train_idx) + len(test_idx) * windows) >= 0
            assert not (set(train_idx) & set(test_idx))
            assert len(train_idx) or len(test_idx)
            if len(train_idx) and len(test_idx):
                train_end_index = sorted(train_idx)[-1] + lookahead * 2
                test_start_index = sorted(test_idx)[0]
                assert dates[train_end_index] == dates[test_start_index]
                assert dates[train_end_index - 2] != dates[test_start_index]
            if train_size == 0.5:
                assert len(train_idx) == len(test_idx)

    @pytest.mark.parametrize(
        "dates_length, windows, lookahead, train_size",
        [
            (11, -1, 1, 0.5),
            (11, 5, 0, 0.5),
            (11, 5, 1, -1),
            (0, 2, 1, 0.5),
            (12, 7, 2, 0.5),
            (1, 1, 2, 0.5),
            (1, 1, 10, 0.5),
            (1, 2, 1, 0.5),
            (10, 2, 11, 0.5),
        ],
    )
    def test_walkforward_split_when_invalid_params_then_error(
        self, dates, dates_length, windows, lookahead, train_size
    ):
        df = self._data_frame(dates, dates_length)
        mixin = WalkforwardMixin()
        with pytest.raises(ValueError):
            list(mixin.walkforward_split(df, windows, lookahead, train_size))

    def _data_frame(self, dates, dates_length):
        dates = dates[:dates_length]
        return pd.DataFrame(
            {"date": dates, "close": np.random.rand(len(dates))}
        )


def pos_size_handler(ctx):
    signals = tuple(ctx.signals())
    ctx.set_shares(signals[0], shares=1000)
    ctx.set_shares(signals[1], shares=2000)


class TestBacktestMixin:
    @pytest.mark.parametrize(
        "pos_size_handler, expected_buy_shares, expected_sell_shares",
        [(None, 200, 100), (pos_size_handler, 1000, 2000)],
    )
    def test_backtest_executions(
        self,
        data_source_df,
        pos_size_handler,
        expected_buy_shares,
        expected_sell_shares,
    ):
        def buy_exec_fn(ctx):
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.buy_limit_price = 100
            ctx.buy_shares = 200

        def sell_exec_fn(ctx):
            ctx.sell_fill_price = PriceType.CLOSE
            ctx.sell_limit_price = 50.5
            ctx.sell_shares = 100

        buy_exec = Execution(
            id=1,
            symbols=frozenset(["SPY"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        sell_exec = Execution(
            id=2,
            symbols=frozenset(["AAPL"]),
            fn=sell_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {buy_exec, sell_exec}
        mock_portfolio = Mock()
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
            buy_delay=1,
            sell_delay=1,
            max_long_positions=None,
            max_short_positions=None,
            pos_size_handler=pos_size_handler,
            exit_dates={},
        )
        buy_df = data_source_df[data_source_df["symbol"] == "SPY"]
        buy_dates = buy_df["date"].unique()[1:]
        assert len(mock_portfolio.buy.call_args_list) == len(buy_dates)
        for i, date in enumerate(buy_dates):
            _, kwargs = mock_portfolio.buy.call_args_list[i]
            assert kwargs["date"] == date
            assert kwargs["symbol"] == "SPY"
            assert kwargs["shares"] == expected_buy_shares
            assert kwargs["fill_price"] == Decimal(
                str(buy_df[buy_df["date"] == date]["close"].values[0])
            )
            assert kwargs["limit_price"] == 100
        sell_df = data_source_df[data_source_df["symbol"] == "AAPL"]
        sell_dates = sell_df["date"].unique()[1:]
        assert len(mock_portfolio.sell.call_args_list) == len(sell_dates)
        for i, date in enumerate(sell_dates):
            _, kwargs = mock_portfolio.sell.call_args_list[i]
            assert kwargs["date"] == date
            assert kwargs["symbol"] == "AAPL"
            assert kwargs["shares"] == expected_sell_shares
            assert kwargs["fill_price"] == Decimal(
                str(sell_df[sell_df["date"] == date]["close"].values[0])
            )
            assert kwargs["limit_price"] == 50.5

    def test_backtest_executions_when_buy_delay(self, data_source_df):
        def buy_exec_fn(ctx):
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.buy_limit_price = 100
            ctx.buy_shares = 200

        buy_exec = Execution(
            id=1,
            symbols=frozenset(["SPY"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {buy_exec}
        mock_portfolio = Mock()
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
            buy_delay=2,
            sell_delay=1,
            max_long_positions=None,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )
        buy_df = data_source_df[data_source_df["symbol"] == "SPY"]
        buy_dates = buy_df["date"].unique()[2:]
        assert len(mock_portfolio.buy.call_args_list) == len(buy_dates)
        for i, date in enumerate(buy_dates):
            _, kwargs = mock_portfolio.buy.call_args_list[i]
            assert kwargs["date"] == date
            assert kwargs["symbol"] == "SPY"
            assert kwargs["shares"] == 200
            assert kwargs["fill_price"] == Decimal(
                str(buy_df[buy_df["date"] == date]["close"].values[0])
            )
            assert kwargs["limit_price"] == 100

    def test_backtest_executions_when_sell_delay(self, data_source_df):
        def sell_exec_fn(ctx):
            ctx.sell_fill_price = PriceType.CLOSE
            ctx.sell_limit_price = 50.5
            ctx.sell_shares = 100

        sell_exec = Execution(
            id=1,
            symbols=frozenset(["AAPL"]),
            fn=sell_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {sell_exec}
        mock_portfolio = Mock()
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
            buy_delay=1,
            sell_delay=2,
            max_long_positions=None,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )
        sell_df = data_source_df[data_source_df["symbol"] == "AAPL"]
        sell_dates = sell_df["date"].unique()[2:]
        assert len(mock_portfolio.sell.call_args_list) == len(sell_dates)
        for i, date in enumerate(sell_dates):
            _, kwargs = mock_portfolio.sell.call_args_list[i]
            assert kwargs["date"] == date
            assert kwargs["symbol"] == "AAPL"
            assert kwargs["shares"] == 100
            assert kwargs["fill_price"] == Decimal(
                str(sell_df[sell_df["date"] == date]["close"].values[0])
            )
            assert kwargs["limit_price"] == 50.5

    def test_backtest_executions_when_invalid_buy_hold_bars_then_error(
        self, data_source_df
    ):
        def buy_exec_fn(ctx):
            ctx.buy_shares = 200
            ctx.hold_bars = 0

        buy_exec = Execution(
            id=1,
            symbols=frozenset(["AAPL"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {buy_exec}
        mixin = BacktestMixin()
        with pytest.raises(
            ValueError, match=re.escape("hold_bars must be greater than 0.")
        ):
            mixin.backtest_executions(
                executions=execs,
                sessions=self._sessions(execs),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Mock(),
                buy_delay=1,
                sell_delay=1,
                max_long_positions=None,
                max_short_positions=None,
                pos_size_handler=None,
                exit_dates={},
            )

    def test_backtest_executions_when_invalid_sell_hold_bars_then_error(
        self, data_source_df
    ):
        def sell_exec_fn(ctx):
            ctx.sell_shares = 100
            ctx.hold_bars = 0

        sell_exec = Execution(
            id=1,
            symbols=frozenset(["AAPL"]),
            fn=sell_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {sell_exec}
        mixin = BacktestMixin()
        with pytest.raises(
            ValueError, match=re.escape("hold_bars must be greater than 0.")
        ):
            mixin.backtest_executions(
                executions=execs,
                sessions=self._sessions(execs),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Mock(),
                buy_delay=1,
                sell_delay=1,
                max_long_positions=None,
                max_short_positions=None,
                pos_size_handler=None,
                exit_dates={},
            )

    def test_backtest_executions_when_no_fn(self, data_source_df):
        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL"]),
            fn=None,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        portfolio = Portfolio(100_000)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
            buy_delay=1,
            sell_delay=1,
            max_long_positions=None,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )
        assert len(portfolio.bars) == len(data_source_df["date"].unique())
        assert not len(portfolio.position_bars)
        assert not len(portfolio.orders)
        assert not len(portfolio.trades)

    def test_backtest_executions_when_empty_symbols(self, data_source_df):
        def buy_exec_fn(ctx):
            ctx.buy_shares = 200

        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        portfolio = Portfolio(100_000)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df[data_source_df["symbol"] != "AAPL"],
            portfolio=portfolio,
            buy_delay=1,
            sell_delay=1,
            max_long_positions=None,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )
        assert len(portfolio.bars) == len(data_source_df["date"].unique())
        assert not len(portfolio.position_bars)
        assert not len(portfolio.orders)
        assert not len(portfolio.trades)

    def test_backtest_executions_when_buy_delay_after_period(
        self, data_source_df
    ):
        def buy_exec_fn(ctx):
            ctx.buy_shares = 200

        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        portfolio = Portfolio(100_000)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
            buy_delay=1000,
            sell_delay=1,
            max_long_positions=None,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )

        assert len(portfolio.bars) == len(data_source_df["date"].unique())
        assert not len(portfolio.position_bars)
        assert not len(portfolio.orders)
        assert not len(portfolio.trades)

    def test_backtest_executions_when_sell_delay_after_period(
        self, data_source_df
    ):
        def sell_exec_fn(ctx):
            ctx.sell_shares = 200

        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL"]),
            fn=sell_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        portfolio = Portfolio(100_000)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
            buy_delay=1,
            sell_delay=1000,
            max_long_positions=None,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )
        assert len(portfolio.bars)
        assert not len(portfolio.position_bars)
        assert not len(portfolio.orders)
        assert not len(portfolio.trades)

    def test_backtest_executions_when_buy_score(self, data_source_df):
        def buy_exec_fn(ctx):
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.buy_shares = 200
            if ctx.symbol == "SPY":
                ctx.score = 1
            else:
                ctx.score = 0

        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL", "SPY"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        mock_portfolio = Mock()
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
            buy_delay=1,
            sell_delay=1,
            max_long_positions=1,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )
        df = data_source_df[data_source_df["symbol"].isin(["AAPL", "SPY"])]
        buy_dates = sorted(df["date"].values)[2:]
        assert len(mock_portfolio.buy.call_args_list) == len(buy_dates)
        for i, date in enumerate(buy_dates):
            sym = "SPY" if i % 2 == 0 else "AAPL"
            _, kwargs = mock_portfolio.buy.call_args_list[i]
            assert kwargs["date"] == date
            assert kwargs["symbol"] == sym
            assert kwargs["shares"] == 200
            assert kwargs["fill_price"] == Decimal(
                str(
                    df[(df["date"] == date) & (df["symbol"] == sym)][
                        "close"
                    ].values[0]
                )
            )
            assert kwargs["limit_price"] is None

    def test_backtest_executions_when_sell_score(self, data_source_df):
        def sell_exec_fn(ctx):
            ctx.sell_fill_price = PriceType.CLOSE
            ctx.sell_shares = 200
            if ctx.symbol == "AAPL":
                ctx.score = 1
            else:
                ctx.score = 0

        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL", "SPY"]),
            fn=sell_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        mock_portfolio = Mock()
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
            buy_delay=1,
            sell_delay=1,
            max_long_positions=None,
            max_short_positions=1,
            pos_size_handler=None,
            exit_dates={},
        )
        df = data_source_df[data_source_df["symbol"].isin(["AAPL", "SPY"])]
        sell_dates = sorted(df["date"].values)[2:]
        assert len(mock_portfolio.sell.call_args_list) == len(sell_dates)
        for i, date in enumerate(sell_dates):
            sym = "AAPL" if i % 2 == 0 else "SPY"
            _, kwargs = mock_portfolio.sell.call_args_list[i]
            assert kwargs["date"] == date
            assert kwargs["symbol"] == sym
            assert kwargs["shares"] == 200
            assert kwargs["fill_price"] == Decimal(
                str(
                    df[(df["date"] == date) & (df["symbol"] == sym)][
                        "close"
                    ].values[0]
                )
            )
            assert kwargs["limit_price"] is None

    @pytest.mark.parametrize(
        "price_type, expected_fill_price",
        [
            (50, 50),
            (Decimal(111.1), Decimal(111.1)),
            (lambda _symbol, _bar_data: 60, 60),
            (PriceType.OPEN, 200),
            (PriceType.HIGH, 400),
            (PriceType.LOW, 100),
            (PriceType.CLOSE, 300),
            (PriceType.MIDDLE, round((100 + (400 - 100) / 2.0), 2)),
            (PriceType.AVERAGE, round((200 + 100 + 400 + 300) / 4.0, 2)),
        ],
    )
    def test_backtest_executions_get_price(
        self, price_type, expected_fill_price
    ):
        dates = pd.date_range(start="1/1/2018", end="1/1/2019").tolist()
        df = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["SPY"] * len(dates),
                "open": np.repeat(200, len(dates)),
                "high": np.repeat(400, len(dates)),
                "low": np.repeat(100, len(dates)),
                "close": np.repeat(300, len(dates)),
            }
        )

        def buy_exec_fn(ctx):
            ctx.buy_shares = 200
            ctx.buy_fill_price = price_type
            ctx.buy_limit_price = 101

        exec = Execution(
            id=1,
            symbols=frozenset(["SPY"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        mock_portfolio = Mock()
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=df,
            portfolio=mock_portfolio,
            buy_delay=1,
            sell_delay=1,
            max_long_positions=None,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )
        buy_dates = dates[1:]
        assert len(mock_portfolio.buy.call_args_list) == len(buy_dates)
        for i, date in enumerate(buy_dates):
            _, kwargs = mock_portfolio.buy.call_args_list[i]
            assert kwargs["date"] == date
            assert kwargs["symbol"] == "SPY"
            assert kwargs["shares"] == 200
            assert kwargs["fill_price"] == expected_fill_price
            assert kwargs["limit_price"] == 101

    def test_backtest_executions_get_price_when_invalid_price_then_error(
        self, data_source_df
    ):
        def buy_exec_fn(ctx):
            ctx.buy_shares = 200
            ctx.buy_fill_price = "invalid"

        exec = Execution(
            id=1,
            symbols=frozenset(["SPY"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        mixin = BacktestMixin()
        with pytest.raises(ValueError, match=r"Unknown price: .*"):
            mixin.backtest_executions(
                executions=execs,
                sessions=self._sessions(execs),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Portfolio(100_000),
                buy_delay=1,
                sell_delay=1,
                max_long_positions=None,
                max_short_positions=None,
                pos_size_handler=None,
                exit_dates={},
            )

    def test_backtest_executions_when_buy_limit_and_no_shares_then_error(
        self, data_source_df
    ):
        def buy_exec_fn(ctx):
            ctx.buy_limit_price = 100

        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL", "SPY"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        mixin = BacktestMixin()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "buy_shares must be set when buy_limit_price is set."
            ),
        ):
            mixin.backtest_executions(
                executions=execs,
                sessions=self._sessions(execs),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Portfolio(100_000),
                buy_delay=1,
                sell_delay=1,
                max_long_positions=1,
                max_short_positions=None,
                pos_size_handler=None,
                exit_dates={},
            )

    def test_backtest_executions_when_sell_limit_and_no_shares_then_error(
        self, data_source_df
    ):
        def sell_exec_fn(ctx):
            ctx.sell_limit_price = 100

        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL", "SPY"]),
            fn=sell_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        mixin = BacktestMixin()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "sell_shares must be set when sell_limit_price is set."
            ),
        ):
            mixin.backtest_executions(
                executions=execs,
                sessions=self._sessions(execs),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Portfolio(100_000),
                buy_delay=1,
                sell_delay=1,
                max_long_positions=1,
                max_short_positions=None,
                pos_size_handler=None,
                exit_dates={},
            )

    def test_backtest_executions_when_buy_order_not_filled(
        self, data_source_df
    ):
        def buy_exec_fn(ctx):
            ctx.buy_fill_price = 100
            ctx.buy_shares = 100

        exec = Execution(
            id=1,
            symbols=frozenset(["SPY"]),
            fn=buy_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        portfolio = Portfolio(1)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
            buy_delay=1,
            sell_delay=1,
            max_long_positions=1,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )
        assert not len(portfolio.orders)

    def test_backtest_executions_when_sell_order_not_filled(
        self, data_source_df
    ):
        def sell_exec_fn(ctx):
            ctx.sell_fill_price = 100
            ctx.sell_limit_price = 200
            ctx.sell_shares = 100

        exec = Execution(
            id=1,
            symbols=frozenset(["SPY"]),
            fn=sell_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        portfolio = Portfolio(1)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            executions=execs,
            sessions=self._sessions(execs),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
            buy_delay=1,
            sell_delay=1,
            max_long_positions=1,
            max_short_positions=None,
            pos_size_handler=None,
            exit_dates={},
        )
        assert not len(portfolio.orders)

    def _sessions(self, execs):
        return {
            ExecSymbol(exec.id, sym): {}
            for exec in execs
            for sym in exec.symbols
        }


@pytest.fixture()
def executions_train_only():
    return [
        {
            "fn": None,
            "symbols": ["AAPL", "MSFT"],
            "models": None,
            "indicators": None,
        },
        {"fn": None, "symbols": "SPY", "models": None, "indicators": None},
        {"fn": None, "symbols": "QQQ", "models": None, "indicators": None},
    ]


@pytest.fixture()
def executions_only(executions_train_only):
    def exec_fn_1(ctx):
        if ctx.long_pos():
            ctx.sell_all_shares()
        else:
            ctx.buy_shares = 100

    def exec_fn_2(ctx):
        ctx.sell_fill_price = PriceType.AVERAGE
        ctx.sell_shares = 10
        ctx.hold_bars = 1

    executions_train_only[0]["fn"] = exec_fn_1
    executions_train_only[1]["fn"] = exec_fn_2
    executions_train_only[2]["fn"] = exec_fn_2
    return executions_train_only


@pytest.fixture()
def executions_with_indicators(executions_only, hhv_ind, llv_ind):
    def exec_fn_1(ctx):
        assert len(ctx.indicator(hhv_ind.name))

    def exec_fn_2(ctx):
        assert len(ctx.indicator(hhv_ind.name))
        assert len(ctx.indicator(llv_ind.name))

    executions_only[0]["indicators"] = hhv_ind
    executions_only[0]["fn"] = exec_fn_1
    executions_only[1]["indicators"] = (hhv_ind, llv_ind)
    executions_only[1]["fn"] = exec_fn_2
    return executions_only


@pytest.fixture()
def executions_with_models(executions_only, model_source):
    def exec_fn(ctx):
        assert type(ctx.model(model_source.name)) == FakeModel

    executions_only[0]["models"] = model_source
    executions_only[0]["fn"] = exec_fn
    return executions_only


@pytest.fixture()
def executions_with_models_and_indicators(
    executions_only, model_source, hhv_ind, llv_ind
):
    def exec_fn_1(ctx):
        assert len(ctx.indicator(llv_ind.name))

    executions_only[0]["indicators"] = llv_ind
    executions_only[0]["fn"] = exec_fn_1

    def exec_fn_2(ctx):
        assert len(ctx.indicator(hhv_ind.name))
        assert type(ctx.model(model_source.name)) == FakeModel

    executions_only[1]["indicators"] = hhv_ind
    executions_only[1]["models"] = model_source
    executions_only[1]["fn"] = exec_fn_2
    return executions_only


@pytest.fixture(
    params=[
        (None, None),
        ("2020/06/01", None),
        (None, "2021-10-31"),
        ("1/1/2021", "2021-09-01"),
    ]
)
def date_range(request):
    return request.param


@pytest.fixture(params=[True, False])
def calc_bootstrap(request):
    return request.param


@pytest.fixture(params=[True, False])
def disable_parallel(request):
    return request.param


@pytest.fixture(params=[None, "weds", ("mon", "fri")])
def days(request):
    return request.param


@pytest.fixture(params=[None, ("10:00", "1:00")])
def between_time(request):
    return request.param


class FakeDataSource(DataSource):
    def _fetch_data(self, symbols, start_date, end_date, timeframe):
        return joblib.load(
            os.path.join(os.path.dirname(__file__), "testdata/daily_1.joblib")
        )


START_DATE = "2020-01-02"
END_DATE = "2021-12-31"


class TestStrategy:
    @pytest.mark.parametrize(
        "data_source",
        [FakeDataSource(), pytest.lazy_fixture("data_source_df")],
    )
    @pytest.mark.parametrize(
        "executions",
        [
            pytest.lazy_fixture("executions_train_only"),
            pytest.lazy_fixture("executions_only"),
            pytest.lazy_fixture("executions_with_indicators"),
            pytest.lazy_fixture("executions_with_models"),
            pytest.lazy_fixture("executions_with_models_and_indicators"),
        ],
    )
    def test_walkforward(
        self,
        data_source,
        executions,
        date_range,
        days,
        between_time,
        calc_bootstrap,
        disable_parallel,
    ):
        config = StrategyConfig(
            bootstrap_samples=100, bootstrap_sample_size=10
        )
        strategy = Strategy(data_source, START_DATE, END_DATE, config)
        for exec in executions:
            strategy.add_execution(**exec)
        result = strategy.walkforward(
            start_date=date_range[0],
            end_date=date_range[1],
            windows=3,
            lookahead=1,
            timeframe="1d",
            days=days,
            between_time=between_time,
            calc_bootstrap=calc_bootstrap,
            disable_parallel=disable_parallel,
        )
        if all(map(lambda e: not e["fn"], executions)):
            assert result is None
            return
        assert type(result) == TestResult
        assert result.metrics is not None
        assert isinstance(result.metrics_df, pd.DataFrame)
        assert not result.metrics_df.empty
        if date_range[0] is None:
            assert result.start_date == datetime.strptime(
                START_DATE, "%Y-%m-%d"
            )
        else:
            assert result.start_date == pd.to_datetime(date_range[0])
        if date_range[1] is None:
            assert result.end_date == datetime.strptime(END_DATE, "%Y-%m-%d")
        else:
            assert result.end_date == pd.to_datetime(date_range[1])
        assert isinstance(result.portfolio, pd.DataFrame)
        assert not result.portfolio.empty
        assert isinstance(result.positions, pd.DataFrame)
        assert isinstance(result.orders, pd.DataFrame)
        if calc_bootstrap:
            assert not result.bootstrap.conf_intervals.empty
            assert not result.bootstrap.drawdown_conf.empty
        else:
            assert result.bootstrap is None

    def test_walkforward_results(self, data_source_df):
        def exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100

        data_source_df = data_source_df[
            data_source_df["date"] <= to_datetime(END_DATE)
        ]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ["AAPL", "SPY"])
        result = strategy.walkforward(windows=3, calc_bootstrap=False)
        dates = set()
        for _, test_idx in strategy.walkforward_split(
            data_source_df, windows=3, lookahead=1, train_size=0.5
        ):
            df = data_source_df.loc[test_idx]
            df = df[df["symbol"].isin(["AAPL", "SPY"])]
            dates.update(df["date"].values)
        assert result.start_date == to_datetime(START_DATE)
        assert result.end_date == to_datetime(END_DATE)
        dates_list = list(dates)
        dates_list.sort()
        assert np.array_equal(result.portfolio.index, dates_list)
        assert len(result.positions) == 2 * len(dates) - 2
        assert np.array_equal(
            result.positions.index.get_level_values(1).unique(), dates_list[1:]
        )
        assert len(result.orders) == 2
        assert not len(result.trades)

    def test_walkforward_when_no_executions_then_error(self, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        with pytest.raises(
            ValueError, match=re.escape("No executions were added.")
        ):
            strategy.walkforward(windows=3, lookahead=1)

    def test_walkforward_when_empty_data_source_then_error(self):
        df = pd.DataFrame(columns=[col.value for col in DataCol])
        strategy = Strategy(df, START_DATE, END_DATE)
        strategy.add_execution(None, "SPY")
        with pytest.raises(
            ValueError, match=re.escape("DataSource is empty.")
        ):
            strategy.walkforward(windows=3, lookahead=1)

    @pytest.mark.parametrize(
        "start_date_1, end_date_1, start_date_2, end_date_2, expected_msg",
        [
            (
                "2020-03-01",
                "2020-02-20",
                None,
                None,
                r"start_date (.*) must be on or before end_date (.*)\.",
            ),
            (
                "2020-03-01",
                "2020-09-30",
                "2020-01-01",
                None,
                r"start_date must be between .* and .*\.",
            ),
            (
                "2020-03-01",
                "2020-09-30",
                "2020-10-01",
                None,
                r"start_date must be between .* and .*\.",
            ),
            (
                "2020-03-01",
                "2020-09-30",
                None,
                "2020-02-01",
                r"end_date must be between .* and .*\.",
            ),
            (
                "2020-03-01",
                "2020-09-30",
                None,
                "2020-10-31",
                r"end_date must be between .* and .*\.",
            ),
            (
                "2020-03-01",
                "2020-09-30",
                "2020-05-01",
                "2020-04-01",
                r"start_date (.*) must be on or before end_date (.*)\.",
            ),
        ],
    )
    def test_walkforward_when_invalid_dates_then_error(
        self,
        executions_only,
        data_source_df,
        start_date_1,
        end_date_1,
        start_date_2,
        end_date_2,
        expected_msg,
    ):
        with pytest.raises(ValueError, match=expected_msg):
            strategy = Strategy(data_source_df, start_date_1, end_date_1)
            for exec in executions_only:
                strategy.add_execution(**exec)
            strategy.walkforward(
                windows=3,
                lookahead=1,
                start_date=start_date_2,
                end_date=end_date_2,
            )

    def test_backtest(self, executions_only, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        for exec in executions_only:
            strategy.add_execution(**exec)
        result = strategy.backtest()
        assert type(result) == TestResult
        assert result.start_date == datetime.strptime(START_DATE, "%Y-%m-%d")
        assert result.end_date == datetime.strptime(END_DATE, "%Y-%m-%d")
        assert not result.portfolio.empty
        assert not result.bootstrap.conf_intervals.empty
        assert not result.bootstrap.drawdown_conf.empty

    @pytest.mark.parametrize("tz", ["UTC", None])
    @pytest.mark.parametrize(
        "between_time, expected_hour",
        [(None, None), (("10:00", "1:00"), (10, 13))],
    )
    @pytest.mark.parametrize(
        "days, expected_days",
        [
            (None, None),
            ("tues", {1}),
            (["weds", "fri"], {2, 4}),
        ],
    )
    def test_filter_dates(
        self,
        tz,
        between_time,
        expected_hour,
        days,
        expected_days,
        data_source_df,
    ):
        data_source_df["date"] = data_source_df["date"].dt.tz_localize(tz)
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        start_date = pd.to_datetime("1/1/2021").to_pydatetime()
        end_date = pd.to_datetime("12/1/2021").to_pydatetime()
        df = strategy._filter_dates(
            data_source_df,
            start_date,
            end_date,
            between_time=between_time,
            days=strategy._to_day_ids(days),
        )
        assert df.iloc[0]["date"] >= start_date
        assert df.iloc[-1]["date"] <= end_date
        row_days = set()
        for _, row in df.iterrows():
            if between_time is not None:
                assert row["date"].hour >= expected_hour[0]
                assert row["date"].hour <= expected_hour[1]
            row_days.add(row["date"].weekday())
        if expected_days is not None:
            assert row_days == expected_days

    def test_filter_dates_when_empty(self, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        start_date = pd.to_datetime("1/1/2021").to_pydatetime()
        end_date = pd.to_datetime("12/1/2021").to_pydatetime()
        df = strategy._filter_dates(
            data_source_df,
            start_date,
            end_date,
            between_time=("9:00", "10:00"),
            days=strategy._to_day_ids("tues"),
        )
        assert df.empty

    def test_filter_dates_when_invalid_between_time_then_error(
        self, data_source_df
    ):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        start_date = pd.to_datetime("1/1/2021").to_pydatetime()
        end_date = pd.to_datetime("12/1/2021").to_pydatetime()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "between_time must be a tuple[str, str] of start time and end"
                " time, received '9:00'."
            ),
        ):
            strategy._filter_dates(
                data_source_df,
                start_date,
                end_date,
                days=None,
                between_time=("9:00"),
            )

    def test_add_execution_when_empty_symbols_then_error(self, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        with pytest.raises(
            ValueError, match=re.escape("symbols cannot be empty.")
        ):
            strategy.add_execution(None, [])

    def test_add_execution_when_duplicate_symbol_then_error(
        self, data_source_df
    ):
        def exec_fn_1(ctx):
            ctx.buy_shares = 100

        def exec_fn_2(ctx):
            ctx.sell_shares = 100

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn_1, ["AAPL", "SPY"])
        with pytest.raises(
            ValueError,
            match=re.escape("AAPL was already added to an execution."),
        ):
            strategy.add_execution(exec_fn_2, "AAPL")

    @pytest.mark.parametrize(
        "initial_cash, max_long_positions, max_short_positions, buy_delay,"
        "sell_delay, bootstrap_samples, bootstrap_sample_size, expected_msg",
        [
            (
                -1,
                None,
                None,
                1,
                1,
                100,
                10,
                "initial_cash must be greater than 0.",
            ),
            (
                10_000,
                0,
                None,
                1,
                1,
                100,
                10,
                "max_long_positions must be greater than 0.",
            ),
            (
                10_000,
                None,
                0,
                1,
                1,
                100,
                10,
                "max_short_positions must be greater than 0.",
            ),
            (
                10_000,
                None,
                None,
                0,
                1,
                100,
                10,
                "buy_delay must be greater than 0.",
            ),
            (
                10_000,
                None,
                None,
                1,
                0,
                100,
                10,
                "sell_delay must be greater than 0.",
            ),
            (
                10_000,
                None,
                None,
                1,
                1,
                0,
                10,
                "bootstrap_samples must be greater than 0.",
            ),
            (
                10_000,
                None,
                None,
                1,
                1,
                100,
                0,
                "bootstrap_sample_size must be greater than 0.",
            ),
        ],
    )
    def test_when_invalid_config_then_error(
        self,
        data_source_df,
        initial_cash,
        max_long_positions,
        max_short_positions,
        buy_delay,
        sell_delay,
        bootstrap_samples,
        bootstrap_sample_size,
        expected_msg,
    ):
        config = StrategyConfig(
            initial_cash=initial_cash,
            max_long_positions=max_long_positions,
            max_short_positions=max_short_positions,
            buy_delay=buy_delay,
            sell_delay=sell_delay,
            bootstrap_samples=bootstrap_samples,
            bootstrap_sample_size=bootstrap_sample_size,
        )
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            Strategy(data_source_df, START_DATE, END_DATE, config)

    def test_when_data_source_missing_columns_then_error(self):
        values = np.repeat(1, 100)
        df = pd.DataFrame(
            {
                "symbol": ["SPY"] * 100,
                "open": values,
                "high": values,
                "low": values,
                "close": values,
            }
        )
        with pytest.raises(
            ValueError,
            match=re.escape("DataFrame is missing required columns: ['date']"),
        ):
            Strategy(df, START_DATE, END_DATE)

    def test_when_invalid_data_source_type_then_error(self):
        with pytest.raises(TypeError, match=r"Invalid data_source type: .*"):
            Strategy({}, START_DATE, END_DATE)

    def test_clear_executions(self):
        df = pd.DataFrame(columns=[col.value for col in DataCol])
        strategy = Strategy(df, START_DATE, END_DATE)
        strategy.add_execution(None, "SPY")
        strategy.clear_executions()
        assert not strategy._executions

    @pytest.mark.parametrize(
        "enable_fractional_shares, expected_shares_type,"
        "expected_short_shares, expected_long_shares",
        [(True, np.float_, 0.1, 3.14), (False, np.int_, 0, 3)],
    )
    def test_to_test_result_when_fractional_shares(
        self,
        data_source_df,
        enable_fractional_shares,
        expected_shares_type,
        expected_long_shares,
        expected_short_shares,
    ):
        portfolio = Portfolio(100_000)
        portfolio.bars = deque(
            (
                PortfolioBar(
                    date=np.datetime64(START_DATE),
                    cash=Decimal(100_000),
                    equity=Decimal(100_000),
                    margin=Decimal(),
                    market_value=Decimal(100_000),
                    pnl=Decimal(1000),
                    fees=Decimal(),
                ),
            )
        )
        portfolio.position_bars = deque(
            (
                PositionBar(
                    symbol="SPY",
                    date=np.datetime64(START_DATE),
                    long_shares=Decimal("3.14"),
                    short_shares=Decimal("0.1"),
                    close=Decimal(100),
                    equity=Decimal(100_000),
                    market_value=Decimal(100_000),
                    margin=Decimal(),
                    unrealized_pnl=Decimal(100),
                ),
            )
        )
        portfolio.orders = deque(
            (
                Order(
                    id=1,
                    type="buy",
                    symbol="SPY",
                    date=np.datetime64(START_DATE),
                    shares=Decimal("3.14"),
                    limit_price=Decimal(100),
                    fill_price=Decimal(99),
                    fees=Decimal(),
                ),
            )
        )
        portfolio.trades = deque(
            (
                Trade(
                    id=1,
                    type="long",
                    symbol="SPY",
                    entry_date=np.datetime64(START_DATE),
                    exit_date=np.datetime64(END_DATE),
                    entry=Decimal(100),
                    exit=Decimal(101),
                    shares=Decimal("3.14"),
                    pnl=Decimal(1000),
                    return_pct=Decimal("10.3"),
                    agg_pnl=Decimal(1000),
                    bars=2,
                    pnl_per_bar=Decimal(500),
                    stop=None,
                ),
            )
        )
        config = StrategyConfig(
            enable_fractional_shares=enable_fractional_shares
        )
        strategy = Strategy(
            data_source_df,
            START_DATE,
            END_DATE,
            config,
        )
        result = strategy._to_test_result(
            START_DATE, END_DATE, portfolio, calc_bootstrap=False
        )
        assert np.issubdtype(
            result.positions["long_shares"].dtype, expected_shares_type
        )
        assert np.issubdtype(
            result.positions["short_shares"].dtype, expected_shares_type
        )
        assert np.issubdtype(
            result.orders["shares"].dtype, expected_shares_type
        )
        assert np.issubdtype(
            result.trades["shares"].dtype, expected_shares_type
        )
        assert (
            result.positions["long_shares"].values[0] == expected_long_shares
        )
        assert (
            result.positions["short_shares"].values[0] == expected_short_shares
        )
        assert result.orders["shares"].values[0] == expected_long_shares
        assert result.trades["shares"].values[0] == expected_long_shares

    def test_to_test_result_when_empty(self, data_source_df):
        portfolio = Portfolio(100_000)
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        result = strategy._to_test_result(
            START_DATE,
            END_DATE,
            portfolio,
            calc_bootstrap=False,
        )
        assert result.portfolio.empty
        assert result.positions.empty
        assert result.orders.empty
        assert result.trades.empty

    def test_backtest_when_exit_long_on_last_bar(self, data_source_df):
        def buy_exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.buy_fill_price = 150

        def sell_fill_price(_symbol, _bar_data):
            return 199.99

        config = StrategyConfig(
            exit_on_last_bar=True, exit_sell_fill_price=sell_fill_price
        )
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        dates = data_source_df[data_source_df["symbol"] == "SPY"][
            "date"
        ].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["type"] == "long"
        assert trade["symbol"] == "SPY"
        assert trade["entry_date"] == dates[1]
        assert trade["exit_date"] == dates[-1]
        assert trade["entry"] == 150
        assert trade["exit"] == 199.99
        assert trade["shares"] == 100

    def test_backtest_when_exit_short_on_last_bar(self, data_source_df):
        def sell_exec_fn(ctx):
            if not ctx.short_pos():
                ctx.sell_shares = 100
                ctx.sell_fill_price = 200

        def buy_fill_price(_symbol, _bar_data):
            return 99.99

        config = StrategyConfig(
            exit_on_last_bar=True, exit_cover_fill_price=buy_fill_price
        )
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(sell_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        dates = data_source_df[data_source_df["symbol"] == "SPY"][
            "date"
        ].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["type"] == "short"
        assert trade["symbol"] == "SPY"
        assert trade["entry_date"] == dates[1]
        assert trade["exit_date"] == dates[-1]
        assert trade["entry"] == 200
        assert trade["exit"] == 99.99
        assert trade["shares"] == 100

    def test_backtest_when_buy_shares_and_sell_shares_then_error(
        self, data_source_df
    ):
        def exec_fn(ctx):
            ctx.buy_shares = 100
            ctx.sell_shares = 100

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ["AAPL", "SPY"])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "For each symbol, only one of buy_shares or sell_shares can be"
                " set per bar."
            ),
        ):
            strategy.backtest()

    def test_backtest_pending_orders(self, data_source_df):
        buy_delay = 2
        dates = data_source_df[data_source_df["symbol"] == "SPY"][
            "date"
        ].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]

        def buy_exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
            elif ctx.bars == 2:
                orders = tuple(ctx.pending_orders())
                assert len(orders) == 1
                assert orders[0] == PendingOrder(
                    id=1,
                    type="buy",
                    symbol="SPY",
                    created=ctx.date[0],
                    exec_date=dates[buy_delay],
                    shares=100,
                    limit_price=None,
                    fill_price=PriceType.MIDDLE,
                )
            else:
                assert not tuple(ctx.pending_orders())

        config = StrategyConfig(buy_delay=buy_delay)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        order = result.orders.iloc[0]
        assert order["type"] == "buy"
        assert order["symbol"] == "SPY"
        assert order["date"] == dates[2]
        assert np.isnan(order["limit_price"])
        assert order["shares"] == 100

    def test_backtest_when_pending_orders_canceled(self, data_source_df):
        dates = data_source_df[data_source_df["symbol"] == "SPY"][
            "date"
        ].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        buy_delay = 10
        sell_delay = 5

        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.buy_limit_price = 99
            elif ctx.bars == 2:
                ctx.sell_shares = 200
                ctx.sell_limit_price = 100
            elif ctx.bars == 3:
                orders = tuple(ctx.pending_orders())
                assert len(orders) == 2
                assert orders[0] == PendingOrder(
                    id=1,
                    type="buy",
                    symbol="SPY",
                    created=ctx.date[0],
                    exec_date=dates[buy_delay],
                    shares=100,
                    limit_price=99,
                    fill_price=PriceType.MIDDLE,
                )
                assert orders[1] == PendingOrder(
                    id=2,
                    type="sell",
                    symbol="SPY",
                    created=ctx.date[1],
                    exec_date=dates[1 + sell_delay],
                    shares=200,
                    limit_price=100,
                    fill_price=PriceType.MIDDLE,
                )
                ctx.cancel_all_pending_orders()
            else:
                assert not tuple(ctx.pending_orders())

        config = StrategyConfig(buy_delay=buy_delay, sell_delay=sell_delay)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.orders)

    def test_backtest_when_buy_hold_bars(self, data_source_df):
        def buy_exec_fn(ctx):
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.sell_fill_price = PriceType.OPEN
            ctx.buy_shares = 100
            ctx.hold_bars = 2

        df = data_source_df[data_source_df["symbol"] == "SPY"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        buy_dates = dates[1:]
        sell_dates = dates[3:]
        config = StrategyConfig(initial_cash=500_000)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        buy_orders = orders[orders["type"] == "buy"]
        assert len(buy_orders) == len(buy_dates)
        for buy_date in buy_dates:
            row = buy_orders[buy_orders["date"] == buy_date]
            assert row["symbol"].item() == "SPY"
            assert row["shares"].item() == 100
            assert np.isnan(row["limit_price"].item())
            assert row["fill_price"].item() == round(
                df[df["date"] == buy_date]["close"].item(), 2
            )
            assert row["fees"].item() == 0
        sell_orders = orders[orders["type"] == "sell"]
        assert len(sell_orders) == len(sell_dates)
        for sell_date in sell_dates:
            row = sell_orders[sell_orders["date"] == sell_date]
            assert row["symbol"].item() == "SPY"
            assert row["shares"].item() == 100
            assert np.isnan(row["limit_price"].item())
            assert row["fill_price"].item() == round(
                df[df["date"] == sell_date]["open"].item(), 2
            )
            assert row["fees"].item() == 0
        assert (result.trades["stop"] == "bar").all()

    def test_backtest_when_sell_hold_bars(self, data_source_df):
        def sell_exec_fn(ctx):
            ctx.sell_fill_price = PriceType.OPEN
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.sell_shares = 100
            ctx.hold_bars = 1

        df = data_source_df[data_source_df["symbol"] == "SPY"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        buy_dates = dates[2:]
        sell_dates = dates[1:]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(sell_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        sell_orders = orders[orders["type"] == "sell"]
        assert len(sell_orders) == len(sell_dates)
        for sell_date in sell_dates:
            row = sell_orders[sell_orders["date"] == sell_date]
            assert row["symbol"].item() == "SPY"
            assert row["shares"].item() == 100
            assert np.isnan(row["limit_price"].item())
            assert row["fill_price"].item() == round(
                df[df["date"] == sell_date]["open"].item(), 2
            )
            assert row["fees"].item() == 0
        buy_orders = orders[orders["type"] == "buy"]
        assert len(buy_orders) == len(buy_dates)
        for buy_date in buy_dates:
            row = buy_orders[buy_orders["date"] == buy_date]
            assert row["symbol"].item() == "SPY"
            assert row["shares"].item() == 100
            assert np.isnan(row["limit_price"].item())
            assert row["fill_price"].item() == round(
                df[df["date"] == buy_date]["close"].item(), 2
            )
            assert row["fees"].item() == 0
        assert len(result.trades) == len(buy_orders)
        assert (result.trades["stop"] == "bar").all()

    def test_backtest_when_stop_loss(self, data_source_df):
        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.stop_loss = 10

        df = data_source_df[data_source_df["symbol"] == "SPY"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["type"] == "long"
        assert trade["symbol"] == "SPY"
        assert trade["entry_date"] == dates[1]
        assert trade["exit"] == trade["entry"] - 10
        assert trade["shares"] == 100
        assert trade["pnl"] == -1000
        assert trade["agg_pnl"] == -1000
        assert trade["pnl_per_bar"] == round(-1000 / trade["bars"], 2)
        assert trade["stop"] == "loss"
        assert len(result.orders) == 2
        buy_order = result.orders.iloc[0]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "SPY"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 100
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0
        sell_order = result.orders.iloc[1]
        assert sell_order["type"] == "sell"
        assert sell_order["symbol"] == "SPY"
        assert sell_order["shares"] == 100
        assert np.isnan(sell_order["limit_price"])
        assert sell_order["fees"] == 0

    def test_backtest_when_sell_before_stop_loss(self, data_source_df):
        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.stop_loss = 10
            elif ctx.bars == 10:
                ctx.sell_all_shares()

        df = data_source_df[data_source_df["symbol"] == "SPY"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["type"] == "long"
        assert trade["symbol"] == "SPY"
        assert trade["entry_date"] == dates[1]
        assert trade["exit_date"] == dates[10]
        assert trade["shares"] == 100
        assert trade["stop"] is None
        assert len(result.orders) == 2
        buy_order = result.orders.iloc[0]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "SPY"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 100
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0
        sell_order = result.orders.iloc[1]
        assert sell_order["type"] == "sell"
        assert sell_order["symbol"] == "SPY"
        assert sell_order["date"] == dates[10]
        assert sell_order["shares"] == 100
        assert np.isnan(sell_order["limit_price"])
        assert sell_order["fees"] == 0

    def test_backtest_when_cancel_stop(self, data_source_df):
        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.stop_loss = 10
            elif ctx.bars == 10:
                assert ctx.cancel_stop(stop_id=1)

        df = data_source_df[data_source_df["symbol"] == "SPY"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.trades)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "SPY"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 100
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0

    def test_backtest_when_cancel_stops(self, data_source_df):
        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.stop_loss = 10
                ctx.stop_trailing = 10
            elif ctx.bars == 10:
                ctx.cancel_stops("SPY")

        df = data_source_df[data_source_df["symbol"] == "SPY"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.trades)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "SPY"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 100
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0

    def test_backtest_when_pos_size_handler_zero_shares(self, data_source_df):
        def buy_exec_fn(ctx):
            ctx.buy_shares = 100

        def sell_exec_fn(ctx):
            ctx.sell_shares = 100

        def pos_size_handler(ctx):
            signals = tuple(ctx.signals())
            ctx.set_shares(signals[0], shares=0)
            ctx.set_shares(signals[1], shares=0)

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(buy_exec_fn, "SPY")
        strategy.add_execution(sell_exec_fn, "AAPL")
        strategy.set_pos_size_handler(pos_size_handler)
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.orders)

    def test_backtest_when_no_stops(self, data_source_df):
        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
            elif ctx.bars > 30:
                ctx.sell_all_shares()

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["stop"] is None
