"""Unit tests for strategy.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import os
import pandas as pd
import pytest
import re
from importlib import import_module
from .fixtures import *  # noqa: F401
from collections import defaultdict, deque
from datetime import datetime
from decimal import Decimal
from pybroker.common import DataCol, PriceType, to_datetime
from pybroker.config import StrategyConfig
from pybroker.context import ExecContext
from pybroker.data import DataSource
from pybroker.eval import EvalMetrics
from pybroker.parallel import get_parallel_config, parallel, set_parallel
from pybroker.portfolio import (
    Order,
    Portfolio,
    PortfolioBar,
    PositionBar,
    Trade,
)
from pybroker.scope import PendingOrder
from pybroker.slippage import SlippageModel
from pybroker.strategy import (
    BacktestMixin,
    Execution,
    Strategy,
    TestResult,
    WalkforwardMixin,
    _is_rankable,
    _rank_by_score,
    _rank_by_short_score,
    _resolve_execution_symbols,
    _resolve_executions,
)
from unittest.mock import Mock, patch


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


def _rotational_test_df(symbols, num_bars=5):
    dates = pd.date_range("2020-01-01", periods=num_bars, freq="D")
    rows = []
    for sym in symbols:
        for date in dates:
            rows.append(
                {
                    "symbol": sym,
                    "date": date,
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "volume": 1000,
                    "adj_close": 100.0,
                }
            )
    return pd.DataFrame(rows)


def _selector_universe_df(
    symbol_volumes: dict[str, int],
    num_bars: int = 20,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=num_bars, freq="D")
    rows = []
    for sym, vol in symbol_volumes.items():
        for i, date in enumerate(dates):
            rows.append(
                {
                    "symbol": sym,
                    "date": date,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": vol + i,
                    "adj_close": 10.0,
                }
            )
    return pd.DataFrame(rows)


class TestSymbolSelector:
    def _execution(self, symbols, fn=None):
        return Execution(
            id=1,
            symbols=symbols,
            fn=fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )

    def test_selector_empty_list_raises(self):
        def empty_selector(_df):
            return []

        with pytest.raises(ValueError, match="empty list"):
            _resolve_execution_symbols(
                self._execution(empty_selector),
                _selector_universe_df({"AAA": 100}),
            )

    def test_selector_unknown_symbols_raises(self):
        def bad_selector(_df):
            return ["UNKNOWN"]

        with pytest.raises(ValueError, match="unknown symbols.*UNKNOWN"):
            _resolve_execution_symbols(
                self._execution(bad_selector),
                _selector_universe_df({"AAA": 100}),
            )

    def test_selector_duplicates_raises(self):
        def dupe_selector(_df):
            return ["AAA", "AAA"]

        with pytest.raises(ValueError, match="duplicate symbols"):
            _resolve_execution_symbols(
                self._execution(dupe_selector),
                _selector_universe_df({"AAA": 100}),
            )

    def test_resolve_executions_overlap_raises(self):
        df = _selector_universe_df({"AAA": 100, "BBB": 200})
        exec_a = self._execution(lambda d: ["AAA"])
        exec_b = self._execution(lambda d: ["AAA", "BBB"])
        with pytest.raises(ValueError, match="AAA was already added"):
            _resolve_executions({exec_a, exec_b}, df)

    def test_selector_receives_train_slice_only(self):
        df = _selector_universe_df(
            {"AAA": 100, "BBB": 200, "CCC": 300}, num_bars=12
        )
        selection_dates: list[set] = []
        test_dates: list[set] = []

        def track_selector(selection_df):
            selection_dates.append(set(selection_df["date"]))
            return ["AAA"]

        def exec_fn(_ctx):
            pass

        strategy = Strategy(df, "2020-01-01", "2020-01-12")
        strategy.add_execution(exec_fn, track_selector)
        mixin = WalkforwardMixin()
        for train_idx, test_idx in mixin.walkforward_split(
            df=df, windows=2, lookahead=1, train_size=0.5, shuffle=False
        ):
            train_data = df.loc[train_idx]
            test_data = df.loc[test_idx]
            test_dates.append(set(test_data["date"]))
            _resolve_executions(strategy._executions, train_data)

        assert len(selection_dates) == 2
        for sel_dates, tst_dates in zip(selection_dates, test_dates):
            assert sel_dates.isdisjoint(tst_dates)

    def test_selector_called_once_per_window(self):
        df = _selector_universe_df(
            {"AAA": 100, "BBB": 200, "CCC": 300}, num_bars=12
        )
        call_count = {"n": 0}

        def counting_selector(_df):
            call_count["n"] += 1
            return ["AAA"]

        def exec_fn(_ctx):
            pass

        strategy = Strategy(df, "2020-01-01", "2020-01-12")
        strategy.add_execution(exec_fn, counting_selector)
        strategy.walkforward(windows=3, train_size=0.5, lookahead=1)
        assert call_count["n"] == 3

    def test_selector_with_dataframe_universe(self):
        df = _selector_universe_df(
            {"AAA": 100, "BBB": 500, "CCC": 300}, num_bars=8
        )
        seen_symbols: list[str] = []

        def top_volume(selection_df):
            adv = selection_df.groupby("symbol")["volume"].mean()
            return [adv.idxmax()]

        def exec_fn(ctx):
            seen_symbols.append(ctx.symbol)

        strategy = Strategy(df, "2020-01-01", "2020-01-08")
        strategy.add_execution(exec_fn, top_volume)
        strategy.walkforward(windows=1, train_size=0.5, lookahead=1)
        assert set(seen_symbols) == {"BBB"}

    def test_selector_with_datasource_raises(self):
        class FakeSource(DataSource):
            def _fetch_data(
                self, symbols, start_date, end_date, timeframe, adjust
            ):
                return pd.DataFrame()

        def exec_fn(_ctx):
            pass

        def selector(_df):
            return ["AAA"]

        strategy = Strategy(
            FakeSource(), "2020-01-01", "2020-01-08", StrategyConfig()
        )
        strategy.add_execution(exec_fn, selector)
        with pytest.raises(
            ValueError, match="Dynamic symbol selection requires"
        ):
            strategy.walkforward(windows=1)

    def test_boundary_liquidation_dropped_symbol(self):
        df = _selector_universe_df({"AAA": 100, "BBB": 200}, num_bars=12)
        window = {"n": 0}
        initial_cash = Decimal(100_000)

        def rotating_selector(_df):
            window["n"] += 1
            return ["AAA"] if window["n"] == 1 else ["BBB"]

        def buy_once(ctx):
            if ctx.long_pos() is None:
                ctx.buy_shares = 100

        strategy = Strategy(
            df,
            "2020-01-01",
            "2020-01-12",
            StrategyConfig(initial_cash=initial_cash, buy_delay=1),
        )
        strategy.add_execution(buy_once, rotating_selector)
        result = strategy.walkforward(windows=2, train_size=0.5, lookahead=1)
        aaa_trades = result.trades[result.trades["symbol"] == "AAA"]
        assert not aaa_trades.empty
        assert pd.notna(aaa_trades.iloc[-1]["exit_date"])
        aaa_sells = result.orders[
            (result.orders["symbol"] == "AAA")
            & (result.orders["type"] == "sell")
        ]
        assert not aaa_sells.empty
        assert result.portfolio.iloc[-1]["cash"] > 0

    def test_mixed_static_and_selector_executions(self):
        df = _selector_universe_df(
            {"AAA": 100, "BBB": 200, "CCC": 300}, num_bars=8
        )
        seen: set[str] = set()

        def pick_aaa(_df):
            return ["AAA"]

        def static_fn(ctx):
            seen.add(ctx.symbol)

        strategy = Strategy(df, "2020-01-01", "2020-01-08")
        strategy.add_execution(static_fn, "BBB")
        strategy.add_execution(static_fn, pick_aaa)
        strategy.walkforward(windows=1, train_size=0.5, lookahead=1)
        assert seen == {"AAA", "BBB"}

    def test_indicators_skipped_for_unselected_symbols(self, scope):
        from pybroker.indicator import indicator
        from pybroker.vect import sumv

        df = _selector_universe_df(
            {"AAA": 100, "BBB": 200, "CCC": 300}, num_bars=8
        )
        sma = indicator(
            "sma_sel", lambda bar_data, n: sumv(bar_data.close, n), n=2
        )

        def pick_aaa(_df):
            return ["AAA"]

        def exec_fn(_ctx):
            pass

        strategy = Strategy(df, "2020-01-01", "2020-01-08")
        strategy.add_execution(exec_fn, pick_aaa, indicators=[sma])
        with patch.object(
            strategy, "compute_indicators", wraps=strategy.compute_indicators
        ) as mock_compute:
            strategy.walkforward(windows=1, train_size=0.5, lookahead=1)
            indicator_syms = mock_compute.call_args.kwargs["indicator_syms"]
            symbols_computed = {pair.symbol for pair in indicator_syms}
            assert symbols_computed == {"AAA"}

    def test_indicator_cache_reuse_across_windows(self, scope, tmp_path):
        from pybroker.cache import (
            clear_indicator_cache,
            disable_indicator_cache,
            enable_indicator_cache,
        )
        from pybroker.indicator import indicator
        from pybroker.vect import sumv

        enable_indicator_cache("selector_cache", str(tmp_path))
        clear_indicator_cache()
        df = _selector_universe_df(
            {"AAA": 100, "BBB": 200, "CCC": 300}, num_bars=15
        )
        sma = indicator(
            "sma_cache", lambda bar_data, n: sumv(bar_data.close, n), n=2
        )
        window = {"n": 0}

        def alternating_selector(_df):
            window["n"] += 1
            if window["n"] in (1, 3):
                return ["AAA"]
            return ["BBB"]

        def exec_fn(_ctx):
            pass

        strategy = Strategy(df, "2020-01-01", "2020-01-15")
        strategy.add_execution(exec_fn, alternating_selector, indicators=[sma])
        aaa_uncached_counts: list[int] = []
        real_get = strategy._get_cached_indicators

        def track_cache(indicator_syms, cache_date_fields):
            data, uncached = real_get(indicator_syms, cache_date_fields)
            aaa_uncached_counts.append(
                sum(1 for p in uncached if p.symbol == "AAA")
            )
            return data, uncached

        try:
            with patch.object(
                strategy, "_get_cached_indicators", side_effect=track_cache
            ):
                strategy.walkforward(windows=3, train_size=0.5, lookahead=1)
        finally:
            disable_indicator_cache()
        assert aaa_uncached_counts == [1, 0, 0]

    def test_static_symbols_unchanged(self, data_source_df):
        def exec_fn(_ctx):
            pass

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.walkforward(windows=1, train_size=0.5)
        assert result.metrics is not None


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
        for i, (train_idx, test_idx) in enumerate(results):
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
            if len(test_idx) and i == len(results) - 1:
                assert dates[dates_length - 1] == dates[sorted(test_idx)[-1]]

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


class TestBacktestMixin:
    def test_backtest_executions(self, data_source_df):
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
            config=StrategyConfig(),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
            exit_dates={},
        )
        buy_df = data_source_df[data_source_df["symbol"] == "SPY"]
        buy_dates = buy_df["date"].unique()[1:]
        assert len(mock_portfolio.buy.call_args_list) == len(buy_dates)
        for i, date in enumerate(buy_dates):
            _, kwargs = mock_portfolio.buy.call_args_list[i]
            assert kwargs["date"] == date
            assert kwargs["symbol"] == "SPY"
            assert kwargs["shares"] == 200
            assert kwargs["fill_price"] == Decimal(
                str(
                    round(buy_df[buy_df["date"] == date]["close"].values[0], 2)
                )
            )
            assert kwargs["limit_price"] == 100
        sell_df = data_source_df[data_source_df["symbol"] == "AAPL"]
        sell_dates = sell_df["date"].unique()[1:]
        assert len(mock_portfolio.sell.call_args_list) == len(sell_dates)
        for i, date in enumerate(sell_dates):
            _, kwargs = mock_portfolio.sell.call_args_list[i]
            assert kwargs["date"] == date
            assert kwargs["symbol"] == "AAPL"
            assert kwargs["shares"] == 100
            assert kwargs["fill_price"] == Decimal(
                str(
                    round(
                        sell_df[sell_df["date"] == date]["close"].values[0], 2
                    )
                )
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
            config=StrategyConfig(buy_delay=2),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
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
                str(
                    round(buy_df[buy_df["date"] == date]["close"].values[0], 2)
                )
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
            config=StrategyConfig(sell_delay=2),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
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
                str(
                    round(
                        sell_df[sell_df["date"] == date]["close"].values[0], 2
                    )
                )
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
                config=StrategyConfig(),
                executions=execs,
                before_exec_fn=None,
                after_exec_fn=None,
                sessions=defaultdict(dict),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Mock(),
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
                config=StrategyConfig(),
                executions=execs,
                before_exec_fn=None,
                after_exec_fn=None,
                sessions=defaultdict(dict),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Mock(),
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
            config=StrategyConfig(),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
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
            config=StrategyConfig(),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df[data_source_df["symbol"] != "AAPL"],
            portfolio=portfolio,
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
            config=StrategyConfig(buy_delay=1_000),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
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
            config=StrategyConfig(sell_delay=1000),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
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
            config=StrategyConfig(max_long_positions=1),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
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
                    round(
                        df[(df["date"] == date) & (df["symbol"] == sym)][
                            "close"
                        ].values[0],
                        2,
                    )
                )
            )
            assert kwargs["limit_price"] is None

    def test_is_rankable(self):
        assert _is_rankable(1.0)
        assert not _is_rankable(None)
        assert not _is_rankable(float("nan"))
        assert not _is_rankable(np.nan)

    def test_rank_by_score(self):
        assert _rank_by_score({"B": 2.0, "A": 3.0, "C": 1.0}) == {
            "A": 1,
            "B": 2,
            "C": 3,
        }

    def test_rank_by_short_score(self):
        assert _rank_by_short_score({"B": 2.0, "A": 3.0, "C": 1.0}) == {
            "C": 1,
            "B": 2,
            "A": 3,
        }

    def test_backtest_executions_when_worst_rank_held_rotates(self):
        symbols = ["S1", "S2", "S3", "S4", "S5", "S6"]
        scores_by_bar = [
            {
                "S1": 60,
                "S2": 50,
                "S3": 40,
                "S4": 30,
                "S5": 20,
                "S6": 10,
            },
            {
                "S1": 55,
                "S2": 45,
                "S3": 35,
                "S4": 25,
                "S5": 15,
                "S6": 5,
            },
            {
                "S1": 1,
                "S2": 2,
                "S3": 3,
                "S4": 4,
                "S5": 5,
                "S6": 60,
            },
        ]

        def exec_fn(ctx):
            idx = min(ctx.bars - 1, len(scores_by_bar) - 1)
            ctx.long_score = scores_by_bar[idx][ctx.symbol]
            if ctx.buy_shares is not None:
                ctx.buy_fill_price = PriceType.CLOSE
            if ctx.sell_shares is not None:
                ctx.sell_fill_price = PriceType.CLOSE

        test_df = _rotational_test_df(symbols, num_bars=4)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(100_000, max_long_positions=2)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(
                max_long_positions=2,
                worst_rank_held=5,
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        buy_symbols = {
            order.symbol for order in portfolio.orders if order.type == "buy"
        }
        assert {"S1", "S2"}.issubset(buy_symbols)
        sell_orders = [
            order for order in portfolio.orders if order.type == "sell"
        ]
        assert len(sell_orders) == 1
        assert sell_orders[0].symbol == "S1"
        assert "S2" in portfolio.long_positions
        assert "S1" not in portfolio.long_positions

    def test_backtest_executions_when_worst_rank_held_unrankable(self):
        symbols = ["S1", "S2", "S3"]
        scores_by_bar = [
            {"S1": 30, "S2": 20, "S3": 10},
            {"S1": float("nan"), "S2": 20, "S3": 10},
        ]

        def exec_fn(ctx):
            idx = min(ctx.bars - 1, len(scores_by_bar) - 1)
            ctx.long_score = scores_by_bar[idx][ctx.symbol]
            if ctx.buy_shares is not None:
                ctx.buy_fill_price = PriceType.CLOSE
            if ctx.sell_shares is not None:
                ctx.sell_fill_price = PriceType.CLOSE

        test_df = _rotational_test_df(symbols, num_bars=3)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(100_000, max_long_positions=2)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(
                max_long_positions=2,
                worst_rank_held=5,
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        assert "S1" not in portfolio.long_positions
        assert "S2" in portfolio.long_positions
        assert any(
            order.type == "sell" and order.symbol == "S1"
            for order in portfolio.orders
        )
        buy_symbols = {
            order.symbol for order in portfolio.orders if order.type == "buy"
        }
        assert "S1" in buy_symbols
        assert "S2" in buy_symbols

    def test_backtest_executions_when_worst_rank_held_long_score_only(self):
        symbols = ["S1", "S2", "S3"]

        def exec_fn(ctx):
            ctx.long_score = {"S1": 30, "S2": 20, "S3": 10}[ctx.symbol]

        test_df = _rotational_test_df(symbols, num_bars=2)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(100_000, max_long_positions=2)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(
                max_long_positions=2,
                worst_rank_held=5,
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        buy_symbols = {
            order.symbol for order in portfolio.orders if order.type == "buy"
        }
        assert buy_symbols == {"S1", "S2"}

    def test_backtest_executions_when_worst_rank_held_short_score_rotates(
        self,
    ):
        symbols = ["S1", "S2", "S3", "S4", "S5", "S6"]
        scores_by_bar = [
            {
                "S1": 10,
                "S2": 20,
                "S3": 30,
                "S4": 40,
                "S5": 50,
                "S6": 60,
            },
            {
                "S1": 15,
                "S2": 25,
                "S3": 35,
                "S4": 45,
                "S5": 55,
                "S6": 65,
            },
            {
                "S1": 60,
                "S2": 50,
                "S3": 40,
                "S4": 30,
                "S5": 20,
                "S6": 10,
            },
        ]

        def exec_fn(ctx):
            idx = min(ctx.bars - 1, len(scores_by_bar) - 1)
            ctx.short_score = scores_by_bar[idx][ctx.symbol]
            if ctx.buy_shares is not None:
                ctx.buy_fill_price = PriceType.CLOSE
            if ctx.sell_shares is not None:
                ctx.sell_fill_price = PriceType.CLOSE

        test_df = _rotational_test_df(symbols, num_bars=4)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(100_000, max_short_positions=2)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(
                max_short_positions=2,
                worst_rank_held=5,
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        short_entries = {
            order.symbol for order in portfolio.orders if order.type == "sell"
        }
        assert {"S1", "S2"}.issubset(short_entries)
        cover_orders = [
            order
            for order in portfolio.orders
            if order.type == "buy" and order.symbol == "S1"
        ]
        assert len(cover_orders) == 1
        assert "S1" not in portfolio.short_positions
        assert "S2" in portfolio.short_positions

    def test_backtest_executions_when_worst_rank_held_short_score_only(self):
        symbols = ["S1", "S2", "S3"]

        def exec_fn(ctx):
            ctx.short_score = {"S1": 10, "S2": 20, "S3": 30}[ctx.symbol]
            if ctx.buy_shares is not None:
                ctx.buy_fill_price = PriceType.CLOSE
            if ctx.sell_shares is not None:
                ctx.sell_fill_price = PriceType.CLOSE

        test_df = _rotational_test_df(symbols, num_bars=2)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(100_000, max_short_positions=2)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(
                max_short_positions=2,
                worst_rank_held=5,
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        short_entries = {
            order.symbol for order in portfolio.orders if order.type == "sell"
        }
        assert short_entries == {"S1", "S2"}

    def test_backtest_executions_when_worst_rank_held_short_score_unrankable(
        self,
    ):
        symbols = ["S1", "S2", "S3"]
        scores_by_bar = [
            {"S1": 10, "S2": 20, "S3": 30},
            {"S1": float("nan"), "S2": 20, "S3": 30},
        ]

        def exec_fn(ctx):
            idx = min(ctx.bars - 1, len(scores_by_bar) - 1)
            ctx.short_score = scores_by_bar[idx][ctx.symbol]
            if ctx.buy_shares is not None:
                ctx.buy_fill_price = PriceType.CLOSE
            if ctx.sell_shares is not None:
                ctx.sell_fill_price = PriceType.CLOSE

        test_df = _rotational_test_df(symbols, num_bars=3)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(100_000, max_short_positions=2)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(
                max_short_positions=2,
                worst_rank_held=5,
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        assert "S1" not in portfolio.short_positions
        assert "S2" in portfolio.short_positions
        assert any(
            order.type == "buy" and order.symbol == "S1"
            for order in portfolio.orders
        )
        short_entries = {
            order.symbol for order in portfolio.orders if order.type == "sell"
        }
        assert {"S1", "S2"}.issubset(short_entries)

    def test_backtest_executions_when_worst_rank_held_long_and_short_score(
        self,
    ):
        long_symbols = ["L1", "L2", "L3", "L4", "L5", "L6"]
        short_symbols = ["S1", "S2", "S3", "S4", "S5", "S6"]
        symbols = long_symbols + short_symbols
        long_scores_by_bar = [
            {
                "L1": 60,
                "L2": 50,
                "L3": 40,
                "L4": 30,
                "L5": 20,
                "L6": 10,
            },
            {
                "L1": 55,
                "L2": 45,
                "L3": 35,
                "L4": 25,
                "L5": 15,
                "L6": 5,
            },
            {
                "L1": 1,
                "L2": 2,
                "L3": 3,
                "L4": 4,
                "L5": 5,
                "L6": 60,
            },
        ]
        short_scores_by_bar = [
            {
                "S1": 60,
                "S2": 50,
                "S3": 40,
                "S4": 30,
                "S5": 20,
                "S6": 10,
            },
            {
                "S1": 65,
                "S2": 55,
                "S3": 45,
                "S4": 35,
                "S5": 25,
                "S6": 15,
            },
            {
                "S1": 1,
                "S2": 2,
                "S3": 3,
                "S4": 4,
                "S5": 5,
                "S6": 60,
            },
        ]

        def exec_fn(ctx):
            idx = min(ctx.bars - 1, len(long_scores_by_bar) - 1)
            if ctx.symbol in long_symbols:
                ctx.long_score = long_scores_by_bar[idx][ctx.symbol]
            if ctx.symbol in short_symbols:
                ctx.short_score = short_scores_by_bar[idx][ctx.symbol]
            if ctx.buy_shares is not None:
                ctx.buy_fill_price = PriceType.CLOSE
            if ctx.sell_shares is not None:
                ctx.sell_fill_price = PriceType.CLOSE

        test_df = _rotational_test_df(symbols, num_bars=4)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(
            100_000, max_long_positions=2, max_short_positions=2
        )
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(
                max_long_positions=2,
                max_short_positions=2,
                worst_rank_held=5,
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        long_buys = {
            order.symbol for order in portfolio.orders if order.type == "buy"
        }
        short_entries = {
            order.symbol for order in portfolio.orders if order.type == "sell"
        }
        assert {"L1", "L2"}.issubset(long_buys)
        assert {"S5", "S6"}.issubset(short_entries)
        assert "L1" not in portfolio.long_positions
        assert "L2" in portfolio.long_positions
        assert "S6" not in portfolio.short_positions
        assert "S5" in portfolio.short_positions

    def test_backtest_executions_when_worst_rank_held_and_score_then_error(
        self,
    ):
        symbols = ["S1", "S2", "S3"]

        def exec_fn(ctx):
            ctx.score = {"S1": 30, "S2": 20, "S3": 10}[ctx.symbol]

        test_df = _rotational_test_df(symbols, num_bars=2)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(100_000, max_long_positions=2)
        mixin = BacktestMixin()
        with pytest.raises(
            ValueError,
            match=(
                "score cannot be used with worst_rank_held; use long_score or "
                "short_score instead."
            ),
        ):
            mixin.backtest_executions(
                config=StrategyConfig(
                    max_long_positions=2,
                    worst_rank_held=5,
                ),
                executions={exec},
                before_exec_fn=None,
                after_exec_fn=None,
                sessions=defaultdict(dict),
                models={},
                indicator_data={},
                test_data=test_df,
                portfolio=portfolio,
                exit_dates={},
            )

    def test_backtest_executions_when_worst_rank_held_user_sell(self):
        symbols = ["S1", "S2", "S3"]
        scores_by_bar = [
            {"S1": 30, "S2": 20, "S3": 10},
            {"S1": 1, "S2": 2, "S3": 60},
        ]
        user_sells = []

        def exec_fn(ctx):
            idx = min(ctx.bars - 1, len(scores_by_bar) - 1)
            ctx.long_score = scores_by_bar[idx][ctx.symbol]
            if ctx.buy_shares is not None:
                ctx.buy_fill_price = PriceType.CLOSE
            if ctx.sell_shares is not None:
                ctx.sell_fill_price = PriceType.CLOSE
            if ctx.bars == 2 and ctx.symbol == "S1":
                ctx.sell_shares = 100
                ctx.sell_fill_price = PriceType.CLOSE
                user_sells.append(ctx.symbol)

        test_df = _rotational_test_df(symbols, num_bars=3)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(100_000, max_long_positions=2)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(
                max_long_positions=2,
                worst_rank_held=5,
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        s1_sells = [
            order
            for order in portfolio.orders
            if order.type == "sell" and order.symbol == "S1"
        ]
        assert len(s1_sells) == 1
        assert s1_sells[0].shares == 100
        assert user_sells == ["S1"]

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
            config=StrategyConfig(max_short_positions=1),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
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
                    round(
                        df[(df["date"] == date) & (df["symbol"] == sym)][
                            "close"
                        ].values[0],
                        2,
                    )
                )
            )
            assert kwargs["limit_price"] is None

    def test_backtest_executions_when_buy_long_score(self, data_source_df):
        def buy_exec_fn(ctx):
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.buy_shares = 200
            if ctx.symbol == "SPY":
                ctx.long_score = 1
            else:
                ctx.long_score = 0

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
            config=StrategyConfig(max_long_positions=1),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
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
                    round(
                        df[(df["date"] == date) & (df["symbol"] == sym)][
                            "close"
                        ].values[0],
                        2,
                    )
                )
            )
            assert kwargs["limit_price"] is None

    def test_backtest_executions_when_sell_short_score(self, data_source_df):
        def sell_exec_fn(ctx):
            ctx.sell_fill_price = PriceType.CLOSE
            ctx.sell_shares = 200
            if ctx.symbol == "AAPL":
                ctx.short_score = 0
            else:
                ctx.short_score = 1

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
            config=StrategyConfig(max_short_positions=1),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=mock_portfolio,
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
                    round(
                        df[(df["date"] == date) & (df["symbol"] == sym)][
                            "close"
                        ].values[0],
                        2,
                    )
                )
            )
            assert kwargs["limit_price"] is None

    def test_backtest_executions_when_max_short_positions_and_cover(
        self, data_source_df
    ):
        def sell_exec_fn(ctx):
            if ctx.symbol == "AAPL":
                if ctx.bars == 1:
                    ctx.sell_shares = 200
                elif ctx.bars == 2:
                    ctx.cover_all_shares()
            else:
                if ctx.bars == 2:
                    ctx.sell_shares = 100

        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL", "SPY"]),
            fn=sell_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        portfolio = Portfolio(100_000, max_short_positions=1)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(max_short_positions=1),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
            exit_dates={},
        )
        assert len(portfolio.short_positions) == 1
        assert not portfolio.long_positions
        assert len(portfolio.orders) == 3
        orders = portfolio.orders
        assert orders[0].symbol == "AAPL"
        assert orders[0].shares == 200
        assert orders[0].type == "sell"
        assert orders[1].symbol == "AAPL"
        assert orders[1].shares == 200
        assert orders[1].type == "buy"
        assert orders[2].symbol == "SPY"
        assert orders[2].shares == 100
        assert orders[2].type == "sell"
        trades = portfolio.trades
        assert len(trades) == 1
        assert trades[0].symbol == "AAPL"
        assert trades[0].type == "short"

    def test_backtest_executions_when_max_long_positions_and_cover(
        self, data_source_df
    ):
        def cover_exec_fn(ctx):
            if ctx.symbol == "AAPL":
                ctx.score = 2
            else:
                ctx.score = 1
            ctx.cover_shares = 100
            ctx.hold_bars = 1

        exec = Execution(
            id=1,
            symbols=frozenset(["AAPL", "SPY"]),
            fn=cover_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        execs = {exec}
        portfolio = Portfolio(100_000, max_long_positions=1)
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(max_long_positions=1),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
            exit_dates={},
        )
        dates = data_source_df["date"].unique()[1:]
        orders = portfolio.orders
        assert (
            len(list(filter(lambda o: o.symbol == "AAPL", orders)))
            == len(dates) * 2 - 1
        )
        trades = portfolio.trades
        assert (
            len(list(filter(lambda t: t.symbol == "AAPL", trades)))
            == len(dates) - 1
        )

    @pytest.mark.parametrize(
        "price_type, expected_fill_price",
        [
            (50, 50),
            (Decimal("111.1"), Decimal("111.1")),
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
            config=StrategyConfig(),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=df,
            portfolio=mock_portfolio,
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
                config=StrategyConfig(),
                executions=execs,
                before_exec_fn=None,
                after_exec_fn=None,
                sessions=defaultdict(dict),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Portfolio(100_000),
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
                config=StrategyConfig(),
                executions=execs,
                before_exec_fn=None,
                after_exec_fn=None,
                sessions=defaultdict(dict),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Portfolio(100_000),
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
                config=StrategyConfig(),
                executions=execs,
                before_exec_fn=None,
                after_exec_fn=None,
                sessions=defaultdict(dict),
                models={},
                indicator_data={},
                test_data=data_source_df,
                portfolio=Portfolio(100_000),
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
            config=StrategyConfig(),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
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
            config=StrategyConfig(),
            executions=execs,
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=data_source_df,
            portfolio=portfolio,
            exit_dates={},
        )
        assert not len(portfolio.orders)


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
def exec_model_source(scope, data_source_df, indicators):
    return model(
        MODEL_NAME,
        lambda sym, *_: FakeModel(
            sym,
            np.full(
                data_source_df[data_source_df["symbol"] == sym].shape[0], 100
            ),
        ),
        indicators,
        pretrained=False,
    )


@pytest.fixture()
def executions_with_models(executions_only, exec_model_source):
    def exec_fn(ctx):
        assert isinstance(ctx.model(exec_model_source.name), FakeModel)

    executions_only[0]["models"] = exec_model_source
    executions_only[0]["fn"] = exec_fn
    return executions_only


@pytest.fixture()
def executions_with_models_and_indicators(
    executions_only, exec_model_source, hhv_ind, llv_ind
):
    def exec_fn_1(ctx):
        assert len(ctx.indicator(llv_ind.name))

    executions_only[0]["indicators"] = llv_ind
    executions_only[0]["fn"] = exec_fn_1

    def exec_fn_2(ctx):
        assert len(ctx.indicator(hhv_ind.name))
        assert isinstance(ctx.model(exec_model_source.name), FakeModel)

    executions_only[1]["indicators"] = hhv_ind
    executions_only[1]["models"] = exec_model_source
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
def disable_parallel_indicators(request):
    return request.param


@pytest.fixture(params=[None, "weds", ("mon", "fri")])
def days(request):
    return request.param


@pytest.fixture(params=[None, ("10:00", "1:00")])
def between_time(request):
    return request.param


class FakeDataSource(DataSource):
    def _fetch_data(
        self, symbols, start_date, end_date, timeframe, adjustment
    ):
        return pd.read_pickle(
            os.path.join(os.path.dirname(__file__), "testdata/daily_1.pkl")
        )


START_DATE = "2020-01-02"
END_DATE = "2021-12-31"


def _limit_order_df(highs, lows):
    dates = pd.date_range("2020-01-02", periods=len(highs), freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["SPY"] * len(highs),
            "open": highs,
            "high": highs,
            "low": lows,
            "close": highs,
            "volume": np.repeat(1_000_000, len(highs)),
        }
    )


def _picklable_train_fake_model(sym, train_data, test_data):
    return FakeModel(
        sym,
        np.full(train_data.shape[0] + test_data.shape[0], 100),
    )


POOLED_MODEL_NAME = "pooled_fake_model"
_pooled_train_calls: list[tuple[int, frozenset[str]]] = []


class PooledFakeModel:
    def __init__(self, symbols):
        self.symbols = frozenset(symbols)


def _train_pooled_fake_model(train_data, test_data):
    _pooled_train_calls.append(
        (
            train_data.shape[0],
            frozenset(train_data[DataCol.SYMBOL.value].unique()),
        )
    )
    return PooledFakeModel(train_data[DataCol.SYMBOL.value].unique())


def _picklable_train_pooled_fake_model(train_data, test_data):
    return PooledFakeModel(train_data[DataCol.SYMBOL.value].unique())


def _picklable_non_pooled_predict_fn(_model, df):
    return np.full(len(df), 50.0)


def _picklable_train_non_pooled_fake_model(sym, train_data, test_data):
    return FakeModel(
        sym,
        np.full(train_data.shape[0] + test_data.shape[0], 50),
    )


def _pooled_predict_fn(_model, df):
    if df.empty:
        return np.array([])
    return np.full(len(df), float(df["close"].iloc[0]))


@pytest.fixture()
def exec_pooled_model_source(scope, indicators):
    _pooled_train_calls.clear()
    return model(
        POOLED_MODEL_NAME,
        _train_pooled_fake_model,
        indicators,
        predict_fn=_pooled_predict_fn,
        pooled=True,
    )


@pytest.fixture()
def exec_picklable_pooled_model_source(scope, indicators):
    return model(
        POOLED_MODEL_NAME,
        _picklable_train_pooled_fake_model,
        indicators,
        predict_fn=_pooled_predict_fn,
        pooled=True,
    )


@pytest.fixture()
def executions_with_pooled_models(
    executions_train_only, exec_pooled_model_source
):
    def exec_fn(ctx):
        preds = ctx.preds(exec_pooled_model_source.name)
        assert len(preds) > 0
        assert isinstance(
            ctx.model(exec_pooled_model_source.name), PooledFakeModel
        )

    executions_train_only[0]["models"] = exec_pooled_model_source
    executions_train_only[0]["fn"] = exec_fn
    return executions_train_only


@pytest.fixture()
def executions_with_picklable_pooled_models(
    executions_train_only, exec_picklable_pooled_model_source
):
    def exec_fn(ctx):
        preds = ctx.preds(exec_picklable_pooled_model_source.name)
        assert len(preds) > 0
        assert isinstance(
            ctx.model(exec_picklable_pooled_model_source.name), PooledFakeModel
        )

    executions_train_only[0]["models"] = exec_picklable_pooled_model_source
    executions_train_only[0]["fn"] = exec_fn
    return executions_train_only


@pytest.fixture()
def executions_with_two_picklable_pooled_groups(
    exec_picklable_pooled_model_source,
):
    def exec_fn(ctx):
        assert isinstance(
            ctx.model(exec_picklable_pooled_model_source.name), PooledFakeModel
        )

    return [
        {
            "fn": exec_fn,
            "symbols": ["AAPL", "MSFT"],
            "models": exec_picklable_pooled_model_source,
            "indicators": None,
        },
        {
            "fn": exec_fn,
            "symbols": ["SPY", "TSLA"],
            "models": exec_picklable_pooled_model_source,
            "indicators": None,
        },
    ]


@pytest.fixture()
def executions_with_picklable_pooled_and_non_pooled_models(
    exec_picklable_pooled_model_source, indicators
):
    non_pooled = model(
        MODEL_NAME,
        _picklable_train_non_pooled_fake_model,
        indicators,
        predict_fn=_picklable_non_pooled_predict_fn,
    )

    def pooled_exec_fn(ctx):
        assert isinstance(
            ctx.model(exec_picklable_pooled_model_source.name), PooledFakeModel
        )

    def non_pooled_exec_fn(ctx):
        assert isinstance(ctx.model(MODEL_NAME), FakeModel)

    return [
        {
            "fn": pooled_exec_fn,
            "symbols": ["AAPL", "MSFT"],
            "models": exec_picklable_pooled_model_source,
            "indicators": None,
        },
        {
            "fn": non_pooled_exec_fn,
            "symbols": "SPY",
            "models": non_pooled,
            "indicators": None,
        },
    ]


@pytest.fixture()
def executions_with_two_pooled_groups(
    executions_train_only, exec_pooled_model_source
):
    def exec_fn(ctx):
        assert isinstance(
            ctx.model(exec_pooled_model_source.name), PooledFakeModel
        )

    return [
        {
            "fn": exec_fn,
            "symbols": ["AAPL", "MSFT"],
            "models": exec_pooled_model_source,
            "indicators": None,
        },
        {
            "fn": exec_fn,
            "symbols": ["SPY", "TSLA"],
            "models": exec_pooled_model_source,
            "indicators": None,
        },
    ]


@pytest.fixture()
def executions_with_pooled_and_non_pooled_models(
    exec_pooled_model_source, indicators
):
    non_pooled = model(
        MODEL_NAME,
        lambda sym, train_data, test_data: FakeModel(
            sym,
            np.full(len(train_data) + len(test_data), 50),
        ),
        indicators,
        predict_fn=lambda _model, df: np.full(len(df), 50.0),
    )

    def pooled_exec_fn(ctx):
        assert isinstance(
            ctx.model(exec_pooled_model_source.name), PooledFakeModel
        )

    def non_pooled_exec_fn(ctx):
        assert isinstance(ctx.model(MODEL_NAME), FakeModel)

    return [
        {
            "fn": pooled_exec_fn,
            "symbols": ["AAPL", "MSFT"],
            "models": exec_pooled_model_source,
            "indicators": None,
        },
        {
            "fn": non_pooled_exec_fn,
            "symbols": "SPY",
            "models": non_pooled,
            "indicators": None,
        },
    ]


@pytest.fixture()
def ray_backend():
    ray = pytest.importorskip("ray")
    from ray.util.joblib import register_ray

    ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False)
    register_ray()
    yield
    ray.shutdown()


@pytest.fixture()
def exec_picklable_model_source(scope, indicators):
    return model(
        MODEL_NAME,
        _picklable_train_fake_model,
        indicators,
        pretrained=False,
    )


@pytest.fixture()
def executions_with_picklable_models(
    executions_only, exec_picklable_model_source
):
    def exec_fn(ctx):
        assert isinstance(
            ctx.model(exec_picklable_model_source.name), FakeModel
        )

    executions_only[0]["models"] = exec_picklable_model_source
    executions_only[0]["fn"] = exec_fn
    return executions_only


class TestStrategy:
    @pytest.mark.parametrize(
        "data_source",
        [FakeDataSource(), LazyFixture("data_source_df")],
    )
    @pytest.mark.parametrize(
        "executions",
        [
            LazyFixture("executions_train_only"),
            LazyFixture("executions_only"),
            LazyFixture("executions_with_indicators"),
            LazyFixture("executions_with_models"),
            LazyFixture("executions_with_models_and_indicators"),
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
        disable_parallel_indicators,
        request,
    ):
        data_source = get_fixture(request, data_source)
        executions = get_fixture(request, executions)
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
            disable_parallel_indicators=disable_parallel_indicators,
            adjust="adjustment",
        )
        if date_range[0] is None:
            expected_start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
        else:
            expected_start_date = pd.to_datetime(date_range[0])
        if date_range[1] is None:
            expected_end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
        else:
            expected_end_date = pd.to_datetime(date_range[1])
        if all(map(lambda e: not e["fn"], executions)):
            assert result.start_date == expected_start_date
            assert result.end_date == expected_end_date
            assert result.portfolio.empty
            assert result.positions.empty
            assert result.orders.empty
            assert result.trades.empty
            assert result.metrics == EvalMetrics()
            assert result.bootstrap is None
            assert result.signals is None
            return
        assert isinstance(result, TestResult)
        assert result.metrics is not None
        assert isinstance(result.metrics_df, pd.DataFrame)
        assert not result.metrics_df.empty
        assert result.start_date == expected_start_date
        assert result.end_date == expected_end_date
        assert isinstance(result.portfolio, pd.DataFrame)
        assert not result.portfolio.empty
        assert isinstance(result.positions, pd.DataFrame)
        assert isinstance(result.orders, pd.DataFrame)
        if calc_bootstrap:
            assert not result.bootstrap.conf_intervals.empty
            assert not result.bootstrap.drawdown_conf.empty
        else:
            assert result.bootstrap is None

    def test_walkforward_enable_parallel_models(
        self, data_source_df, executions_with_models
    ):
        _parallel_mod = import_module("pybroker.parallel")
        saved = get_parallel_config()
        try:
            set_parallel(n_jobs=2, backend="threading")
            config = StrategyConfig()
            strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
            for exec in executions_with_models:
                strategy.add_execution(**exec)
            serial_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=False,
                seed=42,
            )
            parallel_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=True,
                seed=42,
            )
            pd.testing.assert_frame_equal(
                serial_result.portfolio, parallel_result.portfolio
            )
            assert serial_result.metrics == parallel_result.metrics
            with patch("pybroker.model.parallel", wraps=parallel) as (
                mock_parallel
            ):
                strategy.walkforward(
                    windows=1,
                    lookahead=1,
                    timeframe="1d",
                    train_size=0.5,
                    enable_parallel_models=True,
                    seed=42,
                )
                mock_parallel.assert_called()
        finally:
            _parallel_mod._config = saved

    def test_walkforward_pooled_model_training(
        self, data_source_df, executions_with_pooled_models
    ):
        _pooled_train_calls.clear()
        config = StrategyConfig()
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        for exec in executions_with_pooled_models:
            strategy.add_execution(**exec)
        strategy.walkforward(
            windows=1,
            lookahead=1,
            timeframe="1d",
            train_size=0.5,
            seed=42,
        )
        assert len(_pooled_train_calls) == 1
        _, symbols = _pooled_train_calls[0]
        assert symbols == frozenset({"AAPL", "MSFT"})

    def test_walkforward_pooled_model_predict(
        self, data_source_df, executions_with_pooled_models
    ):
        config = StrategyConfig()
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        for exec in executions_with_pooled_models:
            strategy.add_execution(**exec)
        result = strategy.walkforward(
            windows=1,
            lookahead=1,
            timeframe="1d",
            train_size=0.5,
            seed=42,
        )
        assert not result.portfolio.empty

    def test_walkforward_pooled_enable_parallel_models(
        self, data_source_df, executions_with_picklable_pooled_models
    ):
        _parallel_mod = import_module("pybroker.parallel")
        saved = get_parallel_config()
        try:
            set_parallel(n_jobs=2)
            config = StrategyConfig()
            strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
            for exec in executions_with_picklable_pooled_models:
                strategy.add_execution(**exec)
            serial_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=False,
                seed=42,
            )
            parallel_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=True,
                seed=42,
            )
            pd.testing.assert_frame_equal(
                serial_result.portfolio, parallel_result.portfolio
            )
            assert serial_result.metrics == parallel_result.metrics
            assert get_parallel_config().backend == "loky"
        finally:
            _parallel_mod._config = saved

    def test_walkforward_multiple_pooled_executions_parallel(
        self, data_source_df, executions_with_two_picklable_pooled_groups
    ):
        _parallel_mod = import_module("pybroker.parallel")
        saved = get_parallel_config()
        try:
            set_parallel(n_jobs=2)
            config = StrategyConfig()
            strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
            for exec in executions_with_two_picklable_pooled_groups:
                strategy.add_execution(**exec)
            serial_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=False,
                seed=42,
            )
            parallel_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=True,
                seed=42,
            )
            pd.testing.assert_frame_equal(
                serial_result.portfolio, parallel_result.portfolio
            )
            assert serial_result.metrics == parallel_result.metrics
            with patch("pybroker.model.parallel", wraps=parallel) as (
                mock_parallel
            ):
                strategy.walkforward(
                    windows=1,
                    lookahead=1,
                    timeframe="1d",
                    train_size=0.5,
                    enable_parallel_models=True,
                    seed=42,
                )
                mock_parallel.assert_called()
            assert get_parallel_config().backend == "loky"
        finally:
            _parallel_mod._config = saved

    def test_walkforward_pooled_mixed_enable_parallel_models(
        self,
        data_source_df,
        executions_with_picklable_pooled_and_non_pooled_models,
    ):
        _parallel_mod = import_module("pybroker.parallel")
        saved = get_parallel_config()
        try:
            set_parallel(n_jobs=2)
            config = StrategyConfig()
            strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
            for exec in executions_with_picklable_pooled_and_non_pooled_models:
                strategy.add_execution(**exec)
            serial_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=False,
                seed=42,
            )
            parallel_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=True,
                seed=42,
            )
            pd.testing.assert_frame_equal(
                serial_result.portfolio, parallel_result.portfolio
            )
            assert serial_result.metrics == parallel_result.metrics
            with patch("pybroker.model.parallel", wraps=parallel) as (
                mock_parallel
            ):
                strategy.walkforward(
                    windows=1,
                    lookahead=1,
                    timeframe="1d",
                    train_size=0.5,
                    enable_parallel_models=True,
                    seed=42,
                )
                mock_parallel.assert_called()
            assert get_parallel_config().backend == "loky"
        finally:
            _parallel_mod._config = saved

    @pytest.mark.xdist_group(name="loky")
    def test_walkforward_enable_parallel_models_loky(
        self, data_source_df, executions_with_picklable_models
    ):
        _parallel_mod = import_module("pybroker.parallel")
        saved = get_parallel_config()
        try:
            set_parallel(n_jobs=2, backend="loky")
            config = StrategyConfig()
            strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
            for exec in executions_with_picklable_models:
                strategy.add_execution(**exec)
            serial_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=False,
                seed=42,
            )
            parallel_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=True,
                seed=42,
            )
            pd.testing.assert_frame_equal(
                serial_result.portfolio, parallel_result.portfolio
            )
            assert serial_result.metrics == parallel_result.metrics
            assert get_parallel_config().backend == "loky"
        finally:
            _parallel_mod._config = saved

    @pytest.mark.xdist_group(name="ray")
    def test_walkforward_enable_parallel_models_ray(
        self, data_source_df, executions_with_picklable_models, ray_backend
    ):
        _parallel_mod = import_module("pybroker.parallel")
        saved = get_parallel_config()
        try:
            set_parallel(n_jobs=2, backend="ray")
            config = StrategyConfig()
            strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
            for exec in executions_with_picklable_models:
                strategy.add_execution(**exec)
            serial_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=False,
                seed=42,
            )
            parallel_result = strategy.walkforward(
                windows=1,
                lookahead=1,
                timeframe="1d",
                train_size=0.5,
                enable_parallel_models=True,
                seed=42,
            )
            pd.testing.assert_frame_equal(
                serial_result.portfolio, parallel_result.portfolio
            )
            assert serial_result.metrics == parallel_result.metrics
        finally:
            _parallel_mod._config = saved

    @pytest.mark.parametrize("return_signals", [True, False])
    @pytest.mark.parametrize("return_stops", [True, False])
    def test_walkforward_results(
        self, data_source_df, return_signals, return_stops
    ):
        def exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_trailing = 100
                ctx.stop_profit_pct = 100

        data_source_df = data_source_df[
            data_source_df["date"] <= to_datetime(END_DATE)
        ]
        config = StrategyConfig(
            return_signals=return_signals, return_stops=return_stops
        )
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
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
        if return_signals:
            assert len(result.signals) == 2
            assert not result.signals["AAPL"].empty
            assert not result.signals["SPY"].empty
        else:
            assert result.signals is None
        if return_stops:
            assert not result.stops.empty
            assert set(result.stops.columns) == {
                "date",
                "symbol",
                "stop_id",
                "stop_type",
                "pos_type",
                "curr_value",
                "curr_bars",
                "percent",
                "points",
                "bars",
                "fill_price",
                "limit_price",
                "exit_price",
            }
        else:
            assert result.stops is None

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
        result = strategy.backtest(calc_bootstrap=True)
        assert isinstance(result, TestResult)
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

    @pytest.mark.parametrize(
        "config_kwargs, expected_msg",
        [
            (
                {"worst_rank_held": 5},
                "worst_rank_held requires max_long_positions or "
                "max_short_positions to be set.",
            ),
            (
                {"max_long_positions": 2, "worst_rank_held": 1},
                "worst_rank_held must be greater than or equal to "
                "max_long_positions.",
            ),
            (
                {"max_short_positions": 2, "worst_rank_held": 1},
                "worst_rank_held must be greater than or equal to "
                "max_short_positions.",
            ),
        ],
    )
    def test_when_invalid_worst_rank_held_config_then_error(
        self, data_source_df, config_kwargs, expected_msg
    ):
        config = StrategyConfig(**config_kwargs)
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            Strategy(data_source_df, START_DATE, END_DATE, config)

    @pytest.mark.parametrize(
        "leverage, expected_msg",
        [
            (0, "leverage must be greater than or equal to 1."),
            (0.5, "leverage must be greater than or equal to 1."),
            (-1, "leverage must be greater than or equal to 1."),
        ],
    )
    def test_when_invalid_leverage_config_then_error(
        self, data_source_df, leverage, expected_msg
    ):
        config = StrategyConfig(leverage=leverage)
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            Strategy(data_source_df, START_DATE, END_DATE, config)

    def test_when_invalid_interest_rate_config_then_error(
        self, data_source_df
    ):
        config = StrategyConfig(interest_rate=-1)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "interest_rate must be greater than or equal to 0."
            ),
        ):
            Strategy(data_source_df, START_DATE, END_DATE, config)

    def test_backtest_when_leverage(self, data_source_df):
        def buy_exec_fn(ctx):
            if ctx.long_pos() is None:
                ctx.buy_shares = ctx.calc_target_shares(2.0)

        config = StrategyConfig(initial_cash=100_000, leverage=2.0)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        leveraged_shares = result.orders[result.orders["type"] == "buy"].iloc[
            0
        ]["shares"]

        config = StrategyConfig(initial_cash=100_000, leverage=1.0)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        cash_shares = result.orders[result.orders["type"] == "buy"].iloc[0][
            "shares"
        ]
        assert leveraged_shares > cash_shares

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
        [(True, np.float64, 0.1, 3.14), (False, np.int_, 0, 3)],
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
                    margin_loan=Decimal(),
                    net_cash_balance=Decimal(100_000),
                    market_value=Decimal(100_000),
                    pnl=Decimal(1000),
                    unrealized_pnl=Decimal(),
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
                    created=None,
                    order_type="market",
                    intent="buy_to_open",
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
                    mae=Decimal(-10),
                    mfe=Decimal(10),
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
            START_DATE,
            END_DATE,
            portfolio,
            calc_bootstrap=False,
            train_only=False,
            signals=None,
            seed=42,
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

    def test_to_result_when_round_test_result_is_false(self, data_source_df):
        portfolio = Portfolio(100_000)
        portfolio.bars = deque(
            (
                PortfolioBar(
                    date=np.datetime64(START_DATE),
                    cash=Decimal(100_000),
                    equity=Decimal(100_000),
                    margin=Decimal(),
                    margin_loan=Decimal(),
                    net_cash_balance=Decimal(100_000),
                    market_value=Decimal(100_000),
                    pnl=Decimal("1000.111"),
                    unrealized_pnl=Decimal(),
                    fees=Decimal(),
                ),
            )
        )
        portfolio.position_bars = deque(
            (
                PositionBar(
                    symbol="SPY",
                    date=np.datetime64(START_DATE),
                    long_shares=Decimal("3.144"),
                    short_shares=Decimal("0.111"),
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
                    created=None,
                    order_type="market",
                    intent="buy_to_open",
                    shares=Decimal("3.144"),
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
                    shares=Decimal("3.144"),
                    pnl=Decimal(1000),
                    return_pct=Decimal("10.33"),
                    agg_pnl=Decimal(1000),
                    bars=2,
                    pnl_per_bar=Decimal(500),
                    stop=None,
                    mae=Decimal(-10),
                    mfe=Decimal(10),
                ),
            )
        )
        config = StrategyConfig(
            enable_fractional_shares=True, round_test_result=False
        )
        strategy = Strategy(
            data_source_df,
            START_DATE,
            END_DATE,
            config,
        )
        result = strategy._to_test_result(
            START_DATE,
            END_DATE,
            portfolio,
            calc_bootstrap=False,
            train_only=False,
            signals=None,
            seed=42,
        )
        assert result.positions["long_shares"].values[0] == 3.144
        assert result.positions["short_shares"].values[0] == 0.111
        assert result.portfolio["pnl"].values[0] == 1000.111
        assert result.orders["shares"].values[0] == 3.144
        assert result.trades["shares"].values[0] == 3.144

    def test_to_test_result_when_empty(self, data_source_df):
        portfolio = Portfolio(100_000)
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        result = strategy._to_test_result(
            START_DATE,
            END_DATE,
            portfolio,
            calc_bootstrap=False,
            train_only=False,
            signals=None,
            seed=42,
        )
        assert result.portfolio.empty
        assert result.positions.empty
        assert result.orders.empty
        assert result.trades.empty
        assert result.signals is None

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

    def test_backtest_when_exit_on_last_bar_with_multi_symbol_executions(
        self, data_source_df
    ):
        def long_exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.buy_fill_price = 150

        def short_exec_fn(ctx):
            if not ctx.short_pos():
                ctx.sell_shares = 100
                ctx.sell_fill_price = 200

        def sell_fill_price(_symbol, _bar_data):
            return 199.99

        def buy_fill_price(_symbol, _bar_data):
            return 99.99

        config = StrategyConfig(
            exit_on_last_bar=True,
            exit_sell_fill_price=sell_fill_price,
            exit_cover_fill_price=buy_fill_price,
        )
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(long_exec_fn, ["SPY", "AAPL"])
        strategy.add_execution(short_exec_fn, ["MSFT", "TSLA"])
        result = strategy.backtest(calc_bootstrap=False)
        traded_symbols = set(result.trades["symbol"])
        assert traded_symbols == {"SPY", "AAPL", "MSFT", "TSLA"}
        for _, trade in result.trades.iterrows():
            sym_dates = data_source_df[
                data_source_df["symbol"] == trade["symbol"]
            ]["date"].unique()
            sym_dates = sym_dates[sym_dates <= np.datetime64(END_DATE)]
            assert trade["exit_date"] == sym_dates[-1]
            if trade["type"] == "long":
                assert trade["exit"] == 199.99
            else:
                assert trade["exit"] == 99.99

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
                    exec_bar=1 + buy_delay,
                    timeout_bars=None,
                    stops=None,
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
                    exec_bar=1 + buy_delay,
                    timeout_bars=None,
                    stops=None,
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
                    exec_bar=2 + sell_delay,
                    timeout_bars=None,
                    stops=None,
                )
                ctx.cancel_all_pending_orders()
            else:
                assert not tuple(ctx.pending_orders())

        config = StrategyConfig(buy_delay=buy_delay, sell_delay=sell_delay)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.orders)

    def test_limit_order_single_attempt_default(self):
        df = _limit_order_df(
            highs=[110] * 5,
            lows=[90] * 5,
        )

        def buy_exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.buy_limit_price = 50
            elif ctx.bars == 2:
                assert len(tuple(ctx.pending_orders())) == 1
            elif ctx.bars == 3:
                assert not tuple(ctx.pending_orders())

        config = StrategyConfig(buy_delay=2)
        strategy = Strategy(df, "2020-01-02", "2020-01-15", config)
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert not len(result.orders)

    def test_limit_order_persists_until_fill(self):
        df = _limit_order_df(
            highs=[110, 110, 110, 110, 50],
            lows=[90, 90, 90, 90, 30],
        )
        pending_after_exec = False
        fill_date = None

        def buy_exec_fn(ctx):
            nonlocal pending_after_exec, fill_date
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.buy_limit_price = 50
                ctx.buy_timeout_bars = -1
            elif ctx.bars == 2:
                pending_after_exec = bool(tuple(ctx.pending_orders()))
            elif ctx.bars == 5:
                fill_date = ctx.date[-1]

        strategy = Strategy(df, "2020-01-02", "2020-01-10")
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert pending_after_exec
        assert len(result.orders) == 1
        order = result.orders.iloc[0]
        assert order["type"] == "buy"
        assert order["date"] == fill_date
        assert order["limit_price"] == 50
        assert order["shares"] == 100

    def test_limit_order_timeout_cancel(self):
        df = _limit_order_df(
            highs=[110] * 6,
            lows=[90] * 6,
        )
        pending_after_retries = False

        def buy_exec_fn(ctx):
            nonlocal pending_after_retries
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.buy_limit_price = 50
                ctx.buy_timeout_bars = 2
            elif ctx.bars == 4:
                pending_after_retries = bool(tuple(ctx.pending_orders()))
            elif ctx.bars == 5:
                assert not tuple(ctx.pending_orders())

        strategy = Strategy(df, "2020-01-02", "2020-01-15")
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert pending_after_retries
        assert not len(result.orders)

    def test_limit_order_manual_cancel_after_failed_attempt(self):
        df = _limit_order_df(
            highs=[110] * 5,
            lows=[90] * 5,
        )
        canceled = False

        def buy_exec_fn(ctx):
            nonlocal canceled
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.buy_limit_price = 50
                ctx.buy_timeout_bars = -1
            elif ctx.bars == 3:
                orders = tuple(ctx.pending_orders())
                assert len(orders) == 1
                canceled = ctx.cancel_pending_order(orders[0].id)
            elif ctx.bars == 4:
                assert not tuple(ctx.pending_orders())

        strategy = Strategy(df, "2020-01-02", "2020-01-15")
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert canceled
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

    def test_backtest_when_slippage(self, data_source_df):
        class FakeSlippageModel(SlippageModel):
            def apply_slippage(
                self, ctx: ExecContext, buy_shares, sell_shares
            ):
                ctx.buy_shares = 99

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
        strategy.set_slippage_model(FakeSlippageModel())
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        buy_orders = orders[orders["type"] == "buy"]
        assert len(buy_orders) == len(buy_dates)
        for buy_date in buy_dates:
            row = buy_orders[buy_orders["date"] == buy_date]
            assert row["symbol"].item() == "SPY"
            assert row["shares"].item() == 99
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
            assert row["shares"].item() == 99
            assert np.isnan(row["limit_price"].item())
            assert row["fill_price"].item() == round(
                df[df["date"] == sell_date]["open"].item(), 2
            )
            assert row["fees"].item() == 0
        assert (result.trades["stop"] == "bar").all()

    def test_backtest_when_slippage_and_sell_all_shares(self, data_source_df):
        class FakeSlippageModel(SlippageModel):
            def apply_slippage(
                self, ctx: ExecContext, buy_shares, sell_shares
            ):
                if sell_shares:
                    ctx.sell_shares = 90

        def buy_exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
            elif ctx.bars == 2:
                ctx.sell_all_shares()

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_slippage_model(FakeSlippageModel())
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        sell_orders = orders[orders["type"] == "sell"]
        assert len(sell_orders) == 1
        assert sell_orders.iloc[0]["shares"] == 100

    def test_backtest_when_slippage_and_cover_all_shares(self, data_source_df):
        class FakeSlippageModel(SlippageModel):
            def apply_slippage(
                self, ctx: ExecContext, buy_shares, sell_shares
            ):
                if buy_shares:
                    ctx.buy_shares = 90

        def buy_exec_fn(ctx):
            if not ctx.short_pos():
                ctx.sell_shares = 100
            elif ctx.bars == 2:
                ctx.cover_all_shares()

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_slippage_model(FakeSlippageModel())
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        buy_orders = orders[orders["type"] == "buy"]
        assert len(buy_orders) == 1
        assert buy_orders.iloc[0]["shares"] == 100

    def test_backtest_when_stop_loss(self, data_source_df):
        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.stop_loss = 10

        df = data_source_df[data_source_df["symbol"].isin(["SPY", "AAPL"])]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ["SPY", "AAPL"])
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades) == 2
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
        trade = result.trades.iloc[1]
        assert trade["type"] == "long"
        assert trade["symbol"] == "AAPL"
        assert trade["entry_date"] == dates[1]
        assert trade["exit"] == trade["entry"] - 10
        assert trade["shares"] == 100
        assert trade["pnl"] == -1000
        assert trade["agg_pnl"] == -2000
        assert trade["pnl_per_bar"] == round(-1000 / trade["bars"], 2)
        assert trade["stop"] == "loss"
        assert len(result.orders) == 4
        buy_order = result.orders.iloc[0]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "AAPL"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 100
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0
        buy_order = result.orders.iloc[1]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "SPY"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 100
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0
        sell_order = result.orders.iloc[2]
        assert sell_order["type"] == "sell"
        assert sell_order["symbol"] == "SPY"
        assert sell_order["shares"] == 100
        assert np.isnan(sell_order["limit_price"])
        assert sell_order["fees"] == 0
        sell_order = result.orders.iloc[3]
        assert sell_order["type"] == "sell"
        assert sell_order["symbol"] == "AAPL"
        assert sell_order["shares"] == 100
        assert np.isnan(sell_order["limit_price"])
        assert sell_order["fees"] == 0

    def test_backtest_when_custom_stop(self, data_source_df):
        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100

                def stop_fn(stop_ctx):
                    if stop_ctx.close[-1] < float(stop_ctx.entry.price) - 10:
                        return PriceType.CLOSE
                    return None

                ctx.stop_fn = stop_fn

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
        assert trade["shares"] == 100
        assert trade["stop"] == "custom"
        assert trade["pnl"] < 0
        assert len(result.orders) == 2
        sell_order = result.orders.iloc[1]
        assert sell_order["type"] == "sell"
        assert sell_order["symbol"] == "SPY"
        assert sell_order["order_type"] == "stop_custom"

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
                entry = tuple(ctx.long_pos().entries)[0]
                stop = next(iter(entry.stops))
                assert ctx.cancel_stop(stop_id=stop.id)

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

    def test_backtest_when_no_stops(self, data_source_df):
        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
            elif ctx.long_pos() and ctx.bars > 30:
                ctx.sell_all_shares()

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["stop"] is None

    def test_backtest_when_before_exec(self, data_source_df):
        def before_exec_fn(ctxs):
            assert len(ctxs) == 2
            assert isinstance(ctxs["SPY"], ExecContext)
            assert isinstance(ctxs["AAPL"], ExecContext)
            ctxs["SPY"].session["foo"] = "bar"

        def exec_fn(ctx):
            if ctx.symbol == "AAPL" and not ctx.long_pos():
                ctx.buy_shares = 200
            if ctx.symbol == "SPY":
                assert ctx.session["foo"] == "bar"

        df = data_source_df[data_source_df["symbol"] == "AAPL"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ["SPY", "AAPL"])
        strategy.set_before_exec(before_exec_fn)
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "AAPL"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 200
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0

    def test_backtest_when_before_exec_and_no_executions(self, data_source_df):
        def before_exec_fn(ctxs):
            assert len(ctxs) == 2
            assert isinstance(ctxs["SPY"], ExecContext)
            assert isinstance(ctxs["AAPL"], ExecContext)
            if not ctxs["AAPL"].long_pos():
                ctxs["AAPL"].buy_shares = 200

        df = data_source_df[data_source_df["symbol"] == "AAPL"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(None, ["SPY", "AAPL"])
        strategy.set_before_exec(before_exec_fn)
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "AAPL"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 200
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0

    def test_backtest_when_after_exec(self, data_source_df):
        def after_exec_fn(ctxs):
            assert len(ctxs) == 2
            assert isinstance(ctxs["SPY"], ExecContext)
            assert isinstance(ctxs["AAPL"], ExecContext)
            if not ctxs["AAPL"].long_pos():
                ctxs["AAPL"].buy_shares = 300

        def exec_fn(ctx):
            if ctx.symbol == "AAPL" and not ctx.long_pos():
                ctx.buy_shares = 200

        df = data_source_df[data_source_df["symbol"] == "AAPL"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ["SPY", "AAPL"])
        strategy.set_after_exec(after_exec_fn)
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "AAPL"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 300
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0

    def test_backtest_when_after_exec_and_no_executions(self, data_source_df):
        def after_exec_fn(ctxs):
            assert len(ctxs) == 2
            assert isinstance(ctxs["SPY"], ExecContext)
            assert isinstance(ctxs["AAPL"], ExecContext)
            if not ctxs["AAPL"].long_pos():
                ctxs["AAPL"].buy_shares = 200

        df = data_source_df[data_source_df["symbol"] == "AAPL"]
        dates = df["date"].unique()
        dates = dates[dates <= np.datetime64(END_DATE)]
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(None, ["SPY", "AAPL"])
        strategy.set_after_exec(after_exec_fn)
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        buy_order = result.orders.iloc[0]
        assert buy_order["type"] == "buy"
        assert buy_order["symbol"] == "AAPL"
        assert buy_order["date"] == dates[1]
        assert buy_order["shares"] == 200
        assert np.isnan(buy_order["limit_price"])
        assert buy_order["fees"] == 0

    def test_backtest_when_warmup(self, data_source_df):
        def exec_fn(ctx):
            if ctx.bars <= 10:
                raise AssertionError("Warmup failed.")
            elif not ctx.long_pos():
                ctx.buy_shares = 100

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(warmup=10)
        assert len(result.orders) == 1

    def test_backtest_when_warmup_invalid_then_error(self, data_source_df):
        def exec_fn(ctx):
            pass

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        with pytest.raises(ValueError, match=re.escape("warmup must be > 0.")):
            strategy.backtest(warmup=-1)

    def test_backtest_when_args_and_kwargs(self, data_source_df):
        def exec_fn(ctx, foo, bar=None):
            assert foo == "foo_value"
            assert bar == "bar_value"

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            foo="foo_value",
            bar="bar_value",
        )
        strategy.backtest(calc_bootstrap=False)

    def test_walkforward_when_args_and_kwargs(self, data_source_df):
        def exec_fn(ctx, foo, bar=None):
            assert foo == "foo_value"
            assert bar == "bar_value"

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            foo="foo_value",
            bar="bar_value",
        )
        strategy.walkforward(windows=2, calc_bootstrap=False)


class TestStrategyTimeframes:
    @pytest.mark.parametrize("disable_parallel_indicators", [True, False])
    def test_weekly_timeframe_backtest(
        self, data_source_df, disable_parallel_indicators, scope
    ):
        from pybroker.indicator import indicator

        def sma(bar_data, period):
            close = bar_data.close
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                out[i] = np.mean(close[i - period + 1 : i + 1])
            return out

        sma_ind = indicator("sma20", sma, period=5)
        seen = []

        def exec_fn(ctx):
            wk = ctx.timeframe("weekly")
            if len(wk.close) > 0:
                seen.append(
                    (len(wk.close), wk.close[-1], len(wk.indicator("sma20")))
                )

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_timeframes("weekly")
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[sma_ind, sma_ind.timeframe("weekly")],
        )
        strategy.walkforward(
            windows=1,
            disable_parallel_indicators=disable_parallel_indicators,
        )
        assert seen
        for n_bars, _close, n_ind in seen:
            assert n_ind == n_bars

    def test_undeclared_timeframe_then_error(self, data_source_df, scope):
        def exec_fn(ctx):
            ctx.timeframe("weekly")

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        with pytest.raises(ValueError, match=re.escape("set_timeframes()")):
            strategy.walkforward(windows=1)

    def test_timeframe_indicator_not_declared_then_error(
        self, data_source_df, scope
    ):
        from pybroker.indicator import indicator

        sma_ind = indicator(
            "sma20",
            lambda bar_data, period: bar_data.close,
            period=5,
        )
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        with pytest.raises(ValueError, match=re.escape("set_timeframes()")):
            strategy.add_execution(
                lambda ctx: None,
                "SPY",
                indicators=[sma_ind.timeframe("weekly")],
            )

    @pytest.mark.parametrize("enable_parallel_models", [True, False])
    def test_weekly_timeframe_model_walkforward(
        self, data_source_df, scope, enable_parallel_models
    ):
        from pybroker.indicator import indicator
        from pybroker.model import model

        def sma(bar_data, period):
            close = bar_data.close
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                out[i] = np.mean(close[i - period + 1 : i + 1])
            return out

        sma_ind = indicator("sma20", sma, period=5)

        def train_fn(sym, train_data, test_data):
            from tests.fixtures import FakeModel

            return FakeModel(sym, np.zeros(len(test_data)))

        wk_model = model(
            "wk_model",
            train_fn,
            [sma_ind],
            predict_fn=lambda _model, df: np.zeros(len(df)),
        )
        seen = []

        def exec_fn(ctx):
            wk = ctx.timeframe("weekly")
            preds = wk.preds("wk_model")
            if len(preds) > 0:
                seen.append(
                    (len(wk.close), len(preds), len(wk.indicator("sma20")))
                )

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_timeframes("weekly")
        strategy.add_execution(
            exec_fn,
            "SPY",
            models=[wk_model.timeframe("weekly")],
        )
        strategy.walkforward(
            windows=1,
            enable_parallel_models=enable_parallel_models,
        )
        assert seen
        for n_bars, n_preds, n_ind in seen:
            assert n_preds == n_bars
            assert n_ind == n_bars

    def test_invalid_timeframe_granularity_raises(self, data_source_df, scope):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_timeframes("daily")
        strategy.add_execution(lambda ctx: None, "SPY")
        with pytest.raises(ValueError, match="Cannot compress daily bars"):
            strategy.walkforward(windows=1)

    @pytest.mark.parametrize(
        "interval,match",
        [
            ("daily", "Cannot compress daily bars"),
            ("1m", "Cannot compress daily bars"),
            ("1h", "Cannot compress daily bars"),
        ],
    )
    def test_invalid_timeframe_granularity_parametrize(
        self, data_source_df, scope, interval, match
    ):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_timeframes(interval)
        strategy.add_execution(lambda ctx: None, "SPY")
        with pytest.raises(ValueError, match=match):
            strategy.walkforward(windows=1)

    @pytest.mark.parametrize("interval", ["weekly", 5])
    def test_valid_timeframe_granularity_walkforward(
        self, data_source_df, scope, interval
    ):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_timeframes(interval)
        strategy.add_execution(lambda ctx: None, "SPY")
        strategy.walkforward(windows=1)

    def test_valid_subdaily_timeframe_walkforward(self, minute_bars_df, scope):
        from pybroker.indicator import indicator

        def sma(bar_data, period):
            close = bar_data.close
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                out[i] = np.mean(close[i - period + 1 : i + 1])
            return out

        sma_ind = indicator("sma20", sma, period=5)
        seen = []

        def exec_fn(ctx):
            tf_ctx = ctx.timeframe("5m")
            if len(tf_ctx.close) > 0:
                seen.append(len(tf_ctx.close))

        strategy = Strategy(
            minute_bars_df,
            minute_bars_df["date"].min(),
            minute_bars_df["date"].max(),
        )
        strategy.set_timeframes("5m")
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[sma_ind, sma_ind.timeframe("5m")],
        )
        strategy.walkforward(windows=1, timeframe="1m")
        assert seen

    @staticmethod
    def _sma_indicator():
        from pybroker.indicator import indicator

        def sma(bar_data, period):
            close = bar_data.close
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                out[i] = np.mean(close[i - period + 1 : i + 1])
            return out

        return indicator("sma20", sma, period=5)

    @pytest.mark.parametrize("interval", ["weekly", "monthly", 5])
    def test_timeframe_indicator_walkforward(
        self, data_source_df, scope, interval
    ):
        sma_ind = self._sma_indicator()
        seen = []

        def exec_fn(ctx):
            timeframe_ctx = ctx.timeframe(interval)
            if len(timeframe_ctx.close) > 0:
                seen.append(
                    (
                        len(timeframe_ctx.close),
                        len(timeframe_ctx.indicator("sma20")),
                    )
                )

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_timeframes(interval)
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[sma_ind, sma_ind.timeframe(interval)],
        )
        strategy.walkforward(windows=1)
        assert seen
        for n_bars, n_ind in seen:
            assert n_ind == n_bars

    @pytest.mark.parametrize("interval", ["weekly", 5])
    @pytest.mark.parametrize("enable_parallel_models", [True, False])
    def test_timeframe_model_walkforward(
        self, data_source_df, scope, interval, enable_parallel_models
    ):
        from pybroker.model import model

        sma_ind = self._sma_indicator()

        def train_fn(sym, train_data, test_data):
            from tests.fixtures import FakeModel

            return FakeModel(sym, np.zeros(len(test_data)))

        wk_model = model(
            "wk_model",
            train_fn,
            [sma_ind],
            predict_fn=lambda _model, df: np.zeros(len(df)),
        )
        seen = []

        def exec_fn(ctx):
            timeframe_ctx = ctx.timeframe(interval)
            preds = timeframe_ctx.preds("wk_model")
            if len(preds) > 0:
                seen.append((len(timeframe_ctx.close), len(preds)))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_timeframes(interval)
        strategy.add_execution(
            exec_fn,
            "SPY",
            models=[wk_model.timeframe(interval)],
        )
        strategy.walkforward(
            windows=1,
            enable_parallel_models=enable_parallel_models,
        )
        assert seen
        for n_bars, n_preds in seen:
            assert n_preds == n_bars

    def test_timeframe_indicator_and_model_same_token(
        self, data_source_df, scope
    ):
        from pybroker.model import model

        sma_ind = self._sma_indicator()

        def train_fn(sym, train_data, test_data):
            from tests.fixtures import FakeModel

            return FakeModel(sym, np.zeros(len(test_data)))

        wk_model = model(
            "wk_model",
            train_fn,
            [sma_ind],
            predict_fn=lambda _model, df: np.zeros(len(df)),
        )
        seen = []

        def exec_fn(ctx):
            timeframe_ctx = ctx.timeframe("weekly")
            preds = timeframe_ctx.preds("wk_model")
            if len(preds) > 0:
                seen.append(
                    (
                        len(timeframe_ctx.close),
                        len(preds),
                        len(timeframe_ctx.indicator("sma20")),
                    )
                )

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_timeframes("weekly")
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[sma_ind.timeframe("weekly")],
            models=[wk_model.timeframe("weekly")],
        )
        strategy.walkforward(windows=1)
        assert seen
        for n_bars, n_preds, n_ind in seen:
            assert n_preds == n_bars
            assert n_ind == n_bars

    def test_multiple_declared_timeframes(self, data_source_df, scope):
        seen = []

        def exec_fn(ctx):
            weekly = ctx.timeframe("weekly")
            every5 = ctx.timeframe(5)
            if len(weekly.close) > 0 and len(every5.close) > 0:
                seen.append((len(weekly.close), len(every5.close)))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_timeframes("weekly", 5)
        strategy.add_execution(exec_fn, "SPY")
        strategy.walkforward(windows=1)
        assert seen
        for n_weekly, n_every5 in seen:
            assert n_weekly > 0
            assert n_every5 > 0
            assert n_weekly != n_every5
