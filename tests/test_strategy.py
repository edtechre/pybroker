"""Unit tests for strategy.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import json
import numpy as np
import os
import pandas as pd
import pytest
import re
import warnings
from importlib import import_module
from .fixtures import *  # noqa: F401
from collections import defaultdict, deque
from datetime import datetime
from decimal import Decimal
from pybroker.common import DataCol, PriceType, to_datetime
from pybroker.config import StrategyConfig
from pybroker.context import ExecContext, RotationContext
from pybroker.indicator import indicator
from pybroker.model import model
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
from pybroker.slippage import (
    FixedSlippageModel,
    SlippageModel,
    VolatilitySlippageModel,
    VolumeSlippageModel,
)
from pybroker.strategy import (
    BacktestMixin,
    BacktestSettings,
    Execution,
    Strategy,
    TestResult,
    WalkforwardMixin,
    _DEFAULT_JSON_INCLUDE,
    _is_rankable,
    _rank_by_score,
    _rank_by_short_score,
)
from pybroker.common import (
    _resolve_execution_symbols,
    _resolve_executions,
    _selection_df,
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


def _run_rotation(
    symbols,
    exec_fn,
    *,
    num_bars,
    worst_rank_held,
    max_long_positions=None,
    max_short_positions=None,
    leverage=1.0,
    config=None,
    rotation_sizer=None,
):
    if config is None:
        config = StrategyConfig(
            max_long_positions=max_long_positions,
            max_short_positions=max_short_positions,
            leverage=leverage,
        )
    portfolio = Portfolio(
        100_000,
        max_long_positions=max_long_positions,
        max_short_positions=max_short_positions,
        leverage=config.leverage,
    )
    exec = Execution(
        id=1,
        symbols=frozenset(symbols),
        fn=exec_fn,
        model_names=frozenset(),
        indicator_names=frozenset(),
    )
    BacktestMixin().backtest_executions(
        config=config,
        backtest_settings=BacktestSettings(
            max_long_positions=max_long_positions,
            max_short_positions=max_short_positions,
            worst_rank_held=worst_rank_held,
        ),
        executions={exec},
        before_exec_fn=None,
        after_exec_fn=None,
        rotation_sizer=rotation_sizer,
        sessions=defaultdict(dict),
        models={},
        indicator_data={},
        test_data=_rotational_test_df(symbols, num_bars=num_bars),
        portfolio=portfolio,
        exit_dates={},
    )
    return portfolio


def _selector_trending_df(
    symbol_bars: dict[str, tuple[float, int]],
    num_bars: int = 12,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Universe of rising prices, where each symbol trades ``bars`` bars.

    A symbol with fewer bars than ``num_bars`` stops trading partway through,
    which is what a :class:`pybroker.common.SymbolSelector` has to cope with.
    """
    dates = pd.date_range(start, periods=num_bars, freq="D")
    rows = []
    for sym, (base, bars) in symbol_bars.items():
        for i, date in enumerate(dates[:bars]):
            rows.append(
                {
                    "symbol": sym,
                    "date": date,
                    "open": base + i,
                    "high": base + i + 1,
                    "low": base + i - 1,
                    "close": base + i,
                    "volume": 100 + i,
                    "adj_close": base + i,
                }
            )
    return pd.DataFrame(rows)


def _rotating_selector(picks: list[list[str]]):
    """Selector returning ``picks[n]`` on its nth call."""
    state = {"n": 0}

    def selector(_df):
        result = picks[min(state["n"], len(picks) - 1)]
        state["n"] += 1
        return result

    return selector


def _buy_once(ctx):
    if ctx.long_pos() is None:
        ctx.buy_shares = 10


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
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
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

        def track_cache(indicator_syms, cache_date_fields, hyperparams=None):
            data, uncached = real_get(
                indicator_syms, cache_date_fields, hyperparams
            )
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

    def test_selection_df_without_train_window_raises(self):
        df = _selector_universe_df({"AAA": 100})
        with pytest.raises(ValueError, match="requires a training window"):
            _selection_df(
                {self._execution(lambda _d: ["AAA"])}, df.iloc[:0], df
            )

    def test_selection_df_without_selector_returns_empty_train(self):
        df = _selector_universe_df({"AAA": 100})
        assert _selection_df(
            {self._execution(frozenset(("AAA",)))}, df.iloc[:0], df
        ).empty

    def test_backtest_with_selector_raises(self):
        df = _selector_universe_df({"AAA": 100, "BBB": 200}, num_bars=12)
        strategy = Strategy(df, "2020-01-01", "2020-01-12")
        strategy.add_execution(lambda _ctx: None, lambda _d: ["AAA"])
        with pytest.raises(ValueError, match="requires a training window"):
            strategy.backtest()

    def test_walkforward_with_selector_and_no_train_size_raises(self):
        df = _selector_universe_df({"AAA": 100, "BBB": 200}, num_bars=12)
        strategy = Strategy(df, "2020-01-01", "2020-01-12")
        strategy.add_execution(lambda _ctx: None, lambda _d: ["AAA"])
        with pytest.raises(ValueError, match="requires a training window"):
            strategy.walkforward(windows=1, train_size=0)

    @pytest.mark.parametrize(
        "selector",
        [
            lambda d: d.groupby("symbol")["volume"].mean().nlargest(1).index,
            lambda d: d["symbol"].unique()[:1],
            lambda d: ("BBB",),
            lambda d: (s for s in ["BBB"]),
        ],
    )
    def test_selector_accepts_any_symbol_sequence(self, selector):
        df = _selector_universe_df({"BBB": 500}, num_bars=8)
        assert _resolve_execution_symbols(
            self._execution(selector), df
        ) == frozenset(("BBB",))

    def test_selector_non_string_symbols_raises(self):
        df = _selector_universe_df({"AAA": 100}, num_bars=8)
        with pytest.raises(TypeError, match="in the returned sequence"):
            _resolve_execution_symbols(
                self._execution(lambda _d: ["AAA", 7]), df
            )

    def test_selector_returning_a_string_raises(self):
        df = _selector_universe_df({"AAA": 100}, num_bars=8)
        with pytest.raises(TypeError, match="sequence of symbols"):
            _resolve_execution_symbols(self._execution(lambda _d: "AAA"), df)

    def test_dropped_symbol_without_new_bars_exits_at_last_bar(self):
        # AAA stops trading before the window that drops it, so there is no
        # bar in the new window to exit on.
        df = _selector_trending_df({"AAA": (100.0, 9), "BBB": (200.0, 12)})
        strategy = Strategy(
            df,
            "2020-01-01",
            "2020-01-12",
            StrategyConfig(initial_cash=Decimal(100_000), buy_delay=1),
        )
        strategy.add_execution(
            _buy_once, _rotating_selector([["AAA"], ["BBB"]])
        )
        result = strategy.walkforward(windows=2, train_size=0.5, lookahead=1)
        aaa_trades = result.trades[result.trades["symbol"] == "AAA"]
        assert len(aaa_trades) == 1
        # AAA's final bar, not a bar of the window that dropped it.
        assert aaa_trades.iloc[0]["exit_date"] == pd.Timestamp("2020-01-09")
        assert float(aaa_trades.iloc[0]["exit"]) == 108.0

    def test_dropped_symbol_with_stops_and_no_new_bars(self):
        df = _selector_trending_df({"AAA": (100.0, 9), "BBB": (200.0, 12)})

        def buy_with_stop(ctx):
            if ctx.long_pos() is None:
                ctx.buy_shares = 10
                ctx.stop_loss_pct = 50

        strategy = Strategy(
            df,
            "2020-01-01",
            "2020-01-12",
            StrategyConfig(initial_cash=Decimal(100_000), buy_delay=1),
        )
        strategy.add_execution(
            buy_with_stop, _rotating_selector([["AAA"], ["BBB"]])
        )
        # A stop on a dropped symbol with no bar left must not raise when the
        # next window checks stops.
        result = strategy.walkforward(windows=2, train_size=0.5, lookahead=1)
        assert not result.trades[result.trades["symbol"] == "AAA"].empty

    def test_boundary_liquidation_applies_slippage(self):
        df = _selector_trending_df({"AAA": (100.0, 12), "BBB": (200.0, 12)})
        calls = []

        class _HalvingSlippage(SlippageModel):
            def process(self, buy_shares, sell_shares, ctx):
                return buy_shares, sell_shares

            def adjust_fill(
                self,
                side,
                symbol,
                shares,
                fill_price,
                col_scope=None,
                ind_scope=None,
                sym_end_index=None,
            ):
                calls.append((side, symbol))
                return shares, fill_price / 2

        strategy = Strategy(
            df,
            "2020-01-01",
            "2020-01-12",
            StrategyConfig(initial_cash=Decimal(100_000), buy_delay=1),
        )
        strategy.set_slippage_model(_HalvingSlippage())
        strategy.add_execution(
            _buy_once, _rotating_selector([["AAA"], ["BBB"]])
        )
        result = strategy.walkforward(windows=2, train_size=0.5, lookahead=1)
        assert ("sell", "AAA") in calls
        aaa_sell = result.orders[
            (result.orders["symbol"] == "AAA")
            & (result.orders["type"] == "sell")
        ]
        assert len(aaa_sell) == 1
        assert float(aaa_sell.iloc[0]["fill_price"]) == 109.0 / 2

    def test_signals_scoped_to_selected_symbols(self):
        df = _selector_universe_df(
            {"AAA": 100, "BBB": 200, "CCC": 300}, num_bars=8
        )
        strategy = Strategy(
            df, "2020-01-01", "2020-01-08", StrategyConfig(return_signals=True)
        )
        strategy.add_execution(lambda _ctx: None, lambda _d: ["BBB"])
        result = strategy.walkforward(windows=1, train_size=0.5, lookahead=1)
        assert set(result.signals) == {"BBB"}

    def test_warns_when_selected_symbol_has_no_test_data(self):
        # AAA trades only during the train half.
        df = _selector_trending_df({"AAA": (100.0, 6), "BBB": (200.0, 12)})
        strategy = Strategy(df, "2020-01-01", "2020-01-12")
        strategy.add_execution(lambda _ctx: None, lambda _d: ["AAA"])
        with pytest.warns(UserWarning, match="no data in this test window"):
            strategy.walkforward(windows=1, train_size=0.5, lookahead=1)

    def test_no_warning_when_selected_symbols_have_test_data(self):
        df = _selector_trending_df({"AAA": (100.0, 12), "BBB": (200.0, 12)})
        strategy = Strategy(df, "2020-01-01", "2020-01-12")
        strategy.add_execution(lambda _ctx: None, lambda _d: ["AAA"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            strategy.walkforward(windows=1, train_size=0.5, lookahead=1)


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
        assert len(portfolio._metrics_bars) == len(
            data_source_df["date"].unique()
        )
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
        assert len(portfolio._metrics_bars) == len(
            data_source_df["date"].unique()
        )
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

        assert len(portfolio._metrics_bars) == len(
            data_source_df["date"].unique()
        )
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
        assert len(portfolio._metrics_bars)
        assert not len(portfolio.position_bars)
        assert not len(portfolio.orders)
        assert not len(portfolio.trades)

    def test_backtest_executions_when_buy_score(self, data_source_df):
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
            config=StrategyConfig(max_long_positions=2),
            backtest_settings=BacktestSettings(
                max_long_positions=2, worst_rank_held=5
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
            config=StrategyConfig(max_long_positions=2),
            backtest_settings=BacktestSettings(
                max_long_positions=2, worst_rank_held=5
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
            config=StrategyConfig(max_long_positions=2),
            backtest_settings=BacktestSettings(
                max_long_positions=2, worst_rank_held=5
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
            config=StrategyConfig(max_short_positions=2),
            backtest_settings=BacktestSettings(
                max_short_positions=2, worst_rank_held=5
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
            config=StrategyConfig(max_short_positions=2),
            backtest_settings=BacktestSettings(
                max_short_positions=2, worst_rank_held=5
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
            config=StrategyConfig(max_short_positions=2),
            backtest_settings=BacktestSettings(
                max_short_positions=2, worst_rank_held=5
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
            100_000, max_long_positions=2, max_short_positions=2, leverage=2.0
        )
        mixin = BacktestMixin()
        mixin.backtest_executions(
            config=StrategyConfig(
                max_long_positions=2,
                max_short_positions=2,
                leverage=2.0,
            ),
            backtest_settings=BacktestSettings(
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
                "score cannot be used with rotation enabled; use long_score or "
                "short_score instead."
            ),
        ):
            mixin.backtest_executions(
                config=StrategyConfig(max_long_positions=2),
                backtest_settings=BacktestSettings(
                    max_long_positions=2, worst_rank_held=5
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

    def test_backtest_executions_when_worst_rank_held_ignores_user_sell(self):
        symbols = ["S1", "S2", "S3"]
        scores_by_bar = [
            {"S1": 30, "S2": 20, "S3": 10},
            {"S1": 1, "S2": 2, "S3": 60},
        ]
        user_sells = []

        def exec_fn(ctx):
            idx = min(ctx.bars - 1, len(scores_by_bar) - 1)
            ctx.long_score = scores_by_bar[idx][ctx.symbol]
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
            config=StrategyConfig(max_long_positions=2),
            backtest_settings=BacktestSettings(
                max_long_positions=2, worst_rank_held=5
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
        # S1 stays inside the hold band, so rotation keeps it and the sell
        # placed by the execution is discarded.
        assert user_sells == ["S1"]
        assert s1_sells == []
        assert "S1" in portfolio.long_positions

    def test_backtest_executions_when_after_rotation_custom_size(self):
        symbols = ["S1", "S2", "S3"]
        weights = {"S1": 0.7, "S2": 0.3}

        def exec_fn(ctx):
            ctx.long_score = {"S1": 30, "S2": 20, "S3": 10}[ctx.symbol]

        def after_rotation_fn(rotation: RotationContext):
            for sym, ctx in rotation.ctxs.items():
                if (
                    ctx.buy_shares
                    and not ctx.long_pos()
                    and ctx.sell_shares is None
                    and sym in weights
                ):
                    ctx.buy_shares = ctx.calc_target_shares(weights[sym])
                    ctx.buy_fill_price = PriceType.CLOSE

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
            config=StrategyConfig(max_long_positions=2),
            backtest_settings=BacktestSettings(
                max_long_positions=2, worst_rank_held=5
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            rotation_sizer=after_rotation_fn,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        buy_orders = [
            order for order in portfolio.orders if order.type == "buy"
        ]
        shares_by_sym = {order.symbol: order.shares for order in buy_orders}
        assert shares_by_sym == {"S1": 700, "S2": 300}

    def test_backtest_executions_when_after_rotation_default_equal_weight(
        self,
    ):
        symbols = ["S1", "S2", "S3"]

        def exec_fn(ctx):
            ctx.long_score = {"S1": 30, "S2": 20, "S3": 10}[ctx.symbol]

        def fill_prices_only(rotation: RotationContext):
            for ctx in rotation.ctxs.values():
                if ctx.buy_shares:
                    ctx.buy_fill_price = PriceType.CLOSE

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
            config=StrategyConfig(max_long_positions=2),
            backtest_settings=BacktestSettings(
                max_long_positions=2, worst_rank_held=5
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            rotation_sizer=fill_prices_only,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        buy_orders = [
            order for order in portfolio.orders if order.type == "buy"
        ]
        shares_by_sym = {order.symbol: order.shares for order in buy_orders}
        assert shares_by_sym == {"S1": 500, "S2": 500}

    def test_backtest_executions_when_after_rotation_does_not_override_sells(
        self,
    ):
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

        def after_rotation_fn(rotation: RotationContext):
            for ctx in rotation.ctxs.values():
                if ctx.sell_shares is not None:
                    continue
                if ctx.buy_shares and not ctx.long_pos():
                    ctx.buy_shares = ctx.calc_target_shares(0.25)
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
            config=StrategyConfig(max_long_positions=2),
            backtest_settings=BacktestSettings(
                max_long_positions=2, worst_rank_held=5
            ),
            executions={exec},
            before_exec_fn=None,
            after_exec_fn=None,
            rotation_sizer=after_rotation_fn,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=test_df,
            portfolio=portfolio,
            exit_dates={},
        )
        sell_orders = [
            order for order in portfolio.orders if order.type == "sell"
        ]
        assert len(sell_orders) == 1
        assert sell_orders[0].symbol == "S1"
        assert "S2" in portfolio.long_positions
        assert "S1" not in portfolio.long_positions

    def test_backtest_executions_when_rotation_sizer_without_rotation_then_error(
        self,
    ):
        symbols = ["S1", "S2", "S3"]

        def exec_fn(ctx):
            ctx.long_score = {"S1": 30, "S2": 20, "S3": 10}[ctx.symbol]

        def rotation_sizer(rotation: RotationContext):
            pass

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
            match="Rotation sizer is set but rotation is not enabled",
        ):
            mixin.backtest_executions(
                config=StrategyConfig(max_long_positions=2),
                backtest_settings=BacktestSettings(max_long_positions=2),
                executions={exec},
                before_exec_fn=None,
                after_exec_fn=None,
                rotation_sizer=rotation_sizer,
                sessions=defaultdict(dict),
                models={},
                indicator_data={},
                test_data=test_df,
                portfolio=portfolio,
                exit_dates={},
            )

    def test_backtest_executions_when_rotation_shared_long_short_universe(
        self,
    ):
        # Ranking one universe from both ends: the strongest names are the best
        # longs and the weakest are the best shorts.
        symbols = ["A", "B", "C", "D"]
        momentum = {"A": 40, "B": 30, "C": 20, "D": 10}

        def exec_fn(ctx):
            ctx.long_score = momentum[ctx.symbol]
            ctx.short_score = momentum[ctx.symbol]

        portfolio = _run_rotation(
            symbols,
            exec_fn,
            num_bars=4,
            max_long_positions=2,
            max_short_positions=2,
            worst_rank_held=3,
            leverage=2.0,
        )
        assert set(portfolio.long_positions) == {"A", "B"}
        assert set(portfolio.short_positions) == {"C", "D"}

    def test_backtest_executions_when_rotation_overlapping_candidates(self):
        # Limits sum to more slots than there are symbols, so B is a candidate
        # on both sides and is resolved to the side it ranks better on.
        symbols = ["A", "B", "C"]
        momentum = {"A": 30, "B": 20, "C": 10}

        def exec_fn(ctx):
            ctx.long_score = momentum[ctx.symbol]
            ctx.short_score = momentum[ctx.symbol]

        portfolio = _run_rotation(
            symbols,
            exec_fn,
            num_bars=4,
            max_long_positions=2,
            max_short_positions=2,
            worst_rank_held=3,
            leverage=2.0,
        )
        # B ranks 2nd both ways, and ties go long.
        assert set(portfolio.long_positions) == {"A", "B"}
        assert set(portfolio.short_positions) == {"C"}

    def test_backtest_executions_when_rotation_buy_delay(self):
        symbols = ["A", "B"]

        def exec_fn(ctx):
            ctx.long_score = {"A": 30, "B": 20}[ctx.symbol]

        portfolio = _run_rotation(
            symbols,
            exec_fn,
            num_bars=6,
            max_long_positions=4,
            worst_rank_held=4,
            config=StrategyConfig(max_long_positions=4, buy_delay=2),
        )
        # A delayed entry stays pending for a bar. Re-issuing it would stack a
        # second order and double the 25% target allocation.
        buy_orders = [
            order for order in portfolio.orders if order.type == "buy"
        ]
        assert len(buy_orders) == 2
        assert {
            sym: pos.shares for sym, pos in portfolio.long_positions.items()
        } == {"A": 250, "B": 250}

    def test_backtest_executions_when_rotation_fills_free_slots_only(self):
        symbols = ["S1", "S2", "S3", "S4", "S5", "S6"]
        scores = {"S1": 60, "S2": 50, "S3": 40, "S4": 30, "S5": 20, "S6": 10}
        entries_per_bar = []

        def exec_fn(ctx):
            ctx.long_score = scores[ctx.symbol]

        def rotation_sizer(rotation: RotationContext):
            entries_per_bar.append(
                sum(
                    1
                    for ctx in rotation.ctxs.values()
                    if ctx.buy_shares is not None
                    or ctx.sell_shares is not None
                )
            )

        _run_rotation(
            symbols,
            exec_fn,
            num_bars=3,
            max_long_positions=2,
            worst_rank_held=5,
            rotation_sizer=rotation_sizer,
        )
        # Two slots are filled on the first bar and both holdings stay inside
        # the band, so no further entries are generated.
        assert entries_per_bar == [2, 0, 0]

    def test_backtest_executions_when_rotation_ignores_user_buy(self):
        symbols = ["A", "B", "C", "D"]
        scores_by_bar = [
            {"A": 40, "B": 30, "C": 20, "D": 10},
            {"A": 40, "B": 30, "C": 20, "D": 10},
            {"A": 1, "B": 2, "C": 40, "D": 30},
        ]

        def exec_fn(ctx):
            idx = min(ctx.bars - 1, len(scores_by_bar) - 1)
            ctx.long_score = scores_by_bar[idx][ctx.symbol]
            if ctx.long_pos() is not None:
                ctx.buy_shares = 1

        portfolio = _run_rotation(
            symbols,
            exec_fn,
            num_bars=5,
            max_long_positions=2,
            worst_rank_held=2,
        )
        # Pyramiding is discarded, so every fill is a full rotation entry.
        assert [
            order.shares for order in portfolio.orders if order.type == "buy"
        ] == [500, 500, 500, 500]
        assert set(portfolio.long_positions) == {"C", "D"}

    def test_backtest_executions_when_rotation_ignores_user_short(self):
        symbols = ["A", "B", "C"]

        def exec_fn(ctx):
            ctx.long_score = {"A": 30, "B": 20, "C": 10}[ctx.symbol]
            if ctx.symbol == "C" and ctx.short_pos() is None:
                ctx.sell_shares = 10
                ctx.sell_fill_price = PriceType.CLOSE

        portfolio = _run_rotation(
            symbols,
            exec_fn,
            num_bars=4,
            max_long_positions=2,
            worst_rank_held=3,
        )
        assert not portfolio.short_positions
        assert not [
            order for order in portfolio.orders if order.type == "sell"
        ]
        assert set(portfolio.long_positions) == {"A", "B"}

    def test_backtest_executions_when_rotation_stops(self):
        symbols = ["S1", "S2", "S3"]

        def exec_fn(ctx):
            ctx.long_score = {"S1": 30, "S2": 20, "S3": 10}[ctx.symbol]
            ctx.stop_loss_pct = 10

        portfolio = _run_rotation(
            symbols,
            exec_fn,
            num_bars=3,
            max_long_positions=2,
            worst_rank_held=3,
        )
        # Stops attach to the positions rotation opens, and are dropped for the
        # symbol it left untraded.
        assert set(portfolio.long_positions) == {"S1", "S2"}
        assert set(portfolio._active_stops) == {"S1", "S2"}

    def test_backtest_executions_when_cover_and_buy_on_same_date(self):
        symbols = ["A", "B", "S"]

        def exec_fn(ctx):
            if ctx.symbol == "S":
                if ctx.bars == 1:
                    ctx.sell_shares = 10
                    ctx.sell_fill_price = PriceType.CLOSE
                elif ctx.bars == 2:
                    ctx.cover_all_shares()
                    ctx.cover_fill_price = PriceType.CLOSE
                return
            if ctx.bars == 2:
                ctx.buy_shares = 100
                ctx.buy_fill_price = PriceType.CLOSE
                ctx.long_score = {"A": 10, "B": 20}[ctx.symbol]

        test_df = _rotational_test_df(symbols, num_bars=6)
        exec = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(
            100_000, max_long_positions=1, max_short_positions=1
        )
        BacktestMixin().backtest_executions(
            config=StrategyConfig(max_long_positions=1, max_short_positions=1),
            backtest_settings=BacktestSettings(
                max_long_positions=1, max_short_positions=1
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
        # The cover lands on the same date as both buys. Buys still need to be
        # ranked, or the position limit hands the slot to whichever symbol was
        # scheduled first.
        assert set(portfolio.long_positions) == {"B"}

    def test_backtest_when_after_rotation(self):
        symbols = ["S1", "S2", "S3"]
        weights = {"S1": 0.7, "S2": 0.3}

        def exec_fn(ctx):
            ctx.long_score = {"S1": 30, "S2": 20, "S3": 10}[ctx.symbol]

        def after_rotation_fn(rotation: RotationContext):
            for sym, ctx in rotation.ctxs.items():
                if (
                    ctx.buy_shares
                    and not ctx.long_pos()
                    and ctx.sell_shares is None
                    and sym in weights
                ):
                    ctx.buy_shares = ctx.calc_target_shares(weights[sym])
                    ctx.buy_fill_price = PriceType.CLOSE

        test_df = _rotational_test_df(symbols, num_bars=2)
        strategy = Strategy(
            test_df,
            "2020-01-01",
            "2020-01-02",
            StrategyConfig(),
        )
        strategy.set_max_long_positions(2)
        strategy.enable_rotation(5, sizer=after_rotation_fn)
        strategy.add_execution(exec_fn, symbols)
        result = strategy.backtest(calc_bootstrap=False)
        buy_orders = result.orders[result.orders["type"] == "buy"]
        shares_by_sym = dict(zip(buy_orders["symbol"], buy_orders["shares"]))
        assert shares_by_sym == {"S1": 700, "S2": 300}

    def test_backtest_executions_when_sell_score(self, data_source_df):
        def sell_exec_fn(ctx):
            ctx.sell_fill_price = PriceType.CLOSE
            ctx.sell_shares = 200
            if ctx.symbol == "AAPL":
                ctx.long_score = 1
            else:
                ctx.long_score = 0

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
                ctx.long_score = 2
            else:
                ctx.long_score = 1
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
            bootstrap_samples=100,
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

    def test_walkforward_pooled_model_on_weekly_interval(
        self, data_source_df, exec_pooled_model_source, scope, indicators
    ):
        _pooled_train_calls.clear()
        saw_weekly_preds = []

        def exec_fn(ctx):
            weekly = ctx.interval("weekly")
            preds = weekly.preds(POOLED_MODEL_NAME)
            if len(preds) > 0:
                saw_weekly_preds.append(len(preds))
                assert isinstance(
                    weekly.model(POOLED_MODEL_NAME), PooledFakeModel
                )
                assert len(preds) == len(weekly.close)

        config = StrategyConfig()
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(
            exec_fn,
            ["AAPL", "MSFT"],
            models=[exec_pooled_model_source],
            indicators=indicators,
            intervals=["weekly"],
        )
        result = strategy.walkforward(
            windows=1,
            lookahead=1,
            timeframe="1d",
            train_size=0.5,
            seed=42,
        )
        assert not result.portfolio.empty
        assert len(_pooled_train_calls) == 2
        assert saw_weekly_preds

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
            return_signals=return_signals,
            return_stops=return_stops,
            record_position_bars=True,
        )
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(exec_fn, ["AAPL", "SPY"])
        result = strategy.walkforward(windows=3, calc_bootstrap=False)
        dates = set()
        for _, test_idx in strategy.walkforward_split(
            data_source_df, windows=3, lookahead=1, train_size=0.5
        ):
            df = data_source_df.iloc[test_idx]
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

    def test_record_portfolio_bars_metrics_parity(self, data_source_df):
        def exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_loss_pct = 5

        data_source_df = data_source_df[
            data_source_df["date"] <= to_datetime(END_DATE)
        ]
        symbols = ["AAPL", "SPY"]
        default_strategy = Strategy(
            data_source_df,
            START_DATE,
            END_DATE,
            StrategyConfig(bootstrap_samples=10),
        )
        default_strategy.add_execution(exec_fn, symbols)
        default_result = default_strategy.walkforward(
            windows=1,
            lookahead=1,
            train_size=0.5,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )
        recorded_strategy = Strategy(
            data_source_df,
            START_DATE,
            END_DATE,
            StrategyConfig(
                bootstrap_samples=10,
                record_portfolio_bars=True,
                record_position_bars=True,
            ),
        )
        recorded_strategy.add_execution(exec_fn, symbols)
        recorded_result = recorded_strategy.walkforward(
            windows=1,
            lookahead=1,
            train_size=0.5,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )
        pd.testing.assert_frame_equal(
            default_result.portfolio, recorded_result.portfolio
        )
        assert default_result.metrics == recorded_result.metrics

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
        payload = result.to_json()
        assert set(payload.keys()) == {
            "start_date",
            "end_date",
            "metrics",
            "trades",
            "orders",
            "bootstrap",
        }
        json.dumps(payload, allow_nan=False)
        assert "portfolio" not in payload
        include_all = _DEFAULT_JSON_INCLUDE | frozenset({"portfolio"})
        payload_with_portfolio = result.to_json(include=include_all)
        assert "portfolio" in payload_with_portfolio
        assert len(payload_with_portfolio["portfolio"]) <= 100
        truncated = result.to_json(max_rows=1)
        assert len(truncated["trades"]) <= 1
        assert len(truncated["orders"]) <= 1
        result.to_json_str()

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
        "sell_delay, bootstrap_samples, expected_msg",
        [
            (
                -1,
                None,
                None,
                1,
                1,
                100,
                "initial_cash must be greater than 0.",
            ),
            (
                10_000,
                0,
                None,
                1,
                1,
                100,
                "max_long_positions must be greater than 0.",
            ),
            (
                10_000,
                None,
                0,
                1,
                1,
                100,
                "max_short_positions must be greater than 0.",
            ),
            (
                10_000,
                None,
                None,
                0,
                1,
                100,
                "buy_delay must be greater than 0.",
            ),
            (
                10_000,
                None,
                None,
                1,
                0,
                100,
                "sell_delay must be greater than 0.",
            ),
            (
                10_000,
                None,
                None,
                1,
                1,
                0,
                "bootstrap_samples must be greater than 0.",
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
        expected_msg,
    ):
        config = StrategyConfig(
            initial_cash=initial_cash,
            max_long_positions=max_long_positions,
            max_short_positions=max_short_positions,
            buy_delay=buy_delay,
            sell_delay=sell_delay,
            bootstrap_samples=bootstrap_samples,
        )
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            Strategy(data_source_df, START_DATE, END_DATE, config)

    @pytest.mark.parametrize(
        "setup_fn, expected_msg",
        [
            (
                lambda s: s.enable_rotation(5),
                "worst_rank_held requires max_long_positions or "
                "max_short_positions to be set.",
            ),
            (
                lambda s: (
                    s.set_max_long_positions(2),
                    s.enable_rotation(1),
                ),
                "worst_rank_held must be greater than or equal to "
                "max_long_positions.",
            ),
            (
                lambda s: (
                    s.set_max_short_positions(2),
                    s.enable_rotation(1),
                ),
                "worst_rank_held must be greater than or equal to "
                "max_short_positions.",
            ),
        ],
    )
    def test_when_invalid_enable_rotation_then_error(
        self, data_source_df, setup_fn, expected_msg
    ):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        # Rotation settings are validated when the backtest resolves them, so
        # that enable_rotation can be called before the position limits.
        setup_fn(strategy)
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            strategy._resolve_backtest_settings()

    def test_when_enable_rotation_before_position_limits(self, data_source_df):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.enable_rotation(5)
        strategy.set_max_long_positions(2)
        settings = strategy._resolve_backtest_settings()
        assert settings.worst_rank_held == 5
        assert settings.max_long_positions == 2

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

    def test_when_interest_rate_without_bars_per_year_then_error(
        self, data_source_df
    ):
        config = StrategyConfig(interest_rate=7.0)
        with pytest.raises(
            ValueError,
            match=re.escape("bars_per_year is required when interest_rate"),
        ):
            Strategy(data_source_df, START_DATE, END_DATE, config)

    @staticmethod
    def _flat_price_df(symbols, periods=20, price=100.0):
        dates = pd.date_range(START_DATE, periods=periods, freq="D")
        return pd.DataFrame(
            [
                {
                    "symbol": sym,
                    "date": date,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000_000,
                }
                for sym in symbols
                for date in dates
            ]
        )

    @pytest.mark.parametrize("dir", ["long", "short"])
    def test_backtest_when_set_target_shares_then_idempotent(self, dir):
        """Re-stating the same target on a flat price must not churn orders."""

        def exec_fn(ctx):
            ctx.set_target_shares(0.5, dir=dir)

        df = self._flat_price_df(["SPY"])
        config = StrategyConfig(initial_cash=100_000, leverage=2.0)
        strategy = Strategy(df, df["date"].min(), df["date"].max(), config)
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.orders) == 1
        # 50% of 200k of deployable capital at $100 a share.
        assert result.orders.iloc[0]["shares"] == 1000

    def test_backtest_when_rotation_and_leverage_then_slots_equal(self):
        """Every rotation slot gets the same notional regardless of entry.

        The last two symbols only become candidates part way through, so a
        sizing base that is net of already-held exposure would starve them.
        """
        symbols = ["S1", "S2", "S3", "S4"]
        df = self._flat_price_df(symbols)

        def exec_fn(ctx):
            if ctx.symbol in ("S1", "S2") or ctx.bars >= 5:
                ctx.long_score = 1

        config = StrategyConfig(initial_cash=100_000, leverage=2.0)
        strategy = Strategy(df, df["date"].min(), df["date"].max(), config)
        strategy.set_max_long_positions(len(symbols))
        strategy.enable_rotation(len(symbols))
        for sym in symbols:
            strategy.add_execution(exec_fn, sym)
        result = strategy.backtest(calc_bootstrap=False)
        buys = result.orders[result.orders["type"] == "buy"]
        assert len(buys) == len(symbols)
        assert buys["shares"].nunique() == 1
        # 4 equal slots of 200k deployable capital at $100 a share.
        assert buys["shares"].iloc[0] == 500

    def test_backtest_when_same_bar_rotation_respects_leverage(self):
        """Exiting one symbol and entering another on the same bar.

        Both fills land before ``capture_bar``, so sizing the entry off the
        last snapshot would value the account as if the exit had not happened
        yet. No stops involved -- plain scheduled orders reach this.
        """
        dates = pd.date_range(START_DATE, periods=12, freq="D")
        # A gaps down 20% on the bar the exit fills; B is flat.
        a_prices = [100.0] * 6 + [80.0] * 6
        rows = [
            {
                "symbol": sym,
                "date": date,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1_000_000,
            }
            for sym, prices in (("A", a_prices), ("B", [100.0] * 12))
            for date, price in zip(dates, prices)
        ]
        df = pd.DataFrame(rows)

        # Orders fill on the bar after the one that schedules them, so both
        # the exit and the entry land on the first gapped bar.
        entry_date, rotate_date = dates[0], dates[5]

        def a_fn(ctx):
            if ctx.dt == entry_date:
                ctx.buy_shares = 2000
            elif ctx.dt == rotate_date:
                ctx.sell_all_shares()

        def b_fn(ctx):
            if ctx.dt == rotate_date:
                ctx.buy_shares = 5000

        config = StrategyConfig(initial_cash=100_000, leverage=2.0)
        strategy = Strategy(df, dates[0], dates[-1], config)
        strategy.add_execution(a_fn, "A")
        strategy.add_execution(b_fn, "B")
        result = strategy.backtest(calc_bootstrap=False)
        pf = result.portfolio
        assert (pf["market_value"] > 0).all()
        assert (pf["margin_loan"] >= 0).all()
        # Gross exposure never exceeds the configured leverage, on any bar.
        gross = pf["market_value"] + pf["margin_loan"] - pf["cash"]
        assert (gross <= 2.0 * pf["market_value"] + 1e-6).all()

    def test_backtest_when_leverage(self, data_source_df):
        def buy_exec_fn(ctx):
            if ctx.long_pos() is None:
                ctx.set_target_shares(1.0, dir="long")

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

    def test_backtest_when_leverage_short(self, data_source_df):
        def short_exec_fn(ctx):
            if ctx.short_pos() is None:
                ctx.set_target_shares(1.0, dir="short")

        config = StrategyConfig(initial_cash=100_000, leverage=2.0)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(short_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        leveraged_shares = result.orders[result.orders["type"] == "sell"].iloc[
            0
        ]["shares"]

        config = StrategyConfig(initial_cash=100_000, leverage=1.0)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.add_execution(short_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        cash_shares = result.orders[result.orders["type"] == "sell"].iloc[0][
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
        bps = 5
        config = StrategyConfig(initial_cash=500_000)
        strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
        strategy.set_slippage_model(FixedSlippageModel(bps=bps))
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        orders = result.orders
        buy_factor = 1 + bps / 10_000
        sell_factor = 1 - bps / 10_000
        buy_orders = orders[orders["type"] == "buy"]
        assert len(buy_orders) == len(buy_dates)
        for buy_date in buy_dates:
            row = buy_orders[buy_orders["date"] == buy_date]
            assert row["symbol"].item() == "SPY"
            assert row["shares"].item() == 100
            assert np.isnan(row["limit_price"].item())
            base = round(df[df["date"] == buy_date]["close"].item(), 2)
            assert row["fill_price"].item() == round(base * buy_factor, 2)
            assert row["fees"].item() == 0
        sell_orders = orders[orders["type"] == "sell"]
        assert len(sell_orders) == len(sell_dates)
        for sell_date in sell_dates:
            row = sell_orders[sell_orders["date"] == sell_date]
            assert row["symbol"].item() == "SPY"
            assert row["shares"].item() == 100
            assert np.isnan(row["limit_price"].item())
            # Bar-stop exits are slipped adversely, same as scheduled orders.
            base = round(df[df["date"] == sell_date]["open"].item(), 2)
            assert row["fill_price"].item() == round(base * sell_factor, 2)
            assert row["fees"].item() == 0
        assert (result.trades["stop"] == "bar").all()

    def test_backtest_and_walkforward_when_slippage_then_uniform(
        self, data_source_df
    ):
        def buy_exec_fn(ctx):
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.buy_shares = 100
            ctx.hold_bars = 2

        config = StrategyConfig(initial_cash=500_000)

        def make_strategy():
            strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
            strategy.set_slippage_model(FixedSlippageModel(bps=5))
            strategy.add_execution(buy_exec_fn, "SPY")
            return strategy

        backtest_result = make_strategy().backtest(calc_bootstrap=False)
        walkforward_result = make_strategy().walkforward(
            windows=1, train_size=0, calc_bootstrap=False
        )
        pd.testing.assert_frame_equal(
            backtest_result.orders.reset_index(drop=True),
            walkforward_result.orders.reset_index(drop=True),
        )

    def test_backtest_when_slippage_then_deterministic(self, data_source_df):
        def buy_exec_fn(ctx):
            ctx.buy_fill_price = PriceType.CLOSE
            ctx.buy_shares = 100
            ctx.hold_bars = 2

        config = StrategyConfig(initial_cash=500_000)

        def run_backtest():
            strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
            strategy.set_slippage_model(FixedSlippageModel(bps=5))
            strategy.add_execution(buy_exec_fn, "SPY")
            return strategy.backtest(calc_bootstrap=False)

        first = run_backtest()
        second = run_backtest()
        pd.testing.assert_frame_equal(
            first.orders.reset_index(drop=True),
            second.orders.reset_index(drop=True),
        )

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
        assert sell_orders.iloc[0]["shares"] == 90

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
        assert buy_orders.iloc[0]["shares"] == 90

    def test_backtest_when_volume_slippage_and_nan_volume(
        self, data_source_df
    ):
        # NaN volume used to raise decimal.InvalidOperation from min().
        df = data_source_df.copy()
        df.loc[df.index[:20], "volume"] = np.nan

        def buy_exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
            elif ctx.bars == 2:
                ctx.sell_all_shares()

        strategy = Strategy(df, START_DATE, END_DATE)
        strategy.set_slippage_model(VolumeSlippageModel())
        strategy.add_execution(buy_exec_fn, "SPY")
        with pytest.warns(UserWarning, match="missing or NaN 'volume'"):
            result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades)
        assert not result.orders["fill_price"].isna().any()

    def test_backtest_when_volume_slippage_and_no_volume_column(
        self, data_source_df
    ):
        # Every order used to be silently cancelled; now it fails upfront.
        df = data_source_df.drop(columns=["volume"])

        def buy_exec_fn(ctx):
            ctx.buy_shares = 100

        strategy = Strategy(df, START_DATE, END_DATE)
        strategy.set_slippage_model(VolumeSlippageModel())
        strategy.add_execution(buy_exec_fn, "SPY")
        with pytest.raises(
            ValueError, match=re.escape("requires a 'volume' data column")
        ):
            strategy.backtest(calc_bootstrap=False)

    def test_backtest_when_volatility_slippage_and_atr_warmup(
        self, data_source_df
    ):
        # The ATR's leading NaNs used to raise decimal.InvalidOperation.
        def atr_fn(data):
            high, low = data.high, data.low
            tr = np.abs(high - low)
            return pd.Series(tr).rolling(14).mean().to_numpy()

        atr_ind = indicator("atr_14", atr_fn)

        def buy_exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
            elif ctx.bars == 2:
                ctx.sell_all_shares()

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.set_slippage_model(
            VolatilitySlippageModel(atr_indicator="atr_14", scale=0.1)
        )
        strategy.add_execution(buy_exec_fn, "SPY", indicators=atr_ind)
        result = strategy.backtest(calc_bootstrap=False)
        assert len(result.trades)
        assert not result.orders["fill_price"].isna().any()
        assert (result.orders["fill_price"] > 0).all()

    def test_backtest_when_slippage_subclass_overrides_apply_at_fill(
        self, data_source_df
    ):
        # A FixedSlippageModel subclass must not be routed to the fast path.
        class DoublingSlippageModel(FixedSlippageModel):
            def apply_at_fill(self, fill_ctx):
                return fill_ctx.shares, fill_ctx.fill_price * Decimal(2)

        def buy_exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_fill_price = PriceType.CLOSE
                ctx.buy_shares = 100

        df = data_source_df[data_source_df["symbol"] == "SPY"]
        strategy = Strategy(
            df, START_DATE, END_DATE, StrategyConfig(initial_cash=1_000_000)
        )
        strategy.set_slippage_model(DoublingSlippageModel(bps=5))
        strategy.add_execution(buy_exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        buy_orders = result.orders[result.orders["type"] == "buy"]
        assert len(buy_orders) == 1
        buy_date = buy_orders.iloc[0]["date"]
        base = df[df["date"] == buy_date]["close"].item()
        assert buy_orders.iloc[0]["fill_price"] == round(base * 2, 2)

    def test_backtest_when_slippage_and_stop_loss(self, data_source_df):
        bps = 50

        def exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_loss_pct = 5

        df = data_source_df[data_source_df["symbol"] == "SPY"]
        strategy = Strategy(df, START_DATE, END_DATE)
        strategy.set_slippage_model(FixedSlippageModel(bps=bps))
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        stop_exits = result.orders[result.orders["type"] == "sell"]
        assert len(stop_exits)
        assert (result.trades["stop"] == "loss").any()

        unslipped = Strategy(df, START_DATE, END_DATE)
        unslipped.add_execution(exec_fn, "SPY")
        base_result = unslipped.backtest(calc_bootstrap=False)
        base_exits = base_result.orders[base_result.orders["type"] == "sell"]
        # Stop exits are now slipped, so they can never fill higher than the
        # unslipped run's first exit.
        assert (
            stop_exits.iloc[0]["fill_price"] < base_exits.iloc[0]["fill_price"]
        )

    def test_backtest_when_slippage_and_exit_on_last_bar(self, data_source_df):
        bps = 50

        def exec_fn(ctx):
            if not ctx.long_pos():
                ctx.buy_shares = 100

        df = data_source_df[data_source_df["symbol"] == "SPY"]
        config = StrategyConfig(exit_on_last_bar=True)
        strategy = Strategy(df, START_DATE, END_DATE, config)
        strategy.set_slippage_model(FixedSlippageModel(bps=bps))
        strategy.add_execution(exec_fn, "SPY")
        result = strategy.backtest(calc_bootstrap=False)
        sell_orders = result.orders[result.orders["type"] == "sell"]
        assert len(sell_orders) == 1

        unslipped = Strategy(df, START_DATE, END_DATE, config)
        unslipped.add_execution(exec_fn, "SPY")
        base_result = unslipped.backtest(calc_bootstrap=False)
        base_sells = base_result.orders[base_result.orders["type"] == "sell"]
        assert (
            sell_orders.iloc[0]["fill_price"]
            < base_sells.iloc[0]["fill_price"]
        )

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

    def test_backtest_when_bar_stop_and_invalid_fill_price_then_error(
        self, data_source_df
    ):
        # A bar stop's fill price never reaches _verify_input, so without a
        # guard a zero or negative silently books a completed trade.
        def exec_fn(ctx):
            if ctx.bars == 1:
                ctx.buy_shares = 100
                ctx.hold_bars = 1
                ctx.sell_fill_price = 0

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        with pytest.raises(ValueError, match=re.escape("must be > 0")):
            strategy.backtest(calc_bootstrap=False)

    def test_backtest_when_multiple_stops_then_deterministic(
        self, data_source_df
    ):
        # Stops reach the portfolio as a frozenset whose iteration order is
        # not reproducible across processes, so which of two stops that hit on
        # the same bar wins used to vary run to run. Sorting by stop id fixes
        # it; here the stop loss is set first and must always win.
        def run():
            def exec_fn(ctx):
                if ctx.long_pos() is None:
                    ctx.buy_shares = 100
                    ctx.stop_loss_pct = 1
                    ctx.stop_profit_pct = 1

            strategy = Strategy(data_source_df, START_DATE, END_DATE)
            strategy.add_execution(exec_fn, "SPY")
            result = strategy.backtest(calc_bootstrap=False)
            return [
                (row["stop"], float(row["exit"]))
                for _, row in result.trades.iterrows()
            ]

        runs = [run() for _ in range(4)]
        assert all(r == runs[0] for r in runs), runs
        # Bars that straddle both levels are the ones that used to flip; a
        # loss exit on at least one of them means the ordering was exercised.
        assert any(stop == "loss" for stop, _ in runs[0])

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


def _sma(bar_data, period):
    close = bar_data.close
    out = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        out[i] = np.mean(close[i - period + 1 : i + 1])
    return out


class TestStrategyIntervals:
    @pytest.mark.parametrize("disable_parallel_indicators", [True, False])
    def test_weekly_interval_backtest(
        self, data_source_df, disable_parallel_indicators, scope
    ):
        def sma(bar_data, period):
            close = bar_data.close
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                out[i] = np.mean(close[i - period + 1 : i + 1])
            return out

        sma_ind = indicator("sma20", sma, period=5)
        seen = []

        def exec_fn(ctx):
            wk = ctx.interval("weekly")
            if len(wk.close) > 0:
                seen.append(
                    (len(wk.close), wk.close[-1], len(wk.indicator("sma20")))
                )

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[sma_ind],
            intervals=["weekly"],
        )
        strategy.walkforward(
            windows=1,
            disable_parallel_indicators=disable_parallel_indicators,
            timeframe="1d",
        )
        assert seen
        for n_bars, _close, n_ind in seen:
            assert n_ind == n_bars

    def test_undeclared_interval_then_error(self, data_source_df, scope):
        def exec_fn(ctx):
            ctx.interval("weekly")

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY")
        with pytest.raises(
            ValueError,
            match=re.escape("add_execution(..., intervals=[...])"),
        ):
            strategy.walkforward(windows=1, timeframe="1d")

    def test_interval_indicator_not_scheduled_then_error(
        self, data_source_df, scope
    ):
        def exec_fn(ctx):
            ctx.interval("weekly").indicator("sma20")

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY", intervals=["weekly"])
        with pytest.raises(
            ValueError, match="Indicator 'sma20@weekly' not found"
        ):
            strategy.walkforward(windows=1, timeframe="1d")

    @pytest.mark.parametrize("enable_parallel_models", [True, False])
    def test_weekly_interval_model_walkforward(
        self, data_source_df, scope, enable_parallel_models
    ):
        def sma(bar_data, period):
            close = bar_data.close
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                out[i] = np.mean(close[i - period + 1 : i + 1])
            return out

        sma_ind = indicator("sma20", sma, period=5)

        def train_fn(sym, train_data, test_data):
            return FakeModel(sym, np.zeros(len(test_data)))

        wk_model = model(
            "wk_model",
            train_fn,
            [sma_ind],
            predict_fn=lambda _model, df: np.zeros(len(df)),
        )
        seen = []

        def exec_fn(ctx):
            wk = ctx.interval("weekly")
            preds = wk.preds("wk_model")
            if len(preds) > 0:
                seen.append(
                    (len(wk.close), len(preds), len(wk.indicator("sma20")))
                )

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            models=[wk_model],
            intervals=["weekly"],
        )
        strategy.walkforward(
            windows=1,
            enable_parallel_models=enable_parallel_models,
            timeframe="1d",
        )
        assert seen
        for n_bars, n_preds, n_ind in seen:
            assert n_preds == n_bars
            assert n_ind == n_bars

    def test_invalid_interval_granularity_raises(self, data_source_df, scope):
        # The base spacing is only known once the backtest runs, so
        # granularity is validated there rather than at declaration.
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, "SPY", intervals=["daily"])
        with pytest.raises(ValueError, match="Cannot compress daily bars"):
            strategy.walkforward(windows=1, timeframe="1d")

    @pytest.mark.parametrize(
        "interval,match",
        [
            ("daily", "Cannot compress daily bars"),
            ("1m", "Cannot compress daily bars"),
            ("1h", "Cannot compress daily bars"),
        ],
    )
    def test_invalid_interval_granularity_parametrize(
        self, data_source_df, scope, interval, match
    ):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, "SPY", intervals=[interval])
        with pytest.raises(ValueError, match=match):
            strategy.walkforward(windows=1, timeframe="1d")

    @pytest.mark.parametrize("interval", ["weekly", 5])
    def test_valid_interval_granularity_walkforward(
        self, data_source_df, scope, interval
    ):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, "SPY", intervals=[interval])
        strategy.walkforward(windows=1, timeframe="1d")

    def test_valid_subdaily_interval_walkforward(self, minute_bars_df, scope):
        def sma(bar_data, period):
            close = bar_data.close
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                out[i] = np.mean(close[i - period + 1 : i + 1])
            return out

        sma_ind = indicator("sma20", sma, period=5)
        seen = []

        def exec_fn(ctx):
            tf_ctx = ctx.interval("5m")
            if len(tf_ctx.close) > 0:
                seen.append(len(tf_ctx.close))

        strategy = Strategy(
            minute_bars_df,
            minute_bars_df["date"].min(),
            minute_bars_df["date"].max(),
        )
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[sma_ind],
            intervals=["5m"],
        )
        strategy.walkforward(windows=1, timeframe="1m")
        assert seen

    @staticmethod
    def _sma_indicator():
        def sma(bar_data, period):
            close = bar_data.close
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                out[i] = np.mean(close[i - period + 1 : i + 1])
            return out

        return indicator("sma20", sma, period=5)

    @pytest.mark.parametrize("interval", ["weekly", "monthly", 5])
    def test_interval_indicator_walkforward(
        self, data_source_df, scope, interval
    ):
        sma_ind = self._sma_indicator()
        seen = []

        def exec_fn(ctx):
            interval_ctx = ctx.interval(interval)
            if len(interval_ctx.close) > 0:
                seen.append(
                    (
                        len(interval_ctx.close),
                        len(interval_ctx.indicator("sma20")),
                    )
                )

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[sma_ind],
            intervals=[interval],
        )
        strategy.walkforward(windows=1, timeframe="1d")
        assert seen
        for n_bars, n_ind in seen:
            assert n_ind == n_bars

    @pytest.mark.parametrize("interval", ["weekly", 5])
    @pytest.mark.parametrize("enable_parallel_models", [True, False])
    def test_interval_model_walkforward(
        self, data_source_df, scope, interval, enable_parallel_models
    ):
        sma_ind = self._sma_indicator()

        def train_fn(sym, train_data, test_data):
            return FakeModel(sym, np.zeros(len(test_data)))

        wk_model = model(
            "wk_model",
            train_fn,
            [sma_ind],
            predict_fn=lambda _model, df: np.zeros(len(df)),
        )
        seen = []

        def exec_fn(ctx):
            interval_ctx = ctx.interval(interval)
            preds = interval_ctx.preds("wk_model")
            if len(preds) > 0:
                seen.append((len(interval_ctx.close), len(preds)))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            models=[wk_model],
            intervals=[interval],
        )
        strategy.walkforward(
            windows=1,
            enable_parallel_models=enable_parallel_models,
            timeframe="1d",
        )
        assert seen
        for n_bars, n_preds in seen:
            assert n_preds == n_bars

    def test_interval_indicator_and_model_same_token(
        self, data_source_df, scope
    ):
        sma_ind = self._sma_indicator()

        def train_fn(sym, train_data, test_data):
            return FakeModel(sym, np.zeros(len(test_data)))

        wk_model = model(
            "wk_model",
            train_fn,
            [sma_ind],
            predict_fn=lambda _model, df: np.zeros(len(df)),
        )
        seen = []

        def exec_fn(ctx):
            interval_ctx = ctx.interval("weekly")
            preds = interval_ctx.preds("wk_model")
            if len(preds) > 0:
                seen.append(
                    (
                        len(interval_ctx.close),
                        len(preds),
                        len(interval_ctx.indicator("sma20")),
                    )
                )

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[sma_ind],
            models=[wk_model],
            intervals=["weekly"],
        )
        strategy.walkforward(windows=1, timeframe="1d")
        assert seen
        for n_bars, n_preds, n_ind in seen:
            assert n_preds == n_bars
            assert n_ind == n_bars

    def test_multiple_declared_intervals(self, data_source_df, scope):
        seen = []

        def exec_fn(ctx):
            weekly = ctx.interval("weekly")
            every5 = ctx.interval(5)
            if len(weekly.close) > 0 and len(every5.close) > 0:
                seen.append((len(weekly.close), len(every5.close)))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, "SPY", intervals=["weekly", 5])
        strategy.walkforward(windows=1, timeframe="1d")
        assert seen
        for n_weekly, n_every5 in seen:
            assert n_weekly > 0
            assert n_every5 > 0
            assert n_weekly != n_every5

    def test_per_bar_model_predicts_one_row_per_compressed_bar(
        self, data_source_df, scope
    ):
        # per_bar predictions are counted in compressed bars, so they must not
        # be routed through the base-bar `completed` map.
        sma_ind = indicator("sma2", _sma, period=2)

        class _Fake:
            def predict(self, X):
                return np.full(len(X), float(len(X)))

        per_bar_model = model(
            "per_bar_m",
            lambda sym, train_data, test_data: _Fake(),
            [sma_ind],
            per_bar=True,
            predict_fn=lambda m, d: m.predict(d),
        )
        seen = []

        def exec_fn(ctx):
            preds = ctx.interval("weekly").preds("per_bar_m")
            if len(preds):
                seen.append(list(preds))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            models=[per_bar_model],
            intervals=["weekly"],
        )
        strategy.walkforward(windows=1, train_size=0.9, timeframe="1d")
        assert seen
        last = seen[-1]
        assert last == [float(i + 1) for i in range(len(last))]

    def test_interval_preds_do_not_see_future_windows(
        self, data_source_df, scope
    ):
        # A cross-row transform must not reach compressed bars belonging to a
        # later walkforward window.
        sma_ind = indicator("sma2", _sma, period=2)

        class _MaxClose:
            def predict(self, X):
                return np.full(len(X), float(np.asarray(X["close"]).max()))

        def make(name):
            return model(
                name,
                lambda sym, train_data, test_data: _MaxClose(),
                [sma_ind],
                predict_fn=lambda m, d: m.predict(d),
            )

        tf_model, base_model = make("tf_max"), make("base_max")
        seen = []

        def exec_fn(ctx):
            tf_preds = ctx.interval("weekly").preds("tf_max")
            base_preds = ctx.preds("base_max")
            if len(tf_preds) and len(base_preds):
                seen.append((tf_preds[-1], base_preds[-1]))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            models=[tf_model, base_model],
            intervals=["weekly"],
        )
        strategy.walkforward(windows=3, train_size=0.5, timeframe="1d")
        assert seen
        for tf_max, base_max in seen:
            assert tf_max <= base_max

    def test_pretrained_model_allowed_with_intervals(
        self, data_source_df, scope
    ):
        # Pretrained models stay bound to the base timeframe; declaring an
        # interval must not try to train them per interval.
        sma_ind = indicator("sma2", _sma, period=2)

        class _Zeros:
            def predict(self, X):
                return np.zeros(len(X))

        pretrained = model(
            "pre",
            lambda sym, *args, **kwargs: _Zeros(),
            [sma_ind],
            pretrained=True,
            predict_fn=lambda m, d: m.predict(d),
        )
        seen = []

        def exec_fn(ctx):
            seen.append(len(ctx.preds("pre")))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            models=[pretrained],
            intervals=["weekly"],
        )
        strategy.walkforward(windows=1, timeframe="1d")
        assert seen
        assert all(n > 0 for n in seen)

    def test_pretrained_model_on_interval_then_error(
        self, data_source_df, scope
    ):
        sma_ind = indicator("sma2", _sma, period=2)

        class _Zeros:
            def predict(self, X):
                return np.zeros(len(X))

        pretrained = model(
            "pre",
            lambda sym, *args, **kwargs: _Zeros(),
            [sma_ind],
            pretrained=True,
            predict_fn=lambda m, d: m.predict(d),
        )
        errors = []

        def exec_fn(ctx):
            try:
                ctx.interval("weekly").preds("pre")
            except ValueError as e:
                errors.append(str(e))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            models=[pretrained],
            intervals=["weekly"],
        )
        strategy.walkforward(windows=1, timeframe="1d")
        assert errors
        assert "not trained per interval" in errors[0]

    def test_interval_indicator_not_readable_from_base_ctx(
        self, data_source_df, scope
    ):
        sma_ind = indicator("sma2", _sma, period=2)
        errors = []

        def exec_fn(ctx):
            try:
                ctx.indicator("sma2@weekly")
            except ValueError as e:
                errors.append(str(e))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[sma_ind],
            intervals=["weekly"],
        )
        strategy.walkforward(windows=1, timeframe="1d")
        assert errors
        assert "ctx.interval('weekly').indicator('sma2')" in errors[0]

    def test_interval_model_predicts_once_per_window(
        self, data_source_df, scope
    ):
        # Compressed data is immutable per window, so predictions are computed
        # once -- not rebuilt on every base bar.
        sma_ind = indicator("sma2", _sma, period=2)
        calls = []

        class _Counting:
            def predict(self, X):
                calls.append(len(X))
                return np.zeros(len(X))

        counted = model(
            "counted",
            lambda sym, train_data, test_data: _Counting(),
            [sma_ind],
            predict_fn=lambda m, d: m.predict(d),
        )

        def exec_fn(ctx):
            ctx.interval("weekly").preds("counted")

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            models=[counted],
            intervals=["weekly"],
        )
        strategy.walkforward(windows=1, timeframe="1d")
        assert len(calls) == 1

    def test_vwap_indicator_with_declared_interval(
        self, data_source_df, scope
    ):
        # Declaring an interval must not strip vwap from compressed bars.
        df = data_source_df.copy()
        df["vwap"] = df["close"] + 0.5
        vwap_ind = indicator("vwap2", lambda bar_data: bar_data.vwap * 2.0)
        seen = []

        def exec_fn(ctx):
            values = ctx.interval("weekly").indicator("vwap2")
            if len(values):
                seen.append(values[-1])

        strategy = Strategy(df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn,
            "SPY",
            indicators=[vwap_ind],
            intervals=["weekly"],
        )
        strategy.walkforward(windows=1, timeframe="1d")
        assert seen
        assert all(np.isfinite(v) for v in seen)

    def test_symbol_with_single_bar_does_not_abort(self, scope):
        # A newly listed or halted symbol has too few bars to contradict the
        # declared base spacing, so it must not kill the whole run.
        dates = pd.date_range(START_DATE, periods=40, freq="B")
        close = np.arange(1.0, len(dates) + 1)
        full = pd.DataFrame(
            {
                "symbol": "AAA",
                "date": dates,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(len(dates)),
            }
        )
        thin = full.iloc[[10]].copy()
        thin["symbol"] = "BBB"
        df = pd.concat([full, thin]).reset_index(drop=True)
        seen = set()

        def exec_fn(ctx):
            seen.add(ctx.symbol)

        strategy = Strategy(df, dates[0], dates[-1])
        strategy.add_execution(exec_fn, ["AAA", "BBB"], intervals=["weekly"])
        strategy.walkforward(windows=1, timeframe="1d")
        assert "AAA" in seen

    def test_days_filter_with_declared_interval(self, scope):
        # `days=` thins the frame on purpose; the resulting weekly gaps are
        # multiples of the base spacing and must still validate.
        dates = pd.date_range(START_DATE, periods=60, freq="B")
        close = np.arange(1.0, len(dates) + 1)
        df = pd.DataFrame(
            {
                "symbol": "AAA",
                "date": dates,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(len(dates)),
            }
        )
        seen = []

        def exec_fn(ctx):
            seen.append(ctx.dt)

        strategy = Strategy(df, dates[0], dates[-1])
        strategy.add_execution(exec_fn, "AAA", intervals=["monthly"])
        strategy.walkforward(windows=1, days="mon", timeframe="1d")
        assert seen

    def test_intervals_without_base_timeframe_then_error(
        self, data_source_df, scope
    ):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, "SPY", intervals=["weekly"])
        with pytest.raises(ValueError, match="pass timeframe="):
            strategy.walkforward(windows=1)

    def test_interval_scoped_to_declaring_execution(
        self, data_source_df, scope
    ):
        # Intervals belong to the execution that declared them, so a sibling
        # execution cannot read them.
        declared, errors = [], []

        def spy_fn(ctx):
            declared.append(len(ctx.interval("weekly").close))

        def aapl_fn(ctx):
            try:
                ctx.interval("weekly")
            except ValueError as e:
                errors.append(str(e))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(spy_fn, "SPY", intervals=["weekly"])
        strategy.add_execution(aapl_fn, "AAPL")
        strategy.walkforward(windows=1, timeframe="1d")
        assert declared
        assert errors
        assert "was not declared for this execution" in errors[0]

    def test_intervals_not_shared_across_executions(
        self, data_source_df, scope
    ):
        # Two executions declaring different intervals each see only their own.
        errors = []

        def spy_fn(ctx):
            ctx.interval("weekly")
            try:
                ctx.interval(5)
            except ValueError as e:
                errors.append(("SPY", str(e)))

        def aapl_fn(ctx):
            ctx.interval(5)
            try:
                ctx.interval("weekly")
            except ValueError as e:
                errors.append(("AAPL", str(e)))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(spy_fn, "SPY", intervals=["weekly"])
        strategy.add_execution(aapl_fn, "AAPL", intervals=[5])
        strategy.walkforward(windows=1, timeframe="1d")
        assert {sym for sym, _ in errors} == {"SPY", "AAPL"}

    def test_only_declaring_symbols_are_compressed(
        self, data_source_df, scope
    ):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, "SPY", intervals=["weekly"])
        strategy.add_execution(lambda ctx: None, "AAPL")
        df = strategy._fetch_data("1d", None)
        interval_data = strategy._build_interval_data(df, "1d")
        assert set(interval_data.compressed) == {("SPY", "weekly")}

    def test_no_intervals_declared_compresses_nothing(
        self, data_source_df, scope
    ):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, "SPY")
        df = strategy._fetch_data("1d", None)
        # No intervals means no base spacing is needed at all.
        assert not strategy._build_interval_data(df, "").compressed

    def test_selector_execution_compresses_whole_frame(
        self, data_source_df, scope
    ):
        # A selector resolves its symbols per window, after compression has
        # already run, so every candidate symbol must be compressed.
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            lambda ctx: None,
            lambda df: ["SPY"],
            intervals=["weekly"],
        )
        df = strategy._fetch_data("1d", None)
        interval_data = strategy._build_interval_data(df, "1d")
        assert set(interval_data.compressed) == {
            (sym, "weekly") for sym in df["symbol"].unique()
        }

    def test_selector_execution_reads_declared_interval(
        self, data_source_df, scope
    ):
        seen = []

        def exec_fn(ctx):
            seen.append(len(ctx.interval("weekly").close))

        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(
            exec_fn, lambda df: ["SPY"], intervals=["weekly"]
        )
        strategy.walkforward(windows=2, train_size=0.5, timeframe="1d")
        assert seen

    def test_duplicate_intervals_then_error(self, data_source_df, scope):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        with pytest.raises(ValueError, match="Duplicate interval"):
            strategy.add_execution(
                lambda ctx: None, "SPY", intervals=["weekly", "weekly"]
            )

    def test_empty_intervals_then_error(self, data_source_df, scope):
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        with pytest.raises(ValueError, match="intervals cannot be empty"):
            strategy.add_execution(lambda ctx: None, "SPY", intervals=[])

    @pytest.mark.parametrize(
        "intervals,expected",
        [
            ("weekly", frozenset({"weekly"})),
            (5, frozenset({5})),
            (["weekly", 5], frozenset({"weekly", 5})),
            (None, frozenset()),
        ],
    )
    def test_scalar_intervals_are_not_iterated(
        self, data_source_df, scope, intervals, expected
    ):
        # str is Iterable, so 'weekly' must not split into characters.
        strategy = Strategy(data_source_df, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, "SPY", intervals=intervals)
        execution = next(iter(strategy._executions))
        assert execution.intervals == expected
