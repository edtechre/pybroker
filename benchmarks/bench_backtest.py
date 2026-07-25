"""Reference backtest benchmark for pybroker.

Pinned scenario: 4 symbols x 2 years daily (2020 rows) x 3 walkforward windows,
hhv20/llv20 indicator crossover strategy with stops. The dataset is the same
checked-in fixture used by the test suite (``tests/testdata/daily_1.pkl``).

``Walkforward`` runs a setup warmup so the timed method measures steady-state
bar-by-bar execution. ``WalkforwardCold`` omits the warmup so the first
invocation pays Numba JIT compile cost — useful for tracking the benefit of
``@njit(cache=True)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pybroker
from pybroker import ExecContext, Strategy, StrategyConfig
from pybroker.strategy import BacktestMixin, Execution
from pybroker.vect import highv, lowv
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "tests" / "testdata" / "daily_1.pkl"
SEED = 42
WINDOWS = 3
LOOKAHEAD = 1


def _load_dataset() -> pd.DataFrame:
    # Trusted pinned fixture checked into the repo; same file used by the
    # test suite via tests/fixtures.py.
    df = pd.read_pickle(DATA_PATH)  # noqa: S301 - trusted test fixture
    df["date"] = pd.to_datetime(df["date"])
    return df


def _build_strategy(df: pd.DataFrame) -> Strategy:
    pybroker.clear_params()
    pybroker.disable_logging()
    pybroker.disable_progress_bar()

    hhv20 = pybroker.indicator(
        "hhv20", lambda bar_data: highv(bar_data.close, 20)
    )
    llv20 = pybroker.indicator(
        "llv20", lambda bar_data: lowv(bar_data.close, 20)
    )

    def exec_fn(ctx: ExecContext) -> None:
        hhv = ctx.indicator("hhv20")
        llv = ctx.indicator("llv20")
        if len(hhv) < 21:
            return
        close = ctx.close[-1]
        prev_hhv = hhv[-2]
        prev_llv = llv[-2]
        if not ctx.long_pos() and close > prev_hhv:
            ctx.buy_shares = 100
            ctx.stop_loss_pct = 5
            ctx.stop_profit_pct = 15
        elif ctx.long_pos() and close < prev_llv:
            ctx.sell_all_shares()

    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    config = StrategyConfig(
        bootstrap_samples=100,
    )
    strategy = Strategy(df, start, end, config)
    strategy.add_execution(
        exec_fn,
        symbols=sorted(df["symbol"].unique().tolist()),
        indicators=[hhv20, llv20],
    )
    return strategy


class Walkforward:
    """Warm steady-state walkforward on the pinned reference scenario."""

    timeout = 300

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _load_dataset()
        _build_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=True,
            disable_parallel_indicators=True,
        )

    def time_walkforward(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=True,
            disable_parallel_indicators=True,
        )

    def peakmem_walkforward(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=True,
            disable_parallel_indicators=True,
        )


class WalkforwardCold:
    """Cold-start walkforward without JIT warmup in setup."""

    timeout = 600

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _load_dataset()

    def time_cold_walkforward(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=True,
            disable_parallel_indicators=True,
        )


def _build_held_stops_strategy(df: pd.DataFrame) -> Strategy:
    """No indicators; buy once per symbol with stops, then hold."""
    pybroker.clear_params()
    pybroker.disable_logging()
    pybroker.disable_progress_bar()

    def exec_fn(ctx: ExecContext) -> None:
        if not ctx.long_pos():
            ctx.buy_shares = 100
            ctx.stop_loss_pct = 5
            ctx.stop_profit_pct = 15

    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    strategy = Strategy(df, start, end, StrategyConfig())
    strategy.add_execution(
        exec_fn,
        symbols=sorted(df["symbol"].unique().tolist()),
    )
    return strategy


class PortfolioHeldStops:
    """Walkforward stressing portfolio check_stops + capture_bar with held stops."""

    timeout = 300

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _load_dataset()
        _build_held_stops_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def time_portfolio_held_stops(self) -> None:
        _build_held_stops_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )


# ---------------------------------------------------------------------------
# Group A — scenario scaling (asv parametrization + large fixture)
# ---------------------------------------------------------------------------


def _synthetic_ohlcv(n_symbols: int, n_days: int, seed: int) -> pd.DataFrame:
    """Build a deterministic OHLCV fixture large enough to stress scaling."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-02", periods=n_days, freq="B")
    rows = []
    for sym_idx in range(n_symbols):
        sym = f"SYM{sym_idx:03d}"
        # Deterministic random walk per symbol so the data exercises
        # indicator crossovers without being pathological.
        price = 100.0 + rng.standard_normal(n_days).cumsum() * 0.5
        high = price + rng.uniform(0.1, 1.0, n_days)
        low = price - rng.uniform(0.1, 1.0, n_days)
        open_ = price + rng.uniform(-0.5, 0.5, n_days)
        volume = rng.integers(1_000, 100_000, n_days).astype(float)
        for i, d in enumerate(dates):
            rows.append(
                (
                    d,
                    open_[i],
                    high[i],
                    low[i],
                    price[i],
                    volume[i],
                    price[i],
                    sym,
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
            "symbol",
        ],
    )


class WalkforwardScaled:
    """Walkforward parametrized on (windows, symbols) so scaling bugs surface.

    asv reports one result per (windows, symbols) combination. A regression
    that's O(n) on windows but not on symbols (or vice versa) shows up
    immediately.
    """

    params = ([1, 3, 10], [4, 12])
    param_names = ("windows", "symbols")
    timeout = 900

    def setup(self, windows: int, symbols: int) -> None:
        np.random.seed(SEED)
        if symbols == 4:
            self.df = _load_dataset()
        else:
            self.df = _synthetic_ohlcv(
                n_symbols=symbols, n_days=252 * 2, seed=SEED
            )
        # Warmup so Numba compile is amortized before the timed call.
        _build_strategy(self.df).walkforward(
            windows=windows,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def time_walkforward_scaled(self, windows: int, symbols: int) -> None:
        _build_strategy(self.df).walkforward(
            windows=windows,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )


class WalkforwardLarge:
    """Larger scenario (10 symbols x 5y daily x 5 windows) on synthetic data.

    Surfaces O(n) patterns the 4-symbol fixture hides. Kept smaller than the
    obvious "20 symbols x 10y" shape so CI time stays bounded.
    """

    timeout = 900

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _synthetic_ohlcv(n_symbols=10, n_days=252 * 5, seed=SEED)
        _build_strategy(self.df).walkforward(
            windows=5,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def time_walkforward_large(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=5,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def peakmem_walkforward_large(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=5,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )


def _noop_exec_fn(_ctx: ExecContext) -> None:
    pass


def _build_noop_strategy(df: pd.DataFrame) -> Strategy:
    pybroker.clear_params()
    pybroker.disable_logging()
    pybroker.disable_progress_bar()
    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    strategy = Strategy(df, start, end, StrategyConfig())
    strategy.add_execution(
        _noop_exec_fn,
        symbols=sorted(df["symbol"].unique().tolist()),
    )
    return strategy


class WalkforwardMultiWindow:
    """Walkforward with many windows stressing per-window store slicing."""

    timeout = 900

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _synthetic_ohlcv(n_symbols=10, n_days=252 * 5, seed=SEED)
        _build_noop_strategy(self.df).walkforward(
            windows=10,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def time_walkforward_multi_window(self) -> None:
        _build_noop_strategy(self.df).walkforward(
            windows=10,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )


class WalkforwardPerBar:
    """Walkforward with per-bar model predictions."""

    timeout = 600

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _load_dataset()
        self._build_strategy().walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def time_walkforward_per_bar(self) -> None:
        self._build_strategy().walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def _build_strategy(self) -> Strategy:
        from pybroker.model import model

        pybroker.clear_params()
        pybroker.disable_logging()
        pybroker.disable_progress_bar()

        class _ConstantModel:
            def predict(self, _x):
                return np.ones(len(_x))

        hhv20 = pybroker.indicator(
            "hhv20", lambda bar_data: highv(bar_data.close, 20)
        )

        def train_fn(_symbol, _train_df, _test_df):
            return _ConstantModel()

        def predict_fn(_instance, df):
            return np.ones(len(df))

        per_bar_model = model(
            "bench_per_bar",
            train_fn,
            indicators=(hhv20,),
            predict_fn=predict_fn,
            per_bar=True,
        )

        def exec_fn(ctx: ExecContext) -> None:
            preds = ctx.preds("bench_per_bar")
            if len(preds) and preds[-1] > 0:
                ctx.buy_shares = 1

        start = self.df["date"].min().strftime("%Y-%m-%d")
        end = self.df["date"].max().strftime("%Y-%m-%d")
        strategy = Strategy(self.df, start, end, StrategyConfig())
        strategy.add_execution(
            exec_fn,
            symbols=sorted(self.df["symbol"].unique().tolist()),
            models=per_bar_model,
            indicators=[hhv20],
        )
        return strategy


class BacktestBarLoop:
    """Isolates backtest_executions bar-loop overhead (date map, kwargs)."""

    timeout = 600

    def setup(self) -> None:
        np.random.seed(SEED)
        pybroker.clear_params()
        pybroker.disable_logging()
        pybroker.disable_progress_bar()
        self.df = _synthetic_ohlcv(n_symbols=50, n_days=252 * 2, seed=SEED)
        self._run_once()

    def time_backtest_bar_loop(self) -> None:
        self._run_once()

    def _run_once(self) -> None:
        from pybroker.portfolio import Portfolio

        df = self.df
        symbols = sorted(df["symbol"].unique().tolist())
        execution = Execution(
            id=1,
            symbols=frozenset(symbols),
            fn=_noop_exec_fn,
            model_names=frozenset(),
            indicator_names=frozenset(),
        )
        portfolio = Portfolio(
            cash=100_000,
            record_portfolio_bars=False,
            record_position_bars=False,
        )
        BacktestMixin().backtest_executions(
            config=StrategyConfig(),
            executions={execution},
            before_exec_fn=None,
            after_exec_fn=None,
            sessions=defaultdict(dict),
            models={},
            indicator_data={},
            test_data=df,
            portfolio=portfolio,
            exit_dates={},
        )


class _PreprocessOnlyStrategy(Strategy):
    """Strategy whose ``walkforward_split`` yields zero windows.

    Used by ``ExitOnLastBarPreprocess`` to isolate the
    ``exit_on_last_bar`` preprocessing block in
    :py:meth:`Strategy._run_walkforward`: the preprocessing fires before
    the windows loop, and the loop becomes a no-op so the timed call
    doesn't pay for trade simulation.
    """

    def walkforward_split(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        windows: int,
        lookahead: int,
        train_size: float = 0.9,
        shuffle: bool = False,
    ):
        return iter(())


def _exit_on_last_bar_noop_fn(_ctx: ExecContext) -> None:
    pass


class ExitOnLastBarPreprocess:
    """Bench the ``exit_on_last_bar`` preprocessing block in
    ``Strategy._run_walkforward``.

    When ``exit_on_last_bar=True``, ``_run_walkforward`` builds a
    ``{symbol: max_date}`` map before the windows loop fires. A naive
    ``O(E*S*N)`` boolean-mask-per-symbol implementation can be replaced
    by a single ``isin`` + ``groupby(symbol).max()``; this bench
    quantifies the difference and catches future regressions.

    Isolation strategy: ``_PreprocessOnlyStrategy.walkforward_split``
    yields nothing, so the timed call exercises ``_filter_dates``,
    indicator setup (empty here), and the preprocessing block, then
    short-circuits before the windows loop.

    Parametrized on ``n_symbols`` so the linear vs roughly quadratic
    scaling shape is visible per-row in the asv-pr output.
    """

    timeout = 300
    params = [50, 200, 500]
    param_names = ("n_symbols",)

    def setup(self, n_symbols: int) -> None:
        np.random.seed(SEED)
        self.df = _synthetic_ohlcv(n_symbols=n_symbols, n_days=252, seed=SEED)
        self._build_strategy().walkforward(
            windows=1,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def time_exit_on_last_bar_preprocess(self, n_symbols: int) -> None:
        self._build_strategy().walkforward(
            windows=1,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def _build_strategy(self) -> "_PreprocessOnlyStrategy":
        pybroker.clear_params()
        pybroker.disable_logging()
        pybroker.disable_progress_bar()
        symbols = sorted(self.df["symbol"].unique().tolist())
        start = self.df["date"].min().strftime("%Y-%m-%d")
        end = self.df["date"].max().strftime("%Y-%m-%d")
        config = StrategyConfig(exit_on_last_bar=True)
        strategy = _PreprocessOnlyStrategy(self.df, start, end, config)
        strategy.add_execution(_exit_on_last_bar_noop_fn, symbols=symbols)
        return strategy


# ---------------------------------------------------------------------------
# Group B — subsystem micro-benches (no vector dependency)
# ---------------------------------------------------------------------------


class IndicatorPreprocess:
    """Bench :py:meth:`pybroker.indicator.IndicatorSet.__call__` on a
    multi-symbol DataFrame.

    Guards two per-symbol mask loops in ``indicator.py``:
    ``IndicatorsMixin.compute_indicators`` slices ``df[df[symbol] ==
    sym]`` once per uncached ``IndicatorSymbol`` and
    ``IndicatorSet.__call__`` repeats the same boolean mask while
    rebuilding the flat output frame. Both are ``O(S * N)`` per call
    and collapse to a single ``groupby(symbol)`` pass.

    Isolation: calling ``IndicatorSet(...)(df)`` directly exercises
    both loops without any ``Strategy`` shim. Parametrized on
    ``(n_symbols, n_indicators)`` so the linear-in-S scan cost and
    the I-amplified rebuild cost both have a row that pins them.
    """

    timeout = 300
    params = ([50, 200, 500], [1, 5])
    param_names = ("n_symbols", "n_indicators")

    def setup(self, n_symbols: int, n_indicators: int) -> None:
        from pybroker.indicator import IndicatorSet
        from pybroker.vect import highv, lowv, sumv

        np.random.seed(SEED)
        pybroker.clear_params()
        pybroker.disable_logging()
        pybroker.disable_progress_bar()
        self.df = _synthetic_ohlcv(n_symbols=n_symbols, n_days=252, seed=SEED)
        kernels = (
            ("hhv20", lambda b: highv(b.close, 20)),
            ("llv20", lambda b: lowv(b.close, 20)),
            ("sum20", lambda b: sumv(b.close, 20)),
            ("hhv50", lambda b: highv(b.close, 50)),
            ("llv50", lambda b: lowv(b.close, 50)),
        )
        ind_set = IndicatorSet()
        for name, fn in kernels[:n_indicators]:
            ind_set.add(pybroker.indicator(name, fn))
        self._ind_set = ind_set
        self._ind_set(self.df, disable_parallel_indicators=True)

    def time_indicator_set_call(
        self, n_symbols: int, n_indicators: int
    ) -> None:
        self._ind_set(self.df, disable_parallel_indicators=True)


class IndicatorKernels:
    """Direct calls into vect.py @njit kernels.

    Measures the fundamental rolling indicator primitives independent of
    the Strategy pipeline. Regressions in these kernels ripple through
    every user indicator, so they deserve a standalone bench.
    """

    timeout = 60
    params = [1_000, 100_000]
    param_names = ("length",)

    def setup(self, length: int) -> None:
        from pybroker.vect import highv, lowv, sumv, returnv, cross

        rng = np.random.default_rng(SEED)
        self._arr = rng.standard_normal(length)
        self._arr2 = rng.standard_normal(length)
        self._highv = highv
        self._lowv = lowv
        self._sumv = sumv
        self._returnv = returnv
        self._cross = cross
        # Warmup compile for each kernel so timings reflect steady state.
        self._highv(self._arr, 20)
        self._lowv(self._arr, 20)
        self._sumv(self._arr, 20)
        self._returnv(self._arr, 1)
        self._cross(self._arr, self._arr2)

    def time_highv(self, length: int) -> None:
        self._highv(self._arr, 20)

    def time_lowv(self, length: int) -> None:
        self._lowv(self._arr, 20)

    def time_sumv(self, length: int) -> None:
        self._sumv(self._arr, 20)

    def time_returnv(self, length: int) -> None:
        self._returnv(self._arr, 1)

    def time_cross(self, length: int) -> None:
        self._cross(self._arr, self._arr2)


class EvalKernels:
    """Direct calls into eval.py @njit metric kernels.

    BCa bootstrap dominates walkforward's ``calc_bootstrap=True`` path;
    max_drawdown / sharpe / profit_factor feed evaluate() on every window.
    Tracking these separately catches regressions that the macro bench
    would average away.
    """

    timeout = 60

    def setup(self) -> None:
        from pybroker.eval import (
            bca_boot_conf,
            drawdown_conf,
            max_drawdown,
            profit_factor,
            sharpe_ratio,
        )

        rng = np.random.default_rng(SEED)
        self._changes = rng.normal(0.0, 1.0, 5_000)
        self._returns = rng.normal(0.0005, 0.01, 5_000)
        self._bca_boot_conf = bca_boot_conf
        self._drawdown_conf = drawdown_conf
        self._max_drawdown = max_drawdown
        self._sharpe_ratio = sharpe_ratio
        self._profit_factor = profit_factor
        # Warmup JIT compile.
        self._max_drawdown(self._changes)
        self._sharpe_ratio(self._returns)
        self._profit_factor(self._changes)
        self._bca_boot_conf(self._changes, 200, self._sharpe_ratio)
        self._drawdown_conf(self._changes, self._returns, 200)

    def time_max_drawdown(self) -> None:
        self._max_drawdown(self._changes)

    def time_sharpe_ratio(self) -> None:
        self._sharpe_ratio(self._returns)

    def time_profit_factor(self) -> None:
        self._profit_factor(self._changes)

    def time_bca_boot_conf(self) -> None:
        self._bca_boot_conf(self._changes, 200, self._sharpe_ratio)

    def time_drawdown_conf(self) -> None:
        self._drawdown_conf(self._changes, self._returns, 200)


class EvalBootstrap:
    """End-to-end evaluate() path with calc_bootstrap=True."""

    timeout = 120

    def setup(self) -> None:
        import pandas as pd
        from pybroker.eval import EvaluateMixin

        testdata = REPO_ROOT / "tests" / "testdata"
        self._portfolio_df = pd.read_pickle(testdata / "portfolio_df.pkl")
        self._trades_df = pd.read_pickle(testdata / "trades_df.pkl")
        self._mixin = EvaluateMixin()
        np.random.seed(SEED)
        self._mixin.evaluate(
            self._portfolio_df,
            self._trades_df,
            calc_bootstrap=True,
            bootstrap_samples=200,
            bars_per_year=252,
            seed=SEED,
        )

    def time_evaluate_bootstrap(self) -> None:
        np.random.seed(SEED)
        self._mixin.evaluate(
            self._portfolio_df,
            self._trades_df,
            calc_bootstrap=True,
            bootstrap_samples=200,
            bars_per_year=252,
            seed=SEED,
        )


class CacheHit:
    """Direct diskcache.Cache.get benchmark on a populated cache.

    Measures the cost of the read path any indicator/model lookup pays.
    V5 (in-memory L1) shows ~100x here once merged.
    """

    timeout = 30

    def setup(self) -> None:
        import tempfile

        from diskcache import Cache

        self._cache_dir = tempfile.mkdtemp(prefix="asv_cache_hit_")
        self._cache = Cache(directory=self._cache_dir)
        rng = np.random.default_rng(SEED)
        self._key = "sym/hhv20/2020-2022"
        self._cache.set(self._key, rng.standard_normal(500))
        # Warmup read path.
        self._cache.get(self._key)
        self._cleanup_dir = self._cache_dir

    def teardown(self) -> None:
        import shutil

        self._cache.close()
        shutil.rmtree(self._cleanup_dir, ignore_errors=True)

    def time_cache_get(self) -> None:
        self._cache.get(self._key)


# ---------------------------------------------------------------------------
# Group F — Numba candidate micro benches (profiling only, not migration gates)
# ---------------------------------------------------------------------------


class TimeseriesKernels:
    """Lag prep kernels in timeseries.py."""

    timeout = 120
    params = ([1_000, 100_000], [1, 5])
    param_names = ("length", "lags")

    def setup(self, length: int, lags: int) -> None:
        from pybroker.timeseries import (
            build_lag_feature_matrix,
            build_lag_feature_matrix_pooled,
            compute_lag_series_cache,
            shift_array,
        )

        rng = np.random.default_rng(SEED)
        self._shift_array = shift_array
        self._build_lag_feature_matrix = build_lag_feature_matrix
        self._build_lag_feature_matrix_pooled = build_lag_feature_matrix_pooled
        self._lag = lags
        self._values = rng.standard_normal(length)
        df = _synthetic_ohlcv(
            n_symbols=4, n_days=max(length // 4, 50), seed=SEED
        )
        self._cache = compute_lag_series_cache(
            df, ("SYM000", "SYM001"), ("close",), lags
        )
        sym = "SYM000"
        sym_df = df[df["symbol"] == sym].drop(columns="symbol")
        self._history_dates = sym_df["date"].to_numpy(dtype="datetime64[ns]")
        self._row_dates = self._history_dates[
            -min(len(self._history_dates), 200) :
        ]
        row_count = len(self._row_dates)
        self._base_arrays = {
            "close": sym_df["close"].to_numpy(dtype=np.float64)[-row_count:],
        }
        self._sym_col = df["symbol"].to_numpy()
        self._all_dates = df["date"].to_numpy(dtype="datetime64[ns]")
        self._pooled_close = df["close"].to_numpy(dtype=np.float64)
        self._history_by_sym = {
            s: df.loc[df["symbol"] == s, "date"].to_numpy(
                dtype="datetime64[ns]"
            )
            for s in df["symbol"].unique()[:2]
        }
        # Warmup (includes Numba compile when enabled).
        self._shift_array(self._values, lags)
        self._build_lag_feature_matrix(
            sym,
            ("close",),
            lags,
            self._base_arrays,
            self._row_dates,
            self._history_dates,
            self._cache,
        )
        self._build_lag_feature_matrix_pooled(
            self._sym_col,
            ("close",),
            lags,
            {"close": self._pooled_close},
            self._all_dates,
            self._history_by_sym,
            self._cache,
            ("SYM000", "SYM001"),
        )

    def time_shift_array(self, length: int, lags: int) -> None:
        self._shift_array(self._values, lags)

    def time_build_lag_feature_matrix(self, length: int, lags: int) -> None:
        self._build_lag_feature_matrix(
            "SYM000",
            ("close",),
            lags,
            self._base_arrays,
            self._row_dates,
            self._history_dates,
            self._cache,
        )

    def time_build_lag_feature_matrix_pooled(
        self, length: int, lags: int
    ) -> None:
        self._build_lag_feature_matrix_pooled(
            self._sym_col,
            ("close",),
            lags,
            {"close": self._pooled_close},
            self._all_dates,
            self._history_by_sym,
            self._cache,
            ("SYM000", "SYM001"),
        )


class StoreBuildKernels:
    """SymbolArrayStore build from flat frames."""

    timeout = 120
    params = ([4, 10], [504, 1260])
    param_names = ("n_symbols", "n_days")

    def setup(self, n_symbols: int, n_days: int) -> None:
        from pybroker.scope import symbol_array_store_from_frame

        self._df = _synthetic_ohlcv(
            n_symbols=n_symbols, n_days=n_days, seed=SEED
        )
        self._build = symbol_array_store_from_frame
        self._build(self._df)

    def time_symbol_array_store_from_frame(
        self, n_symbols: int, n_days: int
    ) -> None:
        self._build(self._df)


class StoreSliceKernels:
    """Repeated date-window slicing of a master SymbolArrayStore."""

    timeout = 120
    params = ([10], [252 * 5])
    param_names = ("n_symbols", "n_days")

    def setup(self, n_symbols: int, n_days: int) -> None:
        from pybroker.common import get_unique_sorted_dates_array
        from pybroker.scope import (
            slice_symbol_array_store_by_dates,
            symbol_array_store_from_frame,
        )

        self._slice = slice_symbol_array_store_by_dates
        self._df = _synthetic_ohlcv(
            n_symbols=n_symbols, n_days=n_days, seed=SEED
        )
        self._master = symbol_array_store_from_frame(self._df)
        all_dates = get_unique_sorted_dates_array(
            self._df["date"].to_numpy(dtype="datetime64[ns]")
        )
        window_size = 252
        self._windows = tuple(
            all_dates[i * window_size : (i + 1) * window_size]
            for i in range(10)
            if (i + 1) * window_size <= len(all_dates)
        )
        for window in self._windows:
            self._slice(self._master, window)

    def time_slice_symbol_array_store_by_dates(
        self, n_symbols: int, n_days: int
    ) -> None:
        for window in self._windows:
            self._slice(self._master, window)


class ModelPrepKernels:
    """Model input indicator alignment."""

    timeout = 60

    def setup(self) -> None:
        from pybroker.model import _indicator_values_for_dates

        df = _load_dataset()
        sym = df["symbol"].iloc[0]
        sym_df = df[df["symbol"] == sym].sort_values("date")
        self._align = _indicator_values_for_dates
        ind = sym_df.set_index("date")["close"]
        self._ind = ind
        self._dates = sym_df["date"].to_numpy(dtype="datetime64[ns]")
        self._align(ind, self._dates)

    def time_indicator_values_for_dates(self) -> None:
        self._align(self._ind, self._dates)


# ---------------------------------------------------------------------------
# Group C — cold-start rigor (properly wipe JIT cache before timing)
# ---------------------------------------------------------------------------


class WalkforwardProperCold:
    """Walkforward that *really* starts cold: clears pybroker's __pycache__
    before every measured call so Numba has to recompile from source.

    This is the regression signal for V1 (``@njit(cache=True)``) — with
    V1 landed, the `.nbi` files in __pycache__ let Numba skip LLVM compile
    on subsequent processes; without V1 they don't exist and every cold
    process pays the full compile. The regular ``WalkforwardCold`` only
    skips setup warmup, which isn't a cold process.
    """

    timeout = 900

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _load_dataset()

    def setup_iteration(self, *_args) -> None:
        """Clear .nbi caches before every timed iteration."""
        for pkg_dir in (REPO_ROOT / "src" / "pybroker" / "__pycache__",):
            if pkg_dir.exists():
                # Remove only the Numba artifacts, not the .pyc cache
                # (removing all .pyc would also force Python reimport).
                for artifact in pkg_dir.glob("*.nbi"):
                    artifact.unlink(missing_ok=True)
                for artifact in pkg_dir.glob("*.nbc"):
                    artifact.unlink(missing_ok=True)

    def time_proper_cold_walkforward(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )


# ---------------------------------------------------------------------------
# Group D — invariants (not perf metrics; track numeric-stability / size)
# ---------------------------------------------------------------------------


class Determinism:
    """Tracks a hash of walkforward output so asv flags any unintended
    numeric divergence across commits. Perf vectors that claim to be
    semantics-preserving (V2, V7, V8, V9) must keep this hash stable.
    """

    timeout = 300

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _load_dataset()

    def track_walkforward_equity_hash(self) -> int:
        import hashlib

        result = _build_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )
        equity = result.portfolio["equity"].to_numpy()
        digest = hashlib.sha256(equity.tobytes()).hexdigest()
        # asv's track_* expects a number; use the first 8 hex chars as int.
        return int(digest[:8], 16)

    track_walkforward_equity_hash.unit = "hash"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Group E — v2 macro scenario benches (migration gates)
# ---------------------------------------------------------------------------


class WalkforwardTimeframes:
    """End-to-end walkforward with multi-timeframe compression enabled."""

    timeout = 600

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _synthetic_ohlcv(n_symbols=10, n_days=252 * 2, seed=SEED)
        self._build_strategy().walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def time_walkforward_timeframes(self) -> None:
        self._build_strategy().walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def _build_strategy(self) -> Strategy:
        pybroker.clear_params()
        pybroker.disable_logging()
        pybroker.disable_progress_bar()

        def exec_fn(ctx: ExecContext) -> None:
            weekly = ctx.timeframe("weekly")
            if len(weekly.close) > 0 and ctx.close[-1] > weekly.close[-1]:
                ctx.buy_shares = 10

        start = self.df["date"].min().strftime("%Y-%m-%d")
        end = self.df["date"].max().strftime("%Y-%m-%d")
        strategy = Strategy(self.df, start, end, StrategyConfig())
        strategy.enable_timeframes("weekly", base_timeframe="1d")
        strategy.add_execution(
            exec_fn, symbols=sorted(self.df["symbol"].unique().tolist())
        )
        return strategy


class WalkforwardModels:
    """End-to-end walkforward with pooled multi-symbol model training."""

    timeout = 600

    def setup(self) -> None:
        np.random.seed(SEED)
        self.df = _load_dataset()
        self._build_strategy().walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def time_walkforward_models(self) -> None:
        self._build_strategy().walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel_indicators=True,
        )

    def _build_strategy(self) -> Strategy:
        from pybroker.model import model

        pybroker.clear_params()
        pybroker.disable_logging()
        pybroker.disable_progress_bar()

        class _ConstantModel:
            def predict(self, _x):
                return np.ones(len(_x))

        hhv20 = pybroker.indicator(
            "hhv20", lambda bar_data: highv(bar_data.close, 20)
        )

        def train_pooled(train_df, test_df):
            del train_df, test_df
            return _ConstantModel()

        bench_pooled = model(
            "bench_pooled", train_pooled, indicators=(hhv20,), pooled=True
        )

        def exec_fn(ctx: ExecContext) -> None:
            preds = ctx.preds("bench_pooled")
            if len(preds) and preds[-1] > 0:
                ctx.buy_shares = 1

        start = self.df["date"].min().strftime("%Y-%m-%d")
        end = self.df["date"].max().strftime("%Y-%m-%d")
        strategy = Strategy(self.df, start, end, StrategyConfig())
        strategy.add_execution(
            exec_fn,
            symbols=sorted(self.df["symbol"].unique().tolist()),
            models=bench_pooled,
            indicators=[hhv20],
        )
        return strategy
