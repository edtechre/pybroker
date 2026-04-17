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
from pybroker.vect import highv, lowv

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
        bootstrap_sample_size=10,
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
            disable_parallel=True,
        )

    def time_walkforward(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=True,
            disable_parallel=True,
        )

    def peakmem_walkforward(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=True,
            disable_parallel=True,
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
            disable_parallel=True,
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
            disable_parallel=True,
        )

    def time_walkforward_scaled(self, windows: int, symbols: int) -> None:
        _build_strategy(self.df).walkforward(
            windows=windows,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel=True,
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
            disable_parallel=True,
        )

    def time_walkforward_large(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=5,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel=True,
        )

    def peakmem_walkforward_large(self) -> None:
        _build_strategy(self.df).walkforward(
            windows=5,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            disable_parallel=True,
        )


# ---------------------------------------------------------------------------
# Group B — subsystem micro-benches (no vector dependency)
# ---------------------------------------------------------------------------


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
            max_drawdown,
            profit_factor,
            sharpe_ratio,
        )

        rng = np.random.default_rng(SEED)
        self._changes = rng.normal(0.0, 1.0, 5_000)
        self._returns = rng.normal(0.0005, 0.01, 5_000)
        self._bca_boot_conf = bca_boot_conf
        self._max_drawdown = max_drawdown
        self._sharpe_ratio = sharpe_ratio
        self._profit_factor = profit_factor
        # Warmup JIT compile.
        self._max_drawdown(self._changes)
        self._sharpe_ratio(self._returns)
        self._profit_factor(self._changes)
        self._bca_boot_conf(self._changes, 500, 200, self._sharpe_ratio)

    def time_max_drawdown(self) -> None:
        self._max_drawdown(self._changes)

    def time_sharpe_ratio(self) -> None:
        self._sharpe_ratio(self._returns)

    def time_profit_factor(self) -> None:
        self._profit_factor(self._changes)

    def time_bca_boot_conf(self) -> None:
        self._bca_boot_conf(self._changes, 500, 200, self._sharpe_ratio)


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
            disable_parallel=True,
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
            disable_parallel=True,
        )
        equity = result.portfolio["equity"].to_numpy()
        digest = hashlib.sha256(equity.tobytes()).hexdigest()
        # asv's track_* expects a number; use the first 8 hex chars as int.
        return int(digest[:8], 16)

    track_walkforward_equity_hash.unit = "hash"  # type: ignore[attr-defined]


class WalkforwardWithCache(Walkforward):
    """V5-specific: walkforward with indicator_cache enabled.

    The parent `Walkforward.setup` runs one warmup pass — that populates
    diskcache with indicator values. The timed method then reads through
    the populated cache, so the in-process L1 (V5) serves hits on the
    second and subsequent runs. Without V5's L1, each hit goes to disk +
    deserialize; with V5, it's an in-memory dict lookup.
    """

    timeout = 300

    def setup(self) -> None:
        import tempfile

        import pybroker

        self._cache_dir = tempfile.mkdtemp(prefix="asv_v5_")
        pybroker.enable_indicator_cache("asv_v5", self._cache_dir)
        # Parent.setup loads dataset + runs one warmup walkforward.
        # With the cache enabled, that warmup populates diskcache.
        super().setup()

    def teardown(self) -> None:
        import shutil

        import pybroker

        pybroker.disable_indicator_cache()
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    # time_walkforward / peakmem_walkforward inherited from Walkforward.
