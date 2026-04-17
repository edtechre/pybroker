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
