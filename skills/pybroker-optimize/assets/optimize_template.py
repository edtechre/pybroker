"""Starter PyBroker optimization script.

Copy this file into a project and adapt the symbols, dates, hyperparams,
score function, and execution rules to the user's strategy.
"""

import numpy as np
import pybroker
from pybroker import (
    ExecContext,
    Strategy,
    StrategyConfig,
    TestResult,
    YFinance,
    hyperparam,
    indicator,
    sumv,
)

# Cache data source queries so reruns skip refetching, and disable the
# progress bar so backtest output stays out of agent context.
pybroker.enable_data_source_cache("optimize_template")
pybroker.disable_progress_bar()
# optimize runs one backtest per trial; per-run logs flood agent context.
pybroker.disable_logging()

SYMBOLS = ["AAPL", "MSFT"]
START_DATE = "1/1/2020"
END_DATE = "1/1/2024"

# The hyperparam NAME is the search-space key; candidate values run from
# low to high inclusive in steps of step. low == high pins a fixed value.
period = hyperparam("period", default=30, low=10, high=50, step=10)
stop_pct = hyperparam("stop_pct", default=6.0, low=2.0, high=10.0, step=2.0)


# Indicator logic is NumPy over BarData arrays — never pandas.
def sma(data, period: int):
    return sumv(data.close, period) / period


# The Hyperparam is passed in place of a concrete period; each trial
# recomputes the indicator with its suggested value.
sma_ind = indicator("sma", sma, period=period)


# Execution logic reads ctx.* NumPy arrays — never pandas.
def exec_fn(ctx: ExecContext):
    sma_vals = ctx.indicator("sma")
    if np.isnan(sma_vals[-1]):
        return
    if not ctx.long_pos() and ctx.close[-1] > sma_vals[-1]:
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        # ctx.hyperparam works because stop_pct is attached below.
        ctx.stop_loss_pct = ctx.hyperparam("stop_pct")
    elif ctx.long_pos() and ctx.close[-1] < sma_vals[-1]:
        ctx.sell_all_shares()


def score_fn(result: TestResult) -> float:
    # Guard Optional metrics: a None or NaN score marks the trial FAILED.
    if result.metrics.sharpe is None:
        return 0.0
    return result.metrics.sharpe


def build_strategy() -> Strategy:
    config = StrategyConfig(
        initial_cash=100_000,
        exit_on_last_bar=True,
    )
    strategy = Strategy(YFinance(), START_DATE, END_DATE, config)
    strategy.add_execution(
        exec_fn, SYMBOLS, indicators=[sma_ind], hyperparams=[stop_pct]
    )
    return strategy


if __name__ == "__main__":
    strategy = build_strategy()
    # Grid search exhausts the lattices; seed= makes the run reproduce.
    # Alternatives: sampler="tpe" with n_trials=25 for large grids, or
    # windows=3 for walkforward tuning (best_params then reflects the
    # LAST window; read per-window values from opt.windows).
    opt = strategy.optimize(score_fn, seed=42, train_size=0.5)
    print(opt.best_params)
    print(opt.best_score)
    # Held-out test metrics — never report the in-sample best_score.
    print(opt.result.metrics_df)
    # Pin the winners into a regular backtest:
    # result = strategy.backtest(params=opt.best_params, seed=42)
