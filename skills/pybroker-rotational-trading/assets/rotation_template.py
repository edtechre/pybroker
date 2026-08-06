"""Starter PyBroker rotational trading script.

Copy this file into a project and adapt the universe, dates, ranking
indicator, position cap, hold band, and sizer to the user's strategy.
"""

import numpy as np
import pybroker
from pybroker import (
    ExecContext,
    RotationContext,
    Strategy,
    StrategyConfig,
    YFinance,
    indicator,
)

# Cache data source queries so reruns skip refetching, and disable the
# progress bar so backtest output stays out of agent context.
pybroker.enable_data_source_cache("rotation_template")
pybroker.disable_progress_bar()

UNIVERSE = [
    "TSLA",
    "NFLX",
    "AAPL",
    "NVDA",
    "AMZN",
    "MSFT",
    "GOOG",
    "AMD",
    "INTC",
    "META",
]
START_DATE = "1/1/2020"
END_DATE = "1/1/2024"
ROC_PERIOD = 20
MAX_LONG_POSITIONS = 2  # slots to hold
WORST_RANK_HELD = 5  # hold band; must be >= MAX_LONG_POSITIONS
STOP_LOSS_PCT = 10


# Indicator logic is NumPy over BarData arrays — never pandas.
def roc(data, period: int):
    # Full-length output, warmup left-padded with NaN. The [:-period]
    # slice aligns past closes with the present — it never reads bars
    # after index i, unlike forbidden negative *indexing* into a
    # full-length array.
    values = np.full(len(data.close), np.nan)
    values[period:] = data.close[period:] / data.close[:-period] - 1.0
    return values


roc_20 = indicator("roc_20", roc, period=ROC_PERIOD)


# Execution logic reads ctx.* NumPy arrays — never pandas. Under
# rotation, orders placed here would be ignored: this function's job is
# scores, stops, and fill prices only.
def rank_by_momentum(ctx: ExecContext):
    # NaN warmup scores are simply unrankable; no position exists yet.
    ctx.long_score = ctx.indicator("roc_20")[-1]
    # Stops set during execution are kept and applied to the entry
    # orders that rotation places.
    ctx.stop_loss_pct = STOP_LOSS_PCT


def size_by_rank(rotation: RotationContext):
    # Called after rotation decides what to trade; ranks are 1-based
    # with 1 the best. Delete sizer= below to fall back to equal weight
    # 1 / slots. Never override rotation's sell or cover signals.
    weights = {1: 0.7, 2: 0.3}
    for symbol, ctx in rotation.ctxs.items():
        # buy_shares is set only on the entries rotation placed.
        if ctx.buy_shares is not None:
            rank = rotation.long_ranks[symbol]
            ctx.buy_shares = ctx.calc_target_shares(weights[rank])


def build_strategy() -> Strategy:
    config = StrategyConfig(
        initial_cash=100_000,
        exit_on_last_bar=True,
    )
    strategy = Strategy(YFinance(), START_DATE, END_DATE, config)
    # Never the deprecated StrategyConfig.max_long_positions field.
    strategy.set_max_long_positions(MAX_LONG_POSITIONS)
    # Hold the top-ranked symbols; liquidate any holding whose rank
    # falls below WORST_RANK_HELD (band must be >= every position cap).
    strategy.enable_rotation(
        worst_rank_held=WORST_RANK_HELD, sizer=size_by_rank
    )
    strategy.add_execution(rank_by_momentum, UNIVERSE, indicators=roc_20)
    return strategy


if __name__ == "__main__":
    strategy = build_strategy()
    result = strategy.backtest(warmup=ROC_PERIOD)
    print(result.metrics_df)
    # Inspect hold-band exits and rotation entries:
    # print(result.orders)
