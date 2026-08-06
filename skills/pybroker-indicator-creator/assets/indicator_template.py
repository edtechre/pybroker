"""Starter PyBroker indicator script.

Copy this file into a project and adapt the indicator functions, symbols,
dates, and execution rules to the user's indicators.
"""

import numpy as np
import pybroker
from numba import njit
from pybroker import (
    ExecContext,
    Strategy,
    StrategyConfig,
    YFinance,
    indicator,
    returns,
    sumv,
)

# Cache data source queries so reruns skip refetching, and disable the
# progress bar so backtest output stays out of agent context.
pybroker.enable_data_source_cache("indicator_template")
pybroker.disable_progress_bar()

SYMBOLS = ["AAPL", "MSFT"]
START_DATE = "1/1/2020"
END_DATE = "1/1/2024"
LOOKBACK = 20
TREND_LOOKBACK = 63


def sma(data, period: int):
    # Vector helpers consume BarData arrays directly.
    return sumv(data.close, period) / period


def cmma(data, lookback: int):
    # BarData cannot cross the @njit boundary; the nested kernel takes
    # plain NumPy arrays.
    @njit
    def kernel(values):
        n = len(values)
        out = np.full(n, np.nan)
        for i in range(lookback, n):
            ma = 0.0
            for j in range(i - lookback, i):
                ma += values[j]
            out[i] = values[i] - ma / lookback
        return out

    return kernel(data.close)


roc_63 = returns("roc_63", "close", period=TREND_LOOKBACK)
sma_20 = indicator("sma_20", sma, period=LOOKBACK)
cmma_20 = indicator("cmma_20", cmma, lookback=LOOKBACK)


def exec_fn(ctx: ExecContext):
    if ctx.bars <= TREND_LOOKBACK:
        return

    if ctx.long_pos():
        # Exit once price closes back above the 20 day moving average.
        if ctx.close[-1] > ctx.indicator("sma_20")[-1]:
            ctx.sell_all_shares()
        return

    # Buy pullbacks (negative CMMA) within a longer-term uptrend.
    if ctx.indicator("roc_63")[-1] > 0 and ctx.indicator("cmma_20")[-1] < 0:
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        ctx.stop_loss_pct = 5


def build_strategy() -> Strategy:
    config = StrategyConfig(
        initial_cash=100_000,
        exit_on_last_bar=True,
    )
    strategy = Strategy(YFinance(), START_DATE, END_DATE, config)
    strategy.add_execution(
        exec_fn, SYMBOLS, indicators=[roc_63, sma_20, cmma_20]
    )
    return strategy


if __name__ == "__main__":
    result = build_strategy().backtest(warmup=TREND_LOOKBACK)
    print(result.metrics_df)
