"""Starter PyBroker strategy script.

Copy this file into a project and adapt the symbols, dates, indicators, and
execution rules to the user's strategy.
"""

import pybroker
from pybroker import ExecContext, Strategy, StrategyConfig, YFinance, highest

# Cache data source queries so reruns skip refetching, and disable the
# progress bar so backtest output stays out of agent context.
pybroker.enable_data_source_cache("strategy_template")
pybroker.disable_progress_bar()

SYMBOLS = ["AAPL", "MSFT"]
START_DATE = "1/1/2020"
END_DATE = "1/1/2024"
LOOKBACK = 20


def exec_fn(ctx: ExecContext):
    if ctx.bars < LOOKBACK + 1:
        return

    high_20 = ctx.indicator("high_20")
    pos = ctx.long_pos()

    if pos:
        if ctx.close[-1] < ctx.low[-2]:
            ctx.sell_all_shares()
        return

    if ctx.close[-1] > high_20[-2]:
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        ctx.stop_loss_pct = 5
        ctx.hold_bars = 10


def build_strategy() -> Strategy:
    config = StrategyConfig(
        initial_cash=100_000,
        exit_on_last_bar=True,
    )
    high_20 = highest("high_20", "high", period=LOOKBACK)
    strategy = Strategy(YFinance(), START_DATE, END_DATE, config)
    strategy.set_max_long_positions(2)
    strategy.add_execution(exec_fn, SYMBOLS, indicators=high_20)
    return strategy


if __name__ == "__main__":
    result = build_strategy().backtest(warmup=LOOKBACK)
    print(result.metrics_df)
