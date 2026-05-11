"""Starter PyBroker strategy script.

Copy this file into a project and adapt the symbols, dates, indicators, and
execution rules to the user's strategy.
"""

from pybroker import ExecContext, Strategy, StrategyConfig, YFinance, highest


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
        max_long_positions=2,
        exit_on_last_bar=True,
    )
    high_20 = highest("high_20", "high", period=LOOKBACK)
    strategy = Strategy(YFinance(), START_DATE, END_DATE, config)
    strategy.add_execution(exec_fn, SYMBOLS, indicators=high_20)
    return strategy


if __name__ == "__main__":
    result = build_strategy().backtest(warmup=LOOKBACK)
    print(result.metrics_df)
