"""Starter PyBroker strategy script.

Copy this file into a project and adapt the symbols, dates, indicators, and
execution rules to the user's strategy.
"""

import pybroker

# YFinance is not a PyBroker dependency: pip install yfinance
from pybroker import ExecContext, Strategy, StrategyConfig, YFinance, highest

# Cache data source queries so reruns skip refetching, and disable the
# progress bar so backtest output stays out of agent context.
pybroker.enable_data_source_cache("strategy_template")
pybroker.disable_progress_bar()

SYMBOLS = ["AAPL", "MSFT"]
START_DATE = "1/1/2020"
END_DATE = "1/1/2024"
LOOKBACK = 20


# Execution logic reads ctx.* NumPy arrays — never pandas. The arrays hold
# completed bars only, so ctx.close[-1] is the current bar, never the future.
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
        # Fills on the NEXT bar (buy_delay=1) at PriceType.MIDDLE, that
        # bar's low/high midpoint, unless ctx.buy_fill_price is set.
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        ctx.stop_loss_pct = 5
        ctx.hold_bars = 10
        # Rank breakout strength; set_max_long_positions keeps the strongest.
        ctx.long_score = ctx.close[-1] / high_20[-2] - 1


def build_strategy() -> Strategy:
    config = StrategyConfig(
        initial_cash=100_000,
        # Defaults to False, which leaves the final position open: it never
        # becomes a Trade, so trade_count, win_rate and total_pnl exclude it
        # and its P&L sits in unrealized_pnl. Exits fill at
        # exit_sell_fill_price / exit_cover_fill_price, both PriceType.MIDDLE.
        exit_on_last_bar=True,
    )
    high_20 = highest("high_20", "high", period=LOOKBACK)
    strategy = Strategy(YFinance(), START_DATE, END_DATE, config)
    strategy.set_max_long_positions(2)
    strategy.add_execution(exec_fn, SYMBOLS, indicators=high_20)
    return strategy


if __name__ == "__main__":
    result = build_strategy().backtest(
        warmup=LOOKBACK,
        # calc_bootstrap is a backtest/walkforward/optimize parameter, not a
        # StrategyConfig field. True adds result.bootstrap: BCa confidence
        # intervals for profit factor and Sharpe, plus percentile bounds on
        # max drawdown. Cost scales with bars x bootstrap_samples (a
        # StrategyConfig field, default 10_000); metrics_df is unchanged.
        # calc_bootstrap=True,
    )
    print(result.metrics_df)
    # Structured output for agents/reports: result.to_json_str() serializes
    # metrics, trades, orders, and bootstrap; cap tables with max_rows=.
