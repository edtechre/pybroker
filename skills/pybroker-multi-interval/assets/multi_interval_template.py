"""Starter PyBroker multi-timeframe strategy script.

Copy this file into a project and adapt the symbols, dates, intervals,
indicator, model, and execution rules to the user's strategy.
"""

import pybroker
from pybroker import ExecContext, Strategy, StrategyConfig, YFinance, sumv
from sklearn.linear_model import LinearRegression

# Cache downloaded data, indicators, and trained models across runs, and
# disable the progress bar so backtest output stays out of agent context.
pybroker.enable_caches("multi_interval_template")
pybroker.disable_progress_bar()

SYMBOLS = ["AAPL", "MSFT"]
START_DATE = "1/1/2020"
END_DATE = "1/1/2024"
BASE_TIMEFRAME = "1d"
TREND_INTERVAL = "weekly"
REGIME_INTERVAL = "monthly"
SMA_PERIOD = 10
MONTHLY_LOOKBACK = 4
MODEL_NAME = "weekly_slr"


# Indicator logic is NumPy over BarData arrays — never pandas. The same
# function runs on whichever interval it is bound to below.
def weekly_sma(data):
    return sumv(data.close, SMA_PERIOD) / SMA_PERIOD


sma_10 = pybroker.indicator("sma_10", weekly_sma)


def train_fn(symbol: str, train_data, test_data):
    # An interval-bound train_fn receives COMPRESSED weekly bars and is
    # the sanctioned pandas boundary. Copy so the input frame is never
    # widened or mutated.
    df = train_data.copy()
    # Target is the NEXT weekly return, matching lookahead=1 measured in
    # the bound interval's compressed (weekly) bars.
    df["target"] = df["close"].pct_change().shift(-1)
    df = df.dropna()
    model = LinearRegression()
    model.fit(df[["close"]], df["target"])
    # Pin the columns used as prediction input.
    return model, ["close"]


model_weekly = pybroker.model(MODEL_NAME, train_fn)


# Execution logic reads ctx.* NumPy arrays — never pandas.
def exec_fn(ctx: ExecContext):
    # ctx.interval() exposes only completed compressed bars; the week or
    # month still forming is never visible.
    weekly = ctx.interval(TREND_INTERVAL)
    monthly = ctx.interval(REGIME_INTERVAL)
    # Interval arrays are EMPTY during warmup — guard lengths first.
    if len(weekly.close) < SMA_PERIOD:
        return
    if len(monthly.close) < MONTHLY_LOOKBACK:
        return
    preds = weekly.preds(MODEL_NAME)
    if len(preds) == 0:
        return
    regime_up = monthly.close[-1] > monthly.close[-MONTHLY_LOOKBACK]
    # Read bound values with the base name — never "sma_10@weekly".
    wk_sma = weekly.indicator("sma_10")
    trend_up = weekly.close[-1] > wk_sma[-1]
    pos = ctx.long_pos()
    # Orders are always set on the base (daily) context; IntervalContext
    # is read-only.
    if not pos and regime_up and trend_up and preds[-1] > 0:
        # Fills on the next BASE (daily) bar at PriceType.MIDDLE, that bar's
        # low/high midpoint. Interval bindings never change fill pricing.
        ctx.buy_shares = ctx.calc_target_shares(0.25)
    elif pos and (not trend_up or preds[-1] < 0):
        ctx.sell_all_shares()


def build_strategy() -> Strategy:
    config = StrategyConfig(
        initial_cash=100_000,
        # Defaults to False, which leaves the final position open: it never
        # becomes a Trade, so trade_count, win_rate and total_pnl exclude it
        # and its P&L sits in unrealized_pnl. The liquidation lands on the
        # final BASE bar, usually mid-way through the last compressed bar.
        exit_on_last_bar=True,
    )
    strategy = Strategy(YFinance(), START_DATE, END_DATE, config)
    strategy.add_execution(
        exec_fn,
        SYMBOLS,
        # Binding computes sma_10 and trains the model on weekly bars,
        # replacing the base-timeframe variants; add "base" to the
        # .intervals() calls to also keep the daily copies.
        indicators=sma_10.intervals(TREND_INTERVAL),
        models=model_weekly.intervals(TREND_INTERVAL),
        # intervals= provides raw monthly bars only; the weekly binding
        # is auto-unioned into ctx.interval, so weekly is not repeated.
        intervals=REGIME_INTERVAL,
    )
    return strategy


if __name__ == "__main__":
    result = build_strategy().walkforward(
        windows=3,
        train_size=0.5,
        lookahead=1,  # measured in the bound interval's (weekly) bars
        # timeframe= declares the base bar spacing and is required once
        # any interval is declared or bound.
        timeframe=BASE_TIMEFRAME,
        # calc_bootstrap is a walkforward/backtest parameter, not a
        # StrategyConfig field. True adds result.bootstrap: BCa confidence
        # intervals for profit factor and Sharpe, plus percentile bounds on
        # max drawdown. Set StrategyConfig.bars_per_year to the BASE
        # timeframe's bar count or the Sharpe intervals stay per-bar.
        # calc_bootstrap=True,
    )
    print(result.metrics_df)
    # Structured output for agents/reports: result.to_json_str() serializes
    # metrics, trades, orders, and bootstrap; cap tables with max_rows=.
