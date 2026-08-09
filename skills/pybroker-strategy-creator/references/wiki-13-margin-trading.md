# Margin Trading

Source: `docs/source/notebooks/13. Margin Trading.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Margin Trading

Until now, our backtests have been limited to cash-only trades. This notebook explores the margin trading features in **PyBroker v2**. We'll use [StrategyConfig](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig) to apply leverage to our buying power, calculate interest on borrowed funds, and set up collateral for short positions.

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance, sumv

pybroker.enable_data_source_cache("margin_trading")
```

## Configuring Leverage

Margin is enabled with the [leverage](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.leverage) config option, which multiplies buying power for both long and short positions.

```python
config = StrategyConfig(initial_cash=100_000, leverage=2.0)
```

The default of `1.0` buys with cash only, and setting `leverage` to `2.0` allows holding positions worth up to 2x our equity. The borrowed half of each position is tracked as a margin loan. **PyBroker** does not model margin calls. Orders are instead limited to the available buying power at fill time.

Next, we write a simple trend-following strategy that holds a symbol while it closes above its 50-day moving average:

```python
sma_50 = pybroker.indicator("sma_50", lambda data: sumv(data.close, 50) / 50)


def trend_follow(ctx):
    sma = ctx.indicator("sma_50")[-1]
    if ctx.long_pos() is None and ctx.close[-1] > sma:
        ctx.buy_shares = ctx.calc_target_shares(0.25)
    elif ctx.long_pos() is not None and ctx.close[-1] < sma:
        ctx.sell_all_shares()
```

Position sizing is where leverage comes into play. The [ctx.calc_target_shares](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.calc_target_shares) method sizes orders as a fraction of your deployable capital, which equals your total equity multiplied by [leverage](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.leverage).

As a result, targeting 25% across four stocks deploys up to roughly 2x our equity when trading on margin:

```python
yfinance = YFinance()
strategy = Strategy(
    yfinance, start_date="1/1/2025", end_date="8/1/2026", config=config
)
strategy.add_execution(
    trend_follow, ["GS", "MS", "C", "USB"], indicators=sma_50
)
result_2x = strategy.backtest()
result_2x.portfolio.tail()
```

[result.portfolio](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.TestResult.portfolio) records the margin balances on every bar. When a levered position is opened, a portion of cash (`entry cost / leverage`) is posted as collateral. The borrowed remainder is tracked in [margin_loan](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.margin_loan), and the [net_cash_balance](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.net_cash_balance) equals `cash - margin_loan`.

The [margin](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.margin) column is separate and tracks the current value of open short positions, so it stays at zero when the strategy is long only.

To see these margin mechanics in action, we filter the output to show only bars with an outstanding loan:

```python
levered = result_2x.portfolio[result_2x.portfolio["margin_loan"] > 0]
levered.head()
```

On the first of these filtered bars, a single entry filled. Cash dropped by `$25,413` to post half of the entry cost as collateral, while the `margin_loan` column records the borrowed remainder. While the collateral and margin loan are fixed at the entry price, the [notional](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.notional) column tracks the position's current market value at each close.

## Comparing Against Cash-Only

To see the effect of leverage on our strategy, we rerun the strategy with a cash only [leverage](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.leverage) of `1.0`. We also disable PyBroker's logging to keep the output clean for the remaining runs using [disable_logging](https://www.pybroker.com/en/latest/reference/pybroker.scope.html#pybroker.scope.disable_logging):

```python
pybroker.disable_logging()


def run_backtest(
    config, exec_fn=trend_follow, symbols=("GS", "MS", "C", "USB")
):
    strategy = Strategy(
        yfinance, start_date="1/1/2025", end_date="8/1/2026", config=config
    )
    strategy.add_execution(exec_fn, symbols, indicators=sma_50)
    return strategy.backtest()


result_1x = run_backtest(StrategyConfig(initial_cash=100_000))
print(f"1x total return: {result_1x.metrics.total_return_pct:.2f}%")
print(f"2x total return: {result_2x.metrics.total_return_pct:.2f}%")
print(f"1x max drawdown: {result_1x.metrics.max_drawdown_pct:.2f}%")
print(f"2x max drawdown: {result_2x.metrics.max_drawdown_pct:.2f}%")
```

## Charging Margin Interest

Leverage more than doubled the return (and nearly doubled the max drawdown), but borrowing is not free. You can simulate this financing cost using the [interest_rate](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.interest_rate) config option. It applies an annual percentage rate to the portfolio's net cash balance, accruing once per bar at `interest_rate / bars_per_year`. To use this feature, you must also set [bars_per_year](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.bars_per_year).

Interest is charged when net cash is negative (the margin loan exceeds cash) and credited when net cash is positive. To see how these costs affect the portfolio, we will backtest a buy-and-hold strategy. Because both runs hold the same positions, any difference in their final market value will be the interest paid:

```python
def buy_and_hold(ctx):
    if ctx.long_pos() is None:
        ctx.buy_shares = ctx.calc_target_shares(0.25)


config_interest = StrategyConfig(
    initial_cash=100_000,
    leverage=2.0,
    interest_rate=6.0,
    bars_per_year=252,
)
result_hold = run_backtest(config, buy_and_hold)
result_interest = run_backtest(config_interest, buy_and_hold)
print(
    "Final market value (no interest):",
    result_hold.portfolio["market_value"].iloc[-1],
)
print(
    "Final market value (6% interest):",
    result_interest.portfolio["market_value"].iloc[-1],
)
```

Looking at the tail of the portfolio, `cash` will stay zero. Meanwhile, the accrued interest is added to the [margin_loan](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.margin_loan) and increases with every bar:

```python
result_interest.portfolio[
    ["cash", "margin_loan", "net_cash_balance", "market_value"]
].tail()
```

## Shorting on Margin

Short selling also uses margin. Shorts require collateral equal to `entry cost / leverage` and use the same buying power as long positions.

To demonstrate this, we will short symbols trading below their moving average. We will also enable [record_position_bars](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.record_position_bars) to capture per-position balances:

```python
def trend_short(ctx):
    sma = ctx.indicator("sma_50")[-1]
    if ctx.short_pos() is None and ctx.close[-1] < sma:
        ctx.sell_shares = ctx.calc_target_shares(0.25)
    elif ctx.short_pos() is not None and ctx.close[-1] > sma:
        ctx.cover_all_shares()


config_short = StrategyConfig(
    initial_cash=100_000, leverage=2.0, record_position_bars=True
)
result_short = run_backtest(
    config_short, trend_short, ["HD", "LOW", "CMCSA", "KHC"]
)
shorted = result_short.portfolio[result_short.portfolio["margin"] > 0]
shorted[
    [
        "cash",
        "equity",
        "margin",
        "margin_loan",
        "net_cash_balance",
        "market_value",
    ]
].head()
```

For open short positions, the [margin](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.margin) column tracks their current notional exposure, which can exceed your total equity when using leverage. 

When you open a short, collateral equal to `entry cost / leverage` is held from your cash, and the [margin_loan](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.margin_loan) column records the borrowed remainder. Because the collateral and the loan are fixed at entry, they do not change with the `margin` column. 

Your [net_cash_balance](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.net_cash_balance) equals your remaining cash minus this margin loan, turning negative if the loan exceeds your available cash. Note that [equity](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.equity) values shorts at their fixed entry cost, while [market_value](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.market_value) includes their unrealized PnL.

Because we enabled [record_position_bars](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.record_position_bars), [result.positions](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.TestResult.positions) tracks the balances for each individual position. This includes each short's specific share of the portfolio's [margin](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.PortfolioBar.margin), as well as its own unrealized PnL:

```python
result_short.positions[
    ["short_shares", "close", "margin", "unrealized_pnl"]
].head()
```
