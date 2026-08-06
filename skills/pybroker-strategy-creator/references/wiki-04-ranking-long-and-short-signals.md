# Ranking Long And Short Signals

Source: `docs/source/notebooks/4. Ranking Long and Short Signals.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Ranking Long and Short Signals

In this notebook, you will learn how to rank long and short signals across ticker symbols.

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance

pybroker.enable_data_source_cache("ranking")
```

## Long Signals

Let's begin with an example of how to rank ticker symbols based on volume when placing buy orders:

```python
def buy_highest_volume(ctx):
    # If there are no long positions across all tickers being traded:
    if not ctx.has_long_positions():
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.long_score = ctx.volume[-1]
        ctx.hold_bars = 2
```

The ```buy_highest_volume``` function allocates 100% of the portfolio and holds for 2 bars. It sets ```ctx.long_score``` to ```ctx.volume[-1]``` so **PyBroker** ranks the buy signals by volume.

```python
config = StrategyConfig(max_long_positions=1)
strategy = Strategy(YFinance(), "6/1/2021", "6/1/2022", config)
strategy.add_execution(buy_highest_volume, ["T", "F", "GM", "PFE"])
```

To limit the number of long positions that can be held at any time to ```1```, we set [max_long_positions](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.max_long_positions) to ```1``` in the [StrategyConfig](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig). This effectively buys the symbol with the highest volume.

```python
result = strategy.backtest()
result.trades
```

## Short Signals

**PyBroker** can also rank short signals using [short_score](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.short_score), where short orders are placed for the ticker symbols with the highest `short_score` values. The following example buys the ticker symbol with the highest 5-day rate of change (ROC) while shorting the ticker symbol with the most negative 5-day ROC:

```python
def long_high_short_low(ctx):
    # Wait for 6 bars of data and skip symbols with an open position:
    if ctx.bars < 6 or ctx.long_pos() or ctx.short_pos():
        return
    # Calculate the 5-day rate of change (ROC):
    roc = (ctx.close[-1] - ctx.close[-6]) / ctx.close[-6]
    if roc > 0 and not ctx.has_long_positions():
        ctx.buy_shares = ctx.calc_target_shares(0.5)
        # Hold the long position for 2 bars
        ctx.hold_bars = 2
        ctx.long_score = roc
    elif roc < 0 and not ctx.has_short_positions():
        ctx.sell_shares = ctx.calc_target_shares(0.5)
        # Hold the short position for 2 bars
        ctx.hold_bars = 2
        ctx.short_score = -roc


strategy = Strategy(YFinance(), "1/1/2025", "1/1/2026")
strategy.add_execution(long_high_short_low, ["T", "F", "GM", "PFE"])
strategy.set_max_long_positions(1)
strategy.set_max_short_positions(1)
result = strategy.backtest()
result.trades
```

[In the next notebook, we will discuss how to implement custom indicators in PyBroker](https://www.pybroker.com/en/latest/notebooks/5.%20Writing%20Indicators.html).
