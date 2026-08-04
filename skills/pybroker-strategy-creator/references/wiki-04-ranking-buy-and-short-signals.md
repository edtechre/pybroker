# Ranking Buy And Short Signals

Source: `docs/source/notebooks/4. Ranking Buy and Short Signals.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Ranking Buy and Short Signals

In this notebook, we will learn about the features of **PyBroker** that enable you to rank ticker symbols in your trading strategy. With these features, you can easily optimize your strategy and manage risk more effectively.

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance

pybroker.enable_data_source_cache("ranking")
```

## Scoring Ticker Symbols

In this section, we will learn about how to rank ticker symbols when placing buy orders. Let's begin with an example of how to rank ticker symbols based on volume when placing buy orders.

```python
def buy_highest_volume(ctx):
    # If there are no long positions across all tickers being traded:
    if not tuple(ctx.long_positions()):
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.hold_bars = 2
        ctx.long_score = ctx.volume[-1]
```

The ```buy_highest_volume``` function ranks ticker symbols by their most recent trading volume and allocates 100% of the portfolio for 2 bars. The ```ctx.score``` is set to ```ctx.volume[-1]```, which is the most recent trading volume.

```python
config = StrategyConfig(max_long_positions=1)
strategy = Strategy(YFinance(), "6/1/2021", "6/1/2022", config)
strategy.add_execution(buy_highest_volume, ["T", "F", "GM", "PFE"])
```

To limit the number of long positions that can be held at any time to ```1```, we set [max_long_positions](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.max_long_positions) to ```1``` in the [StrategyConfig](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig). In this example, we add the ```buy_highest_volume``` function to the [Strategy](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy) object and specify the ticker symbols to trade: ```['T', 'F', 'GM', 'PFE']```.

```python
result = strategy.backtest()
result.trades
```

## Shorting the Lowest Scores

**PyBroker** can also rank short orders using [short_score](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.short_score), where orders are placed for the ticker symbols with the *lowest* values. The following example buys the ticker symbol with the highest 5-day rate of change (ROC) while shorting the ticker symbol with the lowest 5-day ROC:

```python
def long_high_short_low(ctx):
    # Wait for 6 bars of data and skip symbols with an open position:
    if ctx.bars < 6 or ctx.long_pos() or ctx.short_pos():
        return
    # Calculate the 5-day rate of change (ROC):
    roc = (ctx.close[-1] - ctx.close[-6]) / ctx.close[-6]
    if roc > 0 and not tuple(ctx.long_positions()):
        ctx.set_target_shares(0.5, dir="long")
        # Hold the long position for 2 bars
        ctx.hold_bars = 2
        ctx.long_score = roc
    elif roc < 0 and not tuple(ctx.short_positions()):
        ctx.set_target_shares(0.5, dir="short")
        # Hold the short position for 2 bars
        ctx.hold_bars = 2
        ctx.short_score = roc


strategy = Strategy(YFinance(), "1/1/2025", "1/1/2026")
strategy.add_execution(long_high_short_low, ["T", "F", "GM", "PFE"])
strategy.set_max_long_positions(1)
strategy.set_max_short_positions(1)
result = strategy.backtest()
result.trades
```
