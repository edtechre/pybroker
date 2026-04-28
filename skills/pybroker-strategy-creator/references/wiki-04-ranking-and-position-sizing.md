# Ranking And Position Sizing

Source: `docs/source/notebooks/4. Ranking and Position Sizing.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Ranking and Position Sizing

In this notebook, we will learn about the features of **PyBroker** that enable you to rank ticker symbols and set position sizes for a group of symbols in your trading strategy. With these features, you can easily optimize your strategy and manage risk more effectively.

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance

pybroker.enable_data_source_cache('ranking_and_pos_sizing')
```

## Ranking Ticker Symbols

In this section, we will learn about how to rank ticker symbols when placing buy orders. Let's begin with an example of how to rank ticker symbols based on volume when placing buy orders.

```python
def buy_highest_volume(ctx):
    # If there are no long positions across all tickers being traded:
    if not tuple(ctx.long_positions()):
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.hold_bars = 2
        ctx.score = ctx.volume[-1]
```

The ```buy_highest_volume``` function ranks ticker symbols by their most recent trading volume and allocates 100% of the portfolio for 2 bars. The ```ctx.score``` is set to ```ctx.volume[-1]```, which is the most recent trading volume.

```python
config = StrategyConfig(max_long_positions=1)
strategy = Strategy(YFinance(), '6/1/2021', '6/1/2022', config)
strategy.add_execution(buy_highest_volume, ['T', 'F', 'GM', 'PFE'])
```

To limit the number of long positions that can be held at any time to ```1```, we set [max_long_positions](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.max_long_positions) to ```1``` in the [StrategyConfig](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig). In this example, we add the ```buy_highest_volume``` function to the [Strategy](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy) object and specify the ticker symbols to trade: ```['T', 'F', 'GM', 'PFE']```.

```python
result = strategy.backtest()
result.trades
```

## Setting Position Sizes

In **PyBroker**, you can set position sizes based on multiple tickers. To illustrate this, let's take a simple buy and hold strategy that starts trading after 100 days and holds positions for 30 days:

```python
def buy_and_hold(ctx):
    if not ctx.long_pos() and ctx.bars >= 100:
        ctx.buy_shares = 100
        ctx.hold_bars = 30
        
strategy = Strategy(YFinance(), '6/1/2021', '6/1/2022')
strategy.add_execution(buy_and_hold, ['T', 'F', 'GM', 'PFE'])
```

This will buy ```100``` shares in each of ```['T', 'F', 'GM', 'PFE']```. But what if you don't want to use equal position sizing? For example, you may want to size positions so that more shares are allocated to tickers with lower volatility to decrease the portfolio's overall volatility.

To customize position sizing for each ticker, we can define a [pos_size_handler](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_pos_size_handler) function that calculates the position size for each ticker:

```python
import numpy as np

def pos_size_handler(ctx):
    # Fetch all buy signals.
    signals = tuple(ctx.signals("buy"))
    # Return if there are no buy signals (i.e. there are only sell signals).
    if not signals:
        return
    # Calculates the inverse volatility, where volatility is defined as the
    # standard deviation of close prices for the last 100 days.
    get_inverse_volatility = lambda signal: 1 / np.std(signal.bar_data.close[-100:])
    # Sums the inverse volatilities for all of the buy signals.
    total_inverse_volatility = sum(map(get_inverse_volatility, signals))
    for signal in signals:
        size = get_inverse_volatility(signal) / total_inverse_volatility
        # Calculate the number of shares given the latest close price.
        shares = ctx.calc_target_shares(size, signal.bar_data.close[-1], cash=95_000)
        ctx.set_shares(signal, shares)
        
strategy.set_pos_size_handler(pos_size_handler)
```

The handler runs on every bar that generates a buy or sell signal when [buy_shares](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.buy_shares) or [sell_shares](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.sell_shares) is set on the [ExecContext](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext):

```python
result = strategy.backtest()
```

```python
result.trades
```

Using this method allows for a lot of possibilities, such as using [Mean-Variance Optimization](https://en.wikipedia.org/wiki/Modern_portfolio_theory) to determine portfolio allocations. 

[In the next notebook, we will discuss how to implement custom indicators in PyBroker](https://www.pybroker.com/en/latest/notebooks/5.%20Writing%20Indicators.html).
