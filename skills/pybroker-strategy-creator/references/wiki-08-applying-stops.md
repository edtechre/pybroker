# Applying Stops

Source: `docs/source/notebooks/8. Applying Stops.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Applying Stops

Stops are used to automatically buy or sell a security once it reaches a specified price level. They can be useful for limiting potential losses by allowing traders to exit bad trades automatically, as well as for taking profits by selling a security automatically when it reaches a certain price level.

**PyBroker** supports the simulation of stops, which is explained in detail in this notebook:

```python
import pybroker
from pybroker import Strategy, YFinance

pybroker.enable_data_source_cache('stops')

strategy = Strategy(YFinance(), '1/1/2018', '1/1/2023')
```

## Stop Loss

A stop loss order is used to automatically exit a trade once the security's price reaches or falls below a specified level. For example, the following code shows an example of a stop loss order set at ``20%`` below the entry price:

```python
def buy_with_stop_loss(ctx):
    if not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_loss_pct = 20
        
strategy.add_execution(buy_with_stop_loss, ['TSLA'])
result = strategy.backtest()
result.trades
```

## Take Profit

Similarly, a take profit order can be used to lock in profits on a trade. The following code adds a take profit order at ``10%`` above the entry price:

```python
def buy_with_stop_loss_and_profit(ctx):
    if not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_loss_pct = 20
        ctx.stop_profit_pct = 10
        
strategy.clear_executions()
strategy.add_execution(buy_with_stop_loss_and_profit, ['TSLA'])
result = strategy.backtest()
result.trades
```

## Trailing Stop

A trailing stop is an order that is used to exit a trade once the instrument's price falls a certain percentage or cash amount below its highest market price. Here is an example of setting a trailing stop at ``20%`` below the highest market price:

```python
def buy_with_trailing_stop_loss_and_profit(ctx):
    if not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_trailing_pct = 20
        ctx.stop_profit_pct = 10
        
strategy.clear_executions()
strategy.add_execution(buy_with_trailing_stop_loss_and_profit, ['TSLA'])
result = strategy.backtest()
result.trades
```

## Setting a Limit Price

A stop order can be combined with a limit price to ensure that the order is executed only at a specific price level. Below shows an example of placing a limit price on a stop order:

```python
def buy_with_trailing_stop_loss_and_profit(ctx):
    if not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_trailing_pct = 20
        ctx.stop_trailing_limit = ctx.close[-1] + 1
        ctx.stop_profit_pct = 10
        ctx.stop_profit_limit = ctx.close[-1] - 1
        
strategy.clear_executions()
strategy.add_execution(buy_with_trailing_stop_loss_and_profit, ['TSLA'])
result = strategy.backtest()
result.trades.head()
```

## Canceling a Stop

The following code shows an example of canceling a stop order:

```python
def buy_with_stop_trailing_and_cancel(ctx):
    pos = ctx.long_pos()
    if not pos:
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_trailing_pct = 20
    elif pos.bars > 60:
        ctx.cancel_stops(ctx.symbol)
        
strategy.clear_executions()
strategy.add_execution(buy_with_stop_trailing_and_cancel, ['TSLA'])
result = strategy.backtest()
result.trades
```

## Setting the Stop Exit Price

By default, stops are checked against the bar's low and high prices, and they are exited at the stop's threshold (e.g., -2%) on the same bar when the stop is triggered.

To set a custom exit price, the "exit_price" fields for each stop type can be used. When these fields are set, the stop will be checked against the ``exit_price``, and it will exit at the ``exit_price`` when triggered. For example, the code below sets the [stop_trailing_exit_price](https://www.pybroker.com/en/dev/reference/pybroker.context.html#pybroker.context.ExecContext.stop_trailing_exit_price) to the open price on the bar that triggers the stop:

```python
from pybroker import PriceType

def buy_with_stop_trailing_and_exit_price(ctx):
    pos = ctx.long_pos()
    if not pos:
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_trailing_pct = 20
        ctx.stop_trailing_exit_price = PriceType.OPEN
        
strategy.clear_executions()
strategy.add_execution(buy_with_stop_trailing_and_exit_price, ['TSLA'])
result = strategy.backtest()
result.trades.head()
```

[For more information on the various attributes that can be used to set stops in PyBroker, you can refer to the ExecContext reference documentation.](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext)
