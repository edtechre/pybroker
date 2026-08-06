# Applying Stops

Source: `docs/source/notebooks/8. Applying Stops.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Applying Stops

Stops automatically buy or sell a security once it reaches a specified price level. They help limit potential losses by exiting bad trades, as well as lock in profits by selling when a security reaches a target price.

This notebook explains how to simulate stops in **PyBroker**:

```python
import pybroker
from pybroker import Strategy, YFinance

pybroker.enable_data_source_cache("stops")

strategy = Strategy(YFinance(), "1/1/2018", "1/1/2023")
```

## Stop Loss

A stop loss order is used to automatically exit a trade once the security's price reaches or falls below a specified level. For example, the following code shows an example of a stop loss order set at ``20%`` below the entry price:

```python
def buy_with_stop_loss(ctx):
    if not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_loss_pct = 20


strategy.add_execution(buy_with_stop_loss, ["TSLA"])
result = strategy.backtest()
result.trades
```

## Take Profit

A take profit order can be used to lock in profits on a trade. The following code adds a take profit order at ``10%`` above the entry price:

```python
def buy_with_stop_loss_and_profit(ctx):
    if not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_loss_pct = 20
        ctx.stop_profit_pct = 10


strategy.clear_executions()
strategy.add_execution(buy_with_stop_loss_and_profit, ["TSLA"])
result = strategy.backtest()
result.trades
```

## Trailing Stop

A trailing stop order automatically exits a trade once the instrument's price falls a specified percentage or cash amount below its highest market price. The following example sets a trailing stop at 20% below the highest market price:

```python
def buy_with_trailing_stop_loss_and_profit(ctx):
    if not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_trailing_pct = 20
        ctx.stop_profit_pct = 10


strategy.clear_executions()
strategy.add_execution(buy_with_trailing_stop_loss_and_profit, ["TSLA"])
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
strategy.add_execution(buy_with_trailing_stop_loss_and_profit, ["TSLA"])
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
strategy.add_execution(buy_with_stop_trailing_and_cancel, ["TSLA"])
result = strategy.backtest()
result.trades
```

## Setting the Stop Exit Price

By default, **PyBroker** checks stops against the bar's low and high prices and exits the trade at the stop's threshold (e.g., -2%) on the same bar the stop triggers.

To set a custom exit price, you can use the `exit_price` fields available for each stop type. When configured, **PyBroker** checks the stop against the `exit_price` and uses that price for the exit when triggered. The following code sets [stop_trailing_exit_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.stop_trailing_exit_price) to the open price of the bar that triggers the stop:

```python
from pybroker import PriceType


def buy_with_stop_trailing_and_exit_price(ctx):
    if not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.stop_trailing_pct = 20
        ctx.stop_trailing_exit_price = PriceType.OPEN


strategy.clear_executions()
strategy.add_execution(buy_with_stop_trailing_and_exit_price, ["TSLA"])
result = strategy.backtest()
result.trades.head()
```

For more details on the attributes available for configuring stops, see the [ExecContext reference documentation](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext).
