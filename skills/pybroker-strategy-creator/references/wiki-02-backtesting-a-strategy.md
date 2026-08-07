# Backtesting A Strategy

Source: `docs/source/notebooks/2. Backtesting a Strategy.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Backtesting a Strategy

We're all set to test a basic trading strategy using **PyBroker**! To get started, we'll import the necessary classes listed below:

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance

pybroker.enable_data_source_cache('my_strategy')
```

For our backtest, we'll be using [Yahoo Finance](https://finance.yahoo.com) as our [DataSource](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.DataSource). We'll also be using data source caching to ensure that we only download the necessary data once when we run our backtests.

The next step is to set up a new instance of the [Strategy](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy) class which will be used to perform a backtest on our trading strategy. Here's how you can do it:

First, you can create a [StrategyConfig](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig) object to configure the ```Strategy```. In this case, we're setting the initial cash to ```500,000```:

```python
config = StrategyConfig(initial_cash=500_000)
```

Then, you can create a new instance of the ```Strategy``` class by passing in the following arguments:

- A data source: In this case, we're using Yahoo Finance as the data source.
- A start date: This is the starting date for the backtest.
- An end date: This is the end date for the backtest.
- The configuration object created earlier.

```python
strategy = Strategy(YFinance(), '3/1/2017', '3/1/2022', config)
```

The ```Strategy``` instance is now ready to download data from Yahoo Finance for the period between March 1, 2017, and March 1, 2022, before running the backtest using the specified configuration options. If you need to modify other configuration options, you can refer to the [StrategyConfig reference documentation](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig).

## Defining Strategy Rules

In this section, you will learn how to implement a basic trading strategy in **PyBroker** with the following rules:

1. Buy shares in a stock if the last close price is less than the low of the previous bar and there is no open long position in that stock.
2. Set the limit price of the buy order to 0.01 less than the last close price.
3. Hold the position for 3 days before liquidating it at market price.
4. Trade the rules on AAPL and MSFT, allocating up to 25% of the portfolio to each.

To accomplish this, you will define a ```buy_low``` function that **PyBroker** will call separately for AAPL and MSFT on every bar of data. Each bar corresponds to a single day of data:

```python
def buy_low(ctx):
    # If shares were already purchased and are currently being held, then return.
    if ctx.long_pos():
        return
    # If the latest close price is less than the previous day's low price,
    # then place a buy order.
    if ctx.bars >= 2 and ctx.close[-1] < ctx.low[-2]:
        # Buy a number of shares that is equal to 25% the portfolio.
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        # Set the limit price of the order.
        ctx.buy_limit_price = ctx.close[-1] - 0.01
        # Hold the position for 3 bars before liquidating (in this case, 3 days).
        ctx.hold_bars = 3
```

That is a lot to unpack! The ```buy_low``` function will receive an [ExecContext](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext) (```ctx```) containing data for the current ticker symbol (AAPL or MSFT). The ```ExecContext``` will contain all of the close prices up until the most recent bar of the current ticker symbol. The latest close price is retrieved with ```ctx.close[-1]```.

The ```buy_low``` function will use the ```ExecContext``` to place a buy order. The number of shares to purchase is set using [ctx.buy_shares](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.buy_shares), which is calculated with [ctx.calc_target_shares](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.calc_target_shares). In this case, the number of shares to buy will be equal to 25% of the portfolio. 

The limit price of the order is set with [buy_limit_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.buy_limit_price). If the criteria are met, the buy order will be filled on the next bar. The time at which the order is filled can be configured with [StrategyConfig.buy_delay](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.buy_delay), and its fill price can be set with [ExecContext.buy_fill_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.buy_fill_price). By default, buy orders are filled on the next bar (```buy_delay=1```) and at a [fill price equal to the midpoint between that bar's low and high price](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.PriceType.MIDDLE).

Finally, [ctx.hold_bars](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.hold_bars) specifies how many bars to hold the position for before liquidating it. When liquidated, the shares are sold at market price equal to [ExecContext.sell_fill_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.sell_fill_price), which is configurable and defaults to the midpoint between the bar's low and high price.

To add the ```buy_low``` rules to the ```Strategy``` for AAPL and MSFT, you will use [add_execution](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution):

```python
strategy.add_execution(buy_low, ['AAPL', 'MSFT'])
```

## Adding a Second Execution

You can use different sets of trading rules for different tickers within the same ```Strategy``` instance. In other words, you are not restricted to using only one set of trading rules for a single group of tickers. 

To demonstrate this, a new set of rules for a short strategy is provided in a function called ```short_high```, which is similar to the previous set of rules:

```python
def short_high(ctx):
    # If shares were already shorted then return.
    if ctx.short_pos():
        return
    # If the latest close price is more than the previous day's high price,
    # then place a sell order.
    if ctx.bars >= 2 and ctx.close[-1] > ctx.high[-2]:
        # Short 100 shares.
        ctx.sell_shares = 100
        # Cover the shares after 2 bars (in this case, 2 days).
        ctx.hold_bars = 2
```

The rules in ```short_high``` will be traded on ```TSLA```:

```python
strategy.add_execution(short_high, ['TSLA'])
```

(Note, you can also retrieve bar data for another symbol by calling [ExecContext#foreign](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.foreign))

## Running a Backtest

To run a backtest, call the [backtest](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest) method on the ```Strategy``` instance. Here is an example:

```python
result = strategy.backtest()
```

That was fast! The ```backtest``` method will return an instance of [TestResult](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.TestResult). You can access various information and metrics about the backtest through this instance. For example, to see the daily balances of the portfolio, you can plot the market value using [Matplotlib](https://matplotlib.org/):

```python
import matplotlib.pyplot as plt

chart = plt.subplot2grid((3, 2), (0, 0), rowspan=3, colspan=2)
chart.plot(result.portfolio.index, result.portfolio['market_value'])
```

You can also access the daily balance of each position that was held, the trades that were made for every entry and exit, and all of the orders that were placed:

```python
result.positions
```

```python
result.trades
```

```python
result.orders
```

Additionally, ```result.metrics_df``` contains a DataFrame of metrics calculated using the returns of the backtest. [You can read about what these metrics mean on the reference documentation](https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.EvalMetrics).

```python
result.metrics_df
```

## Filtering Backtest Data

You can filter the data used for the backtest to only include specific bars. For example, you can limit the strategy to trade only on Mondays by filtering the data to only contain bars for Mondays:

```python
result = strategy.backtest(days='mon')
result.orders
```

The data doesn't need to be downloaded again from Yahoo Finance because caching is enabled and the cached data only needs to be filtered.

You can also filter the data by time range, such as 9:30-10:30 AM, using the [between_time](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest) argument.

Although the metrics earlier indicate that we have a profitable strategy, we may be misled by randomness. [In the next notebook, we'll discuss how to use bootstrapping to further evaluate our trading strategies](https://www.pybroker.com/en/latest/notebooks/3.%20Evaluating%20with%20Bootstrap%20Metrics.html).
