# Backtesting A Strategy

Source: `docs/source/notebooks/2. Backtesting a Strategy.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Backtesting a Strategy

We are now ready to test a basic trading strategy using **PyBroker**. To get started, import the following classes:

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance

pybroker.enable_data_source_cache("my_strategy")
```

For the backtest, we will use [Yahoo Finance](https://finance.yahoo.com) as our [DataSource](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.DataSource). We will also enable data caching to ensure we only download the required data once.

First, create a [StrategyConfig](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig) object to configure the `Strategy`. In this example, we set the initial cash to `500,000`:

```python
config = StrategyConfig(initial_cash=500_000)
```

Next, create a new instance of the `Strategy` class by passing the following arguments:

* **Data source**: We use Yahoo Finance for this example.
* **Start date**: The starting date for the backtest.
* **End date**: The ending date for the backtest.
* **Config**: The configuration object created earlier.

```python
strategy = Strategy(YFinance(), "3/1/2017", "3/1/2022", config)
```

The `Strategy` instance is now ready to download data from Yahoo Finance for the period between March 1, 2017, and March 1, 2022. To modify other settings, refer to the [StrategyConfig reference documentation](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig).

## Defining Strategy Rules

In this section, you will implement a basic trading strategy in **PyBroker** using the following rules:

1. Buy a stock if its last close price is lower than the previous bar's low, provided there is no open long position for that stock.
2. Set the buy order's limit price to `0.01` below the last close price.
3. Hold the position for 3 days before liquidating it at market price.
4. Apply these rules to AAPL and MSFT, allocating up to 25% of the portfolio to each.

To accomplish this, define a `buy_low` function. **PyBroker** will call this function separately for AAPL and MSFT on every data bar (where each bar represents a single trading day):

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

The `buy_low` function receives an [ExecContext](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext) (`ctx`) containing historical data for the current ticker (AAPL or MSFT). You can access the most recent close price via `ctx.close[-1]`.

To place an order, calculate the 25% portfolio allocation using [ctx.calc_target_shares](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.calc_target_shares) and assign it to [ctx.buy_shares](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.buy_shares). Then, set the limit price using [buy_limit_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.buy_limit_price). 

By default, buy orders fill on the following bar (`buy_delay=1`) at the [bar's midpoint price](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.PriceType.MIDDLE). You can customize this behavior using [StrategyConfig.buy_delay](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.buy_delay) and [ExecContext.buy_fill_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.buy_fill_price). 

Next, use [ctx.hold_bars](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.hold_bars) to specify the holding period. Upon liquidation, shares are sold at the [ExecContext.sell_fill_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.sell_fill_price) (which also defaults to the bar's midpoint).

To apply these `buy_low` rules to AAPL and MSFT, use the [add_execution](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution) method:

```python
strategy.add_execution(buy_low, ["AAPL", "MSFT"])
```

## Adding a Second Execution

You can apply different trading rules to different tickers within the same `Strategy` instance. To demonstrate this, let's define a new set of rules for a short strategy in a function called `short_high`, which functions similarly to our previous rules:

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
strategy.add_execution(short_high, ["TSLA"])
```

(Note, you can also retrieve bar data for another symbol by calling [ExecContext#foreign](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.foreign))

## Running a Backtest

To run a backtest, call the [backtest](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest) method on the ```Strategy``` instance. Here is an example:

```python
result = strategy.backtest()
```

The `backtest` method returns a [TestResult](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.TestResult) object containing detailed metrics from your backtest. For instance, you can use [Matplotlib](https://matplotlib.org/) to plot the portfolio's daily market value:

```python
import matplotlib.pyplot as plt

chart = plt.subplot2grid((3, 2), (0, 0), rowspan=3, colspan=2)
chart.plot(result.portfolio.index, result.portfolio["market_value"])
```

You can also access the executed trades for every entry and exit, along with all placed orders:

```python
result.trades
```

```python
result.orders
```

Additionally, `result.metrics_df` provides a DataFrame of metrics calculated from the backtest returns. You can find detailed explanations of these metrics in the [reference documentation](https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.EvalMetrics).

```python
result.metrics_df
```

## Filtering Backtest Data

You can filter the backtest data to include only specific bars. For example, to restrict the strategy to trading on Mondays, simply filter the data to keep only Monday bars:

```python
result = strategy.backtest(days="mon")
result.orders
```

Because data caching is enabled, **PyBroker** filtered the local data without re-downloading it from Yahoo Finance.

You can also filter data by specific time ranges (such as 9:30 to 10:30 AM) using the [between_time](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest) argument.

Although the metrics earlier indicate that we have a profitable strategy, we may have been misled by randomness. [In the next notebook, we'll discuss how to use bootstrapping to further evaluate our trading strategies](https://www.pybroker.com/en/latest/notebooks/3.%20Evaluating%20with%20Bootstrap%20Metrics.html).
