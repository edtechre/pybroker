# Evaluating With Bootstrap Metrics

Source: `docs/source/notebooks/3. Evaluating with Bootstrap Metrics.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Evaluating with Bootstrap Metrics

Bootstrap metrics provide a more thorough evaluation of a trading strategy by simulating a wide range of potential market outcomes. Instead of relying on a single historical path, this approach reveals how robust your edge is.

[In the previous notebook](https://www.pybroker.com/en/latest/notebooks/2.%20Backtesting%20a%20Strategy.html), we implemented and backtested a trading strategy. Here is that implementation again:

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance

pybroker.enable_data_source_cache("my_strategy")


def buy_low(ctx):
    if ctx.long_pos():
        return
    if ctx.bars >= 2 and ctx.close[-1] < ctx.low[-2]:
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        ctx.buy_limit_price = ctx.close[-1] - 0.01
        ctx.hold_bars = 3


def short_high(ctx):
    if ctx.short_pos():
        return
    if ctx.bars >= 2 and ctx.close[-1] > ctx.high[-2]:
        ctx.sell_shares = 100
        ctx.hold_bars = 2
```

As before, we create a new [Strategy](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy) instance with the given configurations:

```python
config = StrategyConfig(initial_cash=500_000)
strategy = Strategy(YFinance(), "3/1/2017", "3/1/2022", config)
strategy.add_execution(buy_low, ["AAPL", "MSFT"])
strategy.add_execution(short_high, ["TSLA"])
```

Next, we run the backtest with bootstrap metrics enabled:

```python
result = strategy.backtest(calc_bootstrap=True)
result.metrics_df
```

While the `total_pnl` metric above suggests a profitable strategy, these results could just be due to chance. To increase confidence in our evaluation, we can use the [bootstrap method](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) to compute metrics.

Bootstrapping works by drawing thousands of random samples from the backtest's returns, computing the metric for each sample, and averaging the results. This provides a more robust and accurate estimate of performance.

## Confidence Intervals

**PyBroker** applies the bootstrap method to calculate [confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval) for two performance metrics: the [Profit Factor](https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.EvalMetrics.profit_factor) and the [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio).

```python
result.bootstrap.conf_intervals
```

**PyBroker** uses the [bias-corrected and accelerated (BCa) bootstrap method](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Bias-corrected_and_accelerated_(BCa)_bootstrap) to calculate confidence intervals for these metrics. Returns are sampled per-bar rather than per-trade to capture more granular data.

The resulting table displays the lower bound of the confidence interval at the specified confidence level, providing a more conservative estimate of the strategy's performance. For example, we can be `97.5%` confident that the Sharpe Ratio is at or above a given value *x*. 

In this example, the Sharpe Ratio has a negative lower bound, and the Profit Factor's lower bound is less than 1, suggesting the strategy is unreliable.

## Maximum Drawdown

Next, we evaluate the maximum drawdown of the strategy using the bootstrap method. The probabilities of the drawdown not exceeding certain thresholds (represented in both cash and percentage of portfolio equity) are displayed below:

```python
result.bootstrap.drawdown_conf
```

These confidence levels were obtained using per-bar returns from the backtest's out-of-sample results, similar to how the Profit Factor and Sharpe Ratio were calculated.

We can observe that the bootstrapped max drawdown of ```-23.1%``` at a ```99.9%``` confidence level is worse than the ```-4.7%``` we saw in our original results. This highlights the importance of using randomized tests to evaluate the performance of your trading strategy.

[In the next notebook, we will discuss how to incorporate ranking long and short signals in your trading strategies](https://www.pybroker.com/en/latest/notebooks/4.%20Ranking%20Long%20and%20Short%20Signals.html).
