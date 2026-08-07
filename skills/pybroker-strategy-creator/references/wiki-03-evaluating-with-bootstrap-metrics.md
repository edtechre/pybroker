# Evaluating With Bootstrap Metrics

Source: `docs/source/notebooks/3. Evaluating with Bootstrap Metrics.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Evaluating with Bootstrap Metrics

Bootstrap metrics can help us to more thoroughly evaluate a trading strategy, as we will see in this notebook.

[In the previous notebook](https://www.pybroker.com/en/latest/notebooks/2.%20Backtesting%20a%20Strategy.html), we implemented a trading strategy and backtested it. Here is the implementation again:

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance

pybroker.enable_data_source_cache('my_strategy')

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
config = StrategyConfig(initial_cash=500_000, bootstrap_sample_size=100)
strategy = Strategy(YFinance(), '3/1/2017', '3/1/2022', config)
strategy.add_execution(buy_low, ['AAPL', 'MSFT'])
strategy.add_execution(short_high, ['TSLA'])
```

This time, the ```Strategy``` is configured with a [bootstrap_sample_size](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.bootstrap_sample_size) of ```100``` because of the small amount of historical data being used. Next, we run the backtest with bootstrap metrics enabled:

```python
result = strategy.backtest(calc_bootstrap=True)
result.metrics_df
```

When we look at the ```total_pnl``` metric above, it seems that we have a profitable trading strategy on our first try. However, we cannot be completely sure that these results are repeatable and not just due to chance. To gain more confidence in our results, we can use the [boostrap method](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) to compute metrics.

The bootstrap method works by repeatedly computing a metric on random samples drawn from the backtest's returns. Then, the metric is computed on each random sample, and the average is taken. By doing this on thousands of random samples, we obtain a more robust and accurate estimate of the metric.


## Confidence Intervals

**PyBroker** applies the bootstrap method to calculate [confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval) for two performance metrics, the [Profit Factor](https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.EvalMetrics.profit_factor) and [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio):

```python
result.bootstrap.conf_intervals
```

**PyBroker** uses the [bias corrected and accelerated (BCa) bootstrap method](https://blogs.sas.com/content/iml/2017/07/12/bootstrap-bca-interval.html) to calculate the confidence intervals for these metrics. The returns are sampled per-bar rather than per-trade to capture more information in the metrics.

The resulting table shows the lower bound of the confidence interval at the given confidence level. This provides a more conservative estimate of the strategy's performance. For example, we can be ```97.5%``` confident that the Sharpe Ratio is at or above a given value of *x*. 

In this example, the Sharpe Ratio has negative lower bounds, and the lower bounds of the Profit Factor are less than 1, which suggests that the strategy is not reliable.

## Maximum Drawdown

In this section, we examine the maximum drawdown of the strategy using the bootstrap method. The probabilities of the drawdown not exceeding certain values, represented in cash and percentage of portfolio equity, are displayed below:

```python
result.bootstrap.drawdown_conf
```

These confidence levels were obtained using per-bar returns from the backtest's out-of-sample results, similar to how the Profit Factor and Sharpe Ratio were calculated.

We can observe that the bootstrapped max drawdown of ```-10.46%``` at a ```99.9%``` confidence level is worse than the ```-4.55%``` we saw in our original results. This highlights the importance of using randomized tests to evaluate the performance of your trading strategy.

[In the next notebook, we will discuss how to incorporate ranking and position sizing in your trading strategies](https://www.pybroker.com/en/latest/notebooks/4.%20Ranking%20and%20Position%20Sizing.html).
