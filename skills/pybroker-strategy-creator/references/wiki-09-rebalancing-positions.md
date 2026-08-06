# Rebalancing Positions

Source: `docs/source/notebooks/9. Rebalancing Positions.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Rebalancing Positions

**PyBroker** allows you to simulate portfolio rebalancing by adjusting your asset allocation to match a desired target. This notebook also demonstrates how to rebalance using [portfolio optimization](https://en.wikipedia.org/wiki/Portfolio_optimization).

```python
import pybroker
from pybroker import ExecContext, Strategy, YFinance

pybroker.enable_data_source_cache("rebalancing")
```

## Equal Position Sizing

Suppose we want to rebalance a long-only portfolio at the beginning of every month to maintain an equal allocation for each stock. 

First, we write a helper function to detect when the current bar is the start of a new month:

```python
def start_of_month(ctxs: dict[str, ExecContext]) -> bool:
    dt = tuple(ctxs.values())[0].dt
    if dt.month != pybroker.param("current_month"):
        pybroker.param("current_month", dt.month)
        return True
    return False
```

Next, we write a `rebalance` function to set an equal target allocation for each asset at the beginning of every month:

```python
def rebalance(ctxs: dict[str, ExecContext]):
    if start_of_month(ctxs):
        target = 1 / len(ctxs)
        for ctx in ctxs.values():
            ctx.set_target_shares(target, dir="long")
```

With the `rebalance` function complete, we can backtest our strategy using a portfolio of five stocks. To process all stocks simultaneously on each bar of data, we use the [Strategy.set_after_exec](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_after_exec) method:

```python
strategy = Strategy(YFinance(), start_date="1/1/2018", end_date="1/1/2023")
strategy.add_execution(None, ["TSLA", "NFLX", "AAPL", "NVDA", "AMZN"])
strategy.set_after_exec(rebalance)
result = strategy.backtest()
```

The `set_after_exec` function runs after all executions added via `add_execution`. Since we passed `None` to `add_execution`, no execution logic runs prior to `after_exec`.

```python
result.orders
```

## Portfolio Optimization

[Portfolio optimization](https://en.wikipedia.org/wiki/Portfolio_optimization) guides rebalancing to meet specific objectives, such as allocating stocks to minimize risk.

[Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/) is a popular Python library for portfolio optimization. You can install it using `pip install riskfolio-lib`.

The following example demonstrates how to construct a minimum risk portfolio by minimizing the [Conditional Value at Risk (CVaR)](https://www.investopedia.com/terms/c/conditional_value_at_risk.asp) based on the past year of returns:

```python
import pandas as pd
import riskfolio as rp

pybroker.param("lookback", 252)  # Use past year of returns.


def calculate_returns(ctxs: dict[str, ExecContext], lookback: int):
    prices = {}
    for ctx in ctxs.values():
        prices[ctx.symbol] = ctx.adj_close[-lookback:]
    df = pd.DataFrame(prices)
    return df.pct_change().dropna()


def optimization(ctxs: dict[str, ExecContext]):
    lookback = pybroker.param("lookback")
    if start_of_month(ctxs):
        Y = calculate_returns(ctxs, lookback)
        port = rp.Portfolio(returns=Y)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w = port.optimization(
            model="Classic",
            rm="CVaR",
            obj="MinRisk",
            rf=0,  # Risk free rate.
            l=0,  # Risk aversion factor.
            hist=True,  # Use historical scenarios.
        )
        for symbol, ctx in ctxs.items():
            target = w.T[symbol].values[0]
            ctx.set_target_shares(target, dir="long")
```

For more information and examples, see the [official Riskfolio-Lib documentation](https://riskfolio-lib.readthedocs.io/). Next, we backtest the strategy:

```python
strategy.set_after_exec(optimization)
result = strategy.backtest(warmup=pybroker.param("lookback"))
```

```python
result.orders.head()
```

The portfolio optimization allocated the entire portfolio to `AAPL`, `AMZN`, and `TSLA` during the first month of the backtest.
