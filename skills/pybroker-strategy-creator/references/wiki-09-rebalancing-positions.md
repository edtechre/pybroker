# Rebalancing Positions

Source: `docs/source/notebooks/9. Rebalancing Positions.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Rebalancing Positions

**PyBroker** can be used to simulate rebalancing a portfolio. This means that **PyBroker** can simulate adjusting the asset allocation of a portfolio to match a desired target allocation. Additionally, our portfolio can be rebalanced using [portfolio optimization](https://en.wikipedia.org/wiki/Portfolio_optimization), as this notebook will demonstrate.

```python
import pybroker as pyb
from datetime import datetime
from pybroker import ExecContext, Strategy, YFinance

pyb.enable_data_source_cache('rebalancing')
```

## Equal Position Sizing

Let's assume that we want to rebalance a long-only portfolio at the beginning of every month to make sure that each stock in our portfolio has a roughly equal allocation.

We first start by writing a helper function to detect when the current bar's date is the start of a new month:

```python
def start_of_month(ctxs: dict[str, ExecContext]) -> bool:
    dt = tuple(ctxs.values())[0].dt
    if dt.month != pyb.param('current_month'):
        pyb.param('current_month', dt.month)
        return True
    return False
```

Next, we implement a function that will either buy or sell enough shares of a stock to reach a target allocation.

```python
def set_target_shares(
    ctxs: dict[str, ExecContext], 
    targets: dict[str, float]
):
    for symbol, target in targets.items():
        ctx = ctxs[symbol]
        target_shares = ctx.calc_target_shares(target)
        pos = ctx.long_pos()
        if pos is None:
            ctx.buy_shares = target_shares
        elif pos.shares < target_shares:
            ctx.buy_shares = target_shares - pos.shares
        elif pos.shares > target_shares:
            ctx.sell_shares = pos.shares - target_shares
```

If the current allocation is above the target level, the function will sell the needed shares of the asset, while if the current allocation is below the target level, the function will buy the needed shares of the asset.

Following that, we write a ``rebalance`` function to target each asset to an equal allocation at the beginning of every month:

```python
def rebalance(ctxs: dict[str, ExecContext]):
    if start_of_month(ctxs):
        target = 1 / len(ctxs)
        set_target_shares(ctxs, {symbol: target for symbol in ctxs.keys()})
```

Now that we have implemented the ``rebalance`` function, the next step is to backtest our rebalancing strategy using a portfolio of five stocks. To process all stocks at once on each bar of data, we will use the [Strategy#set_after_exec](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_after_exec) method:

```python
strategy = Strategy(YFinance(), start_date='1/1/2018', end_date='1/1/2023')
strategy.add_execution(None, ['TSLA', 'NFLX', 'AAPL', 'NVDA', 'AMZN'])
strategy.set_after_exec(rebalance)
result = strategy.backtest()
```

```python
result.orders
```

## Portfolio Optimization

[Portfolio optimization](https://en.wikipedia.org/wiki/Portfolio_optimization) can guide our rebalancing in order to meet some objective for our portfolio. For instance, we can use portfolio optimization with the goal of allocating stocks in a way to minimize risk.

[Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/) is a popular Python library for performing portfolio optimization. Below shows how to use it to construct a minimum risk portfolio by minimizing the portfolio's [Conditional Value at Risk (CVar)](https://www.investopedia.com/terms/c/conditional_value_at_risk.asp) based on the past year of returns:

```python
import pandas as pd
import riskfolio as rp

pyb.param('lookback', 252)  # Use past year of returns.

def calculate_returns(ctxs: dict[str, ExecContext], lookback: int):
    prices = {}
    for ctx in ctxs.values():
        prices[ctx.symbol] = ctx.adj_close[-lookback:]
    df = pd.DataFrame(prices)
    return df.pct_change().dropna()

def optimization(ctxs: dict[str, ExecContext]):
    lookback = pyb.param('lookback')
    if start_of_month(ctxs):
        Y = calculate_returns(ctxs, lookback)
        port = rp.Portfolio(returns=Y)
        port.assets_stats(method_mu='hist', method_cov='hist')
        w = port.optimization(
            model='Classic', 
            rm='CVaR', 
            obj='MinRisk', 
            rf=0,      # Risk free rate.
            l=0,       # Risk aversion factor.
            hist=True  # Use historical scenarios.
        )
        targets = {
            symbol: w.T[symbol].values[0]
            for symbol in ctxs.keys()
        }
        set_target_shares(ctxs, targets)
```

You can find more information and examples of using [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/) on its official documentation. Now, let's move on to backtesting the strategy!

```python
strategy.set_after_exec(optimization)
result = strategy.backtest(warmup=pyb.param('lookback'))
```

```python
result.orders.head()
```

Above, we can observe that the portfolio optimization resulted in allocating the entire portfolio to 3 of the 5 stocks during the first month of the backtest.
