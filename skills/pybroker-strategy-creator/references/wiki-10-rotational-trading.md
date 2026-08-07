# Rotational Trading

Source: `docs/source/notebooks/10. Rotational Trading.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Rotational Trading

Rotational trading involves purchasing the best-performing assets while selling underperforming ones. As you may have guessed, **PyBroker** is an excellent tool for backtesting such strategies. So, let's dive in and get started with testing our rotational trading strategy!

```python
import pybroker as pyb
from pybroker import ExecContext, Strategy, StrategyConfig, YFinance
```

Our strategy will involve ranking and buying stocks with the highest [price rate-of-change (ROC)](https://www.investopedia.com/terms/p/pricerateofchange.asp). To start, we'll define a 20-day ROC indicator using [TA-Lib](https://github.com/TA-Lib/ta-lib-python):

```python
import talib as ta

roc_20 = pyb.indicator(
    'roc_20', lambda data: ta.ROC(data.adj_close, timeperiod=20))
```

Next, let's define the rules of our strategy:

- Buy the two stocks with the highest 20-day ROC.
- Allocate 50% of our capital to each stock.
- If either of the stocks is no longer ranked among the top five 20-day ROCs, then we will liquidate that stock.
- Trade these rules daily.

Let's set up our config and some parameters for the above rules:

```python
config = StrategyConfig(max_long_positions=2)
pyb.param('target_size', 1 / config.max_long_positions)
pyb.param('rank_threshold', 5)
```

To proceed with our strategy, we will implement a ``rank`` function that ranks each stock by their 20-day ROC in descending order, from highest to lowest.

```python
def rank(ctxs: dict[str, ExecContext]):
    scores = {
        symbol: ctx.indicator('roc_20')[-1]
        for symbol, ctx in ctxs.items()
    }
    sorted_scores = sorted(
        scores.items(), 
        key=lambda score: score[1],
        reverse=True
    )
    threshold = pyb.param('rank_threshold')
    top_scores = sorted_scores[:threshold]
    top_symbols = [score[0] for score in top_scores]
    pyb.param('top_symbols', top_symbols)
```

The ``top_symbols`` global parameter contains the symbols of the stocks with the top five highest 20-day ROCs.

Now that we have a method for ranking stocks by their ROC, we can proceed with implementing a ``rotate`` function to manage the rotational trading.

```python
def rotate(ctx: ExecContext):
    if ctx.long_pos():
        if ctx.symbol not in pyb.param('top_symbols'):
            ctx.sell_all_shares()
    else:
        target_size = pyb.param('target_size')
        ctx.buy_shares = ctx.calc_target_shares(target_size)
        ctx.score = ctx.indicator('roc_20')[-1]
```

We liquidate the currently held stock if it is no longer ranked among the top five 20-day ROCs. Otherwise, we rank all stocks by their 20-day ROC and buy up to the top two ranked. For more information on ranking when placing buy orders, see the [Ranking and Position Sizing notebook](https://www.pybroker.com/en/latest/notebooks/4.%20Ranking%20and%20Position%20Sizing.html).

We will use the [set_before_exec](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_before_exec) method to execute our ranking with ``rank`` before running the ``rotate`` function. For this backtest, we will use a universe of 10 stocks:

```python
strategy = Strategy(
    YFinance(), 
    start_date='1/1/2018', 
    end_date='1/1/2023', 
    config=config
)
strategy.set_before_exec(rank)
strategy.add_execution(rotate, [
    'TSLA', 
    'NFLX', 
    'AAPL', 
    'NVDA', 
    'AMZN',
    'MSFT',
    'GOOG',
    'AMD',
    'INTC',
    'META'
], indicators=roc_20)
result = strategy.backtest(warmup=20)
```

```python
result.orders
```
