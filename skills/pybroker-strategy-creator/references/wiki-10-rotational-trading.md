# Rotational Trading

Source: `docs/source/notebooks/10. Rotational Trading.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Rotational Trading

Rotational trading involves buying top-performing assets and selling underperforming ones. **PyBroker** can be used for backtesting these strategies.

```python
import pybroker
from pybroker import ExecContext, Strategy, StrategyConfig, YFinance
```

Our strategy will involve ranking and buying stocks with the highest [price rate-of-change (ROC)](https://www.investopedia.com/terms/p/pricerateofchange.asp). To start, we'll define a 20-day ROC indicator using [TA-Lib](https://github.com/TA-Lib/ta-lib-python):

```python
import talib as ta

roc_20 = pybroker.indicator(
    "roc_20", lambda data: ta.ROC(data.adj_close, timeperiod=20)
)
```

Next, let's define the rules of our strategy:

- Buy the two stocks with the highest 20-day ROC.
- Allocate 50% of our capital to each stock.
- If either of the stocks is no longer ranked among the top five 20-day ROCs, then we will liquidate that stock.
- Trade these rules daily.

Let’s set up our config for the above rules:

```python
config = StrategyConfig(max_long_positions=2)
```

To implement the strategy, we write a `rotate` function that sets each stock's [long_score](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.long_score) to its 20-day ROC. **PyBroker** then ranks the stocks by their `long_score` in descending order.

```python
def rotate(ctx: ExecContext):
    ctx.long_score = ctx.indicator("roc_20")[-1]
```

Now that we have a method for scoring stocks by their ROC, we can use the [enable_rotation](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.enable_rotation) method to turn on rotational trading.

Setting `worst_rank_held` to `5` liquidates any currently held stock that falls outside the top five 20-day ROC rankings. Otherwise, **PyBroker** buys up to the top two ranked stocks, allocating 50% of the capital to each. This backtest uses a universe of 10 stocks:

```python
strategy = Strategy(
    YFinance(), start_date="1/1/2018", end_date="1/1/2023", config=config
)
strategy.enable_rotation(worst_rank_held=5)
strategy.add_execution(
    rotate,
    [
        "TSLA",
        "NFLX",
        "AAPL",
        "NVDA",
        "AMZN",
        "MSFT",
        "GOOG",
        "AMD",
        "INTC",
        "META",
    ],
    indicators=roc_20,
)
result = strategy.backtest(warmup=20)
```

```python
result.orders
```

## Custom Position Sizing

**PyBroker** will allocate our capital equally between positions by default. To customize this, we can pass a ``sizer`` function to [enable_rotation](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.enable_rotation). The ``sizer`` is called with a [RotationContext](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.RotationContext) after rotation has decided which stocks to buy, allowing us to override the size of each entry. The ``long_ranks`` attribute contains the rank of each stock, where ``1`` is the highest ranked.

Let's reuse our strategy, but this time allocate 70% of our capital to the top-ranked stock and 30% to the second:

```python
from pybroker import RotationContext


def size_by_rank(rotation: RotationContext):
    weights = {1: 0.7, 2: 0.3}
    for symbol, ctx in rotation.ctxs.items():
        if ctx.buy_shares is not None:
            rank = rotation.long_ranks[symbol]
            ctx.buy_shares = ctx.calc_target_shares(weights[rank])


strategy.enable_rotation(worst_rank_held=5, sizer=size_by_rank)
result = strategy.backtest(warmup=20)
result.orders
```
