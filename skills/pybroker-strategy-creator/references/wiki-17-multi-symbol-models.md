# Multi-Symbol Models

Source: `docs/source/notebooks/17. Multi-Symbol Models.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Multi-Symbol Models

The models trained so far have trained a separate instance for every ticker symbol. Symbols that share the same dynamics, such as stocks in a single industry, can instead be modeled jointly. This approach provides a single model with much more data to learn from.

**PyBroker v2** supports training one model across all symbols passed to [add_execution](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution), as shown in this notebook.

```python
import pybroker
from pybroker import Strategy, YFinance
from sklearn.linear_model import LinearRegression

pybroker.enable_data_source_cache("multi_symbol_models")
```

## Training One Model on Multiple Symbols

This notebook repurposes the linear regression example from [Training a Model](https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html). Below, one shared [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) model is trained with the [close_minus_ma](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.close_minus_ma) indicator on four chip stocks:

```python
from pybroker.indicator import close_minus_ma

cmma_20 = close_minus_ma("cmma_20", lookback=20, atr_length=14)


def train_slr(symbols, train_data, test_data):
    # Shift within symbols so returns never cross a symbol boundary.
    next_close = train_data.groupby("symbol")["close"].shift(-1)
    train_data["target"] = next_close / train_data["close"] - 1
    train_data = train_data.dropna()
    model = LinearRegression()
    model.fit(train_data[["cmma_20"]], train_data["target"])
    return model, ["cmma_20"]


model_slr = pybroker.model("slr", train_slr, indicators=[cmma_20], pooled=True)

SYMBOLS = ["MU", "TXN", "ADI", "AMAT"]
```

Registering the model with `pooled=True` via [model](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.model) trains it only once per execution. When you enable this, the training function replaces its single `symbol` argument with a tuple `symbols` and receives their combined train and test splits.

During the backtest, the trained model is shared across symbols:

```python
POS_SIZE = 1 / len(SYMBOLS)


def hold_long(ctx):
    pred = ctx.preds("slr")[-1]
    if not ctx.long_pos():
        if pred > 0:
            ctx.buy_shares = ctx.calc_target_shares(POS_SIZE)
    elif pred < 0:
        ctx.sell_all_shares()


strategy = Strategy(YFinance(), start_date="1/1/2021", end_date="1/1/2026")
strategy.add_execution(hold_long, SYMBOLS, models=model_slr)
result = strategy.walkforward(
    warmup=20, windows=3, train_size=0.5, lookahead=1
)
result.metrics_df.head(20)
```
