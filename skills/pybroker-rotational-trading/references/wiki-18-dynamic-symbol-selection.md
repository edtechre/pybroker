# Dynamic Symbol Selection

Source: `docs/source/notebooks/18. Dynamic Symbol Selection.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Dynamic Symbol Selection

Every strategy so far has traded a fixed list of ticker symbols chosen before the backtest began. Instead, we may want a strategy to target whichever symbols look best right now. These could be the most liquid names, or the ones with the highest momentum or value.

**PyBroker v2** supports dynamic symbol selection with a [SymbolSelector](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.SymbolSelector), as shown in this notebook.

## Loading the Candidate Universe

Below, twenty liquid large-caps are downloaded from [YFinance](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.YFinance):

```python
import pandas as pd
import numpy as np
import pybroker
from pybroker import Strategy, YFinance, highv, lowv

pybroker.enable_data_source_cache("dynamic_symbol_selection")

UNIVERSE = [
    "AAPL",
    "AMZN",
    "AVGO",
    "COST",
    "CRM",
    "GOOG",
    "JNJ",
    "JPM",
    "KO",
    "LLY",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "PG",
    "PLTR",
    "QCOM",
    "TSLA",
    "WMT",
    "XOM",
]
start_date = "1/1/2021"
end_date = "1/1/2026"
yfinance = YFinance()
df = yfinance.query(UNIVERSE, start_date=start_date, end_date=end_date)
df.head()
```

## Selecting Symbols by Liquidity

Dynamic symbol selection is done with a [SymbolSelector](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.SymbolSelector), which can be any callable that takes a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) and returns a sequence of symbols. Pass it to [add_execution](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution) in place of a fixed symbol list.

The example below ranks the universe by average dollar volume and keeps the top three symbols:

```python
TOP_N = 3


def top_dollar_volume(df: pd.DataFrame):
    dollar_volume = (df["close"] * df["volume"]).groupby(df["symbol"]).mean()
    selected = dollar_volume.nlargest(TOP_N).index
    return selected
```

## Running the Strategy

The example below implements a simple breakout strategy. It buys when a symbol closes above its previous 20-day high, calculated using [highv](https://www.pybroker.com/en/latest/reference/pybroker.vect.html#pybroker.vect.highv), and it sells when the symbol closes below its previous 20-day low, calculated using [lowv](https://www.pybroker.com/en/latest/reference/pybroker.vect.html#pybroker.vect.lowv). Those values are registered as [indicators](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.indicator) and read in the execution with [ctx.indicator](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.indicator). The strategy splits capital equally across the top three selected stocks:

```python
high_20 = pybroker.indicator("high_20", lambda data: highv(data.high, 20))
low_20 = pybroker.indicator("low_20", lambda data: lowv(data.low, 20))

POS_SIZE = 1.0 / TOP_N


def breakout(ctx):
    highs = ctx.indicator("high_20")
    lows = ctx.indicator("low_20")
    if len(highs) < 2 or np.isnan(highs[-2]):
        return
    if not ctx.long_pos():
        if ctx.close[-1] > highs[-2]:
            ctx.buy_shares = ctx.calc_target_shares(POS_SIZE)
    elif ctx.close[-1] < lows[-2]:
        ctx.sell_all_shares()


strategy = Strategy(df, start_date=start_date, end_date=end_date)
strategy.add_execution(
    breakout, top_dollar_volume, indicators=[high_20, low_20]
)
```

For each [walkforward](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.walkforward) window, the `top_dollar_volume` selector runs on the train split and selects the top three stocks to trade in the subsequent test split:

```python
result = strategy.walkforward(windows=4, train_size=0.5)
result.metrics_df.head(10)
```
