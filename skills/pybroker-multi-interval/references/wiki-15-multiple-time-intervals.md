# Multiple Time Intervals

Source: `docs/source/notebooks/15. Multiple Time Intervals.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Multiple Time Intervals

You may want to make trading decisions on a different timeframe than your underlying data source. For instance, you might choose to execute trades on daily bars after confirming the trend on weekly or monthly bars. **PyBroker v2** supports compressing backtest data into longer intervals and making those compressed bars available to your strategy.

## Interval Types

You can define an interval using any of these three formats:

*   **Every-n-bars** (`int` greater than `1`): Compresses every `n` base bars into one bar. Using `5` on daily data produces one bar per five trading days.
*   **Duration** (`str`): A fixed time span written as digits followed by a single unit letter (`s`, `m`, `h`, or `d`). Passing `"5m"` compresses 1-minute bars into 5-minute bars.
*   **Calendar** (`str`): Aligns compressed bars to calendar boundaries using one of the following options:

| Calendar String | Boundary Alignment |
| :--- | :--- |
| `"daily"` | Standard daily boundary. |
| `"weekly"` | Starts on Monday. |
| `"monthly"` | Starts on the 1st of the month. |
| `"quarterly"` | Begins in January, April, July, and October. |
| `"yearly"` | Starts on January 1. |

Your chosen interval must always be longer than the bars being compressed. For example, if you fetch daily bars from [YFinance](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.YFinance), then `"weekly"` and `"monthly"` are valid intervals. Attempting to use `"daily"` or `"1h"` will raise a `ValueError`.

Before using intervals in a strategy, let's build some intuition by compressing bars directly. We will start by downloading daily data:

```python
import pybroker
from pybroker import Strategy, YFinance

pybroker.enable_data_source_cache("multiple_time_intervals")

yfinance = YFinance()
df = yfinance.query(
    ["AMD", "NVDA", "INTC"], start_date="1/1/2021", end_date="1/1/2026"
)
df.head()
```

## Compressing Bars

The [compress_bars](https://www.pybroker.com/en/latest/reference/pybroker.interval.html#pybroker.interval.compress_bars) function converts OHLCV data (either a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) or [BarData](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.BarData)) to a longer interval, returning the result as a new [BarData](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.BarData) object. Every compressed bar is timestamped with the date of the last base bar it contains.

When grouping base bars into a compressed bar, the data is aggregated as follows:
*   **Open:** Taken from the first base bar.
*   **High / Low:** The highest high and lowest low.
*   **Close:** Taken from the last base bar.
*   **Volume:** The sum of the volumes.
*   **VWAP:** The volume-weighted average.
*   **Custom columns:** The last value in the period (e.g., [YFinance](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.YFinance)'s `adj_close`).

You must also supply the `base_timeframe` parameter to declare the spacing of your input bars (for example, `"1d"` for daily data). 

Let's compress AMD into calendar weeks and view the result as a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with [bars_to_df](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.bars_to_df):

```python
from pybroker import compress_bars
from pybroker.common import bars_to_df


amd_df = df[df["symbol"] == "AMD"]
bars_to_df(compress_bars(amd_df, "weekly", base_timeframe="1d")).head()
```

Every-n-bars compression works the same way. In this example, every `5` daily bars become one bar:

```python
bars_to_df(compress_bars(amd_df, 5, base_timeframe="1d")).head()
```

## A Multi-Timeframe Strategy

To use higher timeframes in your backtest, pass the `intervals` parameter to [add_execution](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution). Your execution function can then access the compressed bars through [ctx.interval](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.interval), which returns a read-only [IntervalContext](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.IntervalContext). 

Using the `intervals` parameter provides compressed bars only. Indicators and models are never computed on these intervals unless you bind them explicitly, as shown later in this notebook.

To prevent look-ahead bias, [ctx.interval](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.interval) only ever exposes *completed* bars. For example, the week or month that is currently forming is never visible, ensuring that future data cannot leak into your daily trading decisions.

In the following strategy, we will execute trades on daily bars while using longer intervals to generate different trading signals:

*   **Monthly (Regime):** Only enter when the last completed monthly close is higher than the close from three months ago.
*   **Weekly (Trend):** Only enter when the last completed weekly close is higher than the close from ten weeks ago, and exit when it falls below.
*   **Daily (Timing):** Enter on the first daily close above the last completed weekly close.

```python
def buy_with_trend(ctx):
    weekly = ctx.interval("weekly")
    monthly = ctx.interval("monthly")
    # Wait until enough completed weekly and monthly bars exist.
    if len(weekly.close) < 10 or len(monthly.close) < 4:
        return
    regime_up = monthly.close[-1] > monthly.close[-4]
    trend_up = weekly.close[-1] > weekly.close[-10]
    pos = ctx.long_pos()
    if not pos and regime_up and trend_up and ctx.close[-1] > weekly.close[-1]:
        ctx.buy_shares = 100
    elif pos and not trend_up:
        ctx.sell_all_shares()


strategy = Strategy(yfinance, start_date="1/1/2021", end_date="1/1/2026")
strategy.add_execution(
    buy_with_trend,
    ["AMD", "NVDA", "INTC"],
    intervals=["weekly", "monthly"],
)
result = strategy.backtest(timeframe="1d")
result.metrics_df.head(20)
```

## Binding an Indicator to an Interval

To compute an indicator on compressed bars, bind it to one or more intervals with [Indicator.intervals(...)](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.Indicator.intervals).

The example below updates the weekly trend rule to compare the weekly close against a 10-bar SMA calculated from the weekly bars:

```python
from pybroker.vect import sumv

sma_10 = pybroker.indicator("sma_10", lambda data: sumv(data.close, 10) / 10)


def buy_with_indicator(ctx):
    weekly = ctx.interval("weekly")
    monthly = ctx.interval("monthly")
    # Wait until enough completed weekly and monthly bars exist.
    if len(weekly.close) < 10 or len(monthly.close) < 4:
        return
    wk_sma = weekly.indicator("sma_10")
    regime_up = monthly.close[-1] > monthly.close[-4]
    trend_up = weekly.close[-1] > wk_sma[-1]
    pos = ctx.long_pos()
    if not pos and regime_up and trend_up and ctx.close[-1] > wk_sma[-1]:
        ctx.buy_shares = 100
    elif pos and not trend_up:
        ctx.sell_all_shares()


strategy = Strategy(yfinance, start_date="1/1/2021", end_date="1/1/2026")
strategy.add_execution(
    buy_with_indicator,
    ["AMD", "NVDA", "INTC"],
    indicators=sma_10.intervals("weekly"),
    intervals="monthly",
)
```

**PyBroker** automatically combines bound intervals with those in the execution's `intervals` parameter. Here, the `"weekly"` interval is made accessible via [ctx.interval("weekly")](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.interval) and the `intervals` parameter only needs to specify `"monthly"` for the raw monthly bars.

Note that the binding will override also computing the indicator on the base timeframe of the data source. To also compute the indicator on the base timeframe of the data source, pass `"base"` to [Indicator.intervals()](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.Indicator.intervals).

```python
result = strategy.backtest(timeframe="1d")
result.metrics_df.head(20)
```

## Training a Model on an Interval

You can bind models in the same way with [ModelSource.intervals(...)](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.ModelSource.intervals). **PyBroker** will then train models for each interval using the interval's compressed bars and any registered indicators. You can then access the per-interval predictions by calling the [preds](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.IntervalContext.preds) method on that interval's context.

This example trains a [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) model to predict the next weekly return from the weekly close:

```python
from sklearn.linear_model import LinearRegression


def train_weekly(symbol, train_data, test_data):
    # Predict the next weekly return from the weekly close.
    returns = train_data["close"].pct_change().shift(-1)
    train_rows = train_data.assign(pred=returns).dropna()
    model = LinearRegression()
    model.fit(train_rows[["close"]], train_rows[["pred"]])
    return model, ["close"]


model_weekly = pybroker.model("weekly_slr", train_weekly)


def hold_with_model(ctx):
    preds = ctx.interval("weekly").preds("weekly_slr")
    if len(preds) == 0:
        return
    if not ctx.long_pos():
        if preds[-1] > 0:
            ctx.buy_shares = 100
    elif preds[-1] < 0:
        ctx.sell_all_shares()


strategy = Strategy(yfinance, start_date="1/1/2021", end_date="1/1/2026")
strategy.add_execution(
    hold_with_model,
    ["AMD", "NVDA", "INTC"],
    models=model_weekly.intervals("weekly"),
)
```

During [Walkforward Analysis](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.walkforward), the `lookahead` between an interval model's train and test data is enforced using the compressed bar units in order to prevent future leakage.

```python
result = strategy.walkforward(windows=3, train_size=0.5, timeframe="1d")
result.metrics_df.head(20)
```
