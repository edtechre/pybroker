# Multiple Time Intervals

Source: `docs/source/notebooks/15. Multiple Time Intervals.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Multiple Time Intervals

You may want to make trading decisions on a different timeframe than your underlying data source. For instance, you might choose to execute trades on daily bars after confirming the trend on weekly or monthly bars. **PyBroker v2** supports compressing backtest data into coarser intervals and making those compressed bars available to your strategy.

## Interval Types

You can define an interval using any of these three formats:

*   **Every-n-bars** (`int` greater than `1`): Compresses every `n` base bars into one bar. For example, using `5` on daily data produces one bar per five trading days.
*   **Duration** (`str`): A fixed time span written as digits followed by a single unit letter (`s`, `m`, `h`, or `d`). For example, `"5m"` compresses 1-minute bars into 5-minute bars.
*   **Calendar** (`str`): Aligns exactly to standard calendar boundaries using one of the following options:

| Calendar String | Boundary Alignment |
| :--- | :--- |
| `"daily"` | Standard daily boundary. |
| `"weekly"` | Starts on Monday. |
| `"monthly"` | Starts on the 1st of the month. |
| `"quarterly"` | Begins in January, April, July, and October. |
| `"yearly"` | Starts on January 1. |

Your chosen interval must always be strictly coarser than the bars being compressed. For example, if you fetch daily bars from [YFinance](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.YFinance), `"weekly"` and `"monthly"` are valid intervals. Attempting to use `"daily"` or `"1h"` will raise a `ValueError`.

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

The [compress_bars](https://www.pybroker.com/en/latest/reference/pybroker.interval.html#pybroker.interval.compress_bars) function converts single-symbol OHLCV data (either a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) or [BarData](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.BarData)) to a coarser interval, returning the result as a new [BarData](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.BarData) object. Every compressed bar is timestamped with the date of the last base bar it contains.

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

When building a backtest, you can declare higher timeframes by passing the `intervals` parameter to [add_execution](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution). Your execution function can then read these compressed bars through [ctx.interval](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.interval), which returns a read-only [IntervalContext](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.IntervalContext).

Both indicators and models attached to the execution are automatically computed on the base timeframe and across every declared interval. For example, an indicator with a `period=10` will instantly have a 10-week version available on the weekly bars, made accessible via the [indicator](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.IntervalContext.indicator) method on that context (`ctx.interval("weekly").indicator(...)`).

To prevent look-ahead bias, [ctx.interval](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.interval) only ever exposes *completed* bars. The week or month that is currently forming is never visible, ensuring that future data cannot leak into your daily trading decisions.

To demonstrate this, we will execute trades on daily bars and assign a specific job to each higher timeframe:

*   **Monthly (Regime):** Only enter when the last completed monthly close is higher than the close from three months ago.
*   **Weekly (Trend):** Only enter when the last completed weekly close is above its 10-week moving average, and exit when it falls below that average.
*   **Daily (Timing):** Enter on the first daily close that crosses above the weekly moving average:

```python
from pybroker.vect import sumv


sma_10 = pybroker.indicator("sma_10", lambda data: sumv(data.close, 10) / 10)


def buy_with_trend(ctx):
    weekly = ctx.interval("weekly")
    monthly = ctx.interval("monthly")
    # Wait until enough completed weekly and monthly bars exist.
    if len(weekly.close) == 0 or len(monthly.close) < 4:
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
    buy_with_trend,
    ["AMD", "NVDA", "INTC"],
    indicators=sma_10,
    intervals=["weekly", "monthly"],
)
result = strategy.backtest(timeframe="1d")
result.metrics_df.head(20)
```
