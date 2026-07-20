# Writing Indicators

Source: `docs/source/notebooks/5. Writing Indicators.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Writing Indicators

This notebook explains how to create and integrate custom stock indicators in **PyBroker**.  Indicators in **PyBroker** are written using [NumPy](https://numpy.org/), a powerful library for numerical computing. To optimize performance, we'll also be utilizing [Numba](https://numba.pydata.org/), a JIT compiler that translates Python code into efficient machine code. Numba is especially helpful for accelerating code that involves loops and NumPy arrays. Here's how we import these libraries:

```python
import numpy as np
from numba import njit
```

The following code shows an indicator function that calculates close prices minus a moving average (CMMA), which can be used for a [mean reversion](https://en.wikipedia.org/wiki/Mean_reversion_(finance)) strategy:

```python
def cmma(bar_data, lookback):

    @njit  # Enable Numba JIT.
    def vec_cmma(values):
        # Initialize the result array.
        n = len(values)
        out = np.array([np.nan for _ in range(n)])
        
        # For all bars starting at lookback:
        for i in range(lookback, n):
            # Calculate the moving average for the lookback.
            ma = 0
            for j in range(i - lookback, i):
                ma += values[j]
            ma /= lookback
            # Subtract the moving average from value.
            out[i] = values[i] - ma
        return out
    
    # Calculate with close prices.
    return vec_cmma(bar_data.close)
```

The ```cmma``` function takes two arguments: ```bar_data```, which is an instance of the [BarData](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.BarData) class that holds OHLCV data and custom fields, and ```lookback```, which is a user-defined argument for the lookback of the moving average.

The ```vec_cmma``` function is JIT-compiled by Numba and nested inside ```cmma```. This is necessary since a Numba compiled function supports a NumPy array as an argument but not an instance of a Python class like ```BarData```. Note the computation of the indicator values is [vectorized](https://en.wikipedia.org/wiki/Array_programming) by Numba, meaning that it's performed on all of the historical data at once. This approach significantly speeds up the backtesting process.

The next step is to register the indicator function with **PyBroker** using the following code:

```python
import pybroker

cmma_20 = pybroker.indicator('cmma_20', cmma, lookback=20)
```

Here, we are giving the name ```cmma_20``` to the indicator function and specifying the ```lookback``` parameter as ```20``` bars. Any arguments in the indicator function that come after ```bar_data``` will be passed as user-defined arguments to [pybroker.indicator](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.indicator). Once the indicator function is registered with **PyBroker**, it will return a new [Indicator](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.Indicator) instance that references the indicator function we defined.

The following is an example of how to use the registered ```Indicator``` in **PyBroker** with some data downloaded from [Yahoo Finance](https://finance.yahoo.com):

```python
from pybroker import YFinance

pybroker.enable_data_source_cache('yfinance')

yfinance = YFinance()
df = yfinance.query('PG', '4/1/2020', '4/1/2022')
```

```python
cmma_20(df)
```

As you can see, the ```Indicator``` instance is a ```Callable```. Once called, the resulting computed indicator values are returned as a [Pandas Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html).

The ```Indicator``` class also provides functions for measuring its information content. For example, you can compute the [interquartile range (IQR)](https://en.wikipedia.org/wiki/Interquartile_range):

```python
cmma_20.iqr(df)
```

Or compute the relative [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)):

```python
cmma_20.relative_entropy(df)
```

## Using the Indicator in a Strategy

After implementing our indicator, the next step is to integrate it into a trading strategy. The following example shows a simple strategy that goes long when the 20-day CMMA is less than 0 - i.e. when the last close price drops below the 20-day moving average:

```python
def buy_cmma_cross(ctx):
    if ctx.long_pos():
        return
    # Place a buy order if the most recent value of the 20 day CMMA is < 0:
    if ctx.indicator('cmma_20')[-1] < 0:
        ctx.buy_shares = ctx.calc_target_shares(1)
        ctx.hold_bars = 3
```

The indicator values are retrieved by calling [ctx.indicator](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.BaseContext.indicator) on the [ExecContext](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext) and passing in the registered name of the ```cmma_20``` indicator.

(Note, you can also retrieve indicator data for another symbol by passing the symbol to [ExecContext#indicator()](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.indicator))

```python
from pybroker import Strategy

strategy = Strategy(yfinance, '4/1/2020', '4/1/2022')
strategy.add_execution(buy_cmma_cross, 'PG', indicators=cmma_20)
```

Here, the ```buy_cmma_cross``` function is added to the [Strategy](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy) along with the ```cmma_20``` indicator. We can enable caching of the computed indicator values to disk with the following:

```python
pybroker.enable_indicator_cache('my_indicators')
```

Finally, we can run the backtest with the following code. The ``warmup`` argument specifies that 20 bars need to pass before running the backtest execution:

```python
result = strategy.backtest(warmup=20)
result.metrics_df.round(4)
```

When the backtest runs, **PyBroker** computes the indicator values. If there are multiple indicators added to the ```Strategy```, then **PyBroker** will compute them in parallel across multiple CPU cores.

## Vectorized Helpers

The **PyBroker** library provides vectorized helper functions to make the process of computing indicators easier. One of these helper functions is [highv](https://www.pybroker.com/en/latest/reference/pybroker.vect.html#pybroker.vect.highv), which calculates the highest value for every period of *n* bars.

In the example code, an indicator function called ```hhv``` is defined that uses ```highv``` to calculate the *highest* high price for every period of 5 bars:

```python
from pybroker import highv

def hhv(bar_data, period):
    return highv(bar_data.high, period)

hhv_5 = pybroker.indicator('hhv_5', hhv, period=5)
hhv_5(df)
```

The [pybroker.vect](https://www.pybroker.com/en/latest/reference/pybroker.vect.html) module also includes other vectorized helpers such as [lowv](https://www.pybroker.com/en/latest/reference/pybroker.vect.html#pybroker.vect.lowv), [sumv](https://www.pybroker.com/en/latest/reference/pybroker.vect.html#pybroker.vect.sumv), [returnv](https://www.pybroker.com/en/latest/reference/pybroker.vect.html#pybroker.vect.returnv), and [cross](https://www.pybroker.com/en/latest/reference/pybroker.vect.html#pybroker.vect.cross), the last of which is used to compute crossovers.

Additionally, **PyBroker** includes convenient wrappers for [highest](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.highest) and [lowest](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.lowest) indicators. Our ``hhv`` indicator can be rewritten as:

```python
from pybroker import highest

hhv_5 = highest('hhv_5', 'high', period=5)
hhv_5(df)
```

## Computing Multiple Indicators

An [IndicatorSet](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.IndicatorSet) can be used to calculate multiple indicators. The ```cmma_20``` and ```hhv_5``` indicators can be computed together by adding them to the ```IndicatorSet```. The resulting output will be a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) containing both:

```python
from pybroker import IndicatorSet

indicator_set = IndicatorSet()
indicator_set.add(cmma_20, hhv_5)
indicator_set(df)
```

## Using TA-Lib

[TA-Lib](https://github.com/TA-Lib/ta-lib-python) is a widely used technical analysis library that implements many financial indicators. Integrating TA-Lib with **PyBroker** is straightforward. Here is an example:

```python
import talib

rsi_20 = pybroker.indicator('rsi_20', lambda data: talib.RSI(data.close, timeperiod=20))
rsi_20(df)
```

## Built-In Indicators

PyBroker also includes built-in indicators that are available in the [indicator module](https://www.pybroker.com/en/dev/reference/pybroker.indicator.html).

[In the next tutorial, you will learn how to train a model using custom indicators in PyBroker](https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html).
