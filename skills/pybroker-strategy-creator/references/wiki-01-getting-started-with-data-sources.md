# Getting Started With Data Sources

Source: `docs/source/notebooks/1. Getting Started with Data Sources.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Getting Started with Data Sources

Welcome to **PyBroker**! The best place to start is by learning about the [DataSource](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.DataSource) class. A `DataSource` fetches data from external providers, allowing you to backtest your trading strategies.

## Yahoo Finance

One of the built-in data sources in **PyBroker** is [Yahoo Finance](https://finance.yahoo.com). To use it, import [YFinance](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.YFinance):

```python
from pybroker import YFinance

yfinance = YFinance()
df = yfinance.query(
    ["AAPL", "MSFT"], start_date="3/1/2025", end_date="3/1/2026"
)
df
```

The code above queries data for AAPL and MSFT, returning the results in a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

## Caching Data

To speed up data retrieval, you can cache your queries using **PyBroker**'s caching system. Enable caching by calling [pybroker.enable_data_source_cache('name')](https://www.pybroker.com/en/latest/reference/pybroker.cache.html#pybroker.cache.enable_data_source_cache), where `name` is the identifier for your cache:

```python
import pybroker

pybroker.enable_data_source_cache("yfinance")
```

The next call to [query](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.DataSource.query) caches the returned data to disk. Each unique combination of ticker symbol and date range is cached separately:

```python
yfinance.query(["TSLA", "IBM"], "3/1/2025", "3/1/2026")
```

Calling `query` again with the same ticker symbols and date range returns the cached data:

```python
df = yfinance.query(["TSLA", "IBM"], "3/1/2025", "3/1/2026")
df
```

You can clear the cache using [pybroker.clear_data_source_cache](https://www.pybroker.com/en/latest/reference/pybroker.cache.html#pybroker.cache.clear_data_source_cache):

```python
pybroker.clear_data_source_cache()
```

Or disable caching altogether using [pybroker.disable_data_source_cache](https://www.pybroker.com/en/latest/reference/pybroker.cache.html#pybroker.cache.disable_data_source_cache):

```python
pybroker.disable_data_source_cache()
```

Note that these calls should be made after first calling [pybroker.enable_data_source_cache](https://www.pybroker.com/en/latest/reference/pybroker.cache.html#pybroker.cache.enable_data_source_cache).

## Alpaca

**PyBroker** also includes an [Alpaca](https://alpaca.markets/) ```DataSource``` for fetching stock data. To use it, you can import [Alpaca](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.Alpaca) and provide your API key and secret:

```python
from pybroker import Alpaca
import os

alpaca = Alpaca(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
```

You can query ```Alpaca``` for stock data using the same syntax as with Yahoo Finance, but Alpaca also supports querying data by different timeframes. For example, to query 1 minute data:

```python
df = alpaca.query(
    ["AAPL", "MSFT"],
    start_date="3/1/2025",
    end_date="3/1/2026",
    timeframe="1m",
)
df
```

## Alpaca Crypto

If you are interested in fetching cryptocurrency data, you can use [AlpacaCrypto](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.AlpacaCrypto). Here's an example of how to query hourly data for BTC/USD:

```python
from pybroker import AlpacaCrypto

crypto = AlpacaCrypto(
    os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"]
)
df = crypto.query(
    "BTC/USD", start_date="1/1/2025", end_date="2/1/2026", timeframe="1h"
)
df
```

## AKShare

**PyBroker** also supports an [AKShare](https://github.com/akfamily/akshare) ```DataSource``` for fetching **Chinese** stock data. AKShare, a widely-used open-source package, is tailored for obtaining financial data, with a focus on the Chinese market. This free tool provides higher quality data for the Chinese market compared to Yahoo Finance.

To use it, run `pip install akshare` and import the [AKShare](https://www.pybroker.com/en/latest/reference/pybroker.ext.data.html#pybroker.ext.data.AKShare) data source included in PyBroker:

```python
from pybroker.ext.data import AKShare

akshare = AKShare()
# You can substitute 000001.SZ with 000001, and it will still work!
# and you can set start_date as "20210301" format
# You can also set adjust to 'qfq' or 'hfq' to adjust the data,
# and set timeframe to '1d', '1w' to get daily, weekly data
df = akshare.query(
    symbols=["000001.SZ", "600000.SH"],
    start_date="3/1/2025",
    end_date="3/1/2026",
    adjust="",
    timeframe="1d",
)
df
```

**NOTE**: If the above causes a  ``Native library not available`` error and you still want to use AKShare, then [see this issue for details on how to resolve it](https://github.com/edtechre/pybroker/issues/36#issuecomment-1605910339).

## Custom Data Sources

If the built-in data sources do not cover your needs, **PyBroker** lets you implement your own [DataSource](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.DataSource). Examples include CSV files or external APIs. See [Creating a Custom Data Source](https://www.pybroker.com/en/latest/notebooks/7.%20Creating%20a%20Custom%20Data%20Source.html) for a full walkthrough.

[In the next notebook, we'll take a look at how to use DataSources to backtest a simple trading strategy](https://www.pybroker.com/en/latest/notebooks/2.%20Backtesting%20a%20Strategy.html).
