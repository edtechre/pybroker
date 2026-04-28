# Creating A Custom Data Source

Source: `docs/source/notebooks/7. Creating a Custom Data Source.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Creating a Custom Data Source

**PyBroker** comes with pre-built [DataSources](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.DataSource) for [Yahoo Finance](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.YFinance), [Alpaca](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.Alpaca), and [AKShare](https://github.com/akfamily/akshare), which you can use right away without any additional setup. But if you have a specific need or want to use a different data source, **PyBroker** also allows you to create your own ```DataSource``` class.


## Extending DataSource

In the example code provided below, a new ```DataSource``` called ```CSVDataSource``` is implemented, which loads data from a CSV file. The ```CSVDataSource``` reads a file named ```prices.csv``` into a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), and then returns the data from this DataFrame based on the input parameters provided:

```python
import pandas as pd
import pybroker
from pybroker.data import DataSource

class CSVDataSource(DataSource):
    
    def __init__(self):
        super().__init__()
        # Register custom columns in the CSV.
        pybroker.register_columns('rsi')
    
    def _fetch_data(self, symbols, start_date, end_date, _timeframe, _adjust):
        df = pd.read_csv('data/prices.csv')
        df = df[df['symbol'].isin(symbols)]
        df['date'] = pd.to_datetime(df['date'])
        return df[(df['date'] >= start_date) & (df['date'] <= end_date)]
```

To make the custom ```'rsi'``` column from the CSV file available to **PyBroker**, we register it using [pybroker.register_columns](https://www.pybroker.com/en/latest/reference/pybroker.scope.html#pybroker.scope.register_columns). This allows **PyBroker** to use this custom column when it processes the data.

It's important to note that when returning the data from your custom DataSource, it must include the following columns: ```symbol```, ```date```, ```open```, ```high```, ```low```, and ```close```, as these columns are expected by **PyBroker**.

Now we can query the CSV data from an instance of ```CSVDataSource```:

```python
csv_data_source = CSVDataSource()
df = csv_data_source.query(['MCD', 'NKE', 'DIS'], '6/1/2021', '12/1/2021')
df
```

To use ```CSVDataSource``` in a backtest, we create a new [Strategy](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy) object and pass the custom ```DataSource```:

```python
from pybroker import Strategy

def buy_low_sell_high_rsi(ctx):
    pos = ctx.long_pos() 
    if not pos and ctx.rsi[-1] < 30:
        ctx.buy_shares = 100
    elif pos and ctx.rsi[-1] > 70:
        ctx.sell_shares = pos.shares

strategy = Strategy(csv_data_source, '6/1/2021', '12/1/2021')
strategy.add_execution(buy_low_sell_high_rsi, ['MCD', 'NKE', 'DIS'])
result = strategy.backtest()
result.orders
```

Note that because we registered the custom ```rsi``` column with **PyBroker**, it can be accessed in the [ExecContext](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext) using ```ctx.rsi```.

## Using a Pandas DataFrame

If you do not need the flexibility of implementing your own [DataSource](https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.DataSource), then you can pass a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) to a ``Strategy`` instead.

To demonstrate, the earlier example can be re-implemented as follows:

```python
df = pd.read_csv('data/prices.csv')
df['date'] = pd.to_datetime(df['date'])

pybroker.register_columns('rsi')

strategy = Strategy(df, '6/1/2021', '12/1/2021')
strategy.add_execution(buy_low_sell_high_rsi, ['MCD', 'NKE', 'DIS'])
result = strategy.backtest()
result.orders
```
