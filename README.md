<h1>
    <img src="https://github.com/edtechre/pybroker/blob/master/docs/_static/pybroker-logo.png?raw=true" alt="PyBroker">
</h1>

## Algorithmic Trading in Python with Machine Learning

**PyBroker** is a Python framework for developing algorithmic trading
strategies, especially those that use machine learning. With PyBroker, it is
easy to write trading rules, build models, and analyze a strategy's
performance.

## Key Features

- Fast backtesting engine built in [NumPy](https://numpy.org/) with [Numba](https://numba.pydata.org/) acceleration.
- Easily write trading rules and models that execute on multiple instruments.
- Download historical data from [Alpaca](https://alpaca.markets/) and [Yahoo Finance](https://finance.yahoo.com/).
- Train and backtest models using [Walkforward Analysis](https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model%20with%20Walkforward%20Analysis.html#Walkforward-Analysis) to simulate real trading.
- Includes robust metrics calculated from randomized [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
- Caching of downloaded data, indicators, and models for faster development.
- Parallelized computations for faster performance.

## Installation

PyBroker supports Python 3.9+ on Windows, Mac, and Linux. You can install
PyBroker using ``pip``:

```bash
    pip install lib-pybroker
```

Or you can clone the Git repository with:

```bash
    git clone https://github.com/edtechre/pybroker
```

## A Quick Example

Code speaks louder than words! Here is a peek at what backtesting with PyBroker
looks like:

**Rule-based Strategy**:

```python
   from pybroker import Strategy, YFinance, highest

   def exec_fn(ctx):
      # Require at least 20 days of data.
      if ctx.bars < 20:
         return
      # Get the rolling 10 day high.
      high_10d = ctx.indicator('high_10d')
      # Buy on a new 10 day high.
      if not ctx.long_pos() and high_10d[-1] > high_10d[-2]:
         ctx.buy_shares = 100
         # Hold the position for 2 days.
         ctx.hold_bars = 2

   strategy = Strategy(YFinance(), start_date='1/1/2022', end_date='7/1/2022')
   strategy.add_execution(exec_fn, ['AAPL', 'MSFT'], indicators=highest('high_10d', 'high', period=10))
   result = strategy.backtest()
```

**Model-based Strategy**:

```python
   import pybroker
   from pybroker import Alpaca, Strategy

   def train_fn(train_data, test_data, ticker):
      # Train the model using indicators stored in train_data.
      ...
      return trained_model

   # Register the model and its training function with PyBroker.
   my_model = pybroker.model('my_model', train_fn, indicators=[...])

   def exec_fn(ctx):
      preds = ctx.preds('my_model')
      # Open a long position given my_model's latest prediction.
      if not ctx.long_pos() and preds[-1] > buy_threshold:
         ctx.buy_shares = 100
      # Close the long position given my_model's latest prediction.
      elif ctx.long_pos() and preds[-1] < sell_threshold:
         ctx.sell_all_shares()

   alpaca = Alpaca(api_key=..., api_secret=...)
   strategy = Strategy(alpaca, start_date='1/1/2022', end_date='7/1/2022')
   strategy.add_execution(exec_fn, ['AAPL', 'MSFT'], models=my_model)
   # Run Walkforward Analysis on 1 minute data using 5 windows with 50/50 train/test data.
   result = strategy.walkforward(timeframe='1m', windows=5, train_size=0.5)
```

## Online Documentation

To learn how to use PyBroker, [**head over to the online documentation.**](http://www.pybroker.com)

## Contact

<img src="https://github.com/edtechre/pybroker/blob/master/docs/_static/email-image.png?raw=true">
