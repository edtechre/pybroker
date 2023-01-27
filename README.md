<h1>
    <img src="https://github.com/edtechre/pybroker/blob/master/docs/_static/pybroker-logo.png?raw=true" alt="PyBroker">
</h1>

## Algorithmic Trading in Python with Machine Learning

**PyBroker** is a Python framework for backtesting algorithmic trading strategies,
including strategies that use machine learning. With PyBroker, it is easy to
write trading rules, build models, and analyze a strategy's performance. And it
is made fast with the help of [NumPy](https://numpy.org/) and
[Numba](https://numba.pydata.org/) acceleration.

Some of PyBroker's key features are:

- Easy reuse of trading rules and models across multiple instruments.
- Model training and backtesting using [Walkforward Analysis](https://www.youtube.com/watch?v=WBZ_Vv-iMv4).
- Extensive coverage of trading metrics, which are calculated out-of-sample.
- Robust performance metrics calculated with randomized [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
- Support for downloading historical data from [Alpaca](https://alpaca.markets/).
- Parallelized computing of indicators for faster performance.
- Quicker development from caching downloaded data, indicators, and trained models.

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
      if not ctx.long_pos() and preds[-1] > threshold:
         ctx.buy_shares = 100
      # Close the long position given my_model's latest prediction.
      elif ctx.long_pos() and preds[-1] < threshold:
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
