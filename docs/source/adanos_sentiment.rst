Adanos Sentiment Features
=========================

PyBroker can use custom data columns as execution context fields and model
features. ``AdanosSentiment`` wraps any existing ``DataSource`` and adds
optional stock sentiment features from the Adanos Market Sentiment API.

The wrapped data source still provides the required OHLCV bars:

.. code-block:: python

   import os

   from pybroker import Strategy, YFinance
   from pybroker.ext.data import AdanosSentiment

   data_source = AdanosSentiment(
      YFinance(),
      api_key=os.environ["ADANOS_API_KEY"],
      sources=("reddit", "x", "news", "polymarket"),
   )

   def exec_fn(ctx):
      if not ctx.long_pos() and ctx.adanos_sentiment[-1] > 0.25:
         ctx.buy_shares = 100
         ctx.hold_bars = 5

   strategy = Strategy(data_source, "1/1/2024", "1/1/2025")
   strategy.add_execution(exec_fn, ["AAPL", "MSFT", "NVDA"])
   result = strategy.backtest()

The adapter registers these aggregate columns by default:

* ``adanos_sentiment``
* ``adanos_buzz``
* ``adanos_mentions``

It also registers per-source columns such as ``adanos_reddit_sentiment``,
``adanos_x_buzz``, ``adanos_news_mentions``, and
``adanos_polymarket_sentiment``. These columns are available in model
``train_data`` and ``test_data`` just like price columns or indicators.
