.. meta::
   :title: PyBroker
   :description: Algorithmic Trading in Python with Machine Learning
   :google-site-verification: XJpTXQaAdlEb2eAbndHa2ZmaUiOixSMaRusk-kKVKOQ

.. title:: Algorithmic Trading in Python with Machine Learning

.. raw:: html

   <style>
      @font-face {
         font-family: Bosun;
         src: url("_static/bosun03.otf") format("opentype");
      }
      #pybroker h1 {
         font-family: Bosun;
         font-weight: 900;
         font-size: 3em;
      }
   </style>

================
PyBroker
================

.. raw:: html

   <embed>
      <a href="https://www.python.org/">
         <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python">
      </a> &nbsp;
      <a href="https://pypi.org/project/lib-pybroker/">
         <img src="https://img.shields.io/badge/pypi-v1.0.12-brightgreen.svg"
            alt="PyPI">
      </a> &nbsp;
      <a href="https://opensource.org/licenses/LGPL-3.0">
        <img src="https://img.shields.io/badge/license-LGPLv3-brightgreen.svg"
            alt="LGPL v3">
      </a> &nbsp;
      <a href="https://www.pybroker.com/en/latest/?badge=latest">
         <img src="https://readthedocs.org/projects/pybroker/badge/?version=latest"
            alt="Documentation Status">
      </a> &nbsp;
      <a href="https://github.com/edtechre/pybroker/">
         <img src="https://github.com/edtechre/pybroker/actions/workflows/main.yml/badge.svg?event=push"
            alt="Package status">
      </a>
   </embed>

Algorithmic Trading in Python with Machine Learning
===================================================

**PyBroker** is a Python framework for developing algorithmic trading
strategies, especially those that use machine learning. With PyBroker, it is
easy to write trading rules, build models, and analyze a strategy's
performance.

Key Features
============

* Fast backtesting engine built in `NumPy <https://numpy.org/>`_ with `Numba <https://numba.pydata.org/>`_ acceleration.
* Easily write trading rules and models that execute on multiple instruments.
* Download historical data from `Alpaca <https://alpaca.markets/>`_ and `Yahoo Finance <https://finance.yahoo.com/>`_.
* Train and backtest models using `Walkforward Analysis <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_ to simulate real trading.
* Includes robust metrics calculated from randomized `bootstrapping <https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>`_.
* Caching of downloaded data, indicators, and models for faster development.
* Parallelized computations for faster performance.

.. include:: install.rst

A Quick Example
===============

Code speaks louder than words! Here is a peek at what backtesting with PyBroker
looks like:

**Rule-based Strategy**::

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
   strategy.add_execution(
      exec_fn, ['AAPL', 'MSFT'], indicators=highest('high_10d', 'high', period=10))
   result = strategy.backtest()

**Model-based Strategy**::

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

To learn how to use PyBroker, see the notebooks under the *User Guide*:

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   Installation <install>
   notebooks/1. Getting Started with Data Sources
   notebooks/2. Backtesting a Strategy
   notebooks/3. Evaluate Using Bootstrap Metrics
   notebooks/4. Ranking and Position Sizing
   notebooks/5. Writing Indicators
   notebooks/6. Training a Model
   notebooks/7. Creating a Custom Data Source

`The notebooks above are also available on Github
<https://github.com/edtechre/pybroker/tree/master/docs/notebooks>`_.

.. toctree::
   :maxdepth: 4
   :caption: Reference

   Configuration Options <reference/pybroker.config>
   Modules <reference/modules>
   Index <genindex>

Recommended Reading
===================

For more background on quantitative finance and algorithmic trading, below is a
list of essential books:

* Lingjie Ma, `Quantitative Investing: From Theory to Industry <https://www.amazon.com/Quantitative-Investing-Industry-Lingjie-Ma/dp/3030472019/>`_

* Timothy Masters, `Testing and Tuning Market Trading Systems: Algorithms in C++ <https://www.amazon.com/Testing-Tuning-Market-Trading-Systems/dp/148424172X/>`_

* Stefan Jansen, `Machine Learning for Algorithmic Trading, 2nd Edition <https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715/>`_

* Ernest P. Chan, `Machine Trading: Deploying Computer Algorithms to Conquer the Markets <https://www.amazon.com/Machine-Trading-Deploying-Computer-Algorithms-ebook/dp/B01N7NKVG0/>`_

* Perry J. Kaufman, `Trading Systems and Methods, 6th Edition <https://www.amazon.com/Trading-Systems-Methods-Wiley-ebook/dp/B08141BBXR/>`_

.. toctree::
      :maxdepth: 1
      :caption: Other Information

      Changelog <changelog>
      License <license>

Contact
=======

.. image:: _static/email-image.png
