#########
Changelog
#########

2.0.0
=====

* Requires Python 3.11+.

* `Multiple time intervals <https://www.pybroker.com/en/latest/notebooks/15.%20Multiple%20Time%20Intervals.html>`_, including ``.intervals()`` binding.

* `Margin trading <https://www.pybroker.com/en/latest/notebooks/13.%20Margin%20Trading.html>`_.

* `Rotational trading <https://www.pybroker.com/en/latest/notebooks/10.%20Rotational%20Trading.html>`_.

* `Parameter optimization with Optuna <https://www.pybroker.com/en/latest/notebooks/12.%20Parameter%20Optimization.html>`_.

* `Time series models and lag features <https://www.pybroker.com/en/latest/notebooks/16.%20Time%20Series%20Models.html>`_.

* `Multi-symbol models <https://www.pybroker.com/en/latest/notebooks/17.%20Multi-Symbol%20Models.html>`_.

* `Dynamic symbol selection <https://www.pybroker.com/en/latest/notebooks/18.%20Dynamic%20Symbol%20Selection.html>`_.

* `Fill-time slippage models <https://www.pybroker.com/en/latest/notebooks/14.%20Modeling%20Slippage.html>`_ via ``apply_slippage`` / ``SlippageContext``.

* `Parallelization configuration <https://www.pybroker.com/en/latest/notebooks/11.%20Configuring%20Parallelization.html>`_ (Ray backend).

* Ranking with ``long_score`` / ``short_score``.

* Position limits via ``Strategy.set_max_long_positions`` / ``set_max_short_positions``.

* ``to_json`` / ``to_json_str`` on backtest and optimize results.

* :doc:`Agent Skills <agent-skills>`.

* ATR indicator; ``bars_to_df`` helper.

* Broad NumPy/Numba performance improvements.

Breaking changes
----------------

* Removes ``PosSizeContext`` and ``set_pos_size_handler``.

* Deprecates ``ExecContext.score`` and ``StrategyConfig.max_*_positions``.

* Renames timeframe module to ``interval``; drops week-duration intervals.

* Unifies slippage API; removes custom stop functions.

* Removes ``multiprocessing`` from ``set_parallel``, plus ``bootstrap_sample_size`` / ``indicator_memo_max``.

* Model lag matrices passed to ``train_fn`` / ``predict_fn`` instead of DataFrame attrs.

1.2.14
======

* Requires Python 3.10+.

* Fixes indicator computation bugs in `vect <https://www.pybroker.com/en/latest/reference/pybroker.vect.html>`_ (Aroon, Laguerre RSI, ADX, price change oscillator, reactivity, trend, and related kernels).

* Fixes duplicate ``volume`` column in the AKShare TX fallback path.

1.2.13
======

* Adds signal provenance fields to `Order <https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.Order>`_:

    * ``created`` - date the order signal was created.
    * ``order_type`` - how the order originated (``market``, ``limit``, ``stop_bar``, ``stop_loss``, ``stop_profit``, ``stop_trailing``).
    * ``intent`` - position intent (``buy_to_open``, ``buy_to_close``, ``sell_to_open``, ``sell_to_close``).

* Adds `OrderType <https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.OrderType>`_ and `PositionIntent <https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.PositionIntent>`_ enums.

* Adds ``order_id`` parameter to `PendingOrderScope.orders() <https://www.pybroker.com/en/latest/reference/pybroker.scope.html#pybroker.scope.PendingOrderScope.orders>`_.

1.2.3
=====

* Adds built-in indicators to the `indicator module <https://www.pybroker.com/en/latest/reference/pybroker.indicator.html>`_.


1.1.0
=====

* `Adds support for the following stop types: <https://www.pybroker.com/en/latest/notebooks/8.%20Applying%20Stops.html>`_

    * Stop loss
    * Trailing stop loss
    * Take profit

* `Allows canceling pending orders. <https://www.pybroker.com/en/latest/notebooks/FAQs.html#...-cancel-pending-orders?>`_

* Upgrades ``alpaca-trade-api-python`` to ``alpaca-py`` package.

1.0.0
=====

* Initial release!