#########
Changelog
#########

2.0.1
=====

* Adds ``use_log`` to the `returns <https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.returns>`_ indicator and `returnv <https://www.pybroker.com/en/latest/reference/pybroker.vect.html#pybroker.vect.returnv>`_ for computing log returns.

* Adds ``typing_extensions`` to install dependencies. It was already imported by :mod:`pybroker.strategy` but never declared, so installs relied on it being pulled in indirectly by another dependency.

* Requires ``optuna>=3.4``. Earlier versions store the sampler's random generator in a different place, where `optimize <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.optimize>`_ could not re-seed it, so passing ``seed`` did not make the search reproducible.

* Fixes `parse_timeframe <https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.parse_timeframe>`_ to reject malformed timeframes instead of silently reading a valid token out of them; ``"1.5h"`` was parsed as 5 hours and ``"-1h"`` as 1 hour.

* Fixes the "Finished training model" log message, which raised a ``TypeError`` instead of reporting the elapsed training time.

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

* Ranking with `long_score <https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.long_score>`_ / `short_score <https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.short_score>`_.

* Position limits via `Strategy.set_max_long_positions <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_max_long_positions>`_ / `set_max_short_positions <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_max_short_positions>`_.

* `to_json <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.TestResult.to_json>`_ / `to_json_str <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.TestResult.to_json_str>`_ on backtest and optimize results.

* :doc:`Agent Skills <agent-skills>`.

* `ATR <https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.atr>`_ indicator; `bars_to_df <https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.bars_to_df>`_ helper.

* Broad NumPy/Numba performance improvements.

Breaking changes
----------------

* Removes ``PosSizeContext``, ``set_pos_size_handler``, and ``ExecSignal``; use `Strategy.enable_rotation <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.enable_rotation>`_ / `RotationContext <https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.RotationContext>`_.

* Deprecates ``ExecContext.score`` and ``StrategyConfig.max_*_positions``; use `long_score <https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.long_score>`_ / `short_score <https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.short_score>`_ and `set_max_long_positions <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_max_long_positions>`_ / `set_max_short_positions <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_max_short_positions>`_.

* Unifies `slippage <https://www.pybroker.com/en/latest/reference/pybroker.slippage.html>`_ API; removes ``RandomSlippageModel``.

* Removes ``bootstrap_sample_size``; BCa and drawdown bootstrap now resample the full backtest series instead of a fixed-size sample, fixing cases that could produce incorrect confidence intervals or degenerate to a single replicate.

* Removes ``disable_parallel`` from `backtest <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest>`_ / `walkforward <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.walkforward>`_; parallel indicator and model work is opt-in via ``parallel_indicators`` / ``parallel_models``.

* ``result.positions`` is opt-in via `StrategyConfig.record_position_bars <https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.record_position_bars>`_; full ``Portfolio.bars`` snapshots are opt-in via `record_portfolio_bars <https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.record_portfolio_bars>`_.

* Removes ``akshare`` from install dependencies; install it separately to use `AKShare <https://www.pybroker.com/en/latest/reference/pybroker.ext.data.html#pybroker.ext.data.AKShare>`_.

* Fixes the `Calmar Ratio <https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.calmar_ratio>`_ to the standard definition of annualized return (CAGR) divided by maximum drawdown percentage; it previously annualized arithmetically and measured drawdown on the cumulative sum of per-bar returns.

* Fixes the `Ulcer Index <https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.ulcer_index>`_ to measure drawdowns from the running peak over the whole equity curve, and the `Ulcer Performance Index <https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.upi>`_ to use the annualized (CAGR) return when ``bars_per_year`` is set; passing a ``period`` to the functions keeps the previous trailing-window behavior.

* Fixes ``unrealized_pnl`` in `EvalMetrics <https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.EvalMetrics>`_, which previously understated unrealized PnL by the total fees paid (per-trade PnL is gross of fees while market values are net of them).

* `annual_total_return_percent <https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.annual_total_return_percent>`_ now counts ``n`` bar values as ``n - 1`` return intervals when annualizing.

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

* Adds :doc:`Agent Skills <agent-skills>`.

* Broad performance improvements to bar capture, scope fetches, and position lookups.

* Improves Alpaca crypto and AKShare reliability.

1.2.12
======

* Supports Pandas 3.

1.2.11
======

* Fixes readonly NumPy arrays returned by Pandas 2.3.

* Adds `clear_params <https://www.pybroker.com/en/latest/reference/pybroker.scope.html#pybroker.scope.clear_params>`_ to clear global parameters.

* Forwards ``*args`` and ``**kwargs`` from `Strategy.add_execution <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution>`_ to the execution function.

* Adds ``seed`` to `backtest <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest>`_ / `walkforward <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.walkforward>`_ for reproducible bootstrap results.

* Removes redundant ``subtract_fees`` config option.

1.2.10
======

* Uses per-bar returns instead of absolute per-bar deltas for Sharpe, Sortino, and Calmar ratios.

1.2.9
=====

* Upgrades to NumPy 2 while still supporting NumPy 1.

* Adds ``LONG_ONLY`` and ``SHORT_ONLY`` `position modes <https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig>`_.

* Adds ``max_drawdown_date`` to `EvalMetrics <https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.EvalMetrics>`_.

* Fixes ``yfinance`` dependency version.

1.2.8
=====

* Fixes NumPy typecheck errors.

1.2.7
=====

* Fixes ``df.loc[index]`` returning a DataFrame when the index is not unique.

1.2.6
=====

* Fixes missing ``Adj Close`` column from `YFinance <https://www.pybroker.com/en/latest/reference/pybroker.data.html#pybroker.data.YFinance>`_; adds ``auto_adjust`` argument.

* Raises an error when ``sell_all_shares`` or ``cover_all_shares`` is called with no open position.

1.2.5
=====

* Adds ``adjust`` parameter to `backtest <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest>`_ / `walkforward <https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.walkforward>`_.

* Fixes mypy typecheck errors.

1.2.4
=====

* Guarantees ``largest_loss_pct`` is always negative and ``largest_win_pct`` is always positive in `EvalMetrics <https://www.pybroker.com/en/latest/reference/pybroker.eval.html#pybroker.eval.EvalMetrics>`_.

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