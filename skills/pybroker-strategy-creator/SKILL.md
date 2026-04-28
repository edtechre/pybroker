---
name: pybroker-strategy-creator
description: Create, adapt, review, and debug PyBroker algorithmic trading strategy and backtest code using the bundled PyBroker wiki references generated from the local docs. Use when Codex needs to turn trading rules into PyBroker Strategy/ExecContext logic, add indicators, models, stops, ranking, position sizing, rebalancing, custom data sources, walkforward analysis, bootstrap metrics, or answer PyBroker usage questions.
---

# PyBroker Strategy Creator

## Overview

Create practical PyBroker strategy code from user intent while preserving backtest hygiene: no lookahead leakage, explicit sizing, clear risk controls, and locally valid PyBroker API usage.

## Workflow

1. Extract the strategy spec: universe, data source, date range, timeframe, long/short permissions, entry and exit rules, sizing, stops, ranking, rebalancing cadence, model training needs, and desired output file/notebook.
2. Ask only for missing blockers. If details are absent but noncritical, make conservative assumptions and state them in the final answer or code comments.
3. Read `references/wiki-index.md` to choose the smallest relevant wiki page. For nontrivial strategy work, also read `references/pybroker-patterns.md`.
4. Build a complete runnable strategy surface:
   - create a `StrategyConfig` when cash, fees, position limits, delays, exits, or returned signals/stops matter
   - define indicators with `highest`, `lowest`, `returns`, or `indicator`
   - define model sources with `pybroker.model` only when training or loading predictions is part of the request
   - write execution functions that use completed-bar arrays such as `ctx.close[-1]`, guard lookbacks with `ctx.bars` or `warmup`, and set at most one order side per symbol per bar
   - add executions with `Strategy.add_execution`
   - run `backtest` for a single train/test pass or `walkforward` for model/walk-forward evaluation
5. Validate the produced code as far as the environment allows. At minimum, run syntax checks for created Python files. Run tests or a small local-data backtest when the repo and data make that practical.

## Implementation Rules

- Treat PyBroker as a backtesting framework, not a source of financial advice. Make strategy assumptions explicit and avoid performance claims that are not supported by the produced backtest.
- Use completed historical bar data only. Do not use future prices, future indicator values, or shuffled time series unless explicitly doing a model training split that PyBroker supports.
- Use `ctx.calc_target_shares(target_size)` for allocation-based sizing. Use fixed `ctx.buy_shares` or `ctx.sell_shares` only when the user asks for fixed share sizing.
- Check `ctx.long_pos()` or `ctx.short_pos()` before entering or exiting positions. Use `ctx.sell_all_shares()` and `ctx.cover_all_shares()` for full exits.
- Set entry-time stops on the same bar as the entry order: `hold_bars`, `stop_loss_pct`, `stop_profit_pct`, or `stop_trailing_pct`.
- Use `ctx.score` with `StrategyConfig.max_long_positions` or `max_short_positions` for ranked selection across symbols.
- Use `strategy.set_before_exec`, `strategy.set_after_exec`, or `strategy.set_pos_size_handler` for cross-symbol portfolio logic instead of hiding global state inside a per-symbol execution function.
- If exact API names, constructor parameters, or methods matter, read `references/api-public-surface.md`.
- If the user wants a standalone file, copy and adapt `assets/strategy_template.py`.

## Common Deliverables

- Standalone `.py` backtest script.
- Notebook-ready PyBroker cells.
- Refactor of an existing strategy file.
- Debugging notes and patches for invalid `ExecContext` usage.
- Focused tests using local DataFrame data when live data sources are unavailable.

## Resources

- `references/wiki-index.md`: start here for topic routing across the bundled PyBroker wiki.
- `references/wiki-01-getting-started-with-data-sources.md`: Yahoo Finance, Alpaca, Alpaca Crypto, AKShare, data caching, and data source setup.
- `references/wiki-02-backtesting-a-strategy.md`: defining execution rules, adding executions, running backtests, and filtering data.
- `references/wiki-03-evaluating-with-bootstrap-metrics.md`: evaluation metrics, confidence intervals, bootstrap metrics, and drawdown.
- `references/wiki-04-ranking-and-position-sizing.md`: ranking symbols, score usage, max positions, and position sizing.
- `references/wiki-05-writing-indicators.md`: custom indicators, vector helpers, TA-Lib, built-in indicators, and indicator sets.
- `references/wiki-06-training-a-model.md`: model training, model predictions, caching, and walkforward analysis.
- `references/wiki-07-creating-a-custom-data-source.md`: extending `DataSource`, DataFrame inputs, CSV inputs, and custom columns.
- `references/wiki-08-applying-stops.md`: stop loss, take profit, trailing stops, limit prices, exit prices, and stop cancellation.
- `references/wiki-09-rebalancing-positions.md`: equal weighting, before/after execution hooks, and portfolio optimization.
- `references/wiki-10-rotational-trading.md`: rotational strategy examples and universe rotation.
- `references/wiki-faqs.md`: common PyBroker usage questions and edge cases.
- `references/api-public-surface.md`: generated public API signatures and first docstring sentences from local source.
- `references/pybroker-patterns.md`: load when writing nontrivial strategy code, debugging PyBroker API usage, or adding indicators, stops, models, ranking, rebalancing, or walkforward analysis.
- `assets/strategy_template.py`: copy and adapt when creating a new standalone strategy script.
