---
name: pybroker-strategy-creator
description: Create, adapt, review, and debug PyBroker algorithmic trading strategy and backtest code using the bundled PyBroker wiki references generated from the local docs. Use when an agent needs to turn trading rules into PyBroker Strategy/ExecContext logic, add indicators, models, stops, ranking, rotation, position sizing, rebalancing, custom data sources, walkforward analysis, bootstrap metrics, parameter optimization, multiple time intervals, slippage modeling, margin trading, parallelization, or dynamic symbol selection, or to answer PyBroker usage questions.
---

# PyBroker Strategy Creator

## Overview

Create practical PyBroker strategy code from user intent while preserving backtest hygiene, including no lookahead leakage, explicit sizing, clear risk controls, and locally valid PyBroker API usage.

## Workflow

1. Extract the strategy spec: universe, data source, date range, timeframe, long/short permissions, entry and exit rules, sizing, stops, ranking, rebalancing cadence, model training needs, and desired output file/notebook.
2. Ask only for missing blockers. If details are absent but noncritical, make conservative assumptions and state them in the final answer or code comments.
3. Read `references/wiki-index.md` to choose the smallest relevant wiki page. For nontrivial strategy work, also read `references/pybroker-patterns.md`.
4. Build a complete runnable strategy surface:
   - start scripts with `pybroker.disable_progress_bar()` and `pybroker.enable_data_source_cache("<name>")`
   - create a `StrategyConfig` when cash, fees, delays, exits, margin, or returned signals/stops/positions matter
   - define indicators with `highest`, `lowest`, `returns`, the built-in factories in `pybroker.indicator` (such as `atr`), or `indicator` with vectorized NumPy/Numba functions
   - define model sources with `pybroker.model` only when training or loading predictions is part of the request
   - write execution functions that use completed-bar arrays such as `ctx.close[-1]`, guard lookbacks with `ctx.bars` or `warmup`, and set at most one order side per symbol per bar
   - add executions with `Strategy.add_execution`, passing `hyperparams=` when optimization is involved, `intervals=` for higher-timeframe bars, and `.intervals(...)`-bound indicators/models for per-interval computation, and cap positions with `strategy.set_max_long_positions` / `set_max_short_positions`
   - run `backtest` for a single train/test pass, `walkforward` for model/walk-forward evaluation, or `optimize` for hyperparameter search; pass `timeframe=` whenever an execution declares `intervals=` or binds a model/indicator to an interval
5. Validate the produced code as far as the environment allows. At minimum, run syntax checks for created Python files. Run tests or a small local-data backtest when the repo and data make that practical.

## Implementation Rules

- Treat PyBroker as a backtesting framework, not a source of financial advice. Make strategy assumptions explicit and avoid performance claims that are not supported by the produced backtest.
- Use completed historical bar data only. Do not use future prices, future indicator values, or shuffled time series unless explicitly doing a model training split that PyBroker supports. An indicator value at bar `i` may depend only on inputs at index `i` and earlier: no backward shifts such as `shift(-1)`, and no negative indexing into full-length arrays inside indicator functions (a negative index silently wraps to the end of the series — the future).
- Start generated scripts with `pybroker.disable_progress_bar()` so backtest progress output does not flood agent context, and `pybroker.enable_data_source_cache("<strategy_name>")` (or `pybroker.enable_caches`) so repeated runs do not refetch data. Add `pybroker.disable_logging()` when running many backtests, such as parameter optimization.
- Report `result.metrics_df` as the human-readable summary. When structured output is needed (agent parsing, saved report files, downstream tools), use `result.to_json()` / `result.to_json_str()`: the default payload serializes metrics, trades, orders, and bootstrap capped at `max_rows=100` rows per table, `symbols=` filters to specific tickers, and `include=` opts into `portfolio`/`positions`/`metrics_df`/`signals`/`stops` (`positions` needs `StrategyConfig(record_position_bars=True)`). Do not replace the `metrics_df` print outright: the default JSON payload (trades plus orders) is usually larger than the metrics table.
- Never use pandas to implement indicator or execution logic. `BarData` and `ExecContext` price fields are NumPy arrays: operate on them directly, prefer the vectorized helpers (`highv`, `lowv`, `sumv`, `returnv`, `cross`, `atr`) when they fit, and JIT-compile explicit loops with Numba `@njit`. Never construct a `pd.Series` or `pd.DataFrame` and never call pandas methods such as `.rolling`, `.ewm`, `.shift`, or `.apply` inside an indicator function or a per-bar execution function.
- If a Numba `@njit` function fails to compile or raises a cryptic error such as a `TypingError`, re-run once with the `NUMBA_DISABLE_JIT=1` environment variable to get a readable Python traceback, fix the underlying code, then remove the variable so the backtest runs compiled.
- Self-test novel indicator logic for lookahead with the bump-last-bar check in `references/pybroker-patterns.md`: recompute after changing only the final input bar and assert every earlier output is unchanged.
- Use `ctx.calc_target_shares(target_size)` for allocation-based sizing and `ctx.set_target_shares(target, dir="long")` to rebalance toward a target allocation. Use fixed `ctx.buy_shares` or `ctx.sell_shares` only when the user asks for fixed share sizing.
- Check `ctx.long_pos()` or `ctx.short_pos()` before entering or exiting positions. Use `ctx.sell_all_shares()` and `ctx.cover_all_shares()` for full exits.
- Set entry-time stops on the same bar as the entry order: `hold_bars`, `stop_loss_pct`, `stop_profit_pct`, or `stop_trailing_pct`.
- Orders fill at `PriceType.MIDDLE` — the midpoint of the low and high of the *execution* bar, which under the default `buy_delay`/`sell_delay` of `1` is the bar after the signal, so `PriceType.CLOSE` means the next bar's close. Override with `ctx.buy_fill_price` / `ctx.sell_fill_price`, which take a `PriceType` (`OPEN`, `HIGH`, `LOW`, `CLOSE`, `MIDDLE`, `AVERAGE`), a number, or a `(symbol, bar_data)` callable, and read back as `None` rather than `MIDDLE` until set. A limit price only *gates* the fill: the order still fills at the fill price, never at the limit, so `ctx.buy_limit_price = 200` against a midpoint of `108` books `108`.
- `StrategyConfig.exit_on_last_bar` defaults to `False`, which leaves any position still open when the data ends. That position never becomes a `Trade`, so `trade_count`, `win_rate`, `total_pnl` and every other trade-level metric silently exclude it while its P&L sits in `unrealized_pnl`. Set `exit_on_last_bar=True` whenever trade statistics are reported; exits fill at `exit_sell_fill_price` / `exit_cover_fill_price`, both `PriceType.MIDDLE`, and in `walkforward` the liquidation fires only on each symbol's true final bar, never at window boundaries. Bar-level metrics (`sharpe`, `max_drawdown`) are computed from per-bar market value and barely move either way.
- `calc_bootstrap` is a `backtest`/`walkforward`/`optimize` parameter defaulting to `False`, not a `StrategyConfig` field. Pass `calc_bootstrap=True` to populate `result.bootstrap` with `conf_intervals` (BCa — bias corrected and accelerated — intervals for profit factor and Sharpe; 6x2, MultiIndexed on `name` then `conf`, columns `lower`/`upper`) and `drawdown_conf` (percentile bounds on max drawdown; 4x2, indexed on `conf`, columns `amount`/`percent`). It leaves `metrics_df` unchanged, costs roughly `bars x StrategyConfig.bootstrap_samples` (default `10_000`, so lower it on intraday data), and needs `StrategyConfig.bars_per_year` or the Sharpe intervals are per-bar rather than annualized.
- Rank symbols with `ctx.long_score` and `ctx.short_score` and cap positions with `strategy.set_max_long_positions(n)` / `strategy.set_max_short_positions(n)`; the `StrategyConfig` fields of the same names are deprecated. Scores rank descending on both sides: short orders go to the symbols with the highest `short_score`, so negate a lowest-wins short signal (for example `ctx.short_score = -roc` to short the most negative momentum). For rank-and-rotate portfolios call `strategy.enable_rotation(worst_rank_held=...)`; rotation is exclusive, so execution functions then only set scores and any order fields they set are ignored.
- For parameter search, register values with `pybroker.hyperparam(name, default=..., low=..., high=..., step=...)`, attach them via indicator kwargs or `add_execution(..., hyperparams=[...])`, read them with `ctx.hyperparam("name")`, and run `strategy.optimize(score_fn, sampler="grid")` (or `"tpe"`/`"random"` with `n_trials=`). Optimization does not support trainable models; use pretrained models or indicator-based rules.
- For multi-timeframe logic, declare `add_execution(..., intervals="weekly")` and read compressed bars with `ctx.interval("weekly")`. `intervals=` provides bars only; to compute an indicator or train a model per interval, bind it with `indicator.intervals("weekly")` / `model_source.intervals("weekly")` in `indicators=`/`models=` — bound intervals are available through `ctx.interval` without declaring them again. Binding is exhaustive: include `"base"` (e.g. `.intervals("base", "weekly")`) to keep the base-timeframe variant; unbound sources default to base. When any execution declares or binds intervals, `backtest`/`walkforward` require `timeframe=` for the base data.
- Model trading costs with `strategy.set_slippage_model(...)`: `FixedSlippageModel(bps=...)` for constant costs, `VolatilitySlippageModel` for ATR-scaled slippage, `VolumeSlippageModel` for volume-capped fills. Only enable margin (`StrategyConfig(leverage=..., interest_rate=...)`, which requires `bars_per_year`) when the user asks for it.
- `result.positions` is empty unless `StrategyConfig(record_position_bars=True)`; enable it only when the user needs per-bar position output. The per-bar `result.portfolio` equity curve is always populated.
- Use `strategy.set_before_exec` or `strategy.set_after_exec` for cross-symbol portfolio logic instead of hiding global state inside a per-symbol execution function.
- Optional packages are not PyBroker dependencies: name the required `pip install` for any data source or library the script imports (for example `pip install yfinance` for `YFinance`) and never assume one is importable. When the network or a data-source package is unavailable, validate with a tiny local DataFrame passed to `Strategy` instead.
- If exact API names, constructor parameters, or methods matter, read `references/api-public-surface.md`.
- For exact type signatures — the writable `ExecContext` order/stop attributes, property and parameter types, enum members — read the matching `references/pybroker_*.pyi` stub (`pybroker_context.pyi` for `ExecContext` and slippage).
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
- `references/wiki-04-ranking-long-and-short-signals.md`: ranking long/short signals by score and max positions.
- `references/wiki-05-writing-indicators.md`: custom indicators, vector helpers, TA-Lib, built-in indicators, and indicator sets.
- `references/wiki-06-training-a-model.md`: model training, model predictions, caching, and walkforward analysis.
- `references/wiki-07-creating-a-custom-data-source.md`: extending `DataSource`, DataFrame inputs, CSV inputs, and custom columns.
- `references/wiki-08-applying-stops.md`: stop loss, take profit, trailing stops, limit prices, stop exit prices, and stop cancellation.
- `references/wiki-09-rebalancing-positions.md`: equal weighting, before/after execution hooks, and portfolio optimization.
- `references/wiki-10-rotational-trading.md`: rotational strategy examples, universe rotation, and custom position sizing.
- `references/wiki-11-configuring-parallelization.md`: worker counts, parallel indicators and model training, and the Ray backend.
- `references/wiki-12-parameter-optimization.md`: hyperparams, grid/TPE/random samplers, and walkforward optimization.
- `references/wiki-13-margin-trading.md`: leverage, buying power, margin interest, and shorting on margin.
- `references/wiki-14-modeling-slippage.md`: fixed, volatility, and volume slippage models plus custom slippage.
- `references/wiki-15-multiple-time-intervals.md`: interval types, compressing bars, and multi-timeframe strategies.
- `references/wiki-16-time-series-models.md`: GARCH volatility forecasting and models on lagged returns.
- `references/wiki-17-multi-symbol-models.md`: pooled models trained across multiple symbols.
- `references/wiki-18-dynamic-symbol-selection.md`: `SymbolSelector` and per-window universe selection.
- `references/wiki-faqs.md`: common PyBroker usage questions and edge cases.
- `references/api-public-surface.md`: generated public API signatures and first docstring sentences from local source.
- `references/pybroker_context.pyi`: generated type stubs for `ExecContext` (including its writable order/stop attributes), `IntervalContext`, `RotationContext`, `ExecResult`, and the slippage models.
- `references/pybroker_strategy.pyi`: generated type stubs for `Strategy`, `StrategyConfig`, `TestResult`, and the optimization types.
- `references/pybroker_types.pyi`: generated type stubs for enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `references/pybroker_model.pyi`: generated type stubs for `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `references/pybroker-patterns.md`: load when writing nontrivial strategy code, debugging PyBroker API usage, or adding indicators, stops, models, ranking, rebalancing, or walkforward analysis.
- `assets/strategy_template.py`: copy and adapt when creating a new standalone strategy script.
