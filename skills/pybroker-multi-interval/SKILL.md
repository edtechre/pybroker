---
name: pybroker-multi-interval
description: Build, wire, and debug multi-timeframe PyBroker strategies using the bundled PyBroker wiki references generated from the local docs. Use when an agent needs to trade a base timeframe with confirmation from coarser weekly or monthly bars, compress bars into higher time intervals, declare compressed bars with the intervals parameter of add_execution, read completed higher-timeframe bars with ctx.interval and IntervalContext, bind indicators to intervals with Indicator.intervals, train models per interval with ModelSource.intervals, choose interval formats such as every-n-bars ints, duration strings like 5m or 1h, or calendar strings like weekly and monthly, pass timeframe to backtest, walkforward, or optimize, compress OHLCV bars standalone with compress_bars, guard warmup while interval arrays are still empty, or debug interval errors such as undeclared intervals, missing timeframe, or intervals not strictly coarser than the base data.
---

# PyBroker Multi-Interval

## Overview

Build multi-timeframe PyBroker strategies that trade a base timeframe while confirming regime and trend on strictly coarser compressed intervals such as weekly and monthly bars. Covers the three interval formats, providing compressed bars with `add_execution(intervals=...)`, computing indicators and training models per interval by binding them with `.intervals(...)`, reading completed bars through `ctx.interval(...)`, and standalone compression with `compress_bars`. Strategy code only ever sees completed compressed bars, which keeps higher-timeframe logic free of partial-bar lookahead.

## Workflow

1. Extract the multi-interval spec: base timeframe and bar spacing, the coarser intervals and their formats, the job of each timeframe (regime, trend, timing), whether each interval needs raw bars only or per-interval indicators and models, whether base-timeframe variants must be kept alongside bound ones, backtest versus walkforward, and desired output file/notebook.
2. Ask only for missing blockers. If details are absent but noncritical, make conservative assumptions and state them in the final answer or code comments.
3. Read `references/wiki-index.md` to choose the smallest relevant wiki page. For nontrivial multi-interval work, also read `references/multi-interval-patterns.md`.
4. Build a complete runnable multi-interval surface:
   - start scripts with `pybroker.disable_progress_bar()` and `pybroker.enable_data_source_cache("<name>")` (or `pybroker.enable_caches` when models are trained)
   - choose each interval's format (an int greater than 1 for every-n-bars, a single-unit duration string such as `"5m"` or `"1h"`, or a calendar string such as `"weekly"`) and keep every interval strictly coarser than the base timeframe
   - declare bars-only intervals with `add_execution(..., intervals=...)`; bind per-interval indicator computation and model training with `ind.intervals(...)` / `model_source.intervals(...)` passed to `indicators=`/`models=`, including `"base"` when the base-timeframe variant must also exist
   - write execution logic that reads `ctx.interval("weekly")` completed bars behind length guards (`if len(weekly.close) < 10: return`) and sets orders on the base `ctx` only
   - run `backtest`/`walkforward` with `timeframe="<base spacing>"`; use `pybroker.compress_bars(df, interval, base_timeframe=...)` for standalone compression and validation
5. Validate the produced code as far as the environment allows. At minimum, run syntax checks for created Python files. When practical, backtest against a small local DataFrame and run the multi-interval bump-last-bar test from `references/multi-interval-patterns.md`.

## Implementation Rules

- Treat PyBroker as a backtesting framework, not a source of financial advice. Make strategy assumptions explicit and avoid performance claims that are not supported by a produced backtest.
- Use completed historical bar data only. `ctx.interval(...)` exposes only completed compressed bars: the week or month still forming is never visible, and execution logic must never reconstruct the forming bin from base bars to peek at it. A value at bar `i` may depend only on inputs at index `i` and earlier, and negative indexing into full-length arrays is forbidden (a negative index silently wraps to the end of the series — the future).
- Interval grammar: an interval is an int greater than 1 (every n base bars), a duration string of digits plus one unit letter `s`/`m`/`h`/`d` (`"5m"`, `"1h"`; never `"5min"` or compound spans like `"1h 30m"`, and week durations such as `"2w"` are rejected — use `"weekly"` or `"14d"`), or a calendar string `"daily"`/`"weekly"`/`"monthly"`/`"quarterly"`/`"yearly"` (weeks start Monday, months on the 1st, quarters in January/April/July/October). Every interval must be strictly coarser than the base timeframe, and empty or duplicate interval lists raise `ValueError`.
- Whenever any execution declares or binds an interval, `backtest`/`walkforward`/`optimize` require `timeframe=` stating the base bar spacing (e.g. `timeframe="1d"`), which is validated against the observed data spacing. The `timeframe` grammar is wider than interval grammar — compound spans like `"1h 30m"` and the `w` unit are legal there — so never reuse a `timeframe` string as an interval.
- `add_execution(..., intervals=...)` provides compressed bars only and is scoped to that execution: sibling executions and `set_before_exec`/`set_after_exec` callbacks cannot read them, and no indicator or model is ever computed on an interval unless bound to it. Bound intervals are automatically unioned into `ctx.interval`, so do not repeat them in `intervals=`.
- Binding is exhaustive: `ind.intervals("weekly")` / `model_source.intervals("weekly")` replaces base-timeframe computation, so include the literal `"base"` (e.g. `.intervals("base", "weekly")`) to keep the base variant; unbound sources default to base. `"base"` is valid only inside a binding — it raises in `intervals=` and in `ctx.interval`.
- `ctx.interval(...)` returns a read-only `IntervalContext` exposing `bars`, `dates`, OHLCV arrays, `indicator(name)`, `model(name)`, `input(name)`, `preds(name)`, and registered custom columns as attributes (there is no `vwap` property). Arrays hold completed compressed bars only and are empty during warmup, so guard every read with a length check such as `if len(weekly.close) < 10: return`. Setting order or stop attributes on an `IntervalContext` raises — set them on the base `ctx`.
- Read interval values with base names only: `weekly.indicator("sma_10")`, never `"sma_10@weekly"`. The `@` character is reserved for the interval-suffixed names PyBroker generates itself and is invalid in registered indicator and model names.
- Only trainable models bind to intervals; a pretrained model (`ModelLoader`) raises `ValueError`. An interval-bound model's `train_fn` receives compressed-bar DataFrames with its registered indicators under their base column names — work on a `.copy()` when adding a target and never widen or mutate the input frame. `lookahead` is measured in the bound interval's compressed bars, and PyBroker warns when the hold-out empties the train set. Read predictions with `ctx.interval(...).preds(name)` behind a length guard; `TestResult.signals` contain base-timeframe values only unless `"base"` is in the binding.
- Compress standalone with `pybroker.compress_bars(df, interval, base_timeframe=...)`, which accepts a single symbol only (multi-symbol frames raise, pointing to `compress_symbol_from_frame` / `compress_intervals_from_frame`). Aggregation: open from the first base bar, high/low extremes, close from the last base bar, volume summed, VWAP volume-weighted, custom columns take the last value, and each compressed bar is dated by the last base bar it contains.
- On any interval `ValueError`, match the message against the Common Errors table in `references/multi-interval-patterns.md` before changing code — the messages name the fix.
- Never use pandas to implement indicator or execution logic. Indicator functions operate on `BarData` NumPy arrays (the same function runs unchanged on whichever interval it is bound to), prefer the vectorized helpers (`highv`, `lowv`, `sumv`, `returnv`, `cross`, `atr`), JIT-compile explicit loops with a nested Numba `@njit` kernel, and return a full-length one-dimensional array with NaN warmup bars — never a shortened array. Never construct a `pd.Series` or `pd.DataFrame` and never call pandas methods such as `.rolling`, `.ewm`, `.shift`, or `.apply` inside an indicator function or a per-bar execution function; the only sanctioned pandas is the `train_fn`/`input_data_fn` model boundary.
- Guard base-timeframe lookbacks with `ctx.bars` or `warmup=` and interval lookbacks with length guards (base-bar `warmup=` is no substitute — compressed arrays fill on their own schedule), and set at most one order side per symbol per bar.
- Start generated scripts with `pybroker.disable_progress_bar()` so progress output does not flood agent context, and `pybroker.enable_data_source_cache("<name>")` (or `pybroker.enable_caches` when models are trained) so repeated runs do not refetch data. Add `pybroker.disable_logging()` when running many backtests, such as parameter optimization.
- If a Numba `@njit` function fails to compile or raises a cryptic error such as a `TypingError`, re-run once with the `NUMBA_DISABLE_JIT=1` environment variable to get a readable Python traceback, fix the underlying code, then remove the variable so the backtest runs compiled.
- Self-test novel multi-interval logic for lookahead with the bump-last-bar check in `references/multi-interval-patterns.md`: change only the final base bar and assert every completed compressed output is unchanged — the trailing partial bin absorbs the bump.
- If exact API names, constructor parameters, or methods matter, read `references/api-public-surface.md`.
- For exact type signatures, read the matching `references/pybroker_*.pyi` stub: `IntervalContext` in `pybroker_context.pyi`; `TimeframeInterval` and `CalendarInterval` in `pybroker_types.pyi`; `compress_bars`, `Indicator.intervals`/`IntervalBoundIndicator`, and `ModelSource.intervals`/`IntervalBoundModel` in `pybroker_model.pyi`; the `backtest`/`walkforward`/`optimize` `timeframe=` parameters in `pybroker_strategy.pyi`.
- If the user wants a standalone file, copy and adapt `assets/multi_interval_template.py`.

## Common Deliverables

- Standalone `.py` multi-timeframe backtest script that trades a base timeframe with higher-interval confirmation.
- Upgrade of a single-timeframe strategy to read higher-interval bars, indicators, or predictions.
- `.intervals(...)` bindings that wire per-interval indicators and models into an existing strategy.
- Standalone bar-compression analysis with `compress_bars` on a local DataFrame.
- Debugging notes and patches for interval ValueErrors, empty warmup arrays, and partial-bar lookahead.

## Resources

- `references/wiki-index.md`: start here for topic routing across the bundled references.
- `references/wiki-15-multiple-time-intervals.md`: interval types, compressing bars, multi-timeframe strategies, and binding indicators and models to intervals.
- `references/wiki-05-writing-indicators.md`: custom indicators, vector helpers, TA-Lib, built-in indicators, and indicator sets.
- `references/wiki-06-training-a-model.md`: model training, model predictions, caching, and walkforward analysis.
- `references/api-public-surface.md`: generated public API signatures and first docstring sentences from local source.
- `references/pybroker_context.pyi`: generated type stubs for `ExecContext` (including its writable order/stop attributes), `IntervalContext`, `RotationContext`, `ExecResult`, and the slippage models.
- `references/pybroker_types.pyi`: generated type stubs for enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `references/pybroker_model.pyi`: generated type stubs for `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `references/pybroker_strategy.pyi`: generated type stubs for `Strategy`, `StrategyConfig`, `TestResult`, and the optimization types.
- `references/multi-interval-patterns.md`: load when writing nontrivial multi-interval code; interval grammar, bars-versus-binding scope, compression semantics, the interval error table, and the validation checklist.
- `assets/multi_interval_template.py`: copy and adapt when creating a new standalone multi-timeframe script.
