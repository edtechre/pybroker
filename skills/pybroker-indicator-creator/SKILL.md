---
name: pybroker-indicator-creator
description: Write, register, and debug PyBroker indicators using the bundled PyBroker wiki references generated from the local docs. Use when an agent needs to write custom indicator functions with pybroker.indicator, vectorize indicator logic with NumPy and Numba @njit kernels, wrap third-party technical analysis libraries such as TA-Lib, pandas-ta, ta, tulipy, or finta, use the built-in indicator factories and vector helpers, compute indicators standalone with IndicatorSet, parameterize indicators with hyperparams for optimization, compute indicators on multiple time intervals, feed custom data columns into indicators, cache indicator computations, or debug Numba compilation errors and parallel indicator failures.
---

# PyBroker Indicator Creator

## Overview

Write fast, correct PyBroker indicators: register vectorized NumPy/Numba functions with `pybroker.indicator`, wire them into strategy executions and models, and keep every value free of lookahead bias. Covers the built-in indicator factories and vector helpers, custom Numba `@njit` kernels, wrapping third-party technical analysis libraries such as TA-Lib and pandas-ta, standalone computation with `IndicatorSet`, hyperparam-driven indicators, and multi-timeframe interval indicators.

## Workflow

1. Extract the indicator spec: formula or source library, input fields (OHLCV or registered custom columns), lookback lengths, fixed kwargs versus `hyperparam` parameterization, interval/timeframe needs, consumers (execution functions, models, or standalone DataFrame output), and desired output file/notebook.
2. Ask only for missing blockers. If details are absent but noncritical, make conservative assumptions and state them in the final answer or code comments.
3. Read `references/wiki-index.md` to choose the smallest relevant wiki page. For nontrivial indicator work, also read `references/indicator-patterns.md`.
4. Build a complete runnable indicator surface:
   - start scripts with `pybroker.disable_progress_bar()` and `pybroker.enable_data_source_cache("<name>")`, adding `pybroker.enable_indicator_cache("<name>")` (or `pybroker.enable_caches`) when indicator computation is expensive
   - prefer built-ins first: `highest`, `lowest`, and `returns` at top level, then the factories in the `pybroker.indicator` module (such as `atr`, `adx`, `macd`, `close_minus_ma`), then the vectorized helpers (`highv`, `lowv`, `sumv`, `returnv`, `cross`, `atr`)
   - write custom functions as `fn(bar_data, **kwargs)` over NumPy arrays — never pandas — that return a full-length one-dimensional array with NaN warmup bars, JIT-compiling explicit loops with a nested Numba `@njit` kernel
   - wrap third-party TA libraries (TA-Lib, pandas-ta, `ta`, tulipy, finta) at the wrapper boundary only, padding outputs to full length and registering one indicator per output column
   - register with `pybroker.indicator(name, fn, **kwargs)`, then attach with `Strategy.add_execution(..., indicators=[...])` and read with `ctx.indicator("name")`, or compute standalone with `ind(df)` / `IndicatorSet`
   - pass `pybroker.hyperparam` values as indicator kwargs for parameter search, and bind multi-timeframe indicators with `add_execution(..., indicators=ind.intervals("weekly"))`, read with `ctx.interval("weekly").indicator("name")` (`timeframe=` is then required on `backtest`/`walkforward`)
5. Validate the produced code as far as the environment allows. At minimum, run syntax checks for created Python files. When practical, compute the indicators on a small local DataFrame and check output length, NaN warmup, and the bump-last-bar lookahead test from `references/indicator-patterns.md`.

## Implementation Rules

- Treat PyBroker as a backtesting framework, not a source of financial advice. Make indicator assumptions explicit and avoid performance claims that are not supported by a produced backtest.
- Use completed historical bar data only. An indicator value at bar `i` may depend only on inputs at index `i` and earlier: no centered or forward-shifted windows, no normalization over the full series, and no negative indexing into full-length arrays inside kernels (a negative index silently wraps to the end of the series — the future).
- An indicator function receives a `BarData` argument plus its registered kwargs and must return a one-dimensional array with one value per input bar. Left-pad warmup bars with NaN and never return a shortened array (pad libraries such as tulipy that drop warmup rows); a returned `pd.Series` is converted automatically.
- Prefer built-ins before custom code: `highest`, `lowest`, and `returns` at top level, and the factories in the `pybroker.indicator` module (`atr`, `adx`, `macd`, `stochastic`, `close_minus_ma`, `laguerre_rsi`, and more). Watch the name collision: top-level `pybroker.atr` is the vectorized function `atr(high, low, close, lookback)`, while the factory is `pybroker.indicator.atr(name, lookback)`.
- Indicator names must not contain `@` (reserved for interval-suffixed names such as `sma_20@weekly`), and re-registering a name silently overwrites the previous indicator.
- Never use pandas to implement indicator or execution logic. Write indicator logic with vectorized NumPy over `BarData` arrays, prefer the vectorized helpers (`highv`, `lowv`, `sumv`, `returnv`, `cross`, `atr`) when they fit, and JIT-compile explicit loops with a nested Numba `@njit` kernel that takes plain NumPy arrays — `BarData` cannot cross the `@njit` boundary. Never construct a `pd.Series` or `pd.DataFrame` and never call pandas methods such as `.rolling`, `.ewm`, `.shift`, or `.apply` inside an indicator function or a per-bar execution function; the only sanctioned pandas is the third-party wrapper boundary in the next rule.
- Wrap third-party TA libraries at the wrapper boundary only: NumPy-native libraries (TA-Lib, tulipy) consume `BarData` arrays directly, while pandas-based libraries (pandas-ta, `ta`, finta) get a minimal `pd.Series`/`pd.DataFrame` built from `BarData` arrays — the only sanctioned pandas in indicator code. Register one indicator per output column for multi-output functions (the `talib.MACD` tuple, pandas-ta DataFrames). None of these libraries is a PyBroker dependency: state the required `pip install` and never assume one is importable.
- Compute indicators standalone with `ind(df)` on a single-symbol DataFrame (returns a date-indexed `pd.Series`) or with `IndicatorSet` for multi-symbol frames (requires a `symbol` column; output columns are `symbol`, `date`, then sorted indicator names). `IndicatorSet` never uses the disk cache.
- For parameter search, pass `pybroker.hyperparam(name, default=..., low=..., high=..., step=...)` objects as indicator kwargs and run `strategy.optimize(...)`; override standalone computation with `ind(df, hyperparams={...})`. Hyperparam-driven indicators are never disk-cached.
- For multi-timeframe indicators, do not pass an interval to `indicator()`; bind the registered indicator with `ind.intervals("weekly")` when passing it to `add_execution(indicators=...)`, and read it with `ctx.interval("weekly").indicator("name")` over completed compressed bars. Binding is exhaustive: include `"base"` (e.g. `ind.intervals("base", "weekly")`) to keep the base-timeframe variant; unbound indicators default to base. The bound interval is available through `ctx.interval` without declaring it in `intervals=` (which provides bars only). `backtest`/`walkforward` then require `timeframe=`, and each interval must be strictly coarser than the base timeframe.
- Register non-OHLCV columns with `pybroker.register_columns` before an indicator reads them; they appear as `BarData` attributes and are `None` when the input data lacks them, so guard for that.
- In execution functions read values with `ctx.indicator("name")` (arrays are truncated to completed bars; pass a symbol for another symbol's values) and guard lookbacks with `ctx.bars` or `warmup=`.
- Start generated scripts with `pybroker.disable_progress_bar()` so progress output does not flood agent context, and `pybroker.enable_data_source_cache("<name>")` so repeated runs do not refetch data; add `pybroker.disable_logging()` when running many backtests, such as parameter optimization.
- If a Numba `@njit` function fails to compile or raises a cryptic error such as a `TypingError`, re-run once with the `NUMBA_DISABLE_JIT=1` environment variable to get a readable Python traceback, fix the underlying code, then remove the variable so the backtest runs compiled. Never leave JIT disabled in the final script.
- Debug indicator failures serially before parallelizing: exceptions surface raw (there is no error handling on the indicator compute path), and under `parallel_indicators=True` they arrive wrapped in joblib worker tracebacks, so reproduce with the default serial path first.
- Self-test novel indicator logic for lookahead with the bump-last-bar check in `references/indicator-patterns.md`: recompute after changing only the final input bar and assert every earlier output is unchanged.
- If exact API names, constructor parameters, or methods matter, read `references/api-public-surface.md`.
- For exact type signatures — `indicator()`, `Indicator`, `IndicatorSet`, the vector helpers, and the cache and parallel functions in `references/pybroker_model.pyi`; `BarData` fields and the column/indicator scopes in `references/pybroker_types.pyi` — read the matching `references/pybroker_*.pyi` stub.
- If the user wants a standalone file, copy and adapt `assets/indicator_template.py`.

## Common Deliverables

- Standalone `.py` script that registers indicators and computes or backtests them.
- Wrapper modules that register TA-Lib, pandas-ta, `ta`, tulipy, or finta outputs as PyBroker indicators.
- Conversion of a pandas-based indicator into vectorized NumPy/Numba.
- Debugging notes and patches for Numba compile errors, output-length mismatches, and lookahead leaks.
- Notebook-ready PyBroker indicator cells.

## Resources

- `references/wiki-index.md`: start here for topic routing across the bundled references.
- `references/wiki-05-writing-indicators.md`: custom indicators, vector helpers, TA-Lib, built-in indicators, and indicator sets.
- `references/wiki-11-configuring-parallelization.md`: worker counts, parallel indicators and model training, and the Ray backend.
- `references/wiki-15-multiple-time-intervals.md`: interval types, compressing bars, and multi-timeframe strategies.
- `references/api-public-surface.md`: generated public API signatures and first docstring sentences from local source.
- `references/pybroker_model.pyi`: generated type stubs for `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `references/pybroker_types.pyi`: generated type stubs for enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `references/pybroker_context.pyi`: generated type stubs for `ExecContext` (including its writable order/stop attributes), `IntervalContext`, `RotationContext`, `ExecResult`, and the slippage models.
- `references/pybroker_strategy.pyi`: generated type stubs for `Strategy`, `StrategyConfig`, `TestResult`, and the optimization types.
- `references/indicator-patterns.md`: load when writing nontrivial indicator code; vectorization patterns, third-party library recipes, session hygiene, and the validation checklist.
- `assets/indicator_template.py`: copy and adapt when creating a new standalone indicator script.
