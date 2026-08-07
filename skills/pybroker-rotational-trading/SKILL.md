---
name: pybroker-rotational-trading
description: Build ranked-signal and rotational PyBroker strategies using the bundled PyBroker wiki references generated from the local docs. Use when an agent needs to rank symbols with ctx.long_score or ctx.short_score, cap positions with Strategy.set_max_long_positions or set_max_short_positions, rotate a portfolio into its top-ranked symbols with Strategy.enable_rotation and a worst_rank_held hold band, write a custom rotation sizer over RotationContext long_ranks and short_ranks, choose between ranked-cap prioritization and full rotation, carry stops and fill prices into rotation orders, handle unrankable NaN scores or long/short overlap, screen a dynamic universe with a SymbolSelector before ranking, search position caps or worst_rank_held as hyperparams, migrate deprecated ctx.score or StrategyConfig.max_long_positions code, or debug rotation errors such as worst_rank_held below a position cap or a sizer without rotation enabled.
---

# PyBroker Rotational Trading

## Overview

Build rotational PyBroker strategies that hold the top-ranked symbols in a universe and rotate out of names that fall from favor. Execution functions score each symbol with `ctx.long_score` and `ctx.short_score`, position slots are capped with `Strategy.set_max_long_positions` and `Strategy.set_max_short_positions`, and `Strategy.enable_rotation(worst_rank_held=...)` turns the scores into trades: each bar, held positions ranked worse than the hold band are liquidated and the best-ranked candidates fill the freed slots at equal weight, unless a custom `sizer` over `RotationContext` overrides the entry sizes. Also covers the simpler ranked-cap mode, where execution functions keep placing their own orders and scores only prioritize signals when a position cap binds, plus dynamic candidate universes selected with a `SymbolSelector`.

## Workflow

1. Extract the rotation spec: the candidate universe (fixed list or a `SymbolSelector` screen), the ranking signal for each side, long and/or short legs, position slots per side, hold band (`worst_rank_held`) versus ranked-cap prioritization only, sizing (default equal weight or a custom `sizer`), stops and fill prices, backtest versus walkforward, and the desired deliverable file.
2. Ask only for missing blockers. If details are absent but noncritical, make conservative assumptions and state them in the final answer or code comments.
3. Read `references/wiki-index.md` to choose the smallest relevant wiki page. For nontrivial rotation work, also read `references/rotational-patterns.md`.
4. Build a complete runnable rotation surface:
   - start scripts with `pybroker.disable_progress_bar()` and `pybroker.enable_data_source_cache("<name>")`
   - compute the ranking indicator as NumPy over `BarData` arrays and set `ctx.long_score`/`ctx.short_score` in the execution function (never the deprecated `ctx.score`)
   - cap slots with `strategy.set_max_long_positions(n)`/`set_max_short_positions(n)` (never the deprecated `StrategyConfig` fields), then either stop there for ranked-cap mode or call `strategy.enable_rotation(worst_rank_held=..., sizer=...)` for hold-band rotation
   - under rotation, let the execution function set only scores, stops, and fill prices — orders it places are ignored; in ranked-cap mode, keep placing orders normally with at most one order side per symbol per bar
   - run `backtest`/`walkforward` with `warmup=` covering the ranking indicator's lookback and inspect `result.orders` to confirm rotation entries and hold-band exits
5. Validate the produced code as far as the environment allows. At minimum, run syntax checks for created Python files. When practical, run against a small local DataFrame and confirm `result.orders` shows entries capped at the position limits and exits for symbols that fall out of the hold band.

## Implementation Rules

- Treat PyBroker as a backtesting framework, not a source of financial advice. State assumptions explicitly (universe, ranking signal, hold band, costs) and make no performance claims unsupported by the produced backtest.
- Use completed historical bar data only. Indicator logic must be lookahead-free: never index a full-length array with a negative index (it silently wraps to the end of the series, the future) and never shift future values backward; a value at bar `i` may depend only on inputs at index `i` and earlier. Self-test novel indicator logic with the bump-last-bar check: change only the final input bar and assert every earlier output is unchanged.
- Two ranking modes exist. Ranked-cap mode (`set_max_*_positions` plus scores, no `enable_rotation`) keeps execution functions in charge of orders and uses scores only to prioritize signals when a cap binds; symbols that set no score sort as `0.0` and unrankable scores sort last. Rotation mode (`enable_rotation`) drives all trading from scores. Choose ranked-cap for prioritizing entry signals, rotation for hold-the-top-N portfolios.
- Rank with `ctx.long_score` (buy and cover signals) and `ctx.short_score` (sell signals), never the deprecated `ctx.score`: setting it warns, mixing it with `long_score`/`short_score` raises, and under rotation it raises `ValueError`. Scores rank the whole portfolio across all executions, descending, with the symbol name as a deterministic tiebreak.
- `strategy.set_max_long_positions(n)`/`set_max_short_positions(n)` accept an int greater than 0, a searchable `Hyperparam`, or `None` for unlimited. The `StrategyConfig` fields of the same names are deprecated and the setters take precedence.
- Rotation mechanics: each bar, held positions ranked worse than `worst_rank_held` — or holding an unrankable score, even when another execution opened them — are liquidated, and the top-ranked candidates fill the remaining free slots at equal weight `1 / (long slots + short slots)`. Candidates ranked outside the hold band are never entered. `enable_rotation(None)` disables rotation and clears the sizer.
- Rotation is exclusive: orders placed by execution functions are discarded, but fill prices and stops (including `hold_bars`) set during execution are kept and applied to the orders rotation places. Under rotation, the execution function's job is scores, stops, and fill prices only.
- A `None` or NaN score excludes the symbol from the rank map, which liquidates a held position. NaN indicator warmup is harmless before positions exist, but an indicator that goes NaN mid-series forces an exit — confirm that is intended.
- A symbol picked by both the long and short leg goes to the side where it ranks better; ties go long. A symbol with no bar on the current date keeps its position slot, and in-flight pending orders hold their slots too.
- A rotation `sizer` is a `Callable[[RotationContext], None]` invoked after rotation decides what to trade; `long_ranks`/`short_ranks` are 1-based with `1` the best. Override entry sizes with `ctx.buy_shares = ctx.calc_target_shares(weight)` (or `ctx.sell_shares` for short entries) guarded by `if ctx.buy_shares is not None:`, and never override the sell or cover signals rotation set. A sizer without rotation enabled raises `ValueError`.
- `worst_rank_held` requires at least one position cap and must be greater than or equal to every cap that is set. On any rotation `ValueError`, match the message against the Common Errors table in `references/rotational-patterns.md` before changing code.
- To rotate within a screened universe, pass a `SymbolSelector` callable as the `add_execution` symbols: it runs once per walkforward window on the window's training data, requires a DataFrame data source and a training window (`backtest` and `train_size=0` raise `ValueError`), and positions in symbols a later window drops are closed at that window's first bar.
- `set_max_long_positions`, `set_max_short_positions`, and `enable_rotation(worst_rank_held=...)` all accept a `pybroker.hyperparam(...)`, so slots and the hold band are searchable with `Strategy.optimize`.
- Never use pandas to implement indicator or execution logic: write indicators as vectorized NumPy over `BarData` arrays (Numba `@njit` for explicit loops) and read `ctx.*` NumPy arrays in execution functions — no `pd.Series`/`pd.DataFrame` construction and no `.rolling`/`.ewm`/`.shift`/`.apply` in either. A `SymbolSelector` is a sanctioned pandas boundary: it receives the DataFrame PyBroker hands it.
- An indicator returns one full-length one-dimensional array with one value per input bar, warmup left-padded with NaN — never a shortened array.
- Keep feature data out-of-band: never widen or mutate the user's input DataFrame.
- Enable caching while iterating: `pybroker.enable_data_source_cache(name)` to skip refetching data, or `pybroker.enable_caches(name)` to also cache indicators. Call `pybroker.disable_progress_bar()` in agent-run scripts, and add `pybroker.disable_logging()` when running many backtests.
- Report `result.metrics_df` as the human-readable summary, and inspect `result.orders` to confirm rotation behavior. When structured output is needed (agent parsing, saved report files, downstream tools), use `result.to_json()` / `result.to_json_str()`: the default payload serializes metrics, trades, orders, and bootstrap capped at `max_rows=100` rows per table, `symbols=` filters to specific tickers, and `include=` opts into `portfolio`/`positions`/`metrics_df`/`signals`/`stops`. Do not replace the `metrics_df` print outright: the default JSON payload (trades plus orders) is usually larger than the metrics table.
- On a Numba compilation or typing error in an `@njit` indicator, re-run once with the environment variable `NUMBA_DISABLE_JIT=1` to get a readable Python traceback, fix the error, then re-run with JIT enabled. Never leave JIT disabled in the final script.
- Guard lookbacks with `ctx.bars` or `warmup=`. In ranked-cap mode set at most one order side per symbol per bar; in rotation mode a symbol scored on both legs is resolved by rotation's overlap rule, never by placing both orders.
- If exact API names, constructor parameters, or methods matter, read `references/api-public-surface.md`.
- For exact type signatures — `set_max_long_positions`, `set_max_short_positions`, and `enable_rotation` in `references/pybroker_strategy.pyi`, `RotationContext` and the `long_score`/`short_score` attributes in `references/pybroker_context.pyi`, `SymbolSelector` in `references/pybroker_types.pyi` — read the matching `references/pybroker_*.pyi` stub.
- If the user wants a standalone file, copy and adapt `assets/rotation_template.py`.

## Common Deliverables

- Standalone `.py` rotational backtest that ranks a universe and holds the top-N inside a hold band.
- Ranked-cap prioritization (`long_score`/`short_score` plus position caps) added to an existing multi-symbol strategy.
- Custom rotation `sizer` implementing rank-weighted or otherwise non-equal entry allocation.
- Long/short rotation with both legs, overlap handling, and stops carried into rotation orders.
- Migration of deprecated `ctx.score` or `StrategyConfig.max_long_positions`/`max_short_positions` code to the current API.
- Debugging notes for rotation `ValueError`s, unrankable-score liquidations, and ignored execution-function orders.

## Resources

- `references/wiki-index.md`: start here for topic routing across the bundled references.
- `references/wiki-10-rotational-trading.md`: hold-band rotation with `enable_rotation` and custom position sizing with a `sizer`.
- `references/wiki-04-ranking-long-and-short-signals.md`: ranking long and short signals with scores and position caps.
- `references/wiki-18-dynamic-symbol-selection.md`: screening a candidate universe with a `SymbolSelector`.
- `references/rotational-patterns.md`: load when writing nontrivial rotation code; the mode decision, rotation mechanics, sizer recipes, the rotation error table, and the validation checklist.
- `references/api-public-surface.md`: generated public API signatures and first docstring sentences from local source.
- `references/pybroker_strategy.pyi`: generated type stubs for `Strategy`, `StrategyConfig`, `TestResult`, and the optimization types.
- `references/pybroker_context.pyi`: generated type stubs for `ExecContext` (including its writable order/stop attributes), `IntervalContext`, `RotationContext`, `ExecResult`, and the slippage models.
- `references/pybroker_model.pyi`: generated type stubs for `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `references/pybroker_types.pyi`: generated type stubs for enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `assets/rotation_template.py`: copy and adapt when creating a new standalone rotational trading script.
