---
name: pybroker-optimize
description: Tune PyBroker strategy hyperparameters with Optuna-backed search using the bundled PyBroker wiki references generated from the local docs. Use when an agent needs to declare tunable values with pybroker.hyperparam, run Strategy.optimize with grid, TPE, or random samplers, choose n_trials, direction, train_size, or seed, write score functions over TestResult metrics, wire hyperparams into indicator kwargs or ctx.hyperparam via add_execution(hyperparams=...), pass custom Optuna samplers or a supplied study, inspect OptimizeResult, WindowOptimizeResult, or study.trials_dataframe(), run walkforward optimization with windows, pin winning values with backtest(params=...), or debug failed trials and grid explosions.
---

# PyBroker Optimizer

## Overview

Tune PyBroker strategy hyperparameters with `Strategy.optimize`: declare tunable values with `pybroker.hyperparam`, wire them into indicators and execution functions, and score each candidate combination on a training window before the winning values are replayed on held-out test data. Covers grid, TPE, and random sampling through the integrated Optuna backend, custom Optuna samplers and studies, walkforward optimization across multiple windows, and reading results from `OptimizeResult` and the underlying `optuna.Study`.

## Workflow

1. Extract the optimization spec: which values to tune with their `low`/`high`/`step` ranges and numeric types, the score metric and `direction`, sampler and trial budget, `train_size`, `windows`, `seed`, caching, and whether the strategy uses models (trainable models are unsupported by `optimize`).
2. Ask only for missing blockers. If details are absent but noncritical, make conservative assumptions and state them in the final answer or code comments.
3. Read `references/wiki-index.md` to choose the smallest relevant wiki page. For nontrivial optimization work, also read `references/optimization-patterns.md`.
4. Build a complete runnable optimization surface:
   - declare tunable values with `pybroker.hyperparam(name, default=..., low=..., high=..., step=...)`, using `low == high` to pin a value without searching it
   - attach each hyperparam where it is consumed: as an indicator keyword argument, through `add_execution(..., hyperparams=[...])` read with `ctx.hyperparam(name)`, or into `set_max_long_positions`/`set_max_short_positions`/`enable_rotation(worst_rank_held=...)`
   - write a `score_fn(result: TestResult) -> float` over `result.metrics` with `None`-guards for metrics that can be undefined
   - run `strategy.optimize(score_fn, sampler=..., n_trials=..., seed=..., train_size=..., windows=...)`, with trial parallelism configured through `pybroker.set_parallel(n_jobs=...)`
   - read `OptimizeResult`: `best_params`, `best_score`, `result` (the held-out test `TestResult`), `study`, and `windows`; pin winning values into later runs with `backtest(params=...)` or `walkforward(..., params=...)`
5. Validate the produced code as far as the environment allows. At minimum, run syntax checks for created Python files. Run a small-grid optimize on small local data when the repo and data make that practical.

## Implementation Rules

- Treat PyBroker as a backtesting framework, not a source of financial advice. State assumptions explicitly, and never present the in-sample `best_score` as expected performance; report the held-out test metrics from `OptimizeResult.result`.
- Use completed historical bar data only. Indicator logic must be lookahead-free: never index a full-length array with a negative index (it silently wraps to the end of the series, the future) and never shift future values backward; a value at bar `i` may depend only on inputs at index `i` and earlier. Self-test novel indicator logic with the bump-last-bar check: change only the final input bar and assert every earlier output is unchanged.
- A `Hyperparam`'s `default`, `low`, `high`, and `step` must all share one numeric type (all int or all float; bools are rejected), `step` must be positive, `low` cannot exceed `high`, and `high - low` must be an exact multiple of `step`. Candidate values run from `low` to `high` inclusive; `low == high` declares a fixed hyperparam that resolves in backtests and appears in `best_params` but is excluded from the search.
- The search-space key is the hyperparam name, not the consuming keyword: with `lookback = pybroker.hyperparam("lookback", ...)`, `pybroker.indicator("sma", sma, period=lookback)` searches `"lookback"`.
- `ctx.hyperparam("name")` works only when the same registered `Hyperparam` object is passed in `add_execution(..., hyperparams=[...])`; reading an unattached name raises `ValueError`.
- Write `score_fn(result: TestResult) -> float` over `result.metrics` fields and guard `Optional` metrics (`lambda r: r.metrics.sharpe if r.metrics.sharpe is not None else 0.0`); a `None` or NaN score marks that trial FAILED instead of aborting the study. Use `direction="minimize"` for objectives such as drawdown.
- Choose the sampler by search-space size: `"grid"` (the default) exhaustively enumerates every lattice combination and evaluates trials in parallel; `"random"` also parallelizes; `"tpe"` adapts to earlier trials and therefore runs sequentially. Any `optuna.samplers.BaseSampler` instance is also accepted; it is deep-copied and reseeded per window and must be picklable when `windows > 1`.
- `n_trials` is required for every sampler except `"grid"`, where it defaults to the full grid size and a smaller value samples that many combinations at random. Heed the grid-explosion warning when `grid_size * windows` exceeds 1000: coarsen `step`, fix values with `low == high`, or switch to `"tpe"`/`"random"` with an explicit `n_trials`.
- Pass `seed=` for reproducible optimization; it seeds the sampler and bootstrap metrics, and window `i` derives `seed + i`. Unlike `backtest`/`walkforward` (default `seed=42`), `optimize` defaults to `seed=None`, which does not reproduce.
- Every trial backtests only the training split (`train_size`, exclusive of `0` and `1`, default `0.5`); the winning combination is then replayed once on the held-out test window, which `score_fn` never sees, producing `OptimizeResult.result`.
- With `windows > 1`, each walkforward window is tuned by its own study and the per-window winners are replayed into one continuous stitched result with cash and positions carried across window boundaries. `best_params`, `best_score`, and `study` describe the last window only; report per-window values from `OptimizeResult.windows` (`WindowOptimizeResult` holds `params`, `study`, `train_score`, and the window dates, but no per-window test result). `study=` is rejected when `windows > 1`.
- `optimize` rejects trainable model sources; pretrained models (`pybroker.model(..., pretrained=True)`) are supported and loaded once per train window, then reused across that window's trials. Tune trainable models inside `train_fn` with a validation split over the train window instead.
- Trial and window parallelism come only from the global `pybroker.set_parallel(n_jobs=...)`; `optimize` has no `n_jobs` parameter.
- `pruner=` is passed through to the created Optuna study but never triggers, because each trial is one complete backtest with no intermediate values to report; do not rely on pruning for budget control.
- Pin tuned values outside of optimization with `strategy.backtest(params={...})` or `strategy.walkforward(..., params=...)`; hyperparam defaults apply when `params` is omitted.
- Never use pandas to implement indicator or execution logic: write indicators as vectorized NumPy over `BarData` arrays (Numba `@njit` for explicit loops) and read `ctx.*` NumPy arrays in execution functions — no `pd.Series`/`pd.DataFrame` construction and no `.rolling`/`.ewm`/`.shift`/`.apply` in either.
- An indicator returns one full-length one-dimensional array with one value per input bar, warmup left-padded with NaN — never a shortened array.
- Keep feature data out-of-band: never widen or mutate the user's input DataFrame.
- Enable caching while iterating: `pybroker.enable_data_source_cache(name)` to skip refetching data, or `pybroker.enable_caches(name)` to also cache indicators. Call `pybroker.disable_progress_bar()` in agent-run scripts, and add `pybroker.disable_logging()` for optimize runs, which backtest once per trial and would otherwise flood context with per-run logs.
- On a Numba compilation or typing error in an `@njit` indicator, re-run once with the environment variable `NUMBA_DISABLE_JIT=1` to get a readable Python traceback, fix the error, then re-run with JIT enabled. Never leave JIT disabled in the final script.
- Guard lookbacks with `ctx.bars` or `warmup`, and set at most one order side per symbol per bar. Use current API only: rank with `ctx.long_score`/`ctx.short_score` (never the deprecated `ctx.score`) and cap positions with `strategy.set_max_long_positions(n)`/`set_max_short_positions(n)`, not the deprecated `StrategyConfig` fields.
- If exact API names, constructor parameters, or methods matter, read `references/api-public-surface.md`.
- For exact type signatures — `Strategy.optimize`, `Hyperparam`, `OptimizeResult`, and `WindowOptimizeResult` in `references/pybroker_strategy.pyi`, `ctx.hyperparam` and the writable order/stop attributes in `references/pybroker_context.pyi` — read the matching `references/pybroker_*.pyi` stub.
- If the user wants a standalone file, copy and adapt `assets/optimize_template.py`.

## Common Deliverables

- Standalone `.py` optimization script reporting `best_params` and held-out test metrics.
- Conversion of a hard-coded strategy to `pybroker.hyperparam`-driven values plus an optimize run.
- Walkforward optimization (`windows > 1`) with per-window parameter reporting.
- Optuna study analysis: `trials_dataframe()` summaries, custom sampler configuration, supplied studies.
- Notebook-ready PyBroker optimization cells.

## Resources

- `references/wiki-index.md`: start here for topic routing across the bundled references.
- `references/wiki-12-parameter-optimization.md`: declaring hyperparameters, grid search, TPE and other samplers, and walkforward optimization.
- `references/wiki-11-configuring-parallelization.md`: worker counts with `set_parallel`, parallel indicators, and the Ray backend.
- `references/wiki-03-evaluating-with-bootstrap-metrics.md`: evaluation metrics, bootstrap confidence intervals, and maximum drawdown.
- `references/optimization-patterns.md`: load when writing nontrivial optimization code; score-function recipes, Optuna integration, walkforward windows, and the optimization checklist.
- `references/api-public-surface.md`: generated public API signatures and first docstring sentences from local source.
- `references/pybroker_strategy.pyi`: generated type stubs for `Strategy`, `StrategyConfig`, `TestResult`, and the optimization types.
- `references/pybroker_context.pyi`: generated type stubs for `ExecContext` (including `ctx.hyperparam` and its writable order/stop attributes), `IntervalContext`, `RotationContext`, `ExecResult`, and the slippage models.
- `references/pybroker_model.pyi`: generated type stubs for `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `references/pybroker_types.pyi`: generated type stubs for enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `assets/optimize_template.py`: copy and adapt when creating a new standalone optimization script.
