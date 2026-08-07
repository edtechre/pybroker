---
name: pybroker-model-trainer
description: Register, train, wire, and debug machine learning models for PyBroker backtests using the bundled PyBroker wiki references generated from the local docs. Use when an agent needs to register a model with pybroker.model, write train_fn/predict_fn code for scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, Keras, statsmodels (ARIMA/SARIMAX), or arch models, build ensembles or regime models, run walkforward analysis, build time-series or lagged-feature models, train pooled multi-symbol models, load pretrained models, cache data and trained models, or prevent lookahead leakage in model-driven backtests.
---

# PyBroker Model Trainer

## Overview

Wire machine learning models into PyBroker backtests: register training and prediction functions with `pybroker.model`, feed them indicator features, and evaluate them with walkforward analysis while keeping the train/test flow free of lookahead leakage. Covers per-symbol, pooled multi-symbol, per-bar time-series, lagged-feature, and pretrained models across common libraries such as scikit-learn, XGBoost, and arch.

## Workflow

1. Extract the modeling spec: library, prediction target and horizon, features (indicators, lagged columns, custom columns), per-symbol vs pooled training, vectorized vs per-bar prediction, walkforward windows and lookahead, caching, and whether the model is trainable or pretrained.
2. Ask only for missing blockers. If details are absent but noncritical, make conservative assumptions and state them in the final answer or code comments.
3. Read `references/wiki-index.md` to choose the smallest relevant wiki page. For nontrivial model work, also read `references/model-training-patterns.md`.
4. Build a complete runnable model surface:
   - define feature indicators with `pybroker.indicator` or built-ins, and register any non-OHLCV data columns with `pybroker.register_columns`
   - write a `train_fn` that builds the target from train data only and returns the model, or `(model, input_cols)` to pin prediction columns
   - register the model with `pybroker.model(name, train_fn, ...)`, choosing `indicators`, `lags`/`lag_cols`, `per_bar`, `pooled`, `pretrained`, `input_data_fn`, and `predict_fn` as needed
   - consume predictions in an execution function via `ctx.preds(name)` and pass the model source to `Strategy.add_execution(models=...)`
   - run `strategy.walkforward(windows, train_size, lookahead)` for evaluation, or `backtest(train_size=...)` for a single train/test pass
5. Validate the produced code as far as the environment allows. At minimum, run syntax checks for created Python files. Run a small local-data walkforward when the repo and data make that practical.

## Implementation Rules

- Treat PyBroker as a backtesting framework, not a source of financial advice. Make modeling assumptions explicit and avoid performance claims that are not supported by the produced backtest, including model fit metrics.
- Use completed historical bar data only. Do not use future prices, future indicator values, or shuffled time series outside the supported train-split shuffle.
- Set `lookahead` to the number of bars ahead of the prediction target (default `1` for next-bar targets). Walkforward holds out `lookahead` bars between each train and test split, so an understated value leaks train-adjacent bars into testing.
- When registering with `lags`, the current bar's value is the first feature of each lag block, so the training target must be the next bar's value (for example `fit(lag_train[:-1], target[1:])`). The training `fn` must accept `lag_train` and `lag_test` keyword arguments.
- Keep feature data out-of-band: never widen or mutate the input DataFrame. Work on a `.copy()` inside `train_fn` when adding a target column.
- An `input_data_fn` must return exactly one row per bar. A vectorized `predict_fn` must return one prediction per input row; for classifiers, slice a single `predict_proba` column. With `per_bar=True`, `predict_fn` is required, receives rows up to and including the current bar, must return a scalar, and cannot be combined with `pooled=True`.
- A pooled `train_fn` receives a sorted `symbols` tuple and combined frames with a `symbol` column. Build targets with per-symbol operations such as `groupby("symbol")[col].shift(-1)` so labels never cross a symbol boundary, and return `(model, input_cols)` to keep `symbol` out of model input.
- To train a model on a longer time interval, bind it with `model_source.intervals("weekly")` when passing it to `add_execution(models=...)`; binding is exhaustive, so the base-timeframe model is trained only when `"base"` is included (e.g. `model_source.intervals("base", "weekly")`), and the bound interval is available through `ctx.interval` without declaring it in `intervals=` (which provides bars only). For interval-bound models, `lookahead` is measured in that interval's compressed bars, and predictions are read with `ctx.interval("...").preds(name)`. `timeframe=` is then required on `backtest`/`walkforward`.
- `strategy.optimize` supports pretrained models only. Tune trainable models inside `train_fn` with a search over the train window, or compare registrations across walkforward runs; `pybroker.hyperparam` is for strategy-level parameters.
- Fit scalers, encoders, and any early-stopping validation splits on train data only.
- Never use pandas to implement indicator or execution logic: write indicators as vectorized NumPy over `BarData` arrays (Numba `@njit` for explicit loops) and read `ctx.*` NumPy arrays in execution functions — no `pd.Series`/`pd.DataFrame` construction and no `.rolling`/`.ewm`/`.shift`/`.apply` in either. Pandas belongs only at the `train_fn`/`input_data_fn` boundary where PyBroker hands you DataFrames; building the target there with `shift(-1)` on a `.copy()` stays sanctioned.
- Enable caching while iterating: `pybroker.enable_data_source_cache(name)` to skip refetching data, or `pybroker.enable_caches(name)` to also cache indicators and trained models.
- Call `pybroker.disable_progress_bar()` in agent-run scripts; progress bar output floods AI token context.
- Report `result.metrics_df` as the human-readable summary. When structured output is needed (agent parsing, saved report files, downstream tools), use `result.to_json()` / `result.to_json_str()`: the default payload serializes metrics, trades, orders, and bootstrap capped at `max_rows=100` rows per table, `symbols=` filters to specific tickers, and `include=` opts into `portfolio`/`positions`/`metrics_df`/`signals`/`stops` (`signals` carries model predictions when `StrategyConfig(return_signals=True)`). Do not replace the `metrics_df` print outright: the default JSON payload (trades plus orders) is usually larger than the metrics table.
- On a Numba compilation or typing error in an `@njit` indicator, re-run once with the environment variable `NUMBA_DISABLE_JIT=1` to get a readable Python traceback, fix the error, then re-run with JIT enabled. Never leave JIT disabled in the final script.
- Guard lookbacks with `ctx.bars` or `warmup`, and set at most one order side per symbol per bar.
- Use `ctx.calc_target_shares(target_size)` for allocation-based sizing. Use fixed `ctx.buy_shares` or `ctx.sell_shares` only when the user asks for fixed share sizing.
- Check `ctx.long_pos()` or `ctx.short_pos()` before entering or exiting positions. Use `ctx.sell_all_shares()` and `ctx.cover_all_shares()` for full exits.
- Set entry-time stops on the same bar as the entry order: `hold_bars`, `stop_loss_pct`, `stop_profit_pct`, or `stop_trailing_pct`.
- Rank by model score with `ctx.long_score` / `ctx.short_score` and cap positions with `strategy.set_max_long_positions(n)` / `set_max_short_positions(n)`; the `StrategyConfig` fields of the same names are deprecated. For score-driven rotation, `strategy.enable_rotation(worst_rank_held=...)` makes scores drive all trading and ignores order fields set in execution functions.
- Use `strategy.set_before_exec` or `strategy.set_after_exec` for cross-symbol portfolio logic instead of hiding global state inside a per-symbol execution function.
- If exact API names, constructor parameters, or methods matter, read `references/api-public-surface.md`.
- For exact type signatures — `pybroker.model()` and `train_fn`/`predict_fn` parameter types in `references/pybroker_model.pyi`, `ExecContext` prediction access and its writable order/stop attributes in `references/pybroker_context.pyi` — read the matching `references/pybroker_*.pyi` stub.
- If the user wants a standalone file, copy and adapt `assets/model_training_template.py`.

## Common Deliverables

- Standalone `.py` walkforward backtest script with a trained model.
- `train_fn`/`predict_fn` pairs for a user's chosen library.
- Conversion of an existing single-symbol model to pooled multi-symbol training.
- Debugging notes and patches for leaking targets, misaligned predictions, or invalid `pybroker.model` registrations.
- Notebook-ready PyBroker model training cells.

## Resources

- `references/wiki-index.md`: start here for topic routing across the bundled references.
- `references/wiki-06-training-a-model.md`: model registration, train/backtest flow, model caching, and walkforward analysis.
- `references/wiki-16-time-series-models.md`: GARCH with `per_bar=True` and Random Forest on lagged returns with `lags`/`lag_cols`.
- `references/wiki-17-multi-symbol-models.md`: pooled multi-symbol training with `pooled=True`.
- `references/model-training-patterns.md`: load when writing nontrivial train/predict code; library recipes, session hygiene, and the leakage checklist.
- `references/api-public-surface.md`: generated public API signatures and first docstring sentences from local source.
- `references/pybroker_model.pyi`: generated type stubs for `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `references/pybroker_context.pyi`: generated type stubs for `ExecContext` (including its writable order/stop attributes), `IntervalContext`, `RotationContext`, `ExecResult`, and the slippage models.
- `references/pybroker_strategy.pyi`: generated type stubs for `Strategy`, `StrategyConfig`, `TestResult`, and the optimization types.
- `references/pybroker_types.pyi`: generated type stubs for enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `assets/model_training_template.py`: copy and adapt when creating a new standalone model training script.
