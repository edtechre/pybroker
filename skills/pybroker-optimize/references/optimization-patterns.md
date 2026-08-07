# PyBroker Optimization Patterns

## `Strategy.optimize` API Map

Declare hyperparams first, then search them with `strategy.optimize`:

```python
hp = pybroker.hyperparam(
    name,          # unique name; the search-space key
    default=...,   # value used by regular backtests
    low=...,       # minimum candidate value (inclusive)
    high=...,      # maximum candidate value (inclusive)
    step=...,      # spacing; high - low must be a multiple of step
)

opt = strategy.optimize(
    score_fn,                  # Callable[[TestResult], float]
    sampler="grid",            # "grid" | "tpe" | "random" | BaseSampler
    n_trials=None,             # required for non-grid samplers
    direction="maximize",      # or "minimize"
    seed=None,                 # None does not reproduce; set for repro
    windows=None,              # >1 = per-window walkforward tuning
    study=None,                # existing optuna.Study (windows <= 1)
    pruner=None,               # attached but never actually triggers
    train_size=0.5,            # trial split; exclusive of 0 and 1
    lookahead=1,               # bars held out between train and test
    start_date=None,           # optional date filters, as walkforward
    end_date=None,
    timeframe="",
    between_time=None,
    days=None,
    warmup=None,               # bars before executions run
    parallel_indicators=False, # compute indicators in parallel
    adjust=None,               # data source adjustment
    calc_bootstrap=False,      # bootstrap metrics on opt.result only
    verbose=False,             # log every trial's backtest
)
```

`opt` is an `OptimizeResult`: `best_params` (dict, includes fixed
hyperparams), `best_score` (train-window score), `result` (the held-out
test `TestResult`), `study` (the `optuna.Study`), and `windows` (tuple
of `WindowOptimizeResult` when `windows > 1`, else `None`).

## Declaring Hyperparams

`pybroker.hyperparam` creates a frozen `Hyperparam` and registers it
globally by name. Validation happens at construction:

- `default`, `low`, `high`, and `step` must all be int or all be float
  (bools are rejected).
- `step` must be positive and `low` cannot exceed `high`.
- `high - low` must be an exact multiple of `step` (checked in
  `Decimal`, so float lattices like `low=2.0, high=10.0, step=2.0` are
  exact); otherwise the largest candidate would fall short of `high`.

```python
period = pybroker.hyperparam("period", default=30, low=10, high=50,
                             step=10)          # 10, 20, 30, 40, 50
stop_pct = pybroker.hyperparam("stop_pct", default=6.0, low=2.0,
                               high=10.0, step=2.0)  # floats only
size = pybroker.hyperparam("size", default=0.25, low=0.25, high=0.25,
                           step=0.25)  # fixed: resolved, not searched
```

`low == high` declares a fixed hyperparam: it resolves in backtests and
appears in `best_params`, but is excluded from the search space. A
searchable `Hyperparam` iterates its candidate lattice (`list(period)`
yields `[10, 20, 30, 40, 50]`; `len(period)` is `5`).

Re-registering a name overwrites the previous registration. A
registered hyperparam that no execution can reach raises a warning at
optimize time because it would only inflate the grid.

## Wiring Hyperparams Into a Strategy

### Indicator Parameters

Pass the `Hyperparam` object where a concrete value would go. The
consuming keyword (`period=`) is the indicator function's parameter;
the search-space key is the hyperparam's registered name:

```python
# Indicator logic is NumPy over BarData arrays — never pandas.
def sma(bar_data, period):
    return sumv(bar_data.close, period) / period

period = pybroker.hyperparam("period", default=30, low=10, high=50,
                             step=10)
sma_ind = pybroker.indicator("sma", sma, period=period)
```

Each trial recomputes the indicator with that trial's suggested value.
Indicators without hyperparams are computed once per window and shared
across trials.

### Execution Function Parameters

`ctx.hyperparam(name)` is gated by `add_execution(hyperparams=[...])`,
which must receive the same registered `Hyperparam` object. Reading a
name that was not attached raises `ValueError`:

```python
stop_pct = pybroker.hyperparam("stop_pct", default=6.0, low=2.0,
                               high=10.0, step=2.0)


# Execution logic reads ctx.* NumPy arrays — never pandas.
def sma_cross_stop(ctx):
    sma = ctx.indicator("sma")
    if np.isnan(sma[-1]):
        return
    if not ctx.long_pos() and ctx.close[-1] > sma[-1]:
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        ctx.stop_loss_pct = ctx.hyperparam("stop_pct")
    elif ctx.long_pos() and ctx.close[-1] < sma[-1]:
        ctx.sell_all_shares()


strategy.add_execution(
    sma_cross_stop, ["AAPL", "MSFT"], indicators=[sma_ind],
    hyperparams=[stop_pct],
)
```

### Position Caps and Rotation

`set_max_long_positions`, `set_max_short_positions`, and
`enable_rotation(worst_rank_held=...)` accept a `Hyperparam` in place
of an int, so position limits can be searched too:

```python
max_long = pybroker.hyperparam("max_long", default=5, low=2, high=8,
                               step=2)
strategy.set_max_long_positions(max_long)
```

## Score Functions

`score_fn` receives the trial's train-window `TestResult` and returns a
float. Any `result.metrics` (`EvalMetrics`) field works: `total_pnl`,
`total_return_pct`, `annual_return_pct`, `max_drawdown_pct`, `win_rate`,
`profit_factor`, `sharpe`, `sortino`, `calmar`, `ulcer_index`, `upi`,
`equity_r2`, `trade_count`, and more (see `api-public-surface.md`).

Guard `Optional` metrics — `sharpe`, `annual_return_pct`, `calmar`, and
others are `None` when undefined (for example, too few bars or no
trades). A `None` or NaN score marks that trial `FAILED` in the study
instead of aborting the optimization:

```python
def score_fn(result):
    # None marks the trial FAILED; return a poor score instead so
    # the sampler still learns from the combination.
    if result.metrics.sharpe is None:
        return 0.0
    return result.metrics.sharpe
```

`direction="minimize"` inverts the objective for loss-style metrics:

```python
opt = strategy.optimize(
    lambda r: r.metrics.max_drawdown_pct, direction="minimize",
    seed=42,
)
```

Composite objectives are plain Python — combine metrics and penalize
degenerate strategies explicitly:

```python
def robust_score(result):
    m = result.metrics
    if m.sharpe is None or m.trade_count < 20:
        return 0.0  # too few trades to trust the estimate
    return m.sharpe - 0.5 * m.max_drawdown_pct / 100
```

## Grid Search

`sampler="grid"` (the default) enumerates every combination of the
hyperparam lattices with a seeded `optuna.samplers.GridSampler`:

```python
opt = strategy.optimize(score_fn, seed=42, train_size=0.5)
print(opt.best_params)  # e.g. {"period": 20, "stop_pct": 4.0}
print(opt.best_score)   # the train-window score of that combination
```

- `n_trials` defaults to the full grid size; a smaller value samples
  that many combinations at random (seeded).
- Grid trials are independent, so they run in parallel on the workers
  configured with `pybroker.set_parallel(n_jobs=...)`.
- Grid explosion guard: when `grid_size * windows` exceeds 1000 and
  `n_trials` is unset, PyBroker warns to set `n_trials=` or switch to
  `sampler="tpe"`. Coarsen `step` or fix values with `low == high` to
  shrink the grid.

## TPE and Random Search

Non-grid samplers require `n_trials`:

```python
opt = strategy.optimize(score_fn, sampler="tpe", n_trials=25, seed=42)
opt = strategy.optimize(score_fn, sampler="random", n_trials=50,
                        seed=42)
```

- `"tpe"` (`optuna.samplers.TPESampler`) proposes each trial from the
  results of earlier ones. Evaluating adaptive trials in batches would
  change the values proposed and tie results to the worker count, so
  TPE trials run sequentially and an info-level log notes that
  parallelism was disabled.
- `"random"` (`optuna.samplers.RandomSampler`) is not adaptive and runs
  trials in parallel like grid.
- Prefer grid when the lattice is small enough to exhaust; prefer TPE
  when the full grid is too large and the surface is smooth enough for
  adaptive search to help; use random as a cheap parallel baseline.

## Optuna Integration

PyBroker depends on `optuna>=3.1,<5` directly, so no extra install is
needed. `strategy.optimize` builds real Optuna objects and exposes
them:

```python
study = opt.study                    # optuna.Study
df = study.trials_dataframe()        # one row per trial
best = study.best_value              # equals opt.best_score
trials = study.trials                # FrozenTrial list; .params/.state
```

Failed trials (score of `None`/NaN) appear with
`state == TrialState.FAIL` in `study.trials`.

A custom `optuna.samplers.BaseSampler` instance can be passed directly.
It is deep-copied and reseeded from `seed=` per window (a multi-window
run ships copies to worker processes, so the instance must be
picklable). Samplers other than grid and random are treated as adaptive
and run sequentially:

```python
from optuna.samplers import TPESampler

opt = strategy.optimize(
    score_fn,
    sampler=TPESampler(n_startup_trials=5),
    n_trials=25,
    seed=42,
)
```

A pre-created study — for example one backed by persistent storage —
can be supplied with `study=`. Its own sampler and pruner are used, its
direction must match `direction` (a mismatch raises), and it is not
supported with `windows > 1`, which runs one study per window:

```python
import optuna

study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///opt.db",
    study_name="sma_cross",
    load_if_exists=True,
)
opt = strategy.optimize(score_fn, study=study, n_trials=25, seed=42)
```

Re-running with the same study resumes it: grid search skips already
completed combinations. `pruner=` is attached to the created study but
never actually triggers — each trial is one complete backtest with no
intermediate values to report — so it cannot be used for budget
control; bound work with `n_trials` instead.

## Walkforward Optimization

`windows > 1` re-tunes the hyperparameters in each walkforward window:

```python
opt = strategy.optimize(score_fn, windows=3, seed=42, train_size=0.5)

for i, window in enumerate(opt.windows):
    print(f"window {i}: {window.params} "
          f"train_score={window.train_score:.2f} "
          f"({window.test_start_date:%Y-%m-%d} to "
          f"{window.test_end_date:%Y-%m-%d})")
print(opt.result.metrics_df)  # stitched across all test windows
```

Mechanics worth knowing:

- Each window runs its own study with its own sampler copy, seeded
  `seed + i`, and the windows themselves are tuned in parallel on the
  configured workers.
- After tuning, each window's winning values are replayed on that
  window's held-out test rows into one shared portfolio — cash and
  positions carry across window boundaries — producing a single
  continuous `opt.result`.
- `opt.best_params`, `opt.best_score`, and `opt.study` describe the
  last window only. Per-window values live in `opt.windows`; each
  `WindowOptimizeResult` holds `params`, `study`, `train_score`, and
  the window's train/test dates. There is deliberately no per-window
  test result — judge the strategy by the stitched `opt.result`.
- `lookahead` bars are held out between each train and test split, in
  the bars of the timeframe each model is fitted on.
- Instability of `window.params` across windows is a red flag: values
  that flip every window are likely fit to noise.

## Using the Results

`best_params` includes fixed (`low == high`) hyperparams as well as
searched ones, so it can be replayed directly:

```python
result = strategy.backtest(params=opt.best_params, seed=42)
result = strategy.walkforward(windows=3, params=opt.best_params,
                              seed=42)
```

Without `params=`, backtests resolve every hyperparam to its
`default`. `opt.result` is a regular `TestResult` with `metrics_df`,
`trades`, `orders`, `positions`, and `portfolio` DataFrames, plus
`bootstrap` when `calc_bootstrap=True`. `opt.to_json()` /
`opt.to_json_str()` serialize `best_params`, `best_score`, a study
summary, the test result, and any windows for reports. The test
result accepts the same `include=`/`max_rows=`/`symbols=` controls as
`TestResult.to_json`, and each window payload carries its winning
params, train score, date bounds, study summary, and any
selector-resolved `execution_symbols`.

## Bootstrap Metrics on Optimize Results

`calc_bootstrap` is a parameter of `optimize` (and of
`backtest`/`walkforward`), **not** a `StrategyConfig` field, and it
defaults to `False`. The `StrategyConfig` knob is `bootstrap_samples`
(default `10_000`); `bootstrap_sample_size` was removed in v2.

Only `opt.result` can carry bootstrap metrics. The per-trial train
replays hardcode `calc_bootstrap=False`, so a `score_fn` never sees
them and cannot rank trials on a confidence interval. Rank on an
`EvalMetrics` field instead, then inspect the interval afterwards:

```python
opt = strategy.optimize(score_fn, sampler="grid", calc_bootstrap=True)
if opt.result.bootstrap is not None:
    print(opt.result.bootstrap.conf_intervals)
    print(opt.result.bootstrap.drawdown_conf)
```

- `conf_intervals` is 6 rows by 2 columns, MultiIndexed on `name`
  (`"Profit Factor"`, `"Sharpe Ratio"`) then `conf` (`"97.5%"`,
  `"95%"`, `"90%"`), with columns `["lower", "upper"]`. Read it as
  `ci.loc[("Sharpe Ratio", "95%"), "lower"]`.
- `drawdown_conf` is 4 rows by 2 columns, indexed on `conf`
  (`"99.9%"`, `"99%"`, `"95%"`, `"90%"`) with columns
  `["amount", "percent"]`. Values are negative upper bounds: the worst
  drawdown you would expect not to exceed at that confidence.
- Profit factor and Sharpe use the **BCa** (bias corrected and
  accelerated) bootstrap; the drawdown bounds are a plain percentile
  bootstrap. Returns are resampled per bar, not per trade.
- Sharpe intervals are annualized only when
  `StrategyConfig.bars_per_year` is set.
- It changes no `metrics_df` value, so it never alters a score.
- Cost is `bars * bootstrap_samples` per `TestResult`. It is paid once
  on the final result, not once per trial, so it is cheap next to the
  search itself.
- `optimize` defaults to `seed=None`, unlike `backtest`/`walkforward`
  which default to `seed=42`. Pass `seed=` for reproducible intervals.

## Fill Prices and End-of-Data Exits

Both matter here because they change what a `score_fn` reads.

Orders fill at `PriceType.MIDDLE` — the midpoint of the low and high
of the **execution** bar, one bar after the signal under the default
`buy_delay`/`sell_delay` of `1` — unless `ctx.buy_fill_price` /
`ctx.sell_fill_price` is set to a `PriceType`, a number, or a
`(symbol, bar_data)` callable. A limit price only gates the fill: the
order still fills at the fill price, never at the limit.

`StrategyConfig.exit_on_last_bar` defaults to `False`, which leaves
the final position open and out of `trade_count`, `win_rate`,
`total_pnl` and every other trade-level metric. A `score_fn` that
reads realized P&L will rank trials on an unclosed book unless it is
turned on, so set `exit_on_last_bar=True` for any realized-P&L score.
Bar-level scores (`sharpe`, `max_drawdown`, `profit_factor`) are
computed from per-bar market value and are barely affected.

`optimize` scopes end-of-data exits deliberately: each tuning trial
liquidates at the end of its own window so trials score against a
closed book, while the stitched `opt.result` uses the whole dataset so
it matches an equivalent `walkforward()` run exactly.

## Models and Optimization

`strategy.optimize` raises for trainable model sources: retraining a
model inside every trial would multiply cost and blur what is being
optimized. Pretrained models (`pybroker.model(..., pretrained=True)`)
are supported and loaded once per train window, then reused across that
window's trials.

Tune trainable model hyperparameters inside `train_fn` with a
validation search over the TRAIN window (for example
`GridSearchCV(cv=TimeSeriesSplit(...))`), or compare registrations
across walkforward runs. `pybroker.hyperparam` is for strategy-level
parameters, not model training.

## Session Hygiene and Debugging

Put these at the top of agent-run optimization scripts:

```python
pybroker.enable_data_source_cache("my_strategy")  # skip refetching
pybroker.disable_progress_bar()  # progress bars flood AI context
pybroker.disable_logging()       # one backtest per trial; keep quiet
```

`optimize` runs one complete backtest per trial, so per-run logging is
multiplied by the trial count — `disable_logging()` matters more here
than anywhere else. `verbose=False` (the default) already suppresses
per-trial logs and Optuna's own trial logging; leave it off in agent
runs. Configure workers once with `pybroker.set_parallel(n_jobs=...)`
(Ray is available as a backend for distributed runs).

Never use pandas to implement indicator or execution logic. Indicators
are plain NumPy over `BarData` arrays (use Numba `@njit` for explicit
loops), and execution functions read `ctx.*` NumPy arrays — no
`pd.Series`/`pd.DataFrame` construction and no
`.rolling`/`.ewm`/`.shift`/`.apply` in either.

Numba debug toggle: a Numba compilation or typing error in an `@njit`
indicator is easier to read as plain Python. Re-run once with JIT
disabled to get a normal traceback at the offending line, then
re-enable JIT for the real run:

```bash
NUMBA_DISABLE_JIT=1 python my_strategy.py
```

Debug tuned-indicator failures serially (`parallel_indicators=False`)
before enabling parallel indicator computation; joblib wraps worker
tracebacks and obscures the original error.

## Optimization Checklist

- Syntax-check created Python files with `python -m py_compile <file>`.
- Every lattice is valid: one numeric type, positive `step`,
  `high - low` a multiple of `step`, and `default` within bounds.
- Every declared hyperparam is reachable from an execution (indicator
  kwarg, `hyperparams=[...]`, or a position-limit setting); dangling
  registrations warn and inflate the grid.
- `score_fn` guards `Optional` metrics and penalizes degenerate
  results (for example, too few trades) instead of returning `None`.
- `StrategyConfig(exit_on_last_bar=True)` is set when `score_fn` reads
  realized P&L or trade counts; the library default of `False` leaves
  each trial's final position out of every trade-level metric.
- `calc_bootstrap` is passed to `optimize`, never to `StrategyConfig`,
  and is not read by `score_fn`.
- `seed=` is set when reproducibility matters; `optimize` does not
  default to a seed.
- Grid size is sane: check `len(hp)` per hyperparam and multiply;
  coarsen `step`, pin values, or switch samplers before exceeding the
  1000-combination warning threshold.
- With `windows > 1`, per-window params come from `opt.windows`, not
  `best_params` (last window only), and stability across windows was
  checked.
- Reported performance comes from the held-out `opt.result` metrics,
  never from the in-sample `best_score`.
- Generated scripts start with `pybroker.disable_progress_bar()` and
  `pybroker.enable_data_source_cache(...)` (or `enable_caches`), plus
  `pybroker.disable_logging()` for optimize runs.
- No pandas in indicator or execution logic; indicators return
  full-length NaN-padded arrays.
- Never assume optional model libraries are installed — name the
  required `pip install`s (Optuna itself ships with PyBroker).
