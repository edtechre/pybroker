# Parameter Optimization

Source: `docs/source/notebooks/12. Parameter Optimization.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Parameter Optimization

**PyBroker v2** supports parameterized strategies. This allows you to backtest strategies using different combinations of parameters and automatically select the best performers. This process is known as **parameter optimization** and is handled via the [Optuna framework](https://optuna.org/).

These strategy parameters are created as hyperparameters, as shown in the next section.

## Declaring Hyperparameters

A hyperparameter is a named, tunable value created with [hyperparam](https://www.pybroker.com/en/latest/reference/pybroker.optimize.html#pybroker.optimize.hyperparam). Each one has a [default](https://www.pybroker.com/en/latest/reference/pybroker.optimize.html#pybroker.optimize.Hyperparam.default) that regular backtests use, and a search range given by [low](https://www.pybroker.com/en/latest/reference/pybroker.optimize.html#pybroker.optimize.Hyperparam.low), [high](https://www.pybroker.com/en/latest/reference/pybroker.optimize.html#pybroker.optimize.Hyperparam.high), and [step](https://www.pybroker.com/en/latest/reference/pybroker.optimize.html#pybroker.optimize.Hyperparam.step). The candidate values start at `low` (inclusive), and then increase by `step` until `high` (inclusive).

A hyperparameter can be used to parameterize:

- **Indicators**: pass it as a keyword argument to [indicator](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.indicator).
- **Executions**: attach it with `hyperparams=` on [add_execution](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution) and read it with [ctx.hyperparam](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.hyperparam) in the execution function.

To demonstrate, we will build a moving average crossover strategy with two hyperparameters: the moving average's `period` and a `stop_pct` stop loss.

```python
import numpy as np
import pybroker
from pybroker import Strategy, YFinance, sumv

pybroker.enable_data_source_cache("parameter_optimization")

period = pybroker.hyperparam("period", default=30, low=10, high=50, step=10)
stop_pct = pybroker.hyperparam(
    "stop_pct", default=6.0, low=2.0, high=10.0, step=2.0
)


def sma(bar_data, period):
    return sumv(bar_data.close, period) / period


# The hyperparam is passed in place of a concrete period.
sma_ind = pybroker.indicator("sma", sma, period=period)


def sma_cross_stop(ctx):
    sma = ctx.indicator("sma")
    if np.isnan(sma[-1]):
        return
    pos = ctx.long_pos()
    if not pos and ctx.close[-1] > sma[-1]:
        ctx.buy_shares = 100
        ctx.stop_loss_pct = ctx.hyperparam("stop_pct")
    elif pos and ctx.close[-1] < sma[-1]:
        ctx.sell_all_shares()


strategy = Strategy(YFinance(), start_date="1/1/2021", end_date="1/1/2026")
strategy.add_execution(
    sma_cross_stop,
    ["MRK", "TGT", "ORCL"],
    indicators=sma_ind,
    hyperparams=[stop_pct],
)

result = strategy.backtest()
print(f"Total return: {result.metrics.total_return_pct:.2f}%")
```

## Optimizing with Grid Search

The [optimize](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.optimize) method splits the data into train and test windows as specified by `train_size` (`0.5` for 50/50 by default). As Optuna searches the parameter space, each selected combination is backtested on the train window and scored with `score_fn`. By default, this score is maximized (pass `direction="minimize"` to minimize instead). To guard against overfitting, the backtests are run exclusively on the training window. Once finished, the best performing parameters on the in-sample train window are evaluated on the out-of-sample test window.

In Optuna, a sampler is the algorithm that decides which parameter combinations to test during the optimization process. The default sampler is `grid`, which will evaluate every possible parameter combination. For our example, this results in 5 × 5 = 25 total trials. These trials are evaluated in parallel:

```python
def score_fn(result):
    return result.metrics.total_return_pct


opt_result = strategy.optimize(score_fn)
print("Best train params:", opt_result.best_params)
print("Best train score:", opt_result.best_score)
print(f"Total return: {opt_result.result.metrics.total_return_pct:.2f}%")
```

[best_params](https://www.pybroker.com/en/latest/reference/pybroker.optimize.html#pybroker.optimize.OptimizeResult.best_params) holds the best in-sample values, and [best_score](https://www.pybroker.com/en/latest/reference/pybroker.optimize.html#pybroker.optimize.OptimizeResult.best_score) is the score they earned on the train window.

Using the optimized parameters, we're able to see an improvment to the total return from before. The [result](https://www.pybroker.com/en/latest/reference/pybroker.optimize.html#pybroker.optimize.OptimizeResult.result) attribute contains the [TestResult](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.TestResult) of using the best parameters on the test window:

```python
opt_result.result.metrics_df.head()
```

## Optimizing with Tree-structured Parzen Estimator (TPE)

Grid search grows multiplicatively with each added hyperparameter. An alternative approach is using `sampler="tpe"`. Optuna's [TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) (Tree-structured Parzen Estimator) uses [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) by fitting a probability model to completed trials to suggest promising values for the next run. Note that `n_trials` is required for every sampler except `"grid"`, and providing a seed makes the parameter search reproducible.

Because TPE adapts based on earlier results, its trials always run sequentially:

```python
opt_result = strategy.optimize(score_fn, sampler="tpe", n_trials=15, seed=2)
print("Best params:", opt_result.best_params)
print("Best train score:", opt_result.best_score)
print(f"Total return: {opt_result.result.metrics.total_return_pct:.2f}%")
```

TPE recovered the same best values as the exhaustive grid search while evaluating only 15 of the 25 combinations.

## Other Samplers

`sampler="random"` chooses combinations uniformly at random with Optuna's [RandomSampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html). Like grid, random trials can be evaluated in parallel:

```python
opt_result = strategy.optimize(score_fn, sampler="random", n_trials=10, seed=1)
print("Best params:", opt_result.best_params)
```

Any [optuna.samplers.BaseSampler](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) instance can also be passed directly to [optimize](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.optimize) to customize the parameter search:

```python
from optuna.samplers import TPESampler

# Use fewer random startup trials before TPE's model takes over.
opt_result = strategy.optimize(
    score_fn, sampler=TPESampler(n_startup_trials=5), n_trials=15, seed=2
)
print("Best params:", opt_result.best_params)
print(f"Total return: {opt_result.result.metrics.total_return_pct:.2f}%")
```

Every optimization also returns the underlying [optuna.Study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html) for inspecting the trials:

```python
opt_result.study.trials_dataframe().head()
```

## Walkforward Optimization

Finally, [optimize](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.optimize) supports walkforward optimization. If you pass `> 1` to the `windows` parameter, PyBroker will independently tune the hyperparameters for each window's in-sample split and then combine the out-of-sample results:

```python
opt_result = strategy.optimize(score_fn, windows=3)

for i, window in enumerate(opt_result.windows):
    print(
        f"Window {i + 1} train: {window.train_start_date:%Y-%m-%d} to "
        f"{window.train_end_date:%Y-%m-%d}, "
        f"test: {window.test_start_date:%Y-%m-%d} to "
        f"{window.test_end_date:%Y-%m-%d}"
    )
    print(f"Window {i + 1} best params:", window.params)
```

Each window is tuned separately, so the best parameters can differ between windows. The combined result contains the [best_params](https://www.pybroker.com/en/latest/reference/pybroker.optimize.html#pybroker.optimize.OptimizeResult.best_params) of the **last** (most recent) window:

```python
print("Best params:", opt_result.best_params)
```
