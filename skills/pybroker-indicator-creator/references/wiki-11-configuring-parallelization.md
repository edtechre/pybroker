# Configuring Parallelization

Source: `docs/source/notebooks/11. Configuring Parallelization.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Configuring Parallelization

**PyBroker** uses [Joblib](https://joblib.readthedocs.io/) to compute indicators, train models, and optimize parameters in parallel.

## Setting Workers

[set_parallel](https://www.pybroker.com/en/latest/reference/pybroker.parallel.html#pybroker.parallel.set_parallel) updates the global Joblib configuration. The [n_jobs](https://www.pybroker.com/en/latest/reference/pybroker.parallel.html#pybroker.parallel.ParallelConfig.n_jobs) parameter specifies the number of worker jobs: `-1` (the default) uses all available CPU cores, and `1` runs sequentially. Read the current settings with [get_parallel_config](https://www.pybroker.com/en/latest/reference/pybroker.parallel.html#pybroker.parallel.get_parallel_config):

```python
from pybroker import set_parallel, get_parallel_config

# Use a fixed number of workers.
set_parallel(n_jobs=4)
print(get_parallel_config())

# Or disable parallel execution entirely.
set_parallel(n_jobs=1)
print(get_parallel_config())
```

## Parallel Indicators

Indicators are computed **per symbol**: all indicators for a given symbol are grouped into a single task, and one task is dispatched per symbol. Passing `parallel_indicators=True` to [backtest](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest), [walkforward](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.walkforward), or [optimize](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.optimize) runs these tasks across the configured workers (defaulting to `False`).

To see this in action, let's backtest a moving average crossover:

```python
import numpy as np
import pybroker
from pybroker import Strategy, YFinance, sumv

pybroker.enable_data_source_cache("parallelization")
set_parallel(n_jobs=-1)


def sma(bar_data, period):
    return sumv(bar_data.close, period) / period


sma_20 = pybroker.indicator("sma_20", sma, period=20)


def sma_cross(ctx):
    sma_vals = ctx.indicator("sma_20")
    if np.isnan(sma_vals[-1]):
        return
    pos = ctx.long_pos()
    if not pos and ctx.close[-1] > sma_vals[-1]:
        ctx.buy_shares = 100
    elif pos and ctx.close[-1] < sma_vals[-1]:
        ctx.sell_all_shares()


yfinance = YFinance()
strategy = Strategy(yfinance, start_date="1/1/2021", end_date="1/1/2026")
strategy.add_execution(sma_cross, ["V", "MA", "AXP"], indicators=sma_20)

result = strategy.backtest(parallel_indicators=True, warmup=20)
result.metrics_df.head()
```

Standalone indicator computation with an [IndicatorSet](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.IndicatorSet) accepts the same `parallel_indicators` flag:

```python
from pybroker import IndicatorSet

df = yfinance.query(
    ["V", "MA", "AXP"], start_date="1/1/2021", end_date="1/1/2026"
)
ind_set = IndicatorSet()
ind_set.add(sma_20)
ind_set(df, parallel_indicators=True).tail()
```

## Parallel Model Training

Model training runs serially by default. Passing `parallel_models=True` to [backtest](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest) or [walkforward](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.walkforward) trains each model in its own task across the configured workers.

This example adapts the linear regression model from [Training a Model](https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html):

```python
from sklearn.linear_model import LinearRegression

from pybroker.indicator import close_minus_ma

cmma_20 = close_minus_ma("cmma_20", lookback=20, atr_length=20)


def train_slr(symbol, train_data, test_data):
    # Predict the next bar's return given the 20-day CMMA.
    prev_close = train_data["close"].shift(1)
    daily_returns = (train_data["close"] - prev_close) / prev_close
    train_data["pred"] = daily_returns.shift(-1)
    train_data = train_data.dropna()
    model = LinearRegression()
    model.fit(train_data[["cmma_20"]], train_data[["pred"]])
    # Return the trained model and columns to use as input data.
    return model, ["cmma_20"]


model_slr = pybroker.model("slr", train_slr, indicators=[cmma_20])


def hold_long(ctx):
    if not ctx.long_pos():
        if ctx.preds("slr")[-1] > 0:
            ctx.buy_shares = 100
    elif ctx.preds("slr")[-1] < 0:
        ctx.sell_all_shares()


model_strategy = Strategy(yfinance, start_date="1/1/2021", end_date="1/1/2026")
model_strategy.add_execution(hold_long, ["V", "MA", "AXP"], models=model_slr)
result = model_strategy.walkforward(
    windows=2,
    train_size=0.5,
    lookahead=1,
    warmup=20,
    parallel_models=True,
)
result.metrics_df.head()
```

## Using Ray as the Backend

[Ray](https://www.ray.io/) can distribute the same work across many cores or an entire cluster. Install it with `pip install ray` and then register it with Joblib via [register_ray](https://docs.ray.io/en/latest/ray-more-libs/joblib.html) (from `ray.util.joblib`):

```python
import ray
from ray.util.joblib import register_ray

ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
register_ray()
```

After registering Ray, pass `backend="ray"` to [set_parallel](https://www.pybroker.com/en/latest/reference/pybroker.parallel.html#pybroker.parallel.set_parallel) to make it available as a backend:

```python
set_parallel(backend="ray", n_jobs=-1)
print(get_parallel_config())
```

**PyBroker** will now use the Ray backend for all parallel tasks. For example, calling [backtest](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.backtest) with `parallel_indicators=True`:

```python
result = strategy.backtest(parallel_indicators=True, warmup=20)
print(f"Total return: {result.metrics.total_return_pct:.2f}%")

ray.shutdown()
```

We will explore [parameter optimization](https://www.pybroker.com/en/latest/notebooks/12.%20Parameter%20Optimization.html) in the next notebook, which can also be parallelized in certain scenarios.
