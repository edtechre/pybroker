# Time Series Models

Source: `docs/source/notebooks/16. Time Series Models.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Time Series Models

**PyBroker v2** introduces support for backtesting time series models. Instead of relying on a single row of features, these models make predictions based on a series' own past values.

To show how this works, we will backtest two different strategies. The first relies on a volatility forecast from a [GARCH(1,1)](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity) model built with the [arch](https://arch.readthedocs.io/) library. The second strategy uses a rolling regression that is refit on every bar. Since **PyBroker** does not include `arch` by default, you must install it first by running `pip install arch`.

```python
import arch
import numpy as np
import pandas as pd
import pybroker
from pybroker import Strategy, YFinance

pybroker.enable_data_source_cache("time_series")
```

## Forecasting Volatility with GARCH

GARCH will model the volatility of a return series, so we start by defining an [indicator](https://www.pybroker.com/en/latest/reference/pybroker.indicator.html#pybroker.indicator.indicator) for log returns:

```python
def log_return(bar_data):
    close = bar_data.close
    ret = np.full_like(close, np.nan)
    ret[1:] = np.diff(np.log(close))
    return ret


log_return_ind = pybroker.indicator("log_return", log_return)
```

The training function scales the log returns to percentages for more reliable estimation:

```python
def train_garch(symbol, train_data, test_data):
    returns = (
        pd.concat((train_data["log_return"], test_data["log_return"]))
        .dropna()
        .to_numpy()
        * 100
    )
    n_train = int(train_data["log_return"].count())
    # Estimate on the train window only; the test returns are held out
    # for forecasting.
    am = arch.arch_model(returns, vol="GARCH", p=1, q=1)
    return am.fit(last_obs=n_train, disp="off")
```

The model is built using the combined returns from both the train and test windows. However, passing `last_obs` ensures that parameter estimation relies solely on the train window and isolates the test returns to prevent data leakage.

For every test data bar, the prediction function uses the trained model to forecast the variance of the next bar:

```python
def predict_garch(model, data):
    # Position of the current bar in the model's return series.
    pos = model.fit_stop + len(data) - 1
    # Forecast the next bar's variance from the trained model.
    forecast = model.forecast(horizon=1, start=pos)
    variance = forecast.variance.to_numpy()[0, 0]
    # Annualize the one-day volatility forecast.
    return np.sqrt(variance) / 100 * np.sqrt(252)
```

Because the model already stores the full return series, the model input is only the location of the current bar. The forecast then only uses returns up to that point.

By default, **PyBroker** passes all test window data to the model in a single call. While this vectorized approach is efficient, it fails for autoregressive models since they rely on the previous step's output to generate their next forecast. Passing `per_bar=True` to [model](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.model) will then call [predict_fn](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.model) once per bar:

```python
garch_model = pybroker.model(
    "garch",
    train_garch,
    predict_fn=predict_garch,
    indicators=[log_return_ind],
    per_bar=True,
)
```

The strategy uses the volatility forecast from [ctx.preds](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.preds) as a regime filter. It enters a long position when forecast volatility is below the threshold, and exits when it rises above:

```python
VOL_THRESHOLD = 0.30


def vol_filter(ctx):
    pred_vol = ctx.preds("garch")[-1]
    if not ctx.long_pos():
        # Enter while forecast volatility is below the threshold.
        if pred_vol < VOL_THRESHOLD:
            ctx.buy_shares = ctx.calc_target_shares(0.5)
    elif pred_vol > VOL_THRESHOLD:
        # Exit when forecast volatility rises above the threshold.
        ctx.sell_all_shares()


strategy = Strategy(YFinance(), start_date="1/1/2021", end_date="1/1/2026")
strategy.add_execution(vol_filter, ["SBUX", "IBM"], models=garch_model)
result = strategy.walkforward(windows=2, train_size=0.5)
result.metrics_df.head(20)
```

## Random Forest on Lagged Returns

The second strategy trains a [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) to predict the next bar's return from lagged returns.

```python
from sklearn.ensemble import RandomForestRegressor


def train_forest(symbol, train_data, test_data, lag_train, lag_test):
    rets = train_data["log_return"].to_numpy()
    # Regress each next-bar return on the bar's return and its lags.
    forest = RandomForestRegressor(random_state=42)
    forest.fit(lag_train[:-1], rets[1:])
    return forest
```

The training function receives `lag_train` and `lag_test` parameters built from the model's [lags](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.model) configuration. Each is a feature matrix with one row per input bar where every row begins with the bar's `log_return` value followed by its lagged values.

Unlike the per-bar GARCH model, the predict function uses **PyBroker's** default behavior and evaluates the entire test window in a single vectorized call:

```python
def predict_forest(model, data):
    return model.predict(data)
```

Registering the model with [lags](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.model) set to `3` will include the past three lagged values for each column in [lag_cols](https://www.pybroker.com/en/latest/reference/pybroker.model.html#pybroker.model.model):

```python
forest_model = pybroker.model(
    "forest",
    train_forest,
    predict_fn=predict_forest,
    lags=3,
    lag_cols=[log_return_ind],
)
```

The strategy buys when [ctx.preds](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.preds) for the next-bar return is positive and exits when it is negative. After [add_execution](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.add_execution), we run a [walkforward](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.walkforward) backtest:

```python
def trade_forest(ctx):
    pred = ctx.preds("forest")[-1]
    if not ctx.long_pos():
        if pred > 0:
            ctx.buy_shares = ctx.calc_target_shares(0.5)
    elif pred < 0:
        ctx.sell_all_shares()


strategy.clear_executions()
strategy.add_execution(trade_forest, ["SBUX", "IBM"], models=forest_model)
result = strategy.walkforward(windows=2, train_size=0.5)
result.metrics_df.head(20)
```
