# PyBroker Model Training Patterns

## `pybroker.model` API Map

Register every model with `pybroker.model` before adding executions:

```python
model_source = pybroker.model(
    name,               # unique model name
    fn,                 # train fn, or load fn when pretrained=True
    indicators=None,    # Indicators computed into train/test DataFrames
    lags=None,          # build a lag feature matrix with this many lags
    lag_cols=None,      # columns/Indicators to lag (requires lags)
    per_bar=False,      # call predict_fn once per bar (requires predict_fn)
    input_data_fn=None, # customize model input; one row per bar
    predict_fn=None,    # custom prediction; default calls model.predict
    pretrained=False,   # fn loads a model instead of training one
    pooled=False,       # train once across all symbols in the execution
)
```

Training function signatures:

- Per-symbol (default): `fn(symbol, train_data, test_data)` where the frames
  are that symbol's train and test windows.
- Pooled (`pooled=True`): `fn(symbols, train_data, test_data)` where
  `symbols` is a tuple sorted in ascending order and both frames contain a
  `symbol` column with each symbol's rows grouped in that order. A listed
  symbol can have zero rows, such as when `lags` drops all of its rows
  with the lag warmup.
- Pretrained (`pretrained=True`): `fn(symbol, train_start_date,
  train_end_date)` loads and returns an already-trained model.
- With `lags`, a training `fn` is additionally called with `lag_train=` and
  `lag_test=` keyword arguments and must accept both.

Return either the trained model, or `(model, input_cols)` to pin the columns
used as prediction input. When only a model is returned, the training
DataFrame's columns are used (for pooled models, minus the `symbol` column).

Lag feature matrix: shape `(n_rows, len(lag_cols) * (lags + 1))`, one block
per `lag_cols` entry in declaration order; each block holds the column's
current value followed by lags `1` through `lags`. Rows whose lags are
undefined are dropped from training data only; at prediction time, test-window
rows use real values carried over from the preceding train window. Because
the current bar's value is the first feature of each block, the prediction
target must be the NEXT bar — using the current bar's value as the target
would leak it.

## Reading Predictions in Executions

```python
def exec_fn(ctx):
    pred = ctx.preds("model_name")[-1]   # predictions up to current bar
    model = ctx.model("model_name")      # the trained model instance
    df = ctx.input("model_name")         # model input rows up to current bar
    wk = ctx.interval("weekly").preds("model_name")  # interval-bound model
```

Bind a model to an interval with `model_source.intervals("weekly")` when
passing it to `add_execution(models=...)`; it then trains on exactly the
listed intervals' compressed bars (together with its registered
indicators) and holds out `lookahead` bars in that interval's units. The
base-timeframe model is trained only when `"base"` is included in the
binding; an unbound model trains on the base timeframe. Read the
per-interval predictions with `ctx.interval(...).preds`, not `ctx.preds`.
The `intervals=` parameter of `add_execution` provides bars only and
never trains models.

## Library Recipes

### scikit-learn regressor (next-bar return)

```python
from sklearn.linear_model import LinearRegression

def train_slr(symbol, train_data, test_data):
    df = train_data.copy()  # never mutate the input frame
    # Target is the NEXT bar's return to match walkforward lookahead=1.
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    df = df.dropna()
    model = LinearRegression()
    model.fit(df[["cmma_20"]], df["target"])
    return model, ["cmma_20"]

model_slr = pybroker.model("slr", train_slr, indicators=[cmma_20])
```

### scikit-learn classifier with predict_proba

`predict_fn` must return exactly one prediction per input row, so slice a
single class-probability column instead of returning the full
`(n_rows, n_classes)` matrix:

```python
from sklearn.ensemble import GradientBoostingClassifier

def train_clf(symbol, train_data, test_data):
    df = train_data.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()
    model = GradientBoostingClassifier(random_state=42)
    model.fit(df[["cmma_20"]], df["target"])
    return model, ["cmma_20"]

def predict_up_prob(model, data):
    return model.predict_proba(data)[:, 1]  # P(next bar up), one per row

clf_model = pybroker.model(
    "clf", train_clf, indicators=[cmma_20], predict_fn=predict_up_prob
)
```

### XGBoost / LightGBM / CatBoost (scikit-learn API)

All three gradient-boosting libraries follow the same scikit-learn shape
(`CatBoostRegressor` works exactly like the example below). Carve any
early-stopping validation split out of the TRAIN window — never pass
`test_data` to `eval_set`:

```python
from xgboost import XGBRegressor  # or: from lightgbm import LGBMRegressor

def train_xgb(symbol, train_data, test_data):
    df = train_data.copy()
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    df = df.dropna()
    split = int(len(df) * 0.8)  # last 20% of TRAIN window for validation
    X, y = df[["cmma_20"]], df["target"]
    model = XGBRegressor(n_estimators=200, early_stopping_rounds=10)
    model.fit(X[:split], y[:split], eval_set=[(X[split:], y[split:])],
              verbose=False)
    return model, ["cmma_20"]

xgb_model = pybroker.model("xgb", train_xgb, indicators=[cmma_20])
```

### PyTorch / Keras neural networks

Return the trained network as the model and supply a `predict_fn` that runs
inference and returns one prediction per row. Seed the framework for
reproducible backtests and keep models on CPU:

```python
import torch
from torch import nn

def train_mlp(symbol, train_data, test_data):
    torch.manual_seed(42)  # deterministic backtests
    df = train_data.copy()
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    df = df.dropna()
    X = torch.tensor(df[["cmma_20"]].to_numpy(), dtype=torch.float32)
    y = torch.tensor(df["target"].to_numpy(), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(X).squeeze(-1), y)
        loss.backward()
        opt.step()
    return net, ["cmma_20"]

def predict_mlp(net, data):
    X = torch.tensor(data.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        return net(X).squeeze(-1).numpy()  # one prediction per row

mlp_model = pybroker.model(
    "mlp", train_mlp, indicators=[cmma_20], predict_fn=predict_mlp
)
```

With `lags`, `predict_fn` receives the lag matrix as a NumPy array — build
the tensor from it directly (`torch.tensor(data, dtype=torch.float32)`). A
Keras model follows the same shape: seed with `keras.utils.set_random_seed`,
fit inside `train_fn`, and use `model.predict(data, verbose=0).ravel()` in
`predict_fn` so the `(n_rows, 1)` output flattens to one value per row. For
recurrent models (LSTM) that need a window of context per prediction, prefer
`lags`/`lag_cols` to build the windows, or `per_bar=True` when each
prediction must consume the full series.

### GARCH / autoregressive models with `per_bar=True` (requires `pip install arch`)

Vectorized prediction fails for autoregressive models, which need the series
up to each bar. `per_bar=True` calls `predict_fn` once per bar with rows up
to and including the current bar; it must return a scalar and cannot be
combined with `pooled=True`:

```python
import arch

def train_garch(symbol, train_data, test_data):
    returns = (
        pd.concat((train_data["log_return"], test_data["log_return"]))
        .dropna().to_numpy() * 100
    )
    n_train = int(train_data["log_return"].count())
    am = arch.arch_model(returns, vol="GARCH", p=1, q=1)
    # last_obs restricts estimation to the train window (no leakage).
    return am.fit(last_obs=n_train, disp="off")

def predict_garch(model, data):
    pos = model.fit_stop + len(data) - 1
    forecast = model.forecast(horizon=1, start=pos)
    return float(np.sqrt(forecast.variance.to_numpy()[0, 0]) / 100
                 * np.sqrt(252))

garch_model = pybroker.model(
    "garch", train_garch, predict_fn=predict_garch,
    indicators=[log_return_ind], per_bar=True,
)
```

### statsmodels ARIMA / SARIMAX with `per_bar=True`

Fit parameters on the train window only, then re-filter those SAME
parameters over the series up to each bar with `results.apply` — no
refitting, no leakage — and forecast one step ahead:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_arima(symbol, train_data, test_data):
    returns = train_data["log_return"].dropna().to_numpy()
    return SARIMAX(returns, order=(1, 0, 1)).fit(disp=False)

def predict_arima(result, data):
    series = data["log_return"].dropna().to_numpy()
    return float(result.apply(series).forecast(1)[0])  # next-bar forecast

arima_model = pybroker.model(
    "arima", train_arima, predict_fn=predict_arima,
    indicators=[log_return_ind], per_bar=True,
)
```

The same fit-then-`apply` pattern covers other statsmodels state-space
models (`UnobservedComponents`, `ExponentialSmoothing` via its statespace
variant).

### Lagged features (`lags` / `lag_cols`)

The training `fn` must accept `lag_train` and `lag_test`; align the target
one bar forward:

```python
from sklearn.ensemble import RandomForestRegressor

def train_forest(symbol, train_data, test_data, lag_train, lag_test):
    rets = train_data["log_return"].to_numpy()
    forest = RandomForestRegressor(random_state=42)
    forest.fit(lag_train[:-1], rets[1:])  # next-bar target alignment
    return forest

forest_model = pybroker.model(
    "forest", train_forest, predict_fn=lambda model, data:
    model.predict(data), lags=3, lag_cols=[log_return_ind],
)
```

With `lags`, `predict_fn` receives the lag feature matrix (a NumPy array)
in place of the input DataFrame.

### Pooled multi-symbol model (`pooled=True`)

Group per-symbol operations so targets never cross a symbol boundary, and
pin `input_cols` to keep the `symbol` column out of model input:

```python
def train_pooled(symbols, train_data, test_data):
    df = train_data.copy()
    next_close = df.groupby("symbol")["close"].shift(-1)
    df["target"] = next_close / df["close"] - 1
    df = df.dropna()
    model = LinearRegression()
    model.fit(df[["cmma_20"]], df["target"])
    return model, ["cmma_20"]

pooled_model = pybroker.model(
    "slr", train_pooled, indicators=[cmma_20], pooled=True
)
```

### Ensembles and custom model objects

scikit-learn ensembles (`VotingRegressor`, `StackingRegressor`) are just
regressors — train and return them like any other model. For arbitrary
blends or libraries without a scikit-learn API, return any object: the
default prediction path calls `model.predict(input_frame)`, and `predict_fn`
covers everything else:

```python
class BlendModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

def train_blend(symbol, train_data, test_data):
    df = train_data.copy()
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    df = df.dropna()
    X, y = df[["cmma_20"]], df["target"]
    models = [
        LinearRegression().fit(X, y),
        RandomForestRegressor(random_state=42).fit(X, y),
    ]
    return BlendModel(models), ["cmma_20"]

blend_model = pybroker.model("blend", train_blend, indicators=[cmma_20])
```

The same wrapper pattern adapts any library — a Prophet forecaster, an ONNX
runtime session, or a hand-written rule — as long as predictions come back
one per input row.

### Regime models (hmmlearn)

Unsupervised regime detectors plug into the same contract; the "prediction"
is a state label per bar. Hidden-state numbering is arbitrary per fit, so
map states to meanings (for example by trained variance) inside `train_fn`
and trade on the mapped label:

```python
from hmmlearn.hmm import GaussianHMM

def train_regimes(symbol, train_data, test_data):
    X = train_data[["log_return"]].dropna().to_numpy()
    hmm = GaussianHMM(n_components=2, random_state=42).fit(X)
    return hmm, ["log_return"]

def predict_regime(hmm, data):
    return hmm.predict(data.to_numpy())  # one state label per row

regime_model = pybroker.model(
    "regime", train_regimes, indicators=[log_return_ind],
    predict_fn=predict_regime,
)
```

### Pretrained model (`pretrained=True`)

```python
import joblib

def load_model(symbol, train_start_date, train_end_date):
    model = joblib.load(f"models/{symbol}.joblib")
    return model, ["cmma_20"]

loaded = pybroker.model("loaded", load_model, indicators=[cmma_20],
                        pretrained=True)
```

Pretrained models are the only kind `strategy.optimize` accepts. The same
shape loads `torch.load(...)` checkpoints or an
`onnxruntime.InferenceSession` — pair those with a `predict_fn` that calls
the runtime and returns one prediction per row.

### Hyperparameter tuning

`strategy.optimize` raises for trainable model sources. Tune model
hyperparameters inside `train_fn` with a validation search over the TRAIN
window, or compare registrations across walkforward runs:

```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def train_tuned(symbol, train_data, test_data):
    df = train_data.copy()
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    df = df.dropna()
    search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"max_depth": [3, 5, 10]},
        cv=TimeSeriesSplit(n_splits=3),
    )
    search.fit(df[["cmma_20"]], df["target"])
    return search.best_estimator_, ["cmma_20"]
```

`pybroker.hyperparam(name, default=..., low=..., high=..., step=...)` is for
strategy-level parameters read with `ctx.hyperparam(name)`, not for model
training.

### Custom feature columns

Register any non-OHLCV DataFrame columns before using them as features:

```python
pybroker.register_columns("sentiment")
# 'sentiment' now flows into train/test frames and inferred input columns.
```

## Fill Prices, End-of-Data Exits, and Bootstrap Metrics

Defaults that decide what a model's backtest actually reports.

Orders fill at `PriceType.MIDDLE` — the midpoint of the low and high
of the **execution** bar, one bar after the prediction under the
default `buy_delay`/`sell_delay` of `1`, so `PriceType.CLOSE` means
the next bar's close. `PriceType` offers `OPEN`, `HIGH`, `LOW`,
`CLOSE`, `MIDDLE` (`low + (high - low) / 2`, the default), and
`AVERAGE` (`(open + low + high + close) / 4`). `ctx.buy_fill_price` /
`ctx.sell_fill_price` also accept a number or a `(symbol, bar_data)`
callable and read back as `None` until set. A limit price only gates
the fill: the order still fills at the fill price, never at the limit.

`StrategyConfig.exit_on_last_bar` defaults to `False`, leaving the
final position open and out of `trade_count`, `win_rate`, `total_pnl`
and every other trade-level metric, with its P&L in `unrealized_pnl`.
Set `exit_on_last_bar=True` whenever trade statistics are reported.
In `walkforward` it fires only on each symbol's true final bar, never
at window boundaries, so positions carry across windows as usual.

`calc_bootstrap` is a parameter of `walkforward`/`backtest`, **not** a
`StrategyConfig` field, and defaults to `False`. It is the natural
companion to walkforward analysis: it puts confidence intervals around
a model's out-of-sample edge instead of a single point estimate.

```python
config = StrategyConfig(bootstrap_samples=10_000, bars_per_year=252)
result = strategy.walkforward(
    windows=3, train_size=0.5, calc_bootstrap=True
)
if result.bootstrap is not None:
    print(result.bootstrap.conf_intervals)  # 6x2, (name, conf) index
    print(result.bootstrap.drawdown_conf)   # 4x2, conf index
```

- `conf_intervals` is MultiIndexed on `name` (`"Profit Factor"`,
  `"Sharpe Ratio"`) then `conf` (`"97.5%"`, `"95%"`, `"90%"`), with
  columns `["lower", "upper"]`. A profit factor interval whose
  `lower` sits below `1` means the model's edge is not distinguishable
  from noise at that confidence.
- `drawdown_conf` is indexed on `conf` (`"99.9%"`, `"99%"`, `"95%"`,
  `"90%"`) with columns `["amount", "percent"]`, both negative upper
  bounds.
- Profit factor and Sharpe use the **BCa** (bias corrected and
  accelerated) bootstrap; drawdown bounds are a plain percentile
  bootstrap. Returns are resampled per bar, not per trade.
- Sharpe intervals are annualized only when `bars_per_year` is set.
- It changes no `metrics_df` value, and `result.bootstrap` stays
  `None` under `train_only=True`.
- Cost is `bars * bootstrap_samples`, paid once on the stitched
  `TestResult` rather than once per walkforward window.

## Reporting Results

Print `result.metrics_df` as the human-readable summary. For
structured output (agent parsing, saved report files, downstream
tools), `result.to_json()` returns a JSON-safe dict and
`result.to_json_str()` strict JSON text. The default payload carries
metrics, trades, orders, and bootstrap capped at `max_rows=100` rows
per table; `symbols=` filters to specific tickers; `include=` opts
into `portfolio`/`positions`/`metrics_df`/`signals`/`stops` (with
`StrategyConfig(return_signals=True)`, `signals` carries per-symbol
model predictions). Dates serialize as naive-UTC ISO strings, NaN as
`null`, and legitimately infinite metrics as
`"Infinity"`/`"-Infinity"`.

## Session Hygiene and Debugging

Put these at the top of agent-run scripts:

```python
pybroker.enable_data_source_cache("my_strategy")  # skip refetching data
pybroker.enable_caches("my_strategy")   # data + indicators + trained models
pybroker.disable_progress_bar()  # progress bars flood AI token context
```

Never use pandas to implement indicator or execution logic. Indicators are
plain NumPy over `BarData` arrays (use Numba `@njit` for explicit loops),
and execution functions read `ctx.*` NumPy arrays — no
`pd.Series`/`pd.DataFrame` construction and no
`.rolling`/`.ewm`/`.shift`/`.apply` in either. Pandas belongs only in
`train_fn`/`input_data_fn`, where PyBroker hands you DataFrames.

```python
# Good: vectorized NumPy indicator.
def log_return(bar_data):
    close = bar_data.close
    ret = np.full_like(close, np.nan)
    ret[1:] = np.diff(np.log(close))
    return ret

# Bad: pandas inside an indicator (slow, unnecessary).
def log_return_slow(bar_data):
    return pd.Series(bar_data.close).apply(np.log).diff().to_numpy()
```

Numba debug toggle: a Numba compilation or typing error in an `@njit`
indicator is easier to read as plain Python. Re-run once with JIT disabled to
get a normal traceback at the offending line, then re-enable JIT for the
real run:

```bash
NUMBA_DISABLE_JIT=1 python my_strategy.py
```

Common causes: mixed dtypes in one array, untyped `np.array([...])`
construction, or Python objects (lists of strings, dicts, pandas) inside the
`@njit` function.

## Leakage Checklist

- `lookahead` equals the number of bars ahead of the prediction target
  (next-bar target => `lookahead=1`).
- The training target is shifted forward (`shift(-1)` / `y[1:]`); no
  forward-shifted values survive into features.
- Feature indicators never negative-index into full-length arrays (a
  negative index silently wraps to the end of the series — the future),
  and novel indicator logic passes the bump-last-bar check: change only
  the final input bar and assert every earlier output is unchanged.
- With `lags`, fit `lag_train[:-1]` against `target[1:]`.
- Pooled targets are built with `groupby("symbol")` so labels never cross a
  symbol boundary.
- Scalers, encoders, and early-stopping validation sets come from the train
  window only.
- `input_data_fn` returns exactly one row per bar; a vectorized `predict_fn`
  returns one prediction per input row; a `per_bar=True` `predict_fn`
  returns a scalar.
- `test_data` is used only for held-out evaluation inside `train_fn`, never
  for fitting, tuning, or early stopping.
- `shuffle` stays `False` unless deliberately shuffling training rows.

## Validation Checklist

- Syntax-check created Python files with `python -m py_compile <file>`.
- Generated scripts call `pybroker.disable_progress_bar()` and
  `pybroker.enable_caches(...)` (or `enable_data_source_cache(...)`).
- Every imported model or data-source library names its `pip install`
  (for example `pip install yfinance scikit-learn`); none is a PyBroker
  dependency, so never assume one is importable.
- When the network or a data-source package is unavailable, run with a
  tiny local DataFrame passed to `Strategy` instead.
- Report `result.metrics_df` as the human-readable summary, with
  `result.to_json()` / `to_json_str()` for structured output.
- The Leakage Checklist above passes.
