# PyBroker Indicator Patterns

## `pybroker.indicator` API Map

Use these imports for most indicator work:

```python
import numpy as np
import pandas as pd  # fixtures/TA wrappers only — never indicator logic
import pybroker
from pybroker import indicator, highest, lowest, returns
from pybroker import highv, lowv, sumv, returnv, cross, atr
from numba import njit
```

Register every custom indicator with `pybroker.indicator`:

```python
ind = pybroker.indicator(
    name,     # unique global name; "@" is reserved (interval suffix)
    fn,       # fn(bar_data, **kwargs) -> one value per input bar
    **kwargs, # fixed values or pybroker.hyperparam objects
)
```

The contract:

- `fn` receives a `BarData` first: `date`, `open`, `high`, `low`, and
  `close` are NumPy float64 arrays in ascending chronological order;
  `volume` and `vwap` may be `None`; registered custom columns appear as
  extra attributes.
- The return value must be one-dimensional with one value per input bar.
  Left-pad warmup bars with NaN; never return a shortened array. A
  returned `pd.Series` is converted automatically.
- Registration is global. Re-registering a name silently overwrites, and
  names containing `@` are rejected.
- Standalone computation: `ind(df)` on a single-symbol DataFrame (needs
  `date`, `open`, `high`, `low`, `close` columns) returns a `pd.Series`
  indexed by date. Multi-symbol frames go through `IndicatorSet`.
- Feature diagnostics: `ind.iqr(df)` and `ind.relative_entropy(df)`
  score an indicator's spread and information content.

Wire indicators into a strategy:

```python
strategy.add_execution(exec_fn, ["AAPL", "MSFT"], indicators=[ind])

def exec_fn(ctx):
    value = ctx.indicator("name")[-1]        # this symbol, current bar
    spy = ctx.indicator("name", "SPY")[-1]   # another symbol's values
```

Indicators declared on a model (`pybroker.model(..., indicators=[...])`)
are computed automatically for that model's execution.

## Vectorized NumPy and Numba Kernels

Never use pandas to implement indicator logic, and never use it inside
per-bar execution functions: no `pd.Series`/`pd.DataFrame`
construction, no `.rolling`/`.ewm`/`.shift`/`.apply`. `BarData` fields
are already NumPy arrays — compute with vectorized NumPy. The only
sanctioned pandas in indicator code is the wrapper boundary in
"Wrapping Third-Party TA Libraries" below:

```python
# Good: vectorized NumPy indicator.
def log_return(data):
    close = data.close
    ret = np.full_like(close, np.nan)
    ret[1:] = np.diff(np.log(close))
    return ret

# Bad: pandas inside an indicator (slow, unnecessary).
def log_return_slow(data):
    return pd.Series(data.close).apply(np.log).diff().to_numpy()
```

For logic that needs an explicit loop, JIT-compile a nested kernel with
Numba `@njit`. `BarData` cannot cross the `@njit` boundary, so the
kernel takes plain NumPy arrays:

```python
def cmma(data, lookback):
    @njit  # bare @njit: Numba cannot disk-cache closures
    def kernel(values):
        n = len(values)
        out = np.full(n, np.nan)
        for i in range(lookback, n):
            ma = 0.0
            for j in range(i - lookback, i):
                ma += values[j]
            out[i] = values[i] - ma / lookback
        return out

    return kernel(data.close)

cmma_20 = pybroker.indicator("cmma_20", cmma, lookback=20)
```

Preallocate with `np.full(n, np.nan)` and index-fill; never grow arrays
in loops. If recompilation time on reruns matters, hoist the kernel to
module level, pass `lookback` as an argument, and decorate it with
`@njit(cache=True)` so the compiled code is cached to disk.

## Vector Helper Catalog

Six helpers are exported at top level and are `@njit`-compiled, so they
work both from plain Python and inside your own `@njit` kernels:

- `highv(array, n)` / `lowv(array, n)` — rolling highest/lowest over
  `n` bars.
- `sumv(array, n)` — rolling sum over `n` bars.
- `returnv(array, n=1)` — rolling returns (NaN, not inf, on zero base).
- `cross(a, b)` — 1 where `a` crosses above `b`, else 0.
- `atr(high, low, close, lookback)` — Average True Range.

```python
sma_50 = indicator("sma_50", lambda data, n: sumv(data.close, n) / n, n=50)
```

`pybroker.vect` holds the full technical set (import from the module):
`detrended_rsi`, `macd`, `stochastic`, `stochastic_rsi`, `linear_trend`,
`quadratic_trend`, `cubic_trend`, `adx`, `aroon_up`, `aroon_down`,
`aroon_diff`, `close_minus_ma`, `linear_deviation`,
`quadratic_deviation`, `cubic_deviation`, `price_intensity`,
`price_change_oscillator`, `intraday_intensity`, `money_flow`,
`reactivity`, `price_volume_fit`, `volume_weighted_ma_ratio`,
`normalized_on_balance_volume`, `delta_on_balance_volume`,
`normalized_positive_volume_index`, `normalized_negative_volume_index`,
`volume_momentum`, `laguerre_rsi`, plus the scalar stats `normal_cdf`
and `inverse_normal_cdf`. Exact signatures are in
`api-public-surface.md` under `src/pybroker/vect.py`.

## Built-In Indicator Factories

Every factory returns a registered `Indicator`. Only `highest`,
`lowest`, and `returns` are top-level; the rest live in the
`pybroker.indicator` module:

```python
from pybroker import highest, lowest, returns
from pybroker.indicator import atr, adx, macd, close_minus_ma

high_20 = highest("high_20", "high", period=20)
ret_5 = returns("ret_5", "close", period=5)
atr_14 = atr("atr_14", lookback=14)
```

The `field` argument of `highest`/`lowest`/`returns` (and the other
field-based factories) is a `BarData` attribute name — `"close"`,
`"high"`, or a registered custom column.

Full factory catalog in `pybroker.indicator`: `detrended_rsi`, `macd`,
`stochastic`, `stochastic_rsi`, `linear_trend`, `quadratic_trend`,
`cubic_trend`, `atr`, `adx`, `aroon_up`, `aroon_down`, `aroon_diff`,
`close_minus_ma`, `linear_deviation`, `quadratic_deviation`,
`cubic_deviation`, `price_intensity`, `price_change_oscillator`,
`intraday_intensity`, `money_flow`, `reactivity`, `price_volume_fit`,
`volume_weighted_ma_ratio`, `normalized_on_balance_volume`,
`delta_on_balance_volume`, `normalized_positive_volume_index`,
`normalized_negative_volume_index`, `volume_momentum`, `laguerre_rsi`.
These reimplement standard indicators with volatility normalization and
robust rescaling so values are comparable across symbols and regimes.

Name collision: top-level `pybroker.atr` is the vectorized kernel
`atr(high, low, close, lookback)`; the factory of the same name is
`pybroker.indicator.atr(name, lookback)`. The same split applies to the
other built-in names (`macd`, `adx`, ...).

## Wrapping Third-Party TA Libraries

None of these libraries is a PyBroker dependency: state the required
`pip install` in your answer and never assume one is importable. Rules
that apply to every wrapper:

- NumPy-native libraries (TA-Lib, tulipy) consume `BarData` arrays
  directly. Pandas-based libraries (pandas-ta, `ta`, finta) get a
  minimal `pd.Series`/`pd.DataFrame` built from `BarData` arrays at the
  wrapper boundary — the only sanctioned pandas in indicator code: it
  carries data into the library call and never implements indicator
  math itself.
- Output must be full length and one-dimensional. A returned
  `pd.Series` is converted automatically; pad libraries that drop
  warmup rows (tulipy).
- Multi-output functions register one indicator per output column. Each
  per-column indicator recomputes the underlying call independently;
  `pybroker.enable_indicator_cache` amortizes that across runs.
- Import the library inside the wrapper function when the script should
  stay loadable without it installed.

### TA-Lib (requires `pip install ta-lib` and the TA-Lib C library)

TA-Lib functions take float64 NumPy arrays and return full-length
arrays with leading NaNs — `BarData` fields pass straight through:

```python
import talib

rsi_14 = pybroker.indicator(
    "rsi_14", lambda data: talib.RSI(data.close, timeperiod=14)
)
```

Multi-output functions return a tuple; parameterize one wrapper by
output and register one indicator per element:

```python
def talib_macd(data, output):
    macd_line, signal, hist = talib.MACD(
        data.close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    return {"macd": macd_line, "signal": signal, "hist": hist}[output]

macd_line = pybroker.indicator("macd_line", talib_macd, output="macd")
macd_signal = pybroker.indicator(
    "macd_signal", talib_macd, output="signal"
)
macd_hist = pybroker.indicator("macd_hist", talib_macd, output="hist")
```

### pandas-ta (requires `pip install pandas-ta`)

Functions take and return `pd.Series`; the returned Series is converted
automatically:

```python
import pandas_ta as ta

def pta_rsi(data, length):
    return ta.rsi(pd.Series(data.close), length=length)

rsi_14 = pybroker.indicator("rsi_14", pta_rsi, length=14)
```

Multi-output functions return a DataFrame — select one column per
registered indicator. Column names vary across pandas-ta versions, so
verify with `list(out.columns)` instead of hard-trusting them:

```python
def pta_macd(data, column):
    out = ta.macd(pd.Series(data.close), fast=12, slow=26, signal=9)
    return out[column]

macd_line = pybroker.indicator(
    "macd_line", pta_macd, column="MACD_12_26_9"
)
```

### ta (requires `pip install ta`)

The `ta` library uses indicator classes over `pd.Series`. Leave
`fillna=False` (the default) so warmup bars stay NaN:

```python
from ta.momentum import RSIIndicator

def ta_rsi(data, window):
    close = pd.Series(data.close)
    return RSIIndicator(close=close, window=window).rsi()

rsi_14 = pybroker.indicator("rsi_14", ta_rsi, window=14)
```

Multi-output classes expose one method per output — for example
`ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)`
with `.macd()`, `.macd_signal()`, and `.macd_diff()` — so call one
method per registered indicator.

### tulipy (requires `pip install tulipy`)

tulipy returns SHORTENED arrays that drop the warmup rows, so
left-padding to full length is mandatory:

```python
import tulipy

def tulipy_rsi(data, period):
    out = np.full(len(data.close), np.nan)
    values = tulipy.rsi(np.ascontiguousarray(data.close), period=period)
    out[-len(values):] = values
    return out

rsi_14 = pybroker.indicator("rsi_14", tulipy_rsi, period=14)
```

Multi-output functions such as `tulipy.macd` return a tuple of
shortened arrays — pad each one the same way.

### finta (requires `pip install finta`)

finta takes an OHLC DataFrame with lowercase column names; build it
minimally at the boundary:

```python
from finta import TA

def finta_rsi(data, period):
    ohlc = pd.DataFrame(
        {
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
        }
    )
    return TA.RSI(ohlc, period=period)

rsi_14 = pybroker.indicator("rsi_14", finta_rsi, period=14)
```

Multi-output calls such as `TA.MACD(ohlc)` return a DataFrame
(`MACD`/`SIGNAL` columns) — select one column per registered indicator.

## Standalone Computation with `IndicatorSet`

Compute many indicators over a multi-symbol DataFrame without running a
backtest:

```python
from pybroker import IndicatorSet

ind_set = IndicatorSet()
ind_set.add(cmma_20, rsi_14)
frame = ind_set(df)  # df must include a "symbol" column
```

The output columns are `symbol`, `date`, then the indicator names in
sorted order, one row per input row. Pass `parallel_indicators=True` to
compute symbols in parallel. `IndicatorSet` never uses the disk cache.
For a single-symbol frame, calling the indicator directly — `ind(df)` —
returns a date-indexed `pd.Series`.

## Hyperparam Indicators

Pass `pybroker.hyperparam` objects as indicator kwargs to make the
indicator tunable:

```python
lookback = pybroker.hyperparam(
    "lookback", default=20, low=10, high=50, step=10
)
cmma_ind = pybroker.indicator("cmma", cmma, lookback=lookback)

strategy.add_execution(exec_fn, ["AAPL"], indicators=[cmma_ind])
opt = strategy.optimize(lambda r: r.metrics.total_return_pct)

series = cmma_ind(df, hyperparams={"lookback": 10})  # standalone
result = strategy.backtest(params={"lookback": 10})  # single run
```

Hyperparam-driven indicators are never disk-cached (an in-memory memo
covers repeated values during `optimize`), so expect them to recompute
across script runs.

## Interval Indicators

`indicator()` takes no interval argument. Bind the registered indicator
to one or more intervals with `Indicator.intervals(...)` when passing it
to `add_execution`; it is then computed on exactly those intervals, and
the base-timeframe variant exists only when `"base"` is included in the
binding (an unbound indicator is computed on the base timeframe).
`intervals=` on `add_execution` provides bars only and never computes
indicators. A bound interval is available through `ctx.interval(...)`
without declaring it in `intervals=`:

```python
sma_10 = pybroker.indicator(
    "sma_10", lambda data: sumv(data.close, 10) / 10
)

def buy_with_trend(ctx):
    weekly = ctx.interval("weekly")
    if len(weekly.close) < 10:
        return
    above = weekly.close[-1] > weekly.indicator("sma_10")[-1]
    if above and not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(0.25)

strategy.add_execution(
    buy_with_trend, ["AMD", "NVDA"], indicators=sma_10.intervals("weekly")
)
result = strategy.backtest(timeframe="1d")
```

- Interval forms: an every-n-bars `int`, a duration string (`"30s"`,
  `"5m"`, `"1h"`, `"1d"`), or a calendar string (`"daily"`, `"weekly"`,
  `"monthly"`, `"quarterly"`, `"yearly"`). Each must be strictly
  coarser than the base timeframe.
- `timeframe=` is required on `backtest`/`walkforward` whenever an
  execution declares `intervals=` or binds an indicator/model to an
  interval.
- `ctx.interval(...)` exposes only completed compressed bars, which is
  what prevents lookahead into a partially formed bar.
- Internally the interval copy is named `"sma_10@weekly"` — never type
  the `@` form yourself; reading it from the base context raises.
- For offline experimentation, compress a single-symbol frame with
  `pybroker.compress_bars(data, "weekly", base_timeframe="1d")`.

## Custom Data Columns

Register non-OHLCV columns before an indicator reads them:

```python
pybroker.register_columns("sentiment")

def sentiment_ma(data, period):
    if data.sentiment is None:  # column missing from the input data
        return np.full(len(data.close), np.nan)
    return sumv(data.sentiment, period) / period
```

Registered columns also appear as `ctx.sentiment` in execution
functions and on `ctx.interval(...)` (compressed as the last value in
each period).

## Reporting Results

Print `result.metrics_df` as the human-readable summary. For
structured output (agent parsing, saved report files, downstream
tools), `result.to_json()` returns a JSON-safe dict and
`result.to_json_str()` strict JSON text. The default payload carries
metrics, trades, orders, and bootstrap capped at `max_rows=100` rows
per table; `symbols=` filters to specific tickers; `include=` opts
into `portfolio`/`positions`/`metrics_df`/`signals`/`stops`. Dates
serialize as naive-UTC ISO strings, NaN as `null`, and legitimately
infinite metrics as `"Infinity"`/`"-Infinity"`.

## Session Hygiene and Debugging

Put these at the top of agent-run scripts:

```python
pybroker.enable_data_source_cache("my_indicators")  # skip refetching
pybroker.enable_indicator_cache("my_indicators")  # reuse computed values
pybroker.disable_progress_bar()  # progress bars flood AI token context
```

- `pybroker.enable_caches(name)` covers data, indicators, and trained
  models at once. The default cache directory is
  `<cwd>/.pybrokercache/<namespace>`; `clear_*_cache()` raises if that
  cache was never enabled.
- The disk cache does not apply to `IndicatorSet` calls or to
  hyperparam-driven indicators.
- Add `pybroker.disable_logging()` when running many backtests, such as
  parameter optimization.
- Parallel computation is opt-in via
  `backtest(..., parallel_indicators=True)` and is grouped per symbol;
  tune workers with `pybroker.set_parallel(n_jobs=...)`.
- Debug serially first: there is no error handling on the indicator
  compute path, and under `parallel_indicators=True` an exception
  arrives wrapped in a joblib worker traceback. Reproduce with the
  default serial path, fix, then re-enable parallelism.

Numba debug toggle: a Numba compilation or typing error in an `@njit`
indicator is easier to read as plain Python. Re-run once with JIT
disabled to get a normal traceback at the offending line, then
re-enable JIT for the real run:

```bash
NUMBA_DISABLE_JIT=1 python my_indicators.py
```

Common causes: mixed dtypes in one array, untyped `np.array([...])`
construction, or Python objects (lists of strings, dicts, pandas)
inside the `@njit` function.

## Lookahead Rules

An indicator value at bar `i` may depend only on inputs at index `i`
and earlier. Forbidden:

- Centered or forward-shifted windows (`center=True` smoothing,
  `shift(-n)`, savgol/filtfilt-style filters that read both sides).
- Normalizing by full-series statistics (whole-array min-max or
  z-score); use rolling statistics instead.
- Shifting future values backward when building lags — lag construction
  moves past values forward (`out[lag:] = values[:-lag]`).
- Negative indexing into full-length arrays inside kernels: a negative
  index silently wraps to the end of the series — the future.

Self-test novel indicator logic with the bump-last-bar check (the same
invariant PyBroker's own indicator test sweep enforces): change only
the final input bar and assert every earlier output is unchanged.

```python
before = cmma_20(df).to_numpy().copy()
bumped = df.copy()
bumped.loc[bumped.index[-1], "close"] *= 1.5
after = cmma_20(bumped).to_numpy()
assert np.array_equal(before[:-1], after[:-1], equal_nan=True)
```

Bump every input the indicator reads (`high`, `low`, `volume`, custom
columns) — not just `close` — so the test covers all of its inputs.

## Validation Checklist

- Syntax-check created Python files with `python -m py_compile <file>`.
- Generated scripts start with `pybroker.disable_progress_bar()` and
  `pybroker.enable_data_source_cache(...)`; add
  `pybroker.enable_indicator_cache(...)` when computation is expensive.
- No pandas in your own indicator logic or per-bar execution functions;
  pandas appears only at a third-party wrapper boundary.
- Output is one-dimensional, full length (`len(data.date)`), and
  NaN-padded over warmup bars — shortened outputs (tulipy) are padded.
- The bump-last-bar lookahead test passes for novel indicator logic.
- Indicator names contain no `@`, and no name is registered twice with
  different logic.
- Imports are correct for the collision-prone names: vect kernels from
  `pybroker`, factories from `pybroker.indicator`.
- `timeframe=` is passed to `backtest`/`walkforward` whenever an
  execution declares `intervals=` or binds an indicator/model to an
  interval.
- Third-party wrappers name their `pip install` and do not assume the
  library is importable; multi-output wrappers register one indicator
  per column.
- Failures were reproduced serially before enabling
  `parallel_indicators=True`, and `NUMBA_DISABLE_JIT` is absent from
  the final script.
- Sanity-check value ranges on a small fixture (an RSI stays within
  0-100; a rolling high is never below `close`).
