# PyBroker Multi-Interval Patterns

## Multi-Interval API Map

Use these imports for most multi-interval work:

```python
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance
from pybroker import compress_bars, indicator, model, sumv
from pybroker.common import bars_to_df  # inspect compressed BarData
```

There are two ways to declare an interval, and they mean different
things:

```python
strategy.add_execution(
    exec_fn,
    ["AMD", "NVDA"],
    # Path 1: BIND a source — computes/trains on exactly these
    # intervals (replacing the base variant unless "base" is listed).
    indicators=sma_10.intervals("weekly"),
    models=model_weekly.intervals("weekly"),
    # Path 2: bars only — compressed OHLCV for this execution, nothing
    # is computed on it. Bound intervals are auto-unioned, so "weekly"
    # is not repeated here.
    intervals="monthly",
)
# timeframe= (the base bar spacing) is required once any interval
# exists, on backtest(), walkforward(), and optimize().
result = strategy.backtest(timeframe="1d")
```

Execution functions read intervals through `ctx.interval(...)`, which
returns a read-only `IntervalContext`:

- `bars` — count of completed compressed bars (`0` during warmup).
- `dates`, `open`, `high`, `low`, `close`, `volume` — NumPy arrays
  truncated to the last completed compressed bar. There is no `vwap`
  property.
- `indicator(name)` — a bound indicator's values on this interval.
- `preds(name)` / `model(name)` / `input(name)` — a bound model's
  predictions, trained instance, and input rows on this interval.
- Registered custom columns appear as attributes (`weekly.sentiment`),
  compressed as the last value in each period.

Standalone compression (no strategy needed):
`compress_bars(df, "weekly", base_timeframe="1d")` — see the dedicated
section below.

Exact signatures: `IntervalContext` in `pybroker_context.pyi`;
`TimeframeInterval`/`CalendarInterval` in `pybroker_types.pyi`;
`compress_bars` and the `Indicator.intervals`/`ModelSource.intervals`
bindings in `pybroker_model.pyi`; the `timeframe=` parameters in
`pybroker_strategy.pyi`.

## Interval Grammar

An interval takes one of exactly three forms:

| Form | Example | Meaning |
| --- | --- | --- |
| `int` greater than 1 | `5` | Every n base bars become one bar. |
| Duration string | `"30s"`, `"5m"`, `"1h"`, `"1d"` | Digits plus ONE unit letter: `s`, `m`, `h`, or `d`. |
| Calendar string | `"daily"`, `"weekly"`, `"monthly"`, `"quarterly"`, `"yearly"` | Aligned to calendar boundaries. |

Calendar alignment: weeks start Monday; months on the 1st; quarters in
January, April, July, and October; years on January 1.

Every interval must be strictly coarser than the base timeframe. For a
daily (`"1d"`) base:

- Valid: `"weekly"`, `"monthly"`, `"quarterly"`, `"yearly"`, `5`, `"5d"`.
- Invalid: `"daily"` (equal, not coarser), `"1h"` (finer), `"5min"`
  (unit must be a single letter), `"1h 30m"` (no compound spans),
  `"2w"` (week durations rejected — use `"weekly"` for calendar weeks
  or `"14d"` for fixed 14-day windows), `1` (n must exceed 1).

Duplicate intervals (`intervals=["weekly", "weekly"]`) and empty lists
(`intervals=[]`) raise `ValueError`.

### `timeframe=` grammar is wider — do not confuse the two

`timeframe=` on `backtest`/`walkforward`/`optimize` describes the base
bar spacing of the data and accepts compound spans and long unit names
(`"1h 30m"`, `"90min"`, and the `w` unit are all legal there). Interval
grammar accepts none of those. Never reuse a `timeframe` string as an
interval or assume the reverse: `timeframe="1h 30m"` is fine while
`intervals="1h 30m"` raises.

## Compression Semantics

When base bars are grouped into one compressed bar:

| Field | Aggregation |
| --- | --- |
| `open` | First base bar's open. |
| `high` / `low` | Highest high / lowest low. |
| `close` | Last base bar's close. |
| `volume` | Sum. |
| `vwap` | Volume-weighted average. |
| Custom columns | Last value in the period. |

Each compressed bar is timestamped with the date of the LAST base bar
it contains, so a `"weekly"` bar over Mon-Fri carries Friday's date.
During a backtest the trailing, still-forming bin is never visible to
strategy code; it completes only when a base bar from the next bin
arrives.

`timeframe=` is validated against the observed data: gaps (holidays,
halts, `days=`/`between_time=` filters) are fine as long as every gap
is a whole multiple of the base spacing; inconsistent spacing raises
`ValueError`. Daily-and-coarser bars get a one-hour DST tolerance.

## Bars Only vs Bound Computation

The central distinction:

| | `add_execution(intervals=...)` | `.intervals(...)` binding |
| --- | --- | --- |
| Provides | Compressed OHLCV bars only | Indicator values / trained models on those intervals |
| Declared on | The execution | The `Indicator` / `ModelSource`, passed to `indicators=`/`models=` |
| Base variant | n/a | Replaced unless `"base"` is listed |
| Reachable via `ctx.interval` | Yes | Yes (auto-unioned; no need to repeat in `intervals=`) |

`intervals=` never computes an indicator and never trains a model.
Reading `weekly.indicator("sma_10")` with only `intervals="weekly"`
declared raises — the indicator must be bound.

Intervals are scoped to their execution. A sibling execution's
`ExecContext` cannot read them, and neither can a `set_before_exec` /
`set_after_exec` callback holding another execution's context; each
execution declares its own.

Combined example (bind weekly computation, monthly bars only):

```python
sma_10 = pybroker.indicator(
    "sma_10", lambda data: sumv(data.close, 10) / 10
)

def exec_fn(ctx):
    weekly = ctx.interval("weekly")    # from the indicator binding
    monthly = ctx.interval("monthly")  # from intervals=
    if len(weekly.close) < 10 or len(monthly.close) < 4:
        return
    regime_up = monthly.close[-1] > monthly.close[-4]
    trend_up = weekly.close[-1] > weekly.indicator("sma_10")[-1]
    if regime_up and trend_up and not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(0.25)

strategy.add_execution(
    exec_fn,
    ["AMD", "NVDA"],
    indicators=sma_10.intervals("weekly"),
    intervals="monthly",
)
result = strategy.backtest(timeframe="1d")
```

## Exhaustive Binding and "base"

Binding replaces base-timeframe computation. After
`sma_10.intervals("weekly")`, the indicator exists ONLY on weekly bars:

```python
# Bad: raises — no base-timeframe variant exists after binding.
def exec_fn(ctx):
    daily_sma = ctx.indicator("sma_10")

# Fix 1: read the interval it is bound to.
def exec_fn(ctx):
    wk_sma = ctx.interval("weekly").indicator("sma_10")

# Fix 2: include "base" in the binding to keep the daily variant too.
strategy.add_execution(
    exec_fn, syms, indicators=sma_10.intervals("base", "weekly")
)
```

- An unbound indicator or model defaults to the base timeframe, exactly
  as if intervals did not exist.
- `"base"` is valid only inside `.intervals(...)`. It raises in
  `add_execution(intervals=...)` and in `ctx.interval("base")` — base
  data is read from `ctx` directly (`ctx.close`, `ctx.indicator(...)`).
- The same rule applies to models: `model_source.intervals("weekly")`
  trains no base-timeframe model, and `ctx.preds(name)` then raises
  until `"base"` is added to the binding.

## Reading Intervals in Executions

`ctx.interval("weekly")` returns a read-only `IntervalContext`. Rules:

- Arrays hold completed compressed bars only and are EMPTY during
  warmup. Guard every read:

  ```python
  weekly = ctx.interval("weekly")
  if len(weekly.close) < 10:
      return
  latest_close = weekly.close[-1]   # last COMPLETED weekly bar
  ten_weeks_ago = weekly.close[-10]
  ```

  Negative indexing is safe here only because the arrays are already
  truncated to completed bars; the guard keeps `[-10]` from wrapping
  on a short array.
- A bound indicator's warmup is measured in compressed bars: a 10-bar
  SMA bound to `"weekly"` is NaN until 10 completed weekly bars exist,
  regardless of how many daily bars have passed. `warmup=` on
  `backtest`/`walkforward` counts base bars and is no substitute for
  interval length guards.
- Read bound values with base names only: `weekly.indicator("sma_10")`
  and `weekly.preds("weekly_slr")`. PyBroker suffixes names internally
  (`"sma_10@weekly"`); never type the `@` form, and never register an
  indicator or model whose name contains `@`.
- `IntervalContext` is read-only. Assigning any order or stop attribute
  (`buy_shares`, `stop_loss_pct`, ...) raises with "IntervalContext is
  read-only; set ... on the base ExecContext instead." Decide with
  interval data, act on `ctx`.
- There is no `weekly.vwap` property. Registered custom columns are
  available as attributes (`weekly.sentiment`), compressed as the last
  value per period; guard for `None` when input data may lack the
  column.
- Give each timeframe one job to keep rules legible: e.g. monthly =
  regime filter, weekly = trend, base = entry timing. Compare only
  completed values (`monthly.close[-1] > monthly.close[-4]`).

## Binding Models to Intervals

Bind a trainable model exactly like an indicator:

```python
from sklearn.linear_model import LinearRegression  # pip install scikit-learn

def train_weekly(symbol, train_data, test_data):
    # train_data holds COMPRESSED weekly bars. Copy so the input frame
    # is never widened or mutated.
    df = train_data.copy()
    # Target is the NEXT weekly return, matching lookahead=1 measured
    # in weekly bars.
    df["target"] = df["close"].pct_change().shift(-1)
    df = df.dropna()
    lr = LinearRegression()
    lr.fit(df[["close"]], df["target"])
    return lr, ["close"]

model_weekly = pybroker.model("weekly_slr", train_weekly)

def exec_fn(ctx):
    preds = ctx.interval("weekly").preds("weekly_slr")
    if len(preds) == 0:
        return
    if not ctx.long_pos() and preds[-1] > 0:
        ctx.buy_shares = ctx.calc_target_shares(0.25)
    elif ctx.long_pos() and preds[-1] < 0:
        ctx.sell_all_shares()

strategy.add_execution(
    exec_fn, ["AMD", "NVDA"], models=model_weekly.intervals("weekly")
)
result = strategy.walkforward(windows=3, train_size=0.5, timeframe="1d")
```

- Only trainable models bind. A pretrained model (`ModelLoader`) raises
  "is not trained per interval and cannot be bound to intervals"; read
  pretrained models on the base timeframe with `ctx.model(...)` /
  `ctx.preds(...)`.
- Indicators registered on the model
  (`pybroker.model(name, train_fn, indicators=[...])`) are computed on
  the bound interval's compressed bars and appear in `train_data` /
  `test_data` under their BASE column names (`"sma_10"`, not
  `"sma_10@weekly"`), ready to use as features.
- `lookahead` is measured in the bound interval's compressed bars: a
  weekly model with `lookahead=1` holds out one weekly bar between
  train and test, not one daily bar. A too-large `lookahead` on a
  coarse interval can empty the train set; PyBroker warns with "has no
  training bars left ... Lower lookahead, raise train_size, or use a
  finer interval."
- Predictions are indexed by completed compressed bar. `preds` is empty
  during warmup and before the first test window — always guard with
  `if len(preds) == 0: return`.
- `TestResult.signals` contain base-timeframe values only; an
  interval-bound indicator or model appears in the signals only when
  `"base"` is included in its binding.
- Walkforward window boundaries are enforced in compressed-bar units,
  so user `train_fn` code cannot see compressed bars from a future
  window.

## Standalone Compression with compress_bars

`compress_bars` converts single-symbol OHLCV data (a DataFrame or
`BarData`) to a coarser interval and returns a new `BarData`. Use it to
build intuition, produce fixtures, and validate compression offline —
no strategy or network required:

```python
from pybroker import compress_bars
from pybroker.common import bars_to_df

amd_df = df[df["symbol"] == "AMD"]
weekly = compress_bars(amd_df, "weekly", base_timeframe="1d")
bars_to_df(weekly).head()

five_day = compress_bars(amd_df, 5, base_timeframe="1d")
```

- `base_timeframe` is required and declares the input bar spacing; the
  input dates are validated against it.
- Single symbol only: a multi-symbol frame raises "compress_bars
  expects data for a single symbol", pointing to
  `compress_symbol_from_frame` / `compress_intervals_from_frame`
  (in `pybroker.interval`) for multi-symbol work.
- The returned arrays include the trailing bin even though a backtest
  would still treat it as forming — which is exactly what the
  bump-last-bar self-test below relies on.

## Common Errors

Match the message before changing code — each one names the fix.

- **"Interval 'weekly' was not declared for this execution."** —
  `ctx.interval("weekly")` on an execution that neither declared
  `intervals="weekly"` nor bound anything to it. Note intervals are
  per-execution: a sibling execution's declaration (or a
  `set_before_exec`/`set_after_exec` callback) does not cover this one.
  Fix: add `intervals=` or a `.intervals()` binding on THIS
  `add_execution` call.
- **"Compression intervals ... need the base bar spacing of the data:
  pass timeframe= to backtest() or walkforward()"** — an interval was
  declared or bound but the run call omitted `timeframe=`. Fix:
  `backtest(timeframe="1d")` (or the actual base spacing). Bound
  intervals trigger this too, not just `intervals=`.
- **"Invalid interval '...'. use an int > 1 (e.g. 5) for n-bar
  compression, '5m'/'1h' for duration intervals (digits + unit
  letter), or 'weekly' for calendar weeks."** — malformed interval
  (`"5min"`, `"1h 30m"`, `"weekly "` variants, unit typos). Fix: use
  one of the three exact forms from the grammar section.
- **"Week durations are not supported: use 'weekly' for calendar weeks
  or '14d' for fixed 14-day windows."** — a `"2w"`-style duration.
  Fix: as the message says.
- **"interval compression requires n > 1."** — every-n-bars with
  `n <= 1`. Fix: use an int of at least 2 (a 1-bar "compression" is
  just the base data).
- **"Cannot compress daily bars to interval '1h'. Compression only
  supports strictly coarser intervals"** — the interval is finer than
  or equal to the base timeframe. Fix: pick a strictly coarser
  interval, or check that `timeframe=` states the real base spacing.
- **"intervals cannot be empty." / "Duplicate interval: 'weekly'."** —
  an empty list, or the same interval listed twice (possibly via
  normalization: `"60m"` and `"1h"` are distinct strings but distinct
  intervals — the duplicate check runs on normalized values). Fix:
  de-duplicate the declaration.
- **"Invalid interval 'base'"** — `"base"` passed to
  `add_execution(intervals=...)` or `ctx.interval("base")`. Fix:
  `"base"` belongs only inside `.intervals(...)` bindings; base data is
  read from `ctx` directly.
- **"Invalid indicator name 'sma@weekly': '@' is reserved for interval
  bindings"** — a registered name contains `@`. Fix: rename; PyBroker
  generates the suffixed names itself.
- **"Indicator 'sma_10' is bound to interval 'weekly' and cannot be
  read from the base context. Use
  ctx.interval('weekly').indicator('sma_10') instead."** — reading an
  interval-bound series through `ctx.indicator`. Interval series are
  indexed by compressed bar, so a base-bar index would expose future
  data. Fix: as the message says.
- **"Indicator 'sma_10' not found for AMD. ... If it is bound with
  Indicator.intervals(), include 'base' in the binding to compute it on
  the base timeframe."** — the base variant was replaced by an
  exhaustive binding. Fix: bind `("base", "weekly")` or read the bound
  interval instead.
- **"Model '...' not found for ... Models are trained on an interval
  only when bound to it with ModelSource.intervals()."** —
  `weekly.preds(name)` on an interval the model is not bound to
  (declaring `intervals="weekly"` is not enough). Fix: bind the model.
- **"Pretrained model '...' is not trained per interval and cannot be
  bound to intervals."** — `.intervals()` on a `ModelLoader`. Fix: use
  a trainable model for per-interval training, or read the pretrained
  model on the base timeframe.
- **"Bar spacing for 'AMD' is inconsistent with base timeframe (86400s
  expected, 60s observed between consecutive bars)."** — the data does
  not match `timeframe=` (e.g. minute data with `timeframe="1d"`, or
  mixed feeds). Fix: pass the true base spacing; gaps are fine only as
  whole multiples of it.
- **Warning: "Model ... has no training bars left ... after holding out
  lookahead=N compressed bars"** — the lookahead hold-out, measured in
  compressed bars, consumed the interval train set. Fix: lower
  `lookahead`, raise `train_size`, or bind a finer interval.
- **`IndexError` on `weekly.close[-1]` (or NaN-driven logic that never
  trades)** — a missing warmup guard; interval arrays are empty until
  the first compressed bar completes. Fix:
  `if len(weekly.close) < N: return` before indexing.

## Fill Prices and End-of-Data Exits

**Fills always price off base-timeframe bars.** Declaring
`intervals="weekly"` or binding an indicator to an interval changes
what the execution function reads, never how an order fills: the fill
price is resolved against the symbol's base bar. A strategy that acts
on weekly signals still fills at daily prices.

Orders fill at `PriceType.MIDDLE` — the midpoint of the low and high
of the **execution** base bar, one bar after the signal under the
default `buy_delay`/`sell_delay` of `1`, so `PriceType.CLOSE` means
the next base bar's close, not the close of the compressed bar.
`PriceType` offers `OPEN`, `HIGH`, `LOW`, `CLOSE`, `MIDDLE`
(`low + (high - low) / 2`, the default), and `AVERAGE`
(`(open + low + high + close) / 4`). `ctx.buy_fill_price` /
`ctx.sell_fill_price` also accept a number or a `(symbol, bar_data)`
callable — whose `bar_data` is likewise base-timeframe — and read back
as `None` until set. A limit price only gates the fill: the order
still fills at the fill price, never at the limit.

`StrategyConfig.exit_on_last_bar` defaults to `False`, which leaves
the final position open and out of `trade_count`, `win_rate`,
`total_pnl` and every other trade-level metric, with its P&L in
`unrealized_pnl`. Set `exit_on_last_bar=True` whenever trade
statistics are reported; the liquidation lands on the symbol's final
**base** bar, which is usually mid-way through the last compressed
bar. In `walkforward` it fires only on each symbol's true final bar,
never at window boundaries.

`calc_bootstrap` is a parameter of `backtest`/`walkforward`, **not** a
`StrategyConfig` field, and defaults to `False`. Pass
`calc_bootstrap=True` to populate `result.bootstrap` with
`conf_intervals` (BCa intervals for profit factor and Sharpe) and
`drawdown_conf` (percentile bounds on max drawdown). Set
`StrategyConfig.bars_per_year` to the **base** timeframe's bar count
or the Sharpe intervals are per-bar rather than annualized.

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
pybroker.enable_data_source_cache("my_strategy")  # skip refetching
pybroker.disable_progress_bar()  # progress bars flood AI token context
```

- `pybroker.enable_caches(name)` covers data, indicators, and trained
  models at once — prefer it when interval-bound models are trained.
- Add `pybroker.disable_logging()` when running many backtests, such as
  parameter optimization.
- Interval-bound indicators run the same indicator functions on
  compressed arrays, so Numba behavior is unchanged: on an `@njit`
  compile or typing error, re-run once with `NUMBA_DISABLE_JIT=1` to
  get a readable traceback, fix, then remove the variable. Debug
  serially before `parallel_indicators=True` (joblib wraps worker
  tracebacks).
- Reproduce interval errors with `backtest` before `walkforward`; the
  walkforward window slicing adds moving parts (per-window compressed
  alignment, lookahead hold-outs) that obscure a plain declaration
  mistake.

## Validation Checklist

- Syntax-check created Python files with `python -m py_compile <file>`.
- Generated scripts start with `pybroker.disable_progress_bar()` and
  `pybroker.enable_data_source_cache(...)` (or `pybroker.enable_caches`
  when models are trained).
- `timeframe=` is passed to `backtest`/`walkforward`/`optimize`
  whenever any execution declares `intervals=` or binds an indicator or
  model to an interval — and the value matches the data's real spacing.
- Every interval is strictly coarser than the base timeframe and uses
  one of the three exact grammar forms.
- Every `ctx.interval(...)` read sits behind a length guard; no
  negative index can wrap on a short compressed array.
- Bound values are read with base names (`weekly.indicator("sma_10")`);
  no registered name contains `@`.
- Orders and stops are set on the base `ctx` only.
- No pandas in indicator or per-bar execution logic; pandas appears
  only in `train_fn`/`input_data_fn` (which receive compressed-bar
  DataFrames for bound models) and third-party TA wrapper boundaries.
- `train_fn` works on a `.copy()` and never widens or mutates the input
  frame; interval-bound targets shift in compressed-bar units.
- The bump-last-bar self-test passes: change only the FINAL base bar
  and assert every earlier compressed value is unchanged (the trailing
  bin absorbs the bump):

  ```python
  import numpy as np
  from pybroker import compress_bars

  before = compress_bars(sym_df, "weekly", base_timeframe="1d")
  bumped = sym_df.copy()
  bumped.loc[bumped.index[-1], "close"] *= 1.5
  after = compress_bars(bumped, "weekly", base_timeframe="1d")
  assert np.array_equal(
      before.close[:-1], after.close[:-1], equal_nan=True
  )
  ```

  Bump every input the strategy reads (`high`, `low`, `volume`, custom
  columns), not just `close`. For a full backtest, the equivalent
  check is that completed `ctx.interval` values on earlier bars are
  identical before and after the bump.
- Compression sanity checks on a small fixture: each weekly `volume`
  equals the sum of its daily volumes; each compressed date is the date
  of the last base bar in its bin; the compressed `open` is the first
  daily open of the bin.
- Required installs are named and never assumed: `pip install yfinance`
  for `YFinance` data, `pip install scikit-learn` for the model
  examples. Prefer a small local DataFrame fixture when the network is
  unavailable.
