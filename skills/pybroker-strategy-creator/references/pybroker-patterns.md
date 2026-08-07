# PyBroker Patterns

## API Map

Use these imports and preamble for most strategy scripts:

```python
import pybroker as pyb
from pybroker import Strategy, StrategyConfig, YFinance

pyb.enable_data_source_cache("my_strategy")  # skip refetching on reruns
pyb.disable_progress_bar()  # keep backtest output out of agent context
```

Common optional imports:

```python
from pybroker import ExecContext, PositionMode, PriceType
from pybroker import highest, lowest, returns, indicator
from pybroker import FixedSlippageModel, hyperparam, set_parallel
from pybroker.indicator import atr  # built-in indicator factories
```

Create a strategy:

```python
config = StrategyConfig(initial_cash=100_000, exit_on_last_bar=True)
strategy = Strategy(YFinance(), start_date="1/1/2024", end_date="1/1/2026", config=config)
strategy.set_max_long_positions(5)
```

Data source choices:

- `YFinance()` for Yahoo Finance historical data.
- `Alpaca(...)` or `AlpacaCrypto(...)` when credentials are provided.
- `pandas.DataFrame` with required columns `date`, `symbol`, `open`, `high`, `low`, `close`; optional columns include `volume`, `vwap`, and registered custom columns.

Backtest choices:

- `strategy.backtest(...)` runs one backtest, optionally with `train_size` for models.
- `strategy.walkforward(windows=..., train_size=..., lookahead=...)` runs walkforward analysis.
- `strategy.optimize(score_fn, ...)` searches registered hyperparams (see Strategy Patterns).
- Use `warmup` at least as large as the largest indicator/model lookback before running entries.
- Indicator computation and model training parallelize only when `parallel_indicators=True` / `parallel_models=True` are passed; tune workers with `pyb.set_parallel(n_jobs=...)`.
- `TestResult` exposes `portfolio`, `positions`, `orders`, `trades`, `metrics`, `metrics_df`, optionally `signals`/`stops`, and `to_json()`/`to_json_str()` for compact agent-readable output: the default payload carries metrics, trades, orders, and bootstrap capped at `max_rows=100` rows per table, `symbols=` filters to specific tickers, and `include=` opts into `portfolio`/`positions`/`metrics_df`/`signals`/`stops`. Dates serialize as naive-UTC ISO strings, NaN as `null`, and legitimately infinite metrics as `"Infinity"`/`"-Infinity"`. `positions` is empty unless `StrategyConfig(record_position_bars=True)`; the per-bar `portfolio` equity curve is always populated.

## ExecContext Rules

Inside `exec_fn(ctx)`:

- Price and custom data are arrays through the latest completed bar: `ctx.close[-1]`, `ctx.high[-2]`, `ctx.volume`, `ctx.adj_close`, etc.
- Scalar accessors return the current bar directly: `ctx.close_price`, `ctx.open_price`, `ctx.high_price`, `ctx.low_price`, `ctx.volume_value`.
- `ctx.bars` is the completed bar count for `ctx.symbol`.
- `ctx.long_pos()` and `ctx.short_pos()` return current positions or `None`; `ctx.has_long_positions()` / `ctx.has_short_positions()` check the whole portfolio.
- `ctx.buy_shares` opens/adds long exposure.
- `ctx.sell_shares` sells long shares or opens/adds short exposure depending on position state.
- `ctx.cover_shares` covers short exposure and places the buy before sell orders.
- `ctx.sell_all_shares()` exits the current long position.
- `ctx.cover_all_shares()` exits the current short position.
- `ctx.calc_target_shares(0.25)` sizes to 25% of deployable capital (equity times `StrategyConfig.leverage`), capped by buying power.
- `ctx.set_target_shares(0.25, dir="long")` places whatever order moves the position to a 25% target allocation; use it for rebalancing.
- `ctx.indicator("name")[-1]` reads indicator output.
- `ctx.preds("model_name")[-1]` reads model predictions.
- `ctx.foreign("SPY", "close")` reads another symbol's completed bars.
- `ctx.hyperparam("name")` reads a hyperparam attached to this execution.
- `ctx.interval("weekly")` returns read-only compressed bars for a declared interval.
- `ctx.session` persists per-symbol state across bars.

Order validation pitfalls:

- Set at most one of `ctx.buy_shares` or `ctx.sell_shares` per symbol per bar.
- `buy_limit_price` requires `buy_shares`; `sell_limit_price` requires `sell_shares`. Unfilled limit orders retry for `ctx.buy_timeout_bars` / `ctx.sell_timeout_bars` bars (`None` = single attempt, `-1` = indefinite).
- `hold_bars` and stops require an entry order on the same bar.
- `hold_bars` must be greater than zero.
- `stop_loss` and `stop_loss_pct` are mutually exclusive. The same applies to profit and trailing stops.
- Buy and sell signals fill on future bars controlled by `StrategyConfig.buy_delay` and `sell_delay`; defaults are one bar.
- When rotation is enabled, order fields set in execution functions are ignored; only scores drive trading (fill prices and stops set there are kept).

## Indicator Patterns

Built-ins:

```python
high_20 = highest("high_20", "high", period=20)
low_20 = lowest("low_20", "low", period=20)
ret_5 = returns("ret_5", "close", period=5)
atr_14 = atr("atr_14", lookback=14)  # from pybroker.indicator
```

More built-in factories (`adx`, `macd`, `stochastic`, `close_minus_ma`, ...)
live in the `pybroker.indicator` module. Note the top-level `pybroker.atr` is
the vectorized kernel `atr(high, low, close, lookback)`, not this factory.

Custom indicator functions receive `BarData` and must return a one-dimensional array aligned to the input dates. `BarData` fields are NumPy arrays: compute with vectorized NumPy or the built-in helpers (`highv`, `lowv`, `sumv`, `returnv`, `cross`, `atr` from `pybroker`) — never pandas. No `pd.Series`/`pd.DataFrame` construction and no `.rolling`/`.ewm`/`.shift`/`.apply` inside an indicator function or a per-bar execution function:

```python
from pybroker import indicator, sumv

def sma(data, period: int):
    return sumv(data.close, period) / period

sma_50 = indicator("sma_50", sma, period=50)

# Bad: pandas inside an indicator (slow, unnecessary).
def sma_slow(data, period: int):
    return pd.Series(data.close).rolling(period).mean().to_numpy()
```

For logic that needs an explicit loop, JIT-compile a nested kernel with Numba `@njit` (see the `cmma` example in `wiki-05-writing-indicators.md`). If an `@njit` function fails to compile or raises a cryptic `TypingError`, re-run once with the `NUMBA_DISABLE_JIT=1` environment variable to get a readable Python traceback, fix the code, then remove the variable.

Indicator values must never look ahead: a value at bar `i` may depend only on inputs at index `i` and earlier. Never negative-index into a full-length array inside an indicator function (a negative index silently wraps to the end of the series — the future), never shift future values backward (`shift(-1)`-style), and never normalize by full-series statistics. Self-test novel indicator logic with the bump-last-bar check — change only the final input bar and assert every earlier output is unchanged:

```python
before = sma_50(df).to_numpy().copy()
bumped = df.copy()
bumped.loc[bumped.index[-1], "close"] *= 1.5
after = sma_50(bumped).to_numpy()
assert np.array_equal(before[:-1], after[:-1], equal_nan=True)
```

Bump every input the indicator reads (`high`, `low`, `volume`, custom columns), not just `close`.

Attach indicators to an execution:

```python
strategy.add_execution(exec_fn, ["AAPL", "MSFT"], indicators=[high_20, low_20])
```

## Strategy Patterns

Long breakout with fixed risk:

```python
def breakout(ctx: ExecContext):
    if ctx.bars < 21:
        return
    high_20 = ctx.indicator("high_20")
    pos = ctx.long_pos()
    if pos:
        if ctx.close[-1] < ctx.low[-2]:
            ctx.sell_all_shares()
        return
    if ctx.close[-1] > high_20[-2]:
        ctx.buy_shares = ctx.calc_target_shares(0.20)
        ctx.stop_loss_pct = 5
        ctx.stop_profit_pct = 12
```

Ranked entries (execution functions still place orders; scores decide
which signals win when a position cap binds). Higher score wins on both
sides: short orders go to the symbols with the highest `short_score`, so
negate a lowest-wins short signal to rank the most negative value first:

```python
strategy.set_max_long_positions(1)
strategy.set_max_short_positions(1)

def long_high_short_low(ctx: ExecContext):
    if ctx.bars < 6 or ctx.long_pos() or ctx.short_pos():
        return
    roc = (ctx.close[-1] - ctx.close[-6]) / ctx.close[-6]
    if roc > 0 and not ctx.has_long_positions():
        ctx.buy_shares = ctx.calc_target_shares(0.5)
        ctx.hold_bars = 2
        ctx.long_score = roc
    elif roc < 0 and not ctx.has_short_positions():
        ctx.sell_shares = ctx.calc_target_shares(0.5)
        ctx.hold_bars = 2
        ctx.short_score = -roc  # highest short_score is shorted first
```

The `if`/`elif` keeps one order side per symbol per bar, and the
`has_long_positions()` / `has_short_positions()` gates are
portfolio-wide, not per-symbol.

Ranking semantics:

- Scores rank descending across all executions, with the symbol name as
  a deterministic tiebreak. `long_score` ranks buy/cover signals;
  `short_score` ranks sell signals. Both default to `None` and reset
  every bar.
- A symbol that sets no score sorts as `0.0`; an unrankable (NaN) score
  sorts last. Orders past a cap are silently dropped (debug log only),
  so set scores whenever a cap is set.
- In ranked-cap mode nothing is ever liquidated by ranking alone; only
  `enable_rotation` liquidates by rank.

Rotation (trading is driven entirely by scores; entries are equal-weighted
across the position slots unless a custom `sizer` is passed). With both
caps set, each leg ranks independently; a symbol picked by both legs on
the same bar goes to the side where it ranks better, and ties go long:

```python
strategy.set_max_long_positions(3)
strategy.enable_rotation(worst_rank_held=5)

def rotate(ctx: ExecContext):
    if ctx.bars >= 21:
        ctx.long_score = ctx.indicator("roc_20")[-1]

strategy.add_execution(rotate, symbols, indicators=roc_20)
result = strategy.backtest(warmup=20)
```

Model-backed execution:

```python
model_source = pyb.model("my_model", train_fn, indicators=[ret_5])

def trade_prediction(ctx: ExecContext):
    if ctx.bars < 5:
        return
    pred = ctx.preds("my_model")[-1]
    pos = ctx.long_pos()
    if pos and pred < 0:
        ctx.sell_all_shares()
    elif not pos and pred > 0:
        ctx.buy_shares = ctx.calc_target_shares(0.25)

strategy.add_execution(trade_prediction, ["AAPL"], models=model_source)
result = strategy.walkforward(windows=5, train_size=0.5, lookahead=1)
```

Model variants:

- `pyb.model(..., lags=3, lag_cols=["close", ret_5])` builds a lagged feature matrix; the training `fn` must accept `lag_train=` and `lag_test=` keyword arguments, and `predict_fn` receives the matrix in place of a DataFrame.
- `pyb.model(..., pooled=True)` trains one model per execution across all of its symbols; `fn` becomes `fn(symbols, train_data, test_data, ...)`.
- `pyb.model(..., per_bar=True, predict_fn=...)` calls `predict_fn` once per bar and expects a scalar; use for stateful time-series models such as GARCH. Not supported with `pooled=True`.

Parameter optimization (register hyperparams, then search; add
`pyb.disable_logging()` first since this runs many backtests):

```python
period = pyb.hyperparam("period", default=30, low=10, high=50, step=10)
stop_pct = pyb.hyperparam("stop_pct", default=6.0, low=2.0, high=10.0, step=2.0)
sma_ind = pyb.indicator("sma", sma, period=period)

def sma_cross(ctx: ExecContext):
    sma_vals = ctx.indicator("sma")
    if np.isnan(sma_vals[-1]):  # warmup guard: period is a hyperparam
        return
    if not ctx.long_pos() and ctx.close[-1] > sma_vals[-1]:
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        ctx.stop_loss_pct = ctx.hyperparam("stop_pct")

strategy.add_execution(sma_cross, ["AAPL"], indicators=sma_ind, hyperparams=[stop_pct])
opt = strategy.optimize(lambda r: r.metrics.total_return_pct)  # grid search
opt = strategy.optimize(lambda r: r.metrics.sharpe, sampler="tpe", n_trials=15, seed=2)
print(opt.best_params, opt.best_score)
```

Optimization does not support trainable models — only `pretrained=True` models
or indicator-based rules.

Multiple time intervals (`timeframe=` is required whenever an execution
declares `intervals=` or binds an indicator/model to an interval;
intervals are readable only by the declaring execution). `intervals=`
provides bars only; bind an indicator/model to an interval with
`.intervals(...)` to compute/train it there — the bound interval is
available through `ctx.interval(...)` without declaring it again.
Binding is exhaustive: include `"base"` (e.g.
`sma_10.intervals("base", "weekly")`) to keep the base-timeframe
variant, while unbound sources default to base:

```python
sma_10 = pyb.indicator("sma_10", lambda data: pyb.sumv(data.close, 10) / 10)

def buy_with_trend(ctx: ExecContext):
    weekly = ctx.interval("weekly")
    if len(weekly.close) < 10:
        return
    if weekly.close[-1] > weekly.indicator("sma_10")[-1] and not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(0.25)

strategy.add_execution(buy_with_trend, ["AMD", "NVDA"], indicators=sma_10.intervals("weekly"))
result = strategy.backtest(timeframe="1d")
```

Slippage and margin:

```python
strategy.set_slippage_model(FixedSlippageModel(bps=5))  # or VolatilitySlippageModel / VolumeSlippageModel
config = StrategyConfig(leverage=2.0, interest_rate=6.25, bars_per_year=252)
```

Dynamic symbol selection (pass a callable instead of a symbol list; requires a
DataFrame data source and a training window, so use `walkforward` or
`train_size > 0`):

```python
def top_liquidity(train_df) -> list[str]:
    volume = train_df.groupby("symbol")["volume"].mean()
    return list(volume.nlargest(5).index)

strategy.add_execution(exec_fn, top_liquidity, indicators=[high_20])
result = strategy.walkforward(windows=3, train_size=0.5)
```

## v1 -> v2 Gotchas

- `StrategyConfig.max_long_positions` / `max_short_positions` are deprecated; call `strategy.set_max_long_positions()` / `set_max_short_positions()`.
- `set_pos_size_handler`, `PosSizeContext`, and `ExecSignal` were removed; use `strategy.enable_rotation(worst_rank_held=..., sizer=...)` with `RotationContext`.
- `RandomSlippageModel` was removed; use `FixedSlippageModel`, `VolatilitySlippageModel`, or `VolumeSlippageModel`, or subclass `SlippageModel`.
- `StrategyConfig.bootstrap_sample_size` was removed (`bootstrap_samples` remains).
- `disable_parallel=` was removed from `backtest`/`walkforward`; parallel indicator/model work is now opt-in via `parallel_indicators=` / `parallel_models=`.
- `result.positions` is opt-in via `record_position_bars=True`. `result.portfolio` is always populated, but full `PortfolioBar` snapshots on `Portfolio.bars` are opt-in via `record_portfolio_bars=True`.

## Validation Checklist

- Syntax-check created Python files with `python -m py_compile <file>`.
- Generated scripts start with `pybroker.disable_progress_bar()` and `pybroker.enable_data_source_cache(...)`.
- No pandas operations inside indicator functions or per-bar execution functions — NumPy/Numba only.
- The bump-last-bar lookahead test passes for novel indicator logic, and no indicator function negative-indexes into a full-length array.
- `timeframe=` is passed to `backtest`/`walkforward` whenever an execution declares `intervals=` or binds an indicator/model to an interval.
- `StrategyConfig(record_position_bars=True)` is set when the user needs `result.positions`.
- Grep deliverables for removed/deprecated names: `bootstrap_sample_size`, `disable_parallel`, `set_pos_size_handler`, `StrategyConfig(max_long_positions=...)`.
- Short signals set `ctx.short_score` with higher-wins ordering: negate a lowest-wins signal (`-roc`), never invert it (`1.0 / roc`).
- If using DataFrame data, include a tiny local fixture and run the backtest without network access.
- If using `YFinance`, name the required `pip install yfinance`, expect network availability to be an execution dependency, and mention when not run.
- Inspect `result.metrics_df`, `result.orders`, and `result.trades` for empty or impossible behavior.
- Keep generated examples reproducible by setting explicit dates, symbols, config, and random seeds where applicable.
