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
- `TestResult` exposes `portfolio`, `positions`, `orders`, `trades`, `metrics`, `metrics_df`, optionally `signals`/`stops`, and `to_json()`/`to_json_str()` for compact agent-readable output. `positions` is empty unless `StrategyConfig(record_position_bars=True)`; the per-bar `portfolio` equity curve is always populated.

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
- `ctx.score` is deprecated; set `ctx.long_score` / `ctx.short_score` instead.
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

Custom indicator functions receive `BarData` and must return a one-dimensional array aligned to the input dates. `BarData` fields are NumPy arrays: compute with vectorized NumPy or the built-in helpers (`highv`, `lowv`, `sumv`, `returnv`, `cross`, `atr` from `pybroker`) — never pandas:

```python
from pybroker import indicator, sumv

def sma(data, period: int):
    return sumv(data.close, period) / period

sma_50 = indicator("sma_50", sma, period=50)
```

For logic that needs an explicit loop, JIT-compile a nested kernel with Numba `@njit` (see the `cmma` example in `wiki-05-writing-indicators.md`). If an `@njit` function fails to compile or raises a cryptic `TypingError`, re-run once with the `NUMBA_DISABLE_JIT=1` environment variable to get a readable Python traceback, fix the code, then remove the variable.

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

Ranked entries (execution functions still place orders; scores break ties for
the position caps):

```python
strategy.set_max_long_positions(3)

def rank_by_momentum(ctx: ExecContext):
    if ctx.bars < 63:
        return
    momentum = ctx.close[-1] / ctx.close[-63] - 1
    ctx.long_score = momentum
    if not ctx.long_pos() and momentum > 0:
        ctx.buy_shares = ctx.calc_target_shares(1 / 3)
        ctx.hold_bars = 21
```

Rotation (trading is driven entirely by scores; entries are equal-weighted
across the position slots unless a custom `sizer` is passed):

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
declares `intervals=`; intervals are readable only by the declaring
execution):

```python
sma_10 = pyb.indicator("sma_10", lambda data: pyb.sumv(data.close, 10) / 10)

def buy_with_trend(ctx: ExecContext):
    weekly = ctx.interval("weekly")
    if len(weekly.close) < 10:
        return
    if weekly.close[-1] > weekly.indicator("sma_10")[-1] and not ctx.long_pos():
        ctx.buy_shares = ctx.calc_target_shares(0.25)

strategy.add_execution(buy_with_trend, ["AMD", "NVDA"], indicators=sma_10, intervals="weekly")
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
- `ctx.score` is deprecated; set `ctx.long_score` / `ctx.short_score`.
- `set_pos_size_handler`, `PosSizeContext`, and `ExecSignal` were removed; use `strategy.enable_rotation(worst_rank_held=..., sizer=...)` with `RotationContext`.
- `RandomSlippageModel` was removed; use `FixedSlippageModel`, `VolatilitySlippageModel`, or `VolumeSlippageModel`, or subclass `SlippageModel`.
- `StrategyConfig.bootstrap_sample_size` was removed (`bootstrap_samples` remains).
- `disable_parallel=` was removed from `backtest`/`walkforward`; parallel indicator/model work is now opt-in via `parallel_indicators=` / `parallel_models=`.
- `result.positions` is opt-in via `record_position_bars=True`. `result.portfolio` is always populated, but full `PortfolioBar` snapshots on `Portfolio.bars` are opt-in via `record_portfolio_bars=True`.

## Validation Checklist

- Syntax-check created Python files with `python -m py_compile <file>`.
- Generated scripts start with `pybroker.disable_progress_bar()` and `pybroker.enable_data_source_cache(...)`.
- No pandas operations inside indicator functions or per-bar execution functions — NumPy/Numba only.
- `timeframe=` is passed to `backtest`/`walkforward` whenever an execution declares `intervals=`.
- `StrategyConfig(record_position_bars=True)` is set when the user needs `result.positions`.
- Grep deliverables for removed/deprecated names: `ctx.score`, `bootstrap_sample_size`, `disable_parallel`, `set_pos_size_handler`, `StrategyConfig(max_long_positions=...)`.
- If using DataFrame data, include a tiny local fixture and run the backtest without network access.
- If using `YFinance`, expect network/package availability to be an execution dependency and mention when not run.
- Inspect `result.metrics_df`, `result.orders`, and `result.trades` for empty or impossible behavior.
- Keep generated examples reproducible by setting explicit dates, symbols, config, and random seeds where applicable.
