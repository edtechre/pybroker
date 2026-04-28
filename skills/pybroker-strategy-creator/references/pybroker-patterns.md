# PyBroker Patterns

## API Map

Use these imports for most strategy scripts:

```python
import pybroker as pyb
from pybroker import Strategy, StrategyConfig, YFinance
```

Common optional imports:

```python
from pybroker import ExecContext, PositionMode, PriceType
from pybroker import highest, lowest, returns, indicator
```

Create a strategy:

```python
config = StrategyConfig(
    initial_cash=100_000,
    max_long_positions=5,
    exit_on_last_bar=True,
)
strategy = Strategy(YFinance(), start_date="1/1/2020", end_date="1/1/2024", config=config)
```

Data source choices:

- `YFinance()` for Yahoo Finance historical data.
- `Alpaca(...)` or `AlpacaCrypto(...)` when credentials are provided.
- `pandas.DataFrame` with required columns `date`, `symbol`, `open`, `high`, `low`, `close`; optional columns include `volume`, `vwap`, and registered custom columns.

Backtest choices:

- `strategy.backtest(...)` runs one backtest, optionally with `train_size` for models.
- `strategy.walkforward(windows=..., train_size=..., lookahead=...)` runs walkforward analysis.
- Use `warmup` at least as large as the largest indicator/model lookback before running entries.
- `TestResult` exposes `portfolio`, `positions`, `orders`, `trades`, `metrics`, `metrics_df`, and optionally `signals`/`stops`.

## ExecContext Rules

Inside `exec_fn(ctx)`:

- Price and custom data are arrays through the latest completed bar: `ctx.close[-1]`, `ctx.high[-2]`, `ctx.volume`, `ctx.adj_close`, etc.
- `ctx.bars` is the completed bar count for `ctx.symbol`.
- `ctx.long_pos()` and `ctx.short_pos()` return current positions or `None`.
- `ctx.buy_shares` opens/adds long exposure.
- `ctx.sell_shares` sells long shares or opens/adds short exposure depending on position state.
- `ctx.cover_shares` covers short exposure and places the buy before sell orders.
- `ctx.sell_all_shares()` exits the current long position.
- `ctx.cover_all_shares()` exits the current short position.
- `ctx.calc_target_shares(0.25)` sizes to 25% of portfolio equity at the current close.
- `ctx.indicator("name")[-1]` reads indicator output.
- `ctx.preds("model_name")[-1]` reads model predictions.
- `ctx.foreign("SPY", "close")` reads another symbol's completed bars.
- `ctx.session` persists per-symbol state across bars.

Order validation pitfalls:

- Set at most one of `ctx.buy_shares` or `ctx.sell_shares` per symbol per bar.
- `buy_limit_price` requires `buy_shares`; `sell_limit_price` requires `sell_shares`.
- `hold_bars` and stops require an entry order on the same bar.
- `hold_bars` must be greater than zero.
- `stop_loss` and `stop_loss_pct` are mutually exclusive. The same applies to profit and trailing stops.
- Buy and sell signals fill on future bars controlled by `StrategyConfig.buy_delay` and `sell_delay`; defaults are one bar.

## Indicator Patterns

Built-ins:

```python
high_20 = highest("high_20", "high", period=20)
low_20 = lowest("low_20", "low", period=20)
ret_5 = returns("ret_5", "close", period=5)
```

Custom indicator functions receive `BarData` and must return a one-dimensional array or Series aligned to the input dates:

```python
import numpy as np

def sma(data, period: int):
    close = data.close
    out = np.full_like(close, np.nan, dtype=float)
    for i in range(period - 1, len(close)):
        out[i] = close[i - period + 1 : i + 1].mean()
    return out

sma_50 = indicator("sma_50", sma, period=50)
```

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

Ranked rotation:

```python
config = StrategyConfig(max_long_positions=3, exit_on_last_bar=True)

def rank_by_momentum(ctx: ExecContext):
    if ctx.bars < 63:
        return
    momentum = ctx.close[-1] / ctx.close[-63] - 1
    ctx.score = momentum
    if not ctx.long_pos() and momentum > 0:
        ctx.buy_shares = ctx.calc_target_shares(1 / 3)
        ctx.hold_bars = 21
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

## Validation Checklist

- Syntax-check created Python files with `python -m py_compile <file>`.
- If using DataFrame data, include a tiny local fixture and run the backtest without network access.
- If using `YFinance`, expect network/package availability to be an execution dependency and mention when not run.
- Inspect `result.metrics_df`, `result.orders`, and `result.trades` for empty or impossible behavior.
- Keep generated examples reproducible by setting explicit dates, symbols, config, and random seeds where applicable.
