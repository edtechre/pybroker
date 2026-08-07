# PyBroker Rotational Trading Patterns

## Rotation API Map

Score symbols in the execution function, cap the slots, then enable
rotation:

```python
def rank(ctx: ExecContext):
    ctx.long_score = ...       # ranks buy/cover signals; higher wins
    ctx.short_score = ...      # ranks sell signals; higher wins

strategy.set_max_long_positions(n)   # int > 0 | Hyperparam | None
strategy.set_max_short_positions(n)  # caps rank the WHOLE portfolio

strategy.enable_rotation(
    worst_rank_held,  # hold band; must be >= every set cap;
                      # None disables rotation and clears the sizer
    sizer=None,       # Callable[[RotationContext], None], invoked
                      # after rotation decides what to trade
)
```

A `sizer` receives a `RotationContext` with `ctxs` (mapping of every
symbol to its `ExecContext`), `portfolio`, `long_ranks`, `short_ranks`
(1-based, `1` is the best score), and `config`. All three settings —
both caps and `worst_rank_held` — accept a `pybroker.optimize.Hyperparam`
in place of an int, so they are searchable with `Strategy.optimize`.

## Two Modes: Ranked Caps vs Rotation

Scores participate in two different modes. Pick deliberately:

**Ranked-cap mode** — `set_max_*_positions` plus scores, without
`enable_rotation`. Execution functions stay in charge: they place their
own buy/sell orders, and scores only decide which signals win when the
position cap binds. Orders past the cap are silently dropped (a debug
log only). A symbol that sets no score sorts as `0.0`; an unrankable
(NaN) score sorts last. Nothing is ever liquidated by ranking alone.

**Rotation mode** — `enable_rotation(worst_rank_held=...)`. Trading is
driven entirely by scores: held positions ranked worse than the hold
band are liquidated each bar, top-ranked candidates fill the freed
slots, and any orders the execution functions placed are discarded.

Choose ranked-cap mode to prioritize entry signals in an
order-placing strategy; choose rotation mode for hold-the-top-N
portfolios where the ranking *is* the strategy.

## Scoring Symbols

`ctx.long_score` ranks buy and cover signals; `ctx.short_score` ranks
sell signals. Both default to `None` and reset every bar. Ranking is
descending — the highest score is rank 1 — with the symbol name as a
deterministic tiebreak, and it spans the whole portfolio across every
`add_execution` call.

```python
roc_20 = pybroker.indicator(
    "roc_20", lambda data: roc(data.close, 20)
)


# Execution logic reads ctx.* NumPy arrays — never pandas.
def rank_by_momentum(ctx: ExecContext):
    ctx.long_score = ctx.indicator("roc_20")[-1]
```

## Position Caps

```python
strategy.set_max_long_positions(2)   # at most 2 long positions
strategy.set_max_short_positions(2)  # at most 2 short positions
```

- Accepted values: an int greater than 0 (zero or negative raises
  `ValueError`), a searchable `Hyperparam`, or `None` for unlimited.
- The `StrategyConfig.max_long_positions` and
  `StrategyConfig.max_short_positions` fields are deprecated. The
  setters take precedence, and using both warns.
- In ranked-cap mode the cap is enforced when orders fill: once the
  portfolio holds the maximum, further entries are dropped with only a
  debug log, so unscored strategies fill slots in arbitrary schedule
  order — set scores whenever a cap is set.

## Hold-Band Rotation Mechanics

`enable_rotation(worst_rank_held=k)` runs this per-bar algorithm after
all execution functions have been called:

1. Rank every symbol with a rankable `long_score` (and, when a short
   cap is set, every symbol with a rankable `short_score`).
2. Liquidate held positions whose rank is missing or worse than `k`
   (`sell_all_shares` for longs, `cover_all_shares` for shorts) — even
   positions that a different execution opened.
3. Fill the remaining free slots with the best-ranked symbols not
   already held or pending. Candidates ranked worse than `k` are never
   entered, since they would be liquidated on the next bar.
4. Size each entry with `set_target_shares` at equal weight
   `1 / (long slots + short slots)`, counting only the caps that are
   set. Two long slots and no short cap means 50% of deployable
   capital per position.
5. Invoke the `sizer`, if one was given, to override entry sizes.

```python
strategy = Strategy(YFinance(), "1/1/2020", "1/1/2024")
strategy.set_max_long_positions(2)
strategy.enable_rotation(worst_rank_held=5)
strategy.add_execution(rank_by_momentum, UNIVERSE, indicators=roc_20)
result = strategy.backtest(warmup=20)
```

This buys the two highest-ranked symbols and holds each until its rank
drops below 5. The gap between the cap (2) and the band (5) is
hysteresis: a wider band trades less and tolerates rank churn; setting
`worst_rank_held` equal to the cap rotates on every rank change.

Slot accounting details that prevent double-ordering:

- A held symbol with no bar on the current date keeps its slot; there
  is nothing to trade it against.
- In-flight pending entry orders (unfilled limit orders, orders whose
  execution bar falls later) hold their slots, and a pending exit is
  not re-issued.

## What Execution Functions Still Control Under Rotation

Rotation is exclusive: after the execution functions run, every order
they placed (`buy_shares`, `sell_shares`, limit prices, timeouts) is
discarded, and rotation places its own orders from the scores. What
survives and shapes rotation's orders:

- **Stops** — `stop_loss`/`stop_profit`/`stop_trailing` (and their
  `_pct`/`_limit`/`_exit_price` variants) and `hold_bars` set during
  execution are applied to the entries rotation places.
- **Fill prices** — `buy_fill_price`/`sell_fill_price` set during
  execution are used for rotation's orders (see Fill Prices and
  End-of-Data Exits for what they default to).

```python
def rank_with_stop(ctx: ExecContext):
    ctx.long_score = ctx.indicator("roc_20")[-1]
    # Kept and applied to the entry order rotation places:
    ctx.stop_loss_pct = 10
```

Stops and fill prices on symbols rotation leaves untraded this bar are
dropped automatically (an order-less stop is rejected otherwise), so
setting them unconditionally alongside the score is safe. A stop
triggered later exits the position as usual; rotation then sees a free
slot and refills it with the best-ranked candidate.

## Fill Prices and End-of-Data Exits

Rotation's orders fill at `PriceType.MIDDLE` unless the execution
function set a fill price. `MIDDLE` is the midpoint of the low and
high of the **execution** bar, one bar after the score under the
default `buy_delay`/`sell_delay` of `1`, so `PriceType.CLOSE` means
the next bar's close. `PriceType` offers `OPEN`, `HIGH`, `LOW`,
`CLOSE`, `MIDDLE` (`low + (high - low) / 2`, the default), and
`AVERAGE` (`(open + low + high + close) / 4`); the attributes also
accept a number or a `(symbol, bar_data)` callable, and read back as
`None` rather than `MIDDLE` until set.

```python
def rank_and_fill_at_open(ctx: ExecContext):
    ctx.long_score = ctx.indicator("roc_20")[-1]
    # Kept and applied to whatever order rotation places:
    ctx.buy_fill_price = PriceType.OPEN
    ctx.sell_fill_price = PriceType.OPEN
```

Because a whole universe rotates on one bar, the fill price choice
moves every leg at once. Setting it unconditionally alongside the
score is safe: fill prices on symbols rotation leaves untraded are
dropped automatically.

`StrategyConfig.exit_on_last_bar` defaults to `False`. A rotational
strategy is usually fully invested at the end of the run, so leaving
it off strands one open position per held slot: none of them become
`Trade`s, so `trade_count`, `win_rate`, `total_pnl` and the rest of
the trade table silently exclude them and their P&L sits in
`unrealized_pnl`. Set `exit_on_last_bar=True` whenever trade
statistics are reported:

```python
config = StrategyConfig(
    exit_on_last_bar=True,
    exit_sell_fill_price=PriceType.MIDDLE,  # default; longs exit here
    exit_cover_fill_price=PriceType.MIDDLE,  # default; shorts cover here
)
```

Both exit fill prices accept a `PriceType` or a `(symbol, bar_data)`
callable, but not a bare number. The liquidation bypasses position
caps and order delays but still records real `Order` and `Trade` rows.
Bar-level metrics (`sharpe`, `max_drawdown`) are computed from per-bar
market value and barely move either way. In `walkforward` it fires
only on each symbol's true final bar, never at window boundaries.

## Unrankable Scores, Warmup, and Forced Exits

A score of `None` or NaN is unrankable and excludes the symbol from
the rank map. Under rotation that has teeth: an unranked held position
is liquidated, because "no rank" and "rank worse than the band" are
treated the same.

- Indicator NaN warmup is harmless: no positions exist yet, and
  `warmup=` should cover the lookback anyway.
- An indicator that goes NaN mid-series (a vendor gap, a division
  guard) force-exits the position on that bar. If that is not
  intended, keep the previous score instead of passing NaN through.
- In ranked-cap mode, unrankable scores merely sort last; nothing is
  liquidated.

## Long/Short Rotation and Overlap

Set both caps and both scores to rotate two legs at once:

```python
def rank_both_sides(ctx: ExecContext):
    roc = ctx.indicator("roc_20")[-1]
    ctx.long_score = roc    # long leg: highest momentum
    ctx.short_score = -roc  # short leg: most negative momentum


strategy.set_max_long_positions(2)
strategy.set_max_short_positions(2)
strategy.enable_rotation(worst_rank_held=5)
```

Each leg ranks independently against its own score and its own cap,
within the shared `worst_rank_held` band (which must be >= both caps).
A symbol picked by both legs on the same bar goes to the side where it
ranks better; ties go long. Overlap is only reachable when the two
position limits together exceed the number of rankable symbols. When
reporting short positions, show `margin` and `unrealized_pnl`.

## Custom Sizers with RotationContext

The default sizing is equal weight across all slots. A `sizer`
overrides entry sizes after rotation has decided what to trade:

```python
from pybroker import RotationContext


def size_by_rank(rotation: RotationContext):
    weights = {1: 0.7, 2: 0.3}
    for symbol, ctx in rotation.ctxs.items():
        # buy_shares is set only on the entries rotation placed.
        if ctx.buy_shares is not None:
            rank = rotation.long_ranks[symbol]
            ctx.buy_shares = ctx.calc_target_shares(weights[rank])


strategy.enable_rotation(worst_rank_held=5, sizer=size_by_rank)
```

The two sizer invariants:

- Guard with `if ctx.buy_shares is not None:` (`sell_shares` for short
  entries). Only symbols rotation is entering carry an order; sizing
  anything else creates trades rotation never decided on.
- Never override the sell or cover signals rotation set — exits are
  rotation's contract, not the sizer's.

A rank-decay alternative that adapts to any slot count:

```python
def linear_decay(rotation: RotationContext):
    n = rotation.config.max_long_positions
    total = n * (n + 1) / 2
    for symbol, ctx in rotation.ctxs.items():
        if ctx.buy_shares is not None:
            rank = rotation.long_ranks[symbol]
            ctx.buy_shares = ctx.calc_target_shares(
                (n - rank + 1) / total
            )
```

`calc_target_shares(target)` sizes against deployable capital
(portfolio equity times `config.leverage`); `rotation.portfolio`
exposes equity and positions for sizing rules that need them.

## Ranked-Cap Recipes

Scores without `enable_rotation` prioritize order-placing strategies.
The execution function still decides when to enter and exit; the score
decides who wins scarce slots:

```python
# Execution logic reads ctx.* NumPy arrays — never pandas.
def buy_breakout(ctx: ExecContext):
    highs = ctx.indicator("high_20")
    if len(highs) < 2 or np.isnan(highs[-2]):
        return
    if not ctx.long_pos():
        if ctx.close[-1] > highs[-2]:
            ctx.buy_shares = ctx.calc_target_shares(0.25)
            ctx.hold_bars = 5
            # Prefer the strongest breakout when slots are scarce.
            ctx.long_score = ctx.indicator("roc_20")[-1]


strategy.set_max_long_positions(2)
strategy.add_execution(
    buy_breakout, UNIVERSE, indicators=[high_20, roc_20]
)
```

Set at most one order side per symbol per bar. For a two-sided ranked
strategy, branch on the signal and set `sell_shares` with
`ctx.short_score` on the short branch (see
`wiki-04-ranking-long-and-short-signals.md` for a worked long/short
example with `hold_bars`).

## Dynamic Universes with SymbolSelector

To rotate within a screened universe, pass a `SymbolSelector` — any
callable `(df) -> Sequence[str]` — as the `add_execution` symbols. The
selector is a sanctioned pandas boundary: it receives the DataFrame
that PyBroker hands it.

```python
def top_dollar_volume(df):
    dollar_volume = (
        (df["close"] * df["volume"]).groupby(df["symbol"]).mean()
    )
    return dollar_volume.nlargest(10).index


strategy = Strategy(df, START_DATE, END_DATE)  # DataFrame source
strategy.set_max_long_positions(2)
strategy.enable_rotation(worst_rank_held=5)
strategy.add_execution(
    rank_by_momentum, top_dollar_volume, indicators=roc_20
)
result = strategy.walkforward(windows=4, train_size=0.5)
```

Selector rules:

- It runs once per walkforward window on that window's **training**
  data, never test data, so it needs a training window: `backtest()`
  and `train_size=0` raise `ValueError`.
- The candidate universe must be supplied as a DataFrame, not a
  `DataSource` — the symbols to query are unknown until a window
  splits.
- A position in a symbol that a later window drops is closed at the
  first bar of that window.
- Avoid `shuffle=True` with selectors that depend on bar order; it
  randomizes the training frame's row order.

## Optimizing Rotation Parameters

Both caps and the hold band accept a `Hyperparam`, so the structure of
the rotation is searchable:

```python
max_long = pybroker.hyperparam("max_long", default=2, low=2, high=4,
                               step=1)
band = pybroker.hyperparam("band", default=5, low=4, high=8, step=1)
strategy.set_max_long_positions(max_long)
strategy.enable_rotation(worst_rank_held=band)
opt = strategy.optimize(score_fn, seed=42)
```

Keep `low` on the band at or above `high` on every cap so no sampled
combination violates `worst_rank_held >= cap`. The full optimization
workflow (samplers, score functions, walkforward windows) is the
`pybroker-optimize` skill's territory.

## Common Errors

Match the exact message before changing code:

| Message | Cause | Fix |
| --- | --- | --- |
| `worst_rank_held requires max_long_positions or max_short_positions to be set.` | `enable_rotation` with no position cap | Call `set_max_long_positions` and/or `set_max_short_positions` first |
| `worst_rank_held must be greater than or equal to max_long_positions.` (also the `max_short_positions` variant) | Hold band tighter than a cap | Widen `worst_rank_held` or lower the cap |
| `Rotation sizer is set but rotation is not enabled; call enable_rotation(worst_rank_held=...) first.` | `sizer=` given with rotation off (or disabled with `None`) | Enable rotation, or drop the sizer |
| `max_long_positions must be greater than 0.` (also the short variant) | Zero or negative cap | Use a positive int, or `None` for unlimited |
| Warning that `Strategy.set_max_long_positions` takes precedence over `StrategyConfig.max_long_positions` | Both the deprecated config field and the setter in use | Delete the `StrategyConfig` field |

## Reporting Results

Print `result.metrics_df` as the human-readable summary and inspect
`result.orders` to confirm rotation entries and hold-band exits. For
structured output (agent parsing, saved report files, downstream
tools), `result.to_json()` returns a JSON-safe dict and
`result.to_json_str()` strict JSON text. The default payload carries
metrics, trades, orders, and bootstrap capped at `max_rows=100` rows
per table; `symbols=` filters to specific tickers; `include=` opts
into `portfolio`/`positions`/`metrics_df`/`signals`/`stops`. Dates
serialize as naive-UTC ISO strings, NaN as `null`, and legitimately
infinite metrics as `"Infinity"`/`"-Infinity"`.

## Session Hygiene and Debugging

Put these at the top of agent-run rotation scripts:

```python
pybroker.enable_data_source_cache("my_rotation")  # skip refetching
pybroker.disable_progress_bar()  # progress bars flood AI context
```

Add `pybroker.disable_logging()` when running many backtests (for
example, optimizing the hold band). To verify rotation behavior,
inspect `result.orders`: entries should never exceed the position
caps, and each exit should line up with the symbol's rank leaving the
hold band or a stop triggering.

Never use pandas to implement indicator or execution logic. Indicators
are plain NumPy over `BarData` arrays (use Numba `@njit` for explicit
loops), and execution functions read `ctx.*` NumPy arrays — no
`pd.Series`/`pd.DataFrame` construction and no
`.rolling`/`.ewm`/`.shift`/`.apply` in either. The only pandas here
belongs in a `SymbolSelector`, which receives a DataFrame by contract.

Numba debug toggle: a Numba compilation or typing error in an `@njit`
indicator is easier to read as plain Python. Re-run once with JIT
disabled to get a normal traceback at the offending line, then
re-enable JIT for the real run:

```bash
NUMBA_DISABLE_JIT=1 python my_rotation.py
```

## Validation Checklist

- Syntax-check created Python files with `python -m py_compile <file>`.
- Generated scripts start with `pybroker.disable_progress_bar()` and
  `pybroker.enable_data_source_cache(...)` (or `enable_caches`), plus
  `pybroker.disable_logging()` for many-backtest runs.
- Scores use `ctx.long_score`/`ctx.short_score`; no
  `StrategyConfig(max_long_positions=...)`/`max_short_positions`
  anywhere in the deliverable.
- `worst_rank_held` is greater than or equal to every position cap
  that is set, and at least one cap is set before `enable_rotation`.
- Under rotation, execution functions set only scores, stops, and fill
  prices; any order they place is dead code.
- `StrategyConfig(exit_on_last_bar=True)` is set whenever trade-level
  metrics are reported; a fully invested rotation otherwise leaves one
  unclosed position per slot out of `trade_count` and `total_pnl`.
- A sizer guards entries with `if ctx.buy_shares is not None:` (or
  `sell_shares` for shorts) and never overrides rotation's exits.
- `warmup=` (or a `ctx.bars` guard) covers the ranking indicator's
  lookback, and the NaN-mid-series liquidation behavior was
  considered.
- No pandas in indicator or execution logic; indicators return
  full-length NaN-padded arrays and pass the bump-last-bar check:
  change only the final input bar and assert every earlier output is
  unchanged.
- Required installs are named and never assumed: `pip install
  yfinance` for the `YFinance` data source, `pip install TA-Lib` only
  if the user insists on TA-Lib indicators.
