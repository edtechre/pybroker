# Modeling Slippage

Source: `docs/source/notebooks/14. Modeling Slippage.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# Modeling Slippage

In live trading, orders rarely fill at the exact price a backtest assumes. Factors like spreads, latency, and the market impact of your own order push fill prices in an adverse direction. This difference is called *slippage*. A backtest that ignores it will overstate a strategy's true performance.

This notebook demonstrates **PyBroker's** three built-in slippage models added in v2. It also introduces the [SlippageModel](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.SlippageModel) base class, which you can use to write your own custom models.

## A Baseline Strategy

To see the effect of each model, we reuse the dip-buying strategy from [Backtesting a Strategy](https://www.pybroker.com/en/latest/notebooks/2.%20Backtesting%20a%20Strategy.html). The rule is simple: buy when the latest close drops below the previous day's low. We allocate 25% of the portfolio to the position with [calc_target_shares](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.calc_target_shares) and hold it for 3 bars via [hold_bars](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.hold_bars). Because this strategy trades frequently, small per-fill costs compound into a visible difference in total return.

```python
import pybroker
from pybroker import Strategy, YFinance

pybroker.enable_data_source_cache("slippage")


def buy_low(ctx):
    # If shares were already purchased and are currently being held, then
    # return.
    if ctx.long_pos():
        return
    # If the latest close price is less than the previous day's low price,
    # then place a buy order.
    if ctx.bars >= 2 and ctx.close[-1] < ctx.low[-2]:
        # Buy a number of shares that is equal to 25% of the portfolio.
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        # Hold the position for 3 bars before liquidating.
        ctx.hold_bars = 3


symbols = ["F", "BAC", "T"]
strategy = Strategy(YFinance(), start_date="1/1/2021", end_date="1/1/2026")
strategy.add_execution(buy_low, symbols)
```

First, we run a baseline backtest with no slippage. Every order fills at the unadjusted fill price. By default, this is the midpoint between the bar's low and high prices ([PriceType.MIDDLE](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.PriceType.MIDDLE)):

```python
result = strategy.backtest()
print(f"Total return: {result.metrics.total_return_pct:.2f}%")
result.orders.head()
```

## Fixed Slippage

The [FixedSlippageModel](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.FixedSlippageModel) applies a deterministic, adverse adjustment measured in basis points (where 1 basis point equals 0.01%). Buy fills are increased by `bps`, while sell fills are decreased. Passing `bps=0` disables the adjustment entirely.

Because the remaining examples run several more backtests, we will also disable logging with [disable_logging](https://www.pybroker.com/en/latest/reference/pybroker.scope.html#pybroker.scope.disable_logging) to keep the output short. Next, we attach the model to a [Strategy](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy) with [set_slippage_model](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_slippage_model):

```python
from pybroker import FixedSlippageModel

pybroker.disable_logging()

strategy.set_slippage_model(FixedSlippageModel(bps=10))
result = strategy.backtest()
print(f"Total return: {result.metrics.total_return_pct:.2f}%")
result.orders[result.orders["symbol"] == "T"].head()
```

## Volatility Slippage

Because slippage tends to grow as volatility rises, the [VolatilitySlippageModel](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.VolatilitySlippageModel) ties its adverse adjustment to the market's movement. It scales the slippage using the [Average True Range (ATR)](https://en.wikipedia.org/wiki/Average_true_range) of the fill bar (see [atr](https://www.pybroker.com/en/latest/reference/pybroker.vect.html#pybroker.vect.atr)), pushing the fill price against your order by `scale * ATR`. The ATR is computed over the `atr_period` bars ending at the fill bar. Fills during the warmup period (before a full ATR window exists), are left unadjusted:

```python
from pybroker import VolatilitySlippageModel

strategy.set_slippage_model(VolatilitySlippageModel(atr_period=14, scale=0.1))
result = strategy.backtest()
print(f"Total return: {result.metrics.total_return_pct:.2f}%")
result.orders.head()
```

## Volume Slippage

The [VolumeSlippageModel](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.VolumeSlippageModel) accounts for finite liquidity by introducing two effects:

1. **Volume limit:** The number of filled shares is capped at a percentage of the bar's total volume (`volume_limit * volume`). Any remaining shares in the order are dropped rather than filled later.
2. **Price impact:** The fill price moves against you based on your order size relative to the market. The adjustment is calculated as `price_impact * (filled_shares / volume) ** 2` multiplied by the fill price.

A $100,000 account will rarely hit these limits when trading liquid large caps. For example, a 25% allocation would just be a rounding error in Ford's daily volume. However, that same allocation can be a significant portion of the day's trading in thinly traded small caps. Without a volume model, the backtest unrealistically assumes the entire order fills at the quoted price:

```python
from pybroker import VolumeSlippageModel

smallcaps = Strategy(YFinance(), start_date="1/1/2021", end_date="1/1/2026")
smallcaps.add_execution(buy_low, ["ESCA", "BSET", "HURC"])
result = smallcaps.backtest()
print(f"Return without a volume model: {result.metrics.total_return_pct:.2f}%")

smallcaps.set_slippage_model(
    VolumeSlippageModel(price_impact=0.1, volume_limit=0.025)
)
result = smallcaps.backtest()
print(f"Return with a volume model: {result.metrics.total_return_pct:.2f}%")
result.orders.head()
```

## Writing a Custom Slippage Model

To create your own model, subclass [SlippageModel](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.SlippageModel) and override the [apply_slippage](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.SlippageModel.apply_slippage) method. This method takes a [SlippageContext](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.SlippageContext) object containing the order's [side](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.SlippageContext.side) (`"buy"` or `"sell"`), [symbol](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.SlippageContext.symbol), [shares](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.SlippageContext.shares), and the initial [fill_price](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.SlippageContext.fill_price). Your method must then return a tuple with the adjusted `(shares, fill_price)`.

The following example demonstrates a model that applies a random amount of adverse slippage to every fill:

```python
from decimal import Decimal

import numpy as np
from pybroker import SlippageContext, SlippageModel


class RandomSlippageModel(SlippageModel):
    """Applies random adverse slippage of up to ``max_bps`` per fill."""

    def __init__(self, max_bps: float = 10, seed: int = 42):
        self.max_bps = max_bps
        self._rng = np.random.default_rng(seed)

    def apply_slippage(self, ctx: SlippageContext) -> tuple[Decimal, Decimal]:
        bps = self._rng.uniform(0, self.max_bps)
        adjustment = ctx.fill_price * Decimal(str(bps / 10_000))
        if ctx.side == "buy":
            fill_price = ctx.fill_price + adjustment
        else:
            fill_price = ctx.fill_price - adjustment
        return ctx.shares, fill_price


strategy.set_slippage_model(RandomSlippageModel(max_bps=10, seed=42))
result = strategy.backtest()
print(f"Total return: {result.metrics.total_return_pct:.2f}%")
```
