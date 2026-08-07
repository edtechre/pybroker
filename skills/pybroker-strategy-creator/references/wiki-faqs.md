# FAQs

Source: `docs/source/notebooks/FAQs.ipynb`

This reference was generated from the local PyBroker documentation notebook. Use it as the detailed wiki page for this topic.

# FAQs

### How to...

### ... get your version of PyBroker?

```python
import pybroker

pybroker.__version__
```

### ... get data for another symbol?

```python
from pybroker import ExecContext, Strategy, YFinance, highest

def exec_fn(ctx: ExecContext):
    if ctx.symbol == 'NVDA':
        other_bar_data = ctx.foreign('AMD')
        other_highest = ctx.indicator('high_10d', 'AMD')
        
strategy = Strategy(YFinance(), start_date='1/1/2022', end_date='1/1/2023')
strategy.add_execution(
   exec_fn, ['NVDA', 'AMD'], indicators=highest('high_10d', 'close', period=10))
result = strategy.backtest()
```

You can also retrieve models, predictions, and other data for other symbols. [For more information, refer to the ExecContext reference documentation.](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext)

### ... set a limit price?

Set [buy_limit_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.buy_limit_price) or [sell_limit_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.sell_limit_price):

```python
from pybroker import ExecContext, Strategy, YFinance

def buy_fn(ctx: ExecContext):
    if not ctx.long_pos():
        ctx.buy_shares = 100
        ctx.buy_limit_price = ctx.close[-1] * 0.99
        ctx.hold_bars = 10
        
strategy = Strategy(YFinance(), start_date='3/1/2022', end_date='1/1/2023')
strategy.add_execution(buy_fn, 'SPY')
result = strategy.backtest()
result.orders.head(10)
```

### ... set the fill price?

Set [buy_fill_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.buy_fill_price) and [sell_fill_price](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.sell_fill_price). See [PriceType](https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.PriceType) for options.

```python
from pybroker import ExecContext, PriceType, Strategy, YFinance

def exec_fn(ctx: ExecContext):
    if ctx.long_pos():
        ctx.buy_shares = 100
        ctx.buy_fill_price = PriceType.AVERAGE
    else:
        ctx.sell_shares = 100
        ctx.sell_fill_price = PriceType.CLOSE
        
strategy = Strategy(YFinance(), start_date='3/1/2022', end_date='1/1/2023')
strategy.add_execution(buy_fn, 'SPY')
result = strategy.backtest()
result.orders.head(10)
```

### ... get current positions?

```python
from pybroker import ExecContext, Strategy, YFinance

def exec_fn(ctx: ExecContext):
    # Get all positions.
    all_positions = tuple(ctx.positions())
    # Get all long positions.
    long_positions = tuple(ctx.long_positions())
    # Get all short positions.
    short_positions = tuple(ctx.short_positions())
    # Get long position for current ctx.symbol.
    long_position = ctx.long_pos()
    # Get short position for a symbol.
    short_position = ctx.short_pos('QQQ')
        
strategy = Strategy(YFinance(), start_date='3/1/2022', end_date='1/1/2023')
strategy.add_execution(exec_fn, ['SPY', 'QQQ'])
result = strategy.backtest()
```

[See the Position class for more information.](https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.Position)

### ... use custom column data?

Register your custom columns with [pybroker.register_columns](https://www.pybroker.com/en/latest/reference/pybroker.scope.html#pybroker.scope.register_columns):

```python
import pybroker
from pybroker import ExecContext, Strategy, YFinance

yf = YFinance()
df = yf.query('SPY', start_date='1/1/2022', end_date='1/1/2023')
df['buy_signal'] = 1

def buy_fn(ctx: ExecContext):
    if not ctx.long_pos() and ctx.buy_signal[-1] == 1:
        ctx.buy_shares = 100
        ctx.hold_bars = 1
        
pybroker.register_columns('buy_signal')
strategy = Strategy(df, start_date='3/1/2022', end_date='1/1/2023')
strategy.add_execution(buy_fn, 'SPY')
result = strategy.backtest()
result.orders.head(10)
```

### ... place an order more than one bar ahead?

Use the [buy_delay](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.buy_delay) and [sell_delay](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.sell_delay) configuration options:

```python
from pybroker import ExecContext, Strategy, StrategyConfig, YFinance

def buy_fn(ctx: ExecContext):
    if not tuple(ctx.pending_orders()) and not ctx.long_pos():
        ctx.buy_shares = 100
        ctx.hold_bars = 1

config = StrategyConfig(buy_delay=5)
strategy = Strategy(YFinance(), start_date='3/1/2022', end_date='1/1/2023', config=config)
strategy.add_execution(buy_fn, 'SPY')
result = strategy.backtest()
result.orders.head(10)
```

### ... cancel pending orders?

See the [cancel_pending_order](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.cancel_pending_order) and [cancel_all_pending_orders](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.cancel_all_pending_orders) methods.

```python
from pybroker import ExecContext, Strategy, StrategyConfig, YFinance

def buy_fn(ctx: ExecContext):
    pending = tuple(ctx.pending_orders())
    if not pending and not ctx.long_pos():
        ctx.buy_shares = 100
        ctx.hold_bars = 1
    if pending and ctx.close[-1] < 430:
        ctx.cancel_all_pending_orders(ctx.symbol)

config = StrategyConfig(buy_delay=5)
strategy = Strategy(YFinance(), start_date='3/1/2022', end_date='1/1/2023', config=config)
strategy.add_execution(buy_fn, 'SPY')
result = strategy.backtest()
result.orders.head(10)
```

### ... persist data across bars?

Use the [ExecContext#session](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.session) dictionary:

```python
from pybroker import ExecContext, Strategy, YFinance

def buy_fn(ctx: ExecContext):
    if not ctx.long_pos():
        ctx.buy_shares = 100
        ctx.hold_bars = 1
        count = ctx.session.get('entry_count', 0)
        ctx.session['entry_count'] = count + 1

strategy = Strategy(YFinance(), start_date='1/1/2022', end_date='1/1/2023')
strategy.add_execution(buy_fn, 'SPY')
result = strategy.backtest()
```

### ... exit a position?

Use [sell_all_shares()](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.sell_all_shares) or [cover_all_shares()](https://www.pybroker.com/en/latest/reference/pybroker.context.html#pybroker.context.ExecContext.cover_all_shares) to liquidate a position:

```python
from pybroker import ExecContext, Strategy, YFinance

def buy_fn(ctx: ExecContext):
    pos = ctx.long_pos()
    if not pos:
        ctx.buy_shares = 100
    elif pos.bars > 30:
        ctx.sell_all_shares()

strategy = Strategy(YFinance(), start_date='1/1/2022', end_date='1/1/2023')
strategy.add_execution(buy_fn, 'SPY')
result = strategy.backtest()
result.trades
```

### ... process multiple symbols at once?

Use [set_before_exec](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_before_exec) or [set_after_exec](https://www.pybroker.com/en/latest/reference/pybroker.strategy.html#pybroker.strategy.Strategy.set_after_exec):

```python
from pybroker import ExecContext, Strategy, YFinance

def long_short_fn(ctxs: dict[str, ExecContext]):
    nvda_ctx = ctxs['NVDA']
    amd_ctx = ctxs['AMD']
    if nvda_ctx.long_pos() or amd_ctx.short_pos():
        return
    if nvda_ctx.bars >= 2 and nvda_ctx.close[-1] < nvda_ctx.low[-2]:
        nvda_ctx.buy_shares = 100
        nvda_ctx.hold_bars = 3
        amd_ctx.sell_shares = 100
        amd_ctx.hold_bars = 3

strategy = Strategy(YFinance(), start_date='1/1/2022', end_date='1/1/2023')
strategy.add_execution(None, ['NVDA', 'AMD'])
strategy.set_after_exec(long_short_fn)
result = strategy.backtest()
result.trades
```

### ... annualize the Sharpe Ratio?

Set the [bars_per_year](https://www.pybroker.com/en/latest/reference/pybroker.config.html#pybroker.config.StrategyConfig.bars_per_year) configuration option. For example, setting a value of ``252`` would be used to annualize daily returns.

```python
from pybroker import ExecContext, Strategy, StrategyConfig, YFinance

def buy_fn(ctx: ExecContext):
    if ctx.long_pos() or ctx.bars < 2:
        return
    if ctx.close[-1] < ctx.high[-2]:
        ctx.buy_shares = 100
        ctx.hold_bars = 1

config = StrategyConfig(bars_per_year=252)
strategy = Strategy(YFinance(), start_date='1/1/2022', end_date='1/1/2023', config=config)
strategy.add_execution(buy_fn, 'SPY')
result = strategy.backtest()
result.metrics.sharpe
```

### ... limit margin used for short selling?

By default, PyBroker does not limit the amount of margin that can be used for short selling. However, you can manually limit the amount of margin that can be used:

```python
import pybroker
from pybroker import ExecContext
from decimal import Decimal

def short_fn(ctx: ExecContext):
    margin_requirement = Decimal('0.25')
    max_margin = ctx.total_equity / margin_requirement - ctx.total_equity
    if not ctx.short_pos():
        available_margin = max_margin - ctx.total_margin
        ctx.sell_shares = ctx.calc_target_shares(0.5, cash=available_margin)
        ctx.hold_bars = 1
        
strategy = Strategy(YFinance(), start_date='1/1/2022', end_date='1/1/2023')
strategy.add_execution(short_fn, ['NVDA', 'AMD'])
result = strategy.backtest()
result.portfolio.head(10)
```

### ... get and set a global parameter?

```python
import pybroker

# Set parameter.
pybroker.param('lookback', 100)

# Get parameter.
pybroker.param('lookback')
```

### ... apply random slippage?

Set a [RandomSlippageModel](https://www.pybroker.com/en/latest/reference/pybroker.slippage.html#pybroker.slippage.RandomSlippageModel):

```python
from pybroker import ExecContext, RandomSlippageModel, Strategy, YFinance

def buy_fn(ctx: ExecContext):
    if not ctx.long_pos():
        ctx.buy_shares = 100
        ctx.hold_bars = 1

slippage = RandomSlippageModel(min_pct=1.0, max_pct=5.0) # Slippage of 1-5%
strategy = Strategy(YFinance(), start_date='3/1/2022', end_date='1/1/2023')
strategy.set_slippage_model(slippage)
strategy.add_execution(buy_fn, 'SPY')
result = strategy.backtest()
result.orders.head(10)
```
