from pybroker import YFinance
from pybroker import AKShare
import akshare as ak

# yfinance = YFinance()
# yfinance_df = yfinance.query(['000001.SZ', '000002.SZ'], start_date='3/1/2000', end_date='1/1/2023')

# akshare = AKShare()
# akshare_df = akshare.query(['000001.SZ', '000002.SZ'], start_date='3/1/2000', end_date='1/1/2023')


from pybroker import ExecContext, Strategy, YFinance

def buy_fn(ctx: ExecContext):
    if not ctx.long_pos():
        ctx.buy_shares = 100
        ctx.buy_limit_price = ctx.close[-1] * 0.99
        ctx.hold_bars = 10

# strategy = Strategy(YFinance(), start_date='3/1/2022', end_date='1/1/2023')
strategy = Strategy(AKShare(), start_date='3/1/2022', end_date='1/1/2023')
# strategy.add_execution(buy_fn, ['000001.SZ', '000002.SZ'])
strategy.add_execution(buy_fn, ['000001', '000002'])
result = strategy.backtest()
result.orders.head(10)