"""Bottleneck benchmarks for pybroker.common hot paths."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from pybroker.common import DataCol, quantize
from pybroker.portfolio import Order, PortfolioBar, Trade
from pybroker.timeframe import compress, compress_timeframes_from_frame

DAILY_SECONDS = 86400.0
LARGE_MINUTE_BARS = 252 * 390


def _minute_bars(n: int = 390) -> tuple[np.ndarray, ...]:
    dates = pd.date_range("2020-01-06 09:30", periods=n, freq="1min").to_numpy(
        dtype="datetime64[ns]"
    )
    close = np.linspace(100.0, 110.0, n)
    return (
        dates,
        close,
        close + 0.5,
        close - 0.5,
        close,
        np.ones(n),
    )


def _multi_symbol_daily_df(
    n_symbols: int = 10, n_days: int = 504
) -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    frames: list[pd.DataFrame] = []
    for i in range(n_symbols):
        sym = f"SYM{i:02d}"
        close = np.linspace(100.0 + i, 150.0 + i, n_days)
        frames.append(
            pd.DataFrame(
                {
                    DataCol.SYMBOL.value: [sym] * n_days,
                    DataCol.DATE.value: dates,
                    DataCol.OPEN.value: close,
                    DataCol.HIGH.value: close + 1.0,
                    DataCol.LOW.value: close - 1.0,
                    DataCol.CLOSE.value: close,
                    DataCol.VOLUME.value: np.ones(n_days) * 1_000_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _decimal_series(n: int, seed: float = 100.0) -> list[Decimal]:
    values = np.linspace(seed, seed + 50.0, n)
    return [Decimal(f"{v:.4f}") for v in values]


class ResultQuantize:
    """Result-export quantize loop on realistic Decimal frames."""

    timeout = 120

    def setup(self) -> None:
        n_portfolio = 12_000
        n_orders = 5_000
        n_trades = 3_000
        dates = pd.date_range("2020-01-02", periods=n_portfolio, freq="B")
        decs = _decimal_series(n_portfolio)
        self._portfolio_template = pd.DataFrame(
            {
                "date": dates.to_numpy(dtype="datetime64[ns]"),
                "cash": decs,
                "equity": decs,
                "margin": decs,
                "margin_loan": decs,
                "net_cash_balance": decs,
                "market_value": decs,
                "pnl": decs,
                "unrealized_pnl": decs,
                "fees": decs,
            }
        ).set_index("date")
        order_decs = _decimal_series(n_orders, seed=50.0)
        self._orders_template = pd.DataFrame(
            {
                "limit_price": order_decs,
                "fill_price": order_decs,
                "fees": order_decs,
            }
        )
        trade_decs = _decimal_series(n_trades, seed=75.0)
        self._trades_template = pd.DataFrame(
            {
                "entry": trade_decs,
                "exit": trade_decs,
                "pnl": trade_decs,
                "return_pct": trade_decs,
                "agg_pnl": trade_decs,
                "pnl_per_bar": trade_decs,
                "mae": trade_decs,
                "mfe": trade_decs,
            }
        )
        _ = (PortfolioBar._fields, Order._fields, Trade._fields)

    def time_result_quantize(self) -> None:
        portfolio_df = self._portfolio_template.copy(deep=True)
        orders_df = self._orders_template.copy(deep=True)
        trades_df = self._trades_template.copy(deep=True)
        for col in (
            "cash",
            "equity",
            "margin",
            "margin_loan",
            "net_cash_balance",
            "market_value",
            "pnl",
            "unrealized_pnl",
            "fees",
        ):
            portfolio_df[col] = quantize(portfolio_df, col, True)
        for col in ("limit_price", "fill_price", "fees"):
            orders_df[col] = quantize(orders_df, col, True)
        for col in (
            "entry",
            "exit",
            "pnl",
            "return_pct",
            "agg_pnl",
            "pnl_per_bar",
            "mae",
            "mfe",
        ):
            trades_df[col] = quantize(trades_df, col, True)


class TimeframeCompression:
    """Large timeframe compression stressing to_seconds / parse_timeframe."""

    timeout = 120

    def setup(self) -> None:
        dates, o, h, low, c, v = _minute_bars(LARGE_MINUTE_BARS)
        self._minute = (dates, o, h, low, c, v)
        self._multi_df = _multi_symbol_daily_df()
        self._symbols = set(self._multi_df[DataCol.SYMBOL.value].unique())
        self._intervals = frozenset({"weekly", 5, "monthly"})
        self._custom_cols: frozenset[str] = frozenset()
        compress(dates, o, h, low, c, v, "5m")
        compress_timeframes_from_frame(
            self._multi_df,
            self._symbols,
            self._intervals,
            self._custom_cols,
            DAILY_SECONDS,
        )

    def time_compress_5m_large(self) -> None:
        dates, o, h, low, c, v = self._minute
        compress(dates, o, h, low, c, v, "5m")

    def time_compress_timeframes_multi(self) -> None:
        compress_timeframes_from_frame(
            self._multi_df,
            self._symbols,
            self._intervals,
            self._custom_cols,
            DAILY_SECONDS,
        )
