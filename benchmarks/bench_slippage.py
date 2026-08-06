"""End-to-end walkforward benchmarks stressing fill-time slippage models."""

from __future__ import annotations

import pybroker
from pybroker import ExecContext, Strategy, StrategyConfig
from pybroker.slippage import VolatilitySlippageModel, VolumeSlippageModel
from pybroker.vect import sumv

from benchmarks.bench_backtest import (
    LOOKAHEAD,
    SEED,
    _synthetic_ohlcv,
)

_WINDOWS = 5
_N_SYMBOLS = 10
_N_DAYS = 252 * 5


def _sma_crossover_exec(ctx: ExecContext) -> None:
    close = ctx.close
    if len(close) < 21:
        return
    sma5 = sumv(close, 5)[-1] / 5
    sma20 = sumv(close, 20)[-1] / 20
    if ctx.long_pos() and sma5 < sma20:
        ctx.sell_all_shares()
    elif not ctx.long_pos() and sma5 > sma20:
        ctx.buy_shares = 100


def _build_slippage_strategy(
    df,
    slippage_model,
    *,
    indicators=None,
) -> Strategy:
    pybroker.clear_params()
    pybroker.disable_logging()
    pybroker.disable_progress_bar()
    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")
    strategy = Strategy(df, start, end, StrategyConfig())
    strategy.set_slippage_model(slippage_model)
    strategy.add_execution(
        _sma_crossover_exec,
        symbols=sorted(df["symbol"].unique().tolist()),
        indicators=indicators or [],
    )
    return strategy


class WalkforwardVolumeSlippage:
    """Walkforward with high fill frequency and VolumeSlippageModel."""

    timeout = 900

    def setup(self) -> None:
        import numpy as np

        np.random.seed(SEED)
        self.df = _synthetic_ohlcv(
            n_symbols=_N_SYMBOLS, n_days=_N_DAYS, seed=SEED
        )
        model = VolumeSlippageModel(price_impact=0.1, volume_limit=0.025)
        _build_slippage_strategy(self.df, model).walkforward(
            windows=_WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            parallel_indicators=False,
        )

    def time_walkforward_volume_slippage(self) -> None:
        model = VolumeSlippageModel(price_impact=0.1, volume_limit=0.025)
        _build_slippage_strategy(self.df, model).walkforward(
            windows=_WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            parallel_indicators=False,
        )


class WalkforwardVolatilitySlippage:
    """Walkforward with high fill frequency and VolatilitySlippageModel."""

    timeout = 900

    def setup(self) -> None:
        import numpy as np

        np.random.seed(SEED)
        self.df = _synthetic_ohlcv(
            n_symbols=_N_SYMBOLS, n_days=_N_DAYS, seed=SEED
        )
        # The model computes ATR from bar data itself; no indicator wiring.
        model = VolatilitySlippageModel(atr_period=14, scale=0.1)
        _build_slippage_strategy(self.df, model).walkforward(
            windows=_WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            parallel_indicators=False,
        )

    def time_walkforward_volatility_slippage(self) -> None:
        model = VolatilitySlippageModel(atr_period=14, scale=0.1)
        _build_slippage_strategy(self.df, model).walkforward(
            windows=_WINDOWS,
            lookahead=LOOKAHEAD,
            calc_bootstrap=False,
            parallel_indicators=False,
        )
