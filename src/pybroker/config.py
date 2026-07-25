"""Contains configuration options."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

from pybroker.common import BarData, FeeInfo, FeeMode, PositionMode, PriceType
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Optional, Union


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration options for :class:`pybroker.strategy.Strategy`.

    Attributes:
        initial_cash: Starting cash of strategy.
        fee_mode: :class:`pybroker.common.FeeMode` for calculating brokerage
            fees. Supports one of:

            - ``ORDER_PERCENT``: Fee is a percentage of order amount.
            - ``PER_ORDER``: Fee is a constant amount per order.
            - ``PER_SHARE``: Fee is a constant amount per share in order.
            - ``Callable[[FeeInfo], Decimal]``: Fees are calculated using a
                custom ``Callable`` that is passed
                :class:`pybroker.common.FeeInfo`.
            - ``None``: Fees are disabled (default).
        fee_amount: Brokerage fee amount.
        enable_fractional_shares: Whether to enable trading fractional shares.
            Set to ``True`` for crypto trading. Defaults to ``False``.
        round_fill_price: Whether to round fill prices to the nearest cent.
            Defaults to ``True``.
        position_mode: Position mode for :class:`pybroker.strategy.Strategy`.
            Supports one of:

            - ``DEFAULT``: Long and short positions.
            - ``LONG_ONLY``: Long-only positions.
            - ``SHORT_ONLY``: Short-only positions.
        max_long_positions: Maximum number of long positions that can be held
            at any time in :class:`pybroker.portfolio.Portfolio`. Unlimited
            when ``None``. Defaults to ``None``.
        max_short_positions: Maximum number of short positions that can be
            held at any time in :class:`pybroker.portfolio.Portfolio`.
            Unlimited when ``None``. Defaults to ``None``.
        worst_rank_held: Worst score rank at which a held position is kept.
            When set, the engine liquidates held positions ranked worse than
            this value and auto-generates equal-weight entry signals.
            Long rotation ranks :attr:`pybroker.context.ExecContext.long_score`
            and requires :attr:`.max_long_positions`. Short rotation ranks
            :attr:`pybroker.context.ExecContext.short_score` and requires
            :attr:`.max_short_positions`. Must be greater than or equal to the
            corresponding position limit when that side is enabled. When
            ``None``, the engine does not apply rotational hold-band logic.
            Defaults to ``None``.
        buy_delay: Number of bars before placing an order for a buy signal. The
            default value of ``1`` places a buy order on the next bar. Must be
            > ``0``.
        sell_delay: Number of bars before placing an order for a sell signal.
            The default value of ``1`` places a sell order on the next bar.
            Must be > ``0``.
        bootstrap_samples: Number of samples used to compute boostrap metrics.
            Defaults to ``10_000``.
        bootstrap_sample_size: Size of each random sample used to compute
            bootstrap metrics. Defaults to ``1_000``.
        exit_on_last_bar: Whether to automatically exit any open positions
            on the last bar of data available for a symbol. Defaults to
            ``False``.
        exit_cover_fill_price: Fill price for covering an open short position
            when :attr:`.exit_on_last_bar` is ``True``. Defaults to
            :attr:`pybroker.common.PriceType.MIDDLE`.
        exit_sell_fill_price: Fill price for selling an open long position when
            :attr:`.exit_on_last_bar` is ``True``. Defaults to
            :attr:`pybroker.common.PriceType.MIDDLE`.
        bars_per_year: Number of observations per year that will be used to
            annualize evaluation metrics. For example, a value of ``252`` would
            be used to annualize the Sharpe Ratio for daily returns.
        return_signals: When ``True``, then bar data, indicator data, and model
            predictions are returned with
            :class:`pybroker.strategy.TestResult`. Defaults to ``False``.
        return_stops: When ``True``, then stop values are returned with
            :class:`pybroker.strategy.TestResult`. Defaults to ``False``.
        round_test_result: When ``True``, round values in
            :class:`pybroker.strategy.TestResult` up to the nearest cent.
            Defaults to ``True``.
        leverage: Account leverage multiplier for buying power on long and
            short positions. Default ``1.0`` uses cash-only buying.
            ``2.0`` allows positions up to 2x equity. Must be ``>= 1.0``.
        interest_rate: Annual interest rate applied to net cash balance
            (``cash - margin_loan``). Charges interest when net cash is
            negative and credits interest when net cash is positive.
            Defaults to ``0`` (disabled).
        record_portfolio_bars: When ``True``, append full
            :class:`pybroker.portfolio.PortfolioBar` snapshots to
            :attr:`pybroker.portfolio.Portfolio.bars` on every bar. When
            ``False`` (default), per-bar metrics are stored in a compact
            buffer used for :class:`pybroker.strategy.TestResult` and
            :class:`pybroker.eval.EvalMetrics`.
        record_position_bars: When ``True``, append full
            :class:`pybroker.portfolio.PositionBar` snapshots to
            :attr:`pybroker.portfolio.Portfolio.position_bars` on every bar.
            When ``False`` (default), :attr:`pybroker.strategy.TestResult.positions`
            is empty.
    """

    initial_cash: float = field(default=100_000)
    fee_mode: Optional[Union[FeeMode, Callable[[FeeInfo], Decimal]]] = field(
        default=None
    )
    fee_amount: float = field(default=0)
    enable_fractional_shares: bool = field(default=False)
    round_fill_price: bool = field(default=True)
    position_mode: PositionMode = field(default=PositionMode.DEFAULT)
    max_long_positions: Optional[int] = field(default=None)
    max_short_positions: Optional[int] = field(default=None)
    worst_rank_held: Optional[int] = field(default=None)
    buy_delay: int = field(default=1)
    sell_delay: int = field(default=1)
    bootstrap_samples: int = field(default=10_000)
    bootstrap_sample_size: int = field(default=1_000)
    exit_on_last_bar: bool = field(default=False)
    exit_cover_fill_price: Union[
        PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
    ] = field(default=PriceType.MIDDLE)
    exit_sell_fill_price: Union[
        PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
    ] = field(default=PriceType.MIDDLE)
    bars_per_year: Optional[int] = field(default=None)
    return_signals: bool = field(default=False)
    return_stops: bool = field(default=False)
    round_test_result: bool = field(default=True)
    leverage: float = field(default=1.0)
    interest_rate: float = field(default=0.0)
    record_portfolio_bars: bool = field(default=False)
    record_position_bars: bool = field(default=False)
