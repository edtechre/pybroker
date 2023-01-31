"""Contains configuration classes."""

"""Copyright (C) 2023 Edward West

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .common import FeeMode
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for :class:`pybroker.strategy.Strategy`.

    Attributes:
        initial_cash: Starting cash of strategy.
        fee_mode: :class:`pybroker.common.FeeMode` for calculating brokerage
            fees, disabled when ``None``. Defaults to ``None``.
        fee_amount: Brokerage fee amount.
        max_long_positions: Maximum number of long positions that can be held
            at any time in :class:`pybroker.portfolio.Portfolio`. Unlimited
            when ``None``. Defaults to ``None``.
        max_short_positions: Maximum number of short positions that can be
            held at any time in :class:`pybroker.portfolio.Portfolio`.
            Unlimited when ``None``. Defaults to ``None``.
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
    """

    initial_cash: float = field(default=100_000)
    fee_mode: Optional[FeeMode] = field(default=None)
    fee_amount: float = field(default=0)
    max_long_positions: Optional[int] = field(default=None)
    max_short_positions: Optional[int] = field(default=None)
    buy_delay: int = field(default=1)
    sell_delay: int = field(default=1)
    bootstrap_samples: int = field(default=10_000)
    bootstrap_sample_size: int = field(default=1_000)
