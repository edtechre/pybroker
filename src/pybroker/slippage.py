"""Implements slippage models."""

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

import random
from pybroker.context import ExecContext
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional


class SlippageModel(ABC):
    """Base class for implementing a slippage model."""

    @abstractmethod
    def apply_slippage(
        self,
        ctx: ExecContext,
        buy_shares: Optional[Decimal] = None,
        sell_shares: Optional[Decimal] = None,
    ):
        """Applies slippage to ``ctx``."""


class RandomSlippageModel(SlippageModel):
    """Implements a simple random slippage model.

    Args:
        min_pct: Min percentage of slippage.
        max_pct: Max percentage of slippage.
    """

    def __init__(self, min_pct: float, max_pct: float):
        if min_pct < 0 or min_pct > 100:
            raise ValueError(r"min_pct must be between 0% and 100%.")
        if max_pct < 0 or max_pct > 100:
            raise ValueError(r"max_pct must be between 0% and 100%.")
        if min_pct >= max_pct:
            raise ValueError("min_pct must be < max_pct.")
        self.min_pct = min_pct / 100.0
        self.max_pct = max_pct / 100.0

    def apply_slippage(
        self,
        ctx: ExecContext,
        buy_shares: Optional[Decimal] = None,
        sell_shares: Optional[Decimal] = None,
    ):
        if buy_shares or sell_shares:
            slippage_pct = Decimal(random.uniform(self.min_pct, self.max_pct))
            if buy_shares:
                ctx.buy_shares = buy_shares - slippage_pct * buy_shares
            if sell_shares:
                ctx.sell_shares = sell_shares - slippage_pct * sell_shares
