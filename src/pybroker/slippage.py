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
from .context import ExecContext
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import NamedTuple, Optional


class SlippageData(NamedTuple):
    """Contains data to use for calculating slippage."""

    buy_shares: Optional[Decimal]
    sell_shares: Optional[Decimal]
    buy_fill_price: Decimal
    sell_fill_price: Decimal


class SlippageModel(ABC):
    """Base class for implementing a slippage model."""

    @abstractmethod
    def apply_slippage(self, data: SlippageData, ctx: ExecContext):
        """Applies slippage to ```ctx```."""


class RandomSlippageModel(SlippageModel):
    """Implements a simple random slippage model.

    Args:
        price_pct: ```tuple[min, max]``` percentages of price slippage.
        shares_pct: ```tuple[min, max]``` percentages of share slippage.
    """

    def __init__(
        self,
        price_pct: Optional[tuple[float, float]] = None,
        shares_pct: Optional[tuple[float, float]] = None,
    ):
        if not price_pct and not shares_pct:
            raise ValueError("Must pass either price_pct or shares_pct.")
        if price_pct is not None:
            if len(price_pct) != 2:
                raise ValueError("price_pct must be a tuple[min, max].")
            price_pct = (price_pct[0] / 100.0, price_pct[1] / 100.0)
        if shares_pct is not None:
            if len(shares_pct) != 2:
                raise ValueError("shares_pct must be a tuple[min, max].")
            shares_pct = (shares_pct[0] / 100.0, shares_pct[1] / 100.0)
        self.price_pct = price_pct
        self.shares_pct = shares_pct

    def apply_slippage(self, data: SlippageData, ctx: ExecContext):
        if self.shares_pct and (data.buy_shares or data.sell_shares):
            slippage_pct = Decimal(random.uniform(*self.shares_pct))
            if data.buy_shares:
                ctx.buy_shares = (
                    data.buy_shares - slippage_pct * data.buy_shares
                )
            if data.sell_shares:
                ctx.sell_shares = (
                    data.sell_shares - slippage_pct * data.sell_shares
                )
        if self.price_pct:
            slippage_pct = Decimal(random.uniform(*self.price_pct))
            ctx.buy_fill_price = (
                data.buy_fill_price + data.buy_fill_price * slippage_pct
            )
            ctx.sell_fill_price = (
                data.sell_fill_price - data.sell_fill_price * slippage_pct
            )
