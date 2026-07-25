"""Implements slippage models."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

from abc import ABC
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, Mapping, Optional

from pybroker.common import DataCol, to_decimal
from pybroker.context import ExecContext
from pybroker.scope import ColumnScope, IndicatorScope

if TYPE_CHECKING:
    from pybroker.strategy import Strategy

_BPS_DIVISOR = Decimal(10_000)
_DECIMAL_ONE = Decimal(1)
_DECIMAL_ZERO = Decimal(0)


@dataclass(frozen=True)
class FillSlippageContext:
    """Context passed to fill-time slippage adjustments.

    Attributes:
        side: Order side, either ``buy`` or ``sell``.
        symbol: Ticker symbol of the order.
        shares: Number of shares to fill before slippage.
        fill_price: Base fill price resolved on the fill bar.
        col_scope: Column scope for bar data on the fill bar.
        ind_scope: Indicator scope for indicator data on the fill bar.
        sym_end_index: Current bar index per symbol.
    """

    side: Literal["buy", "sell"]
    symbol: str
    shares: Decimal
    fill_price: Decimal
    col_scope: ColumnScope
    ind_scope: IndicatorScope
    sym_end_index: Mapping[str, int]


def _adverse_price(
    side: Literal["buy", "sell"], base: Decimal, amount: Decimal
) -> Decimal:
    """Returns ``base`` adjusted adversely by ``amount`` for ``side``."""
    if side == "buy":
        return base + amount
    return base - amount


def _fetch_scalar(
    col_scope: ColumnScope,
    symbol: str,
    name: str,
    end_index: int,
) -> float:
    """Returns the last scalar value for ``name`` on the fill bar."""
    value = col_scope.fetch_value(symbol, name, end_index)
    if value is None:
        return 0.0
    return value


class SlippageModel(ABC):
    """Base class for implementing a slippage model.

    A slippage model may adjust fill price, share quantity, or both. Signal-
    time adjustments are made in :meth:`apply_slippage`; fill-time
    adjustments (using fill-bar price, volume, and indicators) are made in
    :meth:`apply_at_fill`.
    """

    @property
    def is_fill_noop(self) -> bool:
        """Whether :meth:`apply_at_fill` is a no-op for this model."""
        return False

    @property
    def uses_signal_slippage(self) -> bool:
        """Whether :meth:`apply_slippage` was overridden by a subclass."""
        return type(self).apply_slippage is not SlippageModel.apply_slippage

    def apply_slippage(
        self,
        ctx: ExecContext,
        buy_shares: Optional[Decimal] = None,
        sell_shares: Optional[Decimal] = None,
    ):
        """Applies signal-time slippage to ``ctx``.

        May adjust ``ctx.buy_shares``, ``ctx.sell_shares``,
        ``ctx.buy_fill_price``, and/or ``ctx.sell_fill_price``.
        """

    def apply_at_fill(
        self, fill_ctx: FillSlippageContext
    ) -> tuple[Decimal, Decimal]:
        """Applies fill-time slippage using data from the fill bar.

        Returns:
            Tuple of ``(shares, fill_price)`` after slippage.
        """
        return fill_ctx.shares, fill_ctx.fill_price

    def validate(self, strategy: "Strategy") -> None:
        """Validates model configuration before a backtest starts."""


class FixedSlippageModel(SlippageModel):
    """Deterministic fixed-basis-point slippage on fill price.

    Buy fills are worsened upward; sell fills are worsened downward. Applies
    to long entries, short entries, long exits, and covers.

    Args:
        bps: Adverse slippage in basis points. ``0`` is a no-op.
    """

    def __init__(self, bps: float = 5):
        if bps < 0:
            raise ValueError("bps must be >= 0.")
        self._bps = bps
        if bps == 0:
            self._buy_multiplier = _DECIMAL_ONE
            self._sell_multiplier = _DECIMAL_ONE
        else:
            factor = Decimal(str(bps)) / _BPS_DIVISOR
            self._buy_multiplier = _DECIMAL_ONE + factor
            self._sell_multiplier = _DECIMAL_ONE - factor

    @property
    def is_fill_noop(self) -> bool:
        return self._bps == 0

    def adjust_fill_price(
        self, side: Literal["buy", "sell"], fill_price: Decimal
    ) -> Decimal:
        """Returns ``fill_price`` adjusted for ``side`` without extra context."""
        if self._bps == 0:
            return fill_price
        if side == "buy":
            return fill_price * self._buy_multiplier
        return fill_price * self._sell_multiplier

    def apply_at_fill(
        self, fill_ctx: FillSlippageContext
    ) -> tuple[Decimal, Decimal]:
        return (
            fill_ctx.shares,
            self.adjust_fill_price(fill_ctx.side, fill_ctx.fill_price),
        )


class VolatilitySlippageModel(SlippageModel):
    """ATR-scaled slippage on fill price.

    Adverse price adjustment equals ``scale * ATR`` at the fill bar.

    Args:
        atr_indicator: Name of the ATR indicator attached to an execution.
        scale: Multiplier applied to the ATR value.
    """

    def __init__(self, atr_indicator: str, scale: float = 0.1):
        if scale < 0:
            raise ValueError("scale must be >= 0.")
        self.atr_indicator = atr_indicator
        self.scale = scale
        self._scale = Decimal(str(scale))

    @property
    def is_fill_noop(self) -> bool:
        return self.scale == 0

    def validate(self, strategy: "Strategy") -> None:
        ind_names: set[str] = set()
        for execution in strategy._executions:
            ind_names.update(execution.indicator_names)
        if self.atr_indicator not in ind_names:
            raise ValueError(
                f"Indicator {self.atr_indicator!r} must be attached to an "
                "execution via add_execution(..., indicators=[...])."
            )

    def apply_at_fill(
        self, fill_ctx: FillSlippageContext
    ) -> tuple[Decimal, Decimal]:
        if self.scale == 0:
            return fill_ctx.shares, fill_ctx.fill_price
        end_index = fill_ctx.sym_end_index[fill_ctx.symbol]
        atr_value = fill_ctx.ind_scope.fetch_value(
            fill_ctx.symbol, self.atr_indicator, end_index
        )
        adjustment = self._scale * to_decimal(atr_value)
        price = _adverse_price(fill_ctx.side, fill_ctx.fill_price, adjustment)
        return fill_ctx.shares, price


class VolumeSlippageModel(SlippageModel):
    """Volume-based participation cap and square-law price impact.

    Share quantity may be capped at ``volume_limit * bar_volume``. Price
    impact is ``price_impact * (filled_shares / bar_volume) ** 2``. Either
    effect can be disabled by passing ``0`` or ``None``.

    Args:
        price_impact: Square-law impact coefficient. ``0`` disables impact.
        volume_limit: Max participation as a fraction of bar volume.
            ``None`` or ``0`` disables the cap.
    """

    def __init__(
        self,
        price_impact: float = 0.1,
        volume_limit: Optional[float] = 0.025,
    ):
        if price_impact < 0:
            raise ValueError("price_impact must be >= 0.")
        if volume_limit is not None and volume_limit < 0:
            raise ValueError("volume_limit must be >= 0.")
        self.price_impact = price_impact
        self.volume_limit = volume_limit
        self._cap_enabled = volume_limit is not None and volume_limit > 0
        self._impact_enabled = price_impact > 0
        self._price_impact = price_impact
        self._volume_limit_dec = (
            Decimal(str(volume_limit)) if self._cap_enabled else None
        )

    @property
    def is_fill_noop(self) -> bool:
        return not self._cap_enabled and not self._impact_enabled

    def apply_at_fill(
        self, fill_ctx: FillSlippageContext
    ) -> tuple[Decimal, Decimal]:
        if self.is_fill_noop:
            return fill_ctx.shares, fill_ctx.fill_price

        volume = _fetch_scalar(
            fill_ctx.col_scope,
            fill_ctx.symbol,
            DataCol.VOLUME.value,
            fill_ctx.sym_end_index[fill_ctx.symbol],
        )
        if volume <= 0:
            return _DECIMAL_ZERO, fill_ctx.fill_price

        shares = fill_ctx.shares
        if self._cap_enabled:
            assert self._volume_limit_dec is not None
            max_shares = self._volume_limit_dec * to_decimal(volume)
            shares = min(shares, max_shares)

        price = fill_ctx.fill_price
        if self._impact_enabled and shares > _DECIMAL_ZERO:
            ratio = float(shares) / volume
            impact = self._price_impact * ratio * ratio
            impact_dec = to_decimal(impact)
            if fill_ctx.side == "buy":
                price = fill_ctx.fill_price * (_DECIMAL_ONE + impact_dec)
            else:
                price = fill_ctx.fill_price * (_DECIMAL_ONE - impact_dec)

        return shares, price
