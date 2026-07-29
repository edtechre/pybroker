"""Implements slippage models."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import math
import pandas as pd
import warnings
from abc import ABC
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, Mapping, Optional

from pybroker.common import DataCol, to_decimal
from pybroker.context import ExecContext
from pybroker.scope import ColumnScope, IndicatorScope

if TYPE_CHECKING:
    from pybroker.strategy import Strategy

_MAX_BPS = 10_000
_BPS_DIVISOR = Decimal(_MAX_BPS)
_DECIMAL_ONE = Decimal(1)
_DECIMAL_ZERO = Decimal(0)
#: Floor applied to adverse price adjustments, as a fraction of the base fill
#: price, so that a fill price can never reach zero or turn negative.
_MIN_PRICE_FACTOR = Decimal("0.01")


@dataclass(frozen=True)
class FillSlippageContext:
    """Context passed to fill-time slippage adjustments.

    Attributes:
        side: Order side, either ``buy`` or ``sell``.
        symbol: Ticker symbol of the order.
        shares: Number of shares to fill before slippage.
        fill_price: Base fill price resolved on the fill bar.
        col_scope: Column scope for bar data on the fill bar. ``None`` when
            bar data is unavailable, in which case volume-based models leave
            the fill unadjusted.
        ind_scope: Indicator scope for indicator data on the fill bar.
            ``None`` when indicator data is unavailable, in which case
            indicator-based models leave the fill unadjusted.
        sym_end_index: Current bar index per symbol, or ``None`` when
            unavailable.
        enable_fractional_shares: Whether fractional shares are enabled. When
            ``False``, a model that reduces ``shares`` must return a whole
            number.
    """

    side: Literal["buy", "sell"]
    symbol: str
    shares: Decimal
    fill_price: Decimal
    col_scope: Optional[ColumnScope]
    ind_scope: Optional[IndicatorScope]
    sym_end_index: Optional[Mapping[str, int]]
    enable_fractional_shares: bool = True


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
) -> Optional[float]:
    """Returns the last scalar value for ``name`` on the fill bar.

    Returns ``None`` when the column is missing or when the value is ``NaN``,
    so that callers can distinguish missing data from a genuine ``0``.
    """
    value = col_scope.fetch_value(symbol, name, end_index)
    if value is None or math.isnan(value):
        return None
    return value


class SlippageModel(ABC):
    """Base class for implementing a slippage model.

    A slippage model may adjust fill price, share quantity, or both. Signal-
    time adjustments are made in :meth:`apply_slippage`; fill-time
    adjustments (using fill-bar price, volume, and indicators) are made in
    :meth:`apply_at_fill`.

    :meth:`apply_at_fill` is called for scheduled buy and sell orders, for
    stop exits (stop loss, take profit, trailing, and bar stops), and
    for :meth:`pybroker.portfolio.Portfolio.exit_position` fills. Stop and
    exit fills apply the returned **fill price only**; a returned share
    quantity is ignored, since those paths exit an entry in full.
    """

    @property
    def is_fill_noop(self) -> bool:
        """Whether :meth:`apply_at_fill` is a no-op for this model."""
        return type(self).apply_at_fill is SlippageModel.apply_at_fill

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

    def adjust_fill(
        self,
        side: Literal["buy", "sell"],
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        col_scope: Optional[ColumnScope] = None,
        ind_scope: Optional[IndicatorScope] = None,
        sym_end_index: Optional[Mapping[str, int]] = None,
        enable_fractional_shares: bool = True,
    ) -> tuple[Decimal, Decimal]:
        """Builds a :class:`.FillSlippageContext` and applies
        :meth:`apply_at_fill` to it.

        Provided so that callers which cannot import :mod:`pybroker.slippage`
        (such as :mod:`pybroker.portfolio`) can still apply fill-time
        slippage.

        Returns:
            Tuple of ``(shares, fill_price)`` after slippage.
        """
        if self.is_fill_noop:
            return shares, fill_price
        return self.apply_at_fill(
            FillSlippageContext(
                side=side,
                symbol=symbol,
                shares=shares,
                fill_price=fill_price,
                col_scope=col_scope,
                ind_scope=ind_scope,
                sym_end_index=sym_end_index,
                enable_fractional_shares=enable_fractional_shares,
            )
        )

    def validate(self, strategy: "Strategy") -> None:
        """Validates model configuration before a backtest starts."""


class FixedSlippageModel(SlippageModel):
    """Deterministic fixed-basis-point slippage on fill price.

    Buy fills are worsened upward; sell fills are worsened downward. Applies
    to long entries, short entries, long exits, and covers.

    Args:
        bps: Adverse slippage in basis points. ``0`` is a no-op. Must be less
            than ``10000`` (100%), which would make sell fills non-positive.
    """

    def __init__(self, bps: float = 5):
        if bps < 0:
            raise ValueError("bps must be >= 0.")
        if bps >= _MAX_BPS:
            raise ValueError(
                f"bps must be < {_MAX_BPS}, otherwise sell fill prices would "
                "be <= 0."
            )
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

    Adverse price adjustment equals ``scale * ATR`` at the fill bar. Bars
    where the ATR is ``NaN`` (such as the indicator's warmup period) are left
    unadjusted.

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
        self._clamped_symbols: set[str] = set()

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
        if fill_ctx.ind_scope is None or fill_ctx.sym_end_index is None:
            return fill_ctx.shares, fill_ctx.fill_price
        end_index = fill_ctx.sym_end_index[fill_ctx.symbol]
        atr_value = fill_ctx.ind_scope.fetch_value(
            fill_ctx.symbol, self.atr_indicator, end_index
        )
        if atr_value is None or math.isnan(atr_value):
            return fill_ctx.shares, fill_ctx.fill_price
        adjustment = self._scale * to_decimal(atr_value)
        price = _adverse_price(fill_ctx.side, fill_ctx.fill_price, adjustment)
        floor = fill_ctx.fill_price * _MIN_PRICE_FACTOR
        if price < floor:
            self._warn_clamped(fill_ctx.symbol)
            price = floor
        return fill_ctx.shares, price

    def _warn_clamped(self, symbol: str):
        if symbol in self._clamped_symbols:
            return
        self._clamped_symbols.add(symbol)
        warnings.warn(
            f"{type(self).__name__}: {self.atr_indicator!r} slippage for "
            f"{symbol!r} exceeded the fill price and was clamped to "
            f"{_MIN_PRICE_FACTOR:%} of it. Consider lowering scale.",
            stacklevel=2,
        )


class VolumeSlippageModel(SlippageModel):
    """Volume-based participation cap and square-law price impact.

    Share quantity may be capped at ``volume_limit * bar_volume``. Price
    impact is ``price_impact * (filled_shares / bar_volume) ** 2``. Either
    effect can be disabled by passing ``0`` or ``None``.

    Requires a ``volume`` data column, which is optional in PyBroker. Bars
    where the volume is missing or ``NaN`` are left unadjusted.

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
        self._warned_symbols: set[str] = set()

    @property
    def is_fill_noop(self) -> bool:
        return not self._cap_enabled and not self._impact_enabled

    def validate(self, strategy: "Strategy") -> None:
        if self.is_fill_noop:
            return
        data_source = getattr(strategy, "_data_source", None)
        if (
            isinstance(data_source, pd.DataFrame)
            and DataCol.VOLUME.value not in data_source.columns
        ):
            raise ValueError(
                f"{type(self).__name__} requires a "
                f"{DataCol.VOLUME.value!r} data column, which is missing from "
                "the backtesting data."
            )

    def _warn_missing_volume(self, symbol: str):
        if symbol in self._warned_symbols:
            return
        self._warned_symbols.add(symbol)
        warnings.warn(
            f"{type(self).__name__}: missing or NaN "
            f"{DataCol.VOLUME.value!r} for {symbol!r}; leaving fills "
            "unadjusted on those bars.",
            stacklevel=2,
        )

    def apply_at_fill(
        self, fill_ctx: FillSlippageContext
    ) -> tuple[Decimal, Decimal]:
        if self.is_fill_noop:
            return fill_ctx.shares, fill_ctx.fill_price
        if fill_ctx.col_scope is None or fill_ctx.sym_end_index is None:
            return fill_ctx.shares, fill_ctx.fill_price

        volume = _fetch_scalar(
            fill_ctx.col_scope,
            fill_ctx.symbol,
            DataCol.VOLUME.value,
            fill_ctx.sym_end_index[fill_ctx.symbol],
        )
        if volume is None:
            self._warn_missing_volume(fill_ctx.symbol)
            return fill_ctx.shares, fill_ctx.fill_price
        if volume <= 0:
            # No volume means no participation to cap against. Only the
            # participation cap can refuse the fill; price impact alone is
            # undefined here and leaves the order untouched.
            if self._cap_enabled:
                return _DECIMAL_ZERO, fill_ctx.fill_price
            return fill_ctx.shares, fill_ctx.fill_price

        shares = fill_ctx.shares
        if self._cap_enabled:
            assert self._volume_limit_dec is not None
            max_shares = self._volume_limit_dec * to_decimal(volume)
            if max_shares < shares:
                shares = max_shares
                if not fill_ctx.enable_fractional_shares:
                    shares = to_decimal(int(shares))

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
