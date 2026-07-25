"""Unit tests for slippage.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import pytest
import re
from decimal import Decimal
from unittest.mock import MagicMock

from pybroker.common import to_decimal

from pybroker.slippage import (
    FillSlippageContext,
    FixedSlippageModel,
    SlippageModel,
    VolatilitySlippageModel,
    VolumeSlippageModel,
)


def _fill_ctx(
    side,
    shares=Decimal(100),
    fill_price=Decimal(100),
    volume=1_000_000.0,
    atr=2.5,
):
    col_scope = MagicMock()
    col_scope.fetch_value.return_value = volume
    ind_scope = MagicMock()
    ind_scope.fetch_value.return_value = atr
    return FillSlippageContext(
        side=side,
        symbol="SPY",
        shares=shares,
        fill_price=fill_price,
        col_scope=col_scope,
        ind_scope=ind_scope,
        sym_end_index={"SPY": 1},
    )


class TestFixedSlippageModel:
    def test_init_when_negative_bps_then_error(self):
        with pytest.raises(ValueError, match=re.escape("bps must be >= 0.")):
            FixedSlippageModel(bps=-1)

    def test_buy_adverse(self):
        model = FixedSlippageModel(bps=5)
        shares, price = model.apply_at_fill(_fill_ctx("buy"))
        assert shares == Decimal(100)
        assert price == Decimal(100) * Decimal("1.0005")

    def test_sell_adverse(self):
        model = FixedSlippageModel(bps=5)
        shares, price = model.apply_at_fill(_fill_ctx("sell"))
        assert shares == Decimal(100)
        assert price == Decimal(100) * Decimal("0.9995")

    def test_bps_zero_is_noop(self):
        model = FixedSlippageModel(bps=0)
        ctx = _fill_ctx("buy")
        assert model.is_fill_noop
        assert model.apply_at_fill(ctx) == (ctx.shares, ctx.fill_price)
        assert model.adjust_fill_price("buy", ctx.fill_price) == ctx.fill_price

    def test_adjust_fill_price_matches_apply_at_fill(self):
        model = FixedSlippageModel(bps=5)
        ctx = _fill_ctx("buy")
        _, price = model.apply_at_fill(ctx)
        assert price == model.adjust_fill_price("buy", ctx.fill_price)

    def test_uses_signal_slippage_false(self):
        assert not FixedSlippageModel(bps=5).uses_signal_slippage

    def test_deterministic(self):
        model = FixedSlippageModel(bps=5)
        ctx = _fill_ctx("buy")
        assert model.apply_at_fill(ctx) == model.apply_at_fill(ctx)


class TestVolatilitySlippageModel:
    def test_init_when_negative_scale_then_error(self):
        with pytest.raises(ValueError, match=re.escape("scale must be >= 0.")):
            VolatilitySlippageModel(atr_indicator="atr_14", scale=-0.1)

    def test_reads_fill_bar_atr(self):
        model = VolatilitySlippageModel(atr_indicator="atr_14", scale=0.1)
        ctx = _fill_ctx("buy", atr=2.5)
        shares, price = model.apply_at_fill(ctx)
        assert shares == Decimal(100)
        assert price == Decimal("100.25")
        ctx.ind_scope.fetch_value.assert_called_once_with("SPY", "atr_14", 1)

    def test_sell_adverse(self):
        model = VolatilitySlippageModel(atr_indicator="atr_14", scale=0.1)
        shares, price = model.apply_at_fill(_fill_ctx("sell", atr=2.5))
        assert price == Decimal("99.75")

    def test_scale_zero_is_fill_noop(self):
        model = VolatilitySlippageModel(atr_indicator="atr_14", scale=0)
        assert model.is_fill_noop

    def test_validate_when_indicator_missing(self):
        strategy = MagicMock()
        execution = MagicMock()
        execution.indicator_names = frozenset({"sma_20"})
        strategy._executions = {execution}
        model = VolatilitySlippageModel(atr_indicator="atr_14")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Indicator 'atr_14' must be attached to an execution"
            ),
        ):
            model.validate(strategy)

    def test_validate_when_indicator_attached(self):
        strategy = MagicMock()
        execution = MagicMock()
        execution.indicator_names = frozenset({"atr_14"})
        strategy._executions = {execution}
        VolatilitySlippageModel(atr_indicator="atr_14").validate(strategy)


class TestVolumeSlippageModel:
    def test_init_when_invalid_params_then_error(self):
        with pytest.raises(
            ValueError, match=re.escape("price_impact must be >= 0.")
        ):
            VolumeSlippageModel(price_impact=-0.1)
        with pytest.raises(
            ValueError, match=re.escape("volume_limit must be >= 0.")
        ):
            VolumeSlippageModel(volume_limit=-0.1)

    def test_zero_volume_no_fill(self):
        model = VolumeSlippageModel()
        shares, _ = model.apply_at_fill(_fill_ctx("buy", volume=0))
        assert shares == Decimal(0)

    def test_cap_before_impact_on_capped_quantity(self):
        model = VolumeSlippageModel(price_impact=0.1, volume_limit=0.025)
        ctx = _fill_ctx("buy", shares=Decimal(100_000), volume=1_000_000)
        shares, price = model.apply_at_fill(ctx)
        assert shares == Decimal(25000)
        ratio = 25000 / 1_000_000
        expected_impact = 0.1 * ratio * ratio
        assert price == Decimal(100) * (
            Decimal(1) + to_decimal(expected_impact)
        )

    def test_volume_limit_none_disables_cap(self):
        model = VolumeSlippageModel(price_impact=0, volume_limit=None)
        ctx = _fill_ctx("buy", shares=Decimal(100_000), volume=1_000)
        shares, price = model.apply_at_fill(ctx)
        assert shares == Decimal(100_000)
        assert price == Decimal(100)

    def test_price_impact_zero_disables_impact(self):
        model = VolumeSlippageModel(price_impact=0, volume_limit=0.025)
        ctx = _fill_ctx("buy", shares=Decimal(100_000), volume=1_000_000)
        shares, price = model.apply_at_fill(ctx)
        assert shares == Decimal(25000)
        assert price == Decimal(100)

    def test_both_disabled_is_fill_noop(self):
        model = VolumeSlippageModel(price_impact=0, volume_limit=None)
        assert model.is_fill_noop
        ctx = _fill_ctx("buy")
        assert model.apply_at_fill(ctx) == (ctx.shares, ctx.fill_price)

    def test_uses_signal_slippage_when_overridden(self):
        class CustomSlippage(SlippageModel):
            def apply_slippage(self, ctx, buy_shares=None, sell_shares=None):
                if buy_shares:
                    ctx.buy_shares = buy_shares - 1

        assert CustomSlippage().uses_signal_slippage

    def test_deterministic(self):
        model = VolumeSlippageModel()
        ctx = _fill_ctx("buy")
        assert model.apply_at_fill(ctx) == model.apply_at_fill(ctx)
