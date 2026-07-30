"""Unit tests for slippage.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import pandas as pd
import pytest
import re
import warnings
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

NAN = float("nan")


def _fill_ctx(
    side,
    shares=Decimal(100),
    fill_price=Decimal(100),
    volume=1_000_000.0,
    atr=2.5,
    enable_fractional_shares=True,
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
        enable_fractional_shares=enable_fractional_shares,
    )


def _strategy_with_data(df):
    strategy = MagicMock()
    strategy._data_source = df
    return strategy


class TestFixedSlippageModel:
    def test_init_when_negative_bps_then_error(self):
        with pytest.raises(ValueError, match=re.escape("bps must be >= 0.")):
            FixedSlippageModel(bps=-1)

    @pytest.mark.parametrize("bps", [10_000, 20_000])
    def test_init_when_bps_at_or_above_max_then_error(self, bps):
        with pytest.raises(ValueError, match=re.escape("bps must be < 10000")):
            FixedSlippageModel(bps=bps)

    def test_sell_price_stays_positive_at_max_bps(self):
        _, price = FixedSlippageModel(bps=9_999).apply_at_fill(
            _fill_ctx("sell")
        )
        assert price > 0

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
        execution.symbols = frozenset({"SPY"})
        strategy._executions = {execution}
        model = VolatilitySlippageModel(atr_indicator="atr_14")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Indicator 'atr_14' must be attached to every execution"
            ),
        ):
            model.validate(strategy)

    def test_validate_when_indicator_missing_from_one_execution(self):
        """The union is not enough: this model prices fills for every
        execution, so one without the indicator raises at its first fill."""
        strategy = MagicMock()
        with_ind = MagicMock()
        with_ind.indicator_names = frozenset({"atr_14"})
        with_ind.symbols = frozenset({"SPY"})
        without_ind = MagicMock()
        without_ind.indicator_names = frozenset()
        without_ind.symbols = frozenset({"AAPL"})
        strategy._executions = {with_ind, without_ind}
        model = VolatilitySlippageModel(atr_indicator="atr_14")
        with pytest.raises(ValueError, match=re.escape("['AAPL']")):
            model.validate(strategy)

    def test_validate_when_indicator_attached(self):
        strategy = MagicMock()
        execution = MagicMock()
        execution.indicator_names = frozenset({"atr_14"})
        strategy._executions = {execution}
        VolatilitySlippageModel(atr_indicator="atr_14").validate(strategy)

    @pytest.mark.parametrize("side", ["buy", "sell"])
    def test_when_atr_nan_then_unadjusted(self, side):
        model = VolatilitySlippageModel(atr_indicator="atr_14")
        ctx = _fill_ctx(side, atr=NAN)
        assert model.apply_at_fill(ctx) == (ctx.shares, ctx.fill_price)

    def test_when_no_ind_scope_then_unadjusted(self):
        model = VolatilitySlippageModel(atr_indicator="atr_14")
        ctx = _fill_ctx("buy")
        ctx = FillSlippageContext(
            side=ctx.side,
            symbol=ctx.symbol,
            shares=ctx.shares,
            fill_price=ctx.fill_price,
            col_scope=ctx.col_scope,
            ind_scope=None,
            sym_end_index=ctx.sym_end_index,
        )
        assert model.apply_at_fill(ctx) == (ctx.shares, ctx.fill_price)

    def test_when_atr_exceeds_price_then_clamped_positive(self):
        model = VolatilitySlippageModel(atr_indicator="atr_14", scale=1)
        with pytest.warns(UserWarning, match="exceeded the fill price"):
            _, price = model.apply_at_fill(
                _fill_ctx("sell", fill_price=Decimal(100), atr=500)
            )
        assert price > 0
        assert price == Decimal(100) * Decimal("0.01")

    def test_clamp_warns_once_per_symbol(self):
        model = VolatilitySlippageModel(atr_indicator="atr_14", scale=1)
        ctx = _fill_ctx("sell", atr=500)
        with pytest.warns(UserWarning):
            model.apply_at_fill(ctx)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.apply_at_fill(ctx)


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

    def test_zero_volume_when_cap_disabled_then_unadjusted(self):
        # Cancelling the order is participation-cap behavior. With the cap
        # off, a zero-volume bar must not silently drop the order.
        model = VolumeSlippageModel(price_impact=0.1, volume_limit=None)
        ctx = _fill_ctx("buy", volume=0)
        assert model.apply_at_fill(ctx) == (ctx.shares, ctx.fill_price)

    @pytest.mark.parametrize("volume", [NAN, None])
    def test_when_volume_missing_then_unadjusted_and_warns(self, volume):
        model = VolumeSlippageModel()
        ctx = _fill_ctx("buy", volume=volume)
        with pytest.warns(UserWarning, match="missing or NaN 'volume'"):
            assert model.apply_at_fill(ctx) == (ctx.shares, ctx.fill_price)

    def test_missing_volume_warns_once_per_symbol(self):
        model = VolumeSlippageModel()
        ctx = _fill_ctx("buy", volume=NAN)
        with pytest.warns(UserWarning):
            model.apply_at_fill(ctx)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.apply_at_fill(ctx)

    def test_when_impact_exceeds_price_then_clamped_and_warns(self):
        # Square-law impact is unbounded above, so an order large relative to
        # bar volume drives the sell price through zero and pays the account
        # to sell. Slippage may only ever worsen a fill.
        model = VolumeSlippageModel(price_impact=0.1, volume_limit=None)
        ctx = _fill_ctx("sell", shares=Decimal(1000), volume=100.0)
        with pytest.warns(UserWarning, match="exceeded the fill price"):
            _, price = model.apply_at_fill(ctx)
        assert price > 0
        assert price == ctx.fill_price * Decimal("0.01")

    def test_impact_clamp_warns_once_per_symbol(self):
        model = VolumeSlippageModel(price_impact=0.1, volume_limit=None)
        ctx = _fill_ctx("sell", shares=Decimal(1000), volume=100.0)
        with pytest.warns(UserWarning):
            model.apply_at_fill(ctx)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.apply_at_fill(ctx)

    def test_when_no_col_scope_then_unadjusted(self):
        model = VolumeSlippageModel()
        base = _fill_ctx("buy")
        ctx = FillSlippageContext(
            side=base.side,
            symbol=base.symbol,
            shares=base.shares,
            fill_price=base.fill_price,
            col_scope=None,
            ind_scope=base.ind_scope,
            sym_end_index=base.sym_end_index,
        )
        assert model.apply_at_fill(ctx) == (ctx.shares, ctx.fill_price)

    def test_validate_when_volume_column_missing(self):
        df = pd.DataFrame(
            {"symbol": ["SPY"], "date": ["2020-01-01"], "close": [100.0]}
        )
        model = VolumeSlippageModel()
        with pytest.raises(
            ValueError,
            match=re.escape("requires a 'volume' data column"),
        ):
            model.validate(_strategy_with_data(df))

    def test_validate_when_volume_column_present(self):
        df = pd.DataFrame(
            {
                "symbol": ["SPY"],
                "date": ["2020-01-01"],
                "close": [100.0],
                "volume": [1_000.0],
            }
        )
        VolumeSlippageModel().validate(_strategy_with_data(df))

    def test_validate_when_fill_noop_then_volume_not_required(self):
        df = pd.DataFrame({"symbol": ["SPY"], "close": [100.0]})
        model = VolumeSlippageModel(price_impact=0, volume_limit=None)
        model.validate(_strategy_with_data(df))

    def test_cap_rounds_down_when_fractional_shares_disabled(self):
        model = VolumeSlippageModel(price_impact=0, volume_limit=0.025)
        ctx = _fill_ctx(
            "buy",
            shares=Decimal(100_000),
            volume=1_000_001,
            enable_fractional_shares=False,
        )
        shares, _ = model.apply_at_fill(ctx)
        assert shares == Decimal(25_000)
        assert shares == shares.to_integral_value()

    def test_cap_keeps_fraction_when_fractional_shares_enabled(self):
        model = VolumeSlippageModel(price_impact=0, volume_limit=0.025)
        shares, _ = model.apply_at_fill(
            _fill_ctx(
                "buy",
                shares=Decimal(100_000),
                volume=1_000_001,
                enable_fractional_shares=True,
            )
        )
        assert shares == Decimal("25000.025")

    def test_when_shares_below_cap_then_not_increased(self):
        model = VolumeSlippageModel(price_impact=0, volume_limit=0.025)
        shares, _ = model.apply_at_fill(
            _fill_ctx("buy", shares=Decimal(10), volume=1_000_000)
        )
        assert shares == Decimal(10)

    def test_sell_price_impact_is_adverse(self):
        model = VolumeSlippageModel(price_impact=0.1, volume_limit=0.025)
        shares, price = model.apply_at_fill(
            _fill_ctx("sell", shares=Decimal(100_000), volume=1_000_000)
        )
        assert shares == Decimal(25_000)
        ratio = 25_000 / 1_000_000
        expected_impact = 0.1 * ratio * ratio
        assert price == Decimal(100) * (
            Decimal(1) - to_decimal(expected_impact)
        )
        assert price < Decimal(100)

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


class TestSlippageModel:
    def test_base_model_is_fill_noop(self):
        assert SlippageModel().is_fill_noop

    def test_signal_only_model_is_fill_noop(self):
        class SignalOnly(SlippageModel):
            def apply_slippage(self, ctx, buy_shares=None, sell_shares=None):
                pass

        model = SignalOnly()
        assert model.is_fill_noop
        assert model.uses_signal_slippage

    def test_fill_model_is_not_fill_noop(self):
        class FillOnly(SlippageModel):
            def apply_at_fill(self, fill_ctx):
                return fill_ctx.shares, fill_ctx.fill_price * Decimal(2)

        model = FillOnly()
        assert not model.is_fill_noop
        assert not model.uses_signal_slippage

    def test_adjust_fill_matches_apply_at_fill(self):
        model = FixedSlippageModel(bps=5)
        ctx = _fill_ctx("buy")
        assert model.adjust_fill(
            side="buy",
            symbol="SPY",
            shares=ctx.shares,
            fill_price=ctx.fill_price,
            col_scope=ctx.col_scope,
            ind_scope=ctx.ind_scope,
            sym_end_index=ctx.sym_end_index,
        ) == model.apply_at_fill(ctx)

    def test_adjust_fill_when_noop_then_unchanged(self):
        model = SlippageModel()
        assert model.adjust_fill(
            side="sell",
            symbol="SPY",
            shares=Decimal(100),
            fill_price=Decimal(100),
        ) == (Decimal(100), Decimal(100))

    def test_adjust_fill_without_scopes(self):
        # Portfolio stop exits may have no scopes available; scope-dependent
        # models must degrade to a no-op instead of raising.
        for model in (
            VolumeSlippageModel(),
            VolatilitySlippageModel(atr_indicator="atr_14"),
        ):
            assert model.adjust_fill(
                side="sell",
                symbol="SPY",
                shares=Decimal(100),
                fill_price=Decimal(100),
            ) == (Decimal(100), Decimal(100))


def test_volatility_validate_ignores_executions_without_a_function():
    """An execution with no function places no orders, so it never reaches
    apply_at_fill and must not be required to attach the ATR indicator."""
    strategy = MagicMock()
    trading = MagicMock()
    trading.fn = lambda ctx: None
    trading.indicator_names = frozenset({"atr_14"})
    trading.symbols = frozenset({"SPY"})
    data_only = MagicMock()
    data_only.fn = None
    data_only.indicator_names = frozenset({"regime"})
    data_only.symbols = frozenset({"AAPL"})
    strategy._executions = {trading, data_only}
    strategy._before_exec_fn = None
    strategy._after_exec_fn = None
    strategy._rotation_sizer = None
    model = VolatilitySlippageModel(atr_indicator="atr_14")
    model.validate(strategy)


def test_volatility_validate_still_rejects_trading_execution_without_atr():
    strategy = MagicMock()
    trading = MagicMock()
    trading.fn = lambda ctx: None
    trading.indicator_names = frozenset({"atr_14"})
    trading.symbols = frozenset({"SPY"})
    other = MagicMock()
    other.fn = lambda ctx: None
    other.indicator_names = frozenset()
    other.symbols = frozenset({"AAPL"})
    strategy._executions = {trading, other}
    model = VolatilitySlippageModel(atr_indicator="atr_14")
    with pytest.raises(ValueError, match=re.escape("['AAPL']")):
        model.validate(strategy)
