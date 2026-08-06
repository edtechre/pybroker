"""Unit tests for indicator.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import pytest
import re
from pybroker.cache import CacheDateFields
from .fixtures import *  # noqa: F401
from pybroker.common import BarData, DataCol, IndicatorSymbol, to_datetime
from pybroker.config import StrategyConfig
from pybroker.optimize import hyperparam
from pybroker.scope import StaticScope
from pybroker.strategy import Strategy
from pybroker.vect import highv
from pybroker.indicator import (
    _to_bar_data,
    Indicator,
    IndicatorsMixin,
    IndicatorSet,
    IntervalBoundIndicator,
    adx,
    aroon_diff,
    aroon_down,
    aroon_up,
    atr,
    close_minus_ma,
    cubic_deviation,
    cubic_trend,
    delta_on_balance_volume,
    detrended_rsi,
    highest,
    indicator,
    intraday_intensity,
    laguerre_rsi,
    linear_deviation,
    linear_trend,
    lowest,
    macd,
    money_flow,
    normalized_negative_volume_index,
    normalized_on_balance_volume,
    normalized_positive_volume_index,
    price_change_oscillator,
    price_intensity,
    price_volume_fit,
    quadratic_deviation,
    quadratic_trend,
    reactivity,
    returns,
    stochastic,
    stochastic_rsi,
    volume_weighted_ma_ratio,
    volume_momentum,
)
from pybroker.vect import lowv

TF_SECONDS = 60
BETWEEN_TIME = ("10:00", "15:30")


@pytest.fixture()
def cache_date_fields(data_source_df):
    return CacheDateFields(
        start_date=to_datetime(sorted(data_source_df["date"].unique())[0]),
        end_date=to_datetime(sorted(data_source_df["date"].unique())[-1]),
        tf_seconds=TF_SECONDS,
        between_time=BETWEEN_TIME,
        days=None,
    )


@pytest.fixture(params=[True, False])
def parallel_indicators(request):
    return request.param


@pytest.fixture()
def ind_syms(hhv_ind, llv_ind, sumv_ind, symbols):
    return [
        IndicatorSymbol(ind.name, sym)
        for sym in symbols
        for ind in (hhv_ind, llv_ind, sumv_ind)
    ]


@pytest.fixture()
def setup_teardown(scope):
    scope.register_custom_cols("adj_close")
    yield
    scope.unregister_custom_cols("adj_close")


@pytest.mark.usefixtures("setup_teardown")
def test_to_bar_data(scope, data_source_df):
    bar_data = _to_bar_data(data_source_df)
    for col in scope.all_data_cols:
        expected = (
            data_source_df[col].to_numpy()
            if col in data_source_df.columns
            else None
        )
        assert np.array_equal(getattr(bar_data, col), expected)


@pytest.mark.parametrize("drop_col", ["date", "open", "high", "low", "close"])
def test_to_bar_data_when_missing_cols_then_error(drop_col, data_source_df):
    with pytest.raises(
        ValueError, match=f"DataFrame is missing required column: {drop_col}"
    ):
        _to_bar_data(data_source_df.drop(columns=drop_col))


@pytest.mark.usefixtures("setup_teardown")
def test_indicator():
    ind = indicator("llv", lowv)
    assert isinstance(ind, Indicator)
    assert ind.name == "llv"


@pytest.mark.usefixtures("setup_teardown")
class TestIndicator:
    def test_call_with_kwargs(self, hhv_ind, data_source_df):
        data = hhv_ind(data_source_df)
        assert len(data) == len(data_source_df["date"])
        assert isinstance(data.index[0], pd.Timestamp)

    def test_call_when_invalid_shape_then_error(self, data_source_df):
        def invalid_shape(_data):
            return np.array([[1, 2, 3], [4, 5, 6]])

        ind_invalid_shape = indicator("invalid_shape", invalid_shape)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Indicator invalid_shape must return a one-dimensional array."
            ),
        ):
            ind_invalid_shape(data_source_df)

    def test_iqr(self, hhv_ind, data_source_df):
        assert isinstance(hhv_ind.iqr(data_source_df), float)

    def test_relative_entropy(self, hhv_ind, data_source_df):
        assert isinstance(hhv_ind.relative_entropy(data_source_df), float)

    def test_repr(self, hhv_ind):
        assert repr(hhv_ind) == "Indicator('hhv', {'n': 5})"

    def test_intervals(self, hhv_ind):
        bound = hhv_ind.intervals("weekly", 5)
        assert isinstance(bound, IntervalBoundIndicator)
        assert bound.indicator is hhv_ind
        assert bound.intervals == frozenset(["weekly", 5])

    def test_intervals_scalar(self, hhv_ind):
        bound = hhv_ind.intervals("monthly")
        assert bound.intervals == frozenset(["monthly"])

    def test_intervals_normalizes_durations(self, hhv_ind):
        assert hhv_ind.intervals("05m").intervals == frozenset(["5m"])

    def test_intervals_when_no_args_then_error(self, hhv_ind):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Indicator.intervals() requires at least one interval."
            ),
        ):
            hhv_ind.intervals()

    def test_intervals_when_duplicate_then_error(self, hhv_ind):
        with pytest.raises(
            ValueError, match=re.escape("Duplicate interval: 'weekly'.")
        ):
            hhv_ind.intervals("weekly", "weekly")

    def test_intervals_base_sentinel(self, hhv_ind):
        bound = hhv_ind.intervals("base", "weekly")
        assert bound.intervals == frozenset(["base", "weekly"])

    def test_intervals_when_duplicate_base_then_error(self, hhv_ind):
        with pytest.raises(
            ValueError, match=re.escape("Duplicate interval: 'base'.")
        ):
            hhv_ind.intervals("base", "base")

    def test_intervals_when_invalid_interval_then_error(self, hhv_ind):
        with pytest.raises(
            ValueError, match="interval compression requires n > 1."
        ):
            hhv_ind.intervals(1)

    def test_intervals_when_list_then_error(self, hhv_ind):
        # Mirroring add_execution(intervals=[...]) into the varargs binder
        # must fail cleanly, not with an unhashable-type TypeError.
        with pytest.raises(
            ValueError, match=re.escape("Invalid interval ['weekly']")
        ):
            hhv_ind.intervals(["weekly"])


@pytest.mark.usefixtures("setup_teardown")
class TestIndicatorsMixin:
    def _assert_indicators(self, ind_data, ind_syms, data_source_df):
        assert set(ind_data.keys()) == set(ind_syms)
        for ind_sym, series in ind_data.items():
            df = data_source_df[data_source_df["symbol"] == ind_sym.symbol]
            assert len(series) == df.shape[0]

    @pytest.mark.usefixtures("setup_ind_cache")
    def test_compute_indicators(
        self,
        ind_syms,
        data_source_df,
        cache_date_fields,
        parallel_indicators,
    ):
        mixin = IndicatorsMixin()
        ind_data = mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            parallel_indicators=parallel_indicators,
        )
        self._assert_indicators(ind_data, ind_syms, data_source_df)

    @pytest.mark.usefixtures("setup_ind_cache")
    def test_compute_indicators_when_empty_data(
        self, ind_syms, cache_date_fields, parallel_indicators
    ):
        mixin = IndicatorsMixin()
        ind_data = mixin.compute_indicators(
            df=pd.DataFrame(columns=[col.value for col in DataCol]),
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            parallel_indicators=parallel_indicators,
        )
        assert len(ind_data) == 0

    @pytest.mark.usefixtures("setup_enabled_ind_cache")
    def test_compute_indicators_data_when_cached(
        self,
        ind_syms,
        cache_date_fields,
        data_source_df,
        parallel_indicators,
    ):
        mixin = IndicatorsMixin()
        mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            parallel_indicators=parallel_indicators,
        )
        ind_data = mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            parallel_indicators=parallel_indicators,
        )
        self._assert_indicators(ind_data, ind_syms, data_source_df)

    @pytest.mark.usefixtures("setup_enabled_ind_cache")
    def test_compute_indicators_when_partial_cached(
        self,
        ind_syms,
        cache_date_fields,
        data_source_df,
        parallel_indicators,
    ):
        mixin = IndicatorsMixin()
        mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms[:1],
            cache_date_fields=cache_date_fields,
            parallel_indicators=parallel_indicators,
        )
        ind_data = mixin.compute_indicators(
            df=data_source_df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            parallel_indicators=parallel_indicators,
        )
        self._assert_indicators(ind_data, ind_syms, data_source_df)


class TestIndicatorSet:
    def test_add_and_remove(self, hhv_ind, llv_ind, sumv_ind):
        ind_set = IndicatorSet()
        ind_set.add(hhv_ind)
        ind_set.add([llv_ind, sumv_ind], hhv_ind)
        assert ind_set._ind_names == set(["llv", "hhv", "sumv"])
        ind_set.remove(llv_ind)
        assert ind_set._ind_names == set(["hhv", "sumv"])
        ind_set.remove(hhv_ind, sumv_ind)
        assert not ind_set._ind_names

    def test_clear(self, hhv_ind, llv_ind, sumv_ind):
        ind_set = IndicatorSet()
        ind_set.add(llv_ind, sumv_ind, hhv_ind)
        assert ind_set._ind_names == set(["llv", "hhv", "sumv"])
        ind_set.clear()
        assert not ind_set._ind_names

    def test_call_when_indicators_empty_then_error(self, data_source_df):
        ind_set = IndicatorSet()
        with pytest.raises(ValueError, match="No indicators were added."):
            ind_set(data_source_df)

    @pytest.mark.parametrize("as_list", [False, True])
    def test_add_when_interval_bound_then_error(self, hhv_ind, as_list):
        bound = hhv_ind.intervals("weekly")
        with pytest.raises(
            ValueError,
            match="IndicatorSet requires Indicators, got "
            "IntervalBoundIndicator",
        ):
            IndicatorSet().add([bound] if as_list else bound)

    @pytest.mark.parametrize(
        "df", [pd.DataFrame(), LazyFixture("data_source_df")]
    )
    def test_call(self, df, hhv_ind, llv_ind, parallel_indicators, request):
        df = get_fixture(request, df)
        ind_set = IndicatorSet()
        ind_set.add([hhv_ind, llv_ind])
        result = ind_set(df, parallel_indicators)
        assert len(result) == len(df)
        assert set(result.columns) == set(["date", "symbol", "hhv", "llv"])

    def test_call_per_symbol_layout_and_values(
        self, data_source_df, hhv_ind, llv_ind, parallel_indicators
    ):
        ind_set = IndicatorSet()
        ind_set.add([hhv_ind, llv_ind])
        result = ind_set(data_source_df, parallel_indicators)

        sym_col = DataCol.SYMBOL.value
        date_col = DataCol.DATE.value

        assert list(result.columns) == [sym_col, date_col, "hhv", "llv"]
        assert len(result) == len(data_source_df)

        syms_in_output = result[sym_col].unique().tolist()
        assert set(syms_in_output) == set(
            data_source_df[sym_col].unique().tolist()
        )
        sym_blocks = result[sym_col].ne(result[sym_col].shift()).cumsum()
        assert sym_blocks.nunique() == len(syms_in_output)

        for sym in syms_in_output:
            sym_rows = result[result[sym_col] == sym].reset_index(drop=True)
            sym_input = data_source_df[
                data_source_df[sym_col] == sym
            ].reset_index(drop=True)

            assert len(sym_rows) == len(sym_input)
            pd.testing.assert_series_equal(
                sym_rows[date_col],
                sym_input[date_col],
                check_names=False,
            )

            for ind in (hhv_ind, llv_ind):
                expected = ind(sym_input).to_numpy()
                actual = sym_rows[ind.name].to_numpy()
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "fn, values, period, expected",
    [
        (
            highest,
            [3, 3, 4, 2, 5, 6, 1, 3],
            3,
            [np.nan, np.nan, 4, 4, 5, 6, 6, 6],
        ),
        (highest, [3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        (highest, [4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 4]),
        (highest, [1], 1, [1]),
        (
            lowest,
            [3, 3, 4, 2, 5, 6, 1, 3],
            3,
            [np.nan, np.nan, 3, 2, 2, 2, 1, 1],
        ),
        (lowest, [3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        (lowest, [4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 1]),
        (lowest, [1], 1, [1]),
        (
            returns,
            [1, 1.5, 1.7, 1.3, 1.2, 1.4],
            1,
            [np.nan, 0.5, 0.13333333, -0.23529412, -0.07692308, 0.16666667],
        ),
        (
            returns,
            [1, 1.5, 1.7, 1.3, 1.2, 1.4],
            2,
            [np.nan, np.nan, 0.7, -0.133333, -0.294118, 0.076923],
        ),
        (returns, [1], 1, [np.nan]),
        (returns, [], 5, []),
    ],
)
def test_wrappers(fn, values, period, expected):
    n = len(values)
    dates = pd.date_range(start="1/1/2018", end="1/1/2019").to_numpy()[:n]
    bar_data = BarData(
        date=dates,
        open=np.zeros(n),
        high=np.zeros(n),
        low=np.zeros(n),
        close=np.array(values),
        volume=None,
        vwap=None,
    )
    indicator = fn("my_indicator", "close", period)
    assert isinstance(indicator, Indicator)
    assert indicator.name == "my_indicator"
    series = indicator(bar_data)
    assert np.array_equal(series.index.to_numpy(), dates)
    assert np.array_equal(
        np.round(series.values, 6), np.round(expected, 6), equal_nan=True
    )


@pytest.mark.parametrize(
    "fn, args",
    [
        (
            detrended_rsi,
            {
                "field": "close",
                "short_length": 5,
                "long_length": 10,
                "reg_length": 20,
            },
        ),
        (macd, {"short_length": 5, "long_length": 10, "smoothing": 2.0}),
        (
            stochastic,
            {
                "lookback": 10,
                "smoothing": 2,
            },
        ),
        (
            stochastic_rsi,
            {
                "field": "close",
                "rsi_lookback": 10,
                "sto_lookback": 10,
                "smoothing": 2.0,
            },
        ),
        (
            linear_trend,
            {"field": "close", "lookback": 10, "atr_length": 20, "scale": 0.5},
        ),
        (
            quadratic_trend,
            {"field": "close", "lookback": 10, "atr_length": 20, "scale": 0.5},
        ),
        (
            cubic_trend,
            {"field": "close", "lookback": 10, "atr_length": 20, "scale": 0.5},
        ),
        (
            atr,
            {
                "lookback": 10,
            },
        ),
        (
            adx,
            {
                "lookback": 10,
            },
        ),
        (
            aroon_up,
            {
                "lookback": 10,
            },
        ),
        (
            aroon_down,
            {
                "lookback": 10,
            },
        ),
        (
            aroon_diff,
            {
                "lookback": 10,
            },
        ),
        (close_minus_ma, {"lookback": 10, "atr_length": 20, "scale": 0.5}),
        (linear_deviation, {"field": "close", "lookback": 10, "scale": 0.5}),
        (
            quadratic_deviation,
            {"field": "close", "lookback": 10, "scale": 0.5},
        ),
        (cubic_deviation, {"field": "close", "lookback": 10, "scale": 0.5}),
        (price_intensity, {"smoothing": 1.0, "scale": 0.5}),
        (
            price_change_oscillator,
            {"short_length": 5, "multiplier": 3, "scale": 0.5},
        ),
        (intraday_intensity, {"lookback": 10, "smoothing": 1.0}),
        (money_flow, {"lookback": 10, "smoothing": 1.0}),
        (reactivity, {"lookback": 10, "smoothing": 1.0, "scale": 0.5}),
        (price_volume_fit, {"lookback": 10, "scale": 0.5}),
        (volume_weighted_ma_ratio, {"lookback": 10, "scale": 0.5}),
        (normalized_on_balance_volume, {"lookback": 10, "scale": 0.5}),
        (
            delta_on_balance_volume,
            {"lookback": 10, "delta_length": 5, "scale": 0.5},
        ),
        (normalized_positive_volume_index, {"lookback": 10, "scale": 0.5}),
        (normalized_negative_volume_index, {"lookback": 10, "scale": 0.5}),
        (volume_momentum, {"short_length": 5, "multiplier": 2, "scale": 2.0}),
        (laguerre_rsi, {"fe_length": 20}),
    ],
)
def test_indicators(fn, args):
    dates = pd.date_range(start="1/1/2018", end="1/1/2019").to_numpy()
    n = len(dates)
    bar_data = BarData(
        date=dates,
        open=np.random.rand(n),
        high=np.random.rand(n),
        low=np.random.rand(n),
        close=np.random.rand(n),
        volume=np.random.rand(n),
        vwap=None,
    )
    indicator = fn(fn.__name__, **args)
    assert isinstance(indicator, Indicator)
    assert indicator.name == fn.__name__
    series = indicator(bar_data)
    assert len(series) == n
    assert np.array_equal(series.index.to_numpy(), dates)


@pytest.fixture(autouse=True)
def clear_hyperparams_for_memo():
    scope = StaticScope.instance()
    scope._hyperparams.clear()
    yield
    scope._hyperparams.clear()


def _hyperparam_hhv_indicator():
    lookback = hyperparam("lookback", default=10, low=5, high=20, step=5)
    hhv = indicator(
        "hhv_memo",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    return hhv


def test_indicator_memo_disabled(data_source_df):
    hhv = _hyperparam_hhv_indicator()
    strategy = Strategy(
        data_source_df,
        "2020-01-02",
        "2021-12-31",
        StrategyConfig(),
    )
    strategy._indicator_memo_max = 0
    ind_sym = IndicatorSymbol(hhv.name, "AAPL")
    strategy.compute_indicators(
        df=data_source_df,
        indicator_syms=[ind_sym],
        cache_date_fields=None,
        parallel_indicators=False,
        hyperparams={"lookback": 10},
    )
    assert len(strategy._indicator_memo_store()) == 0


def test_indicator_memo_eviction(data_source_df):
    hhv = _hyperparam_hhv_indicator()
    strategy = Strategy(
        data_source_df,
        "2020-01-02",
        "2021-12-31",
        StrategyConfig(),
    )
    strategy._indicator_memo_max = 2
    ind_sym = IndicatorSymbol(hhv.name, "AAPL")
    for lookback in (5, 10, 15):
        strategy.compute_indicators(
            df=data_source_df,
            indicator_syms=[ind_sym],
            cache_date_fields=None,
            parallel_indicators=False,
            hyperparams={"lookback": lookback},
        )
    memo = strategy._indicator_memo_store()
    assert len(memo) == 2
    cached = {key[2] for key in memo}
    assert (("lookback", 5),) not in cached
    assert (("lookback", 10),) in cached
    assert (("lookback", 15),) in cached


def test_indicator_hyperparam_defaults_via_backtest(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=20, step=5)
    hhv = indicator(
        "hhv_default",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = Strategy(
        data_source_df,
        "2020-01-02",
        "2021-12-31",
        StrategyConfig(),
    )
    last_values: list[float] = []

    def exec_fn(ctx):
        vals = ctx.indicator(hhv.name)
        if len(vals) > 0:
            last_values.append(float(vals[-1]))

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv])
    strategy.backtest(parallel_indicators=False)
    default_last = last_values[-1]

    last_values.clear()
    strategy.backtest(params={"lookback": 10}, parallel_indicators=False)
    explicit_default_last = last_values[-1]

    last_values.clear()
    strategy.backtest(params={"lookback": 5}, parallel_indicators=False)
    alternate_last = last_values[-1]

    sym_df = data_source_df[data_source_df[DataCol.SYMBOL.value] == "AAPL"]
    series_default = hhv(sym_df)
    series_alternate = hhv(sym_df, hyperparams={"lookback": 5})
    expected = float(series_default.iloc[-1])

    assert default_last == explicit_default_last
    assert default_last == expected
    assert alternate_last == float(series_alternate.iloc[-1])
    assert not np.allclose(
        series_default.to_numpy(),
        series_alternate.to_numpy(),
        equal_nan=True,
    )
