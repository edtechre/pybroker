"""Unit tests for vect.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pytest
import re
from pybroker.vect import (
    adx,
    aroon_diff,
    aroon_down,
    aroon_up,
    close_minus_ma,
    cross,
    cubic_deviation,
    cubic_trend,
    delta_on_balance_volume,
    detrended_rsi,
    highv,
    intraday_intensity,
    laguerre_rsi,
    linear_deviation,
    linear_trend,
    lowv,
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
    returnv,
    stochastic,
    stochastic_rsi,
    sumv,
    volume_momentum,
    volume_weighted_ma_ratio,
)

np.random.seed(42)


@pytest.mark.parametrize(
    "array, n, expected",
    [
        ([3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 3, 2, 2, 2, 1, 1]),
        ([3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        ([4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 1]),
        ([1], 1, [1]),
        ([], 5, []),
    ],
)
def test_lowv(array, n, expected):
    assert np.array_equal(lowv(np.array(array), n), expected, equal_nan=True)


@pytest.mark.parametrize(
    "array, n, expected",
    [
        ([3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 4, 4, 5, 6, 6, 6]),
        ([3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        ([4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 4]),
        ([1], 1, [1]),
        ([], 5, []),
    ],
)
def test_highv(array, n, expected):
    assert np.array_equal(highv(np.array(array), n), expected, equal_nan=True)


@pytest.mark.parametrize(
    "array, n, expected",
    [
        ([3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 10, 9, 11, 13, 12, 10]),
        ([3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        ([4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 10]),
        ([1], 1, [1]),
        ([], 5, []),
    ],
)
def test_sumv(array, n, expected):
    assert np.array_equal(sumv(np.array(array), n), expected, equal_nan=True)


@pytest.mark.parametrize(
    "array, n, expected",
    [
        (
            [1, 1.5, 1.7, 1.3, 1.2, 1.4],
            1,
            [np.nan, 0.5, 0.13333333, -0.23529412, -0.07692308, 0.16666667],
        ),
        (
            [1, 1.5, 1.7, 1.3, 1.2, 1.4],
            2,
            [np.nan, np.nan, 0.7, -0.133333, -0.294118, 0.076923],
        ),
        ([1], 1, [np.nan]),
        ([], 5, []),
    ],
)
def test_returnv(array, n, expected):
    assert np.array_equal(
        np.round(returnv(np.array(array), n), 6),
        np.round(expected, 6),
        equal_nan=True,
    )


@pytest.mark.parametrize("fnv", [lowv, highv, sumv, returnv])
@pytest.mark.parametrize(
    "array, n, expected_msg",
    [
        ([1, 2, 3], 10, "n is greater than array length."),
        ([1, 2, 3], 0, "n needs to be >= 1."),
        ([1, 2, 3], -1, "n needs to be >= 1."),
    ],
)
def test_when_n_invalid_then_error(fnv, array, n, expected_msg):
    with pytest.raises(AssertionError, match=re.escape(expected_msg)):
        fnv(np.array(array), n)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
            [3, 3, 4, 2, 5, 6, 1, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [0, 0, 1, 0, 1, 0, 0, 0],
        ),
        (
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 4, 2, 5, 6, 1, 3],
            [0, 0, 0, 1, 0, 0, 1, 0],
        ),
        ([1, 1], [1, 1], [0, 0]),
    ],
)
def test_cross(a, b, expected):
    assert np.array_equal(
        cross(np.array(a), np.array(b)), expected, equal_nan=True
    )


@pytest.mark.parametrize(
    "a, b, expected_msg",
    [
        ([1, 2, 3], [3, 3, 3, 3], "a and b must be same length."),
        ([3, 3, 3, 3], [1, 2, 3], "a and b must be same length."),
        ([1, 2, 3], [], "b cannot be empty."),
        ([], [1, 2, 3], "a cannot be empty."),
        ([1], [1], "a and b must have length >= 2."),
    ],
)
def test_cross_when_invalid_input_then_error(a, b, expected_msg):
    with pytest.raises(AssertionError, match=re.escape(expected_msg)):
        cross(np.array(a), np.array(b))


@pytest.mark.parametrize(
    "fn, args, expected_length",
    [
        # Detrended RSI
        (
            detrended_rsi,
            {
                "values": np.random.rand(1_000),
                "short_length": 2,
                "long_length": 4,
                "reg_length": 30,
            },
            1_000,
        ),
        (
            detrended_rsi,
            {
                "values": np.array([]),
                "short_length": 2,
                "long_length": 4,
                "reg_length": 30,
            },
            0,
        ),
        (
            detrended_rsi,
            {
                "values": np.random.rand(10),
                "short_length": 2,
                "long_length": 4,
                "reg_length": 30,
            },
            10,
        ),
        # MACD
        (
            macd,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 2,
                "long_length": 4,
                "smoothing": 0.1,
            },
            1_000,
        ),
        (
            macd,
            {
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "short_length": 2,
                "long_length": 4,
                "smoothing": 0.1,
            },
            0,
        ),
        (
            macd,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "short_length": 2,
                "long_length": 50,
                "smoothing": 0.1,
            },
            10,
        ),
        # Stochastic
        (
            stochastic,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": 0,
            },
            1_000,
        ),
        (
            stochastic,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": 1,
            },
            1_000,
        ),
        (
            stochastic,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": 2,
            },
            1_000,
        ),
        (
            stochastic,
            {
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "lookback": 5,
                "smoothing": 0,
            },
            0,
        ),
        (
            stochastic,
            {
                "high": np.array([1.0]),
                "low": np.array([1.0]),
                "close": np.array([1.0]),
                "lookback": 5,
                "smoothing": 0,
            },
            1,
        ),
        # Stochastic RSI
        (
            stochastic_rsi,
            {
                "values": np.random.rand(1_000),
                "rsi_lookback": 5,
                "sto_lookback": 5,
            },
            1_000,
        ),
        (
            stochastic_rsi,
            {
                "values": np.random.rand(1_000),
                "rsi_lookback": 5,
                "sto_lookback": 5,
                "smoothing": 0.5,
            },
            1_000,
        ),
        (
            stochastic_rsi,
            {
                "values": np.array([]),
                "rsi_lookback": 5,
                "sto_lookback": 5,
            },
            0,
        ),
        (
            stochastic_rsi,
            {
                "values": np.random.rand(10),
                "rsi_lookback": 5,
                "sto_lookback": 20,
            },
            10,
        ),
        (
            stochastic_rsi,
            {
                "values": np.random.rand(10),
                "rsi_lookback": 20,
                "sto_lookback": 5,
            },
            10,
        ),
        # Linear Trend
        (
            linear_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
            1_000,
        ),
        (
            linear_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
                "scale": 0.5,
            },
            1_000,
        ),
        (
            linear_trend,
            {
                "values": np.array([]),
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "lookback": 20,
                "atr_length": 10,
            },
            0,
        ),
        (
            linear_trend,
            {
                "values": np.random.rand(10),
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "lookback": 20,
                "atr_length": 10,
            },
            10,
        ),
        (
            linear_trend,
            {
                "values": np.random.rand(10),
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "lookback": 10,
                "atr_length": 20,
            },
            10,
        ),
        # Quadratic Trend
        (
            quadratic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
            1_000,
        ),
        (
            quadratic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
                "scale": 0.5,
            },
            1_000,
        ),
        (
            quadratic_trend,
            {
                "values": np.array([]),
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "lookback": 20,
                "atr_length": 10,
            },
            0,
        ),
        (
            quadratic_trend,
            {
                "values": np.random.rand(10),
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "lookback": 20,
                "atr_length": 10,
            },
            10,
        ),
        (
            quadratic_trend,
            {
                "values": np.random.rand(10),
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "lookback": 10,
                "atr_length": 20,
            },
            10,
        ),
        # Cubic Trend
        (
            cubic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
            1_000,
        ),
        (
            cubic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
                "scale": 0.5,
            },
            1_000,
        ),
        (
            cubic_trend,
            {
                "values": np.array([]),
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "lookback": 20,
                "atr_length": 10,
            },
            0,
        ),
        (
            cubic_trend,
            {
                "values": np.random.rand(10),
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "lookback": 20,
                "atr_length": 10,
            },
            10,
        ),
        (
            cubic_trend,
            {
                "values": np.random.rand(10),
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "lookback": 10,
                "atr_length": 20,
            },
            10,
        ),
        # ADX
        (
            adx,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 10,
            },
            1_000,
        ),
        (
            adx,
            {
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "lookback": 10,
            },
            0,
        ),
        (
            adx,
            {
                "high": np.array([1.0]),
                "low": np.array([1.0]),
                "close": np.array([1.0]),
                "lookback": 10,
            },
            1,
        ),
        # Aroon Up
        (
            aroon_up,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "lookback": 10,
            },
            1_000,
        ),
        (
            aroon_up,
            {
                "high": np.array([]),
                "low": np.array([]),
                "lookback": 10,
            },
            0,
        ),
        (
            aroon_up,
            {
                "high": np.array([1.0]),
                "low": np.array([1.0]),
                "lookback": 10,
            },
            1,
        ),
        # Aroon Down
        (
            aroon_down,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "lookback": 10,
            },
            1_000,
        ),
        (
            aroon_down,
            {
                "high": np.array([]),
                "low": np.array([]),
                "lookback": 10,
            },
            0,
        ),
        (
            aroon_down,
            {
                "high": np.array([1.0]),
                "low": np.array([1.0]),
                "lookback": 10,
            },
            1,
        ),
        # Aroon Diff
        (
            aroon_diff,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "lookback": 10,
            },
            1_000,
        ),
        (
            aroon_diff,
            {
                "high": np.array([]),
                "low": np.array([]),
                "lookback": 10,
            },
            0,
        ),
        (
            aroon_diff,
            {
                "high": np.array([1.0]),
                "low": np.array([1.0]),
                "lookback": 10,
            },
            1,
        ),
        # Close Minus MA
        (
            close_minus_ma,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
            1_000,
        ),
        (
            close_minus_ma,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
                "scale": 0.5,
            },
            1_000,
        ),
        (
            close_minus_ma,
            {
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "lookback": 20,
                "atr_length": 10,
            },
            0,
        ),
        (
            close_minus_ma,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "lookback": 20,
                "atr_length": 10,
            },
            10,
        ),
        (
            close_minus_ma,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
                "lookback": 10,
                "atr_length": 20,
            },
            10,
        ),
        # Linear Deviation
        (
            linear_deviation,
            {
                "values": np.random.rand(1_000),
                "lookback": 10,
            },
            1_000,
        ),
        (
            linear_deviation,
            {
                "values": np.random.rand(1_000),
                "lookback": 10,
                "scale": 1.0,
            },
            1_000,
        ),
        (
            linear_deviation,
            {
                "values": np.array([]),
                "lookback": 10,
            },
            0,
        ),
        (
            linear_deviation,
            {
                "values": np.array([1.0]),
                "lookback": 10,
            },
            1,
        ),
        # Quadratic Deviation
        (
            quadratic_deviation,
            {
                "values": np.random.rand(1_000),
                "lookback": 10,
            },
            1_000,
        ),
        (
            quadratic_deviation,
            {
                "values": np.random.rand(1_000),
                "lookback": 10,
                "scale": 1.0,
            },
            1_000,
        ),
        (
            quadratic_deviation,
            {
                "values": np.array([]),
                "lookback": 10,
            },
            0,
        ),
        (
            quadratic_deviation,
            {
                "values": np.array([1.0]),
                "lookback": 10,
            },
            1,
        ),
        # Cubic Deviation
        (
            cubic_deviation,
            {
                "values": np.random.rand(1_000),
                "lookback": 10,
            },
            1_000,
        ),
        (
            cubic_deviation,
            {
                "values": np.random.rand(1_000),
                "lookback": 10,
                "scale": 1.0,
            },
            1_000,
        ),
        (
            cubic_deviation,
            {
                "values": np.array([]),
                "lookback": 10,
            },
            0,
        ),
        (
            cubic_deviation,
            {
                "values": np.array([1.0]),
                "lookback": 10,
            },
            1,
        ),
        # Price Intensity
        (
            price_intensity,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
            },
            1_000,
        ),
        (
            price_intensity,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "smoothing": 0.1,
            },
            1_000,
        ),
        (
            price_intensity,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "scale": 0.5,
            },
            1_000,
        ),
        (
            price_intensity,
            {
                "open": np.array([]),
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
            },
            0,
        ),
        # Price Change Oscillator
        (
            price_change_oscillator,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 5,
                "multiplier": 2,
            },
            1_000,
        ),
        (
            price_change_oscillator,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 5,
                "multiplier": 2,
                "scale": 1.0,
            },
            1_000,
        ),
        (
            price_change_oscillator,
            {
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "short_length": 5,
                "multiplier": 2,
            },
            0,
        ),
        (
            price_change_oscillator,
            {
                "high": np.array([1.0]),
                "low": np.array([1.0]),
                "close": np.array([1.0]),
                "short_length": 5,
                "multiplier": 2,
            },
            1,
        ),
        # Intraday Intensity
        (
            intraday_intensity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
            1_000,
        ),
        (
            intraday_intensity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": 1.1,
            },
            1_000,
        ),
        (
            intraday_intensity,
            {
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "volume": np.array([]),
                "lookback": 5,
            },
            0,
        ),
        (
            intraday_intensity,
            {
                "high": np.array([1.0]),
                "low": np.array([1.0]),
                "close": np.array([1.0]),
                "volume": np.array([1.0]),
                "lookback": 5,
            },
            1,
        ),
        # Money Flow
        (
            money_flow,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
            1_000,
        ),
        (
            money_flow,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": 1.1,
            },
            1_000,
        ),
        (
            money_flow,
            {
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "volume": np.array([]),
                "lookback": 5,
            },
            0,
        ),
        (
            money_flow,
            {
                "high": np.array([1.0]),
                "low": np.array([1.0]),
                "close": np.array([1.0]),
                "volume": np.array([1.0]),
                "lookback": 5,
            },
            1,
        ),
        # Reactivity
        (
            reactivity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
            1_000,
        ),
        (
            reactivity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": 2.0,
            },
            1_000,
        ),
        (
            reactivity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 1.0,
            },
            1_000,
        ),
        (
            reactivity,
            {
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "volume": np.array([]),
                "lookback": 5,
            },
            0,
        ),
        (
            reactivity,
            {
                "high": np.array([1.0]),
                "low": np.array([1.0]),
                "close": np.array([1.0]),
                "volume": np.array([1.0]),
                "lookback": 5,
            },
            1,
        ),
        # Price Volume Fit
        (
            price_volume_fit,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
            1_000,
        ),
        (
            price_volume_fit,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 1.5,
            },
            1_000,
        ),
        (
            price_volume_fit,
            {
                "close": np.array([]),
                "volume": np.array([]),
                "lookback": 5,
            },
            0,
        ),
        (
            price_volume_fit,
            {
                "close": np.array([1.0]),
                "volume": np.array([1.0]),
                "lookback": 5,
            },
            1,
        ),
        # Volume Weighted MA Ratio
        (
            volume_weighted_ma_ratio,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
            1_000,
        ),
        (
            volume_weighted_ma_ratio,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 1.5,
            },
            1_000,
        ),
        (
            volume_weighted_ma_ratio,
            {
                "close": np.array([]),
                "volume": np.array([]),
                "lookback": 5,
            },
            0,
        ),
        (
            volume_weighted_ma_ratio,
            {
                "close": np.array([1.0]),
                "volume": np.array([1.0]),
                "lookback": 5,
            },
            1,
        ),
        # Normalized On Balance Volume
        (
            normalized_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
            1_000,
        ),
        (
            normalized_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 1.5,
            },
            1_000,
        ),
        (
            normalized_on_balance_volume,
            {
                "close": np.array([]),
                "volume": np.array([]),
                "lookback": 5,
            },
            0,
        ),
        (
            normalized_on_balance_volume,
            {
                "close": np.array([1.0]),
                "volume": np.array([1.0]),
                "lookback": 5,
            },
            1,
        ),
        # Delta On Balance Volume
        (
            delta_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
            1_000,
        ),
        (
            delta_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "delta_length": 10,
            },
            1_000,
        ),
        (
            delta_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 1.0,
            },
            1_000,
        ),
        (
            delta_on_balance_volume,
            {
                "close": np.array([1.0]),
                "volume": np.array([1.0]),
                "lookback": 5,
            },
            1,
        ),
        (
            delta_on_balance_volume,
            {
                "close": np.array([]),
                "volume": np.array([]),
                "lookback": 5,
            },
            0,
        ),
        # Normalized Positive Volume Index
        (
            normalized_positive_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
            1_000,
        ),
        (
            normalized_positive_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 1.5,
            },
            1_000,
        ),
        (
            normalized_positive_volume_index,
            {
                "close": np.array([]),
                "volume": np.array([]),
                "lookback": 5,
            },
            0,
        ),
        (
            normalized_positive_volume_index,
            {
                "close": np.array([1.0]),
                "volume": np.array([1.0]),
                "lookback": 5,
            },
            1,
        ),
        # Normalized Negative Volume Index
        (
            normalized_negative_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
            1_000,
        ),
        (
            normalized_negative_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 1.5,
            },
            1_000,
        ),
        (
            normalized_negative_volume_index,
            {
                "close": np.array([]),
                "volume": np.array([]),
                "lookback": 5,
            },
            0,
        ),
        (
            normalized_negative_volume_index,
            {
                "close": np.array([1.0]),
                "volume": np.array([1.0]),
                "lookback": 5,
            },
            1,
        ),
        # Volume Momentum
        (
            volume_momentum,
            {
                "volume": np.random.rand(1_000),
                "short_length": 5,
            },
            1_000,
        ),
        (
            volume_momentum,
            {
                "volume": np.random.rand(1_000),
                "short_length": 5,
                "multiplier": 3,
            },
            1_000,
        ),
        (
            volume_momentum,
            {"volume": np.random.rand(1_000), "short_length": 5, "scale": 1.0},
            1_000,
        ),
        (
            volume_momentum,
            {
                "volume": np.array([1.0]),
                "short_length": 5,
            },
            1,
        ),
        (
            volume_momentum,
            {
                "volume": np.array([]),
                "short_length": 5,
            },
            0,
        ),
        # Laguerre RSI
        (
            laguerre_rsi,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
            },
            1_000,
        ),
        (
            laguerre_rsi,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "fe_length": 20,
            },
            1_000,
        ),
        (
            laguerre_rsi,
            {
                "open": np.random.rand(10),
                "high": np.random.rand(10),
                "low": np.random.rand(10),
                "close": np.random.rand(10),
            },
            10,
        ),
        (
            laguerre_rsi,
            {
                "open": np.array([]),
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
            },
            0,
        ),
    ],
)
def test_indicators(fn, args, expected_length):
    assert len(fn(**args)) == expected_length


@pytest.mark.parametrize(
    "fn, args",
    [
        # Detrended RSI
        (
            detrended_rsi,
            {
                "values": np.random.rand(100),
                "short_length": 1,
                "long_length": 4,
                "reg_length": 30,
            },
        ),
        (
            detrended_rsi,
            {
                "values": np.random.rand(100),
                "short_length": 1,
                "long_length": 1,
                "reg_length": 30,
            },
        ),
        (
            detrended_rsi,
            {
                "values": np.random.rand(100),
                "short_length": 5,
                "long_length": 4,
                "reg_length": 30,
            },
        ),
        (
            detrended_rsi,
            {
                "values": np.random.rand(100),
                "short_length": 2,
                "long_length": 4,
                "reg_length": 0,
            },
        ),
        # MACD
        (
            macd,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 2,
                "long_length": 4,
            },
        ),
        (
            macd,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "short_length": 2,
                "long_length": 4,
            },
        ),
        (
            macd,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "short_length": 2,
                "long_length": 4,
            },
        ),
        (
            macd,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 0,
                "long_length": 4,
            },
        ),
        (
            macd,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 2,
                "long_length": 0,
            },
        ),
        (
            macd,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 2,
                "long_length": 1,
            },
        ),
        (
            macd,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 2,
                "long_length": 4,
                "smoothing": -0.1,
            },
        ),
        (
            macd,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 2,
                "long_length": 4,
                "scale": 0,
            },
        ),
        # Stochastic
        (
            stochastic,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": 0,
            },
        ),
        (
            stochastic,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": 0,
            },
        ),
        (
            stochastic,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "lookback": 5,
                "smoothing": 0,
            },
        ),
        (
            stochastic,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 0,
                "smoothing": 0,
            },
        ),
        (
            stochastic,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": 3,
            },
        ),
        # Stochastic RSI
        (
            stochastic_rsi,
            {
                "values": np.random.rand(1_000),
                "rsi_lookback": 0,
                "sto_lookback": 5,
            },
        ),
        (
            stochastic_rsi,
            {
                "values": np.random.rand(1_000),
                "rsi_lookback": 5,
                "sto_lookback": 0,
            },
        ),
        (
            stochastic_rsi,
            {
                "values": np.random.rand(1_000),
                "rsi_lookback": 5,
                "sto_lookback": 5,
                "smoothing": -0.1,
            },
        ),
        # Linear Trend
        (
            linear_trend,
            {
                "values": np.random.rand(10),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            linear_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            linear_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            linear_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            linear_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 0,
                "atr_length": 10,
            },
        ),
        (
            linear_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 0,
            },
        ),
        (
            linear_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
                "scale": 0,
            },
        ),
        # Quadratic Trend
        (
            quadratic_trend,
            {
                "values": np.random.rand(10),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            quadratic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            quadratic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            quadratic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            quadratic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 0,
                "atr_length": 10,
            },
        ),
        (
            quadratic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 0,
            },
        ),
        (
            quadratic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
                "scale": 0,
            },
        ),
        # Cubic Trend
        (
            cubic_trend,
            {
                "values": np.random.rand(10),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            cubic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            cubic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            cubic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            cubic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 0,
                "atr_length": 10,
            },
        ),
        (
            cubic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 0,
            },
        ),
        (
            cubic_trend,
            {
                "values": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
                "scale": 0,
            },
        ),
        # ADX
        (
            adx,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 10,
            },
        ),
        (
            adx,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "lookback": 10,
            },
        ),
        (
            adx,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "lookback": 10,
            },
        ),
        (
            adx,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        # Aroon Up
        (
            aroon_up,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "lookback": 10,
            },
        ),
        (
            aroon_up,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "lookback": 10,
            },
        ),
        (
            aroon_up,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        # Aroon Down
        (
            aroon_down,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "lookback": 10,
            },
        ),
        (
            aroon_down,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "lookback": 10,
            },
        ),
        (
            aroon_down,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        # Aroon Diff
        (
            aroon_diff,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "lookback": 10,
            },
        ),
        (
            aroon_diff,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "lookback": 10,
            },
        ),
        (
            aroon_diff,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        # Close Minus MA
        (
            close_minus_ma,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            close_minus_ma,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            close_minus_ma,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "lookback": 20,
                "atr_length": 10,
            },
        ),
        (
            close_minus_ma,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 0,
                "atr_length": 10,
            },
        ),
        (
            close_minus_ma,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 0,
            },
        ),
        (
            close_minus_ma,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "lookback": 20,
                "atr_length": 10,
                "scale": 0,
            },
        ),
        # Linear Deviation
        (
            linear_deviation,
            {
                "values": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        # Quadratic Deviation
        (
            quadratic_deviation,
            {
                "values": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        # Cubic Deviation
        (
            cubic_deviation,
            {
                "values": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        # Price Intensity
        (
            price_intensity,
            {
                "open": np.random.rand(10),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
            },
        ),
        (
            price_intensity,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
            },
        ),
        (
            price_intensity,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
            },
        ),
        (
            price_intensity,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
            },
        ),
        (
            price_intensity,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "smoothing": -1,
            },
        ),
        (
            price_intensity,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "scale": 0,
            },
        ),
        # Price Change Oscillator
        (
            price_change_oscillator,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 5,
                "multiplier": 2,
            },
        ),
        (
            price_change_oscillator,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "short_length": 5,
                "multiplier": 2,
            },
        ),
        (
            price_change_oscillator,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "short_length": 5,
                "multiplier": 2,
            },
        ),
        (
            price_change_oscillator,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 0,
                "multiplier": 2,
            },
        ),
        (
            price_change_oscillator,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 5,
                "multiplier": 0,
            },
        ),
        (
            price_change_oscillator,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "short_length": 5,
                "multiplier": 2,
                "scale": 0,
            },
        ),
        # Intraday Intensity
        (
            intraday_intensity,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            intraday_intensity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            intraday_intensity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            intraday_intensity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(10),
                "lookback": 5,
            },
        ),
        (
            intraday_intensity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        (
            intraday_intensity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": -1,
            },
        ),
        # Money Flow
        (
            money_flow,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            money_flow,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            money_flow,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            money_flow,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(10),
                "lookback": 5,
            },
        ),
        (
            money_flow,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        (
            money_flow,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": -1,
            },
        ),
        # Reactivity
        (
            reactivity,
            {
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            reactivity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            reactivity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            reactivity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(10),
                "lookback": 5,
            },
        ),
        (
            reactivity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        (
            reactivity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "smoothing": -1,
            },
        ),
        (
            reactivity,
            {
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 0,
            },
        ),
        # Price Volume Fit
        (
            price_volume_fit,
            {
                "close": np.random.rand(10),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            price_volume_fit,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(10),
                "lookback": 5,
            },
        ),
        (
            price_volume_fit,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        # Volume Weighted MA Ratio
        (
            volume_weighted_ma_ratio,
            {
                "close": np.random.rand(10),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            volume_weighted_ma_ratio,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(10),
                "lookback": 5,
            },
        ),
        (
            volume_weighted_ma_ratio,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        # Normalized On Balance Volume
        (
            normalized_on_balance_volume,
            {
                "close": np.random.rand(10),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            normalized_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(10),
                "lookback": 5,
            },
        ),
        (
            normalized_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        (
            normalized_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 0,
            },
        ),
        # Delta On Balance Volume
        (
            delta_on_balance_volume,
            {
                "close": np.random.rand(10),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            delta_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(10),
                "lookback": 5,
            },
        ),
        (
            delta_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        (
            delta_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 0,
            },
        ),
        (
            delta_on_balance_volume,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "delta_length": -1,
            },
        ),
        # Normalized Positive Volume Index
        (
            normalized_positive_volume_index,
            {
                "close": np.random.rand(10),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            normalized_positive_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(10),
                "lookback": 5,
            },
        ),
        (
            normalized_positive_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        (
            normalized_positive_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 0,
            },
        ),
        # Normalized Negative Volume Index
        (
            normalized_negative_volume_index,
            {
                "close": np.random.rand(10),
                "volume": np.random.rand(1_000),
                "lookback": 5,
            },
        ),
        (
            normalized_negative_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(10),
                "lookback": 5,
            },
        ),
        (
            normalized_negative_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 0,
            },
        ),
        (
            normalized_negative_volume_index,
            {
                "close": np.random.rand(1_000),
                "volume": np.random.rand(1_000),
                "lookback": 5,
                "scale": 0,
            },
        ),
        # Volume Momentum
        (
            volume_momentum,
            {
                "volume": np.random.rand(1_000),
                "short_length": 0,
            },
        ),
        (
            volume_momentum,
            {
                "volume": np.random.rand(1_000),
                "short_length": 5,
                "multiplier": 0,
            },
        ),
        (
            volume_momentum,
            {
                "volume": np.random.rand(1_000),
                "short_length": 5,
                "scale": 0,
            },
        ),
        # Laguerre RSI
        (
            laguerre_rsi,
            {
                "open": np.random.rand(10),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
            },
        ),
        (
            laguerre_rsi,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(10),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
            },
        ),
        (
            laguerre_rsi,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(10),
                "close": np.random.rand(1_000),
            },
        ),
        (
            laguerre_rsi,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(10),
            },
        ),
        (
            laguerre_rsi,
            {
                "open": np.random.rand(1_000),
                "high": np.random.rand(1_000),
                "low": np.random.rand(1_000),
                "close": np.random.rand(1_000),
                "fe_length": 0,
            },
        ),
    ],
)
def test_indicators_when_assertion_error(fn, args):
    with pytest.raises(AssertionError):
        fn(**args)


# --- Value-level regression tests -------------------------------------------
# The parametrized `test_indicators` above asserts only output LENGTH, which is
# why a family of indexing and window bugs shipped undetected. These pin the
# invariants that actually caught them.


def _ohlcv(n=300, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = np.abs(rng.normal(1e6, 2e5, n))
    return (
        np.ascontiguousarray(open_),
        np.ascontiguousarray(high),
        np.ascontiguousarray(low),
        np.ascontiguousarray(close),
        np.ascontiguousarray(volume),
    )


@pytest.mark.parametrize("lookback", [5, 14, 25])
def test_aroon_stays_within_its_defined_range(lookback):
    """Aroon is 0-100 by construction. Reusing the outer bar index as the
    inner scan variable made the writes land on the wrong bars and produced
    readings up to 200."""
    _, high, low, _, _ = _ohlcv()
    up = np.asarray(aroon_up(high, low, lookback))
    down = np.asarray(aroon_down(high, low, lookback))
    diff = np.asarray(aroon_diff(high, low, lookback))
    assert up.min() >= 0 and up.max() <= 100
    assert down.min() >= 0 and down.max() <= 100
    assert diff.min() >= -100 and diff.max() <= 100
    np.testing.assert_allclose(diff, up - down, rtol=0, atol=1e-9)


def test_laguerre_rsi_computes_every_bar_after_warmup():
    """The inner loop used to leave `output[fe_length:]` unwritten, so the
    whole series was zero apart from one warmup slot."""
    open_, high, low, close, _ = _ohlcv()
    fe_length = 13
    values = np.asarray(laguerre_rsi(open_, high, low, close, fe_length))
    assert np.all(values[:fe_length] == 0)
    body = values[fe_length:]
    assert np.count_nonzero(body) > 0.5 * len(body)
    assert body.min() >= 0 and body.max() <= 100


@pytest.mark.parametrize("fe_length", [1, 2])
def test_laguerre_rsi_short_fe_length_does_not_raise(fe_length):
    """fe_length of 1 divided by log(1) == 0."""
    open_, high, low, close, _ = _ohlcv(n=50)
    assert len(laguerre_rsi(open_, high, low, close, fe_length)) == 50


@pytest.mark.parametrize(
    "short_length, multiplier", [(5, 3), (10, 3), (20, 4)]
)
def test_price_change_oscillator_is_zero_on_constant_log_returns(
    short_length, multiplier
):
    """Short- and long-term |log return| averages are equal on a constant-rate
    series, so the oscillator must be 0. Summing short_length + 2 terms over a
    short_length divisor left a large constant offset."""
    n = 200
    close = np.ascontiguousarray(1.001 ** np.arange(n))
    high = np.ascontiguousarray(close * 1.002)
    low = np.ascontiguousarray(close * 0.998)
    values = np.asarray(
        price_change_oscillator(high, low, close, short_length, multiplier)
    )
    # The residual is the normal_cdf polynomial's error at 0, not the window.
    np.testing.assert_allclose(values[-20:], 0.0, atol=1e-6)


def test_adx_is_zero_when_directional_movement_is_symmetric():
    """Each bar makes an equal new high and new low, so +DM and -DM tie on
    every bar and net directional movement is zero. Counting a tie as an up
    move reported maximum trend strength instead."""
    n = 200
    mid = np.full(n, 100.0)
    high = np.ascontiguousarray(mid + np.arange(n) * 0.5)
    low = np.ascontiguousarray(mid - np.arange(n) * 0.5)
    close = np.ascontiguousarray(mid)
    values = np.asarray(adx(high, low, close, 14))
    np.testing.assert_allclose(values[30:], 0.0, atol=1e-9)


@pytest.mark.parametrize(
    "fn", [linear_deviation, quadratic_deviation, cubic_deviation]
)
def test_deviation_is_zero_on_a_flat_window(fn):
    """A window of identical prices has no deviation from its own trend. The
    residual RMS is float rounding noise, and dividing by it amplified that
    noise to a full-scale reading."""
    lookback = 20
    values = np.ascontiguousarray(np.full(100, 50.0))
    np.testing.assert_allclose(np.asarray(fn(values, lookback)), 0.0, atol=0)


def test_delta_on_balance_volume_emits_no_undifferenced_levels():
    """The bars below the differenced front are levels that were never
    differenced, and used to be left in among genuine deltas."""
    _, _, _, close, volume = _ohlcv()
    lookback, delta_length = 21, 5
    levels = np.asarray(normalized_on_balance_volume(close, volume, lookback))
    deltas = np.asarray(
        delta_on_balance_volume(close, volume, lookback, delta_length)
    )
    overlap = [
        i
        for i in range(len(deltas))
        if deltas[i] != 0 and deltas[i] == levels[i]
    ]
    assert not overlap


def test_reactivity_alpha_stays_a_valid_ema_coefficient():
    """alpha = 2 / (lookback * smoothing + 1) exceeds 1 for small products,
    which amplifies instead of smoothing and inverts the indicator."""
    _, high, low, close, volume = _ohlcv()
    for lookback in (1, 5, 10, 20):
        for smoothing in (0.0, 0.1, 0.5, 1.0, 2.0):
            values = np.asarray(
                reactivity(high, low, close, volume, lookback, smoothing)
            )
            assert np.all(np.isfinite(values))
            assert values.min() >= -50 and values.max() <= 50


def test_cubic_trend_is_not_the_linear_trend():
    """_legendre_3's degree-3 basis degenerates at lookback 3, so cubic_trend
    silently returned linear_trend."""
    _, high, low, close, _ = _ohlcv()
    cubic = np.asarray(cubic_trend(close, high, low, close, 3, 5))
    linear = np.asarray(linear_trend(close, high, low, close, 3, 5))
    assert not np.array_equal(cubic, linear)


@pytest.mark.parametrize("lookback", [1, 2, 3])
def test_trend_short_lookback_does_not_raise(lookback):
    _, high, low, close, _ = _ohlcv(n=60)
    for fn in (linear_trend, quadratic_trend, cubic_trend):
        assert len(fn(close, high, low, close, lookback, 5)) == 60


def test_stochastic_rsi_degenerate_window_does_not_read_out_of_bounds():
    """front_bad is clamped up to n, so seeding the smoother from
    output[front_bad] read one element past the end."""
    _, _, _, close, _ = _ohlcv(n=20)
    assert len(stochastic_rsi(close, 15, 10, 3.0)) == 20


def test_returnv_when_base_is_zero_then_undefined():
    """A zero base price has no defined return; it used to raise."""
    values = np.ascontiguousarray([0.0, 1.0, 2.0, 3.0])
    result = np.asarray(returnv(values, 1))
    assert np.isnan(result[1])
    np.testing.assert_allclose(result[2:], [1.0, 0.5])


def test_adx_computes_at_exactly_two_lookbacks_of_data():
    """The insufficient-data guard was one bar too strict."""
    _, high, low, close, _ = _ohlcv(n=28)
    assert np.count_nonzero(np.asarray(adx(high, low, close, 14)))


def test_normalized_volume_index_does_not_fabricate_zero_from_nan():
    """A NaN in the volatility window made the value 0.0, indistinguishable
    from a genuine neutral reading, for the whole window."""
    _, _, _, close, volume = _ohlcv(n=400)
    close = close.copy()
    close[300] = np.nan
    for fn in (
        normalized_positive_volume_index,
        normalized_negative_volume_index,
    ):
        values = np.asarray(fn(close, volume, 20))
        assert np.isnan(values[300:]).any()
        assert not (values[300:] == 0).any()


def _indicator_args(fn_name, open_, high, low, close, volume):
    """Argument sets for the no-lookahead sweep, keyed by function name."""
    return {
        "adx": ((high, low, close, 14), {}),
        "aroon_up": ((high, low, 25), {}),
        "aroon_down": ((high, low, 25), {}),
        "aroon_diff": ((high, low, 25), {}),
        "close_minus_ma": ((high, low, close, 20, 10), {}),
        "cubic_deviation": ((close, 20), {}),
        "cubic_trend": ((close, high, low, close, 20, 10), {}),
        "delta_on_balance_volume": ((close, volume, 21, 5), {}),
        "detrended_rsi": ((close, 10, 30, 20), {}),
        "intraday_intensity": ((high, low, close, volume, 20), {}),
        "laguerre_rsi": ((open_, high, low, close, 13), {}),
        "linear_deviation": ((close, 20), {}),
        "linear_trend": ((close, high, low, close, 20, 10), {}),
        "macd": ((high, low, close, 5, 20), {}),
        "money_flow": ((high, low, close, volume, 20), {}),
        "normalized_negative_volume_index": ((close, volume, 20), {}),
        "normalized_on_balance_volume": ((close, volume, 20), {}),
        "normalized_positive_volume_index": ((close, volume, 20), {}),
        # short_length of 1 is the parameterization whose short-term loop used
        # to reach index 0 and read close[-1]; the small scale keeps the early
        # bars off the normal_cdf ceiling, where the leak is invisible.
        "price_change_oscillator": ((high, low, close, 1, 2), {"scale": 0.05}),
        "price_intensity": ((open_, high, low, close, 20), {}),
        "price_volume_fit": ((close, volume, 20), {}),
        "quadratic_deviation": ((close, 20), {}),
        "quadratic_trend": ((close, high, low, close, 20, 10), {}),
        "reactivity": ((high, low, close, volume, 20), {}),
        "returnv": ((close, 5), {}),
        "stochastic": ((high, low, close, 20), {}),
        "stochastic_rsi": ((close, 20, 20), {}),
        "volume_momentum": ((volume, 20, 3), {}),
        "volume_weighted_ma_ratio": ((close, volume, 20, 1.0), {}),
    }.get(fn_name)


@pytest.mark.parametrize(
    "fn_name",
    [
        "adx",
        "aroon_up",
        "aroon_down",
        "aroon_diff",
        "close_minus_ma",
        "cubic_deviation",
        "cubic_trend",
        "delta_on_balance_volume",
        "detrended_rsi",
        "intraday_intensity",
        "laguerre_rsi",
        "linear_deviation",
        "linear_trend",
        "macd",
        "money_flow",
        "normalized_negative_volume_index",
        "normalized_on_balance_volume",
        "normalized_positive_volume_index",
        "price_change_oscillator",
        "price_intensity",
        "price_volume_fit",
        "quadratic_deviation",
        "quadratic_trend",
        "reactivity",
        "returnv",
        "stochastic",
        "stochastic_rsi",
        "volume_momentum",
        "volume_weighted_ma_ratio",
    ],
)
def test_indicator_does_not_look_ahead(fn_name):
    """No indicator value may depend on a future bar.

    Changing only the FINAL bar must leave every earlier output untouched.
    This is the check that catches an inner loop reading a negative index,
    which wraps around to the end of the series.
    """
    import pybroker.vect as vect_module

    fn = getattr(vect_module, fn_name)
    open_, high, low, close, volume = _ohlcv(n=400)
    args, kwargs = _indicator_args(fn_name, open_, high, low, close, volume)
    before = np.asarray(fn(*args, **kwargs), dtype=np.float64).copy()

    bumped = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            arg = arg.copy()
            arg[-1] *= 1.5
        bumped.append(arg)
    after = np.asarray(fn(*bumped, **kwargs), dtype=np.float64)

    head_before, head_after = before[:-1], after[:-1]
    mismatch = ~(
        np.isclose(head_before, head_after, rtol=0, atol=0, equal_nan=True)
    )
    assert not mismatch.any(), (
        f"{fn_name} leaked the final bar into "
        f"{int(mismatch.sum())} earlier output(s), first at index "
        f"{int(np.argmax(mismatch))}"
    )


# Adapted from v2_preview's TestRollingWindowKernels: these two are
# kernel-independent and pin the NaN semantics of the brute-force
# lowv/highv/sumv. (The exact-summation and fsum-parity kernel tests do not
# apply to the np.sum-based implementations here.)
_ADVERSARIAL_ARRAYS = [
    [np.nan, 10, 20, 5, 30, 15, 25],
    [1, np.nan, 3, 2, np.nan, 5, 0.5, 4, np.nan, 2],
    [1, 2, 3, np.nan],
    [np.nan, np.nan, np.nan],
    [1, np.inf, 3, -np.inf, 2, 5],
    [np.inf, np.inf, 1, 2],
    [3, 3, 3, 3, 3],
    [1, 2, 3, 4, 5, 6],
    [6, 5, 4, 3, 2, 1],
    [1, 1, 2, 2, 1, 1, 2, 2],
    [1e16, 1, -1e16, 2, 1e16, 3],
    [7],
]


@pytest.mark.parametrize("array", _ADVERSARIAL_ARRAYS)
def test_lowv_never_exceeds_highv(array):
    arr = np.array(array, dtype=np.float64)
    for n in range(1, len(arr) + 1):
        low = np.asarray(lowv(arr, n))
        high = np.asarray(highv(arr, n))
        both = ~np.isnan(low) & ~np.isnan(high)
        assert np.all(low[both] <= high[both]), f"n={n}"


def test_sumv_when_nan_then_later_windows_unaffected():
    # A running accumulator cannot hold the NaN: nan - nan is nan, so one
    # NaN would otherwise blank every remaining window.
    arr = np.array([1.0, np.nan, 3.0, 2.0, 5.0], dtype=np.float64)
    np.testing.assert_allclose(
        sumv(arr, 1), arr, rtol=0, atol=0, equal_nan=True
    )
    np.testing.assert_allclose(
        sumv(arr, 2),
        [np.nan, np.nan, np.nan, 5.0, 7.0],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
