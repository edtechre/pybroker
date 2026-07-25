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


def _brute_lowv(array: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(array), np.nan)
    for i in range(n, len(array) + 1):
        out[i - 1] = np.min(array[i - n : i])
    return out


def _brute_highv(array: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(array), np.nan)
    for i in range(n, len(array) + 1):
        out[i - 1] = np.max(array[i - n : i])
    return out


def _brute_sumv(array: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(array), np.nan)
    for i in range(n, len(array) + 1):
        out[i - 1] = np.sum(array[i - n : i])
    return out


class TestRollingWindowKernels:
    @pytest.mark.parametrize(
        "array, n",
        [
            ([3, 3, 4, 2, 5, 6, 1, 3], 3),
            ([3, 3, 4, 2, 5, 6, 1, 3], 1),
            ([4, 3, 2, 1], 4),
            ([1], 1),
        ],
    )
    def test_rolling_kernels_match_brute_force_fixtures(self, array, n):
        arr = np.array(array, dtype=np.float64)
        np.testing.assert_allclose(
            lowv(arr, n), _brute_lowv(arr, n), rtol=0, atol=0, equal_nan=True
        )
        np.testing.assert_allclose(
            highv(arr, n), _brute_highv(arr, n), rtol=0, atol=0, equal_nan=True
        )
        np.testing.assert_allclose(
            sumv(arr, n), _brute_sumv(arr, n), rtol=0, atol=0, equal_nan=True
        )

    @pytest.mark.parametrize(
        "length, window",
        [
            (100, 2),
            (100, 20),
            (100, 50),
            (10_000, 2),
            (10_000, 20),
            (10_000, 50),
            (10_000, 200),
        ],
    )
    def test_rolling_kernels_match_brute_force_random(self, length, window):
        rng = np.random.default_rng(42 + length + window)
        arr = rng.standard_normal(length)
        np.testing.assert_allclose(
            lowv(arr, window),
            _brute_lowv(arr, window),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            highv(arr, window),
            _brute_highv(arr, window),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            sumv(arr, window),
            _brute_sumv(arr, window),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def _brute_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    smoothing: int = 0,
) -> np.ndarray:
    """Pre-optimization reference for stochastic regression tests."""
    n = len(close)
    front_bad = lookback - 1
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        min_val = 1.0e60
        max_val = -1.0e60
        for j in range(lookback):
            if high[i - j] > max_val:
                max_val = high[i - j]
            if low[i - j] < min_val:
                min_val = low[i - j]
        sto_0 = (close[i] - min_val) / (max_val - min_val + 1.0e-60)
        if smoothing == 0:
            output[i] = 100.0 * sto_0 - 50
        else:
            if i == front_bad:
                sto_1 = sto_0
                output[i] = 100.0 * sto_0 - 50
            else:
                sto_1 = 0.33333333 * sto_0 + 0.66666667 * sto_1
                if smoothing == 1:
                    output[i] = 100.0 * sto_1 - 50
                else:
                    if i == front_bad + 1:
                        sto_2 = sto_1
                        output[i] = 100.0 * sto_1 - 50
                    else:
                        sto_2 = 0.33333333 * sto_1 + 0.66666667 * sto_2
                        output[i] = 100.0 * sto_2 - 50
    return output


class TestStochasticKernels:
    @pytest.mark.parametrize(
        "high, low, close, lookback, smoothing",
        [
            (
                [10, 12, 11, 13, 12, 14],
                [8, 9, 8, 10, 9, 11],
                [9, 11, 10, 12, 11, 13],
                3,
                0,
            ),
            (
                [10, 12, 11, 13, 12, 14],
                [8, 9, 8, 10, 9, 11],
                [9, 11, 10, 12, 11, 13],
                5,
                1,
            ),
            (
                [10, 12, 11, 13, 12, 14],
                [8, 9, 8, 10, 9, 11],
                [9, 11, 10, 12, 11, 13],
                5,
                2,
            ),
        ],
    )
    def test_stochastic_matches_brute_force_fixtures(
        self, high, low, close, lookback, smoothing
    ):
        high_arr = np.array(high, dtype=np.float64)
        low_arr = np.array(low, dtype=np.float64)
        close_arr = np.array(close, dtype=np.float64)
        expected = _brute_stochastic(
            high_arr, low_arr, close_arr, lookback, smoothing
        )
        result = stochastic(high_arr, low_arr, close_arr, lookback, smoothing)
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "length, lookback, smoothing",
        [
            (100, 5, 0),
            (100, 20, 1),
            (100, 50, 2),
            (10_000, 5, 0),
            (10_000, 20, 1),
            (10_000, 200, 2),
        ],
    )
    def test_stochastic_matches_brute_force_random(
        self, length, lookback, smoothing
    ):
        rng = np.random.default_rng(42 + length + lookback + smoothing)
        close = rng.standard_normal(length) + 100.0
        high = close + rng.uniform(0.1, 2.0, length)
        low = close - rng.uniform(0.1, 2.0, length)
        expected = _brute_stochastic(high, low, close, lookback, smoothing)
        result = stochastic(high, low, close, lookback, smoothing)
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)

    def test_stochastic_empty_arrays(self):
        high = np.array([], dtype=np.float64)
        low = np.array([], dtype=np.float64)
        close = np.array([], dtype=np.float64)
        result = stochastic(high, low, close, 5, 0)
        assert len(result) == 0

    def test_stochastic_single_bar(self):
        high = np.array([10.0])
        low = np.array([8.0])
        close = np.array([9.0])
        result = stochastic(high, low, close, 5, 0)
        np.testing.assert_array_equal(result, np.zeros(1))

    def test_stochastic_lookback_greater_than_length(self):
        high = np.array([10.0, 12.0, 11.0])
        low = np.array([8.0, 9.0, 8.0])
        close = np.array([9.0, 11.0, 10.0])
        result = stochastic(high, low, close, 5, 1)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_stochastic_lookback_equals_length(self):
        high = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
        low = np.array([8.0, 9.0, 8.0, 10.0, 9.0])
        close = np.array([9.0, 11.0, 10.0, 12.0, 11.0])
        expected = _brute_stochastic(high, low, close, 5, 0)
        result = stochastic(high, low, close, 5, 0)
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)


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
