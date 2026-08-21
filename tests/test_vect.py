"""Unit tests for vect.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import math
import numpy as np
import pytest
import re
from pybroker.vect import (
    adx,
    aroon_diff,
    aroon_down,
    aroon_up,
    atr,
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


@pytest.mark.parametrize(
    "array, n, expected",
    [
        (
            [1, 1.5, 1.7, 1.3, 1.2, 1.4],
            1,
            [
                np.nan,
                np.log(1.5 / 1),
                np.log(1.7 / 1.5),
                np.log(1.3 / 1.7),
                np.log(1.2 / 1.3),
                np.log(1.4 / 1.2),
            ],
        ),
        (
            [1, 1.5, 1.7, 1.3, 1.2, 1.4],
            2,
            [
                np.nan,
                np.nan,
                np.log(1.7 / 1),
                np.log(1.3 / 1.5),
                np.log(1.2 / 1.7),
                np.log(1.4 / 1.3),
            ],
        ),
        ([1], 1, [np.nan]),
        ([], 5, []),
    ],
)
def test_returnv_when_use_log(array, n, expected):
    assert np.array_equal(
        np.round(returnv(np.array(array), n, True), 6),
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


def _reference_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int
) -> np.ndarray:
    """Textbook ATR: rolling mean of true range, where true range is the
    greatest of ``high - low``, ``abs(high - prev close)``, and
    ``abs(low - prev close)``. Bar 0 has no previous close, so outputs start
    once a full window of defined true ranges exists.
    """
    n = len(close)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        out[i] = np.mean(tr[i - lookback + 1 : i + 1])
    return out


@pytest.mark.parametrize("lookback", [1, 2, 14, 99, 100])
def test_atr_matches_reference(lookback):
    n = 100
    base = np.random.rand(n) * 10 + 10
    spread = np.random.rand(n) + 0.1
    high = base + spread
    low = base - spread
    close = base + (np.random.rand(n) - 0.5) * spread
    result = atr(high, low, close, lookback)
    assert np.allclose(
        result, _reference_atr(high, low, close, lookback), equal_nan=True
    )
    assert np.all(np.isnan(result[:lookback]))
    if lookback < n:
        assert not np.any(np.isnan(result[lookback:]))


def test_atr_when_prev_close_gaps_outside_bar_range():
    high = np.array([10.0, 5.0, 30.0])
    low = np.array([9.0, 4.0, 29.0])
    close = np.array([9.5, 4.5, 29.5])
    result = atr(high, low, close, 1)
    assert np.isnan(result[0])
    assert result[1] == 5.5
    assert result[2] == 25.5


def test_atr_when_empty_then_empty():
    empty = np.array([])
    assert not len(atr(empty, empty, empty, 5))


@pytest.mark.parametrize(
    "n, expected_msg",
    [
        (10, "n is greater than array length."),
        (0, "n needs to be >= 1."),
        (-1, "n needs to be >= 1."),
    ],
)
def test_atr_when_n_invalid_then_error(n, expected_msg):
    array = np.array([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError, match=re.escape(expected_msg)):
        atr(array, array, array, n)


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


def _exact_sumv(array: np.ndarray, n: int) -> np.ndarray:
    """Like :func:`._brute_sumv` but exact for finite windows.

    ``np.sum`` adds left to right, so it drops low-order bits when a window
    mixes magnitudes; ``math.fsum`` does not. Non-finite windows fall back to
    ``np.sum`` because ``fsum`` raises on them.
    """
    out = np.full(len(array), np.nan)
    for i in range(n, len(array) + 1):
        window = array[i - n : i]
        if np.all(np.isfinite(window)):
            out[i - 1] = math.fsum(window)
        else:
            # inf + -inf is a legitimate NaN result here, not a test failure.
            with np.errstate(invalid="ignore"):
                out[i - 1] = np.sum(window)
    return out


# Inputs that a monotonic-deque or running-sum kernel is most likely to get
# wrong: non-finite values interior to the series, runs that never evict, and
# magnitudes far enough apart to cancel.
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

    @pytest.mark.parametrize("array", _ADVERSARIAL_ARRAYS)
    def test_rolling_kernels_match_brute_force_when_non_finite(self, array):
        arr = np.array(array, dtype=np.float64)
        for n in range(1, len(arr) + 1):
            np.testing.assert_allclose(
                lowv(arr, n),
                _brute_lowv(arr, n),
                rtol=0,
                atol=0,
                equal_nan=True,
                err_msg=f"lowv n={n}",
            )
            np.testing.assert_allclose(
                highv(arr, n),
                _brute_highv(arr, n),
                rtol=0,
                atol=0,
                equal_nan=True,
                err_msg=f"highv n={n}",
            )
            np.testing.assert_allclose(
                sumv(arr, n),
                _exact_sumv(arr, n),
                rtol=0,
                atol=0,
                equal_nan=True,
                err_msg=f"sumv n={n}",
            )

    @pytest.mark.parametrize("array", _ADVERSARIAL_ARRAYS)
    def test_lowv_never_exceeds_highv(self, array):
        arr = np.array(array, dtype=np.float64)
        for n in range(1, len(arr) + 1):
            low = np.asarray(lowv(arr, n))
            high = np.asarray(highv(arr, n))
            both = ~np.isnan(low) & ~np.isnan(high)
            assert np.all(low[both] <= high[both]), f"n={n}"

    def test_sumv_when_nan_then_later_windows_unaffected(self):
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

    def test_sumv_when_mixed_magnitudes_then_exact(self):
        arr = np.array([1e16, 1.0, -1e16, 2.0, 1e16, 3.0], dtype=np.float64)
        assert sumv(arr, 1)[1] == 1.0
        assert sumv(arr, 3)[2] == 1.0
        assert sumv(arr, 5)[5] == 6.0

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_rolling_kernels_when_composed_with_returnv(self, n):
        # returnv emits a leading NaN by design, so composing it with these
        # kernels is the most common way a NaN reaches them.
        close = np.array(
            [10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0], dtype=np.float64
        )
        returns = returnv(close)
        for fnv, brute in (
            (lowv, _brute_lowv),
            (highv, _brute_highv),
            (sumv, _exact_sumv),
        ):
            actual = np.asarray(fnv(returns, n))
            np.testing.assert_allclose(
                actual, brute(returns, n), rtol=0, atol=0, equal_nan=True
            )
            # Only the warmup, not the whole series, may be NaN.
            assert not np.isnan(actual[n:]).any()


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


def test_returnv_when_use_log_and_non_positive_then_undefined():
    """A log return is undefined at or below zero, and must be NaN rather
    than the -inf np.log yields at zero, which a rolling window could never
    recover from."""
    values = np.ascontiguousarray([0.0, 1.0, 0.0, -1.0, 2.0, 4.0])
    result = np.asarray(returnv(values, 1, True))
    # Zero base, zero value, negative value, negative base.
    assert np.isnan(result[1:5]).all()
    assert not np.isinf(result).any()
    np.testing.assert_allclose(result[5], np.log(2.0))


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
    """Argument sets for the no-lookahead sweep, keyed by function name.

    A ``"fn[variant]"`` key sweeps one kernel a second time under a
    different flag setting; the function is resolved from the part before
    the bracket.
    """
    return {
        "adx": ((high, low, close, 14), {}),
        "aroon_up": ((high, low, 25), {}),
        "aroon_down": ((high, low, 25), {}),
        "aroon_diff": ((high, low, 25), {}),
        "atr": ((high, low, close, 14), {}),
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
        "returnv[log]": ((close, 5), {"use_log": True}),
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
        "atr",
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
        "returnv[log]",
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

    # A "fn[variant]" key sweeps one kernel under a second flag setting.
    fn = getattr(vect_module, fn_name.partition("[")[0])
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
