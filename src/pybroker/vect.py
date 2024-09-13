"""Contains vectorized utility functions."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Literal


@njit
def _verify_input(array: NDArray[np.float64], n: int):
    assert n > 0, "n needs to be >= 1."
    assert n <= len(array), "n is greater than array length."


@njit
def lowv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Calculates the lowest values for every ``n`` period in ``array``.

    Args:
        array: :class:`numpy.ndarray` of data.
        n: Length of period.

    Returns:
        :class:`numpy.ndarray` of the lowest values for every ``n`` period in
        ``array``.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.min(array[i - n : i])
    return out


@njit
def highv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Calculates the highest values for every ``n`` period in ``array``.

    Args:
        array: :class:`numpy.ndarray` of data.
        n: Length of period.

    Returns:
        :class:`numpy.ndarray` of the highest values for every ``n`` period in
        ``array``.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.max(array[i - n : i])
    return out


@njit
def sumv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Calculates the sums for every ``n`` period in ``array``.

    Args:
        array: :class:`numpy.ndarray` of data.
        n: Length of period.

    Returns:
        :class:`numpy.ndarray` of the sums for every ``n`` period in ``array``.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.sum(array[i - n : i])
    return out


@njit
def returnv(array: NDArray[np.float64], n: int = 1) -> NDArray[np.float64]:
    """Calculates returns.

    Args:
        n: Return period. Defaults to 1.

    Returns:
        :class:`numpy.ndarray` of returns.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len):
        out[i] = (array[i] - array[i - n]) / array[i - n]
    return out


@njit
def cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Checks for crossover of ``a`` above ``b``.

    Args:
        a: :class:`numpy.ndarray` of data.
        b: :class:`numpy.ndarray` of data.

    Returns:
        :class:`numpy.ndarray` containing values of ``1`` when ``a`` crosses
        above ``b``, otherwise values of ``0``.
    """
    assert len(a), "a cannot be empty."
    assert len(b), "b cannot be empty."
    assert len(a) == len(b), "a and b must be same length."
    assert len(a) >= 2, "a and b must have length >= 2."
    crossed = np.where(a > b, 1, 0)
    return (sumv(crossed > 0, 2) == 1) * crossed


@njit
def normal_cdf(z: float) -> float:
    """Computes the CDF of the standard normal distribution."""
    zz = np.fabs(z)
    pdf = np.exp(-0.5 * zz * zz) / np.sqrt(2 * np.pi)
    t = 1 / (1 + zz * 0.2316419)
    poly = (
        (((1.330274429 * t - 1.821255978) * t + 1.781477937) * t - 0.356563782)
        * t
        + 0.319381530
    ) * t
    return 1 - pdf * poly if z > 0 else pdf * poly


@njit
def inverse_normal_cdf(p: float) -> float:
    """Computes the inverse CDF of the standard normal distribution."""
    pp = p if p <= 0.5 else 1 - p
    if pp == 0:
        pp = 1.0e-10
    t = np.sqrt(np.log(1 / (pp * pp)))
    numer = (0.010328 * t + 0.802853) * t + 2.515517
    denom = ((0.001308 * t + 0.189269) * t + 1.432788) * t + 1
    x = t - numer / denom
    return -x if p <= 0.5 else x


@njit
def _atr(
    last_bar: int,
    lookback: int,
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    use_log: bool = False,
) -> float:
    """Computes Average True Range.

    Args:
        last_bar: Index of last bar for ATR calculation.
        lookback: Number of lookback bars.
        high: High prices.
        low: Low prices.
        close: Close prices.
        use_log: Whether to log transform. Defaults to ``False``.

    Returns:
        The computed ATR.
    """
    assert last_bar >= lookback
    if lookback == 0:
        if use_log:
            return np.log(high[last_bar] / low[last_bar])
        else:
            return high[last_bar] - low[last_bar]
    total = 0.0
    for i in range(last_bar - lookback + 1, last_bar + 1):
        if use_log:
            term = high[i] / low[i]
            if high[i] / close[i - 1] > term:
                term = high[i] / close[i - 1]
            if close[i - 1] / low[i] > term:
                term = close[i - 1] / low[i]
            total += np.log(term)
        else:
            term = high[i] - low[i]
            if high[i] - close[i - 1] > term:
                term = high[i] - close[i - 1]
            if close[i - 1] - low[i] > term:
                term = close[i - 1] - low[i]
            total += term
    return total / lookback


@njit
def _variance(
    use_change: bool, last_bar: int, length: int, prices: NDArray[np.float64]
) -> float:
    if use_change:
        assert last_bar >= length
    else:
        assert last_bar >= length - 1
    total = 0.0
    for i in range(last_bar - length + 1, last_bar + 1):
        if use_change:
            term = np.log(prices[i] / prices[i - 1])
        else:
            term = np.log(prices[i])
        total += term
    mean = total / length
    total = 0.0
    for i in range(last_bar - length + 1, last_bar + 1):
        if use_change:
            term = np.log(prices[i] / prices[i - 1]) - mean
        else:
            term = np.log(prices[i]) - mean
        total += term * term
    return total / length


@njit
def detrended_rsi(
    values: NDArray[np.float64],
    short_length: int,
    long_length: int,
    reg_length: int,
) -> NDArray[np.float64]:
    """Computes Detrended Relative Strength Index (RSI).

    Args:
        values: :class:`numpy.ndarray` of input.
        short_length: Lookback for the short-term RSI.
        long_length: Lookback for the long-term RSI.
        reg_length: Number of bars used for linear regressions.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    assert short_length > 0
    assert short_length <= long_length
    assert short_length > 1
    assert long_length > 1
    assert reg_length >= 1
    n = len(values)
    front_bad = long_length + reg_length - 1
    output = np.zeros(n)
    if front_bad >= n:
        return output
    work1 = np.zeros(n)
    for i in range(short_length):
        work1[i] = 1.0e90
    up_sum = dn_sum = 1.0e-60
    for i in range(1, short_length):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum += diff
        else:
            dn_sum -= diff
    up_sum /= short_length - 1
    dn_sum /= short_length - 1
    for i in range(short_length, n):
        diff = values[i] - values[i - 1]
        if diff > 0:
            up_sum = ((short_length - 1.0) * up_sum + diff) / short_length
            dn_sum *= (short_length - 1.0) / short_length
        else:
            dn_sum = ((short_length - 1.0) * dn_sum - diff) / short_length
            up_sum *= (short_length - 1.0) / short_length
        work1[i] = 100.0 * up_sum / (up_sum + dn_sum)
        if short_length == 2:
            work1[i] = -10.0 * np.log(
                2.0 / (1 + 0.00999 * (2 * work1[i] - 100)) - 1
            )
    work2 = np.zeros(n)
    for i in range(long_length):
        work2[i] = -1.0e90
    up_sum = dn_sum = 1.0e-60
    for i in range(1, long_length):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum += diff
        else:
            dn_sum -= diff
    up_sum /= long_length - 1
    dn_sum /= long_length - 1
    for i in range(long_length, n):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum = ((long_length - 1.0) * up_sum + diff) / long_length
            dn_sum *= (long_length - 1.0) / long_length
        else:
            dn_sum = ((long_length - 1.0) * dn_sum - diff) / long_length
            up_sum *= (long_length - 1.0) / long_length
        work2[i] = 100.0 * up_sum / (up_sum + dn_sum)
    for i in range(front_bad, n):
        x_mean = y_mean = 0.0
        for j in range(reg_length):
            k = i - j
            x_mean += work2[k]
            y_mean += work1[k]
        x_mean /= reg_length
        y_mean /= reg_length
        xss = xy = 0.0
        for j in range(reg_length):
            k = i - j
            x_diff = work2[k] - x_mean
            y_diff = work1[k] - y_mean
            xss += x_diff * x_diff
            xy += x_diff * y_diff
        coef = xy / (xss + 1.0e-60)
        x_diff = work2[i] - x_mean
        y_diff = work1[i] - y_mean
        output[i] = y_diff - coef * x_diff
    return output


@njit
def macd(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    short_length: int,
    long_length: int,
    smoothing: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Computes Moving Average Convergence Divergence.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        short_length: Short-term lookback.
        long_length: Long-term lookback.
        smoothing: Compute MACD minus smoothed if >= 2.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert short_length > 0
    assert short_length <= long_length
    assert smoothing >= 0
    assert scale > 0
    n = len(close)
    output = np.zeros(n)
    long_alpha = 2.0 / (long_length + 1.0)
    short_alpha = 2.0 / (short_length + 1.0)
    long_sum = short_sum = close[0]
    for i in range(1, n):
        long_sum = long_alpha * close[i] + (1.0 - long_alpha) * long_sum
        short_sum = short_alpha * close[i] + (1.0 - short_alpha) * short_sum
        diff = 0.5 * (long_length - 1.0)
        diff -= 0.5 * (short_length - 1.0)
        denom = np.sqrt(np.fabs(diff))
        k = long_length + smoothing
        if k > i:
            k = i
        denom *= _atr(i, k, high, low, close, False)
        output[i] = (short_sum - long_sum) / (denom + 1.0e-15)
        output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
    if smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = output[0]
        for i in range(1, n):
            smoothed = alpha * output[i] + (1.0 - alpha) * smoothed
            output[i] -= smoothed
    return output


@njit
def stochastic(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
    smoothing: int = 0,
) -> NDArray[np.float64]:
    """Computes Stochastic.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        lookback: Number of lookback bars.
        smoothing: Number of times the raw stochastic is smoothed, either 0,
            1, or 2 times. Defaults to ``0``.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert lookback > 0
    assert smoothing == 0 or smoothing == 1 or smoothing == 2
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


@njit
def stochastic_rsi(
    values: NDArray[np.float64],
    rsi_lookback: int,
    sto_lookback: int,
    smoothing: float = 0.0,
) -> NDArray[np.float64]:
    """Computes Stochastic Relative Strength Index (RSI).

    Args:
        values: :class:`numpy.ndarray` of input.
        rsi_lookback: Lookback length for RSI calculation.
        sto_lookback: Lookback length for Stochastic calculation.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    assert rsi_lookback > 0
    assert sto_lookback > 0
    assert smoothing >= 0
    n = len(values)
    front_bad = rsi_lookback + sto_lookback - 1
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    if rsi_lookback >= n:
        return output
    for i in range(front_bad):
        output[i] = 0
    up_sum = dn_sum = 1.0e-60
    for i in range(1, rsi_lookback):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum += diff
        else:
            dn_sum -= diff
    up_sum /= rsi_lookback - 1
    dn_sum /= rsi_lookback - 1
    work1 = np.zeros(n)
    for i in range(rsi_lookback, n):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum = ((rsi_lookback - 1) * up_sum + diff) / rsi_lookback
            dn_sum *= (rsi_lookback - 1.0) / rsi_lookback
        else:
            dn_sum = ((rsi_lookback - 1) * dn_sum - diff) / rsi_lookback
            up_sum *= (rsi_lookback - 1.0) / rsi_lookback
        work1[i] = 100.0 * up_sum / (up_sum + dn_sum)
    for i in range(front_bad, n):
        min_val = 1.0e60
        max_val = -1.0e60
        for j in range(sto_lookback):
            if work1[i - j] > max_val:
                max_val = work1[i - j]
            if work1[i - j] < min_val:
                min_val = work1[i - j]
        output[i] = (
            100.0 * (work1[i] - min_val) / (max_val - min_val + 1.0e-60) - 50.0
        )
    if smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = output[front_bad]
        for i in range(front_bad + 1, n):
            smoothed = alpha * output[i] + (1.0 - alpha) * smoothed
            output[i] = smoothed
    return output


@njit
def _legendre_1(n: int) -> NDArray[np.float64]:
    c1 = np.zeros(n)
    total = 0.0
    for i in range(n):
        c1[i] = 2.0 * i / (n - 1.0) - 1.0
        total += c1[i] * c1[i]
    total = np.sqrt(total)
    for i in range(n):
        c1[i] /= total
    return c1


@njit
def _legendre_2(n: int) -> tuple[NDArray, NDArray]:
    c1 = _legendre_1(n)
    c2 = np.zeros(n)
    total = 0.0
    for i in range(n):
        c2[i] = c1[i] * c1[i]
        total += c2[i]
    mean = total / n
    total = 0.0
    for i in range(n):
        c2[i] -= mean
        total += c2[i] * c2[i]
    total = np.sqrt(total)
    for i in range(n):
        c2[i] /= total
    return c1, c2


@njit
def _legendre_3(n: int) -> tuple[NDArray, NDArray, NDArray]:
    """Computes the first three Legendre polynomials.

    The first polynomial measures linear trend, the second measures the
    quadratic trend, and the third measures the cubic trend.

    Args:
        n: Length of result.

    Returns:
        Tuple of first three Legendre polynomials.
    """
    c1, c2 = _legendre_2(n)
    c3 = np.zeros(n)
    total = 0.0
    for i in range(n):
        c3[i] = c1[i] * c1[i] * c1[i]
        total += c3[i]
    mean = total / n
    total = 0.0
    for i in range(n):
        c3[i] -= mean
        total += c3[i] * c3[i]
    total = np.sqrt(total)
    for i in range(n):
        c3[i] /= total
    proj = 0.0
    for i in range(n):
        proj += c1[i] * c3[i]
    total = 0.0
    for i in range(n):
        c3[i] -= proj * c1[i]
        total += c3[i] * c3[i]
    total = np.sqrt(total)
    for i in range(n):
        c3[i] /= total
    return c1, c2, c3


@njit
def _trend(
    values: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
    atr_length: int,
    scale: float,
    trend_type: Literal["linear", "quadratic", "cubic"],
) -> NDArray[np.float64]:
    assert (
        len(values) == len(high)
        and len(values) == len(low)
        and len(values) == len(close)
    )
    assert lookback > 0
    assert atr_length > 0
    assert scale > 0
    n = len(values)
    front_bad = lookback - 1 if ((lookback - 1) > atr_length) else atr_length
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    dptr = None
    for i in range(front_bad, n):
        if trend_type == "linear":
            dptr = _legendre_1(lookback)
        elif trend_type == "quadratic":
            _, dptr = _legendre_2(lookback)
        else:
            _, _, dptr = _legendre_3(lookback)
        dptr_i = 0
        dot_prod = 0.0
        mean = 0.0
        for j in range(i - lookback + 1, i + 1):
            price = np.log(values[j])
            mean += price
            dot_prod += price * dptr[dptr_i]
            dptr_i += 1
        mean /= lookback
        dptr_i -= lookback
        k = lookback - 1
        if lookback == 2:
            k = 2
        denom = _atr(i, atr_length, high, low, close, True) * k
        output[i] = dot_prod * 2.0 / (denom + 1.0e-60)
        yss = rsq = 0.0
        for j in range(i - lookback + 1, i + 1):
            price = np.log(values[j])
            diff = price - mean
            yss += diff * diff
            pred = dot_prod * dptr[dptr_i]
            dptr_i += 1
            diff = diff - pred
            rsq += diff * diff
        rsq = 1 - rsq / (yss + 1.0e-60)
        if rsq < 0:
            rsq = 0
        output[i] *= rsq
        output[i] = 100 * normal_cdf(scale * output[i]) - 50
    return output


def linear_trend(
    values: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
    atr_length: int,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Computes Linear Trend Strength.

    Args:
        values: :class:`numpy.ndarray` of input.
        high: High prices.
        low: Low prices.
        close: Close prices.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _trend(
        values, high, low, close, lookback, atr_length, scale, "linear"
    )


def quadratic_trend(
    values: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
    atr_length: int,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Computes Quadratic Trend Strength.

    Args:
        values: :class:`numpy.ndarray` of input.
        high: High prices.
        low: Low prices.
        close: Close prices.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _trend(
        values, high, low, close, lookback, atr_length, scale, "quadratic"
    )


def cubic_trend(
    values: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
    atr_length: int,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Computes Cubic Trend Strength.

    Args:
        values: :class:`numpy.ndarray` of input.
        high: High prices.
        low: Low prices.
        close: Close prices.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _trend(
        values, high, low, close, lookback, atr_length, scale, "cubic"
    )


@njit
def adx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
) -> NDArray[np.float64]:
    """Computes Average Directional Movement Index.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        lookback: Number of lookback bars.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert lookback > 0
    n = len(close)
    output = np.zeros(n)
    if n <= 2 * lookback:
        return output
    output[0] = 0
    dms_plus = dms_minus = atr_ = 0.0
    for i in range(1, lookback + 1):
        dm_plus = high[i] - high[i - 1]
        dm_minus = low[i - 1] - low[i]
        if dm_plus >= dm_minus:
            dm_minus = 0.0
        else:
            dm_plus = 0.0
        if dm_plus < 0.0:
            dm_plus = 0.0
        if dm_minus < 0.0:
            dm_minus = 0.0
        dms_plus += dm_plus
        dms_minus += dm_minus
        term = high[i] - low[i]
        if high[i] - close[i - 1] > term:
            term = high[i] - close[i - 1]
        if close[i - 1] - low[i] > term:
            term = close[i - 1] - low[i]
        atr_ += term
        di_plus = dms_plus / (atr_ + 1.0e-10)
        di_minus = dms_minus / (atr_ + 1.0e-10)
        adx_ = np.fabs(di_plus - di_minus) / (di_plus + di_minus + 1.0e-10)
        output[i] = 100 * adx_
    for i in range(lookback + 1, 2 * lookback):
        dm_plus = high[i] - high[i - 1]
        dm_minus = low[i - 1] - low[i]
        if dm_plus >= dm_minus:
            dm_minus = 0.0
        else:
            dm_plus = 0.0
        if dm_plus < 0.0:
            dm_plus = 0.0
        if dm_minus < 0.0:
            dm_minus = 0.0
        dms_plus = (lookback - 1.0) / lookback * dms_plus + dm_plus
        dms_minus = (lookback - 1.0) / lookback * dms_minus + dm_minus
        term = high[i] - low[i]
        if high[i] - close[i - 1] > term:
            term = high[i] - close[i - 1]
        if close[i - 1] - low[i] > term:
            term = close[i - 1] - low[i]
        atr_ = (lookback - 1.0) / lookback * atr_ + term
        di_plus = dms_plus / (atr_ + 1.0e-10)
        di_minus = dms_minus / (atr_ + 1.0e-10)
        adx_ += np.fabs(di_plus - di_minus) / (di_plus + di_minus + 1.0e-10)
        output[i] = 100 * adx_ / (i - lookback + 1)
    adx_ /= lookback
    for i in range(2 * lookback, n):
        dm_plus = high[i] - high[i - 1]
        dm_minus = low[i - 1] - low[i]
        if dm_plus >= dm_minus:
            dm_minus = 0.0
        else:
            dm_plus = 0.0
        if dm_plus < 0.0:
            dm_plus = 0.0
        if dm_minus < 0.0:
            dm_minus = 0.0
        dms_plus = (lookback - 1.0) / lookback * dms_plus + dm_plus
        dms_minus = (lookback - 1.0) / lookback * dms_minus + dm_minus
        term = high[i] - low[i]
        if high[i] - close[i - 1] > term:
            term = high[i] - close[i - 1]
        if close[i - 1] - low[i] > term:
            term = close[i - 1] - low[i]
        atr_ = (lookback - 1.0) / lookback * atr_ + term
        di_plus = dms_plus / (atr_ + 1.0e-10)
        di_minus = dms_minus / (atr_ + 1.0e-10)
        term = np.fabs(di_plus - di_minus) / (di_plus + di_minus + 1.0e-10)
        adx_ = (lookback - 1.0) / lookback * adx_ + term / lookback
        output[i] = 100 * adx_
    return output


@njit
def _aroon(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    lookback: int,
    aroon_type: Literal["up", "down", "diff"],
) -> NDArray[np.float64]:
    assert len(high) == len(low)
    assert lookback > 0
    n = len(high)
    output = np.zeros(n)
    if aroon_type == "up" or aroon_type == "down":
        output[0] = 50
    elif aroon_type == "diff":
        output[0] = 0
    for i in range(1, n):
        if aroon_type == "up" or aroon_type == "diff":
            i_max = i
            x_max = high[i]
            for i in range(i - 1, i - lookback - 1, -1):
                if i < 0:
                    break
                if high[i] > x_max:
                    x_max = high[i]
                    i_max = i
        if aroon_type == "down" or aroon_type == "diff":
            i_min = i
            x_min = low[i]
            for i in range(i - 1, i - lookback - 1, -1):
                if i < 0:
                    break
                if low[i] < x_min:
                    x_min = low[i]
                    i_min = i
        if aroon_type == "up":
            output[i] = 100 * (lookback - (i - i_max)) / lookback
        elif aroon_type == "down":
            output[i] = 100 * (lookback - (i - i_min)) / lookback
        else:
            max_val = 100 * (lookback - (i - i_max)) / lookback
            min_val = 100 * (lookback - (i - i_min)) / lookback
            output[i] = max_val - min_val
    return output


@njit
def aroon_up(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    lookback: int,
) -> NDArray[np.float64]:
    """Computes Aroon Upward Trend.

    Args:
        high: High prices.
        low: Low prices.
        lookback: Number of lookback bars.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _aroon(high, low, lookback, "up")


@njit
def aroon_down(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    lookback: int,
) -> NDArray[np.float64]:
    """Computes Aroon Downward Trend.

    Args:
        high: High prices.
        low: Low prices.
        lookback: Number of lookback bars.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _aroon(high, low, lookback, "down")


@njit
def aroon_diff(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    lookback: int,
) -> NDArray[np.float64]:
    """Computes Aroon Upward Trend minus Aroon Downward Trend.

    Args:
        high: High prices.
        low: Low prices.
        lookback: Number of lookback bars.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _aroon(high, low, lookback, "diff")


@njit
def close_minus_ma(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
    atr_length: int,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Computes Close Minus Moving Average.

    Args:
        close: Close prices.
        high: High prices.
        low: Low prices.
        lookback: Number of lookback bars.
        atr_length: Lookback length used for Average True Range (ATR)
            normalization.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert lookback > 0
    assert atr_length > 0
    assert scale > 0
    n = len(close)
    front_bad = max(lookback, atr_length)
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        total = 0.0
        for j in range(i - lookback, i):
            total += np.log(close[j])
        total /= lookback
        denom = _atr(i, atr_length, high, low, close, True)
        if denom > 0.0:
            denom *= np.sqrt(lookback + 1.0)
            output[i] = (np.log(close[i]) - total) / denom
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def _deviation(
    values: NDArray[np.float64],
    lookback: int,
    scale: float,
    dev_type: Literal["linear", "quadratic", "cubic"],
) -> NDArray[np.float64]:
    assert lookback > 0
    assert scale > 0
    n = len(values)
    if dev_type == "linear" and lookback < 3:
        lookback = 3
    if dev_type == "quadratic" and lookback < 4:
        lookback = 4
    if dev_type == "cubic" and lookback < 5:
        lookback = 5
    front_bad = lookback - 1
    if front_bad > n:
        front_bad = n
    if dev_type == "quadratic" or dev_type == "cubic":
        work1, work2, work3 = _legendre_3(lookback)
    else:
        work1 = _legendre_1(lookback)
    output = np.zeros(n)
    for i in range(front_bad, n):
        c0 = c1 = c2 = c3 = 0.0
        dptr = work1
        dptr_i = 0
        for j in range(i - lookback + 1, i + 1):
            price = np.log(values[j])
            c0 += price
            c1 += price * dptr[dptr_i]
            dptr_i += 1
        c0 /= lookback
        if dev_type == "quadratic" or dev_type == "cubic":
            dptr = work2
            dptr_i = 0
            for j in range(i - lookback + 1, i + 1):
                price = np.log(values[j])
                c2 += price * dptr[dptr_i]
                dptr_i += 1
        if dev_type == "cubic":
            dptr = work3
            dptr_i = 0
            for j in range(i - lookback + 1, i + 1):
                price = np.log(values[j])
                c3 += price * dptr[dptr_i]
                dptr_i += 1
        j = 0
        total = 0.0
        for k in range(i - lookback + 1, i + 1):
            pred = c0 + c1 * work1[j]
            if dev_type == "quadratic" or dev_type == "cubic":
                pred += c2 * work2[j]
            if dev_type == "cubic":
                pred += c3 * work3[j]
            diff = np.log(values[k]) - pred
            total += diff * diff
            j += 1
        denom = np.sqrt(total / lookback)
        if denom > 0.0:
            pred = c0 + c1 * work1[lookback - 1]
            if dev_type == "quadratic" or dev_type == "cubic":
                pred += c2 * work2[lookback - 1]
            if dev_type == "cubic":
                pred += c3 * work3[lookback - 1]
            output[i] = (np.log(values[i]) - pred) / denom
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def linear_deviation(
    values: NDArray[np.float64],
    lookback: int,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """Computes Deviation from Linear Trend.

    Args:
        values: :class:`numpy.ndarray` of input.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _deviation(values, lookback, scale, "linear")


@njit
def quadratic_deviation(
    values: NDArray[np.float64],
    lookback: int,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """Computes Deviation from Quadratic Trend.

    Args:
        values: :class:`numpy.ndarray` of input.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _deviation(values, lookback, scale, "quadratic")


@njit
def cubic_deviation(
    values: NDArray[np.float64],
    lookback: int,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """Computes Deviation from Cubic Trend.

    Args:
        values: :class:`numpy.ndarray` of input.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _deviation(values, lookback, scale, "cubic")


@njit
def price_intensity(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    smoothing: float = 0.0,
    scale: float = 0.8,
) -> NDArray[np.float64]:
    """Computes Price Intensity.

    Args:
        open: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        smoothing: Amount of smoothing. Defaults to ``0``.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.8``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    assert (
        len(open) == len(high)
        and len(open) == len(low)
        and len(open) == len(close)
    )
    assert smoothing >= 0
    assert scale > 0
    n = len(close)
    if smoothing < 1:
        smoothing = 1
    output = np.zeros(n)
    denom = high[0] - low[0]
    if denom < 1.0e-60:
        denom = 1.0e-60
    output[0] = (close[0] - open[0]) / denom
    for i in range(1, n):
        denom = high[i] - low[i]
        if high[i] - close[i - 1] > denom:
            denom = high[i] - close[i - 1]
        if close[i - 1] - low[i] > denom:
            denom = close[i - 1] - low[i]
        if denom < 1.0e-60:
            denom = 1.0e-60
        output[i] = (close[i] - open[i]) / denom
    if smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = output[0]
        for i in range(1, n):
            smoothed = alpha * output[i] + (1.0 - alpha) * smoothed
            output[i] = smoothed
    for i in range(n):
        output[i] = (
            100.0 * normal_cdf(scale * np.sqrt(smoothing) * output[i]) - 50.0
        )
    return output


@njit
def price_change_oscillator(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    short_length: int,
    multiplier: int,
    scale: float = 4.0,
) -> NDArray[np.float64]:
    """Computes Price Change Oscillator.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        short_length: Number of short lookback bars.
        multiplier: Multiplier used to compute number of long lookback bars =
            ``multiplier * short_length``.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``4.0``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert short_length > 0
    assert multiplier > 0
    assert scale > 0
    n = len(close)
    if multiplier < 2:
        multiplier = 2
    long_length = short_length * multiplier
    front_bad = long_length
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        short_sum = 0.0
        for j in range(i - short_length - 1, i + 1):
            short_sum += np.fabs(np.log(close[j] / close[j - 1]))

        long_sum = short_sum
        for j in range(i - long_length + 1, i - short_length + 1):
            long_sum += np.fabs(np.log(close[j] / close[j - 1]))
        short_sum /= short_length
        long_sum /= long_length
        denom = 0.36 + 1.0 / short_length
        v = np.log(0.5 * multiplier) / 1.609
        denom += 0.7 * v
        denom *= _atr(i, long_length, high, low, close, True)
        if denom > 1.0e-20:
            output[i] = (short_sum - long_sum) / denom
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def _flow(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    smoothing: float,
    flow_type: Literal["intraday", "money_flow"],
) -> NDArray[np.float64]:
    assert (
        len(high) == len(low)
        and len(high) == len(close)
        and len(high) == len(volume)
    )
    assert lookback > 0
    assert smoothing >= 0
    n = len(close)
    front_bad = lookback - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(first_volume, n):
        if high[i] > low[i]:
            output[i] = (
                100.0
                * (2.0 * close[i] - high[i] - low[i])
                / (high[i] - low[i])
                * volume[i]
            )
        else:
            output[i] = 0.0
    if lookback > 1:
        for i in range(n - 1, front_bad - 1, -1):
            total = 0.0
            for j in range(lookback):
                total += output[i - j]
            output[i] = total / lookback
    if flow_type == "money_flow":
        for i in range(front_bad, n):
            total = 0.0
            for j in range(lookback):
                total += volume[i - j]
            total /= lookback
            if total > 0.0:
                output[i] /= total
            else:
                output[i] = 0.0
    elif smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = volume[first_volume]
        for i in range(first_volume, n):
            smoothed = alpha * volume[i] + (1.0 - alpha) * smoothed
            if smoothed > 0.0:
                output[i] /= smoothed
            else:
                output[i] = 0.0
    for i in range(front_bad):
        output[i] = 0.0
    return output


@njit
def intraday_intensity(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    smoothing: float = 0.0,
) -> NDArray[np.float64]:
    """Computes Intraday Intensity.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _flow(high, low, close, volume, lookback, smoothing, "intraday")


@njit
def money_flow(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    smoothing: float = 0.0,
) -> NDArray[np.float64]:
    """Computes Chaikin's Money Flow.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        smoothing: Amount of smoothing; <= 1 for none. Defaults to ``0``.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    return _flow(high, low, close, volume, lookback, smoothing, "money_flow")


@njit
def reactivity(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    smoothing: float = 0.0,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """Computes Reactivity.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        smoothing: Smoothing multiplier.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    assert (
        len(high) == len(low)
        and len(high) == len(close)
        and len(high) == len(volume)
    )
    assert lookback > 0
    assert smoothing >= 0
    assert scale > 0
    n = len(close)
    front_bad = lookback
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad):
        output[i] = 0.0
    alpha = 2.0 / (lookback * smoothing + 1)
    lowest = low[first_volume]
    highest = high[first_volume]
    smoothed_range = highest - lowest
    smoothed_volume = volume[first_volume]
    if smoothed_range == 0:
        return output
    if first_volume + 1 >= n or first_volume + lookback >= n:
        return output
    for i in range(first_volume + 1, first_volume + lookback):
        if high[i] > highest:
            highest = high[i]
        if low[i] < lowest:
            lowest = low[i]
        smoothed_range = (
            alpha * (highest - lowest) + (1.0 - alpha) * smoothed_range
        )
        smoothed_volume = alpha * volume[i] + (1.0 - alpha) * smoothed_volume
    for i in range(front_bad, n):
        lowest = low[i]
        highest = high[i]
        for j in range(1, lookback + 1):
            if high[i - j] > highest:
                highest = high[i - j]
            if low[i - j] < lowest:
                lowest = low[i - j]
        smoothed_range = (
            alpha * (highest - lowest) + (1.0 - alpha) * smoothed_range
        )
        smoothed_volume = alpha * volume[i] + (1.0 - alpha) * smoothed_volume
        aspect_ratio = (highest - lowest) / smoothed_range
        if volume[i] > 0.0 and smoothed_volume > 0.0:
            aspect_ratio /= volume[i] / smoothed_volume
        else:
            aspect_ratio = 1.0
        output[i] = aspect_ratio * (close[i] - close[i - lookback])
        output[i] /= smoothed_range
        output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
    return output


@njit
def price_volume_fit(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 9.0,
) -> NDArray[np.float64]:
    """Computes Price Volume Fit.

    Args:
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``9.0``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    assert len(close) == len(volume)
    assert lookback > 0
    assert scale > 0
    n = len(close)
    front_bad = lookback - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        x_mean = y_mean = 0.0
        for j in range(lookback):
            k = i - j
            x_mean += np.log(volume[k] + 1.0)
            y_mean += np.log(close[k])
        x_mean /= lookback
        y_mean /= lookback
        xss = xy = 0.0
        for j in range(lookback):
            k = i - j
            x_diff = np.log(volume[k] + 1.0) - x_mean
            y_diff = np.log(close[k]) - y_mean
            xss += x_diff * x_diff
            xy += x_diff * y_diff
        coef = xy / (xss + 1.0e-30)
        output[i] = 100.0 * normal_cdf(scale * coef) - 50.0
    return output


@njit
def volume_weighted_ma_ratio(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Computes Volume-Weighted Moving Average Ratio.

    Args:
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``1.0``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    assert len(close) == len(volume)
    assert lookback > 0
    assert scale > 0
    n = len(close)
    front_bad = lookback - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        total = numer = denom = 0.0
        for j in range(i - lookback + 1, i + 1):
            numer += volume[j] * close[j]
            denom += close[j]
            total += volume[j]
        if total > 0.0:
            output[i] = (
                1000.0
                * np.log(lookback * numer / (total * denom))
                / np.sqrt(lookback)
            )
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def _on_balance_volume(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    delta_length: int,
    scale: float,
    volume_type: Literal["normalized", "delta"],
) -> NDArray[np.float64]:
    assert len(close) == len(volume)
    assert lookback > 0
    assert delta_length >= 0
    assert scale > 0
    n = len(close)
    front_bad = lookback
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        signed_volume = total_volume = 0.0
        for j in range(lookback):
            if close[i - j] > close[i - j - 1]:
                signed_volume += volume[i - j]
            elif close[i - j] < close[i - j - 1]:
                signed_volume -= volume[i - j]
            total_volume += volume[i - j]
        if total_volume <= 0.0:
            output[i] = 0.0
            continue
        value = signed_volume / total_volume
        value *= np.sqrt(lookback)
        output[i] = 100.0 * normal_cdf(scale * value) - 50.0
    if volume_type == "delta":
        if delta_length < 1:
            delta_length = 1
        front_bad += delta_length
        if front_bad > n:
            front_bad = n
        for i in range(n - 1, front_bad - 1, -1):
            output[i] -= output[i - delta_length]
    return output


@njit
def normalized_on_balance_volume(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """Computes Normalized On-Balance Volume.

    Args:
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _on_balance_volume(close, volume, lookback, 0, scale, "normalized")


@njit
def delta_on_balance_volume(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    delta_length: int = 0,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """Computes Delta On-Balance Volume.

    Args:
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        delta_length: Lag for differencing.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.6``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _on_balance_volume(
        close, volume, lookback, delta_length, scale, "delta"
    )


@njit
def _normalized_volume_index(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float,
    volume_type: Literal["positive", "negative"],
) -> NDArray[np.float64]:
    assert len(close) == len(volume)
    assert lookback > 0
    assert scale > 0
    n = len(close)
    volatility_length = 2 * lookback
    if volatility_length < 250:
        volatility_length = 250
    front_bad = volatility_length
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        total = 0.0
        if volume_type == "positive":
            for j in range(lookback):
                if volume[i - j] > volume[i - j - 1]:
                    total += np.log(close[i - j] / close[i - j - 1])
        else:
            for j in range(lookback):
                if volume[i - j] < volume[i - j - 1]:
                    total += np.log(close[i - j] / close[i - j - 1])
        total /= np.sqrt(lookback)
        denom = np.sqrt(_variance(True, i, volatility_length, close))
        if denom > 0.0:
            total /= denom
            output[i] = 100.0 * normal_cdf(scale * total) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def normalized_positive_volume_index(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 0.5,
) -> NDArray[np.float64]:
    """Computes Normalized Positive Volume Index.

    Args:
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.5``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _normalized_volume_index(close, volume, lookback, scale, "positive")


@njit
def normalized_negative_volume_index(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 0.5,
) -> NDArray[np.float64]:
    """Computes Normalized Negative Volume Index.

    Args:
        close: Close prices.
        volume: Trading volume.
        lookback: Number of lookback bars.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``0.5``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    return _normalized_volume_index(close, volume, lookback, scale, "negative")


@njit
def volume_momentum(
    volume: NDArray[np.float64],
    short_length: int,
    multiplier: int = 2,
    scale: float = 3.0,
) -> NDArray[np.float64]:
    """Computes Volume Momentum.

    Args:
        volume: Trading volume.
        short_length: Number of short lookback bars.
        multiplier: Lookback multiplier. Defaults to ``2``.
        scale: Increase > 1.0 for more compression of return values,
            decrease < 1.0 for less. Defaults to ``3.0``.

    Returns:
        :class:`numpy.ndarray` of computed values ranging [-50, 50].
    """
    assert short_length > 0
    assert multiplier >= 1
    assert scale > 0
    n = len(volume)
    if multiplier < 2:
        multiplier = 2
    long_length = short_length * multiplier
    front_bad = long_length - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    denom = np.exp(np.log(multiplier) / 3.0)
    for i in range(front_bad, n):
        short_sum = 0.0
        for j in range(i - short_length + 1, i + 1):
            short_sum += volume[j]
        long_sum = short_sum
        for j in range(i - long_length + 1, i - short_length + 1):
            long_sum += volume[j]
        short_sum /= short_length
        long_sum /= long_length
        if long_sum > 0.0 and short_sum > 0.0:
            output[i] = np.log(short_sum / long_sum) / denom
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def laguerre_rsi(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    fe_length: int = 13,
) -> NDArray[np.float64]:
    """Computes Laguerre Relative Strength Index (RSI).

    Args:
        open: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        fe_length: Fractal Energy length. Defaults to ``13``.

    Returns:
        :class:`numpy.ndarray` of computed values.
    """
    assert (
        len(open) == len(high)
        and len(open) == len(low)
        and len(open) == len(close)
    )
    assert fe_length > 0
    n = len(close)
    output = np.zeros(n)
    if n <= fe_length:
        return output
    alpha = np.zeros(n)
    L0_1, L1_1, L2_1, L3_1 = 0.0, 0.0, 0.0, 0.0
    for i in range(fe_length, n):
        OC = (open[i] + close[i - 1]) / 2.0
        HC = max(high[i], close[i - 1])
        LC = min(low[i], close[i - 1])
        fe_src = (OC + HC + LC + close[i]) / 4.0
        highest = max(high[i + 1 - fe_length : i + 1])
        lowest = min(low[i + 1 - fe_length : i + 1])
        denom = highest - lowest
        if denom == 0:
            output[i] = alpha[i] = 0
            continue
        s = 0
        for i in range(fe_length):
            diff = max(high[i - i], close[i - i - 1]) - min(
                low[i - i], close[i - i - 1]
            )
            s += diff / denom
        fe_alpha = np.log(s) / np.log(fe_length)
        alpha[i] = fe_alpha * 100
        L0 = fe_alpha * fe_src + (1 - fe_alpha) * L0_1
        L1 = -(1 - fe_alpha) * L0 + L0_1 + (1 - fe_alpha) * L1_1
        L2 = -(1 - fe_alpha) * L1 + L1_1 + (1 - fe_alpha) * L2_1
        L3 = -(1 - fe_alpha) * L2 + L2_1 + (1 - fe_alpha) * L3_1
        CU = (
            (L0 - L1 if L0 >= L1 else 0)
            + (L1 - L2 if L1 >= L2 else 0)
            + (L2 - L3 if L2 >= L3 else 0)
        )
        CD = (
            (0 if L0 >= L1 else L1 - L0)
            + (0 if L1 >= L2 else L2 - L1)
            + (0 if L2 >= L3 else L3 - L2)
        )
        lrsi = CU / (CU + CD) if CU + CD != 0 else 0
        output[i] = lrsi * 100
        L0_1, L1_1, L2_1, L3_1 = L0, L1, L2, L3
    return output
