"""Contains implementation of evaluation metrics."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
from pybroker.common import _dataframe_records, _json_safe
from pybroker.scope import StaticScope
from pybroker.vect import highv, inverse_normal_cdf, normal_cdf
from collections import deque
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from numba import njit
from numpy.typing import NDArray
from typing import Any, Callable, NamedTuple, Optional


class BootConfIntervals(NamedTuple):
    """Holds confidence intervals of bootstrap tests.

    Attributes:
        low_2p5: Lower bound of 97.5% confidence interval.
        high_2p5: Upper bound of 97.5% confidence interval.
        low_5: Lower bound of 95% confidence interval.
        high_5: Upper bound of 95% confidence interval.
        low_10: Lower bound of 90% confidence interval.
        high_10: Upper bound of 90% confidence interval.
    """

    low_2p5: float
    high_2p5: float
    low_5: float
    high_5: float
    low_10: float
    high_10: float


@njit(cache=True)
def _fill_bootstrap_row(idx: NDArray[np.int64], n: int) -> None:
    """Draws one bootstrap replicate's indices into ``idx``.

    Materializing the whole ``(n_boot, n)`` matrix costs ``8 * n_boot * n``
    bytes -- 8 GB for a year of minute bars at the default sample count --
    while the resampling loops only ever read one row at a time. Drawing row
    by row consumes the RNG in exactly the same order, so every seeded result
    is unchanged.
    """
    for j in range(n):
        idx[j] = np.random.randint(0, n)


@njit(cache=True)
def _seed_bootstrap(seed: int) -> None:
    """Seeds the RNG used for bootstrap resampling.

    Numba keeps a random state that is separate from NumPy's, so calling
    :func:`numpy.random.seed` from Python has no effect on the resampling done
    by :func:`._fill_bootstrap_row`. Seeding must happen from within compiled
    code to be effective.
    """
    np.random.seed(seed)


@njit(cache=True)
def _bca_clamp(k: int, n_boot: int) -> int:
    return min(max(k, 0), n_boot - 1)


@njit(cache=True)
def _jackknife_log_profit_factor(
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    n = len(x)
    result = np.zeros(n)
    win_sum = 0.0
    loss_sum = 0.0
    for v in x:
        if v > 0:
            win_sum += v
        elif v < 0:
            loss_sum += v
    for i in range(n):
        w = win_sum
        loss_total = loss_sum
        if x[i] > 0:
            w -= x[i]
        elif x[i] < 0:
            loss_total -= x[i]
        numer = 1.0e-10 + w
        denom = 1.0e-10 - loss_total
        result[i] = np.log(numer / denom)
    return result


@njit(cache=True)
def _jackknife_profit_factor(
    x: NDArray[np.float64], use_log: bool
) -> NDArray[np.float64]:
    n = len(x)
    result = np.zeros(n)
    win_sum = 0.0
    loss_sum = 0.0
    for v in x:
        if v > 0:
            win_sum += v
        elif v < 0:
            loss_sum += v
    for i in range(n):
        w = win_sum
        loss_total = loss_sum
        if x[i] > 0:
            w -= x[i]
        elif x[i] < 0:
            loss_total -= x[i]
        numer = 1.0e-10 + w
        denom = 1.0e-10 - loss_total
        if use_log:
            result[i] = np.log(numer / denom)
        else:
            result[i] = numer / denom
    return result


@njit(cache=True)
def _jackknife_sharpe(x: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(x)
    result = np.zeros(n)
    if n <= 1:
        return result
    total = 0.0
    total_sq = 0.0
    for v in x:
        total += v
        total_sq += v * v
    for i in range(n):
        m = n - 1
        mean = (total - x[i]) / m
        var = (total_sq - x[i] * x[i]) / m - mean * mean
        if var <= 0:
            result[i] = 0.0
        else:
            result[i] = mean / np.sqrt(var)
    return result


@njit(cache=True)
def _bca_intervals_from_boot(
    boot: NDArray[np.float64],
    z0_count: int,
    jackknife: NDArray[np.float64],
    n: int,
) -> BootConfIntervals:
    n_boot = len(boot)
    z0_count = min(z0_count, n_boot - 1)
    z0_count = max(z0_count, 1)
    z0 = inverse_normal_cdf(z0_count / n_boot)
    theta_dot = 0.0
    for i in range(n):
        theta_dot += jackknife[i]
    theta_dot /= n
    numer = denom = 0.0
    for i in range(n):
        diff = theta_dot - jackknife[i]
        diff_sq = diff * diff
        denom += diff_sq
        numer += diff_sq * diff
    denom = np.power(np.sqrt(denom), 3)
    accel = numer / (6 * denom + 1.0e-60)
    boot.sort()
    zlo = inverse_normal_cdf(0.025)
    zhi = inverse_normal_cdf(0.975)
    alo = normal_cdf(z0 + (z0 + zlo) / (1 - accel * (z0 + zlo)))
    ahi = normal_cdf(z0 + (z0 + zhi) / (1 - accel * (z0 + zhi)))
    k = _bca_clamp(int((alo * (n_boot + 1))) - 1, n_boot)
    low_2p5 = boot[k]
    k = _bca_clamp(int(((1 - ahi) * (n_boot + 1))) - 1, n_boot)
    high_2p5 = boot[n_boot - 1 - k]
    zlo = inverse_normal_cdf(0.05)
    zhi = inverse_normal_cdf(0.95)
    alo = normal_cdf(z0 + (z0 + zlo) / (1 - accel * (z0 + zlo)))
    ahi = normal_cdf(z0 + (z0 + zhi) / (1 - accel * (z0 + zhi)))
    k = _bca_clamp(int((alo * (n_boot + 1))) - 1, n_boot)
    low_5 = boot[k]
    k = _bca_clamp(int(((1 - ahi) * (n_boot + 1))) - 1, n_boot)
    high_5 = boot[n_boot - 1 - k]
    zlo = inverse_normal_cdf(0.1)
    zhi = inverse_normal_cdf(0.9)
    alo = normal_cdf(z0 + (z0 + zlo) / (1 - accel * (z0 + zlo)))
    ahi = normal_cdf(z0 + (z0 + zhi) / (1 - accel * (z0 + zhi)))
    k = _bca_clamp(int((alo * (n_boot + 1))) - 1, n_boot)
    low_10 = boot[k]
    k = _bca_clamp(int(((1 - ahi) * (n_boot + 1))) - 1, n_boot)
    high_10 = boot[n_boot - 1 - k]
    return BootConfIntervals(low_2p5, high_2p5, low_5, high_5, low_10, high_10)


@njit(cache=True)
def _bca_boot_conf_pf(
    x: NDArray[np.float64], n_boot: int, use_log: bool
) -> BootConfIntervals:
    if n_boot <= 0:
        raise ValueError("Number of boostrap samples must be greater than 0.")
    n = len(x)
    if not n:
        return BootConfIntervals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    idx = np.zeros(n, dtype=np.int64)
    x_buff = np.zeros(n)
    boot = np.zeros(n_boot)
    theta_hat = profit_factor(x, use_log)
    z0_count = 0
    for i in range(n_boot):
        _fill_bootstrap_row(idx, n)
        for j in range(n):
            x_buff[j] = x[idx[j]]
        param = profit_factor(x_buff, use_log)
        boot[i] = param
        if param < theta_hat:
            z0_count += 1
    jackknife = _jackknife_profit_factor(x, use_log)
    return _bca_intervals_from_boot(boot, z0_count, jackknife, n)


@njit(cache=True)
def _bca_boot_conf_sharpe(
    x: NDArray[np.float64], n_boot: int
) -> BootConfIntervals:
    if n_boot <= 0:
        raise ValueError("Number of boostrap samples must be greater than 0.")
    n = len(x)
    if not n:
        return BootConfIntervals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    idx = np.zeros(n, dtype=np.int64)
    x_buff = np.zeros(n)
    boot = np.zeros(n_boot)
    theta_hat = sharpe_ratio(x, None)
    z0_count = 0
    for i in range(n_boot):
        _fill_bootstrap_row(idx, n)
        for j in range(n):
            x_buff[j] = x[idx[j]]
        param = sharpe_ratio(x_buff, None)
        boot[i] = param
        if param < theta_hat:
            z0_count += 1
    jackknife = _jackknife_sharpe(x)
    return _bca_intervals_from_boot(boot, z0_count, jackknife, n)


@njit(cache=True)
def _bca_boot_conf_generic(
    x: NDArray[np.float64],
    n_boot: int,
    fn: Callable[[NDArray[np.float64]], float],
) -> BootConfIntervals:
    if n_boot <= 0:
        raise ValueError("Number of boostrap samples must be greater than 0.")
    n = len(x)
    if not n:
        return BootConfIntervals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    idx = np.zeros(n, dtype=np.int64)
    x_buff = np.zeros(n)
    boot = np.zeros(n_boot)
    theta_hat = fn(x)
    z0_count = 0
    for i in range(n_boot):
        _fill_bootstrap_row(idx, n)
        for j in range(n):
            x_buff[j] = x[idx[j]]
        param = fn(x_buff)
        boot[i] = param
        if param < theta_hat:
            z0_count += 1
    jackknife = np.zeros(n)
    if n > 1:
        # Build each leave-one-out sample into a scratch buffer rather than
        # swapping elements within x. Swapping would mutate the caller's array
        # (np.ascontiguousarray does not copy an already-contiguous float64
        # array) and would leave it corrupted if fn raised. It would also
        # reorder the sample, making the acceleration constant wrong for
        # order-dependent statistics such as max_drawdown.
        buff = np.empty(n - 1, dtype=np.float64)
        for i in range(n):
            for j in range(i):
                buff[j] = x[j]
            for j in range(i + 1, n):
                buff[j - 1] = x[j]
            jackknife[i] = fn(buff)
    return _bca_intervals_from_boot(boot, z0_count, jackknife, n)


def bca_boot_conf(
    x: NDArray[np.float64],
    n_boot: int,
    fn: Callable[[NDArray[np.float64]], float],
) -> BootConfIntervals:
    """Computes confidence intervals for a user-defined parameter using the
    `bias corrected and accelerated (BCa) bootstrap method.
    <https://blogs.sas.com/content/iml/2017/07/12/bootstrap-bca-interval.html>`_

    Args:
        x: :class:`numpy.ndarray` containing the data for the randomized
            bootstrap sampling.
        n_boot: Number of random bootstrap samples to use.
        fn: :class:`Callable` for computing the parameter used for the
            confidence intervals.

    Returns:
        :class:`.BootConfIntervals` containing the computed confidence
        intervals.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    if fn is profit_factor:
        return _bca_boot_conf_pf(x, n_boot, False)
    if fn is log_profit_factor:
        return _bca_boot_conf_pf(x, n_boot, True)
    if fn is sharpe_ratio:
        return _bca_boot_conf_sharpe(x, n_boot)
    return _bca_boot_conf_generic(x, n_boot, fn)


@njit(cache=True)
def profit_factor(
    changes: NDArray[np.float64], use_log: bool = False
) -> float:
    """Computes the profit factor, which is the ratio of gross profit to gross
    loss.

    Args:
        changes: Array of differences between each bar and the previous bar.
        use_log: Whether to log transform the profit factor. Defaults to False.
    """
    win_sum = 0.0
    loss_sum = 0.0
    has_win = False
    has_loss = False
    for change in changes:
        if change > 0:
            win_sum += change
            has_win = True
        elif change < 0:
            loss_sum += change
            has_loss = True
    if not has_win and not has_loss:
        return np.float64(0)
    numer = 1.0e-10 + win_sum
    denom = 1.0e-10 - loss_sum
    if use_log:
        return np.log(numer / denom)
    return np.divide(numer, denom)


@njit(cache=True)
def log_profit_factor(changes: NDArray[np.float64]) -> float:
    """Computes the log transformed profit factor, which is the ratio of gross
    profit to gross loss.

    Args:
        changes: Array of differences between each bar and the previous bar.
    """
    return profit_factor(changes, use_log=True)


@njit(cache=True)
def sharpe_ratio(
    returns: NDArray[np.float64],
    obs: Optional[int] = None,
) -> float:
    """Computes the
    `Sharpe Ratio <https://en.wikipedia.org/wiki/Sharpe_ratio>`_, using the
    population standard deviation of ``returns``.

    Args:
        returns: Array of returns centered at 0.
        obs: Number of observations used to annualize the Sharpe Ratio. For
            example, a value of ``252`` would be used to annualize daily
            returns.
    """
    n = len(returns)
    if not n:
        return 0.0
    std = np.std(returns)
    if std == 0:
        return 0.0
    sr = np.mean(returns) / std
    if obs is not None:
        sr *= np.sqrt(obs)
    return float(sr)


@njit(cache=True)
def downside_deviation(returns: NDArray[np.float64]) -> float:
    """Computes downside deviation, the denominator of the
    `Sortino Ratio <https://en.wikipedia.org/wiki/Sortino_ratio>`_.

    Negative returns are squared about zero and averaged over **all**
    observations, not just the negative ones.

    Args:
        returns: Array of returns centered at 0.
    """
    n = len(returns)
    if not n:
        return 0.0
    downside_sq = 0.0
    for r in returns:
        if r < 0:
            downside_sq += r * r
    return np.sqrt(downside_sq / n)


@njit(cache=True)
def sortino_ratio(
    returns: NDArray[np.float64], obs: Optional[int] = None
) -> float:
    """Computes the
    `Sortino Ratio <https://en.wikipedia.org/wiki/Sortino_ratio>`_.

    With no downside to divide by the ratio is unbounded, so a positive mean
    returns ``inf`` -- genuinely the best score, and one Optuna accepts -- while
    a non-positive mean returns ``0``. ``NaN`` is reserved for returns that are
    not computable at all, because a NaN score marks an Optuna trial *failed*
    and would discard the run rather than rank it.

    Args:
        returns: Array of returns centered at 0.
        obs: Number of observations used to annualize the Sortino Ratio. For
            example, a value of ``252`` would be used to annualize daily
            returns.
    """
    n = len(returns)
    if not n:
        return 0.0
    mean = np.mean(returns)
    if np.isnan(mean):
        # Every ``r < 0`` test against NaN is False, so dd comes out 0 and the
        # degenerate branch below would report the best possible score for
        # returns that are not computable at all.
        return np.nan
    dd = downside_deviation(returns)
    if dd == 0:
        # No losing bar, so the ratio is unbounded rather than undefined.
        # Ranking a flawless curve best is correct; ranking it 0 put it below
        # every curve that did lose, and NaN failed the Optuna trial outright.
        return 0.0 if mean <= 0 else np.inf
    sr = mean / dd
    if obs is not None:
        sr *= np.sqrt(obs)
    return float(sr)


def conf_profit_factor(
    x: NDArray[np.float64], n_boot: int
) -> BootConfIntervals:
    """Computes confidence intervals for ``profit_factor``."""
    intervals = bca_boot_conf(x, n_boot, log_profit_factor)
    return BootConfIntervals(
        low_2p5=np.exp(intervals.low_2p5),
        high_2p5=np.exp(intervals.high_2p5),
        low_5=np.exp(intervals.low_5),
        high_5=np.exp(intervals.high_5),
        low_10=np.exp(intervals.low_10),
        high_10=np.exp(intervals.high_10),
    )


def conf_sharpe_ratio(
    x: NDArray[np.float64], n_boot: int, obs: Optional[int] = None
) -> BootConfIntervals:
    """Computes confidence intervals for :func:`.sharpe_ratio`."""
    intervals = bca_boot_conf(x, n_boot, sharpe_ratio)
    if obs is not None:
        factor = np.sqrt(obs)
        intervals = BootConfIntervals(
            low_2p5=intervals.low_2p5 * factor,
            high_2p5=intervals.high_2p5 * factor,
            low_5=intervals.low_5 * factor,
            high_5=intervals.high_5 * factor,
            low_10=intervals.low_10 * factor,
            high_10=intervals.high_10 * factor,
        )
    return intervals


@njit(cache=True)
def max_drawdown(changes: NDArray[np.float64]) -> float:
    """Computes maximum drawdown, measured in cash.

    Args:
        changes: Array of differences between each bar and the previous bar.
    """
    n = len(changes)
    if not n:
        return 0
    cumulative = 0
    max_equity = 0
    dd = 0
    for change in changes:
        cumulative += change
        if cumulative > max_equity:
            max_equity = cumulative
        else:
            loss = max_equity - cumulative
            if loss > dd:
                dd = loss
    return -dd


def calmar_ratio(returns: NDArray[np.float64], bars_per_year: int) -> float:
    """Computes the Calmar Ratio, defined as the annualized return (CAGR of
    the compounded ``returns``) divided by the maximum drawdown percentage
    of the compounded ``returns``.

    Args:
        returns: Array of returns centered at 0.
        bars_per_year: Number of bars per annum.
    """
    if not len(returns):
        return 0.0
    growth = float(np.prod(1.0 + returns))
    if np.isnan(growth):
        # Mirrors sortino_ratio: NaN comparisons are all False, so falling
        # through would score a non-computable input inf -- the best rank.
        return np.nan
    max_dd_pct, _ = max_drawdown_percent(returns)
    max_dd = abs(max_dd_pct) / 100.0
    if max_dd == 0:
        # No drawdown to divide by, so the ratio is unbounded. A zero
        # drawdown means no bar ever declined, hence growth >= 1. See
        # sortino_ratio: inf for a gain, 0 for no gain, never NaN -- a NaN
        # score marks an Optuna trial failed.
        return 0.0 if growth <= 1.0 else np.inf
    if growth <= 0.0:
        # A bar losing 100% or more zeroes out (or flips) the compounded
        # curve; a fractional power of a non-positive base is NaN, and NaN
        # fails an Optuna trial. A wiped-out account is a total annual loss.
        cagr = -1.0
    else:
        cagr = growth ** (bars_per_year / len(returns)) - 1.0
    return float(cagr / max_dd)


@njit(cache=True)
def max_drawdown_percent(
    returns: NDArray[np.float64],
) -> tuple[float, Optional[int]]:
    """Computes maximum drawdown, measured in percentage loss.

    Args:
        returns: Array of returns centered at 0.

    Returns:
        - Maximum drawdown, measured in percentage loss.
        - Index of the maximum drawdown.
    """
    n = len(returns)
    if not n:
        return 0, None
    cumulative = 1.0
    max_equity = 1.0
    dd = 0.0
    index = None
    for i in range(n):
        cumulative *= returns[i] + 1.0
        if cumulative > max_equity:
            max_equity = cumulative
        elif max_equity > 0:
            loss = (cumulative / max_equity - 1) * 100
            if loss < dd:
                dd = loss
                index = i
    return dd, index


@njit(cache=True)
def _dd_conf(q: float, boot: NDArray[np.float64]) -> float:
    k = int((q * (len(boot) + 1)) - 1)
    k = max(k, 0)
    return boot[k]


class DrawdownConfs(NamedTuple):
    """Contains upper bounds of confidence intervals for maximum drawdown.

    Attributes:
        q_001: 99.9% confidence upper bound.
        q_01: 99% confidence upper bound.
        q_05: 95% confidence upper bound.
        q_10: 90% confidence upper bound.
    """

    q_001: float
    q_01: float
    q_05: float
    q_10: float


class DrawdownMetrics(NamedTuple):
    """Contains drawdown metrics.

    Attributes:
        confs: Upper bounds of confidence intervals for maximum
            drawdown, measured in cash.
        pct_confs: Upper bounds of confidence intervals for maximum
            drawdown, measured in percentage.
    """

    confs: DrawdownConfs
    pct_confs: DrawdownConfs


@njit(cache=True)
def _dd_confs(boot: NDArray[np.float64]) -> DrawdownConfs:
    boot.sort()
    return DrawdownConfs(
        _dd_conf(0.001, boot),
        _dd_conf(0.01, boot),
        _dd_conf(0.05, boot),
        _dd_conf(0.1, boot),
    )


@njit(cache=True)
def drawdown_conf(
    changes: NDArray[np.float64],
    returns: NDArray[np.float64],
    n_boot: int,
) -> DrawdownMetrics:
    """Computes upper bounds of confidence intervals for maximum drawdown using
    the bootstrap method.

    Args:
        changes: Array of differences between each bar and the previous bar.
        returns: Array of returns centered at 0.
        n_boot: Number of random bootstrap samples to use.

    Returns:
        :class:`.DrawdownMetrics` containing the confidence bounds.
    """
    if n_boot <= 0:
        raise ValueError("Number of boostrap samples must be greater than 0.")
    n_changes = len(changes)
    if n_changes != len(returns):
        raise ValueError("Param changes length does not match returns length.")
    if not n_changes:
        empty = DrawdownConfs(0.0, 0.0, 0.0, 0.0)
        return DrawdownMetrics(empty, empty)
    n_trades = n_changes
    idx = np.zeros(n_trades, dtype=np.int64)
    changes_sample = np.zeros(n_trades)
    returns_sample = np.zeros(n_trades)
    boot_dd = np.zeros(n_boot)
    boot_dd_pct = np.zeros(n_boot)
    for i in range(n_boot):
        _fill_bootstrap_row(idx, n_trades)
        for j in range(n_trades):
            k = idx[j]
            changes_sample[j] = changes[k]
            returns_sample[j] = returns[k]
        boot_dd[i] = max_drawdown(changes_sample)
        boot_dd_pct[i], _ = max_drawdown_percent(returns_sample)
    return DrawdownMetrics(_dd_confs(boot_dd), _dd_confs(boot_dd_pct))


@njit(cache=True)
def bootstrap_eval_all(
    changes: NDArray[np.float64],
    returns: NDArray[np.float64],
    n_boot: int,
    bars_per_year: int,
) -> tuple[BootConfIntervals, BootConfIntervals, DrawdownMetrics]:
    """Computes all bootstrap metrics in one shared resampling pass."""
    if n_boot <= 0:
        raise ValueError("Number of boostrap samples must be greater than 0.")
    n = len(changes)
    if n != len(returns):
        raise ValueError("Param changes length does not match returns length.")
    if not n:
        empty_ci = BootConfIntervals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        empty_dd = DrawdownConfs(0.0, 0.0, 0.0, 0.0)
        return empty_ci, empty_ci, DrawdownMetrics(empty_dd, empty_dd)
    idx = np.zeros(n, dtype=np.int64)
    changes_sample = np.zeros(n)
    returns_sample = np.zeros(n)
    boot_log_pf = np.zeros(n_boot)
    boot_sharpe = np.zeros(n_boot)
    boot_dd = np.zeros(n_boot)
    boot_dd_pct = np.zeros(n_boot)
    theta_log_pf = log_profit_factor(changes)
    theta_sharpe = sharpe_ratio(returns, None)
    z0_log_pf = 0
    z0_sharpe = 0
    for i in range(n_boot):
        _fill_bootstrap_row(idx, n)
        for j in range(n):
            k = idx[j]
            changes_sample[j] = changes[k]
            returns_sample[j] = returns[k]
        boot_log_pf[i] = log_profit_factor(changes_sample)
        boot_sharpe[i] = sharpe_ratio(returns_sample, None)
        boot_dd[i] = max_drawdown(changes_sample)
        boot_dd_pct[i], _ = max_drawdown_percent(returns_sample)
        if boot_log_pf[i] < theta_log_pf:
            z0_log_pf += 1
        if boot_sharpe[i] < theta_sharpe:
            z0_sharpe += 1
    log_pf_intervals = _bca_intervals_from_boot(
        boot_log_pf,
        z0_log_pf,
        _jackknife_log_profit_factor(changes),
        n,
    )
    sharpe_intervals = _bca_intervals_from_boot(
        boot_sharpe,
        z0_sharpe,
        _jackknife_sharpe(returns),
        n,
    )
    if bars_per_year > 0:
        factor = np.sqrt(float(bars_per_year))
        sharpe_intervals = BootConfIntervals(
            sharpe_intervals.low_2p5 * factor,
            sharpe_intervals.high_2p5 * factor,
            sharpe_intervals.low_5 * factor,
            sharpe_intervals.high_5 * factor,
            sharpe_intervals.low_10 * factor,
            sharpe_intervals.high_10 * factor,
        )
    dd_metrics = DrawdownMetrics(_dd_confs(boot_dd), _dd_confs(boot_dd_pct))
    return log_pf_intervals, sharpe_intervals, dd_metrics


@njit(cache=True)
def relative_entropy(values: NDArray[np.float64]) -> float:
    """Computes the relative `entropy
    <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.
    """
    # isfinite, not just ~isnan: with an infinite value ``factor`` becomes
    # 0.0, ``0.0 * inf`` is NaN, and int(NaN) in numba is INT64_MIN -- an
    # unchecked out-of-bounds write into ``count`` in a kernel compiled
    # without boundscheck.
    x = values[np.isfinite(values)]
    n = len(x)
    if not n:
        return 0
    n_bins = 3
    if n >= 10000:
        n_bins = 20
    elif n >= 1000:
        n_bins = 10
    elif n >= 100:
        n_bins = 5
    min_val = float(np.min(x))
    max_val = float(np.max(x))
    factor = (n_bins - 1.0e-10) / (max_val - min_val + 1.0e-60)
    count = np.zeros(n_bins)
    for v in x:
        k = int(factor * (v - min_val))
        if k < 0:
            k = 0
        elif k >= n_bins:
            k = n_bins - 1
        count[k] += 1
    sum_ = 0
    for c in count:
        if c == 0:
            continue
        p = c / n
        sum_ += p * np.log(p)
    return -sum_ / np.log(n_bins)


def iqr(values: NDArray[np.float64]) -> float:
    """Computes the `interquartile range (IQR)
    <https://en.wikipedia.org/wiki/Interquartile_range>`_ of ``values``."""
    return _iqr(np.ascontiguousarray(values, dtype=np.float64))


@njit(cache=True)
def _percentile_midpoint(sorted_x: NDArray[np.float64], q: float) -> float:
    n = len(sorted_x)
    if n == 0:
        return 0.0
    rank = q / 100.0 * (n - 1)
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))
    if lo == hi:
        return sorted_x[lo]
    return 0.5 * (sorted_x[lo] + sorted_x[hi])


@njit(cache=True)
def _iqr(values: NDArray[np.float64]) -> float:
    x = values[~np.isnan(values)]
    if len(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    return _percentile_midpoint(sorted_x, 75) - _percentile_midpoint(
        sorted_x, 25
    )


@njit(cache=True)
def ulcer_index(
    values: NDArray[np.float64], period: Optional[int] = None
) -> float:
    """Computes the
    `Ulcer Index <https://en.wikipedia.org/wiki/Ulcer_index>`_ of ``values``:
    the root mean square of percentage drawdowns from the running peak.

    Args:
        values: Array of values.
        period: When ``None``, drawdowns are measured against the running
            peak over all of ``values``. When set, drawdowns are measured
            against a trailing ``period``-bar high instead.
    """
    n = len(values)
    if period is None:
        if not n:
            return 0
        dd = np.zeros(n)
        peak = values[0]
        for i in range(n):
            if values[i] > peak:
                peak = values[i]
            if peak == 0:
                dd[i] = 0
            else:
                dd[i] = (values[i] - peak) / peak * 100
        return np.sqrt(np.mean(np.square(dd)))
    if n <= period:
        return 0
    start = period - 1
    dd = np.zeros(n - start)
    max_values = highv(values, period)
    for i in range(start, n):
        if max_values[i] == 0:
            dd[i - start] = 0
            continue
        dd[i - start] = (values[i] - max_values[i]) / max_values[i] * 100
    return np.sqrt(np.mean(np.square(dd)))


@njit(cache=True)
def upi(
    values: NDArray[np.float64],
    period: Optional[int] = None,
    ui: Optional[float] = None,
    bars_per_year: Optional[int] = None,
) -> float:
    """Computes the `Ulcer Performance Index
    <https://en.wikipedia.org/wiki/Ulcer_index>`_ of ``values``: return
    divided by the :func:`.ulcer_index`.

    Args:
        values: Array of values.
        period: Passed to :func:`.ulcer_index` when ``ui`` is ``None``.
        ui: Precomputed :func:`.ulcer_index` of ``values``.
        bars_per_year: Number of bars per annum. When set, the numerator is
            the annualized return (CAGR) of ``values``, measured in
            percentage; when ``None``, the numerator is the mean per-bar
            return percentage.
    """
    if len(values) <= 1:
        return 0.0
    if ui is None:
        ui = ulcer_index(values, period)
    if ui == 0:
        # A rolling-window ulcer_index skips its first ``period`` bars, so a
        # zero ulcer does not prove the curve never drew down -- a drawdown
        # confined to that warmup is invisible to it (and a passed-in ``ui``
        # proves nothing at all). inf is awarded only to a genuinely
        # drawdown-free gain; a rankable curve scores 0.
        if np.isnan(values).any():
            # Scanned before the direction test: a NaN bar is invisible to
            # every comparison below, and testing direction first scored the
            # same corrupt series 0.0 on a net loss but NaN on a net gain --
            # one Optuna trial COMPLETE, the other FAILED, for identical
            # data. Not-computable is NaN in both directions, matching
            # sortino_ratio and calmar_ratio.
            return np.nan
        if values[-1] <= values[0]:
            return 0.0
        peak = values[0]
        for i in range(1, len(values)):
            if values[i] < peak:
                return 0.0
            peak = values[i]
        return np.inf
    if bars_per_year is not None:
        # A zero starting value cannot produce a return; see the guard on
        # the per-bar path below.
        if values[0] == 0:
            return 0.0
        ratio = values[-1] / values[0]
        if ratio <= 0:
            # An end value wiped out to zero or below: a fractional power
            # of a non-positive base is NaN, and NaN fails an Optuna trial.
            # A wiped-out account is a total annual loss.
            cagr_pct = -100.0
        else:
            cagr_pct = (
                ratio ** (bars_per_year / (len(values) - 1)) - 1.0
            ) * 100.0
        return float(cagr_pct / ui)
    r = np.zeros(len(values) - 1)
    for i in range(len(r)):
        prev = values[i]
        # A zero portfolio cannot produce a return. Unguarded this raises
        # ZeroDivisionError under numba's default error model and discards a
        # TestResult the bar loop already finished computing -- and a quantized
        # market value of exactly 0.00 on a wipeout is an attractor, not a
        # knife-edge. ulcer_index guards the same division.
        r[i] = 0.0 if prev == 0 else (values[i + 1] - prev) / prev * 100
    return float(np.mean(r) / ui)


def win_loss_rate(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes the win rate and loss rate as percentages.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of win rate and loss rate.
    """
    pnls = pnls[pnls != 0]
    n = len(pnls)
    if not n:
        return 0, 0
    win_rate = len(pnls[pnls > 0]) / n * 100
    loss_rate = len(pnls[pnls < 0]) / n * 100
    return win_rate, loss_rate


def winning_losing_trades(pnls: NDArray[np.float64]) -> tuple[int, int]:
    """Returns the number of winning and losing trades.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[int, int]`` containing numbers of winning and losing trades.
    """
    pnls = pnls[pnls != 0]
    if not len(pnls):
        return 0, 0
    return len(pnls[pnls > 0]), len(pnls[pnls < 0])


def total_profit_loss(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes total profit and loss.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of total profit and total loss.
    """
    profits = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return (
        np.sum(profits) if len(profits) else 0,
        np.sum(losses) if len(losses) else 0,
    )


def avg_profit_loss(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes the average profit and average loss per trade.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of average profit and average loss.
    """

    profits = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return (
        float(np.mean(profits)) if len(profits) else 0,
        float(np.mean(losses)) if len(losses) else 0,
    )


def largest_win_loss(pnls: NDArray[np.float64]) -> tuple[float, float]:
    """Computes the largest profit and largest loss of all trades.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[float, float]`` of largest profit and largest loss.
    """
    profits = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return (
        np.max(profits) if len(profits) else 0,
        np.min(losses) if len(losses) else 0,
    )


@njit(cache=True)
def max_wins_losses(pnls: NDArray[np.float64]) -> tuple[int, int]:
    """Computes the max consecutive wins and max consecutive losses.

    Args:
        pnls: Array of profits and losses (PnLs) per trade.

    Returns:
        ``tuple[int, int]`` of max consecutive wins and max consecutive losses.
    """
    max_wins = max_losses = wins = losses = 0
    for pnl in pnls:
        if pnl > 0:
            wins += 1
            max_wins = max(max_wins, wins)
        else:
            wins = 0
        if pnl < 0:
            losses += 1
            max_losses = max(max_losses, losses)
        else:
            losses = 0
    return max_wins, max_losses


def total_return_percent(initial_value: float, pnl: float) -> float:
    """Computes total return as percentage.

    Args:
        initial_value: Initial value.
        pnl: Total profit and loss (PnL).
    """
    if initial_value == 0:
        return 0
    return ((pnl + initial_value) / initial_value - 1) * 100


def annual_total_return_percent(
    initial_value: float, pnl: float, bars_per_year: int, total_bars: int
) -> float:
    """Computes annualized total return as percentage.

    Args:
        initial_value: Initial value.
        pnl: Total profit and loss (PnL).
        bars_per_year: Number of bars per annum.
        total_bars: Total number of bars of the return.
    """
    if initial_value == 0 or total_bars <= 1:
        return 0
    # total_bars values span total_bars - 1 return intervals.
    return (
        np.power(
            (pnl + initial_value) / initial_value,
            bars_per_year / (total_bars - 1),
        )
        - 1
    ) * 100


def r_squared(values: NDArray[np.float64]) -> float:
    """Computes R-squared of ``values``."""
    n = len(values)
    if not n:
        return 0
    x = np.arange(n)
    try:
        coeffs = np.polyfit(x, values, 1)
        pred = np.poly1d(coeffs)(x)
        y_hat = np.mean(values)
        ssres = float(np.sum((values - pred) ** 2))
        sstot = float(np.sum((values - y_hat) ** 2))
        if sstot == 0:
            return 0
        return 1 - ssres / sstot
    except Exception:
        return 0


class TradeStats(NamedTuple):
    """Trade statistics computed in a single pass."""

    trade_count: int
    win_rate: float
    loss_rate: float
    winning_trades: int
    losing_trades: int
    total_profit: float
    total_loss: float
    avg_profit: float
    avg_loss: float
    avg_profit_pct: float
    avg_loss_pct: float
    largest_win: float
    largest_loss: float
    largest_win_pct: float
    largest_loss_pct: float
    largest_win_bars: int
    largest_loss_bars: int
    max_wins: int
    max_losses: int
    avg_pnl: float
    avg_return_pct: float
    avg_trade_bars: float
    avg_winning_trade_bars: float
    avg_losing_trade_bars: float
    total_pnl: float


@njit(cache=True)
def _trade_stats(
    pnls: NDArray[np.float64],
    return_pcts: NDArray[np.float64],
    bars: NDArray[np.int_],
) -> TradeStats:
    n = len(pnls)
    if n == 0:
        return TradeStats(
            0,
            0.0,
            0.0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    nonzero = wins = losses = 0
    total_profit = total_loss = 0.0
    profit_count = loss_count = 0
    profit_pct_sum = loss_pct_sum = 0.0
    win_bars_sum = loss_bars_sum = 0
    largest_win = 0.0
    largest_loss = 0.0
    largest_win_pct = 0.0
    largest_loss_pct = 0.0
    largest_win_bars = 0
    largest_loss_bars = 0
    pnl_sum = return_pct_sum = bars_sum = 0.0
    max_wins = max_losses = cur_wins = cur_losses = 0
    for i in range(n):
        pnl = pnls[i]
        pnl_sum += pnl
        return_pct_sum += return_pcts[i]
        bars_sum += bars[i]
        if pnl > 0:
            cur_wins += 1
            cur_losses = 0
            if cur_wins > max_wins:
                max_wins = cur_wins
            total_profit += pnl
            profit_count += 1
            profit_pct_sum += return_pcts[i]
            win_bars_sum += bars[i]
            if pnl > largest_win:
                largest_win = pnl
                largest_win_pct = return_pcts[i]
                largest_win_bars = bars[i]
        elif pnl < 0:
            cur_losses += 1
            cur_wins = 0
            if cur_losses > max_losses:
                max_losses = cur_losses
            total_loss += pnl
            loss_count += 1
            loss_pct_sum += return_pcts[i]
            loss_bars_sum += bars[i]
            if pnl < largest_loss:
                largest_loss = pnl
                largest_loss_pct = return_pcts[i]
                largest_loss_bars = bars[i]
        else:
            cur_wins = 0
            cur_losses = 0
        if pnl != 0:
            nonzero += 1
            if pnl > 0:
                wins += 1
            else:
                losses += 1
    win_rate = wins / nonzero * 100 if nonzero else 0.0
    loss_rate = losses / nonzero * 100 if nonzero else 0.0
    return TradeStats(
        n,
        win_rate,
        loss_rate,
        wins,
        losses,
        total_profit,
        total_loss,
        total_profit / profit_count if profit_count else 0.0,
        total_loss / loss_count if loss_count else 0.0,
        profit_pct_sum / profit_count if profit_count else 0.0,
        loss_pct_sum / loss_count if loss_count else 0.0,
        largest_win,
        largest_loss,
        largest_win_pct,
        largest_loss_pct,
        largest_win_bars,
        largest_loss_bars,
        max_wins,
        max_losses,
        pnl_sum / n,
        return_pct_sum / n,
        bars_sum / n,
        win_bars_sum / profit_count if profit_count else 0.0,
        loss_bars_sum / loss_count if loss_count else 0.0,
        pnl_sum,
    )


@njit(cache=True)
def _bar_returns_and_changes(
    market_values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = len(market_values)
    if n <= 1:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    m = n - 1
    returns = np.empty(m, dtype=np.float64)
    changes = np.empty(m, dtype=np.float64)
    for i in range(1, n):
        prev = market_values[i - 1]
        cur = market_values[i]
        changes[i - 1] = cur - prev
        if prev == 0:
            # A zero portfolio cannot produce a return. Emitting NaN here would
            # silently poison sharpe/sortino/calmar/volatility downstream, and
            # would freeze max_drawdown_percent at a stale value, since every
            # comparison against NaN is False.
            returns[i - 1] = 0.0
        else:
            returns[i - 1] = (cur - prev) / prev
    return returns, changes


class BootstrapResult(NamedTuple):
    """Contains results of bootstrap tests.

    Attributes:
        conf_intervals: :class:`pandas.DataFrame` containing confidence
            intervals for :func:`.log_profit_factor` and :func:`.sharpe_ratio`.
        drawdown_conf: :class:`pandas.DataFrame` containing upper bounds of
            confidence intervals for maximum drawdown.
        profit_factor: Contains profit factor confidence intervals.
        sharpe: Contains Sharpe Ratio confidence intervals.
        drawdown: Contains drawdown confidence intervals.
    """

    conf_intervals: pd.DataFrame
    drawdown_conf: pd.DataFrame
    profit_factor: BootConfIntervals
    sharpe: BootConfIntervals
    drawdown: DrawdownMetrics

    def to_json(self) -> dict[str, Any]:
        """Returns JSON-serializable bootstrap evaluation metrics."""
        return {
            "conf_intervals": _dataframe_records(self.conf_intervals),
            "drawdown_conf": _dataframe_records(self.drawdown_conf),
            "profit_factor": _json_safe(self.profit_factor._asdict()),
            "sharpe": _json_safe(self.sharpe._asdict()),
            "drawdown": _json_safe(self.drawdown._asdict()),
        }


@dataclass(frozen=True)
class EvalMetrics:
    """Contains metrics for evaluating a :class:`pybroker.strategy.Strategy`.

    Attributes:
        trade_count: Number of trades that were filled.
        initial_market_value: Initial market value of the
            :class:`pybroker.portfolio.Portfolio`.
        end_market_value: Ending market value of the
            :class:`pybroker.portfolio.Portfolio`.
        total_pnl: Total realized profit and loss (PnL), gross of fees.
        unrealized_pnl: Total unrealized profit and loss (PnL).
        total_return_pct: Total realized return measured in percentage,
            gross of fees. Fees are reported separately in ``total_fees``.
        annual_return_pct: Annualized total realized return measured in
            percentage, gross of fees.
        total_profit: Total realized profit.
        total_loss: Total realized loss.
        total_fees: Total brokerage fees. See
            :attr:`pybroker.config.StrategyConfig.fee_mode` for more info.
        max_drawdown: Maximum drawdown, measured in cash.
        max_drawdown_pct: Maximum drawdown, measured in percentage.
        max_drawdown_date: Date of maximum drawdown.
        win_rate: Win rate of trades.
        loss_rate: Loss rate of trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        avg_pnl: Average profit and loss (PnL) per trade, measured in cash.
        avg_return_pct: Average return per trade, measured in percentage.
        avg_trade_bars: Average number of bars per trade.
        avg_profit: Average profit per trade, measured in cash.
        avg_profit_pct: Average profit per trade, measured in percentage.
        avg_winning_trade_bars: Average number of bars per winning trade.
        avg_loss: Average loss per trade, measured in cash.
        avg_loss_pct: Average loss per trade, measured in percentage.
        avg_losing_trade_bars: Average number of bars per losing trade.
        largest_win: Largest profit of a trade, measured in cash.
        largest_win_pct: Largest profit of a trade, measured in percentage
        largest_win_bars: Number of bars in the largest winning trade.
        largest_loss: Largest loss of a trade, measured in cash.
        largest_loss_pct: Largest loss of a trade, measured in percentage.
        largest_loss_bars: Number of bars in the largest losing trade.
        max_wins: Maximum number of consecutive winning trades.
        max_losses: Maximum number of consecutive losing trades.
        sharpe: `Sharpe Ratio <https://en.wikipedia.org/wiki/Sharpe_ratio>`_,
            computed per bar.
        sortino: `Sortino Ratio
            <https://en.wikipedia.org/wiki/Sortino_ratio>`_, computed per bar.
            ``inf`` when there are no losing bars and the mean return is
            positive.
        calmar: Calmar Ratio: annualized return (CAGR) of the equity curve
            divided by its maximum drawdown percentage. ``None`` when
            :attr:`pybroker.config.StrategyConfig.bars_per_year` is not set,
            since the ratio annualizes; ``inf`` when there is no drawdown and
            the equity curve gained.
        profit_factor: Ratio of gross profit to gross loss, computed per bar.
        ulcer_index: `Ulcer Index
            <https://en.wikipedia.org/wiki/Ulcer_index>`_: root mean square
            of the equity curve's percentage drawdowns from its running peak.
        upi: `Ulcer Performance Index
            <https://en.wikipedia.org/wiki/Ulcer_index>`_: return divided by
            the Ulcer Index. The return is annualized (CAGR) when
            :attr:`pybroker.config.StrategyConfig.bars_per_year` is set, and
            is the mean per-bar return otherwise. ``inf`` for a genuinely
            drawdown-free gain.
        equity_r2: R^2 of the equity curve, computed per bar on market values
            of portfolio.
        std_error: Standard deviation of the portfolio market values across
            all bars. This measures dispersion of the equity curve's level;
            it is not a regression standard error.
        annual_std_error: ``std_error`` multiplied by the square root of
            :attr:`pybroker.config.StrategyConfig.bars_per_year`.
        annual_volatility_pct: Annualized volatility percentage, computed per
            bar on market values of portfolio.
    """

    trade_count: int = field(default=0)
    initial_market_value: float = field(default=0)
    end_market_value: float = field(default=0)
    total_pnl: float = field(default=0)
    unrealized_pnl: float = field(default=0)
    total_return_pct: float = field(default=0)
    annual_return_pct: Optional[float] = field(default=None)
    total_profit: float = field(default=0)
    total_loss: float = field(default=0)
    total_fees: float = field(default=0)
    max_drawdown: float = field(default=0)
    max_drawdown_pct: float = field(default=0)
    max_drawdown_date: Optional[datetime] = field(default=None)
    win_rate: float = field(default=0)
    loss_rate: float = field(default=0)
    winning_trades: int = field(default=0)
    losing_trades: int = field(default=0)
    avg_pnl: float = field(default=0)
    avg_return_pct: float = field(default=0)
    avg_trade_bars: float = field(default=0)
    avg_profit: float = field(default=0)
    avg_profit_pct: float = field(default=0)
    avg_winning_trade_bars: float = field(default=0)
    avg_loss: float = field(default=0)
    avg_loss_pct: float = field(default=0)
    avg_losing_trade_bars: float = field(default=0)
    largest_win: float = field(default=0)
    largest_win_pct: float = field(default=0)
    largest_win_bars: int = field(default=0)
    largest_loss: float = field(default=0)
    largest_loss_pct: float = field(default=0)
    largest_loss_bars: int = field(default=0)
    max_wins: int = field(default=0)
    max_losses: int = field(default=0)
    sharpe: float = field(default=0)
    sortino: float = field(default=0)
    calmar: Optional[float] = field(default=None)
    profit_factor: float = field(default=0)
    ulcer_index: float = field(default=0)
    upi: float = field(default=0)
    equity_r2: float = field(default=0)
    std_error: float = field(default=0)
    annual_std_error: Optional[float] = field(default=None)
    annual_volatility_pct: Optional[float] = field(default=None)

    def to_json(self) -> dict[str, Any]:
        """Returns JSON-serializable evaluation metrics."""
        return _json_safe(dataclasses.asdict(self))


class ConfInterval(NamedTuple):
    """Confidence interval upper and low bounds.

    Attributes:
        name: Parameter name.
        conf: Confidence interval percentage represented as a ``str``.
        lower: Lower bound.
        upper: Upper bound.
    """

    name: str
    conf: str
    lower: float
    upper: float


class EvalResult(NamedTuple):
    """Contains evaluation result.

    Attributes:
        metrics: Evaluation metrics.
        bootstrap: Randomized bootstrap metrics.
    """

    metrics: EvalMetrics
    bootstrap: Optional[BootstrapResult]


class EvaluateMixin:
    """Mixin for computing evaluation metrics."""

    def evaluate(
        self,
        portfolio_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        calc_bootstrap: bool,
        bootstrap_samples: int,
        bars_per_year: Optional[int],
        seed: Optional[int] = 42,
    ) -> EvalResult:
        """Computes evaluation metrics.

        Args:
            portfolio_df: :class:`pandas.DataFrame` of portfolio market values
                per bar.
            trades_df: :class:`pandas.DataFrame` of trades.
            calc_bootstrap: ``True`` to calculate randomized bootstrap metrics.
            bootstrap_samples: Number of random bootstrap samples to use.
            bars_per_year: Number of observations per years that will be used
                to annualize evaluation metrics. For example, a value of
                ``252`` would be used to annualize the Sharpe Ratio for daily
                returns.
            seed: Random seed for reproducibility. Defaults to 42.

        Returns:
            :class:`.EvalResult` containing evaluation metrics.
        """
        market_values = portfolio_df["market_value"].to_numpy(
            dtype=np.float64, copy=True
        )
        fees = portfolio_df["fees"].to_numpy(dtype=np.float64, copy=True)
        bar_returns, bar_changes = _bar_returns_and_changes(market_values)
        bar_return_dates = portfolio_df.index[1:]
        if (
            not len(market_values)
            or not len(bar_returns)
            or not len(bar_changes)
        ):
            return EvalResult(EvalMetrics(), None)
        pnls = trades_df["pnl"].to_numpy(dtype=np.float64, copy=True)
        return_pcts = trades_df["return_pct"].to_numpy(
            dtype=np.float64, copy=True
        )
        bars = trades_df["bars"].to_numpy(dtype=np.int64, copy=True)
        trade_stats = _trade_stats(pnls, return_pcts, bars)
        if len(pnls):
            profits = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            profit_pcts = return_pcts[pnls > 0]
            loss_pcts = return_pcts[pnls < 0]
            win_bars = bars[pnls > 0]
            loss_bars = bars[pnls < 0]
            trade_stats = trade_stats._replace(
                total_pnl=float(np.sum(pnls)),
                total_profit=(float(np.sum(profits)) if len(profits) else 0.0),
                total_loss=float(np.sum(losses)) if len(losses) else 0.0,
                avg_pnl=float(np.mean(pnls)),
                avg_return_pct=float(np.mean(return_pcts)),
                avg_trade_bars=float(np.mean(bars)),
                avg_profit=(float(np.mean(profits)) if len(profits) else 0.0),
                avg_loss=float(np.mean(losses)) if len(losses) else 0.0,
                avg_profit_pct=(
                    float(np.mean(profit_pcts)) if len(profit_pcts) else 0.0
                ),
                avg_loss_pct=(
                    float(np.mean(loss_pcts)) if len(loss_pcts) else 0.0
                ),
                avg_winning_trade_bars=(
                    float(np.mean(win_bars)) if len(win_bars) else 0.0
                ),
                avg_losing_trade_bars=(
                    float(np.mean(loss_bars)) if len(loss_bars) else 0.0
                ),
            )
        metrics = self._calc_eval_metrics(
            market_values,
            bar_changes,
            bar_returns,
            bar_return_dates,
            trade_stats,
            fees=fees,
            bars_per_year=bars_per_year,
        )
        logger = StaticScope.instance().logger
        if not calc_bootstrap:
            return EvalResult(metrics, None)
        logger.calc_bootstrap_metrics_start(
            samples=bootstrap_samples, bars=len(bar_returns)
        )
        annualize = bars_per_year if bars_per_year is not None else 0
        if seed is not None:
            _seed_bootstrap(seed)
        log_pf_intervals, sharpe_intervals, dd_metrics = bootstrap_eval_all(
            bar_changes,
            bar_returns,
            bootstrap_samples,
            annualize,
        )
        pf_intervals = BootConfIntervals(
            low_2p5=np.exp(log_pf_intervals.low_2p5),
            high_2p5=np.exp(log_pf_intervals.high_2p5),
            low_5=np.exp(log_pf_intervals.low_5),
            high_5=np.exp(log_pf_intervals.high_5),
            low_10=np.exp(log_pf_intervals.low_10),
            high_10=np.exp(log_pf_intervals.high_10),
        )
        pf_conf = self._to_conf_intervals("Profit Factor", pf_intervals)
        sharpe_conf = self._to_conf_intervals("Sharpe Ratio", sharpe_intervals)
        conf_intervals = pd.DataFrame.from_records(
            pf_conf + sharpe_conf, columns=ConfInterval._fields
        )
        conf_intervals.set_index(["name", "conf"], inplace=True)
        drawdown_conf = pd.DataFrame(
            zip(("99.9%", "99%", "95%", "90%"), *dd_metrics),
            columns=("conf", "amount", "percent"),
        )
        drawdown_conf.set_index("conf", inplace=True)
        bootstrap = BootstrapResult(
            conf_intervals=conf_intervals,
            drawdown_conf=drawdown_conf,
            profit_factor=pf_intervals,
            sharpe=sharpe_intervals,
            drawdown=dd_metrics,
        )
        logger.calc_bootstrap_metrics_completed()
        return EvalResult(metrics, bootstrap)

    def _calc_eval_metrics(
        self,
        market_values: NDArray[np.float64],
        bar_changes: NDArray[np.float64],
        bar_returns: NDArray[np.float64],
        bar_return_dates: pd.Series,
        trade_stats: TradeStats,
        fees: NDArray[np.float64],
        bars_per_year: Optional[int],
    ) -> EvalMetrics:
        total_fees = fees[-1] if len(fees) else 0
        max_dd = max_drawdown(bar_changes)
        max_dd_pct, max_dd_index = max_drawdown_percent(bar_returns)
        max_dd_date = (
            bar_return_dates[max_dd_index].to_pydatetime()
            if max_dd_index is not None
            else None
        )
        sharpe = sharpe_ratio(bar_returns, bars_per_year)
        sortino = sortino_ratio(bar_returns, bars_per_year)
        pf = profit_factor(bar_changes)
        r2 = r_squared(market_values)
        ui = ulcer_index(market_values)
        upi_ = upi(market_values, ui=ui, bars_per_year=bars_per_year)
        std_error = float(np.std(market_values))
        total_pnl = float(trade_stats.total_pnl)
        total_return_pct = total_return_percent(
            initial_value=market_values[0], pnl=total_pnl
        )
        # Market values are net of fees while per-trade PnL is gross of them
        # (fees only ever reduce cash), so the difference of the two absorbs
        # every fee paid. Add the fees accrued over the run back so this
        # reports unrealized PnL alone. The fees column is cumulative;
        # subtracting fees[0] keeps the identity exact even if the first
        # recorded bar carried fees.
        unrealized_pnl = (
            market_values[-1]
            - market_values[0]
            - total_pnl
            + (float(fees[-1]) - float(fees[0]) if len(fees) else 0.0)
        )
        annual_return_pct = None
        annual_std_error = None
        annual_volatility_pct = None
        calmar = None
        if bars_per_year is not None:
            annual_return_pct = annual_total_return_percent(
                initial_value=market_values[0],
                pnl=total_pnl,
                bars_per_year=bars_per_year,
                total_bars=len(market_values),
            )
            annual_std_error = std_error * np.sqrt(bars_per_year)
            annual_volatility_pct = float(
                np.std(bar_returns * 100) * np.sqrt(bars_per_year)
            )
            calmar = calmar_ratio(bar_returns, bars_per_year)
        return EvalMetrics(
            trade_count=trade_stats.trade_count,
            initial_market_value=market_values[0],
            end_market_value=market_values[-1],
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_date=max_dd_date,
            largest_win=trade_stats.largest_win,
            largest_win_pct=trade_stats.largest_win_pct,
            largest_win_bars=trade_stats.largest_win_bars,
            largest_loss=trade_stats.largest_loss,
            largest_loss_pct=trade_stats.largest_loss_pct,
            largest_loss_bars=trade_stats.largest_loss_bars,
            max_wins=trade_stats.max_wins,
            max_losses=trade_stats.max_losses,
            win_rate=trade_stats.win_rate,
            loss_rate=trade_stats.loss_rate,
            winning_trades=trade_stats.winning_trades,
            losing_trades=trade_stats.losing_trades,
            avg_pnl=trade_stats.avg_pnl,
            avg_return_pct=trade_stats.avg_return_pct,
            avg_trade_bars=trade_stats.avg_trade_bars,
            avg_profit=trade_stats.avg_profit,
            avg_profit_pct=trade_stats.avg_profit_pct,
            avg_winning_trade_bars=trade_stats.avg_winning_trade_bars,
            avg_loss=trade_stats.avg_loss,
            avg_loss_pct=trade_stats.avg_loss_pct,
            avg_losing_trade_bars=trade_stats.avg_losing_trade_bars,
            total_profit=trade_stats.total_profit,
            total_loss=trade_stats.total_loss,
            total_pnl=total_pnl,
            unrealized_pnl=unrealized_pnl,
            total_return_pct=total_return_pct,
            annual_return_pct=annual_return_pct,
            total_fees=total_fees,
            sharpe=float(sharpe),
            sortino=sortino,
            calmar=calmar,
            profit_factor=float(pf),
            equity_r2=r2,
            ulcer_index=ui,
            upi=upi_,
            std_error=std_error,
            annual_std_error=annual_std_error,
            annual_volatility_pct=annual_volatility_pct,
        )

    def _to_conf_intervals(
        self, name: str, conf: BootConfIntervals
    ) -> deque[ConfInterval]:
        results: deque[ConfInterval] = deque()
        results.append(
            ConfInterval(name, "97.5%", conf.low_2p5, conf.high_2p5)
        )
        results.append(ConfInterval(name, "95%", conf.low_5, conf.high_5))
        results.append(ConfInterval(name, "90%", conf.low_10, conf.high_10))
        return results
