"""Unit tests for eval.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import math
import numpy as np
import os
import pandas as pd
import pytest
import re
from datetime import datetime
from numba import njit
from pybroker.eval import (
    BootstrapResult,
    BootConfIntervals,
    DrawdownConfs,
    DrawdownMetrics,
    EvalMetrics,
    EvaluateMixin,
    annual_total_return_percent,
    avg_profit_loss,
    bca_boot_conf,
    calmar_ratio,
    conf_profit_factor,
    conf_sharpe_ratio,
    drawdown_conf,
    iqr,
    largest_win_loss,
    max_drawdown,
    max_drawdown_percent,
    max_wins_losses,
    profit_factor,
    r_squared,
    relative_entropy,
    sharpe_ratio,
    sortino_ratio,
    total_profit_loss,
    total_return_percent,
    ulcer_index,
    upi,
    win_loss_rate,
    winning_losing_trades,
)
from typing import get_type_hints

np.random.seed(42)


@pytest.fixture(params=[0, 1, 2])
def value_type(request):
    return request.param


@pytest.fixture(params=[0, 1, 2, 10, 1000])
def rand_values(value_type, request):
    if not request.param:
        return np.empty(0)
    if value_type == 0:
        return np.zeros(request.param)
    elif value_type == 1:
        return np.ones(request.param)
    return np.random.rand(request.param)


@pytest.fixture(params=[True, False])
def calc_bootstrap(request):
    return request.param


@pytest.fixture()
def portfolio_df():
    return pd.read_pickle(
        os.path.join(os.path.dirname(__file__), "testdata/portfolio_df.pkl")
    )


@pytest.fixture()
def trades_df():
    return pd.read_pickle(
        os.path.join(os.path.dirname(__file__), "testdata/trades_df.pkl")
    )


def truncate(value, n):
    return math.floor(value * 10**n) / 10**n


def assert_metric(actual, expected):
    """Compares a ratio, allowing ``inf`` (unbounded) and ``None`` (NaN)."""
    if expected is None:
        assert math.isnan(actual)
    elif math.isinf(expected):
        assert actual == expected
    else:
        assert truncate(actual, 6) == truncate(expected, 6)


@pytest.mark.parametrize(
    "n_boot, expected_msg",
    [
        (0, "Number of boostrap samples must be greater than 0."),
        (-1, "Number of boostrap samples must be greater than 0."),
    ],
)
def test_bca_boot_conf_when_invalid_params_then_error(n_boot, expected_msg):
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        bca_boot_conf(np.random.rand(100), n_boot, profit_factor)


@pytest.mark.parametrize("n_boot", [1, 100])
def test_conf_profit_factor(n_boot, rand_values):
    intervals = conf_profit_factor(rand_values, n_boot)
    assert len(intervals) == 6


@pytest.mark.parametrize("n_boot", [1, 100])
def test_conf_sharpe_ratio(n_boot, rand_values):
    intervals = conf_sharpe_ratio(rand_values, n_boot)
    assert len(intervals) == 6


@pytest.mark.parametrize("n_boot", [1, 100])
def test_drawdown_conf(n_boot, rand_values):
    dd, dd_pct = drawdown_conf(rand_values * 1000, rand_values, n_boot)
    assert len(dd) == 4
    assert len(dd_pct) == 4


@pytest.mark.parametrize(
    "n_boot, expected_msg",
    [
        (0, "Number of boostrap samples must be greater than 0."),
        (-1, "Number of boostrap samples must be greater than 0."),
    ],
)
def test_drawdown_conf_when_invalid_params_then_error(n_boot, expected_msg):
    values = np.random.rand(100)
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        drawdown_conf(values, values, n_boot)


def test_drawdown_conf_when_length_mismatch_then_error():
    with pytest.raises(
        ValueError,
        match=re.escape("Param changes length does not match returns length."),
    ):
        drawdown_conf(np.random.rand(100), np.random.rand(101), 100)


def test_bca_boot_conf_uses_full_data():
    np.random.seed(123)
    x = np.random.rand(50)
    full = bca_boot_conf(x, 200, profit_factor)
    np.random.seed(123)
    prefix = bca_boot_conf(x[:10], 200, profit_factor)
    assert full != prefix


def test_bca_boot_conf_short_data_keeps_n_boot():
    np.random.seed(456)
    x = np.random.rand(5)
    intervals = bca_boot_conf(x, 100, profit_factor)
    assert intervals.low_2p5 != intervals.high_2p5


def test_bca_boot_conf_does_not_mutate_input():
    """The generic path jackknifes by building leave-one-out samples in a
    scratch buffer. Swapping within x would mutate the caller's array, since
    np.ascontiguousarray does not copy an already-contiguous float64 array."""
    x = np.array([1.0, 2.0, -5.0, 3.0, -1.0, 4.0, -2.0])
    original = x.copy()
    bca_boot_conf(x, 100, max_drawdown)
    assert np.array_equal(x, original)


def test_bca_boot_conf_does_not_corrupt_input_when_fn_raises():
    @njit
    def raises_midway(a):
        if len(a) == 6:
            raise ValueError("boom")
        return 0.0

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0])
    original = x.copy()
    with pytest.raises(ValueError):
        bca_boot_conf(x, 10, raises_midway)
    assert np.array_equal(x, original)


def test_drawdown_conf_uses_full_history():
    np.random.seed(789)
    changes = np.concatenate(
        [np.full(20, 100.0), np.full(10, -150.0), np.full(20, 50.0)]
    )
    returns = changes / 10_000.0
    full = drawdown_conf(changes, returns, 200)
    np.random.seed(789)
    short = drawdown_conf(changes[:10], returns[:10], 200)
    assert full.confs.q_001 != short.confs.q_001


@pytest.mark.parametrize(
    "values, expected_pf",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], 1.499999),
        ([1, 1, 1, 1], 40000000001),
        ([1], 10000000001),
        ([-1], 0),
        ([0, 0, 0, 0], 0),
        ([], 0),
    ],
)
def test_profit_factor(values, expected_pf):
    pf = profit_factor(np.array(values))
    assert truncate(pf, 6) == truncate(expected_pf, 6)


@pytest.mark.parametrize(
    "values, obs, expected_sharpe",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], None, 0.167443),
        (
            [0.1, -0.2, 0.3, 0, -0.4, 0.5],
            252,
            0.16744367165578425 * np.sqrt(252),
        ),
        ([1, 1, 1, 1], None, 0),
        ([1], None, 0),
        ([], None, 0),
    ],
)
def test_sharpe_ratio(values, obs, expected_sharpe):
    sharpe = sharpe_ratio(np.array(values), obs)
    assert truncate(sharpe, 6) == truncate(expected_sharpe, 6)


@pytest.mark.parametrize(
    "values, obs, expected_sortino",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], None, 0.273861),
        (
            [0.1, -0.2, 0.3, 0, -0.4, 0.5],
            252,
            0.273861278752583 * np.sqrt(252),
        ),
        ([-0.01, -0.02], None, -0.9486832980505139),
        # No downside and no gain: 0, not unbounded.
        ([0, 0, 0, 0], None, 0),
        ([], None, 0),
    ],
)
def test_sortino_ratio(values, obs, expected_sortino):
    assert_metric(sortino_ratio(np.array(values), obs), expected_sortino)


@pytest.mark.parametrize("values", [[1, 1, 1, 1], [1], [0.01, 0.02, 0.03]])
def test_sortino_ratio_when_no_downside_then_inf(values):
    """A gain with no downside is unbounded, so it ranks best.

    Returning 0 ranks the best possible input alongside a flat one. NaN is
    worse still: a NaN score marks an Optuna trial FAILED, so the winning
    candidate is discarded rather than ranked.
    """
    assert sortino_ratio(np.array(values, dtype=np.float64)) == np.inf


def test_sortino_ratio_when_nan_returns_then_nan():
    """Every ``r < 0`` test against NaN is False, so the downside deviation
    comes out 0 and the degenerate branch reported the best possible score for
    returns that are not computable at all."""
    values = np.array([0.01, np.nan, 0.02], dtype=np.float64)
    assert math.isnan(sortino_ratio(values))


def test_sortino_ratio_matches_reference_definition():
    """Denominator is downside deviation: negative returns squared about zero,
    averaged over *all* observations. Matches quantstats.stats.sortino."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0005, 0.01, 1000)
    expected = returns.mean() / np.sqrt(
        (returns[returns < 0] ** 2).sum() / len(returns)
    )
    assert sortino_ratio(returns) == pytest.approx(expected, rel=1e-12)


def test_sortino_ratio_is_continuous_in_loss_count():
    """Regression: np.std of a one-element array is 0, which used to trip the
    zero-variance guard and collapse a single-loss series to 0."""
    one_loss = np.concatenate([np.full(999, 0.001), [-0.0005]])
    two_losses = np.concatenate([np.full(998, 0.001), [-0.0005, -0.0006]])
    assert sortino_ratio(one_loss) > sortino_ratio(two_losses) > 0


def test_sortino_ratio_accounts_for_loss_depth():
    """Regression: a single loss used to score 0 regardless of magnitude."""
    small = np.concatenate([np.full(99, 0.001), [-0.0005]])
    large = np.concatenate([np.full(99, 0.001), [-0.5]])
    assert sortino_ratio(small) > sortino_ratio(large)


@pytest.mark.parametrize(
    "values, expected_dd",
    [
        ([0.1, 0.15, -0.05, 0.1, -0.25, -0.15, 0], -0.4),
        ([0.1, -0.4], -0.4),
        ([-0.1], -0.1),
        ([1, 1, 1, 1], 0),
        ([1], 0),
        ([], 0),
    ],
)
def test_max_drawdown(values, expected_dd):
    changes = np.array(values)
    assert max_drawdown(changes) == expected_dd


@pytest.mark.parametrize(
    "values, bars_per_year, expected_calmar",
    [
        ([0.1, 0.15, -0.05, 0.1, -0.25, -0.15, 0], 252, -2.75279396935151),
        ([0.1, -0.4], 252, -2.5),
        # A bar losing 100% zeroes the compounded curve: CAGR is a total
        # annual loss, not NaN.
        ([0.05, -1.0], 252, -1.0),
        # No drawdown to divide by: unbounded, not worst-possible.
        ([1, 1, 1, 1], 252, float("inf")),
        ([1], 252, float("inf")),
        ([], 252, 0),
    ],
)
def test_calmar_ratio(values, bars_per_year, expected_calmar):
    assert_metric(
        calmar_ratio(np.array(values), bars_per_year), expected_calmar
    )


@pytest.mark.parametrize(
    "values, expected_dd, expected_index",
    [
        ([0, 0.1, 0.15, -0.05, 0.1, -0.25, -0.15, 0], -36.25, 6),
        ([0, -0.2], -20, 1),
        ([-0.1], -10, 0),
        ([0, 0, 0, 0], 0, None),
        ([0], 0, None),
        ([], 0, None),
    ],
)
def test_max_drawdown_percent(values, expected_dd, expected_index):
    returns = np.array(values)
    dd, index = max_drawdown_percent(returns)
    assert round(dd, 2) == expected_dd
    if expected_index is None:
        assert index is None
    else:
        assert index == expected_index


@pytest.mark.parametrize(
    "values, expected_iqr",
    [
        ([1, 3, 5, 7, 8, 10, 11, 13], 6.5),
        ([1], 0),
        ([1, 2], 0),
        ([1, 1, 1, 1, 1], 0),
        ([], 0),
    ],
)
def test_iqr(values, expected_iqr):
    assert iqr(np.array(values)) == expected_iqr


@pytest.mark.parametrize(
    "values, expected_entropy",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 0.782775),
        ([1, 1, 1, 1], 0),
        ([1], 0),
        ([], 0),
    ],
)
def test_relative_entropy(values, expected_entropy):
    entropy = relative_entropy(np.array(values))
    assert truncate(entropy, 6) == expected_entropy


@pytest.mark.parametrize(
    "values, period, expected_ui",
    [
        # period=None measures drawdowns against the running peak over the
        # whole series.
        ([100, 101, 102, 100, 99, 103, 103, 102], None, 1.296041),
        ([100, 90, 80], None, 12.909944),
        ([0, 0, 0, 0, 0], None, 0),
        ([1, 1, 1, 1, 1], None, 0),
        ([100], None, 0),
        ([], None, 0),
        # An explicit period keeps the trailing-window variant.
        ([100, 101, 102, 100, 99, 103, 103, 102], 2, 0.909259),
        ([100, 101, 102, 100, 99, 103, 103, 102], 1, 0),
        ([0, 0, 0, 0, 0], 2, 0),
        ([1, 1, 1, 1, 1], 2, 0),
        ([100], 14, 0),
        ([100], 1, 0),
        ([], 2, 0),
    ],
)
def test_ulcer_index(values, period, expected_ui):
    assert truncate(ulcer_index(np.array(values), period), 6) == expected_ui


@pytest.mark.parametrize(
    "values, period", [([100, 101, 102], 0), ([100, 101, 102], -1)]
)
def test_ulcer_index_when_invalid_period_then_error(values, period):
    with pytest.raises(AssertionError, match=re.escape("n needs to be >= 1.")):
        ulcer_index(np.array(values), period)


@pytest.mark.parametrize(
    "values, period, ui, expected_upi",
    [
        # period=None divides the mean per-bar return percentage by the
        # whole-series ulcer_index.
        ([100, 101, 102, 100, 99, 103, 103, 102], None, None, 0.231346),
        ([100, 101], None, None, float("inf")),
        ([100, 101, 102, 100, 99, 103, 103, 102], 2, None, 0.329757),
        # Explicit ui=0 with real mid-curve drawdowns: not genuinely
        # drawdown-free, so no inf.
        ([100, 101, 102, 100, 99, 103, 103, 102], 2, 0, 0),
        ([100, 101, 102, 100, 99, 103, 103, 102], 2, 1, 0.299834),
        ([100, 101, 102, 100, 99, 103, 103, 102], 1, None, 0),
        ([0, 0, 0, 0, 0], 2, None, 0),
        ([1, 1, 1, 1, 1], 2, None, 0),
        ([100], 14, None, 0),
        ([100], 1, None, 0),
        ([], 2, None, 0),
        ([], 14, None, 0),
        ([], 14, 0, 0),
        ([], 14, 1.5, 0),
        ([100], 14, None, 0),
        ([100], 14, 0, 0),
        ([100], 14, 1.5, 0),
        ([100], 1, None, 0),
        ([100, 101], 14, None, float("inf")),
        ([100, 101], 14, 0, float("inf")),
        ([100, 101, 102], 2, 0, float("inf")),
    ],
)
def test_upi(values, period, ui, expected_upi):
    assert_metric(upi(np.array(values), period=period, ui=ui), expected_upi)


@pytest.mark.parametrize(
    "values, ui, expected_upi",
    [
        # Annualized (CAGR) return percentage over the whole-series
        # ulcer_index: (102 / 100) ** (252 / 7) - 1 is a 103.99% CAGR.
        ([100, 101, 102, 100, 99, 103, 103, 102], None, 80.23564719096491),
        ([100, 101, 102, 100, 99, 103, 103, 102], 1.0, 103.98873437157054),
        ([100, 90, 80], None, -7.745966692410065),
        # An end value wiped out to zero: CAGR is a total annual loss of
        # -100%, not NaN.
        ([100, 50, 0], None, -1.5491933384829666),
        # The degenerate drawdown-free branch ignores annualization.
        ([100, 101], None, float("inf")),
        ([], None, 0),
    ],
)
def test_upi_annualized(values, ui, expected_upi):
    assert_metric(
        upi(np.array(values), ui=ui, bars_per_year=252), expected_upi
    )


@pytest.mark.parametrize(
    "values, period", [([100, 101, 102], 0), ([100, 101, 102], -1)]
)
def test_upi_when_invalid_period_then_error(values, period):
    with pytest.raises(AssertionError, match=re.escape("n needs to be >= 1.")):
        upi(np.array(values), period)


@pytest.mark.parametrize(
    "values, expected_win_rate, expected_loss_rate",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 70, 30),
        ([0.1], 100, 0),
        ([-0.1], 0, 100),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_win_loss_rate(values, expected_win_rate, expected_loss_rate):
    pnls = np.array(values)
    win_rate, loss_rate = win_loss_rate(pnls)
    assert win_rate == expected_win_rate
    assert loss_rate == expected_loss_rate


@pytest.mark.parametrize(
    "values, expected_winning_trades, expected_losing_trades",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 7, 3),
        ([0.1], 1, 0),
        ([-0.1], 0, 1),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_winning_losing_trades(
    values, expected_winning_trades, expected_losing_trades
):
    pnls = np.array(values)
    winning_trades, losing_trades = winning_losing_trades(pnls)
    assert winning_trades == expected_winning_trades
    assert losing_trades == expected_losing_trades


@pytest.mark.parametrize(
    "values, expected_profit, expected_loss",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], 0.9, -0.6),
        ([0, 0, 0, 0, 0], 0, 0),
        ([0.1], 0.1, 0),
        ([-0.1], 0, -0.1),
        ([], 0, 0),
    ],
)
def test_total_profit_loss(values, expected_profit, expected_loss):
    pnls = np.array(values)
    profit, loss = total_profit_loss(pnls)
    assert profit == expected_profit
    assert round(loss, 2) == expected_loss


@pytest.mark.parametrize(
    "values, expected_profit, expected_loss",
    [
        ([0.1, -0.2, 0.3, 0, -0.4, 0.5], 0.3, -0.3),
        ([1, 1, 1, 1, 1], 1, 0),
        ([-1, -1, -1, -1, -1], 0, -1),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_avg_profit_loss(values, expected_profit, expected_loss):
    pnls = np.array(values)
    profit, loss = avg_profit_loss(pnls)
    assert profit == expected_profit
    assert round(loss, 2) == expected_loss


@pytest.mark.parametrize(
    "values, expected_win, expected_loss",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 0.3, -0.4),
        ([1, 1, 1, 1, 1], 1, 0),
        ([-1, -1, -1, -1, -1], 0, -1),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_largest_win_loss(values, expected_win, expected_loss):
    pnls = np.array(values)
    win, loss = largest_win_loss(pnls)
    assert win == expected_win
    assert loss == expected_loss


@pytest.mark.parametrize(
    "values, expected_wins, expected_losses",
    [
        ([0.1, 0.2, 0.3, -0.2, 0.11, -0.3, -0.4, 0, 0.1, 0.2, 0.2], 3, 2),
        ([1, 1, 1, 1, 1], 5, 0),
        ([-1, -1, -1, -1, -1], 0, 5),
        ([0, 0, 0, 0, 0], 0, 0),
        ([], 0, 0),
    ],
)
def test_max_wins_losses(values, expected_wins, expected_losses):
    pnls = np.array(values)
    wins, losses = max_wins_losses(pnls)
    assert wins == expected_wins
    assert losses == expected_losses


@pytest.mark.parametrize(
    "values, expected_r2",
    [
        ([1, 3, 5, 7, 8, 10, 11, 13], 0.992907),
        ([1], 0),
        ([-1], 0),
        ([1, 1, 1, 1, 1], 0),
        ([0, 0, 0, 0, 0], 0),
        ([], 0),
    ],
)
def test_r_squared(values, expected_r2):
    r2 = r_squared(np.array(values))
    assert truncate(r2, 6) == expected_r2


@pytest.mark.parametrize(
    "initial_value, pnl, expected_return", [(100, 10, 10), (0, 10, 0)]
)
def test_total_return_percent(initial_value, pnl, expected_return):
    return_pct = total_return_percent(initial_value, pnl)
    assert truncate(return_pct, 2) == expected_return


@pytest.mark.parametrize(
    "initial_value, pnl, bars_per_year, total_bars, expected_return",
    [
        # 756 bar values span 755 return intervals.
        (100, 10, 252, 756, 3.23),
        (0, 10, 252, 756, 0),
        (100, 10, 252, 0, 0),
        # A single bar spans no return interval to annualize.
        (100, 10, 252, 1, 0),
    ],
)
def test_annual_total_return_percent(
    initial_value, pnl, bars_per_year, total_bars, expected_return
):
    return_pct = annual_total_return_percent(
        initial_value, pnl, bars_per_year, total_bars
    )
    assert truncate(return_pct, 2) == expected_return


class TestEvaluateMixin:
    @pytest.mark.parametrize(
        "bars_per_year, expected_sharpe, expected_sortino",
        [
            (None, 0.026013464180574847, 0.037930595687473444),
            (
                252,
                0.026013464180574847 * np.sqrt(252),
                0.037930595687473444 * np.sqrt(252),
            ),
        ],
    )
    @pytest.mark.parametrize("bootstrap_samples", [10, 100])
    def test_evaluate(
        self,
        bootstrap_samples,
        portfolio_df,
        trades_df,
        calc_bootstrap,
        bars_per_year,
        expected_sharpe,
        expected_sortino,
    ):
        mixin = EvaluateMixin()
        result = mixin.evaluate(
            portfolio_df,
            trades_df,
            calc_bootstrap,
            bootstrap_samples=bootstrap_samples,
            bars_per_year=bars_per_year,
        )
        assert result.metrics is not None
        if not calc_bootstrap:
            assert result.bootstrap is None
        else:
            assert result.bootstrap is not None
            assert result.bootstrap.conf_intervals is not None
            assert result.bootstrap.drawdown_conf is not None
            assert result.bootstrap.profit_factor is not None
            assert result.bootstrap.sharpe is not None
            assert result.bootstrap.drawdown is not None
            ci = result.bootstrap.conf_intervals
            assert ci.columns.tolist() == ["lower", "upper"]
            names = ci.index.get_level_values(0).unique().tolist()
            assert names == ["Profit Factor", "Sharpe Ratio"]
            for name in names:
                df = ci[ci.index.get_level_values(0) == name]
                confs = df.index.get_level_values(1).tolist()
                assert confs == ["97.5%", "95%", "90%"]
            dd = result.bootstrap.drawdown_conf
            assert dd.columns.tolist() == ["amount", "percent"]
            conf = dd.index.get_level_values(0).tolist()
            assert conf == ["99.9%", "99%", "95%", "90%"]
        metrics = result.metrics
        assert metrics.initial_market_value == 500000
        assert metrics.end_market_value == 693111.87
        assert metrics.total_pnl == 165740.2
        assert (
            metrics.unrealized_pnl
            == metrics.end_market_value
            - metrics.initial_market_value
            - metrics.total_pnl
        )
        assert metrics.total_return_pct == 33.14804
        assert metrics.total_profit == 403511.07999999996
        assert metrics.total_loss == -237770.88
        assert metrics.max_drawdown == -56721.59999999998
        assert metrics.max_drawdown_pct == -7.908428778116649
        assert metrics.max_drawdown_date == datetime(2022, 1, 25, 5, 0)
        assert metrics.win_rate == 52.57731958762887
        assert metrics.loss_rate == 47.42268041237113
        assert metrics.winning_trades == 204
        assert metrics.losing_trades == 184
        assert metrics.avg_pnl == 427.1654639175258
        assert metrics.avg_return_pct == 0.279639175257732
        assert metrics.avg_trade_bars == 2.4149484536082473
        assert metrics.avg_profit == 1977.9954901960782
        assert metrics.avg_profit_pct == 3.1687745098039217
        assert metrics.avg_winning_trade_bars == 2.465686274509804
        assert metrics.avg_loss == -1292.233043478261
        assert metrics.avg_loss_pct == -2.9235326086956523
        assert metrics.avg_losing_trade_bars == 2.358695652173913
        assert metrics.largest_win == 21069.63
        assert metrics.largest_win_pct == 14.49
        assert metrics.largest_win_bars == 3
        assert metrics.largest_loss == -11487.43
        assert metrics.largest_loss_pct == -6.49
        assert metrics.largest_loss_bars == 3
        assert metrics.max_wins == 7
        assert metrics.max_losses == 7
        assert metrics.sharpe == expected_sharpe
        assert metrics.sortino == expected_sortino
        assert metrics.profit_factor == 1.0759385033768167
        assert metrics.ulcer_index == 2.351691016175545
        assert metrics.equity_r2 == 0.8979045919638434
        assert metrics.std_error == 69646.36129687089
        assert metrics.total_fees == 0
        if bars_per_year is not None:
            assert metrics.calmar == 0.854883652700917
            assert metrics.upi == 2.874861720548727
            assert truncate(metrics.annual_return_pct, 6) == truncate(
                5.9025675998261695, 6
            )
            assert metrics.annual_std_error == 1105601.710272446
            assert metrics.annual_volatility_pct == 21.36797425126505
        else:
            assert metrics.calmar is None
            assert metrics.upi == 0.014889530774200667
            assert metrics.annual_return_pct is None
            assert metrics.annual_std_error is None
            assert metrics.annual_volatility_pct is None

    def test_evaluate_when_portfolio_empty(self, trades_df, calc_bootstrap):
        mixin = EvaluateMixin()
        result = mixin.evaluate(
            pd.DataFrame(columns=["market_value", "fees"]),
            trades_df,
            calc_bootstrap,
            bootstrap_samples=100,
            bars_per_year=None,
        )
        assert result.metrics is not None
        for field in get_type_hints(EvalMetrics):
            if field in (
                "calmar",
                "annual_return_pct",
                "annual_std_error",
                "annual_volatility_pct",
                "max_drawdown_date",
            ):
                assert getattr(result.metrics, field) is None
            else:
                assert getattr(result.metrics, field) == 0
        assert result.bootstrap is None

    def test_evaluate_when_single_market_value(
        self, trades_df, calc_bootstrap
    ):
        mixin = EvaluateMixin()
        result = mixin.evaluate(
            pd.DataFrame(
                [[1000, 0]],
                columns=["market_value", "fees"],
                index=[pd.Timestamp("2023-04-12 00:00:00")],
            ),
            trades_df,
            calc_bootstrap,
            bootstrap_samples=100,
            bars_per_year=None,
        )
        assert result.metrics is not None
        for field in get_type_hints(EvalMetrics):
            if field in (
                "calmar",
                "annual_return_pct",
                "annual_std_error",
                "annual_volatility_pct",
                "max_drawdown_date",
            ):
                assert getattr(result.metrics, field) is None
            else:
                assert getattr(result.metrics, field) == 0
        assert result.bootstrap is None

    def test_evaluate_when_trades_empty(self, portfolio_df, calc_bootstrap):
        mixin = EvaluateMixin()
        result = mixin.evaluate(
            portfolio_df,
            pd.DataFrame(columns=["pnl", "return_pct", "bars"]),
            calc_bootstrap,
            bootstrap_samples=100,
            bars_per_year=None,
        )
        metrics = result.metrics
        assert metrics is not None
        assert metrics.total_pnl == 0
        assert metrics.total_return_pct == 0
        assert metrics.total_profit == 0
        assert metrics.total_loss == 0
        assert metrics.win_rate == 0
        assert metrics.loss_rate == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.avg_pnl == 0
        assert metrics.avg_return_pct == 0
        assert metrics.avg_trade_bars == 0
        assert metrics.avg_profit == 0
        assert metrics.avg_profit_pct == 0
        assert metrics.avg_winning_trade_bars == 0
        assert metrics.avg_loss == 0
        assert metrics.avg_loss_pct == 0
        assert metrics.avg_losing_trade_bars == 0
        assert metrics.largest_win == 0
        assert metrics.largest_win_pct == 0
        assert metrics.largest_win_bars == 0
        assert metrics.largest_loss == 0
        assert metrics.largest_loss_pct == 0
        assert metrics.largest_loss_bars == 0
        assert metrics.max_wins == 0
        assert metrics.max_losses == 0
        assert metrics.total_fees == 0
        if calc_bootstrap:
            assert result.bootstrap is not None
            assert result.bootstrap.conf_intervals is not None
            assert result.bootstrap.drawdown_conf is not None
            assert result.bootstrap.profit_factor is not None
            assert result.bootstrap.sharpe is not None
            assert result.bootstrap.drawdown is not None
        else:
            assert result.bootstrap is None

    def test_evaluate_unrealized_pnl_excludes_fees(self, calc_bootstrap):
        """Market values are net of fees while per-trade PnL is gross of
        them, so ``mv[-1] - mv[0] - total_pnl`` understated unrealized PnL
        by every fee paid; a fully closed portfolio with fees reported a
        negative unrealized PnL equal to ``-total_fees``."""
        mixin = EvaluateMixin()
        portfolio_df = pd.DataFrame(
            {
                "market_value": [1000.0, 1010.0, 1005.0, 1030.0],
                "fees": [0.0, 2.0, 5.0, 9.0],
            },
            index=pd.date_range("2023-04-12", periods=4),
        )
        result = mixin.evaluate(
            portfolio_df,
            pd.DataFrame(columns=["pnl", "return_pct", "bars"]),
            calc_bootstrap,
            bootstrap_samples=100,
            bars_per_year=None,
        )
        metrics = result.metrics
        # A 30 market-value gain despite 9 of fees: unrealized PnL is 39.
        assert metrics.unrealized_pnl == 1030.0 - 1000.0 + 9.0
        assert metrics.total_fees == 9.0

    def test_evaluate_bootstrap_is_reproducible_with_seed(
        self, portfolio_df, trades_df
    ):
        """Resampling happens inside @njit code, and Numba keeps a random state
        separate from NumPy's. Seeding via np.random.seed() from Python left
        the bootstrap non-reproducible despite the documented seed argument."""
        mixin = EvaluateMixin()

        def run(seed):
            return mixin.evaluate(
                portfolio_df,
                trades_df,
                calc_bootstrap=True,
                bootstrap_samples=100,
                bars_per_year=252,
                seed=seed,
            ).bootstrap

        first, second = run(42), run(42)
        assert first.profit_factor == second.profit_factor
        assert first.sharpe == second.sharpe
        assert first.drawdown == second.drawdown
        assert run(7).sharpe != first.sharpe
        assert run(None).sharpe != run(None).sharpe

    @pytest.mark.parametrize("calc_bootstrap", [True, False])
    def test_evaluate_preserves_global_numpy_random_state(
        self, portfolio_df, trades_df, calc_bootstrap
    ):
        """evaluate() used to seed the process-global NumPy RNG and only
        restore it on the bootstrap path, silently hijacking the caller's
        stream on every other path."""
        mixin = EvaluateMixin()
        np.random.seed(999)
        expected = np.random.rand(3)
        np.random.seed(999)
        mixin.evaluate(
            portfolio_df,
            trades_df,
            calc_bootstrap,
            bootstrap_samples=50,
            bars_per_year=252,
            seed=42,
        )
        assert np.array_equal(np.random.rand(3), expected)

    def test_evaluate_when_market_value_reaches_zero(self, trades_df):
        """A zero market value used to emit NaN returns, which poisoned
        sharpe/sortino/calmar/volatility and desynced the date index used to
        label max_drawdown_date."""
        market_values = [100.0, 0.0, 0.0, 0.0, 100.0, 120.0, 150.0, 60.0]
        index = pd.date_range("2023-04-12", periods=len(market_values))
        mixin = EvaluateMixin()
        metrics = mixin.evaluate(
            pd.DataFrame(
                {"market_value": market_values, "fees": 0.0}, index=index
            ),
            trades_df,
            calc_bootstrap=False,
            bootstrap_samples=100,
            bars_per_year=252,
        ).metrics
        for field in (
            "sharpe",
            "sortino",
            "calmar",
            "annual_volatility_pct",
            "max_drawdown_pct",
        ):
            assert np.isfinite(getattr(metrics, field)), field
        # Ruin happens on the second bar; the date must name that bar.
        assert metrics.max_drawdown_pct == -100.0
        assert metrics.max_drawdown_date == index[1].to_pydatetime()


def test_eval_metrics_to_json():
    metrics = EvalMetrics(
        trade_count=5,
        sharpe=1.5,
        max_drawdown_date=datetime(2023, 1, 15),
        calmar=None,
    )
    payload = metrics.to_json()
    assert payload["trade_count"] == 5
    assert payload["sharpe"] == 1.5
    assert payload["max_drawdown_date"] == "2023-01-15T00:00:00"
    assert payload["calmar"] is None


def test_eval_metrics_to_json_when_inf_then_sentinel():
    """A legitimately infinite metric (e.g. Sortino with no losing bars)
    must stay distinguishable from a missing metric, which serializes as
    null."""
    import json

    inf_metrics = EvalMetrics(sortino=float("inf"), calmar=float("-inf"))
    nan_metrics = EvalMetrics(sortino=float("nan"), calmar=None)
    inf_payload = inf_metrics.to_json()
    nan_payload = nan_metrics.to_json()
    assert inf_payload["sortino"] == "Infinity"
    assert inf_payload["calmar"] == "-Infinity"
    assert nan_payload["sortino"] is None
    assert inf_payload != nan_payload
    json.dumps(inf_payload, allow_nan=False)


def test_bootstrap_result_to_json():
    conf_intervals = pd.DataFrame(
        [{"name": "sharpe", "conf": "95%", "lower": 0.1, "upper": 2.0}]
    )
    drawdown_conf = pd.DataFrame(
        [{"name": "max_drawdown", "conf": "95%", "upper": 0.05}]
    )
    empty_dd = DrawdownConfs(0.0, 0.0, 0.0, 0.0)
    bootstrap = BootstrapResult(
        conf_intervals=conf_intervals,
        drawdown_conf=drawdown_conf,
        profit_factor=BootConfIntervals(0.1, 1.0, 0.2, 0.9, 0.3, 0.8),
        sharpe=BootConfIntervals(0.0, 2.0, 0.1, 1.9, 0.2, 1.8),
        drawdown=DrawdownMetrics(empty_dd, empty_dd),
    )
    payload = bootstrap.to_json()
    assert len(payload["conf_intervals"]) == 1
    assert payload["profit_factor"]["low_2p5"] == 0.1
    assert payload["drawdown"]["confs"]["q_001"] == 0.0


def test_upi_when_zero_market_value_bar_then_no_zero_division():
    """A wipeout bar must not discard a completed TestResult.

    ulcer_index guards the same division, and its guard means ui is non-zero
    in exactly the case that used to kill upi, so the ``ui == 0`` early-out
    does not cover it.
    """
    values = np.array([100.0, 50.0, 0.0, 10.0, 20.0])
    result = upi(values, period=2)
    assert np.isfinite(result) or np.isnan(result)


def test_relative_entropy_ignores_non_finite_values():
    """+/-inf must not index out of bounds.

    ``factor`` becomes 0.0 with an infinite value, ``0.0 * inf`` is NaN, and
    int(NaN) in numba is INT64_MIN -- an unchecked out-of-bounds write in a
    kernel compiled without boundscheck.
    """
    finite = relative_entropy(np.array([1.0, 2.0, 3.0]))
    with_inf = relative_entropy(np.array([1.0, 2.0, 3.0, np.inf]))
    with_neg_inf = relative_entropy(np.array([1.0, 2.0, 3.0, -np.inf]))
    assert np.isfinite(with_inf)
    assert np.isfinite(with_neg_inf)
    # The infinite observation is excluded, so the result matches the finite
    # sample rather than being computed over a partially counted one.
    assert with_inf == pytest.approx(finite)
    assert with_neg_inf == pytest.approx(finite)


@pytest.mark.parametrize(
    "returns, market_values, label",
    [
        (np.zeros(50), np.full(50, 100_000.0), "no-trade"),
        (
            np.full(50, 0.01),
            np.cumprod(np.full(50, 1.01)) * 100_000.0,
            "all-winning",
        ),
        (
            np.full(50, -0.01),
            np.cumprod(np.full(50, 0.99)) * 100_000.0,
            "all-losing",
        ),
    ],
)
def test_degenerate_metrics_never_fail_an_optuna_trial(
    returns, market_values, label
):
    """An undefined ratio must rank, not abort the study.

    optuna rejects a NaN objective outright, so pybroker records such a trial
    as FAILED. Returning NaN for a degenerate-but-legitimate result therefore
    discarded it -- including an all-winning trial, the one worth keeping --
    and a window whose every trial failed made ``study.best_trial`` raise.
    """
    from pybroker.optimize import _is_failed_score

    for name, value in (
        ("sortino", sortino_ratio(returns)),
        ("calmar", calmar_ratio(returns, 252)),
        ("upi", upi(market_values)),
        ("sharpe", sharpe_ratio(returns)),
    ):
        assert not _is_failed_score(value), f"{label}: {name} = {value!r}"


def test_calmar_ratio_nan_returns_score_nan_not_inf():
    """NaN comparisons are all False, so without the isnan guard a
    non-computable input fell into the drawdown-free branch and scored inf --
    the best possible rank. sortino_ratio carries the same guard."""
    assert math.isnan(calmar_ratio(np.array([0.01, np.nan, 0.02]), 252))


def test_upi_warmup_drawdown_does_not_score_inf():
    """ulcer_index skips its first ``period`` bars, so a zero ulcer does not
    prove the curve never drew down. A 50% drawdown confined to that warmup
    must not score inf."""
    values = np.array([100.0, 50.0, 100.0, 200.0])
    assert ulcer_index(values, 3) == 0
    assert upi(values, 3) == 0
    # A genuinely drawdown-free gain still ranks best.
    assert upi(np.array([100.0, 110.0, 120.0, 130.0]), 2) == np.inf


def test_upi_nan_values_score_nan_regardless_of_direction():
    """NaN means not-computable in both directions.

    Testing direction before scanning for NaN scored the same corrupt series
    0.0 on a net loss but NaN on a net gain -- one Optuna trial COMPLETE, the
    other FAILED, for identical data -- and diverged from sortino_ratio and
    calmar_ratio, which return NaN for both.
    """
    assert math.isnan(upi(np.array([100.0, np.nan, 200.0]), 3))
    assert math.isnan(upi(np.array([100.0, 50.0, np.nan]), 3))
    # Net loss and flat with an interior NaN: NaN, not rankable-worst.
    assert math.isnan(upi(np.array([100.0, np.nan, 90.0]), 3))
    assert math.isnan(upi(np.array([100.0, np.nan, 100.0]), 3))
