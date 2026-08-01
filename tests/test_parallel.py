"""Unit tests for parallel.py module."""

"""Copyright (C) 2026 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import logging

import numpy as np
import pandas as pd
import pytest
import re
from importlib import import_module
from joblib import Parallel
from unittest.mock import patch

from pybroker.indicator import IndicatorSet, indicator
from pybroker.parallel import (
    ParallelConfig,
    get_parallel_config,
    parallel,
    set_parallel,
)
from pybroker.scope import StaticScope
from pybroker.vect import highv

_parallel_mod = import_module("pybroker.parallel")


@pytest.fixture(autouse=True)
def reset_parallel_config():
    saved = get_parallel_config()
    yield
    _parallel_mod._config = saved


def test_get_parallel_config_defaults():
    # The suite-wide conftest fixture pins the global config to sequential,
    # so assert the library default on a fresh instance.
    config = ParallelConfig()
    assert config.n_jobs == -1
    assert config.backend == "loky"
    assert config.parallel is None


def test_set_parallel_updates_n_jobs_and_backend():
    set_parallel(n_jobs=2, backend="threading")
    config = get_parallel_config()
    assert config.n_jobs == 2
    assert config.backend == "threading"
    assert config.parallel is None


def test_set_parallel_partial_update_preserves_fields():
    set_parallel(n_jobs=2, backend="threading")
    set_parallel(n_jobs=4)
    config = get_parallel_config()
    assert config.n_jobs == 4
    assert config.backend == "threading"


def test_set_parallel_unknown_backend_raises():
    with pytest.raises(
        ValueError, match=re.escape("Unknown joblib backend 'not_a_backend'")
    ):
        set_parallel(backend="not_a_backend")


def test_set_parallel_multiprocessing_backend_raises():
    """Dispatched indicator and optimize-trial work is made of closures,
    which the multiprocessing backend's standard pickle cannot serialize.
    """
    before = get_parallel_config()
    with pytest.raises(
        ValueError, match="'multiprocessing' backend is not supported"
    ):
        set_parallel(n_jobs=2, backend="multiprocessing")
    assert get_parallel_config() is before


@pytest.fixture()
def ray_backend():
    import ray
    from ray.util.joblib import register_ray

    ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False)
    register_ray()
    yield
    ray.shutdown()


@pytest.mark.xdist_group(name="ray")
def test_set_parallel_ray_backend_when_registered(ray_backend):
    set_parallel(backend="ray", n_jobs=2)
    config = get_parallel_config()
    assert config.backend == "ray"
    assert config.n_jobs == 2


@pytest.fixture()
def scope():
    yield StaticScope.instance()
    StaticScope.set_instance(None)


@pytest.mark.xdist_group(name="ray")
def test_ray_backend_dispatches_parallel_indicators(ray_backend, scope):
    """Runs indicator dispatch end-to-end through the Ray backend.

    Dispatch ships _decorate_indicator_fn's closure plus a lambda indicator
    to workers, so this locks in the cloudpickle serialization that loky and
    Ray provide and the rejected multiprocessing backend does not.
    """
    ind = indicator(
        "ray_hhv", lambda bar_data, n: highv(bar_data.high, n), n=3
    )
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-02", periods=20)
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                }
            )
            for sym, close in (
                (sym, rng.uniform(10, 20, len(dates))) for sym in ("A", "B")
            )
        ],
        ignore_index=True,
    )
    ind_set = IndicatorSet()
    ind_set.add(ind)
    expected = ind_set(df)
    set_parallel(n_jobs=2, backend="ray")

    # Ray warns once per process that joblib's 'context' argument is
    # unsupported; parallel() filters it since the argument comes from
    # joblib's plumbing, not the caller. Record straight off Ray's pool
    # logger (logger-level filters gate handlers, so a broken filter shows
    # up here regardless of Ray's propagate settings) and reset log_once so
    # the warning would definitely fire unfiltered.
    from ray.util.debug import reset_log_once

    class _RecordingHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    handler = _RecordingHandler()
    pool_logger = logging.getLogger("ray.util.multiprocessing.pool")
    pool_logger.addHandler(handler)
    reset_log_once("context_argument_warning")
    try:
        result = ind_set(df, enable_parallel_indicators=True)
    finally:
        pool_logger.removeHandler(handler)
    assert not [
        r
        for r in handler.records
        if "'context' argument" in r.getMessage()
    ]
    sort_cols = ["symbol", "date"]
    pd.testing.assert_frame_equal(
        result.sort_values(sort_cols).reset_index(drop=True),
        expected.sort_values(sort_cols).reset_index(drop=True),
    )


def test_set_parallel_parallel_mutually_exclusive_with_n_jobs():
    custom = Parallel(n_jobs=1, backend="threading")
    with pytest.raises(
        ValueError,
        match="parallel is mutually exclusive with n_jobs and backend",
    ):
        set_parallel(n_jobs=2, parallel=custom)


def test_set_parallel_parallel_mutually_exclusive_with_backend():
    custom = Parallel(n_jobs=1, backend="threading")
    with pytest.raises(
        ValueError,
        match="parallel is mutually exclusive with n_jobs and backend",
    ):
        set_parallel(backend="threading", parallel=custom)


def test_parallel_context_yields_injected_instance():
    custom = Parallel(n_jobs=1, backend="threading")
    set_parallel(parallel=custom)
    with parallel() as p:
        assert p is custom


def test_parallel_context_uses_configured_backend():
    set_parallel(n_jobs=2, backend="threading")
    with patch("pybroker.parallel.joblib.parallel_backend") as mock_backend:
        mock_backend.return_value.__enter__ = lambda self: self
        mock_backend.return_value.__exit__ = lambda *args: None
        with parallel() as p:
            assert type(p) is Parallel
        mock_backend.assert_called_once_with("threading", n_jobs=2)


def test_set_parallel_rejects_unordered_return_as():
    """Every dispatch site pairs results to inputs by position.

    train_models zips tasks to results to recover predict_fn and lag_columns,
    and optimize zips grid points to scores. An unordered return binds each
    result to the wrong input, and the mismatched lag_columns is written to the
    model cache where it outlives the run.
    """
    with pytest.raises(ValueError, match="submission order"):
        set_parallel(
            parallel=Parallel(n_jobs=2, return_as="generator_unordered")
        )
    # Ordered returns are accepted.
    for return_as in ("list", "generator"):
        set_parallel(parallel=Parallel(n_jobs=2, return_as=return_as))
        assert get_parallel_config().parallel is not None
