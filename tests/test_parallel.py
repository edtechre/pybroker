"""Unit tests for parallel.py module."""

"""Copyright (C) 2026 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import pytest
import re
from importlib import import_module
from joblib import Parallel
from unittest.mock import patch

from pybroker.parallel import (
    get_parallel_config,
    parallel,
    set_parallel,
)

_parallel_mod = import_module("pybroker.parallel")


@pytest.fixture(autouse=True)
def reset_parallel_config():
    saved = get_parallel_config()
    yield
    _parallel_mod._config = saved


def test_get_parallel_config_defaults():
    config = get_parallel_config()
    assert config.n_jobs is None
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
