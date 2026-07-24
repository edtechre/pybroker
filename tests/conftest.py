"""Pytest configuration for the test suite."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import os
import tempfile

import numpy as np
import pytest


def _patch_numpy_seed_for_randomly() -> None:
    """Wrap ``numpy.random.seed`` for pytest-randomly + thinc entrypoints.

    pytest-randomly passes session seeds with per-test offsets that can exceed
    NumPy's legacy ``2**32 - 1`` limit to third-party reseed hooks (e.g. thinc).
    """
    _original_seed = np.random.seed

    def _seed(seed=None):
        if seed is not None and isinstance(seed, (int, np.integer)):
            seed = int(seed) % (2**32)
        return _original_seed(seed)

    np.random.seed = _seed  # type: ignore[method-assign]


_patch_numpy_seed_for_randomly()


def _configure_numba_cache_dir() -> None:
    """Give each pytest worker its own Numba cache before numba is imported."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    cache_dir = tempfile.mkdtemp(prefix=f"numba_cache_{worker}_")
    os.environ["NUMBA_CACHE_DIR"] = cache_dir


_configure_numba_cache_dir()


@pytest.fixture(scope="session", autouse=True)
def _isolated_numba_cache():
    """Ensures the Numba cache directory from :func:`_configure_numba_cache_dir`."""
    cache_dir = os.environ.get("NUMBA_CACHE_DIR", "")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
