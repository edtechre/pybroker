"""Pytest configuration for the test suite."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import os
import tempfile

import pytest


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
