"""Pytest configuration for the test suite."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_numba_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("numba_cache")
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
