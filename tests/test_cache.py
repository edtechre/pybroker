"""Unit tests for cache.py module."""

"""Copyright (C) 2023 Edward West

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .fixtures import *
from diskcache import Cache
from pybroker.cache import (
    clear_caches,
    clear_data_source_cache,
    clear_indicator_cache,
    clear_model_cache,
    disable_caches,
    enable_caches,
    enable_data_source_cache,
    enable_model_cache,
    enable_indicator_cache,
)
from unittest import mock
import os
import pytest
import re


@pytest.fixture()
def setup_teardown(scope, tmp_path):
    with mock.patch.object(os, "getcwd", return_value=tmp_path):
        yield
    scope.data_source_cache = None
    scope.data_source_cache_ns = None
    scope.indicator_cache = None
    scope.indicator_cache_ns = None
    scope.model_cache = None
    scope.model_cache_ns = None


@pytest.fixture(params=["cache", None])
def cache_dir(request, tmp_path):
    return tmp_path / request.param if request.param is not None else None


@pytest.fixture()
def cache_path(tmp_path, cache_dir):
    return tmp_path / ".pybrokercache" if cache_dir is None else cache_dir


@pytest.mark.usefixtures("setup_teardown")
@pytest.mark.parametrize(
    "enable_fn, disable_fn, cache_attr",
    [
        (
            enable_data_source_cache,
            disable_data_source_cache,
            "data_source_cache",
        ),
        (enable_indicator_cache, disable_indicator_cache, "indicator_cache"),
        (enable_model_cache, disable_model_cache, "model_cache"),
    ],
)
def test_enable_and_disable_cache(
    scope, enable_fn, disable_fn, cache_attr, cache_dir, cache_path
):
    cache = enable_fn("test", cache_dir)
    assert cache is not None
    assert cache.directory
    assert len(list(cache_path.iterdir())) == 1
    assert isinstance(getattr(scope, cache_attr), Cache)
    assert getattr(scope, f"{cache_attr}_ns") == "test"
    disable_fn()
    assert getattr(scope, cache_attr) is None
    assert getattr(scope, f"{cache_attr}_ns") == ""


@pytest.mark.usefixtures("setup_teardown")
@pytest.mark.parametrize(
    "enable_fn, clear_fn, cache_attr",
    [
        (
            enable_data_source_cache,
            clear_data_source_cache,
            "data_source_cache",
        ),
        (enable_indicator_cache, clear_indicator_cache, "indicator_cache"),
        (enable_model_cache, clear_model_cache, "model_cache"),
    ],
)
def test_clear_cache_when_enabled_then_success(
    scope, enable_fn, clear_fn, cache_attr, cache_dir
):
    enable_fn("test", cache_dir)
    with mock.patch.object(scope, cache_attr) as cache:
        clear_fn()
        cache.clear.assert_called_once()


@pytest.mark.usefixtures("setup_teardown")
@pytest.mark.parametrize(
    "clear_fn, expected_msg",
    [
        (
            clear_data_source_cache,
            "Data source cache needs to be enabled before clearing.",
        ),
        (
            clear_indicator_cache,
            "Indicator cache needs to be enabled before clearing.",
        ),
        (
            clear_model_cache,
            "Model cache needs to be enabled before clearing.",
        ),
    ],
)
def test_clear_cache_when_not_enabled_then_error(clear_fn, expected_msg):
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        clear_fn()


@pytest.mark.usefixtures("setup_teardown")
@pytest.mark.parametrize(
    "enable_fn",
    [
        enable_data_source_cache,
        enable_indicator_cache,
        enable_model_cache,
        enable_caches,
    ],
)
def test_enable_cache_when_namespace_empty_then_error(enable_fn):
    with pytest.raises(
        ValueError, match=re.escape("Cache namespace cannot be empty.")
    ):
        enable_fn("")


@pytest.mark.usefixtures("setup_teardown")
def test_enable_and_disable_all_caches(scope, cache_dir, cache_path):
    enable_caches("test", cache_dir)
    assert len(list(cache_path.iterdir())) == 1
    assert isinstance(scope.data_source_cache, Cache)
    assert isinstance(scope.indicator_cache, Cache)
    assert isinstance(scope.model_cache, Cache)
    assert scope.data_source_cache_ns == "test"
    assert scope.indicator_cache_ns == "test"
    assert scope.model_cache_ns == "test"
    disable_caches()
    assert scope.data_source_cache is None
    assert scope.indicator_cache is None
    assert scope.model_cache is None
    assert scope.data_source_cache_ns == ""
    assert scope.indicator_cache_ns == ""
    assert scope.model_cache_ns == ""


@pytest.mark.usefixtures("setup_teardown")
def test_clear_all_caches(scope, cache_dir):
    enable_caches("test", cache_dir)
    with mock.patch.object(
        scope, "data_source_cache"
    ) as data_source_cache, mock.patch.object(
        scope, "indicator_cache"
    ) as ind_cache, mock.patch.object(
        scope, "model_cache"
    ) as model_cache:
        clear_caches()
        data_source_cache.clear.assert_called_once()
        ind_cache.clear.assert_called_once()
        model_cache.clear.assert_called_once()
