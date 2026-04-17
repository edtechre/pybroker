"""Unit tests for cache.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import os
import pytest
import re
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
    with (
        mock.patch.object(scope, "data_source_cache") as data_source_cache,
        mock.patch.object(scope, "indicator_cache") as ind_cache,
        mock.patch.object(scope, "model_cache") as model_cache,
    ):
        clear_caches()
        data_source_cache.clear.assert_called_once()
        ind_cache.clear.assert_called_once()
        model_cache.clear.assert_called_once()


@pytest.mark.usefixtures("setup_teardown")
def test_l1_cache_serves_repeated_get_from_memory(scope, cache_dir):
    cache = enable_indicator_cache("test", cache_dir)
    cache.set("k1", "v1")
    assert cache.get("k1") == "v1"
    # Second .get must not hit diskcache — patch the parent class .get.
    with mock.patch("diskcache.Cache.get") as disk_get:
        assert cache.get("k1") == "v1"
        disk_get.assert_not_called()


@pytest.mark.usefixtures("setup_teardown")
def test_l1_cache_clear_drops_memory_copy(scope, cache_dir):
    cache = enable_indicator_cache("test", cache_dir)
    cache.set("k1", "v1")
    assert cache.get("k1") == "v1"
    cache.clear()
    # After clear, a repeated get must fall through to diskcache (which is
    # also empty) — we verify by patching the parent .get and observing it
    # gets called exactly once.
    with mock.patch("diskcache.Cache.get", return_value=None) as disk_get:
        assert cache.get("k1") is None
        disk_get.assert_called_once()


@pytest.mark.usefixtures("setup_teardown")
def test_l1_cache_lru_evicts_oldest(scope, cache_dir):
    from pybroker.cache import _L1Cache

    cache = _L1Cache(directory=str(cache_dir / "l1") if cache_dir else None,
                     l1_maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # evicts "a" from L1
    # "a" is evicted from L1 but still on disk; patch disk to prove the
    # fallthrough actually happens.
    with mock.patch("diskcache.Cache.get", return_value=1) as disk_get:
        assert cache.get("a") == 1
        disk_get.assert_called_once()
    cache.close()
