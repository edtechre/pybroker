"""Contains caching utilities."""

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

from .scope import StaticScope
from dataclasses import dataclass
from datetime import datetime
from diskcache import Cache
from typing import Final, Optional
import os

_DEFAULT_CACHE_DIRNAME: Final = ".pybrokercache"


@dataclass(frozen=True)
class CacheDateFields:
    """Date fields for keying cache data.

    Attributes:
        start_date: Start date of cache data.
        end_date: End date of cache data.
        tf_seconds: Timeframe resolution of cache data represented in seconds.
        between_time: ``tuple[str, str]`` of times of day (e.g. 9:00-9:30 AM)
            that were used to filter the cache data.
        days: Days (e.g. ``"mon"``, ``"tues"`` etc.) that were used to filter
            the cache data.
    """

    start_date: datetime
    end_date: datetime
    tf_seconds: int
    between_time: Optional[tuple[str, str]]
    days: Optional[tuple[int]]


@dataclass(frozen=True)
class DataSourceCacheKey:
    """Cache key used for :class:`pybroker.data.DataSource` data."""

    symbol: str
    tf_seconds: int
    start_date: datetime
    end_date: datetime


@dataclass(frozen=True)
class IndicatorCacheKey:
    """Cache key used for indicator data."""

    symbol: str
    tf_seconds: int
    start_date: datetime
    end_date: datetime
    between_time: Optional[tuple[str, str]]
    days: Optional[tuple[int]]
    ind_name: str


@dataclass(frozen=True)
class ModelCacheKey:
    """Cache key used for trained models."""

    symbol: str
    tf_seconds: int
    start_date: datetime
    end_date: datetime
    between_time: Optional[tuple[str, str]]
    days: Optional[tuple[int]]
    model_name: str


def _get_cache_dir(
    cache_dir: Optional[str], namespace: str, sub_dir: str
) -> str:
    if not namespace:
        raise ValueError("Cache namespace cannot be empty.")
    base_dir = (
        os.path.join(os.getcwd(), _DEFAULT_CACHE_DIRNAME)
        if cache_dir is None
        else cache_dir
    )
    return os.path.join(base_dir, namespace, sub_dir)


def enable_data_source_cache(
    namespace: str, cache_dir: Optional[str] = None
) -> Cache:
    r"""Enables caching of data retrieved from
    :class:`pybroker.data.DataSource`\ s.

    Args:
        namespace: Namespace of the cache.
        cache_dir: Directory used to store cached data.

    Returns:
        :class:`diskcache.Cache` instance.
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, "data_source")
    scope.data_source_cache_ns = namespace
    cache = Cache(directory=cache_dir)
    scope.data_source_cache = cache
    scope.logger.debug_enable_data_source_cache(namespace, cache_dir)
    return cache


def disable_data_source_cache():
    r"""Disables caching data retrieved from
    :class:`pybroker.data.DataSource`\ s.
    """
    scope = StaticScope.instance()
    scope.data_source_cache = None
    scope.data_source_cache_ns = ""
    scope.logger.debug_disable_data_source_cache()


def clear_data_source_cache():
    r"""Clears data cached from :class:`pybroker.data.DataSource`\ s.
    :meth:`enable_data_source_cache` must be called first before clearing.
    """
    scope = StaticScope.instance()
    cache = scope.data_source_cache
    if cache is None:
        raise ValueError(
            "Data source cache needs to be enabled before clearing."
        )
    cache.clear()
    scope.logger.debug_clear_data_source_cache(cache.directory)


def enable_indicator_cache(
    namespace: str, cache_dir: Optional[str] = None
) -> Cache:
    """Enables caching indicator data.

    Args:
        namespace: Namespace of the cache.
        cache_dir: Directory used to store cached data.

    Returns:
        :class:`diskcache.Cache` instance.
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, "indicator")
    scope.indicator_cache_ns = namespace
    cache = Cache(directory=cache_dir)
    scope.indicator_cache = cache
    scope.logger.debug_enable_indicator_cache(namespace, cache_dir)
    return cache


def disable_indicator_cache():
    """Disables caching indicator data."""
    scope = StaticScope.instance()
    scope.indicator_cache = None
    scope.indicator_cache_ns = ""
    scope.logger.debug_disable_indicator_cache()


def clear_indicator_cache():
    """Clears cached indicator data. :meth:`enable_indicator_cache` must be
    called first before clearing.
    """
    scope = StaticScope.instance()
    cache = scope.indicator_cache
    if cache is None:
        raise ValueError(
            "Indicator cache needs to be enabled before clearing."
        )
    cache.clear()
    scope.logger.debug_clear_indicator_cache(cache.directory)


def enable_model_cache(
    namespace: str, cache_dir: Optional[str] = None
) -> Cache:
    """Enables caching trained models.

    Args:
        namespace: Namespace of the cache.
        cache_dir: Directory used to store cached models.

    Returns:
        :class:`diskcache.Cache` instance.
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, "model")
    scope.model_cache_ns = namespace
    cache = Cache(directory=cache_dir)
    scope.model_cache = cache
    scope.logger.debug_enable_model_cache(namespace, cache_dir)
    return cache


def disable_model_cache():
    """Disables caching trained models."""
    scope = StaticScope.instance()
    scope.model_cache = None
    scope.model_cache_ns = ""
    scope.logger.debug_disable_model_cache()


def clear_model_cache():
    """Clears cached trained models. :meth:`enable_model_cache` must be called
    first before clearing.
    """
    scope = StaticScope.instance()
    cache = scope.model_cache
    if cache is None:
        raise ValueError("Model cache needs to be enabled before clearing.")
    cache.clear()
    scope.logger.debug_clear_model_cache(cache.directory)


def enable_caches(namespace, cache_dir: Optional[str] = None):
    """Enables all caches.

    Args:
        namespace: Namespace shared by cached data.
        cache_dir: Directory used to store cached data.
    """
    enable_data_source_cache(namespace, cache_dir)
    enable_indicator_cache(namespace, cache_dir)
    enable_model_cache(namespace, cache_dir)


def disable_caches():
    """Disables all caches."""
    disable_data_source_cache()
    disable_indicator_cache()
    disable_model_cache()


def clear_caches():
    """Clears cached data from all caches. :meth:`enable_caches` must be
    called first before clearing."""
    clear_data_source_cache()
    clear_indicator_cache()
    clear_model_cache()
