"""Contains caching utilities.

Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pybroker.scope import StaticScope
from dataclasses import dataclass, is_dataclass
from datetime import datetime
from diskcache import Cache
from threading import RLock
from collections.abc import Iterable
from typing import Any, Final, Optional

_DEFAULT_CACHE_DIRNAME: Final = ".pybrokercache"

_L1_DEFAULT_MAXSIZE: Final = 1024


def _legacy_key(key: Any) -> Optional[Any]:
    """Return the pre-optimization ``repr(key)`` form, if applicable."""
    if is_dataclass(key) and not isinstance(key, type):
        legacy = repr(key)
        if legacy != key:
            return legacy
    return None


class _L1Cache(Cache):
    """:class:`diskcache.Cache` fronted by an in-process LRU of recent values.

    Repeated ``.get()`` for the same key within a single Python process is
    served from memory, skipping the disk I/O (and unpickling) diskcache does
    on every hit. This matters most during walkforward where the same
    indicator/model keys are re-read across windows.

    The L1 is bounded to ``l1_maxsize`` entries and evicts LRU on overflow.
    It is cleared alongside the underlying disk cache. Cross-process workers
    (joblib loky) do not share the L1; each worker has its own.
    """

    def __init__(
        self,
        *args: Any,
        l1_maxsize: int = _L1_DEFAULT_MAXSIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._l1: OrderedDict[Any, Any] = OrderedDict()
        self._l1_maxsize = l1_maxsize
        self._l1_lock = RLock()

    def _l1_put(self, key: Any, value: Any) -> None:
        with self._l1_lock:
            self._l1[key] = value
            self._l1.move_to_end(key)
            while len(self._l1) > self._l1_maxsize:
                self._l1.popitem(last=False)

    def _l1_evict(self, key: Any) -> None:
        with self._l1_lock:
            self._l1.pop(key, None)
            legacy = _legacy_key(key)
            if legacy is not None:
                self._l1.pop(legacy, None)

    def _l1_get(self, key: Any) -> tuple[bool, Any]:
        with self._l1_lock:
            try:
                value = self._l1[key]
            except KeyError:
                return False, None
            self._l1.move_to_end(key)
            return True, value

    def get(
        self, key: Any, default: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        found, value = self._l1_get(key)
        if found:
            return value
        value = super().get(key, default, *args, **kwargs)
        if value is default:
            legacy = _legacy_key(key)
            if legacy is not None:
                found, l1_value = self._l1_get(legacy)
                if found:
                    self._l1_put(key, l1_value)
                    return l1_value
                value = super().get(legacy, default, *args, **kwargs)
        if value is not default and value is not None:
            self._l1_put(key, value)
        return value

    def set(self, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        # Disk write first: populating the L1 before a failed write would
        # leave a phantom in-process entry that later reads "find" while
        # nothing was ever persisted.
        try:
            result = super().set(key, value, *args, **kwargs)
        except BaseException:
            self._l1_evict(key)
            raise
        if result:
            self._l1_put(key, value)
        else:
            self._l1_evict(key)
        return result

    def delete(self, key: Any, *args: Any, **kwargs: Any) -> Any:
        self._l1_evict(key)
        result = super().delete(key, *args, **kwargs)
        legacy = _legacy_key(key)
        if legacy is not None:
            super().delete(legacy, *args, **kwargs)
        return result

    def pop(self, key: Any, *args: Any, **kwargs: Any) -> Any:
        self._l1_evict(key)
        if args or kwargs:
            result = super().pop(key, *args, **kwargs)
        else:
            result = super().pop(key)
        legacy = _legacy_key(key)
        if legacy is not None:
            with self._l1_lock:
                self._l1.pop(legacy, None)
            try:
                super().pop(legacy)
            except KeyError:
                pass
        return result

    def __delitem__(self, key: Any, retry: bool = True) -> None:
        self._l1_evict(key)
        super().__delitem__(key, retry=retry)
        legacy = _legacy_key(key)
        if legacy is not None:
            try:
                super().__delitem__(legacy, retry=retry)
            except KeyError:
                pass

    def clear(self, *args: Any, **kwargs: Any) -> Any:
        with self._l1_lock:
            self._l1.clear()
        return super().clear(*args, **kwargs)


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
    adjust: Optional[str]
    # Identifies which DataSource produced the data: without it, two
    # sources sharing a cache namespace cross-serve each other's bars.
    source: str

    @classmethod
    def from_date_fields(
        cls,
        *,
        symbol: str,
        adjust: Optional[str],
        source: str,
        fields: CacheDateFields,
    ) -> DataSourceCacheKey:
        return cls(
            symbol=symbol,
            tf_seconds=fields.tf_seconds,
            start_date=fields.start_date,
            end_date=fields.end_date,
            adjust=adjust,
            source=source,
        )


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

    @classmethod
    def from_date_fields(
        cls,
        *,
        symbol: str,
        ind_name: str,
        fields: CacheDateFields,
    ) -> IndicatorCacheKey:
        return cls(
            symbol=symbol,
            tf_seconds=fields.tf_seconds,
            start_date=fields.start_date,
            end_date=fields.end_date,
            between_time=fields.between_time,
            days=fields.days,
            ind_name=ind_name,
        )


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
    # Composition of the pooled group this model was fit on, or ``None`` for a
    # per-symbol model. A pooled model is stored under one key per member, so
    # without this a run over {SPY, AAPL, TSLA} overwrites the entries written
    # by a run over {SPY, AAPL}, and the smaller run is then served a model fit
    # on a symbol set it never asked for. Two executions sharing one pooled
    # model over overlapping symbols clobber each other the same way.
    pooled_symbols: Optional[tuple[str, ...]] = None
    # Interval-bound models hold out ``lookahead`` *compressed* bars from the
    # train set, so two lookaheads that yield the same base-timeframe train
    # end_date still fit on different data. Base-timeframe models are fully
    # described by CacheDateFields' train start/end, so their keys stay
    # lookahead-free (``None``).
    lookahead: Optional[int] = None

    @classmethod
    def from_date_fields(
        cls,
        *,
        symbol: str,
        model_name: str,
        fields: CacheDateFields,
        pooled_symbols: Optional[Iterable[str]] = None,
        lookahead: Optional[int] = None,
    ) -> ModelCacheKey:
        return cls(
            symbol=symbol,
            tf_seconds=fields.tf_seconds,
            start_date=fields.start_date,
            end_date=fields.end_date,
            between_time=fields.between_time,
            days=fields.days,
            model_name=model_name,
            pooled_symbols=(
                None
                if pooled_symbols is None
                else tuple(sorted(pooled_symbols))
            ),
            lookahead=lookahead,
        )


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
    namespace: str,
    cache_dir: Optional[str] = None,
    l1_maxsize: int = _L1_DEFAULT_MAXSIZE,
) -> Cache:
    r"""Enables caching of data retrieved from
    :class:`pybroker.data.DataSource`\ s.

    Args:
        namespace: Namespace of the cache.
        cache_dir: Directory used to store cached data.
        l1_maxsize: Maximum in-process L1 cache entries.

    Returns:
        :class:`diskcache.Cache` instance.
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, "data_source")
    scope.data_source_cache_ns = namespace
    cache = _L1Cache(directory=cache_dir, l1_maxsize=l1_maxsize)
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
    namespace: str,
    cache_dir: Optional[str] = None,
    l1_maxsize: int = _L1_DEFAULT_MAXSIZE,
) -> Cache:
    """Enables caching indicator data.

    Args:
        namespace: Namespace of the cache.
        cache_dir: Directory used to store cached indicator data.
        l1_maxsize: Maximum in-process L1 cache entries.

    Returns:
        :class:`diskcache.Cache` instance.
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, "indicator")
    scope.indicator_cache_ns = namespace
    cache = _L1Cache(directory=cache_dir, l1_maxsize=l1_maxsize)
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
    namespace: str,
    cache_dir: Optional[str] = None,
    l1_maxsize: int = _L1_DEFAULT_MAXSIZE,
) -> Cache:
    """Enables caching trained models.

    Args:
        namespace: Namespace of the cache.
        cache_dir: Directory used to store cached models.
        l1_maxsize: Maximum in-process L1 cache entries.

    Returns:
        :class:`diskcache.Cache` instance.
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, "model")
    scope.model_cache_ns = namespace
    cache = _L1Cache(directory=cache_dir, l1_maxsize=l1_maxsize)
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


def enable_caches(
    namespace: str,
    cache_dir: Optional[str] = None,
    l1_maxsize: int = _L1_DEFAULT_MAXSIZE,
):
    """Enables all caches.

    Args:
        namespace: Namespace shared by cached data.
        cache_dir: Directory used to store cached data.
        l1_maxsize: Maximum in-process L1 cache entries per cache.
    """
    enable_data_source_cache(namespace, cache_dir, l1_maxsize=l1_maxsize)
    enable_indicator_cache(namespace, cache_dir, l1_maxsize=l1_maxsize)
    enable_model_cache(namespace, cache_dir, l1_maxsize=l1_maxsize)


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
