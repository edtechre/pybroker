"""Contains parallel execution configuration."""

"""Copyright (C) 2026 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from joblib import Parallel
from typing import Iterator, Optional

import joblib
from joblib.parallel import BACKENDS, EXTERNAL_BACKENDS, get_active_backend


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel execution used by PyBroker.

    PyBroker can compute indicators, train models, and run optimizations in
    parallel using `joblib <https://joblib.readthedocs.io/>`_. Optimization
    trials run on all available cores by default; indicator computation and
    model training additionally require their ``parallel_indicators``/
    ``parallel_models`` flags. Call ``set_parallel(n_jobs=1)`` to run
    everything sequentially, and read the current configuration with
    :func:`get_parallel_config`.

    Attributes:
        n_jobs: Number of worker jobs. ``-1`` (the default) uses all
            available cores; ``1`` runs sequentially.
        backend: joblib backend name. ``'loky'`` (the default) runs work in
            separate processes. Any backend registered with joblib is also
            accepted (e.g. ``'ray'`` after ``ray.util.joblib.register_ray()``).
        parallel: Optional pre-constructed :class:`joblib.Parallel` instance
            that overrides ``n_jobs`` and ``backend`` entirely. When set,
            PyBroker uses it directly and the caller owns its lifecycle.
            Defaults to ``None``.
    """

    n_jobs: Optional[int] = -1
    backend: Optional[str] = "loky"
    parallel: Optional[Parallel] = None


_config = ParallelConfig()


def set_parallel(
    n_jobs: Optional[int] = None,
    backend: Optional[str] = None,
    parallel: Optional[Parallel] = None,
) -> None:
    """Configures parallel execution used by PyBroker.

    PyBroker uses all available cores by default; call
    ``set_parallel(n_jobs=1)`` to run sequentially.

    Args:
        n_jobs: Number of workers. ``-1`` uses all cores (the default);
            ``1`` runs sequentially. Leave as ``None`` to keep the currently
            configured value.
        backend: joblib backend name: ``'loky'`` (default) or any backend
            registered via :func:`joblib.register_parallel_backend` (e.g.
            ``'ray'`` after ``ray.util.joblib.register_ray()``). The
            ``'multiprocessing'`` backend is rejected because its standard
            pickle serialization cannot ship PyBroker's dispatch closures.
        parallel: Pre-constructed :class:`joblib.Parallel` instance. Mutually
            exclusive with ``n_jobs``/``backend``; caller owns its lifecycle.

    Raises:
        ValueError: If ``parallel`` is passed together with ``n_jobs`` or
            ``backend``, if ``backend`` is not a registered joblib backend or
            is ``'multiprocessing'``, or if ``parallel`` returns results out
            of submission order.
    """
    global _config
    if parallel is not None:
        if n_jobs is not None or backend is not None:
            raise ValueError(
                "parallel is mutually exclusive with n_jobs and backend"
            )
        return_as = getattr(parallel, "return_as", "list")
        if return_as not in ("list", "generator"):
            # Every dispatch site pairs results back to their inputs by
            # position -- train_models zips tasks to models to recover
            # predict_fn and lag_columns, and optimize zips grid points to
            # scores. An unordered return silently binds each result to the
            # wrong input, and for models the mismatched lag_columns is then
            # written to the model cache and outlives the run.
            raise ValueError(
                f"parallel must return results in submission order; "
                f"return_as={return_as!r} does not. Use 'list' or "
                f"'generator'."
            )
        _config = ParallelConfig(parallel=parallel)
        return
    if backend is not None:
        if backend == "multiprocessing":
            # Indicator and optimize-trial work is dispatched as closures,
            # which the standard pickle used by this backend cannot
            # serialize -- every parallel run would fail with a pickling
            # error. loky ships closures via cloudpickle.
            raise ValueError(
                "The 'multiprocessing' backend is not supported: PyBroker "
                "dispatches work as closures, which its standard pickle "
                "serialization cannot handle. Use 'loky' (the default) for "
                "process-based parallelism."
            )
        registered = set(BACKENDS) | set(EXTERNAL_BACKENDS)
        if backend not in registered:
            raise ValueError(
                f"Unknown joblib backend {backend!r}. Registered backends: "
                f"{sorted(registered)}. Third-party backends (e.g. 'ray') "
                f"must be registered first — for Ray: ray.init() and "
                f"register_ray() from ray.util.joblib."
            )
    _config = ParallelConfig(
        n_jobs=_config.n_jobs if n_jobs is None else n_jobs,
        backend=backend if backend is not None else _config.backend,
        parallel=None,
    )


def get_parallel_config() -> ParallelConfig:
    """Returns the current parallel configuration"""
    return _config


class _RayContextWarningFilter(logging.Filter):
    """Drops Ray's warning that joblib's ``context`` argument is unsupported.

    joblib's process-backend plumbing passes ``context`` to every pool it
    configures; Ray's pool ignores it and warns once per process. The
    argument comes from joblib, not the caller, so the warning is noise for
    every Ray-backed run and actionable in none of them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "'context' argument is not supported" not in record.getMessage()


@contextmanager
def _silenced_ray_pool_logger() -> Iterator[None]:
    logger = logging.getLogger("ray.util.multiprocessing.pool")
    log_filter = _RayContextWarningFilter()
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        logger.removeFilter(log_filter)


@contextmanager
def parallel() -> Iterator[Parallel]:
    if _is_nested():
        # Already running inside a worker. Naming a backend explicitly
        # bypasses joblib's nesting protection, so guard it here: without
        # this, each of the outer workers would spawn its own full-width
        # pool.
        yield joblib.Parallel(n_jobs=1)
        return
    if _config.parallel is not None:
        yield _config.parallel
        return
    with joblib.parallel_backend(_config.backend, n_jobs=_config.n_jobs):
        if _config.backend == "ray":
            with _silenced_ray_pool_logger():
                yield joblib.Parallel(n_jobs=_config.n_jobs)
        else:
            yield joblib.Parallel(n_jobs=_config.n_jobs)


def _is_nested() -> bool:
    """Whether the caller is running inside a joblib worker.

    joblib swaps the active backend for a nested one inside workers, which
    bumps ``nesting_level`` above zero.
    """
    backend, _ = get_active_backend()
    return bool(getattr(backend, "nesting_level", 0))


def _effective_n_jobs() -> int:
    """Number of workers a :func:`parallel` pool would actually use.

    Returns ``1`` when work would run sequentially, so callers can skip
    fanning out rather than pay to set up a pool of one.
    """
    if _is_nested():
        return 1
    n_jobs = (
        _config.parallel.n_jobs
        if _config.parallel is not None
        else _config.n_jobs
    )
    return max(1, joblib.effective_n_jobs(n_jobs))
