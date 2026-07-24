"""Contains parallel execution configuration."""

"""Copyright (C) 2026 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

from contextlib import contextmanager
from dataclasses import dataclass
from joblib import Parallel
from typing import Iterator, Optional

import joblib
from joblib.parallel import BACKENDS, EXTERNAL_BACKENDS


@dataclass(frozen=True)
class ParallelConfig:
    n_jobs: Optional[int] = None
    backend: Optional[str] = "loky"
    parallel: Optional[Parallel] = None


_config = ParallelConfig()


def set_parallel(
    n_jobs: Optional[int] = None,
    backend: Optional[str] = None,
    parallel: Optional[Parallel] = None,
) -> None:
    """Configures parallel execution used by PyBroker

    Args:
        n_jobs: Number of workers. ``-1`` uses all cores. Defaults to ``-1``
            when unset.
        backend: joblib backend name: ``'loky'`` (default), ``'threading'``,
            ``'multiprocessing'``, or any backend registered via
            :func:`joblib.register_parallel_backend` (e.g. ``'ray'`` after
            ``ray.util.joblib.register_ray()``).
        parallel: Pre-constructed :class:`joblib.Parallel` instance. Mutually
            exclusive with ``n_jobs``/``backend``; caller owns its lifecycle.

    Raises:
        ValueError: If ``parallel`` is passed together with ``n_jobs`` or
            ``backend``, or if ``backend`` is not a registered joblib backend.
    """
    global _config
    if parallel is not None:
        if n_jobs is not None or backend is not None:
            raise ValueError(
                "parallel is mutually exclusive with n_jobs and backend"
            )
        _config = ParallelConfig(parallel=parallel)
        return
    if backend is not None:
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


@contextmanager
def parallel() -> Iterator[Parallel]:
    if _config.parallel is not None:
        yield _config.parallel
        return
    with joblib.parallel_backend(_config.backend, n_jobs=_config.n_jobs):
        yield joblib.Parallel(n_jobs=_config.n_jobs)
