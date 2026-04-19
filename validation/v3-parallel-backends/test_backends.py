#!/usr/bin/env python3
"""Validate joblib.Parallel across backends.

One backend per invocation. Appends a single markdown row to stdout.

Usage:
    python test_backends.py <backend>

Backends: loky, threading, multiprocessing, ray, dask, spark

Connection endpoints are taken from env vars so the same script works
whether the cluster services are reachable via podman-network hostnames
or localhost:
    RAY_ADDRESS (default: ray://ray-head:10001)
    DASK_SCHEDULER (default: tcp://dask-scheduler:8786)
    SPARK_MASTER (default: spark://spark-master:7077)
"""
import os
import sys
import time
import traceback

import numpy as np
from joblib import Parallel, delayed


def workload(i):
    time.sleep(0.2)
    return np.arange(i * 10, (i + 1) * 10, dtype=np.float64)


def setup_ray():
    import ray
    from ray.util.joblib import register_ray

    ray.init(address=os.environ.get("RAY_ADDRESS", "ray://ray-head:10001"))
    register_ray()


def setup_dask():
    import joblib
    from dask.distributed import Client
    from joblib._dask import DaskDistributedBackend

    # dask-distributed no longer ships its own joblib entry point; joblib
    # itself ships the backend class in joblib._dask. Register explicitly.
    joblib.register_parallel_backend("dask", DaskDistributedBackend)
    Client(os.environ.get("DASK_SCHEDULER", "tcp://dask-scheduler:8786"))


def setup_spark():
    from joblibspark import register_spark
    from pyspark.sql import SparkSession

    (
        SparkSession.builder.master(
            os.environ.get("SPARK_MASTER", "spark://spark-master:7077")
        )
        .appName("pybroker-validator")
        .getOrCreate()
    )
    register_spark()


BACKENDS = {
    "loky": (None, "default; memmap for NumPy"),
    "threading": (None, "GIL-bound"),
    "multiprocessing": (None, "legacy; loky supersedes"),
    "ray": (setup_ray, "ray://ray-head:10001"),
    "dask": (setup_dask, "tcp://dask-scheduler:8786"),
    "spark": (setup_spark, "spark://spark-master:7077"),
}


def run(backend):
    setup_fn, note = BACKENDS[backend]
    row = {"backend": backend, "wall_s": "-", "correct": "no", "note": note, "err": ""}
    try:
        if setup_fn:
            setup_fn()
        n = 10
        start = time.perf_counter()
        outputs = Parallel(backend=backend, n_jobs=4)(
            delayed(workload)(i) for i in range(n)
        )
        wall = time.perf_counter() - start
        expected = [workload(i).tolist() for i in range(n)]
        got = [
            o.tolist() if hasattr(o, "tolist") else list(o) for o in outputs
        ]
        row["wall_s"] = f"{wall:.3f}s"
        row["correct"] = "yes" if got == expected else "mismatch"
    except Exception as e:
        row["err"] = f"{type(e).__name__}: {str(e)[:120]}"
    return row


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in BACKENDS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(BACKENDS)}>", file=sys.stderr)
        sys.exit(2)
    row = run(sys.argv[1])
    note = row["note"]
    if row["err"]:
        note = f"{note} - {row['err']}"
    print(f"| {row['backend']:<15} | {row['wall_s']:>9} | {row['correct']:>9} | {note} |")
    if row["err"]:
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
