# V3 parallel-training API validation harness

Reproduces the results table posted to
[edtechre/pybroker#231](https://github.com/edtechre/pybroker/issues/231#issuecomment-4275860862).

**Not part of pybroker.** This directory is scaffolding to empirically
validate that every `joblib.Parallel` backend works against the proposed
`StrategyConfig.parallel` API shape, before writing the V3 PR.

## What it tests

One dummy workload (10 tasks, 0.2s sleep each, returning a NumPy array)
run through `Parallel(backend=<name>, n_jobs=4)` across:

- `loky` (default, fork + memmap)
- `threading`
- `multiprocessing`
- `ray` (cluster)
- `dask` (cluster)
- `spark` (cluster via `joblibspark`)

## Requirements on the host

- `podman` (tested on 5.8.x) — `docker` also works, swap the `podman`
  calls in `run_one.sh` for `docker`.
- No Python / joblib / anything else on the host. Everything runs in
  containers.

## Running

```bash
# One-time: build the two container images
podman build -t pybroker-validator:latest -f Containerfile .
podman build -t spark-with-joblib:latest -f Containerfile.spark .

# Run all backends, appending rows to results.md
./run_all.sh

# Or one at a time
./run_one.sh loky
./run_one.sh ray
# ...
```

Expected runtime: ~2 minutes total on a recent multi-core box. Cluster
backends (ray/dask/spark) bring up their containers, run one test, tear
down.

## Files

- `Containerfile` — tester image (`pybroker-validator`). Python 3.10 +
  joblib + numpy + ray + dask + pyspark + joblibspark. Used as the
  tester *and* as the Dask scheduler/worker image so joblib is on both
  sides of the wire (workers can't deserialize submitted callables
  otherwise).
- `Containerfile.spark` — Spark image extended with joblib, so Spark
  executors can deserialize batches.
- `test_backends.py` — one-shot test for a single backend; appends
  a markdown row to `results.md` on success.
- `run_one.sh <backend>` — orchestrates cluster containers (if any) +
  runs the tester + cleans up.
- `run_all.sh` — iterates all six backends.

## Notes / snags for reproducibility

- **joblib/dask API mismatch** (`joblib 1.5.3` + `dask 2026.3.0`):
  `joblib._dask.Batch.__call__` has signature `(self, tasks=None)`, but
  joblib's `Parallel` dispatch path passes extra kwargs (`nesting_level`)
  through `submit_kwargs`, raising `TypeError` worker-side. The
  `Containerfile` applies a one-line `sed` to widen the signature to
  `(self, tasks=None, **_kwargs)` — see the comment in the file. Needs
  filing upstream on `joblib/joblib`.
- **Cluster workers need joblib + user modules installed.** Standard
  Ray/Dask/Spark deployment constraint — not an API issue. The two
  Containerfiles here handle it for validation.
- Wall times are illustrative on an ~8-core box. Distribution overhead
  dominates at this tiny workload size; the validation is about "does
  the API wire up correctly", not performance.

## Cleanup

All cluster containers use `--rm` and are torn down by `run_one.sh`'s
`trap EXIT`. After `run_all.sh` ends:

- No running containers.
- No named volumes.
- The shared `pybroker-net` network is removed.
- Pulled/built images remain as cache — `podman rmi <image>` if a
  clean slate is wanted.
