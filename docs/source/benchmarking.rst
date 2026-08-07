Benchmarking with asv
=====================

PyBroker uses `asv (Airspeed Velocity) <https://asv.readthedocs.io/>`_ to
track backtest performance across commits. asv runs benchmarks, stores
per-commit results under ``.asv/results/``, and produces an HTML
dashboard for visual regression tracking.

The benchmark suite lives in ``benchmarks/`` and is exercised in CI on
every pull request via ``.github/workflows/asv-pr.yml`` (uses
``asv continuous origin/master HEAD`` and posts a sticky PR comment with
the diff).

Installation
------------

.. code-block:: bash

   pip install asv
   asv machine --yes          # one-time per machine

Running
-------

Benchmark the current working tree:

.. code-block:: bash

   asv run --quick            # one sample per benchmark, fastest feedback
   asv run                    # calibrated samples, publication quality

Compare two commits:

.. code-block:: bash

   asv continuous master HEAD   # local equivalent of the CI gate
   asv compare master HEAD      # diff table

The PR gate uses two thresholds. It blocks at ``--factor 1.25`` and
reports everything that moved by ``1.1`` or more, resolving the base as
``origin/<base branch>``:

.. code-block:: bash

   asv continuous origin/master HEAD --factor 1.25 --interleave-rounds
   asv compare origin/master HEAD --factor 1.1 --only-changed

The second command re-reads the results the first one stored, so it costs
no extra benchmarking.

Why the gate is looser than the report: ``1.1`` is below a shared
runner's noise floor. Across six ``asv continuous`` runs whose ``src/``
and ``benchmarks/`` were byte-identical between base and head, one run
still flagged a regression — always a sub-2ms microbenchmark, at ratios
up to ``1.18``. The walkforward macrobenchmarks (100ms and up) never
moved. Gating at ``1.25`` clears the measured noise; the ``1.1`` table
keeps the smaller movements visible for a human to judge.

Two flags do part of the work but are not sufficient alone.
``--interleave-rounds`` alternates rounds between the two commits instead
of running each commit's rounds in a block, so drift over the job
(thermal throttling, noisy neighbours, page cache) hits both sides
equally instead of landing entirely on whichever commit ran second; it
reuses the existing rounds, so it is free. ``--no-stats`` is deliberately
*not* used: it disables significance testing, comparing raw medians
against ``--factor`` alone.

If the noise floor rises, raise the sampling
(``--attribute rounds=N``) rather than the gate factor — loosening the
factor trades away real coverage.

Generate and preview the HTML dashboard:

.. code-block:: bash

   asv publish
   asv preview                # serves at http://127.0.0.1:8080

Benchmark Suite
---------------

The asv suite lives in ``benchmarks/`` across four modules. CI fails PRs on
regressions greater than 1.25x unless the PR carries the ``bench-override``
label, and reports anything above 1.1x without blocking. New hot paths
should add a benchmark.

- ``bench_backtest.py`` - end-to-end walkforward (warm, cold, scaled,
  models, intervals, slippage-free) plus microbenchmarks for the indicator
  and eval kernels, ``SymbolArrayStore``, lag prep, and the caches. Also
  tracks a hash of walkforward equity so numeric divergence is flagged.
- ``bench_common.py`` - result-export quantize and interval compression.
- ``bench_data.py`` - data-source cache I/O and yfinance reshape, on pinned
  fixtures only.
- ``bench_slippage.py`` - walkforward under the volume and volatility
  slippage models.

Walkforward benches run on ``tests/testdata/daily_1.pkl`` (4 symbols, 2
years daily, 2020 rows), the fixture the test suite uses via
``tests/fixtures.py``; larger scenarios use synthetic OHLCV.
``WalkforwardCold`` and ``WalkforwardProperCold`` deliberately pay Numba
JIT compile cost, so never add warmup to either.

Environment
-----------

``asv.conf.json`` uses ``environment_type: virtualenv`` so each commit is
benchmarked in a fresh virtualenv built from ``setup.cfg``. The install
command is ``python -mpip install -e .``: no Poetry, no tox; just pip.

For local ad-hoc benchmarking you can switch the config to
``environment_type: existing`` (uses the currently activated venv) to
skip the per-commit env rebuild. Revert before committing if you edit
``asv.conf.json``.

Adding a Benchmark
------------------

Create a new file under ``benchmarks/`` (or add a class to an existing
one). asv picks up any class with ``time_``, ``peakmem_``, or ``track_``
methods. ``setup`` runs before each benchmark method, ``teardown`` after.

See `the asv writing-benchmarks guide
<https://asv.readthedocs.io/en/latest/writing_benchmarks.html>`_ for
parametrized benchmarks, timeouts, and custom tracking metrics.
