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

   asv continuous master HEAD   # same invocation CI uses
   asv compare master HEAD      # diff table

Generate and preview the HTML dashboard:

.. code-block:: bash

   asv publish
   asv preview                # serves at http://127.0.0.1:8080

Benchmark suite
---------------

Currently one module: ``benchmarks/bench_backtest.py``.

- ``Walkforward.time_walkforward`` — steady-state walkforward on the pinned
  4-symbol / 2-year scenario. ``setup`` runs one warmup so Numba JIT
  compile is not counted in the timed method.
- ``Walkforward.peakmem_walkforward`` — peak RSS during the same call.
- ``WalkforwardCold.time_cold_walkforward`` — same scenario without setup
  warmup, so JIT compile is counted. Tracks the win from
  ``@njit(cache=True)``.

Reference dataset: ``tests/testdata/daily_1.pkl`` (4 symbols, 2 years daily,
2020 rows). Same fixture the test suite uses via ``tests/fixtures.py``.

Environment
-----------

``asv.conf.json`` uses ``environment_type: virtualenv`` so each commit is
benchmarked in a fresh virtualenv built from ``setup.cfg``. The install
command is ``python -mpip install .`` — no Poetry, no tox; just pip.

For local ad-hoc benchmarking you can switch the config to
``environment_type: existing`` (uses the currently activated venv) to
skip the per-commit env rebuild. Revert before committing if you edit
``asv.conf.json``.

Adding a benchmark
------------------

Create a new file under ``benchmarks/`` (or add a class to an existing
one). asv picks up any class with ``time_``, ``peakmem_``, or ``track_``
methods. ``setup`` runs before each benchmark method, ``teardown`` after.

See `the asv writing-benchmarks guide
<https://asv.readthedocs.io/en/latest/writing_benchmarks.html>`_ for
parametrized benchmarks, timeouts, and custom tracking metrics.
