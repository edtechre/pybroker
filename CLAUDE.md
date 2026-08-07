# PyBroker — Core Library Guide

PyBroker is a Python framework for developing and backtesting algorithmic
trading strategies, with a focus on strategies driven by machine learning.
Its backtesting engine is built on NumPy and accelerated with Numba. Users
define per-bar execution functions that place orders through an
`ExecContext` (`Strategy.add_execution`); the engine replays historical
bars for multiple instruments, simulates fills with fees, slippage, stops,
and position limits through a `Decimal`-based `Portfolio`, and reports
results as `TestResult` DataFrames with bootstrapped confidence intervals
on the metrics. Models are trained and evaluated with Walkforward
Analysis: the data is split into successive train/test windows so models
only ever predict on bars that came after their training data. On top of
that sit user-defined and built-in indicators, hyperparameter optimization
(Optuna), multi-timeframe intervals, rotational trading, ranked position
sizing, disk caching of data/indicators/models, and parallelized
computation. Bar data comes from built-in data sources (Alpaca, Yahoo
Finance, AKShare) or any user-supplied DataFrame/`DataSource`.

Import name `pybroker`, PyPI name `lib-pybroker`. Version is
single-sourced at `src/pybroker/__init__.py:__version__` (setup.cfg reads
it via `attr:`). Active branch is `v2_preview`; PRs target `master`. This
file governs changes to the core library (`src/pybroker/`) and the
distributable agent skills (`skills/`).

## Commands

```bash
# Setup (once): Python 3.11+ venv, then editable install with test deps
pip install -e ".[test]"

# Tests (~5,000). Local venvs are gitignored (.venv*) — use the project's
# venv python if the checkout has one (e.g. .venv12/bin/python -m pytest).
python -m pytest                           # full suite
python -m pytest tests/test_<module>.py    # one module (1:1 with src)
python -m pytest -n auto --dist loadgroup  # parallel; keeps xdist_group pins (ray/loky)
python -m pytest -p no:randomly ...        # deterministic order when bisecting failures

# Quality gates (tox envs defined in setup.cfg)
tox -e format             # ruff format --diff — CHECK ONLY; `tox -e format -- src tests` to write
tox -e lint               # ruff check src tests
tox -e typecheck          # mypy on src (mypy version pinned in the tox env)
tox -e py311,py312,py313  # full test matrix

# Benchmarks (asv; see Performance & Benchmarks)
asv run --quick             # fast feedback, one sample per benchmark
asv continuous master HEAD  # what the CI PR gate runs (--factor 1.1)

# Docs — CI runs `tox -e docs` on Python 3.12; it is STRICT
# (`sphinx-build -n -W --keep-going`), so any warning fails the build.
# tox needs 3.11+, so run the same flags from a project venv instead;
# a bare `sphinx-build -b html` hides warnings that fail CI.
python -m sphinx -n -W --keep-going -b html docs/source/ docs/_build/
```

## Iron Rules

1. **Never leak future bars.** No negative indexing or backward shifts that
   read past the current bar; new indicators must join the no-lookahead
   sweep. (§ Lookahead-Bias Guardrails)
2. **Pandas only at the I/O boundary.** Core computation is NumPy + Numba;
   six modules are pandas-free and must stay that way. (§ Pandas Boundary)
3. **Every compiled kernel is `@njit(cache=True)`.** 87/87 today; zero
   exceptions. (§ NumPy + Numba Core)
4. **Never widen, mutate, or copy the user's input DataFrame.** Feature
   data stays numpy-backed and out-of-band. (§ Project Rules)
5. **Do not "clean up" mid-file or lazy imports.** They break intentional
   cycles; E402 is disabled in ruff for exactly this. (§ Architecture)
6. **Decimal for money and share counts; float64 for everything
   vectorized.** Quantize to cents only at the output boundary. (§ Money,
   Floats & Determinism)
7. **Identical results every run.** Preserve every determinism rationale
   comment; never iterate an unsorted set into results. (§ Money, Floats &
   Determinism)
8. **Never `git stash`; commit/push only when asked.** Use a detached
   worktree for comparisons. (§ Project Rules)

## Architecture & Layering

The layering ladder below is also the module inventory of `src/pybroker/`:

```
L0  common, vect, parallel   — import nothing from pybroker
L1  interval, log, config    — import common only
L2  scope                    — common, interval, log
L3  cache, portfolio, eval, slippage, data
L4  model, indicator, optimize
L5  context
L6  strategy                 — the only module that may import everything
    ext/data.py              — common + data only (optional data sources)
```

- **Runtime imports point downward only.** If a change needs an upward
  import, the code is in the wrong module — move it, don't import it.
- **Intentional cycle breaks — do not "fix" (as of 2.0.0):**
  - scope↔model: importlib indirection in `scope.py` (`_ModelImports`,
    lines 62-96) plus a mid-file `from pybroker.scope import ...` in
    `model.py` (~line 826).
  - optimize↔strategy: mid-file import block in `optimize.py` (~line 316)
    plus `TYPE_CHECKING`-only imports of `strategy`.
  - portfolio→slippage and slippage→strategy are `TYPE_CHECKING`-only.
  Ruff ignores E402 globally to permit these. Moving them to the top of
  the file creates real import cycles.
- **Public API:** everything public is re-exported from
  `src/pybroker/__init__.py` via the `import X as X` form; there is no
  `__all__` anywhere. A new public name means adding an aliased import
  there — that file *is* the export list.
- **Global state:** module-level convenience functions (`param`,
  `register_columns`, `enable_*_cache`, `hyperparam`, ...) delegate to the
  `StaticScope` singleton in `scope.py`. Exception: `set_parallel` mutates
  module-level config in `parallel.py`. Custom data columns must go
  through `register_columns`; `StaticScope` freezes columns while a
  strategy runs and keeps `ordered_data_cols` deterministic — never
  iterate the unordered `all_data_cols` into model input.

## NumPy + Numba Core

- **Every compiled function is decorated exactly `@njit(cache=True)`** —
  no bare `@njit`, no object mode, no exceptions. Import only
  `from numba import njit`; the codebase uses no `prange`, no
  `numba.typed`, no `objmode`. Kernels live in `vect.py` (indicators),
  `eval.py` (metrics), and a handful in `model.py`/`scope.py`/`interval.py`.
- **Boundary contract:** only scalars, ndarrays, and `NamedTuple`s of
  float/int cross the njit boundary (canonical: the result tuples in
  `eval.py`). Never pass dicts, dataclasses, or Python objects.
- **Validate outside, index inside:** njit kernels index without bounds
  checking, so bounds validation and float64/C-contiguity coercion happen
  in the Python caller (canonical: `_checked_stacked_lags` in `model.py`
  and its docstring).
- **Numba semantics traps:**
  - Division by zero *raises* `ZeroDivisionError` instead of returning
    inf — guard divisors explicitly (see `returnv` in `vect.py`).
  - `int(nan)` yields INT64_MIN, not an error — an unchecked
    out-of-bounds-write hazard.
  - Numba's RNG state is separate from NumPy's — seeding must happen
    *inside* compiled code (see `_seed_bootstrap` in `eval.py`).
- **Array discipline:** preallocate with `np.empty`/`np.full` and
  index-fill; never grow arrays in loops. Prefer O(n) algorithms — the
  house patterns are monotonic-deque rolling min/max and
  Neumaier-compensated rolling sums (`vect.py`).
- **dtypes:** float64 for numerics, int64 for indices, `datetime64[ns]`
  for dates.

## Pandas Boundary Policy

Pandas is an I/O format in this codebase, not a compute engine. It appears
where user data enters and where user-facing results leave; everything in
between runs on NumPy arrays and Numba kernels.

- **Pandas-free modules (must stay that way):** `portfolio`, `vect`,
  `cache`, `config`, `log`, `parallel`. Litmus test: if your diff adds
  `import pandas` to a module that doesn't already import it, the design
  is wrong — stop and restructure.
- **Sanctioned ingress** (DataFrame → ndarray): `DataSource.query`,
  `Strategy._fetch_data`, the `scope.py` frame→`SymbolArrayStore`
  converters (`symbol_array_store_from_frame` and siblings — the *only*
  DataFrame→ndarray conversion sites), and `indicator._to_bar_data`.
- **Sanctioned egress** (results → user): `Strategy._to_test_result` (the
  `TestResult` frames), `eval`'s `BootstrapResult` frames, `get_signals`,
  `ModelInput.to_dataframe` (for the user's `predict_fn`), and
  `ExecContext.input()`.
- **Grandfathered interior uses — frozen.** These exist, are closed to
  extension, and are not precedent for new pandas: indicator values
  carried as `pd.Series` between compute and `IndicatorScope.fetch` (which
  converts to ndarray and caches); `evaluate`'s immediate `to_numpy`
  ingress; `optimize` slicing walkforward windows as DataFrames;
  `pd.isna` in `_is_rankable` (`strategy.py`). Do not add to this list.
  Shrinking an entry toward pure ndarray is welcome only when results are
  bit-identical.
- **Never** in the per-bar loop, in any njit kernel or its per-bar caller,
  or in per-symbol inner loops.
- **Example code follows the same rule:** indicator functions and
  per-bar execution functions written anywhere — docstrings, docs,
  notebooks, skill content, answers — are implemented with NumPy
  (+ `@njit`), never pandas. Pandas is confined to the third-party TA
  wrapper boundary and the `train_fn`/`input_data_fn` model boundary.
- `SymbolArrayStore` hands out **read-only views** (buffers frozen with
  `writeable=False`; custom `__getstate__` rebuilds views after pickling).
  Never flip the writeable flag — copy if you must mutate.

## Lookahead-Bias Guardrails

- **The invariant:** every array observable by strategy code is pre-sliced
  `array[:end_index]` (exclusive right bound at the current bar).
  `ExecContext` holds no arrays — every property fetches through the
  scopes with the symbol's `sym_end_index`. `ctx.close[-1]` is the
  *current* bar precisely because the slice already happened.
- **Forbidden:** negative indexing into full-length arrays (a negative
  index silently wraps to the end of the series — the future); shifting
  future values backward; any indicator whose value at bar `i` depends on
  input at index > `i`.
- **Defensive patterns to imitate, not remove:**
  - `ColumnScope.fetch_value` raises on `end_index <= 0` and clamps
    overshoot instead of letting an index wrap (`scope.py:989-993`).
  - `IntervalScope.completed_index` clamps rather than allowing a
    negative index to wrap to a future compressed bar.
  - `IndicatorScope.fetch` raises `ValueError` rather than truncating an
    interval-bound indicator with a base bar index.
- **The regression net:** `test_indicator_does_not_look_ahead` in
  `tests/test_vect.py` (arguments in `_indicator_args`) runs every
  indicator kernel, bumps only the final bar, and asserts all earlier
  outputs are bit-identical — it caught a real negative-index wraparound
  bug in `price_change_oscillator`. **Every new indicator kernel must be
  registered in this sweep.**
- **Walkforward boundary:** the `lookahead` parameter enforces
  `test_start = train_end + lookahead`; the history store spans train
  through end-of-test *contiguously* so lag-1 features never silently
  reach `lookahead` bars back (`strategy.py`, `_build_window_stores`
  comment). Do not "simplify" the contiguity.
- **Legitimate patterns that are NOT lookahead** (do not flag or "fix"):
  post-backtest evaluation over the completed equity curve (`eval.py`);
  lag construction shifting past→present
  (`shifted[lag:] = values[:-lag]`); sortedness checks
  (`arr[:-1] <= arr[1:]`); `[-1]` on already-truncated context arrays.

## Money, Floats & Determinism

- **Decimal** is for money and share counts (`Portfolio`, `Order`,
  `Trade`, `Entry`, `Position`, `FeeInfo`/`fee_mode`). **float64** is for
  prices in transit, signals, scores, and all vectorized math. Convert
  with `to_decimal` (string round-trip); quantize to cents
  `ROUND_HALF_UP` **only at the output boundary** (`common.quantize`).
  The `Portfolio.capture_bar` pattern — float accumulation with
  `math.fsum` over `sorted` symbols, converted to Decimal once — is the
  template; don't invent new Decimal/float mixing.
- **Determinism is a shipped feature.** Backtests must produce identical
  results across runs and be independent of `PYTHONHASHSEED`. House
  patterns: iterate `sorted(symbols)`, never a raw set; `math.fsum` for
  order-independent sums; stops sorted by monotonic id; `-inf` (never
  NaN) as an unrankable sort key; bootstrap default `seed=42` applied
  inside njit; Optuna samplers explicitly seeded (`optimize.py`).
- The dense determinism rationale comments at these sites are
  load-bearing — never delete, shorten, or reword them.

## Code Style

- ruff format + check: line length **79**, double quotes, 4-space indent,
  target py312; lint select E4/E7/E9/F with E402 off (see Architecture).
  mypy must pass on `src` (`tox -e typecheck`).
- Typing: pre-PEP-604 `Optional[X]`/`Union[X, Y]`, but modern builtin
  generics (`dict[str, int]`, `tuple[str, ...]`); `NDArray[np.float64]`
  from `numpy.typing`; `Final` for module constants; `# type: ignore[code]`
  with the specific code named.
- Containers by intent: `NamedTuple` for immutable records (and the only
  struct allowed across the njit boundary); `@dataclass(frozen=True)` for
  configs and cache keys; a mutable dataclass only when mutation is
  required; hand-written `__init__` for hot-path stateful classes
  (`ExecContext`, `Portfolio`, the `*Scope` classes).
- Docstrings: Google style rendered by napoleon, with Sphinx roles
  (`` :class:`pybroker.scope.ColumnScope` ``); class-level `Attributes:`
  sections on dataclasses/NamedTuples. Module header is two string
  literals — the module docstring plus a *separate* copyright literal
  (Apache 2.0 with Commons Clause) — keep both, in that order.

## Testing Conventions

- `tests/test_<module>.py` maps 1:1 to `src/pybroker/<module>.py`. Shared
  fixtures live in `tests/fixtures.py` and are star-imported
  (`from .fixtures import *` — F403/F405 are per-file-ignored on purpose).
- Golden numbers are **computed, not hardcoded**: recompute expectations
  from the fixture DataFrame and compare against `round(x, 2)` to match
  money quantization. Use `assert_metrics_equal` (`tests/test_strategy.py`)
  and `assert_metric` (`tests/test_eval.py`) — never `==` on `EvalMetrics`
  (NaN fields).
- **No network in tests.** yfinance/alpaca are mocked; pinned pickles live
  in `tests/testdata/` (`daily_1.pkl` is the canonical dataset, shared
  with the benchmarks).
- Tests import private `_underscore` symbols from `pybroker.*` directly —
  that is the convention, not a smell.
- `tests/conftest.py` forces `ParallelConfig(n_jobs=1)` (autouse) and
  gives each xdist worker its own `NUMBA_CACHE_DIR`. JIT stays **on** —
  do not add `NUMBA_DISABLE_JIT` shortcuts.
- A new indicator needs golden-value tests plus registration in the
  no-lookahead sweep (§ Lookahead-Bias Guardrails).

## Performance & Benchmarks

- The asv suite lives in `benchmarks/` (`bench_backtest`, `bench_common`,
  `bench_data`, `bench_slippage`; config in `asv.conf.json`, compared
  against `master`). Process doc: `docs/source/benchmarking.rst`.
- **Perf-sensitive change → run the relevant benches before and after:**
  `asv continuous master HEAD` (a targeted `--bench <pattern>` pass first
  is fine). CI fails PRs on regressions > 1.1× unless the PR carries the
  `bench-override` label. **New hot path → add a benchmark.**
- `WalkforwardCold` intentionally includes Numba JIT compile time — it
  validates the `cache=True` contract. Never add warmup to it.
- Ad-hoc JSON-baseline runners exist for targeted comparisons
  (`scripts/bench_interval.py` + `.bench/timeframe-baseline.json`, and
  `benchmarks/run_*.py`); keep their baselines valid when touching those
  paths.

## Docs

- Docstrings *are* the API reference (Sphinx autodoc) — write them to
  publication quality.
- `docs/source/reference/pybroker.strategy.rst` carries a hand-curated
  `:exclude-members:` list — update it whenever public dataclass fields
  change.
- Build with `sphinx-build -n -W --keep-going -b html docs/source/
  docs/_build/` (see Commands) — the strict flags CI runs via
  `tox -e docs`. Any warning is a build failure; a bare
  `sphinx-build -b html` will not catch what CI catches.
- An include-only `.rst` under `docs/source/` is still discovered as its
  own document and ships as a `<no title>` page; add it to
  `exclude_patterns` in `conf.py` (`.. include::` still resolves it).
- Never create or edit `docs/source/notebooks/*.ipynb` unless explicitly
  requested — document in docstrings instead.

## Agent Skills

`skills/pybroker-{strategy-creator,indicator-creator,model-trainer,optimize,multi-interval,rotational-trading}/`
are distributable skills that teach downstream coding agents PyBroker usage;
users symlink them into their agent's skills directory
(`docs/source/agent-skills.rst`).

- **Generated vs hand-authored.** `references/wiki-*.md`,
  `references/api-public-surface.md`, and `references/pybroker_*.pyi` are
  generated from the local notebooks and source — never hand-edit them;
  regenerate with `.venv12/bin/python scripts/gen_skill_refs.py` and
  verify with `--check`. Hand-authored: `SKILL.md`, `assets/*_template.py`,
  the `*-patterns.md` references, and `wiki-index.md` outside the strategy
  creator's generated `## User Guide Wiki` block.
- **`SKILL.md` Overviews ship to the docs verbatim.**
  `docs/source/agent-skills.rst` includes the slice between the literal
  headings `## Overview` and `## Workflow` — keep both headings intact and
  put new guidance in `## Implementation Rules`, never in the Overview.
- **Shared skeleton & progressive disclosure.** Every skill keeps the
  frontmatter (`name` + trigger-phrase `description`) and the section
  order `Overview / Workflow / Implementation Rules / Common Deliverables
  / Resources`. `SKILL.md` stays lean; depth lives in the `*-patterns.md`
  reference, routed via `wiki-index.md` → smallest relevant wiki page.
  `assets/*_template.py` is the executable embodiment of the rules — keep
  it runnable and in sync when rules change.
- **Project Rules apply to skill content** (skills are public docs): no
  competitor platform names; parallelism documented via
  `set_parallel(n_jobs=...)` with Ray the only named backend;
  short-position docs show only `margin`/`unrealized_pnl`.
- **Non-negotiable requirements every skill must keep teaching** in its
  `SKILL.md` rules, pattern reference, and template:
  1. No lookahead in indicator logic: strictly forbid negative indexing
     into full-length arrays (a negative index silently wraps to the end
     of the series — the future) and backward shifts such as `shift(-1)`;
     a value at bar `i` may depend only on inputs at index `i` and
     earlier. (A per-symbol `shift(-1)` building the training *target*
     inside `train_fn` remains the sanctioned pattern.) Novel indicator
     logic self-tests with the bump-last-bar check: change only the final
     input bar and assert every earlier output is unchanged.
  2. Never use pandas to implement indicator or execution logic:
     indicator and execution-function logic is NumPy + Numba `@njit` —
     no `pd.Series`/`pd.DataFrame` construction and no pandas calls
     such as `.rolling`/`.ewm`/`.shift`/`.apply` inside indicator
     functions or per-bar execution functions. The only sanctioned
     pandas: the minimal frame built at a third-party TA wrapper
     boundary (indicator skill), and the `train_fn`/`input_data_fn`
     frames PyBroker hands to model code.
  3. Indicator output contract: a full-length one-dimensional array, one
     value per input bar, warmup left-padded with NaN — never a shortened
     array (pad third-party TA library outputs).
  4. Session hygiene: generated scripts start with
     `pybroker.disable_progress_bar()` (progress output floods agent
     context) and `pybroker.enable_data_source_cache("<name>")` (or
     `pybroker.enable_caches`) so reruns do not refetch data; add
     `pybroker.disable_logging()` for many-backtest runs such as
     `optimize`.
  5. Numba debug toggle: on an `@njit` compile or typing error, re-run
     once with the `NUMBA_DISABLE_JIT=1` environment variable to get a
     readable Python traceback, fix the code, then remove the variable —
     never leave JIT disabled in a final script. Debug indicator failures
     serially before `parallel_indicators=True` (joblib wraps worker
     tracebacks).
  6. Exact API shapes come from the bundled references: agents read the
     matching `references/pybroker_*.pyi` stub and
     `references/api-public-surface.md` instead of guessing signatures,
     and use current API only (`ctx.long_score`/`ctx.short_score`, never
     the deprecated `ctx.score`; `strategy.set_max_*_positions`, not the
     deprecated `StrategyConfig` fields).
  7. Never widen or mutate the user's input DataFrame; feature data stays
     out-of-band (work on a `.copy()` inside `train_fn` when adding a
     target column).
  8. Execution-function hygiene: guard lookbacks with `ctx.bars` or
     `warmup=`, and set at most one order side per symbol per bar.
  9. Backtesting framework, not financial advice: state assumptions
     explicitly and make no performance claims unsupported by the
     produced backtest.
  10. Validation without network: syntax-check generated files, prefer
      tiny local DataFrame fixtures for runs, and never assume optional
      packages (yfinance, TA-Lib, ML libraries) are installed — name the
      required `pip install`s.
  11. Machine-readable results: report `result.metrics_df` as the
      human-readable summary and teach `result.to_json()` /
      `result.to_json_str()` (and `opt.to_json()` for optimization) as
      the structured output path, with the `include=`/`max_rows=`/
      `symbols=` controls — not a blanket replacement for the metrics
      print, since the default JSON payload is usually larger.

## Project Rules

- **Never add columns to, widen, or copy the user's input DataFrame.**
  Feature and derived data stays numpy-backed, out-of-band (see the
  model-input docstring contract in `model.py`, ~line 1548).
- API design: obvious names; no user-side assembly of intermediate
  objects; reuse existing parameters before adding new ones; `predict_fn`
  uses the trained model's own API. If correct usage would need a
  documented workaround, **fix the API instead of documenting the
  workaround.**
- Never reference competitor backtesting platforms by name in code,
  docstrings, docs, or commit messages.
- Docs describe parallelism via `set_parallel(n_jobs=...)` only; Ray is
  the only backend that may be named.
- `PositionBar` short semantics: `equity` and `market_value` swap roles
  for short positions; docs and examples show only `margin` and
  `unrealized_pnl` for shorts.
- Git: never `git stash` (use a detached worktree for comparisons);
  commit/push only when asked; PRs target `master`.

## Before You Claim Done

Run these in order. "It compiles and the one test I wrote passes" is not
done.

1. **Blast radius:** enumerate every call site of each changed function
   (`grep -rn` across `src/` and `tests/`); trace consumers of changed
   return values.
2. **Entry-point parity:** confirm consistent behavior across `backtest`,
   `walkforward`, and `optimize`, and across pooled vs per-symbol model
   configurations.
3. `tox -e format` (apply with `tox -e format -- src tests` if it reports
   diffs)
4. `tox -e lint`
5. `tox -e typecheck`
6. Targeted tests, then the full suite: `python -m pytest`
7. If indicators, scopes, or context slicing were touched:
   `python -m pytest tests/test_vect.py -k look_ahead`
8. If perf-sensitive: `asv continuous master HEAD` — no regression
   > 1.1×.
9. If the public API changed: export added in `__init__.py`, docstrings
   complete, `:exclude-members:` in `pybroker.strategy.rst` updated.
10. If `skills/`, public signatures/docstrings in `src/`, or the doc
    notebooks changed: `.venv12/bin/python scripts/gen_skill_refs.py
    --check`; regenerate on drift.
