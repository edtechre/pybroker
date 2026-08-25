# Case studies: what this skill exists to catch (and not overclaim)

Three real findings from this repo, each illustrating a different failure mode the phases in
SKILL.md are designed to catch. Read before Phase 4/5 if a result feels uncertain, or as a
sanity check on your own findings before writing them up.

## 1. Conceptual/behavioral change with no signature change (mypy)

Two separate mypy version bumps broke this codebase's typecheck step with **zero config
changes on this repo's side** — proof that a version bump can change what "correct" code
looks like without touching any function signature this repo calls.

- Commit `5d65b0c` ("Fix typecheck broken by mypy 2.0 ndarray narrowing"): mypy 2.0 reworked
  how it narrows `numpy.ndarray` generic types. Code that had type-checked cleanly for years
  suddenly didn't, because mypy got *more precise* about what it could prove — not because
  any pybroker code changed.
- Commit `bfd8172` ("Fix mypy 2.3 errors and pin mypy 2.3 release"): a second, unrelated
  tightening — this time around how `@njit`-wrapped function return types
  (`np.floating` vs `float`) flow through `Callable[...]` parameter matching.

The lesson: when triaging a type-checker, linter, or any tool whose whole job is "decide
whether code is correct," don't just check for removed functions or renamed parameters —
ask whether the tool's *definition of correct* changed. Running the new version against the
real codebase (Phase 4) is the only way to find this; changelog-reading alone under-reports
it unless the changelog explicitly calls out "stricter narrowing" or similar, and even then
you won't know which lines it touches without running it.

Related, sharper finding from the same investigation: running `mypy --show-error-codes
--warn-unused-ignores` (a flag this repo's own CI does *not* use) against the exact
CI-pinned environment found that **18 of 32 existing `# type: ignore` suppressions were
already dead** under the currently-pinned version — including two added by `bfd8172` itself,
already stale again by the time they were checked. The generalizable lesson: don't just ask
"does this pass," ask "would anything today catch drift going forward" — and if the answer
is no (as it was here, since `--warn-unused-ignores` wasn't enabled), that's a finding in its
own right, independent of the specific bump being triaged.

## 2. Coverage illusion (akshare)

`src/pybroker/ext/data.py`'s AKShare TX-fallback path renamed an incoming `"amount"` column
to `"volume"` unconditionally. As of akshare 1.18.74, the TX API started returning its own
real `"volume"` column, with `"amount"` repurposed as a distinct RMB figure — so the
unconditional rename produced **two columns both named `"volume"`** after the bump, silently
corrupting the output instead of raising.

The existing test for this exact code path (`test_query_when_em_unavailable_then_uses_tx_fallback`,
before the fix in PR #250) passed at 100% line coverage on every run, before and after the
akshare bump — because it mocked the fallback API's return value using the **old** schema.
The line executed every time; the mock never reflected the new reality, so the test could
never have caught this regression no matter how many times it ran.

The lesson (Phase 5): "this line has test coverage" and "a test here would catch a real
regression" are different claims. The only way to tell them apart is to read what shape of
data the test actually constructs and ask whether that shape still matches what the
dependency's new version actually returns — not whether the assertion count is nonzero.

Fixed in PR https://github.com/edtechre/pybroker/pull/250 — only remap the legacy `amount`
alias when a real `volume` column isn't already present, plus two tests pinning both the old
and new schema, each asserting `volume` appears exactly once (the literal thing that broke).

## 3. Overclaiming a lint finding (ruff)

`ruff`'s `B905` rule flagged `eval.py:1146` — `zip(("99.9%", "99%", "95%", "90%"), *metrics)`
— for missing `strict=True`, which in general risks silently truncating data on a length
mismatch. Read in isolation, that looks like a real bug and was initially reported as one.

Tracing the actual dataflow showed it wasn't: `metrics` is a `DrawdownMetrics` NamedTuple
containing two `DrawdownConfs` NamedTuples, both **hardcoded to exactly 4 fields** by their
type definitions. The three operands being zipped are therefore *statically guaranteed*
equal length — there is no code path where they could ever differ, so `B905`'s "silent
truncation" concern doesn't apply here no matter what future code changes happen elsewhere.

The lesson: a static/lint tool's finding describes a *pattern*, not a proof about this
specific code. Before writing "X is a bug" in a report, trace where the actual values come
from and confirm the failure mode the tool warns about is actually reachable. Getting this
wrong in the other direction — dismissing a real finding as "probably fine" without tracing
it — is just as much a failure of this phase; the discipline is doing the trace either way,
not defaulting to belief or disbelief.

## 4. Verifying only one direction (nbsphinx) — a miss this skill made against itself

During this skill's own calibration, one run triaged an `nbsphinx` bump that had zero test or
CI coverage (docs-tooling-only, nothing in the pipeline builds the docs). Correctly following
Phase 5's instinct to become the coverage, it built the actual documentation with the *new*
nbsphinx version, watched it succeed with zero warnings, and concluded "safe, nothing to
fix" — a clean, confident verdict, arrived at by actually executing something rather than
just reading a changelog.

It was still wrong. A parallel run (no skill at all, just told to check the same PR) happened
to build the docs with *both* the old and new versions to compare, and found the old version
**crashes outright** against the currently-installed Sphinx (`sphinx.util.status_iterator`
was removed from Sphinx; old nbsphinx still called it). The bump wasn't a no-op — it was the
only thing keeping the documentation buildable at all. The skill-following run never
discovered this because it only ever asked "does the new version work," never "was the old
version already broken."

The lesson (why Phase 5 now says to check both directions): confirming the new version works
answers half the question. It catches regressions but is blind to the mirror case — a bump
that's actually a load-bearing fix — because "safe, nothing to fix" and "safe, and here's
what it silently fixed" look identical from the forward-only check alone. Whenever you're
about to spend effort empirically verifying a version (because coverage is absent and
changelog-reading alone can't settle it), that same effort spent on the *old* version too is
usually cheap relative to what it can reveal, and skipping it means half of what an empirical
check could tell you never gets asked.
