---
name: dependency-migration-triage
description: >
  Rigorously triage a single Dependabot (or Renovate) dependency version-bump PR: pull the
  real changelog/release notes across the full old->new version range (not the PyPI summary
  blurb), always read the actual source diff between the two tags as well - a changelog
  records what changed and can never record what stayed the same, which is what most
  "we are unaffected" arguments actually rest on - map both against
  actual usage sites in this codebase, write a migration plan that distinguishes
  API-signature changes from conceptual/behavioral ones (a version bump can
  quietly change defaults or strictness with no signature change at all), verify whether
  existing tests would actually catch a regression in that exact spot or merely execute the
  line with stale data, write a regression test that fails on the old behavior when a real
  issue is found, fix what's fixable, and open a PR mirroring the migration that links back
  to the originating Dependabot PR. Use this whenever the user wants to review, assess,
  migrate, or avoid "merging blindly" a Dependabot/Renovate PR; asks "what changed" or "what
  needs to adapt" for a dependency bump; wants confidence a version bump is safe beyond "CI
  is green"; or just pastes a Dependabot PR number/link and asks to check it out. Trigger
  even for dependencies that look boring (docs tooling, linters, CI actions) — some of the
  highest-value findings here come from bumps everyone assumes are safe.
metadata:
  internal: true
---

# Dependency Migration Triage

Request: **$ARGUMENTS** (a Dependabot PR number/URL, or a dependency name + old version ->
new version)

You're triaging one dependency version bump. The goal is not "does it still import" — it's
"what actually changed underneath this version number, and does anything in *this* codebase
need to adapt because of it." A green CI run on the bump PR itself is necessary but not
sufficient: CI only proves the currently-selected checks still pass, and says nothing about
whether those checks were ever capable of catching the specific thing that changed. Two real
findings from this repo's own history illustrate why that distinction matters — read
`references/case-studies.md` before Phase 4 if this is your first time running this skill,
or whenever a phase result feels uncertain.

Work through these phases in order. Scale depth to actual signal (see the "Right-size the
investigation" note before Phase 1) — a trivial patch bump of a barely-used tool doesn't
warrant the same effort as a 4-major-version jump of a runtime dependency, and pretending
otherwise produces padded reports, not better decisions.

## Phase 0 — Identify the bump

Resolve the PR (or dependency name) to: package name, old version, new version, and where in
the repo it's declared (`requirements.txt`, `setup.cfg`, `pyproject.toml`, `.github/workflows/*.yml`
for Actions). If given a PR number, `gh pr view <n> --repo <owner>/<repo>` gets you the title
and diff.

**Right-size the investigation.** Before going deep, get a fast read on stakes:
- Is this a runtime dependency (imported by `src/`) or dev/docs tooling only (linters, type
  checkers, doc builders, CI actions)? Runtime deps that ship in the actual package deserve
  more scrutiny than a docs-only tool nobody's users ever touch.
- How big is the jump — a patch release, or several majors/many minors? Bigger jumps need
  more changelog reading, not more assumption.
- Does this repo's test suite even exercise the dependency's usage sites at all (see Phase
  5)? If a dependency has zero test coverage today, that's itself a finding worth surfacing
  regardless of whether this particular bump breaks anything.
A one-line patch bump of an unused-by-default dev tool can reasonably get a light pass. A
major-version bump of something `src/` imports directly cannot.

## Phase 1 — Get the real changelog, not the PyPI blurb

PyPI's description field and a Dependabot PR's auto-generated "Changelog" section are
starting points, not the source of truth — they're often truncated, and they only show the
*target* version's notes, not the full range you're crossing. Go to the dependency's actual
GitHub repo and read the CHANGELOG.md / release notes / "what's new" docs covering **every**
minor/major version between old and new (patch releases are usually folded into their minor
version's notes — confirm this is true for the specific project rather than assuming). Use
WebFetch/WebSearch. For a wide range, you don't need to quote every entry — you need to have
actually read enough to know whether something in it touches what this repo uses.

## Phase 2 — Read the diff between the two tags

Do this on every bump, including ones with a good changelog. A changelog records what
changed; it can never record what *stayed the same* — and most reachability arguments are
claims about the absence of change ("we're fine because the class we subclass is
untouched"). No prose will ever state that for you. The diff is also the only thing that
pins a change to an exact version and an exact symbol.

So the diff does two different jobs depending on what else exists:

- **No usable changelog** — an absent or empty GitHub release, notes that live only in an
  in-repo `CHANGES.rst`/`HISTORY.rst`, a bare dump of PR titles. Then the diff is the only
  primary source there is, and "I read the changelog" is a claim you are not in a position
  to make.
- **A usable changelog** — then it gives you leads and the diff turns them into proof, far
  more cheaply, because the prose already told you which files to open. An entry saying
  "add lazy random state in samplers" becomes "`sampler._rng` changed from
  `numpy.random.RandomState` to a `LazyRandomState` wrapper, at exactly v3.4.0" — which is
  the difference between suspecting a problem and being able to act on one.

Compare the tags directly — no clone needed:

```
gh api repos/<owner>/<repo>/compare/<old-tag>...<new-tag> \
  --jq '.commits[] | "- " + (.commit.message | split("\n")[0])'
gh api repos/<owner>/<repo>/compare/<old-tag>...<new-tag> \
  --jq '.files[] | "\(.status) \(.filename) +\(.additions)/-\(.deletions)"'
```

Add `| select(.filename == "<path>") | .patch` to read one file's hunks.

**`compare` truncates silently on wide ranges, and the failure mode is a false negative.**
Over many versions GitHub drops the `patch` field and then drops files from `.files`
altogether — with no error and no marker. Measured on joblib: `1.2.0...1.5.3` returns 134
files, 37 with `patch == null`, and omits `joblib/parallel.py` **entirely** despite it
changing by 542 lines; narrowing to `1.4.2...1.5.3` returns it with its patch intact. Read
naively, "the file isn't in the list" becomes "the file didn't change" — the exact opposite
of the truth, on the file that mattered most.

Two habits make that safe:
- **Step through the range** one minor at a time rather than spanning it in one call, and
  treat a single wide compare as a survey, never as evidence of absence.
- **Never conclude "unchanged" from a file's absence.** If a file you expected to change
  isn't listed, narrow the range until it appears, or fetch it at both tags and diff
  locally (`gh api repos/<o>/<r>/contents/<path>?ref=<tag> --jq .content | base64 -d`).
  Absence of evidence here really is not evidence of absence.

Extract, in the order that decides things:

1. **Which files changed.** Cross-reference against Phase 3's usage sites. A change confined
   to a subpackage this codebase never imports is unreachable, full stop.
2. **The patch for the files that *are* reachable** — and, for anything you intend to call
   safe, confirmation that its source is genuinely untouched rather than merely unmentioned.
3. **Packaging metadata** — `requires-python`, dependency pins, version ceilings. These
   decide what can resolve at all and are routinely absent from the prose.

Worked outcomes, all from real runs of this skill:

- **Sole source.** mypy 2.3.0 → 2.3.1 had no GitHub release. The diff was six commits; the
  only non-mypyc change replaced an `assert isinstance(...)` with a conditional to fix a
  crash — a *proof* the bump cannot newly reject previously-accepted code, which no
  changelog existed to state.
- **Scoping a vague warning.** alpaca-py 0.44.0's notes flagged one item "Breaking Change"
  with no scope. The file list showed it touching `alpaca/trading/requests.py` only, against
  a codebase that imports `alpaca.data` exclusively. Unreachable, settled in one command.
- **Proving non-change.** For a pytest-randomly 3.12 → 4.1 major bump, the questions that
  mattered were whether `-p no:randomly` still worked and whether xdist seed propagation
  survived. The diff answered both by showing the `pytest11` entry-point name and the
  `XdistHooks` class body byte-identical across the tags. A changelog cannot tell you that
  something is unchanged; only the diff can.

Track which source each later claim rests on. "Read in the diff" and "stated in the
changelog" are different strengths of evidence, and Phase 8 asks you to report them apart.

## Phase 3 — Map to actual usage

Grep `src/`, `tests/`, and (if that's the *only* place the dependency shows up) `docs/`
notebooks for every real call site — imports, function/class usage, config file settings
that reference the tool. Don't reason from the dependency's own docs in the abstract; ground
every claim in this codebase's actual call sites, with file:line references. If a dependency
isn't used anywhere in `src/` or `tests/`, say so plainly — that changes everything
downstream (see Phase 5).

## Phase 4 — Migration plan: API changes AND conceptual/behavioral ones

For each usage site found in Phase 3, cross-reference against Phase 1's changelog and
Phase 2's diff, and classify: **safe as-is** / **trivial fix** (renamed param, deprecated
arg, config key rename) / **real code change needed** / **new concept to adopt**.

The trap to avoid: only checking whether a function signature changed. Some of the most
consequential changes in a version bump are changes to *default behavior* with no signature
change at all — stricter type-checking defaults, changed numeric precision, a narrower
default timeout, a schema a third-party API now returns that your code doesn't expect. Read
`references/case-studies.md` for two concrete examples of exactly this class of bug in this
repo's own history. Ask explicitly, for each usage site: "if I changed nothing in my code,
could this dependency's new version make this line do something different than before?" —
that question catches what a signature diff alone won't.

## Phase 5 — Coverage check: does a test actually cover *this*, or just execute the line?

For every usage site flagged as anything other than "safe as-is" in Phase 4, find the
test(s) that exercise it and read them — don't just check whether coverage tooling marks the
line as executed. Ask: does the test construct the *specific input/shape* that the changed
behavior would affect, with *live* logic (or a mock/fixture reflecting the *new* reality), or
does it pass with data/mocks that predate the change and therefore can't reveal a regression
even though the line technically runs? A line at 100% coverage can still be worthless for
catching this exact class of bug — `references/case-studies.md` has a worked example. This
phase's output is not a percentage; it's a specific yes/no per flagged usage site, with
reasoning.

**When coverage turns out to be zero or inadequate — which is common for docs tooling, CI
actions, and other things outside the test suite's reach — you have to become the coverage
yourself, and that means verifying *both directions*, not just one.** The natural instinct is
to check "does the new version work" and stop once it does. That only catches regressions; it
silently misses the mirror case, where the *old* version was already broken and the bump is
actually a load-bearing fix rather than a no-op. The only way to tell those apart is to run
the exact same check against both versions and compare — if you're about to spend the effort
building/executing something to confirm the new version is fine, spend the same few minutes
running the identical check against the old version first. `references/case-studies.md`
(case study 4) documents this skill catching itself getting this wrong: an earlier run
verified the new version of a dependency built cleanly, concluded "safe, nothing to fix," and
never noticed the old version actually crashed against the current environment — something a
same-effort comparison against the old version would have caught immediately.

## Phase 6 — Write a regression test, only when there's a real finding

If Phase 4/5 turned up an actual breaking change with inadequate coverage: write a test that
constructs the exact input/shape that exposes it, and **prove it's a real regression test, not
a plausible-sounding one** — reproduce the old buggy behavior directly (revert your Phase 7
fix locally, or hand-run the old logic in a scratch snippet) and confirm the new test fails
against it, then reinstate the fix and confirm it passes. This is the difference between "I
think this would have failed" and "I watched it fail."

If Phase 4/5 found nothing wrong: say so and stop here. Don't manufacture a test to look
thorough — a fabricated regression test for a non-issue is noise that looks like rigor, and
future readers can't tell the difference between "this guards something real" and "this pads
the diff" unless you're honest about which is which right now.

## Phase 7 — Fix what's fixable

Mechanical/trivial migrations (deprecation warnings, renamed parameters, config key renames,
newly-required explicit arguments) get fixed inline as part of this same pass — don't leave
free wins for a human to redo. Real breaking changes get a proper code fix paired with the
Phase 6 regression test. If something needs a genuinely new concept adopted (not just a
find-replace), implement it, but flag in the PR description that this is a judgment call the
maintainer may want to review more closely than a mechanical fix.

**If this phase produced a real code change** (skip this entirely for a comment-only, nothing-
to-fix outcome — running either of these against a no-op finding is pure overhead), close the
loop on your own work before moving to Phase 8:

- Run the `silent-failure-hunter` agent (from the `pr-review-toolkit` plugin, if available in
  this environment — otherwise apply the same lens yourself: does the fix swallow an error
  anywhere, degrade to a fallback without surfacing it, or introduce a new way for something
  to fail quietly?) against the diff. This is the same failure class the whole skill exists to
  catch in third-party dependencies — pointing it at your own fix before a maintainer sees it
  is the same discipline turned inward, not a generic review step.
- If the fix touches secrets, authentication, or a publish/release pipeline (long-lived
  tokens, credential handling, OIDC/trusted-publishing config, CI steps that push artifacts
  somewhere), run `/security-review` (or apply that scrutiny yourself if unavailable) before
  proceeding. Skip this for everything else — most triages never go near credentials, and
  running it by default would be blanket overhead for no signal.

## Phase 8 — Open a PR that mirrors the migration

Open a PR (same repo/branch conventions as the rest of this project — check recent merged
PRs for the base branch, typically `dev`) containing the Phase 7 fix and Phase 6 test. The
PR description must:
- Link/reference the originating Dependabot PR number explicitly, so the maintainer can find
  it from either direction.
- State plainly which of three tiers each claim rests on, and don't blur them: *verified by
  executing something* (a test that really failed then passed, a notebook actually run,
  `mypy`/`ruff` actually invoked with the new version), *read in the diff* (Phase 2 — a file
  list proving unreachability, a hunk read directly), or *inferred from changelogs*. The
  middle tier is the one that gets silently promoted to the first or demoted to the third;
  name it as its own thing. If nothing needed fixing,
  the PR (or a comment on the Dependabot PR, if a whole new PR would be empty) should say
  plainly "verified — the following was checked and nothing needs to change," not stay
  silent.
- Let the maintainer choose their own depth: deep-review this PR, or just trust it and merge
  the Dependabot PR directly.

If this run is a **dry-run / calibration** (no real PR should be opened — e.g. while testing
this skill itself), stop after producing the would-be PR title + body + diff, and say so
explicitly instead of calling `gh pr create`.

## A note on trusting static tools

Lint/type-checker findings (ruff rules, mypy errors, security scanners) are hypotheses, not
verdicts, until you've traced the actual dataflow. A rule can flag a pattern that's provably
safe in context (e.g. a `zip()` over two sequences whose lengths are structurally guaranteed
equal by their type definitions) — `references/case-studies.md` has a worked example of
exactly this. Before calling anything a "bug" in your report, verify it against the real code
path, not just the rule's generic description of what it usually catches.
