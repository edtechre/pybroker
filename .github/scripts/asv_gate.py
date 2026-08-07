#!/usr/bin/env python3
"""Decide the benchmark gate from asv's stored results.

A plain ``--factor`` gate does not work on shared CI runners. Measured
across roughly ten ``asv continuous`` legs whose ``src/`` and
``benchmarks/`` were byte-identical between base and head -- so every
flagged result was noise:

    1.29  EvalKernels.time_sharpe_ratio                 16.5us
    1.18  CacheDiskHit.time_cache_disk_get               429us
    1.17  StoreBuildKernels...from_frame(10, 504)       1.39ms
    1.12  IndicatorKernels.time_cross(1000)             7.25us
    1.11  ModelPrepKernels.time_indicator_values...       60us
    1.11  ModelTrainPrep.time_train_models_per_symbol   1.92ms
    0.65  CacheDiskHit.time_cache_disk_get               633us

Every one is under 2ms, and the same benchmark has read 1.12 on one leg
and 0.83 on another *in the same run*. The walkforward macrobenchmarks
(100ms and up) have never been flagged. So the noise is a property of how
short the benchmark is, not of the threshold: no factor separates signal
from noise down there, it only trades false positives for blindness.

This gates on both axes. A benchmark blocks only when it is slow enough
to measure on a noisy runner *and* it regressed past the factor.
Everything else is reported, so a real microbenchmark regression stays
visible to a human -- it just does not block a PR on a coin flip.

Results come from asv's own API (``asv.results``), which reads the JSON
it already wrote under ``.asv/results/``. Values arrive as floats in
seconds, so there is no rendered table to parse and no units to convert.
"""

import argparse
import itertools
import sys
from typing import NamedTuple

from asv.config import Config
from asv.results import iter_results_for_machine_and_hash


def format_seconds(seconds: float) -> str:
    """Renders a duration in whichever unit reads most naturally."""
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:g}{unit}"
    return f"{seconds / 1e-9:g}ns"


class Change(NamedTuple):
    """One benchmark's before/after timing, both in seconds."""

    name: str
    before: float
    after: float

    @property
    def ratio(self) -> float:
        return self.after / self.before


def load_timings(
    results_dir: str, machine: str, commit: str
) -> dict[str, float]:
    """Maps benchmark name (with parameters) to its time in seconds.

    Parameterized benchmarks yield one entry per parameter combination,
    keyed the way asv names them -- ``bench.Class.time_x(4, 12)`` -- so
    a regression in one combination is not averaged away by the others.
    """
    timings: dict[str, float] = {}
    for result in iter_results_for_machine_and_hash(
        results_dir, machine, commit
    ):
        for name in result.get_all_result_keys():
            params = result.get_result_params(name)
            # Always a list -- one entry per parameter combination, and a
            # single-entry list for an unparameterized benchmark. Entries
            # are None where the benchmark failed or was skipped.
            values = result.get_result_value(name, params)
            if not params:
                if values and isinstance(values[0], float):
                    timings[name] = values[0]
                continue
            # asv flattens parameterized results with itertools.product
            # (see asv/results.py `_compatible_results`), so zipping the
            # same product is what keeps each value with its own
            # parameter combination.
            for combo, value in zip(itertools.product(*params), values):
                if isinstance(value, float):
                    timings[f"{name}({', '.join(combo)})"] = value
    return timings


class Verdict(NamedTuple):
    """How each changed benchmark was classified."""

    blocking: list[Change]
    reported: list[Change]
    improved: list[Change]


def classify(
    before: dict[str, float],
    after: dict[str, float],
    factor: float,
    floor: float,
    report_factor: float,
) -> Verdict:
    """Splits changed benchmarks into blocking, reported and improved.

    A benchmark blocks only if it is both slower than ``factor`` and slow
    enough overall to be measurable (``floor``). Everything that moved by
    ``report_factor`` either way is still reported, so a microbenchmark
    regression is visible even though it cannot block.
    """
    blocking: list[Change] = []
    reported: list[Change] = []
    improved: list[Change] = []
    # Sorted so the report is stable run to run, not dict-order dependent.
    for name in sorted(before.keys() & after.keys()):
        old, new = before[name], after[name]
        if not old > 0.0:
            continue
        change = Change(name, old, new)
        if change.ratio >= factor and old >= floor:
            blocking.append(change)
        elif change.ratio >= report_factor:
            reported.append(change)
        elif change.ratio <= 1.0 / report_factor:
            improved.append(change)
    return Verdict(blocking, reported, improved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="asv.conf.json")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--base", required=True, help="Base commit hash.")
    parser.add_argument("--head", required=True, help="Head commit hash.")
    parser.add_argument(
        "--factor",
        type=float,
        required=True,
        help="Ratio at or above which a slowdown blocks.",
    )
    parser.add_argument(
        "--report-factor",
        type=float,
        required=True,
        help="Ratio at or above which a movement is reported (never blocks).",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        required=True,
        help=(
            "Baseline runtime a benchmark must reach before it can block. "
            "Anything faster is reported only."
        ),
    )
    args = parser.parse_args()

    results_dir = Config.load(args.config).results_dir
    before = load_timings(results_dir, args.machine, args.base)
    after = load_timings(results_dir, args.machine, args.head)
    if not before or not after:
        print(
            f"::error::No stored results for "
            f"{'base' if not before else 'head'} on machine "
            f"{args.machine!r}. Nothing to gate on.",
            file=sys.stderr,
        )
        return 1

    verdict = classify(
        before, after, args.factor, args.min_seconds, args.report_factor
    )
    floor = format_seconds(args.min_seconds)

    if verdict.improved:
        print("Faster (never blocks):")
        for change in verdict.improved:
            print(f"  {change.ratio:>5.2f}  {change.name}")
    if verdict.reported:
        print(
            f"\nMoved, but not blocking (under {args.factor}x, or "
            f"under the {floor} floor where this runner's noise exceeds "
            "the factor):"
        )
        for change in verdict.reported:
            print(
                f"  {change.ratio:>5.2f}  {change.name}  "
                f"(baseline {format_seconds(change.before)})"
            )
    if verdict.blocking:
        print(
            f"\n::error::{len(verdict.blocking)} benchmark(s) at or above "
            f"{floor} regressed by {args.factor}x or more:"
        )
        for change in verdict.blocking:
            print(
                f"  {change.ratio:>5.2f}  {change.name}  "
                f"({format_seconds(change.before)} -> "
                f"{format_seconds(change.after)})"
            )
        return 1

    print(
        f"\nNo benchmark at or above {floor} regressed by "
        f"{args.factor}x or more ({len(before)} benchmarks compared)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
