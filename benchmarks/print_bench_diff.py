"""Compare asv benchmark results against a saved baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / ".asv" / "baseline-pre-numpy.json"
RESULTS_DIR = REPO_ROOT / ".asv" / "results"

MIGRATION_BENCH_RE = re.compile(
    r"^bench_backtest\.("
    r"Walkforward|WalkforwardLarge|WalkforwardScaled|PortfolioHeldStops|"
    r"WalkforwardMultiWindow|BacktestBarLoop|WalkforwardPerBar|"
    r"StoreSliceKernels|"
    r"WalkforwardTimeframes|WalkforwardModels|WalkforwardModelsPerSymbol|Determinism"
    r")\."
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _latest_result_file(machine: str, commit: str) -> Path | None:
    base = RESULTS_DIR / machine
    if not base.is_dir():
        return None
    prefix = commit[:8]
    matches = sorted(
        p for p in base.glob(f"{prefix}*.json") if p.name != "machine.json"
    )
    return matches[-1] if matches else None


def _parse_asv_results(path: Path) -> dict[str, dict[str, float | int | None]]:
    data = _load_json(path)
    columns = data.get("result_columns", ["result"])
    result_idx = columns.index("result") if "result" in columns else 0
    out: dict[str, dict[str, float | int | None]] = {}
    for bench_name, entries in data.get("results", {}).items():
        if not MIGRATION_BENCH_RE.match(bench_name):
            continue
        if not entries or len(entries) <= result_idx:
            continue
        result_vals = entries[result_idx]
        if bench_name.startswith("bench_backtest.Determinism."):
            time_val = None
            track_val = (
                result_vals[0]
                if isinstance(result_vals, list) and result_vals
                else result_vals
            )
        elif isinstance(result_vals, list) and len(result_vals) > 1:
            time_val = max(v for v in result_vals if v is not None)
            track_val = None
        else:
            time_val = (
                result_vals[0]
                if isinstance(result_vals, list) and result_vals
                else result_vals
            )
            track_val = None
        out[bench_name] = {
            "time": time_val,
            "peakmem": None,
            "track": track_val,
        }
    return out


def _collect_from_results_dir(
    commit: str, machine: str | None
) -> dict[str, dict[str, float | int | None]]:
    if machine is None:
        machines = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())
        if not machines:
            return {}
        machine = machines[-1]
    path = _latest_result_file(machine, commit)
    if path is None:
        return {}
    return _parse_asv_results(path)


def save_baseline(path: Path, commit: str, machine: str | None = None) -> None:
    metrics = _collect_from_results_dir(commit, machine)
    payload = {"commit": commit, "metrics": metrics}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"Saved baseline ({len(metrics)} benchmarks) to {path}")


def _fmt_time(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and value > 1:
        return f"{value:.2f}s"
    if isinstance(value, (int, float)):
        return f"{value * 1e6:.1f}us"
    return str(value)


def _pct_change(base: float | int | None, cur: float | int | None) -> str:
    if base is None or cur is None or base == 0:
        return "—"
    return f"{((cur - base) / base) * 100:+.1f}%"


def print_diff(
    baseline_path: Path,
    current_commit: str,
    machine: str | None = None,
    regression_threshold: float = 0.05,
) -> int:
    baseline = _load_json(baseline_path)
    base_metrics: dict = baseline.get("metrics", {})
    cur_metrics = _collect_from_results_dir(current_commit, machine)

    if not cur_metrics:
        print(
            "No current asv results found. Run:\n"
            "  asv run --bench "
            "'^(Walkforward|WalkforwardLarge|WalkforwardTimeframes|"
            "WalkforwardModels|Determinism)$'",
            file=sys.stderr,
        )
        return 1

    names = sorted(set(base_metrics) | set(cur_metrics))
    print(f"{'Benchmark':<42} {'Baseline':>10} {'Current':>10} {'Change':>8}")
    print("-" * 74)

    failed = False
    for name in names:
        b = base_metrics.get(name, {})
        c = cur_metrics.get(name, {})
        b_time = b.get("time")
        c_time = c.get("time")
        if name.startswith("bench_backtest.Determinism."):
            b_hash = b.get("track")
            c_hash = c.get("track")
            status = "OK" if b_hash == c_hash else "FAIL"
            if status == "FAIL":
                failed = True
            print(
                f"{name:<42} {str(b_hash):>10} {str(c_hash):>10} {status:>8}"
            )
            continue
        change = _pct_change(b_time, c_time)
        print(
            f"{name:<42} {_fmt_time(b_time):>10} {_fmt_time(c_time):>10} {change:>8}"
        )
        if (
            b_time is not None
            and c_time is not None
            and name.startswith(
                (
                    "bench_backtest.Walkforward.",
                    "bench_backtest.WalkforwardLarge.",
                )
            )
            and c_time > b_time * (1 + regression_threshold)
        ):
            failed = True

    if failed:
        print(
            "\nRegression detected (>5% on primary benches or hash mismatch)."
        )
        return 1
    print("\nNo regression detected on migration gate benchmarks.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--current", help="git commit hash for current results"
    )
    parser.add_argument("--machine", default=None)
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save baseline JSON from current asv results",
    )
    parser.add_argument(
        "--commit",
        help="Commit for --save-baseline (defaults to HEAD via git)",
    )
    args = parser.parse_args(argv)

    if args.save_baseline:
        commit = args.commit
        if commit is None:
            import subprocess

            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        save_baseline(args.baseline, commit, args.machine)
        return 0

    current = args.current
    if current is None:
        import subprocess

        current = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    return print_diff(args.baseline, current, args.machine)


if __name__ == "__main__":
    raise SystemExit(main())
