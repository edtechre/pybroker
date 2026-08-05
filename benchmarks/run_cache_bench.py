"""Run cache bottleneck benchmarks and compare to a saved baseline."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / ".asv" / "baseline-pre-cache-opt.json"


def _median_ms(
    fn: Callable[[], None], warmup: int = 2, repeats: int = 7
) -> float:
    times: list[float] = []
    for i in range(warmup + repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(ms)
    return statistics.median(times)


def _run_cache_micro() -> dict[str, float]:
    from benchmarks.bench_backtest import (
        CacheDiskHit,
        CacheHit,
        CacheL1Hit,
    )

    out: dict[str, float] = {}

    # Legacy baseline bench (raw diskcache) — present before optimization.
    try:
        legacy = CacheHit()
        legacy.setup()
        try:
            out["CacheHit.time_cache_get"] = _median_ms(legacy.time_cache_get)
        finally:
            legacy.teardown()
    except Exception:
        pass

    disk = CacheDiskHit()
    disk.setup()
    try:
        out["CacheDiskHit.time_cache_disk_get"] = _median_ms(
            disk.time_cache_disk_get
        )
    finally:
        disk.teardown()

    l1 = CacheL1Hit()
    l1.setup()
    try:
        out["CacheL1Hit.time_cache_l1_get"] = _median_ms(l1.time_cache_l1_get)
    finally:
        l1.teardown()

    return out


def _run_cache_macro() -> dict[str, float]:
    from benchmarks.bench_backtest import (
        WalkforwardModels,
        WalkforwardModelsCached,
    )

    out: dict[str, float] = {}

    uncached = WalkforwardModels()
    uncached.setup()
    out["WalkforwardModels.time_walkforward_models"] = _median_ms(
        uncached.time_walkforward_models
    )

    cached = WalkforwardModelsCached()
    cached.setup()
    try:
        out["WalkforwardModelsCached.time_walkforward_models_cached"] = (
            _median_ms(cached.time_walkforward_models_cached)
        )
    finally:
        cached.teardown()

    return out


def run_all() -> dict[str, Any]:
    micro = _run_cache_micro()
    macro = _run_cache_macro()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    return {"commit": commit, "micro": micro, "macro": macro}


def save_baseline(path: Path) -> None:
    results = run_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "commit": results["commit"],
        "micro": results["micro"],
        "macro": results["macro"],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"Saved cache baseline to {path}")


def _pct(base: float, cur: float) -> str:
    if base == 0:
        return "—"
    return f"{((cur - base) / base) * 100:+.1f}%"


def compare(baseline_path: Path) -> int:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    cur = run_all()
    base_micro = base.get("micro", {})
    base_macro = base.get("macro", {})
    cur_micro = cur["micro"]
    cur_macro = cur["macro"]

    print(f"{'Benchmark':<56} {'Before':>10} {'After':>10} {'Change':>8}")
    print("-" * 88)

    all_micro = sorted(set(base_micro) | set(cur_micro))
    for name in all_micro:
        b, c = base_micro.get(name), cur_micro.get(name)
        if b is None:
            print(f"{name:<56} {'—':>10} {c:>10.2f} {'new':>8}")
        elif c is None:
            print(f"{name:<56} {b:>10.2f} {'—':>10} {'removed':>8}")
        else:
            print(f"{name:<56} {b:>10.2f} {c:>10.2f} {_pct(b, c):>8}")

    print()
    print("Macro benchmarks:")
    print(f"{'Benchmark':<56} {'Before':>10} {'After':>10} {'Change':>8}")
    print("-" * 88)

    all_macro = sorted(set(base_macro) | set(cur_macro))
    for name in all_macro:
        b, c = base_macro.get(name), cur_macro.get(name)
        if b is None:
            print(f"{name:<56} {'—':>10} {c:>10.2f} {'new':>8}")
        elif c is None:
            print(f"{name:<56} {b:>10.2f} {'—':>10} {'removed':>8}")
        else:
            print(f"{name:<56} {b:>10.2f} {c:>10.2f} {_pct(b, c):>8}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-baseline",
        type=Path,
        nargs="?",
        const=DEFAULT_BASELINE,
        default=None,
    )
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="?",
        const=DEFAULT_BASELINE,
        default=None,
    )
    args = parser.parse_args(argv)

    if args.save_baseline is not None:
        save_baseline(args.save_baseline)
        return 0
    if args.compare is not None:
        return compare(args.compare)

    results = run_all()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
