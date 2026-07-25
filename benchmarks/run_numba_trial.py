"""Run Numba candidate micro + macro benchmarks and compare to baseline."""

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
DEFAULT_BASELINE = REPO_ROOT / ".asv" / "baseline-pre-numba.json"
DEFAULT_MICRO = REPO_ROOT / ".asv" / "baseline-pre-numba-micro.json"

MICRO_SPECS: list[tuple[str, Callable[[], None]]] = []
MACRO_SPECS: list[tuple[str, Callable[[], None]]] = []


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


def _run_lag_prep_bottlenecks() -> dict[str, float]:
    from benchmarks.bench_backtest import LagPrepBottlenecks

    b = LagPrepBottlenecks()
    b.setup()
    key = "LagPrepBottlenecks"
    return {
        f"{key}.time_merge_lag_cache_from_store": _median_ms(
            b.time_merge_lag_cache_from_store
        ),
        f"{key}.time_build_lag_feature_matrix_pooled": _median_ms(
            b.time_build_lag_feature_matrix_pooled
        ),
        f"{key}.time_apply_lags_pooled": _median_ms(b.time_apply_lags_pooled),
    }


def _run_model_train_prep_lags() -> dict[str, float]:
    from benchmarks.bench_backtest import ModelTrainPrepLags

    b = ModelTrainPrepLags()
    b.setup()
    return {
        "ModelTrainPrepLags.time_train_models_pooled_lags": _median_ms(
            b.time_train_models_pooled_lags
        )
    }


def _run_store_micro() -> dict[str, float]:
    from benchmarks.bench_backtest import StoreBuildKernels

    out: dict[str, float] = {}
    for n_symbols, n_days in [(4, 504), (10, 1260)]:
        b = StoreBuildKernels()
        b.setup(n_symbols, n_days)
        key = f"StoreBuildKernels.n_symbols={n_symbols}.n_days={n_days}"
        out[f"{key}.time_symbol_array_store_from_frame"] = _median_ms(
            lambda b=b, ns=n_symbols, nd=n_days: (
                b.time_symbol_array_store_from_frame(ns, nd)
            )
        )
    return out


def _run_model_prep_micro() -> dict[str, float]:
    from benchmarks.bench_backtest import ModelPrepKernels, ModelTrainPrep

    out: dict[str, float] = {}
    b = ModelPrepKernels()
    b.setup()
    out["ModelPrepKernels.time_indicator_values_for_dates"] = _median_ms(
        b.time_indicator_values_for_dates
    )
    prep = ModelTrainPrep()
    prep.setup()
    out["ModelTrainPrep.time_train_models_per_symbol"] = _median_ms(
        prep.time_train_models_per_symbol
    )
    out["ModelTrainPrep.time_train_models_pooled"] = _median_ms(
        prep.time_train_models_pooled
    )
    return out


def _run_macro() -> dict[str, float]:
    from benchmarks.bench_backtest import (
        Determinism,
        PortfolioHeldStops,
        Walkforward,
        WalkforwardLarge,
        WalkforwardModels,
        WalkforwardModelsPerSymbol,
        WalkforwardTimeframes,
    )

    specs = [
        ("Walkforward.time_walkforward", Walkforward),
        (
            "PortfolioHeldStops.time_portfolio_held_stops",
            PortfolioHeldStops,
        ),
        ("WalkforwardLarge.time_walkforward_large", WalkforwardLarge),
        (
            "WalkforwardTimeframes.time_walkforward_timeframes",
            WalkforwardTimeframes,
        ),
        ("WalkforwardModels.time_walkforward_models", WalkforwardModels),
        (
            "WalkforwardModelsPerSymbol.time_walkforward_models_per_symbol",
            WalkforwardModelsPerSymbol,
        ),
    ]
    out: dict[str, float] = {}
    for name, cls in specs:
        b = cls()
        b.setup()
        method = name.split(".", 1)[1]
        out[name] = _median_ms(lambda b=b, m=method: getattr(b, m)())

    det = Determinism()
    det.setup()
    t0 = time.perf_counter()
    h = det.track_walkforward_equity_hash()
    out["Determinism.track_walkforward_equity_hash"] = (
        time.perf_counter() - t0
    ) * 1000
    out["Determinism.hash"] = float(h) if h is not None else 0.0
    return out


def run_all() -> dict[str, Any]:
    micro: dict[str, float] = {}
    micro.update(_run_lag_prep_bottlenecks())
    micro.update(_run_model_train_prep_lags())
    micro.update(_run_store_micro())
    micro.update(_run_model_prep_micro())
    macro = _run_macro()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    return {"commit": commit, "micro": micro, "macro": macro}


def save_baseline(path: Path, micro_path: Path) -> None:
    results = run_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    macro_payload = {"commit": results["commit"], "metrics": results["macro"]}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(macro_payload, fh, indent=2, sort_keys=True)
    micro_payload = {"commit": results["commit"], "metrics": results["micro"]}
    with micro_path.open("w", encoding="utf-8") as fh:
        json.dump(micro_payload, fh, indent=2, sort_keys=True)
    print(f"Saved macro baseline to {path}")
    print(f"Saved micro baseline to {micro_path}")


def _pct(base: float, cur: float) -> str:
    if base == 0:
        return "—"
    return f"{((cur - base) / base) * 100:+.1f}%"


def compare(
    baseline_path: Path,
    micro_path: Path,
    macro_regression: float = 0.05,
    micro_improve: float = 0.05,
    macro_improve: float = 0.03,
) -> int:
    base_macro = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_micro = json.loads(micro_path.read_text(encoding="utf-8"))
    cur = run_all()
    cur_macro = cur["macro"]
    cur_micro = cur["micro"]
    base_m = base_macro.get("metrics", {})
    base_u = base_micro.get("metrics", {})

    failed = False
    print(f"{'Benchmark':<56} {'Before':>10} {'After':>10} {'Change':>8}")
    print("-" * 88)
    for name in sorted(set(base_m) | set(cur_macro)):
        if name == "Determinism.hash":
            b, c = base_m.get(name), cur_macro.get(name)
            ok = b == c
            if not ok:
                failed = True
            print(
                f"{name:<56} {str(b):>10} {str(c):>10} {'OK' if ok else 'FAIL':>8}"
            )
            continue
        b, c = base_m.get(name), cur_macro.get(name)
        if b is None or c is None:
            continue
        ch = _pct(b, c)
        print(f"{name:<56} {b:>10.2f} {c:>10.2f} {ch:>8}")
        if name.startswith(
            ("Walkforward.", "WalkforwardLarge.", "PortfolioHeldStops.")
        ) and c > b * (1 + macro_regression):
            failed = True

    print()
    print("Micro benchmarks:")
    print(f"{'Benchmark':<56} {'Before':>10} {'After':>10} {'Change':>8}")
    print("-" * 88)
    micro_improved = False
    for name in sorted(set(base_u) | set(cur_micro)):
        b, c = base_u.get(name), cur_micro.get(name)
        if b is None or c is None:
            continue
        ch = _pct(b, c)
        print(f"{name:<56} {b:>10.2f} {c:>10.2f} {ch:>8}")
        if c < b * (1 - micro_improve):
            micro_improved = True

    macro_improved = any(
        cur_macro.get(k, 1e9) < base_m.get(k, 0) * (1 - macro_improve)
        for k in base_m
        if k.startswith("Walkforward") and k != "Determinism.hash"
    )
    print()
    if failed:
        print("FAIL: macro regression or determinism hash mismatch")
        return 1
    if micro_improved or macro_improved:
        print("PASS: measurable improvement without macro regression")
        return 0
    print("FAIL: no meaningful improvement")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--micro-baseline", type=Path, default=DEFAULT_MICRO)
    args = parser.parse_args(argv)

    if args.save_baseline:
        save_baseline(args.baseline, args.micro_baseline)
        return 0
    if args.compare:
        return compare(args.baseline, args.micro_baseline)
    results = run_all()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
