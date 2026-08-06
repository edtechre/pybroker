# PyBroker Indicator Index

Use this index to choose the smallest relevant reference file before writing or debugging PyBroker indicator code.

## User Guide Wiki

- `wiki-05-writing-indicators.md` - Writing Indicators; 15 code cells.
  Topics: Writing Indicators; Using the Indicator in a Strategy; Vectorized Helpers; Computing Multiple Indicators; Using TA-Lib; Built-In Indicators.
- `wiki-11-configuring-parallelization.md` - Configuring Parallelization; 7 code cells.
  Topics: Configuring Parallelization; Setting Workers; Parallel Indicators; Parallel Model Training; Using Ray as the Backend.
- `wiki-15-multiple-time-intervals.md` - Multiple Time Intervals; 4 code cells.
  Topics: Multiple Time Intervals; Interval Types; Compressing Bars; A Multi-Timeframe Strategy.

## API And Pattern References

- `api-public-surface.md` - generated local public classes, functions, and methods.
- `pybroker_model.pyi` - generated type stubs: `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `pybroker_types.pyi` - generated type stubs: enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `pybroker_context.pyi` - generated type stubs: `ExecContext` writable order/stop attributes and properties, `IntervalContext`, `RotationContext`, `ExecResult`, and slippage models.
- `pybroker_strategy.pyi` - generated type stubs: `Strategy`, `StrategyConfig`, `TestResult`, and optimization types.
- `indicator-patterns.md` - vectorization patterns, third-party library recipes, session hygiene, and the validation checklist.

## Topic Routing

- First custom indicator, vector helpers, built-in factories, TA-Lib, or `IndicatorSet`: read `wiki-05-writing-indicators.md`.
- Parallel indicator computation, worker counts, or the Ray backend: read `wiki-11-configuring-parallelization.md`.
- Weekly/monthly indicators, bar compression, or multi-timeframe rules: read `wiki-15-multiple-time-intervals.md`.
- Third-party wrappers (TA-Lib, pandas-ta, ta, tulipy, finta), nested @njit kernels, hyperparam indicators, custom columns, lookahead self-tests, caching, Numba debugging: read `indicator-patterns.md`.
- Exact signatures for indicator(), IndicatorSet, the built-in factories, vect helpers, caches, or compress_bars: read `api-public-surface.md`.
- Exact type stubs for indicator()/Indicator/IndicatorSet and vector helpers: read `pybroker_model.pyi`; for `BarData` fields and the column/indicator scopes: read `pybroker_types.pyi`.
