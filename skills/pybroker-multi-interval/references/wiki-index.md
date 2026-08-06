# PyBroker Multi-Interval Index

Use this index to choose the smallest relevant reference file before writing or debugging PyBroker multi-timeframe code.

## User Guide Wiki

- `wiki-15-multiple-time-intervals.md` - Multiple Time Intervals; 8 code cells.
  Topics: Multiple Time Intervals; Interval Types; Compressing Bars; A Multi-Timeframe Strategy; Binding an Indicator to an Interval; Training a Model on an Interval.
- `wiki-05-writing-indicators.md` - Writing Indicators; 15 code cells.
  Topics: Writing Indicators; Using the Indicator in a Strategy; Vectorized Helpers; Computing Multiple Indicators; Using TA-Lib; Built-In Indicators.
- `wiki-06-training-a-model.md` - Training A Model; 13 code cells.
  Topics: Training a Model; Train and Backtest; Walkforward Analysis.

## API And Pattern References

- `api-public-surface.md` - generated local public classes, functions, and methods.
- `pybroker_context.pyi` - generated type stubs: `ExecContext` writable order/stop attributes and properties, `IntervalContext`, `RotationContext`, `ExecResult`, and slippage models.
- `pybroker_types.pyi` - generated type stubs: enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `pybroker_model.pyi` - generated type stubs: `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `pybroker_strategy.pyi` - generated type stubs: `Strategy`, `StrategyConfig`, `TestResult`, and optimization types.
- `multi-interval-patterns.md` - interval grammar, bars-versus-binding scope, compression semantics, the interval error table, and the validation checklist.

## Topic Routing

- Interval formats, compressing bars, a first multi-timeframe strategy, or the binding walkthroughs: read `wiki-15-multiple-time-intervals.md`.
- Writing the indicator function itself (vector helpers, nested @njit kernels, TA wrappers, custom columns): read `wiki-05-writing-indicators.md`.
- train_fn basics, walkforward mechanics, or model caching: read `wiki-06-training-a-model.md`.
- Grammar edge cases, "base" and exhaustive binding, bars-versus-binding scope, warmup guards, compressed-bar lookahead, the error table, or the bump-last-bar self-test: read `multi-interval-patterns.md`.
- Exact signatures for compress_bars, Indicator.intervals, ModelSource.intervals, or timeframe= parameters: read `api-public-surface.md`.
- Exact type stubs for IntervalContext and prediction access: read `pybroker_context.pyi`; for TimeframeInterval and BarData fields: read `pybroker_types.pyi`; for the binding types and compress_bars: read `pybroker_model.pyi`.
