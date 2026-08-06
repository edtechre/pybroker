# PyBroker Optimization Index

Use this index to choose the smallest relevant reference file before writing or debugging PyBroker optimization code.

## User Guide Wiki

- `wiki-03-evaluating-with-bootstrap-metrics.md` - Evaluating With Bootstrap Metrics; 5 code cells.
  Topics: Evaluating with Bootstrap Metrics; Confidence Intervals; Maximum Drawdown.
- `wiki-11-configuring-parallelization.md` - Configuring Parallelization; 7 code cells.
  Topics: Configuring Parallelization; Setting Workers; Parallel Indicators; Parallel Model Training; Using Ray as the Backend.
- `wiki-12-parameter-optimization.md` - Parameter Optimization; 9 code cells.
  Topics: Parameter Optimization; Declaring Hyperparameters; Optimizing with Grid Search; Optimizing with Tree-structured Parzen Estimator (TPE); Other Samplers; Walkforward Optimization; hyperparam; ctx.hyperparam; OptimizeResult.

## API And Pattern References

- `api-public-surface.md` - generated local public classes, functions, and methods.
- `pybroker_strategy.pyi` - generated type stubs: `Strategy`, `StrategyConfig`, `TestResult`, and optimization types.
- `pybroker_context.pyi` - generated type stubs: `ExecContext` writable order/stop attributes and properties, `IntervalContext`, `RotationContext`, `ExecResult`, and slippage models.
- `pybroker_model.pyi` - generated type stubs: `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `pybroker_types.pyi` - generated type stubs: enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `optimization-patterns.md` - score-function recipes, Optuna integration, walkforward windows, and the optimization checklist.

## Topic Routing

- Declaring hyperparams, grid search, TPE, samplers, walkforward optimization basics: read `wiki-12-parameter-optimization.md`.
- Worker counts, parallel trials and indicators, the Ray backend: read `wiki-11-configuring-parallelization.md`.
- Score metric definitions, bootstrap confidence intervals, drawdown: read `wiki-03-evaluating-with-bootstrap-metrics.md`.
- Score-function recipes, Optuna studies and custom samplers, `windows` semantics, fixed hyperparams, model exclusion, grid explosions, debugging: read `optimization-patterns.md`.
- Exact signatures for optimize(), hyperparam(), walkforward(), or ExecContext: read `api-public-surface.md`.
- Exact type stubs for Strategy.optimize/Hyperparam/OptimizeResult/WindowOptimizeResult: read `pybroker_strategy.pyi`; for ctx.hyperparam and order attributes: read `pybroker_context.pyi`.
