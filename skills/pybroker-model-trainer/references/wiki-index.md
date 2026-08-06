# PyBroker Model Training Index

Use this index to choose the smallest relevant reference file before writing or debugging PyBroker model code.

## User Guide Wiki

- `wiki-06-training-a-model.md` - Training A Model; 13 code cells.
  Topics: Training a Model; Train and Backtest; Walkforward Analysis.
- `wiki-16-time-series-models.md` - Time Series Models; 10 code cells.
  Topics: Time Series Models; Forecasting Volatility with GARCH; per-bar predictions; Random Forest on Lagged Returns; lags and lag_cols.
- `wiki-17-multi-symbol-models.md` - Multi-Symbol Models; 3 code cells.
  Topics: Multi-Symbol Models; Training One Model on Multiple Symbols; pooled training.

## API And Pattern References

- `api-public-surface.md` - generated local public classes, functions, and methods.
- `pybroker_model.pyi` - generated type stubs: `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `pybroker_context.pyi` - generated type stubs: `ExecContext` writable order/stop attributes and properties, `IntervalContext`, `RotationContext`, `ExecResult`, and slippage models.
- `pybroker_strategy.pyi` - generated type stubs: `Strategy`, `StrategyConfig`, `TestResult`, and optimization types.
- `pybroker_types.pyi` - generated type stubs: enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `model-training-patterns.md` - library recipes, session hygiene, and the leakage checklist.

## Topic Routing

- First model, train_fn basics, backtest vs walkforward, model caching: read `wiki-06-training-a-model.md`.
- GARCH or other autoregressive models, per-bar prediction, lagged features: read `wiki-16-time-series-models.md`.
- One model shared across symbols, pooled train_fn signature: read `wiki-17-multi-symbol-models.md`.
- Library recipes (scikit-learn, XGBoost/LightGBM/CatBoost, PyTorch/Keras, arch, statsmodels ARIMA, ensembles, regime models, classifiers, pretrained), tuning, leakage checks, Numba debugging: read `model-training-patterns.md`.
- Exact signatures for model(), walkforward(), caches, or ExecContext: read `api-public-surface.md`.
- Exact type stubs for model()/train_fn/predict_fn parameters: read `pybroker_model.pyi`; for ExecContext order attributes and prediction access: read `pybroker_context.pyi`.
