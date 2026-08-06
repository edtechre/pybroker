# PyBroker Wiki Index

Use this index to choose the smallest relevant reference file before writing or debugging PyBroker strategy code.

## User Guide Wiki

- `wiki-01-getting-started-with-data-sources.md` - Getting Started With Data Sources; 10 code cells.
  Topics: Getting Started with Data Sources; Yahoo Finance; Caching Data; Alpaca; Alpaca Crypto; AKShare; Custom Data Sources.

- `wiki-02-backtesting-a-strategy.md` - Backtesting A Strategy; 13 code cells.
  Topics: Backtesting a Strategy; Defining Strategy Rules; Adding a Second Execution; Running a Backtest; Filtering Backtest Data.

- `wiki-03-evaluating-with-bootstrap-metrics.md` - Evaluating With Bootstrap Metrics; 5 code cells.
  Topics: Evaluating with Bootstrap Metrics; Confidence Intervals; Maximum Drawdown.

- `wiki-04-ranking-long-and-short-signals.md` - Ranking Long And Short Signals; 5 code cells.
  Topics: Ranking Long and Short Signals; Long Signals; Short Signals.

- `wiki-05-writing-indicators.md` - Writing Indicators; 15 code cells.
  Topics: Writing Indicators; Using the Indicator in a Strategy; Vectorized Helpers; Computing Multiple Indicators; Using TA-Lib; Built-In Indicators.

- `wiki-06-training-a-model.md` - Training A Model; 13 code cells.
  Topics: Training a Model; Train and Backtest; Walkforward Analysis.

- `wiki-07-creating-a-custom-data-source.md` - Creating A Custom Data Source; 4 code cells.
  Topics: Creating a Custom Data Source; Extending DataSource; Using a Pandas DataFrame.

- `wiki-08-applying-stops.md` - Applying Stops; 7 code cells.
  Topics: Applying Stops; Stop Loss; Take Profit; Trailing Stop; Setting a Limit Price; Canceling a Stop; Setting the Stop Exit Price.

- `wiki-09-rebalancing-positions.md` - Rebalancing Positions; 8 code cells.
  Topics: Rebalancing Positions; Equal Position Sizing; Portfolio Optimization.

- `wiki-10-rotational-trading.md` - Rotational Trading; 7 code cells.
  Topics: Rotational Trading; Custom Position Sizing.

- `wiki-11-configuring-parallelization.md` - Configuring Parallelization; 7 code cells.
  Topics: Configuring Parallelization; Setting Workers; Parallel Indicators; Parallel Model Training; Using Ray as the Backend.

- `wiki-12-parameter-optimization.md` - Parameter Optimization; 9 code cells.
  Topics: Parameter Optimization; Declaring Hyperparameters; Optimizing with Grid Search; Optimizing with Tree-structured Parzen Estimator (TPE); Other Samplers; Walkforward Optimization.

- `wiki-13-margin-trading.md` - Margin Trading; 10 code cells.
  Topics: Margin Trading; Configuring Leverage; Comparing Against Cash-Only; Charging Margin Interest; Shorting on Margin.

- `wiki-14-modeling-slippage.md` - Modeling Slippage; 6 code cells.
  Topics: Modeling Slippage; A Baseline Strategy; Fixed Slippage; Volatility Slippage; Volume Slippage; Writing a Custom Slippage Model.

- `wiki-15-multiple-time-intervals.md` - Multiple Time Intervals; 8 code cells.
  Topics: Multiple Time Intervals; Interval Types; Compressing Bars; A Multi-Timeframe Strategy; Binding an Indicator to an Interval; Training a Model on an Interval.

- `wiki-16-time-series-models.md` - Time Series Models; 10 code cells.
  Topics: Time Series Models; Forecasting Volatility with GARCH; Random Forest on Lagged Returns.

- `wiki-17-multi-symbol-models.md` - Multi-Symbol Models; 3 code cells.
  Topics: Multi-Symbol Models; Training One Model on Multiple Symbols.

- `wiki-18-dynamic-symbol-selection.md` - Dynamic Symbol Selection; 4 code cells.
  Topics: Dynamic Symbol Selection; Loading the Candidate Universe; Selecting Symbols by Liquidity; Running the Strategy.

- `wiki-faqs.md` - FAQs; 15 code cells.
  Topics: FAQs; How to...; See your version of PyBroker; Get data for another symbol; Set a limit price; Set the fill price; Get current positions; Use custom column data; Place an order more than one bar ahead; Cancel pending orders; Persist data across bars; Exit a position; Process multiple symbols at once; Annualize the Sharpe Ratio; Get and set a global parameter; Set a target allocation; Record position bars.

## API And Pattern References

- `api-public-surface.md` - generated local public classes, functions, and methods.
- `pybroker_context.pyi` - generated type stubs: `ExecContext` writable order/stop attributes and properties, `IntervalContext`, `RotationContext`, `ExecResult`, and slippage models.
- `pybroker_strategy.pyi` - generated type stubs: `Strategy`, `StrategyConfig`, `TestResult`, and optimization types.
- `pybroker_types.pyi` - generated type stubs: enums, `BarData`, `Portfolio`, order/trade/position records, and evaluation result types.
- `pybroker_model.pyi` - generated type stubs: `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `pybroker-patterns.md` - concise implementation patterns and validation checklist.

## Topic Routing

- Data downloads, caches, Alpaca/YFinance, or DataFrame inputs: read `wiki-01-getting-started-with-data-sources.md`.
- Core Strategy/ExecContext order logic: read `wiki-02-backtesting-a-strategy.md` and `pybroker-patterns.md`.
- Metrics, randomized bootstrap, or result inspection: read `wiki-03-evaluating-with-bootstrap-metrics.md`.
- Ranking long/short signals, scores, or max positions: read `wiki-04-ranking-long-and-short-signals.md`.
- Built-in or custom indicators: read `wiki-05-writing-indicators.md`.
- ML models, prediction inputs, caching, or walkforward training: read `wiki-06-training-a-model.md`.
- Custom DataSource, CSV, DataFrame, or custom columns: read `wiki-07-creating-a-custom-data-source.md`.
- Stop loss, profit stop, trailing stop, stop limits, or stop cancellation: read `wiki-08-applying-stops.md`.
- Rebalancing, before/after exec hooks, or portfolio-wide logic: read `wiki-09-rebalancing-positions.md`.
- Rotational strategies or ranking across a universe: read `wiki-10-rotational-trading.md`.
- Parallel execution, worker counts, or the Ray backend: read `wiki-11-configuring-parallelization.md`.
- Hyperparameters, parameter optimization, grid search, or TPE: read `wiki-12-parameter-optimization.md`.
- Margin, leverage, buying power, or interest charges: read `wiki-13-margin-trading.md`.
- Slippage or fill-price impact modeling: read `wiki-14-modeling-slippage.md`.
- Weekly/monthly bars, bar compression, or multi-timeframe rules: read `wiki-15-multiple-time-intervals.md`.
- GARCH, time-series forecasts, or per-bar model predictions: read `wiki-16-time-series-models.md`.
- Pooled or multi-symbol model training: read `wiki-17-multi-symbol-models.md`.
- Dynamic universes or `SymbolSelector`: read `wiki-18-dynamic-symbol-selection.md`.
- Edge cases and common questions: read `wiki-faqs.md`.
