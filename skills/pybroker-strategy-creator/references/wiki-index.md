# PyBroker Wiki Index

Use this index to choose the smallest relevant reference file before writing or debugging PyBroker strategy code.

## User Guide Wiki

- `wiki-01-getting-started-with-data-sources.md` - Getting Started With Data Sources; 10 code cells.
  Topics: Getting Started with Data Sources; Yahoo Finance; Caching Data; Alpaca; Alpaca Crypto; AKShare.
- `wiki-02-backtesting-a-strategy.md` - Backtesting A Strategy; 14 code cells.
  Topics: Backtesting a Strategy; Defining Strategy Rules; Adding a Second Execution; Running a Backtest; Filtering Backtest Data.
- `wiki-03-evaluating-with-bootstrap-metrics.md` - Evaluating With Bootstrap Metrics; 5 code cells.
  Topics: Evaluating with Bootstrap Metrics; Confidence Intervals; Maximum Drawdown.
- `wiki-04-ranking-and-position-sizing.md` - Ranking And Position Sizing; 8 code cells.
  Topics: Ranking and Position Sizing; Ranking Ticker Symbols; Setting Position Sizes.
- `wiki-05-writing-indicators.md` - Writing Indicators; 15 code cells.
  Topics: Writing Indicators; Using the Indicator in a Strategy; Vectorized Helpers; Computing Multiple Indicators; Using TA-Lib; Built-In Indicators.
- `wiki-06-training-a-model.md` - Training A Model; 13 code cells.
  Topics: Training a Model; Train and Backtest; Walkforward Analysis.
- `wiki-07-creating-a-custom-data-source.md` - Creating A Custom Data Source; 4 code cells.
  Topics: Creating a Custom Data Source; Extending DataSource; Using a Pandas DataFrame.
- `wiki-08-applying-stops.md` - Applying Stops; 7 code cells.
  Topics: Applying Stops; Stop Loss; Take Profit; Trailing Stop; Setting a Limit Price; Canceling a Stop.
- `wiki-09-rebalancing-positions.md` - Rebalancing Positions; 9 code cells.
  Topics: Rebalancing Positions; Equal Position Sizing; Portfolio Optimization.
- `wiki-10-rotational-trading.md` - Rotational Trading; 7 code cells.
  Topics: Rotational Trading.
- `wiki-faqs.md` - FAQs; 15 code cells.
  Topics: FAQs; How to...; ... get your version of PyBroker?; ... get data for another symbol?; ... set a limit price?; ... set the fill price?.

## API And Pattern References

- `api-public-surface.md` - generated local public classes, functions, and methods.
- `pybroker-patterns.md` - concise implementation patterns and validation checklist.

## Topic Routing

- Data downloads, caches, Alpaca/YFinance, or DataFrame inputs: read `wiki-01-getting-started-with-data-sources.md`.
- Core Strategy/ExecContext order logic: read `wiki-02-backtesting-a-strategy.md` and `pybroker-patterns.md`.
- Metrics, randomized bootstrap, or result inspection: read `wiki-03-evaluating-with-bootstrap-metrics.md`.
- Ranking, scores, max positions, or position sizing: read `wiki-04-ranking-and-position-sizing.md`.
- Built-in or custom indicators: read `wiki-05-writing-indicators.md`.
- ML models, prediction inputs, caching, or walkforward training: read `wiki-06-training-a-model.md`.
- Custom DataSource, CSV, DataFrame, or custom columns: read `wiki-07-creating-a-custom-data-source.md`.
- Stop loss, profit stop, trailing stop, stop limits, or stop cancellation: read `wiki-08-applying-stops.md`.
- Rebalancing, before/after exec hooks, or portfolio-wide logic: read `wiki-09-rebalancing-positions.md`.
- Rotational strategies or ranking across a universe: read `wiki-10-rotational-trading.md`.
- Edge cases and common questions: read `wiki-faqs.md`.
