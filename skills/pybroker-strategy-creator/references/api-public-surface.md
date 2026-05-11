# Public API Surface

Source: `src/pybroker`

Generated from local source signatures and first docstring sentences. Use this with the wiki pages when exact PyBroker names, parameters, constructors, and methods matter.

## `src/pybroker/cache.py`

Contains caching utilities.

### `class CacheDateFields`

Date fields for keying cache data.


### `class DataSourceCacheKey`

Cache key used for :class:`pybroker.data.DataSource` data.


### `class IndicatorCacheKey`

Cache key used for indicator data.


### `class ModelCacheKey`

Cache key used for trained models.


- `enable_data_source_cache(namespace: str, cache_dir: Optional[str]=None) -> Cache`: Enables caching of data retrieved from :class:`pybroker.data.DataSource`\ s.
- `disable_data_source_cache()`: Disables caching data retrieved from :class:`pybroker.data.DataSource`\ s.
- `clear_data_source_cache()`: Clears data cached from :class:`pybroker.data.DataSource`\ s.
- `enable_indicator_cache(namespace: str, cache_dir: Optional[str]=None) -> Cache`: Enables caching indicator data.
- `disable_indicator_cache()`: Disables caching indicator data.
- `clear_indicator_cache()`: Clears cached indicator data.
- `enable_model_cache(namespace: str, cache_dir: Optional[str]=None) -> Cache`: Enables caching trained models.
- `disable_model_cache()`: Disables caching trained models.
- `clear_model_cache()`: Clears cached trained models.
- `enable_caches(namespace, cache_dir: Optional[str]=None)`: Enables all caches.
- `disable_caches()`: Disables all caches.
- `clear_caches()`: Clears cached data from all caches.

## `src/pybroker/common.py`

Contains common classes and utilities.

### `class IndicatorSymbol`

:class:`pybroker.indicator.Indicator`/symbol identifier.


### `class ModelSymbol`

:class:`pybroker.model.ModelSource`/symbol identifier.


### `class TrainedModel`

Trained model/symbol identifier.


### `class DataCol`

Default data column names.


### `class Day`

Enumeration of days.


### `class PriceType`

Enumeration of price types used to specify fill price with :class:`pybroker.context.ExecContext`.


### `class StopType`

Stop types.


### `class FeeMode`

Brokerage fee mode to use for backtesting.


### `class FeeInfo`

Contains info for custom fee calculations.


### `class PositionMode`

Position mode for backtesting.


### `class BarData(date: NDArray[np.datetime64], open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: Optional[NDArray[np.float64]], vwap: Optional[NDArray[np.float64]], **kwargs)`

Contains data for a series of bars.


- `to_datetime(date: Union[str, datetime, np.datetime64, pd.Timestamp]) -> datetime`: Converts ``date`` to :class:`datetime`.
- `to_decimal(value: Union[int, float, Decimal]) -> Decimal`: Converts ``value`` to :class:`Decimal`.
- `parse_timeframe(timeframe: str) -> list[tuple[int, str]]`: Parses timeframe string with the following units: - ``"s"``/``"sec"``: seconds - ``"m"``/``"min"``: minutes - ``"h"``/``"hour"``: hours - ``"d"``/``"day"``: days - ``"w"``/``"week"``: weeks An example timeframe string is ``1h 30m``.
- `to_seconds(timeframe: Optional[str]) -> int`: Converts a timeframe string to seconds, where ``timeframe`` supports the following units: - ``"s"``/``"sec"``: seconds - ``"m"``/``"min"``: minutes - ``"h"``/``"hour"``: hours - ``"d"``/``"day"``: days - ``"w"``/``"week"``: weeks An example timeframe string is ``1h 30m``.
- `quantize(df: pd.DataFrame, col: str, round: bool) -> pd.Series`: Quantizes a :class:`pandas.DataFrame` column by rounding values to the nearest cent.
- `verify_data_source_columns(df: pd.DataFrame)`: Verifies that a :class:`pandas.DataFrame` contains all of the columns required by a :class:`pybroker.data.DataSource`.
- `verify_date_range(start_date: datetime, end_date: datetime)`: Verifies date range bounds.
- `default_parallel() -> Parallel`: Returns a :class:`joblib.Parallel` instance with ``n_jobs`` equal to the number of CPUs on the host machine.
- `get_unique_sorted_dates(col: pd.Series) -> Sequence[np.datetime64]`: Returns sorted unique values from a DataFrame column of dates.

## `src/pybroker/config.py`

Contains configuration options.

### `class StrategyConfig`

Configuration options for :class:`pybroker.strategy.Strategy`.



## `src/pybroker/context.py`

Contains context related classes.

### `class BaseContext(config: StrategyConfig, portfolio: Portfolio, col_scope: ColumnScope, ind_scope: IndicatorScope, input_scope: ModelInputScope, pred_scope: PredictionScope, pending_order_scope: PendingOrderScope, models: Mapping[ModelSymbol, TrainedModel], sym_end_index: Mapping[str, int])`

Base context class.

- `total_equity() -> Decimal`: Total equity currently held in the :class:`pybroker.portfolio.Portfolio`.
- `cash() -> Decimal`: Total cash currently held in the :class:`pybroker.portfolio.Portfolio`.
- `total_margin() -> Decimal`: Total amount of margin currently held in the :class:`pybroker.portfolio.Portfolio`.
- `total_market_value() -> Decimal`: Total market value currently held in the :class:`pybroker.portfolio.Portfolio`.
- `win_rate() -> Decimal`: Running win rate of trades.
- `loss_rate() -> Decimal`: Running loss rate of trades.
- `orders() -> Iterator[Order]`: :class:`Iterator` of all :class:`pybroker.portfolio.Order`\ s that have been placed and filled.
- `pending_orders(symbol: Optional[str]=None) -> Iterator[PendingOrder]`
- `trades() -> Iterator[Trade]`: :class:`Iterator` of all :class:`pybroker.portfolio.Trade`\ s that have been completed.
- `pos(symbol: str, pos_type: Literal['long', 'short']) -> Optional[Position]`: Retrieves a current long or short :class:`pybroker.portfolio.Position` for a ``symbol``.
- `positions(symbol: Optional[str]=None, pos_type: Optional[Literal['long', 'short']]=None) -> Iterator[Position]`: Retrieves all current positions.
- `long_positions(symbol: Optional[str]=None) -> Iterator[Position]`: Retrieves all current long positions.
- `short_positions(symbol: Optional[str]=None) -> Iterator[Position]`: Retrieves all current short positions.
- `calc_target_shares(target_size: float, price: float, cash: Optional[float]=None) -> Union[Decimal, int]`: Calculates the number of shares given a ``target_size`` allocation and share ``price``.
- `model(name: str, symbol: str) -> Any`: Returns a trained model.
- `indicator(name: str, symbol: str) -> NDArray[np.float64]`: Returns indicator data.
- `input(model_name: str, symbol: str) -> pd.DataFrame`: Returns model input data for making predictions.
- `preds(model_name: str, symbol: str) -> NDArray`: Returns model predictions.

### `class ExecResult`

Holds data that was set during the execution of a :class:`pybroker.strategy.Strategy`.


### `class ExecSignal`

Holds data of a buy/sell signal.


### `class PosSizeContext(config: StrategyConfig, portfolio: Portfolio, col_scope: ColumnScope, ind_scope: IndicatorScope, input_scope: ModelInputScope, pred_scope: PredictionScope, pending_order_scope: PendingOrderScope, models: Mapping[ModelSymbol, TrainedModel], sessions: Mapping[str, Mapping], sym_end_index: Mapping[str, int])`

Holds data for a position size handler set with :meth:`pybroker.Strategy.set_pos_size_handler`.

- `signals(signal_type: Optional[Literal['buy', 'sell']]=None) -> Iterator[ExecSignal]`: Returns :class:`Iterator` of :class:`.ExecSignal`\ s containing data for buy and sell signals.
- `set_shares(signal: ExecSignal, shares: Union[int, float, Decimal])`: Sets the number of shares of an order for the buy or sell signal.

- `set_pos_size_ctx_data(ctx: PosSizeContext, buy_results: Optional[list[ExecResult]], sell_results: Optional[list[ExecResult]])`: Sets data on a :class:`.PosSizeContext` instance.
### `class ExecContext(symbol: str, config: StrategyConfig, portfolio: Portfolio, col_scope: ColumnScope, ind_scope: IndicatorScope, input_scope: ModelInputScope, pred_scope: PredictionScope, pending_order_scope: PendingOrderScope, models: Mapping[ModelSymbol, TrainedModel], sym_end_index: Mapping[str, int], session: MutableMapping)`

Contains context data during the execution of a :class:`pybroker.strategy.Strategy`.

- `bars() -> int`: Number of bars of data that have completed.
- `dt() -> datetime`: Current bar's date expressed as a ``datetime``.
- `date() -> NDArray[np.datetime64]`: Current bar's date expressed as a ``numpy.datetime64``.
- `open() -> NDArray[np.float64]`: Current bar's open price.
- `high() -> NDArray[np.float64]`: Current bar's high price.
- `low() -> NDArray[np.float64]`: Current bar's low price.
- `close() -> NDArray[np.float64]`: Current bar's close price.
- `volume() -> Optional[NDArray[np.float64]]`: Current bar's volume.
- `vwap() -> Optional[NDArray[np.float64]]`: Current bar's volume-weighted average price (VWAP).
- `cover_fill_price() -> Optional[Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]]`: Alias for :attr:`.buy_fill_price`.
- `cover_fill_price(fill_price: Optional[Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]])`
- `cover_shares() -> Optional[Union[int, float, Decimal]]`: Alias for :attr:`.buy_shares`.
- `cover_shares(shares: Optional[Union[int, float, Decimal]])`
- `cover_limit_price() -> Optional[Union[int, float, Decimal]]`: Alias for :attr:`.buy_limit_price`.
- `cover_limit_price(limit_price: Optional[Union[int, float, Decimal]])`
- `sell_all_shares()`: Sells all long shares of :attr:`.ExecContext.symbol`.
- `cover_all_shares()`: Covers all short shares of :attr:`.ExecContext.symbol`.
- `foreign(symbol: str, col: Optional[str]=None) -> Union[BarData, Optional[NDArray]]`: Retrieves bar data for another ticker symbol.
- `model(name: str, symbol: Optional[str]=None) -> Any`: Returns a trained model.
- `indicator(name: str, symbol: Optional[str]=None) -> NDArray[np.float64]`: Returns indicator data.
- `input(model_name: str, symbol: Optional[str]=None) -> pd.DataFrame`: Returns model input data for making predictions.
- `preds(model_name: str, symbol: Optional[str]=None) -> NDArray`: Returns model predictions.
- `long_pos(symbol: Optional[str]=None) -> Optional[Position]`: Retrieves a current long :class:`pybroker.portfolio.Position` for a ``symbol``.
- `short_pos(symbol: Optional[str]=None) -> Optional[Position]`: Retrieves a current short :class:`pybroker.portfolio.Position` for a ``symbol``.
- `calc_target_shares(target_size: float, price: Optional[float]=None, cash: Optional[float]=None) -> Union[Decimal, int]`: Calculates the number of shares given a ``target_size`` allocation and share ``price``.
- `cancel_pending_order(order_id: int) -> bool`: Cancels a :class:`pybroker.scope.PendingOrder` with ``order_id``.
- `cancel_all_pending_orders(symbol: Optional[str]=None)`: Cancels all :class:`pybroker.scope.PendingOrder`\ s for ``symbol``.
- `cancel_stop(stop_id: int) -> bool`: Cancels a :class:`pybroker.portfolio.Stop` with ``stop_id``.
- `cancel_stops(val: Union[str, Position, Entry], stop_type: Optional[StopType]=None)`: Cancels :class:`pybroker.portfolio.Stop`\ s.
- `to_result() -> Optional[ExecResult]`: Creates an :class:`.ExecResult` from the data set on :class:`.ExecContext`.

- `set_exec_ctx_data(ctx: ExecContext, date: np.datetime64)`: Sets data on an :class:`.ExecContext` instance.

## `src/pybroker/data.py`

Contains :class:`.DataSource`\ s used to fetch external data.

### `class DataSourceCacheMixin`

Mixin that implements fetching and storing cached :class:`.DataSource` data.

- `get_cached(symbols: Iterable[str], timeframe: str, start_date: Union[str, datetime, pd.Timestamp, np.datetime64], end_date: Union[str, datetime, pd.Timestamp, np.datetime64], adjust: Optional[Any]) -> tuple[pd.DataFrame, Iterable[str]]`: Retrieves cached data from disk when caching is enabled with :meth:`pybroker.cache.enable_data_source_cache`.
- `set_cached(timeframe: str, start_date: Union[str, datetime, pd.Timestamp, np.datetime64], end_date: Union[str, datetime, pd.Timestamp, np.datetime64], adjust: Optional[Any], data: pd.DataFrame)`: Stores data to disk cache when caching is enabled with :meth:`pybroker.cache.enable_data_source_cache`.

### `class DataSource()`

Base class for querying data from an external source.

- `query(symbols: Union[str, Iterable[str]], start_date: Union[str, datetime], end_date: Union[str, datetime], timeframe: Optional[str]='', adjust: Optional[Any]=None) -> pd.DataFrame`: Queries data.

### `class Alpaca(api_key: str, api_secret: str)`

Retrieves stock data from `Alpaca <https://alpaca.markets/>`_.

- `query(symbols: Union[str, Iterable[str]], start_date: Union[str, datetime], end_date: Union[str, datetime], timeframe: Optional[str]='1d', adjust: Optional[Any]=None) -> pd.DataFrame`

### `class AlpacaCrypto(api_key: str, api_secret: str)`

Retrieves crypto data from `Alpaca <https://alpaca.markets/>`_.

- `query(symbols: Union[str, Iterable[str]], start_date: Union[str, datetime], end_date: Union[str, datetime], timeframe: Optional[str]='1d', _adjust: Optional[str]=None) -> pd.DataFrame`

### `class YFinance(auto_adjust: bool=False)`

Retrieves data from `Yahoo Finance <https://finance.yahoo.com/>`_\ .

- `query(symbols: Union[str, Iterable[str]], start_date: Union[str, datetime], end_date: Union[str, datetime], _timeframe: Optional[str]='', _adjust: Optional[Any]=None) -> pd.DataFrame`: Queries data from `Yahoo Finance <https://finance.yahoo.com/>`_\ .


## `src/pybroker/eval.py`

Contains implementation of evaluation metrics.

### `class BootConfIntervals`

Holds confidence intervals of bootstrap tests.


- `bca_boot_conf(x: NDArray[np.float64], n: int, n_boot: int, fn: Callable[[NDArray[np.float64]], float]) -> BootConfIntervals`: Computes confidence intervals for a user-defined parameter using the `bias corrected and accelerated (BCa) bootstrap method.
- `profit_factor(changes: NDArray[np.float64], use_log: bool=False) -> np.floating`: Computes the profit factor, which is the ratio of gross profit to gross loss.
- `log_profit_factor(changes: NDArray[np.float64]) -> np.floating`: Computes the log transformed profit factor, which is the ratio of gross profit to gross loss.
- `sharpe_ratio(returns: NDArray[np.float64], obs: Optional[int]=None, downside_only: bool=False) -> np.floating`: Computes the `Sharpe Ratio <https://en.wikipedia.org/wiki/Sharpe_ratio>`_.
- `sortino_ratio(returns: NDArray[np.float64], obs: Optional[int]=None) -> float`: Computes the `Sortino Ratio <https://en.wikipedia.org/wiki/Sortino_ratio>`_.
- `conf_profit_factor(x: NDArray[np.float64], n: int, n_boot: int) -> BootConfIntervals`: Computes confidence intervals for :func:`.profit_factor`.
- `conf_sharpe_ratio(x: NDArray[np.float64], n: int, n_boot: int, obs: Optional[int]=None) -> BootConfIntervals`: Computes confidence intervals for :func:`.sharpe_ratio`.
- `max_drawdown(changes: NDArray[np.float64]) -> float`: Computes maximum drawdown, measured in cash.
- `calmar_ratio(returns: NDArray[np.float64], bars_per_year: int) -> float`: Computes the Calmar Ratio.
- `max_drawdown_percent(returns: NDArray[np.float64]) -> tuple[float, Optional[int]]`: Computes maximum drawdown, measured in percentage loss.
### `class DrawdownConfs`

Contains upper bounds of confidence intervals for maximum drawdown.


### `class DrawdownMetrics`

Contains drawdown metrics.


- `drawdown_conf(changes: NDArray[np.float64], returns: NDArray[np.float64], n: int, n_boot: int) -> DrawdownMetrics`: Computes upper bounds of confidence intervals for maximum drawdown using the bootstrap method.
- `relative_entropy(values: NDArray[np.float64]) -> float`: Computes the relative `entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.
- `iqr(values: NDArray[np.float64]) -> float`: Computes the `interquartile range (IQR) <https://en.wikipedia.org/wiki/Interquartile_range>`_ of ``values``.
- `ulcer_index(values: NDArray[np.float64], period: int=14) -> float`: Computes the `Ulcer Index <https://en.wikipedia.org/wiki/Ulcer_index>`_ of ``values``.
- `upi(values: NDArray[np.float64], period: int=14, ui: Optional[float]=None) -> float`: Computes the `Ulcer Performance Index <https://en.wikipedia.org/wiki/Ulcer_index>`_ of ``values``.
- `win_loss_rate(pnls: NDArray[np.float64]) -> tuple[float, float]`: Computes the win rate and loss rate as percentages.
- `winning_losing_trades(pnls: NDArray[np.float64]) -> tuple[int, int]`: Returns the number of winning and losing trades.
- `total_profit_loss(pnls: NDArray[np.float64]) -> tuple[float, float]`: Computes total profit and loss.
- `avg_profit_loss(pnls: NDArray[np.float64]) -> tuple[float, float]`: Computes the average profit and average loss per trade.
- `largest_win_loss(pnls: NDArray[np.float64]) -> tuple[float, float]`: Computes the largest profit and largest loss of all trades.
- `max_wins_losses(pnls: NDArray[np.float64]) -> tuple[int, int]`: Computes the max consecutive wins and max consecutive losses.
- `total_return_percent(initial_value: float, pnl: float) -> float`: Computes total return as percentage.
- `annual_total_return_percent(initial_value: float, pnl: float, bars_per_year: int, total_bars: int) -> float`: Computes annualized total return as percentage.
- `r_squared(values: NDArray[np.float64]) -> float`: Computes R-squared of ``values``.
### `class BootstrapResult`

Contains results of bootstrap tests.


### `class EvalMetrics`

Contains metrics for evaluating a :class:`pybroker.strategy.Strategy`.


### `class ConfInterval`

Confidence interval upper and low bounds.


### `class EvalResult`

Contains evaluation result.


### `class EvaluateMixin`

Mixin for computing evaluation metrics.

- `evaluate(portfolio_df: pd.DataFrame, trades_df: pd.DataFrame, calc_bootstrap: bool, bootstrap_sample_size: int, bootstrap_samples: int, bars_per_year: Optional[int], seed: Optional[int]=42) -> EvalResult`: Computes evaluation metrics.


## `src/pybroker/indicator.py`

Contains indicator related functionality.

### `class Indicator(name: str, fn: Callable[..., NDArray[np.float64]], kwargs: dict[str, Any])`

Class representing an indicator.

- `relative_entropy(data: Union[BarData, pd.DataFrame]) -> float`: Generates indicator data with ``data`` and computes its relative `entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.
- `iqr(data: Union[BarData, pd.DataFrame]) -> float`: Generates indicator data with ``data`` and computes its `interquartile range (IQR) <https://en.wikipedia.org/wiki/Interquartile_range>`_.
- `__call__(data: Union[BarData, pd.DataFrame]) -> pd.Series`: Computes indicator values.

- `indicator(name: str, fn: Callable[..., NDArray[np.float64]], **kwargs) -> Indicator`: Creates an :class:`.Indicator` instance and registers it globally with ``name``.
### `class IndicatorsMixin`

Mixin implementing indicator related functionality.

- `compute_indicators(df: pd.DataFrame, indicator_syms: Iterable[IndicatorSymbol], cache_date_fields: Optional[CacheDateFields], disable_parallel: bool) -> dict[IndicatorSymbol, pd.Series]`: Computes indicator data for the provided :class:`pybroker.common.IndicatorSymbol` pairs.

### `class IndicatorSet()`

Computes data for multiple indicators.

- `add(indicators: Union[Indicator, Iterable[Indicator]], *args)`: Adds indicators.
- `remove(indicators: Union[Indicator, Iterable[Indicator]], *args)`: Removes indicators.
- `clear()`: Removes all indicators.
- `__call__(df: pd.DataFrame, disable_parallel: bool=False) -> pd.DataFrame`: Computes indicator data.

- `highest(name: str, field: str, period: int) -> Indicator`: Creates a rolling high :class:`.Indicator`.
- `lowest(name: str, field: str, period: int) -> Indicator`: Creates a rolling low :class:`.Indicator`.
- `returns(name: str, field: str, period: int=1) -> Indicator`: Creates a rolling returns :class:`.Indicator`.
- `detrended_rsi(name: str, field: str, short_length: int, long_length: int, reg_length: int) -> Indicator`: Detrended Relative Strength Index (RSI).
- `macd(name: str, short_length: int, long_length: int, smoothing: float=0.0, scale: float=1.0) -> Indicator`: Moving Average Convergence Divergence.
- `stochastic(name: str, lookback: int, smoothing: int=0) -> Indicator`: Stochastic.
- `stochastic_rsi(name: str, field: str, rsi_lookback: int, sto_lookback: int, smoothing: float=0.0) -> Indicator`: Stochastic Relative Strength Index (RSI).
- `linear_trend(name: str, field: str, lookback: int, atr_length: int, scale: float=1.0) -> Indicator`: Linear Trend Strength.
- `quadratic_trend(name: str, field: str, lookback: int, atr_length: int, scale: float=1.0) -> Indicator`: Quadratic Trend Strength.
- `cubic_trend(name: str, field: str, lookback: int, atr_length: int, scale: float=1.0) -> Indicator`: Cubic Trend Strength.
- `adx(name: str, lookback: int) -> Indicator`: Average Directional Movement Index.
- `aroon_up(name: str, lookback: int) -> Indicator`: Aroon Upward Trend.
- `aroon_down(name: str, lookback: int) -> Indicator`: Aroon Downward Trend.
- `aroon_diff(name: str, lookback: int) -> Indicator`: Aroon Upward Trend minus Aroon Downward Trend.
- `close_minus_ma(name: str, lookback: int, atr_length: int, scale: float=1.0) -> Indicator`: Close Minus Moving Average.
- `linear_deviation(name: str, field: str, lookback: int, scale: float=0.6) -> Indicator`: Deviation from Linear Trend.
- `quadratic_deviation(name: str, field: str, lookback: int, scale: float=0.6) -> Indicator`: Deviation from Quadratic Trend.
- `cubic_deviation(name: str, field: str, lookback: int, scale: float=0.6) -> Indicator`: Deviation from Cubic Trend.
- `price_intensity(name: str, smoothing: float=0.0, scale: float=0.8) -> Indicator`: Price Intensity.
- `price_change_oscillator(name: str, short_length: int, multiplier: int, scale: float=4.0) -> Indicator`: Price Change Oscillator.
- `intraday_intensity(name: str, lookback: int, smoothing: float=0.0) -> Indicator`: Intraday Intensity.
- `money_flow(name: str, lookback: int, smoothing: float=0.0) -> Indicator`: Chaikin's Money Flow.
- `reactivity(name: str, lookback: int, smoothing: float=0.0, scale: float=0.6) -> Indicator`: Reactivity.
- `price_volume_fit(name: str, lookback: int, scale: float=9.0) -> Indicator`: Price Volume Fit.
- `volume_weighted_ma_ratio(name: str, lookback: int, scale: float=1.0) -> Indicator`: Volume-Weighted Moving Average Ratio.
- `normalized_on_balance_volume(name: str, lookback: int, scale: float=0.6) -> Indicator`: Normalized On-Balance Volume.
- `delta_on_balance_volume(name: str, lookback: int, delta_length: int=0, scale: float=0.6) -> Indicator`: Delta On-Balance Volume.
- `normalized_positive_volume_index(name: str, lookback: int, scale: float=0.5) -> Indicator`: Normalized Positive Volume Index.
- `normalized_negative_volume_index(name: str, lookback: int, scale: float=0.5) -> Indicator`: Normalized Negative Volume Index.
- `volume_momentum(name: str, short_length: int, multiplier: int=2, scale: float=3.0) -> Indicator`: Volume Momentum.
- `laguerre_rsi(name: str, fe_length: int=13) -> Indicator`: Laguerre Relative Strength Index (RSI).

## `src/pybroker/log.py`

Logging module.

### `class Logger(scope)`

Class for logging information about triggered events.

- `disable()`: Disables logging.
- `enable()`: Enables logging.
- `disable_progress_bar()`: Disables logging a progress bar.
- `enable_progress_bar()`: Enables logging a progress bar.
- `download_bar_data_start()`
- `info_download_bar_data_start(symbols: Iterable[str], start_date: datetime.datetime, end_date: datetime.datetime, timeframe: str)`
- `loaded_bar_data()`
- `info_loaded_bar_data(symbols: Iterable[str], start_date: datetime.datetime, end_date: datetime.datetime, timeframe: str)`
- `info_invalidate_data_source_cache()`
- `debug_get_data_source_cache(cache_key)`
- `debug_set_data_source_cache(cache_key)`
- `download_bar_data_completed()`
- `indicator_data_start(ind_syms: Sized)`
- `info_indicator_data_start(ind_syms: Iterable[IndicatorSymbol])`
- `loaded_indicator_data()`
- `info_loaded_indicator_data(ind_syms: Iterable[IndicatorSymbol])`
- `indicator_data_loading(count: int)`
- `debug_get_indicator_cache(cache_key)`
- `debug_set_indicator_cache(cache_key)`
- `debug_compute_indicators(is_parallel: bool)`
- `train_split_start(train_dates: Sequence[np.datetime64])`
- `info_train_split_start(model_syms: Iterable[ModelSymbol])`
- `loaded_models()`
- `info_loaded_models(model_syms: Iterable[ModelSymbol])`
- `info_train_model_start(model_sym: ModelSymbol)`
- `info_train_model_completed(model_sym: ModelSymbol)`
- `info_loaded_model(model_sym: ModelSymbol)`
- `debug_get_model_cache(cache_key)`
- `debug_set_model_cache(cache_key)`
- `train_split_completed()`
- `backtest_executions_start(test_dates: Sequence[np.datetime64])`
- `backtest_executions_loading(count: int)`
- `walkforward_start(start_date: datetime.datetime, end_date: datetime.datetime)`
- `info_walkforward_between_time(between_time: tuple[str, str])`
- `info_walkforward_on_days(days: tuple[int])`
- `walkforward_completed()`
- `calc_bootstrap_metrics_start(samples, sample_size)`
- `calc_bootstrap_metrics_completed()`
- `debug_place_buy_order(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal])`
- `debug_buy_shares_exceed_cash(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal], cash: Decimal, clamped_shares: Decimal)`
- `debug_filled_buy_order(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal])`
- `debug_unfilled_buy_order(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal])`
- `debug_place_sell_order(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal])`
- `debug_filled_sell_order(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal])`
- `debug_unfilled_sell_order(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal])`
- `debug_schedule_order(date: np.datetime64, exec_result)`
- `debug_unscheduled_order(exec_result)`
- `warn_bootstrap_sample_size(n: int, sample_size: int)`
- `debug_enable_data_source_cache(ns: str, cache_dir: str)`
- `debug_disable_data_source_cache()`
- `debug_clear_data_source_cache(cache_dir: str)`
- `debug_enable_indicator_cache(ns: str, cache_dir: str)`
- `debug_disable_indicator_cache()`
- `debug_clear_indicator_cache(cache_dir: str)`
- `debug_enable_model_cache(ns: str, cache_dir: str)`
- `debug_disable_model_cache()`
- `debug_clear_model_cache(cache_dir: str)`


## `src/pybroker/model.py`

Contains model related functionality.

### `class ModelSource(name: str, indicator_names: Iterable[str], input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]], predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]], kwargs: dict[str, Any])`

Base class of a model source.

- `prepare_input_data(df: pd.DataFrame) -> pd.DataFrame`: Prepares a :class:`pandas.DataFrame` of input data for passing to a model when making predictions.

### `class ModelLoader(name: str, load_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]], indicator_names: Iterable[str], input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]], predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]], kwargs: dict[str, Any])`

Loads a pre-trained model.

- `__call__(symbol: str, train_start_date: datetime, train_end_date: datetime) -> Union[Any, tuple[Any, Iterable[str]]]`: Loads pre-trained model.

### `class ModelTrainer(name: str, train_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]], indicator_names: Iterable[str], input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]], predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]], kwargs: dict[str, Any])`

Trains a model.

- `__call__(symbol: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Union[Any, tuple[Any, Iterable[str]]]`: Trains model.

- `model(name: str, fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]], indicators: Optional[Iterable[Indicator]]=None, input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]]=None, predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]]=None, pretrained: bool=False, **kwargs) -> ModelSource`: Creates a :class:`.ModelSource` instance and registers it globally with ``name``.
### `class CachedModel`

Stores cached model data.


### `class ModelsMixin`

Mixin implementing model related functionality.

- `train_models(model_syms: Iterable[ModelSymbol], train_data: pd.DataFrame, test_data: pd.DataFrame, indicator_data: Mapping[IndicatorSymbol, pd.Series], cache_date_fields: CacheDateFields) -> dict[ModelSymbol, TrainedModel]`: Trains models for the provided :class:`pybroker.common.ModelSymbol` pairs.


## `src/pybroker/portfolio.py`

Contains portfolio related functionality, such as portfolio metrics and placing orders.

### `class Stop`

Contains information about a stop set on :class:`.Entry`.


### `class StopRecord`

Records per-bar data about a stop.


### `class Entry`

Contains information about an entry into a :class:`.Position`.


### `class Position`

Contains information about an open position in ``symbol``.


### `class Trade`

Holds information about a completed trade (entry and exit).


### `class Order`

Holds information about a filled order.


### `class PortfolioBar`

Snapshot of :class:`.Portfolio` state, captured per bar.


### `class PositionBar`

Snapshot of an open :class:`.Position`\ 's state, captured per bar.


### `class Portfolio(cash: float, fee_mode: Optional[Union[FeeMode, Callable[[FeeInfo], Decimal], None]]=None, fee_amount: Optional[float]=None, enable_fractional_shares: bool=False, position_mode: PositionMode=PositionMode.DEFAULT, max_long_positions: Optional[int]=None, max_short_positions: Optional[int]=None, record_stops: Optional[bool]=False)`

Class representing a portfolio of holdings.

- `buy(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal]=None, stops: Optional[Iterable[Stop]]=None) -> Optional[Order]`: Places a buy order.
- `sell(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal]=None, stops: Optional[Iterable[Stop]]=None) -> Optional[Order]`: Places a sell order.
- `exit_position(date: np.datetime64, symbol: str, buy_fill_price: Decimal, sell_fill_price: Decimal)`: Exits any long and short positions for ``symbol`` at ``buy_fill_price`` and ``sell_fill_price``.
- `capture_bar(date: np.datetime64, df: pd.DataFrame)`: Captures portfolio state of the current bar.
- `incr_bars()`: Increments the number of bars held by every trade entry.
- `remove_stop(stop_id: int) -> bool`: Removes a :class:`.Stop` with ``stop_id``.
- `remove_stops(val: Union[str, Position, Entry], stop_type: Optional[StopType]=None)`: Removes :class:`.Stop`\ s.
- `check_stops(date: np.datetime64, price_scope: PriceScope)`: Checks whether stops are triggered.


## `src/pybroker/scope.py`

Contains scopes that store data and object references used to execute a :class:`pybroker.strategy.Strategy`.

### `class StaticScope()`

A static registry of data and object references.

- `set_indicator(indicator)`: Stores :class:`pybroker.indicator.Indicator` in static scope.
- `has_indicator(name: str) -> bool`: Whether :class:`pybroker.indicator.Indicator` is stored in static scope.
- `get_indicator(name: str)`: Retrieves a :class:`pybroker.indicator.Indicator` from static scope.
- `get_indicator_names(model_name: str) -> tuple[str]`: Returns a ``tuple[str]`` of all :class:`pybroker.indicator.Indicator` names that are registered with :class:`pybroker.model.ModelSource` having ``model_name``.
- `set_model_source(source)`: Stores :class:`pybroker.model.ModelSource` in static scope.
- `has_model_source(name: str) -> bool`: Whether :class:`pybroker.model.ModelSource` is stored in static scope.
- `get_model_source(name: str)`: Retrieves a :class:`pybroker.model.ModelSource` from static scope.
- `register_custom_cols(names: Union[str, Iterable[str]], *args)`: Registers user-defined column names.
- `unregister_custom_cols(names: Union[str, Iterable[str]], *args)`: Unregisters user-defined column names.
- `all_data_cols() -> frozenset[str]`: All registered data column names.
- `freeze_data_cols()`: Prevents additional data columns from being registered.
- `unfreeze_data_cols()`: Allows additional data columns to be registered if :func:`pybroker.scope.StaticScope.freeze_data_cols` was called.
- `param(name: str, value: Optional[Any]=_EMPTY_PARAM) -> Optional[Any]`: Get or set a global parameter.
- `clear_params()`: Clears all global parameters.
- `instance() -> 'StaticScope'`: Returns singleton instance.

- `disable_logging()`: Disables event logging.
- `enable_logging()`: Enables event logging.
- `disable_progress_bar()`: Disables logging a progress bar.
- `enable_progress_bar()`: Enables logging a progress bar.
- `register_columns(names: Union[str, Iterable[str]], *args)`: Registers ``names`` of user-defined data columns.
- `unregister_columns(names: Union[str, Iterable[str]], *args)`: Unregisters ``names`` of user-defined data columns.
- `param(name: str, value: Optional[Any]=_EMPTY_PARAM) -> Optional[Any]`: Get or set a global parameter.
- `clear_params()`: Clears all global parameters.
### `class ColumnScope(df: pd.DataFrame)`

Caches and retrieves column data queried from :class:`pandas.DataFrame`.

- `fetch_dict(symbol: str, names: Iterable[str], end_index: Optional[int]=None) -> dict[str, Optional[NDArray]]`: Fetches a ``dict`` of column data for ``symbol``.
- `fetch(symbol: str, name: str, end_index: Optional[int]=None) -> Optional[NDArray]`: Fetches a :class:`numpy.ndarray` of column data for ``symbol``.
- `bar_data_from_data_columns(symbol: str, end_index: int) -> BarData`: Returns a new :class:`pybroker.common.BarData` instance containing column data of default and custom data columns registered with :class:`.StaticScope`.

### `class IndicatorScope(indicator_data: Mapping[IndicatorSymbol, pd.Series], filter_dates: Sequence[np.datetime64])`

Caches and retrieves :class:`pybroker.indicator.Indicator` data.

- `fetch(symbol: str, name: str, end_index: Optional[int]=None) -> NDArray[np.float64]`: Fetches :class:`pybroker.indicator.Indicator` data.

### `class ModelInputScope(col_scope: ColumnScope, ind_scope: IndicatorScope, models: Mapping[ModelSymbol, TrainedModel])`

Caches and retrieves model input data.

- `fetch(symbol: str, name: str, end_index: Optional[int]=None) -> pd.DataFrame`: Fetches model input data.

### `class PredictionScope(models: Mapping[ModelSymbol, TrainedModel], input_scope: ModelInputScope)`

Caches and retrieves model predictions.

- `fetch(symbol: str, name: str, end_index: Optional[int]=None) -> NDArray`: Fetches model predictions.

### `class PriceScope(col_scope: ColumnScope, sym_end_index: Mapping[str, int], round_fill_price: bool)`

Retrieves most recent prices.

- `fetch(symbol: str, price: Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]) -> Decimal`

### `class PendingOrder`

Holds data for a pending order.


### `class PendingOrderScope()`

Stores :class:`.PendingOrder`\ s

- `contains(order_id: int) -> bool`: Returns whether a :class:`.PendingOrder` exists with ``order_id``.
- `add(type: Literal['buy', 'sell'], symbol: str, created: np.datetime64, exec_date: np.datetime64, shares: Decimal, limit_price: Optional[Decimal], fill_price: Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]) -> int`: Creates a :class:`.PendingOrder`.
- `remove(order_id: int) -> bool`: Removes a :class:`.PendingOrder` with ``order_id```.
- `remove_all(symbol: Optional[str]=None)`: Removes all :class:`.PendingOrder`\ s.
- `orders(symbol: Optional[str]=None) -> Iterable[PendingOrder]`: Returns an :class:`Iterable` of :class:`.PendingOrder`\ s.

- `get_signals(symbols: Iterable[str], col_scope: ColumnScope, ind_scope: IndicatorScope, pred_scope: PredictionScope) -> dict[str, pd.DataFrame]`: Retrieves dictionary of :class:`pandas.DataFrame`\ s containing bar data, indicator data, and model predictions for each symbol.

## `src/pybroker/slippage.py`

Implements slippage models.

### `class SlippageModel`

Base class for implementing a slippage model.

- `apply_slippage(ctx: ExecContext, buy_shares: Optional[Decimal]=None, sell_shares: Optional[Decimal]=None)`: Applies slippage to ``ctx``.

### `class RandomSlippageModel(min_pct: float, max_pct: float)`

Implements a simple random slippage model.

- `apply_slippage(ctx: ExecContext, buy_shares: Optional[Decimal]=None, sell_shares: Optional[Decimal]=None)`


## `src/pybroker/strategy.py`

Contains implementation for backtesting trading strategies.

### `class Execution`

Represents an execution of a :class:`.Strategy`.


### `class BacktestMixin`

Mixin implementing backtesting functionality.

- `backtest_executions(config: StrategyConfig, executions: set[Execution], before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]], after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]], sessions: Mapping[str, MutableMapping], models: Mapping[ModelSymbol, TrainedModel], indicator_data: Mapping[IndicatorSymbol, pd.Series], test_data: pd.DataFrame, portfolio: Portfolio, pos_size_handler: Optional[Callable[[PosSizeContext], None]], exit_dates: Mapping[str, np.datetime64], train_only: bool=False, slippage_model: Optional[SlippageModel]=None, enable_fractional_shares: bool=False, round_fill_price: bool=True, warmup: Optional[int]=None) -> dict[str, pd.DataFrame]`: Backtests a ``set`` of :class:`.Execution`\ s that implement trading logic.

### `class WalkforwardWindow`

Contains ``train_data`` and ``test_data`` of a time window used for `Walkforward Analysis <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.


### `class WalkforwardMixin`

Mixin implementing logic for `Walkforward Analysis <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

- `walkforward_split(df: pd.DataFrame, windows: int, lookahead: int, train_size: float=0.9, shuffle: bool=False) -> Iterator[WalkforwardWindow]`: Splits a :class:`pandas.DataFrame` containing data for multiple ticker symbols into an :class:`Iterator` of train/test time windows for `Walkforward Analysis <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

### `class TestResult`

Contains the results of backtesting a :class:`.Strategy`.


### `class Strategy(data_source: Union[DataSource, pd.DataFrame], start_date: Union[str, datetime], end_date: Union[str, datetime], config: Optional[StrategyConfig]=None)`

Class representing a trading strategy to backtest.

- `set_slippage_model(slippage_model: Optional[SlippageModel])`: Sets :class:`pybroker.slippage.SlippageModel`.
- `add_execution(fn: Optional[Callable[Concatenate[ExecContext, P], None]], symbols: Union[str, Iterable[str]], models: Optional[Union[ModelSource, Iterable[ModelSource]]]=None, indicators: Optional[Union[Indicator, Iterable[Indicator]]]=None, *args: P.args, **kwargs: P.kwargs)`: Adds an execution to backtest.
- `set_before_exec(fn: Optional[Callable[[Mapping[str, ExecContext]], None]])`: :class:`Callable[[Mapping[str, ExecContext]]` that runs before all execution functions.
- `set_after_exec(fn: Optional[Callable[[Mapping[str, ExecContext]], None]])`: :class:`Callable[[Mapping[str, ExecContext]]` that runs after all execution functions.
- `clear_executions()`: Clears executions that were added with :meth:`.add_execution`.
- `set_pos_size_handler(fn: Optional[Callable[[PosSizeContext], None]])`: Sets a :class:`Callable` that determines position sizes to use for buy and sell signals.
- `backtest(start_date: Optional[Union[str, datetime]]=None, end_date: Optional[Union[str, datetime]]=None, timeframe: str='', between_time: Optional[tuple[str, str]]=None, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]=None, lookahead: int=1, train_size: float=0, shuffle: bool=False, calc_bootstrap: bool=False, disable_parallel: bool=False, warmup: Optional[int]=None, portfolio: Optional[Portfolio]=None, adjust: Optional[Any]=None, seed: Optional[int]=42) -> TestResult`: Backtests the trading strategy by running executions that were added with :meth:`.add_execution`.
- `walkforward(windows: int, lookahead: int=1, start_date: Optional[Union[str, datetime]]=None, end_date: Optional[Union[str, datetime]]=None, timeframe: str='', between_time: Optional[tuple[str, str]]=None, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]=None, train_size: float=0.5, shuffle: bool=False, calc_bootstrap: bool=False, disable_parallel: bool=False, warmup: Optional[int]=None, portfolio: Optional[Portfolio]=None, adjust: Optional[Any]=None, seed: Optional[int]=42) -> TestResult`: Backtests the trading strategy using `Walkforward Analysis <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.


## `src/pybroker/vect.py`

Contains vectorized utility functions.

- `lowv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]`: Calculates the lowest values for every ``n`` period in ``array``.
- `highv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]`: Calculates the highest values for every ``n`` period in ``array``.
- `sumv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]`: Calculates the sums for every ``n`` period in ``array``.
- `returnv(array: NDArray[np.float64], n: int=1) -> NDArray[np.float64]`: Calculates returns.
- `cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.bool_]`: Checks for crossover of ``a`` above ``b``.
- `normal_cdf(z: float) -> float`: Computes the CDF of the standard normal distribution.
- `inverse_normal_cdf(p: float) -> float`: Computes the inverse CDF of the standard normal distribution.
- `detrended_rsi(values: NDArray[np.float64], short_length: int, long_length: int, reg_length: int) -> NDArray[np.float64]`: Computes Detrended Relative Strength Index (RSI).
- `macd(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], short_length: int, long_length: int, smoothing: float=0.0, scale: float=1.0) -> NDArray[np.float64]`: Computes Moving Average Convergence Divergence.
- `stochastic(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], lookback: int, smoothing: int=0) -> NDArray[np.float64]`: Computes Stochastic.
- `stochastic_rsi(values: NDArray[np.float64], rsi_lookback: int, sto_lookback: int, smoothing: float=0.0) -> NDArray[np.float64]`: Computes Stochastic Relative Strength Index (RSI).
- `linear_trend(values: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], lookback: int, atr_length: int, scale: float=1.0) -> NDArray[np.float64]`: Computes Linear Trend Strength.
- `quadratic_trend(values: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], lookback: int, atr_length: int, scale: float=1.0) -> NDArray[np.float64]`: Computes Quadratic Trend Strength.
- `cubic_trend(values: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], lookback: int, atr_length: int, scale: float=1.0) -> NDArray[np.float64]`: Computes Cubic Trend Strength.
- `adx(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], lookback: int) -> NDArray[np.float64]`: Computes Average Directional Movement Index.
- `aroon_up(high: NDArray[np.float64], low: NDArray[np.float64], lookback: int) -> NDArray[np.float64]`: Computes Aroon Upward Trend.
- `aroon_down(high: NDArray[np.float64], low: NDArray[np.float64], lookback: int) -> NDArray[np.float64]`: Computes Aroon Downward Trend.
- `aroon_diff(high: NDArray[np.float64], low: NDArray[np.float64], lookback: int) -> NDArray[np.float64]`: Computes Aroon Upward Trend minus Aroon Downward Trend.
- `close_minus_ma(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], lookback: int, atr_length: int, scale: float=1.0) -> NDArray[np.float64]`: Computes Close Minus Moving Average.
- `linear_deviation(values: NDArray[np.float64], lookback: int, scale: float=0.6) -> NDArray[np.float64]`: Computes Deviation from Linear Trend.
- `quadratic_deviation(values: NDArray[np.float64], lookback: int, scale: float=0.6) -> NDArray[np.float64]`: Computes Deviation from Quadratic Trend.
- `cubic_deviation(values: NDArray[np.float64], lookback: int, scale: float=0.6) -> NDArray[np.float64]`: Computes Deviation from Cubic Trend.
- `price_intensity(open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], smoothing: float=0.0, scale: float=0.8) -> NDArray[np.float64]`: Computes Price Intensity.
- `price_change_oscillator(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], short_length: int, multiplier: int, scale: float=4.0) -> NDArray[np.float64]`: Computes Price Change Oscillator.
- `intraday_intensity(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, smoothing: float=0.0) -> NDArray[np.float64]`: Computes Intraday Intensity.
- `money_flow(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, smoothing: float=0.0) -> NDArray[np.float64]`: Computes Chaikin's Money Flow.
- `reactivity(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, smoothing: float=0.0, scale: float=0.6) -> NDArray[np.float64]`: Computes Reactivity.
- `price_volume_fit(close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, scale: float=9.0) -> NDArray[np.float64]`: Computes Price Volume Fit.
- `volume_weighted_ma_ratio(close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, scale: float=1.0) -> NDArray[np.float64]`: Computes Volume-Weighted Moving Average Ratio.
- `normalized_on_balance_volume(close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, scale: float=0.6) -> NDArray[np.float64]`: Computes Normalized On-Balance Volume.
- `delta_on_balance_volume(close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, delta_length: int=0, scale: float=0.6) -> NDArray[np.float64]`: Computes Delta On-Balance Volume.
- `normalized_positive_volume_index(close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, scale: float=0.5) -> NDArray[np.float64]`: Computes Normalized Positive Volume Index.
- `normalized_negative_volume_index(close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, scale: float=0.5) -> NDArray[np.float64]`: Computes Normalized Negative Volume Index.
- `volume_momentum(volume: NDArray[np.float64], short_length: int, multiplier: int=2, scale: float=3.0) -> NDArray[np.float64]`: Computes Volume Momentum.
- `laguerre_rsi(open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], fe_length: int=13) -> NDArray[np.float64]`: Computes Laguerre Relative Strength Index (RSI).

## `src/pybroker/ext/data.py`

Contains extension classes.

### `class AKShare`

Retrieves data from `AKShare <https://akshare.akfamily.xyz/>`_.


### `class YQuery(proxies: Optional[dict]=None)`

Retrieves data from Yahoo Finance using `Yahooquery <https://github.com/dpguthrie/yahooquery>`_\ .
