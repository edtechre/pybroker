# Public API Surface

Source: `src/pybroker`

Generated from local source signatures and first docstring sentences. Use this with the wiki pages when exact PyBroker names, parameters, constructors, and methods matter.

## `src/pybroker/cache.py`

Contains caching utilities.

### `class CacheDateFields`

Date fields for keying cache data.

- `start_date: datetime`
- `end_date: datetime`
- `tf_seconds: int`
- `between_time: Optional[tuple[str, str]]`
- `days: Optional[tuple[int]]`

### `class DataSourceCacheKey`

Cache key used for :class:`pybroker.data.DataSource` data.

- `symbol: str`
- `tf_seconds: int`
- `start_date: datetime`
- `end_date: datetime`
- `adjust: Optional[str]`
- `from_date_fields(*, symbol: str, adjust: Optional[str], fields: CacheDateFields) -> DataSourceCacheKey`

### `class IndicatorCacheKey`

Cache key used for indicator data.

- `symbol: str`
- `tf_seconds: int`
- `start_date: datetime`
- `end_date: datetime`
- `between_time: Optional[tuple[str, str]]`
- `days: Optional[tuple[int]]`
- `ind_name: str`
- `from_date_fields(*, symbol: str, ind_name: str, fields: CacheDateFields) -> IndicatorCacheKey`

### `class ModelCacheKey`

Cache key used for trained models.

- `symbol: str`
- `tf_seconds: int`
- `start_date: datetime`
- `end_date: datetime`
- `between_time: Optional[tuple[str, str]]`
- `days: Optional[tuple[int]]`
- `model_name: str`
- `pooled_symbols: Optional[tuple[str, ...]] = None`
- `lookahead: Optional[int] = None`
- `from_date_fields(*, symbol: str, model_name: str, fields: CacheDateFields, pooled_symbols: Optional[Iterable[str]]=None, lookahead: Optional[int]=None) -> ModelCacheKey`

- `enable_data_source_cache(namespace: str, cache_dir: Optional[str]=None, l1_maxsize: int=_L1_DEFAULT_MAXSIZE) -> Cache`: Enables caching of data retrieved from :class:`pybroker.data.DataSource`\ s.
- `disable_data_source_cache()`: Disables caching data retrieved from :class:`pybroker.data.DataSource`\ s.
- `clear_data_source_cache()`: Clears data cached from :class:`pybroker.data.DataSource`\ s.
- `enable_indicator_cache(namespace: str, cache_dir: Optional[str]=None, l1_maxsize: int=_L1_DEFAULT_MAXSIZE) -> Cache`: Enables caching indicator data.
- `disable_indicator_cache()`: Disables caching indicator data.
- `clear_indicator_cache()`: Clears cached indicator data.
- `enable_model_cache(namespace: str, cache_dir: Optional[str]=None, l1_maxsize: int=_L1_DEFAULT_MAXSIZE) -> Cache`: Enables caching trained models.
- `disable_model_cache()`: Disables caching trained models.
- `clear_model_cache()`: Clears cached trained models.
- `enable_caches(namespace: str, cache_dir: Optional[str]=None, l1_maxsize: int=_L1_DEFAULT_MAXSIZE)`: Enables all caches.
- `disable_caches()`: Disables all caches.
- `clear_caches()`: Clears cached data from all caches.

## `src/pybroker/common.py`

Contains common classes and utilities.

### `class IndicatorSymbol`

:class:`pybroker.indicator.Indicator`/symbol identifier.

- `ind_name: str`
- `symbol: str`

### `class ModelSymbol`

:class:`pybroker.model.ModelSource`/symbol identifier.

- `model_name: str`
- `symbol: str`

### `class TrainedModel`

Trained model/symbol identifier.

- `name: str`
- `instance: Any`
- `predict_fn: Optional[Callable[[Any, Union[pd.DataFrame, NDArray]], NDArray]]`
- `input_cols: Optional[tuple[str]]`
- `per_bar: bool = False`
- `lag_columns: Optional[tuple[str, ...]] = None`

### `class DataCol`

Default data column names.


### `class Day`

Enumeration of days.


### `class PriceType`

Enumeration of price types used to specify fill price with :class:`pybroker.context.ExecContext`.


### `class StopType`

Stop types.


### `class OrderType`

Order type classifications.


### `class PositionIntent`

Position intent of an order.


### `class FeeMode`

Brokerage fee mode to use for backtesting.


### `class FeeInfo`

Contains info for custom fee calculations.

- `symbol: str`
- `shares: Decimal`
- `fill_price: Decimal`
- `order_type: Literal['buy', 'sell']`

### `class PositionMode`

Position mode for backtesting.


### `class BarData(date: NDArray[np.datetime64], open: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: Optional[NDArray[np.float64]], vwap: Optional[NDArray[np.float64]], **kwargs)`

Contains data for a series of bars.

- `date: NDArray[np.datetime64]`
- `open: NDArray[np.float64]`
- `high: NDArray[np.float64]`
- `low: NDArray[np.float64]`
- `close: NDArray[np.float64]`
- `volume: Optional[NDArray[np.float64]]`
- `vwap: Optional[NDArray[np.float64]]`

- `bars_to_df(bar_data: BarData) -> pd.DataFrame`: Converts a :class:`.BarData` instance to a :class:`pandas.DataFrame`.
- `to_datetime(date: Union[str, datetime, np.datetime64, pd.Timestamp]) -> datetime`: Converts ``date`` to :class:`datetime.datetime`.
- `to_decimal(value: Union[int, float, Decimal]) -> Decimal`: Converts ``value`` to :class:`decimal.Decimal`.
- `parse_timeframe(timeframe: str) -> list[tuple[int, str]]`: Parses timeframe string with the following units: - ``"s"``/``"sec"``: seconds - ``"m"``/``"min"``: minutes - ``"h"``/``"hour"``: hours - ``"d"``/``"day"``: days - ``"w"``/``"week"``: weeks An example timeframe string is ``1h 30m``.
- `to_seconds(timeframe: Optional[str]) -> int`: Converts a timeframe string to seconds, where ``timeframe`` supports the following units: - ``"s"``/``"sec"``: seconds - ``"m"``/``"min"``: minutes - ``"h"``/``"hour"``: hours - ``"d"``/``"day"``: days - ``"w"``/``"week"``: weeks An example timeframe string is ``1h 30m``.
- `quantize(df: pd.DataFrame, col: str, round: bool) -> pd.Series`: Quantizes a :class:`pandas.DataFrame` column by rounding values to the nearest cent.
- `verify_data_source_columns(df: pd.DataFrame)`: Verifies that a :class:`pandas.DataFrame` contains all of the columns required by a :class:`pybroker.data.DataSource`.
- `verify_date_range(start_date: datetime, end_date: datetime)`: Verifies date range bounds.
- `get_unique_sorted_dates_array(dates: Union[pd.Series, NDArray[np.datetime64], Sequence[np.datetime64]]) -> NDArray[np.datetime64]`: Returns sorted unique dates from a numpy date array or Series.
- `get_unique_sorted_dates(col: pd.Series) -> Sequence[np.datetime64]`: Returns sorted unique values from a DataFrame column of dates.

## `src/pybroker/config.py`

Contains configuration options.

### `class StrategyConfig`

Configuration options for :class:`pybroker.strategy.Strategy`.

- `initial_cash: float = 100000`
- `fee_mode: Optional[Union[FeeMode, Callable[[FeeInfo], Decimal]]] = None`
- `fee_amount: float = 0`
- `enable_fractional_shares: bool = False`
- `round_fill_price: bool = True`
- `position_mode: PositionMode = PositionMode.DEFAULT`
- `max_long_positions: Optional[int] = None`
- `max_short_positions: Optional[int] = None`
- `buy_delay: int = 1`
- `sell_delay: int = 1`
- `bootstrap_samples: int = 10000`
- `exit_on_last_bar: bool = False`
- `exit_cover_fill_price: Union[PriceType, Callable[[str, BarData], Union[int, float, Decimal]]] = PriceType.MIDDLE`
- `exit_sell_fill_price: Union[PriceType, Callable[[str, BarData], Union[int, float, Decimal]]] = PriceType.MIDDLE`
- `bars_per_year: Optional[int] = None`
- `return_signals: bool = False`
- `return_stops: bool = False`
- `round_test_result: bool = True`
- `leverage: float = 1.0`
- `interest_rate: float = 0.0`
- `record_portfolio_bars: bool = False`
- `record_position_bars: bool = False`

## `src/pybroker/context.py`

Contains context related classes.

### `class ExecResult`

Holds data that was set during the execution of a :class:`pybroker.strategy.Strategy`.

- `symbol: str`
- `date: np.datetime64`
- `buy_fill_price: Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]`
- `sell_fill_price: Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]`
- `score: Optional[float]`
- `long_score: Optional[float]`
- `short_score: Optional[float]`
- `hold_bars: Optional[int]`
- `buy_shares: Optional[Decimal]`
- `buy_limit_price: Optional[Decimal]`
- `buy_timeout_bars: Optional[int]`
- `sell_shares: Optional[Decimal]`
- `sell_limit_price: Optional[Decimal]`
- `sell_timeout_bars: Optional[int]`
- `long_stops: Optional[frozenset[Stop]]`
- `short_stops: Optional[frozenset[Stop]]`
- `cover: bool = False`
- `pending_order_id: Optional[int] = None`
- `exit_pos_type: Optional[Literal['long', 'short']] = None`

### `class IntervalContext(symbol: str, interval: TimeframeInterval, interval_scope: IntervalScope, sym_end_index: Mapping[str, int], models: Mapping[ModelSymbol, TrainedModel])`

Read-only view of compressed bar data for a coarser interval.

- `bars() -> int`
- `dates() -> NDArray[np.datetime64]`
- `open() -> NDArray[np.float64]`
- `high() -> NDArray[np.float64]`
- `low() -> NDArray[np.float64]`
- `close() -> NDArray[np.float64]`
- `volume() -> NDArray[np.float64]`
- `indicator(name: str) -> NDArray[np.float64]`: Returns indicator values on the compressed interval.
- `model(name: str) -> Any`: Returns a trained model on the compressed interval.
- `input(model_name: str) -> pd.DataFrame`: Returns model input data on the compressed interval.
- `preds(model_name: str) -> NDArray`: Returns model predictions on the compressed interval.

### `class ExecContext(symbol: str, config: StrategyConfig, portfolio: Portfolio, col_scope: ColumnScope, ind_scope: IndicatorScope, interval_scope: IntervalScope, declared_intervals: frozenset[TimeframeInterval], input_scope: ModelInputScope, pred_scope: PredictionScope, pending_order_scope: PendingOrderScope, models: Mapping[ModelSymbol, TrainedModel], sym_end_index: Mapping[str, int], session: MutableMapping, run_hyperparams: Optional[Mapping[str, Any]]=None, allowed_hyperparam_names: frozenset[str]=frozenset(), rotation_enabled: bool=False)`

Contains context data during the execution of a :class:`pybroker.strategy.Strategy`.

- `config: StrategyConfig`
- `rotation_enabled: bool`
- `symbol: str`
- `buy_fill_price: Optional[Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]]`
- `buy_shares: Optional[Union[int, float, Decimal]]`
- `buy_limit_price: Optional[Union[int, float, Decimal]]`
- `buy_timeout_bars: Optional[int]`
- `sell_fill_price: Optional[Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]]`
- `sell_shares: Optional[Union[int, float, Decimal]]`
- `sell_limit_price: Optional[Union[int, float, Decimal]]`
- `sell_timeout_bars: Optional[int]`
- `hold_bars: Optional[int]`
- `long_score: Optional[float]`
- `short_score: Optional[float]`
- `session: MutableMapping`
- `stop_loss: Optional[Union[int, float, Decimal]]`
- `stop_loss_pct: Optional[Union[int, float, Decimal]]`
- `stop_loss_limit: Optional[Union[int, float, Decimal]]`
- `stop_loss_exit_price: Optional[PriceType]`
- `stop_profit: Optional[Union[int, float, Decimal]]`
- `stop_profit_pct: Optional[Union[int, float, Decimal]]`
- `stop_profit_limit: Optional[Union[int, float, Decimal]]`
- `stop_profit_exit_price: Optional[PriceType]`
- `stop_trailing: Optional[Union[int, float, Decimal]]`
- `stop_trailing_pct: Optional[Union[int, float, Decimal]]`
- `stop_trailing_limit: Optional[Union[int, float, Decimal]]`
- `stop_trailing_exit_price: Optional[PriceType]`
- `score() -> Optional[float]`: Deprecated ranking field; prefer :attr:`long_score` / :attr:`short_score`.
- `score(value: Optional[float])`
- `total_equity() -> Decimal`: Total equity currently held in the :class:`pybroker.portfolio.Portfolio`.
- `cash() -> Decimal`: Total cash currently held in the :class:`pybroker.portfolio.Portfolio`.
- `total_margin() -> Decimal`: Total amount of margin currently held in the :class:`pybroker.portfolio.Portfolio`.
- `buying_power() -> Decimal`: Available buying power for long and short orders given :attr:`pybroker.config.StrategyConfig.leverage`.
- `margin_loan() -> Decimal`: Borrowed funds used for leveraged long and short positions.
- `net_cash_balance() -> Decimal`: Net cash balance (``cash - margin_loan``).
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
- `has_long_positions() -> bool`: Returns whether any long positions are currently open.
- `has_short_positions() -> bool`: Returns whether any short positions are currently open.
- `bars() -> int`: Number of bars of data that have completed.
- `dt() -> datetime`: Current bar's date expressed as a ``datetime``.
- `date() -> NDArray[np.datetime64]`: Current bar's date expressed as a ``numpy.datetime64``.
- `open() -> NDArray[np.float64]`: Current bar's open price.
- `high() -> NDArray[np.float64]`: Current bar's high price.
- `low() -> NDArray[np.float64]`: Current bar's low price.
- `close() -> NDArray[np.float64]`: Current bar's close price.
- `volume() -> Optional[NDArray[np.float64]]`: Current bar's volume.
- `vwap() -> Optional[NDArray[np.float64]]`: Current bar's volume-weighted average price (VWAP).
- `open_price() -> float`: Current bar's open price as a scalar.
- `high_price() -> float`: Current bar's high price as a scalar.
- `low_price() -> float`: Current bar's low price as a scalar.
- `close_price() -> float`: Current bar's close price as a scalar.
- `volume_value() -> Optional[float]`: Current bar's volume as a scalar.
- `vwap_value() -> Optional[float]`: Current bar's VWAP as a scalar.
- `cover_fill_price() -> Optional[Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]]`: Alias for :attr:`.buy_fill_price`.
- `cover_fill_price(fill_price: Optional[Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]])`
- `cover_shares() -> Optional[Union[int, float, Decimal]]`: Alias for :attr:`.buy_shares`.
- `cover_shares(shares: Optional[Union[int, float, Decimal]])`
- `cover_limit_price() -> Optional[Union[int, float, Decimal]]`: Alias for :attr:`.buy_limit_price`.
- `cover_limit_price(limit_price: Optional[Union[int, float, Decimal]])`
- `sell_all_shares()`: Sells all long shares of :attr:`.ExecContext.symbol`.
- `cover_all_shares()`: Covers all short shares of :attr:`.ExecContext.symbol`.
- `foreign(symbol: str, col: Optional[str]=None) -> Union[BarData, Optional[NDArray]]`: Retrieves bar data for another ticker symbol.
- `interval(interval: TimeframeInterval) -> IntervalContext`: Returns a read-only view of compressed bar data for ``interval``.
- `model(name: str, symbol: Optional[str]=None) -> Any`: Returns a trained model.
- `hyperparam(name: str) -> Any`: Returns a hyperparameter value for this execution.
- `indicator(name: str, symbol: Optional[str]=None) -> NDArray[np.float64]`: Returns indicator data.
- `input(model_name: str, symbol: Optional[str]=None) -> pd.DataFrame`: Returns model input data for making predictions.
- `preds(model_name: str, symbol: Optional[str]=None) -> NDArray`: Returns model predictions.
- `long_pos(symbol: Optional[str]=None) -> Optional[Position]`: Retrieves a current long :class:`pybroker.portfolio.Position` for a ``symbol``.
- `short_pos(symbol: Optional[str]=None) -> Optional[Position]`: Retrieves a current short :class:`pybroker.portfolio.Position` for a ``symbol``.
- `calc_target_shares(target_size: float, price: Optional[float]=None, cash: Optional[float]=None) -> Union[Decimal, int]`: Calculates the number of shares given a ``target_size`` allocation and share ``price``.
- `set_target_shares(target: float, *, dir: Literal['long', 'short'])`: Sets orders to reach a target allocation for long or short exposure.
- `cancel_pending_order(order_id: int) -> bool`: Cancels a :class:`pybroker.scope.PendingOrder` with ``order_id``.
- `cancel_all_pending_orders(symbol: Optional[str]=None)`: Cancels all :class:`pybroker.scope.PendingOrder`\ s for ``symbol``.
- `cancel_stop(stop_id: int) -> bool`: Cancels a :class:`pybroker.portfolio.Stop` with ``stop_id``.
- `cancel_stops(val: Union[str, Position, Entry], stop_type: Optional[StopType]=None)`: Cancels :class:`pybroker.portfolio.Stop`\ s.
- `to_result() -> Optional[ExecResult]`: Creates an :class:`.ExecResult` from the data set on :class:`.ExecContext`.

### `class RotationContext`

Context passed to a rotation sizer set with :meth:`pybroker.strategy.Strategy.enable_rotation`.

- `ctxs: Mapping[str, ExecContext]`
- `portfolio: Portfolio`
- `long_ranks: Mapping[str, int]`
- `short_ranks: Mapping[str, int]`
- `config: StrategyConfig`

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

- `auto_adjust: bool`
- `query(symbols: Union[str, Iterable[str]], start_date: Union[str, datetime], end_date: Union[str, datetime], _timeframe: Optional[str]='', _adjust: Optional[Any]=None) -> pd.DataFrame`: Queries data from `Yahoo Finance <https://finance.yahoo.com/>`_\ .

## `src/pybroker/eval.py`

Contains implementation of evaluation metrics.

### `class BootConfIntervals`

Holds confidence intervals of bootstrap tests.

- `low_2p5: float`
- `high_2p5: float`
- `low_5: float`
- `high_5: float`
- `low_10: float`
- `high_10: float`

- `bca_boot_conf(x: NDArray[np.float64], n_boot: int, fn: Callable[[NDArray[np.float64]], float]) -> BootConfIntervals`: Computes confidence intervals for a user-defined parameter using the `bias corrected and accelerated (BCa) bootstrap method.
- `profit_factor(changes: NDArray[np.float64], use_log: bool=False) -> float`: Computes the profit factor, which is the ratio of gross profit to gross loss.
- `log_profit_factor(changes: NDArray[np.float64]) -> float`: Computes the log transformed profit factor, which is the ratio of gross profit to gross loss.
- `sharpe_ratio(returns: NDArray[np.float64], obs: Optional[int]=None) -> float`: Computes the `Sharpe Ratio <https://en.wikipedia.org/wiki/Sharpe_ratio>`_.
- `downside_deviation(returns: NDArray[np.float64]) -> float`: Computes downside deviation, the denominator of the `Sortino Ratio <https://en.wikipedia.org/wiki/Sortino_ratio>`_.
- `sortino_ratio(returns: NDArray[np.float64], obs: Optional[int]=None) -> float`: Computes the `Sortino Ratio <https://en.wikipedia.org/wiki/Sortino_ratio>`_.
- `conf_profit_factor(x: NDArray[np.float64], n_boot: int) -> BootConfIntervals`: Computes confidence intervals for ``profit_factor``.
- `conf_sharpe_ratio(x: NDArray[np.float64], n_boot: int, obs: Optional[int]=None) -> BootConfIntervals`: Computes confidence intervals for :func:`.sharpe_ratio`.
- `max_drawdown(changes: NDArray[np.float64]) -> float`: Computes maximum drawdown, measured in cash.
- `calmar_ratio(returns: NDArray[np.float64], bars_per_year: int) -> float`: Computes the Calmar Ratio.
- `max_drawdown_percent(returns: NDArray[np.float64]) -> tuple[float, Optional[int]]`: Computes maximum drawdown, measured in percentage loss.

### `class DrawdownConfs`

Contains upper bounds of confidence intervals for maximum drawdown.

- `q_001: float`
- `q_01: float`
- `q_05: float`
- `q_10: float`

### `class DrawdownMetrics`

Contains drawdown metrics.

- `confs: DrawdownConfs`
- `pct_confs: DrawdownConfs`

- `drawdown_conf(changes: NDArray[np.float64], returns: NDArray[np.float64], n_boot: int) -> DrawdownMetrics`: Computes upper bounds of confidence intervals for maximum drawdown using the bootstrap method.
- `bootstrap_eval_all(changes: NDArray[np.float64], returns: NDArray[np.float64], n_boot: int, bars_per_year: int) -> tuple[BootConfIntervals, BootConfIntervals, DrawdownMetrics]`: Computes all bootstrap metrics in one shared resampling pass.
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

### `class TradeStats`

Trade statistics computed in a single pass.

- `trade_count: int`
- `win_rate: float`
- `loss_rate: float`
- `winning_trades: int`
- `losing_trades: int`
- `total_profit: float`
- `total_loss: float`
- `avg_profit: float`
- `avg_loss: float`
- `avg_profit_pct: float`
- `avg_loss_pct: float`
- `largest_win: float`
- `largest_loss: float`
- `largest_win_pct: float`
- `largest_loss_pct: float`
- `largest_win_bars: int`
- `largest_loss_bars: int`
- `max_wins: int`
- `max_losses: int`
- `avg_pnl: float`
- `avg_return_pct: float`
- `avg_trade_bars: float`
- `avg_winning_trade_bars: float`
- `avg_losing_trade_bars: float`
- `total_pnl: float`

### `class BootstrapResult`

Contains results of bootstrap tests.

- `conf_intervals: pd.DataFrame`
- `drawdown_conf: pd.DataFrame`
- `profit_factor: BootConfIntervals`
- `sharpe: BootConfIntervals`
- `drawdown: DrawdownMetrics`
- `to_json() -> dict[str, Any]`: Returns JSON-serializable bootstrap evaluation metrics.

### `class EvalMetrics`

Contains metrics for evaluating a :class:`pybroker.strategy.Strategy`.

- `trade_count: int = 0`
- `initial_market_value: float = 0`
- `end_market_value: float = 0`
- `total_pnl: float = 0`
- `unrealized_pnl: float = 0`
- `total_return_pct: float = 0`
- `annual_return_pct: Optional[float] = None`
- `total_profit: float = 0`
- `total_loss: float = 0`
- `total_fees: float = 0`
- `max_drawdown: float = 0`
- `max_drawdown_pct: float = 0`
- `max_drawdown_date: Optional[datetime] = None`
- `win_rate: float = 0`
- `loss_rate: float = 0`
- `winning_trades: int = 0`
- `losing_trades: int = 0`
- `avg_pnl: float = 0`
- `avg_return_pct: float = 0`
- `avg_trade_bars: float = 0`
- `avg_profit: float = 0`
- `avg_profit_pct: float = 0`
- `avg_winning_trade_bars: float = 0`
- `avg_loss: float = 0`
- `avg_loss_pct: float = 0`
- `avg_losing_trade_bars: float = 0`
- `largest_win: float = 0`
- `largest_win_pct: float = 0`
- `largest_win_bars: int = 0`
- `largest_loss: float = 0`
- `largest_loss_pct: float = 0`
- `largest_loss_bars: int = 0`
- `max_wins: int = 0`
- `max_losses: int = 0`
- `sharpe: float = 0`
- `sortino: float = 0`
- `calmar: Optional[float] = None`
- `profit_factor: float = 0`
- `ulcer_index: float = 0`
- `upi: float = 0`
- `equity_r2: float = 0`
- `std_error: float = 0`
- `annual_std_error: Optional[float] = None`
- `annual_volatility_pct: Optional[float] = None`
- `to_json() -> dict[str, Any]`: Returns JSON-serializable evaluation metrics.

### `class ConfInterval`

Confidence interval upper and low bounds.

- `name: str`
- `conf: str`
- `lower: float`
- `upper: float`

### `class EvalResult`

Contains evaluation result.

- `metrics: EvalMetrics`
- `bootstrap: Optional[BootstrapResult]`

### `class EvaluateMixin`

Mixin for computing evaluation metrics.

- `evaluate(portfolio_df: pd.DataFrame, trades_df: pd.DataFrame, calc_bootstrap: bool, bootstrap_samples: int, bars_per_year: Optional[int], seed: Optional[int]=42) -> EvalResult`: Computes evaluation metrics.

## `src/pybroker/indicator.py`

Contains indicator related functionality.

### `class Indicator(name: str, fn: Callable[..., NDArray[np.float64]], kwargs: dict[str, Any])`

Class representing an indicator.

- `name: str`
- `hyperparam_names() -> frozenset[str]`
- `relative_entropy(data: Union[BarData, pd.DataFrame]) -> float`: Generates indicator data with ``data`` and computes its relative `entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.
- `iqr(data: Union[BarData, pd.DataFrame]) -> float`: Generates indicator data with ``data`` and computes its `interquartile range (IQR) <https://en.wikipedia.org/wiki/Interquartile_range>`_.
- `__call__(data: Union[BarData, pd.DataFrame], hyperparams: Optional[dict[str, Any]]=None) -> pd.Series`: Computes indicator values.

- `indicator(name: str, fn: Callable[..., NDArray[np.float64]], **kwargs) -> Indicator`: Creates an :class:`.Indicator` instance and registers it globally with ``name``.

### `class IndicatorsMixin`

Mixin implementing indicator related functionality.

- `compute_indicators(df: pd.DataFrame, indicator_syms: Iterable[IndicatorSymbol], cache_date_fields: Optional[CacheDateFields], parallel_indicators: bool, interval_data: Optional[IntervalData]=None, symbol_store: Optional[SymbolArrayStore]=None, hyperparams: Optional[dict[str, Any]]=None) -> dict[IndicatorSymbol, pd.Series]`: Computes indicator data for the provided :class:`pybroker.common.IndicatorSymbol` pairs.

### `class IndicatorSet()`

Computes data for multiple indicators.

- `add(indicators: Union[Indicator, Iterable[Indicator]], *args)`: Adds indicators.
- `remove(indicators: Union[Indicator, Iterable[Indicator]], *args)`: Removes indicators.
- `clear()`: Removes all indicators.
- `__call__(df: pd.DataFrame, parallel_indicators: bool=False) -> pd.DataFrame`: Computes indicator data.

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
- `atr(name: str, lookback: int) -> Indicator`: Average True Range (ATR).
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
- `reactivity(name: str, lookback: int, smoothing: float=1.0, scale: float=0.6) -> Indicator`: Reactivity.
- `price_volume_fit(name: str, lookback: int, scale: float=9.0) -> Indicator`: Price Volume Fit.
- `volume_weighted_ma_ratio(name: str, lookback: int, scale: float=1.0) -> Indicator`: Volume-Weighted Moving Average Ratio.
- `normalized_on_balance_volume(name: str, lookback: int, scale: float=0.6) -> Indicator`: Normalized On-Balance Volume.
- `delta_on_balance_volume(name: str, lookback: int, delta_length: int=0, scale: float=0.6) -> Indicator`: Delta On-Balance Volume.
- `normalized_positive_volume_index(name: str, lookback: int, scale: float=0.5) -> Indicator`: Normalized Positive Volume Index.
- `normalized_negative_volume_index(name: str, lookback: int, scale: float=0.5) -> Indicator`: Normalized Negative Volume Index.
- `volume_momentum(name: str, short_length: int, multiplier: int=2, scale: float=3.0) -> Indicator`: Volume Momentum.
- `laguerre_rsi(name: str, fe_length: int=13) -> Indicator`: Laguerre Relative Strength Index (RSI).

## `src/pybroker/interval.py`

Multi-interval bar compression utilities.

### `class CompressedBars`

OHLCV and custom columns aggregated into compressed bars.

- `open: NDArray[np.float64]`
- `high: NDArray[np.float64]`
- `low: NDArray[np.float64]`
- `close: NDArray[np.float64]`
- `volume: NDArray[np.float64]`
- `dates: NDArray[np.datetime64]`
- `custom: Mapping[str, NDArray[np.float64]] = dict()`
- `vwap: Optional[NDArray[np.float64]] = None`
- `slice_by_dates(dates: Iterable[np.datetime64]) -> 'CompressedBars'`: Returns compressed bars restricted to ``dates``.

### `class CompressedSymbolData`

Compressed bar data and alignment map for one symbol.

- `bars: CompressedBars`
- `completed: NDArray[np.int64]`
- `base_dates: NDArray[np.datetime64]`

### `class IntervalData`

Compressed data keyed by ``(symbol, interval)``.

- `compressed: dict[tuple[str, TimeframeInterval], CompressedSymbolData] = dict()`
- `slice_for_test(test_symbol_dates: Mapping[str, NDArray[np.datetime64]]) -> 'IntervalData'`: Returns a copy with ``completed`` arrays aligned to test dates.

- `normalize_interval(interval: TimeframeInterval) -> TimeframeInterval`: Normalizes and validates a compression interval.
- `format_interval(interval: TimeframeInterval) -> str`: Returns a stable string representation of ``interval``.
- `validate_source_name(name: str, kind: str) -> None`: Raises if ``name`` cannot be used as an indicator or model name.
- `indicator_interval_name(base: str, interval: TimeframeInterval) -> str`: Returns the suffixed indicator name for an interval binding.
- `parse_indicator_interval_name(name: str) -> tuple[str, Optional[TimeframeInterval]]`: Parses a suffixed indicator name into base name and interval.
- `model_interval_name(base: str, interval: TimeframeInterval) -> str`: Returns the suffixed model name for an interval binding.
- `parse_model_interval_name(name: str) -> tuple[str, Optional[TimeframeInterval]]`: Parses a suffixed model name into base name and interval.
- `symbol_dates_from_frame(df: pd.DataFrame) -> dict[str, NDArray[np.datetime64]]`: Extracts per-symbol test dates from a multi-symbol frame.
- `build_compressed_symbol_arrays(symbol: str, interval: TimeframeInterval, compressed: CompressedSymbolData, indicator_data: Mapping[IndicatorSymbol, pd.Series], indicator_names: Iterable[str], custom_cols: Iterable[str]) -> tuple[tuple[str, ...], dict[str, NDArray], NDArray[np.datetime64]]`: Builds compressed-bar column arrays with base indicator names.
- `slice_arrays_by_dates(columns: tuple[str, ...], arrays: Mapping[str, NDArray], dates: NDArray[np.datetime64], selected: Iterable[np.datetime64]) -> tuple[tuple[str, ...], dict[str, NDArray], NDArray[np.datetime64]]`: Filters column arrays to rows whose dates are in ``selected``.
- `lookahead_train_dates(bar_dates: NDArray[np.datetime64], train_dates: Iterable[np.datetime64], test_dates: Iterable[np.datetime64], lookahead: int) -> tuple[NDArray[np.datetime64], int]`: Trims compressed train bar dates so the train/test hold-out is ``lookahead`` compressed bars wide.
- `build_compressed_symbol_df(symbol: str, interval: TimeframeInterval, compressed: CompressedSymbolData, indicator_data: Mapping[IndicatorSymbol, pd.Series], indicator_names: Iterable[str], custom_cols: Iterable[str]) -> pd.DataFrame`: Builds a compressed-bar DataFrame with base indicator column names.
- `slice_compressed_df_by_dates(df: pd.DataFrame, dates: Iterable[np.datetime64]) -> pd.DataFrame`: Filters a compressed DataFrame to rows whose dates are in ``dates``.
- `base_timeframe_to_seconds(base_timeframe: str) -> float`: Converts a base timeframe string to seconds.
- `validate_base_timeframe_data(df: pd.DataFrame, base_bar_seconds: float) -> None`: Raises if bar timestamps are inconsistent with ``base_bar_seconds``.
- `compressed_bars_to_bar_data(bars: CompressedBars) -> BarData`: Converts compressed OHLCV arrays to :class:`~pybroker.common.BarData`.
- `validate_interval(interval: TimeframeInterval, base_bar_seconds: float) -> None`: Validates an interval against the base feed bar spacing.
- `is_valid_interval(interval: TimeframeInterval, base_bar_seconds: float) -> bool`: Returns whether ``interval`` is valid for the base feed bar spacing.
- `compress(dates: NDArray[np.datetime64], open_: NDArray[np.float64], high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64], interval: TimeframeInterval, custom_cols: Optional[Mapping[str, NDArray[np.float64]]]=None, vwap: Optional[NDArray[np.float64]]=None) -> tuple[CompressedBars, NDArray[np.int64]]`: Compresses base bars into coarser interval bars.
- `compress_bars(data: Union[BarData, pd.DataFrame], interval: TimeframeInterval, *, base_timeframe: str) -> BarData`: Compresses base OHLCV bars to a coarser ``interval``.
- `compress_symbol_intervals_from_frame(df: pd.DataFrame, symbol: str, intervals: Iterable[TimeframeInterval], custom_cols: Iterable[str], base_bar_seconds: float, *, validate_dates: bool=True, rows: Optional[NDArray[np.int64]]=None) -> dict[TimeframeInterval, CompressedSymbolData]`: Compresses one symbol to multiple intervals with a single OHLCV extract.
- `compress_intervals_from_frame(df: pd.DataFrame, symbol_intervals: Mapping[str, Iterable[TimeframeInterval]], custom_cols: Iterable[str], base_bar_seconds: float) -> IntervalData`: Compresses each symbol to the intervals declared for it.
- `compress_symbol_from_frame(df: pd.DataFrame, symbol: str, interval: TimeframeInterval, custom_cols: Iterable[str], base_bar_seconds: float, *, validate_dates: bool=True) -> CompressedSymbolData`: Compresses one symbol from a multi-symbol frame without copying rows.
- `compress_symbol_df(sym_df: pd.DataFrame, interval: TimeframeInterval, custom_cols: Iterable[str], base_bar_seconds: float, *, validate_dates: bool=True) -> CompressedSymbolData`: Compresses a single-symbol DataFrame.

## `src/pybroker/log.py`

Logging module.

### `class Logger(scope)`

Class for logging information about triggered events.

_Status-logging methods omitted (not needed for strategy code)._

## `src/pybroker/model.py`

Contains model related functionality.

### `class LagSeriesKey`

Internal cache key for a full-history lagged series.

- `symbol: str`
- `column: str`
- `lag: int`
- `interval: Optional[str] = None`

### `class ModelInput`

Internal numpy-backed model input with optional lag feature metadata.

- `columns: tuple[str, ...]`
- `arrays: ArrayDict`
- `dates: np.ndarray`
- `lag_features: Optional[np.ndarray] = None`
- `lags: Optional[int] = None`
- `lag_columns: Optional[tuple[str, ...]] = None`
- `empty() -> bool`
- `slice(end_index: Optional[int]=None) -> ModelInput`: Returns a row slice sharing backing array memory.
- `slice_range(start: int, end_index: Optional[int]=None) -> ModelInput`: Returns a row range sharing backing array memory.
- `lag_warmup_len() -> int`: Returns how many leading rows have undefined lag features.
- `select_columns(columns: tuple[str, ...]) -> ModelInput`: Returns a view restricted to ``columns``.
- `drop_lag_warmup() -> ModelInput`: Drops the leading rows whose lag features are not yet defined.
- `to_dataframe() -> pd.DataFrame`: Materializes a DataFrame of the input columns.

- `shift_array(values: np.ndarray, lag: int) -> np.ndarray`: Returns ``values`` shifted forward by ``lag`` bars with NaN warmup.
- `cached_stacked_lags(cache: LagSeriesCache, symbol: str, col: str, lags: int, interval: Optional[str]=None) -> Optional[np.ndarray]`: Returns a cached stacked array deep enough for ``lags``, else ``None``.
- `model_input_from_frame(df: pd.DataFrame, columns: Optional[tuple[str, ...]]=None, dates: Optional[np.ndarray]=None) -> ModelInput`: Builds a :class:`ModelInput` from a DataFrame without copying columns.
- `model_input_from_arrays(columns: tuple[str, ...], arrays: ArrayDict, dates: np.ndarray) -> ModelInput`: Builds a :class:`ModelInput` from column arrays.
- `symbol_history_arrays(history_df: pd.DataFrame, symbol: str, columns: tuple[str, ...]) -> tuple[np.ndarray, ArrayDict]`: Extracts sorted full-history date and column arrays for one symbol.
- `compute_lag_series_cache(df: pd.DataFrame, symbols: Iterable[str], columns: tuple[str, ...], lags: int) -> LagSeriesCache`: Computes full-history lag arrays for daily/base bars.
- `merge_lag_series_cache_from_store(cache: LagSeriesCache, store: 'SymbolArrayStore', symbols: Iterable[str], columns: tuple[str, ...], lags: int, history_dates: Optional[dict[str, np.ndarray]]=None, indicators: tuple[str, ...]=(), indicator_data: Optional[Mapping[IndicatorSymbol, pd.Series]]=None) -> LagSeriesCache`: Adds full-history lag arrays from a :class:`pybroker.scope.SymbolArrayStore`.
- `merge_lag_series_cache(cache: LagSeriesCache, history_df: pd.DataFrame, symbols: Iterable[str], columns: tuple[str, ...], lags: int, history_dates: Optional[dict[str, np.ndarray]]=None) -> LagSeriesCache`: Adds full-history lag arrays for ``columns`` into ``cache``.
- `merge_lag_series_cache_from_arrays(cache: LagSeriesCache, symbol: str, columns: tuple[str, ...], lags: int, history_dates: np.ndarray, column_arrays: Mapping[str, np.ndarray]) -> None`: Adds full-history lag arrays built from numpy column data.
- `merge_interval_lag_series_cache(cache: LagSeriesCache, symbols: Iterable[str], columns: tuple[str, ...], lags: int, interval: str, bars_by_symbol, arrays_by_symbol=None) -> LagSeriesCache`: Adds full-history interval lag arrays into ``cache``.
- `history_date_offset(history_dates: np.ndarray, row_dates: np.ndarray) -> int`: Returns the start index of ``row_dates`` inside ``history_dates``.
- `build_lag_feature_matrix(symbol: str, columns: tuple[str, ...], lags: int, row_dates: np.ndarray, history_dates: np.ndarray, lag_cache: LagSeriesCache, interval: Optional[str]=None) -> np.ndarray`: Builds a lag-expanded feature matrix from numpy arrays.
- `build_lag_feature_matrix_pooled(sym_col: np.ndarray, columns: tuple[str, ...], lags: int, row_dates: np.ndarray, history_dates_by_symbol: dict[str, np.ndarray], lag_cache: LagSeriesCache, symbols: Iterable[str], interval: Optional[str]=None) -> np.ndarray`: Builds a lag-expanded feature matrix for pooled multi-symbol data.
- `apply_lags_to_model_input(model_input: ModelInput, lag_columns: tuple[str, ...], lags: int, lag_cache: LagSeriesCache, symbol: str, history_dates: np.ndarray, interval: Optional[str]=None) -> ModelInput`: Attaches lag feature metadata to ``model_input``.
- `apply_lags_to_model_input_pooled(model_input: ModelInput, lag_columns: tuple[str, ...], lags: int, lag_cache: LagSeriesCache, history_dates_by_symbol: dict[str, np.ndarray], symbols: Iterable[str], interval: Optional[str]=None) -> ModelInput`: Attaches lag feature metadata to pooled ``model_input``.
- `apply_prepare_input_data(model_input: ModelInput, prepare_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> ModelInput`: Applies a DataFrame-only prepare function to ``model_input``.

### `class ModelSource(name: str, indicator_names: Iterable[str], input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]], predict_fn: Optional[Callable[[Any, Union[pd.DataFrame, NDArray]], NDArray]], pooled: bool, kwargs: dict[str, Any], lags: Optional[int]=None, lag_cols: tuple[str, ...]=(), per_bar: bool=False)`

Base class of a model source.

- `name: str`
- `lags: Optional[int]`
- `per_bar: bool`
- `pooled: bool`
- `prepare_input_data(df: pd.DataFrame) -> pd.DataFrame`: Prepares a :class:`pandas.DataFrame` of input data for passing to a model when making predictions.

### `class ModelLoader(name: str, load_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]], indicator_names: Iterable[str], input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]], predict_fn: Optional[Callable[[Any, Union[pd.DataFrame, NDArray]], NDArray]], pooled: bool, kwargs: dict[str, Any], lags: Optional[int]=None, lag_cols: tuple[str, ...]=(), per_bar: bool=False)`

Loads a pre-trained model.

- `__call__(symbol: str, train_start_date: datetime, train_end_date: datetime) -> Union[Any, tuple[Any, Iterable[str]]]`: Loads pre-trained model.

### `class ModelTrainer(name: str, train_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]], indicator_names: Iterable[str], input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]], predict_fn: Optional[Callable[[Any, Union[pd.DataFrame, NDArray]], NDArray]], pooled: bool, kwargs: dict[str, Any], lags: Optional[int]=None, lag_cols: tuple[str, ...]=(), per_bar: bool=False)`

Trains a model.

- `__call__(symbol: str, train_data: pd.DataFrame, test_data: pd.DataFrame, *, lag_train: Optional[NDArray]=None, lag_test: Optional[NDArray]=None) -> Union[Any, tuple[Any, Iterable[str]]]`: Trains model per symbol.
- `train_pooled(symbols: Sequence[str], train_data: pd.DataFrame, test_data: pd.DataFrame, *, lag_train: Optional[NDArray]=None, lag_test: Optional[NDArray]=None) -> Union[Any, tuple[Any, Iterable[str]]]`: Trains model using combined multi-symbol data.

- `model(name: str, fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]], indicators: Optional[Iterable[Indicator]]=None, lags: Optional[int]=None, lag_cols: Optional[Iterable[Union[str, Indicator]]]=None, per_bar: bool=False, input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]]=None, predict_fn: Optional[Callable[[Any, Union[pd.DataFrame, NDArray]], NDArray]]=None, pretrained: bool=False, pooled: bool=False, **kwargs) -> ModelSource`: Creates a :class:`.ModelSource` instance and registers it globally with ``name``.

### `class CachedModel`

Stores cached model data.

- `model: Any`
- `input_cols: Optional[tuple[str]]`
- `lag_columns: Optional[tuple[str, ...]] = None`

### `class ModelsMixin`

Mixin implementing model related functionality.

- `train_models(model_syms: Iterable[ModelSymbol], train_data: pd.DataFrame, test_data: pd.DataFrame, indicator_data: Mapping[IndicatorSymbol, pd.Series], cache_date_fields: CacheDateFields, parallel_models: bool=False, pooled_model_groups: Optional[Mapping[tuple[str, int], frozenset[str]]]=None, interval_data: Optional[IntervalData]=None, *, history_store: Optional[SymbolArrayStore]=None, train_store: Optional[SymbolArrayStore]=None, test_store: Optional[SymbolArrayStore]=None, lookahead: int=1) -> dict[ModelSymbol, TrainedModel]`: Trains models for the provided :class:`pybroker.common.ModelSymbol` pairs.

## `src/pybroker/optimize.py`

Hyperparameter declaration and optimization with Optuna.

### `class Hyperparam`

Declares a named hyperparameter with bounds and step size.

- `name: str`
- `default: Union[int, float]`
- `low: Union[int, float]`
- `high: Union[int, float]`
- `step: Union[int, float]`

- `hyperparam(name: str, *, default: Union[int, float], low: Union[int, float], high: Union[int, float], step: Union[int, float]) -> Hyperparam`: Creates and registers a :class:`Hyperparam`.

### `class SearchSpace`

Searchable hyperparameters collected from a strategy.

- `hyperparams: frozenset[str]`
- `specs: Mapping[str, Hyperparam]`
- `grid_size() -> int`: Total number of grid combinations.

- `build_run_hyperparams(specs: Mapping[str, Hyperparam], overrides: Optional[dict[str, Any]]=None) -> dict[str, Any]`: Builds the hyperparam dict for a single backtest or trial run.
- `collect_hyperparams(strategy: _ExecutionsHost) -> dict[str, Hyperparam]`: Collects all hyperparams reachable from ``strategy``.
- `collect_search_space(strategy: _ExecutionsHost) -> SearchSpace`: Collects searchable hyperparams reachable from ``strategy``.

### `class WindowOptimizeResult`

Per-window walk-forward optimization result.

- `params: dict[str, Any]`
- `study: optuna.Study`
- `train_score: float`
- `train_start_date: Optional[datetime] = None`
- `train_end_date: Optional[datetime] = None`
- `test_start_date: Optional[datetime] = None`
- `test_end_date: Optional[datetime] = None`
- `execution_symbols: Optional[dict[int, frozenset[str]]] = None`
- `to_json() -> dict[str, Any]`: Returns JSON-serializable walk-forward optimization window results.
- `to_json_str() -> str`: Returns strict JSON text from :meth:`to_json`.

### `class OptimizeResult`

Result of ``Strategy.optimize()``.

- `best_params: dict[str, Any]`
- `best_score: float`
- `result: TestResult`
- `study: optuna.Study`
- `windows: Optional[tuple[WindowOptimizeResult, ...]] = None`
- `to_json(*, include: Optional[frozenset[str]]=None, max_rows: Optional[int]=100, symbols: Optional[frozenset[str]]=None) -> dict[str, Any]`: Returns JSON-serializable optimization results.
- `to_json_str(*, include: Optional[frozenset[str]]=None, max_rows: Optional[int]=100, symbols: Optional[frozenset[str]]=None) -> str`: Returns strict JSON text from :meth:`to_json`.

### `class ObjectiveBundle`

Return value of :func:`make_objective`.

- `objective: Callable[[optuna.Trial], float]`
- `search_space: SearchSpace`
- `score_overrides: Callable[[dict[str, Any]], float]`

- `make_objective(strategy: _OptimizeTrialHost, score_fn: Callable[[TestResult], float], *, train_rows: np.ndarray, df: pd.DataFrame, hyperparams: Mapping[str, Hyperparam], search_space: SearchSpace, invariant_indicator_data: dict[IndicatorSymbol, pd.Series], window_executions: set[Execution], master_store: Any, interval_data: Any, parallel_indicators: bool, warmup: Optional[int], pretrained_models: Mapping[ModelSymbol, TrainedModel], exit_dates: Mapping[str, np.datetime64], verbose: bool=False) -> ObjectiveBundle`: Builds an Optuna objective for train-window scoring.

### `class OptimizeMixin`

Mixin implementing hyperparameter optimization.

- `optimize(score_fn: Callable[[TestResult], float], *, sampler: Union[str, BaseSampler]='grid', n_trials: Optional[int]=None, direction: str='maximize', seed: Optional[int]=None, windows: Optional[int]=None, study: Optional[optuna.Study]=None, pruner: Optional[optuna.pruners.BasePruner]=None, train_size: float=0.5, lookahead: int=1, start_date: Optional[Union[str, datetime]]=None, end_date: Optional[Union[str, datetime]]=None, timeframe: str='', between_time: Optional[tuple[str, str]]=None, days: Optional[Any]=None, warmup: Optional[int]=None, parallel_indicators: bool=False, adjust: Optional[Any]=None, calc_bootstrap: bool=False, verbose: bool=False) -> OptimizeResult`: Searches :func:`pybroker.optimize.hyperparam` values on a training window, then evaluates the best values on the held out test window.

## `src/pybroker/parallel.py`

Contains parallel execution configuration.

### `class ParallelConfig`

Configuration for parallel execution used by PyBroker.

- `n_jobs: Optional[int] = -1`
- `backend: Optional[str] = 'loky'`
- `parallel: Optional[Parallel] = None`

- `set_parallel(n_jobs: Optional[int]=None, backend: Optional[str]=None, parallel: Optional[Parallel]=None) -> None`: Configures parallel execution used by PyBroker.
- `get_parallel_config() -> ParallelConfig`: Returns the current parallel configuration
- `parallel() -> Iterator[Parallel]`

## `src/pybroker/portfolio.py`

Contains portfolio related functionality, such as portfolio metrics and placing orders.

### `class Stop`

Contains information about a stop set on :class:`.Entry`.

- `id: int`
- `symbol: str`
- `stop_type: StopType`
- `pos_type: Literal['long', 'short']`
- `percent: Optional[Decimal]`
- `points: Optional[Decimal]`
- `bars: Optional[int]`
- `fill_price: Optional[Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]]`
- `limit_price: Optional[Decimal]`
- `exit_price: Optional[PriceType]`

### `class StopRecord`

Records per-bar data about a stop.

- `date: np.datetime64`
- `symbol: str`
- `stop_id: int`
- `stop_type: str`
- `pos_type: Literal['long', 'short']`
- `curr_value: Optional[Decimal]`
- `curr_bars: Optional[int]`
- `percent: Optional[Decimal]`
- `points: Optional[Decimal]`
- `bars: Optional[int]`
- `fill_price: Optional[Decimal]`
- `limit_price: Optional[Decimal]`
- `exit_price: Optional[PriceType]`

### `class Entry`

Contains information about an entry into a :class:`.Position`.

- `id: int`
- `date: np.datetime64`
- `symbol: str`
- `shares: Decimal`
- `price: Decimal`
- `type: Literal['long', 'short']`
- `bars: int = 0`
- `sym_bars: int = 0`
- `stops: list[Stop] = list()`
- `mae: Decimal = Decimal()`
- `mfe: Decimal = Decimal()`

### `class Position`

Contains information about an open position in ``symbol``.

- `symbol: str`
- `shares: Decimal`
- `type: Literal['long', 'short']`
- `close: Decimal = Decimal()`
- `equity: Decimal = Decimal()`
- `market_value: Decimal = Decimal()`
- `margin: Decimal = Decimal()`
- `pnl: Decimal = Decimal()`
- `entries: deque[Entry] = deque()`
- `bars: int = 0`
- `entry_notional: Decimal = Decimal()`
- `unmarked_shares: Decimal = Decimal()`
- `unmarked_notional: Decimal = Decimal()`

### `class Trade`

Holds information about a completed trade (entry and exit).

- `id: int`
- `type: Literal['long', 'short']`
- `symbol: str`
- `entry_date: np.datetime64`
- `exit_date: np.datetime64`
- `entry: Decimal`
- `exit: Decimal`
- `shares: Decimal`
- `pnl: Decimal`
- `return_pct: Decimal`
- `agg_pnl: Decimal`
- `bars: int`
- `pnl_per_bar: Decimal`
- `stop: Optional[Literal['bar', 'loss', 'profit', 'trailing']]`
- `mae: Decimal`
- `mfe: Decimal`

### `class Order`

Holds information about a filled order.

- `id: int`
- `type: Literal['buy', 'sell']`
- `symbol: str`
- `date: np.datetime64`
- `created: Optional[np.datetime64]`
- `order_type: Literal['market', 'limit', 'stop_bar', 'stop_loss', 'stop_profit', 'stop_trailing']`
- `intent: Literal['buy_to_open', 'buy_to_close', 'sell_to_open', 'sell_to_close']`
- `shares: Decimal`
- `limit_price: Optional[Decimal]`
- `market_price: Decimal`
- `fill_price: Decimal`
- `fees: Decimal`

### `class PortfolioBar`

Snapshot of :class:`.Portfolio` state, captured per bar.

- `date: np.datetime64`
- `cash: Decimal`
- `equity: Decimal`
- `notional: Decimal`
- `margin: Decimal`
- `margin_loan: Decimal`
- `net_cash_balance: Decimal`
- `market_value: Decimal`
- `pnl: Decimal`
- `unrealized_pnl: Decimal`
- `fees: Decimal`

### `class PositionBar`

Snapshot of an open :class:`.Position`\ 's state, captured per bar.

- `symbol: str`
- `date: np.datetime64`
- `long_shares: Decimal`
- `short_shares: Decimal`
- `close: Decimal`
- `equity: Decimal`
- `market_value: Decimal`
- `margin: Decimal`
- `unrealized_pnl: Decimal`

### `class Portfolio(cash: float, fee_mode: Optional[Union[FeeMode, Callable[[FeeInfo], Decimal], None]]=None, fee_amount: Optional[float]=None, enable_fractional_shares: bool=False, position_mode: PositionMode=PositionMode.DEFAULT, max_long_positions: Optional[int]=None, max_short_positions: Optional[int]=None, record_stops: Optional[bool]=False, leverage: float=1.0, interest_rate: float=0.0, bars_per_year: Optional[int]=None, record_portfolio_bars: bool=False, record_position_bars: bool=False)`

Class representing a portfolio of holdings.

- `cash: Decimal`
- `equity: Decimal`
- `market_value: Decimal`
- `orders: deque[Order]`
- `trades: deque[Trade]`
- `margin: Decimal`
- `pnl: Decimal`
- `long_positions: dict[str, Position]`
- `short_positions: dict[str, Position]`
- `symbols: set[str]`
- `bars: deque[PortfolioBar]`
- `position_bars: deque[PositionBar]`
- `margin_loan() -> Decimal`: Borrowed funds used for leveraged long and short positions.
- `win_rate() -> Decimal`
- `loss_rate() -> Decimal`
- `buy(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal]=None, stops: Optional[Iterable[Stop]]=None, created: Optional[np.datetime64]=None, order_type: OrderType=OrderType.MARKET, market_price: Optional[Decimal]=None) -> Optional[Order]`: Places a buy order.
- `sell(date: np.datetime64, symbol: str, shares: Decimal, fill_price: Decimal, limit_price: Optional[Decimal]=None, stops: Optional[Iterable[Stop]]=None, created: Optional[np.datetime64]=None, order_type: OrderType=OrderType.MARKET, market_price: Optional[Decimal]=None) -> Optional[Order]`: Places a sell order.
- `exit_position(date: np.datetime64, symbol: str, buy_fill_price: Decimal, sell_fill_price: Decimal, col_scope: Optional[ColumnScope]=None, ind_scope: Optional[IndicatorScope]=None, sym_end_index: Optional[Mapping[str, int]]=None, slippage_model: Optional['SlippageModel']=None)`: Exits any long and short positions for ``symbol`` at ``buy_fill_price`` and ``sell_fill_price``.
- `capture_bar(date: np.datetime64, col_scope: ColumnScope, sym_end_index: Mapping[str, int], price_scope: Optional[PriceScope]=None)`: Captures portfolio state of the current bar.
- `incr_bars(date: Optional[np.datetime64]=None, price_scope: Optional[PriceScope]=None)`: Increments the number of bars held by every trade entry.
- `remove_stop(stop_id: int) -> bool`: Removes a :class:`.Stop` with ``stop_id``.
- `remove_stops(val: Union[str, Position, Entry], stop_type: Optional[StopType]=None)`: Removes :class:`.Stop`\ s.
- `check_stops(date: np.datetime64, price_scope: PriceScope, col_scope: Optional[ColumnScope]=None, sym_end_index: Optional[Mapping[str, int]]=None, ind_scope: Optional[IndicatorScope]=None, slippage_model: Optional['SlippageModel']=None)`: Checks whether stops are triggered.

## `src/pybroker/scope.py`

Contains scopes that store data and object references used to execute a :class:`pybroker.strategy.Strategy`.

### `class StaticScope()`

A static registry of data and object references.

- `data_source_cache: Optional[Cache]`
- `data_source_cache_ns: str`
- `indicator_cache: Optional[Cache]`
- `indicator_cache_ns: str`
- `model_cache: Optional[Cache]`
- `model_cache_ns: str`
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
- `ordered_data_cols() -> tuple[str, ...]`: All registered data column names in deterministic order.
- `freeze_data_cols()`: Prevents additional data columns from being registered.
- `unfreeze_data_cols()`: Allows additional data columns to be registered if :func:`pybroker.scope.StaticScope.freeze_data_cols` was called.
- `param(name: str, value: Optional[Any]=_EMPTY_PARAM) -> Optional[Any]`: Get or set a global parameter.
- `clear_params()`: Clears all global parameters.
- `set_hyperparam(hyperparam: Any) -> None`: Stores a :class:`pybroker.optimize.Hyperparam` in static scope.
- `has_hyperparam(name: str) -> bool`: Whether a hyperparam is stored in static scope.
- `get_hyperparam(name: str) -> Any`: Retrieves a hyperparam from static scope.
- `iter_hyperparams() -> Iterable[Any]`: Iterates registered hyperparams.
- `instance() -> 'StaticScope'`: Returns singleton instance.
- `set_instance(scope: Optional['StaticScope']) -> None`: Replaces the singleton instance, or clears it when ``scope`` is ``None``.

- `run_with_scope(scope: StaticScope, fn: Callable[..., Any], *args: Any) -> Any`: Installs ``scope`` as this process' scope, then runs ``fn``.
- `disable_logging()`: Disables event logging.
- `enable_logging()`: Enables event logging.
- `disable_progress_bar()`: Disables logging a progress bar.
- `enable_progress_bar()`: Enables logging a progress bar.
- `register_columns(names: Union[str, Iterable[str]], *args)`: Registers ``names`` of user-defined data columns.
- `unregister_columns(names: Union[str, Iterable[str]], *args)`: Unregisters ``names`` of user-defined data columns.
- `param(name: str, value: Optional[Any]=_EMPTY_PARAM) -> Optional[Any]`: Get or set a global parameter.
- `clear_params()`: Clears all global parameters.

### `class SymbolArrayStore`

Internal numpy-backed OHLCV/custom columns keyed by symbol.

- `symbols: frozenset[str]`
- `sym_arrays: Mapping[str, Mapping[str, NDArray]]`
- `backing: Optional[_StoreBacking] = None`
- `unique_dates() -> NDArray[np.datetime64]`: Returns sorted unique dates across every symbol.

- `symbol_array_store_from_indexed_df(df: pd.DataFrame) -> SymbolArrayStore`: Builds a :class:`SymbolArrayStore` from a sorted MultiIndex frame.
- `symbol_array_store_from_flat_frame(df: pd.DataFrame, sym_col: str=DataCol.SYMBOL.value, date_col: str=DataCol.DATE.value, symbols: Optional[frozenset[str]]=None) -> SymbolArrayStore`: Builds a store from a flat frame via numpy lex-sort and bin slicing.
- `symbol_array_store_from_frame(df: pd.DataFrame, sym_col: str=DataCol.SYMBOL.value, date_col: str=DataCol.DATE.value, symbols: Optional[frozenset[str]]=None) -> SymbolArrayStore`: Builds a store from a flat or MultiIndex OHLCV frame.
- `sym_data_from_store(store: SymbolArrayStore, data_cols: Iterable[str]) -> dict[str, dict[str, Optional[NDArray]]]`: Converts a :class:`SymbolArrayStore` to per-symbol column arrays.
- `slice_symbol_array_store_by_dates(store: SymbolArrayStore, selected_dates: Union[Sequence[np.datetime64], NDArray[np.datetime64]]) -> SymbolArrayStore`: Filters a store to rows whose dates are in ``selected_dates``.
- `merge_symbol_array_stores(left: SymbolArrayStore, right: SymbolArrayStore) -> SymbolArrayStore`: Concatenates per-symbol column arrays from two stores.
- `column_scope_from_frame(df: pd.DataFrame, sym_col: str=DataCol.SYMBOL.value, date_col: str=DataCol.DATE.value) -> 'ColumnScope'`: Creates a :class:`ColumnScope` with upfront numpy extraction.
- `sym_exec_dates_from_store(store: SymbolArrayStore) -> dict[str, frozenset[np.datetime64]]`: Returns per-symbol test dates from a column store.

### `class ColumnScope(store: Union[SymbolArrayStore, pd.DataFrame])`

Caches and retrieves column data from a :class:`SymbolArrayStore`.

- `store() -> SymbolArrayStore`
- `symbols() -> frozenset[str]`: Symbols held by the underlying store.
- `unique_dates() -> NDArray[np.datetime64]`: Returns sorted unique dates across every symbol in the store.
- `fetch_dict(symbol: str, names: Iterable[str], end_index: Optional[int]=None) -> dict[str, Optional[NDArray]]`: Fetches a ``dict`` of column data for ``symbol``.
- `fetch(symbol: str, name: str, end_index: Optional[int]=None) -> Optional[NDArray]`: Fetches a :class:`numpy.ndarray` of column data for ``symbol``.
- `fetch_value(symbol: str, name: str, end_index: int) -> Optional[float]`: Returns the scalar value at ``end_index - 1`` without slicing.
- `bar_data_from_data_columns(symbol: str, end_index: int) -> BarData`: Returns a new :class:`pybroker.common.BarData` instance containing column data of default and custom data columns registered with :class:`.StaticScope`.

### `class IndicatorScope(indicator_data: Mapping[IndicatorSymbol, pd.Series], filter_dates: Sequence[np.datetime64])`

Caches and retrieves :class:`pybroker.indicator.Indicator` data.

- `fetch(symbol: str, name: str, end_index: Optional[int]=None) -> NDArray[np.float64]`: Fetches :class:`pybroker.indicator.Indicator` data.
- `fetch_full(symbol: str, name: str) -> NDArray[np.float64]`: Fetches the full indicator array without truncation.
- `has_indicator(symbol: str, name: str) -> bool`: Whether :class:`pybroker.indicator.Indicator` data is registered for ``symbol``.
- `fetch_history(symbol: str, name: str, dates: NDArray[Any]) -> Optional[NDArray[np.float64]]`: Aligns full-history indicator values to ``dates``.
- `fetch_value(symbol: str, name: str, end_index: int) -> float`: Returns the scalar value at ``end_index - 1`` without slicing.

### `class IntervalScope(interval_data: IntervalData, ind_scope: IndicatorScope, models: Optional[Mapping[ModelSymbol, TrainedModel]]=None, test_dates: Optional[Sequence[np.datetime64]]=None)`

Serves compressed bar and indicator data through alignment maps.

- `window_len(symbol: str, interval: TimeframeInterval) -> int`: Returns the compressed bar count visible in the current window.
- `completed_index(symbol: str, interval: TimeframeInterval, end_index: int) -> int`
- `fetch_bar(symbol: str, interval: TimeframeInterval, col: str, end_index: int) -> NDArray[Any]`
- `fetch_indicator(symbol: str, interval: TimeframeInterval, base_name: str, end_index: int) -> NDArray[np.float64]`
- `fetch_input(symbol: str, interval: TimeframeInterval, base_model_name: str, end_index: int) -> pd.DataFrame`
- `fetch_preds(symbol: str, interval: TimeframeInterval, base_model_name: str, end_index: int) -> NDArray`
- `clear_cache()`: Drops every cached array.

### `class ModelInputScope(col_scope: ColumnScope, ind_scope: IndicatorScope, models: Mapping[ModelSymbol, TrainedModel], history_col_scope: Optional['ColumnScope']=None, test_dates: Optional[Sequence[np.datetime64]]=None)`

Caches and retrieves model input data.

- `fetch(symbol: str, name: str, end_index: Optional[int]=None) -> pd.DataFrame`: Fetches model input data.
- `fetch_model_input(symbol: str, name: str, end_index: Optional[int]=None) -> ModelInput`: Fetches model input as internal :class:`pybroker.model.ModelInput` (no DataFrame).

### `class PredictionScope(models: Mapping[ModelSymbol, TrainedModel], input_scope: ModelInputScope)`

Caches and retrieves model predictions.

- `fetch(symbol: str, name: str, end_index: Optional[int]=None) -> NDArray`: Fetches model predictions.

### `class PriceScope(col_scope: ColumnScope, sym_end_index: Mapping[str, int], round_fill_price: bool)`

Retrieves most recent prices.

- `reset_bar() -> None`: Clears the per-bar OHLC cache.
- `has_bar(symbol: str) -> bool`: Returns whether ``symbol`` has a bar that can be priced.
- `has_bar_on(symbol: str, date: np.datetime64) -> bool`: Returns whether ``symbol``'s current bar falls on ``date``.
- `fetch_float(symbol: str, price: Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]) -> float`: Returns a bar price as ``float`` using the per-bar cache when possible.
- `fetch_bar_ohlc(symbol: str, date: np.datetime64) -> tuple[Optional[float], Optional[float], Optional[float]]`: Returns ``(close, low, high)`` for ``symbol`` on ``date``, or Nones.
- `fetch(symbol: str, price: Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]) -> Decimal`

### `class PendingOrder`

Holds data for a pending order.

- `id: int`
- `type: Literal['buy', 'sell']`
- `symbol: str`
- `created: np.datetime64`
- `exec_date: np.datetime64`
- `shares: Decimal`
- `limit_price: Optional[Decimal]`
- `fill_price: Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]]`
- `exec_bar: int`
- `timeout_bars: Optional[int]`
- `stops: Optional[frozenset['Stop']]`
- `exit_pos_type: Optional[Literal['long', 'short']] = None`

### `class PendingOrderScope()`

Stores :class:`.PendingOrder`\ s

- `mark_attempted(order_id: int) -> None`: Records that ``order_id`` has had its first fill attempt.
- `was_attempted(order_id: int) -> bool`: Returns whether ``order_id`` has had its first fill attempt.
- `contains(order_id: int) -> bool`: Returns whether a :class:`.PendingOrder` exists with ``order_id``.
- `has_orders() -> bool`: Returns whether any pending orders exist.
- `get(order_id: int) -> Optional[PendingOrder]`: Returns a :class:`.PendingOrder` with ``order_id``.
- `add(type: Literal['buy', 'sell'], symbol: str, created: np.datetime64, exec_date: np.datetime64, shares: Decimal, limit_price: Optional[Decimal], fill_price: Union[int, float, np.floating, Decimal, PriceType, Callable[[str, BarData], Union[int, float, Decimal]]], exec_bar: int, timeout_bars: Optional[int], stops: Optional[frozenset['Stop']]=None, exit_pos_type: Optional[Literal['long', 'short']]=None) -> int`: Creates a :class:`.PendingOrder`.
- `retry_bars(order_id: int) -> int`: Returns how many bars ``order_id`` has been retried for.
- `advance_retry_bars(order_id: int) -> None`: Records that ``order_id`` was attempted on a bar.
- `remove(order_id: int) -> bool`: Removes a :class:`.PendingOrder` with ``order_id```.
- `remove_all(symbol: Optional[str]=None)`: Removes all :class:`.PendingOrder`\ s.
- `orders(symbol: Optional[str]=None, order_id: Optional[int]=None) -> Iterable[PendingOrder]`: Returns an :class:`Iterable` of :class:`.PendingOrder`\ s.

- `get_signals(symbols: Iterable[str], col_scope: ColumnScope, ind_scope: IndicatorScope, pred_scope: PredictionScope) -> dict[str, pd.DataFrame]`: Retrieves dictionary of :class:`pandas.DataFrame`\ s containing bar data, indicator data, and model predictions for each symbol.

## `src/pybroker/slippage.py`

Implements slippage models.

### `class SlippageContext`

Context passed to slippage adjustments.

- `side: Literal['buy', 'sell']`
- `symbol: str`
- `shares: Decimal`
- `fill_price: Decimal`
- `col_scope: Optional[ColumnScope]`
- `ind_scope: Optional[IndicatorScope]`
- `sym_end_index: Optional[Mapping[str, int]]`
- `enable_fractional_shares: bool = True`

### `class SlippageModel`

Base class for implementing a slippage model.

- `is_fill_noop() -> bool`: Whether :meth:`apply_slippage` is a no-op for this model.
- `apply_slippage(ctx: SlippageContext) -> tuple[Decimal, Decimal]`: Applies slippage using data from the fill bar.
- `adjust_fill(side: Literal['buy', 'sell'], symbol: str, shares: Decimal, fill_price: Decimal, col_scope: Optional[ColumnScope]=None, ind_scope: Optional[IndicatorScope]=None, sym_end_index: Optional[Mapping[str, int]]=None, enable_fractional_shares: bool=True) -> tuple[Decimal, Decimal]`: Builds a :class:`.SlippageContext` and applies :meth:`apply_slippage` to it.
- `validate(strategy: 'Strategy') -> None`: Validates model configuration before a backtest starts.

### `class FixedSlippageModel(bps: float=5)`

Deterministic fixed-basis-point slippage on fill price.

- `is_fill_noop() -> bool`
- `adjust_fill_price(side: Literal['buy', 'sell'], fill_price: Decimal) -> Decimal`: Returns ``fill_price`` adjusted for ``side`` without extra context.
- `apply_slippage(ctx: SlippageContext) -> tuple[Decimal, Decimal]`

### `class VolatilitySlippageModel(atr_period: int=14, scale: float=0.1)`

ATR-scaled slippage on fill price.

- `atr_period: int`
- `scale: float`
- `is_fill_noop() -> bool`
- `apply_slippage(ctx: SlippageContext) -> tuple[Decimal, Decimal]`

### `class VolumeSlippageModel(price_impact: float=0.1, volume_limit: Optional[float]=0.025)`

Volume-based participation cap and square-law price impact.

- `price_impact: float`
- `volume_limit: Optional[float]`
- `is_fill_noop() -> bool`
- `validate(strategy: 'Strategy') -> None`
- `apply_slippage(ctx: SlippageContext) -> tuple[Decimal, Decimal]`

## `src/pybroker/strategy.py`

Contains implementation for backtesting trading strategies.

### `class BacktestSettings`

- `max_long_positions: Optional[int] = None`
- `max_short_positions: Optional[int] = None`
- `worst_rank_held: Optional[int] = None`

### `class Execution`

Represents an execution of a :class:`.Strategy`.

- `id: int`
- `symbols: Union[frozenset[str], SymbolSelector]`
- `fn: Optional[Callable[[ExecContext], None]]`
- `model_names: frozenset[str]`
- `indicator_names: frozenset[str]`
- `intervals: frozenset[TimeframeInterval] = frozenset()`
- `hyperparam_names: frozenset[str] = frozenset()`
- `args: tuple[Any, ...] = tuple()`
- `kwargs: tuple[tuple[str, Any], ...] = tuple()`

### `class BacktestMixin`

Mixin implementing backtesting functionality.

- `backtest_executions(config: StrategyConfig, executions: set[Execution], before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]], after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]], sessions: Mapping[str, MutableMapping], models: Mapping[ModelSymbol, TrainedModel], indicator_data: Mapping[IndicatorSymbol, pd.Series], test_data: pd.DataFrame, portfolio: Portfolio, exit_dates: Mapping[str, np.datetime64], backtest_settings: BacktestSettings=BacktestSettings(), rotation_sizer: Optional[Callable[[RotationContext], None]]=None, train_only: bool=False, slippage_model: Optional[SlippageModel]=None, enable_fractional_shares: bool=False, round_fill_price: bool=True, warmup: Optional[int]=None, interval_data: IntervalData=IntervalData(), history_col_scope: Optional[ColumnScope]=None, test_col_scope: Optional[ColumnScope]=None, run_hyperparams: Optional[dict[str, Any]]=None, pending_order_scope: Optional[PendingOrderScope]=None, master_col_scope: Optional[ColumnScope]=None) -> dict[str, pd.DataFrame]`: Backtests a ``set`` of :class:`.Execution`\ s that implement trading logic.

### `class WalkforwardWindow`

Contains train/test row indices for a walkforward window.

- `train_data: NDArray[np.int_]`
- `test_data: NDArray[np.int_]`

### `class WalkforwardMixin`

Mixin implementing logic for `Walkforward Analysis <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

- `walkforward_split(df: pd.DataFrame, windows: int, lookahead: int, train_size: float=0.9, shuffle: bool=False) -> Iterator[WalkforwardWindow]`: Splits a :class:`pandas.DataFrame` containing data for multiple ticker symbols into an :class:`Iterator` of train/test time windows for `Walkforward Analysis <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

### `class TestResult`

Contains the results of backtesting a :class:`.Strategy`.

- `start_date: datetime`
- `end_date: datetime`
- `portfolio: pd.DataFrame`
- `positions: pd.DataFrame`
- `orders: pd.DataFrame`
- `trades: pd.DataFrame`
- `metrics: EvalMetrics`
- `metrics_df: pd.DataFrame`
- `bootstrap: Optional[BootstrapResult]`
- `signals: Optional[dict[str, pd.DataFrame]]`
- `stops: Optional[pd.DataFrame]`
- `to_json(*, include: frozenset[str]=_DEFAULT_JSON_INCLUDE, max_rows: Optional[int]=100, symbols: Optional[frozenset[str]]=None) -> dict[str, Any]`: Returns JSON-serializable backtest results.
- `to_json_str(*, include: frozenset[str]=_DEFAULT_JSON_INCLUDE, max_rows: Optional[int]=100, symbols: Optional[frozenset[str]]=None) -> str`: Returns strict JSON text from :meth:`to_json`.

### `class Strategy(data_source: Union[DataSource, pd.DataFrame], start_date: Union[str, datetime], end_date: Union[str, datetime], config: Optional[StrategyConfig]=None)`

Class representing a trading strategy to backtest.

- `set_max_long_positions(max_long: StrategySetting) -> None`: Sets the maximum number of long positions held at any time.
- `set_max_short_positions(max_short: StrategySetting) -> None`: Sets the maximum number of short positions held at any time.
- `enable_rotation(worst_rank_held: StrategySetting, sizer: Optional[Callable[[RotationContext], None]]=None) -> None`: Enables rotational hold-band logic and optional custom sizing.
- `set_slippage_model(slippage_model: Optional[SlippageModel])`: Sets :class:`pybroker.slippage.SlippageModel`.
- `add_execution(fn: Optional[Callable[Concatenate[ExecContext, P], None]], symbols: Union[str, Iterable[str], SymbolSelector], models: Optional[Union[ModelSource, Iterable[ModelSource]]]=None, indicators: Optional[Union[Indicator, Iterable[Indicator]]]=None, hyperparams: Optional[Iterable[Hyperparam]]=None, intervals: Optional[Union[TimeframeInterval, Iterable[TimeframeInterval]]]=None, *args: P.args, **kwargs: P.kwargs)`: Adds an execution to backtest.
- `set_before_exec(fn: Optional[Callable[[Mapping[str, ExecContext]], None]])`: ``Callable[[Mapping[str, ExecContext]], None]`` that runs before all execution functions.
- `set_after_exec(fn: Optional[Callable[[Mapping[str, ExecContext]], None]])`: ``Callable[[Mapping[str, ExecContext]], None]`` that runs after all execution functions.
- `clear_executions()`: Clears executions that were added with :meth:`.add_execution`.
- `backtest(start_date: Optional[Union[str, datetime]]=None, end_date: Optional[Union[str, datetime]]=None, timeframe: str='', between_time: Optional[tuple[str, str]]=None, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]=None, lookahead: int=1, train_size: float=0, shuffle: bool=False, calc_bootstrap: bool=False, parallel_indicators: bool=False, parallel_models: bool=False, warmup: Optional[int]=None, portfolio: Optional[Portfolio]=None, adjust: Optional[Any]=None, seed: Optional[int]=42, params: Optional[dict[str, Any]]=None) -> TestResult`: Backtests the trading strategy by running executions that were added with :meth:`.add_execution`.
- `walkforward(windows: int, lookahead: int=1, start_date: Optional[Union[str, datetime]]=None, end_date: Optional[Union[str, datetime]]=None, timeframe: str='', between_time: Optional[tuple[str, str]]=None, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]=None, train_size: float=0.5, shuffle: bool=False, calc_bootstrap: bool=False, parallel_indicators: bool=False, parallel_models: bool=False, warmup: Optional[int]=None, portfolio: Optional[Portfolio]=None, adjust: Optional[Any]=None, seed: Optional[int]=42, params: Optional[dict[str, Any]]=None) -> TestResult`: Backtests the trading strategy using `Walkforward Analysis <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

## `src/pybroker/vect.py`

Contains vectorized utility functions.

- `lowv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]`: Calculates the lowest values for every ``n`` period in ``array``.
- `highv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]`: Calculates the highest values for every ``n`` period in ``array``.
- `sumv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]`: Calculates the sums for every ``n`` period in ``array``.
- `returnv(array: NDArray[np.float64], n: int=1) -> NDArray[np.float64]`: Calculates returns.
- `cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.bool_]`: Checks for crossover of ``a`` above ``b``.
- `normal_cdf(z: float) -> float`: Computes the CDF of the standard normal distribution.
- `inverse_normal_cdf(p: float) -> float`: Computes the inverse CDF of the standard normal distribution.
- `atr(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], lookback: int) -> NDArray[np.float64]`: Computes Average True Range (ATR).
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
- `reactivity(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], volume: NDArray[np.float64], lookback: int, smoothing: float=1.0, scale: float=0.6) -> NDArray[np.float64]`: Computes Reactivity.
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

- `proxies: Optional[dict]`
