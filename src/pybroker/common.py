"""Contains common classes and utilities."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import re
import warnings
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from numpy.typing import NDArray
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Final,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    TypeGuard,
    Union,
    cast,
)

if TYPE_CHECKING:
    from pybroker.strategy import Execution

SymbolSelector = Callable[[pd.DataFrame], Sequence[str]]
"""Chooses which ticker symbols an execution trades, per walkforward window.

Passed to :meth:`pybroker.strategy.Strategy.add_execution` in place of a fixed
symbol list. Called once per walkforward window with the window's **training**
:class:`pandas.DataFrame` — never with test data — and must return a non-empty
sequence of unique symbols that the frame contains. Any sequence of ``str`` is
accepted, including a ``list``, ``tuple``, :class:`pandas.Index`, or
:class:`numpy.ndarray`, so ``ranked.nlargest(10).index`` works as written.

Notes:
    - A training window is required. :meth:`pybroker.strategy.Strategy.backtest`
      and ``train_size=0`` raise :class:`ValueError`, since selecting from test
      data would look ahead at the bars about to be traded.
    - The candidate universe must come from a :class:`pandas.DataFrame` data
      source; a :class:`pybroker.data.DataSource` cannot be used, because the
      symbols to query are not known until a window is split.
    - ``shuffle=True`` randomizes the training frame's row order, so a selector
      that depends on bars being in date order should not be combined with it.
    - When a window drops a symbol that is still held, the position is closed at
      the first bar of the new test window using
      :attr:`pybroker.config.StrategyConfig.exit_sell_fill_price` and
      :attr:`pybroker.config.StrategyConfig.exit_cover_fill_price`. A symbol that
      has no bars left at all is closed at its final bar instead.
"""


_tf_pattern: Final = re.compile(r"(\d+)([A-Za-z]+)")
_tf_abbr: Final = {
    "s": "sec",
    "m": "min",
    "h": "hour",
    "d": "day",
    "w": "week",
}
_TF_UNITS: Final = frozenset(_tf_abbr.values())
_TF_SECONDS: Final = {
    "sec": 1,
    "min": 60,
    "hour": 60 * 60,
    "day": 24 * 60 * 60,
    "week": 7 * 24 * 60 * 60,
}
_CENTS: Final = Decimal(".01")


class IndicatorSymbol(NamedTuple):
    """:class:`pybroker.indicator.Indicator`/symbol identifier.

    Attributes:
        ind_name: Indicator name.
        symbol: Ticker symbol.
    """

    ind_name: str
    symbol: str


class ModelSymbol(NamedTuple):
    """:class:`pybroker.model.ModelSource`/symbol identifier.

    Attributes:
        model_name: Model name.
        symbol: Ticker symbol.
    """

    model_name: str
    symbol: str


class TrainedModel(NamedTuple):
    """Trained model/symbol identifier.

    Attributes:
        name: Trained model name.
        instance: Trained model instance.
        predict_fn: :class:`Callable` that overrides calling the model's
            default ``predict`` function.
        input_cols: Names of the columns to be used as input for the model when
            making predictions.
        per_bar: If ``True``, predictions are made incrementally once per bar.
        lag_columns: Names of the columns that lag features were built from at
            training time, in feature block order. Reused when making
            predictions so that the lag features match the ones the model was
            trained on. ``None`` when the model was not trained with ``lags``.
    """

    name: str
    instance: Any
    predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]]
    input_cols: Optional[tuple[str]]
    per_bar: bool = False
    lag_columns: Optional[tuple[str, ...]] = None


class DataCol(Enum):
    """Default data column names."""

    DATE = "date"
    SYMBOL = "symbol"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    VWAP = "vwap"


class Day(Enum):
    """Enumeration of days."""

    MON = 0
    TUES = 1
    WEDS = 2
    THURS = 3
    FRI = 4
    SAT = 5
    SUN = 6


class PriceType(Enum):
    """Enumeration of price types used to specify fill price with
    :class:`pybroker.context.ExecContext`.

    Attributes:
        OPEN: Open price of the current bar.
        LOW: Low price of the current bar.
        HIGH: High price of the current bar.
        CLOSE: Close price of the current bar.
        MIDDLE: Midpoint between low price and high price of the current bar.
        AVERAGE: Average of open, low, high, and close prices of the current
            bar.
    """

    OPEN = "open"
    LOW = "low"
    HIGH = "high"
    CLOSE = "close"
    MIDDLE = "middle"
    AVERAGE = "average"


class StopType(Enum):
    """Stop types.

    Attributes:
        BAR: Stop that triggers after n bars.
        LOSS: Stop loss.
        PROFIT: Take profit.
        TRAILING: Trailing stop loss.
        CUSTOM: User-defined stop function.
    """

    BAR = "bar"
    LOSS = "loss"
    PROFIT = "profit"
    TRAILING = "trailing"
    CUSTOM = "custom"


class OrderType(Enum):
    """Order type classifications.

    Attributes:
        MARKET: Market order.
        LIMIT: Limit order.
        STOP_BAR: Bar stop triggered order.
        STOP_LOSS: Stop loss triggered order.
        STOP_PROFIT: Take profit triggered order.
        STOP_TRAILING: Trailing stop triggered order.
        STOP_CUSTOM: Custom stop function triggered order.
    """

    MARKET = "market"
    LIMIT = "limit"
    STOP_BAR = "stop_bar"
    STOP_LOSS = "stop_loss"
    STOP_PROFIT = "stop_profit"
    STOP_TRAILING = "stop_trailing"
    STOP_CUSTOM = "stop_custom"


class PositionIntent(Enum):
    """Position intent of an order.

    Attributes:
        BUY_TO_OPEN: Buy to open a long position.
        BUY_TO_CLOSE: Buy to close a short position.
        SELL_TO_OPEN: Sell to open a short position.
        SELL_TO_CLOSE: Sell to close a long position.
    """

    BUY_TO_OPEN = "buy_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"
    SELL_TO_CLOSE = "sell_to_close"


class FeeMode(Enum):
    """Brokerage fee mode to use for backtesting.

    Attributes:
        ORDER_PERCENT: Fee is a percentage of order amount, where order amount
            is fill_price * shares.
        PER_ORDER: Fee is a constant amount per order.
        PER_SHARE: Fee is a constant amount per share in order.
    """

    ORDER_PERCENT = "order_percent"
    PER_ORDER = "per_order"
    PER_SHARE = "per_share"


class FeeInfo(NamedTuple):
    """Contains info for custom fee calculations.

    Attributes:
        symbol: Trading symbol.
        shares: Number of shares in order.
        fill_price: Fill price of order.
        order_type: Type of order, either "buy" or "sell".
    """

    symbol: str
    shares: Decimal
    fill_price: Decimal
    order_type: Literal["buy", "sell"]


class PositionMode(Enum):
    """Position mode for backtesting.

    Attributes:
        DEFAULT: Long and short positions.
        LONG_ONLY: Long-only positions.
        SHORT_ONLY: Short-only positions.
    """

    DEFAULT = "default"
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"


class BarData:
    r"""Contains data for a series of bars. Each field is a
    :class:`numpy.ndarray` that contains bar values in the series. The values
    are sorted in ascending chronological order.

    Args:
        date: Timestamps of each bar.
        open: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Trading volumes.
        vwap: Volume-weighted average prices (VWAP).
        \**kwargs: Custom data fields.
    """

    def __init__(
        self,
        date: NDArray[np.datetime64],
        open: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        volume: Optional[NDArray[np.float64]],
        vwap: Optional[NDArray[np.float64]],
        **kwargs,
    ):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap = vwap
        self._custom_col_data = kwargs

    def __getattr__(self, attr):
        if self._custom_col_data and attr in self._custom_col_data:
            return self._custom_col_data[attr]
        raise AttributeError(f"Attribute {attr!r} not found.")


def to_datetime(
    date: Union[str, datetime, np.datetime64, pd.Timestamp],
) -> datetime:
    """Converts ``date`` to :class:`datetime`."""
    if isinstance(date, pd.Timestamp):
        return date.to_pydatetime()  # type: ignore[union-attr]
    elif isinstance(date, datetime):
        return date  # type: ignore[return-value]
    elif isinstance(date, np.datetime64):
        return pd.Timestamp(date).to_pydatetime()
    elif isinstance(date, str):
        return pd.to_datetime(date).to_pydatetime()
    else:
        raise TypeError(f"Unsupported date type: {type(date)}")


def to_decimal(value: Union[int, float, Decimal]) -> Decimal:
    """Converts ``value`` to :class:`Decimal`."""
    value_type = type(value)
    if value_type == Decimal:
        return value  # type: ignore[return-value]
    elif value_type is int:
        return Decimal(value)
    return Decimal(str(value))


def parse_timeframe(timeframe: str) -> list[tuple[int, str]]:
    """Parses timeframe string with the following units:

    - ``"s"``/``"sec"``: seconds
    - ``"m"``/``"min"``: minutes
    - ``"h"``/``"hour"``: hours
    - ``"d"``/``"day"``: days
    - ``"w"``/``"week"``: weeks

    An example timeframe string is ``1h 30m``.

    Returns:
        ``list`` of ``tuple[int, str]``, where each tuple contains an ``int``
        value and ``str`` unit of one of the following: ``sec``, ``min``,
        ``hour``, ``day``, ``week``.
    """
    parts = _tf_pattern.findall(timeframe)
    tokens = timeframe.split()
    if not parts or len(parts) != len(tokens):
        raise ValueError("Invalid timeframe format.")
    result = []
    seen_units = set()
    for part in parts:
        unit = part[1].lower()
        if unit in _tf_abbr:
            unit = _tf_abbr[unit]
        if unit not in _TF_UNITS:
            raise ValueError("Invalid timeframe format.")
        if unit in seen_units:
            raise ValueError("Invalid timeframe format.")
        result.append((int(part[0]), unit))
        seen_units.add(unit)
    return result


def to_seconds(timeframe: Optional[str]) -> int:
    """Converts a timeframe string to seconds, where ``timeframe`` supports the
    following units:

    - ``"s"``/``"sec"``: seconds
    - ``"m"``/``"min"``: minutes
    - ``"h"``/``"hour"``: hours
    - ``"d"``/``"day"``: days
    - ``"w"``/``"week"``: weeks

    An example timeframe string is ``1h 30m``.

    Returns:
        The converted number of seconds.
    """
    if not timeframe:
        return 0
    return sum(
        part[0] * _TF_SECONDS[part[1]] for part in parse_timeframe(timeframe)
    )


def quantize(df: pd.DataFrame, col: str, round: bool) -> pd.Series:
    """Quantizes a :class:`pandas.DataFrame` column by rounding values to the
    nearest cent.

    Returns:
        The quantized column converted to ``float`` values.
    """
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not found in DataFrame.")
    values = df[col].dropna()
    if not round:
        return values.astype(float)
    raw = values.to_numpy(dtype=object, copy=False)
    out = [float(val.quantize(_CENTS, ROUND_HALF_UP)) for val in raw]
    return pd.Series(out, index=values.index)


def verify_data_source_columns(df: pd.DataFrame):
    """Verifies that a :class:`pandas.DataFrame` contains all of the
    columns required by a :class:`pybroker.data.DataSource`.
    """
    required_cols = (
        DataCol.SYMBOL,
        DataCol.DATE,
        DataCol.OPEN,
        DataCol.HIGH,
        DataCol.LOW,
        DataCol.CLOSE,
    )
    missing = []
    for col in required_cols:
        if col.value not in df.columns:
            missing.append(col.value)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing!r}")


def verify_date_range(start_date: datetime, end_date: datetime):
    """Verifies date range bounds."""
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must be on or before end_date "
            f"({end_date})."
        )


def get_unique_sorted_dates_array(
    dates: Union[pd.Series, NDArray[np.datetime64], Sequence[np.datetime64]],
) -> NDArray[np.datetime64]:
    """Returns sorted unique dates from a numpy date array or Series."""
    if isinstance(dates, pd.Series):
        arr = dates.to_numpy(copy=False)
    else:
        arr = np.asarray(dates, dtype="datetime64[ns]")
    if arr.size == 0:
        return arr
    return np.unique(arr)


def get_unique_sorted_dates(col: pd.Series) -> Sequence[np.datetime64]:
    """Returns sorted unique values from a DataFrame column of dates.
    Guarantees compatability between Pandas 1 and 2.
    """
    return list(get_unique_sorted_dates_array(col))


def _json_safe(value: Any) -> Any:
    """Recursively converts a value to JSON-serializable Python types."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "_asdict"):
        return _json_safe(value._asdict())
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _dataframe_records(
    df: pd.DataFrame,
    *,
    max_rows: Optional[int] = None,
    reset_index: bool = True,
) -> list[dict[str, Any]]:
    """Converts a :class:`pandas.DataFrame` to JSON-safe record dicts."""
    if df.empty:
        return []
    out = df.reset_index() if reset_index else df
    if max_rows is not None:
        out = out.head(max_rows)
    return [_json_safe(record) for record in out.to_dict(orient="records")]


def _is_symbol_selector(
    symbols: Union[frozenset[str], SymbolSelector],
) -> TypeGuard[SymbolSelector]:
    """Returns whether ``symbols`` is a :class:`.SymbolSelector`."""
    return callable(symbols) and not isinstance(symbols, (str, bytes))


def _static_symbols(
    symbols: Union[frozenset[str], SymbolSelector],
) -> frozenset[str]:
    """Returns ``symbols`` when fixed, or an empty ``frozenset`` for a
    :class:`.SymbolSelector`, whose symbols are not known until a walkforward
    window is split."""
    if _is_symbol_selector(symbols):
        return frozenset()
    return cast(frozenset[str], symbols)


def _ensure_range_index(df: pd.DataFrame) -> pd.DataFrame:
    if (
        isinstance(df.index, pd.RangeIndex)
        and df.index.start == 0
        and df.index.step == 1
        and len(df.index) == len(df)
    ):
        return df
    return df.reset_index(drop=True)


def _selected_symbol_list(selected: Any) -> list[str]:
    """Normalizes a :class:`.SymbolSelector` return value to ``list[str]``."""
    if isinstance(selected, (str, bytes)) or not hasattr(selected, "__iter__"):
        raise TypeError(
            "symbol selector must return a sequence of symbols, "
            f"received {type(selected)!r}."
        )
    result: list[str] = []
    for sym in selected:
        if not isinstance(sym, (str, np.str_)):
            raise TypeError(
                "symbol selector must return a sequence of symbols, "
                f"received {type(sym)!r} in the returned sequence."
            )
        result.append(str(sym))
    return result


def _resolve_execution_symbols(
    execution: "Execution",
    selection_df: pd.DataFrame,
) -> frozenset[str]:
    """Resolves an :class:`pybroker.strategy.Execution`'s symbols against
    ``selection_df``, running its :class:`.SymbolSelector` when it has one."""
    if _is_symbol_selector(execution.symbols):
        selected = _selected_symbol_list(execution.symbols(selection_df))
        if not selected:
            raise ValueError("symbol selector returned an empty list.")
        if len(selected) != len(set(selected)):
            seen: set[str] = set()
            dupes = []
            for sym in selected:
                if sym in seen:
                    dupes.append(sym)
                seen.add(sym)
            raise ValueError(
                f"symbol selector returned duplicate symbols: {sorted(set(dupes))}."
            )
        loaded = set(selection_df[DataCol.SYMBOL.value].unique())
        unknown = set(selected) - loaded
        if unknown:
            raise ValueError(
                f"symbol selector returned unknown symbols: {sorted(unknown)}."
            )
        return frozenset(selected)
    return cast(frozenset[str], execution.symbols)


def _resolve_executions(
    executions: set["Execution"],
    selection_df: pd.DataFrame,
) -> set["Execution"]:
    r"""Resolves every :class:`pybroker.strategy.Execution`\ 's symbols against
    ``selection_df``, verifying that no symbol is claimed twice."""
    resolved: set["Execution"] = set()
    seen_syms: set[str] = set()
    for execution in executions:
        syms = _resolve_execution_symbols(execution, selection_df)
        overlap = seen_syms & syms
        if overlap:
            sym = sorted(overlap)[0]
            raise ValueError(f"{sym} was already added to an execution.")
        seen_syms.update(syms)
        resolved.add(execution._replace(symbols=syms))
    return resolved


def _selected_symbols(
    executions: Iterable["Execution"],
    test_data: pd.DataFrame,
    warn_unbacked: bool,
) -> set[str]:
    r"""Returns the symbols ``executions`` resolved to.

    When ``warn_unbacked``, warns about symbols that ``test_data`` holds no bars
    for. A :class:`.SymbolSelector` picks from training data, so it can name a
    symbol that stops trading at the train/test boundary; that symbol would
    otherwise run nothing and report nothing.
    """
    syms = {sym for e in executions for sym in _static_symbols(e.symbols)}
    if warn_unbacked and syms and not test_data.empty:
        unbacked = syms - set(test_data[DataCol.SYMBOL.value].unique())
        if unbacked:
            warnings.warn(
                "Selected symbols have no data in this test window and will "
                f"not be traded: {sorted(unbacked)}.",
                stacklevel=3,
            )
    return syms


def _selection_df(
    executions: Iterable["Execution"],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Returns the frame a :class:`.SymbolSelector` selects from.

    Selection always reads training data. When a window has none, there is no
    lookahead-free frame to select from, so this raises instead of silently
    handing the selector the test bars it is about to trade.
    """
    if not train_data.empty:
        return train_data
    if any(_is_symbol_selector(e.symbols) for e in executions):
        raise ValueError(
            "Dynamic symbol selection requires a training window: selecting "
            "from test data would look ahead at the bars being traded. Use "
            "walkforward(train_size=...) with train_size > 0 instead of "
            "backtest() or train_size=0."
        )
    return train_data
