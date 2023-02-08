"""Contains common classes and utilities."""

"""Copyright (C) 2023 Edward West

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from joblib import Parallel
from numpy.typing import NDArray
from typing import Any, Callable, Final, NamedTuple, Optional, Union
import numpy as np
import pandas as pd
import os
import re

_tf_pattern: Final = re.compile(r"(\d+)([A-Za-z]+)")
_tf_abbr: Final = {
    "s": "sec",
    "m": "min",
    "h": "hour",
    "d": "day",
    "w": "week",
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
    """

    name: str
    instance: Any
    predict_fn: Optional[Callable[[Any, pd.DataFrame], Any]]


class ExecSymbol(NamedTuple):
    """:class:`pybroker.strategy.Execution`/symbol identifier.

    Attributes:
        exec_id: ID of :class:`.Execution`.
        symbol: Ticker symbol.
    """

    exec_id: int
    symbol: str


class DataCol(Enum):
    """Names of default data columns."""

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


class BarData:
    r"""Contains data of a series of bars. Each field is a
    :class:`numpy.ndarray` that holds values of the bars in the series. The
    values are sorted in ascending chronological order.

    Args:
        date: Timestamps of each bar in the series.
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
        open: NDArray[np.float_],
        high: NDArray[np.float_],
        low: NDArray[np.float_],
        close: NDArray[np.float_],
        volume: Optional[NDArray[np.float_]],
        vwap: Optional[NDArray[np.float_]],
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
    date: Union[str, datetime, np.datetime64, pd.Timestamp]
) -> datetime:
    """Converts ``date`` to :class:`datetime`."""
    date_type = type(date)
    if date_type == pd.Timestamp:
        return date.to_pydatetime()  # type: ignore[union-attr]
    elif date_type == datetime:
        return date  # type: ignore[return-value]
    elif date_type == str:
        return pd.to_datetime(date).to_pydatetime()
    elif date_type == np.datetime64:
        return pd.Timestamp(date).to_pydatetime()
    else:
        raise TypeError(f"Unsupported date type: {date_type}")


def to_decimal(value: Union[int, float, Decimal]) -> Decimal:
    """Converts ``value`` to :class:`Decimal`."""
    value_type = type(value)
    if value_type == Decimal:
        return value  # type: ignore[return-value]
    elif value_type == int:
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
        value and ``str`` unit.
    """
    parts = _tf_pattern.findall(timeframe)
    if not parts or len(parts) != len(timeframe.split()):
        raise ValueError("Invalid timeframe format.")
    result = []
    units = frozenset(_tf_abbr.values())
    seen_units = set()
    for part in parts:
        unit = part[1].lower()
        if unit in _tf_abbr:
            unit = _tf_abbr[unit]
        if unit not in units:
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
    seconds = {
        "sec": 1,
        "min": 60,
        "hour": 60 * 60,
        "day": 24 * 60 * 60,
        "week": 7 * 24 * 60 * 60,
    }
    return sum(
        part[0] * seconds[part[1]] for part in parse_timeframe(timeframe)
    )


def quantize(df: pd.DataFrame, col: str) -> pd.Series:
    """Quantizes a :class:`pandas.DataFrame` column by rounding values to the
    nearest cent.

    Returns:
        The quantized column converted to ``float`` values.
    """
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not found in DataFrame.")
    df = df[~df[col].isna()]
    values = df[col].apply(lambda d: d.quantize(_CENTS, ROUND_HALF_UP))
    return values.astype(float)


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
    for col in required_cols:
        if col.value not in df.columns:
            raise ValueError(
                f"DataFrame is missing required column: {col.value!r}"
            )


def default_parallel() -> Parallel:
    """Returns a :class:`joblib.Parallel` instance with ``n_jobs`` equal to
    the number of CPUs on the host machine.
    """
    return Parallel(n_jobs=os.cpu_count(), prefer="processes", backend="loky")
