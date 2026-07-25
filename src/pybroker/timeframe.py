"""Multi-timeframe bar compression utilities.

Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass, field
from numpy.typing import NDArray
from pybroker.common import BarData, DataCol, IndicatorSymbol, to_seconds
from typing import Iterable, Literal, Mapping, Optional, Union, cast

_BASE_TIMEFRAME_TOLERANCE_SECONDS = 1.0

CalendarInterval = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]

TimeframeInterval = Union[int, CalendarInterval, str]
"""Compression interval for multi-timeframe data.

- ``int`` (``n > 1``): every ``n`` base bars (e.g. ``5``).
- ``str`` duration: digits plus one unit letter — ``"5m"``, ``"1h"``,
  ``"30s"``, ``"1d"``, or ``"1w"`` (letters: ``s``, ``m``, ``h``, ``d``, ``w``).
- ``str`` calendar: ``"daily"``, ``"weekly"``, ``"monthly"``,
  ``"quarterly"``, or ``"yearly"``.
"""

_CALENDAR_INTERVALS: frozenset[str] = frozenset(
    ("daily", "weekly", "monthly", "quarterly", "yearly")
)

_CALENDAR_INTERVAL_SECONDS: dict[str, int] = {
    "daily": 86400,
    "weekly": 7 * 86400,
    "monthly": 28 * 86400,
    "quarterly": 90 * 86400,
    "yearly": 365 * 86400,
}

_INTERVAL_HELP = (
    "use timeframe(5) for 5-bar compression, '5m'/'1h' for duration intervals "
    "(digits + unit letter), or 'weekly' for calendar weeks."
)

_DURATION_PATTERN = re.compile(r"^(\d+)([smhdw])$", re.IGNORECASE)


@dataclass(frozen=True)
class CompressedBars:
    """OHLCV and custom columns aggregated into compressed bars."""

    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]
    dates: NDArray[np.datetime64]
    custom: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)

    def slice_by_dates(
        self, dates: Iterable[np.datetime64]
    ) -> "CompressedBars":
        """Returns compressed bars restricted to ``dates``."""
        if len(self.dates) == 0:
            return self
        target = np.asarray(list(dates), dtype="datetime64[ns]")
        if len(target) == 0:
            empty_f = np.array([], dtype=np.float64)
            empty_d = np.array([], dtype="datetime64[ns]")
            return CompressedBars(
                open=empty_f,
                high=empty_f,
                low=empty_f,
                close=empty_f,
                volume=empty_f,
                dates=empty_d,
            )
        mask = np.isin(self.dates, target)
        custom = {col: values[mask] for col, values in self.custom.items()}
        return CompressedBars(
            open=self.open[mask],
            high=self.high[mask],
            low=self.low[mask],
            close=self.close[mask],
            volume=self.volume[mask],
            dates=self.dates[mask],
            custom=custom,
        )


@dataclass(frozen=True)
class CompressedSymbolData:
    """Compressed bar data and alignment map for one symbol."""

    bars: CompressedBars
    completed: NDArray[np.int64]
    base_dates: NDArray[np.datetime64]


@dataclass
class TimeframeData:
    """Compressed data keyed by ``(symbol, interval)``."""

    compressed: dict[tuple[str, TimeframeInterval], CompressedSymbolData] = (
        field(default_factory=dict)
    )

    def slice_for_test(
        self,
        test_symbol_dates: Mapping[str, NDArray[np.datetime64]],
    ) -> "TimeframeData":
        """Returns a copy with ``completed`` arrays aligned to test dates."""
        if not test_symbol_dates or not self.compressed:
            return TimeframeData()
        result: dict[tuple[str, TimeframeInterval], CompressedSymbolData] = {}
        for (symbol, interval), data in self.compressed.items():
            if symbol not in test_symbol_dates:
                continue
            test_dates = np.asarray(
                test_symbol_dates[symbol], dtype="datetime64[ns]"
            )
            if len(test_dates) == 0:
                continue
            idx = np.searchsorted(data.base_dates, test_dates)
            if not np.array_equal(data.base_dates[idx], test_dates):
                raise ValueError(
                    f"Test dates for {symbol!r} are not a subset of compressed "
                    "base history."
                )
            result[(symbol, interval)] = CompressedSymbolData(
                bars=data.bars,
                completed=data.completed[idx],
                base_dates=test_dates,
            )
        return TimeframeData(compressed=result)


def _normalize_duration_string(value: str) -> str:
    """Normalizes a duration string in ``<digits><unit>`` form (e.g. ``'5m'``)."""
    stripped = value.strip()
    if not stripped or " " in stripped:
        raise ValueError(
            f"Invalid timeframe interval {value!r}. {_INTERVAL_HELP}"
        )
    match = _DURATION_PATTERN.fullmatch(stripped)
    if not match:
        raise ValueError(
            f"Invalid timeframe interval {value!r}. {_INTERVAL_HELP}"
        )
    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError(
            f"Invalid timeframe interval {value!r}. {_INTERVAL_HELP}"
        )
    unit_letter = match.group(2).lower()
    canonical = f"{amount}{unit_letter}"
    if to_seconds(canonical) <= 0:
        raise ValueError(
            f"Invalid timeframe interval {value!r}. {_INTERVAL_HELP}"
        )
    return canonical


def _is_duration_interval(value: str) -> bool:
    if value in _CALENDAR_INTERVALS:
        return False
    try:
        _normalize_duration_string(value)
    except ValueError:
        return False
    return True


def normalize_timeframe_interval(
    interval: TimeframeInterval,
) -> TimeframeInterval:
    """Normalizes and validates a compression timeframe interval."""
    if isinstance(interval, int):
        if interval <= 1:
            raise ValueError("timeframe compression requires n > 1.")
        return interval
    if interval in _CALENDAR_INTERVALS:
        return interval
    return _normalize_duration_string(interval)


def format_timeframe_interval(interval: TimeframeInterval) -> str:
    """Returns a stable string representation of ``interval``."""
    interval = normalize_timeframe_interval(interval)
    if isinstance(interval, int):
        return str(interval)
    return interval


def indicator_timeframe_name(base: str, interval: TimeframeInterval) -> str:
    """Returns the suffixed indicator name for a timeframe binding."""
    return f"{base}@{format_timeframe_interval(interval)}"


def parse_indicator_timeframe_name(
    name: str,
) -> tuple[str, Optional[TimeframeInterval]]:
    """Parses a suffixed indicator name into base name and interval."""
    if "@" not in name:
        return name, None
    base, suffix = name.rsplit("@", 1)
    if suffix.isdigit():
        return base, int(suffix)
    if suffix in _CALENDAR_INTERVALS:
        return base, suffix  # type: ignore[return-value]
    if _is_duration_interval(suffix):
        return base, _normalize_duration_string(suffix)
    return name, None


def model_timeframe_name(base: str, interval: TimeframeInterval) -> str:
    """Returns the suffixed model name for a timeframe binding."""
    return indicator_timeframe_name(base, interval)


def parse_model_timeframe_name(
    name: str,
) -> tuple[str, Optional[TimeframeInterval]]:
    """Parses a suffixed model name into base name and interval."""
    return parse_indicator_timeframe_name(name)


def symbol_dates_from_frame(
    df: pd.DataFrame,
) -> dict[str, NDArray[np.datetime64]]:
    """Extracts per-symbol test dates from a multi-symbol frame."""
    if df.empty:
        return {}
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    symbols = df[sym_col].to_numpy()
    dates = df[date_col].to_numpy(dtype="datetime64[ns]")
    return {str(sym): dates[symbols == sym] for sym in np.unique(symbols)}


def build_compressed_symbol_arrays(
    symbol: str,
    interval: TimeframeInterval,
    compressed: CompressedSymbolData,
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicator_names: Iterable[str],
    custom_cols: Iterable[str],
) -> tuple[tuple[str, ...], dict[str, NDArray], NDArray[np.datetime64]]:
    """Builds compressed-bar column arrays with base indicator names."""
    interval = normalize_timeframe_interval(interval)
    bars = compressed.bars
    columns: list[str] = [
        DataCol.DATE.value,
        DataCol.OPEN.value,
        DataCol.HIGH.value,
        DataCol.LOW.value,
        DataCol.CLOSE.value,
        DataCol.VOLUME.value,
    ]
    arrays: dict[str, NDArray] = {
        DataCol.DATE.value: bars.dates,
        DataCol.OPEN.value: bars.open,
        DataCol.HIGH.value: bars.high,
        DataCol.LOW.value: bars.low,
        DataCol.CLOSE.value: bars.close,
        DataCol.VOLUME.value: bars.volume,
    }
    for col in custom_cols:
        if col in bars.custom:
            columns.append(col)
            arrays[col] = bars.custom[col]
    for ind_name in indicator_names:
        suffixed = indicator_timeframe_name(ind_name, interval)
        columns.append(ind_name)
        arrays[ind_name] = indicator_data[
            IndicatorSymbol(suffixed, symbol)
        ].to_numpy(copy=True)
    return tuple(columns), arrays, bars.dates


def slice_arrays_by_dates(
    columns: tuple[str, ...],
    arrays: Mapping[str, NDArray],
    dates: NDArray[np.datetime64],
    selected: Iterable[np.datetime64],
) -> tuple[tuple[str, ...], dict[str, NDArray], NDArray[np.datetime64]]:
    """Filters column arrays to rows whose dates are in ``selected``."""
    if len(dates) == 0:
        empty = np.array([], dtype=np.float64)
        return columns, {col: empty for col in columns if col in arrays}, dates
    target = np.asarray(list(selected), dtype="datetime64[ns]")
    mask = np.isin(dates, target)
    sliced = {
        col: np.asarray(arrays[col])[mask] for col in columns if col in arrays
    }
    return columns, sliced, dates[mask]


def build_compressed_symbol_df(
    symbol: str,
    interval: TimeframeInterval,
    compressed: CompressedSymbolData,
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicator_names: Iterable[str],
    custom_cols: Iterable[str],
) -> pd.DataFrame:
    """Builds a compressed-bar DataFrame with base indicator column names."""
    columns, arrays, _dates = build_compressed_symbol_arrays(
        symbol,
        interval,
        compressed,
        indicator_data,
        indicator_names,
        custom_cols,
    )
    data = {col: arrays[col] for col in columns}
    return pd.DataFrame(data)


def slice_compressed_df_by_dates(
    df: pd.DataFrame, dates: Iterable[np.datetime64]
) -> pd.DataFrame:
    """Filters a compressed DataFrame to rows whose dates are in ``dates``."""
    if df.empty:
        return df
    date_col = DataCol.DATE.value
    columns = tuple(df.columns)
    arrays = {
        col: df[col].to_numpy(copy=False)
        for col in columns
        if col in df.columns
    }
    bar_dates = arrays[date_col]
    _, sliced_arrays, sliced_dates = slice_arrays_by_dates(
        columns,
        arrays,
        bar_dates,
        dates,
    )
    data = {date_col: sliced_dates, **sliced_arrays}
    return pd.DataFrame(
        {col: data[col] for col in columns if col in data}
    ).reset_index(drop=True)


def _coarser_interval_seconds(interval: str) -> float:
    """Returns comparison seconds for a calendar or duration interval string."""
    if interval in _CALENDAR_INTERVALS:
        return float(_CALENDAR_INTERVAL_SECONDS[interval])
    return float(to_seconds(interval))


def _bar_seconds_label(seconds: float) -> str:
    if seconds >= 86400 * 365:
        return "yearly bars"
    if seconds >= 86400 * 90:
        return "quarterly bars"
    if seconds >= 86400 * 28:
        return "monthly bars"
    if seconds >= 86400 * 7:
        return "weekly bars"
    if seconds >= 86400:
        return "daily bars"
    if seconds >= 3600:
        hours = int(round(seconds / 3600))
        return f"{hours}-hour bars" if hours > 1 else "1-hour bars"
    if seconds >= 60:
        minutes = int(round(seconds / 60))
        return f"{minutes}-minute bars" if minutes > 1 else "1-minute bars"
    secs = int(round(seconds))
    return f"{secs}-second bars" if secs != 1 else "1-second bars"


def base_timeframe_to_seconds(base_timeframe: str) -> float:
    """Converts a base timeframe string to seconds."""
    if not base_timeframe or not base_timeframe.strip():
        raise ValueError("base_timeframe cannot be empty.")
    seconds = to_seconds(base_timeframe)
    if seconds <= 0:
        raise ValueError(f"Invalid base_timeframe {base_timeframe!r}.")
    return float(seconds)


def resolve_base_bar_seconds(
    base_timeframe: Optional[str],
    backtest_timeframe: str,
) -> float:
    """Resolves explicit base bar spacing from declared sources."""
    from_enable = (
        base_timeframe_to_seconds(base_timeframe) if base_timeframe else None
    )
    from_backtest = (
        base_timeframe_to_seconds(backtest_timeframe)
        if backtest_timeframe
        else None
    )
    if from_enable is not None and from_backtest is not None:
        if from_enable != from_backtest:
            raise ValueError(
                f"base_timeframe {base_timeframe!r} does not match backtest "
                f"timeframe {backtest_timeframe!r}."
            )
        return from_enable
    if from_enable is not None:
        return from_enable
    if from_backtest is not None:
        return from_backtest
    raise ValueError(
        "Multi-timeframe strategies require base_timeframe in "
        "enable_timeframes() or timeframe in backtest()/walkforward()."
    )


def _min_bar_seconds_from_dates(
    dates: NDArray[np.datetime64],
) -> Optional[float]:
    if len(dates) < 2:
        return None
    unique_dates = np.unique(dates.astype("datetime64[ns]"))
    if len(unique_dates) < 2:
        return None
    deltas_ns = np.diff(unique_dates.astype("datetime64[ns]").astype(np.int64))
    positive = deltas_ns[deltas_ns > 0]
    if len(positive) == 0:
        return None
    return float(positive.min() / 1_000_000_000)


def validate_base_timeframe_data(
    df: pd.DataFrame, base_bar_seconds: float
) -> None:
    """Raises if bar timestamps are inconsistent with ``base_bar_seconds``."""
    if df.empty:
        return
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    if sym_col in df.columns:
        symbol_groups = [
            (str(sym), group)
            for sym, group in df.groupby(sym_col, sort=False, observed=True)
        ]
    else:
        symbol_groups = [("data", df)]
    for label, group in symbol_groups:
        dates = group[date_col].to_numpy(dtype="datetime64[ns]")
        observed = _min_bar_seconds_from_dates(dates)
        if observed is None:
            raise ValueError(
                f"Need at least 2 bars to validate base timeframe for "
                f"{label!r}."
            )
        if (
            abs(observed - base_bar_seconds)
            > _BASE_TIMEFRAME_TOLERANCE_SECONDS
        ):
            raise ValueError(
                f"Bar spacing for {label!r} is inconsistent with base "
                f"timeframe ({int(base_bar_seconds)}s expected, "
                f"{int(observed)}s observed minimum spacing)."
            )


def compressed_bars_to_bar_data(bars: CompressedBars) -> BarData:
    """Converts compressed OHLCV arrays to :class:`~pybroker.common.BarData`."""
    return BarData(
        date=bars.dates,
        open=bars.open,
        high=bars.high,
        low=bars.low,
        close=bars.close,
        volume=bars.volume,
        vwap=None,
        **bars.custom,
    )


def validate_timeframe_interval(
    interval: TimeframeInterval, base_bar_seconds: float
) -> None:
    """Validates an interval against the base feed bar spacing."""
    interval = normalize_timeframe_interval(interval)
    if isinstance(interval, int):
        return
    interval_seconds = _coarser_interval_seconds(interval)
    if interval_seconds <= base_bar_seconds:
        base_label = _bar_seconds_label(base_bar_seconds)
        raise ValueError(
            f"Cannot compress {base_label} to timeframe {interval!r}. "
            "Compression only supports strictly coarser timeframes "
            "(e.g. 'weekly', '5m', timeframe(5))."
        )


def is_valid_timeframe_interval(
    interval: TimeframeInterval, base_bar_seconds: float
) -> bool:
    """Returns whether ``interval`` is valid for the base feed bar spacing."""
    try:
        validate_timeframe_interval(interval, base_bar_seconds)
    except ValueError:
        return False
    return True


def _calendar_bin_ids(
    dates: NDArray[np.datetime64], interval: CalendarInterval
) -> NDArray[np.int64]:
    if interval == "daily":
        return dates.astype("datetime64[D]").astype(np.int64)
    if interval == "weekly":
        d = dates.astype("datetime64[D]")
        return (d.astype(np.int64) + 3) // 7
    if interval == "monthly":
        return dates.astype("datetime64[M]").astype(np.int64)
    if interval == "quarterly":
        return dates.astype("datetime64[M]").astype(np.int64) // 3
    return dates.astype("datetime64[Y]").astype(np.int64)


def _duration_bin_ids(
    dates: NDArray[np.datetime64], interval: str
) -> NDArray[np.int64]:
    seconds = to_seconds(interval)
    epoch_ns = dates.astype("datetime64[ns]").astype(np.int64)
    return epoch_ns // (seconds * 1_000_000_000)


def compress(
    dates: NDArray[np.datetime64],
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    interval: TimeframeInterval,
    custom_cols: Optional[Mapping[str, NDArray[np.float64]]] = None,
) -> tuple[CompressedBars, NDArray[np.int64]]:
    """Compresses base bars into higher-timeframe bars.

    Returns compressed bars and a ``completed`` alignment map where
    ``completed[t]`` is the index of the last *completed* compressed bar at
    base bar ``t``, or ``-1`` during warmup.
    """
    interval = normalize_timeframe_interval(interval)
    n = len(dates)
    if n == 0:
        empty_f = np.array([], dtype=np.float64)
        empty_d = np.array([], dtype="datetime64[ns]")
        empty_i = np.array([], dtype=np.int64)
        return (
            CompressedBars(
                open=empty_f,
                high=empty_f,
                low=empty_f,
                close=empty_f,
                volume=empty_f,
                dates=empty_d,
            ),
            empty_i,
        )

    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    dates = np.asarray(dates, dtype="datetime64[ns]")

    bin_ids: NDArray[np.int64]
    if isinstance(interval, int):
        bin_ids = np.arange(n) // interval
    elif interval in _CALENDAR_INTERVALS:
        bin_ids = _calendar_bin_ids(dates, cast(CalendarInterval, interval))
    else:
        bin_ids = _duration_bin_ids(dates, interval)

    starts = np.flatnonzero(np.r_[True, bin_ids[1:] != bin_ids[:-1]])
    ends = np.r_[starts[1:], len(bin_ids)] - 1

    o = open_[starts]
    h = np.maximum.reduceat(high, starts)
    lows = np.minimum.reduceat(low, starts)
    c = close[ends]
    v = np.add.reduceat(volume, starts)
    timeframe_dates = dates[ends]

    custom: dict[str, NDArray[np.float64]] = {}
    if custom_cols:
        for col, values in custom_cols.items():
            arr = np.asarray(values, dtype=np.float64)
            custom[col] = arr[ends]

    completed = np.searchsorted(ends, np.arange(n), side="right") - 1

    if isinstance(interval, int) and len(ends) > 0:
        last_bin = len(ends) - 1
        last_bin_size = ends[last_bin] - starts[last_bin] + 1
        if last_bin_size < interval:
            completed = np.where(
                completed == last_bin, last_bin - 1, completed
            )

    bars = CompressedBars(
        open=o,
        high=h,
        low=lows,
        close=c,
        volume=v,
        dates=timeframe_dates,
        custom=custom,
    )
    return bars, completed.astype(np.int64)


def compress_bars(
    data: Union[BarData, pd.DataFrame],
    timeframe: TimeframeInterval,
    *,
    base_timeframe: str,
) -> BarData:
    """Compresses base OHLCV bars to a coarser ``timeframe``.

    Args:
        data: Single-symbol :class:`~pybroker.common.BarData` or OHLCV
            :class:`pandas.DataFrame`.
        timeframe: Target compression interval.
        base_timeframe: Declared base bar spacing (e.g. ``"1m"``, ``"1d"``).

    Returns:
        Compressed :class:`~pybroker.common.BarData`.
    """
    base_bar_seconds = base_timeframe_to_seconds(base_timeframe)
    interval = normalize_timeframe_interval(timeframe)
    validate_timeframe_interval(interval, base_bar_seconds)
    if isinstance(data, BarData):
        volume = data.volume
        if volume is None:
            volume = np.zeros(len(data.date), dtype=np.float64)
        frame_data: dict[str, NDArray] = {
            DataCol.DATE.value: data.date,
            DataCol.OPEN.value: data.open,
            DataCol.HIGH.value: data.high,
            DataCol.LOW.value: data.low,
            DataCol.CLOSE.value: data.close,
            DataCol.VOLUME.value: volume,
            **data._custom_col_data,
        }
        sym_df = pd.DataFrame(frame_data)
    else:
        sym_df = data.copy()
    if sym_df.empty:
        return compressed_bars_to_bar_data(
            compress(
                np.array([], dtype="datetime64[ns]"),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                interval,
            )[0]
        )
    validate_base_timeframe_data(sym_df, base_bar_seconds)
    custom_cols = [
        col
        for col in sym_df.columns
        if col
        not in (
            DataCol.DATE.value,
            DataCol.OPEN.value,
            DataCol.HIGH.value,
            DataCol.LOW.value,
            DataCol.CLOSE.value,
            DataCol.VOLUME.value,
            DataCol.SYMBOL.value,
        )
    ]
    custom_col_data = {
        col: sym_df[col].to_numpy(copy=True)
        for col in custom_cols
        if col in sym_df.columns
    }
    vol = (
        sym_df[DataCol.VOLUME.value].to_numpy(copy=True)
        if DataCol.VOLUME.value in sym_df.columns
        else np.zeros(len(sym_df), dtype=np.float64)
    )
    bars, _ = compress(
        dates=sym_df[DataCol.DATE.value].to_numpy(dtype="datetime64[ns]"),
        open_=sym_df[DataCol.OPEN.value].to_numpy(copy=True),
        high=sym_df[DataCol.HIGH.value].to_numpy(copy=True),
        low=sym_df[DataCol.LOW.value].to_numpy(copy=True),
        close=sym_df[DataCol.CLOSE.value].to_numpy(copy=True),
        volume=vol,
        interval=interval,
        custom_cols=custom_col_data,
    )
    return compressed_bars_to_bar_data(bars)


def compress_symbol_df(
    sym_df: pd.DataFrame,
    interval: TimeframeInterval,
    custom_cols: Iterable[str],
    base_bar_seconds: float,
) -> CompressedSymbolData:
    """Compresses a single-symbol DataFrame."""
    validate_timeframe_interval(interval, base_bar_seconds)
    validate_base_timeframe_data(sym_df, base_bar_seconds)
    custom_col_data = {
        col: sym_df[col].to_numpy(copy=True)
        for col in custom_cols
        if col in sym_df.columns
    }
    vol = (
        sym_df[DataCol.VOLUME.value].to_numpy(copy=True)
        if DataCol.VOLUME.value in sym_df.columns
        else np.zeros(len(sym_df), dtype=np.float64)
    )
    base_dates = sym_df[DataCol.DATE.value].to_numpy(copy=True)
    bars, completed = compress(
        dates=base_dates,
        open_=sym_df[DataCol.OPEN.value].to_numpy(copy=True),
        high=sym_df[DataCol.HIGH.value].to_numpy(copy=True),
        low=sym_df[DataCol.LOW.value].to_numpy(copy=True),
        close=sym_df[DataCol.CLOSE.value].to_numpy(copy=True),
        volume=vol,
        interval=interval,
        custom_cols=custom_col_data,
    )
    return CompressedSymbolData(
        bars=bars, completed=completed, base_dates=base_dates
    )
