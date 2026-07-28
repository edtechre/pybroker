"""Multi-timeframe bar compression utilities.

Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass, field
from numba import njit
from numpy.typing import NDArray
from pybroker.common import BarData, DataCol, IndicatorSymbol, to_seconds
from typing import Any, Iterable, Literal, Mapping, Optional, Union, cast

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

_RESERVED_OHLCV_COLS: frozenset[str] = frozenset(
    {
        DataCol.DATE.value,
        DataCol.OPEN.value,
        DataCol.HIGH.value,
        DataCol.LOW.value,
        DataCol.CLOSE.value,
        DataCol.VOLUME.value,
        DataCol.VWAP.value,
        DataCol.SYMBOL.value,
    }
)


@dataclass(frozen=True)
class _OhlcvArrays:
    """Zero-copy OHLCV column views extracted from a frame or BarData."""

    date: NDArray[np.datetime64]
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]
    custom: Mapping[str, NDArray[np.float64]]
    vwap: Optional[NDArray[np.float64]] = None


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
    vwap: Optional[NDArray[np.float64]] = None

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
            vwap=None if self.vwap is None else self.vwap[mask],
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


TIMEFRAME_NAME_SEPARATOR = "@"
"""Separator reserved for timeframe bindings in indicator and model names."""


def validate_source_name(name: str, kind: str) -> None:
    """Raises if ``name`` cannot be used as an indicator or model name.

    Args:
        name: Name being registered.
        kind: ``'indicator'`` or ``'model'``, used in the error message.
    """
    if TIMEFRAME_NAME_SEPARATOR in name:
        base = name.split(TIMEFRAME_NAME_SEPARATOR, 1)[0]
        raise ValueError(
            f"Invalid {kind} name {name!r}: "
            f"{TIMEFRAME_NAME_SEPARATOR!r} is reserved for timeframe "
            f"bindings, which PyBroker generates itself (e.g. {base!r} on "
            f"'weekly' becomes "
            f"{base + TIMEFRAME_NAME_SEPARATOR + 'weekly'!r}). Rename the "
            f"{kind} and read higher timeframes with "
            f"ctx.timeframe(interval)."
        )


def indicator_timeframe_name(base: str, interval: TimeframeInterval) -> str:
    """Returns the suffixed indicator name for a timeframe binding."""
    return (
        f"{base}{TIMEFRAME_NAME_SEPARATOR}"
        f"{format_timeframe_interval(interval)}"
    )


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


def _symbol_row_groups(df: pd.DataFrame) -> dict[str, NDArray[np.int64]]:
    """Returns row indices per symbol from a single sort of the symbol column.

    One ``argsort`` plus a boundary scan replaces a full-length boolean mask
    per symbol, which was ``O(n_symbols * n_rows)``.
    """
    sym_col = DataCol.SYMBOL.value
    symbols = np.asarray(df[sym_col].to_numpy(copy=False), dtype=object)
    if len(symbols) == 0:
        return {}
    order = np.argsort(symbols, kind="stable")
    ordered = symbols[order]
    # Boundaries where the sorted symbol changes.
    starts = np.flatnonzero(
        np.concatenate(([True], ordered[1:] != ordered[:-1]))
    )
    ends = np.append(starts[1:], len(ordered))
    return {
        str(ordered[start]): order[start:end]
        for start, end in zip(starts, ends)
    }


def _iter_symbol_date_groups(
    df: pd.DataFrame,
) -> Iterable[tuple[str, NDArray[np.datetime64]]]:
    """Yields per-symbol date arrays without building per-symbol frames."""
    date_col = DataCol.DATE.value
    dates = df[date_col].to_numpy(copy=False, dtype="datetime64[ns]")
    sym_col = DataCol.SYMBOL.value
    if sym_col not in df.columns:
        yield "data", dates
        return
    for sym, rows in _symbol_row_groups(df).items():
        yield sym, dates[rows]


def _extract_ohlcv_arrays(
    df: pd.DataFrame,
    extra_custom_cols: Optional[Iterable[str]] = None,
    row_mask: Optional[NDArray[np.bool_]] = None,
) -> _OhlcvArrays:
    """Extracts OHLCV column views from a frame without copying the frame."""
    if row_mask is not None:
        return _extract_ohlcv_arrays_masked(df, row_mask, extra_custom_cols)

    n = len(df)
    empty_f = np.array([], dtype=np.float64)
    empty_d = np.array([], dtype="datetime64[ns]")
    if n == 0:
        return _OhlcvArrays(
            empty_d, empty_f, empty_f, empty_f, empty_f, empty_f, {}, None
        )

    date_col = DataCol.DATE.value
    dates = df[date_col].to_numpy(copy=False, dtype="datetime64[ns]")
    open_ = df[DataCol.OPEN.value].to_numpy(copy=False, dtype=np.float64)
    high = df[DataCol.HIGH.value].to_numpy(copy=False, dtype=np.float64)
    low = df[DataCol.LOW.value].to_numpy(copy=False, dtype=np.float64)
    close = df[DataCol.CLOSE.value].to_numpy(copy=False, dtype=np.float64)
    vol_col = DataCol.VOLUME.value
    if vol_col in df.columns:
        volume = df[vol_col].to_numpy(copy=False, dtype=np.float64)
    else:
        volume = np.zeros(n, dtype=np.float64)
    vwap_col = DataCol.VWAP.value
    vwap = (
        df[vwap_col].to_numpy(copy=False, dtype=np.float64)
        if vwap_col in df.columns
        else None
    )

    custom: dict[str, NDArray[np.float64]] = {}
    if extra_custom_cols is not None:
        for col in extra_custom_cols:
            if col in df.columns:
                custom[col] = df[col].to_numpy(copy=False, dtype=np.float64)
    else:
        for col in df.columns:
            if col not in _RESERVED_OHLCV_COLS:
                custom[col] = df[col].to_numpy(copy=False, dtype=np.float64)
    return _OhlcvArrays(
        date=dates,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        custom=custom,
        vwap=vwap,
    )


def _extract_ohlcv_arrays_masked(
    df: pd.DataFrame,
    row_mask: NDArray[Any],
    extra_custom_cols: Optional[Iterable[str]] = None,
) -> _OhlcvArrays:
    """Extracts OHLCV views for selected rows without building a sub-frame.

    ``row_mask`` may be a boolean mask or an array of row indices.
    """
    empty_f = np.array([], dtype=np.float64)
    empty_d = np.array([], dtype="datetime64[ns]")
    n_rows = (
        int(row_mask.sum()) if row_mask.dtype == np.bool_ else len(row_mask)
    )
    if n_rows == 0:
        return _OhlcvArrays(
            empty_d, empty_f, empty_f, empty_f, empty_f, empty_f, {}, None
        )

    date_col = DataCol.DATE.value
    dates = df[date_col].to_numpy(copy=False, dtype="datetime64[ns]")[row_mask]
    open_ = df[DataCol.OPEN.value].to_numpy(copy=False, dtype=np.float64)[
        row_mask
    ]
    high = df[DataCol.HIGH.value].to_numpy(copy=False, dtype=np.float64)[
        row_mask
    ]
    low = df[DataCol.LOW.value].to_numpy(copy=False, dtype=np.float64)[
        row_mask
    ]
    close = df[DataCol.CLOSE.value].to_numpy(copy=False, dtype=np.float64)[
        row_mask
    ]
    vol_col = DataCol.VOLUME.value
    if vol_col in df.columns:
        volume = df[vol_col].to_numpy(copy=False, dtype=np.float64)[row_mask]
    else:
        volume = np.zeros(n_rows, dtype=np.float64)
    vwap_col = DataCol.VWAP.value
    vwap = (
        df[vwap_col].to_numpy(copy=False, dtype=np.float64)[row_mask]
        if vwap_col in df.columns
        else None
    )

    custom: dict[str, NDArray[np.float64]] = {}
    if extra_custom_cols is not None:
        for col in extra_custom_cols:
            if col in df.columns:
                custom[col] = df[col].to_numpy(copy=False, dtype=np.float64)[
                    row_mask
                ]
    return _OhlcvArrays(
        date=dates,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        custom=custom,
        vwap=vwap,
    )


def _ohlcv_from_bar_data(data: BarData) -> _OhlcvArrays:
    """Builds OHLCV arrays from BarData without an intermediate DataFrame."""
    volume = data.volume
    if volume is None:
        volume = np.zeros(len(data.date), dtype=np.float64)
    custom = {
        col: values
        for col, values in data._custom_col_data.items()
        if values is not None
    }
    return _OhlcvArrays(
        date=data.date,
        open=data.open,
        high=data.high,
        low=data.low,
        close=data.close,
        volume=volume,
        custom=custom,
        vwap=data.vwap,
    )


def symbol_dates_from_frame(
    df: pd.DataFrame,
) -> dict[str, NDArray[np.datetime64]]:
    """Extracts per-symbol test dates from a multi-symbol frame."""
    if len(df) == 0:
        return {}
    return {
        label: sym_dates for label, sym_dates in _iter_symbol_date_groups(df)
    }


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
        ].to_numpy(copy=False)
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
    """Builds a compressed-bar DataFrame with base indicator column names.

    Not used on the backtest hot path; prefer :func:`build_compressed_symbol_arrays`.
    """
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
    """Filters a compressed DataFrame to rows whose dates are in ``dates``.

    Not used on the backtest hot path; prefer :func:`slice_arrays_by_dates`.
    """
    if len(df) == 0:
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


def _base_spacing_tolerance(base_bar_seconds: float) -> float:
    """Returns the allowed deviation from an exact base-spacing multiple.

    Daily and coarser bars are commonly stamped in a local timezone and
    normalized to UTC, so a DST transition shifts them by an hour. Sub-daily
    bars keep the tight tolerance.
    """
    if base_bar_seconds >= 86400:
        return 3600.0 + _BASE_TIMEFRAME_TOLERANCE_SECONDS
    return _BASE_TIMEFRAME_TOLERANCE_SECONDS


def _validate_symbol_dates_for_base(
    label: str, dates: NDArray[np.datetime64], base_bar_seconds: float
) -> None:
    """Raises if any gap between bars is not a multiple of the base spacing.

    Gaps wider than the base spacing are expected and allowed: sessions close,
    symbols halt, illiquid feeds drop empty intervals, and ``days=`` /
    ``between_time=`` thin the frame on purpose. Only a gap *finer* than the
    declared base -- or one that is not a whole multiple of it -- contradicts
    the declared timeframe.
    """
    if len(dates) < 2:
        # A symbol with a single bar cannot contradict the declared spacing.
        return
    unique_dates = np.unique(np.asarray(dates, dtype="datetime64[ns]"))
    if len(unique_dates) < 2:
        return
    epoch_ns: NDArray[np.int64] = unique_dates.astype(np.int64)
    deltas_ns: NDArray[np.int64] = np.diff(epoch_ns)
    positive = deltas_ns[deltas_ns > 0]
    if len(positive) == 0:
        return
    gaps: NDArray[np.float64] = positive.astype(np.float64) / 1e9
    tolerance = _base_spacing_tolerance(base_bar_seconds)
    multiples = np.maximum(np.round(gaps / base_bar_seconds), 1.0)
    residuals = np.abs(gaps - multiples * base_bar_seconds)
    bad = np.flatnonzero(residuals > tolerance * multiples)
    if len(bad) == 0:
        return
    observed = float(gaps[bad[0]])
    raise ValueError(
        f"Bar spacing for {label!r} is inconsistent with base "
        f"timeframe ({int(base_bar_seconds)}s expected, "
        f"{int(observed)}s observed between consecutive bars). Gaps must be "
        "whole multiples of the base timeframe."
    )


def validate_base_timeframe_data(
    df: pd.DataFrame, base_bar_seconds: float
) -> None:
    """Raises if bar timestamps are inconsistent with ``base_bar_seconds``."""
    if len(df) == 0:
        return
    for label, dates in _iter_symbol_date_groups(df):
        _validate_symbol_dates_for_base(label, dates, base_bar_seconds)


def compressed_bars_to_bar_data(bars: CompressedBars) -> BarData:
    """Converts compressed OHLCV arrays to :class:`~pybroker.common.BarData`."""
    return BarData(
        date=bars.dates,
        open=bars.open,
        high=bars.high,
        low=bars.low,
        close=bars.close,
        volume=bars.volume,
        vwap=bars.vwap,
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


def _ascontiguous_f64(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    if arr.dtype == np.float64 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _ascontiguous_dt64(
    arr: NDArray[np.datetime64],
) -> NDArray[np.datetime64]:
    if arr.dtype == np.dtype("datetime64[ns]") and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype="datetime64[ns]")


def _empty_compressed_symbol_data() -> CompressedSymbolData:
    empty_f = np.array([], dtype=np.float64)
    empty_d = np.array([], dtype="datetime64[ns]")
    empty_i = np.array([], dtype=np.int64)
    bars = CompressedBars(
        open=empty_f,
        high=empty_f,
        low=empty_f,
        close=empty_f,
        volume=empty_f,
        dates=empty_d,
    )
    return CompressedSymbolData(
        bars=bars, completed=empty_i, base_dates=empty_d
    )


@njit(cache=True)
def _find_bin_starts_ends(
    bin_ids: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    n = len(bin_ids)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    n_bins = 1
    for i in range(1, n):
        if bin_ids[i] != bin_ids[i - 1]:
            n_bins += 1
    starts = np.empty(n_bins, dtype=np.int64)
    ends = np.empty(n_bins, dtype=np.int64)
    bin_idx = 0
    starts[0] = 0
    for i in range(1, n):
        if bin_ids[i] != bin_ids[i - 1]:
            ends[bin_idx] = i - 1
            bin_idx += 1
            starts[bin_idx] = i
    ends[bin_idx] = n - 1
    return starts, ends


@njit(cache=True)
def _aggregate_bins(
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    n_bins = len(starts)
    o = np.empty(n_bins, dtype=np.float64)
    h = np.empty(n_bins, dtype=np.float64)
    lows = np.empty(n_bins, dtype=np.float64)
    c = np.empty(n_bins, dtype=np.float64)
    v = np.empty(n_bins, dtype=np.float64)
    for i in range(n_bins):
        s = starts[i]
        e = ends[i]
        o[i] = open_[s]
        c[i] = close[e]
        hi = high[s]
        lo = low[s]
        vol_sum = volume[s]
        for j in range(s + 1, e + 1):
            if high[j] > hi:
                hi = high[j]
            if low[j] < lo:
                lo = low[j]
            vol_sum += volume[j]
        h[i] = hi
        lows[i] = lo
        v[i] = vol_sum
    return o, h, lows, c, v


@njit(cache=True)
def _aggregate_custom_cols(
    custom_2d: NDArray[np.float64],
    ends: NDArray[np.int64],
) -> NDArray[np.float64]:
    n_custom, _n_bars = custom_2d.shape
    n_bins = len(ends)
    out = np.empty((n_custom, n_bins), dtype=np.float64)
    for i in range(n_custom):
        for b in range(n_bins):
            out[i, b] = custom_2d[i, ends[b]]
    return out


def _aggregate_vwap(
    vwap: Optional[NDArray[np.float64]],
    volume: NDArray[np.float64],
    starts: NDArray[np.int64],
) -> Optional[NDArray[np.float64]]:
    """Aggregates VWAP per bin, weighted by volume.

    Falls back to the arithmetic mean for bins with no volume, matching what a
    volume-weighted average degenerates to when every weight is zero.
    """
    if vwap is None:
        return None
    if len(starts) == 0:
        return np.array([], dtype=np.float64)
    weighted = np.add.reduceat(vwap * volume, starts)
    vol_sums = np.add.reduceat(volume, starts)
    counts = np.diff(np.append(starts, len(vwap))).astype(np.float64)
    plain = np.add.reduceat(vwap, starts) / counts
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(vol_sums > 0, weighted / vol_sums, plain)
    return np.asarray(out, dtype=np.float64)


def _compute_completed(
    n: int,
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    interval_n: int,
) -> NDArray[np.int64]:
    """Maps each base bar to the last *completed* compressed bar.

    ``interval_n > 0`` is the every-``n``-bars case, where a trailing bin is
    known to be short when it holds fewer than ``n`` base bars. For calendar
    and duration bins (``interval_n <= 0``) the trailing bin's boundary is never
    observed — no later bar exists to close it — so it is always treated as
    still forming. Interior bins are unaffected: they close at their final base
    bar, matching ``ctx.close[-1]`` being the current bar's close.
    """
    completed = (
        np.searchsorted(ends, np.arange(n, dtype=np.int64), side="right") - 1
    )
    if len(ends) == 0:
        return completed
    last_bin = len(ends) - 1
    if interval_n > 0:
        last_bin_size = ends[last_bin] - starts[last_bin] + 1
        if last_bin_size >= interval_n:
            return completed
    mask = completed == last_bin
    completed = completed.copy()
    completed[mask] = last_bin - 1
    return completed


def _compress_every_n_bars(
    k: int,
    dates: NDArray[np.datetime64],
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    custom_cols: Optional[Mapping[str, NDArray[np.float64]]] = None,
    vwap: Optional[NDArray[np.float64]] = None,
) -> tuple[CompressedBars, NDArray[np.int64]]:
    n = len(dates)
    n_full = (n // k) * k
    o_parts: list[NDArray[np.float64]] = []
    h_parts: list[NDArray[np.float64]] = []
    l_parts: list[NDArray[np.float64]] = []
    c_parts: list[NDArray[np.float64]] = []
    v_parts: list[NDArray[np.float64]] = []
    d_parts: list[NDArray[np.datetime64]] = []
    ends_parts: list[NDArray[np.int64]] = []

    if n_full > 0:
        o_r = open_[:n_full].reshape(-1, k)
        h_r = high[:n_full].reshape(-1, k)
        l_r = low[:n_full].reshape(-1, k)
        c_r = close[:n_full].reshape(-1, k)
        v_r = volume[:n_full].reshape(-1, k)
        o_parts.append(o_r[:, 0])
        h_parts.append(h_r.max(axis=1))
        l_parts.append(l_r.min(axis=1))
        c_parts.append(c_r[:, -1])
        v_parts.append(v_r.sum(axis=1))
        full_ends = np.arange(k - 1, n_full, k, dtype=np.int64)
        d_parts.append(dates[full_ends])
        ends_parts.append(full_ends)

    if n_full < n:
        tail = slice(n_full, n)
        o_parts.append(np.array([open_[n_full]], dtype=np.float64))
        h_parts.append(np.array([high[tail].max()], dtype=np.float64))
        l_parts.append(np.array([low[tail].min()], dtype=np.float64))
        c_parts.append(np.array([close[n - 1]], dtype=np.float64))
        v_parts.append(np.array([volume[tail].sum()], dtype=np.float64))
        d_parts.append(np.array([dates[n - 1]], dtype="datetime64[ns]"))
        ends_parts.append(np.array([n - 1], dtype=np.int64))

    ends = np.concatenate(ends_parts)
    custom: dict[str, NDArray[np.float64]] = {}
    if custom_cols:
        for col, values in custom_cols.items():
            custom[col] = values[ends]

    idx = np.arange(n, dtype=np.int64)
    bin_idx = idx // k
    pos_in_bin = idx % k
    completed = np.where(pos_in_bin == k - 1, bin_idx, bin_idx - 1)
    if n % k != 0:
        last_bin = bin_idx[-1]
        completed[bin_idx == last_bin] = last_bin - 1

    starts = np.concatenate(
        [np.arange(0, n_full, k, dtype=np.int64)]
        + ([np.array([n_full], dtype=np.int64)] if n_full < n else [])
    )
    bars = CompressedBars(
        open=np.concatenate(o_parts),
        high=np.concatenate(h_parts),
        low=np.concatenate(l_parts),
        close=np.concatenate(c_parts),
        volume=np.concatenate(v_parts),
        dates=np.concatenate(d_parts),
        custom=custom,
        vwap=_aggregate_vwap(vwap, volume, starts),
    )
    return bars, completed


def compress(
    dates: NDArray[np.datetime64],
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    interval: TimeframeInterval,
    custom_cols: Optional[Mapping[str, NDArray[np.float64]]] = None,
    vwap: Optional[NDArray[np.float64]] = None,
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

    open_ = _ascontiguous_f64(open_)
    high = _ascontiguous_f64(high)
    low = _ascontiguous_f64(low)
    close = _ascontiguous_f64(close)
    volume = _ascontiguous_f64(volume)
    dates = _ascontiguous_dt64(dates)
    if vwap is not None:
        vwap = _ascontiguous_f64(vwap)

    if isinstance(interval, int):
        return _compress_every_n_bars(
            interval,
            dates,
            open_,
            high,
            low,
            close,
            volume,
            custom_cols,
            vwap,
        )

    bin_ids: NDArray[np.int64]
    if interval in _CALENDAR_INTERVALS:
        bin_ids = _calendar_bin_ids(dates, cast(CalendarInterval, interval))
    else:
        bin_ids = _duration_bin_ids(dates, interval)
    bin_ids = np.ascontiguousarray(bin_ids, dtype=np.int64)

    starts, ends = _find_bin_starts_ends(bin_ids)
    o, h, lows, c, v = _aggregate_bins(
        starts, ends, open_, high, low, close, volume
    )
    timeframe_dates = dates[ends]

    custom: dict[str, NDArray[np.float64]] = {}
    if custom_cols:
        col_names = tuple(custom_cols.keys())
        custom_2d = np.empty((len(col_names), n), dtype=np.float64)
        for i, col in enumerate(col_names):
            custom_2d[i] = np.ascontiguousarray(
                custom_cols[col], dtype=np.float64
            )
        aggregated = _aggregate_custom_cols(custom_2d, ends)
        for i, col in enumerate(col_names):
            custom[col] = aggregated[i]

    interval_n = interval if isinstance(interval, int) else -1
    completed = _compute_completed(n, starts, ends, interval_n)

    bars = CompressedBars(
        open=o,
        high=h,
        low=lows,
        close=c,
        volume=v,
        dates=timeframe_dates,
        custom=custom,
        vwap=_aggregate_vwap(vwap, volume, starts),
    )
    return bars, completed


def _compress_ohlcv(
    arrays: _OhlcvArrays,
    interval: TimeframeInterval,
) -> tuple[CompressedBars, NDArray[np.int64]]:
    """Compresses extracted OHLCV arrays."""
    return compress(
        dates=arrays.date,
        open_=arrays.open,
        high=arrays.high,
        low=arrays.low,
        close=arrays.close,
        volume=arrays.volume,
        interval=interval,
        custom_cols=arrays.custom,
        vwap=arrays.vwap,
    )


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
        arrays = _ohlcv_from_bar_data(data)
    else:
        if len(data) == 0:
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
        sym_col = DataCol.SYMBOL.value
        if sym_col in data.columns:
            symbols = pd.unique(data[sym_col])
            if len(symbols) > 1:
                found = ", ".join(repr(str(sym)) for sym in symbols[:5])
                raise ValueError(
                    f"compress_bars expects data for a single symbol but "
                    f"found {len(symbols)}: {found}"
                    f"{', ...' if len(symbols) > 5 else ''}. Use "
                    "compress_symbol_from_frame or "
                    "compress_timeframes_from_frame for multi-symbol frames."
                )
        validate_base_timeframe_data(data, base_bar_seconds)
        arrays = _extract_ohlcv_arrays(data)
    bars, _ = _compress_ohlcv(arrays, interval)
    return compressed_bars_to_bar_data(bars)


def compress_symbol_intervals_from_frame(
    df: pd.DataFrame,
    symbol: str,
    intervals: Iterable[TimeframeInterval],
    custom_cols: Iterable[str],
    base_bar_seconds: float,
    *,
    validate_dates: bool = True,
    rows: Optional[NDArray[np.int64]] = None,
) -> dict[TimeframeInterval, CompressedSymbolData]:
    """Compresses one symbol to multiple intervals with a single OHLCV extract.

    ``rows`` optionally supplies this symbol's precomputed row indices, so a
    caller compressing many symbols groups the frame once instead of scanning
    the symbol column per symbol.
    """
    interval_tuple = tuple(intervals)
    if rows is None:
        sym_col = DataCol.SYMBOL.value
        symbols = df[sym_col].to_numpy(copy=False)
        rows = np.flatnonzero(symbols == symbol)
    if len(rows) == 0:
        empty = _empty_compressed_symbol_data()
        return {interval: empty for interval in interval_tuple}

    arrays = _extract_ohlcv_arrays_masked(
        df, rows, extra_custom_cols=custom_cols
    )
    if validate_dates:
        _validate_symbol_dates_for_base(symbol, arrays.date, base_bar_seconds)

    result: dict[TimeframeInterval, CompressedSymbolData] = {}
    for interval in interval_tuple:
        validate_timeframe_interval(interval, base_bar_seconds)
        bars, completed = _compress_ohlcv(arrays, interval)
        result[interval] = CompressedSymbolData(
            bars=bars, completed=completed, base_dates=arrays.date
        )
    return result


def compress_timeframes_from_frame(
    df: pd.DataFrame,
    symbols: Iterable[str],
    intervals: Iterable[TimeframeInterval],
    custom_cols: Iterable[str],
    base_bar_seconds: float,
) -> TimeframeData:
    """Compresses multiple symbols and intervals from one multi-symbol frame."""
    interval_tuple = tuple(intervals)
    symbol_set = {str(sym) for sym in symbols}
    timeframe_data = TimeframeData()
    for sym_str, rows in _symbol_row_groups(df).items():
        if sym_str not in symbol_set:
            continue
        compressed = compress_symbol_intervals_from_frame(
            df,
            sym_str,
            interval_tuple,
            custom_cols,
            base_bar_seconds,
            validate_dates=False,
            rows=rows,
        )
        for interval, data in compressed.items():
            timeframe_data.compressed[(sym_str, interval)] = data
    return timeframe_data


def compress_symbol_from_frame(
    df: pd.DataFrame,
    symbol: str,
    interval: TimeframeInterval,
    custom_cols: Iterable[str],
    base_bar_seconds: float,
    *,
    validate_dates: bool = True,
) -> CompressedSymbolData:
    """Compresses one symbol from a multi-symbol frame without copying rows."""
    return compress_symbol_intervals_from_frame(
        df,
        symbol,
        (interval,),
        custom_cols,
        base_bar_seconds,
        validate_dates=validate_dates,
    )[interval]


def compress_symbol_df(
    sym_df: pd.DataFrame,
    interval: TimeframeInterval,
    custom_cols: Iterable[str],
    base_bar_seconds: float,
    *,
    validate_dates: bool = True,
) -> CompressedSymbolData:
    """Compresses a single-symbol DataFrame."""
    validate_timeframe_interval(interval, base_bar_seconds)
    if len(sym_df) == 0:
        return _empty_compressed_symbol_data()
    if validate_dates:
        validate_base_timeframe_data(sym_df, base_bar_seconds)
    arrays = _extract_ohlcv_arrays(sym_df, extra_custom_cols=custom_cols)
    bars, completed = _compress_ohlcv(arrays, interval)
    return CompressedSymbolData(
        bars=bars, completed=completed, base_dates=arrays.date
    )
