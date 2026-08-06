"""Contains model related functionality."""

from __future__ import annotations

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import functools
import inspect
import warnings
import numpy as np
import pandas as pd
from numba import njit
from pybroker.cache import CacheDateFields, ModelCacheKey
from pybroker.common import (
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    TrainedModel,
    get_unique_sorted_dates,
    to_datetime,
)
from pybroker.indicator import Indicator
from pybroker.parallel import _effective_n_jobs, parallel
from pybroker.interval import (
    IntervalData,
    TimeframeInterval,
    build_compressed_symbol_arrays,
    lookahead_train_dates,
    parse_indicator_interval_name,
    parse_model_interval_name,
    format_interval,
    slice_arrays_by_dates,
    validate_source_name,
)
from dataclasses import dataclass
from datetime import datetime
from joblib import delayed
from numpy.typing import NDArray
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    cast,
)

if TYPE_CHECKING:
    from pybroker.scope import SymbolArrayStore

# --- Model input and lag helpers (formerly timeseries.py) ---

ArrayDict = dict[str, np.ndarray]


@dataclass(frozen=True)
class LagSeriesKey:
    """Internal cache key for a full-history lagged series."""

    symbol: str
    column: str
    lag: int
    interval: Optional[str] = None


LagSeriesCache = dict[LagSeriesKey, np.ndarray]


@dataclass
class ModelInput:
    """Internal numpy-backed model input with optional lag feature metadata.

    Not part of the public API. User-facing code receives
    :class:`pandas.DataFrame` instances materialized via
    :meth:`to_dataframe`, with any lag feature matrix passed explicitly as a
    separate :class:`numpy.ndarray` argument.
    """

    columns: tuple[str, ...]
    arrays: ArrayDict
    dates: np.ndarray
    lag_features: Optional[np.ndarray] = None
    lags: Optional[int] = None
    lag_columns: Optional[tuple[str, ...]] = None

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self.arrays

    def empty(self) -> bool:
        return len(self.dates) == 0

    def slice(self, end_index: Optional[int] = None) -> ModelInput:
        """Returns a row slice sharing backing array memory."""
        if end_index is None:
            return self
        sliced_arrays = {
            col: values[:end_index] for col, values in self.arrays.items()
        }
        lag_features = (
            None
            if self.lag_features is None
            else self.lag_features[:end_index]
        )
        return ModelInput(
            self.columns,
            sliced_arrays,
            self.dates[:end_index],
            lag_features,
            self.lags,
            self.lag_columns,
        )

    def slice_range(
        self, start: int, end_index: Optional[int] = None
    ) -> ModelInput:
        """Returns a row range sharing backing array memory."""
        if not start:
            return self.slice(end_index)
        rows = slice(start, end_index)
        sliced_arrays = {
            col: values[rows] for col, values in self.arrays.items()
        }
        lag_features = (
            None if self.lag_features is None else self.lag_features[rows]
        )
        return ModelInput(
            self.columns,
            sliced_arrays,
            self.dates[rows],
            lag_features,
            self.lags,
            self.lag_columns,
        )

    def lag_warmup_len(self) -> int:
        """Returns how many leading rows have undefined lag features.

        These rows cannot be handed to an estimator: sklearn rejects NaN
        outright, and a NaN-tolerant predict_fn quietly produces a prediction
        from features that do not exist yet.
        """
        if self.lag_features is None or self.empty():
            return 0
        valid = ~np.isnan(self.lag_features).any(axis=1)
        if valid.all():
            return 0
        if not valid.any():
            return len(valid)
        return int(np.argmax(valid))

    def select_columns(self, columns: tuple[str, ...]) -> ModelInput:
        """Returns a view restricted to ``columns``."""
        arrays = {
            col: self.arrays[col] for col in columns if col in self.arrays
        }
        return ModelInput(
            columns,
            arrays,
            self.dates,
            self.lag_features,
            self.lags,
            self.lag_columns,
        )

    def drop_lag_warmup(self) -> ModelInput:
        """Drops the leading rows whose lag features are not yet defined.

        Only each symbol's warmup region is trimmed. Dropping every row with a
        NaN lag feature would also remove rows from the middle of the series
        whenever a lag column is sparse -- an event column registered with
        :func:`pybroker.scope.register_columns` is NaN except on the bars it
        fires, and an all-NaN ``volume`` column is routine for index, FX and
        CFD series. Either silently shrinks the training set, or empties it.

        Pooled input stacks one block per symbol, so each block carries its own
        warmup at its own offset. Trimming only the front of the matrix would
        leave every block after the first with its NaN rows intact, which
        ``sklearn.fit`` rejects outright.
        """
        if self.lag_features is None or self.empty():
            return self
        valid = ~np.isnan(self.lag_features).any(axis=1)
        if valid.all():
            return self
        keep = self._lag_warmup_keep_mask(valid)
        if not keep.any():
            offenders = self._all_nan_lag_columns()
            detail = (
                f" Lag columns with no finite values: {offenders}."
                if offenders
                else ""
            )
            raise ValueError(
                "Lag features are undefined for every training row, so there "
                f"is nothing left to train on with lags={self.lags}."
                f"{detail}"
            )
        if keep.all():
            return self
        arrays = {col: values[keep] for col, values in self.arrays.items()}
        return ModelInput(
            self.columns,
            arrays,
            self.dates[keep],
            self.lag_features[keep],
            self.lags,
            self.lag_columns,
        )

    def _lag_warmup_keep_mask(
        self, valid: NDArray[np.bool_]
    ) -> NDArray[np.bool_]:
        """Returns a row mask dropping each symbol block's leading warmup.

        Falls back to a single block when the input is not pooled.
        """
        keep = np.zeros(len(valid), dtype=bool)
        for start, stop in self._symbol_blocks():
            block = valid[start:stop]
            if not block.any():
                continue
            first_valid = int(np.argmax(block))
            keep[start + first_valid : stop] = True
        return keep

    def _symbol_blocks(self) -> list[tuple[int, int]]:
        """Returns ``(start, stop)`` row ranges, one per symbol block.

        Pooled model input is sorted by symbol then date, so each symbol owns a
        contiguous run of rows.
        """
        n_rows = len(self.dates)
        sym_col = DataCol.SYMBOL.value
        symbols = self.arrays.get(sym_col)
        if symbols is None or not n_rows:
            return [(0, n_rows)]
        changes = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1
        bounds = [0, *changes.tolist(), n_rows]
        return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    def _all_nan_lag_columns(self) -> tuple[str, ...]:
        """Returns the lag columns that hold no finite value at all."""
        if not self.lag_columns:
            return ()
        return tuple(
            col
            for col in self.lag_columns
            if col in self.arrays and not np.isfinite(self.arrays[col]).any()
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Materializes a DataFrame of the input columns."""
        data = {
            col: self.arrays[col] for col in self.columns if col in self.arrays
        }
        return pd.DataFrame(data)


def shift_array(values: np.ndarray, lag: int) -> np.ndarray:
    """Returns ``values`` shifted forward by ``lag`` bars with NaN warmup."""
    shifted = np.empty(len(values), dtype=np.float64)
    shifted[:lag] = np.nan
    shifted[lag:] = values[:-lag]
    return shifted


def _as_float64_contiguous(values: np.ndarray) -> np.ndarray:
    """Returns a C-contiguous float64 array, copying only when needed."""
    if values.dtype == np.float64 and values.flags.c_contiguous:
        return values
    return np.ascontiguousarray(values, dtype=np.float64)


@njit(cache=True)
def _build_stacked_lags_njit(
    values: NDArray[np.float64], lags: int
) -> NDArray[np.float64]:
    n = len(values)
    stacked = np.empty((lags + 1, n), dtype=np.float64)
    stacked[0] = values
    for lag in range(1, lags + 1):
        stacked[lag, :lag] = np.nan
        stacked[lag, lag:] = values[:-lag]
    return stacked


@njit(cache=True)
def _fill_lag_feature_block_njit(
    matrix: NDArray[np.float64],
    col_start: int,
    lags: int,
    stacked: NDArray[np.float64],
    offset: int,
    n_rows: int,
) -> None:
    # Row k of ``stacked`` holds the series lagged by k bars, so the block's
    # ``lags + 1`` columns are the unlagged series followed by lags 1 through
    # ``lags``. Including row 0 makes the matrix a complete feature set that
    # models can fit and predict against directly.
    for r in range(n_rows):
        for c in range(lags + 1):
            matrix[r, col_start + c] = stacked[c, offset + r]


@njit(cache=True)
def _scatter_matrix_rows_njit(
    matrix: NDArray[np.float64],
    idx: NDArray[np.int64],
    sym_matrix: NDArray[np.float64],
) -> None:
    for i in range(len(idx)):
        matrix[idx[i]] = sym_matrix[i]


def _build_stacked_lags(values: np.ndarray, lags: int) -> np.ndarray:
    """Returns lag-expanded history with shape ``(lags + 1, len(values))``."""
    return _build_stacked_lags_njit(_as_float64_contiguous(values), lags)


def _store_stacked_lags_in_cache(
    cache: LagSeriesCache,
    symbol: str,
    col: str,
    lags: int,
    stacked: np.ndarray,
    interval: Optional[str] = None,
) -> None:
    """Stores stacked lag rows in ``cache`` under ``LagSeriesKey`` entries.

    Keeps the tallest stacked array seen for a key. A taller array serves every
    shallower request without copying, so a model with fewer lags must not
    replace the entry a model with more lags depends on.
    """
    key = LagSeriesKey(symbol, col, 0, interval)
    existing = cache.get(key)
    if (
        existing is not None
        and existing.ndim == 2
        and existing.shape[0] >= stacked.shape[0]
    ):
        stacked = existing
    else:
        cache[key] = stacked
    for lag in range(1, stacked.shape[0]):
        cache[LagSeriesKey(symbol, col, lag, interval)] = stacked[lag]


def cached_stacked_lags(
    cache: LagSeriesCache,
    symbol: str,
    col: str,
    lags: int,
    interval: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Returns a cached stacked array deep enough for ``lags``, else ``None``."""
    existing = cache.get(LagSeriesKey(symbol, col, 0, interval))
    if (
        existing is not None
        and existing.ndim == 2
        and existing.shape[0] >= lags + 1
    ):
        return existing
    return None


def _bars_column_array(bars, col: str):
    if col == DataCol.OPEN.value:
        return bars.open
    if col == DataCol.HIGH.value:
        return bars.high
    if col == DataCol.LOW.value:
        return bars.low
    if col == DataCol.CLOSE.value:
        return bars.close
    if col == DataCol.VOLUME.value:
        return bars.volume
    if col == DataCol.VWAP.value:
        return bars.vwap
    if col in bars.custom:
        return bars.custom[col]
    return None


def model_input_from_frame(
    df: pd.DataFrame,
    columns: Optional[tuple[str, ...]] = None,
    dates: Optional[np.ndarray] = None,
) -> ModelInput:
    """Builds a :class:`ModelInput` from a DataFrame without copying columns."""
    if df.empty:
        cols = columns if columns is not None else tuple(df.columns)
        arrays = {col: np.array([], dtype=np.float64) for col in cols}
        return ModelInput(cols, arrays, np.array([], dtype="datetime64[ns]"))
    date_col = DataCol.DATE.value
    if dates is None:
        dates = (
            df[date_col].to_numpy()
            if date_col in df.columns
            else np.arange(len(df), dtype=np.int64).astype("datetime64[ns]")
        )
    cols = columns if columns is not None else tuple(df.columns)
    arrays = {
        col: df[col].to_numpy(copy=False) for col in cols if col in df.columns
    }
    return ModelInput(cols, arrays, dates)


def model_input_from_arrays(
    columns: tuple[str, ...],
    arrays: ArrayDict,
    dates: np.ndarray,
) -> ModelInput:
    """Builds a :class:`ModelInput` from column arrays."""
    return ModelInput(columns, arrays, dates)


def symbol_history_arrays(
    history_df: pd.DataFrame,
    symbol: str,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, ArrayDict]:
    """Extracts sorted full-history date and column arrays for one symbol."""
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    sym_arr = history_df[sym_col].to_numpy()
    mask = sym_arr == symbol
    if not mask.any():
        return np.array([], dtype="datetime64[ns]"), {}
    rows = np.flatnonzero(mask)
    dates = history_df[date_col].to_numpy(copy=False)[rows]
    order = np.argsort(dates)
    rows = rows[order]
    dates = dates[order]
    arrays = {
        col: history_df[col].to_numpy(dtype=np.float64, copy=False)[rows]
        for col in columns
        if col in history_df.columns
    }
    for col in columns:
        if col not in arrays:
            raise ValueError(
                f"Column {col!r} not found in data for {symbol!r}."
            )
    return dates, arrays


def compute_lag_series_cache(
    df: pd.DataFrame,
    symbols: Iterable[str],
    columns: tuple[str, ...],
    lags: int,
) -> LagSeriesCache:
    """Computes full-history lag arrays for daily/base bars."""
    cache: LagSeriesCache = {}
    merge_lag_series_cache(cache, df, symbols, columns, lags)
    return cache


def merge_lag_series_cache_from_store(
    cache: LagSeriesCache,
    store: "SymbolArrayStore",
    symbols: Iterable[str],
    columns: tuple[str, ...],
    lags: int,
    history_dates: Optional[dict[str, np.ndarray]] = None,
    indicators: tuple[str, ...] = (),
    indicator_data: Optional[Mapping[IndicatorSymbol, pd.Series]] = None,
) -> LagSeriesCache:
    """Adds full-history lag arrays from a :class:`pybroker.scope.SymbolArrayStore`.

    A :class:`pybroker.scope.SymbolArrayStore` holds data columns only, so indicator values
    are aligned onto the store's dates from ``indicator_data``, which covers
    each symbol's full history. Lagging an indicator would otherwise fail even
    though it is a column of the model input.
    """
    if history_dates is None:
        history_dates = {}
    date_col = DataCol.DATE.value
    indicator_set = frozenset(indicators)
    for sym in symbols:
        if sym not in store.symbols:
            continue
        sym_data = store.sym_arrays[sym]
        dates = sym_data.get(date_col)
        if dates is None or len(dates) == 0:
            continue
        sym_dates = np.asarray(dates, dtype="datetime64[ns]")
        history_dates[sym] = sym_dates
        col_arrays: dict[str, np.ndarray] = {}
        for col in columns:
            if col in sym_data:
                col_arrays[col] = sym_data[col]
            elif col in indicator_set and indicator_data is not None:
                ind_sym = IndicatorSymbol(col, sym)
                if ind_sym in indicator_data:
                    col_arrays[col] = _indicator_values_for_dates(
                        indicator_data[ind_sym], sym_dates
                    )
        merge_lag_series_cache_from_arrays(
            cache, sym, columns, lags, history_dates[sym], col_arrays
        )
    return cache


def merge_lag_series_cache(
    cache: LagSeriesCache,
    history_df: pd.DataFrame,
    symbols: Iterable[str],
    columns: tuple[str, ...],
    lags: int,
    history_dates: Optional[dict[str, np.ndarray]] = None,
) -> LagSeriesCache:
    """Adds full-history lag arrays for ``columns`` into ``cache``."""
    if history_dates is None:
        history_dates = {}
    for sym in symbols:
        dates, col_arrays = symbol_history_arrays(history_df, sym, columns)
        if dates.size == 0:
            continue
        history_dates[sym] = dates
        merge_lag_series_cache_from_arrays(
            cache, sym, columns, lags, dates, col_arrays
        )
    return cache


def merge_lag_series_cache_from_arrays(
    cache: LagSeriesCache,
    symbol: str,
    columns: tuple[str, ...],
    lags: int,
    history_dates: np.ndarray,
    column_arrays: Mapping[str, np.ndarray],
) -> None:
    """Adds full-history lag arrays built from numpy column data."""
    del history_dates
    for col in columns:
        # Use get() so a column the caller could not resolve reports the
        # cause instead of raising a bare KeyError.
        values = column_arrays.get(col)
        if values is None:
            raise ValueError(
                f"Column {col!r} not found for {symbol!r}. lag_cols must "
                "name a data column or an Indicator registered on the model."
            )
        values = _as_float64_contiguous(values)
        stacked = _build_stacked_lags(values, lags)
        _store_stacked_lags_in_cache(cache, symbol, col, lags, stacked)


def merge_interval_lag_series_cache(
    cache: LagSeriesCache,
    symbols: Iterable[str],
    columns: tuple[str, ...],
    lags: int,
    interval: str,
    bars_by_symbol,
    arrays_by_symbol=None,
) -> LagSeriesCache:
    """Adds full-history interval lag arrays into ``cache``.

    :class:`pybroker.interval.CompressedBars` holds data columns only, so
    ``arrays_by_symbol`` supplies columns the bars cannot -- indicator values
    in particular -- and is consulted first when given.
    """
    for sym in symbols:
        bars = bars_by_symbol(sym)
        sym_arrays = (
            None if arrays_by_symbol is None else arrays_by_symbol(sym)
        )
        if bars is None and sym_arrays is None:
            continue
        for col in columns:
            if col == DataCol.DATE.value:
                continue
            col_data = None
            if sym_arrays is not None:
                col_data = sym_arrays.get(col)
            if col_data is None and bars is not None:
                col_data = _bars_column_array(bars, col)
            if col_data is None:
                raise ValueError(
                    f"Column {col!r} not found for {sym!r} on interval "
                    f"{interval!r}. lag_cols must name a data column or an "
                    "Indicator registered on the model."
                )
            values = _as_float64_contiguous(np.asarray(col_data))
            stacked = _build_stacked_lags(values, lags)
            _store_stacked_lags_in_cache(
                cache, sym, col, lags, stacked, interval
            )
    return cache


def history_date_offset(
    history_dates: np.ndarray, row_dates: np.ndarray
) -> int:
    """Returns the start index of ``row_dates`` inside ``history_dates``."""
    if row_dates.size == 0:
        return 0
    offset = int(np.searchsorted(history_dates, row_dates[0]))
    end = offset + len(row_dates)
    if end > len(history_dates):
        raise ValueError("Row dates exceed available history.")
    if not np.array_equal(history_dates[offset:end], row_dates):
        raise ValueError("Row dates are not contiguous in history.")
    return offset


def _empty_interval_train_warning(
    model_name: str,
    interval: TimeframeInterval,
    symbols_desc: str,
    lookahead: int,
    dropped: int,
) -> str:
    """Returns the warning for a train set emptied by the lookahead
    hold-out.

    Without it, the downstream symptom is unrecognizable: the pooled path
    raises a lag-column error that never mentions lookahead, and the
    per-symbol path silently hands ``train_fn`` an empty frame.
    """
    return (
        f"Model {model_name!r} on interval {format_interval(interval)!r} "
        f"has no training bars left for {symbols_desc} after holding out "
        f"lookahead={lookahead} compressed bars ({dropped} dropped). Lower "
        "lookahead, raise train_size, or use a finer interval."
    )


def _model_cache_lookahead(model_name: str, lookahead: int) -> Optional[int]:
    """Returns the ``lookahead`` to key a model cache entry with.

    Interval-bound models hold out ``lookahead`` *compressed* bars from the
    train set, so two lookaheads that yield the same base-timeframe train_end
    still fit on different data. Base-timeframe models are fully described by
    :class:`pybroker.cache.CacheDateFields`' train start/end, so their keys
    stay lookahead-free.
    """
    _, interval = parse_model_interval_name(model_name)
    return None if interval is None else lookahead


def _checked_stacked_lags(
    lag_cache: LagSeriesCache,
    symbol: str,
    col: str,
    lags: int,
    offset: int,
    n_rows: int,
    interval: Optional[str] = None,
) -> np.ndarray:
    """Returns the cached stacked lag rows for ``col``, validating bounds.

    The njit fill kernels index ``stacked`` without bounds checking, so an
    undersized cache entry would silently read out of bounds instead of
    raising. Validate here, outside the kernel's inner loop.
    """
    key = LagSeriesKey(symbol, col, 0, interval)
    stacked = lag_cache.get(key)
    interval_msg = f" on interval {interval!r}" if interval else ""
    if stacked is None:
        raise ValueError(
            f"Lag history missing for {symbol!r} column {col!r}{interval_msg}."
        )
    n_lag_rows = stacked.shape[0] if stacked.ndim == 2 else 0
    if n_lag_rows < lags + 1:
        raise ValueError(
            f"Lag history for {symbol!r} column {col!r}{interval_msg} holds "
            f"{n_lag_rows} lag rows but {lags + 1} are required."
        )
    if offset < 0 or offset + n_rows > stacked.shape[1]:
        raise ValueError(
            f"Lag history for {symbol!r} column {col!r}{interval_msg} covers "
            f"{stacked.shape[1]} bars but rows through {offset + n_rows} "
            "were requested."
        )
    return stacked


def build_lag_feature_matrix(
    symbol: str,
    columns: tuple[str, ...],
    lags: int,
    row_dates: np.ndarray,
    history_dates: np.ndarray,
    lag_cache: LagSeriesCache,
    interval: Optional[str] = None,
) -> np.ndarray:
    """Builds a lag-expanded feature matrix from numpy arrays.

    Returns a matrix of shape ``(len(row_dates), len(columns) * (lags + 1))``,
    laid out as one contiguous block per column, each block holding the
    column's current value followed by lags 1 through ``lags``.
    """
    n_rows = len(row_dates)
    n_features = len(columns) * (lags + 1)
    if n_rows == 0:
        return np.empty((0, n_features), dtype=np.float64)
    offset = history_date_offset(history_dates, row_dates)
    matrix = np.empty((n_rows, n_features), dtype=np.float64)
    col_idx = 0
    for col in columns:
        stacked = _checked_stacked_lags(
            lag_cache, symbol, col, lags, offset, n_rows, interval
        )
        _fill_lag_feature_block_njit(
            matrix,
            col_idx,
            lags,
            stacked,
            offset,
            n_rows,
        )
        col_idx += lags + 1
    return matrix


def build_lag_feature_matrix_pooled(
    sym_col: np.ndarray,
    columns: tuple[str, ...],
    lags: int,
    row_dates: np.ndarray,
    history_dates_by_symbol: dict[str, np.ndarray],
    lag_cache: LagSeriesCache,
    symbols: Iterable[str],
    interval: Optional[str] = None,
) -> np.ndarray:
    """Builds a lag-expanded feature matrix for pooled multi-symbol data."""
    n_rows = len(sym_col)
    n_features = len(columns) * (lags + 1)
    if n_rows == 0:
        return np.empty((0, n_features), dtype=np.float64)
    matrix = np.empty((n_rows, n_features), dtype=np.float64)
    sym_col_arr = np.asarray(sym_col)
    order = np.argsort(sym_col_arr, kind="stable")
    sorted_syms = sym_col_arr[order]
    unique_syms, start_indices = np.unique(sorted_syms, return_index=True)
    end_indices = np.append(start_indices[1:], len(sorted_syms))
    symbol_set = set(symbols)
    for sym, start, end in zip(unique_syms, start_indices, end_indices):
        if sym not in symbol_set:
            continue
        idx = order[start:end]
        sym_dates = row_dates[idx]
        sym_matrix = build_lag_feature_matrix(
            sym,
            columns,
            lags,
            sym_dates,
            history_dates_by_symbol[sym],
            lag_cache,
            interval,
        )
        _scatter_matrix_rows_njit(matrix, idx.astype(np.int64), sym_matrix)
    return matrix


def apply_lags_to_model_input(
    model_input: ModelInput,
    lag_columns: tuple[str, ...],
    lags: int,
    lag_cache: LagSeriesCache,
    symbol: str,
    history_dates: np.ndarray,
    interval: Optional[str] = None,
) -> ModelInput:
    """Attaches lag feature metadata to ``model_input``."""
    n_features = len(lag_columns) * (lags + 1)
    if model_input.empty():
        model_input.lag_features = np.empty((0, n_features), dtype=np.float64)
        model_input.lags = lags
        model_input.lag_columns = lag_columns
        return model_input
    matrix = build_lag_feature_matrix(
        symbol,
        lag_columns,
        lags,
        model_input.dates,
        history_dates,
        lag_cache,
        interval,
    )
    model_input.lag_features = matrix
    model_input.lags = lags
    model_input.lag_columns = lag_columns
    return model_input


def apply_lags_to_model_input_pooled(
    model_input: ModelInput,
    lag_columns: tuple[str, ...],
    lags: int,
    lag_cache: LagSeriesCache,
    history_dates_by_symbol: dict[str, np.ndarray],
    symbols: Iterable[str],
    interval: Optional[str] = None,
) -> ModelInput:
    """Attaches lag feature metadata to pooled ``model_input``."""
    n_features = len(lag_columns) * (lags + 1)
    if model_input.empty():
        model_input.lag_features = np.empty((0, n_features), dtype=np.float64)
        model_input.lags = lags
        model_input.lag_columns = lag_columns
        return model_input
    sym_col = model_input.arrays[DataCol.SYMBOL.value]
    matrix = build_lag_feature_matrix_pooled(
        sym_col,
        lag_columns,
        lags,
        model_input.dates,
        history_dates_by_symbol,
        lag_cache,
        symbols,
        interval,
    )
    model_input.lag_features = matrix
    model_input.lags = lags
    model_input.lag_columns = lag_columns
    return model_input


def apply_prepare_input_data(
    model_input: ModelInput,
    prepare_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> ModelInput:
    """Applies a DataFrame-only prepare function to ``model_input``."""
    n_rows = len(model_input.dates)
    df = prepare_fn(model_input.to_dataframe())
    if len(df.columns) and len(df) != n_rows:
        raise ValueError(
            f"input_data_fn returned {len(df)} rows for {n_rows} bars. "
            "Model input must stay aligned one row per bar; return NaN "
            "warmup rows instead of dropping them, or use lags= which "
            "drops warmup rows from training data only."
        )
    result = model_input_from_frame(
        df, columns=tuple(df.columns), dates=model_input.dates
    )
    result.lag_features = model_input.lag_features
    result.lags = model_input.lags
    result.lag_columns = model_input.lag_columns
    return result


from pybroker.scope import (
    StaticScope,
    SymbolArrayStore,
    merge_symbol_array_stores,
    run_with_scope,
    symbol_array_store_from_frame,
)


@njit(cache=True)
def _indicator_values_for_dates_njit(
    ind_dates: NDArray,
    values: NDArray[np.float64],
    dates: NDArray,
) -> NDArray[np.float64]:
    """Aligns indicator values to ``dates`` via batched sorted datetime search."""
    n = len(dates)
    result = np.full(n, np.nan, dtype=np.float64)
    pos = np.searchsorted(ind_dates, dates)
    m = len(ind_dates)
    for i in range(n):
        p = pos[i]
        if p < m and ind_dates[p] == dates[i]:
            result[i] = values[p]
    return result


def _indicator_values_for_dates(
    ind_series: pd.Series, dates: np.ndarray
) -> NDArray[np.float64]:
    """Aligns indicator values to ``dates`` without pandas indexing."""
    if len(dates) == 0:
        return np.array([], dtype=np.float64)
    values = ind_series.to_numpy(dtype=np.float64, copy=False)
    index = ind_series.index
    if getattr(index, "is_monotonic_increasing", False):
        ind_dates = index.to_numpy(dtype="datetime64[ns]", copy=False)
        pos = np.searchsorted(ind_dates, dates)
        valid = pos < len(ind_dates)
        matched = np.zeros(len(dates), dtype=bool)
        matched[valid] = ind_dates[pos[valid]] == dates[valid]
        result = np.full(len(dates), np.nan, dtype=np.float64)
        result[matched] = values[pos[matched]]
        return result
    positions = ind_series.index.get_indexer(dates)
    result = np.full(len(dates), np.nan, dtype=np.float64)
    valid = positions >= 0
    result[valid] = values[positions[valid]]
    return result


def _model_input_columns(
    indicators: tuple[str, ...],
    available: frozenset[str],
    *,
    pooled: bool = False,
) -> tuple[str, ...]:
    """Returns ordered model input columns present in ``available``."""
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    scope = StaticScope.instance()
    columns: list[str] = [sym_col, date_col] if pooled else [date_col]
    for col in scope.ordered_data_cols:
        if col in (sym_col, date_col) or col not in available:
            continue
        columns.append(col)
    for ind_name in indicators:
        columns.append(ind_name)
    return tuple(dict.fromkeys(columns))


def _empty_model_input(
    columns: tuple[str, ...], *, pooled: bool
) -> ModelInput:
    sym_col = DataCol.SYMBOL.value
    arrays = {col: np.array([], dtype=np.float64) for col in columns}
    if pooled and sym_col in arrays:
        arrays[sym_col] = np.array([], dtype=object)
    return model_input_from_arrays(
        columns, arrays, np.array([], dtype="datetime64[ns]")
    )


def _symbol_model_input_from_store(
    store: SymbolArrayStore,
    symbol: str,
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicators: tuple[str, ...],
) -> ModelInput:
    """Builds per-symbol :class:`ModelInput` from a :class:`pybroker.scope.SymbolArrayStore`."""
    date_col = DataCol.DATE.value
    if symbol not in store.sym_arrays:
        available: frozenset[str] = frozenset()
        for sym_data in store.sym_arrays.values():
            available |= frozenset(sym_data.keys())
        columns = _model_input_columns(indicators, available, pooled=False)
        return _empty_model_input(columns, pooled=False)
    sym_data = store.sym_arrays[symbol]
    available = frozenset(sym_data.keys())
    columns_tuple = _model_input_columns(indicators, available, pooled=False)
    dates_arr = sym_data.get(date_col)
    if dates_arr is None or len(dates_arr) == 0:
        return _empty_model_input(columns_tuple, pooled=False)
    dates = np.asarray(dates_arr, dtype="datetime64[ns]")
    arrays: dict[str, NDArray] = {date_col: dates}
    for col in columns_tuple:
        if col == date_col:
            continue
        if col in sym_data:
            arrays[col] = sym_data[col]
        elif col in indicators:
            arrays[col] = _indicator_values_for_dates(
                indicator_data[IndicatorSymbol(col, symbol)], dates
            )
    return model_input_from_arrays(columns_tuple, arrays, dates)


def _pooled_model_input_from_store(
    store: SymbolArrayStore,
    symbols: frozenset[str],
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicators: tuple[str, ...],
) -> ModelInput:
    """Builds pooled multi-symbol :class:`ModelInput` from a store."""
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    available: set[str] = set()
    for sym in symbols:
        if sym in store.sym_arrays:
            available.update(store.sym_arrays[sym].keys())
    columns_tuple = _model_input_columns(
        indicators, frozenset(available), pooled=True
    )
    sym_parts: list[NDArray] = []
    date_parts: list[NDArray] = []
    col_parts: dict[str, list[NDArray]] = {
        col: [] for col in columns_tuple if col not in (sym_col, date_col)
    }
    has_rows = False
    for sym in symbols:
        if sym not in store.sym_arrays:
            continue
        sym_data = store.sym_arrays[sym]
        dates_arr = sym_data.get(date_col)
        if dates_arr is None or len(dates_arr) == 0:
            continue
        has_rows = True
        dates = np.asarray(dates_arr, dtype="datetime64[ns]")
        n = len(dates)
        sym_parts.append(np.full(n, sym, dtype=object))
        date_parts.append(dates)
        for col in col_parts:
            if col in sym_data:
                col_parts[col].append(sym_data[col])
            elif col in indicators:
                col_parts[col].append(
                    _indicator_values_for_dates(
                        indicator_data[IndicatorSymbol(col, sym)], dates
                    )
                )
    if not has_rows:
        return _empty_model_input(columns_tuple, pooled=True)
    sym_vals = np.concatenate(sym_parts)
    dates = np.concatenate(date_parts)
    order = np.lexsort((dates, sym_vals))
    sym_vals = sym_vals[order]
    dates = dates[order]
    arrays: dict[str, NDArray] = {sym_col: sym_vals, date_col: dates}
    for col, parts in col_parts.items():
        arrays[col] = np.concatenate(parts)[order]
    return model_input_from_arrays(columns_tuple, arrays, dates)


def _symbol_model_input(
    symbol: str,
    df: pd.DataFrame,
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicators: tuple[str, ...],
) -> ModelInput:
    """Builds per-symbol :class:`ModelInput` via boolean masks."""
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    if df.empty:
        return ModelInput((), {}, np.array([], dtype="datetime64[ns]"))
    available = frozenset(df.columns)
    columns_tuple = _model_input_columns(indicators, available, pooled=False)
    sym_arr = df[sym_col].to_numpy()
    mask = sym_arr == symbol
    if not mask.any():
        return _empty_model_input(columns_tuple, pooled=False)
    rows = np.flatnonzero(mask)
    dates = df[date_col].to_numpy(copy=False)[rows]
    order = np.argsort(dates)
    rows = rows[order]
    dates = dates[order]
    arrays: dict[str, NDArray] = {date_col: dates}
    for col in columns_tuple:
        if col == date_col:
            continue
        if col in df.columns:
            arrays[col] = df[col].to_numpy(copy=False)[rows]
        elif col in indicators:
            arrays[col] = _indicator_values_for_dates(
                indicator_data[IndicatorSymbol(col, symbol)], dates
            )
    return model_input_from_arrays(columns_tuple, arrays, dates)


def _pooled_model_input(
    df: pd.DataFrame,
    symbols: frozenset[str],
    indicator_data: Mapping[IndicatorSymbol, pd.Series],
    indicators: tuple[str, ...],
) -> ModelInput:
    """Builds pooled multi-symbol :class:`ModelInput` without frame copies."""
    sym_col = DataCol.SYMBOL.value
    date_col = DataCol.DATE.value
    if df.empty:
        available: frozenset[str] = frozenset()
        columns = _model_input_columns(indicators, available, pooled=True)
        return _empty_model_input(columns, pooled=True)
    available = frozenset(df.columns)
    columns_tuple = _model_input_columns(indicators, available, pooled=True)
    sym_arr = df[sym_col].to_numpy()
    mask = np.isin(sym_arr, tuple(symbols))
    if not mask.any():
        return _empty_model_input(columns_tuple, pooled=True)
    rows = np.flatnonzero(mask)
    dates = df[date_col].to_numpy(copy=False)[rows]
    sym_vals = sym_arr[rows]
    order = np.lexsort((dates, sym_vals))
    rows = rows[order]
    dates = dates[order]
    sym_vals = sym_vals[order]
    arrays: dict[str, NDArray] = {sym_col: sym_vals, date_col: dates}
    for col in columns_tuple:
        if col in (sym_col, date_col):
            continue
        if col in df.columns:
            arrays[col] = df[col].to_numpy(copy=False)[rows]
        elif col in indicators:
            ind_values = np.empty(len(rows), dtype=np.float64)
            for sym in symbols:
                sym_mask = sym_vals == sym
                if not sym_mask.any():
                    continue
                ind_values[sym_mask] = _indicator_values_for_dates(
                    indicator_data[IndicatorSymbol(col, sym)],
                    dates[sym_mask],
                )
            arrays[col] = ind_values
    return model_input_from_arrays(columns_tuple, arrays, dates)


def _history_store(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *,
    train_store: Optional[SymbolArrayStore] = None,
    test_store: Optional[SymbolArrayStore] = None,
) -> SymbolArrayStore:
    if train_store is not None and test_store is not None:
        if train_data.empty:
            return test_store
        if test_data.empty:
            return train_store
        return merge_symbol_array_stores(train_store, test_store)
    if train_store is not None:
        if test_data.empty:
            return train_store
        return merge_symbol_array_stores(
            train_store, symbol_array_store_from_frame(test_data)
        )
    if test_store is not None:
        if train_data.empty:
            return test_store
        return merge_symbol_array_stores(
            symbol_array_store_from_frame(train_data), test_store
        )
    if train_data.empty:
        return symbol_array_store_from_frame(test_data)
    if test_data.empty:
        return symbol_array_store_from_frame(train_data)
    return merge_symbol_array_stores(
        symbol_array_store_from_frame(train_data),
        symbol_array_store_from_frame(test_data),
    )


class ModelSource:
    r"""Base class of a model source. A model source provides a model instance
    either by training one or by loading a pre-trained model.

    Args:
        name: Name of model.
        indicator_names: :class:`Iterable` of names of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: ``Callable[[DataFrame], DataFrame]`` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data.
        predict_fn: ``Callable[[Model, DataFrame], ndarray]`` that
            overrides calling the model's default ``predict`` function. If
            set, ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data. When ``lags``
            is set, it is instead called with the lag feature matrix
            (:class:`numpy.ndarray`) in place of the DataFrame.
        lags: Number of lagged values to include for each column in
            ``lag_cols``, producing a ``(n_rows, len(lag_cols) * (lags + 1))``
            feature matrix that is passed to ``train_fn`` as
            ``lag_train``/``lag_test`` and to ``predict_fn`` in place of the
            input DataFrame, rather than added as columns.
        lag_cols: Columns to compute lagged values for. Defaults to the data
            columns of the training data, excluding ``date`` and ``symbol``;
            indicators are lagged only when named here.
        per_bar: If ``True``, ``predict_fn`` is called once per bar with input
            truncated to rows up to and including the current bar.
        pooled: If ``True``, the model is trained once per execution using
            combined multi-symbol data. Defaults to ``False``.
        kwargs: ``dict`` of additional kwargs.
    """

    def __init__(
        self,
        name: str,
        indicator_names: Iterable[str],
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        predict_fn: Optional[
            Callable[[Any, Union[pd.DataFrame, NDArray]], NDArray]
        ],
        pooled: bool,
        kwargs: dict[str, Any],
        lags: Optional[int] = None,
        lag_cols: tuple[str, ...] = (),
        per_bar: bool = False,
    ):
        self.name = name
        self.indicators = tuple(indicator_names)
        self._input_data_fn = input_data_fn
        self._predict_fn = predict_fn
        self.lags = lags
        self.lag_cols = tuple(lag_cols)
        self.per_bar = per_bar
        self.pooled = pooled
        self._kwargs = kwargs

    def prepare_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares a :class:`pandas.DataFrame` of input data for passing to a
        model when making predictions. If set, the ``input_data_fn``
        is used to preprocess the input data. If ``False``, then indicator
        columns in ``df`` are used as input features.
        """
        if df.empty:
            return df
        if self._input_data_fn is None:
            df_cols = frozenset(df.columns)
            for ind_name in self.indicators:
                if ind_name not in df_cols:
                    raise ValueError(
                        f"Indicator {ind_name!r} not found in DataFrame."
                    )
            return df[[*self.indicators]]
        return self._input_data_fn(df)


class ModelLoader(ModelSource):
    r"""Loads a pre-trained model.

    Args:
        name: Name of model.
        load_fn: ``Callable[[symbol: str, train_start_date: datetime,
            train_end_date: datetime, ...], DataFrame]`` used to load and
            return a pre-trained model. This is expected to
            return either a trained model instance, or a tuple containing a
            trained model instance and a :class:`Iterable` of column names to
            to be used as input for the model when making predictions.
        indicator_names: :class:`Iterable` of names of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: ``Callable[[DataFrame], DataFrame]`` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data.
        predict_fn: ``Callable[[Model, DataFrame], ndarray]`` that
            overrides calling the model's default ``predict`` function. If
            set, ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data. When ``lags``
            is set, it is instead called with the lag feature matrix
            (:class:`numpy.ndarray`) in place of the DataFrame.
        pooled: If ``True``, the model is trained once per execution using
            combined multi-symbol data. Defaults to ``False``.
        kwargs: ``dict`` of kwargs to pass to ``load_fn``.
    """

    def __init__(
        self,
        name: str,
        load_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]],
        indicator_names: Iterable[str],
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        predict_fn: Optional[
            Callable[[Any, Union[pd.DataFrame, NDArray]], NDArray]
        ],
        pooled: bool,
        kwargs: dict[str, Any],
        lags: Optional[int] = None,
        lag_cols: tuple[str, ...] = (),
        per_bar: bool = False,
    ):
        super().__init__(
            name,
            indicator_names,
            input_data_fn,
            predict_fn,
            pooled,
            kwargs,
            lags=lags,
            lag_cols=lag_cols,
            per_bar=per_bar,
        )
        self._load_fn = functools.partial(load_fn, **kwargs)

    def __call__(
        self, symbol: str, train_start_date: datetime, train_end_date: datetime
    ) -> Union[Any, tuple[Any, Iterable[str]]]:
        """Loads pre-trained model.

        Args:
            symbol: Ticker symbol for loading the pre-trained model.
            train_start_date: Start date of training window.
            train_end_date: End date of training window.

        Returns:
            Pre-trained model.
        """
        return self._load_fn(symbol, train_start_date, train_end_date)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelLoader({self.name!r}, {self._kwargs})"


class ModelTrainer(ModelSource):
    r"""Trains a model.

    Args:
        name: Name of model.
        train_fn: When ``pooled`` is ``False``, ``Callable[[symbol: str,
            train_data: DataFrame, test_data: DataFrame, ...], DataFrame]``.
            When ``pooled`` is ``True``, ``Callable[[symbols: Sequence[str],
            train_data: DataFrame, test_data: DataFrame, ...], DataFrame]``.
            When ``lags`` is set,
            ``train_fn`` is additionally called with ``lag_train=`` and
            ``lag_test=`` keyword arguments holding the lag feature matrices
            aligned one row per ``train_data``/``test_data`` row, and must
            accept both. This is expected to
            return either a trained model instance, or a tuple containing a
            trained model instance and a :class:`Iterable` of column names to
            to be used as input for the model when making predictions.
        indicator_names: :class:`Iterable` of names of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        input_data_fn: ``Callable[[DataFrame], DataFrame]`` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data.
        predict_fn: ``Callable[[Model, DataFrame], ndarray]`` that
            overrides calling the model's default ``predict`` function. If
            set, ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data. When ``lags``
            is set, it is instead called with the lag feature matrix
            (:class:`numpy.ndarray`) in place of the DataFrame.
        pooled: If ``True``, the model is trained once per execution using
            combined multi-symbol data. Defaults to ``False``.
        kwargs: ``dict`` of kwargs to pass to ``train_fn``.
    """

    def __init__(
        self,
        name: str,
        train_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]],
        indicator_names: Iterable[str],
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        predict_fn: Optional[
            Callable[[Any, Union[pd.DataFrame, NDArray]], NDArray]
        ],
        pooled: bool,
        kwargs: dict[str, Any],
        lags: Optional[int] = None,
        lag_cols: tuple[str, ...] = (),
        per_bar: bool = False,
    ):
        super().__init__(
            name,
            indicator_names,
            input_data_fn,
            predict_fn,
            pooled,
            kwargs,
            lags=lags,
            lag_cols=lag_cols,
            per_bar=per_bar,
        )
        self._train_fn = functools.partial(train_fn, **kwargs)

    def __call__(
        self,
        symbol: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        *,
        lag_train: Optional[NDArray] = None,
        lag_test: Optional[NDArray] = None,
    ) -> Union[Any, tuple[Any, Iterable[str]]]:
        """Trains model per symbol.

        Args:
            symbol: Ticker symbol of model (models are trained per symbol).
            train_data: Train data.
            test_data: Test data.
            lag_train: Lag feature matrix aligned one row per ``train_data``
                row. Passed to ``train_fn`` as ``lag_train=`` when the model
                is registered with ``lags``.
            lag_test: Lag feature matrix aligned one row per ``test_data``
                row. Passed to ``train_fn`` as ``lag_test=`` when the model
                is registered with ``lags``.

        Returns:
            Trained model.
        """
        if self.lags is None:
            return self._train_fn(symbol, train_data, test_data)
        return self._train_fn(
            symbol,
            train_data,
            test_data,
            lag_train=lag_train,
            lag_test=lag_test,
        )

    def train_pooled(
        self,
        symbols: Sequence[str],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        *,
        lag_train: Optional[NDArray] = None,
        lag_test: Optional[NDArray] = None,
    ) -> Union[Any, tuple[Any, Iterable[str]]]:
        """Trains model using combined multi-symbol data.

        Args:
            symbols: Ticker symbols of the pooled group, sorted in ascending
                order to match the order that symbol blocks appear in
                ``train_data`` and ``test_data``. A listed symbol can have no
                rows in a frame, such as when ``lags`` drops all of its rows
                with the lag warmup.
            train_data: Train data containing a ``symbol`` column.
            test_data: Test data containing a ``symbol`` column.
            lag_train: Lag feature matrix aligned one row per ``train_data``
                row. Passed to ``train_fn`` as ``lag_train=`` when the model
                is registered with ``lags``.
            lag_test: Lag feature matrix aligned one row per ``test_data``
                row. Passed to ``train_fn`` as ``lag_test=`` when the model
                is registered with ``lags``.

        Returns:
            Trained model.
        """
        if self.lags is None:
            return self._train_fn(symbols, train_data, test_data)
        return self._train_fn(
            symbols,
            train_data,
            test_data,
            lag_train=lag_train,
            lag_test=lag_test,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelTrainer({self.name!r}, {self._kwargs})"


def _validate_lagged_train_fn(
    name: str, fn: Callable, kwargs: Mapping[str, Any]
):
    """Validates that a lagged ``train_fn`` accepts lag matrix kwargs."""
    for reserved in ("lag_train", "lag_test"):
        if reserved in kwargs:
            raise ValueError(
                f"Model {name!r}: {reserved!r} is reserved for the lag "
                "feature matrix passed to train_fn and cannot be used as a "
                "model kwarg."
            )
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return
    params = sig.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return
    accepted = {
        p.name
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if "lag_train" not in accepted or "lag_test" not in accepted:
        raise ValueError(
            f"Model {name!r} is registered with lags= but its train_fn does "
            "not accept the lag feature matrices. Expected a signature like "
            "train_fn(symbol, train_data, test_data, lag_train, lag_test)"
            " (pooled models take symbols instead of symbol)."
        )


def _parse_lag_cols(
    lag_cols: Optional[Iterable[Union[str, Indicator]]],
    lags: Optional[int],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalizes ``lag_cols`` to column names and implied indicator names."""
    if lag_cols is None:
        return tuple(), tuple()
    if lags is None:
        raise ValueError("lag_cols requires lags to be set, e.g. lags=3.")
    reserved = (DataCol.DATE.value, DataCol.SYMBOL.value)
    names: list[str] = []
    ind_names: list[str] = []
    for col in lag_cols:
        if isinstance(col, Indicator):
            # Declaring an Indicator here also schedules it for computation.
            ind_names.append(col.name)
            name = col.name
        elif isinstance(col, str):
            name = col
        else:
            raise ValueError(
                "lag_cols must contain column names or Indicators, got "
                f"{type(col).__name__}."
            )
        if not name:
            raise ValueError("lag_cols cannot contain an empty column name.")
        if name in reserved:
            raise ValueError(
                f"lag_cols cannot contain reserved column {name!r}."
            )
        names.append(name)
    if not names:
        raise ValueError("lag_cols cannot be empty.")
    # Declaration order sets the feature block order, so dedupe in place.
    return tuple(dict.fromkeys(names)), tuple(dict.fromkeys(ind_names))


def model(
    name: str,
    fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]],
    indicators: Optional[Iterable[Indicator]] = None,
    lags: Optional[int] = None,
    lag_cols: Optional[Iterable[Union[str, Indicator]]] = None,
    per_bar: bool = False,
    input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    predict_fn: Optional[
        Callable[[Any, Union[pd.DataFrame, NDArray]], NDArray]
    ] = None,
    pretrained: bool = False,
    pooled: bool = False,
    **kwargs,
) -> ModelSource:
    r"""Creates a :class:`.ModelSource` instance and registers it globally with
    ``name``.

    Args:
        name: Name for referencing the model globally.
        fn: :class:`Callable` used to either train or load a model instance. If
            for training with ``pooled=False``, then ``fn`` has signature
            ``Callable[[symbol: str, train_data: DataFrame, test_data:
            DataFrame, ...], DataFrame]``. If for training with
            ``pooled=True``, then ``fn`` has signature ``Callable[[symbols:
            Sequence[str], train_data: DataFrame, test_data: DataFrame, ...],
            DataFrame]`` where ``symbols`` contains the pooled symbols sorted
            in ascending order, and both frames contain a ``symbol`` column
            with each symbol's rows grouped together in that order. A listed
            symbol can have no rows in a frame, such as when ``lags`` drops
            all of its rows with the lag warmup. If for loading, then ``fn``
            has signature
            ``Callable[[symbol: str, train_start_date: datetime,
            train_end_date: datetime, ...], DataFrame]``. When ``lags`` is
            set, a training ``fn`` is additionally called with ``lag_train=``
            and ``lag_test=`` keyword arguments holding the prebuilt lag
            feature matrices, and must accept both. This is expected to
            return either a trained model instance, or a tuple containing a
            trained model instance and a :class:`Iterable` of column names to
            to be used as input for the model when making predictions. When
            only a model instance is returned, columns from the training
            DataFrame are used for prediction. For pooled models, the
            ``symbol`` column is omitted from inferred prediction columns.
        indicators: :class:`Iterable` of
            :class:`pybroker.indicator.Indicator`\ s used as features of the
            model.
        lags: Number of lagged values to include for each column in
            ``lag_cols``. The lagged values are built into a feature matrix
            of shape ``(n_rows, len(lag_cols) * (lags + 1))``: one contiguous
            block per column in ``lag_cols`` declaration order, each block
            holding the column's current value followed by lags ``1`` through
            ``lags``, where lag ``1`` is the value from the previous bar. The
            matrix is passed to a training ``fn`` as the ``lag_train`` and
            ``lag_test`` keyword arguments, aligned one row per
            ``train_data``/``test_data`` row, and to ``predict_fn`` (or the
            model's default ``predict``) in place of the input DataFrame.
            Because the current bar's value is the first feature of each
            block, the intended prediction target is the *next* bar — using
            the current bar's value as the target would leak it. Lag data is
            kept separate from model input rather than added as columns, so
            the input :class:`pandas.DataFrame` is never widened or copied.
            Lagged values are computed from each symbol's full history, so
            rows at the start of a test window use real values carried over
            from the preceding train window instead of ``NaN``. Rows whose
            lags are undefined are dropped from training data only.
        lag_cols: Column names and/or
            :class:`pybroker.indicator.Indicator`\ s to compute lagged values
            for. :class:`pybroker.indicator.Indicator`\ s passed here are added
            to ``indicators``. Declaration order sets the order of the feature
            blocks. Defaults to the data columns of the training data,
            excluding ``date`` and ``symbol``; indicators are lagged only when
            named here.
        per_bar: If ``True``, ``predict_fn`` is called once per bar with input
            truncated to rows up to and including the current bar, and must
            return a scalar prediction for that bar. With ``lags``, the input
            is the lag feature matrix truncated the same way, with the
            current bar as its last row. Use for models that are
            refit or updated every bar, such as GARCH or state space models.
            Note this makes one model call per bar per symbol, which is far
            slower than a single vectorized ``predict_fn`` call over the whole
            test window. Requires ``predict_fn`` and is not supported with
            ``pooled=True``.
        input_data_fn: ``Callable[[DataFrame], DataFrame]`` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data, including when
            ``per_bar=True``. It must return one row per bar; adding or
            dropping rows would misalign predictions with bars and raises a
            :class:`ValueError`. For models registered with ``lags``, it
            shapes :meth:`pybroker.context.ExecContext.input` only, since
            predictions are made from the lag feature matrix.
        predict_fn: ``Callable[[Model, DataFrame], ndarray]`` that
            overrides calling the model's default ``predict`` function. If set,
            ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data. When ``lags``
            is set, it is instead called with the lag feature matrix
            (:class:`numpy.ndarray`) in place of the DataFrame. When
            ``per_bar=True``, ``predict_fn`` receives input rows up to and
            including the current bar and must return a scalar prediction.
        pretrained: If ``True``, then ``fn`` is used to load and return a
            pre-trained model. If ``False``, ``fn`` is used to train and return
            a new model. Defaults to ``False``.
        pooled: If ``True``, the model is trained once per execution using
            combined multi-symbol data. Defaults to ``False``.
        \**kwargs: Additional arguments to pass to ``fn``.

    Returns:
        :class:`.ModelSource` instance.
    """
    if lags is not None:
        if not isinstance(lags, int) or lags <= 0:
            raise ValueError("lags must be a positive integer.")
        if not pretrained:
            _validate_lagged_train_fn(name, fn, kwargs)
    if per_bar and pooled:
        raise ValueError("per_bar=True is not supported with pooled=True.")
    if per_bar and predict_fn is None:
        raise ValueError("per_bar=True requires predict_fn to be set.")
    validate_source_name(name, "model")
    scope = StaticScope.instance()
    lag_col_names, lag_col_inds = _parse_lag_cols(lag_cols, lags)
    indicator_names = tuple(
        sorted(
            (
                set(ind.name for ind in indicators)
                if indicators is not None
                else set()
            )
            | set(lag_col_inds)
        )
    )
    if pretrained:
        loader = ModelLoader(
            name=name,
            load_fn=fn,
            indicator_names=indicator_names,
            input_data_fn=input_data_fn,
            predict_fn=predict_fn,
            pooled=pooled,
            kwargs=kwargs,
            lags=lags,
            lag_cols=lag_col_names,
            per_bar=per_bar,
        )
        scope.set_model_source(loader)
        return loader
    else:
        trainer = ModelTrainer(
            name=name,
            train_fn=fn,
            indicator_names=indicator_names,
            input_data_fn=input_data_fn,
            predict_fn=predict_fn,
            pooled=pooled,
            kwargs=kwargs,
            lags=lags,
            lag_cols=lag_col_names,
            per_bar=per_bar,
        )
        scope.set_model_source(trainer)
        return trainer


class CachedModel(NamedTuple):
    """Stores cached model data.

    Attributes:
        input_cols: Names of the columns to be used as input for the model when
            making predictions.
        lag_columns: Names of the columns that lag features were built from at
            training time, in feature block order. ``None`` when the model was
            not trained with ``lags``, and when loading a model cached before
            this field existed.
    """

    model: Any
    """Trained model instance."""

    input_cols: Optional[tuple[str]]
    lag_columns: Optional[tuple[str, ...]] = None


class _TrainerTask(NamedTuple):
    pooled: bool
    source: ModelTrainer
    model_name: str
    symbols: frozenset[str]
    model_sym: Optional[ModelSymbol]
    train_data: ModelInput
    test_data: ModelInput


PooledTrainResult = tuple[str, frozenset[str], Any, Optional[tuple[str]]]
SymTrainResult = tuple[ModelSymbol, Any, Optional[tuple[str]]]
PooledTrainerReturn = tuple[Literal["pooled"], PooledTrainResult]
SymTrainerReturn = tuple[Literal["sym"], SymTrainResult]
TrainerReturn = Union[PooledTrainerReturn, SymTrainerReturn]


def _infer_input_cols(
    train_data: ModelInput, pooled: bool, indicators: tuple[str, ...]
) -> tuple[str, ...]:
    # Columns registered with register_columns() are part of the training
    # frame, so they belong in the inferred input columns too. Filtering on
    # the DataCol enum instead would drop them here while the prediction side
    # keeps them, feeding the model fewer features than it trained on.
    data_cols = StaticScope.instance().all_data_cols
    cols = [
        col
        for col in train_data.columns
        if col in data_cols or col in indicators
    ]
    if pooled:
        symbol_col = DataCol.SYMBOL.value
        cols = [col for col in cols if col != symbol_col]
    return tuple(cols)


def _lag_feature_cols(
    train_data: ModelInput,
    pooled: bool,
    indicators: tuple[str, ...],
    lag_cols: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Resolves the columns to build lag features from.

    This is the authoritative resolver: the result is recorded on
    :attr:`pybroker.common.TrainedModel.lag_columns` and reused when making
    predictions, so that lag features are built the same way at both ends.

    Without ``lag_cols`` this infers data columns only. Indicators are
    typically engineered features that a model does not want lagged, and
    including them by default makes the feature matrix grow with every
    registered indicator. Name an indicator in ``lag_cols`` to lag it.
    """
    if lag_cols:
        missing = tuple(col for col in lag_cols if col not in train_data)
        if missing:
            available = sorted(train_data.columns)
            raise ValueError(
                f"Column {missing[0]!r} in lag_cols not found in model input. "
                f"Available columns: {available}."
            )
        return lag_cols
    date_col = DataCol.DATE.value
    indicator_set = frozenset(indicators)
    return tuple(
        col
        for col in _infer_input_cols(train_data, pooled, indicators)
        if col != date_col and col not in indicator_set
    )


def _parse_model_result(
    model_result: Union[Any, tuple[Any, Iterable[str]]],
    train_data: ModelInput,
    pooled: bool,
    indicators: tuple[str, ...],
) -> tuple[Any, Optional[tuple[str]]]:
    if isinstance(model_result, tuple):
        model = model_result[0]
        input_cols = cast(tuple[str], tuple(model_result[1]))
    else:
        model = model_result
        input_cols = cast(
            tuple[str], _infer_input_cols(train_data, pooled, indicators)
        )
    return model, input_cols


def _train_model_sym(
    source: ModelTrainer,
    model_sym: ModelSymbol,
    sym_train_data: ModelInput,
    sym_test_data: ModelInput,
) -> SymTrainResult:
    model_name, sym = model_sym
    model_result = source(
        sym,
        sym_train_data.to_dataframe(),
        sym_test_data.to_dataframe(),
        lag_train=sym_train_data.lag_features,
        lag_test=sym_test_data.lag_features,
    )
    model, input_cols = _parse_model_result(
        model_result,
        sym_train_data,
        pooled=False,
        indicators=source.indicators,
    )
    return model_sym, model, input_cols


def _train_model_pooled(
    source: ModelTrainer,
    model_name: str,
    symbols: frozenset[str],
    pooled_train_data: ModelInput,
    pooled_test_data: ModelInput,
) -> PooledTrainResult:
    model_result = source.train_pooled(
        tuple(sorted(symbols)),
        pooled_train_data.to_dataframe(),
        pooled_test_data.to_dataframe(),
        lag_train=pooled_train_data.lag_features,
        lag_test=pooled_test_data.lag_features,
    )
    model, input_cols = _parse_model_result(
        model_result,
        pooled_train_data,
        pooled=True,
        indicators=source.indicators,
    )
    return model_name, symbols, model, input_cols


def _run_trainer_task(task: _TrainerTask) -> TrainerReturn:
    if task.pooled:
        return (
            "pooled",
            _train_model_pooled(
                task.source,
                task.model_name,
                task.symbols,
                task.train_data,
                task.test_data,
            ),
        )
    assert task.model_sym is not None
    return (
        "sym",
        _train_model_sym(
            task.source,
            task.model_sym,
            task.train_data,
            task.test_data,
        ),
    )


class ModelsMixin:
    """Mixin implementing model related functionality."""

    def train_models(
        self,
        model_syms: Iterable[ModelSymbol],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        cache_date_fields: CacheDateFields,
        parallel_models: bool = False,
        pooled_model_groups: Optional[
            Mapping[tuple[str, int], frozenset[str]]
        ] = None,
        interval_data: Optional[IntervalData] = None,
        *,
        history_store: Optional[SymbolArrayStore] = None,
        train_store: Optional[SymbolArrayStore] = None,
        test_store: Optional[SymbolArrayStore] = None,
        lookahead: int = 1,
    ) -> dict[ModelSymbol, TrainedModel]:
        """Trains models for the provided :class:`pybroker.common.ModelSymbol`
        pairs.

        Args:
            model_syms: ``Iterable`` of
                :class:`pybroker.common.ModelSymbol` pairs of models to train.
            train_data: :class:`pandas.DataFrame` of training data.
            test_data: :class:`pandas.DataFrame` of test data.
            indicator_data: ``Mapping`` of
                :class:`pybroker.common.IndicatorSymbol` pairs to
                ``pandas.Series`` of :class:`pybroker.indicator.Indicator`
                values.
            cache_date_fields: Date fields used to key cache data.
            parallel_models: If ``True``, :class:`.ModelTrainer` models
                are trained in parallel using multiple processes. Defaults to
                ``False``.
            pooled_model_groups: ``Mapping`` of ``(model_name, execution_id)``
                pairs to ``frozenset[str]`` of symbols for pooled training.
                Defaults to ``None``.
            lookahead: Number of bars in the future of the target prediction,
                expressed in the bars of the timeframe each model is fitted
                on: a model bound to an interval holds out ``lookahead``
                compressed bars between its train and test rows. Defaults to
                ``1``.

        Returns:
            ``dict`` mapping each :class:`pybroker.common.ModelSymbol` pair
            to a :class:`pybroker.common.TrainedModel`.
        """
        if train_data.empty or not model_syms:
            return {}
        if train_store is None and not train_data.empty:
            train_store = symbol_array_store_from_frame(train_data)
        if test_store is None and not test_data.empty:
            test_store = symbol_array_store_from_frame(test_data)
        resolved_history_store = history_store

        def get_history_store() -> SymbolArrayStore:
            """Returns the merged train+test store, building it on first use.

            Only models with ``lags`` read it, and callers that already hold
            the merge pass it in, so neither pays for a second full
            concatenation of every symbol column.
            """
            nonlocal resolved_history_store
            if resolved_history_store is None:
                resolved_history_store = _history_store(
                    train_data,
                    test_data,
                    train_store=train_store,
                    test_store=test_store,
                )
            return resolved_history_store

        lag_series_cache: LagSeriesCache = {}
        history_dates: dict[str, np.ndarray] = {}
        if pooled_model_groups is None:
            pooled_model_groups = {}
        scope = StaticScope.instance()
        train_dates = get_unique_sorted_dates(train_data[DataCol.DATE.value])
        test_dates = get_unique_sorted_dates(test_data[DataCol.DATE.value])
        scope.logger.train_split_start(train_dates)
        scope.logger.info_train_split_start(model_syms)
        models, uncached_model_syms = self._get_cached_models(
            model_syms, cache_date_fields, pooled_model_groups, lookahead
        )
        if not uncached_model_syms and not self._has_uncached_pooled_groups(
            model_syms, models, pooled_model_groups
        ):
            scope.logger.loaded_models()
            scope.logger.info_loaded_models(model_syms)
            return models
        if models:
            scope.logger.info_loaded_models(models.keys())
        start_date = to_datetime(train_dates[0])
        end_date = to_datetime(train_dates[-1])
        uncached_model_sym_set = set(uncached_model_syms)
        trainer_tasks: list[_TrainerTask] = []
        loader_syms: list[tuple[ModelLoader, ModelSymbol]] = []
        covered_pooled_model_syms: set[ModelSymbol] = set()

        for (model_name, _), symbols in pooled_model_groups.items():
            group_model_syms = {
                ModelSymbol(model_name, sym) for sym in symbols
            }
            if group_model_syms.issubset(models.keys()):
                continue
            if not group_model_syms & uncached_model_sym_set:
                continue
            base_name, interval = parse_model_interval_name(model_name)
            source = scope.get_model_source(base_name)
            if not isinstance(source, ModelTrainer) or not source.pooled:
                raise TypeError(
                    f"ModelSource {model_name!r} is not a pooled ModelTrainer."
                )
            if interval is not None:
                if interval_data is None:
                    raise ValueError(
                        f"Timeframe data required to train model {model_name!r}."
                    )
                pooled_train_data, pooled_test_data = (
                    self._prepare_pooled_interval_data(
                        symbols,
                        interval,
                        train_dates,
                        test_dates,
                        indicator_data,
                        source,
                        interval_data,
                        lag_series_cache,
                        lookahead,
                    )
                )
            else:
                pooled_train_data, pooled_test_data = (
                    self._prepare_pooled_data(
                        symbols,
                        train_data,
                        test_data,
                        indicator_data,
                        source,
                        train_dates,
                        test_dates,
                        get_history_store,
                        lag_series_cache,
                        history_dates,
                        train_store=train_store,
                        test_store=test_store,
                    )
                )
            trainer_tasks.append(
                _TrainerTask(
                    pooled=True,
                    source=source,
                    model_name=model_name,
                    symbols=symbols,
                    model_sym=None,
                    train_data=pooled_train_data,
                    test_data=pooled_test_data,
                )
            )
            covered_pooled_model_syms.update(group_model_syms)

        for model_sym in uncached_model_syms:
            if model_sym in models or model_sym in covered_pooled_model_syms:
                continue
            model_name, sym = model_sym
            base_name, interval = parse_model_interval_name(model_name)
            source = scope.get_model_source(base_name)
            if interval is not None:
                if isinstance(source, ModelLoader):
                    raise ValueError(
                        f"Pretrained model {base_name!r} does not support "
                        f"multi-interval training on {interval!r}."
                    )
                if interval_data is None:
                    raise ValueError(
                        f"Timeframe data required to train model {model_name!r}."
                    )
                sym_train_data, sym_test_data = (
                    self._prepare_interval_symbol_data(
                        sym,
                        interval,
                        train_dates,
                        test_dates,
                        indicator_data,
                        source,
                        interval_data,
                        lag_series_cache,
                        lookahead,
                    )
                )
            elif isinstance(source, ModelTrainer):
                if source.pooled:
                    continue
                if train_store is not None:
                    sym_train_data = _symbol_model_input_from_store(
                        train_store, sym, indicator_data, source.indicators
                    )
                else:
                    sym_train_data = _symbol_model_input(
                        sym, train_data, indicator_data, source.indicators
                    )
                if test_store is not None:
                    sym_test_data = _symbol_model_input_from_store(
                        test_store, sym, indicator_data, source.indicators
                    )
                else:
                    sym_test_data = _symbol_model_input(
                        sym, test_data, indicator_data, source.indicators
                    )
                if source.lags is not None:
                    lag_cols = _lag_feature_cols(
                        sym_train_data,
                        pooled=False,
                        indicators=source.indicators,
                        lag_cols=source.lag_cols,
                    )
                    merge_lag_series_cache_from_store(
                        lag_series_cache,
                        get_history_store(),
                        (sym,),
                        lag_cols,
                        source.lags,
                        history_dates,
                        source.indicators,
                        indicator_data,
                    )
                    apply_lags_to_model_input(
                        sym_train_data,
                        lag_cols,
                        source.lags,
                        lag_series_cache,
                        sym,
                        history_dates[sym],
                    )
                    apply_lags_to_model_input(
                        sym_test_data,
                        lag_cols,
                        source.lags,
                        lag_series_cache,
                        sym,
                        history_dates[sym],
                    )
                    sym_train_data = sym_train_data.drop_lag_warmup()
            else:
                sym_train_data = ModelInput(
                    (), {}, np.array([], dtype="datetime64[ns]")
                )
                sym_test_data = ModelInput(
                    (), {}, np.array([], dtype="datetime64[ns]")
                )
            if isinstance(source, ModelTrainer):
                trainer_tasks.append(
                    _TrainerTask(
                        pooled=False,
                        source=source,
                        model_name=model_name,
                        symbols=frozenset(),
                        model_sym=model_sym,
                        train_data=sym_train_data,
                        test_data=sym_test_data,
                    )
                )
            elif isinstance(source, ModelLoader):
                if interval is not None:
                    raise ValueError(
                        f"Pretrained model {base_name!r} does not support "
                        f"multi-interval training on {interval!r}."
                    )
                loader_syms.append((source, model_sym))
            else:
                raise TypeError(f"Invalid ModelSource type: {type(source)}")

        trainer_results = self._run_model_trainers(
            trainer_tasks, parallel_models
        )
        for task, trainer_result in zip(trainer_tasks, trainer_results):
            if trainer_result[0] == "pooled":
                _, pooled_result = cast(PooledTrainerReturn, trainer_result)
                model_name, symbols, model, input_cols = pooled_result
                # Recorded so prediction builds lag features from the same
                # columns the model was trained on.
                lag_columns = task.train_data.lag_columns
                for sym in symbols:
                    model_sym = ModelSymbol(model_name, sym)
                    scope.logger.info_train_model_start(model_sym)
                    models[model_sym] = TrainedModel(
                        name=model_name,
                        instance=model,
                        predict_fn=task.source._predict_fn,
                        input_cols=input_cols,
                        per_bar=task.source.per_bar,
                        lag_columns=lag_columns,
                    )
                    self._set_cached_model(
                        model,
                        input_cols,
                        model_sym,
                        cache_date_fields,
                        lag_columns,
                        pooled_symbols=frozenset(symbols),
                        lookahead=lookahead,
                    )
                    scope.logger.info_train_model_completed(model_sym)
            else:
                _, sym_result = cast(SymTrainerReturn, trainer_result)
                model_sym, model, input_cols = sym_result
                model_name, _ = model_sym
                lag_columns = task.train_data.lag_columns
                scope.logger.info_train_model_start(model_sym)
                models[model_sym] = TrainedModel(
                    name=model_name,
                    instance=model,
                    predict_fn=task.source._predict_fn,
                    input_cols=input_cols,
                    per_bar=task.source.per_bar,
                    lag_columns=lag_columns,
                )
                self._set_cached_model(
                    model,
                    input_cols,
                    model_sym,
                    cache_date_fields,
                    lag_columns,
                    lookahead=lookahead,
                )
                scope.logger.info_train_model_completed(model_sym)
        for source, model_sym in loader_syms:
            model_name, sym = model_sym
            scope.logger.info_loaded_model(model_sym)
            model_result = source(sym, start_date, end_date)
            input_cols = None
            if isinstance(model_result, tuple):
                model = model_result[0]
                input_cols = tuple(model_result[1])
            else:
                model = model_result
            # A loader declaring lag_cols must lag exactly those columns. Left
            # unset, the lag matrix is built from whatever columns load_fn
            # returned -- every OHLCV column, say -- so predict_fn receives a
            # differently shaped matrix than the model was fitted on.
            lag_columns = (
                tuple(source.lag_cols)
                if source.lags is not None and source.lag_cols
                else None
            )
            models[model_sym] = TrainedModel(
                name=model_name,
                instance=model,
                predict_fn=source._predict_fn,
                input_cols=input_cols,
                per_bar=source.per_bar,
                lag_columns=lag_columns,
            )
            self._set_cached_model(
                model,
                input_cols,
                model_sym,
                cache_date_fields,
                lag_columns=lag_columns,
                lookahead=lookahead,
            )
        scope.logger.train_split_completed()
        return models

    def _has_uncached_pooled_groups(
        self,
        model_syms: Iterable[ModelSymbol],
        models: Mapping[ModelSymbol, TrainedModel],
        pooled_model_groups: Mapping[tuple[str, int], frozenset[str]],
    ) -> bool:
        uncached_model_sym_set = set(model_syms) - set(models.keys())
        for (model_name, _), symbols in pooled_model_groups.items():
            group_model_syms = {
                ModelSymbol(model_name, sym) for sym in symbols
            }
            if group_model_syms & uncached_model_sym_set:
                return True
        return False

    def _run_model_trainers(
        self,
        trainer_tasks: Collection[_TrainerTask],
        parallel_models: bool,
    ) -> list[TrainerReturn]:
        if (
            parallel_models
            and len(trainer_tasks) > 1
            and _effective_n_jobs() > 1
        ):
            # Workers start with an empty StaticScope, so ship the caller's
            # along with each task: a train_fn reading pybroker.param() must
            # see the same values it would see running sequentially.
            scope = StaticScope.instance()
            with parallel() as pool:
                return pool(
                    delayed(run_with_scope)(scope, _run_trainer_task, task)
                    for task in trainer_tasks
                )
        return [_run_trainer_task(task) for task in trainer_tasks]

    def _prepare_pooled_data(
        self,
        symbols: frozenset[str],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        source: ModelTrainer,
        train_dates: Collection,
        test_dates: Collection,
        get_history_store: Callable[[], SymbolArrayStore],
        lag_series_cache: LagSeriesCache,
        history_dates: dict[str, np.ndarray],
        *,
        train_store: Optional[SymbolArrayStore] = None,
        test_store: Optional[SymbolArrayStore] = None,
    ) -> tuple[ModelInput, ModelInput]:
        del train_dates, test_dates
        if train_store is not None:
            pooled_train_input = _pooled_model_input_from_store(
                train_store, symbols, indicator_data, source.indicators
            )
        else:
            pooled_train_input = _pooled_model_input(
                train_data, symbols, indicator_data, source.indicators
            )
        if test_store is not None:
            pooled_test_input = _pooled_model_input_from_store(
                test_store, symbols, indicator_data, source.indicators
            )
        else:
            pooled_test_input = _pooled_model_input(
                test_data, symbols, indicator_data, source.indicators
            )
        if source.lags is not None:
            lag_cols = _lag_feature_cols(
                pooled_train_input,
                pooled=True,
                indicators=source.indicators,
                lag_cols=source.lag_cols,
            )
            merge_lag_series_cache_from_store(
                lag_series_cache,
                get_history_store(),
                symbols,
                lag_cols,
                source.lags,
                history_dates,
                source.indicators,
                indicator_data,
            )
            apply_lags_to_model_input_pooled(
                pooled_train_input,
                lag_cols,
                source.lags,
                lag_series_cache,
                history_dates,
                symbols,
            )
            apply_lags_to_model_input_pooled(
                pooled_test_input,
                lag_cols,
                source.lags,
                lag_series_cache,
                history_dates,
                symbols,
            )
            pooled_train_input = pooled_train_input.drop_lag_warmup()
        return pooled_train_input, pooled_test_input

    def _prepare_pooled_interval_data(
        self,
        symbols: frozenset[str],
        interval: TimeframeInterval,
        train_dates: Collection,
        test_dates: Collection,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        source: ModelTrainer,
        interval_data: IntervalData,
        lag_series_cache: LagSeriesCache,
        lookahead: int = 1,
    ) -> tuple[ModelInput, ModelInput]:
        sym_col = DataCol.SYMBOL.value
        scope = StaticScope.instance()
        train_parts: dict[str, list[NDArray]] = {}
        test_parts: dict[str, list[NDArray]] = {}
        columns: tuple[str, ...] = ()
        history_dates: dict[str, np.ndarray] = {}
        full_arrays: dict[str, ArrayDict] = {}
        total_dropped = 0
        for sym in symbols:
            key = (sym, interval)
            if key not in interval_data.compressed:
                raise ValueError(
                    f"Interval {interval!r} data not found for {sym!r}."
                )
            compressed = interval_data.compressed[key]
            sym_columns, arrays, bar_dates = build_compressed_symbol_arrays(
                sym,
                interval,
                compressed,
                indicator_data,
                source.indicators,
                sorted(scope.custom_data_cols),
            )
            columns = sym_columns + (sym_col,)
            history_dates[sym] = np.asarray(bar_dates, dtype="datetime64[ns]")
            # Retained for the lag cache: these hold full-history indicator
            # values, which compressed bars do not carry.
            full_arrays[sym] = arrays
            # Per symbol, not hoisted: each symbol has its own compressed bar
            # history and therefore its own first test compressed index.
            effective_train_dates, dropped = lookahead_train_dates(
                bar_dates, train_dates, test_dates, lookahead
            )
            total_dropped += dropped
            _, train_arrays, train_dates_arr = slice_arrays_by_dates(
                sym_columns,
                arrays,
                bar_dates,
                effective_train_dates,
            )
            _, test_arrays, test_dates_arr = slice_arrays_by_dates(
                sym_columns,
                arrays,
                bar_dates,
                test_dates,
            )
            if train_dates_arr.size:
                for col in sym_columns:
                    train_parts.setdefault(col, []).append(train_arrays[col])
                train_parts.setdefault(sym_col, []).append(
                    np.full(train_dates_arr.size, sym)
                )
            if test_dates_arr.size:
                for col in sym_columns:
                    test_parts.setdefault(col, []).append(test_arrays[col])
                test_parts.setdefault(sym_col, []).append(
                    np.full(test_dates_arr.size, sym)
                )

        def _concat_pooled(
            parts: dict[str, list[NDArray]],
        ) -> ModelInput:
            if not parts or sym_col not in parts:
                return ModelInput((), {}, np.array([], dtype="datetime64[ns]"))
            date_col = DataCol.DATE.value
            pooled: dict[str, NDArray] = {
                col: np.concatenate(arrs) for col, arrs in parts.items()
            }
            order = np.lexsort((pooled[date_col], pooled[sym_col]))
            for col in pooled:
                pooled[col] = pooled[col][order]
            return model_input_from_arrays(columns, pooled, pooled[date_col])

        pooled_train_input = _concat_pooled(train_parts)
        pooled_test_input = _concat_pooled(test_parts)
        if source.lags is not None:
            lag_cols = _lag_feature_cols(
                pooled_train_input,
                pooled=True,
                indicators=source.indicators,
                lag_cols=source.lag_cols,
            )
            interval_str = format_interval(interval)

            def bars_by_symbol(sym, interval=interval):
                key = (sym, interval)
                if key not in interval_data.compressed:
                    return None
                return interval_data.compressed[key].bars

            merge_interval_lag_series_cache(
                lag_series_cache,
                tuple(symbols),
                lag_cols,
                source.lags,
                interval_str,
                bars_by_symbol,
                full_arrays.get,
            )
            apply_lags_to_model_input_pooled(
                pooled_train_input,
                lag_cols,
                source.lags,
                lag_series_cache,
                history_dates,
                symbols,
                interval=interval_str,
            )
            apply_lags_to_model_input_pooled(
                pooled_test_input,
                lag_cols,
                source.lags,
                lag_series_cache,
                history_dates,
                symbols,
                interval=interval_str,
            )
            pooled_train_input = pooled_train_input.drop_lag_warmup()
        if total_dropped and pooled_train_input.empty():
            warnings.warn(
                _empty_interval_train_warning(
                    source.name,
                    interval,
                    ", ".join(repr(sym) for sym in sorted(symbols)),
                    lookahead,
                    total_dropped,
                )
            )
        return pooled_train_input, pooled_test_input

    def _prepare_interval_symbol_data(
        self,
        symbol: str,
        interval: TimeframeInterval,
        train_dates: Collection,
        test_dates: Collection,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        source: ModelSource,
        interval_data: IntervalData,
        lag_series_cache: LagSeriesCache,
        lookahead: int = 1,
    ) -> tuple[ModelInput, ModelInput]:
        scope = StaticScope.instance()
        key = (symbol, interval)
        if key not in interval_data.compressed:
            raise ValueError(
                f"Interval {interval!r} data not found for {symbol!r}."
            )
        compressed = interval_data.compressed[key]
        columns, arrays, bar_dates = build_compressed_symbol_arrays(
            symbol,
            interval,
            compressed,
            indicator_data,
            source.indicators,
            sorted(scope.custom_data_cols),
        )
        # The walkforward split holds out lookahead bars of the base
        # timeframe, but this model is fitted on compressed bars — re-measure
        # the hold-out in compressed-bar units or it silently collapses.
        effective_train_dates, dropped = lookahead_train_dates(
            bar_dates, train_dates, test_dates, lookahead
        )
        _, train_arrays, train_dates = slice_arrays_by_dates(
            columns,
            arrays,
            bar_dates,
            effective_train_dates,
        )
        _, test_arrays, test_dates = slice_arrays_by_dates(
            columns,
            arrays,
            bar_dates,
            test_dates,
        )
        sym_train_data = model_input_from_arrays(
            columns, train_arrays, train_dates
        )
        sym_test_data = model_input_from_arrays(
            columns, test_arrays, test_dates
        )
        if source.lags is not None:
            lag_cols = _lag_feature_cols(
                sym_train_data,
                pooled=False,
                indicators=source.indicators,
                lag_cols=source.lag_cols,
            )
            interval_str = format_interval(interval)

            def bars_by_symbol(sym, interval=interval):
                key = (sym, interval)
                if key not in interval_data.compressed:
                    return None
                return interval_data.compressed[key].bars

            merge_interval_lag_series_cache(
                lag_series_cache,
                (symbol,),
                lag_cols,
                source.lags,
                interval_str,
                bars_by_symbol,
                # Full-history arrays include indicators, which compressed
                # bars do not carry.
                lambda sym, arrays=arrays: arrays if sym == symbol else None,
            )
            history_dates = np.asarray(compressed.bars.dates)
            apply_lags_to_model_input(
                sym_train_data,
                lag_cols,
                source.lags,
                lag_series_cache,
                symbol,
                history_dates,
                interval_str,
            )
            apply_lags_to_model_input(
                sym_test_data,
                lag_cols,
                source.lags,
                lag_series_cache,
                symbol,
                history_dates,
                interval_str,
            )
            sym_train_data = sym_train_data.drop_lag_warmup()
        if dropped and sym_train_data.empty():
            warnings.warn(
                _empty_interval_train_warning(
                    source.name, interval, repr(symbol), lookahead, dropped
                )
            )
        return sym_train_data, sym_test_data

    def _load_pooled_group_cache(
        self,
        model_name: str,
        symbols: frozenset[str],
        cache_date_fields: CacheDateFields,
        lookahead: int = 1,
    ) -> tuple[bool, dict[ModelSymbol, TrainedModel]]:
        """Loads a fully cached pooled group in a single cache pass."""
        scope = StaticScope.instance()
        model_cache = scope.model_cache
        if model_cache is None:
            return False, {}
        cached_by_sym: dict[ModelSymbol, Union[CachedModel, Any]] = {}
        for sym in symbols:
            group_model_sym = ModelSymbol(model_name, sym)
            cache_key = ModelCacheKey.from_date_fields(
                symbol=group_model_sym.symbol,
                model_name=group_model_sym.model_name,
                fields=cache_date_fields,
                pooled_symbols=symbols,
                lookahead=_model_cache_lookahead(
                    group_model_sym.model_name, lookahead
                ),
            )
            scope.logger.debug_get_model_cache(cache_key)
            cached_data = model_cache.get(cache_key)
            if cached_data is None:
                return False, {}
            cached_by_sym[group_model_sym] = cached_data
        loaded: dict[ModelSymbol, TrainedModel] = {}
        for group_model_sym, cached_data in cached_by_sym.items():
            input_cols = None
            lag_columns = None
            if isinstance(cached_data, CachedModel):
                model = cached_data.model
                input_cols = cached_data.input_cols
                # Absent on models cached before the field existed.
                lag_columns = getattr(cached_data, "lag_columns", None)
            else:
                model = cached_data
            # Interval-bound models are keyed by a suffixed name, which is not
            # a registered source. Strip it as the training paths do.
            base_name, _ = parse_model_interval_name(
                group_model_sym.model_name
            )
            source = scope.get_model_source(base_name)
            loaded[group_model_sym] = TrainedModel(
                name=group_model_sym.model_name,
                instance=model,
                predict_fn=source._predict_fn,
                input_cols=input_cols,
                per_bar=source.per_bar,
                lag_columns=lag_columns,
            )
        return True, loaded

    @staticmethod
    def _uses_hyperparams(model_name: str) -> bool:
        """Whether ``model_name`` is trained on a hyperparameterized indicator.

        :class:`ModelCacheKey` carries no hyperparameter values, so a model
        trained under one value would be served for another. The indicator
        disk cache is skipped for the same reason; mirror it here rather than
        return a model fitted on different features.
        """
        scope = StaticScope.instance()
        base_name, _ = parse_model_interval_name(model_name)
        try:
            source = scope.get_model_source(base_name)
        except ValueError:
            return False
        for ind_name in getattr(source, "indicators", ()):
            ind_base, _ = parse_indicator_interval_name(ind_name)
            try:
                indicator = scope.get_indicator(ind_base)
            except ValueError:
                continue
            if indicator.hyperparam_names:
                return True
        return False

    def _get_cached_models(
        self,
        model_syms: Iterable[ModelSymbol],
        cache_date_fields: CacheDateFields,
        pooled_model_groups: Mapping[tuple[str, int], frozenset[str]],
        lookahead: int = 1,
    ) -> tuple[dict[ModelSymbol, TrainedModel], list[ModelSymbol]]:
        model_syms = sorted(model_syms)
        models: dict[ModelSymbol, TrainedModel] = {}
        scope = StaticScope.instance()
        if scope.model_cache is None:
            return models, model_syms
        # Trained under a hyperparameter the key does not carry, so the cache
        # cannot tell two fits apart. Treat them as uncached throughout.
        hyperparam_syms = [
            model_sym
            for model_sym in model_syms
            if self._uses_hyperparams(model_sym.model_name)
        ]
        if hyperparam_syms:
            skipped = set(hyperparam_syms)
            model_syms = [
                model_sym
                for model_sym in model_syms
                if model_sym not in skipped
            ]
        uncached_model_syms: list[ModelSymbol] = []
        pooled_groups_by_model_sym: dict[ModelSymbol, frozenset[str]] = {}
        for (model_name, _), symbols in pooled_model_groups.items():
            for sym in symbols:
                pooled_groups_by_model_sym[ModelSymbol(model_name, sym)] = (
                    symbols
                )
        processed_pooled_groups: set[tuple[str, frozenset[str]]] = set()
        for model_sym in model_syms:
            if model_sym in pooled_groups_by_model_sym:
                symbols = pooled_groups_by_model_sym[model_sym]
                group_key = (model_sym.model_name, symbols)
                if group_key in processed_pooled_groups:
                    continue
                processed_pooled_groups.add(group_key)
                group_cached, loaded = self._load_pooled_group_cache(
                    model_sym.model_name, symbols, cache_date_fields, lookahead
                )
                if group_cached:
                    models.update(loaded)
                else:
                    uncached_model_syms.append(model_sym)
                continue
            cache_key = ModelCacheKey.from_date_fields(
                symbol=model_sym.symbol,
                model_name=model_sym.model_name,
                fields=cache_date_fields,
                lookahead=_model_cache_lookahead(
                    model_sym.model_name, lookahead
                ),
            )
            scope.logger.debug_get_model_cache(cache_key)
            cached_data = scope.model_cache.get(cache_key)
            if cached_data is not None:
                input_cols = None
                lag_columns = None
                if isinstance(cached_data, CachedModel):
                    model = cached_data.model
                    input_cols = cached_data.input_cols
                    # Absent on models cached before the field existed.
                    lag_columns = getattr(cached_data, "lag_columns", None)
                else:
                    model = cached_data
                # See _load_pooled_group_cache: strip the interval suffix.
                base_name, _ = parse_model_interval_name(model_sym.model_name)
                source = scope.get_model_source(base_name)
                models[model_sym] = TrainedModel(
                    name=model_sym.model_name,
                    instance=model,
                    predict_fn=source._predict_fn,
                    input_cols=input_cols,
                    per_bar=source.per_bar,
                    lag_columns=lag_columns,
                )
            else:
                uncached_model_syms.append(model_sym)
        if hyperparam_syms:
            uncached_model_syms = sorted(uncached_model_syms + hyperparam_syms)
        return models, uncached_model_syms

    def _set_cached_model(
        self,
        model: Any,
        input_cols: Optional[tuple[str]],
        model_sym: ModelSymbol,
        cache_date_fields: CacheDateFields,
        lag_columns: Optional[tuple[str, ...]] = None,
        pooled_symbols: Optional[frozenset[str]] = None,
        lookahead: int = 1,
    ):
        scope = StaticScope.instance()
        if scope.model_cache is None:
            return
        if self._uses_hyperparams(model_sym.model_name):
            # See _uses_hyperparams: writing this would make the next run with
            # a different hyperparameter value read back the wrong fit.
            return
        cache_key = ModelCacheKey.from_date_fields(
            symbol=model_sym.symbol,
            model_name=model_sym.model_name,
            fields=cache_date_fields,
            pooled_symbols=pooled_symbols,
            lookahead=_model_cache_lookahead(model_sym.model_name, lookahead),
        )
        cached_model = CachedModel(model, input_cols, lag_columns)
        scope.logger.debug_set_model_cache(cache_key)
        scope.model_cache.set(cache_key, cached_model)
