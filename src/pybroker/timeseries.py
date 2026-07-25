"""Time series model helpers (internal numpy lag transforms and full-history context)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pybroker.common import DataCol
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pybroker.scope import SymbolArrayStore

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

    Not part of the public API. User-facing code receives :class:`pandas.DataFrame`
    instances materialized via :func:`model_input_to_dataframe`.
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
        """Drops rows with NaN lag features."""
        if self.lag_features is None or self.empty():
            return self
        valid = ~np.isnan(self.lag_features).any(axis=1)
        if valid.all():
            return self
        arrays = {col: values[valid] for col, values in self.arrays.items()}
        return ModelInput(
            self.columns,
            arrays,
            self.dates[valid],
            self.lag_features[valid],
            self.lags,
            self.lag_columns,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Materializes a DataFrame for legacy call sites."""
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
) -> LagSeriesCache:
    """Adds full-history lag arrays from a :class:`SymbolArrayStore`."""
    if history_dates is None:
        history_dates = {}
    date_col = DataCol.DATE.value
    for sym in symbols:
        if sym not in store.symbols:
            continue
        sym_data = store.sym_arrays[sym]
        dates = sym_data.get(date_col)
        if dates is None or len(dates) == 0:
            continue
        history_dates[sym] = np.asarray(dates, dtype="datetime64[ns]")
        col_arrays = {col: sym_data[col] for col in columns if col in sym_data}
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
        values = column_arrays[col]
        if values is None:
            raise ValueError(f"Column {col!r} not found for {symbol!r}.")
        values = np.asarray(values, dtype=np.float64)
        for lag in range(1, lags + 1):
            cache[LagSeriesKey(symbol, col, lag)] = shift_array(values, lag)


def merge_timeframe_lag_series_cache(
    cache: LagSeriesCache,
    symbols: Iterable[str],
    columns: tuple[str, ...],
    lags: int,
    interval: str,
    bars_by_symbol,
) -> LagSeriesCache:
    """Adds full-history timeframe lag arrays into ``cache``."""
    for sym in symbols:
        bars = bars_by_symbol(sym)
        if bars is None:
            continue
        for col in columns:
            if col == DataCol.DATE.value:
                continue
            col_data = _bars_column_array(bars, col)
            if col_data is None:
                continue
            values = np.asarray(col_data, dtype=np.float64)
            for lag in range(1, lags + 1):
                cache[LagSeriesKey(sym, col, lag, interval)] = shift_array(
                    values, lag
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


def build_lag_feature_matrix(
    symbol: str,
    columns: tuple[str, ...],
    lags: int,
    base_arrays: ArrayDict,
    row_dates: np.ndarray,
    history_dates: np.ndarray,
    lag_cache: LagSeriesCache,
    interval: Optional[str] = None,
) -> np.ndarray:
    """Builds a lag-expanded feature matrix from numpy arrays."""
    n_rows = len(row_dates)
    n_features = len(columns) * (lags + 1)
    if n_rows == 0:
        return np.empty((0, n_features), dtype=np.float64)
    offset = history_date_offset(history_dates, row_dates)
    matrix = np.empty((n_rows, n_features), dtype=np.float64)
    col_idx = 0
    for col in columns:
        matrix[:, col_idx] = np.asarray(base_arrays[col], dtype=np.float64)
        col_idx += 1
        for lag in range(1, lags + 1):
            lag_arr = lag_cache[LagSeriesKey(symbol, col, lag, interval)]
            matrix[:, col_idx] = lag_arr[offset : offset + n_rows]
            col_idx += 1
    return matrix


def build_lag_feature_matrix_pooled(
    sym_col: np.ndarray,
    columns: tuple[str, ...],
    lags: int,
    base_arrays: ArrayDict,
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
    date_col = DataCol.DATE.value
    for sym in symbols:
        mask = sym_col == sym
        if not mask.any():
            continue
        sym_dates = row_dates[mask]
        sym_base = {col: base_arrays[col][mask] for col in columns}
        if date_col in base_arrays:
            sym_base[date_col] = sym_dates
        sym_matrix = build_lag_feature_matrix(
            sym,
            columns,
            lags,
            sym_base,
            sym_dates,
            history_dates_by_symbol[sym],
            lag_cache,
            interval,
        )
        matrix[mask] = sym_matrix
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
    base_arrays = {col: model_input.arrays[col] for col in lag_columns}
    matrix = build_lag_feature_matrix(
        symbol,
        lag_columns,
        lags,
        base_arrays,
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
        model_input.arrays,
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


LAG_FEATURES_ATTR = "lag_features"
LAGS_ATTR = "lags"
LAG_COLUMNS_ATTR = "lag_columns"


def model_input_to_dataframe(model_input: ModelInput) -> pd.DataFrame:
    """Materializes a DataFrame with optional lag metadata in ``attrs``."""
    df = model_input.to_dataframe()
    if model_input.lag_features is not None:
        df.attrs[LAG_FEATURES_ATTR] = model_input.lag_features
        df.attrs[LAGS_ATTR] = model_input.lags
        df.attrs[LAG_COLUMNS_ATTR] = model_input.lag_columns
    return df


def apply_prepare_input_data(
    model_input: ModelInput,
    prepare_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> ModelInput:
    """Applies a DataFrame-only prepare function to ``model_input``."""
    lag_features = model_input.lag_features
    lags = model_input.lags
    lag_columns = model_input.lag_columns
    df = prepare_fn(model_input.to_dataframe())
    result = model_input_from_frame(
        df, columns=tuple(df.columns), dates=model_input.dates
    )
    if lag_features is not None:
        result.lag_features = lag_features
        result.lags = lags
        result.lag_columns = lag_columns
    return result


def feature_matrix_from_model_input(
    data: Union[ModelInput, pd.DataFrame],
) -> Optional[np.ndarray]:
    """Returns lag feature matrix for ``data``, if any."""
    if isinstance(data, ModelInput):
        return (
            None
            if data.lag_features is None
            else np.asarray(data.lag_features)
        )
    matrix = data.attrs.get(LAG_FEATURES_ATTR)
    return None if matrix is None else np.asarray(matrix)


def model_input_lags(data: Union[ModelInput, pd.DataFrame]) -> Optional[int]:
    """Returns the lag count for ``data``, if any."""
    if isinstance(data, ModelInput):
        return data.lags
    return data.attrs.get(LAGS_ATTR)
