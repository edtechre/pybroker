"""Contains scopes that store data and object references used to execute a
:class:`pybroker.strategy.Strategy`.
"""

from __future__ import annotations

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
from numba import njit
from pybroker.common import (
    BarData,
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    PriceType,
    TrainedModel,
    to_decimal,
)
from pybroker.log import Logger
from pybroker.interval import (
    IntervalData,
    TimeframeInterval,
    _find_bin_starts_ends,
    format_interval,
    indicator_interval_name,
    model_interval_name,
    normalize_interval,
    parse_indicator_interval_name,
)
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from diskcache import Cache
from importlib import import_module
from numpy.typing import NDArray
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    from pybroker.model import LagSeriesCache, ModelInput
    from pybroker.portfolio import Stop


@dataclass(frozen=True)
class _ModelImports:
    """Lazy model helpers to avoid cache -> scope -> model import cycles."""

    model_input_cls: type[ModelInput]
    apply_lags_to_model_input: Callable[..., ModelInput]
    apply_prepare_input_data: Callable[..., ModelInput]
    merge_lag_series_cache_from_arrays: Callable[..., None]
    merge_interval_lag_series_cache: Callable[..., LagSeriesCache]
    model_input_to_dataframe: Callable[..., pd.DataFrame]
    model_trainer_cls: type
    _indicator_values_for_dates: Callable[..., NDArray[np.float64]]


_model_imports: _ModelImports | None = None


def _model() -> _ModelImports:
    global _model_imports
    if _model_imports is None:
        model_mod = import_module("pybroker.model")

        _model_imports = _ModelImports(
            model_input_cls=model_mod.ModelInput,
            apply_lags_to_model_input=model_mod.apply_lags_to_model_input,
            apply_prepare_input_data=model_mod.apply_prepare_input_data,
            merge_lag_series_cache_from_arrays=(
                model_mod.merge_lag_series_cache_from_arrays
            ),
            merge_interval_lag_series_cache=(
                model_mod.merge_interval_lag_series_cache
            ),
            model_input_to_dataframe=model_mod.model_input_to_dataframe,
            model_trainer_cls=model_mod.ModelTrainer,
            _indicator_values_for_dates=(
                model_mod._indicator_values_for_dates
            ),
        )
    return _model_imports


_EMPTY_PARAM: Final = object()

# Cached enum accesses. Hot paths (ColumnScope.bar_data_from_data_columns,
# PriceScope.fetch) hit these 10_000+ times per backtest; binding once here
# skips the enum descriptor call per access.
_COL_DATE: Final = DataCol.DATE.value
_COL_OPEN: Final = DataCol.OPEN.value
_COL_HIGH: Final = DataCol.HIGH.value
_COL_LOW: Final = DataCol.LOW.value
_COL_CLOSE: Final = DataCol.CLOSE.value
_COL_VOLUME: Final = DataCol.VOLUME.value
_COL_VWAP: Final = DataCol.VWAP.value
_PRICE_OPEN: Final = PriceType.OPEN
_PRICE_HIGH: Final = PriceType.HIGH
_PRICE_LOW: Final = PriceType.LOW
_PRICE_CLOSE: Final = PriceType.CLOSE
_PRICE_MIDDLE: Final = PriceType.MIDDLE
_PRICE_AVERAGE: Final = PriceType.AVERAGE
_BAR_OHLC_COLS: Final = (_COL_DATE, _COL_CLOSE, _COL_LOW, _COL_HIGH)


_UNPICKLED_CACHES: Final = (
    "data_source_cache",
    "indicator_cache",
    "model_cache",
)


class StaticScope:
    """A static registry of data and object references.

    Attributes:
        logger: :class:`pybroker.log.Logger`
        data_source_cache: :class:`diskcache.Cache` that stores data retrieved
            from :class:`pybroker.data.DataSource`.
        data_source_cache_ns: Namespace set for  :attr:`.data_source_cache`.
        indicator_cache: :class:`diskcache.Cache` that stores
            :class:`pybroker.indicator.Indicator` data.
        indicator_cache_ns: Namespace set for :attr:`.indicator_cache`.
        model_cache: :class:`diskcache.Cache` that stores trained models.
        model_cache_ns: Namespace set for :attr:`.model_cache`.
        default_data_cols: Default data columns in :class:`pandas.DataFrame`
            retrieved from a :class:`pybroker.data.DataSource`.
        custom_data_cols: User-defined data columns in
            :class:`pandas.DataFrame` retrieved from a
            :class:`pybroker.data.DataSource`.
    """

    __instance = None

    def __init__(self):
        self.logger = Logger(self)
        self.data_source_cache: Optional[Cache] = None
        self.data_source_cache_ns: str = ""
        self.indicator_cache: Optional[Cache] = None
        self.indicator_cache_ns: str = ""
        self.model_cache: Optional[Cache] = None
        self.model_cache_ns: str = ""
        self._indicators = {}
        self._model_sources = {}
        self.default_data_cols = frozenset(
            (
                DataCol.DATE.value,
                DataCol.OPEN.value,
                DataCol.HIGH.value,
                DataCol.LOW.value,
                DataCol.CLOSE.value,
                DataCol.VOLUME.value,
                DataCol.VWAP.value,
            )
        )
        self.custom_data_cols = set()
        self._cols_frozen: bool = False
        self._all_data_cols: Optional[frozenset[str]] = None
        self._ordered_data_cols: Optional[tuple[str, ...]] = None
        self._bar_data_cols: Optional[tuple[str, ...]] = None
        self._params: dict[str, Any] = {}
        self._hyperparams: dict[str, Any] = {}

    def set_indicator(self, indicator):
        """Stores :class:`pybroker.indicator.Indicator` in static scope."""
        self._indicators[indicator.name] = indicator

    def has_indicator(self, name: str) -> bool:
        """Whether :class:`pybroker.indicator.Indicator` is stored in static
        scope.
        """
        return name in self._indicators

    def get_indicator(self, name: str):
        """Retrieves a :class:`pybroker.indicator.Indicator` from static
        scope."""
        if not self.has_indicator(name):
            raise ValueError(f"Indicator {name!r} does not exist.")
        return self._indicators[name]

    def get_indicator_names(self, model_name: str) -> tuple[str]:
        """Returns a ``tuple[str]`` of all
        :class:`pybroker.indicator.Indicator` names that are registered with
        :class:`pybroker.model.ModelSource` having ``model_name``.
        """
        return self._model_sources[model_name].indicators

    def set_model_source(self, source):
        """Stores :class:`pybroker.model.ModelSource` in static scope."""
        self._model_sources[source.name] = source

    def has_model_source(self, name: str) -> bool:
        """Whether :class:`pybroker.model.ModelSource` is stored in static
        scope.
        """
        return name in self._model_sources

    def get_model_source(self, name: str):
        """Retrieves a :class:`pybroker.model.ModelSource` from static
        scope.
        """
        if not self.has_model_source(name):
            raise ValueError(f"ModelSource {name!r} does not exist.")
        return self._model_sources[name]

    def register_custom_cols(self, names: Union[str, Iterable[str]], *args):
        """Registers user-defined column names."""
        self._verify_unfrozen_cols()
        if isinstance(names, str):
            names = (names, *args)
        else:
            names = (*names, *args)
        names = filter(lambda col: col not in self.default_data_cols, names)
        self.custom_data_cols.update(names)

    def unregister_custom_cols(self, names: Union[str, Iterable[str]], *args):
        """Unregisters user-defined column names."""
        self._verify_unfrozen_cols()
        if isinstance(names, str):
            names = (names, *args)
        else:
            names = (*names, *args)
        self.custom_data_cols.difference_update(names)

    @property
    def all_data_cols(self) -> frozenset[str]:
        """All registered data column names. Unordered; use
        :attr:`ordered_data_cols` when iteration order is significant.
        """
        if self._all_data_cols is not None:
            return self._all_data_cols
        return self.default_data_cols | self.custom_data_cols

    @property
    def ordered_data_cols(self) -> tuple[str, ...]:
        """All registered data column names in deterministic order. Iterating
        :attr:`all_data_cols` instead yields a process-dependent order, which
        makes column-order sensitive output such as model input data
        irreproducible across runs.
        """
        if self._ordered_data_cols is not None:
            return self._ordered_data_cols
        return self._build_ordered_data_cols()

    def _build_ordered_data_cols(self) -> tuple[str, ...]:
        return (
            _COL_DATE,
            _COL_OPEN,
            _COL_HIGH,
            _COL_LOW,
            _COL_CLOSE,
            _COL_VOLUME,
            _COL_VWAP,
            *sorted(self.custom_data_cols),
        )

    def _verify_unfrozen_cols(self):
        if self._cols_frozen:
            raise ValueError("Cannot modify columns when strategy is running.")

    def freeze_data_cols(self):
        """Prevents additional data columns from being registered."""
        self._cols_frozen = True
        self._all_data_cols = self.default_data_cols | self.custom_data_cols
        self._ordered_data_cols = self._build_ordered_data_cols()
        self._bar_data_cols = self._ordered_data_cols

    def unfreeze_data_cols(self):
        """Allows additional data columns to be registered if
        :func:`pybroker.scope.StaticScope.freeze_data_cols` was called.
        """
        self._cols_frozen = False
        self._all_data_cols = None
        self._ordered_data_cols = None
        self._bar_data_cols = None

    def param(
        self, name: str, value: Optional[Any] = _EMPTY_PARAM
    ) -> Optional[Any]:
        """Get or set a global parameter."""
        if value is _EMPTY_PARAM:
            return self._params.get(name, None)
        self._params[name] = value
        return value

    def clear_params(self):
        """Clears all global parameters."""
        self._params.clear()

    def set_hyperparam(self, hyperparam: Any) -> None:
        """Stores a :class:`pybroker.optimize.Hyperparam` in static scope."""
        self._hyperparams[hyperparam.name] = hyperparam

    def has_hyperparam(self, name: str) -> bool:
        """Whether a hyperparam is stored in static scope."""
        return name in self._hyperparams

    def get_hyperparam(self, name: str) -> Any:
        """Retrieves a hyperparam from static scope."""
        if not self.has_hyperparam(name):
            raise ValueError(f"Hyperparam {name!r} does not exist.")
        return self._hyperparams[name]

    def iter_hyperparams(self) -> Iterable[Any]:
        """Iterates registered hyperparams."""
        return self._hyperparams.values()

    def __getstate__(self) -> dict[str, Any]:
        """Returns picklable state, for shipping this scope to a worker
        process.

        Caches are per-process resources tied to a diskcache directory, and
        carrying them would ship their in-memory L1 layer too, so the
        :class:`diskcache.Cache` references are dropped. The namespaces are
        kept, so a worker can reopen them if needed.
        """
        return {**self.__dict__, **{k: None for k in _UNPICKLED_CACHES}}

    @classmethod
    def instance(cls) -> "StaticScope":
        """Returns singleton instance."""
        if cls.__instance is None:
            cls.__instance = StaticScope()
        return cls.__instance

    @classmethod
    def set_instance(cls, scope: Optional["StaticScope"]) -> None:
        """Replaces the singleton instance, or clears it when ``scope`` is
        ``None``.

        Used to install a scope that was pickled from another process, so that
        worker tasks see the caller's registered indicators, model sources and
        params instead of an empty scope. Replacing wholesale (rather than
        merging) also keeps stale registrations from surviving in a worker that
        is reused across runs.
        """
        cls.__instance = scope


def run_with_scope(
    scope: StaticScope, fn: Callable[..., Any], *args: Any
) -> Any:
    """Installs ``scope`` as this process' scope, then runs ``fn``.

    :class:`StaticScope` is a per-process singleton, so a worker process starts
    with an empty one and would not see the caller's registered indicators,
    model sources, params or custom columns. Wrap work dispatched to
    :func:`pybroker.parallel.parallel` in this to ship the caller's scope along
    with it. Running sequentially, ``scope`` is already the installed instance
    and this is a no-op.
    """
    StaticScope.set_instance(scope)
    return fn(*args)


def disable_logging():
    """Disables event logging."""
    StaticScope.instance().logger.disable()


def enable_logging():
    """Enables event logging."""
    StaticScope.instance().logger.enable()


def disable_progress_bar():
    """Disables logging a progress bar."""
    StaticScope.instance().logger.disable_progress_bar()


def enable_progress_bar():
    """Enables logging a progress bar."""
    StaticScope.instance().logger.enable_progress_bar()


def register_columns(names: Union[str, Iterable[str]], *args):
    """Registers ``names`` of user-defined data columns."""
    StaticScope.instance().register_custom_cols(names, *args)


def unregister_columns(names: Union[str, Iterable[str]], *args):
    """Unregisters ``names`` of user-defined data columns."""
    StaticScope.instance().unregister_custom_cols(names, *args)


def param(name: str, value: Optional[Any] = _EMPTY_PARAM) -> Optional[Any]:
    """Get or set a global parameter."""
    return StaticScope.instance().param(name, value)


def clear_params():
    """Clears all global parameters."""
    StaticScope.instance().clear_params()


@dataclass(frozen=True)
class _StoreBacking:
    """Contiguous buffers spanning every symbol, plus per-symbol row ranges.

    Per-symbol arrays are views into these buffers. Keeping the buffers whole
    is what makes a store cheap to send to a worker process: joblib memmaps
    numpy arrays above its ``max_nbytes`` threshold (1MB by default), so a few
    large buffers are written once and mapped by every worker, whereas the
    per-symbol arrays are individually far below the threshold and would each
    be copied.

    Attributes:
        stack: ``(n_float_cols, n_rows)`` array of the numeric columns.
        stack_cols: Column names, positionally matching ``stack`` rows.
        other: Non-numeric columns (dates included), each spanning all rows.
        offsets: Maps symbol to its ``(start, stop)`` row range.
    """

    stack: NDArray
    stack_cols: tuple[str, ...]
    other: Mapping[str, NDArray]
    offsets: Mapping[str, tuple[int, int]]

    def __post_init__(self):
        # Freeze the buffers before any view is taken: a view inherits
        # writeability at creation time, so freezing afterwards would leave
        # already-created views writable and let one symbol corrupt another.
        self.stack.flags.writeable = False
        for arr in self.other.values():
            arr.flags.writeable = False

    def views(self) -> dict[str, dict[str, NDArray]]:
        """Returns per-symbol column views into the backing buffers."""
        sym_arrays: dict[str, dict[str, NDArray]] = {}
        for sym, (start, stop) in self.offsets.items():
            arrays: dict[str, NDArray] = {
                col: self.stack[c, start:stop]
                for c, col in enumerate(self.stack_cols)
            }
            for col, arr in self.other.items():
                arrays[col] = arr[start:stop]
            sym_arrays[sym] = arrays
        return sym_arrays


@dataclass(frozen=True)
class SymbolArrayStore:
    """Internal numpy-backed OHLCV/custom columns keyed by symbol."""

    symbols: frozenset[str]
    sym_arrays: Mapping[str, Mapping[str, NDArray]]
    backing: Optional[_StoreBacking] = None

    def __post_init__(self):
        # Input data is read-only. Marking it so turns an accidental in-place
        # write into a loud ValueError, and lets per-symbol arrays be views
        # into a shared buffer rather than copies: a view of a read-only array
        # is itself read-only, so one symbol cannot corrupt its neighbours.
        # Backed stores are already frozen by _StoreBacking.__post_init__.
        if self.backing is not None:
            return
        for arrays in self.sym_arrays.values():
            for arr in arrays.values():
                if arr is not None and arr.flags.owndata:
                    arr.flags.writeable = False

    def unique_dates(self) -> NDArray[np.datetime64]:
        """Returns sorted unique dates across every symbol."""
        date_col = DataCol.DATE.value
        if self.backing is not None and date_col in self.backing.other:
            return np.unique(self.backing.other[date_col])
        if not self.sym_arrays:
            return np.array([], dtype="datetime64[ns]")
        return np.unique(
            np.concatenate(
                [
                    arrays[date_col]
                    for arrays in self.sym_arrays.values()
                    if arrays.get(date_col) is not None
                ]
            )
        )

    def __getstate__(self) -> dict[str, Any]:
        # numpy pickles a view as an independent copy, so when this store is
        # backed by contiguous buffers, send those plus the row ranges and
        # rebuild the views on the other side.
        if self.backing is None:
            return {
                "symbols": self.symbols,
                "sym_arrays": self.sym_arrays,
                "backing": None,
            }
        return {
            "symbols": self.symbols,
            "sym_arrays": None,
            "backing": self.backing,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        backing = state["backing"]
        sym_arrays = state["sym_arrays"]
        if sym_arrays is None:
            sym_arrays = backing.views()
        object.__setattr__(self, "symbols", state["symbols"])
        object.__setattr__(self, "sym_arrays", sym_arrays)
        object.__setattr__(self, "backing", backing)


@njit(cache=True)
def _sorted_dates_indices_njit(
    dates: NDArray[np.datetime64],
    target: NDArray[np.datetime64],
) -> NDArray[np.int64]:
    n_dates = len(dates)
    n_target = len(target)
    if n_dates == 0 or n_target == 0:
        return np.empty(0, dtype=np.int64)
    out = np.empty(n_target, dtype=np.int64)
    n_matches = 0
    i = 0
    j = 0
    while i < n_dates and j < n_target:
        if dates[i] == target[j]:
            out[n_matches] = i
            n_matches += 1
            i += 1
            j += 1
        elif dates[i] < target[j]:
            i += 1
        else:
            j += 1
    return out[:n_matches]


@njit(cache=True)
def _gather_f64_by_indices_njit(
    col_stack: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    n_cols = col_stack.shape[0]
    n_out = len(indices)
    out = np.empty((n_cols, n_out), dtype=np.float64)
    for c in range(n_cols):
        for j in range(n_out):
            out[c, j] = col_stack[c, indices[j]]
    return out


@njit(cache=True)
def _gather_dt64_by_indices_njit(
    dates: NDArray[np.datetime64],
    indices: NDArray[np.int64],
) -> NDArray[np.datetime64]:
    n_out = len(indices)
    out = np.empty(n_out, dtype=dates.dtype)
    for j in range(n_out):
        out[j] = dates[indices[j]]
    return out


def _build_sliced_sym_arrays(
    sym_data: Mapping[str, NDArray],
    indices: NDArray[np.int64],
    date_col: str,
) -> dict[str, NDArray]:
    """Builds per-symbol column arrays for ``indices`` row selection."""
    if len(indices) == 0:
        return {}
    if len(indices) > 0 and indices[-1] - indices[0] + 1 == len(indices):
        start = int(indices[0])
        end = int(indices[-1]) + 1
        return {
            col: np.asarray(arr[start:end], copy=True)
            for col, arr in sym_data.items()
        }
    float_cols: list[str] = []
    other_cols: list[str] = []
    for col, arr in sym_data.items():
        if col == date_col:
            continue
        if np.issubdtype(np.asarray(arr).dtype, np.number):
            float_cols.append(col)
        else:
            other_cols.append(col)
    result: dict[str, NDArray] = {}
    if float_cols:
        sample = sym_data[float_cols[0]]
        n_rows = len(sample)
        col_stack = np.empty((len(float_cols), n_rows), dtype=np.float64)
        for c, col in enumerate(float_cols):
            col_stack[c] = np.ascontiguousarray(
                sym_data[col], dtype=np.float64
            )
        gathered = _gather_f64_by_indices_njit(col_stack, indices)
        for c, col in enumerate(float_cols):
            result[col] = gathered[c].copy()
    for col in other_cols:
        result[col] = np.asarray(sym_data[col][indices], copy=True)
    if date_col in sym_data:
        dates_arr = np.ascontiguousarray(
            sym_data[date_col], dtype="datetime64[ns]"
        )
        result[date_col] = _gather_dt64_by_indices_njit(
            dates_arr, indices
        ).copy()
    return result


def symbol_array_store_from_indexed_df(df: pd.DataFrame) -> SymbolArrayStore:
    """Builds a :class:`SymbolArrayStore` from a sorted MultiIndex frame."""
    df = df.sort_index()
    sym_arrays: dict[str, dict[str, NDArray]] = {}
    date_col = DataCol.DATE.value
    for sym in df.index.get_level_values(0).unique():
        sym_key = str(sym)
        sym_df = df.loc[pd.IndexSlice[sym_key, :]]
        sym_arrays[sym_key] = {
            col: np.asarray(sym_df[col].to_numpy(copy=True))
            for col in sym_df.columns
        }
        if date_col not in sym_arrays[sym_key]:
            idx = sym_df.index
            if isinstance(idx, pd.MultiIndex):
                sym_arrays[sym_key][date_col] = np.asarray(
                    idx.get_level_values(-1).to_numpy(copy=True)
                )
            else:
                sym_arrays[sym_key][date_col] = np.asarray(
                    idx.to_numpy(copy=True)
                )
    return SymbolArrayStore(frozenset(sym_arrays.keys()), sym_arrays)


def symbol_array_store_from_flat_frame(
    df: pd.DataFrame,
    sym_col: str = DataCol.SYMBOL.value,
    date_col: str = DataCol.DATE.value,
    symbols: Optional[frozenset[str]] = None,
) -> SymbolArrayStore:
    """Builds a store from a flat frame via numpy lex-sort and bin slicing."""
    if df.empty:
        return SymbolArrayStore(frozenset(), {})
    sym_values = df[sym_col].astype(str).to_numpy()
    date_arr = df[date_col].to_numpy(dtype="datetime64[ns]", copy=False)
    unique_syms, sym_ids = np.unique(sym_values, return_inverse=True)
    order = np.lexsort((date_arr, sym_ids.astype(np.int64)))
    sorted_sym_ids = sym_ids[order].astype(np.int64)
    starts, ends = _find_bin_starts_ends(sorted_sym_ids)
    data_cols = [col for col in df.columns if col != sym_col]
    float_cols: list[str] = []
    other_arrays: dict[str, NDArray] = {}
    for col in data_cols:
        if col == date_col:
            continue
        col_arr = np.asarray(df[col].to_numpy(copy=True)[order])
        if np.issubdtype(col_arr.dtype, np.number):
            float_cols.append(col)
        else:
            other_arrays[col] = col_arr
    n_rows = len(order)
    sorted_dates = np.ascontiguousarray(
        date_arr[order], dtype="datetime64[ns]"
    )
    col_stack = np.empty((len(float_cols), n_rows), dtype=np.float64)
    for c, col in enumerate(float_cols):
        col_stack[c] = np.ascontiguousarray(
            df[col].to_numpy(copy=True)[order], dtype=np.float64
        )
    # Rows are lex-sorted by (symbol, date), so each symbol owns a contiguous
    # range. Keep the whole-frame buffers and describe symbols as ranges into
    # them, rather than copying each symbol out: the buffers stay large enough
    # for joblib to memmap when this store is sent to a worker.
    selected = [
        (
            str(unique_syms[sorted_sym_ids[starts[i]]]),
            int(starts[i]),
            int(ends[i]) + 1,
        )
        for i in range(len(starts))
    ]
    if symbols is not None:
        selected = [item for item in selected if item[0] in symbols]
    if not selected:
        return SymbolArrayStore(frozenset(), {})
    other: dict[str, NDArray] = {date_col: sorted_dates, **other_arrays}
    stack: NDArray = col_stack
    offsets: dict[str, tuple[int, int]] = {}
    if len(selected) == len(starts):
        offsets = {sym: (start, end) for sym, start, end in selected}
    else:
        # Compact to just the requested symbols so the buffers do not carry
        # rows nobody asked for.
        keep = np.concatenate(
            [np.arange(start, end) for _, start, end in selected]
        )
        stack = np.ascontiguousarray(col_stack[:, keep])
        other = {
            col: np.ascontiguousarray(arr[keep]) for col, arr in other.items()
        }
        pos = 0
        for sym, start, end in selected:
            width = end - start
            offsets[sym] = (pos, pos + width)
            pos += width
    backing = _StoreBacking(
        stack=stack,
        stack_cols=tuple(float_cols),
        other=other,
        offsets=offsets,
    )
    sym_arrays = backing.views()
    return SymbolArrayStore(
        frozenset(sym_arrays.keys()), sym_arrays, backing=backing
    )


def symbol_array_store_from_frame(
    df: pd.DataFrame,
    sym_col: str = DataCol.SYMBOL.value,
    date_col: str = DataCol.DATE.value,
    symbols: Optional[frozenset[str]] = None,
) -> SymbolArrayStore:
    """Builds a store from a flat or MultiIndex OHLCV frame."""
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2:
        store = symbol_array_store_from_indexed_df(df)
        if symbols is None:
            return store
        filtered = {
            sym: arrays
            for sym, arrays in store.sym_arrays.items()
            if sym in symbols
        }
        return SymbolArrayStore(frozenset(filtered.keys()), filtered)
    if sym_col in df.columns and date_col in df.columns:
        return symbol_array_store_from_flat_frame(
            df, sym_col, date_col, symbols=symbols
        )
    indexed = df.set_index([sym_col, date_col]).sort_index()
    store = symbol_array_store_from_indexed_df(indexed)
    if symbols is None:
        return store
    filtered = {
        sym: arrays
        for sym, arrays in store.sym_arrays.items()
        if sym in symbols
    }
    return SymbolArrayStore(frozenset(filtered.keys()), filtered)


def sym_data_from_store(
    store: SymbolArrayStore,
    data_cols: Iterable[str],
) -> dict[str, dict[str, Optional[NDArray]]]:
    """Converts a :class:`SymbolArrayStore` to per-symbol column arrays."""
    sym_data: dict[str, dict[str, Optional[NDArray]]] = {}
    for sym, arrays in store.sym_arrays.items():
        sym_data[sym] = {col: arrays.get(col) for col in data_cols}
    return sym_data


def _dates_in_target_mask(
    dates: NDArray[np.datetime64],
    target: NDArray[np.datetime64],
) -> NDArray[np.bool_]:
    """Returns a boolean mask of ``dates`` present in ``target``."""
    if len(dates) == 0 or len(target) == 0:
        return np.zeros(len(dates), dtype=bool)
    if len(target) > 1 and np.all(target[:-1] <= target[1:]):
        if len(dates) > 1 and np.all(dates[:-1] <= dates[1:]):
            _, idx_in_dates, _ = np.intersect1d(
                dates,
                target,
                assume_unique=True,
                return_indices=True,
            )
            if len(idx_in_dates) == 0:
                return np.zeros(len(dates), dtype=bool)
            mask = np.zeros(len(dates), dtype=bool)
            mask[idx_in_dates] = True
            return mask
    return np.isin(dates, target)


def slice_symbol_array_store_by_dates(
    store: SymbolArrayStore,
    selected_dates: Union[Sequence[np.datetime64], NDArray[np.datetime64]],
) -> SymbolArrayStore:
    """Filters a store to rows whose dates are in ``selected_dates``."""
    if not store.symbols:
        return SymbolArrayStore(frozenset(), {})
    date_col = DataCol.DATE.value
    target = np.asarray(selected_dates, dtype="datetime64[ns]")
    if len(target) == 0:
        return SymbolArrayStore(frozenset(), {})
    target_sorted = len(target) <= 1 or bool(np.all(target[:-1] <= target[1:]))
    sym_arrays: dict[str, dict[str, NDArray]] = {}
    for sym in store.symbols:
        sym_data = store.sym_arrays[sym]
        dates = sym_data.get(date_col)
        if dates is None or len(dates) == 0:
            continue
        dates_arr = np.ascontiguousarray(dates, dtype="datetime64[ns]")
        dates_sorted = len(dates_arr) <= 1 or bool(
            np.all(dates_arr[:-1] <= dates_arr[1:])
        )
        if target_sorted and dates_sorted:
            indices = _sorted_dates_indices_njit(dates_arr, target)
        else:
            mask = _dates_in_target_mask(dates_arr, target)
            if not mask.any():
                continue
            indices = np.flatnonzero(mask).astype(np.int64)
        sliced = _build_sliced_sym_arrays(sym_data, indices, date_col)
        if sliced:
            sym_arrays[sym] = sliced
    return SymbolArrayStore(frozenset(sym_arrays.keys()), sym_arrays)


def merge_symbol_array_stores(
    left: SymbolArrayStore,
    right: SymbolArrayStore,
) -> SymbolArrayStore:
    """Concatenates per-symbol column arrays from two stores."""
    all_symbols = left.symbols | right.symbols
    merged: dict[str, dict[str, NDArray]] = {}
    for sym in all_symbols:
        cols: set[str] = set()
        if sym in left.sym_arrays:
            cols.update(left.sym_arrays[sym].keys())
        if sym in right.sym_arrays:
            cols.update(right.sym_arrays[sym].keys())
        merged[sym] = {}
        for col in cols:
            parts: list[NDArray] = []
            if sym in left.sym_arrays and col in left.sym_arrays[sym]:
                parts.append(left.sym_arrays[sym][col])
            if sym in right.sym_arrays and col in right.sym_arrays[sym]:
                parts.append(right.sym_arrays[sym][col])
            if len(parts) == 1:
                merged[sym][col] = parts[0]
            else:
                merged[sym][col] = np.concatenate(parts)
    return SymbolArrayStore(all_symbols, merged)


def column_scope_from_frame(
    df: pd.DataFrame,
    sym_col: str = DataCol.SYMBOL.value,
    date_col: str = DataCol.DATE.value,
) -> "ColumnScope":
    """Creates a :class:`ColumnScope` with upfront numpy extraction."""
    return ColumnScope(symbol_array_store_from_frame(df, sym_col, date_col))


def sym_exec_dates_from_store(
    store: SymbolArrayStore,
) -> dict[str, frozenset[np.datetime64]]:
    """Returns per-symbol test dates from a column store."""
    date_col = DataCol.DATE.value
    result: dict[str, frozenset[np.datetime64]] = {}
    for sym in store.symbols:
        dates = store.sym_arrays[sym].get(date_col)
        if dates is not None:
            result[sym] = frozenset(np.asarray(dates, dtype="datetime64[ns]"))
    return result


class ColumnScope:
    """Caches and retrieves column data from a :class:`SymbolArrayStore`.

    Args:
        store: Pre-built numpy column store, or a MultiIndex
            :class:`pandas.DataFrame` (legacy convenience).
    """

    def __init__(
        self,
        store: Union[SymbolArrayStore, pd.DataFrame],
    ):
        if isinstance(store, pd.DataFrame):
            self._store = symbol_array_store_from_frame(store)
        else:
            self._store = store
        self._symbols = self._store.symbols

    @property
    def store(self) -> SymbolArrayStore:
        return self._store

    @property
    def symbols(self) -> frozenset[str]:
        """Symbols held by the underlying store."""
        return self._symbols

    def unique_dates(self) -> NDArray[np.datetime64]:
        """Returns sorted unique dates across every symbol in the store."""
        return self._store.unique_dates()

    def fetch_dict(
        self,
        symbol: str,
        names: Iterable[str],
        end_index: Optional[int] = None,
    ) -> dict[str, Optional[NDArray]]:
        r"""Fetches a ``dict`` of column data for ``symbol``.

        Args:
            symbol: Ticker symbol to query.
            names: Names of columns to query.
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.

        Returns:
            ``dict`` mapping column names to :class:`numpy.ndarray`\ s of
            column values.
        """
        result: dict[str, Optional[NDArray]] = {}
        if not names:
            return result
        if symbol not in self._symbols:
            raise ValueError(f"Symbol not found: {symbol}.")
        sym_data = self._store.sym_arrays[symbol]
        for name in names:
            if name not in sym_data:
                result[name] = None
                continue
            array = sym_data[name]
            result[name] = array if end_index is None else array[:end_index]
        return result

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> Optional[NDArray]:
        """Fetches a :class:`numpy.ndarray` of column data for ``symbol``.

        Args:
            symbol: Ticker symbol to query.
            name: Name of column to query.
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.

        Returns:
            :class:`numpy.ndarray` of column data for every bar until
            ``end_index`` (when specified).
        """
        if symbol not in self._symbols:
            raise ValueError(f"Symbol not found: {symbol}.")
        array = self._store.sym_arrays[symbol].get(name)
        if array is None:
            return None
        return array if end_index is None else array[:end_index]

    def fetch_value(
        self, symbol: str, name: str, end_index: int
    ) -> Optional[float]:
        """Returns the scalar value at ``end_index - 1`` without slicing."""
        if symbol not in self._symbols:
            raise ValueError(f"Symbol not found: {symbol}.")
        array = self._store.sym_arrays[symbol].get(name)
        if array is None:
            return None
        if end_index <= 0:
            raise ValueError(f"{name!r} value not found.")
        if end_index > len(array):
            end_index = len(array)
        return float(array[end_index - 1])

    def bar_data_from_data_columns(
        self, symbol: str, end_index: int
    ) -> BarData:
        """Returns a new :class:`pybroker.common.BarData` instance containing
        column data of default and custom data columns registered with
        :class:`.StaticScope`.

        Args:
            symbol: Ticker symbol to query.
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.
        """
        static_scope = StaticScope.instance()
        bar_data_cols = static_scope._bar_data_cols
        if bar_data_cols is None:
            bar_data_cols = static_scope.ordered_data_cols
        if symbol not in self._symbols:
            raise ValueError(f"Symbol not found: {symbol}.")
        sym_data = self._store.sym_arrays[symbol]
        default_col_data: dict[str, Optional[NDArray]] = {}
        custom_col_data: dict[str, NDArray] = {}
        for col in bar_data_cols:
            array = sym_data.get(col)
            if array is None:
                if col in static_scope.default_data_cols:
                    default_col_data[col] = None
                continue
            sliced = array if end_index is None else array[:end_index]
            if col in static_scope.default_data_cols:
                default_col_data[col] = sliced
            else:
                custom_col_data[col] = sliced
        return BarData(
            **default_col_data,  # type: ignore[arg-type]
            **custom_col_data,  # type: ignore[arg-type]
        )


class IndicatorScope:
    """Caches and retrieves :class:`pybroker.indicator.Indicator` data.

    Args:
        indicator_data: :class:`Mapping` of
            :class:`pybroker.common.IndicatorSymbol` pairs to ``pandas.Series``
            of :class:`pybroker.indicator.Indicator` values.
        filter_dates: Filters :class:`pybroker.indicator.Indicator` data on
            :class:`Sequence` of dates.
    """

    def __init__(
        self,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        filter_dates: Sequence[np.datetime64],
    ):
        self._indicator_data = indicator_data
        self._filter_dates = filter_dates
        self._sym_inds: dict[IndicatorSymbol, NDArray[np.float64]] = {}

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Fetches :class:`pybroker.indicator.Indicator` data.

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.indicator.Indicator` to query.
            end_index: Truncates the array of
                :class:`pybroker.indicator.Indicator` data returned
                (exclusive). If ``None``, then indicator data is not truncated.

        Returns:
            :class:`numpy.ndarray` of :class:`pybroker.indicator.Indicator`
            data for every bar until ``end_index`` (when specified).
        """
        _, token = parse_indicator_interval_name(name)
        if token is not None and end_index is not None:
            # Interval series are indexed by compressed bar, so truncating one
            # with a base bar index would expose future data.
            base_name, _ = parse_indicator_interval_name(name)
            raise ValueError(
                f"Indicator {name!r} is bound to interval {token!r} and "
                "cannot be read from the base context. Use "
                f"ctx.interval({token!r}).indicator({base_name!r}) instead."
            )
        ind_sym = IndicatorSymbol(name, symbol)
        if ind_sym in self._sym_inds:
            cached = self._sym_inds[ind_sym]
            return cached if end_index is None else cached[:end_index]
        if ind_sym not in self._indicator_data:
            raise ValueError(f"Indicator {name!r} not found for {symbol}.")
        raw = self._indicator_data[ind_sym]
        if isinstance(raw, np.ndarray):
            ind_data = np.asarray(raw, dtype=np.float64)
        elif token is not None:
            ind_data = np.asarray(raw.to_numpy(copy=False), dtype=np.float64)
        else:
            if isinstance(raw, pd.Series):
                filter_arr = np.asarray(
                    self._filter_dates, dtype="datetime64[ns]"
                )
                ind_dates = raw.index.to_numpy(dtype="datetime64[ns]")
                ind_values = raw.to_numpy(copy=False)
                mask = np.isin(ind_dates, filter_arr)
                ind_data = np.asarray(ind_values[mask], dtype=np.float64)
            else:
                ind_data = np.asarray(raw, dtype=np.float64)
        self._sym_inds[ind_sym] = ind_data
        return ind_data if end_index is None else ind_data[:end_index]

    def fetch_full(self, symbol: str, name: str) -> NDArray[np.float64]:
        """Fetches the full indicator array without truncation."""
        return self.fetch(symbol, name, end_index=None)

    def has_indicator(self, symbol: str, name: str) -> bool:
        """Whether :class:`pybroker.indicator.Indicator` data is registered
        for ``symbol``.
        """
        return IndicatorSymbol(name, symbol) in self._indicator_data

    def fetch_history(
        self, symbol: str, name: str, dates: NDArray[Any]
    ) -> Optional[NDArray[np.float64]]:
        """Aligns full-history indicator values to ``dates``.

        :meth:`fetch` masks base timeframe indicators to ``filter_dates``, so
        it cannot serve data from before the current window. Lag features need
        history that reaches back into the train window, which this reads from
        the unfiltered series.

        Returns:
            :class:`numpy.ndarray` of values aligned to ``dates``, or ``None``
            when the indicator is not registered for ``symbol``.
        """
        ind_sym = IndicatorSymbol(name, symbol)
        if ind_sym not in self._indicator_data:
            return None
        raw = self._indicator_data[ind_sym]
        if not isinstance(raw, pd.Series):
            return None
        return _model()._indicator_values_for_dates(raw, dates)

    def fetch_value(self, symbol: str, name: str, end_index: int) -> float:
        """Returns the scalar value at ``end_index - 1`` without slicing."""
        array = self.fetch_full(symbol, name)
        if end_index <= 0:
            raise ValueError(f"{name!r} value not found.")
        if end_index > len(array):
            end_index = len(array)
        return float(array[end_index - 1])


def _resolve_lag_cols(
    model_source, trained_model: TrainedModel, model_name: str
) -> Optional[tuple[str, ...]]:
    """Returns the columns to build lag features from, or ``None``.

    Lag features must be built from the same columns at prediction time as at
    training time, otherwise the model is handed a feature matrix of a
    different width than it was fit on. The training-time columns are recorded
    on :attr:`pybroker.common.TrainedModel.lag_columns`; the fallback to
    ``input_cols`` only applies to models cached before that field existed.
    """
    if model_source.lags is None:
        return None
    if trained_model.lag_columns is not None:
        return trained_model.lag_columns
    if trained_model.input_cols is None:
        raise ValueError(
            f"Model {model_name!r} requires input columns from training "
            "before applying lags."
        )
    date_col = DataCol.DATE.value
    return tuple(col for col in trained_model.input_cols if col != date_col)


class IntervalScope:
    """Serves compressed bar and indicator data through alignment maps."""

    def __init__(
        self,
        interval_data: IntervalData,
        ind_scope: IndicatorScope,
        models: Optional[Mapping[ModelSymbol, TrainedModel]] = None,
        test_dates: Optional[Sequence[np.datetime64]] = None,
    ):
        self._interval_data = interval_data
        self._ind_scope = ind_scope
        self._models = models or {}
        self._lag_series_cache: LagSeriesCache = {}
        self._lag_cache_keys: set[tuple[str, str, tuple[str, ...], int]] = (
            set()
        )
        self._test_dates = [] if test_dates is None else test_dates
        self._scope = StaticScope.instance()
        self._bar_cache: dict[
            tuple[str, TimeframeInterval, str], NDArray[Any]
        ] = {}
        self._sym_inputs: dict[ModelSymbol, ModelInput] = {}
        self._sym_preds: dict[ModelSymbol, NDArray] = {}

    def _ensure_lag_cache(
        self,
        symbol: str,
        interval: TimeframeInterval,
        lag_cols: tuple[str, ...],
        lags: int,
    ) -> None:
        model = _model()
        interval = normalize_interval(interval)
        interval_str = format_interval(interval)
        memo_key = (symbol, interval_str, lag_cols, lags)
        if memo_key in self._lag_cache_keys:
            return

        def bars_by_symbol(sym, interval_str=interval_str, interval=interval):
            key = (sym, interval)
            if key not in self._interval_data.compressed:
                return None
            return self._interval_data.compressed[key].bars

        def arrays_by_symbol(sym, interval=interval):
            # Compressed bars carry no indicator values. Interval indicators
            # are not masked to the test window, so fetch_full is already the
            # full compressed history the lag cache needs.
            arrays: dict[str, NDArray[Any]] = {}
            for col in lag_cols:
                name = indicator_interval_name(col, interval)
                if self._ind_scope.has_indicator(sym, name):
                    arrays[col] = self._ind_scope.fetch_full(sym, name)
            return arrays or None

        model.merge_interval_lag_series_cache(
            self._lag_series_cache,
            (symbol,),
            lag_cols,
            lags,
            interval_str,
            bars_by_symbol,
            arrays_by_symbol,
        )
        # Memoized only after a successful build so a failure is retried.
        self._lag_cache_keys.add(memo_key)

    def window_len(self, symbol: str, interval: TimeframeInterval) -> int:
        """Returns the compressed bar count visible in the current window.

        ``completed`` is realigned to the walkforward test window by
        :meth:`pybroker.interval.IntervalData.slice_for_test`, so its last
        entry is the newest compressed bar that completes within the window.
        Model input and predictions are capped here so user callbacks never see
        compressed bars belonging to a future window.
        """
        interval = normalize_interval(interval)
        key = (symbol, interval)
        if key not in self._interval_data.compressed:
            raise ValueError(
                f"Timeframe {interval!r} data not found for {symbol!r}."
            )
        data = self._interval_data.compressed[key]
        if len(data.completed) == 0:
            return 0
        last = int(data.completed[-1])
        if last < 0:
            return 0
        return min(last + 1, len(data.bars.dates))

    def _missing_model_error(self, base_model_name: str, symbol: str) -> str:
        """Returns the error for a model missing on a compressed interval."""
        if self._scope.has_model_source(base_model_name):
            source = self._scope.get_model_source(base_model_name)
            if not isinstance(source, _model().model_trainer_cls):
                return (
                    f"Pretrained model {base_model_name!r} is not trained per "
                    f"interval. Access it on the base timeframe with "
                    f"ctx.preds({base_model_name!r})."
                )
        return f"Model {base_model_name!r} not found for {symbol}."

    def completed_index(
        self, symbol: str, interval: TimeframeInterval, end_index: int
    ) -> int:
        interval = normalize_interval(interval)
        key = (symbol, interval)
        if key not in self._interval_data.compressed:
            raise ValueError(
                f"Timeframe {interval!r} data not found for {symbol!r}."
            )
        completed = self._interval_data.compressed[key].completed
        if end_index <= 0 or len(completed) == 0:
            return -1
        # Clamp instead of allowing a negative index to wrap around to the last
        # completed bar of the window, which would expose future data.
        if end_index > len(completed):
            end_index = len(completed)
        return int(completed[end_index - 1])

    def fetch_bar(
        self,
        symbol: str,
        interval: TimeframeInterval,
        col: str,
        end_index: int,
    ) -> NDArray[Any]:
        interval = normalize_interval(interval)
        cache_key = (symbol, interval, col)
        data: NDArray[Any]
        if cache_key not in self._bar_cache:
            key = (symbol, interval)
            if key not in self._interval_data.compressed:
                raise ValueError(
                    f"Timeframe {interval!r} data not found for {symbol!r}."
                )
            bars = self._interval_data.compressed[key].bars
            if col == DataCol.DATE.value:
                data = bars.dates
            elif col == DataCol.OPEN.value:
                data = bars.open
            elif col == DataCol.HIGH.value:
                data = bars.high
            elif col == DataCol.LOW.value:
                data = bars.low
            elif col == DataCol.CLOSE.value:
                data = bars.close
            elif col == DataCol.VOLUME.value:
                data = bars.volume
            elif col == DataCol.VWAP.value and bars.vwap is not None:
                data = bars.vwap
            elif col in bars.custom:
                data = bars.custom[col]
            else:
                raise ValueError(
                    f"Column {col!r} not found for interval {interval!r}."
                )
            self._bar_cache[cache_key] = data
        data = self._bar_cache[cache_key]
        idx = self.completed_index(symbol, interval, end_index)
        if idx < 0:
            return np.array([], dtype=data.dtype)
        return data[: idx + 1]

    def fetch_indicator(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_name: str,
        end_index: int,
    ) -> NDArray[np.float64]:
        interval = normalize_interval(interval)
        name = indicator_interval_name(base_name, interval)
        values = self._ind_scope.fetch_full(symbol, name)
        idx = self.completed_index(symbol, interval, end_index)
        if idx < 0:
            return np.array([], dtype=np.float64)
        return values[: idx + 1]

    def fetch_input(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_model_name: str,
        end_index: int,
    ) -> pd.DataFrame:
        model = _model()
        interval = normalize_interval(interval)
        idx = self.completed_index(symbol, interval, end_index)
        model_input = self._prepare_full_input(
            symbol, interval, base_model_name
        )
        if idx < 0:
            return model.model_input_to_dataframe(model_input.slice(0))
        return model.model_input_to_dataframe(model_input.slice(idx + 1))

    def fetch_preds(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_model_name: str,
        end_index: int,
    ) -> NDArray:
        interval = normalize_interval(interval)
        model_sym = ModelSymbol(
            model_interval_name(base_model_name, interval), symbol
        )
        trained_model = self._models.get(model_sym)
        if trained_model is None:
            raise ValueError(
                self._missing_model_error(base_model_name, symbol)
            )
        if trained_model.per_bar:
            return self._fetch_preds_per_bar(
                symbol,
                interval,
                base_model_name,
                model_sym,
                trained_model,
                end_index,
            )
        idx = self.completed_index(symbol, interval, end_index)
        if idx < 0:
            return np.array([], dtype=np.float64)
        if model_sym not in self._sym_preds:
            input_ = self._prepare_full_input(
                symbol, interval, base_model_name
            )
            if input_.empty() or not input_.columns:
                raise ValueError(
                    f"No input data found for model {base_model_name!r}. "
                    "Consider passing input_data_fn to pybroker#model() if "
                    "custom columns were registered."
                )
            pred = self._run_predict(trained_model, input_)
            self._sym_preds[model_sym] = pred
        pred = self._sym_preds[model_sym]
        return pred[: idx + 1]

    def _fetch_preds_per_bar(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_model_name: str,
        model_sym: ModelSymbol,
        trained_model: TrainedModel,
        end_index: int,
    ) -> NDArray:
        if model_sym not in self._sym_preds:
            self._sym_preds[model_sym] = np.array([], dtype=np.float64)
        pred = self._sym_preds[model_sym]
        target_len = self.completed_index(symbol, interval, end_index) + 1
        if target_len <= 0:
            return np.array([], dtype=pred.dtype)
        if len(pred) >= target_len:
            return pred[:target_len]
        model_input = self._prepare_full_input(
            symbol, interval, base_model_name
        )
        pred_values = pred.tolist()
        while len(pred_values) < target_len:
            # ``pred_values`` counts compressed bars, so slice the compressed
            # input directly. Routing this through ``completed_index`` would mix
            # it with the base-bar index space.
            sliced = model_input.slice(len(pred_values) + 1)
            scalar = self._run_predict_scalar(trained_model, sliced)
            pred_values.append(scalar)
        pred = np.asarray(pred_values, dtype=np.float64)
        self._sym_preds[model_sym] = pred
        return pred[:target_len]

    @staticmethod
    def _run_predict(
        trained_model: TrainedModel,
        input_: Union[ModelInput, pd.DataFrame],
    ) -> NDArray:
        return PredictionScope._run_predict(trained_model, input_)

    @staticmethod
    def _run_predict_scalar(
        trained_model: TrainedModel,
        input_: Union[ModelInput, pd.DataFrame],
    ) -> float:
        return PredictionScope._run_predict_scalar(trained_model, input_)

    def _prepare_full_input(
        self,
        symbol: str,
        interval: TimeframeInterval,
        base_model_name: str,
    ) -> ModelInput:
        model = _model()
        interval = normalize_interval(interval)
        model_sym = ModelSymbol(
            model_interval_name(base_model_name, interval), symbol
        )
        if model_sym in self._sym_inputs:
            return self._sym_inputs[model_sym]
        if not self._scope.has_model_source(base_model_name):
            raise ValueError(f"Model {base_model_name!r} not found.")
        source = self._scope.get_model_source(base_model_name)
        model_input = self._build_compressed_model_input(
            symbol, interval, source
        )
        if model_sym not in self._models:
            raise ValueError(
                self._missing_model_error(base_model_name, symbol)
            )
        trained_model = self._models[model_sym]
        lag_cols = _resolve_lag_cols(
            source, trained_model, model_sym.model_name
        )
        # Lag features are attached before narrowing to input_cols so that they
        # can be built from columns the model does not take as input.
        if lag_cols is not None:
            assert source.lags is not None
            for lag_col in lag_cols:
                if lag_col not in model_input:
                    raise ValueError(
                        f"Missing lag column {lag_col!r} for input data to "
                        f"model {model_sym.model_name!r}."
                    )
            self._ensure_lag_cache(symbol, interval, lag_cols, source.lags)
            model.apply_lags_to_model_input(
                model_input,
                lag_cols,
                source.lags,
                self._lag_series_cache,
                symbol,
                np.asarray(
                    self._interval_data.compressed[
                        (symbol, interval)
                    ].bars.dates
                )[: self.window_len(symbol, interval)],
                format_interval(interval),
            )
        if trained_model.input_cols is not None:
            for input_col in trained_model.input_cols:
                if input_col not in model_input:
                    raise ValueError(
                        f"Missing column {input_col!r} for input data to "
                        f"model {model_sym.model_name!r}."
                    )
            model_input = model_input.select_columns(trained_model.input_cols)
        if not trained_model.input_cols or source._input_data_fn:
            model_input = model.apply_prepare_input_data(
                model_input, source.prepare_input_data
            )
        self._sym_inputs[model_sym] = model_input
        return model_input

    def _build_compressed_model_input(
        self,
        symbol: str,
        interval: TimeframeInterval,
        source,
    ) -> ModelInput:
        model = _model()
        interval = normalize_interval(interval)
        key = (symbol, interval)
        if key not in self._interval_data.compressed:
            raise ValueError(
                f"Timeframe {interval!r} data not found for {symbol!r}."
            )
        bars = self._interval_data.compressed[key].bars
        # Cap at the window so cross-row user transforms (normalization,
        # ranking, fillna(mean)) cannot read compressed bars from a future
        # walkforward window.
        cap = self.window_len(symbol, interval)
        dates = bars.dates[:cap]
        arrays: dict[str, NDArray[Any]] = {
            DataCol.DATE.value: dates,
            DataCol.OPEN.value: bars.open[:cap],
            DataCol.HIGH.value: bars.high[:cap],
            DataCol.LOW.value: bars.low[:cap],
            DataCol.CLOSE.value: bars.close[:cap],
            DataCol.VOLUME.value: bars.volume[:cap],
        }
        if bars.vwap is not None:
            arrays[DataCol.VWAP.value] = bars.vwap[:cap]
        for col in sorted(self._scope.custom_data_cols):
            if col in bars.custom:
                arrays[col] = bars.custom[col][:cap]
        for ind_name in source.indicators:
            arrays[ind_name] = self._ind_scope.fetch_full(
                symbol, indicator_interval_name(ind_name, interval)
            )[:cap]
        columns = tuple(arrays.keys())
        return model.model_input_cls(columns, arrays, dates)

    def clear_cache(self):
        """Drops every cached array.

        Compressed data is immutable for the lifetime of a scope (a new one is
        built per walkforward window), and each cache is keyed independently of
        the current bar, so this is only for tearing a scope down -- calling it
        per bar would rebuild model input and rerun ``predict`` on every bar.
        """
        self._bar_cache.clear()
        self._sym_inputs.clear()
        self._sym_preds.clear()
        self._lag_series_cache.clear()
        self._lag_cache_keys.clear()


class ModelInputScope:
    r"""Caches and retrieves model input data.

    Args:
        col_scope: :class:`.ColumnScope`.
        ind_scope: :class:`.IndicatorScope`.
        models: :class:`Mapping` of
            :class:`pybroker.common.ModelSymbol` pairs to
            :class:`pybroker.common.TrainedModel`\ s.
    """

    def __init__(
        self,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        models: Mapping[ModelSymbol, TrainedModel],
        history_col_scope: Optional["ColumnScope"] = None,
        test_dates: Optional[Sequence[np.datetime64]] = None,
    ):
        self._col_scope = col_scope
        self._ind_scope = ind_scope
        self._models = models
        self._history_col_scope = history_col_scope
        self._lag_series_cache: LagSeriesCache = {}
        self._history_dates: dict[str, np.ndarray] = {}
        self._test_dates = [] if test_dates is None else test_dates
        self._sym_inputs: dict[ModelSymbol, ModelInput] = {}
        self._lag_cache_depth: dict[tuple[str, str], int] = {}
        self._scope = StaticScope.instance()

    def _ensure_lag_cache(
        self,
        symbol: str,
        lag_cols: tuple[str, ...],
        lags: int,
    ) -> None:
        # Track the depth built per column: a shallower cache entry from
        # another model would otherwise be reused for a deeper request.
        missing = tuple(
            col
            for col in lag_cols
            if self._lag_cache_depth.get((symbol, col), -1) < lags
        )
        if not missing:
            return
        if self._history_col_scope is None:
            raise ValueError(
                f"History data required to compute lags for {symbol!r}."
            )
        dates = self._history_dates.get(symbol)
        if dates is not None:
            self._merge_lag_cache(symbol, missing, lags, dates)
            return
        dates = self._history_col_scope.fetch(symbol, DataCol.DATE.value)
        if dates is None:
            raise ValueError(f"History dates not found for {symbol!r}.")
        self._history_dates[symbol] = dates
        self._merge_lag_cache(symbol, missing, lags, dates)

    def _merge_lag_cache(
        self,
        symbol: str,
        lag_cols: tuple[str, ...],
        lags: int,
        dates: NDArray[Any],
    ) -> None:
        model = _model()
        assert self._history_col_scope is not None
        column_arrays: dict[str, NDArray[Any]] = {}
        for col in lag_cols:
            col_data = self._history_col_scope.fetch(symbol, col)
            if col_data is None:
                # A ColumnScope serves data columns only, so an indicator lag
                # column is read from full indicator history instead. Using
                # IndicatorScope.fetch here would mask to the test window and
                # leave test bar 0 without the train window's lag values.
                col_data = self._ind_scope.fetch_history(symbol, col, dates)
            if col_data is None:
                raise ValueError(
                    f"History column {col!r} not found for {symbol!r}. "
                    "lag_cols must name a data column or an Indicator "
                    "registered on the model."
                )
            column_arrays[col] = col_data
        model.merge_lag_series_cache_from_arrays(
            self._lag_series_cache,
            symbol,
            lag_cols,
            lags,
            dates,
            column_arrays,
        )
        for col in lag_cols:
            self._lag_cache_depth[(symbol, col)] = max(
                self._lag_cache_depth.get((symbol, col), -1), lags
            )

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetches model input data.

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.model.ModelSource` to query input
                data.
            end_index: Truncates the array of model input data returned
                (exclusive). If ``None``, then model input data is not
                truncated.

        Returns:
            :class:`pandas.DataFrame` of model input data for every bar until
            ``end_index`` (when specified).
        """
        model = _model()
        model_input = self._fetch_model_input(symbol, name, end_index)
        return model.model_input_to_dataframe(model_input)

    def fetch_model_input(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> ModelInput:
        """Fetches model input as internal :class:`ModelInput` (no DataFrame).

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.model.ModelSource` to query input
                data.
            end_index: Truncates the array of model input data returned
                (exclusive). If ``None``, then model input data is not
                truncated.

        Returns:
            :class:`pybroker.model.ModelInput` for every bar until
            ``end_index`` (when specified).
        """
        return self._fetch_model_input(symbol, name, end_index)

    def _fetch_model_input(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> ModelInput:
        model = _model()
        model_sym = ModelSymbol(name, symbol)
        if model_sym in self._sym_inputs:
            model_input = self._sym_inputs[model_sym]
            return (
                model_input
                if end_index is None
                else model_input.slice(end_index)
            )
        if symbol not in self._col_scope._symbols:
            raise ValueError(f"Symbol not found: {symbol}.")
        if not self._scope.has_model_source(name):
            raise ValueError(f"Model {name!r} not found.")
        model_source = self._scope.get_model_source(name)
        if model_sym not in self._models:
            raise ValueError(f"Model {name!r} not found for {symbol}.")
        trained_model = self._models[model_sym]
        date_col = DataCol.DATE.value
        ind_names = self._scope.get_indicator_names(name)
        lag_cols = _resolve_lag_cols(model_source, trained_model, name)
        if (
            trained_model.input_cols is not None
            and not model_source._input_data_fn
        ):
            # Lag columns are needed to build the lag features even when they
            # are not part of the model's input columns.
            needed = [date_col, *trained_model.input_cols]
            if lag_cols is not None:
                needed.extend(lag_cols)
            needed_set = frozenset(needed)
            ordered = self._scope.ordered_data_cols
            # Registered columns first, in their canonical order, then any
            # remaining names in declaration order. Both are deterministic:
            # iterating the registered column set directly is not.
            data_cols: Iterable[str] = tuple(
                dict.fromkeys(
                    [col for col in ordered if col in needed_set] + needed
                )
            )
        else:
            data_cols = self._scope.ordered_data_cols
        input_: dict[str, NDArray[Any]] = {}
        for col in data_cols:
            data = self._col_scope.fetch(symbol, col)
            if data is not None:
                input_[col] = data
        for ind_name in ind_names:
            input_[ind_name] = self._ind_scope.fetch(symbol, ind_name)
        row_dates = input_.get(date_col)
        if row_dates is None:
            row_dates = self._col_scope.fetch(symbol, date_col)
        assert row_dates is not None
        columns = tuple(input_.keys())
        model_input = model.model_input_cls(columns, input_, row_dates)
        # Lag features are attached before narrowing to input_cols so that
        # they can be built from columns the model does not take as input.
        if lag_cols is not None:
            assert model_source.lags is not None
            for lag_col in lag_cols:
                if lag_col not in model_input:
                    raise ValueError(
                        f"Missing lag column {lag_col!r} for input data to "
                        f"model {model_sym.model_name!r}."
                    )
            self._ensure_lag_cache(symbol, lag_cols, model_source.lags)
            model.apply_lags_to_model_input(
                model_input,
                lag_cols,
                model_source.lags,
                self._lag_series_cache,
                symbol,
                self._history_dates[symbol],
            )
        if trained_model.input_cols is not None:
            for input_col in trained_model.input_cols:
                if input_col not in model_input:
                    raise ValueError(
                        f"Missing column {input_col!r} for input data to "
                        f"model {model_sym.model_name!r}."
                    )
            model_input = model_input.select_columns(trained_model.input_cols)
        if not trained_model.input_cols or model_source._input_data_fn:
            model_input = model.apply_prepare_input_data(
                model_input, model_source.prepare_input_data
            )
        self._sym_inputs[model_sym] = model_input
        return (
            model_input if end_index is None else model_input.slice(end_index)
        )


class PredictionScope:
    r"""Caches and retrieves model predictions.

    Args:
        models: :class:`Mapping` of
            :class:`pybroker.common.ModelSymbol` pairs to
            :class:`pybroker.common.TrainedModel`\ s.
        input_scope: :class:`.ModelInputScope`.
    """

    def __init__(
        self,
        models: Mapping[ModelSymbol, TrainedModel],
        input_scope: ModelInputScope,
    ):
        self._models = models
        self._input_scope = input_scope
        self._sym_preds: dict[ModelSymbol, NDArray] = {}

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> NDArray:
        """Fetches model predictions.

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.model.ModelSource` that made the
                predictions.
            end_index: Truncates the array of predictions returned (exclusive).
                If ``None``, then predictions are not truncated.

        Returns:
            :class:`numpy.ndarray` of model predictions for every bar until
            ``end_index`` (when specified).
        """
        model_sym = ModelSymbol(name, symbol)
        trained_model = self._models.get(model_sym)
        if trained_model is not None and trained_model.per_bar:
            return self._fetch_per_bar(
                symbol, name, model_sym, trained_model, end_index
            )
        if model_sym in self._sym_preds:
            return self._sym_preds[model_sym][:end_index]
        model_input = self._input_scope._fetch_model_input(symbol, name)
        if model_input.empty() or not model_input.columns:
            raise ValueError(
                f"No input data found for model {name!r}. Consider "
                "passing input_data_fn to pybroker#model() if custom columns "
                "were registered."
            )
        if model_sym not in self._models:
            raise ValueError(f"Model {name!r} not found for {symbol}.")
        trained_model = self._models[model_sym]
        pred = self._run_predict(trained_model, model_input)
        self._sym_preds[model_sym] = pred
        return pred[:end_index]

    def _fetch_per_bar(
        self,
        symbol: str,
        name: str,
        model_sym: ModelSymbol,
        trained_model: TrainedModel,
        end_index: Optional[int],
    ) -> NDArray:
        if model_sym not in self._sym_preds:
            self._sym_preds[model_sym] = np.array([], dtype=np.float64)
        pred = self._sym_preds[model_sym]
        if end_index is None:
            input_full = self._input_scope._fetch_model_input(symbol, name)
            target_len = len(input_full.dates)
        else:
            target_len = end_index
        if len(pred) >= target_len:
            return pred if end_index is None else pred[:end_index]
        pred_values = pred.tolist()
        while len(pred_values) < target_len:
            bar_end_index = len(pred_values) + 1
            model_input = self._input_scope._fetch_model_input(
                symbol, name, bar_end_index
            )
            scalar = self._run_predict_scalar(trained_model, model_input)
            pred_values.append(scalar)
        pred = np.asarray(pred_values, dtype=np.float64)
        self._sym_preds[model_sym] = pred
        return pred if end_index is None else pred[:end_index]

    @staticmethod
    def _predict_input_df(
        input_: Union[ModelInput, pd.DataFrame],
    ) -> pd.DataFrame:
        model = _model()
        if isinstance(input_, model.model_input_cls):
            return model.model_input_to_dataframe(input_)
        return input_

    @staticmethod
    def _run_predict(
        trained_model: TrainedModel,
        input_: Union[ModelInput, pd.DataFrame],
    ) -> NDArray:
        input_df = PredictionScope._predict_input_df(input_)
        if trained_model.predict_fn is not None:
            pred = trained_model.predict_fn(trained_model.instance, input_df)
        else:
            predict_fn = getattr(trained_model.instance, "predict", None)
            if predict_fn is not None and callable(predict_fn):
                pred = trained_model.instance.predict(input_df)
            else:
                raise ValueError(
                    f"Model instance trained for {trained_model.name!r} "
                    "does not define a predict function. Please pass a "
                    "predict_fn to pybroker.model()."
                )
        pred_arr = np.asarray(pred)
        if pred_arr.ndim == 0:
            return np.array([pred_arr.item()])
        if len(pred_arr.shape) > 1:
            pred_arr = np.squeeze(pred_arr)
        return pred_arr

    @staticmethod
    def _run_predict_scalar(
        trained_model: TrainedModel,
        input_: Union[ModelInput, pd.DataFrame],
    ) -> float:
        n_rows = len(input_)
        pred = PredictionScope._run_predict(trained_model, input_)
        flat = np.asarray(pred).reshape(-1)
        if not flat.size:
            raise ValueError(
                f"predict_fn for per_bar model {trained_model.name!r} "
                "returned no predictions. Expected a scalar prediction for "
                "the current bar."
            )
        if flat.size != 1 and flat.size != n_rows:
            raise ValueError(
                f"predict_fn for per_bar model {trained_model.name!r} "
                f"returned {flat.size} predictions for {n_rows} input rows. "
                "Expected a scalar prediction for the current bar, e.g. "
                "return preds[-1]."
            )
        # The current bar is the last row of the input, so take the last
        # prediction when predict_fn returns one value per row.
        return float(flat[-1])


class PriceScope:
    """Retrieves most recent prices."""

    def __init__(
        self,
        col_scope: ColumnScope,
        sym_end_index: Mapping[str, int],
        round_fill_price: bool,
    ):
        self._col_scope = col_scope
        self._sym_end_index = sym_end_index
        self._round_fill_price = round_fill_price
        self._bar_cache: dict[tuple[str, PriceType], float] = {}

    def reset_bar(self) -> None:
        """Clears the per-bar OHLC cache. Call once at the start of each bar."""
        self._bar_cache.clear()

    def has_bar(self, symbol: str) -> bool:
        """Returns whether ``symbol`` has a bar that can be priced.

        ``False`` for a symbol absent from the current test window -- one that
        stopped trading, or that a :class:`pybroker.common.SymbolSelector`
        dropped -- whose prices would otherwise raise.
        """
        return self._sym_end_index.get(symbol, 0) > 0

    def _column_value(self, symbol: str, col: str) -> float:
        end_index = self._sym_end_index[symbol]
        if end_index <= 0:
            raise ValueError(f"{col} price not found.")
        sym_data = self._col_scope.store.sym_arrays[symbol]
        if col not in sym_data:
            raise ValueError(f"{col} price not found.")
        array = sym_data[col]
        if end_index > len(array):
            end_index = len(array)
        return float(array[end_index - 1])

    def _round_float(self, fill_price: float) -> float:
        if not self._round_fill_price:
            return fill_price
        if fill_price >= 0.0:
            return int(fill_price * 100.0 + 0.5) / 100.0
        return -int(-fill_price * 100.0 + 0.5) / 100.0

    def _fetch_price_type(self, symbol: str, price: PriceType) -> float:
        key = (symbol, price)
        cached = self._bar_cache.get(key)
        if cached is not None:
            return cached
        if price is _PRICE_OPEN:
            fill_price = self._column_value(symbol, _COL_OPEN)
        elif price is _PRICE_HIGH:
            fill_price = self._column_value(symbol, _COL_HIGH)
        elif price is _PRICE_LOW:
            fill_price = self._column_value(symbol, _COL_LOW)
        elif price is _PRICE_CLOSE:
            fill_price = self._column_value(symbol, _COL_CLOSE)
        elif price is _PRICE_MIDDLE:
            low = self._fetch_price_type(symbol, _PRICE_LOW)
            high = self._fetch_price_type(symbol, _PRICE_HIGH)
            fill_price = low + (high - low) / 2.0
        elif price is _PRICE_AVERAGE:
            open_ = self._fetch_price_type(symbol, _PRICE_OPEN)
            low = self._fetch_price_type(symbol, _PRICE_LOW)
            high = self._fetch_price_type(symbol, _PRICE_HIGH)
            close = self._fetch_price_type(symbol, _PRICE_CLOSE)
            fill_price = (open_ + low + high + close) / 4.0
        else:
            raise ValueError(f"Unknown price: {price!r}")
        self._bar_cache[key] = fill_price
        return fill_price

    def fetch_float(
        self,
        symbol: str,
        price: Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ],
    ) -> float:
        """Returns a bar price as ``float`` using the per-bar cache when possible."""
        if isinstance(price, PriceType):
            fill_price = self._fetch_price_type(symbol, price)
        elif isinstance(price, (int, float, np.floating, Decimal)):
            fill_price = float(price)
        elif callable(price):
            bar_data = self._col_scope.bar_data_from_data_columns(
                symbol, self._sym_end_index[symbol]
            )
            fill_price = float(price(symbol, bar_data))
        else:
            raise ValueError(f"Unknown price: {type(price)!r}")
        return self._round_float(fill_price)

    def fetch_bar_ohlc(
        self,
        symbol: str,
        date: np.datetime64,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Returns ``(close, low, high)`` for ``symbol`` on ``date``, or Nones."""
        end_index = self._sym_end_index[symbol]
        if end_index <= 0:
            return None, None, None
        cols = self._col_scope.fetch_dict(symbol, _BAR_OHLC_COLS)
        date_arr = cols[_COL_DATE]
        if date_arr is None:
            return None, None, None
        if end_index > len(date_arr):
            end_index = len(date_arr)
        idx = end_index - 1
        if date_arr[idx] != date:
            return None, None, None
        close = low = high = None
        close_arr = cols[_COL_CLOSE]
        if close_arr is not None:
            close = float(close_arr[idx])
        low_arr = cols[_COL_LOW]
        if low_arr is not None:
            low = float(low_arr[idx])
        high_arr = cols[_COL_HIGH]
        if high_arr is not None:
            high = float(high_arr[idx])
        return close, low, high

    def fetch(
        self,
        symbol: str,
        price: Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ],
    ) -> Decimal:
        return to_decimal(self.fetch_float(symbol, price))


class PendingOrder(NamedTuple):
    """Holds data for a pending order.

    Attributes:
        id: Unique ID.
        type: Type of order, either ``buy`` or ``sell``.
        symbol: Ticker symbol of the order.
        created: Date the order was created.
        exec_date: Date the order will be executed.
        shares: Number of shares to be bought or sold.
        limit_price: Limit price to use for the order.
        fill_price: Price that the order will be filled at.
        exec_bar: Symbol bar index when the order will first be attempted.
        timeout_bars: Number of bars to retry after the first attempt.
            ``None`` for a single attempt, ``-1`` for indefinite persistence,
            or a positive integer for a limited number of retry bars.
        stops: Stops to attach when the order is filled.
    """

    id: int
    type: Literal["buy", "sell"]
    symbol: str
    created: np.datetime64
    exec_date: np.datetime64
    shares: Decimal
    limit_price: Optional[Decimal]
    fill_price: Union[
        int,
        float,
        np.floating,
        Decimal,
        PriceType,
        Callable[[str, BarData], Union[int, float, Decimal]],
    ]
    exec_bar: int
    timeout_bars: Optional[int]
    stops: Optional[frozenset["Stop"]]


class PendingOrderScope:
    r"""Stores :class:`.PendingOrder`\ s"""

    _order_id: int = 0

    def __init__(self):
        self._orders: dict[int, PendingOrder] = {}
        self._sym_orders: dict[str, set[PendingOrder]] = defaultdict(set)

    def contains(self, order_id: int) -> bool:
        """Returns whether a :class:`.PendingOrder` exists with
        ``order_id``.
        """
        return order_id in self._orders

    def has_orders(self) -> bool:
        """Returns whether any pending orders exist."""
        return bool(self._orders)

    def get(self, order_id: int) -> Optional[PendingOrder]:
        """Returns a :class:`.PendingOrder` with ``order_id``."""
        return self._orders.get(order_id)

    def add(
        self,
        type: Literal["buy", "sell"],
        symbol: str,
        created: np.datetime64,
        exec_date: np.datetime64,
        shares: Decimal,
        limit_price: Optional[Decimal],
        fill_price: Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ],
        exec_bar: int,
        timeout_bars: Optional[int],
        stops: Optional[frozenset["Stop"]] = None,
    ) -> int:
        """Creates a :class:`.PendingOrder`.

        Args:
            type: Type of order, either ``buy`` or ``sell``.
            symbol: Ticker symbol of the order.
            created: Date the order was created.
            exec_date: Date the order will be executed.
            shares: Number of shares to be bought or sold.
            limit_price: Limit price to use for the order.
            fill_price: Price that the order will be filled at.
            exec_bar: Symbol bar index when the order will first be attempted.
            timeout_bars: Number of bars to retry after the first attempt.
            stops: Stops to attach when the order is filled.

        Returns:
            ID of the :class:`.PendingOrder`.
        """
        self._order_id += 1
        order = PendingOrder(
            id=self._order_id,
            type=type,
            symbol=symbol,
            created=created,
            exec_date=exec_date,
            shares=shares,
            limit_price=limit_price,
            fill_price=fill_price,
            exec_bar=exec_bar,
            timeout_bars=timeout_bars,
            stops=stops,
        )
        self._orders[self._order_id] = order
        self._sym_orders[symbol].add(order)
        return order.id

    def remove(self, order_id: int) -> bool:
        """Removes a :class:`.PendingOrder` with ``order_id```."""
        if order_id in self._orders:
            order = self._orders[order_id]
            del self._orders[order_id]
            if (
                order.symbol in self._sym_orders
                and order in self._sym_orders[order.symbol]
            ):
                self._sym_orders[order.symbol].remove(order)
            return True
        return False

    def remove_all(self, symbol: Optional[str] = None):
        r"""Removes all :class:`.PendingOrder`\ s."""
        if symbol is None:
            cancel_ids = tuple(self._orders.keys())
            for order_id in cancel_ids:
                self.remove(order_id)
        elif symbol in self._sym_orders:
            cancel_ids = tuple(order.id for order in self._sym_orders[symbol])
            for order_id in cancel_ids:
                self.remove(order_id)

    def orders(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[int] = None,
    ) -> Iterable[PendingOrder]:
        r"""Returns an :class:`Iterable` of :class:`.PendingOrder`\ s.

        Args:
            symbol: Filter by ticker symbol.
            order_id: Filter by order ID.
        """
        if order_id is not None and symbol is not None:
            order = self._orders.get(order_id)
            if order is not None and order.symbol == symbol:
                return [order]
            return []
        elif order_id is not None:
            order = self._orders.get(order_id)
            if order is not None:
                return [order]
            return []
        elif symbol is not None:
            if symbol not in self._sym_orders:
                return []
            return self._sym_orders[symbol]
        else:
            return self._orders.values()


def get_signals(
    symbols: Iterable[str],
    col_scope: ColumnScope,
    ind_scope: IndicatorScope,
    pred_scope: PredictionScope,
) -> dict[str, pd.DataFrame]:
    r"""Retrieves dictionary of :class:`pandas.DataFrame`\ s
    containing bar data, indicator data, and model predictions for each symbol.
    """
    static_scope = StaticScope.instance()
    cols = static_scope.ordered_data_cols
    inds = static_scope._indicators.keys()
    models = static_scope._model_sources.keys()
    dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        dates_arr = col_scope.fetch(sym, DataCol.DATE.value)
        data: dict[str, Any] = {DataCol.DATE.value: dates_arr}
        for col in cols:
            if col == DataCol.DATE.value:
                continue
            data[col] = col_scope.fetch(sym, col)
        for ind in inds:
            try:
                data[ind] = ind_scope.fetch(sym, ind)
            except ValueError:
                continue
        for model in models:
            try:
                data[f"{model}_pred"] = pred_scope.fetch(sym, model)
            except ValueError:
                continue
        dfs[sym] = pd.DataFrame(data)
    return dfs
