"""Unit tests for model lag transform metadata."""

import numpy as np
import pandas as pd
import pytest
from pybroker.common import DataCol, ModelSymbol, TrainedModel
from pybroker.timeseries import (
    LagSeriesKey,
    apply_lags_to_model_input,
    build_lag_feature_matrix,
    build_lag_feature_matrix_pooled,
    compute_lag_series_cache,
    feature_matrix_from_model_input,
    model_input_from_frame,
    model_input_lags,
    symbol_history_arrays,
    _build_stacked_lags,
)
from pybroker.model import model
from pybroker.scope import (
    ColumnScope,
    IndicatorScope,
    ModelInputScope,
    StaticScope,
)
from .fixtures import *  # noqa: F401


@pytest.fixture(autouse=True)
def clear_model_sources():
    scope = StaticScope.instance()
    scope._model_sources.clear()
    yield
    scope._model_sources.clear()


class TestLagHelpers:
    def test_lags_must_be_positive(self):
        with pytest.raises(
            ValueError, match="lags must be a positive integer"
        ):
            model(
                "m",
                lambda s, t, u: None,
                lags=0,
            )


class TestLagData:
    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame(
            {
                DataCol.SYMBOL.value: ["SPY"] * 5 + ["AAPL"] * 5,
                DataCol.DATE.value: pd.date_range(
                    "2020-01-01", periods=5
                ).tolist()
                * 2,
                DataCol.CLOSE.value: [
                    100,
                    101,
                    102,
                    103,
                    104,
                    200,
                    201,
                    202,
                    203,
                    204,
                ],
            }
        )

    def test_compute_lag_series_cache_per_symbol(self, sample_df):
        cache = compute_lag_series_cache(
            sample_df, ("SPY", "AAPL"), ("close",), 2
        )
        spy_lag1 = cache[LagSeriesKey("SPY", "close", 1)]
        aapl_lag1 = cache[LagSeriesKey("AAPL", "close", 1)]
        assert np.isnan(spy_lag1[0])
        assert spy_lag1[1] == 100
        assert np.isnan(aapl_lag1[0])
        assert aapl_lag1[1] == 200

    def test_apply_lags_no_column_expansion(self, sample_df):
        cache = compute_lag_series_cache(sample_df, ("SPY",), ("close",), 2)
        sym_df = sample_df[sample_df[DataCol.SYMBOL.value] == "SPY"].drop(
            columns=DataCol.SYMBOL.value
        )
        history_dates_arr, _ = symbol_history_arrays(
            sample_df, "SPY", ("close",)
        )
        model_input = model_input_from_frame(sym_df)
        apply_lags_to_model_input(
            model_input,
            ("close",),
            2,
            cache,
            "SPY",
            history_dates_arr,
        )
        assert model_input.columns == tuple(sym_df.columns)
        matrix = feature_matrix_from_model_input(model_input)
        assert matrix is not None
        assert matrix.shape == (len(sym_df), 3)
        assert model_input_lags(model_input) == 2

    def test_drop_lag_warmup(self, sample_df):
        cache = compute_lag_series_cache(sample_df, ("SPY",), ("close",), 2)
        sym_df = sample_df[sample_df[DataCol.SYMBOL.value] == "SPY"].drop(
            columns=DataCol.SYMBOL.value
        )
        history_dates_arr, _ = symbol_history_arrays(
            sample_df, "SPY", ("close",)
        )
        model_input = model_input_from_frame(sym_df)
        apply_lags_to_model_input(
            model_input,
            ("close",),
            2,
            cache,
            "SPY",
            history_dates_arr,
        )
        trimmed = model_input.drop_lag_warmup()
        assert len(trimmed) == 3
        assert trimmed["close"][0] == 102
        matrix = feature_matrix_from_model_input(trimmed)
        assert matrix is not None
        assert not np.isnan(matrix).any()


class TestModelInputScopeLags:
    @pytest.fixture()
    def lag_setup(self, data_source_df, symbols):
        scope = StaticScope.instance()
        scope._model_sources.clear()
        seen_cols = []

        def input_data_fn(df):
            seen_cols.append(list(df.columns))
            return df

        model(
            "lag_model",
            lambda s, t, u: (object(), ("close",)),
            lags=2,
            input_data_fn=input_data_fn,
        )
        sym = symbols[0]
        sym_df = data_source_df[data_source_df["symbol"] == sym]
        test_df = sym_df.iloc[len(sym_df) // 2 :].set_index(["symbol", "date"])
        dates = sorted(test_df.index.get_level_values(1).unique())
        trained = {
            ModelSymbol("lag_model", sym): TrainedModel(
                name="lag_model",
                instance=object(),
                predict_fn=None,
                input_cols=("close",),
            )
        }
        col_scope = ColumnScope(test_df)
        history_col_scope = ColumnScope(
            sym_df.set_index(["symbol", "date"]).sort_index()
        )
        ind_scope = IndicatorScope({}, dates)
        input_scope = ModelInputScope(
            col_scope, ind_scope, trained, history_col_scope, dates
        )
        return input_scope, sym, dates, seen_cols, sym_df

    def test_lag_features_before_input_data_fn(self, lag_setup):
        input_scope, sym, _, seen_cols, _ = lag_setup
        model_input = input_scope.fetch(sym, "lag_model")
        assert seen_cols[0] == ["close"]
        matrix = feature_matrix_from_model_input(model_input)
        assert matrix is not None
        assert matrix.shape[1] == 3

    def test_test_bar_zero_has_train_tail_lags(self, lag_setup):
        input_scope, sym, _, _, sym_df = lag_setup
        train_tail = sym_df.iloc[len(sym_df) // 2 - 1]
        model_input = input_scope.fetch(sym, "lag_model", end_index=1)
        matrix = feature_matrix_from_model_input(model_input)
        assert matrix is not None
        assert matrix[0, 0] == model_input["close"].iloc[0]
        assert matrix[0, 1] == train_tail["close"]
        assert matrix[0, 2] == sym_df.iloc[len(sym_df) // 2 - 2]["close"]

    def test_slice_preserves_lag_features(self, lag_setup):
        input_scope, sym, _, _, _ = lag_setup
        full = input_scope.fetch(sym, "lag_model")
        sliced = input_scope.fetch(sym, "lag_model", end_index=2)
        full_matrix = feature_matrix_from_model_input(full)
        sliced_matrix = feature_matrix_from_model_input(sliced)
        assert full_matrix is not None
        assert sliced_matrix is not None
        np.testing.assert_array_equal(sliced_matrix, full_matrix[:2])


def _build_stacked_lags_reference(values: np.ndarray, lags: int) -> np.ndarray:
    """Pure-Python reference for stacked lag history."""
    n = len(values)
    stacked = np.empty((lags + 1, n), dtype=np.float64)
    stacked[0] = values
    for lag in range(1, lags + 1):
        row = stacked[lag]
        row[:lag] = np.nan
        row[lag:] = values[:-lag]
    return stacked


def _build_lag_feature_matrix_reference(
    symbol: str,
    columns: tuple[str, ...],
    lags: int,
    base_arrays: dict[str, np.ndarray],
    row_dates: np.ndarray,
    history_dates: np.ndarray,
    lag_cache: dict,
    interval=None,
) -> np.ndarray:
    """Pure-Python reference for lag feature matrix assembly."""
    n_rows = len(row_dates)
    n_features = len(columns) * (lags + 1)
    if n_rows == 0:
        return np.empty((0, n_features), dtype=np.float64)
    offset = int(np.searchsorted(history_dates, row_dates[0]))
    end = offset + n_rows
    if end > len(history_dates):
        raise ValueError("Row dates exceed available history.")
    if not np.array_equal(history_dates[offset:end], row_dates):
        raise ValueError("Row dates are not contiguous in history.")
    matrix = np.empty((n_rows, n_features), dtype=np.float64)
    col_idx = 0
    for col in columns:
        stacked = lag_cache[LagSeriesKey(symbol, col, 0, interval)]
        matrix[:, col_idx : col_idx + lags + 1] = stacked[
            :, offset : offset + n_rows
        ].T
        matrix[:, col_idx] = np.ascontiguousarray(
            base_arrays[col], dtype=np.float64
        )
        col_idx += lags + 1
    return matrix


class TestLagNumbaKernels:
    @pytest.mark.parametrize(
        "values, lags",
        [
            (np.array([], dtype=np.float64), 1),
            (np.array([1.0], dtype=np.float64), 1),
            (np.array([1.0, 2.0], dtype=np.float64), 5),
            (np.linspace(1.0, 20.0, 20), 1),
            (np.linspace(1.0, 50.0, 50), 5),
        ],
    )
    def test_build_stacked_lags_matches_reference(self, values, lags):
        expected = _build_stacked_lags_reference(values, lags)
        actual = _build_stacked_lags(values, lags)
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=0, equal_nan=True
        )

    @pytest.fixture()
    def multi_col_df(self):
        return pd.DataFrame(
            {
                DataCol.SYMBOL.value: ["SPY"] * 6 + ["AAPL"] * 6,
                DataCol.DATE.value: pd.date_range(
                    "2020-01-01", periods=6
                ).tolist()
                * 2,
                DataCol.CLOSE.value: [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    200,
                    201,
                    202,
                    203,
                    204,
                    205,
                ],
                DataCol.OPEN.value: [
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    199,
                    200,
                    201,
                    202,
                    203,
                    204,
                ],
            }
        )

    def test_build_lag_feature_matrix_matches_reference(self, multi_col_df):
        lags = 2
        columns = ("close", "open")
        cache = compute_lag_series_cache(multi_col_df, ("SPY",), columns, lags)
        sym_df = multi_col_df[
            multi_col_df[DataCol.SYMBOL.value] == "SPY"
        ].drop(columns=DataCol.SYMBOL.value)
        history_dates, _ = symbol_history_arrays(multi_col_df, "SPY", columns)
        base_arrays = {
            col: sym_df[col].to_numpy(dtype=np.float64) for col in columns
        }
        row_dates = sym_df[DataCol.DATE.value].to_numpy()
        expected = _build_lag_feature_matrix_reference(
            "SPY",
            columns,
            lags,
            base_arrays,
            row_dates,
            history_dates,
            cache,
        )
        actual = build_lag_feature_matrix(
            "SPY",
            columns,
            lags,
            base_arrays,
            row_dates,
            history_dates,
            cache,
        )
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=0, equal_nan=True
        )

    def test_build_lag_feature_matrix_pooled_matches_reference(
        self, multi_col_df
    ):
        lags = 2
        columns = ("close", "open")
        symbols = ("SPY", "AAPL")
        cache = compute_lag_series_cache(multi_col_df, symbols, columns, lags)
        history_dates_by_symbol = {}
        for sym in symbols:
            dates, _ = symbol_history_arrays(multi_col_df, sym, columns)
            history_dates_by_symbol[sym] = dates
        sym_col = multi_col_df[DataCol.SYMBOL.value].to_numpy()
        row_dates = multi_col_df[DataCol.DATE.value].to_numpy(
            dtype="datetime64[ns]"
        )
        base_arrays = {
            col: multi_col_df[col].to_numpy(dtype=np.float64)
            for col in columns
        }
        base_arrays[DataCol.SYMBOL.value] = sym_col
        base_arrays[DataCol.DATE.value] = row_dates
        n_rows = len(sym_col)
        n_features = len(columns) * (lags + 1)
        expected = np.empty((n_rows, n_features), dtype=np.float64)
        order = np.argsort(sym_col, kind="stable")
        sorted_syms = sym_col[order]
        unique_syms, start_indices = np.unique(sorted_syms, return_index=True)
        end_indices = np.append(start_indices[1:], len(sorted_syms))
        for sym, start, end in zip(unique_syms, start_indices, end_indices):
            idx = order[start:end]
            sym_dates = row_dates[idx]
            sym_base = {col: base_arrays[col][idx] for col in columns}
            sym_matrix = _build_lag_feature_matrix_reference(
                sym,
                columns,
                lags,
                sym_base,
                sym_dates,
                history_dates_by_symbol[sym],
                cache,
            )
            expected[idx] = sym_matrix
        actual = build_lag_feature_matrix_pooled(
            sym_col,
            columns,
            lags,
            base_arrays,
            row_dates,
            history_dates_by_symbol,
            cache,
            symbols,
        )
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=0, equal_nan=True
        )
