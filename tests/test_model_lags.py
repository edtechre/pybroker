"""Unit tests for model lag transform metadata."""

import numpy as np
import pandas as pd
import pytest
from pybroker.common import DataCol, ModelSymbol, TrainedModel
from pybroker.timeseries import (
    LagSeriesKey,
    apply_lags_to_model_input,
    compute_lag_series_cache,
    feature_matrix_from_model_input,
    model_input_from_frame,
    model_input_lags,
    symbol_history_arrays,
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
