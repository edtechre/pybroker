"""Unit tests for model lag transform metadata."""

import numpy as np
import pandas as pd
import pytest
from pybroker.common import DataCol, ModelSymbol, TrainedModel
from pybroker.model import (
    LagSeriesKey,
    ModelInput,
    apply_lags_to_model_input,
    build_lag_feature_matrix,
    build_lag_feature_matrix_pooled,
    compute_lag_series_cache,
    model_input_from_frame,
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
        matrix = model_input.lag_features
        assert matrix is not None
        # lags=2 yields the current value plus lags 1 and 2 per column.
        assert matrix.shape == (len(sym_df), 3)
        close = sym_df[DataCol.CLOSE.value].to_numpy(dtype=float)
        np.testing.assert_array_equal(matrix[:, 0], close)
        np.testing.assert_array_equal(matrix[2:, 1], close[1:-1])
        np.testing.assert_array_equal(matrix[2:, 2], close[:-2])
        assert model_input.lags == 2

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
        matrix = trimmed.lag_features
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
            lambda s, t, u, **kwargs: (object(), ("close",)),
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
        model_input = input_scope.fetch_model_input(sym, "lag_model")
        assert seen_cols[0] == ["close"]
        matrix = model_input.lag_features
        assert matrix is not None
        assert matrix.shape[1] == 3

    def test_test_bar_zero_has_train_tail_lags(self, lag_setup):
        input_scope, sym, _, _, sym_df = lag_setup
        train_tail = sym_df.iloc[len(sym_df) // 2 - 1]
        model_input = input_scope.fetch_model_input(
            sym, "lag_model", end_index=1
        )
        matrix = model_input.lag_features
        assert matrix is not None
        # Test bar 0 carries real lag values from the tail of the train
        # window rather than NaN warmup.
        assert matrix[0, 0] == sym_df.iloc[len(sym_df) // 2]["close"]
        assert matrix[0, 1] == train_tail["close"]
        assert matrix[0, 2] == sym_df.iloc[len(sym_df) // 2 - 2]["close"]

    def test_slice_preserves_lag_features(self, lag_setup):
        input_scope, sym, _, _, _ = lag_setup
        full = input_scope.fetch_model_input(sym, "lag_model")
        sliced = input_scope.fetch_model_input(sym, "lag_model", end_index=2)
        full_matrix = full.lag_features
        sliced_matrix = sliced.lag_features
        assert full_matrix is not None
        assert sliced_matrix is not None
        np.testing.assert_array_equal(sliced_matrix, full_matrix[:2])

    def test_fetch_returns_plain_dataframe(self, lag_setup):
        # Lag matrices are never attached to user-facing DataFrames, so a
        # frame does not keep the matrix alive after the input is released.
        input_scope, sym, _, _, _ = lag_setup
        df = input_scope.fetch(sym, "lag_model")
        assert isinstance(df, pd.DataFrame)
        assert df.attrs == {}


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
        # Row 0 of ``stacked`` is the unlagged series, included as the
        # block's first column.
        matrix[:, col_idx : col_idx + lags + 1] = stacked[
            : lags + 1, offset : offset + n_rows
        ].T
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
        row_dates = sym_df[DataCol.DATE.value].to_numpy()
        expected = _build_lag_feature_matrix_reference(
            "SPY",
            columns,
            lags,
            row_dates,
            history_dates,
            cache,
        )
        actual = build_lag_feature_matrix(
            "SPY",
            columns,
            lags,
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
            sym_matrix = _build_lag_feature_matrix_reference(
                sym,
                columns,
                lags,
                sym_dates,
                history_dates_by_symbol[sym],
                cache,
            )
            expected[idx] = sym_matrix
        actual = build_lag_feature_matrix_pooled(
            sym_col,
            columns,
            lags,
            row_dates,
            history_dates_by_symbol,
            cache,
            symbols,
        )
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=0, equal_nan=True
        )


class TestLagCacheDepth:
    """Regression tests for lag caches shared across models on one symbol."""

    @pytest.fixture()
    def scope_setup(self, data_source_df, symbols):
        def build(specs):
            scope = StaticScope.instance()
            scope._model_sources.clear()
            sym = symbols[0]
            sym_df = data_source_df[data_source_df["symbol"] == sym]
            test_df = sym_df.iloc[len(sym_df) // 2 :].set_index(
                ["symbol", "date"]
            )
            dates = sorted(test_df.index.get_level_values(1).unique())
            trained = {}
            for name, cols, lags in specs:
                model(
                    name,
                    lambda s, t, u, c=cols, **kwargs: (object(), c),
                    lags=lags,
                    lag_cols=cols,
                )
                trained[ModelSymbol(name, sym)] = TrainedModel(
                    name=name,
                    instance=object(),
                    predict_fn=None,
                    input_cols=cols,
                    lag_columns=cols,
                )
            input_scope = ModelInputScope(
                ColumnScope(test_df),
                IndicatorScope({}, dates),
                trained,
                ColumnScope(sym_df.set_index(["symbol", "date"]).sort_index()),
                dates,
            )
            return input_scope, sym, sym_df

        return build

    @staticmethod
    def _expected_lags(sym_df, col, lags, n_test_rows):
        """Returns the lag block expected for the test window."""
        values = sym_df[col].to_numpy(dtype=float)
        start = len(values) - n_test_rows
        expected = np.empty((n_test_rows, lags + 1), dtype=float)
        for r in range(n_test_rows):
            for lag in range(lags + 1):
                src = start + r - lag
                expected[r, lag] = values[src] if src >= 0 else np.nan
        return expected

    def test_two_models_different_lags_same_symbol(self, scope_setup):
        # A shallower cache entry must not be reused for a deeper request:
        # the njit fill kernel would read out of bounds without raising.
        input_scope, sym, sym_df = scope_setup(
            [("lag2", ("close",), 2), ("lag6", ("close",), 6)]
        )
        shallow_matrix = input_scope.fetch_model_input(
            sym, "lag2"
        ).lag_features
        deep_matrix = input_scope.fetch_model_input(sym, "lag6").lag_features
        assert shallow_matrix.shape[1] == 3
        assert deep_matrix.shape[1] == 7
        np.testing.assert_array_equal(
            deep_matrix,
            self._expected_lags(sym_df, "close", 6, len(deep_matrix)),
        )
        np.testing.assert_array_equal(
            shallow_matrix,
            self._expected_lags(sym_df, "close", 2, len(shallow_matrix)),
        )

    def test_deep_model_first_then_shallow(self, scope_setup):
        input_scope, sym, sym_df = scope_setup(
            [("deep", ("close",), 6), ("shallow", ("close",), 2)]
        )
        deep_matrix = input_scope.fetch_model_input(sym, "deep").lag_features
        shallow_matrix = input_scope.fetch_model_input(
            sym, "shallow"
        ).lag_features
        np.testing.assert_array_equal(
            deep_matrix,
            self._expected_lags(sym_df, "close", 6, len(deep_matrix)),
        )
        np.testing.assert_array_equal(
            shallow_matrix,
            self._expected_lags(sym_df, "close", 2, len(shallow_matrix)),
        )

    def test_two_models_different_lag_columns_same_symbol(self, scope_setup):
        input_scope, sym, sym_df = scope_setup(
            [("on_close", ("close",), 2), ("on_volume", ("volume",), 2)]
        )
        close_matrix = input_scope.fetch_model_input(
            sym, "on_close"
        ).lag_features
        volume_matrix = input_scope.fetch_model_input(
            sym, "on_volume"
        ).lag_features
        np.testing.assert_array_equal(
            close_matrix,
            self._expected_lags(sym_df, "close", 2, len(close_matrix)),
        )
        np.testing.assert_array_equal(
            volume_matrix,
            self._expected_lags(sym_df, "volume", 2, len(volume_matrix)),
        )


class TestLagCacheBounds:
    def test_store_stacked_lags_keeps_tallest(self):
        from pybroker.model import _store_stacked_lags_in_cache

        cache = {}
        values = np.arange(10, dtype=float)
        _store_stacked_lags_in_cache(
            cache, "SPY", "close", 6, _build_stacked_lags(values, 6)
        )
        _store_stacked_lags_in_cache(
            cache, "SPY", "close", 2, _build_stacked_lags(values, 2)
        )
        assert cache[LagSeriesKey("SPY", "close", 0)].shape[0] == 7

    def test_build_lag_feature_matrix_rejects_shallow_cache(self):
        values = np.arange(10, dtype=float)
        dates = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-11"),
        )
        cache = {}
        from pybroker.model import _store_stacked_lags_in_cache

        _store_stacked_lags_in_cache(
            cache, "SPY", "close", 2, _build_stacked_lags(values, 2)
        )
        with pytest.raises(ValueError, match="lag rows"):
            build_lag_feature_matrix("SPY", ("close",), 6, dates, dates, cache)

    def test_build_lag_feature_matrix_rejects_missing_column(self):
        dates = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-11"),
        )
        with pytest.raises(ValueError, match="Lag history missing"):
            build_lag_feature_matrix("SPY", ("close",), 2, dates, dates, {})


class TestLaggedTrainFnSignature:
    """A lagged train_fn must accept the lag matrix kwargs to register."""

    def test_missing_lag_params_raises(self):
        with pytest.raises(ValueError, match="lag_train"):
            model("m", lambda s, t, u: None, lags=2)

    def test_var_keyword_registers(self):
        assert model("m_kw", lambda s, t, u, **kwargs: None, lags=2)

    def test_explicit_lag_params_register(self):
        assert model(
            "m_explicit",
            lambda s, t, u, lag_train, lag_test: None,
            lags=2,
        )

    def test_reserved_model_kwargs_raise(self):
        with pytest.raises(ValueError, match="reserved"):
            model(
                "m_reserved",
                lambda s, t, u, lag_train, lag_test: None,
                lags=2,
                lag_train=1,
            )

    def test_non_lagged_fn_is_not_checked(self):
        assert model("m_no_lags", lambda s, t, u: None)


class TestColumnOrderDeterminism:
    def test_model_input_columns_stable_across_hash_seeds(self):
        # Iterating the registered column frozenset yields a different order
        # per process, which makes the positionally-consumed lag matrix
        # irreproducible across runs.
        import subprocess
        import sys

        code = (
            "import pybroker;"
            "from pybroker.model import _model_input_columns;"
            "from pybroker.scope import StaticScope;"
            "pybroker.register_columns('zeta', 'alpha');"
            "s = StaticScope.instance();"
            "print(_model_input_columns(('ind',), s.all_data_cols))"
        )
        outs = set()
        for seed in ("0", "1", "12345"):
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            assert result.returncode == 0, result.stderr
            outs.add(result.stdout.strip())
        assert len(outs) == 1, outs


def test_drop_lag_warmup_keeps_mid_series_nan_rows():
    """A sparse lag column must not thin the middle of the training set."""
    n = 12
    dates = (
        np.arange(n).astype("timedelta64[D]") + np.datetime64("2021-01-04")
    ).astype("datetime64[ns]")
    # Leading warmup NaN, then a NaN in the middle of the series.
    lag_features = np.arange(n, dtype=float).reshape(n, 1)
    lag_features[0] = np.nan
    lag_features[5] = np.nan
    model_input = ModelInput(
        ("date", "close"),
        {"date": dates, "close": np.arange(n, dtype=float)},
        dates,
        lag_features=lag_features,
        lags=1,
        lag_columns=("close",),
    )
    trimmed = model_input.drop_lag_warmup()
    # Only the leading row goes; the mid-series NaN row stays.
    assert len(trimmed) == n - 1
    assert np.isnan(trimmed.lag_features[4, 0])


def test_drop_lag_warmup_raises_when_every_row_is_nan():
    """An all-NaN lag column must name itself rather than empty the frame."""
    n = 6
    dates = (
        np.arange(n).astype("timedelta64[D]") + np.datetime64("2021-01-04")
    ).astype("datetime64[ns]")
    model_input = ModelInput(
        ("date", "volume"),
        {"date": dates, "volume": np.full(n, np.nan)},
        dates,
        lag_features=np.full((n, 1), np.nan),
        lags=1,
        lag_columns=("volume",),
    )
    with pytest.raises(ValueError, match="volume"):
        model_input.drop_lag_warmup()


def test_drop_lag_warmup_is_pooled_aware():
    """Each symbol block carries its own warmup at its own offset.

    Pooled model input stacks one contiguous block per symbol. Trimming only
    the front of the matrix leaves every block after the first with its NaN
    lag rows intact, which sklearn.fit rejects outright.
    """
    n_per, lags = 6, 2
    rows = 2 * n_per
    dates = np.tile(
        (
            np.arange(n_per).astype("timedelta64[D]")
            + np.datetime64("2021-01-04")
        ).astype("datetime64[ns]"),
        2,
    )
    symbols = np.array(["AAA"] * n_per + ["BBB"] * n_per)
    lag_features = np.arange(rows, dtype=float).reshape(rows, 1)
    lag_features[0:lags] = np.nan
    lag_features[n_per : n_per + lags] = np.nan
    model_input = ModelInput(
        ("date", "symbol", "close"),
        {
            "date": dates,
            "symbol": symbols,
            "close": np.arange(rows, dtype=float),
        },
        dates,
        lag_features=lag_features,
        lags=lags,
        lag_columns=("close",),
    )
    trimmed = model_input.drop_lag_warmup()
    assert not np.isnan(trimmed.lag_features).any()
    # Both blocks keep exactly their post-warmup rows.
    kept, counts = np.unique(trimmed.arrays["symbol"], return_counts=True)
    assert list(kept) == ["AAA", "BBB"]
    assert list(counts) == [n_per - lags, n_per - lags]
