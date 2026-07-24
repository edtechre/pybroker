"""Unit tests for per-bar model prediction."""

import numpy as np
import pytest
from pybroker.common import ModelSymbol, TrainedModel
from pybroker.model import model
from pybroker.scope import (
    ModelInputScope,
    PredictionScope,
    StaticScope,
    get_signals,
)
from .fixtures import *  # noqa: F401


@pytest.fixture(autouse=True)
def clear_model_sources():
    scope = StaticScope.instance()
    scope._model_sources.clear()
    yield
    scope._model_sources.clear()


class TestPerBarValidation:
    def test_per_bar_and_pooled_rejected(self):
        with pytest.raises(
            ValueError, match="per_bar=True is not supported with pooled=True"
        ):
            model(
                "m",
                lambda t, u: None,
                predict_fn=lambda m, d: np.array([1.0]),
                per_bar=True,
                pooled=True,
            )

    def test_per_bar_requires_predict_fn(self):
        with pytest.raises(
            ValueError, match="per_bar=True requires predict_fn"
        ):
            model(
                "m",
                lambda s, t, u: None,
                per_bar=True,
            )


class TestPredictionScopePerBar:
    @pytest.fixture()
    def per_bar_scope(self, data_source_df, symbols):
        StaticScope.instance()._model_sources.clear()
        calls = []

        def predict_fn(model, data):
            calls.append(len(data))
            return float(len(data))

        model(
            "pb",
            lambda s, t, u: (object(), ("close",)),
            predict_fn=predict_fn,
            per_bar=True,
        )
        sym = symbols[0]
        sym_df = data_source_df[data_source_df["symbol"] == sym]
        test_df = sym_df.iloc[len(sym_df) // 2 :].set_index(["symbol", "date"])
        dates = sorted(test_df.index.get_level_values(1).unique())
        trained = {
            ModelSymbol("pb", sym): TrainedModel(
                name="pb",
                instance=object(),
                predict_fn=predict_fn,
                input_cols=("close",),
                per_bar=True,
            )
        }
        from pybroker.scope import ColumnScope, IndicatorScope

        col_scope = ColumnScope(test_df)
        ind_scope = IndicatorScope({}, dates)
        input_scope = ModelInputScope(col_scope, ind_scope, trained, {}, dates)
        pred_scope = PredictionScope(trained, input_scope)
        return pred_scope, sym, calls, len(dates)

    def test_strict_call_order_and_inclusive_slice(self, per_bar_scope):
        pred_scope, sym, calls, n = per_bar_scope
        for end_index in range(1, n + 1):
            pred = pred_scope.fetch(sym, "pb", end_index)
            assert len(pred) == end_index
            assert pred[-1] == float(end_index)
        assert calls == list(range(1, n + 1))

    def test_incremental_cache_matches_hand_rolled(self, per_bar_scope):
        pred_scope, sym, _, n = per_bar_scope
        full = pred_scope.fetch(sym, "pb", n)
        again = pred_scope.fetch(sym, "pb", n)
        assert np.array_equal(full, again)

    def test_get_signals_materializes_full_array(self, per_bar_scope):
        pred_scope, sym, _, n = per_bar_scope
        col_scope = pred_scope._input_scope._col_scope
        ind_scope = pred_scope._input_scope._ind_scope
        signals = get_signals([sym], col_scope, ind_scope, pred_scope)
        assert len(signals[sym]["pb_pred"]) == n

    def test_input_scope_serves_incremental_slices(self, per_bar_scope):
        pred_scope, sym, _, _ = per_bar_scope
        input_scope = pred_scope._input_scope
        full = input_scope.fetch(sym, "pb")
        slice1 = input_scope.fetch(sym, "pb", end_index=1)
        assert len(slice1) == 1
        assert slice1["close"].iloc[0] == full["close"].iloc[0]
