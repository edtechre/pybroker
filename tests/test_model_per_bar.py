"""Unit tests for per-bar model prediction."""

import numpy as np
import pandas as pd
import pytest
from pybroker import Strategy
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
                lambda s, t, u: None,
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


class TestPerBarScalarContract:
    """``predict_fn`` must yield the current bar's prediction, not bar 0's."""

    @pytest.fixture()
    def df(self):
        dates = pd.date_range("2020-01-01", periods=40)
        close = 100 + np.arange(40, dtype=float)
        return pd.DataFrame(
            {
                "symbol": "SPY",
                "date": dates,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1e6,
            }
        )

    @staticmethod
    def _run(df, predict_fn, capture=None):
        StaticScope.instance()._model_sources.clear()
        m = model(
            "pb_contract",
            lambda s, t, u: object(),
            predict_fn=predict_fn,
            per_bar=True,
        )
        strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])

        def exec_fn(ctx):
            pred = ctx.preds("pb_contract")[-1]
            if capture is not None:
                capture.append((float(ctx.close[-1]), float(pred)))

        strategy.add_execution(exec_fn, ["SPY"], models=m)
        strategy.walkforward(windows=1, train_size=0.5)

    def test_vector_predict_uses_current_bar(self, df):
        # Taking the first element would freeze every bar at bar 0's value.
        seen = []
        self._run(df, lambda _m, data: data["close"].to_numpy(), seen)
        assert seen
        for close, pred in seen:
            assert pred == close

    def test_scalar_predict_still_works(self, df):
        seen = []
        self._run(df, lambda _m, data: float(data["close"].iloc[-1]), seen)
        assert seen
        for close, pred in seen:
            assert pred == close

    def test_wrong_length_predict_raises(self, df):
        with pytest.raises(
            ValueError, match="Expected a scalar prediction for the current"
        ):
            self._run(df, lambda _m, _d: np.array([1.0, 2.0]))

    def test_empty_predict_raises(self, df):
        with pytest.raises(ValueError, match="returned no predictions"):
            self._run(df, lambda _m, _d: np.array([]))


class TestInputDataFnRowContract:
    """``input_data_fn`` must return one row per bar."""

    @pytest.fixture()
    def df(self):
        dates = pd.date_range("2020-01-01", periods=60)
        close = 100 + np.arange(60, dtype=float)
        return pd.DataFrame(
            {
                "symbol": "SPY",
                "date": dates,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1e6,
            }
        )

    @staticmethod
    def _run(df, per_bar):
        StaticScope.instance()._model_sources.clear()
        m = model(
            "row_contract",
            lambda s, t, u: object(),
            predict_fn=(
                (lambda _m, d: 1.0)
                if per_bar
                else (lambda _m, d: np.arange(len(d), dtype=float))
            ),
            per_bar=per_bar,
            input_data_fn=lambda d: d.assign(
                r=d["close"].pct_change()
            ).dropna(),
        )
        strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
        strategy.add_execution(
            lambda ctx: ctx.preds("row_contract"), ["SPY"], models=m
        )
        strategy.walkforward(windows=1, train_size=0.5)

    @pytest.mark.parametrize("per_bar", [False, True])
    def test_dropping_rows_raises(self, df, per_bar):
        # Dropping head rows would shift every prediction one bar forward.
        with pytest.raises(ValueError, match="one row per bar"):
            self._run(df, per_bar)
