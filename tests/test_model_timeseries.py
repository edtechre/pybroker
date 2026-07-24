"""Integration tests for time series model support."""

import arch
import numpy as np
import pytest
from pybroker import Strategy, model
from pybroker.common import DataCol
from pybroker.config import StrategyConfig
from pybroker.timeseries import (
    feature_matrix_from_model_input,
    model_input_lags,
)
from pybroker.scope import StaticScope
from .fixtures import *  # noqa: F401

START_DATE = np.datetime64("2019-01-01")
END_DATE = np.datetime64("2020-12-31")


@pytest.fixture(autouse=True)
def clear_model_sources():
    scope = StaticScope.instance()
    scope._model_sources.clear()
    yield
    scope._model_sources.clear()


class TestStatefulWalkforward:
    def test_mock_stateful_per_bar_walkforward(self, data_source_df):
        calls = []

        def train_fn(symbol, train_data, test_data):
            return {"id": symbol, "count": 0}

        def predict_fn(model, data):
            calls.append(len(data))
            model["count"] += 1
            return float(model["count"])

        m = model(
            "stateful",
            train_fn,
            predict_fn=predict_fn,
            per_bar=True,
        )

        captured = []

        def exec_fn(ctx):
            pred = ctx.preds("stateful", ctx.symbol)[-1]
            captured.append(pred)

        ds = data_source_df[data_source_df[DataCol.SYMBOL.value] == "SPY"]
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ["SPY"], models=m)
        strategy.walkforward(windows=2, train_size=0.5)
        assert captured
        assert captured[0] == 1.0
        assert len(calls) == len(captured)
        assert all(c >= 1 for c in calls)

    def test_lags_walkforward_feature_metadata(self, data_source_df):
        seen_train_cols = []
        seen_train_matrix = []
        seen_input = []

        def train_fn(symbol, train_data, test_data):
            seen_train_cols.append(set(train_data.columns))
            seen_train_matrix.append(
                feature_matrix_from_model_input(train_data)
            )
            return object(), ("close",)

        def predict_fn(model, data):
            return np.full(len(data), 1.0)

        def exec_fn(ctx):
            df = ctx.input("lag_wf", ctx.symbol)
            if len(df) == 1:
                seen_input.append(
                    (
                        set(df.columns),
                        feature_matrix_from_model_input(df),
                        model_input_lags(df),
                    )
                )

        m = model(
            "lag_wf",
            train_fn,
            predict_fn=predict_fn,
            lags=2,
        )
        ds = data_source_df[data_source_df[DataCol.SYMBOL.value] == "SPY"]
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ["SPY"], models=m)
        strategy.walkforward(windows=2, train_size=0.5)
        assert seen_train_cols
        for cols in seen_train_cols:
            assert "close" in cols
            assert not any(name.endswith("_lag1") for name in cols)
        for matrix in seen_train_matrix:
            assert matrix is not None
        assert seen_input
        cols, matrix, lags = seen_input[0]
        assert cols == {"close"}
        assert matrix is not None
        assert matrix.shape == (1, 3)
        assert lags == 2

    def test_per_bar_fresh_model_each_walkforward_window(self, data_source_df):
        train_counts = []

        def train_fn(symbol, train_data, test_data):
            train_counts.append(symbol)
            return {"symbol": symbol}

        def predict_fn(model, data):
            return 1.0

        m = model(
            "wf_reset",
            train_fn,
            predict_fn=predict_fn,
            per_bar=True,
        )
        ds = data_source_df[data_source_df[DataCol.SYMBOL.value] == "SPY"]
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, ["SPY"], models=m)
        strategy.walkforward(windows=2, train_size=0.5)
        assert len(train_counts) == 2


class TestGarchIntegration:
    @pytest.fixture()
    def spy_df(self, data_source_df):
        return data_source_df[data_source_df[DataCol.SYMBOL.value] == "SPY"]

    def test_garch_vol_walkforward(self, spy_df):

        def train_fn(symbol, train_data, test_data):
            returns = train_data["close"].pct_change().dropna() * 100
            return arch.arch_model(returns, vol="GARCH", p=1, q=1).fit(
                disp="off"
            )

        def predict_fn(model, data):
            returns = data["close"].pct_change().dropna() * 100
            res = model.model.fix(model.params.values, returns)
            fcast = res.forecast(horizon=1)
            daily_vol = np.sqrt(fcast.variance.iloc[-1, 0]) / 100
            return daily_vol * np.sqrt(252)

        preds = []

        def exec_fn(ctx):
            vol = ctx.preds("garch_vol", ctx.symbol)[-1]
            preds.append(vol)

        m = model(
            "garch_vol",
            train_fn,
            predict_fn=predict_fn,
            per_bar=True,
        )
        ds = spy_df.copy()
        strategy = Strategy(
            ds,
            START_DATE,
            END_DATE,
            config=StrategyConfig(max_long_positions=1),
        )
        strategy.add_execution(exec_fn, ["SPY"], models=m)
        strategy.walkforward(windows=2, train_size=0.5)
        assert preds
        assert all(np.isfinite(p) and p > 0 for p in preds)

    def test_garch_with_lags(self, spy_df):

        def train_fn(symbol, train_data, test_data):
            returns = train_data["close"].pct_change().dropna() * 100
            model = arch.arch_model(returns, vol="GARCH", p=1, q=1).fit(
                disp="off"
            )
            return model, ("close",)

        def predict_fn(model, data):
            matrix = feature_matrix_from_model_input(data)
            assert matrix is not None
            assert matrix.shape[1] == 4
            returns = data["close"].pct_change().dropna() * 100
            res = model.model.fix(model.params.values, returns)
            fcast = res.forecast(horizon=1)
            daily_vol = np.sqrt(fcast.variance.iloc[-1, 0]) / 100
            return daily_vol * np.sqrt(252)

        m = model(
            "garch_lags",
            train_fn,
            predict_fn=predict_fn,
            lags=3,
            per_bar=True,
        )
        ds = spy_df.copy()
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, ["SPY"], models=m)
        strategy.walkforward(windows=2, train_size=0.5)
