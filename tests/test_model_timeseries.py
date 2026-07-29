"""Integration tests for time series model support."""

import arch
import numpy as np
import pandas as pd
import pytest
from pybroker import Strategy, model
from pybroker.common import DataCol
from pybroker.config import StrategyConfig
from pybroker.model import (
    feature_matrix_from_model_input,
    model_input_lag_columns,
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
            lag_cols=("close",),
        )
        ds = data_source_df[data_source_df[DataCol.SYMBOL.value] == "SPY"]
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(exec_fn, ["SPY"], models=m)
        strategy.walkforward(windows=2, train_size=0.5)
        assert seen_train_cols
        for cols in seen_train_cols:
            assert "close" in cols
            # Lag features ride alongside model input rather than widening it.
            assert not any(name.endswith("_lag1") for name in cols)
        for matrix in seen_train_matrix:
            assert matrix is not None
            assert matrix.shape[1] == 2
        assert seen_input
        cols, matrix, lags = seen_input[0]
        assert cols == {"close"}
        assert matrix is not None
        # lags=2 over one lag column: lag 1 and lag 2, matching training.
        assert matrix.shape == (1, 2)
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
            assert matrix.shape[1] == 3
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
            lag_cols=("close",),
            per_bar=True,
        )
        ds = spy_df.copy()
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, ["SPY"], models=m)
        strategy.walkforward(windows=2, train_size=0.5)


class TestLagColumnContract:
    """Lag features must be built identically at training and prediction."""

    @staticmethod
    def _widths(data_source_df, pooled, lag_cols=None):
        widths = {}

        def capture_train(train_data):
            widths["train"] = feature_matrix_from_model_input(
                train_data
            ).shape[1]
            widths["train_cols"] = model_input_lag_columns(train_data)

        def train_fn(symbol, train_data, test_data):
            capture_train(train_data)
            return object(), ("close",)

        def pooled_train_fn(train_data, test_data):
            capture_train(train_data)
            return object(), ("close",)

        def predict_fn(model, data):
            widths.setdefault(
                "predict", feature_matrix_from_model_input(data).shape[1]
            )
            widths.setdefault("predict_cols", model_input_lag_columns(data))
            return np.zeros(len(data))

        m = model(
            "lag_contract",
            pooled_train_fn if pooled else train_fn,
            predict_fn=predict_fn,
            lags=3,
            lag_cols=lag_cols,
            pooled=pooled,
        )
        syms = ["SPY", "AAPL"] if pooled else ["SPY"]
        ds = data_source_df[data_source_df[DataCol.SYMBOL.value].isin(syms)]
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(
            lambda ctx: ctx.preds("lag_contract"), syms, models=m
        )
        strategy.walkforward(windows=1, train_size=0.5)
        return widths

    @pytest.mark.parametrize("pooled", [False, True])
    def test_train_and_predict_widths_match(self, data_source_df, pooled):
        # A narrower input_cols return value must not narrow the lag matrix,
        # or the model is handed a different shape than it was fit on.
        widths = self._widths(data_source_df, pooled)
        assert widths["train"] == widths["predict"]
        assert widths["train_cols"] == widths["predict_cols"]
        assert widths["train"] == len(widths["train_cols"]) * 3

    @pytest.mark.parametrize("pooled", [False, True])
    def test_lag_cols_narrows_both_ends(self, data_source_df, pooled):
        widths = self._widths(data_source_df, pooled, lag_cols=("close",))
        assert widths["train"] == 3
        assert widths["predict"] == 3
        assert widths["train_cols"] == ("close",)
        assert widths["predict_cols"] == ("close",)

    def test_lag_cols_requires_lags(self):
        with pytest.raises(ValueError, match="lag_cols requires lags"):
            model("bad_lag_cols", lambda s, t, u: None, lag_cols=("close",))

    def test_lag_cols_rejects_reserved_column(self):
        with pytest.raises(ValueError, match="reserved column 'date'"):
            model(
                "bad_reserved",
                lambda s, t, u: None,
                lags=2,
                lag_cols=("date",),
            )

    def test_lag_cols_rejects_unknown_column(self, data_source_df):
        m = model(
            "bad_unknown",
            lambda s, t, u: object(),
            predict_fn=lambda _m, d: np.zeros(len(d)),
            lags=2,
            lag_cols=("not_a_column",),
        )
        ds = data_source_df[data_source_df[DataCol.SYMBOL.value] == "SPY"]
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: None, ["SPY"], models=m)
        with pytest.raises(ValueError, match="lag_cols not found"):
            strategy.walkforward(windows=1, train_size=0.5)


class TestIntervalPerBar:
    """Per-bar predictions on a compressed interval must not look ahead."""

    @pytest.fixture()
    def df(self):
        dates = pd.date_range("2020-01-01", periods=120)
        close = 100 + np.arange(120, dtype=float)
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

    def test_no_lookahead(self, df):
        # Anchoring each prediction to its own bar's close makes a prediction
        # built from future rows immediately visible.
        m = model(
            "tf_pb",
            lambda s, t, u: object(),
            predict_fn=lambda _m, data: float(data["close"].iloc[-1]),
            per_bar=True,
        )
        strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
        seen = []

        def exec_fn(ctx):
            tf = ctx.interval("weekly")
            preds = tf.preds("tf_pb")
            if len(preds):
                seen.append((preds.copy(), tf.close.copy()))

        strategy.add_execution(
            exec_fn,
            ["SPY"],
            models=m,
            intervals=["weekly"],
        )
        strategy.walkforward(windows=1, train_size=0.5, timeframe="1d")
        assert seen
        for preds, closes in seen:
            np.testing.assert_array_equal(preds, closes)

    def test_predict_called_once_per_compressed_bar(self, df):
        # Rebuilding the prediction history each base bar would replay
        # earlier bars and corrupt stateful models.
        calls = []

        def predict_fn(mdl, data):
            calls.append(len(data))
            mdl["count"] += 1
            return float(mdl["count"])

        m = model(
            "tf_pb_state",
            lambda s, t, u: {"count": 0},
            predict_fn=predict_fn,
            per_bar=True,
        )
        strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
        final = []

        def exec_fn(ctx):
            preds = ctx.interval("weekly").preds("tf_pb_state")
            if len(preds):
                final.append(preds.copy())

        strategy.add_execution(
            exec_fn,
            ["SPY"],
            models=m,
            intervals=["weekly"],
        )
        strategy.walkforward(windows=1, train_size=0.5, timeframe="1d")
        assert calls
        assert calls == list(range(1, len(calls) + 1))
        assert final
        np.testing.assert_array_equal(
            final[-1], np.arange(1.0, len(final[-1]) + 1)
        )

    @pytest.mark.parametrize("interval", ["weekly", 5])
    def test_lagged_model_on_interval(self, df, interval):
        # The interval lag cache is keyed by the *string* form of the
        # interval. Writing it under one key type and reading it under
        # another silently misses, surfacing as "Lag history missing".
        lags_seen = []

        def predict_fn(_m, data):
            lags_seen.append(model_input_lags(data))
            return np.zeros(len(data))

        m = model(
            "tf_lag",
            lambda s, t, u: object(),
            predict_fn=predict_fn,
            lags=2,
            lag_cols=["close"],
        )
        strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
        seen = []

        def exec_fn(ctx):
            preds = ctx.interval(interval).preds("tf_lag")
            if len(preds):
                seen.append(len(preds))

        strategy.add_execution(
            exec_fn, ["SPY"], models=m, intervals=[interval]
        )
        strategy.walkforward(windows=1, train_size=0.5, timeframe="1d")
        assert seen
        assert lags_seen
        assert all(n == 2 for n in lags_seen)
