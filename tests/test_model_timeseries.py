"""Integration tests for time series model support."""

import arch
import numpy as np
import pandas as pd
import pytest
from pybroker import Strategy, indicator, model
from pybroker.common import DataCol
from pybroker.config import StrategyConfig
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
        seen_train = []
        seen_predict = []
        seen_input = []

        def train_fn(symbol, train_data, test_data, lag_train, lag_test):
            seen_train_cols.append(set(train_data.columns))
            seen_train.append(
                (lag_train, len(train_data), lag_test, len(test_data))
            )
            return object(), ("close",)

        def predict_fn(model, data):
            seen_predict.append(data)
            return np.full(len(data), 1.0)

        def exec_fn(ctx):
            ctx.preds("lag_wf", ctx.symbol)
            df = ctx.input("lag_wf", ctx.symbol)
            if len(df) == 1:
                seen_input.append((set(df.columns), dict(df.attrs)))

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
        for lag_train, n_train, lag_test, n_test in seen_train:
            # lags=2 over one lag column: the current value plus lags 1 and
            # 2, aligned one row per train/test row.
            assert isinstance(lag_train, np.ndarray)
            assert lag_train.shape == (n_train, 3)
            assert not np.isnan(lag_train).any()
            assert isinstance(lag_test, np.ndarray)
            assert lag_test.shape == (n_test, 3)
        assert seen_predict
        for matrix in seen_predict:
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape[1] == 3
        assert seen_input
        cols, attrs = seen_input[0]
        assert cols == {"close"}
        # The lag matrix is never attached to user-facing DataFrames.
        assert attrs == {}

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

    def test_per_bar_lag_features_composition(self, data_source_df):
        calls = []

        def train_fn(symbol, train_data, test_data, lag_train, lag_test):
            return object()

        def predict_fn(model, matrix):
            # Per-bar predictions receive the growing lag matrix directly.
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape[1] == 3
            # Last row is the current bar's value followed by its lags,
            # with the first test bar's lags carried from the train window.
            if len(matrix) > 1:
                assert matrix[-1, 1] == matrix[-2, 0]
            assert np.isfinite(matrix[-1]).all()
            calls.append(len(matrix))
            return 1.0

        m = model(
            "per_bar_lags",
            train_fn,
            predict_fn=predict_fn,
            lags=2,
            lag_cols=("close",),
            per_bar=True,
        )
        ds = data_source_df[data_source_df[DataCol.SYMBOL.value] == "SPY"]
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(
            lambda ctx: ctx.preds("per_bar_lags"), ["SPY"], models=m
        )
        strategy.walkforward(windows=2, train_size=0.5)
        assert calls
        assert max(calls) > 1


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
        def train_fn(symbol, train_data, test_data, lag_train, lag_test):
            returns = train_data["close"].pct_change().dropna() * 100
            model = arch.arch_model(returns, vol="GARCH", p=1, q=1).fit(
                disp="off"
            )
            return model, ("close",)

        def predict_fn(model, matrix):
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape[1] == 4
            # The first column of the lag block is the current close series.
            returns = pd.Series(matrix[:, 0]).pct_change().dropna() * 100
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

        def capture_train(train_data, lag_train):
            widths["train"] = lag_train.shape[1]
            widths["data_cols"] = [
                col
                for col in train_data.columns
                if col not in ("date", "symbol")
            ]

        def train_fn(symbol, train_data, test_data, lag_train, lag_test):
            capture_train(train_data, lag_train)
            return object(), ("close",)

        def pooled_train_fn(train_data, test_data, lag_train, lag_test):
            capture_train(train_data, lag_train)
            return object(), ("close",)

        def predict_fn(model, data):
            widths.setdefault("predict", data.shape[1])
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
        # Inferred lag columns are the data columns excluding date/symbol.
        assert widths["train"] == len(widths["data_cols"]) * 4

    @pytest.mark.parametrize("pooled", [False, True])
    def test_lag_cols_narrows_both_ends(self, data_source_df, pooled):
        widths = self._widths(data_source_df, pooled, lag_cols=("close",))
        assert widths["train"] == 4
        assert widths["predict"] == 4

    def test_lag_cols_requires_lags(self):
        with pytest.raises(ValueError, match="lag_cols requires lags"):
            model("bad_lag_cols", lambda s, t, u: None, lag_cols=("close",))

    def test_lag_cols_rejects_reserved_column(self):
        with pytest.raises(ValueError, match="reserved column 'date'"):
            model(
                "bad_reserved",
                lambda s, t, u, **kwargs: None,
                lags=2,
                lag_cols=("date",),
            )

    def test_lag_cols_rejects_unknown_column(self, data_source_df):
        m = model(
            "bad_unknown",
            lambda s, t, u, **kwargs: object(),
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

    def test_interval_lag_warmup_preserves_prediction_shape(self, df):
        """Warmup padding must keep the estimator's own output shape.

        A classifier's predict_proba is (n_rows, n_classes). Padding with a
        1-D float array raised on the concatenate, and coercing the dtype
        broke non-float outputs -- so lags silently worked for regressors and
        crashed for classifiers.
        """

        def predict_fn(_m, data):
            # Two-column probabilities, one row per input row.
            return np.column_stack((np.zeros(len(data)), np.ones(len(data))))

        m = model(
            "tf_proba",
            lambda s, t, u, **kwargs: object(),
            predict_fn=predict_fn,
            lags=3,
            lag_cols=["close"],
        )
        strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
        shapes = []

        def exec_fn(ctx):
            preds = ctx.interval("weekly").preds("tf_proba")
            if len(preds):
                shapes.append(np.asarray(preds).shape)

        strategy.add_execution(
            exec_fn, ["SPY"], models=m, intervals=["weekly"]
        )
        strategy.walkforward(windows=1, train_size=0.5, timeframe="1d")
        assert shapes
        # Two columns survive, and the warmup rows are NaN in both.
        longest = max(shapes, key=lambda sh: sh[0])
        assert longest[1] == 2

    def test_interval_lag_warmup_rejects_wrong_length_predictions(self, df):
        """A wrong-length result must raise, not silently misalign.

        pred[: idx + 1] indexes by compressed bar, so quietly padding a
        short result shifts every prediction for the rest of the run.
        """

        def predict_fn(_m, data):
            return np.zeros(max(len(data) - 1, 1))

        m = model(
            "tf_shortpred",
            lambda s, t, u, **kwargs: object(),
            predict_fn=predict_fn,
            lags=3,
            lag_cols=["close"],
        )
        strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])

        def exec_fn(ctx):
            ctx.interval("weekly").preds("tf_shortpred")

        strategy.add_execution(
            exec_fn, ["SPY"], models=m, intervals=["weekly"]
        )
        with pytest.raises(ValueError, match="predictions for"):
            strategy.walkforward(windows=1, train_size=0.5, timeframe="1d")

    @pytest.mark.parametrize("interval", ["weekly", 5])
    def test_interval_lag_warmup_rows_are_not_predicted(self, df, interval):
        """Warmup rows must not reach the estimator.

        Compressed model input starts at compressed row 0 with no earlier
        history to draw lags from, so rows 0..lags-1 have all-NaN lag
        features. Training drops them; prediction did not, and a strict
        estimator raises on the first test bar while a NaN-tolerant one
        quietly predicts from features that do not exist yet.
        """
        seen_matrices = []

        def predict_fn(_m, matrix):
            assert isinstance(matrix, np.ndarray)
            seen_matrices.append(matrix)
            if not np.isfinite(matrix).all():
                raise ValueError("Input X contains NaN")
            return np.zeros(len(matrix))

        m = model(
            "tf_lag_warmup",
            lambda s, t, u, **kwargs: object(),
            predict_fn=predict_fn,
            lags=3,
            lag_cols=["close"],
        )
        strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
        preds_seen = []

        def exec_fn(ctx):
            preds = ctx.interval(interval).preds("tf_lag_warmup")
            if len(preds):
                preds_seen.append(np.asarray(preds))

        strategy.add_execution(
            exec_fn, ["SPY"], models=m, intervals=[interval]
        )
        strategy.walkforward(windows=1, train_size=0.5, timeframe="1d")
        assert seen_matrices
        assert preds_seen
        # The warmup bars are reported as NaN rather than fabricated.
        longest = max(preds_seen, key=len)
        assert np.isnan(longest[:3]).all()
        assert np.isfinite(longest[3:]).all()

    @pytest.mark.parametrize("interval", ["weekly", 5])
    def test_lagged_model_on_interval(self, df, interval):
        # The interval lag cache is keyed by the *string* form of the
        # interval. Writing it under one key type and reading it under
        # another silently misses, surfacing as "Lag history missing".
        widths_seen = []

        def predict_fn(_m, matrix):
            widths_seen.append(matrix.shape[1])
            return np.zeros(len(matrix))

        m = model(
            "tf_lag",
            lambda s, t, u, **kwargs: object(),
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
        assert widths_seen
        # lags=2 over one lag column: current value plus lags 1 and 2.
        assert all(n == 3 for n in widths_seen)


def _sma(bar_data, period):
    close = bar_data.close
    out = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        out[i] = np.mean(close[i - period + 1 : i + 1])
    return out


class TestIndicatorLagColumns:
    """``lags=`` must compose with ``indicators=`` on every path.

    Lag caches are built from raw data sources that hold no indicator values,
    so an indicator lag column used to fail with a bare ``KeyError`` during
    training and ``History column not found`` at prediction.
    """

    @pytest.fixture()
    def spy_df(self, data_source_df):
        return data_source_df[data_source_df[DataCol.SYMBOL.value] == "SPY"]

    @staticmethod
    def _walkforward(m, ds, syms, intervals=None):
        strategy = Strategy(ds, START_DATE, END_DATE)
        if intervals:
            strategy.add_execution(
                lambda ctx: ctx.interval(intervals[0]).preds(m.name),
                syms,
                models=m,
                intervals=intervals,
            )
            strategy.walkforward(windows=1, train_size=0.5, timeframe="1d")
        else:
            strategy.add_execution(
                lambda ctx: ctx.preds(m.name), syms, models=m
            )
            strategy.walkforward(windows=1, train_size=0.5)

    def test_filed_repro_indicators_with_lags(self, spy_df):
        # indicators= plus lags= and no lag_cols raised KeyError('sma5').
        ind = indicator("sma5", _sma, period=5)
        m = model(
            "ind_lags",
            lambda s, t, u, **kwargs: object(),
            [ind],
            predict_fn=lambda _m, d: np.zeros(len(d)),
            lags=2,
        )
        self._walkforward(m, spy_df, ["SPY"])

    def test_inferred_lag_cols_exclude_indicators(self, spy_df):
        ind = indicator("sma5", _sma, period=5)
        seen = {}

        def train_fn(symbol, train_data, test_data, lag_train, lag_test):
            seen["width"] = lag_train.shape[1]
            seen["cols"] = list(train_data.columns)
            return object()

        m = model(
            "inferred_lags",
            train_fn,
            [ind],
            predict_fn=lambda _m, d: np.zeros(len(d)),
            lags=2,
        )
        self._walkforward(m, spy_df, ["SPY"])
        assert seen["cols"]
        # Inferred lag columns are the data columns, excluding date and the
        # sma5 indicator; a width including sma5 would be one block wider.
        data_cols = [
            col for col in seen["cols"] if col not in ("date", "sma5")
        ]
        assert data_cols
        assert seen["width"] == len(data_cols) * 3

    @pytest.mark.parametrize("intervals", [None, ["weekly"]])
    @pytest.mark.parametrize("pooled", [False, True])
    def test_indicator_lag_cols_all_paths(
        self, data_source_df, pooled, intervals
    ):
        ind = indicator("sma5", _sma, period=5)
        widths = {}

        def train_fn(symbol, train_data, test_data, lag_train, lag_test):
            widths["train"] = lag_train.shape[1]
            return object(), ("close",)

        def pooled_train_fn(train_data, test_data, lag_train, lag_test):
            widths["train"] = lag_train.shape[1]
            return object(), ("close",)

        def predict_fn(_m, data):
            widths.setdefault("predict", data.shape[1])
            return np.zeros(len(data))

        m = model(
            "ind_lag_cols",
            pooled_train_fn if pooled else train_fn,
            [ind],
            predict_fn=predict_fn,
            lags=2,
            lag_cols=(ind,),
            pooled=pooled,
        )
        syms = ["SPY", "AAPL"] if pooled else ["SPY"]
        ds = data_source_df[data_source_df[DataCol.SYMBOL.value].isin(syms)]
        self._walkforward(m, ds, syms, intervals)
        # Training and prediction must build the same shape, or the model is
        # fed a matrix it was never fit on. lags=2 over the single sma5 lag
        # column yields width 3.
        assert widths["train"] == widths["predict"] == 3

    def test_test_bar_zero_uses_train_window_indicator_history(self, spy_df):
        # The lag cache must reach back past the test window. Sourcing from
        # IndicatorScope.fetch would mask to the test window and leave bar 0
        # as NaN instead of carrying the train window's values.
        ind = indicator("sma5", _sma, period=5)
        seen = {}

        def train_fn(symbol, train_data, test_data, lag_train, lag_test):
            # The split boundary comes from the data the model actually sees.
            seen["train_tail"] = train_data[DataCol.DATE.value].to_numpy()[-2:]
            return object(), ("close",)

        def predict_fn(_m, matrix):
            seen.setdefault("row0", matrix[0].copy())
            return np.zeros(len(matrix))

        m = model(
            "ind_history",
            train_fn,
            [ind],
            predict_fn=predict_fn,
            lags=2,
            lag_cols=(ind,),
        )
        self._walkforward(m, spy_df, ["SPY"])
        close = spy_df[DataCol.CLOSE.value].to_numpy(dtype=float)
        full = pd.Series(
            _sma(type("B", (), {"close": close})(), 5),
            index=spy_df[DataCol.DATE.value].to_numpy(),
        )
        prev, last = seen["train_tail"]
        assert not np.isnan(seen["row0"]).any()
        dates = spy_df[DataCol.DATE.value].to_numpy()
        bar0 = dates[np.searchsorted(dates, last) + 1]
        np.testing.assert_allclose(
            seen["row0"], [full.loc[bar0], full.loc[last], full.loc[prev]]
        )

    def test_unknown_lag_col_raises_clear_error(self, spy_df):
        m = model(
            "bad_lag_col",
            lambda s, t, u, **kwargs: object(),
            predict_fn=lambda _m, d: np.zeros(len(d)),
            lags=2,
            lag_cols=("not_a_column",),
        )
        with pytest.raises(ValueError, match="lag_cols not found"):
            self._walkforward(m, spy_df, ["SPY"])

    def test_vwap_lag_col_on_interval(self, spy_df):
        ds = spy_df.assign(
            vwap=spy_df[DataCol.CLOSE.value].to_numpy(dtype=float) * 1.01
        )
        m = model(
            "vwap_lags",
            lambda s, t, u, **kwargs: (object(), ("close",)),
            predict_fn=lambda _m, d: np.zeros(len(d)),
            lags=2,
            lag_cols=("vwap",),
        )
        self._walkforward(m, ds, ["SPY"], ["weekly"])


class TestLagMatrixArguments:
    """Lag matrices are passed explicitly, never through the DataFrames."""

    @staticmethod
    def _run(m, data_source_df, syms):
        ds = data_source_df[data_source_df[DataCol.SYMBOL.value].isin(syms)]
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(lambda ctx: ctx.preds(m.name), syms, models=m)
        strategy.walkforward(windows=1, train_size=0.5)

    def test_train_fn_receives_aligned_matrices(self, data_source_df):
        seen = {}

        def train_fn(symbol, train_data, test_data, lag_train, lag_test):
            seen["lag_train"] = lag_train
            seen["lag_test"] = lag_test
            seen["n_train"] = len(train_data)
            seen["n_test"] = len(test_data)
            seen["train_close"] = train_data["close"].to_numpy()
            return object(), ("close",)

        m = model(
            "aligned_lags",
            train_fn,
            predict_fn=lambda _m, d: np.zeros(len(d)),
            lags=2,
            lag_cols=("close",),
        )
        self._run(m, data_source_df, ["SPY"])
        lag_train, lag_test = seen["lag_train"], seen["lag_test"]
        assert isinstance(lag_train, np.ndarray)
        assert isinstance(lag_test, np.ndarray)
        assert lag_train.shape == (seen["n_train"], 3)
        assert lag_test.shape == (seen["n_test"], 3)
        # Row i of the matrix belongs to row i of the frame, and lag 1 of a
        # bar is the previous bar's current value.
        np.testing.assert_array_equal(lag_train[:, 0], seen["train_close"])
        np.testing.assert_array_equal(lag_train[1:, 1], lag_train[:-1, 0])
        # Warmup rows are already dropped from training data only.
        assert not np.isnan(lag_train).any()

    def test_pooled_train_fn_receives_matrices(self, data_source_df):
        seen = {}

        def pooled_train_fn(train_data, test_data, lag_train, lag_test):
            seen["lag_train"] = lag_train
            seen["lag_test"] = lag_test
            seen["n_train"] = len(train_data)
            seen["n_test"] = len(test_data)
            assert "symbol" in train_data.columns
            return object(), ("close",)

        m = model(
            "pooled_lags",
            pooled_train_fn,
            predict_fn=lambda _m, d: np.zeros(len(d)),
            lags=2,
            lag_cols=("close",),
            pooled=True,
        )
        self._run(m, data_source_df, ["SPY", "AAPL"])
        assert seen["lag_train"].shape == (seen["n_train"], 3)
        assert seen["lag_test"].shape == (seen["n_test"], 3)
        assert not np.isnan(seen["lag_train"]).any()

    def test_predict_input_type_by_lags(self, data_source_df):
        seen = {}

        def lag_predict_fn(_m, data):
            seen.setdefault("lagged", type(data))
            return np.zeros(len(data))

        def plain_predict_fn(_m, data):
            seen.setdefault("plain", type(data))
            return np.zeros(len(data))

        m_lag = model(
            "typed_lags",
            lambda s, t, u, **kwargs: (object(), ("close",)),
            predict_fn=lag_predict_fn,
            lags=2,
            lag_cols=("close",),
        )
        m_plain = model(
            "typed_plain",
            lambda s, t, u: (object(), ("close",)),
            predict_fn=plain_predict_fn,
        )
        ds = data_source_df[data_source_df[DataCol.SYMBOL.value] == "SPY"]
        strategy = Strategy(ds, START_DATE, END_DATE)
        strategy.add_execution(
            lambda ctx: (ctx.preds("typed_lags"), ctx.preds("typed_plain")),
            ["SPY"],
            models=[m_lag, m_plain],
        )
        strategy.walkforward(windows=1, train_size=0.5)
        assert seen["lagged"] is np.ndarray
        assert seen["plain"] is pd.DataFrame

    def test_default_predict_receives_matrix(self, data_source_df):
        # Without a predict_fn, the model's own predict gets the matrix.
        instances = []

        class RecordingModel:
            def __init__(self):
                self.seen = []

            def predict(self, X):
                self.seen.append((type(X), np.asarray(X).shape[1]))
                return np.zeros(len(X))

        def train_fn(symbol, train_data, test_data, lag_train, lag_test):
            instance = RecordingModel()
            instances.append(instance)
            return instance, ("close",)

        m = model("default_predict", train_fn, lags=2, lag_cols=("close",))
        self._run(m, data_source_df, ["SPY"])
        assert instances
        assert instances[-1].seen
        for input_type, width in instances[-1].seen:
            assert input_type is np.ndarray
            assert width == 3

    def test_strict_train_fn_without_lags_gets_no_kwargs(self, data_source_df):
        # A three-positional-arg train_fn (no **kwargs) must keep working
        # for models without lags.
        def train_fn(symbol, train_data, test_data):
            return object(), ("close",)

        m = model(
            "strict_no_lags",
            train_fn,
            predict_fn=lambda _m, d: np.zeros(len(d)),
        )
        self._run(m, data_source_df, ["SPY"])
