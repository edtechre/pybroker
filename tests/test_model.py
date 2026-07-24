"""Unit tests for model.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import pytest
import re
from .fixtures import *  # noqa: F401
from unittest.mock import Mock, patch
from pybroker.cache import CacheDateFields
from pybroker.common import (
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    TrainedModel,
    to_datetime,
)
from pybroker.model import ModelLoader, ModelsMixin, ModelTrainer, model
from pybroker.timeframe import TimeframeData, compress_symbol_df
from pybroker.scope import ModelInputScope

TF_SECONDS = 60
BETWEEN_TIME = ("10:00", "15:30")


@pytest.fixture(params=[True, False])
def enable_parallel_models(request):
    return request.param


@pytest.fixture()
def train_data(data_source_df):
    return data_source_df.iloc[: data_source_df.shape[0] // 2]


@pytest.fixture()
def test_data(data_source_df):
    return data_source_df.iloc[data_source_df.shape[0] // 2 :]


@pytest.fixture()
def cache_date_fields(train_data):
    return CacheDateFields(
        start_date=to_datetime(sorted(train_data["date"].unique())[0]),
        end_date=to_datetime(sorted(train_data["date"].unique())[-1]),
        tf_seconds=TF_SECONDS,
        between_time=BETWEEN_TIME,
        days=None,
    )


@pytest.fixture()
def start_date(train_data):
    return to_datetime(sorted(train_data["date"].unique())[0])


@pytest.fixture()
def end_date(train_data):
    return to_datetime(sorted(train_data["date"].unique())[-1])


@pytest.fixture()
def model_loader():
    return model(
        "loader",
        lambda symbol, train_start_date, train_end_date: FakeModel(
            symbol=symbol, preds=[]
        ),
        [],
        pretrained=True,
    )


@pytest.fixture()
def model_syms(train_data, model_source, model_loader):
    return [
        ModelSymbol(model_source.name, sym)
        for sym in train_data["symbol"].unique()
    ] + [
        ModelSymbol(model_loader.name, sym)
        for sym in train_data["symbol"].unique()
    ]


@pytest.mark.parametrize(
    "pretrained, input_cols",
    [
        (True, False),
        (True, False),
    ],
)
def test_model(indicators, pretrained, input_cols):
    def input_data_fn(df):
        pass

    def predict_fn(model, df):
        pass

    name = f"pretrained={pretrained}"
    source = model(
        name,
        lambda x: (x, ["hhv", "llv"]) if input_cols else x,
        indicators,
        input_data_fn=input_data_fn,
        predict_fn=predict_fn,
        pretrained=pretrained,
    )
    assert isinstance(source, ModelLoader if pretrained else ModelTrainer)
    assert source.name == name
    assert source.indicators == ("hhv", "llv", "sumv")
    assert source._input_data_fn is input_data_fn
    assert source._predict_fn is predict_fn


@pytest.mark.parametrize("pooled", [True, False])
def test_model_pooled_flag(indicators, pooled):
    source = model(
        f"pooled={pooled}",
        lambda *args: args,
        indicators,
        pooled=pooled,
    )
    assert source.pooled is pooled


class TestModelSource:
    @pytest.mark.parametrize("clazz", [ModelLoader, ModelTrainer])
    def test_model_prepare_input_fn(self, data_source_df, clazz):
        prepare_fn = Mock()
        source = clazz(
            "model_source", lambda x: x, [], prepare_fn, None, False, {}
        )
        source.prepare_input_data(data_source_df)
        prepare_fn.assert_called_once_with(data_source_df)

    @pytest.mark.parametrize("clazz", [ModelLoader, ModelTrainer])
    def test_model_prepare_input_fn_when_empty_data(self, clazz):
        source = clazz("model_source", lambda x: x, [], None, None, False, {})
        df = source.prepare_input_data(pd.DataFrame())
        assert df.empty

    @pytest.mark.parametrize("clazz", [ModelLoader, ModelTrainer])
    def test_model_prepare_input_fn_when_fn_none(
        self, ind_df, ind_names, clazz
    ):
        source = clazz(
            "model_source", lambda x: x, ind_names, None, None, False, {}
        )
        df = source.prepare_input_data(ind_df)
        assert df.equals(ind_df)

    @pytest.mark.parametrize("clazz", [ModelLoader, ModelTrainer])
    def test_model_prepare_input_fn_when_indicators_not_found_then_error(
        self, ind_df, clazz
    ):
        source = clazz(
            "model_source", lambda x: x, ["foo"], None, None, False, {}
        )
        with pytest.raises(
            ValueError,
            match=re.escape("Indicator 'foo' not found in DataFrame."),
        ):
            source.prepare_input_data(ind_df)

    def test_model_loader_call_with_kwargs(self, start_date, end_date):
        load_fn = Mock()
        kwargs = {"a": 1, "b": 2}
        ModelLoader("loader", load_fn, [], None, None, False, kwargs)(
            "SPY", start_date, end_date
        )
        load_fn.assert_called_once_with("SPY", start_date, end_date, **kwargs)

    def test_model_trainer_call_with_kwargs(self, train_data, test_data):
        train_fn = Mock()
        kwargs = {"a": 1, "b": 2}
        ModelTrainer("trainer", train_fn, [], None, None, False, kwargs)(
            "SPY", train_data, test_data
        )
        train_fn.assert_called_once_with(
            "SPY", train_data, test_data, **kwargs
        )

    def test_model_trainer_repr(self):
        trainer = ModelTrainer(
            "trainer", lambda x: x, [], None, None, False, {"a": 1}
        )
        assert repr(trainer) == "ModelTrainer('trainer', {'a': 1})"

    def test_model_loader_repr(self):
        trainer = ModelLoader(
            "loader", lambda x: x, [], None, None, False, {"a": 1}
        )
        assert repr(trainer) == "ModelLoader('loader', {'a': 1})"


class TestModelsMixin:
    def _assert_models(self, models, expected_model_syms):
        assert set(models.keys()) == set(expected_model_syms)
        for model_sym in expected_model_syms:
            model = models[model_sym]
            assert isinstance(model, TrainedModel)
            assert model.name == model_sym.model_name
            assert model.instance.symbol == model_sym.symbol

    @pytest.mark.usefixtures("setup_model_cache")
    @pytest.mark.parametrize(
        "param_test_data",
        [
            pd.DataFrame(columns=["symbol", "date"]),
            LazyFixture("test_data"),
        ],
    )
    def test_train_models(
        self,
        model_syms,
        train_data,
        param_test_data,
        ind_data,
        cache_date_fields,
        enable_parallel_models,
        request,
    ):
        param_test_data = get_fixture(request, param_test_data)
        mixin = ModelsMixin()
        models = mixin.train_models(
            model_syms,
            train_data,
            param_test_data,
            ind_data,
            cache_date_fields,
            enable_parallel_models=enable_parallel_models,
        )
        self._assert_models(models, model_syms)

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_when_empty_train_data(
        self, model_syms, test_data, ind_data, cache_date_fields
    ):
        mixin = ModelsMixin()
        models = mixin.train_models(
            model_syms, pd.DataFrame(), test_data, ind_data, cache_date_fields
        )
        assert len(models) == 0

    @pytest.mark.usefixtures("setup_enabled_model_cache")
    def test_train_models_when_cached(
        self, model_syms, train_data, test_data, ind_data, cache_date_fields
    ):
        mixin = ModelsMixin()
        mixin.train_models(
            model_syms, train_data, test_data, ind_data, cache_date_fields
        )
        models = mixin.train_models(
            model_syms, train_data, test_data, ind_data, cache_date_fields
        )
        self._assert_models(models, model_syms)

    @pytest.mark.usefixtures("setup_enabled_model_cache")
    def test_train_models_when_partial_cached(
        self, model_syms, train_data, test_data, ind_data, cache_date_fields
    ):
        mixin = ModelsMixin()
        mixin.train_models(
            model_syms[:1], train_data, test_data, ind_data, cache_date_fields
        )
        models = mixin.train_models(
            model_syms, train_data, test_data, ind_data, cache_date_fields
        )
        self._assert_models(models, model_syms)

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_parallel_invokes_pool(
        self,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        indicators,
    ):
        trainer = model(
            "parallel_trainer",
            lambda sym, *_: FakeModel(sym, np.array([1.0])),
            indicators,
            pretrained=False,
        )
        trainer_syms = sorted(
            ModelSymbol(trainer.name, sym)
            for sym in train_data["symbol"].unique()
        )
        fake_results = [
            (
                "sym",
                (
                    model_sym,
                    FakeModel(model_sym.symbol, np.array([1.0])),
                    None,
                ),
            )
            for model_sym in trainer_syms
        ]
        mixin = ModelsMixin()
        with patch("pybroker.model.parallel") as mock_parallel:
            mock_pool = Mock(return_value=fake_results)
            mock_parallel.return_value.__enter__ = Mock(return_value=mock_pool)
            mock_parallel.return_value.__exit__ = Mock(return_value=False)
            models = mixin.train_models(
                trainer_syms,
                train_data,
                test_data,
                ind_data,
                cache_date_fields,
                enable_parallel_models=True,
            )
            mock_parallel.assert_called_once()
            mock_pool.assert_called_once()
        self._assert_models(models, trainer_syms)


class TestPooledModelsMixin:
    @pytest.fixture()
    def pooled_symbols(self):
        return frozenset({"SPY", "AAPL"})

    @pytest.fixture()
    def pooled_model_syms(self, pooled_symbols):
        return [
            ModelSymbol("pooled_model", sym) for sym in sorted(pooled_symbols)
        ]

    @pytest.fixture()
    def pooled_model_groups(self, pooled_symbols):
        return {("pooled_model", 1): pooled_symbols}

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_pooled_calls_once(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_symbols,
        pooled_model_syms,
        pooled_model_groups,
    ):
        train_fn = Mock(return_value=FakeModel("pooled", np.array([1.0])))
        model(
            "pooled_model",
            train_fn,
            indicators,
            pooled=True,
        )
        mixin = ModelsMixin()
        models = mixin.train_models(
            pooled_model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            pooled_model_groups=pooled_model_groups,
        )
        train_fn.assert_called_once()
        call_args = train_fn.call_args[0]
        assert len(call_args) == 2
        pooled_train, pooled_test = call_args
        assert DataCol.SYMBOL.value in pooled_train.columns
        assert set(pooled_train[DataCol.SYMBOL.value].unique()) == set(
            pooled_symbols
        )
        assert len(models) == len(pooled_symbols)

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_pooled_data_has_symbol_column(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_symbols,
        pooled_model_syms,
        pooled_model_groups,
    ):
        captured = {}

        def train_pooled(train_df, test_df):
            captured["train"] = train_df
            captured["test"] = test_df
            return FakeModel("pooled", np.array([1.0]))

        model("pooled_model", train_pooled, indicators, pooled=True)
        mixin = ModelsMixin()
        mixin.train_models(
            pooled_model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            pooled_model_groups=pooled_model_groups,
        )
        assert DataCol.SYMBOL.value in captured["train"].columns
        assert DataCol.SYMBOL.value in captured["test"].columns
        assert len(captured["train"][DataCol.SYMBOL.value].unique()) > 1

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_pooled_stores_shared_instance(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_model_syms,
        pooled_model_groups,
    ):
        shared = FakeModel("pooled", np.array([1.0]))
        model(
            "pooled_model",
            lambda train_df, test_df: shared,
            indicators,
            pooled=True,
        )
        mixin = ModelsMixin()
        models = mixin.train_models(
            pooled_model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            pooled_model_groups=pooled_model_groups,
        )
        instances = [
            models[model_sym].instance for model_sym in pooled_model_syms
        ]
        assert all(instance is shared for instance in instances)

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_pooled_non_pooled_mixed(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_model_syms,
        pooled_model_groups,
    ):
        model(
            "pooled_model",
            lambda train_df, test_df: FakeModel("pooled", np.array([1.0])),
            indicators,
            pooled=True,
        )
        per_sym = model(
            "per_sym_model",
            lambda sym, train_df, test_df: FakeModel(sym, np.array([1.0])),
            indicators,
        )
        per_sym_syms = [
            ModelSymbol(per_sym.name, sym)
            for sym in train_data["symbol"].unique()
        ]
        mixin = ModelsMixin()
        models = mixin.train_models(
            pooled_model_syms + per_sym_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            pooled_model_groups=pooled_model_groups,
        )
        pooled_instances = {
            models[model_sym].instance for model_sym in pooled_model_syms
        }
        assert len(pooled_instances) == 1
        for model_sym in per_sym_syms:
            assert models[model_sym].instance.symbol == model_sym.symbol

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_pooled_infers_input_cols_without_symbol(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_model_syms,
        pooled_model_groups,
    ):
        model(
            "pooled_model",
            lambda train_df, test_df: FakeModel("pooled", np.array([1.0])),
            indicators,
            pooled=True,
        )
        mixin = ModelsMixin()
        models = mixin.train_models(
            pooled_model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            pooled_model_groups=pooled_model_groups,
        )
        input_cols = models[pooled_model_syms[0]].input_cols
        assert input_cols is not None
        assert DataCol.SYMBOL.value not in input_cols
        assert "hhv" in input_cols

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_pooled_explicit_input_cols_unchanged(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_model_syms,
        pooled_model_groups,
    ):
        explicit_cols = ("hhv", "symbol_id")
        model(
            "pooled_model",
            lambda train_df, test_df: (
                FakeModel("pooled", np.array([1.0])),
                explicit_cols,
            ),
            indicators,
            pooled=True,
        )
        mixin = ModelsMixin()
        models = mixin.train_models(
            pooled_model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            pooled_model_groups=pooled_model_groups,
        )
        assert models[pooled_model_syms[0]].input_cols == explicit_cols

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_non_pooled_infers_input_cols_from_train_df(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
    ):
        per_sym = model(
            "per_sym_model",
            lambda sym, train_df, test_df: FakeModel(sym, np.array([1.0])),
            indicators,
        )
        model_sym = ModelSymbol(per_sym.name, train_data["symbol"].unique()[0])
        mixin = ModelsMixin()
        models = mixin.train_models(
            [model_sym],
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
        )
        input_cols = models[model_sym].input_cols
        assert input_cols is not None
        assert DataCol.SYMBOL.value not in input_cols
        assert "hhv" in input_cols
        assert "date" in input_cols

    @pytest.mark.usefixtures("setup_model_cache")
    def test_pooled_predict_input_omits_symbol(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_model_syms,
        pooled_model_groups,
        col_scope,
        ind_scope,
    ):
        model(
            "pooled_model",
            lambda train_df, test_df: FakeModel("pooled", np.array([1.0])),
            indicators,
            pooled=True,
        )
        mixin = ModelsMixin()
        models = mixin.train_models(
            pooled_model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            pooled_model_groups=pooled_model_groups,
        )
        model_sym = pooled_model_syms[0]
        input_scope = ModelInputScope(col_scope, ind_scope, models)
        df = input_scope.fetch(model_sym.symbol, model_sym.model_name)
        assert DataCol.SYMBOL.value not in df.columns
        assert list(df.columns) == list(models[model_sym].input_cols)

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_single_pooled_group_parallel_skips_pool(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_model_syms,
        pooled_model_groups,
    ):
        model(
            "pooled_model",
            lambda train_df, test_df: FakeModel("pooled", np.array([1.0])),
            indicators,
            pooled=True,
        )
        mixin = ModelsMixin()
        with patch("pybroker.model.parallel") as mock_parallel:
            mixin.train_models(
                pooled_model_syms,
                train_data,
                test_data,
                ind_data,
                cache_date_fields,
                enable_parallel_models=True,
                pooled_model_groups=pooled_model_groups,
            )
            mock_parallel.assert_not_called()

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_multiple_pooled_groups_parallel_invokes_pool(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
    ):
        model(
            "pooled_model",
            lambda train_df, test_df: FakeModel("pooled", np.array([1.0])),
            indicators,
            pooled=True,
        )
        symbols = sorted(train_data["symbol"].unique())
        group_one = frozenset({symbols[0]})
        group_two = frozenset({symbols[1]})
        pooled_model_groups = {
            ("pooled_model", 1): group_one,
            ("pooled_model", 2): group_two,
        }
        model_syms = [
            ModelSymbol("pooled_model", sym)
            for sym in sorted(group_one | group_two)
        ]
        mixin = ModelsMixin()
        fake_results = [
            (
                "pooled",
                (
                    "pooled_model",
                    group_one,
                    FakeModel("pooled", np.array([1.0])),
                    None,
                ),
            ),
            (
                "pooled",
                (
                    "pooled_model",
                    group_two,
                    FakeModel("pooled", np.array([1.0])),
                    None,
                ),
            ),
        ]
        with patch("pybroker.model.parallel") as mock_parallel:
            mock_pool = Mock(return_value=fake_results)
            mock_parallel.return_value.__enter__ = Mock(return_value=mock_pool)
            mock_parallel.return_value.__exit__ = Mock(return_value=False)
            models = mixin.train_models(
                model_syms,
                train_data,
                test_data,
                ind_data,
                cache_date_fields,
                enable_parallel_models=True,
                pooled_model_groups=pooled_model_groups,
            )
            mock_parallel.assert_called_once()
            mock_pool.assert_called_once()
        assert len(models) == len(model_syms)

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_pooled_mixed_parallel_invokes_pool(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_model_syms,
        pooled_model_groups,
    ):
        model(
            "pooled_model",
            lambda train_df, test_df: FakeModel("pooled", np.array([1.0])),
            indicators,
            pooled=True,
        )
        per_sym = model(
            "per_sym_model",
            lambda sym, train_df, test_df: FakeModel(sym, np.array([1.0])),
            indicators,
        )
        per_sym_syms = [
            ModelSymbol(per_sym.name, sym)
            for sym in train_data["symbol"].unique()
        ]
        mixin = ModelsMixin()
        fake_results = [
            (
                "pooled",
                (
                    "pooled_model",
                    frozenset({"SPY", "AAPL"}),
                    FakeModel("pooled", np.array([1.0])),
                    None,
                ),
            ),
            *[
                (
                    "sym",
                    (
                        model_sym,
                        FakeModel(model_sym.symbol, np.array([1.0])),
                        None,
                    ),
                )
                for model_sym in per_sym_syms
            ],
        ]
        with patch("pybroker.model.parallel") as mock_parallel:
            mock_pool = Mock(return_value=fake_results)
            mock_parallel.return_value.__enter__ = Mock(return_value=mock_pool)
            mock_parallel.return_value.__exit__ = Mock(return_value=False)
            models = mixin.train_models(
                pooled_model_syms + per_sym_syms,
                train_data,
                test_data,
                ind_data,
                cache_date_fields,
                enable_parallel_models=True,
                pooled_model_groups=pooled_model_groups,
            )
            mock_parallel.assert_called_once()
            mock_pool.assert_called_once()
        assert len(models) == len(pooled_model_syms) + len(per_sym_syms)

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_pooled_mixed_serial_parallel_equivalent(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled_model_syms,
        pooled_model_groups,
    ):
        model(
            "pooled_model",
            lambda train_df, test_df: FakeModel("pooled", np.array([1.0])),
            indicators,
            pooled=True,
        )
        per_sym = model(
            "per_sym_model",
            lambda sym, train_df, test_df: FakeModel(sym, np.array([1.0])),
            indicators,
        )
        per_sym_syms = [
            ModelSymbol(per_sym.name, sym)
            for sym in train_data["symbol"].unique()
        ]
        all_model_syms = pooled_model_syms + per_sym_syms
        mixin = ModelsMixin()
        serial_models = mixin.train_models(
            all_model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            enable_parallel_models=False,
            pooled_model_groups=pooled_model_groups,
        )
        parallel_models = mixin.train_models(
            all_model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            enable_parallel_models=True,
            pooled_model_groups=pooled_model_groups,
        )
        assert set(serial_models.keys()) == set(parallel_models.keys())
        for model_sym in all_model_syms:
            if model_sym in pooled_model_syms:
                assert (
                    serial_models[model_sym].instance
                    is parallel_models[model_sym].instance
                    or serial_models[model_sym].instance.symbol
                    == parallel_models[model_sym].instance.symbol
                )
            else:
                assert (
                    serial_models[model_sym].instance.symbol
                    == parallel_models[model_sym].instance.symbol
                )


class TestTimeframeModels:
    def test_model_timeframe_binds_name(self, scope, indicators):
        m = model(
            "tf_model",
            lambda sym, train, test: FakeModel(sym, np.array([1.0])),
            indicators,
        )
        tf_m = m.timeframe("weekly")
        assert tf_m.name == "tf_model@weekly"
        assert tf_m.base is m

    def test_train_models_timeframe(self, scope, cache_date_fields):
        sym = "SPY"
        dates = np.array(
            ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"],
            dtype="datetime64[D]",
        )
        n = len(dates)
        close = np.arange(n, dtype=np.float64) + 1
        df = pd.DataFrame(
            {
                DataCol.SYMBOL.value: [sym] * n,
                DataCol.DATE.value: dates,
                DataCol.OPEN.value: close,
                DataCol.HIGH.value: close + 1,
                DataCol.LOW.value: close - 1,
                DataCol.CLOSE.value: close,
                DataCol.VOLUME.value: np.ones(n),
            }
        )
        timeframe_data = TimeframeData()
        timeframe_data.compressed[(sym, 2)] = compress_symbol_df(
            df, 2, frozenset()
        )
        from pybroker.indicator import IndicatorsMixin, indicator

        sma_ind = indicator(
            "sma2",
            lambda bar_data, period: bar_data.close,
            period=2,
        )
        ind_data = IndicatorsMixin().compute_indicators(
            df=df,
            indicator_syms=[IndicatorSymbol(sma_ind.timeframe(2).name, sym)],
            cache_date_fields=cache_date_fields,
            disable_parallel_indicators=True,
            timeframe_data=timeframe_data,
        )
        m = model(
            "tf_model",
            lambda sym, train, test: FakeModel(sym, np.zeros(len(test))),
            [sma_ind],
        )
        mixin = ModelsMixin()
        train_dates = dates[:2]
        test_dates = dates[2:]
        models = mixin.train_models(
            model_syms=[ModelSymbol(m.timeframe(2).name, sym)],
            train_data=df[df[DataCol.DATE.value].isin(train_dates)],
            test_data=df[df[DataCol.DATE.value].isin(test_dates)],
            indicator_data=ind_data,
            cache_date_fields=cache_date_fields,
            timeframe_data=timeframe_data,
        )
        model_sym = ModelSymbol("tf_model@2", sym)
        assert model_sym in models
        assert models[model_sym].name == "tf_model@2"


class TestTimeSeriesModelOptions:
    def test_lags_metadata_on_train(
        self,
        scope,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
    ):
        sym = train_data[DataCol.SYMBOL.value].iloc[0]
        model(
            "lag_train",
            lambda sym, train, test: (object(), ("close",)),
            lags=2,
        )
        mixin = ModelsMixin()
        models = mixin.train_models(
            [ModelSymbol("lag_train", sym)],
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
        )
        assert models[ModelSymbol("lag_train", sym)].input_cols == ("close",)

    def test_per_bar_stored_on_trained_model(
        self,
        scope,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
    ):
        sym = train_data[DataCol.SYMBOL.value].iloc[0]
        model(
            "pb_model",
            lambda sym, train, test: object(),
            predict_fn=lambda m, d: np.array([1.0]),
            per_bar=True,
        )
        mixin = ModelsMixin()
        models = mixin.train_models(
            [ModelSymbol("pb_model", sym)],
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
        )
        assert models[ModelSymbol("pb_model", sym)].per_bar is True
