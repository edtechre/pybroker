"""Unit tests for model.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import pytest
import re
import warnings
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
from pybroker.indicator import IndicatorsMixin, indicator
from pybroker.model import ModelLoader, ModelsMixin, ModelTrainer, model
from pybroker.parallel import set_parallel
from pybroker.interval import (
    IntervalData,
    compress,
    compress_symbol_df,
    indicator_interval_name,
    model_interval_name,
)
from pybroker.scope import ModelInputScope, param, register_columns

TF_SECONDS = 60
BETWEEN_TIME = ("10:00", "15:30")


@pytest.fixture(params=[True, False])
def parallel_models(request):
    return request.param


@pytest.fixture()
def multi_worker_parallel_config():
    """Configures more than one worker for the duration of a test.

    ``_run_model_trainers`` skips the pool when the configured worker count is
    ``1`` (the default), since building one would just run the tasks
    sequentially anyway. Tests that assert the pool *is* entered have to opt
    in.
    """
    import pybroker.parallel as parallel_mod

    saved = parallel_mod._config
    set_parallel(n_jobs=2)
    try:
        yield
    finally:
        parallel_mod._config = saved


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


def _param_reading_train_fn(symbol, train_df, test_df):
    """Reads a param set by the parent. Module-level so it pickles to loky."""
    return param("scope_probe")


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
        parallel_models,
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
            parallel_models=parallel_models,
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

    @pytest.mark.usefixtures(
        "setup_model_cache", "multi_worker_parallel_config"
    )
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
                parallel_models=True,
            )
            mock_parallel.assert_called_once()
            mock_pool.assert_called_once()
        self._assert_models(models, trainer_syms)

    @pytest.mark.usefixtures("setup_model_cache")
    def test_train_models_parallel_propagates_static_scope(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
    ):
        """A ``train_fn`` must see the caller's params in a worker process.

        Worker processes start with an empty
        :class:`pybroker.scope.StaticScope`, so without shipping the caller's
        scope alongside each task ``pybroker.param()`` silently returns
        ``None`` under parallel training while returning the real value when
        run serially.
        """
        import pybroker.parallel as parallel_mod

        saved = parallel_mod._config
        param("scope_probe", "from-parent")
        source = model(
            "scope_probe_model", _param_reading_train_fn, indicators
        )
        model_syms = sorted(
            ModelSymbol(source.name, sym)
            for sym in train_data["symbol"].unique()
        )
        # More than one task, otherwise the pool is skipped entirely.
        assert len(model_syms) > 1
        set_parallel(n_jobs=2)
        try:
            # loky, i.e. genuinely separate processes rather than threads.
            assert parallel_mod._config.backend == "loky"
            models = ModelsMixin().train_models(
                model_syms,
                train_data,
                test_data,
                ind_data,
                cache_date_fields,
                parallel_models=True,
            )
        finally:
            parallel_mod._config = saved
        for model_sym in model_syms:
            assert models[model_sym].instance == "from-parent"


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
        assert len(call_args) == 3
        symbols_arg, pooled_train, pooled_test = call_args
        assert symbols_arg == tuple(sorted(pooled_symbols))
        assert DataCol.SYMBOL.value in pooled_train.columns
        assert list(pooled_train[DataCol.SYMBOL.value].unique()) == list(
            symbols_arg
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

        def train_pooled(symbols, train_df, test_df):
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
            lambda symbols, train_df, test_df: shared,
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
            lambda symbols, train_df, test_df: FakeModel(
                "pooled", np.array([1.0])
            ),
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
            lambda symbols, train_df, test_df: FakeModel(
                "pooled", np.array([1.0])
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
            lambda symbols, train_df, test_df: (
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
    @pytest.mark.parametrize("pooled", [False, True])
    def test_train_models_infers_registered_custom_data_cols(
        self,
        scope,
        indicators,
        train_data,
        test_data,
        ind_data,
        cache_date_fields,
        pooled,
    ):
        """Columns from ``register_columns`` are trained on, so they belong in
        the inferred input columns.

        Dropping them here while the prediction side keeps them would feed the
        model fewer features than it saw during training.
        """
        register_columns("adj_close")
        seen_train_cols = []

        def train_fn(*args):
            seen_train_cols.append(tuple(args[-2].columns))
            return FakeModel("custom_col", np.array([1.0]))

        source = model("custom_col_model", train_fn, indicators, pooled=pooled)
        symbols = frozenset(train_data["symbol"].unique())
        model_syms = [ModelSymbol(source.name, sym) for sym in sorted(symbols)]
        pooled_model_groups = {(source.name, 1): symbols} if pooled else None
        mixin = ModelsMixin()
        models = mixin.train_models(
            model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            pooled_model_groups=pooled_model_groups,
        )
        assert seen_train_cols
        assert all("adj_close" in cols for cols in seen_train_cols)
        for model_sym in model_syms:
            input_cols = models[model_sym].input_cols
            assert input_cols is not None
            assert "adj_close" in input_cols
            assert DataCol.SYMBOL.value not in input_cols

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
            lambda symbols, train_df, test_df: FakeModel(
                "pooled", np.array([1.0])
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
        model_sym = pooled_model_syms[0]
        input_scope = ModelInputScope(col_scope, ind_scope, models)
        df = input_scope.fetch(model_sym.symbol, model_sym.model_name)
        assert DataCol.SYMBOL.value not in df.columns
        assert list(df.columns) == list(models[model_sym].input_cols)

    @pytest.mark.usefixtures(
        "setup_model_cache", "multi_worker_parallel_config"
    )
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
            lambda symbols, train_df, test_df: FakeModel(
                "pooled", np.array([1.0])
            ),
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
                parallel_models=True,
                pooled_model_groups=pooled_model_groups,
            )
            mock_parallel.assert_not_called()

    @pytest.mark.usefixtures(
        "setup_model_cache", "multi_worker_parallel_config"
    )
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
            lambda symbols, train_df, test_df: FakeModel(
                "pooled", np.array([1.0])
            ),
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
                parallel_models=True,
                pooled_model_groups=pooled_model_groups,
            )
            mock_parallel.assert_called_once()
            mock_pool.assert_called_once()
        assert len(models) == len(model_syms)

    @pytest.mark.usefixtures(
        "setup_model_cache", "multi_worker_parallel_config"
    )
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
            lambda symbols, train_df, test_df: FakeModel(
                "pooled", np.array([1.0])
            ),
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
                parallel_models=True,
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
            lambda symbols, train_df, test_df: FakeModel(
                "pooled", np.array([1.0])
            ),
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
            parallel_models=False,
            pooled_model_groups=pooled_model_groups,
        )
        parallel_models = mixin.train_models(
            all_model_syms,
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            parallel_models=True,
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


class TestIntervalModels:
    def test_model_interval_name_helper(self):
        assert model_interval_name("tf_model", "weekly") == "tf_model@weekly"

    def test_train_models_interval(self, scope, cache_date_fields):
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
        interval_data = IntervalData()
        interval_data.compressed[(sym, 2)] = compress_symbol_df(
            df, 2, frozenset(), 86400.0
        )

        sma_ind = indicator(
            "sma2",
            lambda bar_data, period: bar_data.close,
            period=2,
        )
        tf_ind_name = indicator_interval_name(sma_ind.name, 2)
        ind_data = IndicatorsMixin().compute_indicators(
            df=df,
            indicator_syms=[IndicatorSymbol(tf_ind_name, sym)],
            cache_date_fields=cache_date_fields,
            parallel_indicators=False,
            interval_data=interval_data,
        )
        m = model(
            "tf_model",
            lambda sym, train, test: FakeModel(sym, np.zeros(len(test))),
            [sma_ind],
        )
        mixin = ModelsMixin()
        train_dates = dates[:2]
        test_dates = dates[2:]
        tf_model_name = model_interval_name(m.name, 2)
        models = mixin.train_models(
            model_syms=[ModelSymbol(tf_model_name, sym)],
            train_data=df[df[DataCol.DATE.value].isin(train_dates)],
            test_data=df[df[DataCol.DATE.value].isin(test_dates)],
            indicator_data=ind_data,
            cache_date_fields=cache_date_fields,
            interval_data=interval_data,
        )
        model_sym = ModelSymbol(tf_model_name, sym)
        assert model_sym in models
        assert models[model_sym].name == tf_model_name

    def test_train_models_pooled_interval(self, scope, cache_date_fields):
        sma_ind = indicator(
            "sma2",
            lambda bar_data, period: bar_data.close,
            period=2,
        )
        symbols = ["SPY", "AAPL"]
        dates = np.array(
            pd.date_range("2020-01-06", periods=20, freq="B"),
            dtype="datetime64[D]",
        )
        n = len(dates)
        close = np.arange(n, dtype=np.float64) + 1
        frames = []
        for sym in symbols:
            frames.append(
                pd.DataFrame(
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
            )
        df = pd.concat(frames, ignore_index=True)
        interval_data = IntervalData()
        for sym in symbols:
            sym_df = df[df[DataCol.SYMBOL.value] == sym].reset_index(drop=True)
            interval_data.compressed[(sym, "weekly")] = compress_symbol_df(
                sym_df, "weekly", frozenset(), 86400.0
            )
        ind_syms = []
        for sym in symbols:
            ind_syms.append(
                IndicatorSymbol(
                    indicator_interval_name(sma_ind.name, "weekly"), sym
                )
            )
        ind_data = IndicatorsMixin().compute_indicators(
            df=df,
            indicator_syms=ind_syms,
            cache_date_fields=cache_date_fields,
            parallel_indicators=False,
            interval_data=interval_data,
        )
        train_fn = Mock(return_value=FakeModel("pooled", np.array([1.0])))
        model(
            "pooled_model",
            train_fn,
            [sma_ind],
            pooled=True,
        )
        tf_model_name = model_interval_name("pooled_model", "weekly")
        pooled_syms = frozenset(symbols)
        pooled_model_syms = [
            ModelSymbol(tf_model_name, sym) for sym in sorted(pooled_syms)
        ]
        pooled_model_groups = {(tf_model_name, 1): pooled_syms}
        weekly_dates = interval_data.compressed[
            (symbols[0], "weekly")
        ].bars.dates
        split = max(len(weekly_dates) // 2, 1)
        train_dates = weekly_dates[:split]
        test_dates = weekly_dates[split:]
        train_data = df[df[DataCol.DATE.value].isin(train_dates)]
        test_data = df[df[DataCol.DATE.value].isin(test_dates)]
        mixin = ModelsMixin()
        models = mixin.train_models(
            model_syms=pooled_model_syms,
            train_data=train_data,
            test_data=test_data,
            indicator_data=ind_data,
            cache_date_fields=cache_date_fields,
            interval_data=interval_data,
            pooled_model_groups=pooled_model_groups,
        )
        train_fn.assert_called_once()
        symbols_arg, pooled_train, _pooled_test = train_fn.call_args[0]
        assert symbols_arg == tuple(sorted(pooled_syms))
        assert DataCol.SYMBOL.value in pooled_train.columns
        assert list(pooled_train[DataCol.SYMBOL.value].unique()) == list(
            symbols_arg
        )
        assert len(pooled_train) > 0
        assert len(models) == len(pooled_model_syms)
        instances = {
            models[model_sym].instance for model_sym in pooled_model_syms
        }
        assert len(instances) == 1

    @staticmethod
    def _make_symbol_df(sym, dates):
        n = len(dates)
        close = np.arange(n, dtype=np.float64) + 1
        return pd.DataFrame(
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

    @staticmethod
    def _compressed_bar_dates(sym_df, interval):
        bars, _ = compress(
            sym_df[DataCol.DATE.value].to_numpy(),
            sym_df[DataCol.OPEN.value].to_numpy(),
            sym_df[DataCol.HIGH.value].to_numpy(),
            sym_df[DataCol.LOW.value].to_numpy(),
            sym_df[DataCol.CLOSE.value].to_numpy(),
            sym_df[DataCol.VOLUME.value].to_numpy(),
            interval,
        )
        return np.asarray(bars.dates, dtype="datetime64[ns]")

    @staticmethod
    def _bar_index(bar_dates, date):
        (indices,) = np.nonzero(bar_dates == date)
        assert len(indices) == 1
        return int(indices[0])

    @pytest.mark.parametrize("interval", ["weekly", 5])
    @pytest.mark.parametrize("lookahead", [1, 2, 3])
    def test_train_models_interval_honors_lookahead(
        self, scope, cache_date_fields, lookahead, interval
    ):
        sym = "SPY"
        dates = np.array(
            pd.date_range("2020-01-06", periods=30, freq="B"),
            dtype="datetime64[D]",
        )
        df = self._make_symbol_df(sym, dates)
        interval_data = IntervalData()
        interval_data.compressed[(sym, interval)] = compress_symbol_df(
            df, interval, frozenset(), 86400.0
        )
        recorded = {}

        def train_fn(symbol, train_df, test_df):
            recorded["train"] = train_df
            recorded["test"] = test_df
            return FakeModel(symbol, np.zeros(len(test_df)))

        m = model("la_model", train_fn, [])
        tf_model_name = model_interval_name(m.name, interval)
        train_dates = dates[:15]
        test_dates = dates[15:]
        mixin = ModelsMixin()
        models = mixin.train_models(
            model_syms=[ModelSymbol(tf_model_name, sym)],
            train_data=df[df[DataCol.DATE.value].isin(train_dates)],
            test_data=df[df[DataCol.DATE.value].isin(test_dates)],
            indicator_data={},
            cache_date_fields=cache_date_fields,
            interval_data=interval_data,
            lookahead=lookahead,
        )
        assert ModelSymbol(tf_model_name, sym) in models
        bar_dates = self._compressed_bar_dates(df, interval)
        train_idx = np.nonzero(np.isin(bar_dates, train_dates))[0]
        test_idx = np.nonzero(np.isin(bar_dates, test_dates))[0]
        got_train = recorded["train"][DataCol.DATE.value].to_numpy()
        got_test = recorded["test"][DataCol.DATE.value].to_numpy()
        # Test input is never truncated.
        assert np.array_equal(got_test, bar_dates[test_idx])
        # Kept train bars are a leading run of the untruncated selection.
        assert len(got_train) > 0
        assert np.array_equal(
            got_train, bar_dates[train_idx][: len(got_train)]
        )
        last_train_i = self._bar_index(bar_dates, got_train[-1])
        first_test_i = self._bar_index(bar_dates, got_test[0])
        assert first_test_i - last_train_i == lookahead
        if lookahead == 1:
            # Regression lock: lookahead=1 passes the train window through
            # unchanged, matching test_train_models_interval.
            assert np.array_equal(got_train, bar_dates[train_idx])

    @pytest.mark.parametrize("lookahead", [1, 2, 3])
    def test_train_models_pooled_interval_honors_lookahead(
        self, scope, cache_date_fields, lookahead
    ):
        # An every-5-bars interval bins each symbol from its own first bar,
        # so histories starting on different dates produce misaligned
        # compressed calendars and different first test compressed indices.
        interval = 5
        dates = np.array(
            pd.date_range("2020-01-06", periods=40, freq="B"),
            dtype="datetime64[D]",
        )
        sym_dates = {"SPY": dates, "AAPL": dates[3:]}
        frames = {
            sym: self._make_symbol_df(sym, sym_dates[sym]) for sym in sym_dates
        }
        df = pd.concat(frames.values(), ignore_index=True)
        interval_data = IntervalData()
        for sym, sym_df in frames.items():
            interval_data.compressed[(sym, interval)] = compress_symbol_df(
                sym_df, interval, frozenset(), 86400.0
            )
        recorded = {}

        def train_fn(symbols_arg, train_df, test_df):
            recorded["train"] = train_df
            recorded["test"] = test_df
            return FakeModel("pooled", np.array([1.0]))

        model("pooled_la_model", train_fn, [], pooled=True)
        tf_model_name = model_interval_name("pooled_la_model", interval)
        pooled_syms = frozenset(sym_dates)
        train_dates = dates[:20]
        test_dates = dates[20:]
        mixin = ModelsMixin()
        models = mixin.train_models(
            model_syms=[
                ModelSymbol(tf_model_name, sym) for sym in sorted(pooled_syms)
            ],
            train_data=df[df[DataCol.DATE.value].isin(train_dates)],
            test_data=df[df[DataCol.DATE.value].isin(test_dates)],
            indicator_data={},
            cache_date_fields=cache_date_fields,
            pooled_model_groups={(tf_model_name, 1): pooled_syms},
            interval_data=interval_data,
            lookahead=lookahead,
        )
        assert len(models) == len(pooled_syms)
        sym_col = DataCol.SYMBOL.value
        date_col = DataCol.DATE.value
        first_test_indices = {}
        for sym, sym_df in frames.items():
            bar_dates = self._compressed_bar_dates(sym_df, interval)
            sym_train = (
                recorded["train"]
                .loc[recorded["train"][sym_col] == sym, date_col]
                .to_numpy()
            )
            sym_test = (
                recorded["test"]
                .loc[recorded["test"][sym_col] == sym, date_col]
                .to_numpy()
            )
            assert len(sym_train) > 0
            assert len(sym_test) > 0
            last_train_i = self._bar_index(bar_dates, sym_train[-1])
            first_test_i = self._bar_index(bar_dates, sym_test[0])
            assert first_test_i - last_train_i == lookahead
            if lookahead == 1:
                train_idx = np.nonzero(np.isin(bar_dates, train_dates))[0]
                assert np.array_equal(sym_train, bar_dates[train_idx])
            first_test_indices[sym] = first_test_i
        # The premise that makes this test meaningful: the symbols reach
        # their test windows at different compressed indices, so a
        # truncation hoisted out of the per-symbol loop cannot satisfy
        # both gaps.
        assert len(set(first_test_indices.values())) == len(frames)

    def test_pooled_interval_cache_distinct_by_lookahead(
        self, scope, setup_enabled_model_cache, cache_date_fields
    ):
        interval = "weekly"
        dates = np.array(
            pd.date_range("2020-01-06", periods=20, freq="B"),
            dtype="datetime64[D]",
        )
        symbols = ["SPY", "AAPL"]
        frames = {sym: self._make_symbol_df(sym, dates) for sym in symbols}
        df = pd.concat(frames.values(), ignore_index=True)
        interval_data = IntervalData()
        for sym, sym_df in frames.items():
            interval_data.compressed[(sym, interval)] = compress_symbol_df(
                sym_df, interval, frozenset(), 86400.0
            )
        calls = []

        def train_fn(symbols_arg, train_df, test_df):
            calls.append(symbols_arg)
            return FakeModel("pooled", np.array([1.0]))

        model("pooled_la_cache", train_fn, [], pooled=True)
        tf_model_name = model_interval_name("pooled_la_cache", interval)
        pooled_syms = frozenset(symbols)
        train_dates = dates[:10]
        test_dates = dates[10:]
        mixin = ModelsMixin()
        kwargs = dict(
            model_syms=[
                ModelSymbol(tf_model_name, sym) for sym in sorted(pooled_syms)
            ],
            train_data=df[df[DataCol.DATE.value].isin(train_dates)],
            test_data=df[df[DataCol.DATE.value].isin(test_dates)],
            indicator_data={},
            cache_date_fields=cache_date_fields,
            pooled_model_groups={(tf_model_name, 1): pooled_syms},
            interval_data=interval_data,
        )
        mixin.train_models(**kwargs, lookahead=2)
        assert len(calls) == 1
        # A different lookahead trims a different train set, so the
        # lookahead=2 fit must not be served here.
        mixin.train_models(**kwargs, lookahead=1)
        assert len(calls) == 2
        # Same lookahead as the first run: served from cache.
        mixin.train_models(**kwargs, lookahead=2)
        assert len(calls) == 2

    @pytest.mark.parametrize("mode", ["per_symbol", "pooled", "lags"])
    def test_train_models_interval_lookahead_empties_train_warns(
        self, scope, cache_date_fields, mode
    ):
        interval = 2
        dates = np.array(
            ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"],
            dtype="datetime64[D]",
        )
        symbols = ["SPY", "AAPL"] if mode == "pooled" else ["SPY"]
        frames = {sym: self._make_symbol_df(sym, dates) for sym in symbols}
        df = pd.concat(frames.values(), ignore_index=True)
        interval_data = IntervalData()
        for sym, sym_df in frames.items():
            interval_data.compressed[(sym, interval)] = compress_symbol_df(
                sym_df, interval, frozenset(), 86400.0
            )
        pooled_model_groups = None
        if mode == "pooled":

            def train_fn(symbols_arg, train_df, test_df):
                return FakeModel("pooled", np.array([1.0]))

            model("empty_la_model", train_fn, [], pooled=True)
        elif mode == "lags":

            def train_fn(symbol, train_df, test_df, **kwargs):
                return FakeModel(symbol, np.zeros(len(test_df)))

            model("empty_la_model", train_fn, [], lags=1)
        else:

            def train_fn(symbol, train_df, test_df):
                return FakeModel(symbol, np.zeros(len(test_df)))

            model("empty_la_model", train_fn, [])
        tf_model_name = model_interval_name("empty_la_model", interval)
        if mode == "pooled":
            pooled_model_groups = {(tf_model_name, 1): frozenset(symbols)}
        model_syms = [
            ModelSymbol(tf_model_name, sym) for sym in sorted(symbols)
        ]
        # Two compressed bars total: one train, one test. lookahead=2
        # exceeds the compressed train span, leaving zero train bars.
        train_dates = dates[:2]
        test_dates = dates[2:]
        mixin = ModelsMixin()
        with pytest.warns(UserWarning, match="lookahead"):
            models = mixin.train_models(
                model_syms=model_syms,
                train_data=df[df[DataCol.DATE.value].isin(train_dates)],
                test_data=df[df[DataCol.DATE.value].isin(test_dates)],
                indicator_data={},
                cache_date_fields=cache_date_fields,
                pooled_model_groups=pooled_model_groups,
                interval_data=interval_data,
                lookahead=2,
            )
        assert set(models.keys()) == set(model_syms)

    def test_pooled_interval_partial_empty_is_silent(
        self, scope, cache_date_fields
    ):
        """One pooled symbol's train contribution emptying is by design
        silent: the symbol is skipped (matching the pre-existing pooled
        behavior for symbols with no train rows) and the group still trains
        on the surviving symbols. Only a fully emptied pooled train set
        warns.
        """
        interval = 2
        lookahead = 2
        long_dates = np.array(
            pd.date_range("2020-01-06", periods=12, freq="B"),
            dtype="datetime64[D]",
        )
        # The short-history symbol's compressed train span is smaller than
        # lookahead, so its whole train contribution is dropped.
        short_dates = long_dates[6:]
        frames = {
            "LONG": self._make_symbol_df("LONG", long_dates),
            "SHRT": self._make_symbol_df("SHRT", short_dates),
        }
        df = pd.concat(frames.values(), ignore_index=True)
        interval_data = IntervalData()
        for sym, sym_df in frames.items():
            interval_data.compressed[(sym, interval)] = compress_symbol_df(
                sym_df, interval, frozenset(), 86400.0
            )
        train_frames = []

        def train_fn(symbols_arg, train_df, test_df):
            train_frames.append(train_df.copy())
            return FakeModel("pooled", np.array([1.0]))

        model("partial_empty_model", train_fn, [], pooled=True)
        tf_model_name = model_interval_name("partial_empty_model", interval)
        symbols = frozenset(frames)
        model_syms = [
            ModelSymbol(tf_model_name, sym) for sym in sorted(symbols)
        ]
        train_dates = long_dates[:8]
        test_dates = long_dates[8:]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            models = ModelsMixin().train_models(
                model_syms=model_syms,
                train_data=df[df[DataCol.DATE.value].isin(train_dates)],
                test_data=df[df[DataCol.DATE.value].isin(test_dates)],
                indicator_data={},
                cache_date_fields=cache_date_fields,
                pooled_model_groups={(tf_model_name, 1): symbols},
                interval_data=interval_data,
                lookahead=lookahead,
            )
        assert set(models.keys()) == set(model_syms)
        assert len(train_frames) == 1
        # SHRT's contribution emptied; LONG's survived, trimmed to the
        # lookahead hold-out computed from its own compressed history.
        assert set(train_frames[0][DataCol.SYMBOL.value]) == {"LONG"}
        bars = interval_data.compressed[("LONG", interval)].bars
        bar_dates = np.asarray(bars.dates)
        train_idx = np.nonzero(
            np.isin(bar_dates, train_dates.astype("datetime64[ns]"))
        )[0]
        test_idx = np.nonzero(
            np.isin(bar_dates, test_dates.astype("datetime64[ns]"))
        )[0]
        kept = np.nonzero(
            np.isin(
                bar_dates,
                train_frames[0][DataCol.DATE.value].to_numpy(
                    dtype="datetime64[ns]"
                ),
            )
        )[0]
        assert kept.size == train_idx.size - 1
        assert test_idx[0] - kept[-1] == lookahead

    def test_interval_cache_not_shared_across_lookaheads(
        self, scope, setup_enabled_model_cache, cache_date_fields
    ):
        sym = "SPY"
        interval = "weekly"
        dates = np.array(
            pd.date_range("2020-01-06", periods=20, freq="B"),
            dtype="datetime64[D]",
        )
        df = self._make_symbol_df(sym, dates)
        interval_data = IntervalData()
        interval_data.compressed[(sym, interval)] = compress_symbol_df(
            df, interval, frozenset(), 86400.0
        )
        base_calls = []
        tf_calls = []

        def base_train_fn(symbol, train_df, test_df):
            base_calls.append(symbol)
            return FakeModel(symbol, np.zeros(len(test_df)))

        def tf_train_fn(symbol, train_df, test_df):
            tf_calls.append(symbol)
            return FakeModel(symbol, np.zeros(len(test_df)))

        model("base_la_cache", base_train_fn, [])
        model("tf_la_cache", tf_train_fn, [])
        tf_model_name = model_interval_name("tf_la_cache", interval)
        train_dates = dates[:10]
        test_dates = dates[10:]
        mixin = ModelsMixin()
        kwargs = dict(
            model_syms=[
                ModelSymbol("base_la_cache", sym),
                ModelSymbol(tf_model_name, sym),
            ],
            train_data=df[df[DataCol.DATE.value].isin(train_dates)],
            test_data=df[df[DataCol.DATE.value].isin(test_dates)],
            indicator_data={},
            cache_date_fields=cache_date_fields,
            interval_data=interval_data,
        )
        mixin.train_models(**kwargs, lookahead=1)
        assert len(base_calls) == 1
        assert len(tf_calls) == 1
        mixin.train_models(**kwargs, lookahead=2)
        # The interval model fits on a lookahead-truncated compressed train
        # set, so its cache entry is keyed by lookahead and it retrains.
        assert len(tf_calls) == 2
        # The base model's train set is unchanged by lookahead: cache hit.
        assert len(base_calls) == 1


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
            lambda sym, train, test, **kwargs: (object(), ("close",)),
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


def test_indicator_values_for_dates_matches_get_indexer(data_source_df):
    from pybroker.model import _indicator_values_for_dates

    sym_df = data_source_df[data_source_df["symbol"] == "SPY"].sort_values(
        "date"
    )
    ind = sym_df.set_index("date")["close"]
    dates = sym_df["date"].to_numpy(dtype="datetime64[ns]")
    aligned = _indicator_values_for_dates(ind, dates)
    positions = ind.index.get_indexer(dates)
    expected = np.full(len(dates), np.nan, dtype=np.float64)
    valid = positions >= 0
    expected[valid] = ind.to_numpy(dtype=np.float64)[positions[valid]]
    assert np.array_equal(aligned, expected, equal_nan=True)

    sparse_dates = dates[::7]
    aligned_sparse = _indicator_values_for_dates(ind, sparse_dates)
    positions_sparse = ind.index.get_indexer(sparse_dates)
    expected_sparse = np.full(len(sparse_dates), np.nan, dtype=np.float64)
    valid_sparse = positions_sparse >= 0
    expected_sparse[valid_sparse] = ind.to_numpy(dtype=np.float64)[
        positions_sparse[valid_sparse]
    ]
    assert np.array_equal(aligned_sparse, expected_sparse, equal_nan=True)


def _model_inputs_equal(left, right) -> bool:
    if left.columns != right.columns:
        return False
    if not np.array_equal(left.dates, right.dates):
        return False
    for col in left.columns:
        if col not in left.arrays or col not in right.arrays:
            return False
        left_arr = left.arrays[col]
        right_arr = right.arrays[col]
        if left_arr.dtype == object or right_arr.dtype == object:
            if not np.array_equal(left_arr, right_arr):
                return False
        elif not np.array_equal(left_arr, right_arr, equal_nan=True):
            return False
    return True


def test_symbol_model_input_from_store_matches_frame(
    data_source_df, ind_data, indicators
):
    from pybroker.model import (
        _symbol_model_input,
        _symbol_model_input_from_store,
    )
    from pybroker.scope import symbol_array_store_from_frame

    sym = "SPY"
    ind_names = tuple(ind.name for ind in indicators)
    store = symbol_array_store_from_frame(data_source_df)
    frame_input = _symbol_model_input(sym, data_source_df, ind_data, ind_names)
    store_input = _symbol_model_input_from_store(
        store, sym, ind_data, ind_names
    )
    assert _model_inputs_equal(frame_input, store_input)


def test_pooled_model_input_from_store_matches_frame(
    data_source_df, ind_data, indicators
):
    from pybroker.model import (
        _pooled_model_input,
        _pooled_model_input_from_store,
    )
    from pybroker.scope import symbol_array_store_from_frame

    symbols = frozenset(data_source_df["symbol"].unique())
    ind_names = tuple(ind.name for ind in indicators)
    store = symbol_array_store_from_frame(data_source_df)
    frame_input = _pooled_model_input(
        data_source_df, symbols, ind_data, ind_names
    )
    store_input = _pooled_model_input_from_store(
        store, symbols, ind_data, ind_names
    )
    assert _model_inputs_equal(frame_input, store_input)


def test_train_models_reuses_history_store(
    scope,
    indicators,
    train_data,
    test_data,
    ind_data,
    cache_date_fields,
):
    from pybroker.scope import (
        merge_symbol_array_stores,
        symbol_array_store_from_frame,
    )

    sym = sorted(train_data["symbol"].unique())[0]
    # lags= is what actually reads the merged store, so without it this would
    # pass on laziness alone and never exercise the supplied store.
    model(
        "store_model",
        lambda symbol, train_df, test_df, **kwargs: FakeModel(
            symbol, np.array([1.0])
        ),
        indicators,
        lags=2,
        lag_cols=("close",),
    )
    train_store = symbol_array_store_from_frame(train_data)
    test_store = symbol_array_store_from_frame(test_data)
    history_store = merge_symbol_array_stores(train_store, test_store)
    mixin = ModelsMixin()
    with (
        patch(
            "pybroker.model.symbol_array_store_from_frame"
        ) as store_from_frame,
        patch("pybroker.model.merge_symbol_array_stores") as merge_stores,
    ):
        mixin.train_models(
            [ModelSymbol("store_model", sym)],
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
            history_store=history_store,
            train_store=train_store,
            test_store=test_store,
        )
        store_from_frame.assert_not_called()
        # The caller already merged train+test; redoing it would concatenate
        # every column of every symbol a second time, on every window.
        merge_stores.assert_not_called()


def test_train_models_skips_history_store_without_lags(
    scope,
    indicators,
    train_data,
    test_data,
    ind_data,
    cache_date_fields,
):
    """Only lagged models read the merged store, so it must stay unbuilt."""
    sym = sorted(train_data["symbol"].unique())[0]
    model(
        "no_lags_model",
        lambda symbol, train_df, test_df: FakeModel(symbol, np.array([1.0])),
        indicators,
    )
    mixin = ModelsMixin()
    with patch("pybroker.model._history_store") as history_store_fn:
        models = mixin.train_models(
            [ModelSymbol("no_lags_model", sym)],
            train_data,
            test_data,
            ind_data,
            cache_date_fields,
        )
        history_store_fn.assert_not_called()
    assert len(models) == 1


def test_get_cached_models_pooled_single_pass(
    scope,
    setup_enabled_model_cache,
    cache_date_fields,
):
    from pybroker.cache import ModelCacheKey
    from pybroker.model import CachedModel, ModelsMixin

    model(
        "pooled_cache",
        lambda symbols, train_df, test_df: FakeModel(
            "pooled_cache", np.array([1.0])
        ),
        [],
        pooled=True,
    )
    pooled_symbols = frozenset({"SPY", "AAPL"})
    model_syms = [
        ModelSymbol("pooled_cache", sym) for sym in sorted(pooled_symbols)
    ]
    pooled_model_groups = {("pooled_cache", 1): pooled_symbols}
    shared = FakeModel("pooled_cache", np.array([1.0]))
    input_cols = ("close",)
    for sym in pooled_symbols:
        cache_key = ModelCacheKey.from_date_fields(
            symbol=sym,
            model_name="pooled_cache",
            fields=cache_date_fields,
            # The key carries the group composition, so a run over a different
            # symbol set cannot be served this fit.
            pooled_symbols=pooled_symbols,
        )
        scope.model_cache.set(cache_key, CachedModel(shared, input_cols))
    get_calls: list[ModelCacheKey] = []
    original_get = scope.model_cache.get

    def counting_get(key):
        get_calls.append(key)
        return original_get(key)

    scope.model_cache.get = counting_get
    mixin = ModelsMixin()
    models, uncached = mixin._get_cached_models(
        model_syms, cache_date_fields, pooled_model_groups
    )
    assert len(models) == 2
    assert uncached == []
    assert len(get_calls) == 2


def test_uses_hyperparams_when_indicator_has_hyperparam(scope):
    """A model trained on a tuned indicator must bypass the model cache.

    ModelCacheKey carries no hyperparameter values, so caching such a model
    would serve a fit made under a different value on the next run.
    """
    from pybroker.optimize import hyperparam

    tuned_ind = indicator(
        "tuned_ind",
        lambda data, period: data.close,
        period=hyperparam("period", default=5, low=5, high=20, step=5),
    )
    plain_ind = indicator("plain_ind", lambda data: data.close)
    model("tuned_model", lambda *a, **k: None, indicators=[tuned_ind])
    model("plain_model", lambda *a, **k: None, indicators=[plain_ind])
    mixin = ModelsMixin()
    assert mixin._uses_hyperparams("tuned_model")
    assert not mixin._uses_hyperparams("plain_model")
    # Interval-suffixed names resolve to the same source.
    assert mixin._uses_hyperparams("tuned_model@weekly")
    assert not mixin._uses_hyperparams("unregistered_model")


def test_logger_getstate_drops_progress_bar(scope):
    """The scope is pickled to workers, and a live progress bar is not
    picklable because it holds the caller's output stream."""
    import pickle

    from pybroker.scope import StaticScope

    static_scope = StaticScope.instance()
    static_scope.logger._start_progress_bar("test", 10)
    assert static_scope.logger._progress_bar is not None
    try:
        restored = pickle.loads(pickle.dumps(static_scope))
    finally:
        static_scope.logger._progress_bar = None
    assert restored.logger._progress_bar is None


def test_pooled_model_cache_key_carries_group_composition(
    scope, setup_enabled_model_cache, cache_date_fields
):
    """A pooled model must not be served to a different symbol set.

    The model is stored under one plain key per group member, so a run over
    {SPY, AAPL, TSLA} overwrites the entries a run over {SPY, AAPL} wrote, and
    the smaller run is then handed a fit it never asked for.
    """
    from pybroker.cache import ModelCacheKey

    small = ModelCacheKey.from_date_fields(
        symbol="SPY",
        model_name="pooled",
        fields=cache_date_fields,
        pooled_symbols=frozenset({"SPY", "AAPL"}),
    )
    large = ModelCacheKey.from_date_fields(
        symbol="SPY",
        model_name="pooled",
        fields=cache_date_fields,
        pooled_symbols=frozenset({"SPY", "AAPL", "TSLA"}),
    )
    per_symbol = ModelCacheKey.from_date_fields(
        symbol="SPY", model_name="pooled", fields=cache_date_fields
    )
    assert small != large
    assert small != per_symbol
    # Composition is order-independent.
    assert small == ModelCacheKey.from_date_fields(
        symbol="SPY",
        model_name="pooled",
        fields=cache_date_fields,
        pooled_symbols=["AAPL", "SPY"],
    )


def test_model_cache_key_distinct_by_lookahead(cache_date_fields):
    """An interval-bound model holds out ``lookahead`` compressed bars, so
    identical date fields still describe different fits across lookaheads.

    Base-timeframe models are fully described by their train start/end, so
    their keys stay lookahead-free.
    """
    from pybroker.cache import ModelCacheKey
    from pybroker.model import _model_cache_lookahead

    tf_model_name = model_interval_name("m", "weekly")
    assert _model_cache_lookahead(tf_model_name, 2) == 2
    assert _model_cache_lookahead("m", 2) is None
    tf_one = ModelCacheKey.from_date_fields(
        symbol="SPY",
        model_name=tf_model_name,
        fields=cache_date_fields,
        lookahead=_model_cache_lookahead(tf_model_name, 1),
    )
    tf_two = ModelCacheKey.from_date_fields(
        symbol="SPY",
        model_name=tf_model_name,
        fields=cache_date_fields,
        lookahead=_model_cache_lookahead(tf_model_name, 2),
    )
    assert tf_one != tf_two
    base_one = ModelCacheKey.from_date_fields(
        symbol="SPY",
        model_name="m",
        fields=cache_date_fields,
        lookahead=_model_cache_lookahead("m", 1),
    )
    base_two = ModelCacheKey.from_date_fields(
        symbol="SPY",
        model_name="m",
        fields=cache_date_fields,
        lookahead=_model_cache_lookahead("m", 2),
    )
    assert base_one.lookahead is None
    assert base_one == base_two
