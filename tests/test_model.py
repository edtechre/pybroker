"""Unit tests for model.py module."""

"""Copyright (C) 2023 Edward West

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from pybroker.cache import CacheDateFields
from .fixtures import *  # noqa: F401
from unittest.mock import Mock
from pybroker.common import ModelSymbol, TrainedModel, to_datetime
from pybroker.model import ModelLoader, ModelsMixin, ModelTrainer, model
import pandas as pd
import pytest
import re

TF_SECONDS = 60
BETWEEN_TIME = ("10:00", "15:30")


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
def end_date(train_data):
    return to_datetime(sorted(train_data["date"].unique())[-1])


@pytest.fixture()
def model_syms(train_data, model_source):
    return [
        ModelSymbol(model_source.name, sym)
        for sym in train_data["symbol"].unique()
    ]


@pytest.mark.parametrize("pretrained", [True, False])
def test_model(indicators, pretrained):
    def input_data_fn(df):
        pass

    def predict_fn(model, df):
        pass

    name = f"pretrained={pretrained}"
    source = model(
        name,
        lambda x: x,
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


class TestModelSource:
    @pytest.mark.parametrize("clazz", [ModelLoader, ModelTrainer])
    def test_model_prepare_input_fn(self, data_source_df, clazz):
        prepare_fn = Mock()
        source = clazz("model_source", lambda x: x, [], prepare_fn, None, {})
        source.prepare_input_data(data_source_df)
        prepare_fn.assert_called_once_with(data_source_df)

    @pytest.mark.parametrize("clazz", [ModelLoader, ModelTrainer])
    def test_model_prepare_input_fn_when_empty_data(self, clazz):
        source = clazz("model_source", lambda x: x, [], None, None, {})
        df = source.prepare_input_data(pd.DataFrame())
        assert df.empty

    @pytest.mark.parametrize("clazz", [ModelLoader, ModelTrainer])
    def test_model_prepare_input_fn_when_fn_none(
        self, ind_df, ind_names, clazz
    ):
        source = clazz("model_source", lambda x: x, ind_names, None, None, {})
        df = source.prepare_input_data(ind_df)
        assert df.equals(ind_df)

    @pytest.mark.parametrize("clazz", [ModelLoader, ModelTrainer])
    def test_model_prepare_input_fn_when_indicators_not_found_then_error(
        self, ind_df, clazz
    ):
        source = clazz("model_source", lambda x: x, ["foo"], None, None, {})
        with pytest.raises(
            ValueError,
            match=re.escape("Indicator 'foo' not found in DataFrame."),
        ):
            source.prepare_input_data(ind_df)

    def test_model_loader_call_with_kwargs(self):
        load_fn = Mock()
        kwargs = {"a": 1, "b": 2}
        ModelLoader("loader", load_fn, [], None, None, kwargs)("SPY")
        load_fn.assert_called_once_with("SPY", **kwargs)

    def test_model_trainer_call_with_kwargs(self, train_data, test_data):
        train_fn = Mock()
        kwargs = {"a": 1, "b": 2}
        ModelTrainer("trainer", train_fn, [], None, None, kwargs)(
            "SPY", train_data, test_data
        )
        train_fn.assert_called_once_with(
            "SPY", train_data, test_data, **kwargs
        )

    def test_model_trainer_repr(self):
        trainer = ModelTrainer(
            "trainer", lambda x: x, [], None, None, {"a": 1}
        )
        assert repr(trainer) == "ModelTrainer('trainer', {'a': 1})"

    def test_model_loader_repr(self):
        trainer = ModelLoader("loader", lambda x: x, [], None, None, {"a": 1})
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
            pytest.lazy_fixture("test_data"),
        ],
    )
    def test_train_models(
        self,
        model_syms,
        train_data,
        param_test_data,
        ind_data,
        cache_date_fields,
    ):
        mixin = ModelsMixin()
        models = mixin.train_models(
            model_syms,
            train_data,
            param_test_data,
            ind_data,
            cache_date_fields,
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
