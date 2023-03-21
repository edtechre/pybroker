"""Contains common test fixtures."""

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

import joblib
import numpy as np
import os
import pandas as pd
import pytest
from decimal import Decimal
from pybroker.cache import (
    clear_data_source_cache,
    clear_indicator_cache,
    clear_model_cache,
    disable_data_source_cache,
    disable_indicator_cache,
    disable_model_cache,
    enable_data_source_cache,
    enable_model_cache,
    enable_indicator_cache,
)
from pybroker.common import PriceType
from pybroker.indicator import IndicatorSymbol, indicator
from pybroker.model import ModelSymbol, TrainedModel, model
from pybroker.scope import (
    ColumnScope,
    IndicatorScope,
    ModelInputScope,
    PendingOrder,
    PendingOrderScope,
    PredictionScope,
    StaticScope,
)
from pybroker.vect import highv, lowv, sumv

MODEL_NAME = "fake_model"


class FakeModel:
    def __init__(self, symbol, preds):
        self.symbol = symbol
        self._preds = preds

    def predict(self, _):
        return self._preds


@pytest.fixture()
def data_source_df():
    return joblib.load(
        os.path.join(os.path.dirname(__file__), "testdata/daily_1.joblib")
    )


@pytest.fixture()
def symbols(data_source_df):
    return list(data_source_df["symbol"].unique())


@pytest.fixture()
def symbol(symbols):
    return symbols[0]


@pytest.fixture()
def dates(data_source_df):
    return list(data_source_df["date"].unique())


@pytest.fixture()
def scope():
    scope = StaticScope.instance()
    yield scope
    StaticScope.__instance = None


@pytest.fixture()
def hhv_ind(scope):
    return indicator("hhv", lambda bar_data, n: highv(bar_data.close, n), n=5)


@pytest.fixture()
def llv_ind(scope):
    return indicator("llv", lambda bar_data, n: lowv(bar_data.close, n), n=3)


@pytest.fixture()
def sumv_ind(scope):
    return indicator("sumv", lambda bar_data, n: sumv(bar_data.close, n), n=2)


@pytest.fixture()
def ind_data(data_source_df, symbols, hhv_ind, llv_ind, sumv_ind):
    return {
        **{
            IndicatorSymbol(ind.name, sym): ind(
                data_source_df[data_source_df["symbol"] == sym]
            )
            for sym in symbols
            for ind in [hhv_ind, llv_ind, sumv_ind]
        }
    }


@pytest.fixture()
def indicators(hhv_ind, llv_ind, sumv_ind):
    return [hhv_ind, llv_ind, sumv_ind]


@pytest.fixture()
def ind_names(indicators):
    return list(map(lambda x: x.name, indicators))


@pytest.fixture()
def ind_name(ind_names):
    return ind_names[0]


@pytest.fixture()
def ind_df(data_source_df, hhv_ind, llv_ind, sumv_ind):
    return pd.DataFrame(
        {
            hhv_ind.name: hhv_ind(data_source_df),
            llv_ind.name: llv_ind(data_source_df),
            sumv_ind.name: sumv_ind(data_source_df),
        }
    )


@pytest.fixture(params=[True, False])
def model_source(scope, data_source_df, indicators, request):
    return model(
        MODEL_NAME,
        lambda sym, *_: FakeModel(
            sym,
            np.full(
                data_source_df[data_source_df["symbol"] == sym].shape[0], 100
            ),
        ),
        indicators,
        pretrained=request.param,
    )


@pytest.fixture()
def preds(symbols, data_source_df):
    return {
        sym: np.random.random(
            data_source_df[data_source_df["symbol"] == sym].shape[0]
        )
        for sym in symbols
    }


@pytest.fixture(params=[True, False])
def trained_models(model_source, preds, symbols, data_source_df, request):
    trained_models = {}
    for sym in symbols:
        model_sym = ModelSymbol(MODEL_NAME, sym)
        if request.param:

            def predict_fn(preds_array):
                def _(model, df):
                    return preds_array

                return _

            trained_models[model_sym] = TrainedModel(
                MODEL_NAME,
                FakeModel(sym, None),
                predict_fn=predict_fn(preds[sym]),
            )
        else:
            trained_models[model_sym] = TrainedModel(
                MODEL_NAME, FakeModel(sym, preds[sym]), predict_fn=None
            )
    return trained_models


@pytest.fixture()
def trained_model(trained_models):
    return list(trained_models.values())[0]


@pytest.fixture()
def col_scope(data_source_df):
    return ColumnScope(data_source_df.set_index(["symbol", "date"]))


@pytest.fixture()
def ind_scope(ind_data, dates):
    return IndicatorScope(ind_data, dates)


@pytest.fixture()
def input_scope(col_scope, ind_scope):
    return ModelInputScope(col_scope, ind_scope)


@pytest.fixture()
def pred_scope(trained_models, input_scope):
    return PredictionScope(trained_models, input_scope)


@pytest.fixture()
def pending_orders():
    return (
        PendingOrder(
            id=1,
            type="buy",
            symbol="SPY",
            created=np.datetime64("2020-01-05"),
            exec_date=np.datetime64("2020-01-10"),
            shares=Decimal(100),
            limit_price=None,
            fill_price=PriceType.MIDDLE,
        ),
        PendingOrder(
            id=2,
            type="sell",
            symbol="AAPL",
            created=np.datetime64("2020-01-06"),
            exec_date=np.datetime64("2020-01-08"),
            shares=Decimal(200),
            limit_price=Decimal(99),
            fill_price=PriceType.AVERAGE,
        ),
    )


@pytest.fixture()
def pending_order_scope(pending_orders):
    scope = PendingOrderScope()
    for order in pending_orders:
        scope.add(
            type=order.type,
            symbol=order.symbol,
            created=order.created,
            exec_date=order.exec_date,
            shares=order.shares,
            limit_price=order.limit_price,
            fill_price=order.fill_price,
        )
    return scope


@pytest.fixture()
def setup_enabled_model_cache(tmp_path):
    enable_model_cache("test", tmp_path)
    yield
    clear_model_cache()
    disable_model_cache()


@pytest.fixture(params=[True, False])
def setup_model_cache(tmp_path, request):
    if request.param:
        enable_model_cache("test", tmp_path)
    else:
        disable_model_cache()
    yield
    if request.param:
        clear_model_cache()
    disable_model_cache()


@pytest.fixture()
def setup_enabled_ds_cache(tmp_path):
    enable_data_source_cache("test", tmp_path)
    yield
    clear_data_source_cache()
    disable_data_source_cache()


@pytest.fixture(params=[True, False])
def setup_ds_cache(tmp_path, request):
    if request.param:
        enable_data_source_cache("test", tmp_path)
    else:
        disable_data_source_cache()
    yield
    if request.param:
        clear_data_source_cache()
    disable_data_source_cache()


@pytest.fixture()
def setup_enabled_ind_cache(tmp_path):
    enable_indicator_cache("test", tmp_path)
    yield
    clear_indicator_cache()
    disable_indicator_cache()


@pytest.fixture(params=[True, False])
def setup_ind_cache(tmp_path, request):
    if request.param:
        enable_indicator_cache("test", tmp_path)
    else:
        disable_indicator_cache()
    yield
    if request.param:
        clear_indicator_cache()
    disable_indicator_cache()
