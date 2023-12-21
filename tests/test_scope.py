"""Unit tests for scope.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import pybroker
import pytest
import re
from .fixtures import *
from decimal import Decimal
from pybroker.common import PriceType
from pybroker.indicator import IndicatorSymbol
from pybroker.model import model
from pybroker.scope import (
    PriceScope,
    enable_logging,
    enable_progress_bar,
    disable_logging,
    disable_progress_bar,
    get_signals,
    param,
    register_columns,
    unregister_columns,
)
from unittest.mock import Mock


@pytest.fixture(params=[10, None])
def end_index(request):
    return request.param


@pytest.fixture()
def mock_logger(scope):
    logger, scope.logger = scope.logger, Mock()
    yield scope.logger
    scope.logger = logger


def test_register_columns(scope):
    scope.custom_data_cols = set()
    register_columns("a")
    register_columns("b", "b", "c")
    register_columns(["d", "e"], "c")
    expected = {"a", "b", "c", "d", "e"}
    assert scope.custom_data_cols == expected
    assert scope.all_data_cols == scope.default_data_cols | expected


def test_register_columns_when_frozen_then_error(scope):
    scope.freeze_data_cols()
    with pytest.raises(
        ValueError,
        match=re.escape("Cannot modify columns when strategy is running."),
    ):
        register_columns("a")
    scope.unfreeze_data_cols()


def test_unregister_columns(scope):
    scope.custom_data_cols = set()
    register_columns("a", "b", "c", "d", "e")
    unregister_columns("a", "b")
    unregister_columns("c")
    unregister_columns(["c"], "d")
    assert scope.custom_data_cols == {"e"}
    assert scope.all_data_cols == scope.default_data_cols | {"e"}


def test_unregister_columns_when_frozen_then_error(scope):
    scope.freeze_data_cols()
    with pytest.raises(
        ValueError,
        match=re.escape("Cannot modify columns when strategy is running."),
    ):
        unregister_columns("a")
    scope.unfreeze_data_cols()


def test_enable_logging(mock_logger):
    enable_logging()
    mock_logger.enable.assert_called_once()


def test_disable_logging(mock_logger):
    disable_logging()
    mock_logger.disable.assert_called_once()


def test_enable_progress_bar(mock_logger):
    enable_progress_bar()
    mock_logger.enable_progress_bar.assert_called_once()


def test_disable_progress_bar(mock_logger):
    disable_progress_bar()
    mock_logger.disable_progress_bar.assert_called_once()


def test_param_when_empty():
    assert param("bar") is None


@pytest.mark.parametrize("value", [42, None])
def test_param_when_set_and_get(value):
    param("foo", value)
    assert param("foo") == value


def test_param_when_set_to_none():
    param("baz", 11)
    assert param("baz") == 11
    param("baz", None)
    assert param("baz") is None


class TestStaticScope:
    def test_set_and_get_indicator(self, scope, hhv_ind):
        scope.set_indicator(hhv_ind)
        assert scope.has_indicator(hhv_ind.name)
        assert scope.get_indicator(hhv_ind.name) == hhv_ind

    def test_get_indicator_when_not_found_then_error(self, scope):
        with pytest.raises(
            ValueError, match=re.escape("Indicator 'foo' does not exist.")
        ):
            scope.get_indicator("foo")

    def test_set_and_get_model_source(self, scope, model_source):
        scope.set_model_source(model_source)
        assert scope.has_model_source(model_source.name)
        assert scope.get_model_source(model_source.name) == model_source

    def test_get_model_source_when_not_found_then_error(self, scope):
        with pytest.raises(
            ValueError, match=re.escape("ModelSource 'foo' does not exist.")
        ):
            scope.get_model_source("foo")

    def test_get_indicator_names(self, scope, model_source, ind_names):
        scope.set_model_source(model_source)
        assert set(scope.get_indicator_names(model_source.name)) == set(
            ind_names
        )


class TestColumnScope:
    def _assert_length(self, values, end_index, data_source_df, sym):
        df = data_source_df[data_source_df["symbol"] == sym]
        expected = df.shape[0] if end_index is None else end_index
        assert len(values) == expected

    def test_fetch_dict(self, col_scope, data_source_df, symbols, end_index):
        cols = ["date", "close"]
        result = col_scope.fetch_dict(symbols[0], cols, end_index)
        assert set(result.keys()) == set(cols)
        for value in result.values():
            self._assert_length(value, end_index, data_source_df, symbols[0])

    def test_fetch(self, col_scope, data_source_df, symbols, end_index):
        values = col_scope.fetch(symbols[0], "close", end_index)
        assert isinstance(values, np.ndarray)
        self._assert_length(values, end_index, data_source_df, symbols[0])

    def test_fetch_when_cached(self, col_scope, data_source_df, symbols):
        col_scope.fetch(symbols[0], "close", 1)
        values = col_scope.fetch(symbols[0], "close", 2)
        assert isinstance(values, np.ndarray)
        self._assert_length(values, 2, data_source_df, symbols[0])

    def test_fetch_dict_when_empty_names(self, col_scope, symbols, end_index):
        result = col_scope.fetch_dict(symbols[0], [], end_index)
        assert not len(result)

    def test_fetch_dict_when_name_not_found(
        self, col_scope, symbols, end_index
    ):
        result = col_scope.fetch_dict(symbols[0], ["foo"], end_index)
        assert result["foo"] is None

    def test_fetch_when_name_not_found(self, col_scope, symbols, end_index):
        assert col_scope.fetch(symbols[0], "foo", end_index) is None

    def test_fetch_when_symbol_not_found_then_error(
        self, col_scope, end_index
    ):
        with pytest.raises(
            ValueError, match=re.escape("Symbol not found: FOO.")
        ):
            col_scope.fetch("FOO", "close", end_index)

    def test_fetch_dict_when_symbol_not_found_then_error(
        self, col_scope, end_index
    ):
        with pytest.raises(
            ValueError, match=re.escape("Symbol not found: FOO.")
        ):
            col_scope.fetch_dict("FOO", ["close"], end_index)

    def test_fetch_dict_when_cached(
        self, col_scope, data_source_df, symbols, end_index
    ):
        cols = ["date", "close"]
        col_scope.fetch_dict(symbols[0], cols, end_index)
        result = col_scope.fetch_dict(symbols[0], cols, end_index)
        assert set(result.keys()) == set(cols)
        for value in result.values():
            self._assert_length(value, end_index, data_source_df, symbols[0])

    def test_bar_data_from_data_columns(
        self, col_scope, data_source_df, symbols, end_index
    ):
        register_columns("adj_close")
        bar_data = col_scope.bar_data_from_data_columns(symbols[0], end_index)
        sym_df = data_source_df[data_source_df["symbol"] == symbols[0]]
        for col in ("open", "high", "low", "close", "volume", "adj_close"):
            assert (
                getattr(bar_data, col) == sym_df[col].to_numpy()[:end_index]
            ).all()
        unregister_columns("adj_close")


class TestIndicatorScope:
    def test_fetch(self, ind_scope, symbol, ind_data, ind_name, end_index):
        result = ind_scope.fetch(symbol, ind_name, end_index)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(
            result,
            ind_data[IndicatorSymbol(ind_name, symbol)].values[:end_index],
            equal_nan=True,
        )

    def test_fetch_when_cached(
        self, ind_scope, symbol, ind_data, ind_name, end_index
    ):
        ind_scope.fetch(symbol, ind_name, end_index)
        result = ind_scope.fetch(symbol, ind_name, end_index)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(
            result,
            ind_data[IndicatorSymbol(ind_name, symbol)].values[:end_index],
            equal_nan=True,
        )

    @pytest.mark.parametrize("sym, name", [("FOO", "hhv"), ("SPY", "foo")])
    def test_fetch_when_not_found_then_error(self, ind_scope, sym, name):
        with pytest.raises(
            ValueError,
            match=re.escape(f"Indicator {name!r} not found for {sym}."),
        ):
            ind_scope.fetch(sym, name)


class TestModelInputScope:
    def test_fetch(
        self, input_scope, model_source, symbol, data_source_df, end_index
    ):
        df = data_source_df[data_source_df["symbol"] == symbol]
        result = input_scope.fetch(symbol, model_source.name, end_index)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == set(model_source.indicators)
        assert (
            result.shape[0] == df.shape[0] if end_index is None else end_index
        )

    def test_fetch_when_input_fn(
        self,
        scope,
        indicators,
        input_scope,
        symbol,
        data_source_df,
        end_index,
        trained_model,
    ):
        scope.custom_data_cols = set()
        expected_cols = {"hhv", "llv", "sumv"}

        def input_fn(df):
            assert set(df.columns) == expected_cols
            df["foo"] = np.ones(len(df["hhv"]))
            return df

        model_source = model(
            trained_model.name,
            lambda *_: trained_model,
            indicators,
            input_data_fn=input_fn,
        )
        df = data_source_df[data_source_df["symbol"] == symbol]
        result = input_scope.fetch(symbol, model_source.name, end_index)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"foo"} | expected_cols
        assert (
            result.shape[0] == df.shape[0] if end_index is None else end_index
        )

    def test_fetch_when_cached(
        self, input_scope, model_source, symbol, data_source_df, end_index
    ):
        input_scope.fetch(symbol, model_source.name, end_index)
        result = input_scope.fetch(symbol, model_source.name, end_index)
        df = data_source_df[data_source_df["symbol"] == symbol]
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == set(model_source.indicators)
        assert (
            result.shape[0] == df.shape[0] if end_index is None else end_index
        )

    @pytest.mark.parametrize(
        "sym, name, expected_msg",
        [
            ("FOO", MODEL_NAME, "Symbol not found: FOO"),
            ("SPY", "foo", "Model 'foo' not found."),
        ],
    )
    def test_fetch_when_not_found_then_error(
        self, input_scope, sym, name, expected_msg
    ):
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_scope.fetch(sym, name)


class TestPredictionScope:
    def test_fetch(
        self,
        pred_scope,
        preds,
        trained_model,
        symbol,
        data_source_df,
        end_index,
    ):
        values = pred_scope.fetch(symbol, trained_model.name, end_index)
        assert isinstance(values, np.ndarray)
        expected = (
            preds[symbol] if end_index is None else preds[symbol][:end_index]
        )
        assert np.array_equal(values, expected, equal_nan=True)
        df = data_source_df[data_source_df["symbol"] == symbol]
        assert len(values) == df.shape[0] if end_index is None else end_index

    def test_fetch_when_cached(
        self,
        pred_scope,
        preds,
        trained_model,
        symbol,
        data_source_df,
        end_index,
    ):
        pred_scope.fetch(symbol, trained_model.name, end_index)
        values = pred_scope.fetch(symbol, trained_model.name, end_index)
        assert isinstance(values, np.ndarray)
        expected = (
            preds[symbol] if end_index is None else preds[symbol][:end_index]
        )
        assert np.array_equal(values, expected, equal_nan=True)
        df = data_source_df[data_source_df["symbol"] == symbol]
        assert len(values) == df.shape[0] if end_index is None else end_index

    @pytest.mark.parametrize(
        "sym, name, expected_msg",
        [
            ("FOO", MODEL_NAME, "Symbol not found: FOO"),
            ("SPY", "foo", "Model 'foo' not found."),
        ],
    )
    def test_fetch_when_not_found_then_error(
        self, pred_scope, sym, name, expected_msg
    ):
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            pred_scope.fetch(sym, name)

    def test_fetch_when_predict_not_defined_then_error(self, input_scope):
        model = TrainedModel(
            name=MODEL_NAME, instance={}, predict_fn=None, input_cols=None
        )
        pred_scope = PredictionScope(
            models={ModelSymbol(MODEL_NAME, "SPY"): model},
            input_scope=input_scope,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Model instance trained for {MODEL_NAME!r} does not define a "
                "predict function. Please pass a predict_fn to "
                "pybroker.model()."
            ),
        ):
            pred_scope.fetch("SPY", MODEL_NAME)

    def test_fetch_when_input_data_empty_then_error(self, col_scope):
        model_name = "no_input_data"
        ind_scope = IndicatorScope({}, [])
        pybroker.model(model_name, lambda sym, train, test: {})
        model = TrainedModel(
            name=model_name, instance={}, predict_fn=None, input_cols=None
        )
        models = {ModelSymbol(model_name, "SPY"): model}
        input_scope = ModelInputScope(col_scope, ind_scope, models)
        pred_scope = PredictionScope(models, input_scope)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"No input data found for model {model_name!r}. Consider "
                "passing input_data_fn to pybroker#model() if custom columns "
                "were registered."
            ),
        ):
            pred_scope.fetch("SPY", model_name)


class TestPriceScope:
    @pytest.mark.parametrize(
        "price, round_fill_price, expected_price",
        [
            (50, True, 50),
            (111.1, True, Decimal("111.1")),
            (np.float32(99.98), True, Decimal("99.98")),
            (lambda _symbol, _bar_data: 60, True, 60),
            (PriceType.OPEN, True, 200),
            (PriceType.HIGH, True, 400),
            (PriceType.LOW, True, 100),
            (PriceType.CLOSE, True, 300),
            (PriceType.MIDDLE, True, round((100 + (400 - 100) / 2.0), 2)),
            (PriceType.MIDDLE, False, (100 + (400 - 100) / 2.0)),
            (PriceType.AVERAGE, True, round((200 + 100 + 400 + 300) / 4.0, 2)),
        ],
    )
    def test_fetch(self, price, round_fill_price, expected_price):
        df = pd.DataFrame(
            {
                "date": [
                    np.datetime64("2020-02-03"),
                    np.datetime64("2020-02-04"),
                    np.datetime64("2020-02-05"),
                ],
                "symbol": ["SPY"] * 3,
                "open": [100, 200, 300],
                "high": [500, 400, 500],
                "low": [200, 100, 200],
                "close": [250, 300, 400],
            }
        )
        col_scope = ColumnScope(df.set_index(["symbol", "date"]))
        price_scope = PriceScope(col_scope, {"SPY": 2}, round_fill_price)
        assert price_scope.fetch("SPY", price) == expected_price


class TestPendingOrderScope:
    def test_remove(self, pending_orders, pending_order_scope):
        assert pending_order_scope.remove(pending_orders[0].id)
        orders = tuple(pending_order_scope.orders())
        assert len(orders) == 1
        assert orders[0] == pending_orders[1]
        assert not pending_order_scope.contains(1)
        assert pending_order_scope.contains(2)

    def test_remove_all(self, pending_order_scope):
        pending_order_scope.remove_all()
        assert not tuple(pending_order_scope.orders())
        assert not pending_order_scope.contains(1)
        assert not pending_order_scope.contains(2)

    def test_remove_all_when_symbol(self, pending_orders, pending_order_scope):
        pending_order_scope.remove_all("AAPL")
        orders = tuple(pending_order_scope.orders())
        assert len(orders) == 1
        assert orders[0] == pending_orders[0]
        assert not pending_order_scope.contains(2)
        assert pending_order_scope.contains(1)

    def test_contains(self, pending_order_scope):
        assert pending_order_scope.contains(1)
        assert not pending_order_scope.contains(3)

    def test_orders(self, pending_orders, pending_order_scope):
        assert tuple(pending_order_scope.orders()) == pending_orders
        assert tuple(pending_order_scope.orders("SPY")) == tuple(
            [pending_orders[0]]
        )
        assert not tuple(pending_order_scope.orders("FOO"))


def test_get_signals(
    symbols,
    scope,
    col_scope,
    ind_scope,
    pred_scope,
    data_source_df,
    ind_data,
    preds,
):
    dfs = get_signals(symbols, col_scope, ind_scope, pred_scope)
    assert set(dfs.keys()) == set(symbols)
    for sym in symbols:
        for col in scope.all_data_cols:
            if col not in data_source_df.columns:
                continue
            assert np.array_equal(
                dfs[sym][col].values,
                data_source_df[data_source_df["symbol"] == sym][col].values,
            )
        assert np.array_equal(
            dfs[sym][f"{MODEL_NAME}_pred"].values, preds[sym], equal_nan=True
        )
    for ind_name, sym in ind_data:
        assert np.array_equal(
            dfs[sym][ind_name].values,
            ind_data[IndicatorSymbol(ind_name, sym)].values,
            equal_nan=True,
        )
