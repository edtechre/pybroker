"""Unit tests for log.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import datetime
import logging
from decimal import Decimal
from unittest.mock import patch

import numpy as np

from .fixtures import *  # noqa: F401
from pybroker.log import Logger
from pybroker.scope import PendingOrder


def _order_kwargs():
    return {
        "date": np.datetime64("2020-01-01"),
        "symbol": "AAPL",
        "shares": Decimal("10"),
        "fill_price": Decimal("100"),
        "limit_price": None,
    }


class TestLogger:
    def test_enable_and_disable(scope, capsys, caplog):
        caplog.set_level(logging.DEBUG)
        logger = Logger(scope)
        logger.disable()
        logger.indicator_data_start([])
        logger.info_indicator_data_start([])
        logger.debug_compute_indicators(is_parallel=False)
        logger.loaded_indicator_data()
        assert capsys.readouterr() == ("", "")
        assert not caplog.record_tuples
        logger.enable()
        logger.indicator_data_start([])
        logger.info_indicator_data_start([])
        logger.debug_compute_indicators(is_parallel=False)
        logger.loaded_indicator_data()
        captured = capsys.readouterr()
        assert captured.out
        assert captured.err == ""
        assert len(caplog.record_tuples) == 2

    def test_enable_and_disable_progress_bar(scope, capsys):
        logger = Logger(scope)
        logger.disable_progress_bar()
        logger._start_progress_bar("start", 10)
        logger._update_progress_bar(1)
        assert capsys.readouterr() == ("start\n", "")
        logger.enable_progress_bar()
        logger._start_progress_bar("start", 10)
        logger._update_progress_bar(1)
        captured = capsys.readouterr()
        assert captured.out
        assert captured.err == ""

    def test_debug_order_skips_when_debug_disabled(self, scope, caplog):
        caplog.set_level(logging.WARNING, logger="pybroker")
        logger = Logger(scope)
        with patch("pybroker.log.to_datetime") as mock_to_datetime:
            logger.debug_place_buy_order(**_order_kwargs())
        assert not caplog.record_tuples
        mock_to_datetime.assert_not_called()

    def test_debug_order_emits_when_debug_enabled(self, scope, caplog):
        caplog.set_level(logging.DEBUG, logger="pybroker")
        logger = Logger(scope)
        logger.debug_place_buy_order(**_order_kwargs())
        assert len(caplog.record_tuples) == 1
        assert caplog.record_tuples[0][0] == "pybroker"
        assert caplog.record_tuples[0][1] == logging.DEBUG
        assert "Placing buy order" in caplog.record_tuples[0][2]

    def test_debug_timeout_respects_disable(self, scope, caplog):
        caplog.set_level(logging.DEBUG, logger="pybroker")
        logger = Logger(scope)
        pending_order = PendingOrder(
            id=1,
            type="buy",
            symbol="AAPL",
            created=np.datetime64("2020-01-01"),
            exec_date=np.datetime64("2020-01-02"),
            shares=Decimal("10"),
            limit_price=Decimal("100"),
            fill_price=Decimal("100"),
            exec_bar=1,
            timeout_bars=5,
            stops=None,
        )
        logger.disable()
        with patch("pybroker.log.to_datetime") as mock_to_datetime:
            logger.debug_timeout_order(
                date=np.datetime64("2020-01-01"),
                pending_order=pending_order,
            )
        assert not caplog.record_tuples
        mock_to_datetime.assert_not_called()

    def test_info_loaded_bar_data_message(self, scope, caplog):
        caplog.set_level(logging.INFO, logger="pybroker")
        scope.data_source_cache_ns = "test-ns"
        logger = Logger(scope)
        start_date = datetime.datetime(2020, 1, 1)
        end_date = datetime.datetime(2020, 12, 31)
        logger.info_loaded_bar_data(
            symbols=["AAPL", "MSFT"],
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
        )
        assert len(caplog.record_tuples) == 1
        message = caplog.record_tuples[0][2]
        assert "namespace=test-ns" in message
        assert "2020-01-01 00:00:00 to 2020-12-31 00:00:00" in message
        assert "timeframe: 1d" in message
        assert "['AAPL', 'MSFT']" in message
