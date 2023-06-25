"""Unit tests for log.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import logging
from .fixtures import *  # noqa: F401
from pybroker.log import Logger


class TestLogger:
    def test_enable_and_disable(scope, capsys, caplog):
        caplog.set_level(logging.DEBUG)
        logger = Logger(scope)
        logger.disable()
        logger.indicator_data_start([])
        logger.info_indicator_data_start([])
        logger.debug_compute_indicators(is_parallel=False)
        logger.loaded_indicator_data()
        logger.warn_bootstrap_sample_size(10, 100)
        assert capsys.readouterr() == ("", "")
        assert not caplog.record_tuples
        logger.enable()
        logger.indicator_data_start([])
        logger.info_indicator_data_start([])
        logger.debug_compute_indicators(is_parallel=False)
        logger.loaded_indicator_data()
        logger.warn_bootstrap_sample_size(10, 100)
        captured = capsys.readouterr()
        assert captured.out
        assert captured.err == ""
        assert len(caplog.record_tuples) == 3

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
