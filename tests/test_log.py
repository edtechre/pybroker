"""Unit tests for log.py module."""

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

from .fixtures import *  # noqa: F401
from pybroker.log import Logger


class TestLogger:
    def test_enable_and_disable(scope, capsys):
        logger = Logger(scope)
        logger.disable()
        logger.indicator_data_start([])
        assert capsys.readouterr() == ("", "")
        logger.enable()
        logger.indicator_data_start([])
        captured = capsys.readouterr()
        assert captured.out
        assert captured.err == ""

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
