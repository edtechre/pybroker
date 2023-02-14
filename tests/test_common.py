"""Unit tests for common.py module."""

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

from datetime import datetime
from decimal import Decimal

from joblib import Parallel
from pybroker.common import (
    BarData,
    default_parallel,
    parse_timeframe,
    quantize,
    to_datetime,
    to_decimal,
    to_seconds,
    verify_data_source_columns,
)
import numpy as np
import pandas as pd
import pytest
import re


def test_bar_data_get_custom_data():
    date = np.full(10, np.datetime64("2022-02-02"))
    open_ = np.full(10, 1)
    high = np.full(10, 2)
    low = np.full(10, 3)
    close = np.full(10, 4)
    foo = np.full(10, 5)
    bar = np.full(10, 6)
    custom_data = {"foo": foo, "bar": bar}
    bar_data = BarData(
        date=date,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=None,
        vwap=None,
        **custom_data,
    )
    assert bar_data.foo is foo
    assert bar_data.bar is bar


def test_bar_data_get_custom_data_when_no_attr_then_error():
    date = np.full(10, np.datetime64("2022-02-02"))
    open_ = np.full(10, 1)
    high = np.full(10, 2)
    low = np.full(10, 3)
    close = np.full(10, 4)
    bar_data = BarData(
        date=date,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=None,
        vwap=None,
    )
    with pytest.raises(
        AttributeError, match=re.escape("Attribute 'foo' not found.")
    ):
        bar_data.foo


@pytest.mark.parametrize(
    "tf, expected",
    [
        ("1day 2h 3min", [(1, "day"), (2, "hour"), (3, "min")]),
        ("10week", [(10, "week")]),
        ("3d 20m", [(3, "day"), (20, "min")]),
        ("30s", [(30, "sec")]),
    ],
)
def test_parse_timeframe_success(tf, expected):
    assert parse_timeframe(tf) == expected


@pytest.mark.parametrize(
    "tf",
    [
        "10foo",
        "20days",
        "10d 5 m",
        "1w 2w 3w 5min",
        "dd ff cc",
        "w d m",
        "1d5m",
        "1d 5mm",
        "",
    ],
)
def test_parse_timeframe_invalid(tf):
    with pytest.raises(
        ValueError, match=re.escape("Invalid timeframe format.")
    ):
        parse_timeframe(tf)


@pytest.mark.parametrize(
    "tf, expected",
    [
        ("1day 2h 3min", 24 * 60 * 60 + 2 * 60 * 60 + 3 * 60),
        ("10week", 10 * 7 * 24 * 60 * 60),
        ("3d 20m", 3 * 24 * 60 * 60 + 20 * 60),
        ("30s", 30),
        (None, 0),
    ],
)
def test_to_seconds(tf, expected):
    assert to_seconds(tf) == expected


@pytest.mark.parametrize(
    "date, expected",
    [
        ("2022-02-02", datetime.strptime("2022-02-02", "%Y-%m-%d")),
        (
            datetime.strptime("2021-05-05", "%Y-%m-%d"),
            datetime.strptime("2021-05-05", "%Y-%m-%d"),
        ),
        (
            np.datetime64("2019-03-03"),
            datetime.strptime("2019-03-03", "%Y-%m-%d"),
        ),
        (
            pd.Timestamp("2020-03-03"),
            datetime.strptime("2020-03-03", "%Y-%m-%d"),
        ),
    ],
)
def test_to_datetime(date, expected):
    dt = to_datetime(date)
    assert type(dt) == datetime
    assert dt == expected


def test_to_datetime_type_error():
    with pytest.raises(TypeError, match=r"Unsupported date type: .*"):
        to_datetime(1000)


def test_quantize():
    df = pd.DataFrame(
        [
            [Decimal("0.9999"), Decimal("1.22222")],
            [Decimal("0.1"), Decimal("0.22")],
            [Decimal("0.33"), Decimal("0.2222")],
            [Decimal(1), Decimal("0.1")],
        ],
        columns=["a", "b"],
    )
    df["a"] = quantize(df, "a")
    assert (df["a"].values == [1.00, 0.1, 0.33, 1]).all()


def test_quantize_when_column_not_found_then_error():
    df = pd.DataFrame(
        [
            [Decimal("0.9999"), Decimal("1.22222")],
            [Decimal("0.1"), Decimal("0.22")],
            [Decimal("0.33"), Decimal("0.2222")],
            [Decimal(1), Decimal("0.1")],
        ],
        columns=["a", "b"],
    )
    with pytest.raises(
        ValueError, match=re.escape("Column 'c' not found in DataFrame.")
    ):
        quantize(df, "c")


@pytest.mark.parametrize(
    "value, expected",
    [
        (1.22222, Decimal("1.22222")),
        (1, Decimal(1)),
        (30.33, Decimal("30.33")),
        (Decimal("10.1"), Decimal("10.1")),
    ],
)
def test_to_decimal(value, expected):
    assert to_decimal(value) == expected


def test_verify_data_source_columns():
    df = pd.DataFrame(
        columns=["symbol", "date", "open", "high", "low", "close"]
    )
    verify_data_source_columns(df)
    assert True


def test_verify_data_source_columns_when_missing_then_error():
    df = pd.DataFrame(columns=["symbol", "date", "open", "high", "low"])
    with pytest.raises(
        ValueError,
        match=re.escape("DataFrame is missing required columns: ['close']"),
    ):
        verify_data_source_columns(df)


def test_default_parallel():
    assert type(default_parallel()) == Parallel
