"""Unit tests for common.py module."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import pytest
import re
from datetime import date, datetime, timedelta
from decimal import Decimal
from pybroker.common import (
    BarData,
    PriceType,
    StopType,
    bars_to_df,
    parse_timeframe,
    quantize,
    to_datetime,
    to_decimal,
    to_seconds,
    verify_data_source_columns,
    verify_date_range,
)


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


def test_bars_to_df():
    date = np.full(10, np.datetime64("2022-02-02"))
    open_ = np.full(10, 1.0)
    high = np.full(10, 2.0)
    low = np.full(10, 3.0)
    close = np.full(10, 4.0)
    volume = np.full(10, 5.0)
    vwap = np.full(10, 6.0)
    foo = np.full(10, 7.0)
    bar_data = BarData(
        date=date,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        foo=foo,
    )
    df = bars_to_df(bar_data)
    assert df.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "foo",
    ]
    assert len(df) == 10
    assert (df["date"].to_numpy() == date).all()
    assert (df["open"].to_numpy() == open_).all()
    assert (df["high"].to_numpy() == high).all()
    assert (df["low"].to_numpy() == low).all()
    assert (df["close"].to_numpy() == close).all()
    assert (df["volume"].to_numpy() == volume).all()
    assert (df["vwap"].to_numpy() == vwap).all()
    assert (df["foo"].to_numpy() == foo).all()


def test_bars_to_df_when_optional_cols_none():
    date = np.full(10, np.datetime64("2022-02-02"))
    open_ = np.full(10, 1.0)
    high = np.full(10, 2.0)
    low = np.full(10, 3.0)
    close = np.full(10, 4.0)
    bar_data = BarData(
        date=date,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=None,
        vwap=None,
    )
    df = bars_to_df(bar_data)
    assert df.columns.tolist() == ["date", "open", "high", "low", "close"]
    assert len(df) == 10


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
    assert isinstance(dt, datetime)
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
    df["a"] = quantize(df, "a", True)
    assert (df["a"].values == [1.00, 0.1, 0.33, 1]).all()


def test_quantize_when_round_is_false():
    df = pd.DataFrame(
        [
            [Decimal("0.9999"), Decimal("1.22222")],
            [Decimal("0.1"), Decimal("0.22")],
            [Decimal("0.33"), Decimal("0.2222")],
            [Decimal(1), Decimal("0.1")],
        ],
        columns=["a", "b"],
    )
    df["a"] = quantize(df, "a", False)
    assert (df["a"].values == [0.9999, 0.1, 0.33, 1]).all()


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
        quantize(df, "c", True)


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


def test_verify_date_range_when_invalid_then_error():
    with pytest.raises(
        ValueError,
        match=r"start_date (.*) must be on or before end_date (.*)\.",
    ):
        verify_date_range("2020-05-01", "2020-04-01")


def test_verify_data_source_columns_when_missing_then_error():
    df = pd.DataFrame(columns=["symbol", "date", "open", "high", "low"])
    with pytest.raises(
        ValueError,
        match=re.escape("DataFrame is missing required columns: ['close']"),
    ):
        verify_data_source_columns(df)


def test_json_safe_when_nat_then_null():
    """NaTType subclasses datetime, so it would otherwise be serialized by the
    datetime branch as the string "NaT" rather than as null."""
    from pybroker.common import _json_safe

    assert _json_safe(pd.NaT) is None
    assert _json_safe(np.datetime64("NaT")) is None
    assert _json_safe(pd.Timestamp("2021-01-04")) == "2021-01-04T00:00:00"


@pytest.mark.parametrize(
    "value, expected",
    [
        (Decimal("NaN"), None),
        (Decimal("Infinity"), "Infinity"),
        (Decimal("-Infinity"), "-Infinity"),
        (Decimal("1.5"), 1.5),
    ],
)
def test_json_safe_non_finite_decimal(value, expected):
    """Decimal('NaN') floats into a raw nan, which
    json.dumps(allow_nan=False) rejects -- so one non-finite Decimal made
    to_json_str() raise on an otherwise valid result. Infinities keep a
    string sentinel so they stay distinguishable from missing values."""
    import json

    from pybroker.common import _json_safe

    assert _json_safe(value) == expected
    json.dumps({"v": _json_safe(value)}, allow_nan=False)


@pytest.mark.parametrize(
    "value, expected",
    [
        (PriceType.CLOSE, "close"),
        (StopType.LOSS, "loss"),
        (date(2021, 2, 15), "2021-02-15"),
        (np.array([1.0, 2.5]), [1.0, 2.5]),
        (np.array([np.nan]), [None]),
        (np.array([[1.0, 2.0], [3.0, np.nan]]), [[1.0, 2.0], [3.0, None]]),
        (np.array(7.0), 7.0),
        (frozenset({"b", "a"}), ["a", "b"]),
        ({10, 2, 1}, [1, 2, 10]),
        (b"bytes", "bytes"),
        (np.bool_(True), True),
        (np.bool_(False), False),
        (timedelta(days=1), "P1DT0H0M0S"),
        (np.timedelta64(1, "D"), "P1DT0H0M0S"),
        (np.timedelta64("NaT"), None),
        (pd.Period("2021Q1", freq="Q"), "2021Q1"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
        (np.float64("inf"), "Infinity"),
        (np.float64("-inf"), "-Infinity"),
        ({np.int64(3): np.float64(1.5)}, {3: 1.5}),
        ({("AAA", "BBB"): 1.5}, {'["AAA", "BBB"]': 1.5}),
        ({frozenset({"b", "a"}): 1}, {'["a", "b"]': 1}),
        (np.timedelta64(3, "M"), "3 months"),
        (timedelta.max, "999999999 days, 23:59:59.999999"),
        (
            np.array(["2021-01-04"], dtype="datetime64[ns]"),
            ["2021-01-04T00:00:00"],
        ),
        (
            np.array([1], dtype="timedelta64[ns]"),
            ["P0DT0H0M0.000000001S"],
        ),
    ],
)
def test_json_safe_total(value, expected):
    """_json_safe is total: every input maps to JSON-serializable Python
    types, so json.dumps(allow_nan=False) never raises on its output."""
    import json

    from pybroker.common import _json_safe

    assert _json_safe(value) == expected
    json.dumps({"v": _json_safe(value)}, allow_nan=False)


def test_json_safe_fallback_deterministic():
    """The str() fallback must not embed memory addresses: identical
    backtests must serialize to identical JSON bytes across runs."""
    from pybroker.common import _json_safe

    class Marker:
        pass

    first = _json_safe(Marker())
    second = _json_safe(Marker())
    assert first == second
    assert "0x" not in first


def test_dataframe_records_drops_unnamed_index():
    from pybroker.common import _dataframe_records

    df = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
    records = _dataframe_records(df)
    assert records == [
        {"name": "a", "value": 1},
        {"name": "b", "value": 2},
    ]


def test_dataframe_records_keeps_named_index():
    from pybroker.common import _dataframe_records

    df = pd.DataFrame({"value": [1, 2]}, index=pd.Index([10, 20], name="id"))
    records = _dataframe_records(df)
    assert records == [
        {"id": 10, "value": 1},
        {"id": 20, "value": 2},
    ]


def test_dataframe_records_when_negative_max_rows_then_error():
    from pybroker.common import _dataframe_records

    df = pd.DataFrame({"value": [1, 2]})
    with pytest.raises(ValueError, match=re.escape("max_rows must be >= 0")):
        _dataframe_records(df, max_rows=-1)
