"""Unit tests for vect.py module."""

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

from pybroker.vect import lowv, highv, sumv, cross
import numpy as np
import pytest
import re


@pytest.mark.parametrize(
    "array, n, expected",
    [
        ([3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 3, 2, 2, 2, 1, 1]),
        ([3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        ([4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 1]),
        ([1], 1, [1]),
    ],
)
def test_lowv(array, n, expected):
    assert np.array_equal(lowv(np.array(array), n), expected, equal_nan=True)


@pytest.mark.parametrize(
    "array, n, expected",
    [
        ([3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 4, 4, 5, 6, 6, 6]),
        ([3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        ([4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 4]),
        ([1], 1, [1]),
    ],
)
def test_highv(array, n, expected):
    assert np.array_equal(highv(np.array(array), n), expected, equal_nan=True)


@pytest.mark.parametrize(
    "array, n, expected",
    [
        ([3, 3, 4, 2, 5, 6, 1, 3], 3, [np.nan, np.nan, 10, 9, 11, 13, 12, 10]),
        ([3, 3, 4, 2, 5, 6, 1, 3], 1, [3, 3, 4, 2, 5, 6, 1, 3]),
        ([4, 3, 2, 1], 4, [np.nan, np.nan, np.nan, 10]),
        ([1], 1, [1]),
    ],
)
def test_sumv(array, n, expected):
    assert np.array_equal(sumv(np.array(array), n), expected, equal_nan=True)


@pytest.mark.parametrize("fnv", [lowv, highv, sumv])
@pytest.mark.parametrize(
    "array, n, expected_msg",
    [
        ([], 0, "Array is empty."),
        ([1, 2, 3], 10, "n is greater than array length."),
        ([1, 2, 3], 0, "n needs to be >= 1."),
        ([1, 2, 3], -1, "n needs to be >= 1."),
    ],
)
def test_when_n_invalid_then_error(fnv, array, n, expected_msg):
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        fnv(np.array(array), n)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
            [3, 3, 4, 2, 5, 6, 1, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [0, 0, 1, 0, 1, 0, 0, 0],
        ),
        (
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 4, 2, 5, 6, 1, 3],
            [0, 0, 0, 1, 0, 0, 1, 0],
        ),
        ([1, 1], [1, 1], [0, 0]),
    ],
)
def test_cross(a, b, expected):
    assert np.array_equal(
        cross(np.array(a), np.array(b)), expected, equal_nan=True
    )


@pytest.mark.parametrize(
    "a, b, expected_msg",
    [
        ([1, 2, 3], [3, 3, 3, 3], "len(a) != len(b)"),
        ([3, 3, 3, 3], [1, 2, 3], "len(a) != len(b)"),
        ([1, 2, 3], [], "b cannot be empty."),
        ([], [1, 2, 3], "a cannot be empty."),
        ([1], [1], "a and b must have length >= 2."),
    ],
)
def test_cross_when_invalid_input_then_error(a, b, expected_msg):
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        cross(np.array(a), np.array(b))
