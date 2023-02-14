"""Contains vectorized utility functions."""

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

from numba import njit
from numpy.typing import NDArray
import numpy as np


@njit
def _verify_input(array: NDArray[np.float_], n: int):
    if n <= 0:
        raise ValueError("n needs to be >= 1.")
    if n > len(array):
        raise ValueError("n is greater than array length.")


@njit
def lowv(array: NDArray[np.float_], n: int) -> NDArray[np.float_]:
    """Calculates the lowest values for every ``n`` period in ``array``.

    Args:
        array: :class:`numpy.ndarray` of data.
        n: Length of period.

    Returns:
        :class:`numpy.ndarray` of the lowest values for every ``n`` period in
        ``array``.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.min(array[i - n : i])
    return out


@njit
def highv(array: NDArray[np.float_], n: int) -> NDArray[np.float_]:
    """Calculates the highest values for every ``n`` period in ``array``.

    Args:
        array: :class:`numpy.ndarray` of data.
        n: Length of period.

    Returns:
        :class:`numpy.ndarray` of the highest values for every ``n`` period in
        ``array``.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.max(array[i - n : i])
    return out


@njit
def sumv(array: NDArray[np.float_], n: int) -> NDArray[np.float_]:
    """Calculates the sums for every ``n`` period in ``array``.

    Args:
        array: :class:`numpy.ndarray` of data.
        n: Length of period.

    Returns:
        :class:`numpy.ndarray` of the sums for every ``n`` period in ``array``.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.sum(array[i - n : i])
    return out


@njit
def returnv(array: NDArray[np.float_], n: int = 1) -> NDArray[np.float_]:
    """Calculates returns.

    Args:
        n: Return period. Defaults to 1.

    Returns:
        :class:`numpy.ndarray` of returns.
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len):
        out[i] = (array[i] - array[i - n]) / array[i - n]
    return out


@njit
def cross(a: NDArray[np.float_], b: NDArray[np.float_]) -> NDArray[np.bool_]:
    """Checks for crossover of ``a`` above ``b``.

    Args:
        a: :class:`numpy.ndarray` of data.
        b: :class:`numpy.ndarray` of data.

    Returns:
        :class:`numpy.ndarray` containing values of ``1`` when ``a`` crosses
        above ``b``, otherwise values of ``0``.
    """
    if not len(a):
        raise ValueError("a cannot be empty.")
    if not len(b):
        raise ValueError("b cannot be empty.")
    if len(a) != len(b):
        raise ValueError("len(a) != len(b)")
    if len(a) < 2:
        raise ValueError("a and b must have length >= 2.")
    crossed = np.where(a > b, 1, 0)
    return (sumv(crossed > 0, 2) == 1) * crossed
