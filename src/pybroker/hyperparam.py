"""Hyperparameter declaration and resolution for strategy optimization.

Hyperparams declare tunable values for indicators and executions. Each
hyperparam is registered globally by name via :func:`hyperparam` and
resolved to a concrete int or float at backtest or optimization time.

Pass hyperparams as keyword arguments to
:func:`pybroker.indicator.indicator`, or list them on
:meth:`pybroker.strategy.Strategy.add_execution` to read them inside an
execution with ``ctx.hyperparam(name)``.
"""

from __future__ import annotations

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Union

from pybroker.scope import StaticScope


@dataclass(frozen=True)
class Hyperparam:
    """Declares a named hyperparameter with bounds and step size.

    Created with :func:`hyperparam` and registered globally by ``name``.

    Attributes:
        name: Unique identifier used in indicator kwargs, execution
            hyperparam lists, and optimization results.
        default: Value for backtests and the baseline during optimization.
            Should lie within ``[low, high]``.
        low: Minimum candidate value searched during optimize (inclusive).
        high: Maximum candidate value searched during optimize (inclusive).
            Candidate values are ``low``, ``low + step``, ... up to the
            largest value not exceeding ``high``.
        step: Spacing between candidate values. Must be positive. Integer
            hyperparams use integer steps; float hyperparams use float
            steps with values rounded to match Optuna stepped suggestions.

    Examples:
        Indicator period from 5 to 50 in steps of 5::

            period = hyperparam("period", default=14, low=5, high=50, step=5)
    """

    name: str
    default: Union[int, float]
    low: Union[int, float]
    high: Union[int, float]
    step: Union[int, float]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("default", self.default),
            ("low", self.low),
            ("high", self.high),
            ("step", self.step),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Hyperparam {self.name!r}: {field_name} must be int or "
                    f"float, got {type(value).__name__}."
                )
        value_types = {
            type(self.default),
            type(self.low),
            type(self.high),
            type(self.step),
        }
        if value_types != {int} and value_types != {float}:
            raise TypeError(
                f"Hyperparam {self.name!r}: default, low, high, and step must "
                "all be int or all be float."
            )
        if self.step <= 0:
            raise ValueError(
                f"Hyperparam {self.name!r}: step must be positive."
            )
        if self.low > self.high:
            raise ValueError(
                f"Hyperparam {self.name!r}: low cannot exceed high."
            )

    def _is_float(self) -> bool:
        return isinstance(self.default, float)

    def _ndigits(self) -> int:
        step = float(self.step)
        if step >= 1:
            return max(0, -int(math.floor(math.log10(step))))
        return max(0, -int(math.floor(math.log10(step))))

    def _within_high(self, val: Union[int, float]) -> bool:
        if self._is_float():
            ndigits = self._ndigits()
            return round(float(val), ndigits) <= round(
                float(self.high), ndigits
            )
        return int(val) <= int(self.high)

    def _lattice_count(self) -> int:
        if self.low == self.high:
            raise TypeError(
                f"Hyperparam {self.name!r} is fixed; lattice is undefined."
            )
        if self._is_float():
            count = 0
            ndigits = self._ndigits()
            i = 0
            while True:
                val = round(float(self.low) + i * float(self.step), ndigits)
                if not self._within_high(val):
                    break
                count += 1
                i += 1
            return count
        low = int(self.low)
        high = int(self.high)
        step = int(self.step)
        return (high - low) // step + 1

    def _lattice_values(self) -> Iterator[Union[int, float]]:
        if self.low == self.high:
            raise TypeError(
                f"Hyperparam {self.name!r} is fixed; lattice is undefined."
            )
        if self._lattice_count() == 0:
            raise ValueError(
                f"Hyperparam {self.name!r}: empty lattice for "
                f"low={self.low}, high={self.high}, step={self.step}."
            )
        if self._is_float():
            ndigits = self._ndigits()
            i = 0
            while True:
                val = round(float(self.low) + i * float(self.step), ndigits)
                if not self._within_high(val):
                    break
                yield val
                i += 1
        else:
            low = int(self.low)
            high = int(self.high)
            step = int(self.step)
            val = low
            while val <= high:
                yield val
                val += step

    def __iter__(self) -> Iterator[Union[int, float]]:
        return self._lattice_values()

    def __len__(self) -> int:
        return self._lattice_count()


def hyperparam(
    name: str,
    *,
    default: Union[int, float],
    low: Union[int, float],
    high: Union[int, float],
    step: Union[int, float],
) -> Hyperparam:
    """Creates and registers a :class:`Hyperparam`.

    Args:
        name: Unique identifier for the hyperparam. Referenced in indicator
            kwargs, ``add_execution(..., hyperparams=[...])``, and
            ``ctx.hyperparam(name)``.
        default: Value used for backtests.
        low: Minimum candidate value searched during optimize (inclusive).
        high: Maximum candidate value searched during optimize (inclusive).
        step: Spacing between candidate values. Must be positive.

    Returns:
        The registered :class:`Hyperparam` instance.
    """
    hp = Hyperparam(name=name, default=default, low=low, high=high, step=step)
    StaticScope.instance().set_hyperparam(hp)
    return hp


def _is_hyperparam(value: Any) -> bool:
    return isinstance(value, Hyperparam)


def _find_hyperparam_names(mapping: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        value.name for value in mapping.values() if _is_hyperparam(value)
    )


def _resolve_hyperparams(
    mapping: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    """Replaces :class:`Hyperparam` values in ``mapping`` with run values.

    Args:
        mapping: Keyword arguments that may contain :class:`Hyperparam`
            instances (for example, indicator ``_kwargs``).
        params: Dict of ``name -> value`` for the current run.

    Returns:
        A new dict with hyperparams replaced by their resolved values.
    """
    resolved: dict[str, Any] = {}
    for key, value in mapping.items():
        if _is_hyperparam(value):
            if value.name not in params:
                raise KeyError(
                    f"Hyperparam {value.name!r} is not in the run hyperparams "
                    "dict."
                )
            resolved[key] = params[value.name]
        else:
            resolved[key] = value
    return resolved


def _hyperparam_specs_from_kwargs(
    mapping: Mapping[str, Any],
) -> dict[str, Hyperparam]:
    return {
        value.name: value
        for value in mapping.values()
        if _is_hyperparam(value)
    }


@dataclass(frozen=True)
class SearchSpace:
    """Searchable hyperparameters collected from a strategy.

    Only includes hyperparams with ``low < high`` that are passed to Optuna
    during :meth:`pybroker.strategy.Strategy.optimize`.

    Attributes:
        hyperparams: Names of hyperparams searched during optimize.
        specs: Mapping of hyperparam name to :class:`Hyperparam` spec.
    """

    hyperparams: frozenset[str]
    specs: Mapping[str, Hyperparam]

    def grid_size(self) -> int:
        """Total number of grid combinations."""
        size = 1
        for name in self.hyperparams:
            size *= len(self.specs[name])
        return size


def build_run_hyperparams(
    specs: Mapping[str, Hyperparam],
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Builds the hyperparam dict for a single backtest or trial run.

    Args:
        specs: All hyperparams reachable from the strategy.
        overrides: Trial or user-supplied values to merge over defaults.

    Returns:
        Dict of ``name -> value`` for every hyperparam in ``specs``.
    """
    result: dict[str, Any] = {name: specs[name].default for name in specs}
    if overrides:
        for name, value in overrides.items():
            if name not in specs:
                raise ValueError(
                    f"Unknown hyperparam override {name!r}. "
                    f"Declared: {sorted(specs)}."
                )
            result[name] = value
    return result
