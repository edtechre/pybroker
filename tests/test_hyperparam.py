"""Tests for hyperparam module."""

import optuna
import pytest

from pybroker.hyperparam import (
    Hyperparam,
    _hyperparam_specs_from_kwargs,
    _resolve_hyperparams,
    build_run_hyperparams,
    hyperparam,
)
from pybroker.scope import StaticScope


@pytest.fixture(autouse=True)
def clear_hyperparams():
    scope = StaticScope.instance()
    scope._hyperparams.clear()
    yield
    scope._hyperparams.clear()


def test_default_only_hyperparam():
    hp = hyperparam("thresh", default=1.0, low=1.0, high=1.0, step=1.0)
    assert hp.default == 1.0
    assert hp.low == hp.high
    with pytest.raises(TypeError):
        len(hp)


def test_int_lattice():
    hp = hyperparam("lookback", default=14, low=5, high=50, step=5)
    values = list(hp)
    assert values == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    assert len(hp) == 10


def test_float_lattice_quantization():
    hp = hyperparam("alpha", default=0.1, low=0.1, high=0.3, step=0.1)
    values = list(hp)
    assert values == [0.1, 0.2, 0.3]


def test_float_lattice_includes_high():
    hp = hyperparam("x", default=0.7, low=0.5, high=1.0, step=0.1)
    values = list(hp)
    assert values[0] == 0.5
    assert values[-1] == 1.0


def test_float_lattice_matches_optuna_snapping():
    hp = hyperparam("x", default=0.7, low=0.5, high=0.9, step=0.1)

    def objective(trial):
        return trial.suggest_float("x", 0.5, 0.9, step=0.1)

    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler({"x": list(hp)})
    )
    study.optimize(objective, n_trials=len(hp))
    optuna_values = sorted(t.params["x"] for t in study.trials)
    assert optuna_values == list(hp)


def test_resolve_hyperparams():
    lookback = hyperparam("lookback", default=14, low=5, high=50, step=5)
    kwargs = {"period": lookback}
    resolved = _resolve_hyperparams(kwargs, {"lookback": 20})
    assert resolved == {"period": 20}


def test_build_run_hyperparams_from_indicator_kwargs():
    lookback = hyperparam("lookback", default=14, low=5, high=50, step=5)
    specs = _hyperparam_specs_from_kwargs({"period": lookback})
    assert build_run_hyperparams(specs) == {"lookback": 14}
    assert build_run_hyperparams(specs, {"lookback": 20}) == {"lookback": 20}


def test_rejects_bool():
    with pytest.raises(TypeError, match="must be int or float"):
        Hyperparam(name="bad", default=True, low=1, high=10, step=1)


def test_rejects_mixed_int_float():
    with pytest.raises(TypeError, match="all be int or all be float"):
        Hyperparam(name="bad", default=1, low=1.0, high=10.0, step=1.0)


def test_build_run_hyperparams_override():
    lookback = hyperparam("lookback", default=14, low=5, high=50, step=5)
    thresh = hyperparam("thresh", default=1.0, low=1.0, high=1.0, step=1.0)
    specs = {"lookback": lookback, "thresh": thresh}
    run = build_run_hyperparams(specs, {"lookback": 20})
    assert run == {"lookback": 20, "thresh": 1.0}


def test_invalid_step():
    with pytest.raises(ValueError, match="step must be positive"):
        Hyperparam(name="bad", default=1, low=1, high=10, step=0)


def test_low_exceeds_high():
    with pytest.raises(ValueError, match="low cannot exceed high"):
        Hyperparam(name="bad", default=1, low=10, high=1, step=1)
