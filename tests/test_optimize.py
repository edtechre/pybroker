"""Tests for optimize module."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import pybroker
from pybroker.config import StrategyConfig
from pybroker.optimize import (
    collect_hyperparams,
    collect_search_space,
    hyperparam,
)
from pybroker.indicator import indicator
from pybroker.model import ModelLoader, model
from pybroker.scope import StaticScope
from pybroker.slippage import (
    VolatilitySlippageModel,
    VolumeSlippageModel,
)
from pybroker.strategy import Strategy
from pybroker.interval import compress_symbol_df
from pybroker.vect import highv
from .fixtures import *  # noqa: F401,F403

START_DATE = "2020-01-02"
END_DATE = "2021-12-31"


@pytest.fixture(autouse=True)
def clear_hyperparams():
    scope = StaticScope.instance()
    scope._hyperparams.clear()
    yield
    scope._hyperparams.clear()


def _make_strategy(data_source_df):
    return Strategy(
        data_source_df,
        START_DATE,
        END_DATE,
        StrategyConfig(initial_cash=100_000),
    )


def test_collect_search_space_excludes_fixed(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    thresh = hyperparam("thresh", default=1.0, low=1.0, high=1.0, step=1.0)
    hhv = indicator(
        "hhv_lb",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)

    def exec_fn(ctx):
        _ = ctx.hyperparam("thresh")

    strategy.add_execution(
        exec_fn, "AAPL", indicators=[hhv], hyperparams=[thresh]
    )
    all_specs = collect_hyperparams(strategy)
    space = collect_search_space(strategy)
    assert set(all_specs) == {"lookback", "thresh"}
    assert space.hyperparams == frozenset({"lookback"})


def test_collect_search_space_from_indicator(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=15, step=5)
    hhv = indicator(
        "hhv_lb",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)

    def exec_fn(ctx):
        _ = ctx.indicator(hhv.name)

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv])
    space = collect_search_space(strategy)
    assert space.hyperparams == frozenset({"lookback"})
    assert space.grid_size() == 3


def test_ctx_hyperparam_gate(data_source_df):
    thresh = hyperparam("thresh", default=1.0, low=1.0, high=1.0, step=1.0)
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_lb",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)
    seen = {}

    def exec_fn(ctx):
        seen["thresh"] = ctx.hyperparam("thresh")
        with pytest.raises(ValueError, match="not attached"):
            ctx.hyperparam("lookback")

    strategy.add_execution(
        exec_fn, "AAPL", indicators=[hhv], hyperparams=[thresh]
    )
    strategy.backtest(params={"lookback": 10})
    assert seen["thresh"] == 1.0


def test_optimize_grid_smoke(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_opt",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)

    def exec_fn(ctx):
        vals = ctx.indicator(hhv.name)
        if len(vals) > 0 and vals[-1] > 0:
            if not ctx.long_pos():
                ctx.buy_shares = 10

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv])
    opt = strategy.optimize(
        lambda r: r.metrics.sharpe if r.metrics.sharpe is not None else 0.0,
        sampler="grid",
        train_size=0.5,
        disable_parallel_indicators=True,
    )
    assert opt.best_params["lookback"] in (5, 10)
    assert opt.result.bootstrap is not None


def test_optimize_result_to_json(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_json",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)

    def exec_fn(ctx):
        vals = ctx.indicator(hhv.name)
        if len(vals) > 0 and vals[-1] > 0:
            if not ctx.long_pos():
                ctx.buy_shares = 10

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv])
    opt = strategy.optimize(
        lambda r: r.metrics.sharpe if r.metrics.sharpe is not None else 0.0,
        sampler="grid",
        train_size=0.5,
        disable_parallel_indicators=True,
    )
    payload = opt.to_json()
    assert isinstance(payload["study"], dict)
    assert "n_trials" in payload["study"]
    assert "best_params" in payload["study"]
    assert set(payload["result"].keys()) == {
        "start_date",
        "end_date",
        "metrics",
        "trades",
        "orders",
        "bootstrap",
    }
    json.dumps(payload, allow_nan=False)
    opt.to_json_str()


def test_optimize_indicator_integration(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    periods_seen: set[int] = set()

    def tracking_hhv(data, period):
        periods_seen.add(period)
        return highv(data.high, period)

    hhv = indicator("hhv_int", tracking_hhv, period=lookback)
    strategy = _make_strategy(data_source_df)

    def exec_fn(ctx):
        vals = ctx.indicator(hhv.name)
        if len(vals) > 0 and vals[-1] > 0:
            if not ctx.long_pos():
                ctx.buy_shares = 10

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv])
    opt = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        train_size=0.5,
        disable_parallel_indicators=True,
    )
    trial_lookbacks = {t.params["lookback"] for t in opt.study.trials}

    assert len(opt.study.trials) == 2
    assert trial_lookbacks == {5, 10}
    assert periods_seen == {5, 10}
    assert opt.best_params["lookback"] in (5, 10)
    best_trial = max(opt.study.trials, key=lambda t: t.value)
    assert opt.best_params["lookback"] == best_trial.params["lookback"]
    assert opt.study.best_value == best_trial.value
    assert np.isfinite(opt.result.metrics.total_pnl)


def test_optimize_logs_trial_count(capsys, data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_log",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(None, "AAPL", indicators=[hhv])
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        train_size=0.5,
        disable_parallel_indicators=True,
    )
    assert "Optimizing: 2 trials (grid)" in capsys.readouterr().out


def test_grid_explosion_guard(data_source_df):
    hp = hyperparam("p", default=1, low=0, high=2000, step=1)
    ind = indicator("i", lambda data, p: data.close * 0 + p, p=hp)
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(None, "AAPL", indicators=[ind])
    study = MagicMock()
    study.best_params = {"p": 1}
    study.best_value = 1.0
    with pytest.warns(UserWarning, match="Grid size"):
        with patch("optuna.create_study", return_value=study):
            with patch.object(study, "optimize"):
                strategy.optimize(
                    lambda r: r.metrics.total_pnl,
                    n_trials=None,
                    train_size=0.5,
                    disable_parallel_indicators=True,
                )


def test_optimize_rejects_models(data_source_df):
    m = model(
        MODEL_NAME,
        lambda sym, *_: FakeModel(
            sym,
            np.full(
                data_source_df[data_source_df["symbol"] == sym].shape[0], 100
            ),
        ),
        pretrained=False,
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(None, "AAPL", models=[m])
    with pytest.raises(ValueError, match="trainable model sources"):
        strategy.optimize(lambda r: r.metrics.total_pnl)


def test_optimize_model_loader_integration(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_model_loader",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    n = data_source_df[data_source_df["symbol"] == "AAPL"].shape[0]
    load_calls: list[tuple[str, object, object]] = []

    def load_fn(sym, train_start_date, train_end_date):
        load_calls.append((sym, train_start_date, train_end_date))
        return FakeModel(sym, np.full(n, 1.0))

    m = model(
        "loader_opt",
        load_fn,
        indicators=[hhv],
        pretrained=True,
    )
    assert isinstance(
        StaticScope.instance().get_model_source(m.name), ModelLoader
    )

    strategy = _make_strategy(data_source_df)
    model_ids: list[int] = []
    pred_lens: list[int] = []

    def exec_fn(ctx):
        model_ids.append(id(ctx.model(m.name)))
        pred_lens.append(len(ctx.preds(m.name)))
        _ = ctx.indicator(hhv.name)

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv], models=[m])
    opt = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        train_size=0.5,
        disable_parallel_indicators=True,
    )

    assert len(opt.study.trials) == 2
    assert {t.params["lookback"] for t in opt.study.trials} == {5, 10}
    assert len(load_calls) == 1
    assert load_calls[0][0] == "AAPL"
    assert len(set(model_ids)) == 1
    assert all(length > 0 for length in pred_lens)
    assert np.isfinite(opt.result.metrics.total_pnl)


def test_optimize_rejects_trainable_with_pretrained(data_source_df):
    n_aapl = data_source_df[data_source_df["symbol"] == "AAPL"].shape[0]
    n_msft = data_source_df[data_source_df["symbol"] == "MSFT"].shape[0]
    pretrained = model(
        "pretrained_model",
        lambda sym, *_: FakeModel(sym, np.full(n_aapl, 1.0)),
        pretrained=True,
    )
    trainable = model(
        MODEL_NAME,
        lambda sym, train, test: FakeModel(sym, np.full(n_msft, 1.0)),
        pretrained=False,
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(None, "AAPL", models=[pretrained])
    strategy.add_execution(None, "MSFT", models=[trainable])
    with pytest.raises(ValueError, match="trainable model sources"):
        strategy.optimize(lambda r: r.metrics.total_pnl)


def test_indicator_resolves_params(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_p", lambda data, period: highv(data.high, period), period=lookback
    )
    strategy = _make_strategy(data_source_df)
    lengths = []

    def exec_fn(ctx):
        lengths.append(len(ctx.indicator(hhv.name)))

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv])
    strategy.backtest(params={"lookback": 5}, disable_parallel_indicators=True)
    assert lengths


def test_make_objective_exports():
    assert callable(pybroker.make_objective)


def test_optimize_indicator_memo_max_validation(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_lb",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)

    def exec_fn(ctx):
        _ = ctx.indicator(hhv.name)

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv])
    with pytest.raises(ValueError, match="indicator_memo_max"):
        strategy.optimize(
            lambda r: r.metrics.total_pnl,
            indicator_memo_max=-1,
            n_trials=1,
            disable_parallel_indicators=True,
        )


def test_optimize_trial_interval_alignment_matches_walkforward(
    data_source_df,
):
    # Trials run over the train window, so their interval data must be
    # realigned to it -- otherwise `completed` still indexes from the very
    # first bar of history and every window after the first is shifted.
    hyperparam("lookback", default=10, low=5, high=10, step=5)
    pretrained = model(
        "tf_align",
        lambda sym, train_start_date, train_end_date: FakeModel(
            sym, np.zeros(1)
        ),
        pretrained=True,
        predict_fn=lambda m, d: np.zeros(len(d)),
    )
    seen: list[tuple[np.datetime64, int]] = []

    def exec_fn(ctx):
        seen.append((np.datetime64(ctx.dt), len(ctx.interval("weekly").close)))

    strategy = _make_strategy(data_source_df)
    strategy.add_execution(
        exec_fn,
        "AAPL",
        models=[pretrained],
        intervals=["weekly"],
    )
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        train_size=0.5,
        disable_parallel_indicators=True,
        timeframe="1d",
    )
    assert seen

    sym_df = data_source_df[data_source_df["symbol"] == "AAPL"]
    sym_df = sym_df[
        (sym_df["date"] >= pd.Timestamp(START_DATE))
        & (sym_df["date"] <= pd.Timestamp(END_DATE))
    ].sort_values("date")
    reference = compress_symbol_df(
        sym_df.drop(columns=["symbol"]), "weekly", frozenset(), 86400.0
    )
    base_dates = reference.base_dates
    for date, count in seen:
        idx = int(np.searchsorted(base_dates, date))
        assert base_dates[idx] == date
        assert count == int(reference.completed[idx]) + 1


def test_optimize_when_slippage_indicator_missing_then_error(data_source_df):
    # optimize() used to skip slippage validation, so a missing indicator
    # surfaced as a failure deep inside an Optuna trial.
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_slip",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)

    def exec_fn(ctx):
        if not ctx.long_pos():
            ctx.buy_shares = 10

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv])
    strategy.set_slippage_model(
        VolatilitySlippageModel(atr_indicator="atr_not_attached")
    )
    with pytest.raises(ValueError, match="must be attached to an execution"):
        strategy.optimize(
            lambda r: 0.0,
            sampler="grid",
            train_size=0.5,
            disable_parallel_indicators=True,
        )


def test_optimize_when_volume_column_missing_then_error(data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_vol_slip",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df.drop(columns=["volume"]))

    def exec_fn(ctx):
        if not ctx.long_pos():
            ctx.buy_shares = 10

    strategy.add_execution(exec_fn, "AAPL", indicators=[hhv])
    strategy.set_slippage_model(VolumeSlippageModel())
    with pytest.raises(ValueError, match="requires a 'volume' data column"):
        strategy.optimize(
            lambda r: 0.0,
            sampler="grid",
            train_size=0.5,
            disable_parallel_indicators=True,
        )


def test_optimize_walkforward_resolves_selector_once_per_window(
    data_source_df,
):
    # The replay that produces the reported result used to re-run the
    # selector, so a stateful one described a different universe than the
    # study had tuned.
    hyperparam("lookback", default=10, low=5, high=10, step=5)
    calls: list[int] = []

    def counting_selector(_selection_df):
        calls.append(len(calls))
        return ["AAPL"]

    def exec_fn(ctx):
        if not ctx.long_pos():
            ctx.buy_shares = 10

    strategy = _make_strategy(data_source_df)
    strategy.add_execution(exec_fn, counting_selector)
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        train_size=0.5,
        disable_parallel_indicators=True,
    )
    assert len(calls) == 2


def test_optimize_with_selector_and_no_train_window_raises(data_source_df):
    hyperparam("lookback", default=10, low=5, high=10, step=5)
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(lambda _ctx: None, lambda _d: ["AAPL"])
    with pytest.raises(ValueError, match="requires a training window"):
        strategy.optimize(
            lambda r: 0.0,
            sampler="grid",
            train_size=0,
            disable_parallel_indicators=True,
        )
