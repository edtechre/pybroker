"""Tests for optimize module."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest

import pybroker
from pybroker.config import StrategyConfig
from pybroker.optimize import (
    SearchSpace,
    _build_sampler,
    collect_hyperparams,
    collect_search_space,
    hyperparam,
)
from pybroker.indicator import indicator
from pybroker.model import ModelLoader, model
from pybroker.scope import StaticScope, symbol_array_store_from_frame
from pybroker.slippage import (
    VolumeSlippageModel,
)
from pybroker.strategy import Strategy
from pybroker.interval import compress_symbol_df, model_interval_name
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


def _add_tuned_execution(strategy, fn=None, symbols="AAPL", **kwargs):
    """Adds an execution with a searchable hyperparam attached to it.

    Declaring a hyperparam without attaching it to an execution is a genuine
    misconfiguration that ``collect_hyperparams`` warns about, so tests that
    only need *some* search space attach it rather than leaving it dangling.
    """
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    strategy.add_execution(
        fn if fn is not None else (lambda _ctx: None),
        symbols,
        hyperparams=[lookback],
        **kwargs,
    )
    return lookback


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
        parallel_indicators=False,
        calc_bootstrap=True,
    )
    assert opt.best_params["lookback"] in (5, 10)
    assert opt.result.bootstrap is not None
    # Off by default, matching Strategy.walkforward.
    assert (
        strategy.optimize(
            lambda r: 0.0,
            sampler="grid",
            train_size=0.5,
            parallel_indicators=False,
        ).result.bootstrap
        is None
    )


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
        parallel_indicators=False,
        calc_bootstrap=True,
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
        parallel_indicators=False,
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
        parallel_indicators=False,
    )
    assert "Optimizing: 2 trials (grid)" in capsys.readouterr().out


def test_optimize_quiet_trials_by_default(capsys, data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_quiet",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(None, "AAPL", indicators=[hhv])
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        train_size=0.5,
        parallel_indicators=False,
    )
    out = capsys.readouterr().out
    assert "Optimizing: 2 trials (grid)" in out
    # Only the final test window evaluation logs; the two trials are silent.
    assert out.count("Test split:") == 1
    assert not StaticScope.instance().logger._disabled


def test_optimize_verbose_logs_trials(capsys, data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_verbose",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(None, "AAPL", indicators=[hhv])
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        train_size=0.5,
        parallel_indicators=False,
        verbose=True,
    )
    out = capsys.readouterr().out
    # Two trial backtests plus the final test window evaluation.
    assert out.count("Test split:") == 3


def test_optimize_windows_quiet_trials_by_default(capsys, data_source_df):
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    hhv = indicator(
        "hhv_quiet_wf",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(None, "AAPL", indicators=[hhv])
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        train_size=0.5,
        parallel_indicators=False,
    )
    out = capsys.readouterr().out
    # Each window's trials and replays are silent; only the stitched replay
    # logs one test split per window.
    assert out.count("Test split:") == 2
    assert not StaticScope.instance().logger._disabled


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
                    parallel_indicators=False,
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
    load_calls: list[tuple[str, object, object]] = []

    def load_fn(sym, train_start_date, train_end_date):
        load_calls.append((sym, train_start_date, train_end_date))
        m = MagicMock()
        # One prediction per input row: PredictionScope.fetch now rejects
        # fixed-length predictions spanning the full history when trials
        # predict over the train window alone.
        m.predict = lambda X: np.full(len(X), 1.0)
        return m

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
        parallel_indicators=False,
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
    strategy.backtest(params={"lookback": 5}, parallel_indicators=False)
    assert lengths


def test_make_objective_exports():
    assert callable(pybroker.make_objective)


def test_optimize_trial_interval_alignment_matches_walkforward(
    data_source_df,
):
    # Trials run over the train window, so their interval data must be
    # realigned to it -- otherwise `completed` still indexes from the very
    # first bar of history and every window after the first is shifted.
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
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
        hyperparams=[lookback],
    )
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        train_size=0.5,
        parallel_indicators=False,
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


def test_load_pretrained_models_threads_lookahead(data_source_df, monkeypatch):
    # optimize() rejects trainable ModelTrainers up front
    # (_validate_optimize_models), so the lookahead threading into
    # train_models is locked by calling _load_pretrained_models directly with
    # a trainable interval model registered on the execution.
    trainable = model(
        "lookahead_thread_model",
        lambda sym, train, test: FakeModel(sym, np.zeros(len(test))),
        pretrained=False,
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(
        lambda _ctx: None,
        "AAPL",
        models=[trainable.intervals("base", "weekly")],
    )
    df = (
        data_source_df[data_source_df["symbol"] == "AAPL"]
        .sort_values("date")
        .reset_index(drop=True)
    )
    window = next(
        iter(
            strategy.walkforward_split(
                df, windows=1, lookahead=3, train_size=0.5
            )
        )
    )
    captured: dict = {}

    def capture_train_models(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(strategy, "train_models", capture_train_models)
    result = strategy._load_pretrained_models(
        df=df,
        train_rows=window.train_data,
        test_rows=window.test_data,
        window_executions=strategy._executions,
        indicator_data={},
        master_store=symbol_array_store_from_frame(df),
        interval_data=IntervalData(),
        tf_seconds=86400,
        between_time=None,
        days=None,
        lookahead=3,
    )
    assert result == {}
    assert captured["lookahead"] == 3
    model_syms = set(captured["model_syms"])
    assert ModelSymbol("lookahead_thread_model", "AAPL") in model_syms
    assert (
        ModelSymbol(
            model_interval_name("lookahead_thread_model", "weekly"), "AAPL"
        )
        in model_syms
    )


def test_load_pretrained_models_pooled_interval_groups(
    data_source_df, monkeypatch
):
    # An interval-bound pooled model must produce a pooled group for its
    # suffixed name, mirroring the walkforward path.
    pooled_model = model(
        "pooled_interval_groups",
        lambda syms, train, test: FakeModel("pooled", np.zeros(len(test))),
        pretrained=False,
        pooled=True,
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(
        lambda _ctx: None,
        "AAPL",
        models=[pooled_model.intervals("base", "weekly")],
    )
    df = (
        data_source_df[data_source_df["symbol"] == "AAPL"]
        .sort_values("date")
        .reset_index(drop=True)
    )
    window = next(
        iter(
            strategy.walkforward_split(
                df, windows=1, lookahead=1, train_size=0.5
            )
        )
    )
    captured: dict = {}

    def capture_train_models(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(strategy, "train_models", capture_train_models)
    strategy._load_pretrained_models(
        df=df,
        train_rows=window.train_data,
        test_rows=window.test_data,
        window_executions=strategy._executions,
        indicator_data={},
        master_store=symbol_array_store_from_frame(df),
        interval_data=IntervalData(),
        tf_seconds=86400,
        between_time=None,
        days=None,
        lookahead=1,
    )
    exec_id = next(iter(strategy._executions)).id
    groups = captured["pooled_model_groups"]
    assert ("pooled_interval_groups", exec_id) in groups
    assert (
        model_interval_name("pooled_interval_groups", "weekly"),
        exec_id,
    ) in groups


def test_collect_hyperparams_model_kwargs_error_names_base(data_source_df):
    # The error must be deterministic and name the registered model, not the
    # internal suffixed per-interval expansion.
    depth = hyperparam("kw_depth", default=1, low=1, high=3, step=1)
    trend = model(
        "kwargs_hp_model",
        lambda sym, train, test: FakeModel(sym, np.zeros(len(test))),
        predict_fn=lambda m, d: np.zeros(len(d)),
        depth=depth,
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(
        lambda _ctx: None, "AAPL", models=[trend.intervals("weekly")]
    )
    with pytest.raises(
        ValueError,
        match="Model 'kwargs_hp_model' has hyperparams in kwargs",
    ):
        collect_hyperparams(strategy)


def test_validate_optimize_models_names_base_only(data_source_df):
    from pybroker.optimize import _validate_optimize_models

    trend = model(
        "trainable_bound",
        lambda sym, train, test: FakeModel(sym, np.zeros(len(test))),
        predict_fn=lambda m, d: np.zeros(len(d)),
    )
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(
        lambda _ctx: None, "AAPL", models=[trend.intervals("weekly")]
    )
    with pytest.raises(ValueError, match="trainable model sources") as exc:
        _validate_optimize_models(strategy)
    assert "trainable_bound" in str(exc.value)
    assert "trainable_bound@weekly" not in str(exc.value)


def test_optimize_trials_never_see_test_bars(data_source_df):
    # Trials tune on the train window only; the test window is reserved for
    # the final replay of the winning params. The losing grid value runs
    # only inside trials, so the dates it saw pin what the trials saw.
    lookback = hyperparam("lookback", default=10, low=5, high=10, step=5)
    pretrained = model(
        "trial_phase_model",
        lambda sym, train_start_date, train_end_date: FakeModel(
            sym, np.zeros(1)
        ),
        pretrained=True,
        predict_fn=lambda m, d: np.zeros(len(d)),
    )
    seen: list[tuple[int, pd.Timestamp]] = []

    def exec_fn(ctx):
        seen.append((ctx.hyperparam("lookback"), pd.Timestamp(ctx.dt)))

    strategy = _make_strategy(data_source_df)
    strategy.add_execution(
        exec_fn,
        "AAPL",
        models=[pretrained],
        intervals=["weekly"],
        hyperparams=[lookback],
    )
    opt = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        train_size=0.5,
        parallel_indicators=False,
        timeframe="1d",
    )
    sym_df = data_source_df[data_source_df["symbol"] == "AAPL"]
    sym_df = (
        sym_df[
            (sym_df["date"] >= pd.Timestamp(START_DATE))
            & (sym_df["date"] <= pd.Timestamp(END_DATE))
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )
    window = next(
        iter(
            strategy.walkforward_split(
                sym_df, windows=1, lookahead=1, train_size=0.5
            )
        )
    )
    train_dates = {
        pd.Timestamp(date) for date in sym_df["date"].iloc[window.train_data]
    }
    test_dates = {
        pd.Timestamp(date) for date in sym_df["date"].iloc[window.test_data]
    }
    assert train_dates and test_dates
    best = opt.best_params["lookback"]
    assert best in (5, 10)
    losing = 5 if best == 10 else 10
    trial_only_dates = {date for value, date in seen if value == losing}
    assert trial_only_dates
    assert not (trial_only_dates & test_dates)
    assert trial_only_dates <= train_dates
    # Sanity: the final replay (best params) did cover the test window, so
    # the disjointness above is not vacuous.
    replay_dates = {date for value, date in seen if value == best}
    assert test_dates <= replay_dates


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
            parallel_indicators=False,
        )


def test_optimize_walkforward_resolves_selector_once_per_window(
    data_source_df,
):
    # The replay that produces the reported result used to re-run the
    # selector, so a stateful one described a different universe than the
    # study had tuned.
    calls: list[int] = []

    def counting_selector(_selection_df):
        calls.append(len(calls))
        return ["AAPL"]

    def exec_fn(ctx):
        if not ctx.long_pos():
            ctx.buy_shares = 10

    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy, exec_fn, counting_selector)
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        train_size=0.5,
        parallel_indicators=False,
    )
    assert len(calls) == 2


def test_optimize_walkforward_keeps_persistent_orders_across_windows():
    """The stitched replay must share one PendingOrderScope across windows.

    A fresh scope per window drops every order still pending at a boundary,
    including the ``timeout_bars=-1`` orders documented to persist
    indefinitely. Only the stitched result is wrong -- the per-window results
    and best_params are correct -- and the stitched result is the one a caller
    acts on.
    """
    dates = pd.date_range("2021-01-04", periods=60, freq="B")
    # Flat, then a late drop that the limit order finally fills against.
    prices = [100.0] * 50 + [50.0] * 10
    df = pd.DataFrame(
        [
            dict(
                symbol="AAPL",
                date=date,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1e6,
            )
            for date, price in zip(dates, prices)
        ]
    )

    def exec_fn(ctx):
        if not ctx.session.get("placed"):
            ctx.session["placed"] = True
            ctx.buy_shares = 10
            ctx.buy_limit_price = 60
            ctx.buy_timeout_bars = -1

    def make():
        return Strategy(
            df,
            str(dates[0].date()),
            str(dates[-1].date()),
            StrategyConfig(initial_cash=100_000),
        )

    baseline = make()
    baseline.add_execution(exec_fn, "AAPL")
    expected = baseline.walkforward(
        windows=3, train_size=0.5, calc_bootstrap=False
    )

    strategy = make()
    _add_tuned_execution(strategy, exec_fn, "AAPL")
    opt = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=3,
        train_size=0.5,
        parallel_indicators=False,
        calc_bootstrap=False,
    )
    assert not expected.orders.empty
    assert len(opt.result.orders) == len(expected.orders)


def test_optimize_with_lagged_pretrained_model(data_source_df):
    # _run_optimize_test used to omit history_col_scope, so ModelInputScope
    # raised "History data required to compute lags" during the final
    # out-of-sample replay -- after every trial had already run.
    lb = hyperparam("lag_lb", default=10, low=5, high=10, step=5)
    ind = indicator(
        "lag_hhv", lambda data, period: highv(data.high, period), period=lb
    )

    def load_fn(symbol, train_start_date, train_end_date, *args, **kwargs):
        m = MagicMock()
        m.predict = lambda X: np.zeros(len(X))
        return m, ("close",)

    m = model(
        "lag_pre",
        load_fn,
        pretrained=True,
        indicators=[ind],
        lags=2,
        lag_cols=["close"],
    )

    def exec_fn(ctx):
        preds = ctx.preds("lag_pre")
        if len(preds) and not ctx.long_pos():
            ctx.buy_shares = 10

    for windows in (None, 2):
        strategy = _make_strategy(data_source_df)
        strategy.add_execution(exec_fn, "AAPL", models=[m], indicators=[ind])
        result = strategy.optimize(
            lambda r: r.metrics.total_pnl,
            sampler="grid",
            windows=windows,
            parallel_indicators=False,
        )
        assert result.best_params["lag_lb"] in (5, 10)


def test_optimize_grid_parallel_matches_sequential(data_source_df):
    # The parallel grid path took the itertools.product prefix, pinning every
    # hyperparam but the last to its lowest values, while the sequential path
    # let GridSampler shuffle.
    from pybroker.parallel import get_parallel_config, set_parallel

    def build():
        aa = hyperparam("aa", default=5, low=5, high=25, step=5)
        bb = hyperparam("bb", default=10, low=10, high=40, step=10)
        ind = indicator(
            "par_hhv",
            lambda data, period, offset: highv(data.high, period) + offset,
            period=aa,
            offset=bb,
        )
        strategy = _make_strategy(data_source_df)
        strategy.add_execution(lambda _ctx: None, "AAPL", indicators=[ind])
        return strategy

    def run():
        return strategy_params(
            build().optimize(
                lambda r: r.metrics.total_pnl,
                sampler="grid",
                n_trials=4,
                seed=42,
                parallel_indicators=False,
            )
        )

    def strategy_params(result):
        return sorted(
            (t.params["aa"], t.params["bb"]) for t in result.study.trials
        )

    prior = get_parallel_config()
    try:
        # Pin the baseline explicitly rather than inheriting ambient config:
        # the library default is n_jobs=-1, so an unpinned run is parallel.
        set_parallel(n_jobs=1)
        sequential = run()
        set_parallel(n_jobs=2, backend="threading")
        # Logger._start_progress_bar is not thread safe, which is a separate
        # defect from the grid subset this test covers.
        pybroker.disable_progress_bar()
        parallel_result = run()
    finally:
        # Both fields: set_parallel keeps the current backend when it is not
        # passed, so restoring n_jobs alone would leak "threading" globally.
        set_parallel(n_jobs=prior.n_jobs, backend=prior.backend)
        pybroker.enable_progress_bar()
    assert len(sequential) == 4
    assert sequential == parallel_result
    # A biased prefix would pin aa to its minimum in every trial.
    assert len({aa for aa, _ in sequential}) > 1


def test_optimize_when_warmup_not_positive_then_error(data_source_df):
    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy)
    with pytest.raises(ValueError, match="warmup must be > 0"):
        strategy.optimize(lambda r: 0.0, warmup=0)


def test_optimize_when_windows_not_positive_then_error(data_source_df):
    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy)
    with pytest.raises(ValueError, match="windows must be > 0"):
        strategy.optimize(lambda r: 0.0, windows=0)


@pytest.mark.parametrize(
    "start_date, end_date",
    [("2019-01-01", None), (None, "2022-06-01")],
)
def test_optimize_when_dates_out_of_range_then_error(
    data_source_df, start_date, end_date
):
    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy)
    with pytest.raises(ValueError, match="must be between"):
        strategy.optimize(
            lambda r: 0.0, start_date=start_date, end_date=end_date
        )


def test_optimize_when_study_direction_conflicts_then_error(data_source_df):
    import optuna

    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy)
    # optuna.create_study defaults to minimize; deferring to it silently would
    # return the worst trial as the best one.
    with pytest.raises(ValueError, match="direction"):
        strategy.optimize(
            lambda r: 0.0,
            direction="maximize",
            study=optuna.create_study(),
            parallel_indicators=False,
        )


def test_optimize_when_study_and_multiple_windows_then_error(data_source_df):
    import optuna

    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy)
    with pytest.raises(ValueError, match="study= is not supported"):
        strategy.optimize(
            lambda r: 0.0,
            windows=2,
            study=optuna.create_study(direction="maximize"),
            parallel_indicators=False,
        )


def test_optimize_when_study_supplied_then_trials_recorded(data_source_df):
    import optuna

    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy)
    study = optuna.create_study(direction="maximize")
    result = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        study=study,
        n_trials=2,
        parallel_indicators=False,
    )
    assert result.study is study
    assert len(study.trials) == 2


def test_optimize_when_model_indicator_has_hyperparam_then_collected(
    data_source_df,
):
    # An indicator registered only on the model source is still expanded by
    # Strategy._fetch_indicators, so its hyperparam has to be collected or the
    # run hyperparams dict is missing a name the indicator asks for.
    period = hyperparam("m_period", default=10, low=5, high=10, step=5)
    ind = indicator(
        "m_hhv", lambda data, period: highv(data.high, period), period=period
    )

    def load_fn(symbol, *args, **kwargs):
        m = MagicMock()
        m.predict = lambda X: np.zeros(len(X))
        return m

    m = model("m_pre", load_fn, pretrained=True, indicators=[ind])
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(lambda _ctx: None, "AAPL", models=[m])

    assert "m_period" in collect_hyperparams(strategy)
    assert "m_period" in collect_search_space(strategy).hyperparams
    result = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        parallel_indicators=False,
    )
    assert result.best_params["m_period"] in (5, 10)


def test_optimize_when_max_long_positions_is_hyperparam_and_windows(
    data_source_df,
):
    # The stitching Portfolio used to be sized with run_hyperparams=None, which
    # raises for a Hyperparam position limit -- after every window study had
    # already run.
    mlp = hyperparam("mlp", default=1, low=1, high=3, step=1)
    strategy = _make_strategy(data_source_df)
    strategy.set_max_long_positions(mlp)
    strategy.add_execution(lambda _ctx: None, "AAPL")
    result = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        parallel_indicators=False,
    )
    assert result.best_params["mlp"] in (1, 2, 3)
    assert result.windows is not None and len(result.windows) == 2


def test_optimize_invariant_indicator_computed_once_per_run(data_source_df):
    # A hyperparam-free indicator is neither disk cached nor memoized in the
    # trial path, so fetching the whole indicator set per trial recomputed it
    # once per trial and discarded the invariant precomputation.
    lb = hyperparam("lb", default=10, low=5, high=25, step=5)
    tuned_calls: list[int] = []
    invariant_calls: list[int] = []

    def tuned_fn(data, period):
        tuned_calls.append(period)
        return highv(data.high, period)

    def invariant_fn(data):
        invariant_calls.append(1)
        return highv(data.high, 20)

    tuned = indicator("tuned_ind", tuned_fn, period=lb)
    invariant = indicator("invariant_ind", invariant_fn)
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(
        lambda _ctx: None, "AAPL", indicators=[tuned, invariant]
    )
    result = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        parallel_indicators=False,
    )
    assert len(result.study.trials) == 5
    # One compute per distinct hyperparam value, memoized across trials.
    assert sorted(set(tuned_calls)) == [5, 10, 15, 20, 25]
    assert len(tuned_calls) == 5
    assert len(invariant_calls) == 1


def test_optimize_clears_indicator_memo_between_calls(data_source_df):
    # The memo key has no notion of the data window, so a surviving memo used
    # to serve the first call's series to a second call over other dates.
    lb = hyperparam("lb", default=10, low=5, high=10, step=5)
    ind = indicator(
        "memo_hhv", lambda data, period: highv(data.high, period), period=lb
    )

    seen: list[int] = []

    def exec_fn(ctx):
        vals = ctx.indicator("memo_hhv")
        seen.append(len(vals))
        if len(vals) and not ctx.long_pos():
            ctx.buy_shares = 10

    def run(strategy):
        return strategy.optimize(
            lambda r: r.metrics.total_pnl,
            sampler="grid",
            start_date="2021-01-04",
            end_date="2021-12-31",
            parallel_indicators=False,
        )

    reused = _make_strategy(data_source_df)
    reused.add_execution(exec_fn, "AAPL", indicators=[ind])
    reused.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        start_date="2020-01-02",
        end_date="2020-06-30",
        parallel_indicators=False,
    )
    assert not reused._indicator_memo
    second = run(reused)

    fresh = _make_strategy(data_source_df)
    fresh.add_execution(exec_fn, "AAPL", indicators=[ind])
    expected = run(fresh)

    assert second.best_params == expected.best_params
    assert second.result.metrics.total_pnl == expected.result.metrics.total_pnl
    assert len(second.result.orders) == len(expected.result.orders)


def test_optimize_honors_exit_on_last_bar(data_source_df):
    # optimize used to hardcode exit_dates={}, so end-of-data liquidation never
    # happened and realized-PnL score_fns ranked trials on an unclosed book.
    def exec_fn(ctx):
        if not ctx.long_pos():
            ctx.buy_shares = 10

    config = StrategyConfig(initial_cash=100_000, exit_on_last_bar=True)
    strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
    _add_tuned_execution(strategy, exec_fn)
    opt = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        parallel_indicators=False,
    )

    wf_strategy = Strategy(data_source_df, START_DATE, END_DATE, config)
    _add_tuned_execution(wf_strategy, exec_fn)
    wf = wf_strategy.walkforward(windows=2, train_size=0.5)

    assert len(opt.result.trades) == len(wf.trades)
    assert len(opt.result.trades) > 0


def test_optimize_sessions_persist_across_windows(data_source_df):
    # The stitched replay carries positions and cash across windows, so
    # ctx.session has to persist across them too.
    def exec_fn(ctx):
        ctx.session["n"] = ctx.session.get("n", 0) + 1

    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy, exec_fn)
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=4,
        parallel_indicators=False,
    )
    wf_strategy = _make_strategy(data_source_df)
    counts: list[int] = []

    def wf_exec_fn(ctx):
        ctx.session["n"] = ctx.session.get("n", 0) + 1
        counts.append(ctx.session["n"])

    _add_tuned_execution(wf_strategy, wf_exec_fn)
    wf_strategy.walkforward(windows=4, train_size=0.5)
    # The stitched replay is the last thing optimize runs, so the session
    # counter it leaves behind must reach the same total walkforward does.
    assert max(counts) > len(counts) // 4


def test_optimize_windows_hold_tuning_results_only(data_source_df):
    """A window holds what it was tuned to, not a backtest of its own: the
    single out-of-sample TestResult is the stitched ``result``, matching the
    one-TestResult shape of ``Strategy.walkforward``.
    """

    def exec_fn(ctx):
        if not ctx.long_pos():
            ctx.buy_shares = 10

    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy, exec_fn)
    result = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        parallel_indicators=False,
    )
    assert len(result.windows) == 2
    for window in result.windows:
        assert window.params["lookback"] in (5, 10)
        assert isinstance(window.train_score, float)
        assert not hasattr(window, "test_result")
        for field in (
            window.train_start_date,
            window.train_end_date,
            window.test_start_date,
            window.test_end_date,
        ):
            assert isinstance(field, datetime)
        # lookahead holds train strictly before test.
        assert window.train_start_date <= window.train_end_date
        assert window.train_end_date < window.test_start_date
        assert window.test_start_date <= window.test_end_date
        payload = window.to_json()
        assert set(payload.keys()) == {
            "params",
            "train_score",
            "train_start_date",
            "train_end_date",
            "test_start_date",
            "test_end_date",
            "study",
        }
        window.to_json_str()
    assert result.windows[0].test_end_date < result.windows[1].test_end_date
    assert not hasattr(result, "walkforward_efficiency")
    # to_json_str uses allow_nan=False, so any NaN in the payload would raise.
    result.to_json_str()


@pytest.mark.parametrize("train_size", [0, 1, -0.5, 1.5])
def test_optimize_when_no_train_or_test_window_then_error(
    data_source_df, train_size
):
    # An empty train window scores every trial identically, so best_params
    # would be decided by enumeration order rather than by the data. An empty
    # test window leaves nothing to evaluate the winner on.
    hyperparam("lookback", default=10, low=5, high=10, step=5)
    strategy = _make_strategy(data_source_df)
    strategy.add_execution(lambda _ctx: None, lambda _d: ["AAPL"])
    with pytest.raises(ValueError, match="0 < train_size < 1"):
        strategy.optimize(
            lambda r: 0.0,
            sampler="grid",
            train_size=train_size,
            parallel_indicators=False,
        )


def test_run_study_when_grid_sampler_missing_hyperparam_then_error():
    """A supplied GridSampler covering part of the space must be rejected on
    both paths: sequentially the objective raises for the missing name, while
    in parallel the point simply lacks it and the default is used silently."""
    import optuna
    from optuna.samplers import GridSampler
    from pybroker.optimize import Hyperparam, SearchSpace, _run_study
    from pybroker.parallel import get_parallel_config, set_parallel

    space = SearchSpace(
        hyperparams=frozenset({"x", "y"}),
        specs={
            "x": Hyperparam("x", default=1, low=1, high=2, step=1),
            "y": Hyperparam("y", default=1, low=1, high=2, step=1),
        },
    )

    class _Bundle:
        search_space = space
        score_overrides = staticmethod(lambda params: 1.0)
        objective = staticmethod(lambda trial: 1.0)

    sampler = GridSampler({"x": [1, 2]}, seed=1)
    prior = get_parallel_config()
    try:
        for n_jobs in (1, 2):
            set_parallel(n_jobs=n_jobs, backend="threading")
            study = optuna.create_study(direction="maximize", sampler=sampler)
            with pytest.raises(ValueError, match="missing declared"):
                _run_study(study, _Bundle(), 2, sampler)
    finally:
        set_parallel(n_jobs=prior.n_jobs, backend=prior.backend)


def test_run_study_when_nan_score_in_parallel_then_trial_fails():
    """study.tell rejects NaN while study.optimize records a failed trial and
    carries on, so one bad trial must not abort a parallel study."""
    import optuna
    from optuna.samplers import RandomSampler
    from pybroker.optimize import Hyperparam, SearchSpace, _run_study
    from pybroker.parallel import get_parallel_config, set_parallel

    space = SearchSpace(
        hyperparams=frozenset({"x"}),
        specs={"x": Hyperparam("x", default=1, low=1, high=3, step=1)},
    )

    def score(params):
        return float("nan") if params.get("x") == 2 else float(params["x"])

    class _Bundle:
        search_space = space
        score_overrides = staticmethod(score)
        objective = staticmethod(
            lambda trial: score({"x": trial.suggest_int("x", 1, 3)})
        )

    prior = get_parallel_config()
    try:
        set_parallel(n_jobs=2, backend="threading")
        sampler = RandomSampler(seed=1)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        _run_study(study, _Bundle(), 6, sampler)
    finally:
        set_parallel(n_jobs=prior.n_jobs, backend=prior.backend)
    assert len(study.trials) == 6
    assert study.best_value == 3.0


def test_run_study_when_grid_sampler_and_nan_score_then_trial_fails():
    """The parallel grid path must tolerate a NaN score as the sequential one
    does: create_trial rejects NaN exactly as study.tell does."""
    import optuna
    from optuna.samplers import GridSampler
    from pybroker.optimize import Hyperparam, SearchSpace, _run_study
    from pybroker.parallel import get_parallel_config, set_parallel

    space = SearchSpace(
        hyperparams=frozenset({"x"}),
        specs={"x": Hyperparam("x", default=1, low=1, high=3, step=1)},
    )

    def score(params):
        return float("nan") if params["x"] == 2 else float(params["x"])

    class _Bundle:
        search_space = space
        score_overrides = staticmethod(score)
        objective = staticmethod(
            lambda trial: score({"x": trial.suggest_int("x", 1, 3)})
        )

    prior = get_parallel_config()
    results = {}
    try:
        for label, n_jobs in (("sequential", 1), ("parallel", 2)):
            set_parallel(n_jobs=n_jobs, backend="threading")
            sampler = GridSampler({"x": [1, 2, 3]}, seed=1)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            _run_study(study, _Bundle(), 3, sampler)
            results[label] = (len(study.trials), study.best_value)
    finally:
        set_parallel(n_jobs=prior.n_jobs, backend=prior.backend)
    assert results["parallel"] == results["sequential"] == (3, 3.0)


def test_run_study_when_sampler_stops_then_batch_still_recorded():
    """A sampler stopping mid-batch must not strand already-scored trials.

    Only RandomSampler (and subclasses) reaches the batched path — adaptive
    samplers are forced sequential — so the mid-batch stop is exercised with
    a RandomSampler whose after_trial calls Study.stop(), the way
    BruteForceSampler does once its space is exhausted. Every score in the
    batch has already been paid for, so all of them must be told before the
    loop exits.
    """
    import optuna
    from optuna.samplers import RandomSampler
    from pybroker.optimize import Hyperparam, SearchSpace, _run_study
    from pybroker.parallel import get_parallel_config, set_parallel

    space = SearchSpace(
        hyperparams=frozenset({"x"}),
        specs={"x": Hyperparam("x", default=1, low=1, high=3, step=1)},
    )

    class _Bundle:
        search_space = space
        score_overrides = staticmethod(lambda params: float(params["x"]))
        objective = staticmethod(
            lambda trial: float(trial.suggest_int("x", 1, 3))
        )

    class _StoppingRandomSampler(RandomSampler):
        tells = 0

        def after_trial(self, study, trial, state, values):
            super().after_trial(study, trial, state, values)
            type(self).tells += 1
            if type(self).tells >= 3:
                study.stop()

    prior = get_parallel_config()
    try:
        set_parallel(n_jobs=4, backend="threading")
        sampler = _StoppingRandomSampler(seed=1)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        _run_study(study, _Bundle(), 12, sampler)
    finally:
        set_parallel(n_jobs=prior.n_jobs, backend=prior.backend)
    states = [trial.state for trial in study.trials]
    # The stop fired on the third tell of a four-trial batch: the whole batch
    # is recorded, and no further batch is drawn.
    assert len(states) == 4
    assert all(state == optuna.trial.TrialState.COMPLETE for state in states)


def test_run_study_adaptive_sampler_forced_sequential_with_info_log(caplog):
    """Adaptive samplers must not fan out: batching would change their
    proposals and tie results to the worker count. They run sequentially,
    noted by an info log rather than a warning, and produce the same trials
    as an n_jobs=1 run.
    """
    import logging
    import optuna
    import warnings as warnings_mod
    from optuna.samplers import TPESampler
    from pybroker.optimize import Hyperparam, SearchSpace, _run_study
    from pybroker.parallel import get_parallel_config, set_parallel

    space = SearchSpace(
        hyperparams=frozenset({"x"}),
        specs={"x": Hyperparam("x", default=1, low=1, high=3, step=1)},
    )

    class _Bundle:
        search_space = space
        score_overrides = staticmethod(lambda params: float(params["x"]))
        objective = staticmethod(
            lambda trial: float(trial.suggest_int("x", 1, 3))
        )

    def run(n_jobs):
        prior = get_parallel_config()
        try:
            set_parallel(n_jobs=n_jobs, backend="threading")
            sampler = TPESampler(seed=7)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            _run_study(study, _Bundle(), 6, sampler)
        finally:
            set_parallel(n_jobs=prior.n_jobs, backend=prior.backend)
        return [trial.params["x"] for trial in study.trials]

    caplog.set_level(logging.INFO, logger="pybroker")
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        parallel_params = run(n_jobs=4)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert any(
        "Running TPESampler trials sequentially" in r.getMessage()
        for r in caplog.records
    )
    sequential_params = run(n_jobs=1)
    assert len(parallel_params) == 6
    # Forced-sequential evaluation is machine-independent: identical to a
    # genuinely sequential run with the same seed.
    assert parallel_params == sequential_params


def test_run_study_random_parallel_matches_sequential(caplog):
    """RandomSampler's draws ignore completed results, so the batched path
    must propose the same points as a sequential run, in the same order,
    without falling back to sequential evaluation.
    """
    import logging
    import optuna
    from optuna.samplers import RandomSampler
    from pybroker.optimize import Hyperparam, SearchSpace, _run_study
    from pybroker.parallel import get_parallel_config, set_parallel

    space = SearchSpace(
        hyperparams=frozenset({"x"}),
        specs={"x": Hyperparam("x", default=1, low=1, high=3, step=1)},
    )

    class _Bundle:
        search_space = space
        score_overrides = staticmethod(lambda params: float(params["x"]))
        objective = staticmethod(
            lambda trial: float(trial.suggest_int("x", 1, 3))
        )

    def run(n_jobs):
        prior = get_parallel_config()
        try:
            set_parallel(n_jobs=n_jobs, backend="threading")
            sampler = RandomSampler(seed=7)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            _run_study(study, _Bundle(), 6, sampler)
        finally:
            set_parallel(n_jobs=prior.n_jobs, backend=prior.backend)
        return [trial.params["x"] for trial in study.trials]

    caplog.set_level(logging.INFO, logger="pybroker")
    parallel_params = run(n_jobs=2)
    assert not [
        r for r in caplog.records if "trials sequentially" in r.getMessage()
    ]
    sequential_params = run(n_jobs=1)
    assert len(parallel_params) == 6
    assert parallel_params == sequential_params


def test_run_study_reraises_unrelated_runtime_error():
    """The stop-signal catch must not swallow a genuine RuntimeError."""
    import optuna
    from optuna.samplers import RandomSampler
    from pybroker.optimize import Hyperparam, SearchSpace, _run_study
    from pybroker.parallel import get_parallel_config, set_parallel

    space = SearchSpace(
        hyperparams=frozenset({"x"}),
        specs={"x": Hyperparam("x", default=1, low=1, high=3, step=1)},
    )

    class _Bundle:
        search_space = space
        score_overrides = staticmethod(lambda params: float(params["x"]))
        objective = staticmethod(
            lambda trial: float(trial.suggest_int("x", 1, 3))
        )

    prior = get_parallel_config()
    try:
        set_parallel(n_jobs=2, backend="threading")
        sampler = RandomSampler(seed=1)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        with patch.object(
            study, "tell", side_effect=RuntimeError("storage exploded")
        ):
            with pytest.raises(RuntimeError, match="storage exploded"):
                _run_study(study, _Bundle(), 2, sampler)
    finally:
        set_parallel(n_jobs=prior.n_jobs, backend=prior.backend)


def test_supplied_sampler_instance_is_reseeded_per_window(data_source_df):
    """A supplied sampler must not be shared by every walkforward window.

    Returning the instance itself hands all windows one advancing RNG
    sequentially, and one pickled copy at the same state under n_jobs > 1 --
    where every window then explores an identical candidate set. Either way
    the per-window seed is discarded.
    """
    search_space = SearchSpace(
        hyperparams=frozenset(("lookback",)),
        specs={"lookback": (5, 10)},
    )
    supplied = optuna.samplers.RandomSampler(seed=1)
    first = _build_sampler(supplied, search_space, seed=5)
    second = _build_sampler(supplied, search_space, seed=6)
    third = _build_sampler(supplied, search_space, seed=5)
    assert first is not supplied
    assert second is not first
    draws = [s._rng.rng.rand() for s in (first, second, third)]
    # Same window seed reproduces; a different one does not.
    assert draws[0] == draws[2]
    assert draws[0] != draws[1]


def test_run_study_parallel_grid_resumes_instead_of_repeating():
    """A resumed grid study must continue where it left off.

    The sequential path lets GridSampler.before_trial map grid_id from
    trial.number, so slicing the sampler's grid from 0 re-evaluates the same
    opening points and never reaches the rest of the grid. add_trial bypasses
    before_trial, so optuna's "grid exhausted" warning never fires either.
    """
    from optuna.samplers import GridSampler
    from pybroker.optimize import Hyperparam, SearchSpace, _run_study
    from pybroker.parallel import get_parallel_config, set_parallel

    space = SearchSpace(
        hyperparams=frozenset({"x"}),
        specs={"x": Hyperparam("x", default=5, low=5, high=25, step=5)},
    )

    class _Bundle:
        search_space = space
        score_overrides = staticmethod(lambda params: float(params["x"]))
        objective = staticmethod(
            lambda trial: float(trial.suggest_int("x", 5, 25, step=5))
        )

    prior = get_parallel_config()
    try:
        set_parallel(n_jobs=2, backend="threading")
        sampler = GridSampler({"x": [5, 10, 15, 20, 25]}, seed=7)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        _run_study(study, _Bundle(), 2, sampler)
        _run_study(study, _Bundle(), 2, sampler)
    finally:
        set_parallel(n_jobs=prior.n_jobs, backend=prior.backend)
    visited = [t.params["x"] for t in study.trials]
    assert len(visited) == 4
    # Four distinct grid points, not the first two evaluated twice.
    assert len(set(visited)) == 4


def test_optimize_returns_signals_when_configured(data_source_df):
    """``return_signals`` must not be silently ignored.

    backtest_executions computes the frames either way, so all three optimize
    result paths were doing the work and discarding it.
    """

    def exec_fn(ctx):
        if not ctx.long_pos():
            ctx.buy_shares = 10

    strategy = Strategy(
        data_source_df,
        START_DATE,
        END_DATE,
        StrategyConfig(initial_cash=100_000, return_signals=True),
    )
    _add_tuned_execution(strategy, exec_fn, "AAPL")
    opt = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        train_size=0.5,
        parallel_indicators=False,
        calc_bootstrap=False,
    )
    assert opt.result.signals is not None
    assert "AAPL" in opt.result.signals
    assert not opt.result.signals["AAPL"].empty


def test_optimize_and_walkforward_agree_on_order_count():
    """Both entry points must trade identically on identical input.

    They run the same bar loop through different call paths, so a fix threaded
    into one and not the other diverges silently -- the cross-window order
    scheduling fix reached Strategy.walkforward and missed all three
    backtest_executions calls in optimize, costing the stitched replay roughly
    one order per symbol per window boundary.
    """
    dates = pd.date_range("2021-01-04", periods=60, freq="B")
    prices = [100.0 + (i % 5) for i in range(60)]
    df = pd.DataFrame(
        [
            dict(
                symbol="AAPL",
                date=date,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1e6,
            )
            for date, price in zip(dates, prices)
        ]
    )

    def exec_fn(ctx):
        # Signals on every bar, so window-boundary bars are exercised.
        if not ctx.long_pos():
            ctx.buy_shares = 10
        else:
            ctx.sell_all_shares()

    def make():
        return Strategy(
            df,
            str(dates[0].date()),
            str(dates[-1].date()),
            StrategyConfig(initial_cash=100_000),
        )

    baseline = make()
    baseline.add_execution(exec_fn, "AAPL")
    expected = baseline.walkforward(
        windows=3, train_size=0.5, calc_bootstrap=False
    )

    strategy = make()
    _add_tuned_execution(strategy, exec_fn, "AAPL")
    opt = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=3,
        train_size=0.5,
        parallel_indicators=False,
        calc_bootstrap=False,
    )
    assert len(expected.orders) > 0
    assert len(opt.result.orders) == len(expected.orders)
    assert len(opt.result.trades) == len(expected.trades)
