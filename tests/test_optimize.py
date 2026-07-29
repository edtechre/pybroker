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
        disable_parallel_indicators=True,
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
            disable_parallel_indicators=True,
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
        disable_parallel_indicators=True,
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
        disable_parallel_indicators=True,
    )
    assert len(calls) == 2


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
            disable_parallel_indicators=True,
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
                disable_parallel_indicators=True,
            )
        )

    def strategy_params(result):
        return sorted(
            (t.params["aa"], t.params["bb"]) for t in result.study.trials
        )

    sequential = run()
    prior = get_parallel_config()
    try:
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
            disable_parallel_indicators=True,
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
            disable_parallel_indicators=True,
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
        disable_parallel_indicators=True,
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
        disable_parallel_indicators=True,
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
        disable_parallel_indicators=True,
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
        disable_parallel_indicators=True,
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
            disable_parallel_indicators=True,
        )

    reused = _make_strategy(data_source_df)
    reused.add_execution(exec_fn, "AAPL", indicators=[ind])
    reused.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        start_date="2020-01-02",
        end_date="2020-06-30",
        disable_parallel_indicators=True,
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
        disable_parallel_indicators=True,
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
        disable_parallel_indicators=True,
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


def test_optimize_walkforward_efficiency_none_when_train_unprofitable(
    data_source_df,
):
    def exec_fn(ctx):
        if not ctx.long_pos():
            ctx.buy_shares = 10

    strategy = _make_strategy(data_source_df)
    _add_tuned_execution(strategy, exec_fn)
    result = strategy.optimize(
        # Minimizing profit drives in-sample PnL to zero or below, which leaves
        # the efficiency ratio undefined rather than negative.
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        direction="minimize",
        windows=2,
        disable_parallel_indicators=True,
    )
    is_pnl = sum(w.train_pnl for w in result.windows)
    if is_pnl > 0:
        assert result.walkforward_efficiency is not None
    else:
        assert result.walkforward_efficiency is None
    # to_json_str uses allow_nan=False, so a NaN efficiency would raise.
    result.to_json_str()
    for window in result.windows:
        window.to_json_str()


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
            disable_parallel_indicators=True,
        )
