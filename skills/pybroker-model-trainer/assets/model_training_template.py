"""Starter PyBroker model training script.

Copy this file into a project and adapt the symbols, dates, features,
training function, and execution rules to the user's model.
"""

import pybroker
from pybroker import ExecContext, Strategy, StrategyConfig, YFinance
from pybroker.indicator import close_minus_ma
from sklearn.linear_model import LinearRegression


SYMBOLS = ["AAPL", "MSFT"]
START_DATE = "1/1/2020"
END_DATE = "1/1/2024"
LOOKBACK = 20
MODEL_NAME = "next_return"

# Cache downloaded data, indicators, and trained models across runs.
pybroker.enable_caches("model_training_template")
# Progress bars flood AI token context and add nothing to a log file.
pybroker.disable_progress_bar()

cmma_20 = close_minus_ma("cmma_20", lookback=LOOKBACK, atr_length=14)


def train_fn(symbol: str, train_data, test_data):
    # Copy so the input frame is never widened or mutated.
    df = train_data.copy()
    # Target is the NEXT bar's return, matching walkforward lookahead=1.
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    df = df.dropna()
    model = LinearRegression()
    model.fit(df[["cmma_20"]], df["target"])
    # Pin the columns used as prediction input.
    return model, ["cmma_20"]


model_source = pybroker.model(MODEL_NAME, train_fn, indicators=[cmma_20])


def exec_fn(ctx: ExecContext):
    pred = ctx.preds(MODEL_NAME)[-1]
    if not ctx.long_pos():
        if pred > 0:
            ctx.buy_shares = ctx.calc_target_shares(0.5)
    elif pred < 0:
        ctx.sell_all_shares()


def build_strategy() -> Strategy:
    config = StrategyConfig(
        initial_cash=100_000,
        exit_on_last_bar=True,
    )
    strategy = Strategy(YFinance(), START_DATE, END_DATE, config)
    strategy.add_execution(exec_fn, SYMBOLS, models=model_source)
    return strategy


if __name__ == "__main__":
    result = build_strategy().walkforward(
        windows=3,
        train_size=0.5,
        lookahead=1,  # bars ahead of the prediction target
        warmup=LOOKBACK,
    )
    print(result.metrics_df)
