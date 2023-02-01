"""Logging module."""

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

from .common import Day, IndicatorSymbol, ModelSymbol, to_datetime
from decimal import Decimal
from progressbar import ProgressBar
from typing import Iterable, Optional, Sequence, Sized, Union
import datetime
import logging
import numpy as np
import time


class Logger:
    """Class for logging information about triggered events.

    Args:
        scope: :class:`pybroker.scope.StaticScope`.
    """

    def __init__(self, scope):
        self._scope = scope
        self._progress_bar: Optional[ProgressBar] = None
        self._download_start_time: Optional[float] = None
        self._train_split_start_time: Optional[float] = None
        self._train_model_start_time: Optional[float] = None
        self._walkforward_start_time: Optional[float] = None
        self._bootstrap_start_time: Optional[float] = None
        self._progress_bar_disabled = False
        self._disabled = False

    def _start_progress_bar(self, message: str, total_count: int):
        if self._disabled:
            return
        if self._progress_bar_disabled:
            print(message, flush=True)
            return
        self._progress_bar = ProgressBar(max_value=total_count)
        self._out(message)
        self._progress_bar.update(0)

    def _update_progress_bar(self, count: int):
        if (
            self._progress_bar is None
            or self._disabled
            or self._progress_bar_disabled
        ):
            return
        self._progress_bar.update(count)
        if count == self._progress_bar.max_value:
            self._progress_bar.finish()
            self._progress_bar = None
            self._out("")

    def disable(self):
        """Disables logging."""
        self._disabled = True

    def enable(self):
        """Enables logging."""
        self._disabled = False

    def disable_progress_bar(self):
        """Disables logging a progress bar."""
        self._progress_bar_disabled = True

    def enable_progress_bar(self):
        """Enables logging a progress bar."""
        self._progress_bar_disabled = False

    def download_bar_data_start(self):
        self._out("Loading bar data...")
        self._download_start_time = time.time()

    def info_download_bar_data_start(
        self,
        symbols: Iterable[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        timeframe: str,
    ):
        self._info(
            "Loading:\n"
            f"{start_date} to {end_date}\n"
            f"timeframe: {timeframe}\n"
            f"{sorted(symbols)}"
        )

    def loaded_bar_data(self):
        self._out("Loaded cached bar data.\n")

    def info_loaded_bar_data(
        self,
        symbols: Iterable[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        timeframe: str,
    ):
        self._info(
            "Loaded:\n"
            f"namespace={self._scope.data_source_cache_ns}\n"
            f"{start_date} to {end_date}\n",
            f"timeframe: {timeframe}\n",
            f"{sorted(symbols)}",
        )

    def info_invalidate_data_source_cache(self):
        self._info(
            "Mismatched columns in data source cache:\n"
            f"namespace={self._scope.data_source_cache_ns}\n"
            "Invalidating cache..."
        )

    def debug_get_data_source_cache(self, cache_key):
        self._debug(f"Fetched data source cache:\n{cache_key}")

    def debug_set_data_source_cache(self, cache_key):
        self._debug(f"Set data source cache:\n{cache_key}")

    def download_bar_data_completed(self):
        if self._download_start_time is None:
            return
        self._out(
            "Loaded bar data:",
            self._format_time(self._download_start_time),
            "\n",
        )
        self._download_start_time = None

    def indicator_data_start(self, ind_syms: Sized):
        self._start_progress_bar("Generating indicators...", len(ind_syms))

    def info_indicator_data_start(self, ind_syms: Iterable[IndicatorSymbol]):
        self._info(f"Indicators: {sorted(ind_syms)}")

    def loaded_indicator_data(self):
        self._out("Loaded cached indicator data.\n")

    def info_loaded_indicator_data(self, ind_syms: Iterable[IndicatorSymbol]):
        self._info(
            f"Loaded:\n"
            f"namespace={self._scope.indicator_cache_ns}\n"
            f"{sorted(ind_syms)}"
        )

    def indicator_data_loading(self, count: int):
        self._update_progress_bar(count)

    def debug_get_indicator_cache(self, cache_key):
        self._debug(f"Fetched indicator cache:\n{cache_key}")

    def debug_set_indicator_cache(self, cache_key):
        self._debug(f"Set indicator cache:\n{cache_key}")

    def debug_compute_indicators(self, is_parallel: bool):
        self._debug(
            "Computing indicators in parallel."
            if is_parallel
            else "Computing indicators in serial."
        )

    def train_split_start(self, train_dates: Sequence[np.datetime64]):
        start_date = to_datetime(train_dates[0])
        end_date = to_datetime(train_dates[-1])
        self._out(f"Train split: {start_date} to {end_date}")
        self._train_split_start_time = time.time()

    def info_train_split_start(self, model_syms: Iterable[ModelSymbol]):
        self._info(f"Models: {sorted(model_syms)}")

    def loaded_models(self):
        self._out("Loaded cached models.\n")

    def info_loaded_models(self, model_syms: Iterable[ModelSymbol]):
        self._info(
            f"Loaded:\n"
            f"namespace={self._scope.model_cache_ns}\n"
            f"{sorted(model_syms)}"
        )

    def info_train_model_start(self, model_sym: ModelSymbol):
        self._info(f"Training model: {model_sym}")
        self._train_model_start_time = time.time()

    def info_train_model_completed(self, model_sym: ModelSymbol):
        if self._train_model_start_time is None:
            return
        self._info(
            f"Finished training model {model_sym}:",
            self._format_time(self._train_model_start_time),
        )
        self._train_model_start_time = None

    def info_loaded_model(self, model_sym: ModelSymbol):
        self._info(f"Loaded model: {model_sym}")

    def debug_get_model_cache(self, cache_key):
        self._debug(f"Fetched model cache:\n{cache_key}")

    def debug_set_model_cache(self, cache_key):
        self._debug(f"Set model cache:\n{cache_key}")

    def train_split_completed(self):
        if self._train_split_start_time is None:
            return
        self._out(
            "Finished training models:",
            self._format_time(self._train_split_start_time),
            "\n",
        )
        self._train_split_start_time = None

    def backtest_executions_start(self, test_dates: Sequence[np.datetime64]):
        if not len(test_dates):
            return
        start_date = to_datetime(test_dates[0])
        end_date = to_datetime(test_dates[-1])
        self._start_progress_bar(
            f"Test split: {start_date} to {end_date}", len(test_dates)
        )

    def backtest_executions_loading(self, count: int):
        self._update_progress_bar(count)

    def walkforward_start(
        self, start_date: datetime.datetime, end_date: datetime.datetime
    ):
        self._out(f"Backtesting: {start_date} to {end_date}\n")
        self._walkforward_start_time = time.time()

    def info_walkforward_between_time(self, between_time: tuple[str, str]):
        self._info(f"Backtest between times: {between_time}")

    def info_walkforward_on_days(self, days: tuple[int]):
        self._info(f"Backtest on days: {map(lambda d: Day(d).name, days)}")

    def walkforward_completed(self):
        if self._walkforward_start_time is None:
            return
        self._out(
            "Finished backtest:",
            self._format_time(self._walkforward_start_time),
        )
        self._walkforward_start_time = None

    def calc_bootstrap_metrics_start(self, samples, sample_size):
        self._out(
            f"Calculating bootstrap metrics: sample_size={sample_size}, "
            f"samples={samples}..."
        )
        self._bootstrap_start_time = time.time()

    def calc_bootstrap_metrics_completed(self):
        if self._bootstrap_start_time is None:
            return
        self._out(
            "Calculated bootstrap metrics:",
            self._format_time(self._bootstrap_start_time),
            "\n",
        )
        self._bootstrap_start_time = None

    def debug_place_buy_order(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Union[int, float, Decimal],
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ):
        order = self._format_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        self._debug(f"Placing buy order:\n{order}")

    def debug_filled_buy_order(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Union[int, float, Decimal],
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ):
        order = self._format_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        self._debug(f"Filled buy order:\n{order}")

    def debug_unfilled_buy_order(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Union[int, float, Decimal],
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ):
        order = self._format_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        self._debug(f"Unfilled buy order:\n{order}")

    def debug_place_sell_order(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Union[int, float, Decimal],
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ):
        order = self._format_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        self._debug(f"Placing sell order:\n{order}")

    def debug_filled_sell_order(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Union[int, float, Decimal],
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ):
        order = self._format_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        self._debug(f"Filled sell order:\n{order}")

    def debug_unfilled_sell_order(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Union[int, float, Decimal],
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ):
        order = self._format_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        self._debug(f"Unfilled sell order:\n{order}")

    def debug_schedule_order(self, date: np.datetime64, exec_result):
        self._debug(f"Scheduling order: {date}\n{exec_result}")

    def debug_unscheduled_order(self, exec_result):
        self._debug(f"Unscheduled order:\n{exec_result}")

    def warn_bootstrap_sample_size(self, n: int, sample_size: int):
        self._warn(
            f"Returns length {n} < sample size {sample_size}.\n"
            "Setting number of bootstraps to 1."
        )

    def debug_enable_data_source_cache(self, ns: str, cache_dir: str):
        self._debug(
            "Enabled data source cache:\n"
            f"namespace={ns}\n"
            f"dir={cache_dir}"
        )

    def debug_disable_data_source_cache(self):
        self._debug("Disabled data source cache.")

    def debug_clear_data_source_cache(self, cache_dir: str):
        self._debug(f"Cleared data source cache: {cache_dir}")

    def debug_enable_indicator_cache(self, ns: str, cache_dir: str):
        self._debug(
            "Enabled indicator cache:\n" f"namespace={ns}\n" f"dir={cache_dir}"
        )

    def debug_disable_indicator_cache(self):
        self._debug("Disabled indicator cache.")

    def debug_clear_indicator_cache(self, cache_dir: str):
        self._debug(f"Cleared indicator cache: {cache_dir}")

    def debug_enable_model_cache(self, ns: str, cache_dir: str):
        self._debug(
            "Enabled model cache:\n" f"namespace={ns}\n" f"dir={cache_dir}"
        )

    def debug_disable_model_cache(self):
        self._debug("Disabled model cache.")

    def debug_clear_model_cache(self, cache_dir: str):
        self._debug(f"Cleared model cache: {cache_dir}")

    def _out(self, msg: str, *args):
        if self._disabled:
            return
        print(msg, *args, flush=True)

    def _info(self, msg: str, *args):
        if self._disabled:
            return
        logging.info(msg, *args)

    def _debug(self, msg: str, *args):
        if self._disabled:
            return
        logging.debug(msg, *args)

    def _warn(self, msg: str, *args):
        if self._disabled:
            return
        logging.warn(msg, *args)

    def _format_order(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Union[int, float, Decimal],
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ):
        return (
            f"date={to_datetime(date)}\n"
            f"symbol={symbol}\n"
            f"shares={shares}\n"
            f"fill_price={fill_price}\n"
            f"limit_price={limit_price}\n"
        )

    def _format_time(self, start_seconds: float) -> str:
        delta = time.time() - start_seconds
        return str(datetime.timedelta(seconds=round(delta)))
