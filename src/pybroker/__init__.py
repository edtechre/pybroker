"""Global imports."""

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

from pybroker.cache import (
    clear_caches,
    clear_data_source_cache,
    clear_indicator_cache,
    clear_model_cache,
    disable_caches,
    disable_data_source_cache,
    disable_indicator_cache,
    disable_model_cache,
    enable_caches,
    enable_data_source_cache,
    enable_indicator_cache,
    enable_model_cache,
)
from pybroker.common import BarData, DataCol, Day, FeeMode, PriceType
from pybroker.context import ExecContext
from pybroker.config import StrategyConfig
from pybroker.data import Alpaca, AlpacaCrypto, YFinance
from pybroker.eval import EvalMetrics, BootstrapResult
from pybroker.indicator import (
    Indicator,
    IndicatorSet,
    highest,
    indicator,
    lowest,
)
from pybroker.model import ModelLoader, ModelSource, ModelTrainer, model
from pybroker.portfolio import Entry, Order, Position, Trade
from pybroker.scope import (
    disable_logging,
    enable_logging,
    param,
    register_columns,
    unregister_columns,
)
from pybroker.slippage import RandomSlippageModel
from pybroker.strategy import Strategy, TestResult
from pybroker.vect import cross, highv, lowv, sumv

# Temporary fix for regression in Numba 0.57.0
# https://github.com/numba/numba/issues/8940
from numba.np.unsafe import ndarray

__version__ = "1.1.24"
