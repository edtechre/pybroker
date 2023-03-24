"""Contains scopes that store data and object references used to execute a
:class:`pybroker.strategy.Strategy`.
"""

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

import numpy as np
import pandas as pd
from .common import (
    BarData,
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    PriceType,
    TrainedModel,
    to_decimal,
)
from .log import Logger
from collections import defaultdict
from decimal import Decimal
from diskcache import Cache
from numpy.typing import NDArray
from typing import (
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)


class StaticScope:
    """A static registry of data and object references.

    Attributes:
        logger: :class:`pybroker.log.Logger`
        data_source_cache: :class:`diskcache.Cache` that stores data retrieved
            from :class:`pybroker.data.DataSource`.
        data_source_cache_ns: Namespace set for  :attr:`.data_source_cache`.
        indicator_cache: :class:`diskcache.Cache` that stores
            :class:`pybroker.indicator.Indicator` data.
        indicator_cache_ns: Namespace set for :attr:`.indicator_cache`.
        model_cache: :class:`diskcache.Cache` that stores trained models.
        model_cache_ns: Namespace set for :attr:`.model_cache`.
        default_data_cols: Default data columns in :class:`pandas.DataFrame`
            retrieved from a :class:`pybroker.data.DataSource`.
        custom_data_cols: User-defined data columns in
            :class:`pandas.DataFrame` retrieved from a
            :class:`pybroker.data.DataSource`.
    """

    __instance = None

    def __init__(self):
        self.logger = Logger(self)
        self.data_source_cache: Optional[Cache] = None
        self.data_source_cache_ns: str = ""
        self.indicator_cache: Optional[Cache] = None
        self.indicator_cache_ns: str = ""
        self.model_cache: Optional[Cache] = None
        self.model_cache_ns: str = ""
        self._indicators = {}
        self._model_sources = {}
        self.default_data_cols = frozenset(
            (
                DataCol.DATE.value,
                DataCol.OPEN.value,
                DataCol.HIGH.value,
                DataCol.LOW.value,
                DataCol.CLOSE.value,
                DataCol.VOLUME.value,
                DataCol.VWAP.value,
            )
        )
        self.custom_data_cols = set()
        self._cols_frozen: bool = False

    def set_indicator(self, indicator):
        """Stores :class:`pybroker.indicator.Indicator` in static scope."""
        self._indicators[indicator.name] = indicator

    def has_indicator(self, name: str) -> bool:
        """Whether :class:`pybroker.indicator.Indicator` is stored in static
        scope.
        """
        return name in self._indicators

    def get_indicator(self, name: str):
        """Retrieves a :class:`pybroker.indicator.Indicator` from static
        scope."""
        if not self.has_indicator(name):
            raise ValueError(f"Indicator {name!r} does not exist.")
        return self._indicators[name]

    def get_indicator_names(self, model_name: str) -> tuple[str]:
        """Returns a ``tuple[str]`` of all
        :class:`pybroker.indicator.Indicator` names that are registered with
        :class:`pybroker.model.ModelSource` having ``model_name``.
        """
        return self._model_sources[model_name].indicators

    def set_model_source(self, source):
        """Stores :class:`pybroker.model.ModelSource` in static scope."""
        self._model_sources[source.name] = source

    def has_model_source(self, name: str) -> bool:
        """Whether :class:`pybroker.model.ModelSource` is stored in static
        scope.
        """
        return name in self._model_sources

    def get_model_source(self, name: str):
        """Retrieves a :class:`pybroker.model.ModelSource` from static
        scope.
        """
        if not self.has_model_source(name):
            raise ValueError(f"ModelSource {name!r} does not exist.")
        return self._model_sources[name]

    def register_custom_cols(self, names: Union[str, Iterable[str]], *args):
        """Registers user-defined column names."""
        self._verify_unfrozen_cols()
        if type(names) == str:
            names = (names, *args)
        else:
            names = (*names, *args)
        names = filter(lambda col: col not in self.default_data_cols, names)
        self.custom_data_cols.update(names)

    def unregister_custom_cols(self, names: Union[str, Iterable[str]], *args):
        """Unregisters user-defined column names."""
        self._verify_unfrozen_cols()
        if type(names) == str:
            names = (names, *args)
        else:
            names = (*names, *args)
        self.custom_data_cols.difference_update(names)

    @property
    def all_data_cols(self) -> frozenset[str]:
        """All registered data column names."""
        return self.default_data_cols | self.custom_data_cols

    def _verify_unfrozen_cols(self):
        if self._cols_frozen:
            raise ValueError("Cannot modify columns when strategy is running.")

    def freeze_data_cols(self):
        """Prevents additional data columns from being registered."""
        self._cols_frozen = True

    def unfreeze_data_cols(self):
        """Allows additional data columns to be registered if
        :func:`pybroker.scope.StaticScope.freeze_data_cols` was called.
        """
        self._cols_frozen = False

    @classmethod
    def instance(cls) -> "StaticScope":
        """Returns singleton instance."""
        if cls.__instance is None:
            cls.__instance = StaticScope()
        return cls.__instance


def disable_logging():
    """Disables event logging."""
    StaticScope.instance().logger.disable()


def enable_logging():
    """Enables event logging."""
    StaticScope.instance().logger.enable()


def disable_progress_bar():
    """Disables logging a progress bar."""
    StaticScope.instance().logger.disable_progress_bar()


def enable_progress_bar():
    """Enables logging a progress bar."""
    StaticScope.instance().logger.enable_progress_bar()


def register_columns(names: Union[str, Iterable[str]], *args):
    """Registers ``names`` of user-defined data columns."""
    StaticScope.instance().register_custom_cols(names, *args)


def unregister_columns(names: Union[str, Iterable[str]], *args):
    """Unregisters ``names`` of user-defined data columns."""
    StaticScope.instance().unregister_custom_cols(names, *args)


class ColumnScope:
    """Caches and retrieves column data queried from :class:`pandas.DataFrame`.

    Args:
        df: :class:`pandas.DataFrame` containing the column data.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.sort_index()
        self._symbols = frozenset(df.index.get_level_values(0).unique())
        self._sym_cols: dict[str, dict[str, Optional[NDArray]]] = defaultdict(
            dict
        )

    def fetch_dict(
        self,
        symbol: str,
        names: Collection[str],
        end_index: Optional[int] = None,
    ) -> dict[str, Optional[NDArray]]:
        r"""Fetches a ``dict`` of column data for ``symbol``.

        Args:
            symbol: Ticker symbol to query.
            names: Names of columns to query.
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.

        Returns:
            ``dict`` mapping column names to :class:`numpy.ndarray`\ s of
            column values.
        """
        result: dict[str, Optional[NDArray]] = {}
        if not names:
            return result
        sym_dfs: dict[str, pd.DataFrame] = {}
        for name in names:
            if symbol in self._sym_cols and name in self._sym_cols[symbol]:
                result[name] = self._sym_cols[symbol][name]
                if result[name] is not None:
                    result[name] = result[name][
                        :end_index
                    ]  # type: ignore[index]
                continue
            if symbol in sym_dfs:
                sym_df = sym_dfs[symbol]
            else:
                if symbol not in self._symbols:
                    raise ValueError(f"Symbol not found: {symbol}.")
                sym_df = self._df.loc[pd.IndexSlice[symbol, :]].reset_index()
                sym_dfs[symbol] = sym_df
            if name not in sym_df.columns:
                self._sym_cols[symbol][name] = None
                result[name] = None
                continue
            array = sym_df[name].to_numpy()
            self._sym_cols[symbol][name] = array
            result[name] = array[:end_index]
        return result

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> Optional[NDArray]:
        """Fetches a :class:`numpy.ndarray` of column data for ``symbol``.

        Args:
            symbol: Ticker symbol to query.
            name: Name of column to query.
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.

        Returns:
            :class:`numpy.ndarray` of column data for every bar until
            ``end_index`` (when specified).
        """
        result = self.fetch_dict(symbol, (name,), end_index)
        return result.get(name, None)

    def bar_data_from_data_columns(
        self, symbol: str, end_index: int
    ) -> BarData:
        """Returns a new :class:`pybroker.common.BarData` instance containing
        column data of default and custom data columns registered with
        :class:`.StaticScope`.

        Args:
            symbol: Ticker symbol to query.
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.
        """
        static_scope = StaticScope.instance()
        default_col_data = self.fetch_dict(
            symbol, static_scope.default_data_cols, end_index
        )
        custom_col_data = self.fetch_dict(
            symbol, static_scope.custom_data_cols, end_index
        )
        return BarData(
            **default_col_data,  # type: ignore[arg-type]
            **custom_col_data,  # type: ignore[arg-type]
        )


class IndicatorScope:
    """Caches and retrieves :class:`pybroker.indicator.Indicator` data.

    Args:
        indicator_data: :class:`Mapping` of
            :class:`pybroker.common.IndicatorSymbol` pairs to ``pandas.Series``
            of :class:`pybroker.indicator.Indicator` values.
        filter_dates: Filters :class:`pybroker.indicator.Indicator` data on
            :class:`Sequence` of dates.
    """

    def __init__(
        self,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        filter_dates: Sequence[np.datetime64],
    ):
        self._indicator_data = indicator_data
        self._filter_dates = filter_dates
        self._sym_inds: dict[IndicatorSymbol, NDArray[np.float_]] = {}

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> NDArray[np.float_]:
        """Fetches :class:`pybroker.indicator.Indicator` data.

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.indicator.Indicator` to query.
            end_index: Truncates the array of
                :class:`pybroker.indicator.Indicator` data returned
                (exclusive). If ``None``, then indicator data is not truncated.

        Returns:
            :class:`numpy.ndarray` of :class:`pybroker.indicator.Indicator`
            data for every bar until ``end_index`` (when specified).
        """
        ind_sym = IndicatorSymbol(name, symbol)
        if ind_sym in self._sym_inds:
            return self._sym_inds[ind_sym][:end_index]
        if ind_sym not in self._indicator_data:
            raise ValueError(f"Indicator {name!r} not found for {symbol}.")
        ind_series = self._indicator_data[ind_sym]
        ind_data = ind_series[ind_series.index.isin(self._filter_dates)].values
        self._sym_inds[ind_sym] = ind_data
        return ind_data[:end_index]


class ModelInputScope:
    """Caches and retrieves model input data.

    Args:
        col_scope: :class:`.ColumnScope`.
        ind_scope: :class:`.IndicatorScope`.
    """

    def __init__(self, col_scope: ColumnScope, ind_scope: IndicatorScope):
        self._col_scope = col_scope
        self._ind_scope = ind_scope
        self._sym_inputs: dict[ModelSymbol, pd.DataFrame] = {}
        self._scope = StaticScope.instance()

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetches model input data.

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.model.ModelSource` to query input
                data.
            end_index: Truncates the array of model input data returned
                (exclusive). If ``None``, then model input data is not
                truncated.

        Returns:
            :class:`numpy.ndarray` of model input data for every bar until
            ``end_index`` (when specified).
        """
        model_sym = ModelSymbol(name, symbol)
        if model_sym in self._sym_inputs:
            df = self._sym_inputs[model_sym]
            return df if end_index is None else df.loc[: end_index - 1]
        input_ = {}
        for col in self._scope.all_data_cols:
            data = self._col_scope.fetch(symbol, col)
            if data is not None:
                input_[col] = data
        if not self._scope.has_model_source(name):
            raise ValueError(f"Model {name!r} not found.")
        for ind_name in self._scope.get_indicator_names(name):
            input_[ind_name] = self._ind_scope.fetch(symbol, ind_name)
        df = pd.DataFrame.from_dict(input_)
        df = self._scope.get_model_source(name).prepare_input_data(df)
        self._sym_inputs[model_sym] = df
        return df if end_index is None else df.loc[: end_index - 1]


class PredictionScope:
    r"""Caches and retrieves model predictions.

    Args:
        models: :class:`Mapping` of
            :class:`pybroker.common.ModelSymbol` pairs to
            :class:`pybroker.common.TrainedModel`\ s.
        input_scope: :class:`.ModelInputScope`.
    """

    def __init__(
        self,
        models: Mapping[ModelSymbol, TrainedModel],
        input_scope: ModelInputScope,
    ):
        self._models = models
        self._input_scope = input_scope
        self._sym_preds: dict[ModelSymbol, NDArray] = {}

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> NDArray:
        """Fetches model predictions.

        Args:
            symbol: Ticker symbol to query.
            name: Name of :class:`pybroker.model.ModelSource` that made the
                predictions.
            end_index: Truncates the array of predictions returned (exclusive).
                If ``None``, then predictions are not truncated.

        Returns:
            :class:`numpy.ndarray` of model predictions for every bar until
            ``end_index`` (when specified).
        """
        model_sym = ModelSymbol(name, symbol)
        if model_sym in self._sym_preds:
            return self._sym_preds[model_sym][:end_index]
        input_ = self._input_scope.fetch(symbol, name)
        if model_sym not in self._models:
            raise ValueError(f"Model {name!r} not found for {symbol}.")
        trained_model = self._models[model_sym]
        if trained_model.predict_fn is not None:
            pred = trained_model.predict_fn(trained_model.instance, input_)
        else:
            predict_fn = getattr(trained_model.instance, "predict", None)
            if predict_fn is not None and callable(predict_fn):
                pred = trained_model.instance.predict(input_)
            else:
                raise ValueError(
                    f"Model instance trained for {model_sym.model_name!r} "
                    "does not define a predict function. Please pass a "
                    "predict_fn to pybroker.model()."
                )
        if len(pred.shape) > 1:
            pred = np.squeeze(pred)
        self._sym_preds[model_sym] = pred
        return pred[:end_index]


class PriceScope:
    """Retrieves most recent prices."""

    def __init__(
        self,
        col_scope: ColumnScope,
        sym_end_index: Mapping[str, int],
    ):
        self._col_scope = col_scope
        self._sym_end_index = sym_end_index

    def fetch(
        self,
        symbol: str,
        price: Union[
            float,
            int,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ],
    ) -> Decimal:
        end_index = self._sym_end_index[symbol]
        price_type = type(price)
        if price_type == PriceType:
            if price == PriceType.OPEN:
                open_ = self._col_scope.fetch(
                    symbol, DataCol.OPEN.value, end_index
                )
                if open_ is None:
                    raise ValueError("Open price not found.")
                return to_decimal(open_[-1])
            elif price == PriceType.HIGH:
                high = self._col_scope.fetch(
                    symbol, DataCol.HIGH.value, end_index
                )
                if high is None:
                    raise ValueError("High price not found.")
                return to_decimal(high[-1])
            elif price == PriceType.LOW:
                low = self._col_scope.fetch(
                    symbol, DataCol.LOW.value, end_index
                )
                if low is None:
                    raise ValueError("Low price not found.")
                return to_decimal(low[-1])
            elif price == PriceType.CLOSE:
                close = self._col_scope.fetch(
                    symbol, DataCol.CLOSE.value, end_index
                )
                if close is None:
                    raise ValueError("Close price not found.")
                return to_decimal(close[-1])
            elif price == PriceType.MIDDLE:
                low = self._col_scope.fetch(
                    symbol, DataCol.LOW.value, end_index
                )
                if low is None:
                    raise ValueError("Low price not found.")
                high = self._col_scope.fetch(
                    symbol, DataCol.HIGH.value, end_index
                )
                if high is None:
                    raise ValueError("High price not found.")
                return to_decimal(
                    round((low[-1] + (high[-1] - low[-1]) / 2.0), 2)
                )
            elif price == PriceType.AVERAGE:
                open_ = self._col_scope.fetch(
                    symbol, DataCol.OPEN.value, end_index
                )
                if open_ is None:
                    raise ValueError("Open price not found.")
                high = self._col_scope.fetch(
                    symbol, DataCol.HIGH.value, end_index
                )
                if high is None:
                    raise ValueError("High price not found.")
                low = self._col_scope.fetch(
                    symbol, DataCol.LOW.value, end_index
                )
                if low is None:
                    raise ValueError("Low price not found.")
                close = self._col_scope.fetch(
                    symbol, DataCol.CLOSE.value, end_index
                )
                if close is None:
                    raise ValueError("Close price not found.")
                return to_decimal(
                    round(
                        (open_[-1] + low[-1] + high[-1] + close[-1]) / 4.0, 2
                    )
                )
            else:
                raise ValueError(f"Unknown price: {price_type}")
        elif price_type == float or price_type == int or price_type == Decimal:
            return to_decimal(price)  # type: ignore[arg-type]
        elif callable(price):
            bar_data = self._col_scope.bar_data_from_data_columns(
                symbol, self._sym_end_index[symbol]
            )
            return to_decimal(price(symbol, bar_data))
        else:
            raise ValueError(f"Unknown price: {price_type}")


class PendingOrder(NamedTuple):
    """Holds data for a pending order.

    Attributes:
        id: Unique ID.
        type: Type of order, either ``buy`` or ``sell``.
        symbol: Ticker symbol of the order.
        created: Date the order was created.
        exec_date: Date the order will be executed.
        shares: Number of shares to be bought or sold.
        limit_price: Limit price to use for the order.
        fill_price: Price that the order will be filled at.
    """

    id: int
    type: Literal["buy", "sell"]
    symbol: str
    created: np.datetime64
    exec_date: np.datetime64
    shares: Decimal
    limit_price: Optional[Decimal]
    fill_price: Union[
        int,
        float,
        Decimal,
        PriceType,
        Callable[[str, BarData], Union[int, float, Decimal]],
    ]


class PendingOrderScope:
    r"""Stores :class:`.PendingOrder`\ s"""

    _order_id: int = 0

    def __init__(self):
        self._orders: dict[int, PendingOrder] = {}
        self._sym_orders: dict[str, set[PendingOrder]] = defaultdict(set)

    def contains(self, order_id: int) -> bool:
        """Returns whether a :class:`.PendingOrder` exists with
        ``order_id``.
        """
        return order_id in self._orders

    def add(
        self,
        type: Literal["buy", "sell"],
        symbol: str,
        created: np.datetime64,
        exec_date: np.datetime64,
        shares: Decimal,
        limit_price: Optional[Decimal],
        fill_price: Union[
            int,
            float,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ],
    ) -> int:
        """Creates a :class:`.PendingOrder`.

        Args:
            type: Type of order, either ``buy`` or ``sell``.
            symbol: Ticker symbol of the order.
            created: Date the order was created.
            exec_date: Date the order will be executed.
            shares: Number of shares to be bought or sold.
            limit_price: Limit price to use for the order.
            fill_price: Price that the order will be filled at.

        Returns:
            ID of the :class:`.PendingOrder`.
        """
        self._order_id += 1
        order = PendingOrder(
            id=self._order_id,
            type=type,
            symbol=symbol,
            created=created,
            exec_date=exec_date,
            shares=shares,
            limit_price=limit_price,
            fill_price=fill_price,
        )
        self._orders[self._order_id] = order
        self._sym_orders[symbol].add(order)
        return order.id

    def remove(self, order_id: int) -> bool:
        """Removes a :class:`.PendingOrder` with ``order_id```."""
        if order_id in self._orders:
            order = self._orders[order_id]
            del self._orders[order_id]
            if (
                order.symbol in self._sym_orders
                and order in self._sym_orders[order.symbol]
            ):
                self._sym_orders[order.symbol].remove(order)
            return True
        return False

    def remove_all(self, symbol: Optional[str] = None):
        r"""Removes all :class:`.PendingOrder`\ s."""
        if symbol is None:
            cancel_ids = tuple(self._orders.keys())
            for order_id in cancel_ids:
                self.remove(order_id)
        elif symbol in self._sym_orders:
            cancel_ids = tuple(order.id for order in self._sym_orders[symbol])
            for order_id in cancel_ids:
                self.remove(order_id)

    def orders(self, symbol: Optional[str] = None) -> Iterable[PendingOrder]:
        r"""Returns an :class:`Iterable` of :class:`.PendingOrder`\ s."""
        if symbol is None:
            return self._orders.values()
        else:
            if symbol not in self._sym_orders:
                return []
            return self._sym_orders[symbol]
