"""Contains context related classes. A context provides data during the
execution of a :class:`pybroker.strategy.Strategy`."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
from pybroker.common import (
    BarData,
    DataCol,
    ModelSymbol,
    PriceType,
    StopType,
    to_datetime,
    to_decimal,
)
from pybroker.config import StrategyConfig
from pybroker.model import TrainedModel
from pybroker.portfolio import Entry, Order, Portfolio, Position, Stop, Trade
from pybroker.scope import (
    ColumnScope,
    IndicatorScope,
    ModelInputScope,
    PendingOrder,
    PendingOrderScope,
    PredictionScope,
    StaticScope,
    TimeframeScope,
)
from pybroker.timeframe import TimeframeInterval, normalize_timeframe_interval
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Union,
)


@dataclass
class ExecResult:
    r"""Holds data that was set during the execution of a
    :class:`pybroker.strategy.Strategy`.

    Attributes:
        symbol: Ticker symbol that was used for the execution.
        date: Timestamp of the bar that was used for the execution.
        buy_fill_price: Fill price to use for a buy (long) order of ``symbol``.
        sell_fill_price: Fill price to use for a sell (short) order of
            ``symbol``.
        score: Score used to rank ``symbol`` when ranking long and short
            signals. Orders are placed for symbols with the highest scores,
            where the number of positions held at any time in the
            :class:`pybroker.portfolio.Portfolio` is specified by
            :attr:`pybroker.config.StrategyConfig.max_long_positions` and
            :attr:`pybroker.config.StrategyConfig.max_short_positions`
            respectively. Buy and sell signals are ranked separately by
            ``score``. Mutually exclusive with ``long_score`` and
            ``short_score``.
        long_score: Score used to rank ``symbol`` when ranking buy and cover
            signals. Orders are placed for symbols with the highest
            ``long_score`` values, where the number of long positions held at
            any time in the :class:`pybroker.portfolio.Portfolio` is specified
            by :attr:`pybroker.config.StrategyConfig.max_long_positions`.
            When :attr:`pybroker.config.StrategyConfig.worst_rank_held` is set,
            ``long_score`` is used for long rotational hold-band logic.
            Mutually exclusive with ``score``.
        short_score: Score used to rank ``symbol`` when ranking sell signals.
            Orders are placed for symbols with the lowest ``short_score``
            values, where the number of short positions held at any time in
            the :class:`pybroker.portfolio.Portfolio` is specified by
            :attr:`pybroker.config.StrategyConfig.max_short_positions`.
            When :attr:`pybroker.config.StrategyConfig.worst_rank_held` is set,
            ``short_score`` is used for short rotational hold-band logic.
            Mutually exclusive with ``score``.
        hold_bars: Number of bars to hold a long or short position for, after
            which the position is automatically liquidated.
        buy_shares: Number of shares to buy of ``symbol``.
        buy_limit_price: Limit price used for a buy (long) order of ``symbol``.
        sell_shares: Number of shares to sell of ``symbol``.
        sell_limit_price: Limit price used for a sell (short) order of
            ``symbol``.
        long_stops: Stops for long :class:`pybroker.portfolio.Entry`\ s.
        short_stops: Stops for short :class:`pybroker.portfolio.Entry`\ s.
        cover: Whether ``buy_shares`` are used to cover a short position. If
            ``True``, the resulting buy order will be placed before sell
            orders.
        pending_order_id: ID of :class:`pybroker.scope.PendingOrder` that was
            created.
    """

    symbol: str
    date: np.datetime64
    buy_fill_price: Union[
        int,
        float,
        np.floating,
        Decimal,
        PriceType,
        Callable[[str, BarData], Union[int, float, Decimal]],
    ]
    sell_fill_price: Union[
        int,
        float,
        np.floating,
        Decimal,
        PriceType,
        Callable[[str, BarData], Union[int, float, Decimal]],
    ]
    score: Optional[float]
    long_score: Optional[float]
    short_score: Optional[float]
    hold_bars: Optional[int]
    buy_shares: Optional[Decimal]
    buy_limit_price: Optional[Decimal]
    sell_shares: Optional[Decimal]
    sell_limit_price: Optional[Decimal]
    long_stops: Optional[frozenset[Stop]]
    short_stops: Optional[frozenset[Stop]]
    cover: bool = field(default=False)
    pending_order_id: Optional[int] = field(default=None)


class TimeframeContext:
    """Read-only view of compressed bar data for a higher timeframe."""

    _symbol: str
    _interval: TimeframeInterval
    _timeframe_scope: TimeframeScope
    _sym_end_index: Mapping[str, int]
    _models: Mapping[ModelSymbol, TrainedModel]
    _scope: StaticScope

    _READ_ONLY_ATTRS: frozenset[str] = frozenset(
        {
            "buy_fill_price",
            "buy_shares",
            "buy_limit_price",
            "sell_fill_price",
            "sell_shares",
            "sell_limit_price",
            "hold_bars",
            "score",
            "long_score",
            "short_score",
            "stop_loss",
            "stop_loss_pct",
            "stop_loss_limit",
            "stop_loss_exit_price",
            "stop_profit",
            "stop_profit_pct",
            "stop_profit_limit",
            "stop_profit_exit_price",
            "stop_trailing",
            "stop_trailing_pct",
            "stop_trailing_limit",
            "stop_trailing_exit_price",
        }
    )

    def __init__(
        self,
        symbol: str,
        interval: TimeframeInterval,
        timeframe_scope: TimeframeScope,
        sym_end_index: Mapping[str, int],
        models: Mapping[ModelSymbol, TrainedModel],
    ):
        object.__setattr__(self, "_symbol", symbol)
        object.__setattr__(
            self, "_interval", normalize_timeframe_interval(interval)
        )
        object.__setattr__(self, "_timeframe_scope", timeframe_scope)
        object.__setattr__(self, "_sym_end_index", sym_end_index)
        object.__setattr__(self, "_models", models)
        object.__setattr__(self, "_scope", StaticScope.instance())

    @property
    def bars(self) -> int:
        end_index = self._sym_end_index[self._symbol]
        data = self._timeframe_scope.fetch_bar(
            self._symbol,
            self._interval,
            DataCol.CLOSE.value,
            end_index,
        )
        return len(data)

    @property
    def dates(self) -> NDArray[np.datetime64]:
        return self._fetch(DataCol.DATE.value)

    @property
    def open(self) -> NDArray[np.float64]:
        return self._fetch(DataCol.OPEN.value)

    @property
    def high(self) -> NDArray[np.float64]:
        return self._fetch(DataCol.HIGH.value)

    @property
    def low(self) -> NDArray[np.float64]:
        return self._fetch(DataCol.LOW.value)

    @property
    def close(self) -> NDArray[np.float64]:
        return self._fetch(DataCol.CLOSE.value)

    @property
    def volume(self) -> NDArray[np.float64]:
        return self._fetch(DataCol.VOLUME.value)

    def indicator(self, name: str) -> NDArray[np.float64]:
        """Returns indicator values on the compressed timeframe."""
        end_index = self._sym_end_index[self._symbol]
        return self._timeframe_scope.fetch_indicator(
            self._symbol, self._interval, name, end_index
        )

    def model(self, name: str) -> Any:
        """Returns a trained model on the compressed timeframe."""
        from pybroker.timeframe import model_timeframe_name

        model_sym = ModelSymbol(
            model_timeframe_name(name, self._interval), self._symbol
        )
        if model_sym not in self._models:
            raise ValueError(f"Model {name!r} not found for {self._symbol}.")
        return self._models[model_sym].instance

    def input(self, model_name: str) -> pd.DataFrame:
        """Returns model input data on the compressed timeframe."""
        end_index = self._sym_end_index[self._symbol]
        return self._timeframe_scope.fetch_input(
            self._symbol, self._interval, model_name, end_index
        )

    def preds(self, model_name: str) -> NDArray:
        """Returns model predictions on the compressed timeframe."""
        end_index = self._sym_end_index[self._symbol]
        return self._timeframe_scope.fetch_preds(
            self._symbol, self._interval, model_name, end_index
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._READ_ONLY_ATTRS:
            raise AttributeError(
                f"TimeframeContext is read-only; set {name!r} on the base "
                "ExecContext instead."
            )
        object.__setattr__(self, name, value)

    def _fetch(self, col: str) -> NDArray:
        end_index = self._sym_end_index[self._symbol]
        return self._timeframe_scope.fetch_bar(
            self._symbol, self._interval, col, end_index
        )

    def __getattr__(self, attr: str) -> NDArray:
        if attr in self._scope.custom_data_cols:
            end_index = self._sym_end_index[self._symbol]
            return self._timeframe_scope.fetch_bar(
                self._symbol, self._interval, attr, end_index
            )
        raise AttributeError(f"Attribute {attr!r} not found.")


class ExecContext:
    r"""Contains context data during the execution of a
    :class:`pybroker.strategy.Strategy`. Includes data about the current bar,
    portfolio positions, and other relevant context. This class is also used to
    set buy and sell signals for placing orders.

    The data contained in this class is for the latest bar that has already
    completed. Placing an order will be executed on a future bar specified by
    :attr:`pybroker.config.StrategyConfig.buy_delay` and
    :attr:`pybroker.config.StrategyConfig.sell_delay`.

    Attributes:
        config: :class:`pybroker.config.StrategyConfig`.
        symbol: Current ticker symbol of the execution.
        buy_fill_price: Fill price to use for a buy (long) order of
            ``symbol``.
        buy_shares: Number of shares to buy of ``symbol``.
        buy_limit_price: Limit price to use for a buy (long) order of
            ``symbol``.
        sell_fill_price: Fill price to use for a sell (short) order of
            ``symbol``.
        sell_shares: Number of shares to sell of ``symbol``.
        sell_limit_price: Limit price to use for a sell (short) order of
            ``symbol``.
        hold_bars: Number of bars to hold a long or short position for, after
            which the position is automatically liquidated.
        score: Score used to rank ``symbol`` when ranking buy and sell signals.
            Orders are placed for symbols with the highest scores, where the
            number of positions held at any time in the
            :class:`pybroker.portfolio.Portfolio` is specified by
            :attr:`pybroker.config.StrategyConfig.max_long_positions` and
            :attr:`pybroker.config.StrategyConfig.max_short_positions`
            respectively. Long and short signals are ranked separately by
            ``score``. Mutually exclusive with ``long_score`` and
            ``short_score``.
        long_score: Score used to rank ``symbol`` when ranking buy and cover
            signals. Orders are placed for symbols with the highest
            ``long_score`` values, where the number of long positions held at
            any time in the :class:`pybroker.portfolio.Portfolio` is specified
            by :attr:`pybroker.config.StrategyConfig.max_long_positions`.
            When :attr:`pybroker.config.StrategyConfig.worst_rank_held` is set,
            ``long_score`` is used for long rotational hold-band logic.
            Mutually exclusive with ``score``.
        short_score: Score used to rank ``symbol`` when ranking sell signals.
            Orders are placed for symbols with the lowest ``short_score``
            values, where the number of short positions held at any time in
            the :class:`pybroker.portfolio.Portfolio` is specified by
            :attr:`pybroker.config.StrategyConfig.max_short_positions`.
            When :attr:`pybroker.config.StrategyConfig.worst_rank_held` is set,
            ``short_score`` is used for short rotational hold-band logic.
            Mutually exclusive with ``score``.
        session: ``dict`` used to store custom data that persists for each
            bar during the :class:`pybroker.strategy.Strategy`\ 's execution.
        stop_loss: Sets stop loss on a new :class:`pybroker.portfolio.Entry`,
            where value is measured in points from entry price.
        stop_loss_pct: Sets stop loss on a new
            :class:`pybroker.portfolio.Entry`, where value is measured in
            percentage from entry price.
        stop_loss_limit: Limit price to use for the stop loss.
        stop_loss_exit_price: Exit :class:`pybroker.common.PriceType` to use
            for the stop loss exit. If set, the stop is checked against the
            ``exit_price`` and exits at the ``exit_price`` when triggered.
        stop_profit: Sets profit stop on a new
            :class:`pybroker.portfolio.Entry`, where value is measured in
            points from entry price.
        stop_profit_pct: Sets profit stop on a new
            :class:`pybroker.portfolio.Entry`, where value is measured in
            percentage from entry price.
        stop_profit_limit: Limit price to use for the profit stop.
        stop_profit_exit_price: Exit :class:`pybroker.common.PriceType` to use
            for the profit stop exit. If set, the stop is checked against the
            ``exit_price`` and exits at the ``exit_price`` when triggered.
        stop_trailing: Sets a trailing stop loss on a new
            :class:`pybroker.portfolio.Entry`, where value is measured in
            points from entry price.
        stop_trailing_pct: Sets a trailing stop loss on a new
            :class:`pybroker.portfolio.Entry`, where value is measured in
            percentage from entry price.
        stop_trailing_limit: Limit price to use for the trailing stop loss.
        stop_trailing_exit_price: Exit :class:`pybroker.common.PriceType` to
            use for the trailing stop exit. If set, the stop is checked against
            the ``exit_price`` and exits at the ``exit_price`` when triggered.
    """

    _stop_id: int = 0

    def __init__(
        self,
        symbol: str,
        config: StrategyConfig,
        portfolio: Portfolio,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        timeframe_scope: TimeframeScope,
        declared_timeframes: frozenset[TimeframeInterval],
        input_scope: ModelInputScope,
        pred_scope: PredictionScope,
        pending_order_scope: PendingOrderScope,
        models: Mapping[ModelSymbol, TrainedModel],
        sym_end_index: Mapping[str, int],
        session: MutableMapping,
    ):
        self.config = config
        self._portfolio = portfolio
        self._col_scope = col_scope
        self._ind_scope = ind_scope
        self._timeframe_scope = timeframe_scope
        self._declared_timeframes = declared_timeframes
        self._input_scope = input_scope
        self._pred_scope = pred_scope
        self._models = models
        self._sym_end_index = sym_end_index
        self._pending_order_scope = pending_order_scope
        self._scope = StaticScope.instance()
        self._curr_date: Optional[np.datetime64] = None
        self._dt: Optional[datetime] = None
        self._foreign: dict[str, pd.DataFrame] = {}
        self._timeframe: dict[TimeframeInterval, TimeframeContext] = {}

        self.symbol: str = symbol
        self.buy_fill_price: Optional[
            Union[
                int,
                float,
                np.floating,
                Decimal,
                PriceType,
                Callable[[str, BarData], Union[int, float, Decimal]],
            ]
        ] = None
        self.buy_shares: Optional[Union[int, float, Decimal]] = None
        self.buy_limit_price: Optional[Union[int, float, Decimal]] = None
        self.sell_fill_price: Optional[
            Union[
                int,
                float,
                np.floating,
                Decimal,
                PriceType,
                Callable[[str, BarData], Union[int, float, Decimal]],
            ]
        ] = None
        self.sell_shares: Optional[Union[int, float, Decimal]] = None
        self.sell_limit_price: Optional[Union[int, float, Decimal]] = None
        self.hold_bars: Optional[int] = None
        self.score: Optional[float] = None
        self.long_score: Optional[float] = None
        self.short_score: Optional[float] = None
        self.session = session

        self.stop_loss: Optional[Union[int, float, Decimal]] = None
        self.stop_loss_pct: Optional[Union[int, float, Decimal]] = None
        self.stop_loss_limit: Optional[Union[int, float, Decimal]] = None
        self.stop_loss_exit_price: Optional[PriceType] = None
        self.stop_profit: Optional[Union[int, float, Decimal]] = None
        self.stop_profit_pct: Optional[Union[int, float, Decimal]] = None
        self.stop_profit_limit: Optional[Union[int, float, Decimal]] = None
        self.stop_profit_exit_price: Optional[PriceType] = None
        self.stop_trailing: Optional[Union[int, float, Decimal]] = None
        self.stop_trailing_pct: Optional[Union[int, float, Decimal]] = None
        self.stop_trailing_limit: Optional[Union[int, float, Decimal]] = None
        self.stop_trailing_exit_price: Optional[PriceType] = None

        self._cover: bool = False
        self._exiting_pos: bool = False

    @property
    def total_equity(self) -> Decimal:
        """Total equity currently held in the
        :class:`pybroker.portfolio.Portfolio`.
        """
        return self._portfolio.equity

    @property
    def cash(self) -> Decimal:
        """Total cash currently held in the
        :class:`pybroker.portfolio.Portfolio`.
        """
        return self._portfolio.cash

    @property
    def total_margin(self) -> Decimal:
        """Total amount of margin currently held in the
        :class:`pybroker.portfolio.Portfolio`.
        """
        return self._portfolio.margin

    @property
    def total_market_value(self) -> Decimal:
        """Total market value currently held in the
        :class:`pybroker.portfolio.Portfolio`. The market value is defined as
        the amount of equity held in cash and long positions added together
        with the unrealized PnL of all open short positions.
        """
        return self._portfolio.market_value

    @property
    def win_rate(self) -> Decimal:
        """Running win rate of trades."""
        return self._portfolio.win_rate

    @property
    def loss_rate(self) -> Decimal:
        """Running loss rate of trades."""
        return self._portfolio.loss_rate

    def orders(self) -> Iterator[Order]:
        r""":class:`Iterator` of all :class:`pybroker.portfolio.Order`\ s that
        have been placed and filled.
        """
        for order in self._portfolio.orders:
            yield order

    def pending_orders(
        self, symbol: Optional[str] = None
    ) -> Iterator[PendingOrder]:
        for order in self._pending_order_scope.orders(symbol):
            yield order

    def trades(self) -> Iterator[Trade]:
        r""":class:`Iterator` of all :class:`pybroker.portfolio.Trade`\ s that
        have been completed.
        """
        for trade in self._portfolio.trades:
            yield trade

    def pos(
        self,
        symbol: str,
        pos_type: Literal["long", "short"],
    ) -> Optional[Position]:
        r"""Retrieves a current long or short
        :class:`pybroker.portfolio.Position` for a ``symbol``.

        Args:
            symbol: Ticker symbol of the position to return.
            pos_type: Specifies whether to return a ``long`` or ``short``
                position.

        Returns:
            :class:`pybroker.portfolio.Position` if one exists, otherwise
            ``None``.
        """
        self._verify_pos_type(pos_type)
        if pos_type == "long" and symbol in self._portfolio.long_positions:
            return self._portfolio.long_positions[symbol]
        elif pos_type == "short" and symbol in self._portfolio.short_positions:
            return self._portfolio.short_positions[symbol]
        return None

    def positions(
        self,
        symbol: Optional[str] = None,
        pos_type: Optional[Literal["long", "short"]] = None,
    ) -> Iterator[Position]:
        r"""Retrieves all current positions.

        Args:
            symbol: Ticker symbol used to filter positions. If ``None``,
                positions for all symbols are returned. Defaults to ``None``.
            pos_type: Type of positions to return. If ``None``, both ``long``
                and ``short`` positions are returned.

        Returns:
            :class:`Iterator` of currently held
            :class:`pybroker.portfolio.Position` \s.
        """
        if pos_type is not None:
            self._verify_pos_type(pos_type)
        if symbol is None:
            if pos_type != "short":
                for pos in self._portfolio.long_positions.values():
                    yield pos
            if pos_type != "long":
                for pos in self._portfolio.short_positions.values():
                    yield pos
        else:
            if (
                pos_type != "short"
                and symbol in self._portfolio.long_positions
            ):
                yield self._portfolio.long_positions[symbol]
            if (
                pos_type != "long"
                and symbol in self._portfolio.short_positions
            ):
                yield self._portfolio.short_positions[symbol]

    def long_positions(
        self, symbol: Optional[str] = None
    ) -> Iterator[Position]:
        r"""Retrieves all current long positions.

        Args:
            symbol: Ticker symbol used to filter positions. If ``None``,
                long positions for all symbols are returned. Defaults to
                ``None``.

        Returns:
            :class:`Iterator` of currently held long
            :class:`pybroker.portfolio.Position` \s.
        """
        return self.positions(symbol, "long")

    def short_positions(
        self, symbol: Optional[str] = None
    ) -> Iterator[Position]:
        r"""Retrieves all current short positions.

        Args:
            symbol: Ticker symbol used to filter positions. If ``None``,
                short positions for all symbols are returned. Defaults to
                ``None``.

        Returns:
            :class:`Iterator` of currently held short
            :class:`pybroker.portfolio.Position` \s.
        """
        return self.positions(symbol, "short")

    def _verify_pos_type(self, pos_type: str):
        if pos_type != "short" and pos_type != "long":
            raise ValueError(f"Unknown pos_type: {pos_type!r}.")

    def _verify_symbol(self):
        if self.symbol is None:
            raise ValueError("symbol is not set.")

    @property
    def bars(self) -> int:
        """Number of bars of data that have completed."""
        return self._sym_end_index[self.symbol]

    @property
    def dt(self) -> datetime:
        """Current bar's date expressed as a ``datetime``."""
        if self._curr_date is None:
            raise ValueError("_curr_date is not set.")
        if self._dt is None:
            self._dt = to_datetime(self._curr_date)
        return self._dt

    @property
    def date(self) -> NDArray[np.datetime64]:
        """Current bar's date expressed as a ``numpy.datetime64``."""
        self._verify_symbol()
        return self._col_scope.fetch(  # type: ignore[return-value]
            self.symbol,  # type: ignore[arg-type]
            DataCol.DATE.value,
            self._sym_end_index[self.symbol],
        )

    @property
    def open(self) -> NDArray[np.float64]:
        """Current bar's open price."""
        self._verify_symbol()
        return self._col_scope.fetch(  # type: ignore[return-value]
            self.symbol,  # type: ignore[arg-type]
            DataCol.OPEN.value,
            self._sym_end_index[self.symbol],
        )

    @property
    def high(self) -> NDArray[np.float64]:
        """Current bar's high price."""
        self._verify_symbol()
        return self._col_scope.fetch(  # type: ignore[return-value]
            self.symbol,  # type: ignore[arg-type]
            DataCol.HIGH.value,
            self._sym_end_index[self.symbol],
        )

    @property
    def low(self) -> NDArray[np.float64]:
        """Current bar's low price."""
        self._verify_symbol()
        return self._col_scope.fetch(  # type: ignore[return-value]
            self.symbol,  # type: ignore[arg-type]
            DataCol.LOW.value,
            self._sym_end_index[self.symbol],
        )

    @property
    def close(self) -> NDArray[np.float64]:
        """Current bar's close price."""
        self._verify_symbol()
        return self._col_scope.fetch(  # type: ignore[return-value]
            self.symbol,  # type: ignore[arg-type]
            DataCol.CLOSE.value,
            self._sym_end_index[self.symbol],
        )

    @property
    def volume(self) -> Optional[NDArray[np.float64]]:
        """Current bar's volume."""
        self._verify_symbol()
        return self._col_scope.fetch(  # type: ignore[return-value]
            self.symbol,  # type: ignore[arg-type]
            DataCol.VOLUME.value,
            self._sym_end_index[self.symbol],
        )

    @property
    def vwap(self) -> Optional[NDArray[np.float64]]:
        """Current bar's volume-weighted average price (VWAP)."""
        self._verify_symbol()
        return self._col_scope.fetch(  # type: ignore[return-value]
            self.symbol,  # type: ignore[arg-type]
            DataCol.VWAP.value,
            self._sym_end_index[self.symbol],
        )

    @property
    def cover_fill_price(
        self,
    ) -> Optional[
        Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ]
    ]:
        """Alias for :attr:`.buy_fill_price`. When set, this causes the buy
        order to be placed before any sell orders.
        """
        return self.buy_fill_price

    @cover_fill_price.setter
    def cover_fill_price(
        self,
        fill_price: Optional[
            Union[
                int,
                float,
                np.floating,
                Decimal,
                PriceType,
                Callable[[str, BarData], Union[int, float, Decimal]],
            ]
        ],
    ):
        self.buy_fill_price = fill_price
        self._cover = True

    @property
    def cover_shares(self) -> Optional[Union[int, float, Decimal]]:
        """Alias for :attr:`.buy_shares`. When set, this causes the buy
        order to be placed before any sell orders.
        """
        return self.buy_shares

    @cover_shares.setter
    def cover_shares(self, shares: Optional[Union[int, float, Decimal]]):
        self.buy_shares = shares
        self._cover = True

    @property
    def cover_limit_price(self) -> Optional[Union[int, float, Decimal]]:
        """Alias for :attr:`.buy_limit_price`. When set, this causes the buy
        order to be placed before any sell orders.
        """
        return self.buy_limit_price

    @cover_limit_price.setter
    def cover_limit_price(
        self, limit_price: Optional[Union[int, float, Decimal]]
    ):
        self.buy_limit_price = limit_price
        self._cover = True

    def sell_all_shares(self):
        """Sells all long shares of :attr:`.ExecContext.symbol`."""
        pos = self.long_pos()
        if pos is None:
            raise ValueError(
                f"sell_all_shares failed: No long position for {self.symbol}"
            )
        self.sell_shares = pos.shares
        self._portfolio.remove_stops(pos)
        self._exiting_pos = True

    def cover_all_shares(self):
        """Covers all short shares of :attr:`.ExecContext.symbol`."""
        pos = self.short_pos()
        if pos is None:
            raise ValueError(
                f"cover_all_shares failed: No short position for {self.symbol}"
            )
        self.cover_shares = pos.shares
        self._portfolio.remove_stops(pos)
        self._exiting_pos = True

    def foreign(
        self, symbol: str, col: Optional[str] = None
    ) -> Union[BarData, Optional[NDArray]]:
        """Retrieves bar data for another ticker symbol.

        Args:
            symbol: Ticker symbol of the bar data.
            col: Name of the data column to retrieve. If ``None``, all data
                columns are returned in :class:`pybroker.common.BarData`.

        Returns:
            If ``col`` is ``None``, a :class:`pybroker.common.BarData`
            instance containing data of all bars up to the current one.
            Otherwise, an :class:`numpy.ndarray` containing values of the
            column ``col``.
        """
        if symbol in self._foreign:
            return self._foreign[symbol]
        if symbol not in self._sym_end_index:
            raise ValueError(f"Symbol {symbol!r} not found.")
        end_index = self._sym_end_index[symbol]
        if col is None:
            bar_data = self._col_scope.bar_data_from_data_columns(
                symbol, end_index
            )
            self._foreign[symbol] = bar_data
            return bar_data
        else:
            return self._col_scope.fetch(symbol, col, end_index)

    def timeframe(self, interval: TimeframeInterval) -> TimeframeContext:
        r"""Returns a read-only view of compressed bar data for ``interval``.

        ``interval`` must match a value previously passed to
        :meth:`pybroker.strategy.Strategy.set_timeframes`. The same
        :class:`~pybroker.TimeframeInterval` forms are supported:

        - **Every-n-bars** (``int``): e.g. ``ctx.timeframe(5)`` for bars
          formed from every 5 base bars.

        - **Duration** (``str``): e.g. ``ctx.timeframe("5m")`` for
          fixed-duration bins (digits plus unit letter).

        - **Calendar** (``str``): e.g. ``ctx.timeframe("weekly")`` for
          calendar weekly bars.

        For example::

            strategy.set_timeframes("weekly", "5m")

            def exec_fn(ctx):
                weekly = ctx.timeframe("weekly")
                five_min = ctx.timeframe("5m")
                if len(weekly.close) > 0:
                    wk_sma = weekly.indicator("sma20")

        Args:
            interval: Compression interval declared with
                :meth:`pybroker.strategy.Strategy.set_timeframes`.

        Returns:
            :class:`pybroker.context.TimeframeContext` exposing read-only
            OHLCV, indicators, and model outputs on the compressed bars.
        """
        interval = normalize_timeframe_interval(interval)
        if interval not in self._declared_timeframes:
            raise ValueError(
                f"Timeframe {interval!r} was not declared with set_timeframes()."
            )
        if interval not in self._timeframe:
            self._timeframe[interval] = TimeframeContext(
                symbol=self.symbol,
                interval=interval,
                timeframe_scope=self._timeframe_scope,
                sym_end_index=self._sym_end_index,
                models=self._models,
            )
        return self._timeframe[interval]

    def model(self, name: str, symbol: Optional[str] = None) -> Any:
        r"""Returns a trained model.

        Args:
            name: Name used to identify the model that was registered with
                :meth:`pybroker.model.model`.
            symbol: Ticker symbol of the data that was used to train the model.
                If ``None``, the ``ExecContext``\ 's :attr:`.symbol` is used.

        Returns:
            Instance of the trained model.
        """
        symbol = self._get_symbol(symbol)
        model_sym = ModelSymbol(name, symbol)
        if model_sym not in self._models:
            raise ValueError(f"Model {name!r} not found for {symbol}.")
        return self._models[model_sym].instance

    def indicator(
        self, name: str, symbol: Optional[str] = None
    ) -> NDArray[np.float64]:
        r"""Returns indicator data.

        Args:
            name: Name used to identify the indicator, registered with
                :meth:`pybroker.indicator.indicator`.
            symbol: Ticker symbol that was used to generate the indicator data.
                If ``None``, the ``ExecContext``\ 's :attr:`.symbol` is used.

        Returns:
            :class:`numpy.ndarray` of indicator values for all bars up to the
            current one, sorted in ascending chronological order.
        """
        symbol = self._get_symbol(symbol)
        end_index = self._sym_end_index[symbol]
        return self._ind_scope.fetch(symbol, name, end_index)

    def input(
        self, model_name: str, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        r"""Returns model input data for making predictions.

        Args:
            model_name: Name of the model for the input data.
            symbol: Ticker symbol of the model for the input data. If ``None``,
                the ``ExecContext``\ 's :attr:`.symbol` is used.

        Returns:
            :class:`pandas.DataFrame` containing the input data, where each row
            represents a bar in the sequence up to the current bar. The rows
            are sorted in ascending chronological order.
        """
        symbol = self._get_symbol(symbol)
        end_index = self._sym_end_index[symbol]
        return self._input_scope.fetch(symbol, model_name, end_index)

    def preds(self, model_name: str, symbol: Optional[str] = None) -> NDArray:
        r"""Returns model predictions.

        Args:
            model_name: Name of the model that made the predictions.
            symbol: Ticker symbol of the model that made the predictions. If
                ``None``, the ``ExecContext``\ 's :attr:`.symbol` is used.

        Returns:
            :class:`numpy.ndarray` containing the sequence of model predictions
            up to the current bar. Sorted in ascending chronological order.
        """
        symbol = self._get_symbol(symbol)
        end_index = self._sym_end_index[symbol]
        return self._pred_scope.fetch(symbol, model_name, end_index)

    def long_pos(
        self,
        symbol: Optional[str] = None,
    ) -> Optional[Position]:
        r"""Retrieves a current long :class:`pybroker.portfolio.Position` for a
        ``symbol``.

        Args:
            symbol: Ticker symbol of the position to return. If ``None``,
                the ``ExecContext``\ 's :attr:`.symbol` is used. Defaults to
                ``None``.

        Returns:
            :class:`pybroker.portfolio.Position` if one exists, otherwise
            ``None``.
        """
        # Fast path inlined — called per bar per active symbol (~7k calls
        # per V0 bench). Skips _get_symbol + super().pos + _verify_pos_type
        # delegation and collapses the dict membership+lookup into one
        # dict.get().
        if symbol is None:
            symbol = self.symbol
            if symbol is None:
                raise ValueError("symbol is not set.")
        return self._portfolio.long_positions.get(symbol)

    def short_pos(
        self,
        symbol: Optional[str] = None,
    ) -> Optional[Position]:
        r"""Retrieves a current short :class:`pybroker.portfolio.Position` for
        a ``symbol``.

        Args:
            symbol: Ticker symbol of the position to return. If ``None``,
                the ``ExecContext``\ 's :attr:`.symbol` is used. Defaults to
                ``None``.

        Returns:
            :class:`pybroker.portfolio.Position` if one exists, otherwise
            ``None``.
        """
        if symbol is None:
            symbol = self.symbol
            if symbol is None:
                raise ValueError("symbol is not set.")
        return self._portfolio.short_positions.get(symbol)

    def calc_target_shares(
        self,
        target_size: float,
        price: Optional[float] = None,
        cash: Optional[float] = None,
    ) -> Union[Decimal, int]:
        r"""Calculates the number of shares given a ``target_size`` allocation
        and share ``price``.

        Args:
            target_size: Proportion of cash used to calculate the number of
                shares, where the max ``target_size`` is ``1``. For example, a
                ``target_size`` of ``0.1`` would represent 10% of cash.
            price: Share price used to calculate the number of shares. If
                ``None``, the share price of the ``ExecContext``\ 's
                :attr:`.symbol` is used.
            cash: Cash used to calculate the number of number of shares. If
                ``None``, then the :class:`pybroker.portfolio.Portfolio` equity
                is used to calculate the number of shares.

        Returns:
            Number of shares given ``target_size`` and share ``price``. If
            :attr:`pybroker.config.StrategyConfig.enable_fractional_shares` is
            ``True``, then a Decimal is returned.
        """
        price = self.close[-1] if price is None else price
        shares = (
            (to_decimal(cash) if cash is not None else self._portfolio.equity)
            * to_decimal(target_size)
            / to_decimal(price)
        )
        if self.config.enable_fractional_shares:
            return shares.max(0)
        return max(int(shares), 0)

    def set_target_shares(self, target: float):
        r"""Sets buy or sell orders to reach a target allocation.

        Calculates the number of shares needed to reach ``target`` using
        :meth:`.calc_target_shares` and sets either :attr:`.buy_shares` or
        :attr:`.sell_shares`.

        Args:
            target: Target allocation size as a proportion of portfolio
                equity, where the max ``target`` is ``1``.
        """
        target_shares = self.calc_target_shares(target)
        pos = self.long_pos()
        if pos is None:
            self.buy_shares = target_shares
        elif pos.shares < target_shares:
            self.buy_shares = target_shares - pos.shares
        elif pos.shares > target_shares:
            self.sell_shares = pos.shares - target_shares

    def cancel_pending_order(self, order_id: int) -> bool:
        """Cancels a :class:`pybroker.scope.PendingOrder` with ``order_id``."""
        return self._pending_order_scope.remove(order_id)

    def cancel_all_pending_orders(self, symbol: Optional[str] = None):
        r"""Cancels all :class:`pybroker.scope.PendingOrder`\ s for ``symbol``.
        When ``symbol`` is ``None``, all pending orders are canceled.
        """
        self._pending_order_scope.remove_all(symbol)

    def cancel_stop(self, stop_id: int) -> bool:
        """Cancels a :class:`pybroker.portfolio.Stop` with ``stop_id``."""
        return self._portfolio.remove_stop(stop_id)

    def cancel_stops(
        self,
        val: Union[str, Position, Entry],
        stop_type: Optional[StopType] = None,
    ):
        r"""Cancels :class:`pybroker.portfolio.Stop`\ s.

        Args:
            val: Ticker symbol, :class:`pybroker.portfolio.Position`, or
                :class:`pybroker.portfolio.Entry` for which to cancel stops.
            stop_type: :class:`pybroker.common.StopType`.
        """
        self._portfolio.remove_stops(val, stop_type)

    def _get_symbol(self, symbol: Optional[str] = None) -> str:
        if symbol is not None:
            return symbol
        if self.symbol is None:
            raise ValueError("symbol is not set.")
        return self.symbol

    def _create_stop(
        self,
        stop_type: StopType,
        pos_type: Literal["long", "short"],
        points: Optional[Union[int, float, Decimal]],
        percent: Optional[Union[int, float, Decimal]],
        bars: Optional[int],
        fill_price: Optional[
            Union[
                int,
                float,
                np.floating,
                Decimal,
                PriceType,
                Callable[[str, BarData], Union[int, float, Decimal]],
            ]
        ],
        limit_price: Optional[Union[int, float, Decimal]],
        exit_price: Optional[PriceType],
    ):
        percent_dec, points_dec, limit_price_dec = None, None, None
        if stop_type != StopType.BAR:
            if percent is None and points is None:
                raise ValueError("Percent or points must be set.")
            if percent is not None:
                percent_dec = to_decimal(percent)
            elif points is not None:
                points_dec = to_decimal(points)
        if limit_price is not None:
            limit_price_dec = to_decimal(limit_price)
        if exit_price is not None and not isinstance(exit_price, PriceType):
            raise ValueError("Stop exit price must be a PriceType.")
        ExecContext._stop_id += 1
        return Stop(
            id=self._stop_id,
            symbol=self._get_symbol(),
            stop_type=stop_type,
            pos_type=pos_type,
            percent=percent_dec,
            points=points_dec,
            bars=bars,
            fill_price=fill_price,
            limit_price=limit_price_dec,
            exit_price=exit_price,
        )

    def _get_stops(
        self,
    ) -> tuple[Optional[frozenset[Stop]], Optional[frozenset[Stop]]]:
        pos_type: Optional[Literal["long", "short"]] = None
        if self.buy_shares is not None:
            pos_type = "long"
        elif self.sell_shares is not None:
            pos_type = "short"
        if pos_type is None:
            return None, None
        stops: deque[Stop] = deque()
        if self.hold_bars is not None:
            if self.hold_bars <= 0:
                raise ValueError("hold_bars must be greater than 0.")
            if pos_type == "long":
                fill_price = (
                    self.sell_fill_price
                    if self.sell_fill_price is not None
                    else PriceType.MIDDLE
                )
            else:
                fill_price = (
                    self.buy_fill_price
                    if self.buy_fill_price is not None
                    else PriceType.MIDDLE
                )
            stops.append(
                self._create_stop(
                    stop_type=StopType.BAR,
                    points=None,
                    percent=None,
                    bars=self.hold_bars,
                    pos_type=pos_type,
                    fill_price=fill_price,
                    limit_price=None,
                    exit_price=None,
                )
            )
        if self.stop_loss is not None and self.stop_loss_pct is not None:
            raise ValueError(
                "Only one of stop_loss or stop_loss_pct can be set."
            )
        if self.stop_loss is not None:
            stops.append(
                self._create_stop(
                    stop_type=StopType.LOSS,
                    points=self.stop_loss,
                    percent=None,
                    bars=None,
                    pos_type=pos_type,
                    fill_price=None,
                    limit_price=self.stop_loss_limit,
                    exit_price=self.stop_loss_exit_price,
                )
            )
        elif self.stop_loss_pct is not None:
            stops.append(
                self._create_stop(
                    stop_type=StopType.LOSS,
                    points=None,
                    percent=self.stop_loss_pct,
                    bars=None,
                    pos_type=pos_type,
                    fill_price=None,
                    limit_price=self.stop_loss_limit,
                    exit_price=self.stop_loss_exit_price,
                )
            )
        if self.stop_profit is not None and self.stop_profit_pct is not None:
            raise ValueError(
                "Only one of stop_profit or stop_profit_pct can be set."
            )
        if self.stop_profit is not None:
            stops.append(
                self._create_stop(
                    stop_type=StopType.PROFIT,
                    points=self.stop_profit,
                    percent=None,
                    bars=None,
                    pos_type=pos_type,
                    fill_price=None,
                    limit_price=self.stop_profit_limit,
                    exit_price=self.stop_profit_exit_price,
                )
            )
        elif self.stop_profit_pct is not None:
            stops.append(
                self._create_stop(
                    stop_type=StopType.PROFIT,
                    points=None,
                    percent=self.stop_profit_pct,
                    bars=None,
                    pos_type=pos_type,
                    fill_price=None,
                    limit_price=self.stop_profit_limit,
                    exit_price=self.stop_profit_exit_price,
                )
            )
        if (
            self.stop_trailing is not None
            and self.stop_trailing_pct is not None
        ):
            raise ValueError(
                "Only one of stop_trailing or stop_trailing_pct can be set."
            )
        if self.stop_trailing is not None:
            stops.append(
                self._create_stop(
                    stop_type=StopType.TRAILING,
                    points=self.stop_trailing,
                    percent=None,
                    bars=None,
                    pos_type=pos_type,
                    fill_price=None,
                    limit_price=self.stop_trailing_limit,
                    exit_price=self.stop_trailing_exit_price,
                )
            )
        elif self.stop_trailing_pct is not None:
            stops.append(
                self._create_stop(
                    stop_type=StopType.TRAILING,
                    points=None,
                    percent=self.stop_trailing_pct,
                    bars=None,
                    pos_type=pos_type,
                    fill_price=None,
                    limit_price=self.stop_trailing_limit,
                    exit_price=self.stop_trailing_exit_price,
                )
            )
        if (
            self.stop_loss_limit is not None
            and self.stop_loss is None
            and self.stop_loss_pct is None
        ):
            raise ValueError(
                "Either stop_loss or stop_loss_pct must be set when "
                "stop_loss_limit is set."
            )
        if (
            self.stop_loss_exit_price is not None
            and self.stop_loss is None
            and self.stop_loss_pct is None
        ):
            raise ValueError(
                "Either stop_loss or stop_loss_pct must be set when "
                "stop_loss_exit_price is set."
            )
        if (
            self.stop_profit_limit is not None
            and self.stop_profit is None
            and self.stop_profit_pct is None
        ):
            raise ValueError(
                "Either stop_profit or stop_profit_pct must be set when "
                "stop_profit_limit is set."
            )
        if (
            self.stop_profit_exit_price is not None
            and self.stop_profit is None
            and self.stop_profit_pct is None
        ):
            raise ValueError(
                "Either stop_profit or stop_profit_pct must be set when "
                "stop_profit_exit_price is set."
            )
        if (
            self.stop_trailing_limit is not None
            and self.stop_trailing is None
            and self.stop_trailing_pct is None
        ):
            raise ValueError(
                "Either stop_trailing or stop_trailing_pct must be set when "
                "stop_trailing_limit is set."
            )
        if (
            self.stop_trailing_exit_price is not None
            and self.stop_trailing is None
            and self.stop_trailing_pct is None
        ):
            raise ValueError(
                "Either stop_trailing or stop_trailing_pct must be set when "
                "stop_trailing_exit_price is set."
            )
        if pos_type == "long":
            return frozenset(stops), None
        else:
            return None, frozenset(stops)

    def to_result(self) -> Optional[ExecResult]:
        """Creates an :class:`.ExecResult` from the data set on
        :class:`.ExecContext`.
        """
        if self._curr_date is None:
            raise ValueError("curr_date is not set.")
        if self.symbol is None:
            raise ValueError("symbol is not set.")
        if self.buy_shares is None:
            if self.buy_limit_price is not None:
                raise ValueError(
                    "buy_shares must be set when buy_limit_price is set."
                )
            if self.buy_fill_price is not None and self.hold_bars is None:
                raise ValueError(
                    "buy_shares or hold_bars must be set when "
                    "buy_fill_price is set."
                )
        if self.sell_shares is None:
            if self.sell_limit_price is not None:
                raise ValueError(
                    "sell_shares must be set when sell_limit_price is set."
                )
            if self.sell_fill_price is not None and self.hold_bars is None:
                raise ValueError(
                    "sell_shares or hold_bars must be set when "
                    "sell_fill_price is set."
                )

        if self.buy_shares is None and self.sell_shares is None:
            if (
                self.stop_loss is not None
                or self.stop_loss_pct is not None
                or self.stop_loss_limit is not None
                or self.stop_profit is not None
                or self.stop_profit_pct is not None
                or self.stop_profit_limit is not None
                or self.stop_trailing is not None
                or self.stop_trailing_pct is not None
                or self.stop_trailing_limit is not None
            ):
                raise ValueError(
                    "Either buy_shares or sell_shares must be set when a stop "
                    "is set."
                )
            if self.hold_bars is not None:
                raise ValueError(
                    "Either buy_shares or sell_shares must be set when "
                    "hold_bars is set."
                )
        if self.buy_shares is not None and self.sell_shares is not None:
            raise ValueError(
                "For each symbol, only one of buy_shares or sell_shares can be"
                " set per bar."
            )
        if self.score is not None and (
            self.long_score is not None or self.short_score is not None
        ):
            raise ValueError(
                "score cannot be set when long_score or short_score is set."
            )
        if self.config.worst_rank_held is not None and self.score is not None:
            raise ValueError(
                "score cannot be used with worst_rank_held; use long_score or "
                "short_score instead."
            )
        if not self.buy_shares and not self.sell_shares:
            return None
        buy_fill_price = (
            self.buy_fill_price
            if self.buy_fill_price is not None
            else PriceType.MIDDLE
        )
        sell_fill_price = (
            self.sell_fill_price
            if self.sell_fill_price is not None
            else PriceType.MIDDLE
        )
        buy_shares = (
            to_decimal(self.buy_shares)
            if self.buy_shares is not None
            else None
        )
        buy_limit_price = (
            to_decimal(self.buy_limit_price)
            if self.buy_limit_price is not None
            else None
        )
        sell_limit_price = (
            to_decimal(self.sell_limit_price)
            if self.sell_limit_price is not None
            else None
        )
        sell_shares = (
            to_decimal(self.sell_shares)
            if self.sell_shares is not None
            else None
        )
        long_stops, short_stops = self._get_stops()
        return ExecResult(
            symbol=self.symbol,
            date=self._curr_date,
            buy_fill_price=buy_fill_price,
            sell_fill_price=sell_fill_price,
            score=self.score,
            long_score=self.long_score,
            short_score=self.short_score,
            hold_bars=self.hold_bars,
            buy_shares=buy_shares,
            buy_limit_price=buy_limit_price,
            sell_shares=sell_shares,
            sell_limit_price=sell_limit_price,
            long_stops=long_stops,
            short_stops=short_stops,
            cover=self._cover,
        )

    def __getattr__(self, attr):
        if attr in self._scope.custom_data_cols:
            if self.symbol is None:
                raise ValueError("symbol is not set.")
            return self._col_scope.fetch(
                self.symbol, attr, self._sym_end_index[self.symbol]
            )
        raise AttributeError(f"Attribute {attr!r} not found.")


def set_exec_ctx_data(ctx: ExecContext, date: np.datetime64):
    """Sets data on an :class:`.ExecContext` instance.

    Args:
        ctx: :class:`.ExecContext`.
        date: Current bar's date.
    """
    ctx._curr_date = date
    ctx._dt = None
    ctx._foreign.clear()
    ctx._timeframe.clear()
    ctx._timeframe_scope.clear_cache()
    ctx._cover = False
    ctx._exiting_pos = False
    ctx.buy_fill_price = None
    ctx.buy_shares = None
    ctx.buy_limit_price = None
    ctx.sell_fill_price = None
    ctx.sell_shares = None
    ctx.sell_limit_price = None
    ctx.hold_bars = None
    ctx.score = None
    ctx.long_score = None
    ctx.short_score = None
    ctx.stop_loss = None
    ctx.stop_loss_pct = None
    ctx.stop_loss_limit = None
    ctx.stop_profit = None
    ctx.stop_profit_pct = None
    ctx.stop_profit_limit = None
    ctx.stop_trailing = None
    ctx.stop_trailing_pct = None
    ctx.stop_trailing_limit = None
