#########
Changelog
#########

1.2.13
======

* Adds signal provenance fields to `Order <https://www.pybroker.com/en/latest/reference/pybroker.portfolio.html#pybroker.portfolio.Order>`_:

    * ``created`` — date the order signal was created.
    * ``order_type`` — how the order originated (``market``, ``limit``, ``stop_bar``, ``stop_loss``, ``stop_profit``, ``stop_trailing``).
    * ``intent`` — position intent (``buy_to_open``, ``buy_to_close``, ``sell_to_open``, ``sell_to_close``).

* Adds `OrderType <https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.OrderType>`_ and `PositionIntent <https://www.pybroker.com/en/latest/reference/pybroker.common.html#pybroker.common.PositionIntent>`_ enums.

* Adds ``order_id`` parameter to `PendingOrderScope.orders() <https://www.pybroker.com/en/latest/reference/pybroker.scope.html#pybroker.scope.PendingOrderScope.orders>`_.

1.2.3
=====

* Adds built-in indicators to the `indicator module <https://www.pybroker.com/en/latest/reference/pybroker.indicator.html>`_.


1.1.0
=====

* `Adds support for the following stop types: <https://www.pybroker.com/en/latest/notebooks/8.%20Applying%20Stops.html>`_

    * Stop loss
    * Trailing stop loss
    * Take profit

* `Allows canceling pending orders. <https://www.pybroker.com/en/latest/notebooks/FAQs.html#...-cancel-pending-orders?>`_

* Upgrades ``alpaca-trade-api-python`` to ``alpaca-py`` package.

1.0.0
=====

* Initial release!