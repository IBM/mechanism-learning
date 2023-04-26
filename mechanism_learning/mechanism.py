# Copyright IBM Corp. 2023
# SPDX-License-Identifier: Apache2.0

import typing as ty
from .trading_network import TradingNetwork


class BaseTradingMechanism:

    def __init__(self, trading_network: TradingNetwork) -> None:
        self.set_trading_network(trading_network)

    def set_trading_network(self, trading_network: TradingNetwork) -> None:
        self._trading_network = trading_network

    def get_trading_network(self) -> TradingNetwork:
        return self._trading_network

    # allocation rule
    def get_quantity(
        self,
        declared_types: ty.Dict[str, str]
    ) -> int:
        production_cost = self._trading_network.get_supplier(declared_types["S"]).get_cost()
        handling_cost = self._trading_network.get_buyer(declared_types["B"]).get_cost()
        if production_cost + handling_cost > 1:
            return 0
        else:
            return 1

    # net payment of supplier to IP
    def _get_supplier_net_payment(
        self,
        quantity: int,
        declared_buyer_type: str
    ) -> float:
        raise NotImplementedError("Not implemented #{self.class}##{__method__}")

    # net payment of buyer to IP
    def _get_buyer_net_payment(
        self,
        quantity: int,
        declared_supplier_type: str
    ) -> float:
        raise NotImplementedError("Not implemented #{self.class}##{__method__}")

    # allocation of quantity and payment (i.e. contract)
    def get_net_payment(
        self,
        declared_types: ty.Dict[str, str],
        quantity: int
    ) -> ty.Dict[str, float]:
        supplier_net_payment = self._get_supplier_net_payment(quantity,
                                                              declared_types["B"])
        buyer_net_payment = self._get_buyer_net_payment(quantity,
                                                        declared_types["S"])
        return {
            "S": supplier_net_payment,
            "B": buyer_net_payment,
        }

    def get_IP_utility(self, net_payment: ty.Dict[str, float]) -> float:
        # sum of the payment to IP
        return sum(net_payment.values())


class GrovesTradingMechanism(BaseTradingMechanism):

    def __init__(
        self,
        trading_network: TradingNetwork,
        additional_payment: ty.Dict[str, ty.Dict[str, float]]
    ) -> None:
        super().__init__(trading_network)
        self.set_additional_payment(additional_payment)

    def set_additional_payment(
        self,
        additional_payment: ty.Dict[str, ty.Dict[str, float]]
    ) -> None:
        self._additional_payment = additional_payment

    # net payment of supplier to IP
    def _get_supplier_net_payment(
        self,
        quantity: int,
        declared_buyer_type: str
    ) -> float:
        declared_buyer = self._trading_network.get_buyer(declared_buyer_type)
        payment = - quantity * (1 - declared_buyer.get_cost())
        payment += self._additional_payment["S"][declared_buyer_type]
        return payment

    # net payment of buyer to IP
    def _get_buyer_net_payment(
        self,
        quantity: int,
        declared_supplier_type: str
    ) -> float:
        payment = self._trading_network.get_supplier(declared_supplier_type).get_cost() * quantity
        payment += self._additional_payment["B"][declared_supplier_type]
        return payment


class AlphaTradingMechanism(BaseTradingMechanism):

    def __init__(
        self,
        trading_network: TradingNetwork,
        weight: float
    ) -> None:
        super().__init__(trading_network)
        self.set_weight(weight)

    def set_weight(self, weight: float) -> None:
        self._weight = weight

    # net payment of supplier to IP
    def _get_supplier_net_payment(
        self,
        quantity: int,
        declared_buyer_type: str
    ) -> float:
        if quantity == 0:
            return 0
        else:
            declared_buyer = self._trading_network.get_buyer(declared_buyer_type)
            return (self._weight - quantity) * (1 - declared_buyer.get_cost())

    # net payment of buyer to IP
    def _get_buyer_net_payment(
        self,
        quantity: int,
        declared_supplier_type: str
    ) -> float:
        return self._trading_network.get_supplier(declared_supplier_type).get_cost() * quantity
