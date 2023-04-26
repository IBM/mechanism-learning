# Copyright IBM Corp. 2023

import typing as ty


class Supplier:

    def __init__(self, production_cost: float) -> None:
        # production cost per unit
        self._cost = production_cost

    def get_cost(self) -> float:
        return self._cost

    def get_value(self, quantity: int) -> float:
        # - production cost per unit * quantity
        return - self.get_cost() * quantity

    def get_utility(
        self,
        quantity: int,
        net_payment: float
    ) -> float:
        # - production cost per unit * quantity - net payment to IP
        return self.get_value(quantity) - net_payment


class Buyer:

    def __init__(self, handling_cost: float) -> None:
        # handling cost per unit
        self._cost = handling_cost

    def get_cost(self) -> float:
        return self._cost

    def get_value(
        self,
        quantity: int,
        retail_price: float = 1.0
    ) -> float:
        # - production cost per unit * quantity
        return (retail_price - self.get_cost()) * quantity

    def get_utility(
        self,
        quantity: int,
        net_payment: float,
        retail_price: float = 1.0
    ) -> float:
        # (retail price - handling cost per unit) * quantity - net payment to IP
        return self.get_value(quantity, retail_price) - net_payment
