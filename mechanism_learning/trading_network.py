# Copyright IBM Corp. 2023

import typing as ty
from .player import Supplier, Buyer


class TradingNetwork:

    def __init__(
        self,
        supplier_cost: ty.Dict[str, float],
        buyer_cost: ty.Dict[str, float]
    ) -> None:

        # suppliers of each type
        self._supplier = dict()
        for supplier_type in supplier_cost:
            self._supplier[supplier_type] = Supplier(supplier_cost[supplier_type])

        # buyers of each type
        self._buyer = dict()
        for buyer_type in buyer_cost:
            self._buyer[buyer_type] = Buyer(buyer_cost[buyer_type])

    def get_supplier_types(self) -> ty.KeysView[str]:
        return self._supplier.keys()

    def get_buyer_types(self) -> ty.KeysView[str]:
        return self._buyer.keys()

    def get_player_types(self, player: str) -> ty.KeysView[str]:
        if player == "S":
            return self.get_supplier_types()
        elif player == "B":
            return self.get_buyer_types()
        else:
            raise ValueError()

    def get_supplier(self, supplier_type: str) -> Supplier:
        return self._supplier[supplier_type]

    def get_buyer(self, buyer_type: str) -> Buyer:
        return self._buyer[buyer_type]

    def get_player(
        self,
        player: str,
        player_type: str
    ) -> ty.Union[Supplier, Buyer]:

        if player == "S":
            return self.get_supplier(player_type)
        elif player == "B":
            return self.get_buyer(player_type)
        else:
            raise ValueError()

    def get_player_cost(self, player: str) -> ty.Dict[str, float]:
        player_cost = {
            p_type: self.get_player(player, p_type).get_cost()
            for p_type in self.get_player_types(player)
        }
        return player_cost
