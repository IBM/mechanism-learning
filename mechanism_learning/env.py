# Copyright IBM Corp. 2023

import random
import itertools
import numpy as np
import typing as ty
from collections import defaultdict, Counter
from .mechanism import BaseTradingMechanism, AlphaTradingMechanism, GrovesTradingMechanism
from .trading_network import TradingNetwork


class BaseSupplierBuyerEnv:

    def __init__(
        self,
        supplier_cost: ty.Dict[str, float],
        buyer_cost: ty.Dict[str, float],
        type_prior: ty.Dict[ty.Tuple[str, str], float]
    ):
        self._trading_network = TradingNetwork(supplier_cost, buyer_cost)
        self._prior = type_prior
        self._mechanism = self.set_mechanism()
        random.seed(0)

    def set_mechanism(self) -> BaseTradingMechanism:
        return BaseTradingMechanism(self._trading_network)

    def get_mechanism(self) -> BaseTradingMechanism:
        return self._mechanism

    def get_trading_network(self) -> TradingNetwork:
        return self._trading_network

    def get_prior(self, types: ty.Dict[str, str]) -> float:
        return self._prior[(types["S"], types["B"])]

    def get_prior_dist(self) -> ty.Dict[ty.Tuple[str, str], float]:
        return self._prior

    def get_sample_prior(self, num: int) -> ty.List[ty.Tuple[str, str]]:
        samples = random.choices(list(self._prior.keys()),
                                 weights=list(self._prior.values()),
                                 k=num)
        return samples

    # utility of each player with given declared types and weight for supplier payment
    def get_utility(self, declared_types: ty.Dict[str, str]) -> ty.Dict[str, float]:
        quantity = self.get_mechanism().get_quantity(declared_types)
        net_payment = self.get_mechanism().get_net_payment(declared_types, quantity)

        utility = dict()
        utility["S"] = self._trading_network.get_supplier(declared_types["S"]).get_utility(quantity, net_payment["S"])
        utility["B"] = self._trading_network.get_buyer(declared_types["B"]).get_utility(quantity, net_payment["B"])
        utility["IP"] = self.get_mechanism().get_IP_utility(net_payment)

        return utility

    # expected utility of each player of each type
    def get_expected_utility(self) -> ty.Dict[str, float]:

        expected_utility: ty.Dict[str, float] = defaultdict(float)
        total_prior: ty.Dict[str, float] = defaultdict(float)

        for (X, Y) in itertools.product(
            self._trading_network.get_supplier_types(),
            self._trading_network.get_buyer_types()
        ):
            declared_types = {"S": X, "B": Y}
            prior = self.get_prior(declared_types)
            utility = self.get_utility(declared_types)
            for player in utility:
                if player == "IP":
                    player_type = player
                else:
                    player_type = player + "-" + declared_types[player]
                expected_utility[player_type] += prior * utility[player]
                total_prior[player_type] += prior

        for key in expected_utility:
            if total_prior[key] == 0:
                continue
            expected_utility[key] /= total_prior[key]

        return expected_utility

    def get_total_value(self, declared_types: ty.Dict[str, str]) -> float:
        quantity = self.get_mechanism().get_quantity(declared_types)
        total_value = self._trading_network.get_supplier(declared_types["S"]).get_value(quantity)
        total_value += self._trading_network.get_buyer(declared_types["B"]).get_value(quantity)
        return total_value

    def get_expected_total_value(self) -> float:
        total_value = 0.
        for types in self._prior:
            declared_types = {"S": types[0], "B": types[1]}
            total_value += self.get_total_value(declared_types) * self._prior[types]        
        return total_value

    def get_expected_total_value_squared(self) -> float:
        total_value_squared = 0.
        for types in self._prior:
            declared_types = {"S": types[0], "B": types[1]}
            total_value_squared += self.get_total_value(declared_types)**2 * self._prior[types]        
        return total_value_squared
    
    def get_conditional_prior_of_b_types(self, s_type: str) -> ty.Dict[str, float]:
        # conditional probability of buyer type given that the supplier type is s_type
        conditional: ty.Dict[str, float] = defaultdict(float)
        for X, Y in self._prior:
            if X == s_type:
                conditional[Y] += self._prior[(X, Y)]
        total_probability = sum(conditional.values())
        for b_type in conditional:
            conditional[b_type] = conditional[b_type] / total_probability
        return conditional

    def get_conditional_prior_of_s_types(self, b_type: str) -> ty.Dict[str, float]:
        # conditional probability of supplier type given that the buyer type is b_type
        conditional: ty.Dict[str, float] = defaultdict(float)
        for X, Y in self._prior:
            if Y == b_type:
                conditional[X] += self._prior[(X, Y)]
        total_probability = sum(conditional.values())
        for s_type in conditional:
            conditional[s_type] = conditional[s_type] / total_probability
        return conditional

    def get_conditional_prior_of_opponent_types(
        self,
        player,
        player_type: str
    ) -> ty.Dict[str, float]:
        if player == "S":
            return self.get_conditional_prior_of_b_types(player_type)
        elif player == "B":
            return self.get_conditional_prior_of_s_types(player_type)
        else:
            raise ValueError()

    def get_expected_total_value_given_s_type(
        self,
        s_type: str,
        prior_given_s_type: ty.Optional[ty.Dict[str, float]] = None
    ) -> float:

        if prior_given_s_type is None:
            conditional = self.get_conditional_prior_of_b_types(s_type)
        else:
            conditional = prior_given_s_type

        expected_total_value = 0.
        for b_type in conditional:
            declared_types = {"S": s_type, "B": b_type}
            expected_total_value += self.get_total_value(declared_types) * conditional[b_type]
        return expected_total_value

    def get_expected_total_value_given_b_type(
        self,
        b_type: str,
        prior_given_b_type: ty.Optional[ty.Dict[str, float]] = None
    ) -> float:

        if prior_given_b_type is None:
            conditional = self.get_conditional_prior_of_s_types(b_type)
        else:
            conditional = prior_given_b_type

        expected_total_value = 0.
        for s_type in conditional:
            declared_types = {"S": s_type, "B": b_type}
            expected_total_value += self.get_total_value(declared_types) * conditional[s_type]
        return expected_total_value

    def get_expected_total_value_given_player_type(
        self,
        player: str,
        player_type: str,
        prior_given_player_type: ty.Optional[ty.Dict[str, float]] = None
    ) -> float:

        if player == "S":
            return self.get_expected_total_value_given_s_type(player_type,
                                                              prior_given_player_type)
        elif player == "B":
            return self.get_expected_total_value_given_b_type(player_type,
                                                              prior_given_player_type)
        else:
            raise ValueError()


class GrovesSupplierBuyerEnv(BaseSupplierBuyerEnv):

    def __init__(
        self,
        supplier_cost: ty.Dict[str, float],
        buyer_cost: ty.Dict[str, float],
        type_prior: ty.Dict[ty.Tuple[str, str], float],
        additional_payment: ty.Dict[str, ty.Dict[str, float]]
    ):
        self._additional_payment = additional_payment
        super().__init__(supplier_cost, buyer_cost, type_prior)

    def set_mechanism(self) -> BaseTradingMechanism:
        return GrovesTradingMechanism(self._trading_network,
                                      self._additional_payment)


class AlphaSupplierBuyerEnv(BaseSupplierBuyerEnv):

    def __init__(
        self,
        supplier_cost: ty.Dict[str, float],
        buyer_cost: ty.Dict[str, float],
        type_prior: ty.Dict[ty.Tuple[str, str], float],
        weight
    ):
        self._weight = weight
        super().__init__(supplier_cost, buyer_cost, type_prior)

    def set_mechanism(self) -> BaseTradingMechanism:
        return AlphaTradingMechanism(self._trading_network, self._weight)
