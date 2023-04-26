# Copyright IBM Corp. 2023
# SPDX-License-Identifier: Apache2.0

from unittest import TestCase
import itertools
import sys
sys.path.append("..")
from mechanism_learning.trading_network import *

random = np.random.RandomState(0)

class TestPayment(TestCase):

    def test_quantity(self):
        for supplier_cost, buyer_cost in [(0.2, 0.3), (0.7, 0.8)]:        
            trading_network = TradingNetwork(
                type_prior = {},
                supplier_cost = {"L": supplier_cost},
                buyer_cost = {"L": buyer_cost},
            )

            declared_types = {"S": "L", "B": "L"}
            result = trading_network.get_quantity_payment(declared_types, 0.5)
            target_quantity = int(supplier_cost + buyer_cost < 1)
            print("test quantity with %f, %f" % (supplier_cost, buyer_cost))
            self.assertEqual(result["quantity"], target_quantity)

    def test_payment(self):
        prior = random.random(4)
        prior = prior / sum(prior)
        type_prior = {
            ("L", "L"): prior[0],
            ("L", "H"): prior[1],
            ("H", "L"): prior[2],
            ("H", "H"): prior[3],
        }
        supplier_cost = {
            "L": random.random() / 4,
            "H": 0.5 + random.random() / 4,
        }
        buyer_cost = {
            "L": random.random() / 4,
            "H": 0.5 + random.random() / 4,
        }
        
        trading_network = TradingNetwork(
            type_prior = type_prior,
            supplier_cost = supplier_cost,
            buyer_cost = buyer_cost,
        )

        for X, Y in itertools.product(["L", "H"], ["L", "H"]):
            declared_types = {"S": X, "B": Y}
            if Y == "H":
                weight = 0
            else:
                weight = (1 - supplier_cost[X] - buyer_cost[Y]) / (1 - buyer_cost[Y])
            result = trading_network.get_quantity_payment(declared_types, weight)
            
            target_supplier_net_payment = (weight - result["quantity"]) * (1 - buyer_cost[Y])
            self.assertAlmostEqual(result["net_payment"]["S"], target_supplier_net_payment)
            
            target_buyer_net_payment = supplier_cost[X] * result["quantity"]
            self.assertAlmostEqual(result["net_payment"]["B"], target_buyer_net_payment)
            
            supplier_utility = trading_network.get_supplier(X).get_utility(result["quantity"], result["net_payment"]["S"])
            print(X, Y, -supplier_cost[X] * result["quantity"] - result["net_payment"]["S"])
            self.assertGreaterEqual(supplier_utility, 0)
