# Copyright IBM Corp. 2023
# SPDX-License-Identifier: Apache2.0

from unittest import TestCase
import itertools
import numpy as np
from mechanism_learning.trading_network import *


random = np.random.RandomState(0)


class TestUtility(TestCase):        

    def _test_supplier(self, cost, quantity, net_payment):
        supplier = Supplier(cost)
        self.assertEqual(cost, supplier.get_cost())
        utility = supplier.get_utility(quantity, net_payment)
        target = - supplier.get_cost() * quantity - net_payment
        print("test supplier utility with (cost, quantity, net_payment) (%f, %d, %f)" % (cost, quantity, net_payment))
        self.assertAlmostEqual(utility, target), "utility, target = %f, %f" % (utility, target)

    def test_supplier(self):
        for cost, quantity, net_payment in itertools.product([0, random.random(), 1], [0, 1], [0, random.random(), 1]):
            self._test_supplier(cost, quantity, net_payment)

    def _test_buyer(self, cost, quantity, net_payment, retail_price):
        buyer = Buyer(cost)
        self.assertEqual(cost, buyer.get_cost())
        utility = buyer.get_utility(quantity, net_payment, retail_price=retail_price)
        target = (retail_price - cost) * quantity - net_payment
        print("test buyer utility with (cost, quantity, net_payment, retail_price) (%f, %d, %f, %f)" % (cost, quantity, net_payment, retail_price))
        self.assertAlmostEqual(utility, target), "utility, target = %f, %f" % (utility, target)

    def test_buyer(self):
        for cost, quantity, net_payment, retail_price in itertools.product([0, random.random(), 1], [0, 1], [0, random.random(), 1], [0, random.random(), 1]):
            self._test_buyer(cost, quantity, net_payment, retail_price)

    def test_IP(self):
        trading_network = TradingNetwork(
            type_prior = dict(),
            supplier_cost = dict(),
            buyer_cost = dict()
        )
        for supplier_net_payment, buyer_net_payment in itertools.product([0, random.random(), 1], [0, random.random(), 1]):
            net_payment = {"S": supplier_net_payment, "B": buyer_net_payment}
            utility = trading_network.get_IP_utility(net_payment)
            target = net_payment["S"] + net_payment["B"]
            self.assertAlmostEqual(utility, target)
