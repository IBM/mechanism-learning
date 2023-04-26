# Copyright IBM Corp. 2023
# SPDX-License-Identifier: Apache2.0

import cplex
import numpy as np
import typing as ty
from collections import defaultdict, Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from .mechanism import BaseTradingMechanism
from .trading_network import TradingNetwork
from .env import BaseSupplierBuyerEnv


def _get_LP_with_obj_WBB(
    env: BaseSupplierBuyerEnv,
    reduce_variables: bool = False,
    confidence_bound: bool = False,
    n_sample: ty.Optional[int] = None,
) -> cplex.Cplex:

    opt = cplex.Cplex()
    opt.set_problem_type(opt.problem_type.LP)

    # set variables and range
    # e.g. variable S_L means the groves payment of Supplier when buyer declares Low
    players = ["S", "B"]  # trading_network._types
    var_names = list()
    for player, other_player in zip(players, reversed(players)):
        other_player_types = env.get_trading_network().get_player_types(other_player)
        if reduce_variables and player == "B":
            var_names.append(player)
        else:
            # TODO: reduce S_Y for C_B^Y > 1
            for player_type in other_player_types:
                var_names.append(player + "_" + player_type)
    lb = [-np.inf] * len(var_names)
    ub = [np.inf] * len(var_names)
    opt.variables.add(names=var_names, lb=lb, ub=ub)

    # prepare marginal probability
    # marginal["S_L"]: marginal probability that seller sees L declared by other
    marginal: ty.Dict[str, float] = defaultdict(float)
    for key in env.get_prior_dist():
        s_type, b_type = key
        proba = env.get_prior_dist()[key]
        # TODO: reduce S_Y for C_B^Y > 1
        marginal["S_" + b_type] += proba
        if reduce_variables:
            marginal["B"] += proba
        else:
            marginal["B_" + s_type] += proba

    # set objective
    # min E[beta_S^Y + beta_B^X]
    opt.objective.set_sense(opt.objective.sense.minimize)
    opt.objective.set_linear([(v, marginal[v]) for v in var_names])

    # prepare expected total value: E[(1 - C_S^X - C_B^Y) * q(X,Y)]
    expected_total_value = env.get_expected_total_value()
    if confidence_bound:
        expected_total_value_squared = env.get_expected_total_value_squared()
        variance = (expected_total_value_squared - expected_total_value**2) / n_sample
        print(expected_total_value, np.sqrt(variance))
        expected_total_value += np.sqrt(variance)

    # set constraints on WBB
    # s.t. E[beta_S^Y + beta_B^X] >= E[(1 - C_S^X - C_B^Y) * q(X,Y)]
    opt.linear_constraints.add(names=["WBB"],
                               lin_expr=[[var_names, [marginal[v] for v in var_names]]],
                               senses=["G"],
                               rhs=[expected_total_value])

    return opt


def _add_IR_constraint_to_learn(
    opt: cplex.Cplex,
    env: BaseSupplierBuyerEnv,
    sample_types: ty.List[ty.Tuple[str, str]],
    sample_supplier_cost: ty.Dict[str, float],
    sample_buyer_cost: ty.Dict[str, float],
    player: str,
    variables_reduced: bool = False,
    confidence_bound: bool = False,
    min_utility_of_player: ty.Optional[ty.Dict[str, float]] = None,
) -> cplex.Cplex:

    # set constraints on IR of player
    # s.t. E[beta_S^Y | X] <= E[(1 - C_S^X - C_B^Y) * q(X,Y) | X]
    # s.t. E[beta_B^X | Y] <= E[(1 - C_S^X - C_B^Y) * q(X,Y) | Y]

    # regressor of the conditional total value (right-hand side)
    if player == "S":
        X = [[sample_supplier_cost[s_type]] for s_type, _ in sample_types]
    elif player == "B":
        X = [[sample_buyer_cost[b_type]] for _, b_type in sample_types]
    else:
        raise ValueError()

    y = [env.get_total_value({"S": s_type, "B": b_type}) for s_type, b_type in sample_types]

    regressor = GaussianProcessRegressor()
    regressor.fit(X, y)

    # for each type of the player, predict the conditional total value
    player_types = env.get_trading_network().get_player_types(player)
    X_pred = [
        [env.get_trading_network().get_player(player, p_type).get_cost()]
        for p_type in player_types
    ]
    conditional_total_value, std = regressor.predict(X_pred, return_std=True)
    if confidence_bound:
        print(conditional_total_value, std)
        conditional_total_value -= std

    for i, p_type in enumerate(player_types):
        if variables_reduced:
            # TODO: reduce S_Y for C_B^Y > 1
            variables = [player]
            coeffs = [1.]
        else:
            conditional = env.get_conditional_prior_of_opponent_types(player, p_type)
            variables = [player + "_" + o_type for o_type in conditional]
            coeffs = [conditional[o_type] for o_type in conditional]

        if min_utility_of_player:
            rhs = [conditional_total_value[i] - min_utility_of_player[p_type]]
        else:
            rhs = [conditional_total_value[i]]

        opt.linear_constraints.add(names=["IR" + player + p_type],
                                   lin_expr=[[variables, coeffs]],
                                   senses=["L"],
                                   rhs=rhs)

    return opt


def learn_groves_LP(
    sample_types: ty.List[ty.Tuple[str, str]],
    sample_supplier_cost: ty.Dict[str, float],
    sample_buyer_cost: ty.Dict[str, float],
    reduce_variables: bool = False,
    confidence_bound: bool = False,
    min_utility: ty.Optional[ty.Dict[str, ty.Dict[str, float]]] = None,
) -> ty.Tuple[cplex.Cplex, BaseSupplierBuyerEnv]:

    empirical_prior: ty.Dict[ty.Tuple[str, str], float] = dict(Counter(sample_types))
    for types in empirical_prior:
        empirical_prior[types] /= len(sample_types)

    env = BaseSupplierBuyerEnv(sample_supplier_cost,
                               sample_buyer_cost,
                               empirical_prior)

    opt = _get_LP_with_obj_WBB(env,
                               reduce_variables=reduce_variables,
                               confidence_bound=confidence_bound,
                               n_sample=len(sample_types))

    for player in ["S", "B"]:
        # set constraints on IR of player
        # s.t. E[beta_S^Y | X] <= E[(1 - C_S^X - C_B^Y) * q(X,Y) | X]
        # s.t. E[beta_B^X | Y] <= E[(1 - C_S^X - C_B^Y) * q(X,Y) | Y]
        if reduce_variables and player == "B":
            # TODO: reduce S_Y for C_B^Y > 1
            reduce = True
        else:
            reduce = False

        if min_utility:
            min_utility_of_player = min_utility[player]
        else:
            min_utility_of_player = None

        opt = _add_IR_constraint_to_learn(opt,
                                          env,
                                          sample_types,
                                          sample_supplier_cost,
                                          sample_buyer_cost,
                                          player,
                                          variables_reduced=reduce,
                                          confidence_bound=confidence_bound,
                                          min_utility_of_player=min_utility_of_player)

    # solve the LP
    opt.solve()

    return opt, env


def _add_IR_constraint(
    opt: cplex.Cplex,
    env: BaseSupplierBuyerEnv,
    player: str,
    variables_reduced: bool = False,
    min_utility_of_player: ty.Optional[ty.Dict[str, float]] = None
) -> cplex.Cplex:

    # set constraints on IR of player
    # s.t. E[beta_S^Y | X] <= E[(1 - C_S^X - C_B^Y) * q(X,Y) | X]
    # s.t. E[beta_B^X | Y] <= E[(1 - C_S^X - C_B^Y) * q(X,Y) | Y]

    for p_type in env.get_trading_network().get_player_types(player):
        # conditional probability of the opponent types given the type of the player
        conditional = env.get_conditional_prior_of_opponent_types(player, p_type)

        # conditional expected total value given the type of the player
        conditional_total_value = env.get_expected_total_value_given_player_type(player,
                                                                                 p_type,
                                                                                 conditional)

        if min_utility_of_player:
            rhs = [conditional_total_value - min_utility_of_player[p_type]]
        else:
            rhs = [conditional_total_value]

        if variables_reduced:
            # TODO: reduce S_Y for C_B^Y > 1
            variables = [player]
            coeffs = [1.]
        else:
            # E[beta_S^Y | X], where X is the player type and Y is the opponent type
            variables = [player + "_" + o_type for o_type in conditional]
            coeffs = [conditional[o_type] for o_type in conditional]

        # rhs: E[(1 - C_S^X - C_B^Y) * q(X,Y) | X]
        opt.linear_constraints.add(names=["IR" + player + p_type],
                                   lin_expr=[[variables, coeffs]],
                                   senses=["L"],
                                   rhs=rhs)

    return opt


def solve_groves_LP(
    env: BaseSupplierBuyerEnv,
    reduce_variables: bool = False,
    min_utility: ty.Optional[ty.Dict[str, ty.Dict[str, float]]] = None,
) -> cplex.Cplex:

    opt = _get_LP_with_obj_WBB(env, reduce_variables)

    for player in ["S", "B"]:
        # set constraints on IR of player
        # s.t. E[beta_S^Y | X] <= E[(1 - C_S^X - C_B^Y) * q(X,Y) | X]
        # s.t. E[beta_B^X | Y] <= E[(1 - C_S^X - C_B^Y) * q(X,Y) | Y]
        if reduce_variables and player == "B":
            # TODO: reduce S_Y for C_B^Y > 1
            reduce = True
        else:
            reduce = False
        if min_utility:
            opt = _add_IR_constraint(opt, env, player, reduce, min_utility[player])
        else:
            opt = _add_IR_constraint(opt, env, player, reduce)

    # solve the LP
    opt.solve()

    return opt


def _LP_to_payment(
    opt: cplex.Cplex,
    env: BaseSupplierBuyerEnv,
    variables_reduced: bool = False
) -> ty.Dict[str, ty.Dict[str, float]]:

    # payment[player][player_type] = payment_to_IP
    #                                (from the player of the player_type)
    payment = dict()
    payment["S"] = {
        Y: opt.solution.get_values("S_%s" % Y)
        for Y in env.get_trading_network().get_buyer_types()
    }
    if variables_reduced:
        payment["B"] = {
            X: opt.solution.get_values("B")
            for X in env.get_trading_network().get_supplier_types()
        }
    else:
        payment["B"] = {
            X: opt.solution.get_values("B_%s" % X)
            for X in env.get_trading_network().get_supplier_types()
        }

    return payment


# Compute Groves payment from the true environment
def compute_groves_payment(
    env: BaseSupplierBuyerEnv,
    reduce_variables: bool = False,
    min_utility: ty.Optional[ty.Dict[str, ty.Dict[str, float]]] = None,  # minimum utility to guarantee for each agent of each type
) -> ty.Tuple[cplex.Cplex, ty.Dict[str, ty.Dict[str, float]]]:

    opt = solve_groves_LP(env, reduce_variables, min_utility)

    if opt.solution.is_primal_feasible():
        additional_payment = _LP_to_payment(opt,
                                            env,
                                            variables_reduced=reduce_variables)
    else:
        additional_payment = {}

    return opt, additional_payment


# Learn Groves payment from samples
def learn_groves_payment(
    sample_types: ty.List[ty.Tuple[str, str]],
    sample_supplier_cost: ty.Dict[str, float],
    sample_buyer_cost: ty.Dict[str, float],
    reduce_variables: bool = False,
    confidence_bound: bool = False,
    min_utility: ty.Optional[ty.Dict[str, ty.Dict[str, float]]] = None,  # minimum utility to guarantee for each agent of each type
) -> ty.Tuple[cplex.Cplex, ty.Dict[str, ty.Dict[str, float]]]:

    opt, env = learn_groves_LP(sample_types,
                               sample_supplier_cost,
                               sample_buyer_cost,
                               reduce_variables,
                               confidence_bound=confidence_bound,
                               min_utility=min_utility)

    if opt.solution.is_primal_feasible():
        additional_payment = _LP_to_payment(opt,
                                            env,
                                            variables_reduced=reduce_variables)
    else:
        additional_payment = {}

    return opt, additional_payment
