# Copyright IBM Corp. 2023

import typing as ty
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from mechanism_learning.env import BaseSupplierBuyerEnv, GrovesSupplierBuyerEnv
from mechanism_learning.compute_groves_payment import compute_groves_payment, learn_groves_payment


def run_compute_groves_payment(
    ser_cost_prob: pd.Series,
    n_sample: ty.Optional[int] = None,
    reduce_variables: bool = False,
    confidence_bound: bool = False,
    min_utility: ty.Optional[ty.Dict[str, ty.Dict[str, float]]] = None,  # minimum utility to guarantee for each agent of each type
) -> pd.Series:

    # get parameters of the instance
    supplier_cost = {
        "L": ser_cost_prob["cost_s_low"],
        "H": ser_cost_prob["cost_s_high"],
    }
    buyer_cost = {
        "L": ser_cost_prob["cost_b_low"],
        "H": ser_cost_prob["cost_b_high"],
    }
    type_prior = {
        ("L", "L"): ser_cost_prob["prob_LL"],
        ("L", "H"): ser_cost_prob["prob_LH"],
        ("H", "L"): ser_cost_prob["prob_HL"],
        ("H", "H"): ser_cost_prob["prob_HH"],
    }

    # compute Groves payment
    env = BaseSupplierBuyerEnv(supplier_cost, buyer_cost, type_prior)
    if n_sample is None:
        opt, additional_payment = compute_groves_payment(env,
                                                         reduce_variables=reduce_variables,
                                                         min_utility=min_utility)
    else:
        # prepare samples
        sample_types = env.get_sample_prior(n_sample)
        s_types = set([s_type for s_type, _ in sample_types])
        b_types = set([b_type for _, b_type in sample_types])
        sample_supplier_cost = {
            s_type: env.get_trading_network().get_supplier(s_type).get_cost()
            for s_type in s_types
        }
        sample_buyer_cost = {
            b_type: env.get_trading_network().get_buyer(b_type).get_cost()
            for b_type in b_types
        }

        opt, additional_payment = learn_groves_payment(sample_types,
                                                       sample_supplier_cost,
                                                       sample_buyer_cost,
                                                       reduce_variables=reduce_variables,
                                                       confidence_bound=confidence_bound,
                                                       min_utility=min_utility)

        # result["additional_payment"]
        if additional_payment:
            # empty if infeasible
            for player in ["S", "B"]:
                if player == "S":
                    player_cost = supplier_cost
                elif player == "B":
                    player_cost = buyer_cost
                else:
                    raise ValueError()

                X = [[player_cost[p_type]] for p_type in additional_payment[player]]
                y = [additional_payment[player][p_type] for p_type in additional_payment[player]]
                regressor = GaussianProcessRegressor()
                regressor.fit(X, y)

                X_pred = [[player_cost[p_type]] for p_type in player_cost]
                z_pred = regressor.predict(X_pred)
                for i, p_type in enumerate(player_cost):
                    if p_type not in additional_payment[player]:
                        additional_payment[player][p_type] = z_pred[i]

    # check whether to satisfy all four properties
    is_feasible = opt.solution.is_primal_feasible()

    if is_feasible:
        groves_env = GrovesSupplierBuyerEnv(supplier_cost,
                                            buyer_cost,
                                            type_prior,
                                            additional_payment)
        expected_utility = groves_env.get_expected_utility()
    else:
        expected_utility = {i: 0. for i in ["S-L", "S-H", "B-L", "B-H", "IP"]}
        additional_payment = {
            player: {
                cost: 0 for cost in ["L", "H"]
            } for player in ["S", "B"]
        }

    df = pd.DataFrame([is_feasible], columns=["is_feasible"])
    for key in expected_utility:
        df["utility_" + key] = expected_utility[key]
    for player in additional_payment:
        for player_type in additional_payment[player]:
            df["payment_" + player + "-" + player_type] = additional_payment[player][player_type]

    return df.iloc[0]


def runall_compute_groves_payment(
    df_cost_prob: pd.DataFrame,
    reduce_variables: bool = False,
    min_utility: ty.Optional[ty.Dict[str, ty.Dict[str, float]]] = None,  # minimum utility to guarantee for each agent of each type
) -> pd.DataFrame:

    df = pd.DataFrame()

    for i in range(len(df_cost_prob)):
        ser_cost_prob = df_cost_prob.iloc[i]
        ser_result = run_compute_groves_payment(ser_cost_prob,
                                                reduce_variables=reduce_variables,
                                                min_utility=min_utility)
        df = pd.concat([df, pd.DataFrame([pd.concat([ser_cost_prob, ser_result])])])

    return df


def runall_compute_learn_groves_payment(
    df_cost_prob: pd.DataFrame,
    n_sample: int,
    reduce_variables: bool = False,
    confidence_bound: bool = False,
    min_utility: ty.Optional[ty.Dict[str, ty.Dict[str, float]]] = None,  # minimum utility to guarantee for each agent of each type
) -> pd.DataFrame:

    df = pd.DataFrame()

    for i in range(len(df_cost_prob)):
        ser_cost_prob = df_cost_prob.iloc[i]

        # Compute
        ser_result = run_compute_groves_payment(ser_cost_prob,
                                                reduce_variables=reduce_variables,
                                                min_utility=min_utility)
        ser_result = ser_result.add_prefix("comp_")

        # Learn
        ser_result_learn = run_compute_groves_payment(ser_cost_prob,
                                                      n_sample=n_sample,
                                                      reduce_variables=reduce_variables,
                                                      confidence_bound=confidence_bound,
                                                      min_utility=min_utility)
        ser_result_learn = ser_result_learn.add_prefix("learn_")

        # Aggregate results
        df = pd.concat([
            df,
            pd.DataFrame([pd.concat([ser_cost_prob, ser_result, ser_result_learn])])
        ])

    return df
