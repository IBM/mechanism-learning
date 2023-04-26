# Copyright IBM Corp. 2023
# SPDX-License-Identifier: Apache2.0

import pathlib
import argparse
import numpy as np
import pandas as pd
from mechanism_learning.experiment import runall_compute_learn_groves_payment

parser = argparse.ArgumentParser(description="Computing groves mechanisms for random instances")
parser.add_argument("n", type=int, help="number of instances")
parser.add_argument("n_sample", type=int, help="number of samples")
parser.add_argument("seed", type=int, help="random number seed")
parser.add_argument("--reduce", action="store_true", help="if reduce variables")
parser.add_argument("--CB", action="store_true", help="if confidence bound")
args = parser.parse_args()

rng = np.random.RandomState(args.seed)

cost = rng.random((args.n, 4))
df = pd.DataFrame(cost, columns=["s1", "s2", "b1", "b2"])
df["cost_s_low"] = df[["s1", "s2"]].min(axis=1)
df["cost_s_high"] = df[["s1", "s2"]].max(axis=1)
df["cost_b_low"] = df[["b1", "b2"]].min(axis=1)
df["cost_b_high"] = df[["b1", "b2"]].max(axis=1)
is_impossible = (df["cost_s_low"] + df["cost_b_high"] < 1) * (df["cost_s_high"] + df["cost_b_low"] < 1) * (df["cost_s_high"] + df["cost_b_high"] > 1)
df_cost = df[is_impossible][["cost_s_low", "cost_s_high", "cost_b_low", "cost_b_high"]]
df_cost = df_cost.reset_index()

prob = rng.random((len(df_cost), 4))
df = pd.DataFrame(prob, columns=["p1", "p2", "p3", "p4"])
df["sum"] = df.sum(axis=1)
df["prob_LL"] = df["p1"] / df["sum"]
df["prob_LH"] = df["p2"] / df["sum"]
df["prob_HL"] = df["p3"] / df["sum"]
df["prob_HH"] = df["p4"] / df["sum"]
df_prob = df[["prob_LL", "prob_LH", "prob_HL", "prob_HH"]]

df_cost_prob = pd.concat([df_cost, df_prob], axis="columns")

df = runall_compute_learn_groves_payment(df_cost_prob,
                                         args.n_sample,
                                         reduce_variables=args.reduce,
                                         confidence_bound=args.CB)
path_name = "./log/random_groves_learn"
if args.reduce:
    path_name += "_reduced"
if args.CB:
    path_name += "_CB"
path = pathlib.Path(path_name)
path.mkdir(parents=True, exist_ok=True)
df.to_csv(path.joinpath("%d_%d_%d.csv" % (args.n, args.n_sample, args.seed)))
