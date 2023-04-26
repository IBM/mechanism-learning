# Copyright IBM Corp. 2023
# SPDX-License-Identifier: Apache2.0
#
# This will run all experiments in Osogami et al. (2022).
# Takayuki Osogami, Segev Wasserkrug, Elisheva S. Shamash, "Mechanism Learning for Trading Networks", arXiv:2208.09222, 2022

for n in 16 32 64 128 256 512 1024 2048 4096
do
    python exp_random_groves_learn.py 10000 $n 1 --CB
    python exp_random_groves_learn.py 10000 $n 1 --reduce --CB
done
