<!-- This should be the location of the title of the repository, normally the short name -->
# Mechanism-Learning

<!-- Build Status, is a great thing to have at the top of your repository, it shows that you take your CI/CD as first class citizens -->
<!-- [![Build Status](https://travis-ci.org/jjasghar/ibm-cloud-cli.svg?branch=master)](https://travis-ci.org/jjasghar/ibm-cloud-cli) -->

<!-- Not always needed, but a scope helps the user understand in a short sentance like below, why this repo exists -->
## Scope

Python code for reproducing the experimental results reported in Takayuki Osogami, Segev Wasserkrug, and Elisheva Shamash, "Learning Efficient Truthful Mechanisms for Trading Networks", in Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI-23).

<!-- A more detailed Usage or detailed explaination of the repository here -->
## Usage

To install the package:

```
conda env create -n mechlearn -f environment.yml
conda activate mechlearn
pip install .
```

To install the package:
```
cd exp
./exp.sh
```

To draw figures in the paper, run the jupyter notebooks under `notebooks`.

## Authors

- Author: Takayuki Osogami <osogami@jp.ibm.com>

[issues]: https://github.com/IBM/repo-template/issues/new
