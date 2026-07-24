# Hyperparameter grids

Each YAML in this directory declares the search space for one axis and the
selection rule used to pick the winning configuration during tuning. Grids
are consumed by `experiments/sensitivity.py` for sweeps and by external
tuning scripts (not included here) for baseline hyperparameter selection.

Selection rule for every grid: best mean ARI on the held-out validation
partition across three seeds; ties broken by lower IR-violation rate.
