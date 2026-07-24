# DNG-Net Experimental Codebase

Reference implementation for the paper "Differentiable Normative Guidance for
Nash Bargaining Solution Recovery in Clustered Federated Learning".

The codebase is organized around the experimental framework described in the
accompanying document. Every experiment listed in Steps 3, 5, 6, and 7 of that
framework has an entry point in `experiments/`.

## Structure

```
dng_net/
  models/          DNG-Net architecture and client models
  mechanisms/      NBS recovery layer, peer-audit graph, projections
  attacks/         mimicry, free-riding, collusion, best-response search
  data_pkg/        synthetic mixture, FEMNIST, CIFAR-10 Dirichlet, Shakespeare
  metrics/         game-theoretic and clustering metrics
  experiments/     experiment scripts (E1 through E14, ablations, sensitivity)
  configs/         yaml configs per experiment
```

## Reproducing Results

```
pip install -r requirements.txt
python -m experiments.run_all --config configs/base.yaml
```

Individual experiments:

```
python -m experiments.e01_signal_separation --config configs/e01.yaml
python -m experiments.e02_admission_rate     --config configs/e02.yaml
python -m experiments.e03_core_audit         --config configs/e03.yaml
...
```

All headline numbers are collected over ten seeds. Confidence intervals are
computed by bootstrap resampling. Statistical tests and effect sizes are
computed in `metrics/statistics.py`.

## Assumptions Reported per Dataset

The theorem-validation script `experiments/theorem_validation.py` measures the
Definition 1 risk margin and the fraction of rounds with non-positive individual
surplus. Both are reported alongside every headline result.

## Note

No pretrained models are used. Every hyperparameter grid is published under
`configs/grids/`. Hardware profile and container image are described in
`REPRODUCIBILITY.md`.
