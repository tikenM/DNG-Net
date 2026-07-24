# Reproducibility

## Hardware and OS

- Apple Silicon (M-series) via MPS backend, macOS 14 or later
- Linux with CUDA 12 GPUs
- Linux CPU-only for small runs

Device selection is automatic. Force it with:

```
export DNG_DEVICE=mps      # or cuda, cpu
```

## Apple Silicon (M4) specifics

Recent PyTorch versions run Gumbel-Softmax, Adam, and standard convolutions on
MPS. For any op that is not yet implemented, PyTorch will fall back to CPU if
this environment variable is set:

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Set this once per shell before running any experiment.

Peak throughput on M4:

- Cache all per-client test tensors on device once (already done in
  `_build_audit_and_M`).
- Prefer `local_epochs: 1` and larger `batch_size` values. MPS overhead per
  kernel launch dominates on small batches.
- Reduce `peer_audit_fanout_c` for early debugging runs; increase for the
  final experiments.

## Software versions

Recorded in `requirements.txt`. Pin exact versions before submission with:

```
pip freeze > requirements.lock.txt
```

## Determinism

Every experiment script accepts `--config` and reads `experiment.seed`. Setting
this seed reproduces a run bit-for-bit only on the same device backend. The
Gumbel-Softmax uses a stochastic sample, so runs are seed-dependent by design.

## What was fixed in this revision

- Partition MLP now receives gradient from the Nash-product loss.
- Core deviation gain uses the correct sum-based coalition value contract.
- IR helper is called from `experiments/e04_e05_ir_bb.py` and includes the
  transfer in the utility.
- Peer-audit and utility now live in matching units (both are cross-entropy
  loss).
- `free_ride_report` and the double-negation ROC construction were removed.
- `mechanisms/nbs_recovery.py` supports a population-independent partition
  MLP via `partition_feature_mode: stats`.
- `metrics/statistics.py` holds the statistical utilities as advertised in
  the framework document.

## Status

All experiment scripts, ablation driver, and sensitivity driver are present
and smoke-tested. Shakespeare loader and CharLSTMClassifier are implemented.
Hyperparameter grids are in `configs/grids/`.

### Files added in this revision

- `experiments/e02_admission_rate.py`
- `experiments/e06_best_response.py`
- `experiments/e07_gumbel_annealing.py`
- `experiments/e08_communication_scaling.py`
- `experiments/e09_server_compute_scaling.py`
- `experiments/e10_collusion_sweep.py`
- `experiments/e11_latent_recovery.py`
- `experiments/e12_surplus_quantification.py`
- `experiments/e13_scalability.py`
- `experiments/e14_cross_dataset.py`
- `experiments/ablations.py`     (A1-A10)
- `experiments/sensitivity.py`   (S1-S10)
- `experiments/_common.py`       (uniform reporting harness)
- `data_pkg/shakespeare.py`
- `models/client_models.py:CharLSTMClassifier`
- `configs/grids/` (5 grid files + README)
- `configs/shakespeare.yaml`
- `configs/e14_cross_dataset.yaml`

### Known runtime notes

- E2, E10, E11 invoke IFCA, FeSEM, and CFL baselines. Runtime scales with
  clients times rounds times baselines.
- E6 uses a numerical projected-gradient search. Set `evaluation.br_steps`
  to trade IC-tightness against runtime.
- E13 with N=1000 requires memory and time proportional to N times K plus
  the audit fan-out. On M4, allow a few minutes per cell.
- Ablations and sensitivity drivers re-run `train_dng_net` once per branch
  or per sweep point. A twelve-branch ablation takes roughly twelve times a
  single training run.
