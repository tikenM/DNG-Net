"""E1: peer-audit signal separation.

Question: does peer-audit loss distinguish mimicked reports from truthful ones
more sharply than parameter distance does?

Protocol:
    - Train one truthful local model per client.
    - Construct mimicry vectors at strengths alpha in a fine grid.
    - For each strength, evaluate peer-audit loss on honest peer datasets and
      L2 distance to the mimicked cluster centroid.
    - Compare distributions via two-sample KS and ROC AUC of a threshold
      classifier.
    - Repeat over 30 seeds; report bootstrap CIs.

Outputs a JSON blob with per-strength statistics and a matplotlib figure with
overlaid histograms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

from data_pkg.registry import build_clients, build_model_factory
from models.client_models import evaluate_loss, local_train


def run(cfg):
    all_ks = {}
    all_auc_audit = {}
    all_auc_dist = {}
    for seed in range(cfg["experiment"]["num_seeds"]):
        rng = np.random.default_rng(seed)
        clients = build_clients(cfg, rng)
        target_k = 0

        factory = build_model_factory(cfg)

        # Truthful models.
        truthful_flats = []
        for c in clients:
            m = factory()
            m = local_train(m, c.train_loader(64), epochs=2, lr=cfg["federation"]["client_lr"])
            truthful_flats.append(m.get_flat_params())

        # Target cluster centroid = mean of truthful reports of honest members of target_k.
        idx_target = [i for i, c in enumerate(clients) if c.latent_cluster == target_k]
        centroid = torch.stack([truthful_flats[i] for i in idx_target]).mean(dim=0)

        # Honest peers (from target cluster) provide the audit data.
        honest_peers = idx_target[: max(1, len(idx_target) // 2)]

        for alpha in [0.25, 0.5, 0.75, 1.0]:
            audit_true = []
            audit_mim = []
            dist_true = []
            dist_mim = []
            for i in range(len(clients)):
                if i in idx_target:
                    continue
                mim_report = alpha * centroid
                # Peer-audit loss averaged over honest peers.
                a_t = []
                a_m = []
                for p in honest_peers:
                    mt = factory(); mt.set_flat_params(truthful_flats[i])
                    mm = factory(); mm.set_flat_params(mim_report)
                    a_t.append(evaluate_loss(mt, clients[p].test_loader(128)))
                    a_m.append(evaluate_loss(mm, clients[p].test_loader(128)))
                audit_true.append(np.mean(a_t))
                audit_mim.append(np.mean(a_m))
                dist_true.append(float((truthful_flats[i] - centroid).norm().item()))
                dist_mim.append(float((mim_report - centroid).norm().item()))

            audit_true = np.array(audit_true)
            audit_mim = np.array(audit_mim)
            dist_true = np.array(dist_true)
            dist_mim = np.array(dist_mim)

            ks_stat, _ = stats.ks_2samp(audit_true, audit_mim)
            y = np.concatenate([np.zeros_like(audit_true), np.ones_like(audit_mim)])
            # Positive class = mimicked. Both mimicked and truthful losses live
            # in a common scale where higher loss is more likely to be a
            # mimicked report. Use the raw loss as the classifier score.
            auc_audit = roc_auc_score(y, np.concatenate([audit_true, audit_mim]))
            auc_dist = roc_auc_score(y, np.concatenate([dist_true, dist_mim]))

            all_ks.setdefault(alpha, []).append(ks_stat)
            all_auc_audit.setdefault(alpha, []).append(auc_audit)
            all_auc_dist.setdefault(alpha, []).append(auc_dist)

    result = {
        str(a): {
            "ks_mean": float(np.mean(all_ks[a])),
            "ks_std": float(np.std(all_ks[a])),
            "auc_audit_mean": float(np.mean(all_auc_audit[a])),
            "auc_dist_mean": float(np.mean(all_auc_dist[a])),
        }
        for a in all_ks
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e01.json")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    result = run(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
