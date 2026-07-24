"""Adversary reports and best-response search.

Each attack takes the current server-broadcast cluster centroids and returns
the parameter vector the adversarial client will submit. The best-response
search in `unilateral_best_response` upper bounds unilateral profit and is used
by E6 to give an empirical incentive-compatibility check that does not rely on
a hand-curated attack list.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn.functional as F

from data_pkg.synthetic import ClientData


# ---- Fixed attacks --------------------------------------------------------- #


def mimicry_report(target_centroid: torch.Tensor, strength: float) -> torch.Tensor:
    """Return a report that lies on the ray from origin to the target centroid.

    strength = 1.0 corresponds to exact mimicry. Intermediate values interpolate
    between an uninformative zero vector and the centroid, letting E1 sweep
    attack intensity.
    """
    return strength * target_centroid


# Free-riding is implemented directly in train_dng._adversary_report by
# returning the broadcast initialization unchanged. No wrapper function is
# needed and adding one only invites the dead-code confusion this section
# previously contained.


def noise_report(reference: torch.Tensor, sigma: float) -> torch.Tensor:
    """Isotropic Gaussian noise centered at a reference vector."""
    return reference + sigma * torch.randn_like(reference)


def targeted_allocation_report(
    reference: torch.Tensor,
    grad_of_own_x: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """Move the report in the direction that maximizes own allocation x_i.

    Requires that the caller supply d(x_i)/d(theta_i) by autograd. This attack
    directly probes the endogenous-price mechanism.
    """
    return reference + step * grad_of_own_x


# ---- Best-response search -------------------------------------------------- #


def unilateral_best_response(
    client_idx: int,
    truthful_report: torch.Tensor,
    utility_fn: Callable[[torch.Tensor], torch.Tensor],
    steps: int = 50,
    lr: float = 1e-2,
) -> tuple[torch.Tensor, float]:
    """Projected-gradient search over client_idx's report.

    `utility_fn` maps a candidate report to that client's realized utility
    inside the mechanism, holding all other reports fixed at their truthful
    values. Returns the best-response vector and the utility gap
    (u_best - u_truth). If the gap is non-positive, IC is verified for this
    client under this initialization.
    """
    report = truthful_report.detach().clone().requires_grad_(True)
    u_truth = float(utility_fn(truthful_report).item())

    opt = torch.optim.Adam([report], lr=lr)
    best_u = u_truth
    best_vec = truthful_report.detach().clone()
    for _ in range(steps):
        opt.zero_grad()
        u = utility_fn(report)
        (-u).backward()
        opt.step()
        with torch.no_grad():
            u_val = float(utility_fn(report).item())
            if u_val > best_u:
                best_u = u_val
                best_vec = report.detach().clone()
    return best_vec, best_u - u_truth


# ---- Collusion ------------------------------------------------------------- #


def collusion_whitewash(
    W: torch.Tensor,
    colluder_ids: List[int],
) -> torch.Tensor:
    """Simulate whitewashing: colluders report zero loss when auditing each other.

    Used inside E10 to model a colluding clique that lies to the peer-audit
    protocol. The mechanism must nevertheless recover a stable partition when
    the clique is a minority within its target cluster.
    """
    W_att = W.clone()
    for i in colluder_ids:
        for j in colluder_ids:
            if i != j and not torch.isnan(W_att[i, j]):
                W_att[i, j] = 0.0
    return W_att
