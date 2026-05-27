"""
Algorithm 2 from Koshal, Nedic, Shanbhag (2016):
Asynchronous gossip-based distributed scheme for aggregative games.

At each *gossip event* k we follow the gossip protocol of Koshal et al.
(2016), Section 4.1, p. 689:

  1. Pick the active agent I^k uniformly from {1, ..., N}.
     (This is the equivalence of the superposed local Poisson clocks
     of rate 1 with a single global clock of rate N.)
  2. I^k contacts a neighbor J^k drawn uniformly from N_{I^k}, i.e. with
     probability p_{I^k j} = 1/d_{I^k} for every j in N_{I^k}. This is
     the simplest valid choice in Koshal's framework, which only
     requires p_{ij} > 0.
  3. I^k and J^k average their local aggregate estimates:
         v_i^+ = v_j^+ = 0.5 * (v_i + v_j)
  4. Both endpoints perform a local projected-gradient step with their
     own stepsize indexed by the local update counter Gamma_i:
         x_l^{k+1} = Pi_{X_l}( x_l - tau(Gamma_l) * F_l(x_l, v_l^+) )
     and update the innovation term v_l^{k+1} = v_l^+ + (s_l^{k+1} - s_l^k).
  5. All other players are idle.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from numpy.typing import NDArray

import config as C
from games.nash_cournot import F_i, project_Xi, aggregate


def run(G: nx.Graph,
        x_star: NDArray,
        max_events: int = C.MAX_GOSSIP,
        eps: float = C.EPS_TOL,
        record_every: int = 200,
        rng: np.random.Generator | None = None,
        x0: NDArray | None = None) -> dict:
    """
    Run the asynchronous gossip algorithm on graph G with the diminishing
    stepsize alpha_{k,i} = 9 / Gamma_i^k (Koshal 2016, Sec. 6.2).

    Parameters
    ----------
    x0 : (N, 2S) or None
        Initial decision. If None, a random initial point is sampled
        uniformly from [0, C.INIT_SCALE]^{2S} per player and then
        projected onto each player's feasible set X_i, using `rng`.

    Notes
    -----
    `max_events` counts individual gossip events, *not* rounds. On a graph
    with |E| edges one "sweep" corresponds to roughly |E| events.
    """
    if rng is None:
        rng = np.random.default_rng(C.SEED)

    # Precompute neighbor lists once. Koshal's protocol (Sec. 4.1, p. 689)
    # picks the active agent I^k uniformly from {0, ..., N-1} and then its
    # contacted neighbor J^k uniformly from N_{I^k}.
    neighbors_list = [np.asarray(list(G.neighbors(i)), dtype=int) for i in range(C.N)]
    N, L = C.N, C.L

    # Initial decision: sample uniformly and project onto X_i.
    if x0 is None:
        raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
        x = np.empty_like(raw)
        for i in range(N):
            x[i] = project_Xi(i, raw[i])
    else:
        x = np.asarray(x0, dtype=float).reshape(N, 2 * L).copy()

    v = x[:, L:].copy()                 # v_i^0 = s_i^0
    s_prev = x[:, L:].copy()
    gamma = np.zeros(N, dtype=int)      # local update counters

    rel_err, cons_err, events = [], [], []

    for k in range(max_events):
        # Koshal Sec. 4.1, p. 689:
        #   I^k ~ Uniform({0, ..., N-1})
        #   J^k ~ Uniform(N_{I^k})           (p_{ij} = 1/d_i, the simplest
        #                                     valid choice; only positivity
        #                                     is required by Koshal Prop. 6.)
        i = int(rng.integers(N))
        nbrs = neighbors_list[i]
        if nbrs.size == 0:                  # isolated node (never happens
            continue                        # on a connected graph)
        j = int(rng.choice(nbrs))

        # 1. Consensus on v
        avg = 0.5 * (v[i] + v[j])
        v[i] = v[j] = avg

        # 2. Local projected-gradient steps with diminishing stepsize
        #    alpha_{k,l} = 9 / Gamma_l^k  (Koshal 2016, Sec. 6.2)
        for l in (i, j):
            gamma[l] += 1
            tau = C.STEP_DIM_NUM / gamma[l]
            grad = F_i(l, x[l], N * v[l])
            x_new_l = project_Xi(l, x[l] - tau * grad)
            # innovation
            v[l] = v[l] + (x_new_l[L:] - s_prev[l])
            s_prev[l] = x_new_l[L:]
            x[l] = x_new_l

        # 3. Book-keeping
        if k % record_every == 0 or k == max_events - 1:
            # Sample error per Koshal et al. (2016), eq. (65):
            # relative max-norm error. Matches sync_koshal.py for a
            # consistent metric across both algorithms.
            denom = max(np.max(np.abs(x_star)), 1e-12)
            e = float(np.max(np.abs(x.ravel() - x_star)) / denom)
            d = np.mean(np.linalg.norm(N * v - aggregate(x.ravel()), axis=1))
            rel_err.append(e)
            cons_err.append(d)
            events.append(k)
            if e < eps:
                break

    return {
        "x_final": x.ravel(),
        "events":  np.asarray(events),
        "rel_err": np.asarray(rel_err),
        "cons_err": np.asarray(cons_err),
        "k_eps":   events[-1] if rel_err and rel_err[-1] < eps else None,
    }
