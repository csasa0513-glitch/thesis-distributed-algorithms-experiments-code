"""
Cournot-specific asynchronous gossip specialization
====================================================

This is Algorithm 2 from Koshal et al. (2016, Sec. 4.1),
specialized to the Nash-Cournot setting (Sec. 6.1).

In the Cournot model, the player gradient depends on the aggregate only
through the sales vector \(\bar{s} = \sum_j s_j\). Therefore, we track
only the sales aggregate instead of the full decision aggregate:

    v_i^k in R^L
    v_i^0 = s_i^0
    v_i approximates \bar{s}^k / N
    F_i(x_i, N v_i^+) = F_i(x_i, \bar{s})

Gossip protocol:
    1. Select an active agent I^k uniformly from {0, ..., N-1}.
    2. Select a neighbor J^k uniformly from N_{I^k}.
    3. I^k and J^k average their sales estimates.
    4. Both agents perform a projected-gradient step with stepsize
       alpha_{k,l} = 9 / Gamma_l^k.
    5. All other agents remain idle.
"""
    
from __future__ import annotations

import numpy as np
import networkx as nx
from numpy.typing import NDArray

import config as C
from games.nash_cournot import F_i, project_Xi, aggregate


def run(G: nx.Graph,
        x_star: NDArray,
        max_events: int = 200_000,
        eps: float = C.EPS_TOL,
        record_every: int = 200,
        rng: np.random.Generator | None = None,
        x0: NDArray | None = None,
        stop_metric: str = "rel_err",
        step_scale: float = C.STEP_DIM_NUM) -> dict:
    """
Run the asynchronous gossip algorithm on graph G with diminishing
stepsize alpha_{k,i} = step_scale / Gamma_i^k. In the thesis
experiments, the default value step_scale = 9 follows Koshal et al.
(2016).

Parameters
----------
x0 : (N, 2L) or None
    Initial decision. If None, a random initial point is sampled from
    [0, C.INIT_SCALE]^{2L} per player and projected onto X_i.
step_scale : float
    Numerator c in alpha_{k,i} = c / Gamma_i^k.
stop_metric : {"rel_err", "concurrence"}
    Early-stopping criterion. "rel_err" uses the sample-path error
    ||x^k - x*||_inf / ||x*||_inf. "concurrence" uses
    max_i ||N v_i - bar_s_star||.
max_events : int
    Number of gossip events to simulate.
"""
    assert stop_metric in {"rel_err", "concurrence"}, \
        f"stop_metric must be 'rel_err' or 'concurrence'; got {stop_metric!r}."
    assert record_every >= 1, \
        f"record_every must be a positive integer; got {record_every}."
    if rng is None:
        rng = np.random.default_rng(C.SEED)

    # --- Graph sanity checks (Koshal Assumption 7) ---
    assert isinstance(G, nx.Graph) and not G.is_directed()
    assert nx.is_connected(G)
    assert nx.number_of_selfloops(G) == 0
    assert G.number_of_nodes() == C.N, f"Expected {C.N} nodes, got {G.number_of_nodes()}."
    assert set(G.nodes()) == set(range(C.N)), "Node labels must be 0..N-1."

    neighbors_list = [np.asarray(list(G.neighbors(i)), dtype=int) for i in range(C.N)]
    N, L = C.N, C.L

    # Initial decision
    if x0 is None:
        raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
    else:
        raw = np.asarray(x0, dtype=float).reshape(N, 2 * L)
    x = np.empty_like(raw)
    for i in range(N):
        x[i] = project_Xi(i, raw[i])

    v = x[:, L:].copy()                 # v_i^0 = s_i^0
    s_prev = x[:, L:].copy()
    gamma = np.zeros(N, dtype=int)      

    s_star_per_agent = x_star.reshape(N, 2 * L)[:, L:]
    bar_s_star = s_star_per_agent.sum(axis=0)

    rel_err, cons_err, concurrence_err, events = [], [], [], []

    for k in range(max_events):
        i = int(rng.integers(N))
        nbrs = neighbors_list[i]
        if nbrs.size == 0:                  
            continue                        
        j = int(rng.choice(nbrs))

        avg = 0.5 * (v[i] + v[j])
        v[i] = v[j] = avg

        for l in (i, j):
            gamma[l] += 1
            alpha_ki = step_scale / gamma[l]
            grad = F_i(l, x[l], N * v[l])
            x_new_l = project_Xi(l, x[l] - alpha_ki * grad)
            # innovation
            v[l] = v[l] + (x_new_l[L:] - s_prev[l])
            s_prev[l] = x_new_l[L:]
            x[l] = x_new_l

        events_done = k + 1
        if events_done % record_every == 0 or events_done == max_events:
            denom = max(np.max(np.abs(x_star)), 1e-12)
            e = float(np.max(np.abs(x.ravel() - x_star)) / denom)
            d = np.mean(np.linalg.norm(N * v - aggregate(x.ravel()), axis=1))
            c = float(np.max(np.linalg.norm(N * v - bar_s_star, axis=1)))
            rel_err.append(e)
            cons_err.append(d)
            concurrence_err.append(c)
            events.append(events_done)
            stop_value = c if stop_metric == "concurrence" else e
            if stop_value < eps:
                break

    # Last recorded value of whichever metric triggered the stop.
    last_stop_value = (concurrence_err[-1] if stop_metric == "concurrence"
                       else rel_err[-1]) if events else None
    return {
        "x_final":         x.ravel(),
        "events":          np.asarray(events),
        "rel_err":         np.asarray(rel_err),
        "cons_err":        np.asarray(cons_err),
        "concurrence_err": np.asarray(concurrence_err),
        "k_eps":           events[-1] if last_stop_value is not None
                                       and last_stop_value < eps else None,
    }
