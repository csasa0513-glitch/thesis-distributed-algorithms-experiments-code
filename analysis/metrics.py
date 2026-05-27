"""
Structural graph and weight-matrix indicators used in RQ3.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from numpy.typing import NDArray


def spectral_gap(W: NDArray) -> float:
    """1 - lambda_2(W), where eigenvalues are sorted by magnitude."""
    eig = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
    # eig[0] = 1, we want 1 - |eig[1]|
    return float(1.0 - eig[1])


def clustering(G: nx.Graph) -> float:
    return float(nx.average_clustering(G))


def degree_heterogeneity(G: nx.Graph) -> float:
    """sigma_d / mean_d."""
    deg = np.asarray([d for _, d in G.degree()], dtype=float)
    return float(deg.std() / max(deg.mean(), 1e-12))


def diameter(G: nx.Graph) -> int:
    return int(nx.diameter(G))


def avg_path_length(G: nx.Graph) -> float:
    """Average shortest-path length L(G) of the unweighted graph."""
    return float(nx.average_shortest_path_length(G))


# --------------------------------------------------------------------------
# Asynchronous gossip spectral indicators (Boyd-Gossip 2006; Koshal 2016)
# --------------------------------------------------------------------------
def expected_gossip_matrix(G):
    """
    Expected one-event averaging matrix E[W(k)] under Koshal's gossip
    protocol (Sec. 4.1, p. 689):

        I^k ~ Uniform({1, ..., N})       => P(I = i)        = 1/N
        J^k | I^k = i ~ Uniform(N_i)     => P(J = j | I=i)  = 1/d_i

    Edge (i,j) activated => state left-multiplied by
        W_ij = I - (1/2)(e_i - e_j)(e_i - e_j)^T.

    Expectation:

        E[W(k)] = I - (1 / (2N)) *
                  sum over edges (i,j) of (1/d_i + 1/d_j)
                                          * (e_i - e_j)(e_i - e_j)^T.

    This is the weighted-Laplacian form of Boyd-Gossip 2006, eq. (7).
    E[W(k)] is symmetric, doubly stochastic, and PSD (convex combination
    of PSD single-edge averaging matrices), so all eigenvalues lie in [0, 1].
    """
    N = G.number_of_nodes()
    deg = dict(G.degree())
    EW = np.eye(N)
    for i, j in G.edges():
        w_ij = 1.0 / deg[i] + 1.0 / deg[j]
        e = np.zeros(N)
        e[i] = 1.0
        e[j] = -1.0
        EW -= (w_ij / (2.0 * N)) * np.outer(e, e)
    return EW


def lambda2_async(G):
    """
    Algebraic second-largest eigenvalue of E[W(k)] under Koshal's gossip.
    E[W(k)] is PSD with the largest eigenvalue 1 (consensus direction),
    so the second-largest controls per-event mean-square consensus rate:
        E[ || v(k) - mean(v(k)) ||^2 ] ~ lambda_2^k.
    """
    EW = expected_gossip_matrix(G)
    eig = np.linalg.eigvalsh(EW)          # ascending order
    return float(eig[-2])                 # second-largest eigenvalue


def rho_async(G):
    """
    Per-event amplitude convergence rate, sqrt(lambda_2(E[W(k)])).

    Matches Koshal, Nedic, Shanbhag (2016, p. 701, Table 9), who report
    the square root of the second-largest eigenvalue of the expected
    weight matrix, putting it on the same amplitude scale as the
    synchronous |lambda_2(W)|.

    max(., 0.0) guards against tiny floating-point negatives near zero;
    theoretically lambda_2 >= 0 always for this protocol.
    """
    return float(np.sqrt(max(lambda2_async(G), 0.0)))


def summary(G, W):
    return {
        "spectral_gap":     spectral_gap(W),
        "clustering":       clustering(G),
        "avg_path_length":  avg_path_length(G),
        "deg_hetero":       degree_heterogeneity(G),
        "diameter":         diameter(G),
        "mean_degree":      float(np.mean([d for _, d in G.degree()])),
        "num_edges":        G.number_of_edges(),
    }
