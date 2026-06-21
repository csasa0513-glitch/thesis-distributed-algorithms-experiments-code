"""
Graph generators and doubly stochastic weight matrices.

Every function returns a tuple (G, W) where
    G  :  networkx.Graph                (undirected, connected, no self-loops)
    W  :  numpy.ndarray, shape (N, N)   (doubly stochastic, symmetric)

We use the weight rule of Koshal, Nedic, Shanbhag (2016, Sec. 6.2, p. 699)
so that the numerical experiments in Chapter 6 can be compared directly to
their Tables 3-6:

    W_ij = delta            if (i,j) in E
    W_ii = 1 - delta * d_i
    W_ij = 0                otherwise
    delta = 0.5 / max_i d_i

This W is symmetric and doubly stochastic for any undirected connected
graph G. The positive diagonal (W_ii >= 0.5) guarantees aperiodicity, so
the scheme works for bipartite graphs (e.g., even cycle, grid) as well.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from numpy.typing import NDArray


# --------------------------------------------------------------------------
# Koshal weight matrix (constant edge weight delta)
# --------------------------------------------------------------------------
def koshal_weights(G: nx.Graph) -> NDArray:
    """
    Koshal et al. (2016, p. 699) weights:
        delta = 0.5 / max_i d_i,
        W_ij = delta            if (i,j) in E,
        W_ii = 1 - delta * d_i,
        W_ij = 0                otherwise.
    """
    N = G.number_of_nodes()
    deg = dict(G.degree())
    d_max = max(deg.values())
    delta = 0.5 / d_max
    W = np.zeros((N, N))
    for i, j in G.edges():
        W[i, j] = W[j, i] = delta
    for i in G.nodes():
        W[i, i] = 1.0 - delta * deg[i]
    return W


# --------------------------------------------------------------------------
# Regular graphs  (baseline for RQ1)
# --------------------------------------------------------------------------
def cycle(N: int):
    G = nx.cycle_graph(N)
    return G, koshal_weights(G)


def wheel(N: int):
    """
    Wheel graph with N total nodes: one central vertex connected to all
    N-1 peripheral vertices. The peripheral vertices have NO edges among
    themselves. This is the graph called "wheel" in Koshal et al. (2016).
    """
    G = nx.star_graph(N - 1)  # nx.star_graph(n) returns n+1 nodes
    return G, koshal_weights(G)


def complete(N: int):
    G = nx.complete_graph(N)
    return G, koshal_weights(G)


def grid(N: int):
    """
    Two-dimensional grid with 5 columns (Koshal et al. 2016, Sec. 6.3).
    The grid has N/5 rows and 5 columns. Vertex (corner) nodes have 2
    neighbors, edge nodes have 3, and interior nodes have 4. Requires N
    divisible by 5; Koshal uses N in {20, 50} which gives 4x5 and 10x5.
    """
    if N % 5 != 0:
        raise ValueError(f"Grid requires N divisible by 5; got N={N}")
    rows = N // 5
    G = nx.grid_2d_graph(rows, 5)
    G = nx.convert_node_labels_to_integers(G)
    return G, koshal_weights(G)


# --------------------------------------------------------------------------
# Complex graphs (main contribution: RQ2)
# --------------------------------------------------------------------------
def watts_strogatz(N: int, K: int, p: float, seed: int | None = None):
    """
    Watts-Strogatz small-world.  Start from a ring lattice with degree K
    (each node connected to its K nearest neighbours), rewire each edge
    with probability p. Uppercase K matches Watts & Strogatz (1998) and
    avoids clashing with the iteration index k used elsewhere in the code.

    Special case p = 0 returns the pure ring lattice (no rewiring), which
    is always connected. For p > 0 we use `connected_watts_strogatz_graph`
    which rejection-samples until the result is connected, so W is well
    defined.
    """
    if p <= 0.0:
        # Pure ring lattice; no rewiring, deterministic and connected.
        G = nx.watts_strogatz_graph(N, K, 0.0, seed=seed)
    else:
        G = nx.connected_watts_strogatz_graph(N, K, p, tries=100, seed=seed)
    return G, koshal_weights(G)


# --------------------------------------------------------------------------
# Registry used by the run_ scripts
# --------------------------------------------------------------------------
# Five-graph baseline (Chapter 6 of the thesis): four regular graphs
# (cycle, wheel, grid, complete) reproducing Koshal et al. (2016) Figure 4,
# plus one complex graph family (Watts-Strogatz) for the sensitivity sweep.
# Note: this "wheel" graph is structurally a star (one center + N-1
# leaves, no peripheral edges); the naming follows the thesis / Koshal (2016).
REGULAR = {
    "cycle":    cycle,
    "wheel":    wheel,
    "grid":     grid,
    "complete": complete,
}
