"""
Empirical (real-world) communication graphs used in Section 6.6 of the
thesis. Each loader returns (G, W) with the same conventions as
`graphs.topologies`:
    G  : networkx.Graph (undirected, connected, no self-loops)
    W  : numpy.ndarray  (Koshal et al. 2016 doubly stochastic weight matrix)

Currently provided:
    karate_club : Zachary's Karate Club (N = 34).
"""
from __future__ import annotations

import networkx as nx
from numpy.typing import NDArray

from graphs.topologies import koshal_weights


def karate_club() -> tuple[nx.Graph, NDArray]:
    """
    Zachary's Karate Club (Zachary 1977).

    34 nodes, 78 undirected edges. Two ground-truth communities centered
    at nodes 0 (Mr. Hi) and 33 (the officer); the graph is connected and
    is among the most studied empirical networks in network science.

    Returned without any node-attribute information (the doubly stochastic
    W only needs the adjacency).
    """
    G = nx.karate_club_graph()
    # Sanity check (networkx ships a connected version)
    if not nx.is_connected(G):
        raise RuntimeError("karate_club_graph is unexpectedly disconnected")
    return G, koshal_weights(G)


# Registry used by run_real_network.py
REAL = {
    "karate_club": karate_club,
}
