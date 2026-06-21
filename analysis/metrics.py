"""
Graph indicators used in Chapter 6.

Assumes a connected undirected graph. The sync indicator is computed
from a symmetric weight matrix, and the async indicator is computed
from the expected gossip matrix.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from numpy.typing import NDArray


def sync(W: NDArray) -> float:
    """Return the sync spectral indicator."""
    assert W.ndim == 2 and W.shape[0] == W.shape[1], \
        f"sync(W) requires a square matrix; got shape {W.shape}."
    assert np.allclose(W, W.T, atol=1e-9), \
        "sync(W) requires a symmetric weight matrix."

    eig = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
    return float(eig[1])


def CG(G: nx.Graph) -> float:
    """Return the average clustering coefficient."""
    return float(nx.average_clustering(G))


def LG(G: nx.Graph) -> float:
    """Return the average shortest-path length."""
    assert nx.is_connected(G), "LG(G) requires a connected graph."
    return float(nx.average_shortest_path_length(G))


def mean_degree(G: nx.Graph) -> float:
    """Return the mean node degree."""
    return float(np.mean([d for _, d in G.degree()]))


def num_edges(G: nx.Graph) -> int:
    """Return the number of edges."""
    return int(G.number_of_edges())


def expected_gossip_matrix(G: nx.Graph) -> NDArray:
    """Return the expected one-step gossip matrix."""
    assert isinstance(G, nx.Graph) and not G.is_directed(), \
        "expected_gossip_matrix(G) requires an undirected graph."
    assert nx.is_connected(G), \
        "expected_gossip_matrix(G) requires a connected graph."

    N = G.number_of_nodes()
    assert set(G.nodes()) == set(range(N)), \
        "expected_gossip_matrix(G) requires node labels 0..N-1."

    deg = dict(G.degree())
    EW = np.eye(N)

    for i, j in G.edges():
        w_ij = 1.0 / deg[i] + 1.0 / deg[j]
        e = np.zeros(N)
        e[i] = 1.0
        e[j] = -1.0
        EW -= (w_ij / (2.0 * N)) * np.outer(e, e)

    return EW


def async_(G: nx.Graph) -> float:
    """Return the async spectral indicator."""
    EW = expected_gossip_matrix(G)
    eig = np.linalg.eigvalsh(EW)
    return float(np.sqrt(max(eig[-2], 0.0)))


def summary(G: nx.Graph, W: NDArray) -> dict:
    """Return the indicators reported in Chapter 6."""
    return {
        "sync": sync(W),
        "async": async_(G),
        "CG": CG(G),
        "LG": LG(G),
        "mean_degree": mean_degree(G),
        "num_edges": num_edges(G),
    }
