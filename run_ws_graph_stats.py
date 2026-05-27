"""
Graph-only structural indicators for the Watts-Strogatz sweep.

For each p in {0, 0.001, 0.01, 0.1, 0.5, 1.0} we draw R=50 independent
realisations of WS(N=50, K=6, p) and record:
    clustering coefficient C
    average shortest-path length L
    diameter
    degree heterogeneity sigma_d / mean_d
    mean degree
    number of edges

No algorithm is run; this fills Table 6 (tab:ws-graph-stats) of Chapter 6.
The |lambda_2| spectral indicator is intentionally NOT reported here because
it is a property of the weight matrix, not of the graph, and is therefore
algorithm-dependent (sync vs async). Spectral indicators belong in Table 7.

Output:
    results/ws_graph_stats.csv      one row per (p, rep)

Usage:
    python run_ws_graph_stats.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

import config as C
from graphs.topologies import watts_strogatz


P_VALUES = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]
N_REPS = C.R              # 50
N = C.N                   # 50
K = C.WS_K                # 6


def main() -> None:
    master_rng = np.random.default_rng(C.SEED)
    rows = []
    for p in P_VALUES:
        for r in range(N_REPS):
            seed = int(master_rng.integers(1 << 31))
            G, _ = watts_strogatz(N, K, p, seed)
            deg = np.asarray([d for _, d in G.degree()], dtype=float)
            rows.append({
                "p":               float(p),
                "rep":             r,
                "clustering":      float(nx.average_clustering(G)),
                "avg_path_length": float(nx.average_shortest_path_length(G)),
                "diameter":        int(nx.diameter(G)),
                "deg_hetero":      float(deg.std() / max(deg.mean(), 1e-12)),
                "mean_degree":     float(deg.mean()),
                "num_edges":       int(G.number_of_edges()),
            })
        print(f"[p={p}] {N_REPS} realisations done", flush=True)

    df = pd.DataFrame(rows)
    out = C.RESULTS / "ws_graph_stats.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    # Aggregated summary
    agg = df.groupby("p").agg(
        C_mean=("clustering", "mean"),
        C_std=("clustering", "std"),
        L_mean=("avg_path_length", "mean"),
        L_std=("avg_path_length", "std"),
        D_mean=("diameter", "mean"),
        D_std=("diameter", "std"),
    ).round(4)
    print("\nAggregated (R=50):")
    print(agg.to_string())


if __name__ == "__main__":
    main()
