"""
Watts-Strogatz structural indicators for Table 7 of the thesis.

For each rewiring probability p_rew in {0, 0.001, 0.005, 0.01, 0.05, 0.1,
0.5, 1.0}, we draw R = 50 independent connected realisations of
WS(N = 50, K = 6, p_rew) and record the clustering coefficient and the
average shortest-path length.

Diagnostic columns are included in the CSV only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

import config as C
from graphs.generators import watts_strogatz


P_VALUES = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
N_REPS = C.R              
N = C.N                   
K = C.WS_K                


def main() -> None:
    master_rng = np.random.default_rng(C.SEED + 43)
    rows = []
    for p_rew in P_VALUES:
        for r in range(N_REPS):
            seed = int(master_rng.integers(1 << 31))
            G, _ = watts_strogatz(N, K, p_rew, seed)
            # Defensive: pin the connectivity contract from generators.py.
            assert nx.is_connected(G), (
                f"watts_strogatz returned a disconnected graph at "
                f"p_rew = {p_rew}, rep = {r}, seed = {seed}. "
                f"Check graphs/generators.py watts_strogatz."
            )
            deg = np.asarray([d for _, d in G.degree()], dtype=float)
            rows.append({
                "p_rew":           float(p_rew),
                "rep":             r,
                "clustering":      float(nx.average_clustering(G)),
                "avg_path_length": float(nx.average_shortest_path_length(G)),
                "deg_hetero":      float(deg.std() / max(deg.mean(), 1e-12)),
                "mean_degree":     float(deg.mean()),
                "num_edges":       int(G.number_of_edges()),
            })
        print(f"[p_rew={p_rew}] {N_REPS} realisations done", flush=True)

    df = pd.DataFrame(rows)
    out = C.RESULTS / "ws_graph_stats.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    # Aggregated summary (matches the column layout of Table 7).
    agg = df.groupby("p_rew").agg(
        C_mean=("clustering", "mean"),
        C_std=("clustering", "std"),
        L_mean=("avg_path_length", "mean"),
        L_std=("avg_path_length", "std"),
    ).round(4)
    print("\nAggregated (R=50):")
    print(agg.to_string())

    print("\nLaTeX rows (paste into tab:ws-graph-stats):")
    c_row = "$C$  & " + " & ".join(f"${agg.loc[p_rew, 'C_mean']:.3f}$"
                                   for p_rew in P_VALUES) + " \\\\"
    l_row = "$L$  & " + " & ".join(f"${agg.loc[p_rew, 'L_mean']:.2f}$"
                                   for p_rew in P_VALUES) + " \\\\"
    print(c_row)
    print(l_row)


if __name__ == "__main__":
    main()
