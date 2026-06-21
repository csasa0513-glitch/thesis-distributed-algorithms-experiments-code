"""
Compute |lambda_2(W)| for each graph in the five-graph setup.

This script generates the data behind Table 1 in Chapter 6 of the thesis.
|lambda_2(W)| is the second-largest eigenvalue (in magnitude) of the
weight matrix W defined by Koshal et al. (2016, p. 699):
    delta = 0.5 / max_i d_i,  W_ij = delta if (i,j) in E,
    W_ii = 1 - delta * d_i.
By Xiao and Boyd (2003, Theorem 1), |lambda_2| controls the asymptotic
convergence rate of the consensus iteration on a static graph.

For the deterministic graphs (cycle, wheel, grid, complete) we report
one value per N. For the random graph (WS) we sample R independent
graphs per N and report the mean and standard deviation over the R
realisations.

Output:
    results/spectral_gap.csv   long-format CSV
        columns: N, graph, mean, std, n_samples

Usage:
    python run_spectral.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from graphs.generators import cycle, wheel, grid, complete, watts_strogatz


def lambda2(W: np.ndarray) -> float:
    """Second-largest eigenvalue of W in magnitude."""
    eig = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
    return float(eig[1])   # eig[0] = 1, eig[1] = |lambda_2|


def deterministic_value(N: int, gen) -> float:
    """Compute lambda_2 for a deterministic graph."""
    _, W = gen(N)
    return lambda2(W)


def random_values(N: int, gen, R: int, seed_base: int) -> np.ndarray:
    """Compute lambda_2 for R independent realisations of a random graph."""
    values = np.empty(R)
    for r in range(R):
        _, W = gen(N, seed=seed_base + r)
        values[r] = lambda2(W)
    return values


def main() -> None:
    R = C.R    # use the same R as the experiments (default 50)
    N_sizes = (20, 50)

    rows = []

    print("=" * 60, flush=True)
    print(f"Computing |lambda_2(W)| for the five-graph setup",
          flush=True)
    print(f"  R (random graph samples) = {R}", flush=True)
    print("=" * 60, flush=True)

    for N in N_sizes:
        # ---- Deterministic graphs ----
        for name, gen in [("cycle", cycle),
                          ("wheel", wheel),
                          ("grid", grid),
                          ("complete", complete)]:
            val = deterministic_value(N, gen)
            rows.append({
                "N":          N,
                "graph":   name,
                "mean":       val,
                "std":        0.0,
                "n_samples":  1,
            })
            print(f"[N={N}]  {name:9s}  |lambda_2| = {val:.4f}", flush=True)

        # ---- Random graph (WS) ----
        ws_gen = lambda N_val, seed: watts_strogatz(N_val, C.WS_K, 0.1,
                                                    seed=seed)

        for name, gen, seed_base in [
            ("WS(p=0.1)", ws_gen, C.SEED + 10_000 * N + 7),
        ]:
            vals = random_values(N, gen, R, seed_base)
            rows.append({
                "N":          N,
                "graph":   name,
                "mean":       float(vals.mean()),
                "std":        float(vals.std(ddof=1)),
                "n_samples":  R,
            })
            print(f"[N={N}]  {name:9s}  <|lambda_2|> = {vals.mean():.4f} "
                  f"+- {vals.std(ddof=1):.4f}", flush=True)

    df = pd.DataFrame(rows)
    out_path = C.RESULTS / "spectral_gap.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    # Pretty print for direct copy into LaTeX Table 1
    print("\n=== Table 1: |lambda_2(W)| by graph and N ===")
    pivot = df.pivot(index="graph", columns="N", values="mean")
    # Preserve the row order used in the thesis (Table 1)
    desired_order = ["cycle", "wheel", "grid", "complete", "WS(p=0.1)"]
    pivot = pivot.reindex([t for t in desired_order if t in pivot.index])
    print(pivot.round(4))


if __name__ == "__main__":
    main()
