"""
Table 1 of the thesis
===============================

Table 1: Centralized synchronous baseline on the static complete graph.

Table 1 reports the mean error and the width of the 90% confidence
interval after k iterations.

In the thesis, this baseline is described as the "Centralized synchronous
baseline on the static complete graph". It is not introduced as a separate
algorithm. Instead, it is the synchronous distributed algorithm specialized
to a static complete graph, where every player has exact access to the
aggregate at every iteration.

On the complete graph, the mixing step makes all local aggregate estimates
identical after one round, so the aggregate used in the update is exact.
Therefore, the implementation below uses the true aggregate directly instead
of explicitly forming the complete-graph weight matrix and updating v.

Algorithm
---------
For each player i in parallel, at every iteration k = 0, 1, ...
    bar_s = sum_j s_j^k
    x_i^{k+1} = Pi_{X_i}(x_i^k - alpha_k F_i(x_i^k, bar_s))

with stepsize
    alpha_k = 1 / (k + 1).

The only source of variability across sample paths is the random initial
point
    x_i^0 ~ Proj_{X_i}(U[0, INIT_SCALE]^{2L}).

For each fixed horizon k, the script records the relative error used in the
thesis at that iteration count and then reports the sample mean and the
width of the 90% confidence interval across R runs.

Output
------
results/sync_complete.csv
    One row per (N, k) with the mean relative error and CI width.

Usage
-----
    python run_Table1.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config as C
from games.nash_cournot import F_i, project_Xi, solve_NE
from analysis.plots import Z_TABLE


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
N_VALUES: tuple[int, ...] = (20, 50)
R: int = C.R                                 # sample paths per (N); shared with config
K_HORIZONS: tuple[int, ...] = (5_000, 10_000)
MAX_ITER: int = K_HORIZONS[-1]


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial decision x_i^0 ~ Proj_{X_i}( U[0, INIT_SCALE]^{2L} )."""
    xi = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))   # xi_i ~ U([0, INIT_SCALE])^{2L}
    for i in range(N):
        xi[i] = project_Xi(i, xi[i])
    return xi


def _run_one_path(x0: np.ndarray, x_star: np.ndarray, N: int, L: int) -> dict:
    """One run of the centralized synchronous baseline on the static complete graph.

    Records the thesis relative error at each horizon in K_HORIZONS.

    NOTE:
    This implementation is equivalent to running the synchronous distributed
    algorithm on a static complete graph, where the aggregate is exact at
    every iteration. For that reason, the code directly substitutes the true
    aggregate bar_s into F_i instead of explicitly updating the consensus
    variable v.
    """
    x = x0.copy()                                         # shape (N, 2L)
    err_at_k = {}
    denom = max(np.max(np.abs(x_star)), 1e-12)

    for k in range(MAX_ITER):
        alpha_k = 1.0 / (k + 1)                               # Koshal 1/k
        bar_s = x[:, L:].sum(axis=0)                      # true aggregate
        x_new = np.empty_like(x)
        for i in range(N):
            grad = F_i(i, x[i], bar_s)
            x_new[i] = project_Xi(i, x[i] - alpha_k * grad)
        x = x_new

        if (k + 1) in K_HORIZONS:
            e = float(np.max(np.abs(x.ravel() - x_star)) / denom)
            err_at_k[k + 1] = e

    return err_at_k


def main() -> None:
    rows = []
    master_rng = np.random.default_rng(C.SEED + 1)

    print("=" * 60, flush=True)
    print("Table 1 -- Centralized synchronous baseline on the static complete graph", flush=True)
    print("Mean error and width of 90% confidence interval after k iterations", flush=True)
    print(f"  R = {R}, horizons = {K_HORIZONS}", flush=True)
    print("=" * 60, flush=True)

    for N_val in N_VALUES:
        print(f"\n--- N = {N_val} ---", flush=True)
        C.resample(N_val)                                 # redraw Cournot coefs
        print("  Solving reference NE ...", flush=True)
        x_star = solve_NE()
        print(f"  ||x*|| = {np.linalg.norm(x_star):.4f}", flush=True)

        errs = {k: np.zeros(R) for k in K_HORIZONS}
        t0 = time.time()

        for r in range(R):
            rng = np.random.default_rng(int(master_rng.integers(1 << 31)))
            x0 = _random_x0(N_val, C.L, rng)
            out = _run_one_path(x0, x_star, N_val, C.L)
            for k in K_HORIZONS:
                errs[k][r] = out[k]
            if (r + 1) % 10 == 0 or (r + 1) == R:
                dt = time.time() - t0
                eta = dt / (r + 1) * (R - r - 1)
                print(f"  rep {r + 1:2d}/{R}  ({dt:.1f}s, ETA {eta:.1f}s)",
                      flush=True)

        for k in K_HORIZONS:
            mean_e = float(errs[k].mean())
            sem = float(errs[k].std(ddof=1) / np.sqrt(R))
            ci_w = 2.0 * Z_TABLE[0.90] * sem
            rows.append({
                "N":         N_val,
                "k":         k,
                "mean_err":  mean_e,
                "ci_width":  ci_w,
            })
            print(f"  k={k:>6}  mean_err={mean_e:.4e}  CI90={ci_w:.4e}",
                  flush=True)

    df = pd.DataFrame(rows)
    out_path = C.RESULTS / "sync_complete.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    print("\n" + "=" * 60)
    print("Table 1 -- Centralized synchronous baseline on the static complete graph")
    print("Mean error and width of 90% confidence interval after k iterations")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4e}"))


if __name__ == "__main__":
    main()
