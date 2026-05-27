"""
Reproduce Koshal et al. (2016), Tables 3 and 4 (static-network,
centralized synchronous algorithm) -- the baseline where every player
already knows the true aggregate s_bar = sum_j s_j and does NOT need
consensus / a graph / a mixing matrix.

Algorithm (per Koshal Sec. 6.2 with known aggregate):
    For each player i in parallel, every round k = 0, 1, ...
        bar_s = sum_j s_j^k                      (true aggregate)
        x_i^{k+1} = Pi_{X_i}( x_i^k - tau_k F_i(x_i^k, bar_s) )
    with stepsize tau_k = 1 / (k + 1).

Variability across the R = 50 sample paths comes only from the random
initial point x_i^0 ~ project_Xi( U[0, INIT_SCALE]^{2L} ).
Each (N, k) cell of Tables 3 and 4 reports the mean relative error and
the 90 % CI width over R samples, at horizons k in {5000, 10000}.

Outputs:
    results/sync_centralized.csv      one row per (N, k) with mean and CI

Usage:
    python run_sync_centralized.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config as C
from games.nash_cournot import F_i, project_Xi, solve_NE


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
N_VALUES: tuple[int, ...] = (20, 50)
R: int = 50                                  # sample paths per (N)
K_HORIZONS: tuple[int, ...] = (5_000, 10_000)
MAX_ITER: int = K_HORIZONS[-1]
Z_90: float = 1.645


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial decision x_i^0 ~ Proj_{X_i}( U[0, INIT_SCALE]^{2L} )."""
    raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
    for i in range(N):
        raw[i] = project_Xi(i, raw[i])
    return raw


def _run_one_path(x0: np.ndarray, x_star: np.ndarray, N: int, L: int) -> dict:
    """One centralized run. Records relative error at every K in K_HORIZONS."""
    x = x0.copy()                                         # shape (N, 2L)
    err_at_k = {}
    denom = max(np.max(np.abs(x_star)), 1e-12)

    for k in range(MAX_ITER):
        tau = 1.0 / (k + 1)                               # Koshal 1/k
        bar_s = x[:, L:].sum(axis=0)                      # true aggregate
        x_new = np.empty_like(x)
        for i in range(N):
            grad = F_i(i, x[i], bar_s)
            x_new[i] = project_Xi(i, x[i] - tau * grad)
        x = x_new

        # Record after k+1 iterations (so k=5000 means 5000 updates done)
        if (k + 1) in K_HORIZONS:
            e = float(np.max(np.abs(x.ravel() - x_star)) / denom)
            err_at_k[k + 1] = e

    return err_at_k


def main() -> None:
    rows = []
    master_rng = np.random.default_rng(C.SEED + 1)

    print("=" * 60, flush=True)
    print("Centralized sync algorithm (Koshal Tables 3, 4)", flush=True)
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

        # Mean + 90 % CI width per horizon
        for k in K_HORIZONS:
            mean_e = float(errs[k].mean())
            sem = float(errs[k].std(ddof=1) / np.sqrt(R))
            ci_w = 2.0 * Z_90 * sem
            rows.append({
                "N":         N_val,
                "k":         k,
                "mean_err":  mean_e,
                "ci_width":  ci_w,
            })
            print(f"  k={k:>6}  mean_err={mean_e:.4e}  CI90={ci_w:.4e}",
                  flush=True)

    df = pd.DataFrame(rows)
    out_path = C.RESULTS / "sync_centralized.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    # Pretty print like Koshal Tables 3, 4
    print("\n" + "=" * 60)
    print("Table 3 (Mean error on termination):")
    pivot_mean = df.pivot(index="N", columns="k", values="mean_err")
    print(pivot_mean.to_string(float_format=lambda x: f"{x:.4e}"))
    print("\nTable 4 (Width of 90% CI of mean error):")
    pivot_ci = df.pivot(index="N", columns="k", values="ci_width")
    print(pivot_ci.to_string(float_format=lambda x: f"{x:.4e}"))


if __name__ == "__main__":
    main()
