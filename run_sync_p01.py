"""
Synchronous algorithm on the representative small-world graph
WS(N=50, K=6, p=0.1) used by Section 6.3 of the thesis.

This script runs ONLY p = 0.1 (no sweep) -- the full sweep is in
`run_sync_sensitivity.py`. We need a single representative graph
setting for the §6.3.2 "Sync vs async convergence on WS(p=0.1)"
analysis, with the full trajectory stored per rep so we can plot
the mean curve plus 90% CI band.

For R = 50 independent realisations of WS(50, 6, 0.1) we record:
    - the full mean-error trajectory (for the §6.3.2 figure)
    - the relative error at horizon \\tilde k = 10^4 (Table 7)
    - the structural indicators of the graph
    - the synchronous spectral indicator |lambda_2(W)|
          (Boyd-Gossip 2006; Koshal 2016)

Outputs (in `results/`):
    sync_p01.csv             one row per rep with summary stats
    sync_p01_trajectories.npz  curves array for plotting (iters, runs)
    ws_sync_p01.png          mean-error figure with 90% CI band

Usage:
    python run_sync_p01.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from games.nash_cournot import solve_NE, project_Xi
from graphs.topologies import watts_strogatz
from algorithms.sync_koshal import run as run_sync
from analysis.metrics import (
    summary as struct_summary,
    spectral_gap,
)
from analysis.plots import mean_ci


# --------------------------------------------------------------------------
# Fixed knobs (§6.3 design: one representative p = 0.1)
# --------------------------------------------------------------------------
P: float = 0.1
N_REPS: int = C.R                 # 50
MAX_ITER: int = 10_000            # \tilde k = 10^4 (Koshal sync horizon)
K_SUMMARY: int = 10_000           # error reported in Table 7
WS_K: int = C.WS_K                # 6


def _fresh_NE(N_val: int = 50) -> np.ndarray:
    """Resample Cournot coefficients and recompute NE -- always fresh,
    no caching, so any change to game parameters or projection takes
    effect immediately."""
    C.resample(N_val)
    return solve_NE()


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial decision, projected onto each player's X_i."""
    raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
    for i in range(N):
        raw[i] = project_Xi(i, raw[i])
    return raw


def main() -> None:
    print(f"[start] sync run at p={P}, R={N_REPS}, max_iter={MAX_ITER:,}",
          flush=True)
    print("[start] solving Nash equilibrium (fresh, N=50) ...", flush=True)
    x_star = _fresh_NE(N_val=50)
    print(f"[start] NE solved, ||x*|| = {np.linalg.norm(x_star):.4f}",
          flush=True)
    master_rng = np.random.default_rng(C.SEED + 43)  # match sync_sensitivity

    rows = []
    trajectories: list[np.ndarray] = []
    iter_axes: list[np.ndarray] = []

    t_start = time.time()
    for r in range(N_REPS):
        t_rep = time.time()
        seed = int(master_rng.integers(1 << 31))
        rng = np.random.default_rng(seed)
        graph_seed = int(rng.integers(1 << 31))

        G, W = watts_strogatz(C.N, WS_K, P, seed=graph_seed)
        x0 = _random_x0(C.N, C.L, rng)
        print(f"  rep {r + 1:2d}/{N_REPS} starting ...", flush=True)

        out = run_sync(
            W, x_star,
            max_iter=MAX_ITER,
            eps=0.0,
            record_every=1,
            x0=x0,
            known_aggregate=False,
        )

        iters = out["iters"]
        rel_err = out["rel_err"]
        idx = int(np.searchsorted(iters, K_SUMMARY, side="right") - 1)
        idx = max(0, min(idx, len(rel_err) - 1))

        # Per-graph spectral indicator |lambda_2(W)|
        lam2 = 1.0 - spectral_gap(W)

        rows.append({
            "rep":                r,
            "p":                  P,
            "mean_error_at_K":    float(rel_err[idx]),
            "abs_lambda2":        lam2,
            **struct_summary(G, W),
        })

        trajectories.append(rel_err)
        iter_axes.append(iters)

        dt_rep = time.time() - t_rep
        elapsed_total = time.time() - t_start
        eta = elapsed_total / (r + 1) * (N_REPS - r - 1)
        print(f"  rep {r + 1:2d}/{N_REPS} done"
              f"  err@1e4={rel_err[idx]:.4f}"
              f"  ({dt_rep:.1f}s, ETA {eta/60:.1f} min)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "sync_p01.csv", index=False)
    print(f"\nSaved results/sync_p01.csv  ({N_REPS} rows)")

    # Save trajectories for the §6.3.2 figure.
    np.savez(
        C.RESULTS / "sync_p01_trajectories.npz",
        iters_per_rep=np.stack(iter_axes),
        rel_err_per_rep=np.stack(trajectories),
    )
    print(f"Saved results/sync_p01_trajectories.npz")

    # --------------------------------------------------------------
    # Figure: mean +/- 90% CI band over the R sample paths
    # Cropped to k >= K_START to drop the early-stage transient of
    # the 1/k diminishing stepsize (first step overshoots).
    # --------------------------------------------------------------
    K_START = 100
    mean, lo, hi = mean_ci(trajectories, confidence=0.90)
    iters_axis = iter_axes[0]
    mask = iters_axis >= K_START
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(iters_axis[mask], mean[mask], color="C0", linewidth=1.4,
            label="mean over $R = 50$ graphs")
    ax.fill_between(iters_axis[mask], lo[mask], hi[mask],
                    color="C0", alpha=0.25, label="90% CI")
    ax.set_yscale("log")
    ax.set_xlabel(f"iteration $k$, $k \\geq {K_START}$")
    ax.set_ylabel("relative error $e(k)$")
    ax.set_title("Synchronous algorithm on $WS(50,\\,6,\\,0.1)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.RESULTS / "ws_sync_p01.png", dpi=200)
    plt.close(fig)
    print(f"Saved results/ws_sync_p01.png")

    # Quick sanity summary
    n = len(df)
    m = df["mean_error_at_K"].mean()
    sd = df["mean_error_at_K"].std(ddof=1)
    sem = sd / np.sqrt(n)
    ci_width = 2 * 1.645 * sem  # 90% CI width
    print("\n=== Sync at p=0.1 (R=50, max_iter=1e4) ===")
    print(f"  mean error at k=1e4    : {m:.4f}")
    print(f"  90% CI width           : {ci_width:.4e}")
    print(f"  |lambda_2|(W) (mean R) : {df['abs_lambda2'].mean():.4f}")


if __name__ == "__main__":
    main()
