"""
Asynchronous gossip algorithm on the representative small-world graph
WS(N=50, K=6, p=0.1) used by Section 6.3 of the thesis.

This script runs ONLY p = 0.1 (no sweep) -- the full sweep is in
`run_async_sensitivity.py`. We need a single representative graph for
the §6.3.2 "Sync vs async convergence on WS(p=0.1)" analysis.

For R = 50 independent realisations of WS(50, 6, 0.1) we record:
    - the full mean-error trajectory (for the combined sync/async figure)
    - the relative error at horizon \tilde k = 1e5 (for Table 7)
    - the structural indicators of the graph
    - the async spectral indicator
          rho_async = sqrt(lambda_2(E[W(k)]))  (Koshal 2016, p.701)

Outputs (in `results/`):
    async_p01.csv             one row per rep with summary stats
    async_p01_trajectories.npz  curves array for plotting (events, runs)

Usage:
    python run_async_p01.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from games.nash_cournot import solve_NE
from graphs.topologies import watts_strogatz
from algorithms.async_gossip import run as run_async
from analysis.metrics import (
    summary as struct_summary,
    rho_async,
    lambda2_async,
)
from analysis.plots import mean_ci


# --------------------------------------------------------------------------
# Fixed knobs (§6.3 design: one representative p = 0.1)
# --------------------------------------------------------------------------
P: float = 0.1
N_REPS: int = C.R                 # 50
MAX_EVENTS: int = 100_000         # \tilde k = 10^5 (Koshal async horizon)
K_SUMMARY: int = 100_000          # error reported in Table 7
WS_K: int = C.WS_K                # 6


def _fresh_NE(N_val: int = 50) -> np.ndarray:
    """Resample Cournot coefficients and recompute NE -- always fresh,
    no caching, so any change to game parameters or projection takes
    effect immediately."""
    C.resample(N_val)
    return solve_NE()


def main() -> None:
    import time, sys
    print(f"[start] async run at p={P}, R={N_REPS}, max_events={MAX_EVENTS:,}",
          flush=True)
    print("[start] solving Nash equilibrium (fresh, N=50) ...", flush=True)
    x_star = _fresh_NE(N_val=50)
    print(f"[start] NE solved, ||x*|| = {np.linalg.norm(x_star):.4f}",
          flush=True)
    # Paired with run_sync_p01.py (same master seed C.SEED + 43)
    # so both algorithms run on the *same* 50 graph instances. This
    # turns §6.3.2 into a clean sync-vs-async comparison on a common
    # graph sample, isolating the algorithmic difference.
    master_rng = np.random.default_rng(C.SEED + 43)

    rows = []
    trajectories: list[np.ndarray] = []
    event_axes: list[np.ndarray] = []

    t_start = time.time()
    for r in range(N_REPS):
        t_rep = time.time()
        seed = int(master_rng.integers(1 << 31))
        rng = np.random.default_rng(seed)
        graph_seed = int(rng.integers(1 << 31))

        G, W = watts_strogatz(C.N, WS_K, P, seed=graph_seed)
        print(f"  rep {r + 1:2d}/{N_REPS} starting ...", flush=True)

        out = run_async(
            G, x_star,
            max_events=MAX_EVENTS,
            eps=0.0,
            rng=rng,
        )

        # Error at the reported horizon K_SUMMARY
        events = out["events"]
        rel_err = out["rel_err"]
        idx = int(np.searchsorted(events, K_SUMMARY, side="right") - 1)
        idx = max(0, min(idx, len(rel_err) - 1))

        # Per-graph spectral indicators
        rho = rho_async(G)
        lam2 = lambda2_async(G)

        rows.append({
            "rep":                r,
            "p":                  P,
            "mean_error_at_K":    float(rel_err[idx]),
            "lambda2_async":      lam2,
            "rho_async":          rho,
            **struct_summary(G, W),
        })

        trajectories.append(rel_err)
        event_axes.append(events)

        dt_rep = time.time() - t_rep
        elapsed_total = time.time() - t_start
        eta = elapsed_total / (r + 1) * (N_REPS - r - 1)
        print(f"  rep {r + 1:2d}/{N_REPS} done"
              f"  err@1e5={rel_err[idx]:.4f}"
              f"  ({dt_rep:.1f}s, ETA {eta/60:.1f} min)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "async_p01.csv", index=False)
    print(f"\nSaved results/async_p01.csv  ({N_REPS} rows)")

    # Save trajectories so we can build the combined sync/async figure later.
    # All event axes are typically identical (record_every fixed), but we
    # store them per-rep for safety.
    np.savez(
        C.RESULTS / "async_p01_trajectories.npz",
        events_per_rep=np.stack(event_axes),
        rel_err_per_rep=np.stack(trajectories),
    )
    print(f"Saved results/async_p01_trajectories.npz")

    # --------------------------------------------------------------
    # Figure: mean +/- 90% CI band over the R sample paths
    # Cropped to k >= K_START to drop the early-stage transient of
    # the 9/Gamma diminishing stepsize (first step overshoots).
    # --------------------------------------------------------------
    K_START = 100
    mean, lo, hi = mean_ci(trajectories, confidence=0.90)
    events_axis = event_axes[0]
    mask = events_axis >= K_START
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(events_axis[mask], mean[mask], color="C1", linewidth=1.4,
            label="mean over $R = 50$ graphs")
    ax.fill_between(events_axis[mask], lo[mask], hi[mask],
                    color="C1", alpha=0.25, label="90% CI")
    ax.set_yscale("log")
    ax.set_xlabel(f"gossip event $k$, $k \\geq {K_START}$")
    ax.set_ylabel("relative error $e(k)$")
    ax.set_title("Asynchronous gossip on $WS(50,\\,6,\\,0.1)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.RESULTS / "ws_async_p01.png", dpi=200)
    plt.close(fig)
    print(f"Saved results/ws_async_p01.png")

    # Quick summary for sanity
    n = len(df)
    m = df["mean_error_at_K"].mean()
    sd = df["mean_error_at_K"].std(ddof=1)
    sem = sd / np.sqrt(n)
    # 90% CI width using normal quantile (Koshal convention)
    ci_width = 2 * 1.645 * sem
    print("\n=== Async at p=0.1 (R=50, max_events=1e5) ===")
    print(f"  mean error at k=1e5    : {m:.4f}")
    print(f"  90% CI width           : {ci_width:.4e}")
    print(f"  rho_async (mean over R): {df['rho_async'].mean():.4f}")
    print(f"  lambda2_async (mean)   : {df['lambda2_async'].mean():.6f}")


if __name__ == "__main__":
    main()
