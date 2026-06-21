"""
Synchronous algorithm on WS(50, 6, 0.1) for Figure 3a.

Runs only p_rew = 0.1 and records the mean-error trajectory and the
error at k = 10^4 for R = C.R independent realisations.

Outputs:
    sync_p01.csv
    sync_p01_trajectories.npz
    ws_sync_p01.png
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import config as C
from games.nash_cournot import solve_NE, project_Xi
from graphs.generators import watts_strogatz
from algorithms.sync_koshal import run as run_sync
from analysis.metrics import (
    summary as struct_summary,
    sync,
)
from analysis.plots import mean_ci


# --------------------------------------------------------------------------
# Fixed knobs
# --------------------------------------------------------------------------
P_REW: float = 0.1
N_REPS: int = C.R                 
MAX_ITER: int = 10_000            
K_SUMMARY: int = 10_000           
WS_K: int = C.WS_K                


def _fresh_NE(N_val: int = 50) -> np.ndarray:
    C.resample(N_val)
    return solve_NE()


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    xi = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))   # xi_i ~ U([0, INIT_SCALE])^{2L}
    for i in range(N):
        xi[i] = project_Xi(i, xi[i])
    return xi


def _rep_plan(master_rng: np.random.Generator) -> tuple[int, int, int, int]:
    rep_seed = int(master_rng.integers(1 << 31))
    rng_plan = np.random.default_rng(rep_seed)
    graph_seed = int(rng_plan.integers(1 << 31))
    x0_seed = int(rng_plan.integers(1 << 31))
    gossip_seed = int(rng_plan.integers(1 << 31))
    return rep_seed, graph_seed, x0_seed, gossip_seed


def main() -> None:
    print(f"[start] sync run at p_rew={P_REW}, R={N_REPS}, max_iter={MAX_ITER:,}",
          flush=True)
    print("[start] solving Nash equilibrium (fresh, N=50) ...", flush=True)
    x_star = _fresh_NE(N_val=50)
    print(f"[start] NE solved, ||x*|| = {np.linalg.norm(x_star):.4f}",
          flush=True)
    master_rng = np.random.default_rng(C.SEED + 43)

    rows = []
    trajectories: list[np.ndarray] = []
    iter_axes: list[np.ndarray] = []
    x0_list: list[np.ndarray] = []   # stored in npz for paired-comparison audit

    t_start = time.time()
    for r in range(N_REPS):
        t_rep = time.time()
        rep_seed, graph_seed, x0_seed, gossip_seed = _rep_plan(master_rng)

        G, W = watts_strogatz(C.N, WS_K, P_REW, seed=graph_seed)
        assert nx.is_connected(G), (
            f"watts_strogatz returned a disconnected graph at "
            f"rep = {r}, graph_seed = {graph_seed}. "
            f"Check graphs/generators.py watts_strogatz."
        )
        x0_rng = np.random.default_rng(x0_seed)
        x0 = _random_x0(C.N, C.L, x0_rng)
        assert x0.shape == (C.N, 2 * C.L), (
            f"x0 shape mismatch at rep = {r}: got {x0.shape}, "
            f"expected ({C.N}, {2 * C.L})."
        )
        assert np.isfinite(x0).all(), (
            f"x0 contains non-finite values at rep = {r}."
        )
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

        # Per-graph spectral indicator |lambda_2(W)| = rho_sync = sync(W)
        lam2 = sync(W)

        rows.append({
            "rep":                r,
            "p_rew":                  P_REW,
            "mean_error_at_K":    float(rel_err[idx]),
            "abs_lambda2":        lam2,
            "rep_seed":           rep_seed,
            "graph_seed":         graph_seed,
            "x0_seed":            x0_seed,
            "gossip_seed":        gossip_seed,
            **struct_summary(G, W),
        })

        trajectories.append(rel_err)
        iter_axes.append(iters)
        x0_list.append(x0.copy())

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

    np.savez(
        C.RESULTS / "sync_p01_trajectories.npz",
        iters_per_rep=np.stack(iter_axes),
        rel_err_per_rep=np.stack(trajectories),
        x0_per_rep=np.stack(x0_list),
    )
    print(f"Saved results/sync_p01_trajectories.npz")

    mean, lo, hi = mean_ci(trajectories, confidence=0.90)
    iters_axis = iter_axes[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(iters_axis, mean, color="C0", linewidth=1.4,
            label="mean over $R = 50$ graphs")
    ax.fill_between(iters_axis, lo, hi,
                    color="C0", alpha=0.25, label="90% CI")
    ax.set_yscale("log")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("relative error $e(k)$")
    ax.set_title("Synchronous algorithm on $WS(50,\\,6,\\,0.1)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.RESULTS / "ws_sync_p01.png", dpi=200)
    plt.close(fig)
    print(f"Saved results/ws_sync_p01.png")


if __name__ == "__main__":
    main()
