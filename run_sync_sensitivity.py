"""
Q3 (sync sensitivity): effect of the rewiring probability p of the
Watts-Strogatz family on the SYNCHRONOUS algorithm.

This is the sync analogue of run_async_sensitivity.py. It supports
Section 6.5 of the thesis ("Algorithm 1 (Synchronous) Sensitivity").

For Watts-Strogatz we sweep the rewiring probability
    p in {0.0, 0.001, 0.01, 0.1, 0.5, 1.0}
on a ring-lattice of base degree K = 6 (uppercase K is the WS ring
degree from Watts & Strogatz 1998; lowercase k is reserved for the
iteration index).

The sweep uses N = 50 and `config.R` repetitions, with a fresh graph
realisation per sample path. The sync stepsize is alpha_k = 1/k.

Outputs written to `results/`:

    sync_sensitivity.csv       one row per (family, parameter, rep);
                               fills the sync sensitivity table of Sec. 6.5
    sync_sensitivity.png       single-panel convergence figure (WS)

Usage:
    python run_sync_sensitivity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from games.nash_cournot import solve_NE, project_Xi
from graphs.topologies import watts_strogatz
from algorithms.sync_koshal import run as run_sync
from analysis.metrics import summary as struct_summary
from analysis.plots import mean_ci


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
N_REPS: int = C.R            # use the same R as config (default 50)
MAX_ITER: int = 10_000       # match run_sync.py horizon
K_SUMMARY: int = 10_000      # horizon reported in the LaTeX sensitivity table
WS_K: int = C.WS_K


def _load_or_solve_NE() -> np.ndarray:
    cache = C.RESULTS / "NE.npz"
    if cache.exists():
        return np.load(cache)["x_star"]
    x_star = solve_NE()
    np.savez(cache, x_star=x_star)
    return x_star


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial decision, projected onto each player's X_i."""
    raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
    for i in range(N):
        raw[i] = project_Xi(i, raw[i])
    return raw


def _run_sweep(family: str, param_values, graph_builder,
               x_star: np.ndarray, master_rng: np.random.Generator):
    rows = []
    curves: dict[str, tuple] = {}
    for pv in param_values:
        err_runs = []
        for r in range(N_REPS):
            seed = int(master_rng.integers(1 << 31))
            rng = np.random.default_rng(seed)
            graph_seed = int(rng.integers(1 << 31))
            G, W = graph_builder(C.N, pv, graph_seed)
            x0 = _random_x0(C.N, C.L, rng)
            out = run_sync(W, x_star,
                           max_iter=MAX_ITER,
                           eps=0.0,
                           record_every=1,
                           x0=x0,
                           known_aggregate=False)
            err_runs.append(out["rel_err"])
            # error at the reported horizon
            iters = out["iters"]
            idx = int(np.searchsorted(iters, K_SUMMARY, side="right") - 1)
            idx = max(0, min(idx, len(out["rel_err"]) - 1))
            rows.append({
                "family":     family,
                "param":      float(pv),
                "rep":        r,
                "mean_error_at_K": float(out["rel_err"][idx]),
                **struct_summary(G, W),
            })
            if (r + 1) % 10 == 0 or (r + 1) == N_REPS:
                print(f"    [{family}={pv}] run {r + 1}/{N_REPS} done", flush=True)
        mean, low, high = mean_ci(err_runs)
        curves[f"{family} {pv}"] = (np.arange(len(mean)), mean, low, high)
        param_rows = [r for r in rows if r["param"] == pv and r["family"] == family]
        mean_err = np.mean([r["mean_error_at_K"] for r in param_rows])
        print(f"[{family}]  param={pv}  mean_err@{K_SUMMARY}={mean_err:.3e}",
              flush=True)
    return rows, curves


def main() -> None:
    # Use N = 50 to match the async sensitivity sweep (config default).
    C.resample(50)
    x_star = _load_or_solve_NE()

    def _ws_builder(N_val, p, seed):
        return watts_strogatz(N_val, WS_K, p, seed=seed)

    rng = np.random.default_rng(C.SEED + 43)   # different from async sweep

    rows_all = []

    print("=" * 70, flush=True)
    print("run_sync_sensitivity.py : sync algorithm sensitivity to WS p",
          flush=True)
    print(f"  N = {C.N}", flush=True)
    print(f"  R (sample paths) = {N_REPS}", flush=True)
    print(f"  Max iterations    = {MAX_ITER}", flush=True)
    print(f"  WS p values       = {C.WS_P}", flush=True)
    print("=" * 70, flush=True)

    # WS sweep
    rows_ws, curves_ws = _run_sweep("WS", C.WS_P, _ws_builder,
                                    x_star, rng)
    rows_all.extend(rows_ws)

    df = pd.DataFrame(rows_all)
    df.to_csv(C.RESULTS / "sync_sensitivity.csv", index=False)
    print("\nSummary (mean over reps of mean_error_at_K):")
    print(df.groupby(["family", "param"])["mean_error_at_K"].mean())

    # Single-panel figure (convergence curves)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
    for label, (x, m, lo, hi) in curves_ws.items():
        ax.plot(x, m, label=label, linewidth=1.3)
        ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("iteration k")
    ax.set_ylabel("relative error e(k)")
    ax.set_title("Watts-Strogatz  (varying p, K=6)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Synchronous algorithm: sensitivity to the rewiring probability p")
    fig.tight_layout()
    fig.savefig(C.RESULTS / "sync_sensitivity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
