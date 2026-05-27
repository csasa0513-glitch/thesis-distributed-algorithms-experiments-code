"""
Q3 (sensitivity): effect of the rewiring probability p of the
Watts-Strogatz family on the asynchronous gossip algorithm. Used by
Sec. 6.4 of the thesis.

PAIRED with run_sync_sensitivity.py: both scripts use master seed
C.SEED + 44 with the SAME (p outer, rep inner) loop and the same per-
rep seed-derivation order, so that within each (p, r) cell the synchronous
and asynchronous algorithms run on the SAME WS realisation and the SAME
initial point x^0. This isolates the algorithmic difference from
graph-sample and initial-condition randomness.

We sweep the rewiring probability
    p in {0.0, 0.001, 0.01, 0.1, 0.5, 1.0}
on a ring-lattice of base degree K = 6.
The sweep uses N = 50, `config.R` repetitions, and the diminishing
stepsize rule of Koshal et al. (alpha_{k,i} = 9 / Gamma_i^k).

Outputs written to `results/`:

    async_sensitivity.csv      one row per (family, parameter, rep);
                               fills the sensitivity table of Sec. 6.4
    async_sensitivity.png      single-panel convergence figure (WS)

Usage:
    python run_async_sensitivity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from games.nash_cournot import solve_NE
from graphs.topologies import watts_strogatz
from algorithms.async_gossip import run as run_async
from analysis.metrics import summary as struct_summary, lambda2_async, rho_async
from analysis.plots import mean_ci


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
N_REPS: int = C.R            # use the same R as config (default 50)
MAX_EVENTS: int = 100_000
K_SUMMARY: int = 100_000     # horizon reported in the LaTeX sensitivity table
WS_K: int = C.WS_K


def _fresh_NE(N_val: int = 50) -> np.ndarray:
    """Resample Cournot coefficients and recompute NE -- always fresh,
    no caching, so any change to game parameters or projection takes
    effect immediately."""
    C.resample(N_val)
    return solve_NE()


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
            out = run_async(G, x_star,
                            max_events=MAX_EVENTS,
                            eps=0.0,
                            rng=rng)
            err_runs.append(out["rel_err"])
            # error at the reported horizon
            events = out["events"]
            idx = int(np.searchsorted(events, K_SUMMARY, side="right") - 1)
            idx = max(0, min(idx, len(out["rel_err"]) - 1))
            struct = struct_summary(G, W)
            rows.append({
                "family":          family,
                "param":           float(pv),
                "rep":             r,
                "mean_error_at_K": float(out["rel_err"][idx]),
                "lambda2_async":   lambda2_async(G),
                "rho_async":       rho_async(G),
                **struct,
            })
            if (r + 1) % 10 == 0 or (r + 1) == N_REPS:
                print(f"    [{family}={pv}] run {r + 1}/{N_REPS} done", flush=True)
        mean, low, high = mean_ci(err_runs)
        curves[f"{family} {pv}"] = (np.arange(len(mean)), mean, low, high)
        mean_err = np.mean([r['mean_error_at_K'] for r in rows if r['param'] == pv])
        print(f"[{family}]  param={pv}  mean_err@{K_SUMMARY}={mean_err:.3e}",
              flush=True)
    return rows, curves


def main() -> None:
    # Use N = 50 to match the sync sensitivity sweep.
    # Fresh NE: no caching, so any change to game parameters takes effect.
    x_star = _fresh_NE(N_val=50)
    print(f"[start] NE solved, ||x*|| = {np.linalg.norm(x_star):.4f}",
          flush=True)

    def _ws_builder(N_val, p, seed):
        return watts_strogatz(N_val, WS_K, p, seed=seed)

    # Paired with run_sync_sensitivity.py (same master seed, same loop
    # order, same per-rep RNG calls) so sync and async use identical
    # WS graphs and identical x^0 for each (p, r) cell.
    rng = np.random.default_rng(C.SEED + 44)

    # WS sweep (diminishing stepsize only; constant variant is out of scope)
    rows_all, curves_ws = _run_sweep("WS", C.WS_P, _ws_builder, x_star, rng)

    df = pd.DataFrame(rows_all)
    df.to_csv(C.RESULTS / "async_sensitivity.csv", index=False)
    print("\nSummary (mean over reps of mean_error_at_K):")
    print(df.groupby(["family", "param"])["mean_error_at_K"].mean())

    # Single-panel figure (convergence curves).
    # Following Watts-Strogatz (1998) Fig. 2 style, we drop p=0 from the
    # plot (no log-scale rendering for p=0); p=0 numbers are still in the
    # CSV / Table.
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
    for label, (x, m, lo, hi) in curves_ws.items():
        pv = float(label.split()[-1])
        if pv == 0.0:
            continue
        ax.plot(x, m, label=f"$p={pv:g}$", linewidth=1.3)
        ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("gossip event $k$")
    ax.set_ylabel("relative error $e(k)$")
    ax.set_title("Watts--Strogatz $(N=50,\\,K=6)$, varying $p$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle("Asynchronous gossip: sensitivity to the rewiring probability $p$")
    fig.tight_layout()
    fig.savefig(C.RESULTS / "async_sensitivity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
