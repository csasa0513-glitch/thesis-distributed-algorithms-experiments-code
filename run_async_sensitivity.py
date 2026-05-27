"""
Q3 (sensitivity): effect of the rewiring probability p of the
Watts-Strogatz family on the asynchronous gossip algorithm.

We sweep the rewiring probability
    p in {0.0, 0.001, 0.01, 0.1, 0.5, 1.0}
on a ring-lattice of base degree K = 6 (uppercase K is the WS ring
degree from Watts & Strogatz 1998; lowercase k is reserved for the
gossip-event index). The grid is taken on a roughly
log scale and brackets both limits of Watts and Strogatz (1998):
    p = 0    -> pure ring lattice (no rewiring)
    p = 1    -> Erdos-Renyi-like random graph
The sweep uses N = 50, `config.R` repetitions, and the diminishing
stepsize rule of Koshal et al. (alpha_{k,i} = 9 / Gamma_i^k).

Outputs written to `results/`:

    async_sensitivity.csv      one row per (family, parameter, rep);
                               fills the sensitivity table of Sec. 6.3
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
from analysis.metrics import summary as struct_summary
from analysis.plots import mean_ci


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
N_REPS: int = C.R            # use the same R as config (default 50)
MAX_EVENTS: int = 100_000
K_SUMMARY: int = 100_000     # horizon reported in the LaTeX sensitivity table
WS_K: int = C.WS_K


def _load_or_solve_NE() -> np.ndarray:
    cache = C.RESULTS / "NE.npz"
    if cache.exists():
        return np.load(cache)["x_star"]
    x_star = solve_NE()
    np.savez(cache, x_star=x_star)
    return x_star


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
            rows.append({
                "family":     family,
                "param":      float(pv),
                "rep":        r,
                "mean_error_at_K": float(out["rel_err"][idx]),
                **struct_summary(G, W),
            })
        mean, low, high = mean_ci(err_runs)
        curves[f"{family} {pv}"] = (np.arange(len(mean)), mean, low, high)
        mean_err = np.mean([r['mean_error_at_K'] for r in rows if r['param'] == pv])
        print(f"[{family}]  param={pv}  mean_err@{K_SUMMARY}={mean_err:.3e}",
              flush=True)
    return rows, curves


def main() -> None:
    x_star = _load_or_solve_NE()

    def _ws_builder(N_val, p, seed):
        return watts_strogatz(N_val, WS_K, p, seed=seed)

    rng = np.random.default_rng(C.SEED + 42)

    # WS sweep (diminishing stepsize only; constant variant is out of scope)
    rows_all, curves_ws = _run_sweep("WS", C.WS_P, _ws_builder, x_star, rng)

    df = pd.DataFrame(rows_all)
    df.to_csv(C.RESULTS / "async_sensitivity.csv", index=False)
    print("\nSummary (mean over reps of mean_error_at_K):")
    print(df.groupby(["family", "param"])["mean_error_at_K"].mean())

    # Single-panel figure (diminishing step, convergence curves)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
    for label, (x, m, lo, hi) in curves_ws.items():
        ax.plot(x, m, label=label, linewidth=1.3)
        ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("recorded event index")
    ax.set_ylabel("relative error e(k)")
    ax.set_title("Watts-Strogatz  (varying p, K=6)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Asynchronous gossip: sensitivity to the rewiring probability p")
    fig.tight_layout()
    fig.savefig(C.RESULTS / "async_sensitivity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
