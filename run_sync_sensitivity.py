"""
Q3 (sync sensitivity): effect of the rewiring probability p_rew of the
Watts-Strogatz family on the synchronous algorithm.

Outputs:
    sync_sensitivity.csv
    sync_sensitivity_trajectories.npz
    sync_sensitivity.png
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import config as C
from games.nash_cournot import solve_NE, project_Xi
from graphs.generators import watts_strogatz
from algorithms.sync_koshal import run as run_sync
from analysis.metrics import summary as struct_summary
from analysis.plots import mean_ci


# Hard-coded §6.4 invariants
N_WS: int = 50
WS_K_FIXED: int = 6
WS_P_VALUES: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05,
                                  0.1, 0.5, 1.0)


def _check_invariants() -> None:
    assert C.N == N_WS, (
        f"§6.4 fixes N = {N_WS} but config.N = {C.N}."
    )
    assert C.WS_K == WS_K_FIXED, (
        f"§6.4 fixes K = {WS_K_FIXED} but config.WS_K = {C.WS_K}."
    )
    assert tuple(C.WS_P) == WS_P_VALUES, (
        f"§6.4 sweep is pinned to {WS_P_VALUES} "
        f"but config.WS_P = {C.WS_P}."
    )

N_REPS: int = C.R
MAX_ITER: int = 10_000
K_SUMMARY: int = 10_000
WS_K: int = WS_K_FIXED

PLOT_P_VALUES: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0)


def _x_star_fingerprint(x_star: np.ndarray) -> tuple[float, str]:
    """Return (||x*||, sha256 hex prefix) of the Nash equilibrium."""
    norm = float(np.linalg.norm(x_star))
    h = hashlib.sha256(np.ascontiguousarray(x_star).tobytes()).hexdigest()[:16]
    return norm, h


def _load_or_compute_NE(N_val: int = 50) -> tuple[np.ndarray, str]:
    """Load shared NE_N{N}.npz if available, otherwise compute NE."""
    C.resample(N_val)
    npz_path = C.RESULTS / f"NE_N{N_val}.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            x_star = np.asarray(data["x_star"], dtype=float).copy()
        return x_star, f"file ({npz_path.name})"
    return solve_NE(), "computed (run compute_NE_reference.py to share)"


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


def _run_sweep(family, param_values, graph_builder, x_star, master_rng):
    rows = []
    curves = {}
    iters_per_rep, rel_err_per_rep, x0_per_rep = [], [], []
    pvs, reps, rseeds, gseeds, xseeds, gosseeds = [], [], [], [], [], []

    for p_rew in param_values:
        err_runs = []
        iters_axis = None
        for r in range(N_REPS):
            rep_seed, graph_seed, x0_seed, gossip_seed = _rep_plan(master_rng)

            G, W = graph_builder(N_WS, p_rew, graph_seed)
            assert nx.is_connected(G), (
                f"watts_strogatz disconnected at p_rew={p_rew}, rep={r}, "
                f"graph_seed={graph_seed}."
            )

            x0 = _random_x0(N_WS, C.L, np.random.default_rng(x0_seed))
            assert x0.shape == (N_WS, 2 * C.L)
            assert np.isfinite(x0).all()

            out = run_sync(W, x_star, max_iter=MAX_ITER, eps=0.0,
                           record_every=1, x0=x0, known_aggregate=False)
            if iters_axis is None:
                iters_axis = out["iters"]
            err_runs.append(out["rel_err"])

            iters = out["iters"]
            idx = int(np.searchsorted(iters, K_SUMMARY, side="right") - 1)
            idx = max(0, min(idx, len(out["rel_err"]) - 1))
            struct = struct_summary(G, W)
            rho_sync = float(struct["sync"])
            struct_rest = {k: v for k, v in struct.items()
                           if k != "sync"}
            rows.append({
                "family": family, "param": float(p_rew), "rep": r,
                "mean_error_at_K": float(out["rel_err"][idx]),
                "rho_sync": rho_sync,
                "rep_seed": rep_seed, "graph_seed": graph_seed,
                "x0_seed": x0_seed, "gossip_seed": gossip_seed,
                **struct_rest,
            })

            iters_per_rep.append(out["iters"])
            rel_err_per_rep.append(out["rel_err"])
            x0_per_rep.append(x0.copy())
            pvs.append(float(p_rew)); reps.append(r)
            rseeds.append(rep_seed); gseeds.append(graph_seed)
            xseeds.append(x0_seed); gosseeds.append(gossip_seed)

            if (r + 1) % 10 == 0 or (r + 1) == N_REPS:
                print(f"    [{family}={p_rew}] run {r + 1}/{N_REPS} done",
                      flush=True)
        mean, low, high = mean_ci(err_runs)
        curves[f"{family} {p_rew}"] = (iters_axis, mean, low, high)
        ms = np.mean([r2["mean_error_at_K"] for r2 in rows
                      if r2["param"] == p_rew and r2["family"] == family])
        print(f"[{family}]  param={p_rew}  mean_err@{K_SUMMARY}={ms:.3e}",
              flush=True)

    traj = {
        "iters_per_rep": np.stack(iters_per_rep),
        "rel_err_per_rep": np.stack(rel_err_per_rep),
        "x0_per_rep": np.stack(x0_per_rep),
        "param_per_rep": np.asarray(pvs, dtype=float),
        "rep_per_rep": np.asarray(reps, dtype=np.int64),
        "rep_seed_per_rep": np.asarray(rseeds, dtype=np.int64),
        "graph_seed_per_rep": np.asarray(gseeds, dtype=np.int64),
        "x0_seed_per_rep": np.asarray(xseeds, dtype=np.int64),
        "gossip_seed_per_rep": np.asarray(gosseeds, dtype=np.int64),
    }
    return rows, curves, traj


def main():
    _check_invariants()
    x_star, x_src = _load_or_compute_NE(N_val=N_WS)
    norm, h = _x_star_fingerprint(x_star)
    print(f"[start] NE source: {x_src}", flush=True)
    print(f"[start] ||x*|| = {norm:.6f}, sha256[:16] = {h}", flush=True)
    print(f"[start] async run must log the SAME sha256 prefix.",
          flush=True)

    def _ws_builder(N_val, p_rew, seed):
        return watts_strogatz(N_val, WS_K, p_rew, seed=seed)

    rng = np.random.default_rng(C.SEED + 44)

    print("=" * 70, flush=True)
    print("run_sync_sensitivity.py", flush=True)
    print(f"  N={N_WS}, K={WS_K_FIXED}, R={N_REPS}, MAX_ITER={MAX_ITER}",
          flush=True)
    print(f"  WS p values = {WS_P_VALUES}", flush=True)
    print("=" * 70, flush=True)

    rows, curves, traj = _run_sweep("WS", WS_P_VALUES, _ws_builder,
                                    x_star, rng)

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "sync_sensitivity.csv", index=False)
    print(f"\nSaved sync_sensitivity.csv ({len(df)} rows)")
    print(df.groupby(["family", "param"])["mean_error_at_K"].mean())

    traj["x_star_norm"] = np.asarray(norm, dtype=float)
    traj["x_star_hash"] = np.asarray(h, dtype="<U16")
    np.savez(C.RESULTS / "sync_sensitivity_trajectories.npz", **traj)
    print(f"Saved sync_sensitivity_trajectories.npz")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, (x, m, lo, hi) in curves.items():
        p_rew = float(label.split()[-1])
        if p_rew not in PLOT_P_VALUES:
            continue
        ax.plot(x, m, label=f"$p_{{rew}}={p_rew:g}$", linewidth=1.3)
        ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("relative error $e(k)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle("Sync sensitivity to rewiring probability $p_{rew}$, "
                 "$WS(50,\\,6,\\,p_{rew})$")
    fig.tight_layout()
    fig.savefig(C.RESULTS / "sync_sensitivity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
