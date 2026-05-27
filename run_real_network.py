"""
Sec. 6.6 of the thesis: distributed algorithms on an empirical network.

We run BOTH the synchronous algorithm (Algorithm 1) and the asynchronous
gossip algorithm (Algorithm 2) of Koshal et al. (2016) on Zachary's
Karate Club (N = 34), the canonical real-world graph in network science.

Because the graph is fixed (no random-graph ensemble), each sample path
shares the same W; variability across the R sample paths is produced by
drawing a fresh random initial decision (sync) and a fresh gossip seed
(async). The Cournot coefficients are redrawn for N = 34 with
config.resample(34) so the game instance is consistent.

Outputs (in results/):
    real_network_sync.csv         long-format: rows are (graph, mode, k)
    real_network_async.csv        long-format: rows are (graph, mode, k)
    real_network_sync_conv.png    sync convergence curve, 90% CI band
    real_network_async_conv.png   async convergence curve, 90% CI band

Usage:
    python run_real_network.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from games.nash_cournot import solve_NE, project_Xi
from graphs.real_networks import REAL
from algorithms.sync_koshal import run as run_sync
from algorithms.async_gossip import run as run_async
from analysis.metrics import summary as struct_summary
from analysis.plots import convergence_plot, mean_ci


# --------------------------------------------------------------------------
# Knobs (match the regular-graph reproduction in run_sync.py / run_async_regular.py)
# --------------------------------------------------------------------------
R_REPS: int = 50                 # sample paths per (graph, algorithm)
SYNC_CHECKPOINTS = C.SYNC_CHECKPOINTS               # (5_000, 10_000)
ASYNC_CHECKPOINTS = (50_000, 100_000)               # Koshal Tables 5-6 horizon
MAX_ASYNC_EVENTS = max(ASYNC_CHECKPOINTS) + 1
Z_90: float = 1.6449             # 90% CI z-score


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial decision uniform in [0, INIT_SCALE], projected onto X_i."""
    raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
    for i in range(N):
        raw[i] = project_Xi(i, raw[i])
    return raw


def _ci_width(values: np.ndarray) -> float:
    n = values.size
    if n < 2:
        return 0.0
    s = float(values.std(ddof=1))
    return 2.0 * Z_90 * s / np.sqrt(n)


def _error_at(events: np.ndarray, errors: np.ndarray, target_k: int) -> float:
    idx = int(np.searchsorted(events, target_k, side="right") - 1)
    idx = max(0, min(idx, len(errors) - 1))
    return float(errors[idx])


# --------------------------------------------------------------------------
# Sync experiment on one real network
# --------------------------------------------------------------------------
def run_sync_on(name: str, G, W) -> tuple[list[dict], tuple]:
    """Run R sync sample paths on (G, W). Return rows + (x_axis, mean, lo, hi)."""
    N = G.number_of_nodes()
    C.resample(N)
    x_star = solve_NE()

    rng = np.random.default_rng(C.SEED + 70_000 + hash(name) % 1_000)
    errs = np.empty((R_REPS, C.MAX_ITER_SYNC))
    for r in range(R_REPS):
        x0 = _random_x0(N, C.L, rng)
        out = run_sync(W, x_star,
                       max_iter=C.MAX_ITER_SYNC,
                       eps=0.0,
                       record_every=1,
                       x0=x0,
                       known_aggregate=False)
        errs[r] = out["rel_err"]
        if (r + 1) % 10 == 0 or (r + 1) == R_REPS:
            print(f"    [sync {name}] run {r + 1}/{R_REPS} done", flush=True)

    rows = []
    for k in SYNC_CHECKPOINTS:
        col = errs[:, k - 1]
        rows.append({
            "graph":     name,
            "algorithm": "sync",
            "k":         k,
            "mean_err":  float(col.mean()),
            "ci_width":  _ci_width(col),
            "n_runs":    R_REPS,
            **struct_summary(G, W),
        })

    mean_traj, low_traj, high_traj = mean_ci(list(errs), confidence=0.90)
    stride = 50
    iters = np.arange(1, C.MAX_ITER_SYNC + 1)
    curve = (iters[::stride],
             mean_traj[::stride],
             low_traj[::stride],
             high_traj[::stride])
    return rows, curve


# --------------------------------------------------------------------------
# Async experiment on one real network
# --------------------------------------------------------------------------
def run_async_on(name: str, G, W) -> tuple[list[dict], tuple]:
    """Run R async sample paths on G. Return rows + (events, mean, lo, hi)."""
    N = G.number_of_nodes()
    C.resample(N)
    x_star = solve_NE()

    err_trajectories = []
    err_at_checkpoints = []
    events_ref = None
    for r in range(R_REPS):
        seed = C.SEED + 80_000 + 17 * r + hash(name) % 1_000
        rng = np.random.default_rng(seed)
        out = run_async(G, x_star,
                        max_events=MAX_ASYNC_EVENTS,
                        eps=0.0,
                        rng=rng)
        err_trajectories.append(out["rel_err"])
        err_at_checkpoints.append([_error_at(out["events"], out["rel_err"], k)
                                   for k in ASYNC_CHECKPOINTS])
        if events_ref is None:
            events_ref = out["events"]
        if (r + 1) % 10 == 0 or (r + 1) == R_REPS:
            print(f"    [async {name}] run {r + 1}/{R_REPS} done", flush=True)

    err_arr = np.asarray(err_at_checkpoints)
    mean = err_arr.mean(axis=0)
    sem  = err_arr.std(axis=0, ddof=1) / np.sqrt(R_REPS)
    ci_w = 2.0 * Z_90 * sem
    rows = []
    for idx, k in enumerate(ASYNC_CHECKPOINTS):
        rows.append({
            "graph":     name,
            "algorithm": "async",
            "k":         int(k),
            "mean_err":  float(mean[idx]),
            "ci_width":  float(ci_w[idx]),
            "n_runs":    R_REPS,
            **struct_summary(G, W),
        })

    mean_traj, low_traj, high_traj = mean_ci(err_trajectories, confidence=0.90)
    x_axis = events_ref if events_ref is not None else np.arange(len(mean_traj))
    curve = (x_axis, mean_traj, low_traj, high_traj)
    return rows, curve


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 70, flush=True)
    print("run_real_network.py : sync + async on empirical networks", flush=True)
    print(f"  R (sample paths) = {R_REPS}", flush=True)
    print(f"  Networks         = {list(REAL.keys())}", flush=True)
    print("=" * 70, flush=True)

    sync_rows, async_rows = [], []
    sync_curves, async_curves = {}, {}

    for name, loader in REAL.items():
        G, W = loader()
        N = G.number_of_nodes()
        print(f"\n--- {name}  (N = {N}, |E| = {G.number_of_edges()}) ---",
              flush=True)

        # Sync
        rows_s, curve_s = run_sync_on(name, G, W)
        sync_rows.extend(rows_s)
        sync_curves[name] = curve_s
        print(f"[sync {name}]  e@5e3={rows_s[0]['mean_err']:.3e}  "
              f"e@1e4={rows_s[1]['mean_err']:.3e}", flush=True)

        # Async
        rows_a, curve_a = run_async_on(name, G, W)
        async_rows.extend(rows_a)
        async_curves[name] = curve_a
        print(f"[async {name}] e@5e4={rows_a[0]['mean_err']:.3e}  "
              f"e@1e5={rows_a[1]['mean_err']:.3e}", flush=True)

    pd.DataFrame(sync_rows).to_csv(C.RESULTS / "real_network_sync.csv", index=False)
    pd.DataFrame(async_rows).to_csv(C.RESULTS / "real_network_async.csv", index=False)

    convergence_plot(
        sync_curves,
        xlabel="iteration k",
        ylabel="mean relative error e_k",
        title=f"Sync on real networks (mean over R={R_REPS})",
        out=C.RESULTS / "real_network_sync_conv.png",
    )
    convergence_plot(
        async_curves,
        xlabel="gossip event k",
        ylabel="mean relative error e_k",
        title=f"Async on real networks (mean over R={R_REPS})",
        out=C.RESULTS / "real_network_async_conv.png",
    )
    print("\nReal-network experiments done.")


if __name__ == "__main__":
    main()
