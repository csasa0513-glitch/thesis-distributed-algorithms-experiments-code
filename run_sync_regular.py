"""
Distributed synchronous algorithm on the four regular baseline graphs
(cycle, star, grid, complete) at N in {20, 50}.

This script is the SYNC analogue of `run_async_regular.py`. Together
they form the §6.2 reproduction-and-extension block:

  * `run_sync_centralized.py`  -> Koshal Tables 3, 4 (centralized sync)
  * `run_async_regular.py`     -> Koshal Tables 5, 6 (async, 4 graphs)
  * `run_sync_regular.py`      -> *this file*: thesis extension --
                                  distributed synchronous algorithm
                                  on the same four static graphs

Koshal et al. (2016) report distributed synchronous results only on
*dynamic* B-strongly-connected networks (their Tables 1, 2); the
distributed synchronous algorithm on static regular graphs is not
covered by their tables. This script fills that gap so that the
sync and async columns in §6.2 can be compared on the same set of
topologies.

Algorithm (per Koshal Sec. 3, Algorithm 1, eqs. 9-11):
    Per round k = 0, 1, ...
        v_hat_i^k = sum_j W_ij v_j^k                          (eq. 9, mix)
        x_i^{k+1} = Pi_{X_i}(x_i^k - tau_k F_i(x_i^k, N v_hat_i^k))  (eq. 10)
        v_i^{k+1} = v_hat_i^k + (s_i^{k+1} - s_i^k)           (eq. 11, innov)
    Stepsize tau_k = 1 / k (Koshal Sec. 6.2).

For every (N, topology) cell we run R = 50 independent sample paths
with a random initial decision
    x^0 ~ Pi_{X_i}( U[0, INIT_SCALE]^{2L} ).
The R sample paths share their initial point batch across the four
topologies for the SAME N (one master_rng per N), which makes the
distributed runs directly comparable to each other within a row;
they are NOT shared across N (different problem dimension).

At each checkpoint tilde_k in SYNC_CHECKPOINTS (5000 and 10000) we
report the mean of the sample error (Koshal eq. 65)

    e_{tilde_k} = max_{i,l} { |g_il^{tilde_k} - g_il*|,
                              |s_il^{tilde_k} - s_il*| }
                / max_{i,l} { |g_il*|, |s_il*| }

and the width of the 90% CI of the mean,

    w_{tilde_k} = 2 * z_{0.95} * s_R / sqrt(R),  z_{0.95} = 1.6449.

Output:
    results/sync_regular.csv          one row per (N, topology, tilde_k)
                                      with mean_err and ci_width.
    results/sync_convergence_N20.png  distributed convergence curves, N=20.
    results/sync_convergence_N50.png  distributed convergence curves, N=50.

Usage:
    python run_sync_regular.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from games.nash_cournot import solve_NE, project_Xi
from graphs.topologies import REGULAR
from algorithms.sync_koshal import run as run_sync
from analysis.plots import convergence_plot, mean_ci


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
N_SIZES: tuple[int, ...] = (20, 50)
Z_90: float = 1.6449   # z_{0.95} for the two-sided 90% CI of the mean


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _solve_NE_for(N_new: int) -> np.ndarray:
    """Resample Cournot coefficients for N = N_new and solve the NE fresh.

    No caching: we always recompute, because any change to the game
    parameters or the projection would otherwise be silently invalidated.
    """
    C.resample(N_new)
    return solve_NE()


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial decision, projected onto each player's X_i."""
    raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
    for i in range(N):
        raw[i] = project_Xi(i, raw[i])
    return raw


def _ci_width(values: np.ndarray) -> float:
    """Two-sided 90% CI width of the mean (normal approximation)."""
    n = values.size
    if n < 2:
        return 0.0
    s = float(values.std(ddof=1))
    return 2.0 * Z_90 * s / np.sqrt(n)


def _run_batch(W: np.ndarray,
               x_star: np.ndarray,
               x0_batch: list[np.ndarray],
               label: str = "") -> np.ndarray:
    """
    Run the distributed synchronous algorithm once per initial condition
    in `x0_batch` with a FIXED weight matrix `W`.
    Returns an (R, MAX_ITER_SYNC) array of sample errors.
    """
    R = len(x0_batch)
    errs = np.empty((R, C.MAX_ITER_SYNC))
    for r, x0 in enumerate(x0_batch):
        out = run_sync(
            W, x_star,
            max_iter=C.MAX_ITER_SYNC,
            eps=0.0,                 # never stop early
            record_every=1,
            x0=x0,
            known_aggregate=False,   # distributed: aggregate via consensus on v
        )
        errs[r] = out["rel_err"]
        if (r + 1) % 10 == 0 or (r + 1) == R:
            print(f"    [{label}] run {r + 1}/{R} done", flush=True)
    return errs


def _checkpoint_rows(errs: np.ndarray,
                     N_new: int,
                     topology: str) -> list[dict]:
    """Extract mean and 90% CI at each Koshal checkpoint as rows."""
    rows = []
    for k in C.SYNC_CHECKPOINTS:
        col = errs[:, k - 1]                 # k updates -> Python index k-1
        rows.append({
            "N":         N_new,
            "topology":  topology,
            "k":         k,
            "mean_err":  float(col.mean()),
            "ci_width":  _ci_width(col),
            "n_runs":    int(errs.shape[0]),
        })
    return rows


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def experiment_regular() -> pd.DataFrame:
    """Distributed sync on the four regular graphs, R = 50 sample paths."""
    rows: list[dict] = []

    for N_new in N_SIZES:
        print(f"\n=== N = {N_new} : solving NE and drawing initial conditions ===",
              flush=True)
        x_star = _solve_NE_for(N_new)
        N, L = C.N, C.L
        print(f"  ||x*|| = {np.linalg.norm(x_star):.4f}", flush=True)

        # Shared initial conditions across the four topologies for this N.
        # Different topologies on the same x^0 isolate the effect of W.
        rng = np.random.default_rng(C.SEED + 1000 + N_new)
        x0_batch = [_random_x0(N, L, rng) for _ in range(C.R_SYNC)]

        stride = 50
        curves: dict[str, tuple] = {}

        for name, gen in REGULAR.items():
            print(f"--- N={N_new}  topology={name}  "
                  f"({C.R_SYNC} runs x {C.MAX_ITER_SYNC} iter) ---",
                  flush=True)
            _, W = gen(N)
            errs_dist = _run_batch(W, x_star, x0_batch,
                                   label=f"N={N_new} {name}")
            rows.extend(_checkpoint_rows(errs_dist, N_new, name))
            # Intermediate save per topology
            pd.DataFrame(rows).to_csv(C.RESULTS / "sync_regular.csv", index=False)

            # Mean + 90% CI band over R reps (for the convergence plot).
            mean_traj, low_traj, high_traj = mean_ci(list(errs_dist),
                                                     confidence=0.90)
            iters = np.arange(1, C.MAX_ITER_SYNC + 1)
            sel = iters[::stride]
            curves[name] = (sel,
                            mean_traj[::stride],
                            low_traj[::stride],
                            high_traj[::stride])

            print(f"[sync] N={N_new:2d}  {name:9s}  "
                  f"k=5e3 mean={errs_dist[:, 4999].mean():.3e}  "
                  f"k=1e4 mean={errs_dist[:, 9999].mean():.3e}")

        convergence_plot(
            curves,
            xlabel="iteration k",
            ylabel="mean relative error e_k",
            title=f"Distributed synchronous algorithm, N={N_new} "
                  f"(mean over R={C.R_SYNC})",
            out=C.RESULTS / f"sync_convergence_N{N_new}.png",
        )
        print(f"[sync] N={N_new}  4 topologies done")

    return pd.DataFrame(rows)


def _pivot(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot: (N, k) rows  x  topology columns, in the canonical order."""
    piv = df.pivot_table(index=["N", "k"], columns="topology", values=field)
    ordered_cols = list(REGULAR.keys())
    return piv[[c for c in ordered_cols if c in piv.columns]]


def main() -> None:
    print("=" * 70, flush=True)
    print("run_sync_regular.py : distributed synchronous algorithm",
          flush=True)
    print("                      on the four regular baseline graphs",
          flush=True)
    print(f"  N sizes           : {N_SIZES}", flush=True)
    print(f"  R (sample paths)  : {C.R_SYNC}", flush=True)
    print(f"  Max iterations    : {C.MAX_ITER_SYNC}", flush=True)
    print(f"  Checkpoints       : {C.SYNC_CHECKPOINTS}", flush=True)
    print(f"  Regular topologies: {list(REGULAR.keys())}", flush=True)
    print("=" * 70, flush=True)

    df = experiment_regular()
    df.to_csv(C.RESULTS / "sync_regular.csv", index=False)

    print("\n==== Distributed sync : mean relative error  (your Table 2) ====")
    print(_pivot(df, "mean_err"))
    print("\n==== Distributed sync : 90% CI width  (your Table 3) ====")
    print(_pivot(df, "ci_width"))


if __name__ == "__main__":
    main()
