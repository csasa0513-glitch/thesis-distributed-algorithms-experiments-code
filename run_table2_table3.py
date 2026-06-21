"""
Synchronous algorithm on the four baseline regular graphs
=========================================================

This script runs the synchronous distributed algorithm from Chapter 4
on the four baseline regular graphs used in Chapter 6 of the thesis:
Cycle, Wheel, Grid, and Complete.

Purpose
-------
The script provides the synchronous results reported for the regular-graph
extension in Section 6.2. For each graph and each network size
N in {20, 50}, it evaluates the algorithm at the fixed iteration
horizons specified in ``C.SYNC_CHECKPOINTS`` and records the mean
relative error together with the width of the two-sided 90% normal-
approximation confidence interval.

The reported error is the relative error defined in Section 6.1 of the
thesis. At a fixed horizon k, it is computed with respect to the
reference Nash equilibrium and then averaged over R runs.

Experimental design
-------------------
For each table cell, the script uses R = C.R independent runs.
For a fixed value of N, the game coefficients are sampled once and then
kept fixed across all graphs, runs, and checkpoints, in line with the
experimental setup described in Section 6.1 of the thesis.

Across runs, the randomness comes from the feasible random initial point.
For each run, the initialization is generated independently and then
projected onto the feasible set of each player.

Output
------
results/sync_regular.csv
    One row per (N, graph, checkpoint k) with the following columns:
    algorithm, budget_type, N, graph, k, mean_err, ci_width, n_runs

Figures
-------
results/sync_convergence_N20.png
results/sync_convergence_N50.png

These plots show the mean relative error together with the 90% confidence
band over the full synchronous trajectory.

Usage
-----
    python run_table2_table3.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from games.nash_cournot import solve_NE, project_Xi
from graphs.generators import REGULAR
from algorithms.sync_koshal import run as run_sync
from analysis.plots import convergence_plot, mean_ci, Z_TABLE


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
N_SIZES: tuple[int, ...] = (20, 50)
ALGORITHM_LABEL: str = "sync"          # constant CSV column; merge key with async file
BUDGET_TYPE: str    = "iteration"      # what `k` counts: one synchronous round

# Explicit graph order. We do NOT iterate over REGULAR.keys() / .items()
# directly because that would couple the experiment ordering -- and therefore
# the per-cell seed assignment via GRAPH_ID -- to the (dictionary literal)
# order of declarations in graphs/generators.py. Pinning the order here
# matches the parallel convention in run_table4_table5.py and makes any
# future addition to REGULAR a deliberate, reviewable change in both files.
GRAPHS: tuple[str, ...] = ("cycle", "wheel", "grid", "complete")

# Deterministic integer ID per graph, used to build reproducible
# per-cell seeds. We do NOT use hash(name) because Python 3 enables
# hash randomization by default (PYTHONHASHSEED).
GRAPH_ID: dict[str, int] = {name: idx for idx, name in enumerate(GRAPHS)}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _solve_NE_for(N_new: int) -> np.ndarray:
    """Resample the game data for N = N_new and solve the reference Nash equilibrium.

    For each fixed network size N, the coefficients are sampled once and
    then kept fixed across all graphs and runs, as in Section 6.1 of the thesis.
    """
    C.resample(N_new)
    return solve_NE()


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one random feasible initial point for the synchronous algorithm."""
    xi = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))   # xi_i ~ U([0, INIT_SCALE])^{2L}
    for i in range(N):
        xi[i] = project_Xi(i, xi[i])
    return xi


def _ci_width(values: np.ndarray) -> float:
    """Return the full width of the two-sided 90% normal-approximation CI of the 
    mean."""
    n = values.size
    if n < 2:
        return 0.0
    s = float(values.std(ddof=1))
    return 2.0 * Z_TABLE[0.90] * s / np.sqrt(n)


def _run_batch_unpaired(W: np.ndarray,
                        x_star: np.ndarray,
                        seeds: list[int],
                        label: str = "") -> np.ndarray:
    """
    Run the synchronous distributed algorithm R times on one fixed graph.

    Each run uses an independently generated feasible initial point.
    The returned array has shape (R, C.MAX_ITER_SYNC) and stores the
    relative error trajectory for every run.
    """
    R = len(seeds)
    N, L = C.N, C.L
    errs = np.empty((R, C.MAX_ITER_SYNC))
    for r, s in enumerate(seeds):
        cell_rng = np.random.default_rng(s)
        x0 = _random_x0(N, L, cell_rng)
        out = run_sync(
            W, x_star,
            max_iter=C.MAX_ITER_SYNC,
            eps=0.0,                 
            record_every=1,
            x0=x0,
            known_aggregate=False,   
        )
        assert out["rel_err"].size == C.MAX_ITER_SYNC, (
            f"Expected {C.MAX_ITER_SYNC} error records; got "
            f"{out['rel_err'].size}. Check eps=0.0 and record_every=1."
        )
        errs[r] = out["rel_err"]
        if (r + 1) % 10 == 0 or (r + 1) == R:
            print(f"    [{label}] run {r + 1}/{R} done", flush=True)
    return errs


def _checkpoint_rows(errs: np.ndarray,
                     N_new: int,
                     graph: str) -> list[dict]:
    """Extract the reported mean error and 90% CI width at the thesis checkpoints."""
    rows = []
    for k in C.SYNC_CHECKPOINTS:
        col = errs[:, k - 1]                 # k updates -> Python index k-1
        rows.append({
            "algorithm":   ALGORITHM_LABEL,
            "budget_type": BUDGET_TYPE,
            "N":           N_new,
            "graph":    graph,
            "k":           k,
            "mean_err":    float(col.mean()),
            "ci_width":    _ci_width(col),
            "n_runs":      int(errs.shape[0]),
        })
    return rows


def experiment_regular() -> pd.DataFrame:
    """Run the synchronous algorithm on Cycle, Wheel, Grid, and Complete."""
    rows: list[dict] = []

    for N_new in N_SIZES:
        print(f"\n=== N = {N_new} : solving NE ===", flush=True)
        x_star = _solve_NE_for(N_new)
        N, L = C.N, C.L
        print(f"  ||x*|| = {np.linalg.norm(x_star):.4f}", flush=True)

        stride = 50
        curves: dict[str, tuple] = {}

        for name in GRAPHS:
            gen = REGULAR[name]
            print(f"--- N={N_new}  graph={name}  "
                  f"({C.R} runs x {C.MAX_ITER_SYNC} iter) ---",
                  flush=True)
            _, W = gen(N)
            seeds = [C.SEED + 1_000_000 + 100_000 * N_new
                     + 1_000 * GRAPH_ID[name] + 10 * r + 1
                     for r in range(C.R)]
            errs_dist = _run_batch_unpaired(W, x_star, seeds,
                                            label=f"N={N_new} {name}")
            rows.extend(_checkpoint_rows(errs_dist, N_new, name))
            pd.DataFrame(rows).to_csv(C.RESULTS / "sync_regular.csv", index=False)

            mean_traj, low_traj, high_traj = mean_ci(list(errs_dist),
                                                     confidence=0.90)
            iters = np.arange(1, C.MAX_ITER_SYNC + 1)
            sel = iters[::stride]
            curves[name] = (sel,
                            mean_traj[::stride],
                            low_traj[::stride],
                            high_traj[::stride])

            ck = list(C.SYNC_CHECKPOINTS)
            print(f"[sync] N={N_new:2d}  {name:9s}  " + "  ".join(
                f"k={kk:>5d} mean error={errs_dist[:, kk - 1].mean():.3e}"
                for kk in ck
            ))

        convergence_plot(
            curves,
            xlabel="iteration k",
            ylabel="mean relative error e_k",
            title=f"Distributed synchronous algorithm, N={N_new} "
                  f"(mean over R={C.R})",
            out=C.RESULTS / f"sync_convergence_N{N_new}.png",
        )
        print(f"[sync] N={N_new}  4 graphs done")

    return pd.DataFrame(rows)


def _pivot(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot the reported results to the table layout used in Chapter 6."""
    piv = df.pivot_table(index=["N", "k"], columns="graph", values=field)
    piv = piv[[c for c in GRAPHS if c in piv.columns]]
    # Display only: capitalize column headers to match the thesis tables
    # (Cycle, Wheel, Grid, Complete). The underlying data/keys stay lowercase.
    piv.columns = [c.capitalize() for c in piv.columns]
    return piv


def main() -> None:
    print("=" * 70, flush=True)
    print("run_table2_table3.py -- Synchronous algorithm on the baseline regular graphs",
          flush=True)
    print("Chapter 6 regular-graph extension: Cycle, Wheel, Grid, Complete",
          flush=True)
    print(f"  N sizes           : {N_SIZES}", flush=True)
    print(f"  R (runs)          : {C.R}", flush=True)
    print(f"  Max iterations    : {C.MAX_ITER_SYNC}", flush=True)
    print(f"  Checkpoints       : {C.SYNC_CHECKPOINTS}", flush=True)
    print(f"  Graphs            : {list(GRAPHS)}", flush=True)
    print(f"  CSV label         : algorithm = {ALGORITHM_LABEL!r}", flush=True)
    print(f"  Budget type       : {BUDGET_TYPE!r}", flush=True)
    print("=" * 70, flush=True)

    df = experiment_regular()
    df.to_csv(C.RESULTS / "sync_regular.csv", index=False)

    print("\n==== Table 2 layout: Synchronous algorithm mean error after k iterations ====")
    print(_pivot(df, "mean_err"))
    print("\n==== Table 3 layout: Synchronous algorithm width of 90% confidence interval")
    print("==== of the mean error after k iterations ====")
    print(_pivot(df, "ci_width"))


if __name__ == "__main__":
    main()