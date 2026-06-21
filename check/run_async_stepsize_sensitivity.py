"""
Asynchronous-gossip stepsize sensitivity sweep (thesis extension)
==================================================================

Purpose
-------
Sweep the numerator c of the diminishing stepsize rule
    alpha_{k,i} = c / Gamma_i^k
over c in {1, 3, 5, 9} on the four regular static graphs (cycle,
wheel, grid, complete) at N in {20, 50}. The output is NOT part of
the Koshal reproduction; it is a thesis-side diagnostic that
strengthens the sync/async comparison story.

Motivation
----------
Koshal et al. (2016) prove convergence of the asynchronous gossip
algorithm for any 1/k-type diminishing rule satisfying the Robbins-
Monro conditions (sum diverges, sum-of-squares converges), and their
*theoretical* analysis takes alpha = 1/k. Their *numerical* tables
(5-8) use the heavier numerator alpha = 9/k. The factor 9 is therefore
a tuning constant whose impact is not separately reported in the
paper. This sweep isolates the c-effect on the same graphs and
horizons as run_async_regular.py so the reader can see exactly how
much of the async performance comes from the algorithm and how much
from the tuned numerator.

This script is intentionally separate from run_async_regular.py:
the regular file is the strict Koshal Tables 5-8 reproduction, and
must keep alpha = 9/k. Mixing the c-sweep into that file would blur
the reproduction-vs-extension boundary.

Design: paired across c
-----------------------
For each (N, graph, rep) cell we sample ONE initial point x_i^0
and then run the algorithm four times -- once per c value -- starting
from this shared x0. This pairs the four c results at the same
sample-path, removing initial-point noise from the c-comparison and
making the (per-rep) gap between e@k for c=1 and c=9 directly
attributable to the stepsize. Across reps, the seeds are independent
(unpaired), so the standard error and 90% CI reflect cross-sample
variability the same way Koshal does.

Algorithm: same as run_async_regular.py except for `step_scale`.
Per-event indexing: k counts pairwise gossip events (one (i, j)
exchange = one event), matching run_async_regular.py.

Output schema (results/async_stepsize_sensitivity.csv)
------------------------------------------------------
One row per (c, N, graph, checkpoint k). Designed to be filtered
on ``experiment_block`` and then optionally concatenated against
``async_regular.csv`` for c = 9 sanity-check.

    experiment_block : str    constant "async_stepsize_sensitivity"
                                 (tag for filtering / merging)
    step_rule        : str    constant "c/local_update_count"
                                 (human-readable identifier; the rule
                                 itself does not vary in this sweep)
    step_scale       : float  in {1.0, 3.0, 5.0, 9.0}  (the c value)
    algorithm        : str    constant "async"
    budget_type      : str    constant "gossip_event"
    N                : int    in {20, 50}
    graph         : str    in {cycle, wheel, grid, complete}
    k                : int    in K_HORIZONS (= 50_000, 100_000)
    mean_err         : float  Koshal eq. (65), averaged over R reps
    ci_width         : float  2 * z_{0.90} * s_R / sqrt(R)
    n_runs           : int    = R

Optional figure (one per N):
    results/async_stepsize_sensitivity_N{20,50}.png
        Mean convergence curves stratified by c, one panel per graph.

Cost note
---------
This sweep performs roughly 4x the work of run_async_regular.py
(four c values for the same (N, graph, rep) grid):

    4 c-values * 2 N * 4 graphs * R reps * MAX_EVENTS

At R = 50 and MAX_EVENTS = 1e5 that is ~1.6e8 gossip events, several
hours on a single CPU. Use R = 10 for a quick sanity check.

Usage:
    python run_async_stepsize_sensitivity.py
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

import config as C
from games.nash_cournot import project_Xi, solve_NE
from graphs.generators import REGULAR
from algorithms.async_gossip import run as run_async
from analysis.plots import convergence_plot, mean_ci, Z_TABLE


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
N_REPS: int = C.R
N_SIZES: tuple[int, ...] = (20, 50)
C_VALUES: tuple[float, ...] = (1.0, 3.0, 5.0, 9.0)
K_HORIZONS: tuple[int, int] = (50_000, 100_000)
MAX_EVENTS: int = max(K_HORIZONS) + 1

# Explicit graph order (decoupled from REGULAR.keys() iteration order).
GRAPHS: tuple[str, ...] = ("cycle", "wheel", "grid", "complete")
GRAPH_ID: dict[str, int] = {name: idx for idx, name in enumerate(GRAPHS)}

# Constant CSV tags (output schema columns).
EXPERIMENT_BLOCK: str = "async_stepsize_sensitivity"
STEP_RULE:        str = "c/local_update_count"
ALGORITHM_LABEL:  str = "async"
BUDGET_TYPE:      str = "gossip_event"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_graph(name: str, N_val: int) -> tuple[nx.Graph, np.ndarray]:
    G, W = REGULAR[name](N_val)
    return G, W


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial decision, projected onto each player's X_i."""
    xi = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))   # xi_i ~ U([0, INIT_SCALE])^{2L}
    for i in range(N):
        xi[i] = project_Xi(i, xi[i])
    return xi


def _error_at(events: np.ndarray, errors: np.ndarray, target_k: int) -> float:
    idx = int(np.searchsorted(events, target_k, side="right") - 1)
    idx = max(0, min(idx, len(errors) - 1))
    return float(errors[idx])


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def experiment_stepsize_sensitivity(
    c_values: tuple[float, ...] = C_VALUES,
) -> pd.DataFrame:
    """Sweep c in `c_values` over the four regular graphs, paired by c.

    For each (N, graph, rep) we sample ONE x0 and then run the
    asynchronous algorithm |c_values| times from this shared x0, varying
    only `step_scale`. The (rng, x0) pairing pinpoints the c-effect
    against a held-fixed sample path.
    """
    rows: list[dict] = []
    # Trajectories collected for the optional sensitivity figure:
    # traj_per_N[N][c][graph] = list of per-rep rel_err arrays.
    traj_per_N: dict[int, dict[float, dict[str, list[np.ndarray]]]] = {
        N: {c: {t: [] for t in GRAPHS} for c in c_values}
        for N in N_SIZES
    }
    events_axes: dict[int, dict[str, np.ndarray | None]] = {
        N: {t: None for t in GRAPHS} for N in N_SIZES
    }

    for N_new in N_SIZES:
        print(f"\n=== N = {N_new} : solving NE ===", flush=True)
        C.resample(N_new)
        x_star = solve_NE()
        print(f"  ||x*|| = {np.linalg.norm(x_star):.4f}", flush=True)
        L = C.L

        for name in GRAPHS:
            print(f"--- N={N_new}  graph={name}  "
                  f"(R={N_REPS} reps x {len(c_values)} c-values) ---",
                  flush=True)
            # Collect per-c errs per checkpoint: shape (R, |c_values|, |K|)
            err_at_horizons = np.empty((N_REPS, len(c_values), len(K_HORIZONS)))

            for r in range(N_REPS):
                # One seed per (N, graph, rep); reused across all c-values.
                # Seed formula mirrors run_async_regular.py for consistency.
                seed = (C.SEED
                        + 100_000 * N_new
                        + 1_000 * GRAPH_ID[name]
                        + 10 * r
                        + 1)
                # Per-rep rng for x0 sampling -- consumed ONCE so the same
                # x0 feeds all 4 c-runs. A fresh rng is then created per
                # c-run so that the gossip path inside `run_async` is
                # reproducible and independent across c.
                x0_rng = np.random.default_rng(seed)
                x0 = _random_x0(C.N, L, x0_rng)

                G, _W = _make_graph(name, C.N)

                for ci, c in enumerate(c_values):
                    # Independent rng per c-run, derived deterministically
                    # from (seed, ci). This isolates the gossip path
                    # randomness from the x0 sampling RNG above and keeps
                    # every (rep, c) cell reproducible.
                    cell_rng = np.random.default_rng(
                        np.random.SeedSequence([seed, int(c * 100), ci])
                    )
                    out = run_async(
                        G, x_star,
                        max_events=MAX_EVENTS,
                        eps=0.0,
                        rng=cell_rng,
                        x0=x0,
                        step_scale=float(c),
                    )
                    for ki, k_val in enumerate(K_HORIZONS):
                        err_at_horizons[r, ci, ki] = _error_at(
                            out["events"], out["rel_err"], k_val
                        )
                    traj_per_N[N_new][c][name].append(out["rel_err"])
                    if events_axes[N_new][name] is None:
                        events_axes[N_new][name] = out["events"]

                if (r + 1) % 5 == 0 or (r + 1) == N_REPS:
                    print(f"    [N={N_new} {name:8s}] rep {r+1:>2d}/{N_REPS} "
                          f"done (all {len(c_values)} c-values)", flush=True)

            # Aggregate: mean and 90% CI per (c, k).
            for ci, c in enumerate(c_values):
                for ki, k_val in enumerate(K_HORIZONS):
                    col = err_at_horizons[:, ci, ki]
                    sem = col.std(ddof=1) / np.sqrt(N_REPS)
                    rows.append({
                        "experiment_block": EXPERIMENT_BLOCK,
                        "step_rule":        STEP_RULE,
                        "step_scale":       float(c),
                        "algorithm":        ALGORITHM_LABEL,
                        "budget_type":      BUDGET_TYPE,
                        "N":                N_new,
                        "graph":         name,
                        "k":                k_val,
                        "mean_err":         float(col.mean()),
                        "ci_width":         float(2.0 * Z_TABLE[0.90] * sem),
                        "n_runs":           int(N_REPS),
                    })
            # Intermediate save per graph so a long run is not lost on
            # interruption.
            pd.DataFrame(rows).to_csv(
                C.RESULTS / "async_stepsize_sensitivity.csv", index=False)

            # Per-graph console summary across c.
            for ci, c in enumerate(c_values):
                m1, m2 = err_at_horizons[:, ci, :].mean(axis=0)
                print(f"  c={c:>4.1f}  k={K_HORIZONS[0]:>6d} mean={m1:.3e}  "
                      f"k={K_HORIZONS[1]:>7d} mean={m2:.3e}", flush=True)

    # Save and emit figures.
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "async_stepsize_sensitivity.csv", index=False)
    _emit_figures(traj_per_N, events_axes, c_values)
    return df


def _emit_figures(
    traj_per_N: dict[int, dict[float, dict[str, list[np.ndarray]]]],
    events_axes: dict[int, dict[str, np.ndarray | None]],
    c_values: tuple[float, ...],
) -> None:
    """One figure per N: mean +/- 90% CI band, one curve per c, one panel per graph.

    We use a separate `convergence_plot` per N (4 curves overlaid -- one per
    c -- for each panel) to keep visual comparison within a panel tight.
    """
    for N_new in traj_per_N:
        # One figure per (N, graph) — 4 c-curves overlaid.
        for name in GRAPHS:
            curves: dict[str, tuple] = {}
            events_ref = events_axes[N_new][name]
            if events_ref is None:
                continue
            for c in c_values:
                run_list = traj_per_N[N_new][c][name]
                if not run_list:
                    continue
                mean_traj, low_traj, high_traj = mean_ci(
                    run_list, confidence=0.90)
                curves[f"c = {c:g}"] = (events_ref, mean_traj, low_traj, high_traj)
            if curves:
                convergence_plot(
                    curves,
                    xlabel="gossip event k",
                    ylabel="mean relative error e_k",
                    title=f"Async stepsize sensitivity, N={N_new}, "
                          f"graph = {name}",
                    out=C.RESULTS / f"async_stepsize_sensitivity_"
                                    f"N{N_new}_{name}.png",
                )


def _pivot(df: pd.DataFrame, field: str, N_val: int) -> pd.DataFrame:
    """For a fixed N, pivot: (step_scale, k) rows  x  graph columns."""
    sub = df[df["N"] == N_val]
    piv = sub.pivot_table(index=["step_scale", "k"],
                           columns="graph", values=field)
    return piv[[c for c in GRAPHS if c in piv.columns]]


def main() -> None:
    print("=" * 70, flush=True)
    print("run_async_stepsize_sensitivity.py -- thesis extension", flush=True)
    print("Sweep c in {1, 3, 5, 9} for alpha_{k,i} = c / Gamma_i^k", flush=True)
    print("on the same 4 regular graphs and horizons as Koshal Tables 5-8.",
          flush=True)
    print(f"  N sizes           : {N_SIZES}", flush=True)
    print(f"  R (sample paths)  : {N_REPS}  [= C.R]", flush=True)
    print(f"  c values          : {C_VALUES}", flush=True)
    print(f"  Checkpoints       : {K_HORIZONS}", flush=True)
    print(f"  Graphs        : {GRAPHS}", flush=True)
    print(f"  Pairing           : paired-by-c, unpaired across reps",
          flush=True)
    print(f"  CSV tags          : experiment_block={EXPERIMENT_BLOCK!r}, "
          f"step_rule={STEP_RULE!r}", flush=True)
    print("=" * 70, flush=True)

    df = experiment_stepsize_sensitivity()

    print("\n==== Stepsize sensitivity, N=20 : mean relative error "
          "(Koshal eq. (65)) ====")
    print(_pivot(df, "mean_err", 20))
    print("\n==== Stepsize sensitivity, N=50 : mean relative error "
          "(Koshal eq. (65)) ====")
    print(_pivot(df, "mean_err", 50))


if __name__ == "__main__":
    main()
