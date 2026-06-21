"""
Asynchronous benchmark on the four baseline graphs.

Uses N = 20 and 50, runs R = 50 sample paths, and reports mean relative
error and 90% CI width at k = 5e4 and k = 1e5.

Output
------
results/async_regular.csv
    One row per (N, graph, k) with the mean relative error and the width of
    the 90% confidence interval.

Usage
-----
    python run_table4_table5.py
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


# Experiment parameters. R = C.R sample paths.
N_REPS: int = C.R
K_HORIZONS: tuple[int, int] = (50_000, 100_000)
ALGORITHM_LABEL: str = "async"          # constant CSV column; merge key with sync file
BUDGET_TYPE: str    = "gossip_event"    # what `k` counts: one pairwise gossip exchange

# Baseline graphs.
GRAPHS: tuple[str, ...] = ("cycle", "wheel", "grid", "complete")

# Stable IDs for reproducible seeds.
GRAPH_ID: dict[str, int] = {name: idx for idx, name in enumerate(GRAPHS)}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_graph(name: str, N_val: int):
    """Return (G, W) for a baseline graph."""
    result = REGULAR[name](N_val)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"REGULAR[{name!r}](N) must return a 2-tuple (G, W); got "
        f"{type(result).__name__}."
    )
    G, W = result
    assert isinstance(G, nx.Graph), \
        f"First element from REGULAR[{name!r}] must be nx.Graph; got {type(G).__name__}."
    assert isinstance(W, np.ndarray) and W.shape == (N_val, N_val), \
        f"Second element from REGULAR[{name!r}] must be N x N np.ndarray; got shape {getattr(W, 'shape', 'N/A')}."
    return G, W


def _error_at(events: np.ndarray, errors: np.ndarray, target_k: int) -> float:
    idx = int(np.searchsorted(events, target_k, side="right") - 1)
    idx = max(0, min(idx, len(errors) - 1))
    return float(errors[idx])


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial decision, projected onto each player's feasible set X_i."""
    xi = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))   # xi_i ~ U([0, INIT_SCALE])^{2L}
    for i in range(N):
        xi[i] = project_Xi(i, xi[i])
    return xi


def _run_one(G, x_star, rng: np.random.Generator,
             max_events: int, eps: float = 0.0,
             x0: np.ndarray | None = None):
    return run_async(G, x_star,
                     max_events=max_events,
                     eps=eps,
                     rng=rng,
                     x0=x0)


def experiment_async_regular() -> pd.DataFrame:
    rows = []
    max_events = max(K_HORIZONS) + 1
    for N_new in (20, 50):
        C.resample(N_new)
        x_star = solve_NE()

        # Collect convergence curves (one per graph) for the plot.
        curves: dict[str, tuple] = {}

        for name in GRAPHS:
            err_runs = []
            err_trajectories = []   
            events_ref = None       
            for r in range(N_REPS):
                seed = (C.SEED
                        + 100_000 * N_new
                        + 1_000 * GRAPH_ID[name]
                        + 10 * r
                        + 1)
                rng = np.random.default_rng(seed)
                x0 = _random_x0(C.N, C.L, rng)
                G, _W = _make_graph(name, C.N)
                out = _run_one(G, x_star, rng, max_events, x0=x0)
                err_runs.append([_error_at(out["events"], out["rel_err"], k)
                                 for k in K_HORIZONS])
                err_trajectories.append(out["rel_err"])
                if events_ref is None:
                    events_ref = out["events"]
                if (r + 1) % 5 == 0 or (r + 1) == N_REPS:
                    print(f"    [N={N_new} {name:8s}] rep {r+1:>2d}/{N_REPS} done",
                          flush=True)
            err_runs = np.asarray(err_runs)
            mean = err_runs.mean(axis=0)
            sem  = err_runs.std(axis=0, ddof=1) / np.sqrt(N_REPS)
            ci_w = 2.0 * Z_TABLE[0.90] * sem
            for idx, k_val in enumerate(K_HORIZONS):
                rows.append({
                    "algorithm":   ALGORITHM_LABEL,
                    "budget_type": BUDGET_TYPE,
                    "N":           N_new,
                    "graph":    name,
                    "k":           k_val,
                    "mean_err":    float(mean[idx]),
                    "ci_width":    float(ci_w[idx]),
                    "n_runs":      int(N_REPS),
                })
            # Convergence curve: mean and 90% CI band over the R reps.
            mean_traj, low_traj, high_traj = mean_ci(
                list(err_trajectories), confidence=0.90)
            # Align with the event index (events_ref); fallback to indices.
            x_axis = (events_ref if events_ref is not None
                      else np.arange(len(mean_traj)))
            curves[name] = (x_axis, mean_traj, low_traj, high_traj)
            print(f"[async] N={N_new:3d}  {name:8s}  "
                  f"e@5e4={mean[0]:.3e}  e@1e5={mean[1]:.3e}",
                  flush=True)
            pd.DataFrame(rows).to_csv(
                C.RESULTS / "async_regular.csv", index=False)

        # Save the convergence plot for this N.
        if curves:
            convergence_plot(
                curves,
                xlabel="gossip event k",
                ylabel="mean relative error e_k",
                title=f"Async algorithm, N={N_new} (mean over R={N_REPS})",
                out=C.RESULTS / f"async_convergence_N{N_new}.png",
            )
            print(f"[async] N={N_new}  convergence plot saved", flush=True)
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _pivot(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot to the thesis Table 4/5 layout: rows (N, k), columns the graphs."""
    piv = df.pivot_table(index=["N", "k"], columns="graph", values=field)
    piv = piv[[c for c in GRAPHS if c in piv.columns]]
    piv.columns = [c.capitalize() for c in piv.columns]
    return piv


def main() -> None:
    print("=" * 70, flush=True)
    print("run_table4_table5.py -- async benchmark on regular graphs", flush=True)
    print("Async gossip on the four baseline graphs", flush=True)
    print("Pair with run_table2_table3.py", flush=True)
    print(f"  R (sample paths)  : {N_REPS}  [= C.R]", flush=True)
    print(f"  Checkpoints       : {K_HORIZONS}", flush=True)
    print(f"  Regular graphs: {GRAPHS}", flush=True)
    print(f"  CSV merge key     : algorithm = {ALGORITHM_LABEL!r}", flush=True)
    print(f"  Budget type       : {BUDGET_TYPE!r}  (one k = one pairwise gossip)", flush=True)
    print("=" * 70, flush=True)

    df = experiment_async_regular()
    df.to_csv(C.RESULTS / "async_regular.csv", index=False)

    print("\n==== Table 4: Asynchronous algorithm: "
          "mean error after gossip events ====")
    print(_pivot(df, "mean_err"))
    print("\n==== Table 5: Asynchronous algorithm: width of 90% confidence")
    print("==== interval of the mean error after gossip events ====")
    print(_pivot(df, "ci_width"))


if __name__ == "__main__":
    main()