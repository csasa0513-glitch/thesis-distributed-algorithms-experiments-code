"""
Asynchronous concurrence experiment on the four baseline graphs.

This script produces the stopping-time results reported in Table 6 of the
thesis. It uses the four baseline graphs (cycle, wheel, grid, complete)
at N = 20 and records the number of gossip events required for the players'
aggregate estimates to agree within 1e-3.

For each graph, the script runs R = 50 sample paths and reports the sample
mean and sample standard deviation of the stopping time. Additional spectral
and diagnostic quantities are also written to the CSV.

Output
------
results/async_table6.csv

Usage
-----
    python run_async_table6.py
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

import config as C
from algorithms.async_gossip import run as run_async
from games.nash_cournot import project_Xi, solve_NE
from graphs.generators import REGULAR
from analysis.metrics import async_



# Knobs. R = C.R sample paths.
N_TEST: int = 20
N_REPS: int = C.R                
EPS: float = 1e-3
MAX_EVENTS: int = 500_000

# Metadata columns for the output CSV.
ALGORITHM_LABEL:  str = "async"
EXPERIMENT_BLOCK: str = "table6"

GRAPHS: tuple[str, ...] = ("cycle", "wheel", "grid", "complete")
GRAPH_ID: dict[str, int] = {name: idx for idx, name in enumerate(GRAPHS)}


# Helpers
def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    xi = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))   # xi_i ~ U([0, INIT_SCALE])^{2L}
    for i in range(N):
        xi[i] = project_Xi(i, xi[i])
    return xi


def _make_graph(name: str, N_val: int):
    result = REGULAR[name](N_val)
    assert isinstance(result, tuple) and len(result) == 2, \
        f"REGULAR[{name!r}] must return (G, W); got {type(result).__name__}."
    G, W = result
    assert isinstance(G, nx.Graph), \
        f"First element must be nx.Graph; got {type(G).__name__}."
    return G, W


def _expected_W_spectrum(G: nx.Graph) -> tuple[float, float]:
    """Return (lambda_2(E[W]), sqrt(lambda_2(E[W])))."""
    return async_(G) ** 2, async_(G)


def main() -> None:
    print("=" * 78)
    print("  Async gossip: Table 6 results")
    print(f"  Concurrence threshold = {EPS:.0e}, N = {N_TEST}, R = {N_REPS}")
    print(f"  max_events = {MAX_EVENTS:,}")
    print(f"  Stepsize: alpha_{{k,i}} = 9 / Gamma_i^k")
    print("=" * 78)

    C.resample(N_TEST)
    print(f"Solving Nash equilibrium (N = {N_TEST}) ...", flush=True)
    x_star = solve_NE()
    print(f"  ||x*|| = {np.linalg.norm(x_star):.4f}\n", flush=True)

    rows: list[dict] = []
    for name in GRAPHS:
        # Graph generation is deterministic, so build it once and reuse it
        # across all reps (the spectral indicator and every gossip run use it).
        G, _ = _make_graph(name, C.N)
        lam2, sqrt_lam2 = _expected_W_spectrum(G)

        k_eps_runs: list[int] = []
        n_converged = 0
        final_rel_err: list[float] = []

        for r in range(N_REPS):
            # Unpaired design: each (graph, rep) cell uses its own random x0.
            seed = C.SEED + 9_000 + 17 * r + 100 * GRAPH_ID[name]
            rng = np.random.default_rng(seed)
            x0 = _random_x0(C.N, C.L, rng)
            out = run_async(
                G,
                x_star,
                max_events=MAX_EVENTS,
                eps=EPS,
                rng=rng,
                x0=x0,
                stop_metric="concurrence",
            )
            # k_eps cap behavior (see docstring "Non-convergence handling"):
            # right-censoring at MAX_EVENTS biases mean_k_eps DOWNWARD when
            # n_converged < N_REPS. Track n_converged explicitly so the
            # downstream reader can compensate.
            if out["k_eps"] is not None:
                k_eps_runs.append(int(out["k_eps"]))
                n_converged += 1
            else:
                k_eps_runs.append(MAX_EVENTS)
            final_rel_err.append(float(out["rel_err"][-1]))
            if (r + 1) % 10 == 0 or (r + 1) == N_REPS:
                print(f"  [{name:9s}] rep {r + 1:>2d}/{N_REPS} done", flush=True)

        k_eps_arr = np.asarray(k_eps_runs, dtype=float)
        rows.append({
            "experiment_block":          EXPERIMENT_BLOCK,
            "algorithm":                 ALGORITHM_LABEL,
            "graph":                  name,
            "n_reps":                    N_REPS,
            "n_reps_converged":          n_converged,
            "mean_k_eps":                float(k_eps_arr.mean()),
            "std_k_eps":                 float(k_eps_arr.std(ddof=1)) if N_REPS > 1 else 0.0,
            "median_k_eps":              float(np.median(k_eps_arr)),
            "min_k_eps":                 int(k_eps_arr.min()),
            "max_k_eps":                 int(k_eps_arr.max()),
            "lambda2_EW":                lam2,
            "sqrt_lambda2_EW":           sqrt_lam2,
            "mean_final_rel_err":        float(np.mean(final_rel_err)),
        })

    df = pd.DataFrame(rows)
    out_path = C.RESULTS / "async_table6.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ------------------------------------------------------------------
    # Pretty-print our own results on the baseline graphs.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Results on the baseline graphs")
    print("  (stopping-time and spectral values)")
    print("=" * 60)
    print(
        f"  {'Graph':10s}  "
        f"{'sqrt(lam)':>11s}  {'mean k_eps':>11s}  "
        f"{'std k_eps':>11s}"
    )
    print("-" * 60)
    for r in rows:
        print(
            f"  {r['graph']:10s}  "
            f"{r['sqrt_lambda2_EW']:>11.4e}  "
            f"{r['mean_k_eps']:>11,.0f}  "
            f"{r['std_k_eps']:>11,.0f}"
        )

    print("\nNotes:")
    print("  - Stopping rule: max_i ||N v_i - sum_j s_j^*|| < 1e-3.")
    print("  - Non-converged runs are capped at MAX_EVENTS.")


if __name__ == "__main__":
    main()
