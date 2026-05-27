"""
Re-run the asynchronous gossip experiment on the grid topology at N = 20
under the Koshal node-then-uniform-neighbor sampling rule.

Background:
    The original N = 20 async data (preserved in
    `async_regular_tables_5_8_N20_only.csv`) was generated under
    Hanzely's uniform-edge scheme. After switching `async_gossip.py`
    to Koshal's node-then-neighbor scheme (Sec. 4.1, p. 689), the
    cycle, star, and complete topologies are unchanged (they are
    regular, so the two schemes are statistically equivalent), but
    the grid has boundary nodes of degree 3 and interior nodes of
    degree 4, so the two schemes differ. This script re-runs only the
    grid at N = 20 to bring the grid data in line with the new
    sampling rule.

Output:
    `results/async_grid_N20_koshal.csv`  — one row per horizon
    (k = 5e4, 1e5) with mean error and 90% CI width over R = 50
    sample paths.

Usage:
    python run_async_grid_N20.py
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

import config as C
from games.nash_cournot import solve_NE
from graphs.topologies import grid
from algorithms.async_gossip import run as run_async


# --------------------------------------------------------------------------
# Knobs (match run_async_regular.py)
# --------------------------------------------------------------------------
N_REPS: int = 50
K_HORIZONS: tuple[int, int] = (50_000, 100_000)
Z_90: float = 1.645


def _error_at(events: np.ndarray, errors: np.ndarray, target_k: int) -> float:
    idx = int(np.searchsorted(events, target_k, side="right") - 1)
    idx = max(0, min(idx, len(errors) - 1))
    return float(errors[idx])


def main() -> None:
    print("=" * 70)
    print(" run_async_grid_N20.py")
    print("=" * 70)

    C.resample(20)
    t0 = time.time()
    x_star = solve_NE()
    print(f"\n[0] Reference NE solved (N = 20): "
          f"||x*|| = {np.linalg.norm(x_star):.4f}, "
          f"elapsed = {time.time() - t0:.2f}s\n")

    G, _W = grid(C.N)
    max_events = max(K_HORIZONS) + 1
    err_runs = []
    t_start = time.time()
    for r in range(N_REPS):
        seed = (C.SEED
                + 100_000 * 20
                + 100 * (hash("grid") % 10_000)
                + 10 * r
                + 1)
        rng = np.random.default_rng(seed)
        out = run_async(G, x_star,
                        max_events=max_events,
                        eps=0.0,
                        rng=rng)
        err_runs.append([_error_at(out["events"], out["rel_err"], k)
                         for k in K_HORIZONS])
        if (r + 1) % 5 == 0 or (r + 1) == N_REPS:
            elapsed = time.time() - t_start
            est_total = elapsed * N_REPS / (r + 1)
            print(f"    [N=20 grid] rep {r+1:>2d}/{N_REPS} done   "
                  f"(elapsed {elapsed/60:.1f} min, "
                  f"estimated total {est_total/60:.1f} min)",
                  flush=True)

    err_runs = np.asarray(err_runs)
    mean = err_runs.mean(axis=0)
    sem = err_runs.std(axis=0, ddof=1) / np.sqrt(N_REPS)
    ci_w = 2.0 * Z_90 * sem

    rows = []
    for idx, k_val in enumerate(K_HORIZONS):
        rows.append({
            "N":          20,
            "topology":   "grid",
            "k":          k_val,
            "mean_error": float(mean[idx]),
            "ci90_width": float(ci_w[idx]),
        })
    df = pd.DataFrame(rows)
    out_path = C.RESULTS / "async_grid_N20_koshal.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[done] saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
