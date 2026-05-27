"""
Quick check: is the centralized baseline at N=50 systematically lower
than at N=20, or is it a random game-instance effect?

For 5 different master seeds we redraw all Cournot coefficients
(D_INTERCEPT, A_COEF, B_COEF), solve the reference NE, and run the
centralised projected-gradient sync algorithm (one path each) at
N=20 and N=50.

The relative error at k=5000 and k=10000 is recorded.

Total wall-clock: about 10-20 minutes (10 NE solves + 10 sync runs).

Usage:
    python verify_centralized_seed_sensitivity.py
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

import config as C
from games.nash_cournot import solve_NE, project_Xi
from algorithms.sync_koshal import run as run_sync


SEEDS = [42, 123, 2024, 7, 999]
N_VALUES = [20, 50]
CHECKPOINTS = (5_000, 10_000)


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial point uniform in [0, INIT_SCALE], projected onto X_i."""
    raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
    x = np.empty_like(raw)
    for i in range(N):
        x[i] = project_Xi(i, raw[i])
    return x


def _reset_game(master_seed: int, N: int) -> None:
    """Redraw all Cournot coefficients with the given master seed."""
    # Override module-level SEED so resample(N) uses SEED + N
    C.SEED = master_seed
    # D_INTERCEPT is set at import time and not touched by resample();
    # redraw it explicitly here for full game-instance independence
    rng_master = np.random.default_rng(master_seed)
    C.D_INTERCEPT = rng_master.uniform(90.0, 100.0, size=C.L)
    # A_COEF, B_COEF, CAP, and N
    C.resample(N)


def main() -> None:
    print("=" * 70)
    print(" verify_centralized_seed_sensitivity.py")
    print("=" * 70)
    print(f" seeds      = {SEEDS}")
    print(f" N values   = {N_VALUES}")
    print(f" checkpoints= {CHECKPOINTS}")
    print()

    rows = []
    t_total = time.time()
    for s_idx, master_seed in enumerate(SEEDS):
        for N in N_VALUES:
            _reset_game(master_seed, N)
            t0 = time.time()
            x_star = solve_NE()
            t_ne = time.time() - t0

            rng = np.random.default_rng(master_seed * 1000 + N + 1)
            x0 = _random_x0(N, C.L, rng)

            # W is irrelevant under known_aggregate=True (the algorithm only
            # uses the true aggregate bar_s for the gradient); provide
            # identity as a valid doubly stochastic placeholder.
            W_dummy = np.eye(N)

            t1 = time.time()
            out = run_sync(W_dummy, x_star,
                           max_iter=max(CHECKPOINTS),
                           eps=0.0,
                           record_every=1,
                           x0=x0,
                           known_aggregate=True)
            t_sync = time.time() - t1

            for k in CHECKPOINTS:
                err = float(out["rel_err"][k - 1])
                rows.append({"seed": master_seed, "N": N,
                             "k": k, "rel_err": err})
            print(f"  [seed={master_seed:5d}, N={N:2d}]  "
                  f"err@5e3={out['rel_err'][4999]:.3e}  "
                  f"err@1e4={out['rel_err'][9999]:.3e}   "
                  f"(NE {t_ne:.1f}s, sync {t_sync:.1f}s)", flush=True)
        print()

    df = pd.DataFrame(rows)
    out_path = C.RESULTS / "centralized_seed_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"saved -> {out_path}")

    print("\n=== Summary (mean / min / max over the 5 seeds) ===")
    for k in CHECKPOINTS:
        for N in N_VALUES:
            errs = df[(df["k"] == k) & (df["N"] == N)]["rel_err"].values
            print(f"  k={k:6d}, N={N:2d}:  mean={errs.mean():.3e}   "
                  f"min={errs.min():.3e}   max={errs.max():.3e}")

    print("\n=== Pairwise (does N=50 < N=20 hold across seeds?) ===")
    for k in CHECKPOINTS:
        wins = 0
        for s in SEEDS:
            e20 = df[(df["seed"] == s) & (df["N"] == 20) & (df["k"] == k)]["rel_err"].iloc[0]
            e50 = df[(df["seed"] == s) & (df["N"] == 50) & (df["k"] == k)]["rel_err"].iloc[0]
            won = "Y" if e50 < e20 else "N"
            if e50 < e20:
                wins += 1
            print(f"  k={k:6d}, seed={s:5d}:  N=20 -> {e20:.3e},   "
                  f"N=50 -> {e50:.3e}   N=50 < N=20? {won}")
        print(f"  -> N=50 won {wins}/{len(SEEDS)} times at k={k}")

    print(f"\nTotal wall-clock: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
