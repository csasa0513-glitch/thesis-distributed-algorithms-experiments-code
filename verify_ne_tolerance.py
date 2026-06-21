"""
Mini-test (~10 minutes): verify the claim that async N=20 errors are
'NE-precision-floor-limited', not algorithm-limited.

How it works
------------
1. Solve NE for N=20 with two different tolerances:
     tol = 1e-9  (current "tight" reference x*_tight)
     tol = 1e-5  (loose reference x*_loose, imitating Koshal's apparent precision)
2. Compute the relative difference between the two reference points:
     floor_predicted := max|x*_tight - x*_loose| / max|x*_tight|
   This is the *predicted* floor any algorithm would see if Koshal's
   loose NE were used as reference.
3. Load the existing async_regular.csv and predict which cells would
   saturate at floor_predicted vs which would not change.
4. (Optional, ~10 min) re-run async on N=20 complete and wheel with the
   loose reference, to verify the predicted floor empirically.

Usage
-----
    python verify_ne_tolerance.py                # prediction only (~5 s)
    python verify_ne_tolerance.py --run-async    # also re-run async (~10 min)

Outputs printed to stdout. Does not modify any other file.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import config as C
from games.nash_cournot import solve_NE


N_TEST: int = 20
TIGHT_TOL: float = 1e-9
LOOSE_TOL: float = 1e-5


def predicted_floor() -> tuple[float, np.ndarray, np.ndarray]:
    """Return (floor, x_tight, x_loose) for N=20."""
    C.resample(N_TEST)

    print(f"Solving NE with tol = {TIGHT_TOL:.0e} ...", flush=True)
    t0 = time.time()
    x_tight = solve_NE(tol=TIGHT_TOL)
    print(f"  done in {time.time() - t0:.1f}s   ||x*_tight|| = "
          f"{np.linalg.norm(x_tight):.6f}", flush=True)

    print(f"Solving NE with tol = {LOOSE_TOL:.0e} ...", flush=True)
    t0 = time.time()
    x_loose = solve_NE(tol=LOOSE_TOL)
    print(f"  done in {time.time() - t0:.1f}s   ||x*_loose|| = "
          f"{np.linalg.norm(x_loose):.6f}", flush=True)

    denom = float(np.max(np.abs(x_tight)))
    diff = float(np.max(np.abs(x_tight - x_loose)))
    floor = diff / denom
    return floor, x_tight, x_loose


def predict_table(floor: float) -> None:
    """Show prediction for each async N=20 cell."""
    df = pd.read_csv(C.RESULTS / "async_regular.csv")
    df_20 = df[df["N"] == N_TEST].copy()
    df_20["current_err"] = df_20["mean_err"]
    df_20["predicted_at_1e-5"] = df_20["mean_err"].apply(
        lambda e: floor if e < floor else e
    )
    df_20["change"] = df_20.apply(
        lambda r: "SATURATE" if r["current_err"] < floor else "no change",
        axis=1,
    )
    print(df_20[
        ["graph", "k", "current_err", "predicted_at_1e-5", "change"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3e}"))


def rerun_async_with_loose_ne(x_loose: np.ndarray) -> None:
    """Optional: rerun async on N=20 complete + wheel, using x_loose as reference."""
    from graphs.generators import REGULAR
    from algorithms.async_gossip import run as run_async

    print("\n" + "=" * 70)
    print(f"  Re-running async N={N_TEST} (complete + wheel) with loose NE")
    print("=" * 70)

    C.resample(N_TEST)
    rng = np.random.default_rng(C.SEED + 2000 + N_TEST)
    R = 10                    # fewer reps for speed (currently 50)
    max_iter = 50_000         # half of full sweep
    denom_loose = float(np.max(np.abs(x_loose)))

    for name in ["complete", "wheel"]:
        gen = REGULAR[name]
        _, W_seed = gen(N_TEST)
        errs = np.zeros(R)
        t0 = time.time()
        for r in range(R):
            x0 = rng.uniform(0.0, C.INIT_SCALE, size=(N_TEST, 2 * C.L))
            out = run_async(
                W=None,
                x_star=x_loose,             # <- use loose NE here
                graph=name,
                N=N_TEST,
                max_iter=max_iter,
                eps=0.0,
                record_every=0,
                x0=x0,
            )
            errs[r] = out["rel_err"][-1]
        dt = time.time() - t0
        print(f"  {name:9s}  R={R}  k={max_iter}  "
              f"mean_err_loose = {errs.mean():.3e}  "
              f"(time: {dt:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-async", action="store_true",
                        help="Also re-run async on N=20 with loose NE (~10 min)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  NE tolerance verification (N = {N_TEST})")
    print(f"    tight tol = {TIGHT_TOL:.0e}")
    print(f"    loose tol = {LOOSE_TOL:.0e}")
    print("=" * 70)

    floor, x_tight, x_loose = predicted_floor()

    print(f"\n  Predicted NE-precision floor at tol={LOOSE_TOL:.0e}:")
    print(f"    floor = max|x_tight - x_loose| / max|x_tight| = {floor:.3e}")
    print(f"    (any algorithm error below this would be invisible)\n")

    print("Prediction table (async N=20 cells):")
    print("-" * 70)
    predict_table(floor)

    if args.run_async:
        rerun_async_with_loose_ne(x_loose)


if __name__ == "__main__":
    main()
