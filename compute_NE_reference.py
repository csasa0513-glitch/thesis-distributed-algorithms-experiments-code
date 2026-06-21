"""
Compute and save Nash equilibrium references for N=20 and N=50.

This is a standalone utility for offline inspection and debugging.
It is not part of the main experiment pipeline.

Output:
    results/NE_N{N}.npz
"""
from __future__ import annotations

import time

import numpy as np

import config as C
from games.nash_cournot import solve_NE


N_SIZES: tuple[int, ...] = (20, 50)


def main() -> None:
    print("=" * 72)
    print("  Computing Nash equilibrium references")
    print(f"  Master seed   : {C.SEED}")
    print(f"  Sizes         : {N_SIZES}")
    print("=" * 72)

    for N_val in N_SIZES:
        print(f"\n--- N = {N_val} ---", flush=True)
        C.resample(N_val)

        t0 = time.time()
        x_star = solve_NE()
        dt = time.time() - t0

        x_norm  = float(np.linalg.norm(x_star))
        x_inf   = float(np.max(np.abs(x_star)))
        x_min   = float(np.min(x_star))

        X = x_star.reshape(N_val, 2 * C.L)
        g = X[:, : C.L]
        s = X[:, C.L :]

        # Player feasibility check: sum_l g_il = sum_l s_il.
        per_player_g = g.sum(axis=1)     
        per_player_s = s.sum(axis=1)     
        feas_residual = float(np.max(np.abs(per_player_g - per_player_s)))

        # Extra diagnostic only: compare market totals by location.
        # This is not the player feasibility constraint from the model.
        bar_g = g.sum(axis=0)            
        bar_s = s.sum(axis=0)            
        loc_clearing = float(np.max(np.abs(bar_g - bar_s)))

        
        out_path = C.RESULTS / f"NE_N{N_val}.npz"
        np.savez(
            out_path,
            x_star=x_star,
            N=np.int64(N_val),
            seed=np.int64(C.SEED),
            norm_x_star=np.float64(x_norm),
            timestamp=np.bytes_(time.strftime("%Y-%m-%d %H:%M:%S")),
        )

        print(f"  solved in              : {dt:.2f}s", flush=True)
        print(f"  ||x*||_2               : {x_norm:.6f}", flush=True)
        print(f"  max_i |x*_i|           : {x_inf:.6f}", flush=True)
        print(f"  min_i x*_i             : {x_min:.6e}", flush=True)
        print(f"  sum_l bar_s_star       : {bar_s.sum():.6f}", flush=True)
        print(f"  sum_l bar_g_star       : {bar_g.sum():.6f}", flush=True)
        print(f"  per-player feasibility : max_i |sum_l g_il - sum_l s_il| = "
              f"{feas_residual:.3e}"
              f"  (Koshal eq. (64) equality; should be ~NE tol)", flush=True)
        print(f"  per-location clearing  : max_l |bar_g[l] - bar_s[l]|    = "
              f"{loc_clearing:.3e}"
              f"  (diagnostic, not a constraint)", flush=True)
        print(f"  Saved                  : {out_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
