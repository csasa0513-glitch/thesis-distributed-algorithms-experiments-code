"""
Diagnostic: is x0 ~ Uniform[0, INIT_SCALE]^{2S} already in X_i ?

The feasible set X_i (Koshal 2016 eq. 64; see games/nash_cournot.py) is
    g_il >= 0,  s_il >= 0,
    g_il <= Cap_il = 500,
    sum_l s_il  <=  sum_l g_il.                    (*)

Constraints 1 and 2 are satisfied by Uniform[0, scale] for scale <= 500.
Only (*) can fail.

When only (*) is active, the projection has a closed form:
    if  diff = sum(s) - sum(g) > 0
        lam        = diff / (2S)
        g_proj     = g + lam            (raise production)
        s_proj     = s - lam            (lower sales)
        ||x - Pi|| = sqrt(2S) * lam     =  diff / sqrt(2S)
    else
        Pi(x) = x

This is fast enough for large M.  We use it because the full Dykstra
projection in nash_cournot.project_Xi is overkill when only the coupling
constraint is violated.

Run:
    python check_init_feasibility.py
"""
from __future__ import annotations

import numpy as np

import config as C


M = 100_000                   # samples per init_scale
INIT_SCALES = (1.0, 5.0, 10.0, 50.0, 100.0)
CAP = 500.0


def measure(init_scale: float, M: int, rng: np.random.Generator) -> dict:
    """Vectorized feasibility / projection-distance measurement."""
    L = C.L
    x = rng.uniform(0.0, init_scale, size=(M, 2 * L))
    g_sum = x[:, :L].sum(axis=1)
    s_sum = x[:, L:].sum(axis=1)
    diff  = s_sum - g_sum                       # > 0 means infeasible

    # Box constraints: g <= 500 and g, s >= 0
    box_ok = (init_scale <= CAP)                # true for all scales in our list
    coupling_ok = diff <= 0.0
    feasible    = box_ok & coupling_ok

    # closed-form half-space projection distance
    proj_dist = np.where(diff > 0.0, diff / np.sqrt(2.0 * L), 0.0)
    proj_dist_infeasible = proj_dist[diff > 0.0]

    return {
        "init_scale":      init_scale,
        "n_samples":       M,
        "p_feasible":      float(feasible.mean()),
        "mean_proj_dist":  (float(proj_dist_infeasible.mean())
                            if proj_dist_infeasible.size else 0.0),
        "max_proj_dist":   (float(proj_dist_infeasible.max())
                            if proj_dist_infeasible.size else 0.0),
        "mean_s_minus_g":  float(diff.mean()),
        "std_s_minus_g":   float(diff.std(ddof=1)),
    }


def main() -> None:
    # Use the same Cournot instance as the experiments at N = 50.
    C.resample(50)
    rng = np.random.default_rng(C.SEED + 99_999)

    print("=" * 90, flush=True)
    print(f"Feasibility of  x0 ~ Uniform[0, init_scale]^(2S)  on X_i,  L = {C.L}",
          flush=True)
    print(f"M = {M} samples per init_scale", flush=True)
    print("=" * 90, flush=True)
    print(f"{'init_scale':>10}  {'P(in X_i)':>10}  "
          f"{'mean dist to X_i':>18}  {'max dist to X_i':>18}  "
          f"{'E[sum s - sum g]':>18}", flush=True)
    print("-" * 90, flush=True)

    for scale in INIT_SCALES:
        out = measure(scale, M, rng)
        print(f"{out['init_scale']:>10.1f}  "
              f"{out['p_feasible']:>10.4f}  "
              f"{out['mean_proj_dist']:>18.4f}  "
              f"{out['max_proj_dist']:>18.4f}  "
              f"{out['mean_s_minus_g']:>18.4f}", flush=True)

    print("", flush=True)
    print("Notes:", flush=True)
    print("* Box constraints g >= 0, s >= 0, g <= 500 are satisfied for every", flush=True)
    print("  init_scale <= 500. Only the coupling sum(s) <= sum(g) can fail.", flush=True)
    print("* sum(s) and sum(g) are i.i.d. sums of L Uniform[0, scale] variables,", flush=True)
    print("  so by symmetry P(sum(s) <= sum(g)) = 1/2 regardless of init_scale.", flush=True)
    print("* The projection therefore IS needed: it costs a small Euclidean", flush=True)
    print("  correction of magnitude ~  scale / sqrt(6S)  (analytic mean of", flush=True)
    print("  |sum(s)-sum(g)|/sqrt(2S) under i.i.d. uniform).", flush=True)
    print("* The smaller init_scale is, the smaller the absolute correction,", flush=True)
    print("  but the feasibility rate stays at 1/2.", flush=True)


if __name__ == "__main__":
    main()
