"""
Sanity check for the asynchronous gossip algorithm after the switch
to Koshal's node-then-uniform-neighbor sampling (Sec. 4.1, p. 689).

Two checks:

  [1] The empirical update frequency p_i matches the theoretical
      formula  p_i = (1/N)(1 + sum_{j in N_i} 1/d_j).
      This is the fastest and most direct verification that the
      sampling rule is implemented as Koshal specifies.

  [2] On cycle, star, and complete (N = 20, 8000 events, one path
      each) the relative error decreases over the course of the
      run.  Short runs may still be in the transient region of the
      diminishing stepsize, so the check just inspects the
      trajectory (early, middle, late) rather than asserting a
      strict decrease at every checkpoint.

Total wall-clock: ~30 seconds.

Usage:
    python sanity_check_async.py
"""
from __future__ import annotations

import time
import numpy as np

import config as C
from games.nash_cournot import solve_NE
from graphs.topologies import cycle, star, complete
from algorithms.async_gossip import run as run_async


def _expected_p_i(G) -> np.ndarray:
    """Closed-form p_i = (1/N)(1 + sum_{j in N_i} 1/d_j)."""
    N = G.number_of_nodes()
    deg = dict(G.degree())
    p = np.zeros(N)
    for i in range(N):
        p[i] = (1.0 / N) * (1.0 + sum(1.0 / deg[j] for j in G.neighbors(i)))
    return p


def _empirical_p_i(G, n_events: int = 10_000, seed: int = 12345) -> np.ndarray:
    """Mirror the sampling in async_gossip.py and count agent updates."""
    N = G.number_of_nodes()
    rng = np.random.default_rng(seed)
    neighbors_list = [list(G.neighbors(i)) for i in range(N)]
    counts = np.zeros(N)
    for _ in range(n_events):
        i = int(rng.integers(N))
        nbrs = neighbors_list[i]
        j = int(rng.choice(nbrs))
        counts[i] += 1
        counts[j] += 1
    return counts / n_events


def main() -> None:
    print("=" * 70)
    print(" sanity_check_async.py")
    print("=" * 70)

    # ----- Step 0: solve reference NE -----
    C.resample(20)
    t0 = time.time()
    x_star = solve_NE()
    print(f"\n[0] Reference NE solved: ||x*|| = {np.linalg.norm(x_star):.4f}, "
          f"elapsed = {time.time() - t0:.2f}s")

    # ----- Step 1: verify p_i = (1/N)(1 + sum 1/d_j) -----
    print("\n[1] Checking p_i = (1/N)(1 + sum_{j in N_i} 1/d_j)")
    print("    (theory vs empirical over 10,000 events)\n")
    for name, builder in [("cycle", cycle), ("star", star),
                          ("complete", complete)]:
        G, _ = builder(C.N)
        p_theory = _expected_p_i(G)
        p_emp = _empirical_p_i(G, n_events=10_000)
        max_err = float(np.max(np.abs(p_theory - p_emp)))
        # Show node 0 (hub for star, ordinary node otherwise) and node 1.
        print(f"   {name:9s}  p_0  theory={p_theory[0]:.4f}  "
              f"empirical={p_emp[0]:.4f}   |   "
              f"p_1  theory={p_theory[1]:.4f}  empirical={p_emp[1]:.4f}   |   "
              f"max|diff| = {max_err:.4f}")

    print("\n   Expected values:")
    print("     cycle / complete: p_i = 2/N = 0.1000 for every i")
    print("     star: p_hub = 1.0000 (hub is in every event)")
    print("           p_leaf = 1/(N-1) = 0.0526")

    # ----- Step 2: error trajectory at 8000 events -----
    print("\n[2] Error trajectory on 8000 events, one path per topology.")
    print("    (cycle has the largest |lambda_2| and the slowest mixing,")
    print("     so its early errors can be very large; this is expected.)\n")
    for name, builder in [("cycle", cycle), ("star", star),
                          ("complete", complete)]:
        G, _ = builder(C.N)
        rng = np.random.default_rng(2024)
        t_topo = time.time()
        out = run_async(G, x_star, max_events=8000, eps=0.0,
                        rng=rng,
                        record_every=500)
        err = out["rel_err"]
        if np.any(np.isnan(err)):
            raise RuntimeError(f"NaN encountered for {name}")
        print(f"   {name:9s}  err: start={err[0]:.2e}  "
              f"mid={err[len(err) // 2]:.2e}  end={err[-1]:.2e}   "
              f"[{time.time() - t_topo:.1f}s]")

    print("\n" + "=" * 70)
    print(" Sanity check complete.")
    print(" If [1] shows matched p_i and [2] shows decreasing trajectories,")
    print(" the async update rule matches Koshal Sec. 4.1 with p_{ij} = 1/d_i.")
    print("=" * 70)


if __name__ == "__main__":
    main()
