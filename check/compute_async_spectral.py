"""
Compute spectral indicators for async gossip on the 4 baseline graphs
(cycle, wheel, grid, complete) at N=20, for comparison with Koshal et al.
2016, Table 9 (page 702).

Koshal's gossip model (Sec. 4.1, p. 689):
  - Each agent has a Poisson clock with rate 1.
  - At a tick, one agent wakes up (uniformly, prob 1/N).
  - Agent i contacts neighbor j with conditional probability p_{ij}.
  - We assume uniform contact: p_{ij} = 1 / d_i for j in N_i, 0 otherwise.

Therefore, the probability that edge {i, j} is activated in one global tick is
    P({i,j}) = (1/N) * ( p_{ij} + p_{ji} ) = (1/N) * ( 1/d_i + 1/d_j ).
Sum over all edges = 1.

Expected weight matrix:
    E[W(k)] = sum over edges  P_e * W^{(e)}
            = I  -  (1/2) * L_w,
where L_w is the weighted Laplacian
    L_w = sum_e P_e * (e_i - e_j)(e_i - e_j)^T.

The spectral indicator (Koshal Lemma 8 / your Lemma 5.10):
    lambda    := second-largest eigenvalue of E[W(k)]
    sqrt(lam) := sqrt(lambda) =  per-step contraction factor in expectation.

Output: results/async_spectral_baselines.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from graphs.generators import REGULAR


N_TEST: int = 20


def edge_activation_probs(G, N: int) -> dict[tuple[int, int], float]:
    """P({i,j} activated in one global tick) = (1/N) * (1/d_i + 1/d_j)."""
    deg = dict(G.degree())
    probs: dict[tuple[int, int], float] = {}
    for u, v in G.edges():
        i, j = (u, v) if u < v else (v, u)
        probs[(i, j)] = (1.0 / N) * (1.0 / deg[i] + 1.0 / deg[j])
    return probs


def expected_W(N: int, edge_probs: dict[tuple[int, int], float]) -> np.ndarray:
    """E[W(k)] = I - (1/2) * L_weighted, with L_w = sum_e P_e (e_i-e_j)(e_i-e_j)^T."""
    L_w = np.zeros((N, N))
    for (i, j), p in edge_probs.items():
        L_w[i, i] += p
        L_w[j, j] += p
        L_w[i, j] -= p
        L_w[j, i] -= p
    return np.eye(N) - L_w / 2.0


def main() -> None:
    N = N_TEST
    C.resample(N)

    print("=" * 78)
    print(f"  Async spectral indicators on 4 baseline graphs (N = {N})")
    print("  Comparison with Koshal et al. 2016, Table 9 (page 702)")
    print("=" * 78)
    header = (
        f"{'Graph':10s}  {'|E|':>3s}  {'max_d':>5s}  "
        f"{'P_edge sum':>10s}  "
        f"{'lambda2':>11s}  {'sqrt(lam)':>9s}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for name, gen in REGULAR.items():
        G, _W_sync = gen(N)

        # 1. edge activation probs and expected weight matrix
        edge_probs = edge_activation_probs(G, N)
        P_sum = sum(edge_probs.values())   # should be ~ 1
        EW = expected_W(N, edge_probs)

        # 2. spectral indicators
        eigs = np.linalg.eigvalsh(EW)      # ascending
        lam2 = float(eigs[-2])             # second-largest eigenvalue
        sqrt_lam = float(np.sqrt(max(lam2, 0.0)))
        max_d = max(dict(G.degree()).values())

        rows.append({
            "N": N,
            "graph": name,
            "num_edges": G.number_of_edges(),
            "max_degree": max_d,
            "P_edge_sum": P_sum,
            "lambda2_EW": lam2,
            "sqrt_lambda2": sqrt_lam,
        })
        print(
            f"{name:10s}  {G.number_of_edges():>3d}  {max_d:>5d}  "
            f"{P_sum:>10.6f}  "
            f"{lam2:>11.6e}  {sqrt_lam:>9.4f}"
        )

    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "results" / "async_spectral_baselines.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    print("\n" + "=" * 78)
    print("  Koshal et al. 2016, Table 9 (N = 20) -- for direct comparison")
    print("=" * 78)
    print(f"  {'Network':10s}  {'lambda':>11s}  {'iters@1e-3':>10s}")
    print(f"  {'Cycle':10s}  {'9.994e-01':>11s}  {'48,818':>10s}")
    print(f"  {'Wheel':10s}  {'1.622e-01':>11s}  {'8,324':>10s}")
    print(f"  {'Grid':10s}  {'3.151e-01':>11s}  {'17,950':>10s}")
    print(f"  {'Complete':10s}  {'1.089e-08':>11s}  {'5,842':>10s}")
    print("\nNotes:")
    print(" - This 'wheel' graph is structurally a star (one hub + N-1 leaves).")
    print(" - 'lambda' column in Koshal Table 9 is *the* second-largest eigenvalue,")
    print("   not the square root. (His text mentions sqrt(lambda) but the table")
    print("   reports lambda itself.)")
    print(" - If our 'lambda2_EW' column matches Koshal's 'lambda' column, the model")
    print("   and convention agree.")


if __name__ == "__main__":
    main()
