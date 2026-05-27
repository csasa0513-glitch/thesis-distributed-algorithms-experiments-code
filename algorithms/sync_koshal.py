"""
Algorithm 1 from Koshal, Nedic, Shanbhag (2016):
Synchronous projected-gradient scheme for aggregative games.

Two modes are supported through the `known_aggregate` flag.

* Distributed mode  (`known_aggregate=False`, default).
  Per round k every player simultaneously performs THREE STEPS in the
  exact order of Koshal eqs. (9), (10), (11), p. 685:
      Step 1 (mixing, eq. 9):    v_hat_i^k = sum_j W_ij v_j^k
      Step 2 (gradient, eq. 10): x_i^{k+1} = Pi_{X_i}(
                                     x_i^k - tau_k F_i(x_i^k, N v_hat_i^k) )
      Step 3 (innovation, 11):   v_i^{k+1} = v_hat_i^k + (s_i^{k+1} - s_i^k)
  The gradient in Step 2 uses the MIXED v_hat_i^k (after applying W), not
  the pre-mix v_i^k. This is the key ordering of Koshal's Algorithm 1.
  Under the Koshal-style doubly stochastic W (see graphs/topologies.py)
  the invariant
      sum_i v_i^k = sum_i s_i^k = bar s^k
  holds for all k, so v_i^k -> bar s^k / N and N v_i^k -> bar s^k.
  The factor N is why v_i is multiplied by N before being passed to F_i.
  Initial condition: v_i^0 = s_i^0.

* Centralized mode  (`known_aggregate=True`).
  Every player is given direct access to the true aggregate
      bar s^k = sum_j s_j^k
  and the gradient step uses it instead of N v_hat_i^k:
      x_i^{k+1} = Pi_{X_i}( x_i^k - tau_k F_i(x_i^k, bar s^k) )
  This is the static baseline of Koshal et al. (2016), Tables 3-4.

Stepsize:   tau_k = 1 / (k + 1)        (Koshal et al. 2016, Sec. 6.2)
            In the paper k starts at 1, so tau_k = 1/k. Here k is 0-indexed
            by Python convention, hence 1/(k+1).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import config as C
from games.nash_cournot import F_i, project_Xi, aggregate


def run(W: NDArray,
        x_star: NDArray,
        max_iter: int = C.MAX_ITER_SYNC,
        eps: float = C.EPS_TOL,
        record_every: int = 1,
        x0: NDArray | None = None,
        known_aggregate: bool = False,
        rng: np.random.Generator | None = None) -> dict:
    """
    Run the synchronous algorithm.

    Parameters
    ----------
    W : (N, N)
        Doubly stochastic weight matrix (ignored when
        `known_aggregate=True`).
    x_star : (N*2S,)
        Nash equilibrium used only for error reporting.
    max_iter : int
        Maximum number of rounds.
    eps : float
        Early-stopping tolerance on the relative error.
    record_every : int
        Record the error every `record_every` rounds.
    x0 : (N, 2S) or None
        Initial decision. If None, a random initial point is sampled
        uniformly from [0, C.INIT_SCALE]^{2S} per player and then
        projected onto each player's feasible set X_i.
    known_aggregate : bool
        If False, players estimate the aggregate via consensus on v.
        If True, players use the exact aggregate bar s^k at every round.
    rng : np.random.Generator or None
        Random number generator. Required when x0 is None.
    """
    N, L = C.N, C.L

    # State
    if x0 is None:
        if rng is None:
            raise ValueError("rng is required when x0 is None.")
        raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
        x = np.empty_like(raw)
        for i in range(N):
            x[i] = project_Xi(i, raw[i])
    else:
        x = np.asarray(x0, dtype=float).reshape(N, 2 * L).copy()

    v = x[:, L:].copy()                # local aggregate estimates v_i^0 = s_i^0
    s_prev = x[:, L:].copy()

    rel_err, cons_err, iters = [], [], []

    for k in range(max_iter):
        # Koshal stepsize tau_k = 1 / (k+1)  (= 1/k in 1-indexed paper)
        tau = 1.0 / (k + 1)

        # ----- Step 1: MIXING (Koshal eq. 9) -------------------------------
        #     v_hat_i^k = sum_j W_ij v_j^k
        v_hat = W @ v

        # ----- Step 2: GRADIENT using the MIXED estimate (Koshal eq. 10) ---
        #     x_i^{k+1} = Pi_{X_i}( x_i^k - tau_k F_i(x_i^k, N v_hat_i^k) )
        if known_aggregate:
            bar_s = x[:, L:].sum(axis=0)     # exact aggregate bar s^k
        x_new = np.empty_like(x)
        for i in range(N):
            if known_aggregate:
                grad = F_i(i, x[i], bar_s)
            else:
                grad = F_i(i, x[i], N * v_hat[i])     # uses v_hat, not v
            x_new[i] = project_Xi(i, x[i] - tau * grad)

        # ----- Step 3: INNOVATION (Koshal eq. 11) --------------------------
        #     v_i^{k+1} = v_hat_i^k + (s_i^{k+1} - s_i^k)
        s_new = x_new[:, L:]
        v_new = v_hat + (s_new - s_prev)

        # Book-keeping.
        # Sample error follows Koshal et al. (2016), eq. (65):
        #     e_k = max_{i,s} { |g_is^k - g_is*|, |s_is^k - s_is*| }
        #         / max_{i,s} { |g_is*|, |s_is*| }
        # which for the stacked vector x = (g_1,..,g_N, s_1,..,s_N) is just
        # the relative max-norm error.
        if k % record_every == 0:
            denom = max(np.max(np.abs(x_star)), 1e-12)
            e = float(np.max(np.abs(x_new.ravel() - x_star)) / denom)
            d = np.mean(np.linalg.norm(N * v_new - aggregate(x_new.ravel()), axis=1))
            rel_err.append(e)
            cons_err.append(d)
            iters.append(k)
            if e < eps:
                x, v, s_prev = x_new, v_new, s_new
                break

        x, v, s_prev = x_new, v_new, s_new

    return {
        "x_final":  x.ravel(),
        "iters":    np.asarray(iters),
        "rel_err":  np.asarray(rel_err),
        "cons_err": np.asarray(cons_err),
        "k_eps":    iters[-1] if rel_err and rel_err[-1] < eps else None,
    }
