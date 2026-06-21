"""
Synchronous distributed algorithm for the Nash-Cournot game.

If known_aggregate=False, the method follows the three-step distributed
scheme: mixing, projected update, and innovation.
If known_aggregate=True, the exact aggregate is used as a centralized
benchmark.
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
    Run the synchronous Nash-Cournot implementation.

    Parameters
    ----------
    W : (N, N)
        Doubly stochastic weight matrix; ignored in centralized mode.
    x_star : (N*2L,)
        Reference equilibrium used for error reporting.
    max_iter : int
        Maximum number of iterations.
    eps : float
        Stopping tolerance for the relative error.
    record_every : int
        Record diagnostics every `record_every` iterations.
    x0 : (N, 2L) or None
        Initial decision vector.
    known_aggregate : bool
        Use the exact aggregate if True.
    rng : np.random.Generator or None
        Used only when x0 is None.
    """
    N, L = C.N, C.L

    # --- Input sanity checks --------------------------------------------
    assert record_every >= 1, \
        f"record_every must be a positive integer; got {record_every}."
    assert x_star.size == N * 2 * L, \
        f"x_star must have N*2L = {N * 2 * L} entries; got {x_star.size}."
    if not known_aggregate:

        assert W.shape == (N, N), \
            f"W must have shape (N, N) = ({N}, {N}); got {W.shape}."
        assert np.all(W >= -1e-12), \
            "W must have non-negative entries."
        assert np.allclose(W.sum(axis=1), 1.0, atol=1e-9), \
            "W must be row-stochastic."
        assert np.allclose(W.sum(axis=0), 1.0, atol=1e-9), \
            "W must be column-stochastic."

    if x0 is None:
        if rng is None:
            raise ValueError("rng is required when x0 is None.")
        raw = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))
    else:
        raw = np.asarray(x0, dtype=float).reshape(N, 2 * L)
    x = np.empty_like(raw)
    for i in range(N):
        x[i] = project_Xi(i, raw[i])

    v = x[:, L:].copy()                
    s_prev = x[:, L:].copy()

    rel_err, cons_err, iters = [], [], []

    for k in range(max_iter):
        alpha_k = 1.0 / (k + 1)

        # Step 1: 
        if known_aggregate:
            bar_s = x[:, L:].sum(axis=0)     
        else:
            v_hat = W @ v

        # Step 2: 
        x_new = np.empty_like(x)
        for i in range(N):
            if known_aggregate:
                grad = F_i(i, x[i], bar_s)
            else:
                grad = F_i(i, x[i], N * v_hat[i])
            x_new[i] = project_Xi(i, x[i] - alpha_k * grad)

        # Step 3: 
        s_new = x_new[:, L:]
        if not known_aggregate:
            v_new = v_hat + (s_new - s_prev)

        iters_done = k + 1
        if iters_done % record_every == 0:
            denom = max(np.max(np.abs(x_star)), 1e-12)
            e = float(np.max(np.abs(x_new.ravel() - x_star)) / denom)
            if known_aggregate:
                d = 0.0
            else:
                d = np.mean(np.linalg.norm(
                    N * v_new - aggregate(x_new.ravel()), axis=1))
            rel_err.append(e)
            cons_err.append(d)
            iters.append(iters_done)
            if e < eps:
                x = x_new
                if not known_aggregate:
                    v, s_prev = v_new, s_new
                break

        x = x_new
        if not known_aggregate:
            v, s_prev = v_new, s_new

    return {
        "x_final":  x.ravel(),
        "iters":    np.asarray(iters),
        "rel_err":  np.asarray(rel_err),
        "cons_err": np.asarray(cons_err),
        "k_eps":    iters[-1] if rel_err and rel_err[-1] < eps else None,
    }
