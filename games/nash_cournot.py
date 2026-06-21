"""
Nash-Cournot game used in Chapter 6.

The numerical experiments use the model
    p_l(\bar s_l) = m_l - \bar s_l,
    c_{i l}(g_{i l}) = a_{i l} g_{i l} + b_{i l} g_{i l}^2.

In the experiments, the feasible set is enforced with the equality
    sum_l s_{i l} = sum_l g_{i l}.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import config as C


def F_i(i: int, x_i: NDArray, u: NDArray) -> NDArray:
    """Return the local mapping F_i(x_i, u) for player i."""
    g_i = x_i[: C.L]
    s_i = x_i[C.L :]
    dF_g = C.A_COEF[i] + 2.0 * C.B_COEF[i] * g_i
    dF_s = u - C.M_INTERCEPT + s_i
    return np.concatenate([dF_g, dF_s])


def F_stack(x: NDArray, u: NDArray) -> NDArray:
    """Return the stacked mapping F(x, u) = [F_1; ...; F_N]."""
    out = np.empty(C.N * 2 * C.L)
    for i in range(C.N):
        out[i * 2 * C.L : (i + 1) * 2 * C.L] = F_i(i, x[i], u)
    return out


def project_Xi(i: int, y: NDArray, tol: float = 1e-12) -> NDArray:
    """
    Return the Euclidean projection of y onto X_i.

    In the numerical experiments, X_i is defined by
        0 <= g_l <= cap_l,
        s_l >= 0,
        sum_l g_l = sum_l s_l.

    The projection has the closed form
        g_l(mu) = clip(y_g_l - mu, 0, cap_l),
        s_l(mu) = max(y_s_l + mu, 0),
    where mu is chosen so that sum(g) - sum(s) = 0.
    """
    y_g = y[: C.L]
    y_s = y[C.L :]
    cap = C.CAP[i]

    def F(mu: float) -> float:
        g = np.minimum(np.maximum(y_g - mu, 0.0), cap)
        s = np.maximum(y_s + mu, 0.0)
        return float(g.sum() - s.sum())

    margin = (float(np.max(cap)) + float(np.max(np.abs(y_g)))
              + float(np.max(np.abs(y_s))) + 1.0)
    mu_lo = -margin
    mu_hi =  margin
    while F(mu_lo) <= 0.0:
        mu_lo *= 2.0
    while F(mu_hi) >= 0.0:
        mu_hi *= 2.0

    for _ in range(100):
        mu = 0.5 * (mu_lo + mu_hi)
        f = F(mu)
        if abs(f) < tol or (mu_hi - mu_lo) < tol:
            break
        if f > 0.0:
            mu_lo = mu
        else:
            mu_hi = mu

    g = np.minimum(np.maximum(y_g - mu, 0.0), cap)
    s = np.maximum(y_s + mu, 0.0)
    return np.concatenate([g, s])


def project_full(y: NDArray) -> NDArray:
    """Project a stacked vector y in R^{N*2L} player-wise."""
    out = np.empty_like(y)
    for i in range(C.N):
        out[i * 2 * C.L : (i + 1) * 2 * C.L] = project_Xi(
            i, y[i * 2 * C.L : (i + 1) * 2 * C.L]
        )
    return out


def aggregate(x: NDArray) -> NDArray:
    """Return the aggregate sales vector \bar s in R^L."""
    X = x.reshape(C.N, 2 * C.L)
    return X[:, C.L :].sum(axis=0)


def solve_NE(max_iter: int = 50_000, tol: float = 1e-9) -> NDArray:
    """Solve the Nash equilibrium by centralized projected gradient descent."""
    L_lip = float(2.0 * np.max(C.B_COEF) + 2.0 * C.N)
    alpha = 1.0 / max(L_lip, 1.0)
    x = np.zeros(C.N * 2 * C.L)
    for _ in range(max_iter):
        u = aggregate(x)
        g = F_stack(x.reshape(C.N, 2 * C.L), u)
        x_new = project_full(x - alpha * g)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


