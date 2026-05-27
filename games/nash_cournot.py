"""
Nash-Cournot aggregative game used in Chapter 6.

Feasible set X_i (matches Koshal 2016, eq. (64), Sec. 6.1):
    0 <= g_il <= cap_il,         l = 1, ..., L
    s_il      >= 0,              l = 1, ..., L
    sum_l g_il == sum_l s_il.    (equality coupling)

Inverse demand:  p_l(u_l) = d_l - u_l              (Koshal 2016, slope = 1)
Cost:            c_il(g)  = a_il * g + b_il * g^2   (no 1/2 coefficient)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import config as C


def F_i(i: int, x_i: NDArray, u: NDArray) -> NDArray:
    """Local gradient F_i(x_i, u) for player i."""
    g_i = x_i[: C.L]
    s_i = x_i[C.L :]
    dF_g = C.A_COEF[i] + 2.0 * C.B_COEF[i] * g_i
    dF_s = u - C.D_INTERCEPT + s_i
    return np.concatenate([dF_g, dF_s])


def F_stack(x: NDArray, u: NDArray) -> NDArray:
    """Stacked gradient F(x, u) = [F_1; ...; F_N], shape (N*2L,)."""
    out = np.empty(C.N * 2 * C.L)
    for i in range(C.N):
        out[i * 2 * C.L : (i + 1) * 2 * C.L] = F_i(i, x[i], u)
    return out


def project_Xi(i: int, y: NDArray, tol: float = 1e-12) -> NDArray:
    """
    Exact Euclidean projection  Pi_{X_i}(y)  of y in R^{2L} onto X_i.

    X_i (Koshal 2016, eq. (64)):
        0 <= g_l <= cap_l,  s_l >= 0,  sum_l g_l == sum_l s_l.

    Closed form via Lagrangian:
        g_l(mu) = clip(y_g_l - mu, 0, cap_l)
        s_l(mu) = max(y_s_l + mu, 0)
    Choose scalar mu s.t. F(mu) = sum(g) - sum(s) = 0 (bisection).
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
    """True aggregate sales bar_s in R^L from a stacked x in R^{N*2L}."""
    X = x.reshape(C.N, 2 * C.L)
    return X[:, C.L :].sum(axis=0)


def solve_NE(max_iter: int = 50_000, tol: float = 1e-9) -> NDArray:
    """Centralized projected-gradient NE solver. Used only for reference x*."""
    L_lip = float(2.0 * np.max(C.B_COEF) + 2.0 * C.N)
    tau = 1.0 / max(L_lip, 1.0)
    x = np.zeros(C.N * 2 * C.L)
    for _ in range(max_iter):
        u = aggregate(x)
        g = F_stack(x.reshape(C.N, 2 * C.L), u)
        x_new = project_full(x - tau * g)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


if __name__ == "__main__":
    x_star = solve_NE()
    np.savez(C.RESULTS / "NE.npz", x_star=x_star)
    print(f"NE computed, ||x*|| = {np.linalg.norm(x_star):.4f}")
