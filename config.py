"""
Central configuration for Chapter 6 experiments.

Cournot coefficients are sampled independently for each N and kept
fixed across all runs for that N. This matches the experimental setup
in Chapter 6, where N=20 and N=50 are treated as separate instances.

Each call to resample(N_new) redraws all random arrays for the new N:
M_INTERCEPT, A_COEF, and B_COEF.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

# Game instance 
SEED: int = 20260327
N: int = 50
L: int = 10

# Initial draw for the default instance N=50 
_SS_INIT = np.random.SeedSequence([SEED, N])
_RNG_INIT = np.random.default_rng(_SS_INIT)
M_INTERCEPT = _RNG_INIT.uniform(90.0, 100.0, size=L)
A_COEF = _RNG_INIT.uniform(2.0, 12.0, size=(N, L))
B_COEF = _RNG_INIT.uniform(2.0, 3.0, size=(N, L))

CAP_CONST: float = 500.0
CAP = np.full((N, L), CAP_CONST)


def resample(N_new: int) -> None:
    """Redraw all Cournot coefficients for a new N."""
    global N, A_COEF, B_COEF, CAP, M_INTERCEPT
    N = N_new
    ss = np.random.SeedSequence([SEED, N_new])
    rng = np.random.default_rng(ss)
    M_INTERCEPT = rng.uniform(90.0, 100.0, size=L)
    A_COEF      = rng.uniform(2.0,  12.0, size=(N, L))
    B_COEF      = rng.uniform(2.0,   3.0, size=(N, L))
    CAP         = np.full((N, L), CAP_CONST)


# Algorithm stepsize constant
STEP_DIM_NUM_ASYNC: float = 9.0
STEP_DIM_NUM: float = STEP_DIM_NUM_ASYNC

MAX_ITER_SYNC: int = 10_000
SYNC_CHECKPOINTS: tuple[int, ...] = (5_000, 10_000)

# Initial decision: uniform on [0, INIT_SCALE]^{2L} then projected.
INIT_SCALE: float = 10.0

# Convergence threshold
EPS_TOL: float = 1e-3

R: int = 50

# Watts-Strogatz sweep parameters.
WS_K: int = 6
WS_P: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)


# Paths
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
