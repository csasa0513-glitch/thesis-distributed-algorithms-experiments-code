"""
Central configuration for all experiments of Chapter 6.

Cournot coefficients are drawn per (SEED + N) so N=20 and N=50 are
independent instances. Each call to resample(N_new) re-draws ALL
random arrays: D_INTERCEPT, A_COEF, B_COEF.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

# Game instance (Koshal 2016, Sec. 6.1)
SEED: int = 20260327
N: int = 50
L: int = 10
RNG = np.random.default_rng(SEED)

D_INTERCEPT = RNG.uniform(90.0, 100.0, size=L)
A_COEF = RNG.uniform(2.0, 12.0, size=(N, L))
B_COEF = RNG.uniform(2.0,  3.0, size=(N, L))

CAP_CONST: float = 500.0
CAP = np.full((N, L), CAP_CONST)


def resample(N_new: int) -> None:
    """Redraw ALL Cournot coefficients (D, A, B) from the MASTER seed
    (not from SEED + N_new). Under this scheme the N=20 game is a
    strict subset of the N=50 game (first 20 players share identical
    a, b coefficients), so N=20 vs N=50 can be compared as a true
    scaling experiment -- which matches the intent of Koshal et al.'s
    Tables 3-8 where larger N is reported as harder."""
    global N, A_COEF, B_COEF, CAP, D_INTERCEPT
    N = N_new
    rng = np.random.default_rng(SEED)
    D_INTERCEPT = rng.uniform(90.0, 100.0, size=L)
    A_COEF      = rng.uniform(2.0,  12.0, size=(N, L))
    B_COEF      = rng.uniform(2.0,   3.0, size=(N, L))
    CAP         = np.full((N, L), CAP_CONST)


# Algorithm parameters
STEP_DIM_NUM: float = 9.0
MAX_ITER_SYNC:  int = 10_000
MAX_GOSSIP:     int = 200_000
EPS_TOL:        float = 1e-3

# Synchronous experiment knobs
R_SYNC: int = 50
SYNC_CHECKPOINTS: tuple[int, ...] = (5_000, 10_000)
INIT_SCALE: float = 10.0

# Replication
R: int = 50

# Watts-Strogatz
WS_K: int = 6
WS_P = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]

# Paths
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)