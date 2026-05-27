"""
Plot helpers. Every function writes a PNG into config.RESULTS.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def convergence_plot(curves: dict, xlabel: str, ylabel: str, title: str,
                     out: Path, logy: bool = True) -> None:
    """
    curves  :  dict label -> (x, mean, low, high)
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for label, (x, mean, low, high) in curves.items():
        ax.plot(x, mean, label=label, linewidth=1.5)
        ax.fill_between(x, low, high, alpha=0.2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# Normal-quantile lookup for the levels used in the thesis / Koshal 2016.
# Koshal reports the *width* of the 90% CI in Tables 2, 4, 7, 8; the plots
# here therefore default to 90% so that the shaded band in every figure
# matches the width reported in every table.
_Z_TABLE = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def mean_ci(runs: list[np.ndarray], confidence: float = 0.90) -> tuple:
    """
    Given a list of 1-D arrays (possibly of different lengths) compute the
    elementwise mean and a symmetric confidence band

        mean +/- z_{alpha/2} * sigma / sqrt(R)      (half-width = z * SEM)

    at the requested ``confidence`` level (default 90%, matching Koshal
    et al. 2016). The returned tuple ``(mean, low, high)`` is intended for
    ``matplotlib.fill_between``; the full width of the band is ``high-low``
    (i.e. 2 * z * SEM), which is exactly the quantity Koshal tabulates.

    Shorter runs (early termination) are right-padded with their last value.
    """
    if confidence not in _Z_TABLE:
        raise ValueError(f"confidence must be one of {sorted(_Z_TABLE)}; "
                         f"got {confidence}")
    z = _Z_TABLE[confidence]

    T = max(len(r) for r in runs)
    pad = np.full((len(runs), T), np.nan)
    for i, r in enumerate(runs):
        pad[i, : len(r)] = r
        pad[i, len(r):] = r[-1]          # carry forward last value
    mean = np.nanmean(pad, axis=0)
    sem  = np.nanstd(pad, axis=0, ddof=1) / np.sqrt(pad.shape[0])
    half = z * sem
    return mean, mean - half, mean + half
