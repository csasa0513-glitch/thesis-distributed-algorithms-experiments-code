"""
Plot helpers.

Each function writes a PNG to config.RESULTS.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def convergence_plot(
    curves: dict[str, tuple[NDArray, NDArray, NDArray, NDArray]],
    xlabel: str,
    ylabel: str,
    title: str,
    out: Path,
    logy: bool = True,
) -> None:
    """
    Plot mean curves with confidence bands.

    curves: dict label -> (x, mean, low, high), all 1-D arrays.
    """
    assert len(curves) > 0, "convergence_plot needs at least one curve."

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for label, (x, mean, low, high) in curves.items():
        ax.plot(x, mean, label=label, linewidth=1.5)
        if logy:
            # Keep the lower band positive for log-scale plots.
            low_band = np.maximum(low, 1e-15)
        else:
            low_band = low
        ax.fill_between(x, low_band, high, alpha=0.2)

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Normal quantiles used in the thesis.
# The default is 90%, matching the tables and figures in Chapter 6.
Z_TABLE = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def mean_ci(
    runs: list[NDArray],
    confidence: float = 0.90,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Return the pointwise mean and confidence band.

    The band is mean +/- z * SEM at the requested confidence level.
    The default is 90%, matching Chapter 6.

    Shorter runs are right-padded with their last value.
    If there is only one run, the band collapses to the mean.
    """
    assert len(runs) >= 1, "mean_ci requires at least one run."
    assert all(len(r) > 0 for r in runs), \
        "mean_ci requires every run to be non-empty."

    if confidence not in Z_TABLE:
        raise ValueError(
            f"confidence must be one of {sorted(Z_TABLE)}; got {confidence}"
        )

    z = Z_TABLE[confidence]
    R = len(runs)

    T = max(len(r) for r in runs)
    pad = np.empty((R, T), dtype=float)
    for i, r in enumerate(runs):
        pad[i, :len(r)] = r
        pad[i, len(r):] = r[-1]   # carry forward last value

    mean = pad.mean(axis=0)

    if R < 2:
        return mean, mean.copy(), mean.copy()

    sem = pad.std(axis=0, ddof=1) / np.sqrt(R)
    half = z * sem
    return mean, mean - half, mean + half