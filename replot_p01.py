"""
Re-draw the §6.3.2 figures (Figure 3a sync, Figure 3b async) from the
existing trajectory .npz files, without re-running the p01 sweeps.

This script plots the full k axis (from k = 0), matching the thesis figures.

Inputs (in `results/`, written by run_*_p01.py):
    sync_p01_trajectories.npz     (iters_per_rep, rel_err_per_rep, ...)
    async_p01_trajectories.npz    (events_per_rep, rel_err_per_rep, ...)

Outputs (overwrites the previous PNGs):
    ws_sync_p01.png
    ws_async_p01.png

Usage:
    python replot_p01.py
"""
from __future__ import annotations

import numpy as np
import matplotlib

# Non-GUI Agg backend so the script does not depend on Tk
# (avoids "Unable to register TclNotifier window class" on Windows).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from analysis.plots import mean_ci


def _replot(npz_path, png_path, axis_key, x_label, title, color):
    """Re-render one §6.3.2 figure from a p01 trajectory .npz, full k.

    axis_key: 'iters_per_rep' (sync) or 'events_per_rep' (async).
    """
    with np.load(npz_path) as data:
        axis_all = data[axis_key]              # (R, T)
        rel_err  = data["rel_err_per_rep"]     # (R, T)
    R, T = rel_err.shape
    axis = axis_all[0]                          # all reps share the axis

    trajectories = [rel_err[i] for i in range(R)]
    mean, lo, hi = mean_ci(trajectories, confidence=0.90)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(axis, mean, color=color, linewidth=1.4,
            label=f"mean over $R = {R}$ graphs")
    ax.fill_between(axis, lo, hi, color=color, alpha=0.25,
                    label="90% CI")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("relative error $e(k)$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved {png_path}  (R={R} reps, T={T} points)", flush=True)


def main() -> None:
    _replot(
        C.RESULTS / "sync_p01_trajectories.npz",
        C.RESULTS / "ws_sync_p01.png",
        axis_key="iters_per_rep",
        x_label="iteration $k$",
        title="Synchronous algorithm on $WS(50,\\,6,\\,0.1)$",
        color="C0",
    )
    _replot(
        C.RESULTS / "async_p01_trajectories.npz",
        C.RESULTS / "ws_async_p01.png",
        axis_key="events_per_rep",
        x_label="gossip event $k$",
        title="Asynchronous gossip on $WS(50,\\,6,\\,0.1)$",
        color="C1",
    )


if __name__ == "__main__":
    main()
