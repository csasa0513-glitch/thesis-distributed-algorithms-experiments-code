"""
Re-draw the §6.4 sensitivity figures from the existing trajectory
.npz files, without re-running the sync / async sweeps.

Useful when only the cosmetics of the plot need to change (title,
axes, line width, legend) -- the underlying mean / CI data is
recomputed from the same trajectories that produced the original
PNG, so this is a strictly cosmetic re-render.

Inputs (in `results/`, written by the original sweep scripts):
    sync_sensitivity_trajectories.npz    (iters_per_rep, rel_err_per_rep,
                                          param_per_rep, ...)
    async_sensitivity_trajectories.npz   (events_per_rep, rel_err_per_rep,
                                          param_per_rep, ...)

Outputs (overwrites the previous PNGs):
    sync_sensitivity.png
    async_sensitivity.png

Usage:
    python replot_sensitivity.py
"""
from __future__ import annotations

import numpy as np
import matplotlib

# Use the non-GUI Agg backend before importing pyplot. This script only
# writes PNGs to disk, never opens a window, so Agg avoids the Windows
# "Unable to register TclNotifier window class" error that the default
# Tk backend can raise depending on the local Tcl/Tk setup.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from analysis.plots import Z_TABLE


# Same subset as the sweep scripts -- kept here so this replot is
# self-contained and does not import from run_*_sensitivity.py.
PLOT_P_VALUES: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0)


def _replot(npz_path, png_path, axis_key, x_label, suptitle):
    """Re-render one sensitivity figure from a trajectory .npz.

    axis_key: 'iters_per_rep' (sync) or 'events_per_rep' (async).
    The full trajectory is plotted (no early-stage crop), so the
    9 / Gamma early transient remains visible in the asynchronous
    panel; this is a deliberate choice for the §6.4 sensitivity
    figures to show the full convergence behaviour.
    """
    with np.load(npz_path) as data:
        axis_all = data[axis_key]            # (N_total, T)
        rel_err  = data["rel_err_per_rep"]   # (N_total, T)
        params   = data["param_per_rep"]     # (N_total,)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    Z = Z_TABLE[0.90]
    n_curves = 0
    for pv in PLOT_P_VALUES:
        mask = np.isclose(params, pv)
        if not mask.any():
            print(f"  WARN: p={pv} not found in {npz_path.name}; skipping",
                  flush=True)
            continue
        axis = axis_all[mask][0]              # all reps share the same axis
        errs = rel_err[mask]                  # (R, T)
        m  = errs.mean(axis=0)
        sd = errs.std(axis=0, ddof=1)
        sem = sd / np.sqrt(len(errs))
        ax.plot(axis, m, label=f"$p={pv:g}$", linewidth=1.3)
        ax.fill_between(axis, m - Z * sem, m + Z * sem, alpha=0.2)
        n_curves += 1
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("relative error $e(k)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved {png_path}  ({n_curves} curves)", flush=True)


def main() -> None:
    _replot(
        C.RESULTS / "sync_sensitivity_trajectories.npz",
        C.RESULTS / "sync_sensitivity.png",
        axis_key="iters_per_rep",
        x_label="iteration $k$",
        suptitle=("Sync sensitivity to rewiring probability $p$, "
                  "$WS(50,\\,6,\\,p)$"),
    )
    _replot(
        C.RESULTS / "async_sensitivity_trajectories.npz",
        C.RESULTS / "async_sensitivity.png",
        axis_key="events_per_rep",
        x_label="gossip event $k$",
        suptitle=("Async sensitivity to rewiring probability $p$, "
                  "$WS(50,\\,6,\\,p)$"),
    )


if __name__ == "__main__":
    main()
