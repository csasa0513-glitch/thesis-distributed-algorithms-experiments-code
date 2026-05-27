"""
Generate the §6.3.2 async convergence figure from cached trajectories.

Loads `results/async_p01_trajectories.npz` (produced by
`run_async_p01.py`) and writes `results/ws_async_p01.png`. Use this
when the async simulation has already been run and only the figure
needs to be (re)generated -- saves ~50-100 minutes of simulation time.

The figure is cropped to k >= K_START to drop the early-stage
transient of the 1/k-type diminishing stepsize (the first projected-
gradient step overshoots because tau_0 is large and F_i is large at
the random initial point).

Usage:
    python plot_async_p01.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import config as C
from analysis.plots import mean_ci

K_START: int = 100   # crop the early transient


def main() -> None:
    npz_path = C.RESULTS / "async_p01_trajectories.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run `python run_async_p01.py` first."
        )

    data = np.load(npz_path)
    events_per_rep = data["events_per_rep"]      # (R, T)
    rel_err_per_rep = data["rel_err_per_rep"]    # (R, T)
    R, T = rel_err_per_rep.shape
    print(f"Loaded {R} sample paths of length {T} from {npz_path.name}")

    trajectories = [rel_err_per_rep[i] for i in range(R)]
    mean, lo, hi = mean_ci(trajectories, confidence=0.90)
    events_axis = events_per_rep[0]

    # Crop the early transient: keep only k >= K_START
    mask = events_axis >= K_START
    print(f"Cropping to k >= {K_START}: {mask.sum()}/{T} points kept")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(events_axis[mask], mean[mask], color="C1", linewidth=1.4,
            label=f"mean over $R = {R}$ graphs")
    ax.fill_between(events_axis[mask], lo[mask], hi[mask],
                    color="C1", alpha=0.25, label="90% CI")
    ax.set_yscale("log")
    ax.set_xlabel(f"gossip event $k$, $k \\geq {K_START}$")
    ax.set_ylabel("relative error $e(k)$")
    ax.set_title("Asynchronous gossip on $WS(50,\\,6,\\,0.1)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out = C.RESULTS / "ws_async_p01.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
