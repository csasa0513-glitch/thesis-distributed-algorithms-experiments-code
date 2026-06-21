"""
Watts-Strogatz illustration: regular ring (p=0), small-world (p in (0,1)),
random (p=1). Used as a textbook-style figure in Chapter 2.

This replaces the earlier K=4 illustration with K=6, matching the
experimental setup used in Chapter 6.

Usage:
    python plot_ws_illustration.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
N: int = 20       # number of nodes (matches the original illustration)
K: int = 6        # base degree of the ring lattice (matches Chapter 6 setup)
SEED: int = 42    # the seed you remember

P_VALUES = [0.0, 0.10, 1.0]
TITLES = [
    r"$p = 0$ (regular)",
    r"$p \in (0, 1)$ (small-world)",
    r"$p = 1$ (random)",
]


def _draw(ax, G: nx.Graph, title: str) -> None:
    pos = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="0.35", width=0.9, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color="#7FB3E6", node_size=160,
                           edgecolors="#3A7AB8", linewidths=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()
    ax.set_aspect("equal")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
    for ax, p, title in zip(axes, P_VALUES, TITLES):
        G = nx.watts_strogatz_graph(n=N, k=K, p=p, seed=SEED)
        _draw(ax, G, title)
    fig.tight_layout()
    out_path = "ws_illustration_K6.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white")
    print(f"Saved figure to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
