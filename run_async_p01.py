"""
Asynchronous gossip algorithm on WS(50, 6, 0.1) for Figure 3a.

Runs only p_rew = 0.1 and records the mean-error trajectory and the
error at k = 10^5 for R = C.R independent realisations.

Outputs:
    async_p01.csv
    async_p01_trajectories.npz
    ws_async_p01.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import config as C
from games.nash_cournot import solve_NE, project_Xi
from graphs.generators import watts_strogatz
from algorithms.async_gossip import run as run_async
from analysis.metrics import summary as struct_summary
from analysis.plots import mean_ci


# --------------------------------------------------------------------------
# Fixed knobs (§6.3 design: one representative p_rew = 0.1)
# --------------------------------------------------------------------------
P_REW: float = 0.1
N_REPS: int = C.R                 # 50
MAX_EVENTS: int = 100_000         # \tilde k = 10^5 (Koshal async horizon)
K_SUMMARY: int = 100_000          # error reported in tab:ws-error
WS_K: int = C.WS_K                # 6


def _fresh_NE(N_val: int = 50) -> np.ndarray:
    """Resample Cournot coefficients and recompute NE -- always fresh,
    no caching, so any change to game parameters or projection takes
    effect immediately."""
    C.resample(N_val)
    return solve_NE()


def _random_x0(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial decision, projected onto each player's X_i.

    Uniform draw on $[0, C.INIT_SCALE]^{2L}$ per player, then
    `project_Xi` for feasibility. This MUST stay byte-identical to the
    same-named helper in `run_sync_p01.py`; otherwise the paired
    comparison breaks even if the seeds are aligned.
    """
    xi = rng.uniform(0.0, C.INIT_SCALE, size=(N, 2 * L))   # xi_i ~ U([0, INIT_SCALE])^{2L}
    for i in range(N):
        xi[i] = project_Xi(i, xi[i])
    return xi


def _rep_plan(master_rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Generate the four seeds for one rep of the §6.3.2 paired experiment.

    Decoupling the randomness sources (graph, x0, gossip stream) by name
    makes the paired comparison auditable: the same `master_rng` in
    `run_sync_p01.py` and `run_async_p01.py` produces the same
    (graph_seed, x0_seed, gossip_seed) for the same rep index r. The
    returned `rep_seed` is also stored in the CSV so a reader can
    replay any single rep without advancing through the master RNG.

    Returns
    -------
    rep_seed : int
        The seed drawn directly from the master RNG; identifies the rep.
    graph_seed : int
        Drives `watts_strogatz` (-> WS realisation).
    x0_seed : int
        Drives `_random_x0` (-> initial point).
    gossip_seed : int
        Drives the asynchronous gossip stream consumed by
        `algorithms.async_gossip.run`.
    """
    rep_seed = int(master_rng.integers(1 << 31))
    rng_plan = np.random.default_rng(rep_seed)
    graph_seed = int(rng_plan.integers(1 << 31))
    x0_seed = int(rng_plan.integers(1 << 31))
    gossip_seed = int(rng_plan.integers(1 << 31))
    return rep_seed, graph_seed, x0_seed, gossip_seed


def main() -> None:
    import time
    print(f"[start] async run at p_rew={P_REW}, R={N_REPS}, max_events={MAX_EVENTS:,}",
          flush=True)
    print("[start] solving Nash equilibrium (fresh, N=50) ...", flush=True)
    x_star = _fresh_NE(N_val=50)
    print(f"[start] NE solved, ||x*|| = {np.linalg.norm(x_star):.4f}",
          flush=True)
    # Master seed shared with run_sync_p01.py so that the (graph, x0)
    # pair at rep r is bit-identical across the two scripts.
    master_rng = np.random.default_rng(C.SEED + 43)

    rows = []
    trajectories: list[np.ndarray] = []
    event_axes: list[np.ndarray] = []
    x0_list: list[np.ndarray] = []   # stored in npz for paired-comparison audit

    t_start = time.time()
    for r in range(N_REPS):
        t_rep = time.time()
        # Four-part rep plan: same call in run_sync_p01.py produces the
        # same (graph_seed, x0_seed, gossip_seed) for the same r.
        rep_seed, graph_seed, x0_seed, gossip_seed = _rep_plan(master_rng)

        G, W = watts_strogatz(C.N, WS_K, P_REW, seed=graph_seed)
        # Defensive: pin the connectivity contract from generators.py;
        # async_gossip.run asserts connectivity internally too, but we
        # add it here so a failure is reported at the call site.
        assert nx.is_connected(G), (
            f"watts_strogatz returned a disconnected graph at "
            f"rep = {r}, graph_seed = {graph_seed}. "
            f"Check graphs/generators.py watts_strogatz."
        )
        # Construct x0 OUTSIDE async_gossip with its own RNG so the
        # initial condition is decoupled from the gossip stream and
        # bit-identical to the x0 used by run_sync_p01.py at the same r.
        x0_rng = np.random.default_rng(x0_seed)
        x0 = _random_x0(C.N, C.L, x0_rng)
        assert x0.shape == (C.N, 2 * C.L), (
            f"x0 shape mismatch at rep = {r}: got {x0.shape}, "
            f"expected ({C.N}, {2 * C.L})."
        )
        assert np.isfinite(x0).all(), (
            f"x0 contains non-finite values at rep = {r}."
        )
        # Gossip stream gets its OWN dedicated RNG. async_gossip.run no
        # longer also has to sample x0, so the two randomness sources
        # cannot accidentally couple.
        gossip_rng = np.random.default_rng(gossip_seed)
        print(f"  rep {r + 1:2d}/{N_REPS} starting ...", flush=True)

        out = run_async(
            G, x_star,
            max_events=MAX_EVENTS,
            eps=0.0,
            x0=x0,
            rng=gossip_rng,
        )

        # Error at the reported horizon K_SUMMARY
        events = out["events"]
        rel_err = out["rel_err"]
        idx = int(np.searchsorted(events, K_SUMMARY, side="right") - 1)
        idx = max(0, min(idx, len(rel_err) - 1))

        rows.append({
            "rep":                r,
            "p_rew":                  P_REW,
            "mean_error_at_K":    float(rel_err[idx]),
            # Per-rep RNG plan (paired-comparison audit columns).
            "rep_seed":           rep_seed,
            "graph_seed":         graph_seed,
            "x0_seed":            x0_seed,
            "gossip_seed":        gossip_seed,
            # lambda2_async and rho_async are now included inside
            # struct_summary, so no manual computation needed here.
            **struct_summary(G, W),
        })

        trajectories.append(rel_err)
        event_axes.append(events)
        x0_list.append(x0.copy())

        dt_rep = time.time() - t_rep
        elapsed_total = time.time() - t_start
        eta = elapsed_total / (r + 1) * (N_REPS - r - 1)
        print(f"  rep {r + 1:2d}/{N_REPS} done"
              f"  err@1e5={rel_err[idx]:.4f}"
              f"  ({dt_rep:.1f}s, ETA {eta/60:.1f} min)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "async_p01.csv", index=False)
    print(f"\nSaved results/async_p01.csv  ({N_REPS} rows)")

    # Save trajectories so we can build the combined sync/async figure
    # later. All event axes are typically identical (record_every fixed),
    # but we store them per-rep for safety. x0_per_rep is included so
    # the paired-comparison claim (sync and async use the same x0 at
    # every rep) can be verified offline:
    #     np.allclose(sync_npz['x0_per_rep'], async_npz['x0_per_rep'])
    np.savez(
        C.RESULTS / "async_p01_trajectories.npz",
        events_per_rep=np.stack(event_axes),
        rel_err_per_rep=np.stack(trajectories),
        x0_per_rep=np.stack(x0_list),
    )
    print(f"Saved results/async_p01_trajectories.npz")

    # --------------------------------------------------------------
    # Figure: mean +/- 90% CI band over the R sample paths
    # Full trajectory from k = 0 (no early-transient crop), matching
    # the thesis figures.
    # --------------------------------------------------------------
    mean, lo, hi = mean_ci(trajectories, confidence=0.90)
    events_axis = event_axes[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(events_axis, mean, color="C1", linewidth=1.4,
            label="mean over $R = 50$ graphs")
    ax.fill_between(events_axis, lo, hi,
                    color="C1", alpha=0.25, label="90% CI")
    ax.set_yscale("log")
    ax.set_xlabel("gossip event $k$")
    ax.set_ylabel("relative error $e(k)$")
    ax.set_title("Asynchronous gossip on $WS(50,\\,6,\\,0.1)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.RESULTS / "ws_async_p01.png", dpi=200)
    plt.close(fig)
    print(f"Saved results/ws_async_p01.png")


if __name__ == "__main__":
    main()
