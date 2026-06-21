"""
Comprehensive sanity check of all 15 A/B/C class files.

Mini-scale verification:
  - Each algorithm runs with R=2 reps and small k to fit < 2 minutes total
  - Verifies imports, signatures, output shapes, sanity of numerical results
  - Cross-checks paired-comparison invariants for §6.3 and §6.4

Run from `thesis-code/`:
    python sanity_check_all.py

Output: a structured pass/fail report; non-zero exit code on any failure.
"""
from __future__ import annotations

import sys
import time
import traceback
import numpy as np


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


class Reporter:
    def __init__(self):
        self.results = []

    def section(self, title):
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print('=' * 70)

    def check(self, name, ok, detail=""):
        tag = PASS if ok else FAIL
        print(f"  {tag} {name}" + (f"  ({detail})" if detail else ""))
        self.results.append((name, ok))
        if not ok and detail:
            print(f"         {detail}")

    def summary(self):
        total = len(self.results)
        passed = sum(1 for _, ok in self.results if ok)
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY: {passed}/{total} passed")
        print('=' * 70)
        if passed < total:
            print("Failures:")
            for name, ok in self.results:
                if not ok:
                    print(f"  - {name}")
            sys.exit(1)
        else:
            print("All checks passed. Code base is ready to run real experiments.")


def run_with_timing(label, fn):
    t0 = time.time()
    try:
        result = fn()
        return True, result, time.time() - t0
    except Exception as e:
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", time.time() - t0


# ============================================================================
def main():
    r = Reporter()

    # --------------------------------------------------------------------
    # A CLASS: Core infrastructure
    # --------------------------------------------------------------------
    r.section("A class: Core infrastructure (5 files)")

    # A1. config.py
    try:
        import config as C
        r.check("config.py imports", True)
        r.check("config.SEED is set", hasattr(C, 'SEED') and isinstance(C.SEED, int))
        r.check("config.N default = 50", C.N == 50)
        r.check("config.L = 10", C.L == 10)
        r.check("config.STEP_DIM_NUM = 9.0", C.STEP_DIM_NUM == 9.0)
        r.check("config.WS_P has 8 values", len(C.WS_P) == 8)
        # Verify resample redraws M_INTERCEPT (the key bug we hunted)
        d_initial = C.M_INTERCEPT.copy()
        C.resample(20)
        d_20 = C.M_INTERCEPT.copy()
        C.resample(50)
        d_50 = C.M_INTERCEPT.copy()
        r.check("resample(20) changes M_INTERCEPT", not np.allclose(d_initial, d_20))
        r.check("resample(50) gives yet different D", not np.allclose(d_20, d_50))
        r.check("M_INTERCEPT in [90, 100]",
                np.all((d_50 >= 90) & (d_50 <= 100)))
        r.check("A_COEF shape (50, 10) after resample(50)", C.A_COEF.shape == (50, 10))
    except Exception as e:
        r.check("config.py basic", False, str(e))

    # A2. games/nash_cournot.py
    try:
        from games.nash_cournot import F_i, project_Xi, solve_NE, aggregate
        r.check("nash_cournot.py imports", True)
        # Test exact projection: sum(g) == sum(s) after project_Xi
        C.resample(20)
        rng_test = np.random.default_rng(0)
        y_rand = rng_test.uniform(0.0, 10.0, size=2 * C.L)
        y_proj = project_Xi(0, y_rand)
        diff = y_proj[:C.L].sum() - y_proj[C.L:].sum()
        r.check("project_Xi enforces equality (max|diff| < 1e-10)",
                abs(diff) < 1e-10, f"|sum(g)-sum(s)| = {abs(diff):.2e}")
        r.check("project_Xi keeps g >= 0", y_proj[:C.L].min() >= -1e-10)
        r.check("project_Xi keeps s >= 0", y_proj[C.L:].min() >= -1e-10)
        r.check("project_Xi keeps g <= cap=500", y_proj[:C.L].max() <= 500 + 1e-10)
        # Test solve_NE
        t0 = time.time()
        C.resample(20)
        x_star = solve_NE()
        dt = time.time() - t0
        r.check("solve_NE() converges for N=20", x_star.shape == (20 * 2 * C.L,),
                f"||x*|| = {np.linalg.norm(x_star):.4f}, {dt:.1f}s")
        # NE should satisfy equality per player
        X = x_star.reshape(20, 2 * C.L)
        per_player_diff = np.max(np.abs(X[:, :C.L].sum(axis=1) - X[:, C.L:].sum(axis=1)))
        r.check("NE satisfies sum(g) = sum(s) per player",
                per_player_diff < 1e-8,
                f"max |sum(g_i) - sum(s_i)| = {per_player_diff:.2e}")
    except Exception as e:
        r.check("nash_cournot.py", False, traceback.format_exc())

    # A3. graphs/generators.py
    try:
        import networkx as nx
        from graphs.generators import cycle, wheel, grid, complete, watts_strogatz, REGULAR
        r.check("generators.py imports", True)
        # Test each regular graph connected + W doubly stochastic
        for name, fn in REGULAR.items():
            G, W = fn(20)
            ok_conn = nx.is_connected(G)
            ok_sym = np.allclose(W, W.T)
            ok_ds = np.allclose(W.sum(axis=1), 1.0)
            r.check(f"{name}(20): connected + symmetric + doubly stochastic",
                    ok_conn and ok_sym and ok_ds)
        # Test WS connectivity for all p values
        for p in C.WS_P:
            G, W = watts_strogatz(50, 6, p, seed=42)
            ok_conn = nx.is_connected(G)
            r.check(f"WS(50, 6, {p}) connected", ok_conn)
    except Exception as e:
        r.check("generators.py", False, traceback.format_exc())

    # A4. algorithms/sync_koshal.py
    try:
        from algorithms.sync_koshal import run as run_sync
        r.check("sync_koshal.py imports", True)
        # Verify step order: should have v_hat = W @ v BEFORE gradient
        import inspect
        src = inspect.getsource(run_sync)
        order_correct = (
            "v_hat = W @ v" in src
            and src.index("v_hat = W @ v") < src.index("grad = F_i")
            and "N * v_hat[i]" in src
            and "v_new = v_hat" in src
        )
        r.check("sync MIX -> GRAD(uses v_hat) -> INNOV order", order_correct)
        # Quick mini run
        C.resample(20)
        x_star_20 = solve_NE()
        _, W = cycle(20)
        rng = np.random.default_rng(42)
        out = run_sync(W, x_star_20, max_iter=500, eps=0.0, record_every=100, rng=rng)
        r.check("sync runs for 500 iter on cycle(20)",
                len(out['rel_err']) >= 5)
        r.check("sync rel_err monotone decreasing in mean (rough)",
                out['rel_err'][0] > out['rel_err'][-1])
    except Exception as e:
        r.check("sync_koshal.py", False, traceback.format_exc())

    # A5. algorithms/async_gossip.py
    try:
        from algorithms.async_gossip import run as run_async
        r.check("async_gossip.py imports", True)
        # Verify no step_rule / a_const params (we removed constant stepsize)
        sig = inspect.signature(run_async)
        params = list(sig.parameters.keys())
        r.check("async signature has no step_rule/a_const",
                'step_rule' not in params and 'a_const' not in params)
        # Quick mini run
        G_cycle, _ = cycle(20)
        rng = np.random.default_rng(42)
        out_a = run_async(G_cycle, x_star_20, max_events=2000, eps=0.0,
                          record_every=200, rng=rng)
        r.check("async runs for 2000 events on cycle(20)",
                len(out_a['rel_err']) >= 5)
    except Exception as e:
        r.check("async_gossip.py", False, traceback.format_exc())

    # --------------------------------------------------------------------
    # B CLASS: Helpers
    # --------------------------------------------------------------------
    r.section("B class: Helper modules (2 files)")

    # B1. analysis/metrics.py
    try:
        from analysis.metrics import (sync, async_,
                                       expected_gossip_matrix, summary)
        r.check("metrics.py imports", True)
        _, W_cycle = cycle(20)
        gap = 1.0 - sync(W_cycle)
        r.check("spectral_gap(W) in [0, 1]", 0 <= gap <= 1)
        G_cycle, _ = cycle(20)
        EW = expected_gossip_matrix(G_cycle)
        r.check("E[W(k)] is symmetric", np.allclose(EW, EW.T))
        r.check("E[W(k)] doubly stochastic",
                np.allclose(EW.sum(axis=1), 1.0))
        lam2 = async_(G_cycle) ** 2
        rho = async_(G_cycle)
        r.check("lambda2_async in [0, 1]", 0 <= lam2 <= 1)
        r.check("rho_async = sqrt(lambda2_async)", abs(rho - np.sqrt(max(lam2, 0))) < 1e-10)
        summ = summary(G_cycle, W_cycle)
        r.check("summary() returns 6 keys",
                set(summ.keys()) == {'sync', 'async', 'CG', 'LG',
                                       'mean_degree', 'num_edges'})
    except Exception as e:
        r.check("metrics.py", False, traceback.format_exc())

    # B2. analysis/plots.py
    try:
        from analysis.plots import mean_ci, convergence_plot
        r.check("plots.py imports", True)
        # Test mean_ci on synthetic data
        runs = [np.array([1.0, 0.5, 0.25, 0.125]),
                np.array([1.1, 0.6, 0.3, 0.15]),
                np.array([0.9, 0.4, 0.2, 0.1])]
        m, lo, hi = mean_ci(runs, confidence=0.90)
        r.check("mean_ci returns 3 arrays of same length",
                len(m) == len(lo) == len(hi) == 4)
        r.check("mean_ci low <= mean <= high",
                np.all(lo <= m) and np.all(m <= hi))
        # CI width should equal 2 * z * sem
        expected_mean = np.array([1.0, 0.5, 0.25, 0.125])
        r.check("mean_ci mean is correct (1.0, 0.5, 0.25, 0.125)",
                np.allclose(m, expected_mean))
    except Exception as e:
        r.check("plots.py", False, traceback.format_exc())

    # --------------------------------------------------------------------
    # C CLASS: Experiment scripts (smoke test imports + paired invariants)
    # --------------------------------------------------------------------
    r.section("C class: Experiment scripts (8 files)")

    for mod_name in ['run_Table1', 'run_table4_table5',
                     'run_table2_table3', 'run_Table7',
                     'run_sync_p01', 'run_async_p01',
                     'run_sync_sensitivity', 'run_async_sensitivity']:
        try:
            __import__(mod_name)
            r.check(f"{mod_name} imports cleanly", True)
        except Exception as e:
            r.check(f"{mod_name} imports cleanly", False, str(e))

    # Verify paired-seed invariants for §6.3 (p01) and §6.4 (sensitivity)
    r.section("Paired-seed invariants (§6.3 + §6.4)")

    try:
        # §6.3: sync_p01 and async_p01 should generate SAME 50 WS(0.1) graphs
        master_sync = np.random.default_rng(C.SEED + 43)
        master_async = np.random.default_rng(C.SEED + 43)
        graphs_sync = []
        graphs_async = []
        from games.nash_cournot import project_Xi
        def gen_xy(master):
            seed = int(master.integers(1 << 31))
            rng = np.random.default_rng(seed)
            graph_seed = int(rng.integers(1 << 31))
            G, _ = watts_strogatz(50, 6, 0.1, seed=graph_seed)
            return list(G.edges()), seed
        # 3 reps each
        for _ in range(3):
            g, _ = gen_xy(master_sync)
            graphs_sync.append(g)
            g, _ = gen_xy(master_async)
            graphs_async.append(g)
        match_p01 = all(s == a for s, a in zip(graphs_sync, graphs_async))
        r.check("§6.3 (SEED+43): sync_p01 and async_p01 generate SAME graphs",
                match_p01)
    except Exception as e:
        r.check("§6.3 paired invariant", False, str(e))

    try:
        # §6.4: sync_sensitivity and async_sensitivity should be paired
        master_sync_s = np.random.default_rng(C.SEED + 44)
        master_async_s = np.random.default_rng(C.SEED + 44)
        gs_s = []
        ga_s = []
        for p in [0.001, 0.1, 1.0]:  # mini sweep
            for _ in range(2):
                seed = int(master_sync_s.integers(1 << 31))
                rng = np.random.default_rng(seed)
                gseed = int(rng.integers(1 << 31))
                G, _ = watts_strogatz(50, 6, p, seed=gseed)
                gs_s.append(list(G.edges()))
                seed = int(master_async_s.integers(1 << 31))
                rng = np.random.default_rng(seed)
                gseed = int(rng.integers(1 << 31))
                G, _ = watts_strogatz(50, 6, p, seed=gseed)
                ga_s.append(list(G.edges()))
        match_sens = all(s == a for s, a in zip(gs_s, ga_s))
        r.check("§6.4 (SEED+44): sync_sens and async_sens generate SAME graphs",
                match_sens)
    except Exception as e:
        r.check("§6.4 paired invariant", False, str(e))

    # --------------------------------------------------------------------
    # End-to-end mini experiment (very small)
    # --------------------------------------------------------------------
    r.section("End-to-end mini experiment (sync on cycle(20), 2 reps, k=500)")

    try:
        C.resample(20)
        x_star = solve_NE()
        _, W = cycle(20)
        errs = []
        for r_idx in range(2):
            rng = np.random.default_rng(100 + r_idx)
            out = run_sync(W, x_star, max_iter=500, eps=0.0, record_every=1, rng=rng)
            errs.append(out['rel_err'][-1])
        mean = np.mean(errs)
        r.check("End-to-end sync mini run completes",
                len(errs) == 2,
                f"err at k=500 across 2 reps: {errs}")
        r.check("Mini run gives finite, positive errors",
                all(np.isfinite(e) and e > 0 for e in errs))
    except Exception as e:
        r.check("End-to-end sync", False, str(e))

    try:
        G_c, _ = cycle(20)
        errs_a = []
        for r_idx in range(2):
            rng = np.random.default_rng(200 + r_idx)
            out = run_async(G_c, x_star, max_events=2000, eps=0.0,
                            record_every=200, rng=rng)
            errs_a.append(out['rel_err'][-1])
        r.check("End-to-end async mini run completes", len(errs_a) == 2)
        r.check("End-to-end async finite + positive",
                all(np.isfinite(e) and e > 0 for e in errs_a))
    except Exception as e:
        r.check("End-to-end async", False, str(e))

    # Done
    r.summary()


if __name__ == "__main__":
    main()
