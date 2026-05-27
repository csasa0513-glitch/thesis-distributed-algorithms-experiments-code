# Code Audit Report — `thesis-code/`

**Date**: 2026-05-27
**Files audited**: 5 (A) + 2 (B) + 8 (C) = **15 files**
**Reference**: Koshal, Nedic, Shanbhag (2016), Op. Res. 64(3)

---

## Executive Summary

| Status | Count | Notes |
|---|---|---|
| ✅ Correct (no action) | 12 | Aligned with Koshal + thesis design |
| 🟡 Minor risk (fix recommended) | 3 | `NE.npz` caching issue + 1 docstring |
| ❌ Wrong (must fix) | 0 | — |

**Bottom line**: Ready to run after **one preventive action** (always delete `results/NE.npz` before any p01/sensitivity script). All algorithms match Koshal one-to-one.

---

## A class — Core infrastructure (5 files)

### A1. `config.py` (1543 bytes) ✅

**Purpose**: Game parameters, master seed, per-N resample function.

**Koshal §6.1 matching**:

| Koshal (eq./section) | Code |
|---|---|
| $d_\ell \sim U[90, 100]$ | `RNG.uniform(90.0, 100.0, size=L)` ✓ |
| $a_{i\ell} \sim U[2, 12]$ | `RNG.uniform(2.0, 12.0, size=(N, L))` ✓ |
| $b_{i\ell} \sim U[2, 3]$ | `RNG.uniform(2.0, 3.0, size=(N, L))` ✓ |
| $\text{cap}_{i\ell} = 500$ | `CAP = np.full((N, L), 500.0)` ✓ |
| $L = 10$ | `L: int = 10` ✓ |
| $N \in \{20, 50\}$ | `resample(N_new)` handles both ✓ |
| Per-N independent Cournot instance | `default_rng(SEED + N_new)` re-seeds for each N ✓ |
| Diminishing stepsize numerator 9 | `STEP_DIM_NUM: float = 9.0` ✓ |

**Verified**: `D_INTERCEPT, A_COEF, B_COEF` all redrawn in `resample()` (user confirmed in PowerShell).

---

### A2. `games/nash_cournot.py` (3472 bytes) ✅

**Purpose**: Gradient $F_i$, exact Euclidean projection $\Pi_{X_i}$, offline NE solver.

**Koshal matching**:

| Koshal (eq./page) | Code |
|---|---|
| eq. (64): $\sum g_{il} = \sum s_{il}$ (equality) | `project_Xi` enforces $\sum g = \sum s$ via Lagrange + bisection ✓ |
| eq. (65) implicit: bounds $0 \le g \le \text{cap}, s \ge 0$ | `np.minimum(np.maximum(y_g - mu, 0), cap)`, `np.maximum(y_s + mu, 0)` ✓ |
| Gradient $\nabla_{x_i} f_i$ at $(x_i, \bar s)$ | `F_i(i, x_i, u)`: cost part $a + 2bg$, price part $u - d + s$ ✓ |
| Algorithm $\Pi_{K_i}[\cdot]$ — exact Euclidean projection | Lagrange + 100-iteration bisection (machine precision) ✓ |

**Strict Euclidean projection** (not Dykstra approximation): KKT closed form parameterised by single $\mu$, bisection finds root in 1D. Mathematically equivalent to the true projection.

**solve_NE**: constant stepsize $\tau = 1/L_{lip}$, projected gradient with true aggregate (centralized). Matches Koshal §6.1 last paragraph ("NE computed by constant steplength gradient projection algorithm assuming each agent has true information of the aggregate"). ✓

---

### A3. `graphs/topologies.py` (4722 bytes) ✅

**Purpose**: Generate cycle/star/grid/complete + WS(N, K, p); compute Koshal weight matrix.

**Koshal matching**:

| Koshal (eq./page) | Code |
|---|---|
| p. 699: $W_{ij} = \delta$ if $(i,j) \in E$, $W_{ii} = 1 - \delta d_i$, $\delta = 0.5/\max_i d_i$ | `koshal_weights(G)` — exactly this formula ✓ |
| Static network connectivity guaranteed | `connected_watts_strogatz_graph(... tries=100)` for $p > 0$ ✓ |
| 4 baseline topologies | `cycle`, `star`, `grid` (N/5 × 5), `complete` ✓ |

**Note**: Koshal calls "wheel" what is structurally a star graph (1 center + N-1 leaves). Code uses `nx.star_graph(N-1)` which gives N nodes. ✓

**Verified**: All R=50 WS(N=50, K=6, p) realisations across 6 p values are connected (tested earlier).

---

### A4. `algorithms/sync_koshal.py` (5587 bytes) ✅

**Purpose**: Algorithm 1 (sync projected gradient with consensus on $v$).

**Koshal Alg 1 (eq. 9-11) one-to-one**:

| Koshal step | Eq. | Code line | Match |
|---|---|---|---|
| 1. Mixing | (9): $\hat v_i^k = \sum_j W_{ij} v_j^k$ | `v_hat = W @ v` (line 101) | ✓ |
| 2. Gradient using **mixed** $\hat v$ | (10): $x_i^{k+1} = \Pi_{K_i}[x_i^k - \alpha_k F_i(x_i^k, N \hat v_i^k)]$ | `grad = F_i(l, x[l], N * v_hat[l])` (line 112) | ✓ |
| 3. Innovation | (11): $v_i^{k+1} = \hat v_i^k + (s_i^{k+1} - s_i^k)$ | `v_new = v_hat + (s_new - s_prev)` (line 118) | ✓ |

**Stepsize**: $\tau_k = 1/k$ (paper, 1-indexed) → `tau = 1.0 / (k + 1)` (Python 0-indexed). ✓

**Centralized mode** (`known_aggregate=True`): uses true $\bar s = \sum_j s_j$ instead of $N \hat v_i$. Matches Koshal §6.1 baseline. ✓

**Initial v**: $v_i^0 = s_i^0$ ✓ (line 84, `v = x[:, L:].copy()`)

---

### A5. `algorithms/async_gossip.py` (4948 bytes) ✅

**Purpose**: Algorithm 2 (gossip-based async with pairwise consensus).

**Koshal Alg 2 (eq. 23-25) one-to-one**:

| Koshal step | Eq. | Code line | Match |
|---|---|---|---|
| Activate $I^k \sim U\{1..N\}$ | p. 689 | `i = int(rng.integers(N))` (line 97) | ✓ |
| Contact $J^k \sim U(\mathcal{N}_{I^k})$, $p_{ij} = 1/d_i$ | p. 689 (simplest valid choice) | `j = int(rng.choice(nbrs))` (line 101) | ✓ |
| Consensus on $v$ | (23): $\hat v_i = \hat v_j = (v_i + v_j)/2$ | `avg = 0.5*(v[i]+v[j]); v[i]=v[j]=avg` (line 104-105) | ✓ |
| Gradient using $\hat v$ | (24a): $x_l^{k+1} = \Pi[x_l^k - \alpha_{k,l} F_l(x_l^k, N \hat v_l^k)]$ | `grad = F_i(l, x[l], N * v[l])` (line 115; v[l] already $\hat v$ after consensus) | ✓ |
| Innovation | (24b): $v_l^{k+1} = \hat v_l^k + (s_l^{k+1} - s_l^k)$ | `v[l] = v[l] + (x_new_l[L:] - s_prev[l])` (line 118) | ✓ |
| Others idle | (25): $v_i^{k+1} = v_i^k, x_i^{k+1} = x_i^k$ for $i \notin \{I^k, J^k\}$ | implicit (only `for l in (i, j)`) | ✓ |

**Stepsize**: $\alpha_{k,i} = 9 / \Gamma_i^k$ (per-node update counter $\Gamma$). ✓ Matches Koshal Tables 5-6.

**Constant stepsize variant**: **removed** (Tables 7-8 out of thesis scope) ✓.

**Verified by user**: paired with sync_p01 produces SAME 50 WS(0.1) graphs (max_diff = 7.5e-12).

---

## B class — Helpers (2 files)

### B1. `analysis/metrics.py` (3720 bytes) ✅

**Purpose**: Spectral indicators + structural graph statistics.

**Koshal matching**:

| Function | Definition | Koshal reference |
|---|---|---|
| `spectral_gap(W)` | $1 - \lvert\lambda_2(W)\rvert$ | Standard convergence rate for sync |
| `lambda2_async(G)` | $\lambda_2(E[W(k)])$ via Boyd-Gossip eq. (7) form | Koshal Table 9 uses this |
| `rho_async(G)` | $\sqrt{\lambda_2(E[W(k)])}$ | Koshal p. 701 (square root convention) |
| `expected_gossip_matrix(G)` | $I - \frac{1}{2N}\sum_{(i,j)\in E}(1/d_i + 1/d_j)(e_i - e_j)(e_i - e_j)^T$ | Boyd 2006 eq. (7) ✓ |

**Verified** mathematically: $E[W(k)]$ is symmetric, doubly stochastic, PSD with eigenvalues in $[0, 1]$.

`summary(G, W)` returns dict with: spectral_gap, clustering, avg_path_length, deg_hetero, diameter, mean_degree, num_edges. Used by p01 + sensitivity scripts to enrich CSV. ✓

---

### B2. `analysis/plots.py` (2378 bytes) ✅

**Purpose**: Generic plotting + CI band helper.

**CI formula** (line 60-66):
$$w_{90} = 2 \cdot z_{0.95} \cdot \frac{s_R}{\sqrt R},\quad z_{0.95} = 1.645$$

Matches Koshal Tables 2, 4, 7, 8 convention. ✓

**`mean_ci()`**: pads shorter runs with last value (handles early termination). Returns `(mean, low, high)` for `fill_between`. Width $= 2 z \cdot \text{SEM}$. ✓

---

## C class — Experiment scripts (8 files)

### C1. `run_sync_centralized.py` (4952 bytes) ✅

**Maps to**: Koshal Tables 3, 4 (centralized sync baseline) → your §6.2.

**Algorithm**: centralized projected gradient with true aggregate (no W, no consensus). $\tau_k = 1/k$.

**Setup**:
- $N \in \{20, 50\}$ ✓
- $R = 50$ ✓
- Horizons $\tilde k \in \{5000, 10000\}$ ✓
- Master seed `C.SEED + 1` (independent of all other scripts) ✓
- Calls `C.resample(N)` + `solve_NE()` per N — fresh ✓ (no NE.npz cache)

**Output**: `results/sync_centralized.csv` with columns `N, k, mean_err, ci_width`. ✓

---

### C2. `run_async_regular.py` (8875 bytes) ✅

**Maps to**: Koshal Tables 5, 6 (async on 4 baselines, diminishing stepsize) → your §6.2.

**Setup**:
- $N \in \{20, 50\}$ ✓ (was `(50,)` only — I fixed this earlier)
- $R = 50$ per (N, topology) ✓
- Horizons $\tilde k \in \{5 \cdot 10^4, 10^5\}$ ✓
- Diminishing stepsize only ✓ (constant variant removed)
- Calls `C.resample(N)` + `solve_NE()` per N — fresh ✓
- Seed scheme: per (N, topology, rep) independent — **marginal** wrt sync ✓

**Output**: `results/async_regular_tables_5_8.csv` with columns `N, topology, k, mean_error, ci90_width`. ✓

---

### C3. `run_sync.py` (10900 bytes) 🟡 **Minor docstring issue**

**Maps to**: Your §6.2 extension — sync on 4 static baselines (NOT in Koshal explicitly).

**Setup**:
- $N \in \{20, 50\}$ ✓
- $R = 50$ ✓
- Horizons 5000, 10000 ✓
- Master seed `C.SEED + 1000 + N_new` — same x⁰ batch shared across distributed + centralized for the SAME N ✓
- Marginal wrt async_regular (different seed scheme) ✓

**🟡 Docstring bug** (line 1-6, 39-40):
> "Q1: synchronous algorithm on the four regular graphs ... reproducing Koshal et al. (2016) Tables 3-4 (static network)..."

This is **wrong** — Koshal Tables 3, 4 are CENTRALIZED. This script runs DISTRIBUTED sync. The docstring should clarify:
- "Tables 3, 4 are reproduced by `run_sync_centralized.py`"
- This script is "our extension: distributed sync on 4 baselines"

**Fix**: I'll update the docstring (see Recommended Fixes section).

**Code itself**: correct.

---

### C4. `run_ws_graph_stats.py` (2487 bytes) ✅

**Maps to**: Your §6.3.1 Table 6 (WS graph structural indicators).

**Setup**:
- $p \in \{0, 0.001, 0.01, 0.1, 0.5, 1.0\}$ ✓
- $R = 50$ per p ✓
- $N = 50$, $K = 6$ ✓
- No algorithm run (pure graph stats); no NE needed ✓
- Master seed `C.SEED` (not `+1` or `+43` — its own scheme, doesn't conflict) ✓

**Output**: `results/ws_graph_stats.csv` with columns p, rep, clustering, avg_path_length, diameter, deg_hetero, mean_degree, num_edges. ✓

**Note**: Doesn't report $\lvert\lambda_2\rvert$ here — correctly relegated to Table 7 (sync) and ws-sensitivity (async). ✓

---

### C5. `run_sync_p01.py` (6341 bytes) ⚠️ **NE cache risk**

**Maps to**: Your §6.3.2 Table 7 sync row + Figure 3.

**Setup**:
- $p = 0.1$ fixed ✓
- $R = 50$ ✓
- Horizon $\tilde k = 10^4$ ✓
- Master seed `C.SEED + 43` — **paired** with `run_async_p01.py` ✓
- $K_{START} = 100$ for figure crop ✓

**🟡 Issue**: `_load_or_solve_NE()` (line 55-61) **caches NE.npz**. If a previous run saved NE.npz with DIFFERENT game parameters (e.g., old code, or different N), it loads stale NE → all error metrics wrong.

**Symptoms** if cache stale:
- Error numbers look unrealistic
- Trajectories don't converge correctly

**Workaround**: Always delete `results/NE.npz` before running this script.

**Better fix**: Remove the caching, always call `C.resample(50); x_star = solve_NE()` fresh. (See Recommended Fixes.)

**Doesn't call `C.resample(50)`**: relies on default `C.N = 50` from config import. This is safe in fresh Python sessions but fragile.

---

### C6. `run_async_p01.py` (6483 bytes) ⚠️ **NE cache risk** (same as C5)

**Maps to**: Your §6.3.2 Table 7 async row + Figure 4.

**Setup**:
- $p = 0.1$ fixed ✓
- $R = 50$ ✓
- Horizon $\tilde k = 10^5$ ✓
- Master seed `C.SEED + 43` — **paired** with `run_sync_p01.py` ✓
- Calls `lambda2_async`, `rho_async` correctly ✓

**🟡 Same NE cache issue as C5.**

---

### C7. `run_sync_sensitivity.py` (5777 bytes) ⚠️ **NE cache risk** (same)

**Maps to**: Your §6.4 sensitivity (sync part).

**Setup**:
- $p \in C.WS\_P = \{0, 0.001, 0.01, 0.1, 0.5, 1.0\}$ ✓ (6 values)
- $R = 50$ per p ✓
- Horizon $\tilde k = 10^4$ ✓
- Master seed `C.SEED + 44` — **paired** with `run_async_sensitivity.py` ✓
- **Loop order**: p OUTER, rep INNER — must match async sensitivity for paired ✓
- CSV columns include `abs_lambda2 = 1 - spectral_gap` ✓
- Figure: skip p=0 ✓
- Calls `C.resample(50)` explicitly ✓

**🟡 Same NE cache issue.**

---

### C8. `run_async_sensitivity.py` (4668 bytes) ⚠️ **NE cache risk** (same)

**Maps to**: Your §6.4 sensitivity (async part).

**Setup**:
- $p \in C.WS\_P$ ✓
- $R = 50$ per p ✓
- Horizon $\tilde k = 10^5$ ✓
- Master seed `C.SEED + 44` — **paired** with `run_sync_sensitivity.py` ✓
- Same loop order ✓
- CSV columns include `lambda2_async, rho_async` ✓
- Figure: skip p=0 ✓
- step_rule parameter **removed** ✓

**🟡 Same NE cache issue.** **Does NOT call `C.resample(50)` explicitly** — relies on import default. (Should add for safety.)

---

## Recommended fixes

### Fix 1 (mandatory): Always delete `NE.npz` before any p01/sensitivity script

Add to your pre-run checklist:

```powershell
cd C:\Claude_BA_thesis\thesis-code
Remove-Item results\NE.npz -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force
```

**Then** run sync_p01 / async_p01 / sync_sensitivity / async_sensitivity.

Alternatively, I can remove the `_load_or_solve_NE` caching from the 4 affected scripts so they always recompute. Recommend doing this — caching saves ~5 sec at most, not worth the bug risk. Say "remove caching" and I'll do it.

### Fix 2 (optional, cosmetic): Update `run_sync.py` docstring

Replace the misleading "reproducing Koshal et al. (2016) Tables 3-4" with "extension: distributed sync on 4 baseline graphs (Koshal Tables 3-4 are the centralized baseline, reproduced separately by `run_sync_centralized.py`)".

Say "fix docstring" and I'll do it.

### Fix 3 (optional, defensive): Add `C.resample(50)` to `run_async_sensitivity.py`

It's safer to be explicit than rely on import default. Say "add resample" and I'll do it.

---

## Recommended run order (to minimize NE.npz risk)

```powershell
cd C:\Claude_BA_thesis\thesis-code

# Step 1: nuke cache + bytecode
Remove-Item results\NE.npz -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force

# Step 2: Koshal Tables 3, 4 (centralized) — already running for you
python -u run_sync_centralized.py

# Step 3: §6.2 extension — distributed sync on 4 baselines
Remove-Item results\NE.npz -ErrorAction SilentlyContinue   # safety
python -u run_sync.py                                       # ~10-30 min

# Step 4: Koshal Tables 5, 6 (async on 4 baselines)
Remove-Item results\NE.npz -ErrorAction SilentlyContinue   # safety
python -u run_async_regular.py                              # ~30-90 min

# Step 5: §6.3 representative WS p=0.1 (sync + async paired)
Remove-Item results\NE.npz -ErrorAction SilentlyContinue
python -u run_sync_p01.py                                   # ~10 min
# DO NOT delete NE.npz here — async should reuse the same NE as sync
python -u run_async_p01.py                                  # ~50-100 min

# Step 6: §6.4 sensitivity (sync + async paired)
Remove-Item results\NE.npz -ErrorAction SilentlyContinue
python -u run_sync_sensitivity.py                           # ~30-60 min
# DO NOT delete NE.npz here either
python -u run_async_sensitivity.py                          # ~5-10 hours
```

**Total runtime budget**: ~10-15 hours (mostly async). Run async overnight.

After each step, `git add . && git commit -m "..." && git push` for safety.

---

## Conclusion

✅ **No algorithmic bugs**. All A class + B class files match Koshal 2016 line-by-line.

🟡 **One systemic concern**: NE.npz caching in 4 scripts can cause stale-reference bug. Easy fix: delete `NE.npz` between major code changes.

🟡 **One cosmetic issue**: `run_sync.py` docstring misleadingly says "Tables 3-4" — these are centralized, but this script runs distributed.

Ready to run after preventive fixes.
