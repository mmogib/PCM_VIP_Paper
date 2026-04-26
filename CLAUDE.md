# DIPCM — Julia Implementation

## Overview
Julia implementation of the double inertial projection-contraction method (IPCMAS1, IPCMAS2, DIPCM) for solving variational inclusion problems. Uses Style B (Flat Include) architecture.

## Structure
```
jcode/
├── Project.toml           # Dependencies
├── CLAUDE.md              # This file
├── src/
│   ├── includes.jl        # Entry point: loads all source files
│   ├── dependences.jl     # Package imports
│   ├── types.jl           # Algorithm types, Problem, Solution
│   ├── algorithms.jl      # All algorithm implementations + parameter functions
│   ├── functions.jl       # Operator definitions, startSolvingExample
│   ├── projections.jl     # Projection operators
│   ├── utils.jl           # CSV/XLSX conversion, parse_args, prepare_filepath
│   └── examples/          # Original experiment scripts (archived)
├── scripts/               # Experiment scripts
│   # ── s* = strong-convergence DIPCM track (USED in current paper §4) ──
│   ├── s10_example1_vip.jl        # VIP example (§4.1)
│   ├── s12_sensitivity.jl         # New DIPCM sensitivity: β̄ × θ̄ grid + ε-scale robustness
│   ├── s15_csv_to_latex.jl        # LaTeX tables (median time, average iter)
│   ├── s20_example2_sfp.jl        # Split feasibility problem (§4.2)
│   ├── s30_example3_en.jl         # Elastic net problem (§4.3)
│   ├── s40_linear_convergence.jl  # R-linear convergence verification (§4.4)
│   # ── w* = weak-convergence comparison track (NOT used in current paper; kept for record) ──
│   ├── w11_sensitivity_ipcmas1.jl
│   ├── w12_weak_comparison_vip.jl
│   ├── w13_analyze_sensitivity.jl
│   ├── w22_weak_comparison_sfp.jl
│   ├── w32_weak_comparison_en.jl
│   ├── w50_perf_profile.jl
│   └── w51_weak_latex.jl
├── results/               # Output data
└── archive/               # Archived code versions
    ├── dipcm_legacy.jl              # Pre-2026-04-25 DIPCM (archived 2026-04-26)
    ├── s12_legacy_sensitivity.jl    # Legacy α × β_zn × θ̄ sensitivity (archived 2026-04-26)
    └── algov{1,2}.jl, example*.jl   # Pre-existing snapshots
```

## Algorithms Implemented (`src/algorithms.jl`)
```
# Used in current paper:
DIPCM            # §3.4 main algorithm (paper line 1833+) — 3-term Halpern–Mann update:
                 #   x_{n+1} = (1 − α_n − σ_n) z_n + σ_n u_n
                 # Adaptive inertial parameters β'_n, θ_n with ε_n ≤ ε'_n;
                 # Strong convergence to P_Ω(0) via Saejung–Yotkaew lemma (no strong monotonicity).
                 # Defaults: β̄=0.3, θ̄=0.9, α_n=1/(n+1), σ_n=0.8−α_n,
                 #           ε_n=5/(n+1)^2.1, ε'_n=10/(n+1)^2.1, ξ_n=100/(n+1)^1.1.
IPCMAS1          # Algorithm 1 — convex combination (1−α) z_n + α u_n (weak convergence under mono+Lip)
IPCMAS2          # Algorithm 2 — single inertial PCM, β=0 (R-linear under strong monotonicity)
DeyHICPP         # Dey (2023) — single inertial Halpern PCM, constant stepsize λ ∈ (0, 1/L)
Suantai2024      # SICIP — Suantai et al. (2024) — combined double inertial PCM, adaptive stepsize
DongIPCA         # Dong et al. (2018) — inertial projection contraction (variational inequalities)

# Archived (not in src/algorithms.jl):
DIPCM_legacy     # Pre-2026-04-25 implementation. Lives at archive/dipcm_legacy.jl.

# Implemented for the shelved §4.4 weak-comparison; NOT in current paper:
TanQin2024       # Tan & Qin (2024), Algorithm 3.1
ChenMiPCA        # Chen, Zhang, Dong (2020), Algorithm 3.1
PeeyadaIMFBSA    # Peeyada et al. (2022), Algorithm 3.1
```

Each algorithm has a `get_<Name>_params(L)` function returning a named tuple of parameters.

## Type Hierarchy
```
Problem             # Test problem definition
  - name, Aλ (resolvent), A, B (operator), L (Lipschitz), x0, x1, n (dim)
  - dot, norm, stopping (customizable)

Solution{T}         # Algorithm output
  - solver, problem, solution, iterations, time, converged
  - parameters, history (:dk, :xk, :err vectors)
```

## Running Scripts
```bash
cd jcode
julia --project=. scripts/s10_example1_vip.jl                    # default comparison
julia --project=. scripts/s10_example1_vip.jl --theta-sync       # controlled θ=0.9
julia --project=. scripts/s12_sensitivity.jl                      # sensitivity analysis
julia --project=. scripts/s12_sensitivity.jl --dim 300            # sensitivity at N=300
julia --project=. scripts/s20_example2_sfp.jl                     # SFP example
julia --project=. scripts/s30_example3_en.jl                      # elastic net
julia --project=. scripts/s40_linear_convergence.jl               # R-linear verification
```

## Key Design Decisions
- DIPCM is the main algorithm for experiments; IPCMAS1 is kept for theory (matches paper's Algorithm 1)
- IPCMAS1 code matches the paper exactly: `(1−α) z_n + α u_n` with constant α
- DIPCM code uses old presentation `α z_n + σ_n u_n`, σ_n = (1−α) − β_n; structurally equivalent to paper's new 3-term form
- Adaptive stepsize: λ₁ defaults to 1/(2L), can be overridden
- Suantai2024 uses non-increasing stepsize: λ_{k+1} = min{μ‖w_k − y_k‖/‖f(w_k) − f(y_k)‖, λ_k}
- TeeIO logging via `src/io_utils.jl` — all scripts log to both stdout and file

## Rules
- **DO NOT run Julia scripts.** Mohammed runs scripts locally. Only create/edit scripts.
- Tests (`runtests.jl`) may be run to verify code changes.
- **jcode/ is self-contained.** No script here may write to a path outside `jcode/` (no `../paper/...` outputs). All generated tables/figures/CSVs live under `jcode/results/`. The paper side copies what it needs manually — tables get **pasted inline** into `paper/main.tex` (no `\input` from jcode/).

## Status (2026-04-26, Phases 1–5 complete; awaiting GitHub commit)

**Phase 1 ✅** — New `DIPCM` in `algorithms.jl` matches paper §3.4. Smoke test PASS.

**Phase 2 ✅** — Sensitivity ran (bundle A3+B1+C1+D1+E2). Defaults locked:
- $\bar\beta = 0.3$, $\bar\theta = 0.9$, $\gamma=1.1$, $\mu=0.5$
- $\alpha_n = 1/(n+1)$, $\sigma_n = 0.8 - \alpha_n$
- $\varepsilon_n = 5/(n+1)^{2.1}$, $\varepsilon'_n = 10/(n+1)^{2.1}$, $\xi_n = 100/(n+1)^{1.1}$
- $\lambda_1 = 1/(1.05L)$ for §4.1 / §4.4, $\lambda_1 = 0.05$ for §4.2 / §4.3.
- Sensitivity CSVs at `results/sensitivity/{sensitivity, epsilon_scale}.csv`.

**Phase 3 ✅** — Main runs complete (with JIT warmup):
- s10 (Ex 1 VIP): DIPCM **44 iters** at N=300, ε=10⁻⁶ vs DeyHICPP **44,418** → over 1000× speedup.
- s20 (Ex 2 SFP): DIPCM avg **3.6 iters** (DeyHICPP 4.6, SICIP 49.9).
- s30 (Ex 3 EN): MSE **10.91 ± 1.68** (matches paper exactly).
- s40 (R-linear): DIPCM ρ=0.9997, R²=0.767 (not R-linear); IPCMAS2 ρ=0.983, R²=0.989 (R-linear ✓).

**Phase 4 ✅** — `paper/main.tex` §4 updated with new numbers, parameter listings, narrative tweaks, sensitivity table+remark, all 5 figures refreshed. Aggregator: `s15` now produces *median CPU time* + *average iterations* in §4.1 / §4.2 tables. Concerns A (`λ_1 = 1/(1.05L)`) and B (`ξ_n = 100/(n+1)^{1.1}` in §4.2) fixed. Algorithm label renamed `algo:DIPCM`. All revision-round material in `\color{blue}` (32 prior `\color{red}` instances converted).

**Phase 5 ✅** — Cleanup done 2026-04-26:
- `DIPCM_legacy` and `get_DIPCM_legacy_params` moved to `archive/dipcm_legacy.jl`.
- `s12_legacy_sensitivity.jl` moved to `archive/`.
- Throwaway helpers (`s00_dipcm_smoke.jl`, `s01_relabel_legacy_dipcm.jl`, `s99_run_all.jl`) deleted.
- Pre-Phase-3 backup, `comparison_legacy.csv` siblings, `sensitivity_legacy/` deleted.
- s10/s20/s30 `ALL_ALGORITHMS` no longer reference `DIPCM_legacy`.

CSV state (clean):
- `results/example_1/comparison_incremental.csv` — 600 rows (4 algorithms × 5 N × 6 ε × 5 instances).
- `results/example_2/comparison_incremental.csv` — 24 rows (3 algos × 8 instances).
- `results/example_3/elastic_net_simulation_20260426_08_59_48.csv` — 50-run summary.
- `results/linear_convergence/linear_convergence_results_20260426_08_59_58.csv`.
- `results/sensitivity/{sensitivity, epsilon_scale}.csv`.

**Next**: Mohammed pushes `jcode/` to GitHub, then drafts response letters anchored to the new `main.tex` numbers.
- w-prefix scripts and TanQin / Chen / Peeyada algorithm implementations remain in the repo for record (shelved §4.4 weak-comparison work). Not invoked from any active s-script.
- Code pushed to GitHub (commit d476e99, 2026-04-07). New commit planned at end of Phase 5.
