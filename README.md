# DIPCM — Double Inertial Projection-Contraction Method

Julia implementation for the paper:
**"Projection and contraction method with double inertial steps for variational inclusion problems on Hilbert spaces"**

## Algorithms

| Function | Paper | Update Rule | Convergence |
|----------|-------|-------------|-------------|
| `DIPCM` | §3.4 (DIPCM) | `x_{n+1} = (1 − α_n − σ_n) z_n + σ_n u_n` | Strong to `P_Ω(0)` (no strong monotonicity) |
| `IPCMAS1` | Algorithm 1 | `x_{n+1} = (1−α) z_n + α u_n` | Weak |
| `IPCMAS2` | Algorithm 2 | `x_{n+1} = (1−α) x_n + α u_n` (single inertial) | R-linear under strong monotonicity |
| `DeyHICPP` | Dey (2023) | `x_{n+1} = (1−θ_n−β_n) x_n + θ_n z_n` | Strong |
| `Suantai2024` | Suantai et al. (2024) | `x_{n+1} = w_n − γ β_n d_n` | Weak |

**DIPCM** is the main algorithm (paper §3.4). It uses adaptive inertial caps `β'_n, θ_n` controlled by sequences `ε_n ≤ ε'_n`, an explicit Halpern weight `α_n`, and a Mann weight `σ_n`. Strong convergence proof uses the Saejung–Yotkaew (2012) convergence lemma — strong monotonicity is **not** required.

Default parameters (locked 2026-04-26 from sensitivity analysis):

```
β̄ = 0.3,  θ̄ = 0.9,  γ = 1.1,  μ = 0.5
α_n = 1/(n+1),       σ_n = 0.8 − α_n
ε_n = 5/(n+1)^2.1,   ε'_n = 10/(n+1)^2.1
ξ_n = 100/(n+1)^1.1
λ_1 = 1/(1.05·L)     # for §4.1 / §4.4 (large L)
λ_1 = 0.05           # for §4.2 / §4.3
```

## Quick Start

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Sensitivity (β̄ × θ̄ grid + ε-scale robustness check)
julia --project=. scripts/s12_sensitivity.jl

# Example 1: VIP comparison (DIPCM vs DeyHICPP vs SICIP vs IPCMAS2)
julia --project=. scripts/s10_example1_vip.jl

# Generate LaTeX tables for §4.1 (median time, average iter)
julia --project=. scripts/s15_csv_to_latex.jl --csv results/example_1/comparison.csv

# Example 2: Split feasibility problem (L²[0,1])
julia --project=. scripts/s20_example2_sfp.jl

# Example 3: Elastic net regularization (50 simulation runs)
julia --project=. scripts/s30_example3_en.jl

# R-linear convergence verification (paper §4.4)
julia --project=. scripts/s40_linear_convergence.jl
```

## Scripts

| Script | Purpose | Flags |
|--------|---------|-------|
| `s10_example1_vip.jl` | VIP comparison (§4.1; 5 dims × 6 tols × 5 instances) | `--dim=`, `--algo=`, `--force`, `--verbose` |
| `s12_sensitivity.jl` | DIPCM sensitivity: β̄ × θ̄ grid + ε-scale at locked best | `--dim=`, `--skip-eps`, `--skip-grid` |
| `s15_csv_to_latex.jl` | LaTeX tables (vip / sfp / weak format) — median time, average iter | `--csv=`, `--format=`, `--output=` |
| `s20_example2_sfp.jl` | SFP comparison (§4.2; L²[0,1], 8 instances) | `--dim=`, `--algo=`, `--force` |
| `s30_example3_en.jl` | Elastic net (§4.3; 50 simulation runs) | `--n_runs=`, `--run=all\|plot` |
| `s40_linear_convergence.jl` | R-linear rate verification (§4.4; 20 runs) | `--n_runs=`, `--seed=` |

All scripts perform a JIT warmup before timing and log to both stdout and file (via TeeIO). `s10` and `s20` support resume (CSV-based; `--force` re-runs everything).

## Structure

```
src/
├── includes.jl        # Entry point
├── dependences.jl     # Package imports
├── algorithms.jl      # DIPCM, IPCMAS1/2, DeyHICPP, Suantai2024, DongIPCA, weak-comparison algos
├── functions.jl       # Comparison infrastructure, solve_problem, resume mechanism
├── types.jl           # Problem, Solution types
├── projections.jl     # Projection operators
├── utils.jl           # CSV/XLSX, performance profiles
├── utils2.jl          # Misc helpers
├── io_utils.jl        # TeeIO dual logging
└── examples/          # Original experiment scripts (archived)

scripts/               # Active experiment scripts (s10/s12/s15/s20/s30/s40 + w-prefix legacy)
results/               # Output (gitignored)
archive/               # Historical snapshots
├── dipcm_legacy.jl              # Pre-2026-04-25 DIPCM (constant inertial params)
├── s12_legacy_sensitivity.jl    # Legacy α × β_zn × θ̄ sensitivity
└── algov{1,2}.jl, example*.jl   # Earlier reference snapshots
```

## Requirements

Julia 1.10 (LTS) or newer. Dependencies in `Project.toml`.
