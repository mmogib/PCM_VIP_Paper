# DIPCM — Double Inertial Projection-Contraction Method

Julia implementation for the paper:  
**"Projection and contraction method with double inertial steps for variational inclusion problems on Hilbert spaces"**

## Algorithms

| Function | Paper | Update Rule | Convergence |
|----------|-------|-------------|-------------|
| `DIPCM` | Algorithm 3 (Section 3.4) | `x_{n+1} = α z_n + σ_n u_n` | Strong (to P_Ω(0)) |
| `IPCMAS1` | Algorithm 1 | `x_{n+1} = (1-α) z_n + α u_n` | Weak |
| `IPCMAS2` | Algorithm 2 | `x_{n+1} = (1-α) x_n + α u_n` | R-linear (strong monotonicity) |
| `DeyHICPP` | Dey (2023) | `x_{n+1} = (1-θ_n-β_n) x_n + θ_n z_n` | Strong |
| `Suantai2024` | Suantai et al. (2024) | `x_{n+1} = w_n - γ β_n d_n` | Weak |

**DIPCM** is the main algorithm. Default parameters: `α=0.2, θ̄=0.9, β_zn=0.2, β_n=1/(n+1)`.

## Quick Start

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Sensitivity analysis (DIPCM parameters)
julia --project=. scripts/s12_sensitivity.jl

# Example 1: VIP comparison (DIPCM vs DeyHICPP vs SICIP vs IPCMAS2)
julia --project=. scripts/s10_example1_vip.jl

# Example 2: Split feasibility problem
julia --project=. scripts/s20_example2_sfp.jl

# Example 3: Elastic net regularization
julia --project=. scripts/s30_example3_en.jl

# R-linear convergence verification
julia --project=. scripts/s40_linear_convergence.jl
```

## Scripts

| Script | Purpose | Flags |
|--------|---------|-------|
| `s10_example1_vip.jl` | VIP comparison (5 dims, 6 tolerances, 5 instances) | `--dim=`, `--algo=`, `--force`, `--verbose` |
| `s12_sensitivity.jl` | DIPCM parameter sensitivity (α, θ̄, β_zn) | `--dim=`, `--maxiter=` |
| `s20_example2_sfp.jl` | SFP in L²[0,1] (8 instances) | `--dim=`, `--algo=`, `--force` |
| `s30_example3_en.jl` | Elastic net (50 simulation runs) | `--n_runs=`, `--run=all\|plot` |
| `s40_linear_convergence.jl` | R-linear rate verification (20 runs) | `--n_runs=`, `--seed=` |

All scripts log to both stdout and file (via TeeIO). Scripts s10/s20 support resume (`--force` to override).

## Structure

```
src/
├── includes.jl        # Entry point
├── io_utils.jl        # TeeIO dual logging
├── algorithms.jl      # DIPCM, IPCMAS1/2, DeyHICPP, Suantai2024, DongIPCA
├── functions.jl       # Comparison infrastructure, solve_problem, resume
├── types.jl           # Problem, Solution types
├── utils.jl           # CSV/XLSX, performance profiles
└── projections.jl     # Projection operators
scripts/               # Active experiment scripts
src/examples/          # Archived original scripts
results/               # Output (gitignored)
```

## Requirements

Julia 1.12+. Dependencies in `Project.toml`.
