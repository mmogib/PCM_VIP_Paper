# DIPCM — Julia Implementation

## Overview
Julia implementation of the double inertial projection-contraction method (IPCMAS1/IPCMAS2) for solving variational inclusion problems. Uses Style B (Flat Include) architecture.

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
├── scripts/               # Active experiment scripts
│   ├── s10_example1_vip.jl        # VIP example (Section 4.1) — supports --theta-sync
│   ├── s12_sensitivity.jl         # Sensitivity analysis for (α, β, θ)
│   ├── s20_example2_sfp.jl        # Split feasibility problem (Section 4.2)
│   ├── s30_example3_en.jl         # Elastic net problem (Section 4.3)
│   └── s40_linear_convergence.jl  # R-linear convergence verification (Section 4.4)
├── results/               # Output data
├── archive/               # Archived code versions
├── table1_results.csv     # Numerical results for Table 1
└── table2_results.csv     # Numerical results for Table 2
```

## Algorithms Implemented
```
DIPCM            # Our main algorithm — double inertial PCM with implicit contraction (strong convergence)
                 #   x_{n+1} = α z_n + σ_n u_n, σ_n = (1-α) - β_n, contracts toward origin
                 #   Default: α=0.2, θ̄=0.9, β_zn=0.2, β_n=1/(n+1)
IPCMAS1          # Our Algorithm 1 — convex combination (1-α)z_n + α u_n (weak convergence, slow in practice)
IPCMAS2          # Our Algorithm 2 — single inertial PCM, β=0 (R-linear convergence)
DeyHICPP         # Dey (2023) — single inertial Halpern PCM, constant stepsize λ∈(0,1/L)
Suantai2024      # SICIP — Suantai et al. (2024) — combined double inertial PCM, adaptive stepsize
DongIPCA         # Dong et al. (2018) — inertial projection contraction (variational inequalities)
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
- IPCMAS1 code now matches the paper exactly: `(1-α)z_n + α u_n` with constant α (no sequences)
- DIPCM: `α z_n + σ_n u_n` with σ_n = (1-α) - β_n, implicit contraction toward origin
- Adaptive stepsize: λ₁ defaults to 1/(2L) but can be overridden
- Suantai2024 uses non-increasing stepsize: λ_{k+1} = min{μ‖w_k-y_k‖/‖f(w_k)-f(y_k)‖, λ_k}
- TeeIO logging via `src/io_utils.jl` — all scripts log to both stdout and file

## Rules
- **DO NOT run Julia scripts.** Mohammed runs scripts locally. Only create/edit scripts.
- Tests (`runtests.jl`) may be run to verify code changes.

## Status
- DIPCM implemented and sensitivity tested (s12 — all 120 configs converge, 43–205 iters)
- IPCMAS1 fixed to match paper (constant α, no sequences)
- s12 rewritten for DIPCM sensitivity; s13 deleted (absorbed)
- Next: update s10, s20, s30, s40 to use DIPCM replacing IPCMAS1 in comparisons
