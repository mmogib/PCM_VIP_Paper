# Projection and Contraction Method with Double Inertial Steps for Monotone Variational Inclusion Problems

## Overview

This codebase reproduces the numerical results from the paper:  
**"Projection and Contraction Method with Double Inertial Steps for Monotone Variational Inclusion Problems on Hilbert Spaces"**

It implements and compares the following algorithms:
- **DeyHICPP** — Dey's Hybrid Inertial Contractive Projection method
- **IPCMAS1** — Inertial Projection Contraction Method with Adaptive Stepsizes (first variant)
- **IPCMAS2** — Inertial Projection Contraction Method with Adaptive Stepsizes (second variant)

Across four key experiments demonstrating the methods' effectiveness on different problem classes.

## Requirements

- **Julia** (recommended: v1.10 or later)
- Project dependencies listed in `Project.toml` and `Manifest.toml`

## Quick Start

### 1. Setup Environment
Open a terminal in the repository root and instantiate Julia's project environment:
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### 2. Run Experiments
Each experiment script is located in `src/examples/` and can be run with optional command-line arguments:

```bash
# Experiment 1: Variational Inclusion Problem (VIP)
julia --project=. src/examples/example_1_vip.jl [options]

# Experiment 2: Split Feasibility Problem (SFP) in L2[0,1]
julia --project=. src/examples/example_2_sfp.jl [options]

# Experiment 3: Elastic Net Regularization
julia --project=. src/examples/example_3_en.jl [options]

# Experiment 4: Verification of R-Linear Convergence
julia --project=. src/examples/linear_convergence.jl [options]
```

---

## Experiments

### Experiment 1: Variational Inclusion Problem (VIP)
**File:** `src/examples/example_1_vip.jl`

Compares algorithms on a synthetic VIP family with varying problem dimensions and tolerance levels.

**Key Parameters:**
- `--maxiter` or `--itr` (default: 50000) — Maximum number of iterations

**Command-line Options:**
- `--verbose` or `-v` — Print detailed logs
- `--no-progress` — Hide progress bars
- `--clear` or `-c` — Clear output folder before running

**Output:**
- Tables and convergence plots saved to `results/example_1/`
- Performance profile plots for iteration counts and computation time

**Example Usage:**
```bash
julia --project=. src/examples/example_1_vip.jl --maxiter 100000 --verbose
```

---

### Experiment 2: Split Feasibility Problem (SFP) in L2[0,1]
**File:** `src/examples/example_2_sfp.jl`

Solves split feasibility problems posed in the Hilbert space L²[0,1] with multiple instances.

**Key Parameters:**
- `--maxiter` or `--itr` (default: 50000) — Maximum number of iterations

**Command-line Options:**
- `--verbose` or `-v` — Print detailed logs
- `--no-progress` — Hide progress bars
- `--clear` or `-c` — Clear output folder before running

**Output:**
- Convergence plots and error histories saved to `results/example_2/`
- Comparison tables for different problem instances

**Example Usage:**
```bash
julia --project=. src/examples/example_2_sfp.jl --clear --verbose
```

---

### Experiment 3: Elastic Net Regularization
**File:** `src/examples/example_3_en.jl`

Full simulation study comparing algorithms on elastic net regularization with hyperparameter tuning via cross-validation.

**Key Parameters:**
- `--n_runs` (default: 50) — Number of simulation runs
- `--n_train` (default: 20) — Training set size
- `--n_val` (default: 20) — Validation set size
- `--n_test` (default: 200) — Test set size
- `--n_features` (default: 8) — Number of features
- `--sigma` (default: 3.0) — Noise standard deviation (σ)
- `--rho` (default: 0.5) — Correlation parameter (ρ)
- `--n_folds` (default: 10) — Number of cross-validation folds
- `--tol` (default: 1e-6) — Convergence tolerance
- `--maxiter` (default: 10000) — Maximum iterations per solve
- `--seed` (default: 2025) — Random seed
- `--run` (default: "all") — Choose "runonly" to skip plotting, or "plot" to plot only

**Command-line Options:**
- `--clear` or `-c` — Clear output folder before running

**Output:**
- Aggregated results (mean, std, median) in CSV and XLSX formats saved to `results/example_3/`
- 8 publication-quality plots:
  1. Test MSE comparison
  2. Coefficient error comparison
  3. Variable selection metrics (precision, recall, F1)
  4. Computational timing comparison
  5. Combined 2×2 grid summary
  6. Radar chart for multi-metric comparison
  7. Accuracy vs. computational cost scatter plot
  8. Relative performance improvements

**Example Usage:**
```bash
# Run with default parameters
julia --project=. src/examples/example_3_en.jl

# Run with custom parameters and clear previous results
julia --project=. src/examples/example_3_en.jl --n_runs 100 --n_train 50 --clear

# Generate plots only (from existing results)
julia --project=. src/examples/example_3_en.jl --run plot
```

---

### Experiment 4: Verification of R-Linear Convergence
**File:** `src/examples/linear_convergence.jl`

Empirically verifies the R-linear convergence rate on elastic net problems under strong convexity.

**Output:**
- Convergence rate analysis saved to `results/examples/linear_convergence/`
- Plots showing error decay and linear convergence verification

**Example Usage:**
```bash
julia --project=. src/examples/linear_convergence.jl
```

---

## Project Structure

```
.
├── README.md                          # Original documentation
├── Project.toml                       # Julia project dependencies
├── Manifest.toml                      # Locked dependency versions
├── src/
│   ├── includes.jl                    # Main include file (imports all modules)
│   ├── dependences.jl                 # External package imports
│   ├── types.jl                       # Problem and Solution types
│   ├── algorithms.jl                  # Algorithm implementations (DeyHICPP, IPCMAS1/2)
│   ├── projections.jl                 # Projection operators
│   ├── functions.jl                   # Core optimization functions
│   ├── utils.jl                       # File I/O, CSV/XLSX conversion, performance profiles
│   ├── utils2.jl                      # Additional utilities
│   └── examples/
│       ├── example_1_vip.jl           # Experiment 1: Variational Inclusion Problem
│       ├── example_2_sfp.jl           # Experiment 2: Split Feasibility Problem
│       ├── example_3_en.jl            # Experiment 3: Elastic Net (NEW simulation format)
│       ├── example_3_elastic.jl       # Experiment 3: Elastic Net (legacy)
│       └── linear_convergence.jl      # Experiment 4: Linear Convergence Verification
├── results/                           # Output directory (created automatically)
│   ├── example_1/                     # Experiment 1 outputs
│   ├── example_2/                     # Experiment 2 outputs
│   ├── example_3/                     # Experiment 3 outputs (CSV, XLSX, plots)
│   └── examples/
│       └── linear_convergence/        # Experiment 4 outputs
└── archive/                           # Legacy and archived example files
```

## Output Locations

All results are saved to the `results/` folder, organized by experiment:

| Experiment | Output Location | File Types |
|-----------|-----------------|-----------|
| 1 (VIP) | `results/example_1/` | CSV tables, PNG plots |
| 2 (SFP) | `results/example_2/` | CSV tables, convergence plots (PNG) |
| 3 (Elastic Net) | `results/example_3/` | CSV, XLSX, 8 PNG plots |
| 4 (Linear Convergence) | `results/examples/linear_convergence/` | CSV, plots |

Each run includes a timestamp in the filename to avoid overwriting previous results.

## Command-Line Options Summary

| Option | Alias | Usage | Purpose |
|--------|-------|-------|---------|
| `--verbose` | `-v` | Any experiment | Print detailed algorithm logs |
| `--no-progress` | — | Exp 1–2 | Hide progress bars |
| `--clear` | `-c` | Any experiment | Clear output folder before running |
| `--maxiter` | `--itr` | Exp 1–2 | Set maximum iterations |
| `--n_runs` | — | Exp 3 | Number of simulation runs |
| `--n_train` | — | Exp 3 | Training set size |
| `--sigma` | — | Exp 3 | Noise standard deviation |
| `--seed` | — | Exp 3 | Random seed for reproducibility |
| `--run` | — | Exp 3 | "runonly", "plot", or "all" |

## Notes for Researchers

1. **First Run**: The first execution may take longer as Julia compiles code. Subsequent runs are faster.

2. **Reproducibility**: Use the same `--seed` value to reproduce exact results.

3. **Custom Parameters**: Modify command-line arguments to test different problem sizes, tolerances, or regularization parameters.

4. **Output Inspection**: 
   - Open CSV/XLSX files in Excel or similar tools for tabular results
   - View PNG plots with any image viewer
   - For detailed logs, use the `--verbose` flag when running experiments

5. **Troubleshooting**:
   - If results don't update, use `--clear` to remove previous outputs
   - On Windows, use forward slashes (/) in file paths within the code
   - Ensure you have write permissions to the `results/` directory

## Key Features

✓ Comprehensive algorithm comparison across diverse problems  
✓ Automatic CSV and XLSX output generation  
✓ Publication-quality visualization plots  
✓ Cross-validation for hyperparameter tuning (Exp 3)  
✓ Convergence analysis and performance profiling  
✓ Configurable parameters via command-line arguments  
✓ Reproducible results with seeded random number generation  

## Additional Utilities

The `src/utils.jl` file provides helper functions:
- `CSV.read()` / `CSV.write()` — Read/write CSV files
- `prepare_filepath()` — Generate timestamped filenames
- `clear_folder_recursive()` — Clean output directories
- Performance profile generation (for Experiment 1)
- CSV to XLSX conversion utilities
