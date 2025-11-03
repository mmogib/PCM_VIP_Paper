**Overview**
- Julia code to experiment with monotone operator/variational inequality algorithms (DeyHICPP, IPCMAS1/2) and small applications (VIP toy problems, split feasibility in L2[0,1], elastic net).
- Contains runnable example scripts that generate CSV/plots into a `results/` folder and small utilities for converting CSV → XLSX and plotting performance profiles.

**Requirements**
- Julia (recommended: a recent 1.x release).
- Project-managed dependencies are listed in `Project.toml`/`Manifest.toml` and are resolved via Julia’s package manager.

**Quick Start**
- Open a terminal in the repository root, then instantiate the environment:
  - `julia --project=. -e "using Pkg; Pkg.instantiate()"`
- Run any example script (writes outputs to `results/`):
  - VIP comparison: `julia --project=. src/examples/example_1_vip.jl`
  - Split feasibility (L2[0,1]): `julia --project=. src/examples/example_2_sfp.jl`
  - Elastic net demo: `julia --project=. src/examples/example_3_elastic.jl`

**Example Options**
- Verbose logs: add `--verbose` or `-v`.
- Hide progress bars (where available): add `--no-progress`.
- Example 1 can clear its output folder before running: add `--clear` or `-c`.

**Outputs**
- CSV tables and figures are written under `results/` (created automatically on first run).
- Example 1 also produces performance profile plots for time and iteration counts.
- Utilities in `src/utils.jl` support converting CSV files to XLSX and combining multiple CSVs into a single workbook.

**Project Layout**
- `src/`
  - `includes.jl` — pulls in dependencies, types, algorithms, utils.
  - `types.jl` — lightweight problem/solution types used across examples.
  - `algorithms.jl` — implementations of DeyHICPP and IPCMAS variants.
  - `utils.jl`, `utils2.jl` — helpers for file handling, CSV→XLSX, performance profiles, etc.
  - `examples/` — runnable example scripts:
    - `example_1_vip.jl` — compares algorithms on a synthetic VIP family and saves tables/plots.
    - `example_2_sfp.jl` — split-feasibility problem posed in L2[0,1].
    - `example_3_elastic.jl` — small elastic-net regression demo.
- `results/` — output directory (kept with `.gitkeep`).

**Notes**
- Run examples with `julia --project=.` to ensure the correct environment.
- If you add new scripts, prefer placing them under `src/examples/` and writing outputs under `results/<example>/`.

