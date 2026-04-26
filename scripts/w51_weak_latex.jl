## w51_weak_latex.jl — Batch-generate the three weak-comparison LaTeX tables
#
# Reads the three weak-comparison CSVs produced by w12/w22/w32 and emits one
# LaTeX table per example, saved NEXT TO its source CSV inside jcode/results/.
# Each table uses the combined-tolerance layout (rows = dim × tol blocks,
# columns = algorithms × {iter, time}), with `$-$` for 0/N convergence and
# starred values for partial convergence.
#
# NOTE: jcode/ is self-contained; scripts never write outside jcode/.
# Copy-paste the generated table bodies manually into `paper/main.tex`.
# Do NOT `\input{}` them — the paper repo is kept independent of jcode.
#
# Usage:
#   julia --project=. scripts/w51_weak_latex.jl              # all 3 tables
#   julia --project=. scripts/w51_weak_latex.jl --example 1  # single example
#
# Output:
#   results/example_{1,2,3}_weak/weak_table.tex

include("../src/includes.jl")

const WEAK_JOBS = [
    (
        ex = 1,
        src = "results/example_1_weak/comparison.csv",
        dst = "results/example_1_weak/weak_table.tex",
        label = "tab:weak_ex1",
        caption = "Weak-convergence regime comparison on the variational inclusion problem (Example~1). Iteration count and CPU time (seconds) per (dimension, tolerance); averaged over instances. `\$-\$' marks 0/N convergence; starred values denote partial convergence across instances.",
    ),
    (
        ex = 2,
        src = "results/example_2_weak/comparison.csv",
        dst = "results/example_2_weak/weak_table.tex",
        label = "tab:weak_ex2",
        caption = "Weak-convergence regime comparison on the split feasibility problem in \$L^{2}[0,1]\$ (Example~2). Iteration count and CPU time (seconds) per (dimension, tolerance); averaged over instances. `\$-\$' marks 0/N convergence; starred values denote partial convergence.",
    ),
    (
        ex = 3,
        src = "results/example_3_weak/comparison.csv",
        dst = "results/example_3_weak/weak_table.tex",
        label = "tab:weak_ex3",
        caption = "Weak-convergence regime comparison on the elastic net problem (Example~3). Iteration count and CPU time (seconds) per (dimension, tolerance); averaged over instances.",
    ),
]

function main()
    opts, _ = parse_args(ARGS)
    example_filter = get(opts, "example", "")
    selected = isempty(example_filter) ? nothing : parse(Int, example_filter)

    println("="^70)
    println("Generating weak-comparison LaTeX tables")
    println("="^70)

    for job in WEAK_JOBS
        if selected !== nothing && job.ex != selected
            continue
        end
        if !isfile(job.src)
            println("  SKIP ex$(job.ex) — $(job.src) not found")
            continue
        end
        mkpath(dirname(job.dst))
        println("  ex$(job.ex):  $(job.src)")
        println("         → $(job.dst)")
        run(`$(Base.julia_cmd()) --project=. scripts/s15_csv_to_latex.jl
             --csv=$(job.src) --format=weak --output=$(job.dst)
             --label=$(job.label) --caption=$(job.caption)`)
    end
    println("Done.")
end

main()
