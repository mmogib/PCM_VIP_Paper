## w12_weak_comparison_vip.jl — Example 1 (VIP) weak-convergence comparison
#
# Compares IPCMAS1 against 3 weak-convergence competitors from the literature
# on the monotone variational inclusion problem:
#
#   TanQin2024     — Tan & Qin (2024), Algorithm 3.1
#   ChenMiPCA      — Chen, Zhang, Dong (2020), Algorithm 3.1 (s=2)
#   PeeyadaIMFBSA  — Peeyada, Suparatulatorn, Cholamjiak (2022), Algorithm 3.1
#   IPCMAS1        — our Algorithm 1
#
# Tolerances: {1e-1, 1e-2, 1e-3} — the regime where weak-convergence methods
# are reported in the literature. DIPCM is NOT included here; see s10 for the
# high-precision comparison that features DIPCM.
#
# Per-iter data is saved to results/example_1_weak/history/ for post-hoc
# analysis (convergence curves, performance profiles).
#
# Usage:
#   julia --project=. scripts/w12_weak_comparison_vip.jl
#   julia --project=. scripts/w12_weak_comparison_vip.jl --dim=100,200
#   julia --project=. scripts/w12_weak_comparison_vip.jl --algo=IPCMAS1,TanQin2024
#   julia --project=. scripts/w12_weak_comparison_vip.jl --force

include("../src/includes.jl")

function setup_example1(n::Int; seed=2025, num_of_instances=5)
    rng = Xoshiro(seed)
    U = Uniform(1, 100)
    problems = Vector{Problem}(undef, num_of_instances)
    for i in 1:num_of_instances
        Z = rand(rng, U, n, n)
        B_matrix = Z' * Z
        L = maximum(abs.(eigvals(B_matrix)))
        A_matrix = triu(ones(n, n))
        B(x) = B_matrix * x
        A_resolvent(x, λ) = (I + λ * A_matrix) \ x
        x0 = rand(rng, n)
        x1 = rand(rng, n)
        problems[i] = Problem(; name="Example 1 ($i)", Aλ=A_resolvent,
            A=x -> A_matrix * x, B=B, L=L, x0=x0, x1=x1, n=n)
    end
    return problems
end

ALL_ALGORITHMS = Dict(
    "IPCMAS1"        => ("IPCMAS1",       IPCMAS1,       L -> get_IPCMAS1_params(L; μ0=0.5, λ0=1 / (1.05 * L))),
    "TanQin2024"     => ("TanQin2024",    TanQin2024,    L -> get_TanQin2024_params(L)),
    "ChenMiPCA"      => ("ChenMiPCA",     ChenMiPCA,     L -> get_ChenMiPCA_params(L; λ0=1 / (1.05 * L))),
    "PeeyadaIMFBSA"  => ("PeeyadaIMFBSA", PeeyadaIMFBSA, L -> get_PeeyadaIMFBSA_params(L)),
)

function main()
    opts, _ = parse_args(ARGS)
    algo_filter = get(opts, "algo", "")
    force = any(x -> x == "--force", ARGS)

    title = "example 1 weak"
    default_keys = ["IPCMAS1", "TanQin2024", "ChenMiPCA", "PeeyadaIMFBSA"]

    if !isempty(algo_filter)
        selected_keys = split(algo_filter, ",") .|> strip .|> String
        for k in selected_keys
            haskey(ALL_ALGORITHMS, k) || error("Unknown algorithm '$k'. Available: $(join(keys(ALL_ALGORITHMS), ", "))")
        end
        title *= " ($(algo_filter))"
    else
        selected_keys = default_keys
    end

    algorithms = [ALL_ALGORITHMS[k] for k in selected_keys if haskey(ALL_ALGORITHMS, k)]

    errors = [1e-1, 1e-2, 1e-3]
    dims_str = get(opts, "dim", "100,150,200,250,300")
    dims = parse.(Int, split(dims_str, ","))
    seed = 2025
    num_of_instances = 5
    maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "20000")))
    verbose = any(x -> x in ("--verbose", "-v"), ARGS)
    show_progress = !any(x -> x == "--no-progress", ARGS)
    clearfolder = any(x -> x in ("--clear", "-c"), ARGS)

    logpath, tee, logfile = setup_logging("w12_weak_vip"; logdir="results/example_1_weak")
    println(tee, "="^70)
    println(tee, "EXAMPLE 1 (VIP) — WEAK-CONVERGENCE COMPARISON")
    println(tee, "="^70)
    println(tee, "  Algorithms: ", join(first.(algorithms), ", "))
    println(tee, "  Tolerances: $errors")
    println(tee, "  Dims:       $dims")
    println(tee, "  Instances:  $num_of_instances (seed=$seed)")
    println(tee, "  MaxIter:    $maxiter")
    println(tee, "  save_history=true → per-iter CSVs enabled")
    force && println(tee, "  --force: re-running all configs (overwrites history CSVs too)")
    println(tee, "="^70)
    flush(tee)

    csv_file, solutions = startSolvingExample(title, algorithms, setup_example1, dims;
        errors=errors,
        seed=seed,
        num_of_instances=num_of_instances,
        maxiter=maxiter,
        verbose=verbose,
        show_progress=show_progress,
        clearfolder=clearfolder,
        plotit=true,
        plot_comparizon=false,
        io=tee,
        force=force,
        save_history=true,
    )

    teardown_logging(tee, logpath)
end

main()
