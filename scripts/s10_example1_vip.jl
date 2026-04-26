## s10_example1_vip.jl — Example 1: Variational Inclusion Problem comparison
#
# Compares DIPCM, DeyHICPP, SICIP, IPCMAS2 on VIP with x* = 0.
# Supports resume (--force to override), TeeIO logging, performance profiles.
#
# Usage:
#   julia --project=. scripts/s10_example1_vip.jl
#   julia --project=. scripts/s10_example1_vip.jl --dim=100,200
#   julia --project=. scripts/s10_example1_vip.jl --algo=DIPCM,DeyHICPP
#   julia --project=. scripts/s10_example1_vip.jl --force        # re-run all
#
# Output:
#   results/example_1/comparison.csv + .xlsx
#   results/example_1/log_s10_*.txt

include("../src/includes.jl")

function setup_example1(n::Int; seed=2025, num_of_instances=1)
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
        problems[i] = Problem(; name="Example 1 ($i)", Aλ=A_resolvent, A=x -> A_matrix * x, B=B, L=L, x0=x0, x1=x1, n=n)
    end
    return problems
end

ALL_ALGORITHMS = Dict(
    "DeyHICPP"       => ("DeyHICPP", DeyHICPP, L -> get_DeyHICPP_params(L; λ0=1 / (1.05 * L))),
    "SICIP"          => ("SICIP", Suantai2024, L -> get_Suantai2024_params(L)),
    "IPCMAS1"        => ("IPCMAS1", IPCMAS1, L -> get_IPCMAS1_params(L; μ0=0.5, λ0=1 / (1.05 * L))),
    "IPCMAS2"        => ("IPCMAS2", IPCMAS2, L -> get_IPCMAS2_params(L; γ=1.1, λ0=1 / (1.05 * L))),
    "DIPCM"          => ("DIPCM", DIPCM, L -> get_DIPCM_params(L; β_bar=0.3, θ_bar=0.9, λ0=1 / (1.05 * L))),
    "TanQin2024"     => ("TanQin2024", TanQin2024, L -> get_TanQin2024_params(L)),
    "ChenMiPCA"      => ("ChenMiPCA", ChenMiPCA, L -> get_ChenMiPCA_params(L; λ0=1 / (1.05 * L))),
    "PeeyadaIMFBSA"  => ("PeeyadaIMFBSA", PeeyadaIMFBSA, L -> get_PeeyadaIMFBSA_params(L)),
)

"""
JIT warmup: compile algorithm specializations on a tiny problem before the timing loop.
Without this, the first call to each algorithm in the main loop pays full method-compilation
cost and inflates the Instance-1 time entry of the comparison CSV.
"""
function _jit_warmup()
    probs = setup_example1(10; seed = 1, num_of_instances = 1)
    prob = probs[1]
    L = prob.L
    DIPCM(prob; get_DIPCM_params(L; β_bar = 0.3, θ_bar = 0.9, λ0 = 1 / (1.05 * L))...,
          tol = 1e-2, maxiter = 1000)
    DeyHICPP(prob; get_DeyHICPP_params(L; λ0 = 1 / (1.05 * L))...,
             tol = 1e-2, maxiter = 1000)
    Suantai2024(prob; get_Suantai2024_params(L)...,
                tol = 1e-2, maxiter = 1000)
    IPCMAS2(prob; get_IPCMAS2_params(L; γ = 1.1, λ0 = 1 / (1.05 * L))...,
            tol = 1e-2, maxiter = 1000)
    return nothing
end

function main()
    _jit_warmup()
    opts, pos = parse_args(ARGS)
    algo_filter = get(opts, "algo", "")
    force = any(x -> x == "--force", ARGS)

    title = "example 1"
    default_keys = ["DeyHICPP", "SICIP", "DIPCM", "IPCMAS2"]

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

    errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    dims_str = get(opts, "dim", "100,150,200,250,300")
    dims = parse.(Int, split(dims_str, ","))
    seed = 2025
    num_of_instances = 5
    maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "50000")))
    verbose = any(x -> x in ("--verbose", "-v"), ARGS)
    show_progress = !any(x -> x == "--no-progress", ARGS)
    clearfolder = any(x -> x in ("--clear", "-c"), ARGS)

    logpath, tee, logfile = setup_logging("s10_example1"; logdir="results/example_1")
    println(tee, "Algorithms: ", join(first.(algorithms), ", "))
    force && println(tee, "  --force: re-running all configs")
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
    )
    teardown_logging(tee, logpath)
end

main()
