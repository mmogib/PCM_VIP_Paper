## w32_weak_comparison_en.jl — Example 3 (Elastic Net) weak-convergence comparison
#
# Compares IPCMAS1 against 3 weak-convergence competitors on the elastic net
# problem:  min ½‖Xw − y‖² + (λ₁/(1+λ₂))‖w‖₁
#
# Unlike s30 (which performs cross-validation + hyperparameter search), this
# script fixes (λ₁, λ₂) and just benchmarks the solvers on multiple data
# instances. The hyperparameter choice is NOT claimed optimal — only fixed so
# we can compare iteration counts / times across algorithms at matched tols.
#
# Algorithms:
#   IPCMAS1, TanQin2024, ChenMiPCA, PeeyadaIMFBSA
#
# Notes on assumptions:
#   F(w) = X'(Xw − y) is the gradient of a convex quadratic with L-Lipschitz
#   gradient where L = λ_max(X'X). By Baillon–Haddad, F is (1/L)-cocoercive,
#   so Peeyada's assumption is satisfied. Chen and Tan-Qin require monotone +
#   Lipschitz, also satisfied.
#
# Tolerances: {1e-1, 1e-2, 1e-3}. Per-iter data to results/example_3_weak/history/.
#
# Usage:
#   julia --project=. scripts/w32_weak_comparison_en.jl
#   julia --project=. scripts/w32_weak_comparison_en.jl --dim=8    # n_features
#   julia --project=. scripts/w32_weak_comparison_en.jl --algo=IPCMAS1,TanQin2024

include("../src/includes.jl")

# ── Elastic net problem generator ──────────────────────────────────────────
# Generates EN data with n_features and varying seed; returns a Problem with
# the correct resolvent (soft-threshold) and gradient operators.
function setup_example3(n_features::Int; seed=2025, num_of_instances=5,
    n_train=20, n_val=20, n_test=200,
    σ=3.0, ρ=0.5, λ₁=1.0, λ₂=0.5)

    problems = Vector{Problem}(undef, num_of_instances)
    for i in 1:num_of_instances
        rng = Xoshiro(seed + i - 1)

        # Default ground-truth coefficients (paper's choice for n=8)
        w_true = n_features == 8 ?
            [3.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0] :
            (v = zeros(n_features); v[1:min(3, n_features)] .= [3.0, 1.5, 2.0][1:min(3, n_features)]; v)

        # Correlated design matrix Σ_ij = ρ^|i-j|
        Σ = [ρ^abs(i - j) for i in 1:n_features, j in 1:n_features]
        Lchol = cholesky(Σ).L
        n_total = n_train + n_val + n_test
        Z = randn(rng, n_total, n_features)
        X = Z * Lchol'
        y = X * w_true + σ * randn(rng, n_total)

        # Use train+val for the solve (common practice when skipping CV)
        X_fit = X[1:n_train+n_val, :]
        y_fit = y[1:n_train+n_val]

        λ₁_scaled = λ₁ / (1 + λ₂)
        Aλ(w, λ) = sign.(w) .* max.(abs.(w) .- λ * λ₁_scaled, 0)
        B(w) = X_fit' * (X_fit * w - y_fit)
        A_id(w) = w
        XtX = X_fit' * X_fit
        L = maximum(abs.(eigvals(XtX)))

        x0 = zeros(n_features)
        x1 = zeros(n_features)

        problems[i] = Problem(;
            name="EN_inst$(i)_λ₁=$(λ₁)_λ₂=$(λ₂)",
            Aλ=Aλ, A=A_id, B=B, L=L,
            x0=x0, x1=x1, n=n_features,
        )
    end
    return problems
end

# IPCMAS1 params tuned by w11 sensitivity: α=0.30, β=0.0119, θ̄=0.9, ξ_n=100/(n+1)^1.1.
function get_IPCMAS1_params_EN(L::Float64; γ=1.1, μ0=0.5, α0=0.30, β0=0.0119, λ0=nothing)
    a_seq(n) = 100 / (n + 1)^(1.1)
    θ_seq(n) = 0.9
    return (
        γ=γ, μ=μ0,
        λ1=isnothing(λ0) ? 1.0 / (2 * L) : λ0,
        β=β0, α=α0, a_seq=a_seq, θ_seq=θ_seq,
    )
end

function main()
    opts, _ = parse_args(ARGS)
    force = any(x -> x == "--force", ARGS)

    title = "example 3 weak"

    ALL_ALGORITHMS = Dict(
        "IPCMAS1"        => ("IPCMAS1",       IPCMAS1,       L -> get_IPCMAS1_params_EN(L; λ0=1 / (1.05 * L))),
        "TanQin2024"     => ("TanQin2024",    TanQin2024,    L -> get_TanQin2024_params(L)),
        "ChenMiPCA"      => ("ChenMiPCA",     ChenMiPCA,     L -> get_ChenMiPCA_params(L; λ0=1 / (1.05 * L))),
        "PeeyadaIMFBSA"  => ("PeeyadaIMFBSA", PeeyadaIMFBSA, L -> get_PeeyadaIMFBSA_params(L)),
    )
    default_keys = ["IPCMAS1", "TanQin2024", "ChenMiPCA", "PeeyadaIMFBSA"]

    algo_filter = get(opts, "algo", "")
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
    dims_str = get(opts, "dim", "8")
    dims = parse.(Int, split(dims_str, ","))
    maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "20000")))
    seed = 2025
    num_of_instances = 5
    verbose = any(x -> x in ("--verbose", "-v"), ARGS)
    show_progress = !any(x -> x == "--no-progress", ARGS)
    clearfolder = any(x -> x in ("--clear", "-c"), ARGS)

    logpath, tee, logfile = setup_logging("w32_weak_en"; logdir="results/example_3_weak")
    println(tee, "="^70)
    println(tee, "EXAMPLE 3 (Elastic Net) — WEAK-CONVERGENCE COMPARISON")
    println(tee, "="^70)
    println(tee, "  Algorithms:   ", join(first.(algorithms), ", "))
    println(tee, "  Tolerances:   $errors")
    println(tee, "  n_features:   $dims")
    println(tee, "  Instances:    $num_of_instances (seed=$seed)")
    println(tee, "  Fixed hyperparams: λ₁=1.0, λ₂=0.5  (no CV in this script)")
    println(tee, "  MaxIter:      $maxiter")
    println(tee, "  save_history=true → per-iter CSVs enabled")
    force && println(tee, "  --force: re-running all configs")
    println(tee, "="^70)
    flush(tee)

    csv_file, solutions = startSolvingExample(title, algorithms, setup_example3, dims;
        errors=errors,
        seed=seed,
        num_of_instances=num_of_instances,
        maxiter=maxiter,
        verbose=verbose,
        show_progress=show_progress,
        clearfolder=clearfolder,
        plotit=false,
        plot_comparizon=false,
        io=tee,
        force=force,
        save_history=true,
    )

    teardown_logging(tee, logpath)
end

main()
