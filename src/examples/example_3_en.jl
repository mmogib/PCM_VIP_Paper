# ============================================================================
# ELASTIC NET SIMULATION STUDY
# ============================================================================
# 
# Logic & Steps:
# 1. Parameter Setup - Configure algorithm parameters for DeyHICPP and IPCMAS1
# 2. Data Generation - Create synthetic correlated elastic net data (train/val/test)
# 3. Elastic Net Solver - Solve: min ½||Xw-y||² + (λ₁/(1+λ₂))||w||₁
# 4. Cross-Validation - k-fold CV to evaluate hyperparameter pairs (λ₁, λ₂)
# 5. Hyperparameter Selection - Grid search to find optimal (λ₁, λ₂)
# 6. Model Evaluation - Compute metrics: test MSE, coef error, precision, recall, F1
# 7. Single Simulation - One run: generate data → select hyperparams → train → evaluate
# 8. Simulation Study - Multiple runs across algorithms, tracking all metrics
# 9. Summarize Results - Compute mean/std/median statistics across runs
# 10. Plotting - Generate 8 comparison plots (MSE, coefficients, selection, timing, etc.)
# 11. Main Execution - Parse CLI args and run simulations or plotting
#
# Output: CSV/XLSX files with results and visualization plots for algorithm comparison
# ============================================================================

# ============================================================================
# TRAINING & CROSS-VALIDATION PROCESS
# ============================================================================
#
# Cross-Validation (cross_validate):
#   - Performs k-fold CV on training data for hyperparameters (λ₁, λ₂)
#   - Splits n samples into k folds with shuffled indices
#   - For each fold: holds out validation fold, trains on k-1 folds, computes validation MSE
#   - Returns average CV error across all folds
#
# Hyperparameter Selection (select_hyperparameters):
#   - Grid search over hyperparameter space
#   - Default: λ₁ ∈ [0.0, 0.01, 0.1, 1.0, 10.0, 100.0], λ₂ ∈ [0.01, 0.1, 1.0, 10.0, 100.0]
#   - Tests all λ₁ × λ₂ combinations (30 pairs)
#   - Calls cross_validate() for each pair to get CV error
#   - Selects pair with lowest CV error
#
# Full Training Cycle (run_single_simulation):
#   1. Generate data → create train/val/test split
#   2. Hyperparameter tuning → find best (λ₁, λ₂) via CV
#   3. Final training → retrain on full training set with best hyperparams
#   4. Test evaluation → compute all metrics (MSE, coef error, precision, recall, F1)
#
# Elastic Net Solver (solve_elastic_net):
#   - Solves: min ½||Xw-y||² + (λ₁/(1+λ₂))||w||₁
#   - Computes Lipschitz constant L from X'X
#   - Calls algorithm (DeyHICPP or IPCMAS1) with convergence tolerance & max iterations
#   - Returns optimal coefficient vector w
#
# ============================================================================

include("../includes.jl")


function get_DeyHICPP_params_EN(L::Float64)
    λ_constant = 0.01  # As specified in the paper (page 21)
    β_seq = n -> 1.0 / sqrt(n + 1)
    return (
        γ=0.01,  # As specified for Example 2
        λ_seq=n -> λ_constant,
        α=0.5,
        τ_seq=n -> 1.0 / n^2,  # Modified for Example 2
        β_seq,
        θ_seq=n -> 0.8 - β_seq(n),
    )
end


function get_IPCMAS1_params_EN(L::Float64; γ=1.1, μ0=0.5, α0=0.5, β0=0.3, λ0=0.5)
    μ = μ0
    λ1 = isnothing(λ0) ? 1.0 / (2 * L) : λ0 #  # Constant step size
    α_fixed = α0
    # rng = Xoshiro(2025)
    # U = Uniform(0, (1 - 3α_fixed) / (3 * (1 - α_fixed)))

    β = β0 #rand(rng, U)
    aseq() = begin
        prefix = Float64[0.0]   # prefix[k+1] stores sum_{i=1}^k 1/i^2
        function (n::Int)
            n ≥ 1 || throw(ArgumentError("n must be ≥ 0"))
            while length(prefix) - 1 < n
                k = length(prefix)          # next i to add
                push!(prefix, prefix[end] + 1.0 / (k^2))
            end
            return prefix[n+1]
        end
    end

    β_seq(n) = 1.0 / (5 * n + 1)
    α_seq(n) = 0.8 - β_seq(n)
    # α_seq(n) = α_fixed #  0.8 - β_seq(n)
    a_seq(n) = 100 / (n)^(2) #aseq()
    # ι_seq = n -> 1.0 / n^2
    θ_seq(n) = 0.9

    return (
        γ=γ,
        μ=μ,
        λ1=λ1,
        β=β,
        α=α_fixed,
        β_seq=β_seq,
        α_seq=α_seq,
        a_seq=a_seq,
        θ_seq=θ_seq,
    )
end


"""
    save_simulation_results(summary, output_dir="results";
                           base_filename="simulation_results",
                           n_runs=50, n_train=20, n_val=20, n_test=200,
                           n_features=8, σ=3.0, ρ=0.5, n_folds=10,
                           tol=1e-6, maxiter=10000, seed=123)

Save simulation study results to CSV and XLSX files.

# Arguments
- `summary`: Dictionary from summarize_results()
- `output_dir`: Directory to save files (default: "results")
- `base_filename`: Base name for output files (default: "simulation_results")
- Simulation parameters: n_runs, n_train, n_val, n_test, n_features, σ, ρ, n_folds, tol, maxiter, seed

# Returns
Named tuple with paths to CSV and XLSX files
"""
function save_simulation_results(summary::Dict, output_dir::String="results/example3";
    base_filename::String="simulation_results",
    n_runs::Int=50,
    n_train::Int=20,
    n_val::Int=20,
    n_test::Int=200,
    n_features::Int=8,
    σ::Float64=3.0,
    ρ::Float64=0.5,
    n_folds::Int=10,
    tol::Float64=1e-6,
    maxiter::Int=10000,
    seed::Int=123,
    clearfolder::Bool=true
)

    # Prepare output directory
    if clearfolder
        clear_folder_recursive("results/example3"; clearSubfolders=false)
    end
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    # Create DataFrame for results
    rows = []

    for (alg_name, stats) in summary
        row = (
            Algorithm=alg_name,

            # Test metrics
            TestMSE_mean=stats.test_mse.mean,
            TestMSE_std=stats.test_mse.std,
            TestMSE_median=stats.test_mse.median,

            # Coefficient error
            CoefError_mean=stats.coef_error.mean,
            CoefError_std=stats.coef_error.std,
            CoefError_median=stats.coef_error.median,

            # Relative coefficient error
            RelCoefError_mean=stats.coef_error_relative.mean,
            RelCoefError_std=stats.coef_error_relative.std,
            RelCoefError_median=stats.coef_error_relative.median,

            # Variable selection metrics
            Precision_mean=stats.precision.mean,
            Precision_std=stats.precision.std,
            Precision_median=stats.precision.median, Recall_mean=stats.recall.mean,
            Recall_std=stats.recall.std,
            Recall_median=stats.recall.median, F1Score_mean=stats.f1_score.mean,
            F1Score_std=stats.f1_score.std,
            F1Score_median=stats.f1_score.median,

            # Timing
            CVTime_mean=stats.cv_time.mean,
            CVTime_std=stats.cv_time.std,
            CVTime_total=stats.cv_time.total, TrainTime_mean=stats.train_time.mean,
            TrainTime_std=stats.train_time.std,
            TrainTime_total=stats.train_time.total,

            # Simulation parameters
            n_runs=n_runs,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_features=n_features,
            sigma=σ,
            rho=ρ,
            n_folds=n_folds,
            tolerance=tol,
            max_iterations=maxiter,
            random_seed=seed
        )
        push!(rows, row)
    end

    # Convert to DataFrame
    df = DataFrame(rows)

    # Prepare file paths with timestamp
    csv_path = prepare_filepath(joinpath(output_dir, base_filename * ".csv"); dated=true)
    xlsx_path = prepare_filepath(joinpath(output_dir, base_filename * ".xlsx"); dated=true)

    # Save as CSV
    CSV.write(csv_path, df)
    println("Saved CSV: $csv_path")

    # Save as XLSX
    XLSX.writetable(xlsx_path, "Results" => df, overwrite=true)
    println("Saved XLSX: $xlsx_path")

    return (csv=csv_path, xlsx=xlsx_path, dataframe=df)
end

# ============================================================================
# 1. DATA GENERATION
# ============================================================================

"""
    generate_elastic_net_data(n_train, n_val, n_test, n_features; 
                              w_true=nothing, σ=3.0, ρ=0.5, rng=Random.GLOBAL_RNG)

Generate correlated data for elastic net simulation.
"""
function generate_elastic_net_data(n_train, n_val, n_test, n_features;
    w_true=nothing, σ=3.0, ρ=0.5,
    rng=Random.GLOBAL_RNG)

    # Set default coefficients if not provided
    if w_true === nothing
        if n_features == 8
            w_true = [3.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        else
            error("Please provide w_true for n_features != 8")
        end
    end

    # Create correlation matrix: corr(i,j) = ρ^|i-j|
    Σ = [ρ^abs(i - j) for i in 1:n_features, j in 1:n_features]

    # Cholesky decomposition for generating correlated data
    L = cholesky(Σ).L

    # Total number of observations
    n_total = n_train + n_val + n_test

    # Generate design matrix with correlation structure
    Z = randn(rng, n_total, n_features)
    X = Z * L'

    # Generate response with noise
    y = X * w_true + σ * randn(rng, n_total)

    # Split into train/validation/test
    X_train = X[1:n_train, :]
    y_train = y[1:n_train]

    X_val = X[n_train+1:n_train+n_val, :]
    y_val = y[n_train+1:n_train+n_val]

    X_test = X[n_train+n_val+1:end, :]
    y_test = y[n_train+n_val+1:end]

    return (
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        w_true=w_true,
        Σ=Σ
    )
end

# ============================================================================
# 2. ELASTIC NET SOLVER
# ============================================================================

"""
    solve_elastic_net(X, y, λ₁, λ₂, algorithm, params_function; 
                     tol=1e-6, maxiter=10000, verbose=false)

Solve elastic net problem using specified algorithm.
Formulation: min (1/2)||Xw - y||² + (λ₁/(1+λ₂))||w||₁

# Arguments
- `X`: Design matrix (n_samples × n_features)
- `y`: Response vector (n_samples)
- `λ₁`: L1 regularization parameter
- `λ₂`: Elastic net mixing parameter
- `algorithm`: Algorithm function (IPCMAS1 or DeyHICPP)
- `params_function`: Function that takes L and returns NamedTuple of parameters

# Keyword Arguments
- `tol`: Convergence tolerance (default: 1e-6)
- `maxiter`: Maximum iterations (default: 10000)
- `verbose`: Print progress information (default: false)
"""
function solve_elastic_net(X::Matrix{Float64}, y::Vector{Float64},
    λ₁::Float64, λ₂::Float64,
    algorithm, params_function;
    tol=1e-6, maxiter=10000, verbose=false)

    n_samples, n_features = size(X)

    # Scaled L1 parameter as per paper: λ₁/(1+λ₂)
    λ₁_scaled = λ₁ / (1 + λ₂)

    # Proximal operator for L1 norm (soft-thresholding)
    Aλ(w, λ) = sign.(w) .* max.(abs.(w) .- λ * λ₁_scaled, 0)

    # Gradient of smooth part: (1/2)||Xw - y||²
    B(w) = X' * (X * w - y)

    # Identity operator
    A(w) = w

    # Lipschitz constant of B: largest eigenvalue of X'X
    XtX = X' * X
    L = maximum(sum(XtX, dims=1))  # Fast approximation (column sum norm)

    # Initial points
    x0 = zeros(n_features)
    x1 = zeros(n_features)

    # Create problem
    problem = Problem(
        name="ElasticNet_λ₁=$(λ₁)_λ₂=$(λ₂)",
        Aλ=Aλ,
        A=A,
        B=B,
        L=L,
        x0=x0,
        x1=x1,
        n=n_features
    )

    # Get algorithm parameters using the params_function with computed L
    algorithm_params = params_function(L)

    # Solve using your solve_problem function
    solution = solve_problem(algorithm, problem, algorithm_params;
        tol=tol, maxiter=maxiter, verbose=verbose)

    return solution.solution
end

# ============================================================================
# 3. CROSS-VALIDATION
# ============================================================================

"""
    cross_validate(X, y, λ₁, λ₂, algorithm, params_function; 
                  n_folds=10, tol=1e-6, maxiter=10000, rng=Random.GLOBAL_RNG)

Perform k-fold cross-validation for given hyperparameters.
"""
function cross_validate(X, y, λ₁, λ₂, algorithm, params_function;
    n_folds=10, tol=1e-6, maxiter=10000,
    rng=Random.GLOBAL_RNG)
    n = size(X, 1)
    fold_size = div(n, n_folds)

    # Shuffle indices
    indices = shuffle(rng, 1:n)

    cv_errors = zeros(n_folds)

    for k in 1:n_folds
        # Define validation fold
        val_start = (k - 1) * fold_size + 1
        val_end = k == n_folds ? n : k * fold_size
        val_idx = indices[val_start:val_end]

        # Training folds
        train_idx = setdiff(indices, val_idx)

        # Split data
        X_train_cv = X[train_idx, :]
        y_train_cv = y[train_idx]
        X_val_cv = X[val_idx, :]
        y_val_cv = y[val_idx]

        # Train model
        ŵ = solve_elastic_net(X_train_cv, y_train_cv, λ₁, λ₂, algorithm, params_function;
            tol=tol, maxiter=maxiter, verbose=false)

        # Compute validation error
        cv_errors[k] = mean((X_val_cv * ŵ - y_val_cv) .^ 2)
    end

    return mean(cv_errors)
end

"""
    select_hyperparameters(X_train, y_train, algorithm, params_function;
                          λ₁_grid=nothing, λ₂_grid=nothing, 
                          n_folds=10, tol=1e-6, maxiter=10000,
                          rng=Random.GLOBAL_RNG)

Select optimal hyperparameters using cross-validation.
"""
function select_hyperparameters(X_train, y_train, algorithm, params_function;
    λ₁_grid=nothing, λ₂_grid=nothing,
    n_folds=10, tol=1e-6, maxiter=10000,
    rng=Random.GLOBAL_RNG)

    # Default grids
    if λ₁_grid === nothing
        λ₁_grid = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    end
    if λ₂_grid === nothing
        λ₂_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    end

    best_cv_error = Inf
    best_λ₁ = nothing
    best_λ₂ = nothing


    for λ₁ in λ₁_grid, λ₂ in λ₂_grid
        cv_error = cross_validate(X_train, y_train, λ₁, λ₂, algorithm, params_function;
            n_folds=n_folds, tol=tol, maxiter=maxiter,
            rng=rng)

        if cv_error < best_cv_error
            best_cv_error = cv_error
            best_λ₁ = λ₁
            best_λ₂ = λ₂
        end

    end

    return (λ₁=best_λ₁, λ₂=best_λ₂, cv_error=best_cv_error)
end

# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================

"""
    evaluate_model(ŵ, w_true, X_test, y_test; threshold=1e-6)

Compute comprehensive evaluation metrics.
"""
function evaluate_model(ŵ, w_true, X_test, y_test; threshold=1e-6)
    # Prediction error (MSE)
    y_pred = X_test * ŵ
    test_mse = mean((y_pred - y_test) .^ 2)

    # Coefficient estimation error
    coef_error = norm(ŵ - w_true)
    coef_error_relative = norm(ŵ - w_true) / norm(w_true)

    # Variable selection metrics
    true_support = findall(abs.(w_true) .> threshold)
    pred_support = findall(abs.(ŵ) .> threshold)

    # True positives, false positives, false negatives
    tp = length(intersect(true_support, pred_support))
    fp = length(setdiff(pred_support, true_support))
    fn = length(setdiff(true_support, pred_support))
    tn = length(w_true) - length(union(true_support, pred_support))

    # Additional metrics
    precision = tp > 0 ? tp / (tp + fp) : 0.0
    recall = tp > 0 ? tp / (tp + fn) : 0.0
    f1_score = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0

    return (
        test_mse=test_mse,
        coef_error=coef_error,
        coef_error_relative=coef_error_relative,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1_score=f1_score
    )
end

# ============================================================================
# 5. SINGLE SIMULATION RUN
# ============================================================================

"""
    run_single_simulation(algorithm, params_function, n_train, n_val, n_test, n_features;
                         w_true=nothing, σ=3.0, ρ=0.5,
                         λ₁_grid=nothing, λ₂_grid=nothing,
                         n_folds=10, tol=1e-6, maxiter=10000,
                         rng=Random.GLOBAL_RNG)

Run one complete simulation: generate data, select hyperparameters, evaluate.
"""
function run_single_simulation(algorithm, params_function, n_train, n_val, n_test, n_features;
    w_true=nothing, σ=3.0, ρ=0.5,
    λ₁_grid=nothing, λ₂_grid=nothing,
    n_folds=10, tol=1e-6, maxiter=10000,
    rng=Random.GLOBAL_RNG)

    # 1. Generate data
    data = generate_elastic_net_data(n_train, n_val, n_test, n_features;
        w_true=w_true, σ=σ, ρ=ρ, rng=rng)

    # 2. Select hyperparameters using cross-validation
    start_time = time()
    best_params = select_hyperparameters(data.X_train, data.y_train, algorithm, params_function;
        λ₁_grid=λ₁_grid, λ₂_grid=λ₂_grid,
        n_folds=n_folds, tol=tol, maxiter=maxiter,
        rng=rng)
    cv_time = time() - start_time

    # 3. Retrain on full training data with best parameters
    start_time = time()
    ŵ = solve_elastic_net(data.X_train, data.y_train,
        best_params.λ₁, best_params.λ₂, algorithm, params_function;
        tol=tol, maxiter=maxiter, verbose=false)
    train_time = time() - start_time

    # 4. Evaluate on test set
    metrics = evaluate_model(ŵ, data.w_true, data.X_test, data.y_test)

    return (
        λ₁=best_params.λ₁,
        λ₂=best_params.λ₂,
        cv_error=best_params.cv_error,
        cv_time=cv_time,
        train_time=train_time,
        metrics...
    )
end

# ============================================================================
# 6. FULL SIMULATION STUDY
# ============================================================================

"""
    run_simulation_study(algorithms, n_runs;
                        n_train=20, n_val=20, n_test=200, n_features=8,
                        w_true=nothing, σ=3.0, ρ=0.5,
                        λ₁_grid=nothing, λ₂_grid=nothing,
                        n_folds=10, tol=1e-6, maxiter=10000, seed=123)

Run complete simulation study with multiple runs and algorithms.

# Arguments
- `algorithms`: Vector of (name, algorithm_function, params_function) tuples
  Example: [("DeyHICPP", DeyHICPP, L -> get_DeyHICPP_params(L; λ0=1/(1.05*L)))]
- `n_runs`: Number of simulation runs
"""
function run_simulation_study(algorithms, n_runs;
    n_train=20, n_val=20, n_test=200, n_features=8,
    w_true=nothing, σ=3.0, ρ=0.5,
    λ₁_grid=nothing, λ₂_grid=nothing,
    n_folds=10, tol=1e-6, maxiter=10000, seed=123)

    results = Dict()

    for (alg_name, algorithm, params_function) in algorithms
        println("Running simulations for $alg_name...")

        run_results = []
        prog = Progress(n_runs, desc="$alg_name: ")
        for run in 1:n_runs
            # Create RNG with different seed for each run
            rng = Xoshiro(seed + run)

            result = run_single_simulation(algorithm, params_function,
                n_train, n_val, n_test, n_features;
                w_true=w_true, σ=σ, ρ=ρ,
                λ₁_grid=λ₁_grid, λ₂_grid=λ₂_grid,
                n_folds=n_folds, tol=tol, maxiter=maxiter,
                rng=rng)

            push!(run_results, result)
            # Show detailed information
            next!(prog; showvalues=[
                (:Run, "$run/$n_runs"),
                (:TestMSE, round(result.test_mse, digits=4)),
                (:F1Score, round(result.f1_score, digits=4)),
                ("Selected λ₁", round(result.λ₁, digits=4)),
                ("Selected λ₂", round(result.λ₂, digits=4)),
                (:CVTime, "$(round(result.cv_time, digits=2))s")
            ])

        end

        results[alg_name] = run_results
    end

    return results
end

# ============================================================================
# 7. SUMMARIZE RESULTS
# ============================================================================

"""
    summarize_results(results)

Compute summary statistics across all simulation runs.
"""
function summarize_results(results)
    summary = Dict()

    for (alg_name, runs) in results
        # Extract metrics from all runs
        test_mse = [r.test_mse for r in runs]
        coef_error = [r.coef_error for r in runs]
        coef_error_rel = [r.coef_error_relative for r in runs]
        precision = [r.precision for r in runs]
        recall = [r.recall for r in runs]
        f1 = [r.f1_score for r in runs]
        cv_time = [r.cv_time for r in runs]
        train_time = [r.train_time for r in runs]

        # Compute std with corrected=true by default, but handle single run case
        n_runs = length(runs)
        std_func = n_runs > 1 ? std : (_ -> 0.0)

        summary[alg_name] = (
            test_mse=(mean=mean(test_mse), std=std_func(test_mse), median=median(test_mse)),
            coef_error=(mean=mean(coef_error), std=std_func(coef_error), median=median(coef_error)),
            coef_error_relative=(mean=mean(coef_error_rel), std=std_func(coef_error_rel), median=median(coef_error_rel)),
            precision=(mean=mean(precision), std=std_func(precision), median=median(precision)),
            recall=(mean=mean(recall), std=std_func(recall), median=median(recall)),
            f1_score=(mean=mean(f1), std=std_func(f1), median=median(f1)),
            cv_time=(mean=mean(cv_time), std=std_func(cv_time), total=sum(cv_time)),
            train_time=(mean=mean(train_time), std=std_func(train_time), total=sum(train_time))
        )
    end

    return summary
end
"""
    print_summary(summary)

Print formatted summary statistics.
"""
function print_summary(summary)
    println("\n" * "="^80)
    println("SIMULATION STUDY RESULTS")
    println("="^80)

    for (alg_name, stats) in summary
        println("\nAlgorithm: $alg_name")
        println("-"^80)
        println("Test MSE:              $(round(stats.test_mse.mean, digits=4)) ± $(round(stats.test_mse.std, digits=4))")
        println("Coefficient Error:     $(round(stats.coef_error.mean, digits=4)) ± $(round(stats.coef_error.std, digits=4))")
        println("Relative Coef Error:   $(round(stats.coef_error_relative.mean, digits=4)) ± $(round(stats.coef_error_relative.std, digits=4))")
        println("Precision:             $(round(stats.precision.mean, digits=4)) ± $(round(stats.precision.std, digits=4))")
        println("Recall:                $(round(stats.recall.mean, digits=4)) ± $(round(stats.recall.std, digits=4))")
        println("F1 Score:              $(round(stats.f1_score.mean, digits=4)) ± $(round(stats.f1_score.std, digits=4))")
        println("CV Time (avg):         $(round(stats.cv_time.mean, digits=2))s")
        println("Training Time (avg):   $(round(stats.train_time.mean, digits=4))s")
    end

    println("\n" * "="^80)
end
function plotit()

    # Read the results
    df = CSV.read("results/example_3/elastic_net_simulation_20251210_07_44_02.csv", DataFrame)

    # Set plot defaults for publication quality
    default(
        fontfamily="Computer Modern",
        framestyle=:box,
        grid=false,
        guidefontsize=11,
        tickfontsize=10,
        legendfontsize=10,
        dpi=300,
        size=(800, 600)
    )

    # Extract data
    algorithms = df.Algorithm
    colors = [:steelblue, :coral]
    bar_width = 0.6

    # Create output directory if needed
    mkpath("results/example_3/plots")

    # ============================================================================
    # Plot 1: Test MSE Comparison
    # ============================================================================
    p1 = bar(1:2, df.TestMSE_mean,
        yerr=df.TestMSE_std,
        ylabel="Test MSE",
        xticks=(1:2, algorithms),
        xrotation=45,
        color=colors,
        alpha=0.8,
        legend=false,
        bar_width=bar_width,
        linecolor=:black,
        linewidth=1.5,
        title="Test MSE Comparison"
    )

    # Add values on bars
    for i in 1:2
        annotate!(i, df.TestMSE_mean[i] + df.TestMSE_std[i] + 1.5,
            text(string(round(df.TestMSE_mean[i], digits=2)), 10, :center))
    end

    savefig(p1, "results/example_3/plots/test_mse_comparison.png")

    # ============================================================================
    # Plot 2: Coefficient Error Comparison
    # ============================================================================
    p2 = bar(1:2, df.CoefError_mean,
        yerr=df.CoefError_std,
        ylabel="Coefficient Error (L2 norm)",
        xticks=(1:2, algorithms),
        xrotation=45,
        color=colors,
        alpha=0.8,
        legend=false,
        bar_width=bar_width,
        linecolor=:black,
        linewidth=1.5,
        title="Coefficient Estimation Error"
    )

    for i in 1:2
        annotate!(i, df.CoefError_mean[i] + df.CoefError_std[i] + 0.15,
            text(string(round(df.CoefError_mean[i], digits=2)), 10, :center))
    end

    savefig(p2, "results/example_3/plots/coef_error_comparison.png")

    # ============================================================================
    # Plot 3: Variable Selection Metrics (Fixed groupedbar)
    # ============================================================================
    # Prepare data for groupedbar
    data_matrix = [df.Precision_mean df.Recall_mean df.F1Score_mean]'  # Transpose to 3x2

    p3 = groupedbar(
        data_matrix,
        bar_position=:dodge,
        ylabel="Score",
        xticks=(1:3, ["Precision", "Recall", "F1-Score"]),
        ylim=(0, 1.1),
        color=[colors[1] colors[2]],
        alpha=0.8,
        label=reshape(algorithms, 1, :),
        legend=:topright,
        bar_width=0.7,
        linecolor=:black,
        linewidth=1.5,
        title="Variable Selection Performance"
    )

    savefig(p3, "results/example_3/plots/selection_metrics.png")

    # ============================================================================
    # Plot 4: Timing Comparison
    # ============================================================================
    p4 = bar(1:2, df.CVTime_mean,
        yerr=df.CVTime_std,
        ylabel="Cross-Validation Time (seconds)",
        xticks=(1:2, algorithms),
        xrotation=45,
        color=colors,
        alpha=0.8,
        legend=false,
        bar_width=bar_width,
        linecolor=:black,
        linewidth=1.5,
        title="Computational Efficiency"
    )

    for i in 1:2
        annotate!(i, df.CVTime_mean[i] + df.CVTime_std[i] + 0.3,
            text(string(round(df.CVTime_mean[i], digits=2)) * "s", 10, :center))
    end

    savefig(p4, "results/example_3/plots/timing_comparison.png")

    # ============================================================================
    # Plot 5: Combined 2x2 Grid
    # ============================================================================
    p_combined = plot(p1, p2, p3, p4,
        layout=(2, 2),
        size=(1200, 1000),
        plot_title="Elastic Net Algorithm Comparison",
        plot_titlefontsize=16,
        left_margin=5Plots.mm,
        bottom_margin=5Plots.mm
    )

    savefig(p_combined, "results/example_3/plots/combined_results.png")

    # ============================================================================
    # Plot 6: Radar Chart for Multi-Metric Comparison
    # ============================================================================
    # Normalize metrics to [0, 1] for radar chart
    function normalize_inverse(x)
        return 1 .- (x .- minimum(x)) ./ (maximum(x) - minimum(x))
    end

    function normalize_direct(x)
        return (x .- minimum(x)) ./ (maximum(x) - minimum(x))
    end

    # Prepare data
    ipcmas_scores = [
        normalize_inverse(df.TestMSE_mean)[1],
        normalize_inverse(df.CoefError_mean)[1],
        normalize_direct(df.Precision_mean)[1],
        normalize_direct(df.Recall_mean)[1],
        normalize_direct(df.F1Score_mean)[1],
        normalize_inverse(df.CVTime_mean)[1]
    ]

    deyhicpp_scores = [
        normalize_inverse(df.TestMSE_mean)[2],
        normalize_inverse(df.CoefError_mean)[2],
        normalize_direct(df.Precision_mean)[2],
        normalize_direct(df.Recall_mean)[2],
        normalize_direct(df.F1Score_mean)[2],
        normalize_inverse(df.CVTime_mean)[2]
    ]

    # Create angles for radar chart
    n_metrics = 6
    angles = range(0, 2π, length=n_metrics + 1)

    # Close the polygon
    ipcmas_scores_closed = vcat(ipcmas_scores, ipcmas_scores[1])
    deyhicpp_scores_closed = vcat(deyhicpp_scores, deyhicpp_scores[1])

    p6 = plot(
        angles, ipcmas_scores_closed,
        proj=:polar,
        label="IPCMAS1",
        linewidth=2,
        color=colors[1],
        fillalpha=0.2,
        size=(700, 700),
        title="Multi-Metric Performance Comparison",
        legend=:topright
    )

    plot!(p6, angles, deyhicpp_scores_closed,
        label="DeyHICPP",
        linewidth=2,
        color=colors[2],
        fillalpha=0.2
    )

    savefig(p6, "results/example_3/plots/radar_comparison.png")

    # ============================================================================
    # Plot 7: Performance vs Computational Cost
    # ============================================================================
    p7 = scatter(
        df.CVTime_mean,
        df.TestMSE_mean,
        xerr=df.CVTime_std,
        yerr=df.TestMSE_std,
        xlabel="Cross-Validation Time (seconds)",
        ylabel="Test MSE",
        markersize=12,
        color=colors,
        markerstrokewidth=2,
        markerstrokecolor=:black,
        label=reshape(algorithms, 1, :),
        legend=:topright,
        title="Prediction Accuracy vs Computational Cost",
        size=(800, 600)
    )

    # Add algorithm names near points
    for i in 1:2
        annotate!(df.CVTime_mean[i] + 0.3, df.TestMSE_mean[i] + 1.5,
            text(algorithms[i], 9, :left))
    end

    savefig(p7, "results/example_3/plots/accuracy_vs_cost.png")

    # ============================================================================
    # Plot 8: Relative Performance Metrics (as percentages)
    # ============================================================================
    baseline_idx = 1  # IPCMAS1
    compare_idx = 2   # DeyHICPP

    improvements = [
        (df.TestMSE_mean[baseline_idx] - df.TestMSE_mean[compare_idx]) / df.TestMSE_mean[compare_idx] * 100,
        (df.CoefError_mean[baseline_idx] - df.CoefError_mean[compare_idx]) / df.CoefError_mean[compare_idx] * 100,
        (df.F1Score_mean[baseline_idx] - df.F1Score_mean[compare_idx]) / df.F1Score_mean[compare_idx] * 100,
        (df.CVTime_mean[baseline_idx] - df.CVTime_mean[compare_idx]) / df.CVTime_mean[compare_idx] * 100
    ]

    metric_labels = ["Test MSE", "Coef. Error", "F1-Score", "Time"]

    p8 = bar(
        1:4,
        improvements,
        ylabel="Relative Improvement of IPCMAS1 over DeyHICPP (%)",
        xticks=(1:4, metric_labels),
        xrotation=45,
        color=[improvements[i] > 0 ? :green : :red for i in 1:4],
        alpha=0.7,
        legend=false,
        bar_width=0.6,
        linecolor=:black,
        linewidth=1.5,
        title="IPCMAS1 Performance Relative to DeyHICPP"
    )

    # Add zero line
    hline!([0], color=:black, linestyle=:dash, linewidth=1.5)

    # Add percentage labels
    for i in 1:4
        y_pos = improvements[i] > 0 ? improvements[i] + 2 : improvements[i] - 2
        annotate!(i, y_pos,
            text(string(round(improvements[i], digits=1)) * "%", 10, :center))
    end

    savefig(p8, "results/example_3/plots/relative_improvement.png")

    println("\n" * "="^80)
    println("All plots saved successfully!")
    println("="^80)
    println("\nGenerated plots:")
    println("  1. test_mse_comparison.png")
    println("  2. coef_error_comparison.png")
    println("  3. selection_metrics.png")
    println("  4. timing_comparison.png")
    println("  5. combined_results.png")
    println("  6. radar_comparison.png")
    println("  7. accuracy_vs_cost.png")
    println("  8. relative_improvement.png")
    println("\nAll files saved to: results/example_3/plots/")
end


# ============================================================================
# 8. USAGE EXAMPLE
# ============================================================================
opts, positional = parse_args(ARGS)

# Extract simulation parameters with defaults
n_runs = parse(Int, get(opts, "n_runs", "50"))
n_train = parse(Int, get(opts, "n_train", "20"))
n_val = parse(Int, get(opts, "n_val", "20"))
n_test = parse(Int, get(opts, "n_test", "200"))
n_features = parse(Int, get(opts, "n_features", "8"))
σ = parse(Float64, get(opts, "sigma", "3.0"))
ρ = parse(Float64, get(opts, "rho", "0.5"))
n_folds = parse(Int, get(opts, "n_folds", "10"))
tol = parse(Float64, get(opts, "tol", "1e-6"))
maxiter = parse(Int, get(opts, "maxiter", "10000"))
seed = parse(Int, get(opts, "seed", "2025"))
clearfolder = any(x -> x in ("--clear", "-c"), ARGS)
runwhat = parse(Int, get(opts, "run", "all"))

if runwhat in ["all", "runonly"]
    # Define algorithms with your configuration format


    algorithms = [
        ("DeyHICPP", DeyHICPP, get_DeyHICPP_params_EN),
        ("IPCMAS1", IPCMAS1, L -> get_IPCMAS1_params_EN(L; γ=1.1, μ0=0.5, α0=0.5, β0=0.3, λ0=0.05)),
    ]

    # Run simulation study
    results = run_simulation_study(
        algorithms,
        n_runs;
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_features=n_features,
        σ=σ,
        ρ=ρ,
        n_folds=n_folds,
        tol=tol,
        maxiter=maxiter,
        seed=seed
    )

    # Compute and print summary statistics
    summary = summarize_results(results)
    print_summary(summary)

    # Access individual algorithm results
    for (alg_name, runs) in results
        println("\\n$alg_name results:")
        println("  First run test MSE: ", runs[1].test_mse)
        println("  First run selected λ₁: ", runs[1].λ₁)
        println("  First run selected λ₂: ", runs[1].λ₂)
    end


    # Save results using the same parsed parameters
    saved_files = save_simulation_results(
        summary, "results/example_3";
        base_filename="elastic_net_simulation",
        n_runs=n_runs,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_features=n_features,
        σ=σ,
        ρ=ρ,
        n_folds=n_folds,
        tol=tol,
        maxiter=maxiter,
        seed=seed,
        clearfolder=clearfolder
    )

    println("\nFiles saved:")
    println("  CSV:  ", saved_files.csv)
    println("  XLSX: ", saved_files.xlsx)



elseif runwhat in ["all", "plot"]
    plotit()
else
    println("Nothing requested ($runwhat)")
end
