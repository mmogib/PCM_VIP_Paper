include("../includes.jl")

"""
    save_convergence_results(convergence_results, output_dir="results/examples/linear_convergence";
                            base_filename="linear_convergence_results",
                            n_runs=20, seed=123)

Save convergence verification results to CSV.
"""
function save_convergence_results(convergence_results::Dict,
    output_dir::String="results/examples/linear_convergence";
    base_filename::String="linear_convergence_results",
    n_runs::Int=20,
    n_train::Int=20,
    n_test::Int=200,
    n_features::Int=8,
    σ::Float64=3.0,
    ρ::Float64=0.5,
    λ₁::Float64=1.0,
    λ₂::Float64=0.1,
    seed::Int=123)

    # Create output directory if needed
    mkpath(output_dir)

    # Create DataFrame for results
    rows = []

    for (alg_name, stats) in convergence_results
        row = (
            Algorithm=alg_name,
            MeanConvergenceRate=stats.mean_rate,
            StdConvergenceRate=stats.std_rate,
            MeanRSquared=stats.mean_r_squared,
            LinearPercentage=stats.linear_percentage,
            n_runs=n_runs,
            n_train=n_train,
            n_test=n_test,
            n_features=n_features,
            sigma=σ,
            rho=ρ,
            lambda1=λ₁,
            lambda2=λ₂,
            random_seed=seed
        )
        push!(rows, row)
    end

    # Convert to DataFrame
    df = DataFrame(rows)

    # Prepare file path with timestamp
    csv_path = prepare_filepath(joinpath(output_dir, base_filename * ".csv"); dated=true)

    # Save as CSV
    CSV.write(csv_path, df)
    println("Saved convergence results: $csv_path")

    return csv_path
end

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
"""
    analyze_convergence_rate(history, w_true=nothing; 
                             skip_initial=5, 
                             use_last_n=nothing)

Analyze convergence rate from algorithm history.
Uses :err from history if available, otherwise computes from :xk.

Returns:
- convergence_rate: Estimated C (should be < 1 for convergence)
- is_linear: Whether convergence appears linear
- r_squared: Goodness of linear fit
"""
function analyze_convergence_rate(history::Dict, w_true::Union{Vector{Float64},Nothing}=nothing;
    skip_initial::Int=5,
    use_last_n::Union{Int,Nothing}=nothing)

    # Get errors from history
    if haskey(history, :err) && !isempty(history[:err])
        errors = history[:err]
    elseif haskey(history, :xk) && !isempty(history[:xk]) && w_true !== nothing
        # Fallback: compute from iterates if available
        iterates = history[:xk]
        errors = [norm(w - w_true) for w in iterates]
    else
        error("History must contain :err or :xk key")
    end

    n_iters = length(errors)

    # Skip initial iterations (often erratic) and optionally use only last n
    start_idx = skip_initial + 1
    end_idx = use_last_n === nothing ? n_iters : min(n_iters, start_idx + use_last_n - 1)

    if end_idx <= start_idx
        return (convergence_rate=NaN, is_linear=false, r_squared=NaN,
            errors=errors, log_errors=log.(errors))
    end

    # Work with log errors
    log_errors = log.(errors[start_idx:end_idx])
    iterations = start_idx:end_idx

    # Linear regression: log(error) = a + b*k
    # where b = log(C) is the convergence rate
    n = length(iterations)
    mean_iter = mean(iterations)
    mean_log_error = mean(log_errors)

    # Slope b = Cov(k, log_error) / Var(k)
    b = sum((iterations .- mean_iter) .* (log_errors .- mean_log_error)) /
        sum((iterations .- mean_iter) .^ 2)
    a = mean_log_error - b * mean_iter

    # Convergence rate C = exp(b)
    C = exp(b)

    # R-squared for goodness of fit
    predicted = a .+ b .* iterations
    ss_res = sum((log_errors .- predicted) .^ 2)
    ss_tot = sum((log_errors .- mean_log_error) .^ 2)
    r_squared = 1 - ss_res / ss_tot

    # Check if convergence is linear (high R²)
    is_linear = r_squared > 0.95

    return (
        convergence_rate=C,
        is_linear=is_linear,
        r_squared=r_squared,
        slope=b,
        intercept=a,
        errors=errors,
        log_errors=log.(errors),
        analysis_range=(start_idx, end_idx)
    )
end

"""
    estimate_convergence_rates(algorithms_results, w_true)

Estimate convergence rates for multiple algorithms.
"""
function estimate_convergence_rates(algorithms_results::Dict, w_true::Vector{Float64})
    rates = Dict()

    for (alg_name, solution) in algorithms_results
        analysis = analyze_convergence_rate(solution.history, w_true)
        rates[alg_name] = analysis
    end

    return rates
end
"""
    plot_convergence_comparison(algorithms_results; 
                               skip_initial=5, 
                               title="Convergence Comparison")

Plot convergence curves for multiple algorithms with better visibility.
"""
function plot_convergence_comparison(algorithms_results::Dict;
    skip_initial::Int=5,
    title::String="Convergence Comparison")

    p = plot(
        xlabel="Iteration",
        ylabel="log(||w - w*||)",
        title=title,
        legend=:topright,
        yscale=:identity,
        framestyle=:box,
        grid=true,
        gridalpha=0.3,
        dpi=300,
        size=(900, 600)
    )

    # More distinct colors and styles
    colors = [:blue, :red, :green, :purple]
    markers = [:circle, :square, :diamond, :utriangle]
    linestyles = [:solid, :dash, :dot, :dashdot]

    for (idx, (alg_name, solution)) in enumerate(algorithms_results)
        history = solution.history

        # Get errors from history
        if !haskey(history, :err) || isempty(history[:err])
            @warn "No error history for $alg_name"
            continue
        end

        errors = history[:err]
        log_errors = log.(errors)

        # Plot from skip_initial onwards
        plot_range = skip_initial+1:length(errors)

        plot!(p, plot_range, log_errors[plot_range],
            label=alg_name,
            linewidth=2.5,
            color=colors[mod1(idx, length(colors))],
            linestyle=linestyles[mod1(idx, length(linestyles))],
            marker=markers[mod1(idx, length(markers))],
            markersize=4,
            markerstrokewidth=1.5,
            markevery=max(1, div(length(plot_range), 15)))

        # Fit and plot linear trend for all algorithms
        analysis = analyze_convergence_rate(history; skip_initial=skip_initial)

        start_idx, end_idx = analysis.analysis_range
        fitted = analysis.intercept .+ analysis.slope .* (start_idx:end_idx)

        plot!(p, start_idx:end_idx, fitted,
            label="$(alg_name) fit (C=$(round(analysis.convergence_rate, digits=3)))",
            linestyle=:dash,
            linewidth=2,
            color=colors[mod1(idx, length(colors))],
            alpha=0.7)
    end

    return p
end
"""
    plot_convergence_comparison_split(algorithms_results; 
                                     skip_initial=5, 
                                     title="Convergence Comparison")

Create separate subplots for each algorithm for better visibility.
"""
function plot_convergence_comparison_split(algorithms_results::Dict;
    skip_initial::Int=5,
    title::String="Convergence Comparison")

    n_algs = length(algorithms_results)
    colors = [:steelblue, :coral, :green, :purple]

    plots = []

    for (idx, (alg_name, solution)) in enumerate(algorithms_results)
        history = solution.history

        if !haskey(history, :err) || isempty(history[:err])
            @warn "No error history for $alg_name"
            continue
        end

        errors = history[:err]
        log_errors = log.(errors)
        plot_range = skip_initial+1:length(errors)

        p_sub = plot(
            plot_range, log_errors[plot_range],
            xlabel="Iteration",
            ylabel="log(||w - w*||)",
            title=alg_name,
            linewidth=2.5,
            color=colors[mod1(idx, length(colors))],
            marker=:circle,
            markersize=3,
            markevery=max(1, div(length(plot_range), 20)),
            legend=false,
            framestyle=:box,
            grid=true,
            gridalpha=0.3
        )

        # Add linear fit
        analysis = analyze_convergence_rate(history; skip_initial=skip_initial)
        start_idx, end_idx = analysis.analysis_range
        fitted = analysis.intercept .+ analysis.slope .* (start_idx:end_idx)

        plot!(p_sub, start_idx:end_idx, fitted,
            linestyle=:dash,
            linewidth=2,
            color=:black,
            alpha=0.7,
            label="C=$(round(analysis.convergence_rate, digits=4))")

        push!(plots, p_sub)
    end

    layout = n_algs == 3 ? (1, 3) : (2, 2)
    p_combined = plot(plots...,
        layout=layout,
        size=(1200, 400),
        plot_title=title,
        plot_titlefontsize=14,
        legend=:topright
    )

    return p_combined
end
"""
    plot_convergence_rate_verification(history, alg_name; skip_initial=5)

Detailed convergence analysis plot for a single algorithm.
"""
function plot_convergence_rate_verification(history::Dict,
    alg_name::String; skip_initial::Int=5)

    analysis = analyze_convergence_rate(history; skip_initial=skip_initial)

    # Create 2x2 subplot
    p1 = plot(1:length(analysis.errors), analysis.errors,
        xlabel="Iteration",
        ylabel="||w - w*||",
        title="Error vs Iteration",
        linewidth=2,
        color=:steelblue,
        legend=false,
        yscale=:log10)

    p2 = plot(1:length(analysis.log_errors), analysis.log_errors,
        xlabel="Iteration",
        ylabel="log(||w - w*||)",
        title="Log Error vs Iteration",
        linewidth=2,
        color=:coral,
        legend=false)

    # Add linear fit
    start_idx, end_idx = analysis.analysis_range
    fitted = analysis.intercept .+ analysis.slope .* (start_idx:end_idx)
    plot!(p2, start_idx:end_idx, fitted,
        linestyle=:dash,
        linewidth=2,
        color=:black,
        label="Linear fit")

    # Ratio plot: ||e_{k+1}|| / ||e_k||
    ratios = analysis.errors[2:end] ./ analysis.errors[1:end-1]
    p3 = plot(1:length(ratios), ratios,
        xlabel="Iteration",
        ylabel="||e(k+1)|| / ||e(k)||",
        title="Convergence Ratio",
        linewidth=2,
        color=:green,
        legend=false)
    hline!(p3, [analysis.convergence_rate],
        linestyle=:dash,
        linewidth=2,
        color=:red,
        label="Estimated C")

    # Residuals from linear fit
    residuals = analysis.log_errors[start_idx:end_idx] .- fitted
    p4 = scatter(start_idx:end_idx, residuals,
        xlabel="Iteration",
        ylabel="Residual",
        title="Residuals from Linear Fit",
        color=:purple,
        markersize=3,
        legend=false)
    hline!(p4, [0], linestyle=:dash, color=:black)

    p = plot(p1, p2, p3, p4,
        layout=(2, 2),
        size=(1200, 1000),
        plot_title="$alg_name Convergence Analysis\nC=$(round(analysis.convergence_rate, digits=4)), R²=$(round(analysis.r_squared, digits=4))",
        plot_titlefontsize=14)

    return p, analysis
end

"""
    run_single_simulation_with_history(algorithm, params_function, 
                                       n_train, n_val, n_test, n_features;
                                       kwargs...)

Run simulation and return full solution with history.
"""
function run_single_simulation_with_history(algorithm, params_function,
    n_train, n_val, n_test, n_features;
    w_true=nothing, σ=3.0, ρ=0.5,
    λ₁=1.0, λ₂=0.1,
    tol=1e-6, maxiter=10000,
    rng=Random.GLOBAL_RNG)

    # Generate data
    data = generate_elastic_net_data(n_train, n_val, n_test, n_features;
        w_true=w_true, σ=σ, ρ=ρ, rng=rng)

    # Solve with fixed hyperparameters
    n_samples, n_features_actual = size(data.X_train)

    # Create elastic net problem
    λ₁_scaled = λ₁ / (1 + λ₂)
    Aλ(w, λ) = sign.(w) .* max.(abs.(w) .- λ * λ₁_scaled, 0)
    B(w) = data.X_train' * (data.X_train * w - data.y_train)
    A(w) = w

    XtX = data.X_train' * data.X_train
    L = maximum(sum(XtX, dims=1))

    x0 = zeros(n_features_actual)
    x1 = zeros(n_features_actual)

    problem = Problem(
        name="ElasticNet_λ₁=$(λ₁)_λ₂=$(λ₂)",
        Aλ=Aλ,
        A=A,
        B=B,
        L=L,
        x0=x0,
        x1=x1,
        n=n_features_actual
    )

    # Get algorithm parameters
    algorithm_params = params_function(L)

    # Solve
    solution = solve_problem(algorithm, problem, algorithm_params;
        tol=tol, maxiter=maxiter, verbose=false)

    return solution, data
end
"""
    verify_linear_convergence(algorithms, n_runs=10; kwargs...)

Run convergence verification study.
"""
function verify_linear_convergence(algorithms, n_runs::Int=10;
    n_train::Int=20, n_test::Int=200,
    n_features::Int=8, σ::Float64=3.0, ρ::Float64=0.5,
    λ₁::Float64=1.0, λ₂::Float64=0.1,
    tol::Float64=1e-6, maxiter::Int=10000,
    seed::Int=123)

    results = Dict()

    for (alg_name, algorithm, params_function) in algorithms
        println("\nAnalyzing convergence for $alg_name...")

        run_analyses = []

        for run in 1:n_runs
            rng = Xoshiro(seed + run)

            # Generate data
            data = generate_elastic_net_data(n_train, 0, n_test, n_features;
                σ=σ, ρ=ρ, rng=rng)

            # Create elastic net problem
            λ₁_scaled = λ₁ / (1 + λ₂)
            Aλ(w, λ) = sign.(w) .* max.(abs.(w) .- λ * λ₁_scaled, 0)
            B(w) = data.X_train' * (data.X_train * w - data.y_train)
            A(w) = w

            XtX = data.X_train' * data.X_train
            L = maximum(sum(XtX, dims=1))

            x0 = zeros(n_features)
            x1 = zeros(n_features)

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

            # Get algorithm parameters and solve
            algorithm_params = params_function(L)
            solution = solve_problem(algorithm, problem, algorithm_params;
                tol=tol, maxiter=maxiter, verbose=false)

            # Analyze convergence
            analysis = analyze_convergence_rate(solution.history)
            push!(run_analyses, analysis)
        end

        # Aggregate results
        rates = [a.convergence_rate for a in run_analyses if !isnan(a.convergence_rate)]
        r_squareds = [a.r_squared for a in run_analyses if !isnan(a.r_squared)]
        is_linear_count = sum([a.is_linear for a in run_analyses])

        results[alg_name] = (
            mean_rate=mean(rates),
            std_rate=std(rates),
            mean_r_squared=mean(r_squareds),
            linear_percentage=is_linear_count / n_runs * 100,
            all_analyses=run_analyses
        )

        println("  Mean convergence rate C: $(round(mean(rates), digits=4)) ± $(round(std(rates), digits=4))")
        println("  Mean R²: $(round(mean(r_squareds), digits=4))")
        println("  Linear convergence: $(is_linear_count)/$n_runs runs ($(round(is_linear_count/n_runs*100, digits=1))%)")
    end

    return results
end



# Define algorithms
algorithms = [
    ("DeyHICPP", DeyHICPP, get_DeyHICPP_params_EN),
    ("IPCMAS1", IPCMAS1, L -> get_IPCMAS1_params_EN(L; γ=1.1, μ0=0.5, α0=0.5, β0=0.3, λ0=0.05)),
    ("IPCMAS2", IPCMAS2, (L) -> get_IPCMAS2_params(L; γ=1.1, λ0=1 / (1.05 * L)))
]
# Verify linear convergence
convergence_results = verify_linear_convergence(algorithms, 20;
    n_train=20, n_test=200,
    seed=123)

# Print summary
println("\n" * "="^80)
println("LINEAR CONVERGENCE VERIFICATION")
println("="^80)
for (alg_name, stats) in convergence_results
    println("\n$alg_name:")
    println("  Convergence Rate C: $(round(stats.mean_rate, digits=4)) ± $(round(stats.std_rate, digits=4))")
    println("  R² (linearity): $(round(stats.mean_r_squared, digits=4))")
    println("  Linear in $(round(stats.linear_percentage, digits=1))% of runs")
end



# Save convergence results
saved_csv = save_convergence_results(
    convergence_results,
    "results/examples/linear_convergence";
    base_filename="linear_convergence_results",
    n_runs=20,
    n_train=20,
    n_test=200,
    n_features=8,
    σ=3.0,
    ρ=0.5,
    λ₁=1.0,
    λ₂=0.1,
    seed=123
)

# Plot comparison for one run
rng = Xoshiro(123)
alg_results = Dict()

# Generate data once for all algorithms
data = generate_elastic_net_data(20, 0, 200, 8; σ=3.0, ρ=0.5, rng=rng)

# Fixed hyperparameters for convergence analysis
λ₁ = 1.0
λ₂ = 0.1
λ₁_scaled = λ₁ / (1 + λ₂)

# Create elastic net problem
Aλ(w, λ) = sign.(w) .* max.(abs.(w) .- λ * λ₁_scaled, 0)
B(w) = data.X_train' * (data.X_train * w - data.y_train)
A(w) = w

XtX = data.X_train' * data.X_train
L = maximum(sum(XtX, dims=1))

x0 = zeros(8)
x1 = zeros(8)

problem = Problem(
    name="ElasticNet_λ₁=$(λ₁)_λ₂=$(λ₂)",
    Aλ=Aλ,
    A=A,
    B=B,
    L=L,
    x0=x0,
    x1=x1,
    n=8
)

# Solve with each algorithm
for (alg_name, algorithm, params_function) in algorithms
    println("Running $alg_name...")
    algorithm_params = params_function(L)
    solution = solve_problem(algorithm, problem, algorithm_params;
        tol=1e-6, maxiter=10000, verbose=false)
    alg_results[alg_name] = solution
end

# Plot convergence comparison
p_compare = plot_convergence_comparison(alg_results;
    title="Linear Convergence Verification")
savefig(p_compare, "results/examples/linear_convergence/convergence_comparison.png")

# Detailed analysis for IPCMAS2
p_detail, analysis = plot_convergence_rate_verification(
    alg_results["IPCMAS2"].history,
    "IPCMAS2"
)
savefig(p_detail, "results/examples/linear_convergence/ipcmas2_convergence_detail.png")

# Also save for IPCMAS1
p_detail1, analysis1 = plot_convergence_rate_verification(
    alg_results["IPCMAS1"].history,
    "IPCMAS1"
)
savefig(p_detail1, "results/examples/linear_convergence/ipcmas1_convergence_detail.png")

# And DeyHICPP
p_detail_dey, analysis_dey = plot_convergence_rate_verification(
    alg_results["DeyHICPP"].history,
    "DeyHICPP"
)
savefig(p_detail_dey, "results/examples/linear_convergence/deyhicpp_convergence_detail.png")

println("\n" * "="^80)
println("All results saved!")
println("="^80)
println("CSV: $saved_csv")
println("Plots saved to: results/examples/linear_convergence/")
println("\nIPCMAS2 Convergence Analysis:")
println("  Convergence Rate C: $(round(analysis.convergence_rate, digits=4))")
println("  R² (linearity): $(round(analysis.r_squared, digits=4))")
println("  Is R-linear: $(analysis.is_linear ? "Yes" : "No")")



# Plot convergence comparison - improved version
p_compare = plot_convergence_comparison(alg_results;
    skip_initial=5,
    title="Linear Convergence Verification")
savefig(p_compare, "results/examples/linear_convergence/convergence_comparison.png")

# Alternative: side-by-side comparison
p_compare_split = plot_convergence_comparison_split(alg_results;
    skip_initial=5,
    title="Linear Convergence Verification")

savefig(p_compare_split, "results/examples/linear_convergence/convergence_comparison_split.png")