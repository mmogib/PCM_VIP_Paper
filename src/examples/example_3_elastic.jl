include("../includes.jl")

function cross_validate(X, y, λ₁, λ₂; n_folds=10, rng=Random.GLOBAL_RNG)
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

        # Training folds (all except k)
        train_idx = setdiff(indices, val_idx)

        # Split data
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_val = X[val_idx, :]
        y_val = y[val_idx]

        # Train model
        ŵ = solve_elastic_net(X_train, y_train, λ₁, λ₂)

        # Compute validation error
        cv_errors[k] = mean((X_val * ŵ - y_val) .^ 2)
    end

    # Return average CV error
    return mean(cv_errors)
end
# Generate data for elastic net simulation with correlated predictors.
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
    Z = randn(rng, n_total, n_features)  # Standard normal
    X = Z * L'  # Transform to have correlation Σ

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

# ============================================
# Elastic Net Problem Setup
# ============================================
function setup_example3(n::Int; seed=2025, num_of_instances=1)
    rng = Xoshiro(seed)
    n_samples = 100
    n_features = 20
    n_nonzero = 5
    noise_level = 0.1
    noise_dist = Normal(0, noise_level)
    λ₁ = 0.1  # L1 penalty
    λ₂ = 0.05 # L2 penalty
    # Generate random matrix Z ∈ [1, 100]
    X, y, w_true, true_indices = generate_elastic_net_data(100, 20, 5, 0.1)
    problems = Vector{Problem}(undef, num_of_instances)
    for i in 1:num_of_instances
        # Z = rand(rng, U, n, n)

        # # f = Z^T * Z
        # B_matrix = Z' * Z

        # # Lipschitz constant L = max eigenvalue of f
        # L = maximum(abs.(eigvals(B_matrix)))

        # # Upper triangular matrix with all entries one
        # A_matrix = triu(ones(n, n))

        # # Define operator f
        # B(x) = B_matrix * x

        # # Define resolvent of A: J^A_λ(x) = (I + λA)^(-1) x
        # function A_resolvent(x, λ)
        #     return (I + λ * A_matrix) \ x
        # end

        # # Generate initial points in [0,1]
        # x0 = rand(rng, n)
        # x1 = rand(rng, n)
        # Generate design matrix
        X = rand(rng, noise_dist, n_samples, n_features)

        # Normalize columns
        X = X ./ sqrt.(sum(X .^ 2, dims=1))

        # Create sparse true coefficients
        w_true = zeros(n_features)
        indices = sort(randperm(rng, n_features)[1:n_nonzero])
        w_true[indices] = randn(rng, n_nonzero) .* 3

        # Generate observations
        y = X * w_true + noise_level * randn(rng, n_samples)


        # Operator A: Resolvent of ∂(λ₁||·||₁) - soft thresholding
        # J^A_λ(x) = prox_{λ*λ₁||·||₁}(x)
        A_resolvent(x, λ) = sign.(x) .* max.(abs.(x) .- λ * λ₁, 0)

        # Operator B: Gradient of smooth part
        # B(w) = ∇f(w) = X'(Xw - y) + λ₂w
        B(w) = X' * (X * w - y)

        # Initial points
        n_features = size(X, 2)
        x0 = zeros(n_features)
        x1 = randn(n_features) * 0.01  # Small random perturbation
        problems[i] = Problem(; name="Example 1 ($i)", Aλ=A_resolvent, A=x -> A_matrix * x, B=B, L=L, x0=x0, x1=x1, n=n)
    end

    return problems
end

# ============================================
# Define Operators for Elastic Net
# ============================================

# Generate the problem




# ============================================
# Run the Algorithms
# ============================================

# println("=" * "="^59)
# println("ELASTIC NET PROBLEM - Using Existing Algorithm Implementations")
# println("=" * "="^59)
# println("\nProblem Setup:")
# println("  Samples: ", size(X, 1))
# println("  Features: ", size(X, 2))
# println("  True non-zero coefficients: ", length(true_indices), " at positions ", true_indices)
# println("  λ₁ (L1 penalty): ", λ₁)
# println("  λ₂ (L2 penalty): ", λ₂)

# println("\n" * "-"^60)
# println("Running Algorithms...")
# println("-"^60)

# # Run DeyHICPP
# const __ex3_verbose = any(x -> x in ("--verbose", "-v"), ARGS)
# const __ex3_show_progress = !any(x -> x == "--no-progress", ARGS)
# p_ex3 = __ex3_show_progress ? Progress(3; desc="Elastic Net") : nothing
# if __ex3_verbose && !__ex3_show_progress
#     println("\n1. Running DeyHICPP...")
# end
# t1 = time()
# sol_dey, iter_dey, conv_dey = DeyHICPP(A, B, x0, x1,
#     γ=1.5, α=0.5, tol=1e-6, maxiter=1000)
# time_dey = time() - t1
# if !__ex3_show_progress
#     println("   Iterations: ", iter_dey, ", Converged: ", conv_dey, ", Time: ", round(time_dey, digits=4), "s")
# end
# if __ex3_show_progress
#     next!(p_ex3; showvalues=[(:algo, "DeyHICPP"), (:iter, iter_dey), (:time, round(time_dey, digits=3))])
# end

# # Run IPCMAS1
# if __ex3_verbose && !__ex3_show_progress
#     println("\n2. Running IPCMAS1...")
# end
# t2 = time()
# sol_ipcmas1, iter_ipcmas1, conv_ipcmas1 = IPCMAS1(A, B, x0, x1,
#     γ=1.5, μ=0.5, λ1=0.01, α=0.5, tol=1e-6, maxiter=1000)
# time_ipcmas1 = time() - t2
# if !__ex3_show_progress
#     println("   Iterations: ", iter_ipcmas1, ", Converged: ", conv_ipcmas1, ", Time: ", round(time_ipcmas1, digits=4), "s")
# end
# if __ex3_show_progress
#     next!(p_ex3; showvalues=[(:algo, "IPCMAS1"), (:iter, iter_ipcmas1), (:time, round(time_ipcmas1, digits=3))])
# end

# # Run IPCMAS2
# if __ex3_verbose && !__ex3_show_progress
#     println("\n3. Running IPCMAS2...")
# end
# t3 = time()
# sol_ipcmas2, iter_ipcmas2, conv_ipcmas2 = IPCMAS2(A, B, x0, x1,
#     γ=1.5, μ=0.5, λ1=0.01, α=0.5, θ=0.5, tol=1e-6, maxiter=1000)
# time_ipcmas2 = time() - t3
# if !__ex3_show_progress
#     println("   Iterations: ", iter_ipcmas2, ", Converged: ", conv_ipcmas2, ", Time: ", round(time_ipcmas2, digits=4), "s")
# end
# if __ex3_show_progress
#     next!(p_ex3; showvalues=[(:algo, "IPCMAS2"), (:iter, iter_ipcmas2), (:time, round(time_ipcmas2, digits=3))])
# end

# # ============================================
# # Compare Results
# # ============================================

# if !__ex3_show_progress
#     println("\n" * "="^60)
#     println("RESULTS COMPARISON")
#     println("="^60)
# end

# # Compute objective values
# obj(w) = 0.5 * norm(X * w - y)^2 + λ₁ * norm(w, 1) + 0.5 * λ₂ * norm(w)^2

# obj_dey = obj(sol_dey)
# obj_ipcmas1 = obj(sol_ipcmas1)
# obj_ipcmas2 = obj(sol_ipcmas2)

# if !__ex3_show_progress
#     println("\nObjective Function Values:")
#     println("-"^40)
#     @printf("  DeyHICPP:   %.6f\n", obj_dey)
#     @printf("  IPCMAS1:    %.6f\n", obj_ipcmas1)
#     @printf("  IPCMAS2:    %.6f\n", obj_ipcmas2)
# end

# # Solution quality
# if !__ex3_show_progress
#     println("\nSolution Quality:")
#     println("-"^40)
#     @printf("  ||w_dey - w_true||/||w_true||:     %.6f\n", norm(sol_dey - w_true) / norm(w_true))
#     @printf("  ||w_ipcmas1 - w_true||/||w_true||: %.6f\n", norm(sol_ipcmas1 - w_true) / norm(w_true))
#     @printf("  ||w_ipcmas2 - w_true||/||w_true||: %.6f\n", norm(sol_ipcmas2 - w_true) / norm(w_true))
# end

# # Sparsity recovery
# threshold = 1e-3
# nnz_dey = sum(abs.(sol_dey) .> threshold)
# nnz_ipcmas1 = sum(abs.(sol_ipcmas1) .> threshold)
# nnz_ipcmas2 = sum(abs.(sol_ipcmas2) .> threshold)

# if !__ex3_show_progress
#     println("\nSparsity Recovery:")
#     println("-"^40)
#     println("  True non-zeros: ", length(true_indices))
#     println("  DeyHICPP non-zeros:   ", nnz_dey)
#     println("  IPCMAS1 non-zeros:    ", nnz_ipcmas1)
#     println("  IPCMAS2 non-zeros:    ", nnz_ipcmas2)
# end

# # Performance summary
# if !__ex3_show_progress
#     println("\n" * "="^60)
#     println("PERFORMANCE SUMMARY")
#     println("="^60)
#     println("Algorithm    Iterations    Time(s)     Objective     Error")
#     println("-"^60)
#     @printf("DeyHICPP     %6d      %.4f    %.6f    %.4f\n",
#         iter_dey, time_dey, obj_dey, norm(sol_dey - w_true) / norm(w_true))
#     @printf("IPCMAS1      %6d      %.4f    %.6f    %.4f\n",
#         iter_ipcmas1, time_ipcmas1, obj_ipcmas1, norm(sol_ipcmas1 - w_true) / norm(w_true))
#     @printf("IPCMAS2      %6d      %.4f    %.6f    %.4f\n",
#         iter_ipcmas2, time_ipcmas2, obj_ipcmas2, norm(sol_ipcmas2 - w_true) / norm(w_true))
# else
#     @printf("Summary: iters=[%d,%d,%d], times=[%.3f,%.3f,%.3f]s\n",
#         iter_dey, iter_ipcmas1, iter_ipcmas2, time_dey, time_ipcmas1, time_ipcmas2)
# end

# if !__ex3_show_progress
#     println("\n" * "="^60)
#     println("KEY FINDINGS:")
#     println("="^60)
#     if iter_ipcmas1 < iter_dey || iter_ipcmas2 < iter_dey
#         println("✓ Double inertial extrapolation shows improved convergence")
#     end
#     if iter_ipcmas2 < iter_ipcmas1
#         println("✓ IPCMAS2's simpler structure achieves faster convergence")
#     end
#     println("✓ All algorithms successfully solve the Elastic Net problem")
#     println("✓ Strong convexity (due to λ₂) ensures linear convergence")
# end

# # ============================================
# # Save Results to CSV
# # ============================================

# # Performance metrics
# results = ["Algorithm" "Iterations" "Time" "Objective" "RelativeError" "NonZeros";
#     "DeyHICPP" iter_dey time_dey obj_dey norm(sol_dey - w_true)/norm(w_true) nnz_dey;
#     "IPCMAS1" iter_ipcmas1 time_ipcmas1 obj_ipcmas1 norm(sol_ipcmas1 - w_true)/norm(w_true) nnz_ipcmas1;
#     "IPCMAS2" iter_ipcmas2 time_ipcmas2 obj_ipcmas2 norm(sol_ipcmas2 - w_true)/norm(w_true) nnz_ipcmas2]

# writedlm("results/elastic_net_results.csv", results, ',')

# # Solutions
# solutions = hcat(sol_dey, sol_ipcmas1, sol_ipcmas2, w_true)
# headers = ["DeyHICPP" "IPCMAS1" "IPCMAS2" "True"]
# writedlm("results/elastic_net_solutions.csv", vcat(headers, solutions), ',')

# println("\nResults saved to results directory as CSV files")
