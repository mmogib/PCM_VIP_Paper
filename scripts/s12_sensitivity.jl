## s12_sensitivity.jl — Sensitivity for new DIPCM (paper §3.4)
#
# Phase A: 32-cell β̄ × θ̄ grid at fixed defaults
#          α_n = 1/(n+1),  σ_n = 0.8 - α_n,
#          ε_n = 5.0/(n+1)^2.1,  ε'_n = 10.0/(n+1)^2.1.
# Phase B: 5-cell ε-scale check (multiplier relative to defaults above) at the
#          PAPER-LOCKED cell (β̄ = 0.3, θ̄ = 0.9), not at the grid argmin.
#          The grid argmin is reported for context but not used for Phase B.
#
# Test problem: Example 1 VIP, x* = 0, W ~ U(0,1) (moderate L), N=200,
#               5 instances, ε=1e-6, maxiter=50000.
#
# Bundle: A3 + B1 + C1 + D1 + E2 (decided 2026-04-25).
#
# Usage:
#   julia --project=. scripts/s12_sensitivity.jl
#   julia --project=. scripts/s12_sensitivity.jl --dim 200
#   julia --project=. scripts/s12_sensitivity.jl --skip-eps    # only run Phase A
#   julia --project=. scripts/s12_sensitivity.jl --skip-grid   # only run Phase B
#
# Output:
#   results/sensitivity/sensitivity.csv          (β̄ × θ̄ grid)
#   results/sensitivity/epsilon_scale.csv        (ε-scale check)
#   results/sensitivity/log_s12_sensitivity_*.txt

include("../src/includes.jl")
using Statistics: mean, std

# ── Problem: VIP with x* = 0, moderate L (W ~ U(0,1)) ──────────────
function setup_example1(n::Int; seed = 2025, num_of_instances = 5)
    rng = Xoshiro(seed)
    U = Uniform(0, 1)
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
        problems[i] = Problem(; name = "Instance $i (L=$(round(L,digits=1)))",
            Aλ = A_resolvent, A = x -> A_matrix * x, B = B, L = L,
            x0 = x0, x1 = x1, n = n)
    end
    return problems
end

# Run the algorithm on every problem instance and aggregate.
function run_one_config(problems, params_for_L; tol, maxiter)
    iters_list = Int[]
    times_list = Float64[]
    converged_count = 0
    for prob in problems
        L = prob.L
        params = params_for_L(L)
        t0 = time()
        sol = DIPCM(prob; params..., tol = tol, maxiter = maxiter)
        elapsed = time() - t0
        push!(iters_list, sol.iterations)
        push!(times_list, elapsed)
        sol.converged && (converged_count += 1)
    end
    return (
        avg_iter = mean(iters_list),
        std_iter = std(iters_list),
        avg_time = mean(times_list),
        std_time = std(times_list),
        conv_rate = converged_count / length(problems),
    )
end

function main()
    opts, _ = parse_args(ARGS)
    dim = parse(Int, get(opts, "dim", "200"))
    maxiter = parse(Int, get(opts, "maxiter", "50000"))
    skip_eps = any(x -> x == "--skip-eps", ARGS)
    skip_grid = any(x -> x == "--skip-grid", ARGS)
    tol = 1e-6
    seed = 2025
    num_of_instances = 5

    # ── Locked defaults (Phase B uses these, not grid argmin) ─
    β_locked = 0.3
    θ_locked = 0.9

    # ── Grid (Phase A) ─────────────────────────────────
    β_bar_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    θ_bar_values = [0.5, 0.7, 0.9, 0.95]
    # ── ε-scale (Phase B) ──────────────────────────────
    ε_scales = [0.01, 0.1, 1.0, 10.0, 100.0]

    output_dir = "results/sensitivity"
    mkpath(output_dir)
    grid_csv_path = joinpath(output_dir, "sensitivity.csv")
    eps_csv_path = joinpath(output_dir, "epsilon_scale.csv")

    logpath, tee, logfile = setup_logging("s12_sensitivity"; logdir = output_dir)

    println(tee, "="^72)
    println(tee, "DIPCM SENSITIVITY (paper §3.4 algorithm)")
    println(tee, "  x_{n+1} = (1 - α_n - σ_n) z_n + σ_n u_n")
    println(tee, "  Fixed: α_n = 1/(n+1),  σ_n = 0.8 - α_n,")
    println(tee, "         ε_n = 5.0/(n+1)^2.1,  ε'_n = 10.0/(n+1)^2.1,")
    println(tee, "         γ = 1.1, μ = 0.5, λ_1 = 1/(1.05L)")
    println(tee, "="^72)
    println(tee, "  Dimension: $dim, Instances: $num_of_instances")
    println(tee, "  MaxIter:   $maxiter, Tol: $tol")
    println(tee, "  β̄ values:  $β_bar_values")
    println(tee, "  θ̄ values:  $θ_bar_values")
    if !skip_eps
        println(tee, "  ε scales:  $ε_scales (Phase B)")
    end
    println(tee, "  Phase B cell (locked): β̄ = $β_locked, θ̄ = $θ_locked")
    println(tee, "="^72)
    flush(tee)

    problems = setup_example1(dim; seed = seed, num_of_instances = num_of_instances)

    # ════════════════════════════════════════════════════
    # Phase A — β̄ × θ̄ grid
    # ════════════════════════════════════════════════════
    grid_rows = []
    if skip_grid
        println(tee, "\n--skip-grid set; Phase A not run.")
        flush(tee)
    else
        println(tee, "\n── Phase A: β̄ × θ̄ grid ─────────────────────────")
        flush(tee)
        for θ̄ in θ_bar_values
            println(tee)
            println(tee, "  θ̄ = $θ̄")
            for β̄ in β_bar_values
                params_for_L = L -> get_DIPCM_params(L; β_bar = β̄, θ_bar = θ̄, λ0 = 1 / (1.05 * L))
                r = run_one_config(problems, params_for_L; tol = tol, maxiter = maxiter)
                push!(grid_rows, (
                    β_bar = β̄, θ_bar = θ̄,
                    avg_iterations = r.avg_iter, std_iterations = r.std_iter,
                    avg_time = r.avg_time, std_time = r.std_time,
                    convergence_rate = r.conv_rate,
                    dim = dim, num_instances = num_of_instances,
                ))
                mark = r.conv_rate == 1.0 ? "✓" : "✗"
                println(tee, @sprintf("    %s  β̄=%.2f  →  iter %7.1f (±%6.1f)  time %.4fs  conv=%.0f%%",
                    mark, β̄, r.avg_iter, r.std_iter, r.avg_time, 100 * r.conv_rate))
                flush(tee)
                CSV.write(grid_csv_path, DataFrame(grid_rows))
            end
        end

        # Report grid argmin for context (not used for Phase B)
        converged_grid = filter(r -> r.convergence_rate == 1.0, grid_rows)
        if !isempty(converged_grid)
            argmin_cell = argmin(r -> r.avg_iterations, converged_grid)
            println(tee)
            println(tee, "─"^72)
            println(tee, @sprintf("  Grid argmin (informational): β̄ = %.2f, θ̄ = %.2f  →  avg %.1f iters  (±%.1f)",
                argmin_cell.β_bar, argmin_cell.θ_bar, argmin_cell.avg_iterations, argmin_cell.std_iterations))
            println(tee, @sprintf("  Phase B will run at LOCKED cell: β̄ = %.2f, θ̄ = %.2f", β_locked, θ_locked))
            println(tee, "─"^72)
        else
            println(tee, "\nWARNING: No β̄ × θ̄ configuration achieved 100% convergence in Phase A.")
        end
    end
    flush(tee)

    if skip_eps
        println(tee, "\n--skip-eps set; Phase B not run.")
        println(tee, "Grid CSV: $grid_csv_path")
        teardown_logging(tee, logpath)
        return
    end

    # ════════════════════════════════════════════════════
    # Phase B — ε-scale check at the LOCKED defaults
    # ════════════════════════════════════════════════════
    println(tee, @sprintf("\n── Phase B: ε-scale robustness at LOCKED β̄=%.2f, θ̄=%.2f ─────────",
        β_locked, θ_locked))
    flush(tee)

    # Phase B scales are relative to the new defaults: ε_n = 5/(n+1)^2.1, ε'_n = 10/(n+1)^2.1
    # so scale = 1.0 reproduces the baseline result at (β_locked, θ_locked).
    eps_rows = []
    for c in ε_scales
        ε_seq  = n -> c * 5.0 / (n + 1)^2.1
        εp_seq = n -> c * 10.0 / (n + 1)^2.1
        params_for_L = L -> get_DIPCM_params(L;
            β_bar = β_locked, θ_bar = θ_locked,
            ε_seq = ε_seq, εp_seq = εp_seq,
            λ0 = 1 / (1.05 * L))
        r = run_one_config(problems, params_for_L; tol = tol, maxiter = maxiter)
        push!(eps_rows, (
            scale = c, β_bar = β_locked, θ_bar = θ_locked,
            avg_iterations = r.avg_iter, std_iterations = r.std_iter,
            avg_time = r.avg_time, std_time = r.std_time,
            convergence_rate = r.conv_rate,
            dim = dim, num_instances = num_of_instances,
        ))
        mark = r.conv_rate == 1.0 ? "✓" : "✗"
        println(tee, @sprintf("  %s  scale = %7.2f  →  iter %7.1f (±%6.1f)  conv=%.0f%%",
            mark, c, r.avg_iter, r.std_iter, 100 * r.conv_rate))
        flush(tee)
        CSV.write(eps_csv_path, DataFrame(eps_rows))
    end

    println(tee)
    println(tee, "Grid CSV:    $grid_csv_path")
    println(tee, "ε-scale CSV: $eps_csv_path")
    teardown_logging(tee, logpath)
end

main()
