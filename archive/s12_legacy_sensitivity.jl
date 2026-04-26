## s12_legacy_sensitivity.jl — Sensitivity analysis for DIPCM_legacy (archived)
#
# Archived 2026-04-26 at end of Phase 5. Sweeps the legacy DIPCM parameter
# space (α × β_zn × θ̄) on Example 1 (VIP, x* = 0) with moderate L (W ~ U(0,1)).
# Calls DIPCM_legacy from this same archive directory.
# Preserved for reproducibility of the pre-2026-04-25 sensitivity table.
# The new DIPCM (paper §3.4) is swept by `../scripts/s12_sensitivity.jl`.
#
# Usage:
#   julia --project=. archive/s12_legacy_sensitivity.jl
#   julia --project=. archive/s12_legacy_sensitivity.jl --dim 200
#
# Output:
#   results/sensitivity_legacy/dipcm_sensitivity_N{dim}.csv
#   results/sensitivity_legacy/log_s12_legacy_sensitivity_*.txt

include("../src/includes.jl")
include("dipcm_legacy.jl")           # DIPCM_legacy + get_DIPCM_legacy_params
using Statistics: mean, std

# ── Problem: VIP with x* = 0, moderate L (W ~ U(0,1)) ──────────────
function setup_example1(n::Int; seed=2025, num_of_instances=5)
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
        problems[i] = Problem(; name="Instance $i (L=$(round(L,digits=1)))", Aλ=A_resolvent,
            A=x -> A_matrix * x, B=B, L=L, x0=x0, x1=x1, n=n)
    end
    return problems
end

function main()
    opts, _ = parse_args(ARGS)
    dim = parse(Int, get(opts, "dim", "200"))
    maxiter = parse(Int, get(opts, "maxiter", "50000"))
    tol = 1e-6
    seed = 2025
    num_of_instances = 5

    # Fixed
    β_decay = n -> 1.0 / (n + 1)

    # Swept
    α_values = [0.2, 0.05, 0.1, 0.15, 0.25, 0.3]
    θ_values = [0.9, 0.7, 0.5, 0.95]
    β_zn_values = [0.0, 0.1, 0.2, 0.3, 0.5]

    # Output (legacy results in their own subdirectory; new DIPCM uses results/sensitivity/)
    output_dir = "results/sensitivity_legacy"
    mkpath(output_dir)
    csv_path = prepare_filepath(joinpath(output_dir, "dipcm_sensitivity_N$(dim).csv"); dated=true)

    logpath, tee, logfile = setup_logging("s12_legacy_sensitivity"; logdir=output_dir)

    println(tee, "="^70)
    println(tee, "DIPCM SENSITIVITY ANALYSIS (Example 1, x* = 0)")
    println(tee, "  x_{n+1} = α z_n + σ_n u_n,  σ_n = (1-α) - β_n")
    println(tee, "  β_n = 1/(n+1),  contraction toward origin")
    println(tee, "="^70)
    println(tee, "  Dimension: $dim, Instances: $num_of_instances")
    println(tee, "  MaxIter: $maxiter, Tol: $tol")
    println(tee, "  α values: $α_values")
    println(tee, "  θ̄ values: $θ_values")
    println(tee, "  β_zn values: $β_zn_values")
    println(tee, "="^70)
    flush(tee)

    problems = setup_example1(dim; seed=seed, num_of_instances=num_of_instances)

    rows = []

    for α in α_values
        println(tee)
        println(tee, "-"^70)
        println(tee, "α = $α")
        println(tee, "-"^70)
        flush(tee)

        for θ in θ_values
            for β_zn in β_zn_values
                iters_list = Int[]
                times_list = Float64[]
                converged_count = 0

                for prob in problems
                    L = prob.L
                    t0 = time()
                    sol = DIPCM_legacy(prob;
                        α=α, β_zn=β_zn, θ_bar=θ,
                        β_decay=β_decay,
                        γ=1.1, μ=0.5,
                        λ1=1 / (1.05 * L),
                        tol=tol, maxiter=maxiter)
                    elapsed = time() - t0

                    push!(iters_list, sol.iterations)
                    push!(times_list, elapsed)
                    if sol.converged
                        converged_count += 1
                    end
                end

                avg_iter = mean(iters_list)
                std_iter = std(iters_list)
                avg_time = mean(times_list)
                std_time = std(times_list)
                conv_rate = converged_count / num_of_instances

                push!(rows, (
                    α=α, θ=θ, β_zn=β_zn,
                    avg_iterations=avg_iter,
                    std_iterations=std_iter,
                    avg_time=avg_time,
                    std_time=std_time,
                    convergence_rate=conv_rate,
                    dim=dim,
                    num_instances=num_of_instances,
                ))

                mark = conv_rate == 1.0 ? "✓" : "✗"
                println(tee, @sprintf("  %s  α=%.2f θ=%.2f β_zn=%.1f → avg_iter=%7.1f (±%6.1f) conv=%.0f%%",
                    mark, α, θ, β_zn, avg_iter, std_iter, 100 * conv_rate))
                flush(tee)

                CSV.write(csv_path, DataFrame(rows))
            end
        end
    end

    # Summary
    println(tee)
    println(tee, "="^70)
    println(tee, "SUMMARY")
    println(tee, "="^70)

    converged_rows = filter(r -> r.convergence_rate == 1.0, rows)
    if !isempty(converged_rows)
        best = argmin(r -> r.avg_iterations, converged_rows)
        println(tee, @sprintf("  Best overall: α=%.2f θ=%.2f β_zn=%.1f → %.1f iters",
            best.α, best.θ, best.β_zn, best.avg_iterations))

        println(tee)
        println(tee, "  Best per α:")
        for α in sort(unique(r.α for r in converged_rows))
            α_rows = filter(r -> r.α == α, converged_rows)
            if !isempty(α_rows)
                b = argmin(r -> r.avg_iterations, α_rows)
                println(tee, @sprintf("    α=%.2f → θ=%.2f β_zn=%.1f  %5.0f iters",
                    α, b.θ, b.β_zn, b.avg_iterations))
            end
        end

        println(tee)
        println(tee, "  Best per β_zn:")
        for β_zn in sort(unique(r.β_zn for r in converged_rows))
            b_rows = filter(r -> r.β_zn == β_zn, converged_rows)
            if !isempty(b_rows)
                b = argmin(r -> r.avg_iterations, b_rows)
                println(tee, @sprintf("    β_zn=%.1f → α=%.2f θ=%.2f  %5.0f iters",
                    β_zn, b.α, b.θ, b.avg_iterations))
            end
        end
    else
        println(tee, "  WARNING: No configurations converged!")
    end

    println(tee)
    println(tee, "CSV saved to: $csv_path")
    teardown_logging(tee, logpath)
end

main()
