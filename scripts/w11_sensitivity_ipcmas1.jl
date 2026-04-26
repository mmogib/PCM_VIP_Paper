## w11_sensitivity_ipcmas1.jl — Sensitivity analysis for IPCMAS1 parameters
#
# Tests IPCMAS1: x_{n+1} = (1-α) z_n + α u_n   with
#   z_n = x_n + β (x_n - x_{n-1}),  w_n = x_n + θ_n (x_n - x_{n-1}),
#   y_n = J^A_λ(w_n - λ B(w_n)),   u_n = w_n - γ η_n d_n,
#   λ_{n+1} = min{μ ‖w_n-y_n‖/‖B(w_n)-B(y_n)‖, λ_n + ξ_n},
#   ξ_n = c/(n+1)^1.1.
#
# Tolerance = 1e-3 (weak-convergence regime).
# Fixed:  γ = 1.8, μ = 0.5, λ₁ = 1/(1.05·L)
# Swept:  α (convex weight), β_fraction (of β_max(α)), θ̄ (constant θ), c (ξ scale)
#
# Feasibility: α ∈ (0, 1/3),  β ∈ [0, β_max(α) = (1-3α)/(3(1-α)))
# We sweep β_fractions ∈ {0.0, 0.25, 0.5, 0.9}; β = β_fraction · β_max(α) stays
# strictly inside the feasibility bound.
#
# Resume: rows already in the CSV are skipped unless --force is passed.
#
# Usage:
#   julia --project=. scripts/w11_sensitivity_ipcmas1.jl
#   julia --project=. scripts/w11_sensitivity_ipcmas1.jl --dim=200
#   julia --project=. scripts/w11_sensitivity_ipcmas1.jl --force
#
# Output:
#   results/sensitivity/ipcmas1_sensitivity_N{dim}_eps1em3.csv
#   results/sensitivity/log_w11_*.txt

include("../src/includes.jl")
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

# β_max(α) = (1 - 3α) / (3(1 - α)) — feasibility bound from Assumption (A4)
β_max(α) = (1 - 3α) / (3 * (1 - α))

function load_done_set(csv_path::String)
    done = Set{NTuple{5,Any}}()  # (α, β_frac, θ, c, instance_idx)
    isfile(csv_path) || return done
    try
        df = CSV.read(csv_path, DataFrame)
        for row in eachrow(df)
            push!(done, (row.α, row.β_fraction, row.θ, row.c, row.instance))
        end
    catch
    end
    return done
end

function append_row(csv_path::String, row::NamedTuple, need_header::Bool)
    open(csv_path, "a") do f
        if need_header
            println(f, join(string.(keys(row)), ","))
        end
        println(f, join([string(v) for v in values(row)], ","))
        flush(f)
    end
end

function main()
    opts, _ = parse_args(ARGS)
    dim = parse(Int, get(opts, "dim", "200"))
    maxiter = parse(Int, get(opts, "maxiter", "20000"))
    tol = 1e-3
    seed = 2025
    num_of_instances = 5
    force = any(x -> x == "--force", ARGS)

    # Swept grid
    α_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    β_fractions = [0.0, 0.25, 0.5, 0.9]
    θ_values = [0.3, 0.5, 0.7, 0.9, 0.99]
    c_values = [1.0, 10.0, 100.0]

    # Fixed
    γ_fixed = 1.8
    μ_fixed = 0.5

    # Output
    output_dir = "results/sensitivity"
    mkpath(output_dir)
    csv_path = joinpath(output_dir, "ipcmas1_sensitivity_N$(dim)_eps1em3.csv")

    logpath, tee, logfile = setup_logging("w11_sensitivity_ipcmas1"; logdir=output_dir)

    println(tee, "="^70)
    println(tee, "IPCMAS1 SENSITIVITY ANALYSIS (Example 1, x* = 0)")
    println(tee, "  x_{n+1} = (1-α) z_n + α u_n,  z_n = x_n + β(x_n-x_{n-1})")
    println(tee, "  Tolerance: $tol  (weak-convergence regime)")
    println(tee, "="^70)
    println(tee, "  Dimension: $dim, Instances: $num_of_instances, Seed: $seed")
    println(tee, "  MaxIter: $maxiter")
    println(tee, "  Fixed: γ=$γ_fixed, μ=$μ_fixed, λ₁=1/(1.05·L), θ_seq(n)=θ̄ (const)")
    println(tee, "  Swept:")
    println(tee, "    α values:          $α_values")
    println(tee, "    β_fraction values: $β_fractions  (β = β_fraction · β_max(α))")
    println(tee, "    θ̄ values:          $θ_values")
    println(tee, "    c values:          $c_values  (ξ_n = c/(n+1)^1.1)")
    n_configs = length(α_values) * length(β_fractions) * length(θ_values) * length(c_values)
    println(tee, "  Total configs: $n_configs × $num_of_instances = $(n_configs * num_of_instances) runs")
    println(tee, "  CSV: $csv_path")
    force && println(tee, "  --force: re-running all configs (ignoring existing CSV)")
    println(tee, "="^70)
    flush(tee)

    # Paper default baseline: α=0.25, β=0.0001, θ̄=0.9, c=100 (from get_IPCMAS1_params)
    println(tee)
    println(tee, "  Paper default baseline: α=0.25, β=0.0001, θ̄=0.9, c=100")
    println(tee, "  (This tuple is NOT exactly in the grid; included for reference only.)")
    println(tee)
    flush(tee)

    problems = setup_example1(dim; seed=seed, num_of_instances=num_of_instances)

    # Resume
    done = force ? Set{NTuple{5,Any}}() : load_done_set(csv_path)
    n_skipped = 0
    need_header = !isfile(csv_path) || filesize(csv_path) == 0 || force
    if force && isfile(csv_path)
        rm(csv_path)
        need_header = true
    end

    total = length(α_values) * length(β_fractions) * length(θ_values) * length(c_values)
    idx = 0

    for α in α_values
        β_upper = β_max(α)
        println(tee)
        println(tee, "-"^70)
        println(tee, @sprintf("α = %.2f   (β_max = %.4f)", α, β_upper))
        println(tee, "-"^70)
        flush(tee)

        for β_frac in β_fractions
            β_val = β_frac * β_upper
            for θ in θ_values
                for c in c_values
                    idx += 1
                    a_seq = n -> c / (n + 1)^1.1

                    iters_list = Int[]
                    times_list = Float64[]
                    converged_count = 0
                    final_errs = Float64[]

                    for (inst_idx, prob) in enumerate(problems)
                        key = (α, β_frac, θ, c, inst_idx)
                        if key in done
                            n_skipped += 1
                            continue
                        end

                        L = prob.L
                        t0 = time()
                        local sol
                        try
                            sol = IPCMAS1(prob;
                                γ=γ_fixed, μ=μ_fixed,
                                λ1=1 / (1.05 * L),
                                α=α, β=β_val,
                                a_seq=a_seq,
                                θ_seq=_ -> θ,
                                tol=tol, maxiter=maxiter)
                        catch e
                            println(tee, "    ERROR α=$α β=$β_val θ=$θ c=$c inst=$inst_idx: $e")
                            flush(tee)
                            continue
                        end
                        elapsed = time() - t0

                        push!(iters_list, sol.iterations)
                        push!(times_list, elapsed)
                        sol.converged && (converged_count += 1)
                        final_err = isempty(sol.history[:err]) ? NaN : last(sol.history[:err])
                        push!(final_errs, final_err)

                        row = (
                            algorithm="IPCMAS1",
                            instance=inst_idx,
                            problem=prob.name,
                            dim=dim,
                            L=prob.L,
                            α=α,
                            β_fraction=β_frac,
                            β=β_val,
                            θ=θ,
                            c=c,
                            γ=γ_fixed,
                            μ=μ_fixed,
                            tol=tol,
                            iters=sol.iterations,
                            time=elapsed,
                            converged=sol.converged,
                            final_err=final_err,
                        )
                        append_row(csv_path, row, need_header)
                        need_header = false
                    end

                    if !isempty(iters_list)
                        conv_rate = converged_count / length(iters_list)
                        avg_iter = mean(iters_list)
                        std_iter = length(iters_list) > 1 ? std(iters_list) : 0.0
                        mark = conv_rate == 1.0 ? "✓" : "✗"
                        println(tee, @sprintf("  [%4d/%4d] %s α=%.2f β_frac=%.2f (β=%.4f) θ=%.2f c=%3.0f → avg_iter=%7.1f (±%6.1f) conv=%.0f%%",
                            idx, total, mark, α, β_frac, β_val, θ, c,
                            avg_iter, std_iter, 100 * conv_rate))
                    else
                        println(tee, @sprintf("  [%4d/%4d] — α=%.2f β_frac=%.2f θ=%.2f c=%3.0f  (all %d instances already done)",
                            idx, total, α, β_frac, θ, c, num_of_instances))
                    end
                    flush(tee)
                end
            end
        end
    end

    println(tee)
    println(tee, "="^70)
    println(tee, "SUMMARY")
    println(tee, "="^70)
    println(tee, "  Skipped (resumed): $n_skipped runs")
    println(tee, "  CSV: $csv_path")
    println(tee)

    # Rank top 10 configs with 100% convergence
    if isfile(csv_path) && filesize(csv_path) > 0
        df = CSV.read(csv_path, DataFrame)
        grouped = groupby(df, [:α, :β_fraction, :θ, :c])
        ranking = DataFrame(
            α=Float64[], β_fraction=Float64[], β=Float64[], θ=Float64[], c=Float64[],
            avg_iter=Float64[], std_iter=Float64[], avg_time=Float64[], conv_rate=Float64[],
            n_instances=Int[])
        for g in grouped
            conv_rate = mean(g.converged)
            push!(ranking, (
                g.α[1], g.β_fraction[1], g.β[1], g.θ[1], g.c[1],
                mean(g.iters), length(g.iters) > 1 ? std(g.iters) : 0.0,
                mean(g.time), conv_rate, nrow(g),
            ))
        end
        full_conv = filter(r -> r.conv_rate == 1.0, ranking)
        sort!(full_conv, [:avg_iter, :std_iter])

        println(tee, "  Top 10 configurations (100% convergence, sorted by avg_iter):")
        n_show = min(10, nrow(full_conv))
        for i in 1:n_show
            r = full_conv[i, :]
            println(tee, @sprintf("    %2d. α=%.2f β_frac=%.2f (β=%.4f) θ=%.2f c=%3.0f → %7.1f iters (±%6.1f), %.4fs",
                i, r.α, r.β_fraction, r.β, r.θ, r.c, r.avg_iter, r.std_iter, r.avg_time))
        end
        if nrow(full_conv) == 0
            println(tee, "  WARNING: No configurations achieved 100% convergence.")
        end
    end

    teardown_logging(tee, logpath)
end

main()
