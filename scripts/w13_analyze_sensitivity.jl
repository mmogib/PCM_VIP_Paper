# Quick analysis of w11 IPCMAS1 sensitivity results.
# Usage: julia --project=. scripts/w13_analyze_sensitivity.jl

include("../src/includes.jl")
using Statistics: mean, std, median

csv = "results/sensitivity/ipcmas1_sensitivity_N200_eps1em3.csv"
df = CSV.read(csv, DataFrame)

println("="^70)
println("IPCMAS1 SENSITIVITY ANALYSIS — SUMMARY (Example 1, eps=1e-3, N=200)")
println("="^70)
println("Total rows: $(nrow(df))")
println("All converged: $(all(df.converged))")
println("Iter range: $(minimum(df.iters)) – $(maximum(df.iters))")
println()

# Marginal effects
function marginal(df, col)
    g = groupby(df, col)
    summary = DataFrame(
        value = Float64[], avg_iter = Float64[], std_iter = Float64[], n = Int[]
    )
    for grp in g
        push!(summary, (grp[1, col], mean(grp.iters), std(grp.iters), nrow(grp)))
    end
    sort!(summary, :value)
    return summary
end

println("── Marginal effect of α (averaged over β, θ, c, instances) ──")
m = marginal(df, :α)
for r in eachrow(m)
    println(@sprintf("  α=%.2f → %7.1f iters (±%6.1f)  [n=%d]", r.value, r.avg_iter, r.std_iter, r.n))
end
println()

println("── Marginal effect of β_fraction ──")
m = marginal(df, :β_fraction)
for r in eachrow(m)
    println(@sprintf("  β_frac=%.2f → %7.1f iters (±%6.1f)  [n=%d]", r.value, r.avg_iter, r.std_iter, r.n))
end
println()

println("── Marginal effect of θ ──")
m = marginal(df, :θ)
for r in eachrow(m)
    println(@sprintf("  θ=%.2f → %7.1f iters (±%6.1f)  [n=%d]", r.value, r.avg_iter, r.std_iter, r.n))
end
println()

println("── Marginal effect of c ──")
m = marginal(df, :c)
for r in eachrow(m)
    println(@sprintf("  c=%5.1f → %7.1f iters (±%6.1f)  [n=%d]", r.value, r.avg_iter, r.std_iter, r.n))
end
println()

# Cross-tabs: α × β_fraction
println("── Average iters: α × β_fraction (min over θ, c) ──")
g = groupby(df, [:α, :β_fraction])
tbl = DataFrame(α=Float64[], β_fraction=Float64[], avg_iter=Float64[], best_iter=Float64[])
for grp in g
    push!(tbl, (grp.α[1], grp.β_fraction[1], mean(grp.iters),
                minimum(combine(groupby(grp, [:θ, :c]), :iters => mean => :m).m)))
end
sort!(tbl, [:α, :β_fraction])
αs = sort(unique(tbl.α))
βs = sort(unique(tbl.β_fraction))
print("  α \\ β_f   "); for β in βs; print(@sprintf("%10.2f", β)); end; println()
for α in αs
    print(@sprintf("  α=%.2f     ", α))
    for β in βs
        cell = filter(r -> r.α == α && r.β_fraction == β, tbl)
        print(@sprintf("%10.1f", cell.avg_iter[1]))
    end
    println()
end
println()

# Top configs (per α — best θ,c,β_frac for each α)
println("── Best configuration per α (over β_frac, θ, c) ──")
by_α = groupby(df, :α)
for grp in by_α
    α = grp.α[1]
    grp_configs = groupby(grp, [:β_fraction, :θ, :c])
    configs = DataFrame(β_fraction=Float64[], θ=Float64[], c=Float64[], avg_iter=Float64[], std_iter=Float64[])
    for g in grp_configs
        push!(configs, (g.β_fraction[1], g.θ[1], g.c[1], mean(g.iters), std(g.iters)))
    end
    sort!(configs, :avg_iter)
    best = configs[1, :]
    println(@sprintf("  α=%.2f → β_frac=%.2f θ=%.2f c=%3.0f → %7.1f iters (±%6.1f)",
        α, best.β_fraction, best.θ, best.c, best.avg_iter, best.std_iter))
end
println()

# Global top 20
println("── Top 20 configurations (sorted by avg_iter, then std) ──")
grouped = groupby(df, [:α, :β_fraction, :θ, :c])
ranking = DataFrame(α=Float64[], β_frac=Float64[], β=Float64[], θ=Float64[], c=Float64[],
                    avg_iter=Float64[], std_iter=Float64[], avg_time=Float64[])
for g in grouped
    push!(ranking, (g.α[1], g.β_fraction[1], g.β[1], g.θ[1], g.c[1],
                    mean(g.iters), std(g.iters), mean(g.time)))
end
sort!(ranking, [:avg_iter, :std_iter])
for i in 1:min(20, nrow(ranking))
    r = ranking[i, :]
    println(@sprintf("  %2d. α=%.2f β_frac=%.2f (β=%.4f) θ=%.2f c=%3.0f → %7.1f (±%6.1f) iters, %.3fs",
        i, r.α, r.β_frac, r.β, r.θ, r.c, r.avg_iter, r.std_iter, r.avg_time))
end
