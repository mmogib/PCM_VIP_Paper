## w50_perf_profile.jl — Dolan–Moré performance profiles from comparison CSV
#
# Reads a summary comparison CSV (produced by startSolvingExample) and
# generates performance profiles:
#   (a) per-tolerance profiles (one PDF each)
#   (b) an aggregated profile pooling all tolerances as separate "problems"
# Both by iteration count and by CPU time.
#
# Non-converged entries are treated as Inf.
#
# Usage:
#   julia --project=. scripts/w50_perf_profile.jl            # all 3 weak CSVs
#   julia --project=. scripts/w50_perf_profile.jl --csv results/example_1_weak/comparison.csv
#   julia --project=. scripts/w50_perf_profile.jl --csv ... --tol 1e-3       # single tolerance only
#   julia --project=. scripts/w50_perf_profile.jl --csv ... --outdir imgs/
#
# Output:
#   {outdir}/perf_{dataset}_tol{tol}_{metric}.pdf   (per-tolerance)
#   {outdir}/perf_{dataset}_aggregate_{metric}.pdf  (aggregated)

include("../src/includes.jl")

const DEFAULT_WEAK_CSVS = [
    "results/example_1_weak/comparison.csv",
    "results/example_2_weak/comparison.csv",
    "results/example_3_weak/comparison.csv",
]

function infer_dataset_name(csv_path::AbstractString)
    # e.g. "results/example_1_weak/comparison.csv" → "example_1_weak"
    dir = dirname(csv_path)
    return basename(dir)
end

function write_filtered_csv(df::DataFrame, path::String)
    CSV.write(path, df)
    return path
end

function process_one(csv_path::AbstractString, outdir_opt::AbstractString, tol_filter::AbstractString)
    isfile(csv_path) || (println("  SKIP $csv_path (not found)"); return)

    dataset = infer_dataset_name(csv_path)
    outdir = isempty(outdir_opt) ? joinpath(dirname(csv_path), "perf_profiles") : outdir_opt
    mkpath(outdir)

    target_tols = isempty(tol_filter) ? nothing : [parse(Float64, tol_filter)]

    df = CSV.read(csv_path, DataFrame)
    all_tols = sort(unique(df.Error), rev=true)  # largest first (1e-1, 1e-2, 1e-3)
    tols_to_plot = target_tols === nothing ? all_tols : target_tols
    algorithms = unique(df.Algorithm)

    println("="^70)
    println("Performance profiles for $dataset")
    println("="^70)
    println("  CSV:        $csv_path")
    println("  Algorithms: ", join(algorithms, ", "))
    println("  Tolerances: ", join(tols_to_plot, ", "))
    println("  Output dir: $outdir")
    println()

    tmpdir = mktempdir()

    # ── (a) per-tolerance profiles ─────────────────────────────────────────
    for tol in tols_to_plot
        tol_str = @sprintf("%.0e", tol)
        df_tol = filter(r -> abs(r.Error - tol) < 1e-15, df)
        if nrow(df_tol) == 0
            println("  SKIP tol=$tol (no rows)")
            continue
        end
        # BenchmarkProfiles expects every (solver, problem) cell filled. Count
        # per-solver row counts; skip if uneven.
        counts = Dict(s => count(df_tol.Algorithm .== s) for s in algorithms)
        if length(unique(values(counts))) > 1
            println("  WARN tol=$tol: uneven rows across solvers $(counts); profile may be misleading.")
        end

        tmp_csv = joinpath(tmpdir, "filtered_tol$(tol_str).csv")
        write_filtered_csv(df_tol, tmp_csv)

        for metric in ("Time", "Iter")
            outfile = joinpath(outdir, "perf_$(dataset)_tol$(tol_str)_$(lowercase(metric)).pdf")
            try
                performance_profile_from_csv(tmp_csv;
                    tag=metric,
                    solvers=String.(algorithms),
                    treat_nonconverged_as_inf=true,
                    savepath=outfile,
                    titled=false,
                )
                println("  ✓ tol=$tol  $metric → $outfile")
            catch e
                println("  ✗ tol=$tol  $metric: $e")
            end
        end
    end

    # ── (b) aggregated profile: pool all tolerances as separate problems ───
    # Relabel each (problem, tol) pair into a synthetic problem name so each
    # tolerance counts as a distinct "problem" in Dolan–Moré.
    if target_tols === nothing && length(tols_to_plot) > 1
        df_agg = copy(df)
        df_agg[!, "Problem Instance"] = string.(df_agg[!, "Problem Instance"]) .* "_tol" .* string.(df_agg.Error)

        tmp_csv = joinpath(tmpdir, "aggregated.csv")
        write_filtered_csv(df_agg, tmp_csv)

        for metric in ("Time", "Iter")
            outfile = joinpath(outdir, "perf_$(dataset)_aggregate_$(lowercase(metric)).pdf")
            try
                performance_profile_from_csv(tmp_csv;
                    tag=metric,
                    solvers=String.(algorithms),
                    treat_nonconverged_as_inf=true,
                    savepath=outfile,
                    titled=false,
                )
                println("  ✓ aggregate  $metric → $outfile")
            catch e
                println("  ✗ aggregate  $metric: $e")
            end
        end
    end

    rm(tmpdir; recursive=true, force=true)
    println()
end

function main()
    opts, _ = parse_args(ARGS)
    csv_opt = get(opts, "csv", "")
    outdir_opt = get(opts, "outdir", "")
    tol_filter = get(opts, "tol", "")

    csv_paths = if isempty(csv_opt)
        DEFAULT_WEAK_CSVS
    else
        [csv_opt]
    end

    for csv_path in csv_paths
        process_one(csv_path, outdir_opt, tol_filter)
    end
    println("Done.")
end

main()
