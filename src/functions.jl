function solve_problem(algorithm, problem::Problem, params::NamedTuple;
    tol=1e-5, maxiter=50000, verbose::Bool=false)

    if verbose
        println(@sprintf("  -> starting: tol=%.1e, maxiter=%d", tol, maxiter))
    end

    elapsed_time = @elapsed begin
        sol = algorithm(
            problem;
            params...,
            tol=tol,
            maxiter=maxiter,
        )
        # (; solution, iterations, converged) = sol
        # x, iter, converged = solution, iterations, converged
        sol
    end

    if verbose
        println(@sprintf("  -> finished: time=%.4fs, iter=%d, converged=%s",
            elapsed_time, iter, string(converged)))
    end
    # return x, iter, converged, elapsed_time
    return Solution(;
        solver=sol.solver,
        problem=sol.problem,
        solution=sol.solution,
        iterations=sol.iterations,
        converged=sol.converged,
        parameters=sol.parameters,
        time=elapsed_time,
        history=sol.history,
    )
end



"""
Build a Set of (algo, dim, problem, error) keys from an existing CSV file for resume.
Returns empty set if file doesn't exist or is empty.
"""
function load_done_set(csv_path::String)
    done = Set{Tuple{String,Int,String,Float64}}()
    isfile(csv_path) || return done
    try
        df = CSV.read(csv_path, DataFrame)
        for row in eachrow(df)
            push!(done, (row.Algorithm, row.Dimension, row[Symbol("Problem Instance")], row.Error))
        end
    catch
    end
    return done
end

function generate_comparison_table(algorithms::Vector,
    example_setup, dims::Vector{Int};
    errors=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    seed=2025,
    num_of_instances=1,
    maxiter=50000,
    verbose::Bool=false,
    show_progress::Bool=true,
    io::IO=stdout,
    resume_file::String="",
    force::Bool=false,
    save_history::Bool=false,
    history_dir::String="",
)
    algorithm_names = first.(algorithms)
    done_set = (isempty(resume_file) || force) ? Set() : load_done_set(resume_file)
    n_skipped = 0

    N_ERROR = length(errors)
    N_ALGORITHMS = length(algorithms)
    N_DIMS = length(dims)
    total_tasks = N_ERROR * N_ALGORITHMS * N_DIMS * num_of_instances
    p = show_progress ? Progress(total_tasks; desc=@sprintf("Comparing (n=%s)", join(dims, ","))) : nothing
    all_results = []

    if !show_progress
        println(io, "\n" * "="^(50 + 25 * length(algorithms)))
        println(io, "Comparison Table: n = $(join(dims, ","))")
        println(io, "="^(50 + 25 * length(algorithms)))
        println(io)
    else
        println(io, @sprintf("\nComparing algorithms (n=%s) ...", join(dims, ",")))
    end
    io isa TeeIO && flush(io)

    for n in dims
        problems = example_setup(n, seed=seed, num_of_instances=num_of_instances)
        for (problem_index, problem) in enumerate(problems)
            if !show_progress
                print(io, @sprintf("%-10s", "Error"))
                for name in algorithm_names
                    print(io, @sprintf(" | %-20s", name))
                end
                println(io)
                print(io, @sprintf("%-10s", ""))
                for _ in algorithm_names
                    print(io, @sprintf(" | %-9s %-9s", "Time", "No. It."))
                end
                println(io)
                println(io, "-"^(50 + 25 * length(algorithms)))
            end

            for err in errors
                if !show_progress
                    print(io, @sprintf("10^(%d)   ", Int(log10(err))))
                end
                for (i, (_, algo_func, param_getter)) in enumerate(algorithms)
                    algo_name = algorithm_names[i]
                    key = (algo_name, n, problem.name, err)

                    # Resume: skip if already done
                    if key in done_set
                        n_skipped += 1
                        if show_progress
                            next!(p; showvalues=[(:status, "skipped"), (:algo, algo_name), (:dim, n), (:tol, err)])
                        end
                        continue
                    end

                    params = param_getter(problem.L)

                    if verbose && !show_progress
                        println(io, @sprintf("Running %-10s at tol=%.1e", algo_name, err))
                    end

                    local iterations, converged, time_elapsed
                    try
                        sol = solve_problem(
                            algo_func, problem, params;
                            tol=err, maxiter=maxiter, verbose=verbose,
                        )
                        iterations = sol.iterations
                        converged = sol.converged
                        time_elapsed = sol.time

                        push!(all_results, (
                            algo_name=algo_name, dim=n, problem_name=problem.name,
                            error=err, time=time_elapsed, iter=iterations, converged=converged,
                            lambda=get(sol.parameters, :λ1, ""),
                            history=sol.history, full_solution=sol,
                        ))

                        if !isempty(resume_file)
                            _save_one_row(resume_file, algo_name, n, problem.name, err, time_elapsed, iterations, converged, get(sol.parameters, :λ1, ""))
                        end

                        if save_history && !isempty(history_dir)
                            try
                                _save_history_csv(history_dir, algo_name, n, problem.name, err, sol.history; overwrite=force)
                            catch e
                                println(io, "\n  WARN: failed to save history for $algo_name ($(problem.name), tol=$err): $e")
                            end
                        end
                    catch e
                        println(io, "\n  ERROR: $algo_name on $(problem.name) at tol=$err: $e")
                        iterations = -1
                        converged = false
                        time_elapsed = 0.0
                    end

                    if !show_progress
                        status = converged ? "" : (iterations == -1 ? "E" : "*")
                        print(io, @sprintf(" | %.4f    %5d%s", time_elapsed, iterations, status))
                    end

                    if show_progress
                        next!(p; showvalues=[
                            (:instance, problem.name), (:dim, n), (:algo, algo_name), (:tol, err), (:iter, iterations),
                            (:max_iters, maxiter),
                        ])
                    end
                end

                if !show_progress
                    println(io)
                end
            end
        end
    end

    if !show_progress
        println(io, "="^(50 + 25 * length(algorithms)))
        println(io, "* = Did not converge")
    else
        println(io, @sprintf("Finished comparisons (n=%s). Skipped %d already-done configs.", join(dims, ","), n_skipped))
    end
    io isa TeeIO && flush(io)
    return all_results
end

"""Append one result row to CSV (for incremental resume)."""
function _save_one_row(filepath, algo, dim, problem, err, time, iter, converged, lambda)
    header = ["Algorithm", "Dimension", "Problem Instance", "Error", "Time", "Iter", "Converged", "lambda1"]
    needs_header = !isfile(filepath) || filesize(filepath) == 0
    open(filepath, "a") do f
        if needs_header
            println(f, join(header, ","))
        end
        println(f, join([algo, dim, problem, err, time, iter, converged, lambda], ","))
        flush(f)
    end
end

"""
Slugify a string for safe use in filenames (replace non-alnum with `_`, trim).
"""
_slugify(s::AbstractString) = replace(strip(s), r"[^A-Za-z0-9]+" => "_")

"""
Write per-iteration history to CSV. Aligns all vectors to length of `:err`
(which is always pushed on every iter, including the terminal break).
Shorter vectors are padded with NaN; longer vectors are truncated.
Skips if file already exists.

File path:
  {history_dir}/{algo}_N{dim}_{problem_slug}_tol{err_sci}.csv
"""
function _save_history_csv(history_dir::String, algo::String, dim::Int,
    problem_name::String, err::Float64, history::Dict{Symbol,Vector{<:Real}};
    overwrite::Bool=false)
    mkpath(history_dir)
    prob_slug = _slugify(problem_name)
    err_str = @sprintf("%.0e", err)  # e.g. "1e-06"
    filepath = joinpath(history_dir, "$(algo)_N$(dim)_$(prob_slug)_tol$(err_str).csv")
    (!overwrite && isfile(filepath)) && return filepath   # resume: skip if exists

    # Determine reference length from :err; fall back to max length among present keys
    N = haskey(history, :err) ? length(history[:err]) :
        (isempty(history) ? 0 : maximum(length(v) for v in values(history)))
    N == 0 && return filepath  # nothing to write

    # Fixed column order; include only keys that are populated
    preferred = [:err, :xk, :dk, :lambda, :eta, :x_norm, :wy_norm, :t_iter]
    keys_present = [k for k in preferred if haskey(history, k) && !isempty(history[k])]
    # Also include any extra keys not in preferred, in sorted order
    extras = sort([k for k in keys(history) if !(k in preferred) && !isempty(history[k])])
    cols = vcat(keys_present, extras)

    open(filepath, "w") do f
        println(f, "iter," * join(string.(cols), ","))
        for i in 1:N
            vals = String["$i"]
            for k in cols
                v = history[k]
                push!(vals, i <= length(v) ? string(v[i]) : "NaN")
            end
            println(f, join(vals, ","))
        end
    end
    return filepath
end

function save_comparison_results(results::Vector, filename::String; io::IO=stdout)
    header = [
        (:algo_name, "Algorithm"),
        (:dim, "Dimension"),
        (:problem_name, "Problem Instance"),
        (:error, "Error"),
        (:time, "Time"),
        (:iter, "Iter"),
        (:converged, "Converged"),
        (:lambda, "lambda1"),
    ]
    headers_symbols = first.(header)
    headers_names = last.(header)
    results = map(x -> x[filter(y -> y in headers_symbols, keys(x))], results)
    writedlm(filename, vcat([headers_names], results), ',')
    println(io, "\nResults saved to $filename")
end

function startSolvingExample(title::String, algorithms::Vector, example_setup, dims::Vector{Int};
    errors=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    seed=2025,
    num_of_instances=1,
    maxiter=50000,
    verbose::Bool=false,
    show_progress::Bool=true, clearfolder::Bool=false,
    plotit=true,
    plot_comparizon=true,
    plot_convergence=(results) -> println("No ploting convergence provided"),
    convergence_dims::Union{Nothing,Vector{Int}}=nothing,
    io::IO=stdout,
    force::Bool=false,
    save_history::Bool=false,
)
    println(io, "\n" * "="^70)
    println(io, "$(uppercase(title)): Algorithm Comparison")
    println(io, "="^70)
    io isa TeeIO && flush(io)

    title_clean = replace(title, " " => "_")
    resume_csv = prepare_filepath("results/$(title_clean)/comparison_incremental.csv", dated=false)
    history_dir = save_history ? joinpath("results", title_clean, "history") : ""
    if save_history
        mkpath(history_dir)
        println(io, "  save_history=true → per-iter CSVs → $history_dir")
    end

    results = generate_comparison_table(
        algorithms, example_setup, dims,
        errors=errors,
        seed=seed,
        num_of_instances=num_of_instances,
        maxiter=maxiter,
        verbose=verbose,
        show_progress=show_progress,
        io=io,
        resume_file=resume_csv,
        force=force,
        save_history=save_history,
        history_dir=history_dir,
    )

    if clearfolder
        clear_folder_recursive("results/$title_clean"; clearSubfolders=false)
    end

    ex1_ns_file = prepare_filepath("results/$(title_clean)/comparison.csv", dated=false)
    ex1_all_file = prepare_filepath("results/$(title_clean)/comparisons.xlsx", dated=false)

    # Use incremental CSV as authoritative source (contains both resumed + new results)
    if isfile(resume_csv)
        cp(resume_csv, ex1_ns_file; force=true)
        println(io, "\nResults saved to $ex1_ns_file (from incremental resume file)")
    else
        save_comparison_results(results, ex1_ns_file; io=io)
    end

    csv_to_xlsx(ex1_ns_file, ex1_all_file, overwrite=true, sheet_name="all_results")
    if plotit
        plotProfiles(title_clean, ex1_ns_file, "Time")
        plotProfiles(title_clean, ex1_ns_file, "Iter")
    end
    solutions = map(x -> x[:full_solution], results)
    if plot_comparizon && length(solutions) >= 2
        savepath = prepare_filepath("results/$(title_clean)/$(title_clean)_comparizon_plot.png", dated=false)
        plt = compare_plot(solutions[1].solution, solutions[2].solution)
        savefig(plt, savepath)
    end

    plot_convergence(results)

    ex1_ns_file, solutions
end

function plotProfiles(title, datafile, tag)
    ex1_plot_file_time = prepare_filepath("results/$(title)/$(title)_profile_$tag.png", dated=false)
    plt = performance_profile_from_csv(datafile; tag=tag, savepath=ex1_plot_file_time)
end


function compare_plot(x1::AbstractVector, x2::AbstractVector)
    n = length(x1)
    @assert length(x2) == n
    t = range(0, 1, length=n)

    p1 = plot(t, x1; label="x1", lw=2)
    p2 = plot(t, x2; label="x2", lw=2)
    p3 = plot(t, x1 .- x2; label="x1 - x2", lw=2)

    plot(p1, p2, p3; layout=(3, 1), legend=:topright)
end

"""
make_convergence_plot(series; savepath=nothing)

`series` :: Vector of (label::String, values::Vector{<:Real})

Example:
	series = [
		("ALGO1", algo1),
		("ALGO2", algo2),
		("ALGO3", algo3),
		("ALGO4", algo4),
		("ALGO5", algo5),
	]
"""
function make_convergence_plot(series::Vector{Tuple{String,Vector{T}}}; xlabel::String="x", ylabel::AbstractString="y", ylimits=(1e-8, 1e0), plothwargs...) where {T<:Real}

    # line / marker styles (cycling)
    colors = [:blue, :red, :green, :black, :orange, :purple, :brown]
    markers = [:circle, :utriangle, :star5, :diamond, :rect, :xcross]
    lstyles = [:solid, :dash, :dot, :dashdot]
    ymin, ymax = ylimits
    plt = plot(; yscale=:log10, dpi=300,
        legend=:topright,
        gridalpha=0.3,
        framestyle=:box)

    for (i, (label, vals)) in enumerate(series)
        vl = length(vals)

        # Determine step size based on length
        step = if vl <= 10
            1
        elseif vl <= 20
            2
        elseif vl <= 30
            5
        elseif vl <= 100
            10
        elseif vl <= 200
            20
        elseif vl <= 300
            50
        else
            100
        end

        # Sample indices from the data
        sampled_indices = 1:step:vl
        sampled_vals = vals[sampled_indices]

        # X-axis labels (actual iteration numbers, 0-indexed)
        kk = sampled_indices .- 1

        # X-axis positions for plotting
        k = 0:length(sampled_vals)-1

        mk = markers[(i-1)%length(markers)+1]
        c = colors[(i-1)%length(colors)+1]
        ls = lstyles[(i-1)%length(lstyles)+1]

        plot!(plt, k, sampled_vals;
            color=c,
            lw=1.5,
            linestyle=ls,
            marker=mk,
            ms=2,
            markevery=30,
            markerstrokecolor=c,
            markerstrokewidth=0.5,
            alpha=0.9,
            label=label,
            xticks=(k, kk),
            plothwargs...
        )
    end

    xlabel!(plt, xlabel)
    ylabel!(plt, ylabel)
    ylims!(plt, ymin, ymax)

    return plt
end
