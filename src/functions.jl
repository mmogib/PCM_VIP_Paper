function solve_problem(algorithm, problem::Problem, params::NamedTuple;
	tol = 1e-5, maxiter = 50000, verbose::Bool = false)

	if verbose
		println(@sprintf("  -> starting: tol=%.1e, maxiter=%d", tol, maxiter))
	end

	elapsed_time = @elapsed begin
		sol = algorithm(
			problem;
			params...,
			tol = tol,
			maxiter = maxiter,
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
		solver = sol.solver,
		problem = sol.problem,
		solution = sol.solution,
		iterations = sol.iterations,
		converged = sol.converged,
		parameters = sol.parameters,
		time = elapsed_time,
		history = sol.history,
	)
end



function generate_comparison_table(algorithms::Vector,
	example_setup, dims::Vector{Int};
	errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
	seed = 2025,
	num_of_instances = 1,
	maxiter = 50000,
	verbose::Bool = false,
	show_progress::Bool = true,
)
	# Algorithm names
	algorithm_names = first.(algorithms)
	# Setup problem once

	# Store results

	# Progress bar over all runs (errors × algorithms)
	N_ERROR = length(errors)
	N_ALGORITHMS = length(algorithms)
	N_DIMS = length(dims)
	total_tasks = N_ERROR * N_ALGORITHMS * N_DIMS * num_of_instances
	p = show_progress ? Progress(total_tasks; desc = @sprintf("Comparing (n=%s)", join(dims, ","))) : nothing
	all_results = []
	# Print header (suppress table if progress shown)
	if !show_progress
		println("\n" * "="^(50 + 25 * length(algorithms)))
		println("Comparison Table: n = $(join(dims, ","))")
		println("="^(50 + 25 * length(algorithms)))
		println()
	else
		println(@sprintf("\nComparing algorithms (n=%s) ...", join(dims, ",")))
	end
	for n in dims
		problems = example_setup(n, seed = seed, num_of_instances = num_of_instances)
		for (problem_index, problem) in enumerate(problems)
			# Print column headers
			if !show_progress
				print(@sprintf("%-10s", "Error"))
				for name in algorithm_names
					print(@sprintf(" | %-20s", name))
				end
				println()

				print(@sprintf("%-10s", ""))
				for _ in algorithm_names
					print(@sprintf(" | %-9s %-9s", "Time", "No. It."))
				end
				println()
				println("-"^(50 + 25 * length(algorithms)))
			end

			# Run each error level
			for err in errors
				if !show_progress
					print(@sprintf("10^(%d)   ", Int(log10(err))))
				end
				for (i, (_, algo_func, param_getter)) in enumerate(algorithms)
					# Get algorithm-specific parameters
					params = param_getter(problem.L)

					# Solve
					if verbose && !show_progress
						println(@sprintf("Running %-10s at tol=%.1e", algorithm_names[i], err))
						println("  params = ", params)
					end

					sol = solve_problem(
						algo_func, problem, params;
						tol = err, maxiter = maxiter, verbose = verbose,
					)
					(; solution, iterations, converged, time, history) = sol
					x, iter, converged = solution, iterations, converged
					# Store results
					algo_name = algorithm_names[i]

					push!(all_results,
						(
							algo_name = algo_name,
							dim = n,
							problem_name = problem.name,
							error = err,
							time = time,
							iter = iter,
							converged = converged,
							lambda = get(sol.parameters, :λ1, ""),
							history = history,
							full_solution = sol,
						),
					)

					# Print
					if !show_progress
						status = converged ? "" : "*"
						print(@sprintf(" | %.4f    %5d%s", time, iter, status))
					end

					# Update progress bar
					# sol_quality1 = norm(x - problem.Aλ(x, 0.5))
					# sol_quality2 = norm(x - problem.Aλ(x, 10.0))
					if show_progress
						next!(p; showvalues = [
							(:λ1, get(sol.parameters, :λ1, "NotThere")),
							(:instance, problem.name), (:dim, n), (:algo, algorithm_names[i]), (:tol, err), (:iter, iter),
							(:max_iters, maxiter),
						]
						)
					end
				end

				if !show_progress
					println()
				end
			end
		end
	end

	if !show_progress
		println("="^(50 + 25 * length(algorithms)))
		println("* = Did not converge")
	else
		println(@sprintf("Finished comparisons (n=%s)", join(dims, ",")))
	end
	return all_results
end

function save_comparison_results(results::Vector, filename::String)

	# Create header

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
	# Save
	writedlm(filename, vcat([headers_names], results), ',')
	println("\nResults saved to $filename")
end

function startSolvingExample(title::String, algorithms::Vector, example_setup, dims::Vector{Int};
	errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
	seed = 2025,
	num_of_instances = 1,
	maxiter = 50000,
	verbose::Bool = false,
	show_progress::Bool = true, clearfolder::Bool = false,
	plotit = true,
	plot_comparizon = true,
	plot_convergence = true,
)

	println("\n" * "="^70)
	println("$(uppercase(title)): Algorithm Comparison")
	println("="^70)

	# Define algorithms to compare
	# Each entry is (algorithm_function, parameter_getter_function)


	# Generate Table 1 (n = 100)
	results = generate_comparison_table(
		algorithms, example_setup, dims,
		errors = errors,
		seed = seed,
		num_of_instances = num_of_instances,
		maxiter = maxiter,
		verbose = verbose,
		show_progress = show_progress,
	)
	title = replace(title, " " => "_")
	# clear previous results
	if clearfolder
		clear_folder_recursive("results/$title"; clearSubfolders = false)
	end
	# Save results
	ex1_ns_file = prepare_filepath("results/$(title)/comparison_all.csv", dated = true)
	# ex1_messages_file = prepare_filepath("results/$(title)/messages.txt", dated = true)
	ex1_all_file = prepare_filepath("results/$(title)/all_comparisons.xlsx", dated = true)

	save_comparison_results(results, ex1_ns_file)
	# writedlm(ex1_messages_file, map(x -> x[:messages], results), ',')

	csv_to_xlsx(ex1_ns_file, ex1_all_file, overwrite = true)
	if plotit
		plotProfiles(title, ex1_ns_file, "Time")
		plotProfiles(title, ex1_ns_file, "Iter")
	end
	solutions = map(x -> x[:full_solution], results)
	if plot_comparizon
		savepath = prepare_filepath("results/$(title)/comparizon_plot.png", dated = true)
		plt = compare_plot(solutions[1].solution, solutions[2].solution)
		savefig(plt, savepath)
	end
	if plot_convergence
		savepath = prepare_filepath("results/$(title)/convergence_plot.png", dated = true)
		names = map(s -> s.solver, solutions) |> unique
		series = map(name -> begin
				hist = filter(s -> s.solver == name, solutions) |> y -> map(x -> x.history[:err], y)
				(name, vcat(hist...))
			end, names)
		plt = make_convergence_plot(series, xlabel = "k", ylabel = "error")
		savefig(plt, savepath)
	end

	ex1_ns_file, solutions
end

function plotProfiles(title, datafile, tag)
	ex1_plot_file_time = prepare_filepath("results/$(title)/profile_$tag.png", dated = true)
	plt = performance_profile_from_csv(datafile; tag = tag, savepath = ex1_plot_file_time)
end


function compare_plot(x1::AbstractVector, x2::AbstractVector)
	n = length(x1)
	@assert length(x2) == n
	t = range(0, 1, length = n)

	p1 = plot(t, x1; label = "x1", lw = 2)
	p2 = plot(t, x2; label = "x2", lw = 2)
	p3 = plot(t, x1 .- x2; label = "x1 - x2", lw = 2)

	plot(p1, p2, p3; layout = (3, 1), legend = :topright)
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
function make_convergence_plot(series::Vector{Tuple{String, Vector{T}}}; xlabel::String = "x", ylabel::String = "y") where {T <: Real}

	# line / marker styles (cycling)
	colors = [:blue, :red, :green, :black, :orange, :purple, :brown]
	markers = [:circle, :utriangle, :star5, :diamond, :rect, :xcross]
	lstyles = [:solid, :dash, :dot, :dashdot]
	ymax = 1e0 #maximum(map(s -> maximum(s[2]), series)) + 0.1
	plt = plot(; yscale = :log10, dpi = 300,
		legend = :topright,
		gridalpha = 0.3,
		framestyle = :box)

	for (i, (label, vals)) in enumerate(series)
		k  = 0:length(vals)-1
		mk = markers[(i-1)%length(markers)+1]
		c  = colors[(i-1)%length(colors)+1]
		ls = lstyles[(i-1)%length(lstyles)+1]
		plot!(plt, k, vals;
			color = c,
			lw = 1.5,
			linestyle = ls,
			marker = mk,
			ms = 2,
			markevery = 30,
			markerstrokecolor = c,
			markerstrokewidth = 0.5,
			alpha = 0.9,
			label = label,
		)
	end

	xlabel!(plt, xlabel)
	ylabel!(plt, ylabel)
	ylims!(plt, 1e-8, ymax)


	return plt
end
