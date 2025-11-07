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
		solution = sol.solution,
		iterations = sol.iterations,
		converged = sol.converged,
		parameters = sol.parameters,
		messages = sol.messages,
		time = elapsed_time)
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
	all_results = []

	# Progress bar over all runs (errors × algorithms)
	N_ERROR = length(errors)
	N_ALGORITHMS = length(algorithms)
	N_DIMS = length(dims)
	total_tasks = N_ERROR * N_ALGORITHMS * N_DIMS * num_of_instances
	p = show_progress ? Progress(total_tasks; desc = @sprintf("Comparing (n=%s)", join(dims, ","))) : nothing
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
					(; solution, iterations, converged, time, messages) = sol
					x, iter, converged = solution, iterations, converged
					# Store results
					algo_name = algorithm_names[i]

					push!(all_results,
						(algo_name = algo_name, dim = n,
							problem_index = problem.name, error = err, time = time, iter = iter, converged = converged,
							lambda = get(sol.parameters, :λ1, ""),
							messages = messages,
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
	header = ["Algorithm", "Dimension", "Problem Instance", "Error", "Time", "Iter", "Converged", "lambda1"]
	results = map(x -> x[filter(y -> y != :messages, keys(x))], results)
	# Save
	writedlm(filename, vcat([header], results), ',')
	println("\nResults saved to $filename")
end

function startSolvingExample(title::String, algorithms::Vector, example_setup, dims::Vector{Int};
	errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
	seed = 2025,
	num_of_instances = 1,
	maxiter = 50000,
	verbose::Bool = false,
	show_progress::Bool = true, clearfolder::Bool = false,
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
	ex1_messages_file = prepare_filepath("results/$(title)/messages.txt", dated = true)
	ex1_all_file = prepare_filepath("results/$(title)/all_comparisons.xlsx", dated = true)

	save_comparison_results(results, ex1_ns_file)
	writedlm(ex1_messages_file, map(x -> x[end], results), ',')

	csv_to_xlsx(ex1_ns_file, ex1_all_file, overwrite = true)
	ex1_ns_file
end

function plotProfiles(title, datafile, tag)
	ex1_plot_file_time = prepare_filepath("results/$(title)/profile_$tag.png", dated = true)
	plt = performance_profile_from_csv(datafile; tag = tag, savepath = ex1_plot_file_time)
end
