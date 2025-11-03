include("../includes.jl")


function setup_example1(n::Int; seed = 2025, num_of_instances = 1)
	rng = Xoshiro(seed)
	U = Uniform(1, 100)
	# Generate random matrix Z ∈ [1, 100]
	problems = Vector{Problem}(undef, num_of_instances)
	for i in 1:num_of_instances
		Z = rand(rng, U, n, n)

		# f = Z^T * Z
		B_matrix = Z' * Z

		# Lipschitz constant L = max eigenvalue of f
		L = maximum(abs.(eigvals(B_matrix)))

		# Upper triangular matrix with all entries one
		A_matrix = triu(ones(n, n))

		# Define operator f
		B(x) = B_matrix * x

		# Define resolvent of A: J^A_λ(x) = (I + λA)^(-1) x
		function A_resolvent(x, λ)
			return (I + λ * A_matrix) \ x
		end

		# Generate initial points in [0,1]
		x0 = rand(rng, n)
		x1 = rand(rng, n)
		problems[i] = Problem("Example 1 ($i)", A_resolvent, x -> A_matrix * x, B, L, x0, x1, n)
	end

	return problems
end

function solve_problem(algorithm, problem::Problem, params::NamedTuple;
	tol = 1e-5, maxiter = 50000, verbose::Bool = false)

	if verbose
		println(@sprintf("  -> starting: tol=%.1e, maxiter=%d", tol, maxiter))
	end

	elapsed_time = @elapsed begin
		sol = algorithm(
			problem.Aλ, problem.B, problem.x0, problem.x1;
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
	return Solution(sol, time = elapsed_time)
end





function generate_comparison_table(algorithms::Vector,
	algorithm_names::Vector{String},
	dims::Vector{Int};
	errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
	seed = 2025,
	num_of_instances = 1,
	maxiter = 50000,
	clear::Bool = false,
	verbose::Bool = false,
	show_progress::Bool = true,
)

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
		problems = setup_example1(n, seed = seed, num_of_instances = num_of_instances)
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
				for (i, (algo_func, param_getter)) in enumerate(algorithms)
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
					(; solution, iterations, converged, time) = sol
					x, iter, converged = solution, iterations, converged
					# Store results
					algo_name = algorithm_names[i]

					push!(all_results,
						(algo_name = algo_name, dim = n, problem_index = problem_index, error = err, time = time, iter = iter, converged = converged, lamnda = get(sol.parameters, :λ1, "")))

					# Print
					if !show_progress
						status = converged ? "" : "*"
						print(@sprintf(" | %.4f    %5d%s", time, iter, status))
					end

					# Update progress bar
					sol_quality1 = norm(x - problem.Aλ(x, 0.5))
					sol_quality2 = norm(x - problem.Aλ(x, 10.0))
					if show_progress
						next!(p; showvalues = [
							(:λ1, get(sol.parameters, :λ1, "NotThere")),
							(:Aλ1, sol_quality1),
							(:Aλ10, sol_quality2),
							(:instance, problem.name), (:dim, n), (:algo, algorithm_names[i]), (:tol, err), (:iter, iter),
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

	# Save
	writedlm(filename, vcat([header], results), ',')
	println("\nResults saved to $filename")
end

function startSolvingExample(; clearfolder::Bool = false)
	println("\n" * "="^70)
	println("Example 1: Algorithm Comparison")
	println("="^70)

	# Define algorithms to compare
	# Each entry is (algorithm_function, parameter_getter_function)
	algorithms = [
		(DeyHICPP, get_DeyHICPP_params),
		(IPCMAS1, get_IPCMAS1_params),
		(IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.5, λ0 = 1e-9)),
		(IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.5, λ0 = 1.0)),
		# (IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.6, λ0 = 1.0)),
		# (IPCMAS1, (L) -> get_IPCMAS1_params(L; λ0 = 0.5)),
		# (IPCMAS1, (L) -> get_IPCMAS1_params(L; λ0 = 0.25)),
		# (IPCMAS1, (L) -> get_IPCMAS1_params(L; λ0 = 1 / 12)),
		# (IPCMAS2, get_IPCMAS2_params),
		# Add more algorithms here as we implement them:
		# (Algorithm_1_15, get_Algorithm_1_15_params),
		# (Algorithm_1_17, get_Algorithm_1_17_params),
	]

	algorithm_names = [
		"DeyHICPP",
		"IPCMAS1(λ1=1/2L, μ=0.5)",
		"IPCMAS1(λ1=1.1920929f-7, μ=0.5)",
		"IPCMAS1(λ1=1, μ=0.5)",
		# "IPCMAS1(λ1=1, μ=0.6)",
		# "IPCMAS2",
		# "Algo (1.15)",
		# "Algo (1.17)",
	]

	# Generate Table 1 (n = 100)
	results = generate_comparison_table(
		algorithms,
		algorithm_names,
		[100, 150, 200];
		errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
		seed = 2025,
		num_of_instances = 1,
		maxiter = 10_000,
		verbose = any(x -> x in ("--verbose", "-v"), ARGS),
		show_progress = !any(x -> x == "--no-progress", ARGS),
	)

	# clear previous results
	if clearfolder
		clear_folder_recursive("results/example1"; clearSubfolders = false)
	end
	# Save results
	ex1_ns_file = prepare_filepath("results/example1/comparison_all.csv", dated = true)
	ex1_all_file = prepare_filepath("results/example1/all_comparisons.xlsx", dated = true)

	save_comparison_results(results, ex1_ns_file)

	csv_to_xlsx(ex1_ns_file, ex1_all_file, overwrite = true)
	ex1_ns_file
end

csv_file = startSolvingExample(; clearfolder = any(x -> x in ("--clear", "-c"), ARGS))

# csv_file = find_newest_csv("results/example1", "comparison_all")
function plotProfiles(datafile, tag)
	ex1_plot_file_time = prepare_filepath("results/example1/profile_$tag.png", dated = true)
	plt = performance_profile_from_csv(datafile; tag = tag, savepath = ex1_plot_file_time)
end

plotProfiles(csv_file, "Time")
plotProfiles(csv_file, "Iter")



# Example usage:
"""
# Convert single CSV to XLSX
# csv_to_xlsx("comparison_n100.csv")

# Convert with custom output name
csv_to_xlsx("comparison_n100.csv", "results_dimension_100.xlsx")

# Convert multiple CSV files to one XLSX with multiple sheets
csv_to_xlsx_multiple(
	["comparison_n100.csv", "comparison_n200.csv"],
	"all_comparisons.xlsx"
)

# Or with custom sheet names
XLSX.writetable("custom_results.xlsx",
	"n=100" => CSV.read("comparison_n100.csv", DataFrame),
	"n=200" => CSV.read("comparison_n200.csv", DataFrame)
)
"""

# # Or on iterations
# ex1_plot_file_Iters = prepare_filepath("results/example1/profile_Iters.png", dated = true)
# plt_iter = performance_profile_from_csv(ex1_n100_file; tag = "Iter", savepath = ex1_plot_file_Iters)

# # Provide explicit solver order (optional)
# ex1_plot_file_time_all = prepare_filepath("results/example1/profile_TimeAll.png", dated = true)
# plt2 = performance_profile_from_csv(ex1_n100_file; tag = "Time",
# 	solvers = ["IPCMAS1", "DeyHICPP", "IPCMAS2"], savepath = ex1_plot_file_time_all)
