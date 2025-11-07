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
		problems[i] = Problem(; name = "Example 1 ($i)", Aλ = A_resolvent, A = x -> A_matrix * x, B = B, L = L, x0 = x0, x1 = x1, n = n)
	end

	return problems
end


opts, pos = parse_args(ARGS)
title = "example 1"
algorithms = [
	("DeyHICPP", DeyHICPP, get_DeyHICPP_params),
	("IPCMAS1(λ1=1/2L, μ=0.5)", IPCMAS1, get_IPCMAS1_params),
	("IPCMAS1(λ1=1.1920929f-7, μ=0.5)", IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.5, λ0 = 1e-9)),
	("IPCMAS1(λ1=1, μ=0.5)", IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.5, λ0 = 1.0))]

errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
dims = [100, 150, 200, 250, 300]
seed = 2025
num_of_instances = 1
maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "50000")))
verbose = any(x -> x in ("--verbose", "-v"), ARGS)
show_progress = !any(x -> x == "--no-progress", ARGS)
clearfolder = any(x -> x in ("--clear", "-c"), ARGS)



csv_file = startSolvingExample(title, algorithms, setup_example1, dims;
	errors = errors,
	seed = seed,
	num_of_instances = num_of_instances,
	maxiter = maxiter,
	verbose = verbose,
	show_progress = show_progress,
	clearfolder = clearfolder,
)

# csv_file = find_newest_csv("results/example1", "comparison_all")

title = replace(title, " " => "_")
plotProfiles(title, csv_file, "Time")
plotProfiles(title, csv_file, "Iter")



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
