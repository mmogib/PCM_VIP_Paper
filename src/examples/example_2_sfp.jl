include("../includes.jl")

"""
	Example2Problem

Structure to hold Example 2 (Split Feasibility Problem) data in L²[0,1].
"""
struct Example2Problem
	A_resolvent::Function  # Resolvent of ∂i_C
	f::Function           # Gradient ∇h(x) = A*(I - P_Q)Ax
	P_C::Function         # Projection onto C
	P_Q::Function         # Projection onto Q
	A_op::Function        # Linear operator A
	A_star::Function      # Adjoint A*
	L::Float64           # Lipschitz constant of ∇h
	x0::Function         # Initial point 1 (function of t)
	x1::Function         # Initial point 2 (function of t)
	u::Function          # Anchor point u (function of t) for some algorithms
	grid::Vector{Float64} # Discretization grid
	n_points::Int        # Number of discretization points
end


"""
	discretize_L2_function(f::Function, grid::Vector{Float64})

Discretize an L²[0,1] function on a grid.
"""
function discretize_L2_function(f::Function, grid::Vector{Float64})
	return [f(t) for t in grid]
end

function L2InnerProduct(grid::Vector{Float64})
	n = length(grid)
	dx = grid[2] - grid[1]  # Assuming uniform grid
	function L2_inner_product(x::Vector, y::Vector)
		# Trapezoidal rule
		result = 0.5 * (x[1] * y[1] + x[end] * y[end])
		for i in 2:n-1
			result += x[i] * y[i]
		end

		return result * dx
	end
end


function L2Norm(grid::Vector{Float64})
	function L2_norm(x::Vector)
		return sqrt(L2InnerProduct(grid)(x, x))
	end
end

function ProjectOnC(grid::Vector{Float64})
	l2prod = L2InnerProduct(grid)
	# Define basis functions for projections
	g_C(t) = 3 * t^2  # For C = {x : ⟨x, 3t²⟩ = 0}

	# Discretize basis functions
	g_C_vec = discretize_L2_function(g_C, grid)

	# Precompute norms squared
	norm_g_C_sq = l2prod(g_C_vec, g_C_vec)

	function P_C(x::Vector)
		inner_prod = l2prod(x, g_C_vec)
		if abs(inner_prod) > 1e-12
			return x - (inner_prod / norm_g_C_sq) * g_C_vec
		else
			return x
		end
	end
end



# Projection onto Q: P_Q(x) = x - max(0, ⟨x, g_Q⟩ + 1)/‖g_Q‖² * g_Q
function ProjectOnQ(grid::Vector{Float64})
	l2prod = L2InnerProduct(grid)
	g_Q(t) = t / 3    # For Q = {x : ⟨x, t/3⟩ ≥ -1}
	g_Q_vec = discretize_L2_function(g_Q, grid)
	norm_g_Q_sq = l2prod(g_Q_vec, g_Q_vec)
	function P_Q(x::Vector)
		inner_prod = l2prod(x, g_Q_vec)
		projection_coeff = (inner_prod + 1) / norm_g_Q_sq
		if inner_prod < -1
			return x - projection_coeff * g_Q_vec
		else
			return x
		end
	end
end

"""
	setup_example2(n_points::Int=101; seed=123)

Setup Example 2: Split Feasibility Problem in L²[0,1].

The problem is to find x ∈ C such that Ax ∈ Q, where:
- C = {x ∈ L²[0,1] : ⟨x, 3t²⟩ = 0} (hyperplane)
- Q = {x ∈ L²[0,1] : ⟨x, t/3⟩ ≥ -1} (halfspace)
- A = I (identity operator)

# Arguments
- `n_points`: Number of discretization points (default: 101)
- `seed`: Random seed for initial points

# Returns
- `Example2Problem`: Structure containing all problem data
"""
function setup_example2_wrapper(initial_points)

	function setup_example2(n_points::Int; seed = 2025, num_of_instances = 1)
		rng = Xoshiro(seed)

		# Create uniform grid on [0, 1]
		grid = range(0.0, 1.0, length = n_points) |> collect

		# Projection onto C: P_C(x) = x - ⟨x, g_C⟩/‖g_C‖² * g_C


		# Linear operator A = I (identity)
		A_op(x) = copy(x)
		A_star(x) = copy(x)  # A* = A for identity

		# Gradient ∇h(x) = A*(I - P_Q)Ax = (I - P_Q)x for A = I
		function grad_h(x::Vector)
			Ax = A_op(x)
			return Ax - ProjectOnQ(grid)(Ax)
		end

		# Lipschitz constant L = ‖A‖² = 1 for identity operator
		L = 1.0

		# Indicator function subdiferential: ∂i_C
		# The resolvent J^{∂i_C}_λ(x) = P_C(x) (independent of λ for normal cone)
		function resolvent_indicator_C(x::Vector, λ::Float64)
			return ProjectOnC(grid)(x)
		end

		l2norm = L2Norm(grid)
		l2dot = L2InnerProduct(grid)
		P_C = ProjectOnC(grid)
		P_Q = ProjectOnQ(grid)
		function compute_error_L2(x::Vector, tol::Float64)
			# Error component 1: distance to C
			P_C_x = P_C(x)
			xPCx = x - P_C_x
			error_C = 0.5 * dot(xPCx, xPCx)
			# Error component 2: distance of Ax to Q
			Ax = A_op(x)
			P_Q_Ax = P_Q(Ax)
			AxPQAx = Ax - P_Q_Ax
			error_Q = 0.5 * dot(AxPQAx, AxPQAx)
			return (error_C + error_Q) < tol
		end

		problems = Vector{Problem}(undef, num_of_instances * length(initial_points))
		# x0_func(t) = generate_smooth_random(t)
		# x1_func(t) = generate_smooth_random(t)
		# u_func(t) = generate_smooth_random(t)
		counter = 1

		for i in 1:num_of_instances
			for (init_name, _, x0_func, x1_func) in initial_points
				problems[counter] = Problem(;
					name = "Example 2 $i ($init_name)",
					Aλ = resolvent_indicator_C,
					A = A_op,
					B = grad_h,
					L = L,
					x0 = discretize_L2_function(x0_func, grid),
					x1 = discretize_L2_function(x1_func, grid),
					n = n_points,
					stopping = (x, tol) -> compute_error_L2(x, 1e-3),
					norm = l2norm,
					dot = l2dot,
				)
				counter += 1
			end
		end
		return problems
	end
end




"""
	solve_problem_L2(algorithm, problem::Example2Problem, params::NamedTuple; 
					 tol=1e-3, maxiter=10000)

Generic solver for Example 2 (L² functions).

# Arguments
- `algorithm`: Algorithm function
- `problem`: Example2Problem structure
- `params`: Named tuple of algorithm-specific parameters
- `tol`: Tolerance
- `maxiter`: Maximum iterations

# Returns
- `x`: Solution (discretized)
- `iter`: Iterations
- `converged`: Convergence flag
- `time`: Elapsed time
"""
function solve_problem_L2(algorithm, problem::Example2Problem, params::NamedTuple;
	tol = 1e-3, maxiter = 10000, verbose::Bool = false)

	# Discretize initial points
	x0_vec = discretize_L2_function(problem.x0, problem.grid)
	x1_vec = discretize_L2_function(problem.x1, problem.grid)

	# Create modified norm function for stopping criterion
	# We need to use L² norm instead of Euclidean norm
	if verbose
		println(@sprintf("  -> starting: tol=%.1e, maxiter=%d", tol, maxiter))
	end

	elapsed_time = @elapsed begin
		x, iter, converged = algorithm(
			problem.A_resolvent,
			problem.f,
			x0_vec,
			x1_vec;
			params...,
			tol = tol,
			maxiter = maxiter,
		)
	end

	if verbose
		println(@sprintf("  -> finished: time=%.4fs, iter=%d, converged=%s",
			elapsed_time, iter, string(converged)))
	end

	return x, iter, converged, elapsed_time
end



"""
	get_DeyHICPP_params_L2(L::Float64)

Get default parameters for DeyHICPP algorithm for Example 2.
"""
function get_DeyHICPP_params_L2(L::Float64)
	λ_constant = 0.01  # As specified in the paper (page 21)
	β_seq = n -> 1.0 / sqrt(n + 1)
	return (
		γ = 0.01,  # As specified for Example 2
		λ_seq = n -> λ_constant,
		α = 0.5,
		τ_seq = n -> 1.0 / n^2,  # Modified for Example 2
		β_seq,
		θ_seq = n -> 0.8 - β_seq(n),
	)
end


function get_IPCMAS1_params_L2(L::Float64; γ = 1.1, μ0 = 0.5, α0 = 0.5, β0 = 0.3, λ0 = 0.5)
	μ = μ0
	λ1 = isnothing(λ0) ? 1.0 / (2 * L) : λ0 #  # Constant step size
	α_fixed = α0
	# rng = Xoshiro(2025)
	# U = Uniform(0, (1 - 3α_fixed) / (3 * (1 - α_fixed)))

	β = β0 #rand(rng, U)
	aseq() = begin
		prefix = Float64[0.0]   # prefix[k+1] stores sum_{i=1}^k 1/i^2
		function (n::Int)
			n ≥ 1 || throw(ArgumentError("n must be ≥ 0"))
			while length(prefix) - 1 < n
				k = length(prefix)          # next i to add
				push!(prefix, prefix[end] + 1.0 / (k^2))
			end
			return prefix[n+1]
		end
	end

	β_seq(n) = 1.0 / (5 * n + 1)
	α_seq(n) = 0.8 - β_seq(n)
	# α_seq(n) = α_fixed #  0.8 - β_seq(n)
	a_seq(n) = 100 / (n)^(2) #aseq()
	# ι_seq = n -> 1.0 / n^2

	return (
		γ = γ,
		μ = μ,
		λ1 = λ1,
		β = β,
		α = α_fixed,
		β_seq = β_seq,
		α_seq = α_seq,
		a_seq = a_seq,
	)
end
"""
	generate_table3(algorithms::Vector, 
				   algorithm_names::Vector{String};
				   initial_points=[
					   ("15t³ + e^(t/22)", "t³e^(t/211) + 5t", "sin(t) + t⁶"),
					   ("e^t", "te^t³", "cos(t)"),
					   ("t + 1", "3t² + t", "t²e^t"),
					   ("11sin(t)", "5t²", "e^(t/2)")
				   ],
				   n_points=101,
				   tol=1e-3,
				   maxiter=10000)

Generate Table 3 from the paper for Example 2.
"""
function generate_table3(algorithms::Vector,
	algorithm_names::Vector{String};
	initial_points = [
		(t -> 15 * t^3 + exp(t / 22), t -> t^3 * exp(t / 211) + 5 * t, t -> sin(t) + t^6),
		(t -> exp(t), t -> t * exp(t^3), t -> cos(t)),
		(t -> t + 1, t -> 3 * t^2 + t, t -> t^2 * exp(t)),
		(t -> 11 * sin(t), t -> 5 * t^2, t -> exp(t / 2)),
	],
	n_points = 101,
	tol = 1e-3,
	maxiter = 10000,
	verbose::Bool = false,
	show_progress::Bool = true)

	# Minimal header if showing progress; full table otherwise
	if !show_progress
		println("\n" * "="^(50 + 25 * length(algorithms)))
		println("Table 3: Split Feasibility Problem in L²[0,1]")
		println("="^(50 + 25 * length(algorithms)))
		println()
	else
		println(@sprintf("\nGenerating Table 3 (n_points=%d, tol=%.1e) ...", n_points, tol))
	end

	# Print header
	if !show_progress
		print(@sprintf("%-15s", "Choice"))
		for name in algorithm_names
			print(@sprintf(" | %-20s", name))
		end
		println()

		print(@sprintf("%-15s", ""))
		for _ in algorithm_names
			print(@sprintf(" | %-9s %-9s", "Time", "No. It."))
		end
		println()
		println("-"^(50 + 25 * length(algorithms)))
	end

	all_results = Dict{String, Vector}()

	# Progress bar over all runs (choices × algorithms)
	total_tasks = length(initial_points) * length(algorithms)
	p = show_progress ? Progress(total_tasks; desc = @sprintf("Table 3 (n=%d)", n_points)) : nothing

	# Run each choice of initial points
	for (choice_idx, (u_func, x0_func, x1_func)) in enumerate(initial_points)
		# Setup problem with these initial points
		problem = setup_example2(n_points)
		problem = Example2Problem(
			problem.A_resolvent,
			problem.f,
			problem.P_C,
			problem.P_Q,
			problem.A_op,
			problem.A_star,
			problem.L,
			x0_func,
			x1_func,
			u_func,
			problem.grid,
			problem.n_points,
		)

		if !show_progress
			print(@sprintf("Choice %d        ", choice_idx))
		end

		for (i, (algo_func, param_getter)) in enumerate(algorithms)
			# Get algorithm-specific parameters
			params = param_getter(problem.L)

			# Solve
			if verbose && !show_progress
				println(@sprintf("Running %-10s for Choice %d", algorithm_names[i], choice_idx))
				println("  params = ", params)
			end
			x, iter, converged, time = solve_problem_L2(
				algo_func, problem, params;
				tol = tol, maxiter = maxiter, verbose = verbose,
			)

			# Store results
			algo_name = algorithm_names[i]
			if !haskey(all_results, algo_name)
				all_results[algo_name] = []
			end

			# Compute final error
			final_error = compute_error_L2(x, problem)

			push!(all_results[algo_name],
				(choice = choice_idx, time = time, iter = iter,
					converged = converged, error = final_error))

			# Print
			if !show_progress
				status = converged ? "" : "*"
				print(@sprintf(" | %.4f    %5d%s", time, iter, status))
			end

			# Update progress bar
			if show_progress
				next!(p; showvalues = [(:choice, choice_idx), (:algo, algorithm_names[i]), (:iter, iter)])
			end
		end
		if !show_progress
			println()
		end
	end

	if !show_progress
		println("="^(50 + 25 * length(algorithms)))
		println("* = Did not converge")
		println("\nNote: Initial point choices correspond to Table 3 in the paper")
		println("Choice 1: u(t)=15t³+e^(t/22), x₀(t)=t³e^(t/211)+5t, x₁(t)=sin(t)+t⁶")
		println("Choice 2: u(t)=e^t, x₀(t)=te^t³, x₁(t)=cos(t)")
		println("Choice 3: u(t)=t+1, x₀(t)=3t²+t, x₁(t)=t²e^t")
		println("Choice 4: u(t)=11sin(t), x₀(t)=5t², x₁(t)=e^(t/2)")
	else
		println(@sprintf("Finished Table 3 (n_points=%d)", n_points))
	end

	return all_results
end

"""
	test_example2_single_run(algorithm, param_getter; 
							choice=1, n_points=101, tol=1e-3)

Quick test of Example 2 with a single choice of initial points.
"""
function test_example2_single_run(algorithm, param_getter;
	choice = 1, n_points = 101, tol = 1e-3)

	# Define initial point choices
	initial_points = [
		(t -> 15 * t^3 + exp(t / 22), t -> t^3 * exp(t / 211) + 5 * t, t -> sin(t) + t^6),
		(t -> exp(t), t -> t * exp(t^3), t -> cos(t)),
		(t -> t + 1, t -> 3 * t^2 + t, t -> t^2 * exp(t)),
		(t -> 11 * sin(t), t -> 5 * t^2, t -> exp(t / 2)),
	]

	u_func, x0_func, x1_func = initial_points[choice]

	println("\n" * "="^70)
	println("Example 2 Test: Split Feasibility Problem")
	println("="^70)
	println("Choice: $choice")
	println("Discretization points: $n_points")
	println("Tolerance: $tol")

	# Setup problem
	problem = setup_example2(n_points)
	problem = Example2Problem(
		problem.A_resolvent,
		problem.f,
		problem.P_C,
		problem.P_Q,
		problem.A_op,
		problem.A_star,
		problem.L,
		x0_func,
		x1_func,
		u_func,
		problem.grid,
		problem.n_points,
	)

	params = param_getter(problem.L)

	x, iter, converged, time = solve_problem_L2(
		algorithm, problem, params;
		tol = tol, maxiter = 10000,
	)

	final_error = compute_error_L2(x, problem)

	println("\nResults:")
	println("  Converged: $converged")
	println("  Iterations: $iter")
	println("  Time: $(round(time, digits=4)) seconds")
	println("  Final error: $(round(final_error, digits=6))")
	println("  Solution L² norm: $(round(l2norm(problem.grid)(x), digits=6))")
	println("="^70)

	return x, iter, converged, time
end

# ============================================================================
# Example usage
# ============================================================================

# println("\n" * "="^70)
# println("Example 2: Split Feasibility Problem in L²[0,1]")
# println("="^70)

# Test single run
# test_example2_single_run(DeyHICPP, get_DeyHICPP_params_L2, choice = 1, tol = 1e-3)

# # Generate full Table 3
# algorithms = [
# 	(DeyHICPP, get_DeyHICPP_params_L2),
# ]

# algorithm_names = [
# 	"DeyHICPP",
# ]

# algorithms = [
# 	(DeyHICPP, get_DeyHICPP_params),
# 	(IPCMAS1, get_IPCMAS1_params),
# 	(IPCMAS2, get_IPCMAS2_params),
# 	# Add more algorithms here as we implement them:
# 	# (Algorithm_1_15, get_Algorithm_1_15_params),
# 	# (Algorithm_1_17, get_Algorithm_1_17_params),
# ]

# algorithm_names = [
# 	"DeyHICPP",
# 	"IPCMAS1",
# 	"IPCMAS2",
# 	# "Algo (1.15)",
# 	# "Algo (1.17)",
# ]

# results_table3 = generate_table3(
# 	algorithms,
# 	algorithm_names;
# 	n_points = 101,
# 	tol = 1e-3,
# 	maxiter = 10000,
# 	verbose = any(x -> x in ("--verbose", "-v"), ARGS),
# 	show_progress = !any(x -> x == "--no-progress", ARGS),
# )

# # Save results
# function save_table3_results(results::Dict, filename::String)
# 	data = []
# 	header = ["Choice", "Algorithm", "Time", "Iterations", "Converged", "Final_Error"]

# 	for algo_name in keys(results)
# 		for r in results[algo_name]
# 			push!(data, [r.choice, algo_name, r.time, r.iter, r.converged, r.error])
# 		end
# 	end

# 	writedlm(filename, vcat([header], data), ',')
# 	println("\nTable 3 results saved to $filename")
# end

# save_table3_results(results_table3, "results/table3_results_2.csv")

# # Convert to Excel
# csv_to_xlsx("results/table3_results.csv", "results/table3_results_2.xlsx", sheet_name = "SFP_L2", overwrite = true)



opts, pos = parse_args(ARGS)
title = "example 2"
initial_points =
	[
		("case 1", t -> 15 * t^3 + exp(t / 22), t -> t^3 * exp(t / 211) + 5 * t, t -> sin(t) + t^6),
		("case 2", t -> exp(t), t -> t * exp(t^3), t -> cos(t)),
		("case 3", t -> t + 1, t -> 3 * t^2 + t, t -> t^2 * exp(t)),
		("case 4", t -> 11 * sin(t), t -> 5 * t^2, t -> exp(t / 2)),
	]
algorithms = [
	("DeyHICPP", DeyHICPP, get_DeyHICPP_params_L2),
	("IPCMAS1(λ1=1/2L, β0 = 0.3)", IPCMAS1, L -> get_IPCMAS1_params_L2(L; γ = 1.1, μ0 = 0.5, α0 = 0.5, β0 = 0.3, λ0 = 0.05)),
]

errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
dims = [1_000]
maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "50000")))
seed = 2025
num_of_instances = 1
verbose = any(x -> x in ("--verbose", "-v"), ARGS)
show_progress = !any(x -> x == "--no-progress", ARGS)
clearfolder = any(x -> x in ("--clear", "-c"), ARGS)



csv_file, solutions = startSolvingExample(title, algorithms, setup_example2_wrapper(initial_points), dims;
	errors = errors,
	seed = seed,
	num_of_instances = num_of_instances,
	maxiter = maxiter,
	verbose = verbose,
	show_progress = show_progress,
	clearfolder = clearfolder,
	plotit = true,
	plot_comparizon = true,
)
println(solutions[1].solver)
# csv_file = find_newest_csv("results/example1", "comparison_all")




# n_points = 1000
# grid = range(0.0, 1.0, length = n_points) |> collect
# p_C = ProjectOnC(grid)
# p_Q = ProjectOnQ(grid)

# f1, f2, f3 = sin, t -> 3t^2, t -> t / 3
# f1_dis = discretize_L2_function(f1, grid)
# f2_dis = discretize_L2_function(f2, grid)
# f3_dis = discretize_L2_function(f3, grid)
# val_f1_f2_A1 = L2InnerProduct(grid)(p_C(f1_dis), f2_dis)
# val_f1_f2_A2 = L2InnerProduct(grid)(p_Q(p_C(f1_dis)), f3_dis)

# val_f1_f2_B1 = L2InnerProduct(ProjectOnC(f1), f2)
# val_f1_f2_B2 = L2InnerProduct(ProjectOnQ(ProjectOnC(f1)), f3)

# printstyled(val_f1_f2_A1, color = :red)
# println()
# print("=="^75)
# println()
# printstyled(val_f1_f2_B1, color = :blue)
# println()
# print("=="^75)
# println()
# printstyled(val_f1_f2_A2, color = :red)
# println()
# print("=="^75)
# println()
# printstyled(val_f1_f2_B2, color = :blue)
