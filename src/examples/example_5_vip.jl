include("../includes.jl")


function setup_example5(n::Int; seed = 2025, num_of_instances = 1)
	rng = Xoshiro(seed)
	# Generate random matrix Z ∈ [1, 100]
	U = Uniform(-10, 10)
	x0s = reshape(hcat([0.5; 0.5], [0.1; 0.2], [0.25; 0.25], [0.75; 0.25], [0.5; -0.6],
			rand(rng, U, 2, num_of_instances)), 2, 5 + num_of_instances)
	problems = Vector{Problem}(undef, 5 + num_of_instances)
	for (i, x0) in enumerate(eachcol(x0s))

		# Lipschitz constant L = max eigenvalue of f
		L = 1.0
		# Upper triangular matrix with all entries one
		A_matrix = I(2)

		# Define operator f
		B(x) = [-x[1] * exp(x[2]); x[2]]
		Pc(x) = project_sphere([max(x[1], 0); x[2]], [0.0; 0.0], 1.0)
		function A_resolvent(x, λ)
			return Pc(x - λ * B(x))
		end

		# Generate initial points in [0,1]
		x1 = Pc(x0)
		problems[i] = Problem(; name = "Example 1 ($i)", Aλ = A_resolvent, A = x -> x, B = B, L = L, x0 = x0, x1 = x1, n = n)
	end

	return problems
end


opts, pos = parse_args(ARGS)
title = "example 5"
algorithms = [
	# ("DeyHICPP", DeyHICPP, get_DeyHICPP_params),
	# ("IPCMAS1(λ1=1/2L, μ=0.5)", IPCMAS1, get_IPCMAS1_params),
	# ("IPCMAS1(λ1=1e-9, μ=0.5)", IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.5, λ0 = 1e-9)),
	# ("IPCMAS1(λ1=1, γ=1.1)", IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.5, λ0 = 1.0)),
	("IPCMAS1(λ1=1, γ=0.4)", IPCMAS1, (L) -> get_IPCMAS1_params(L; γ = 1.1, μ0 = 0.5, λ0 = 1.0)),
	# ("IPCMAS2(λ1=1e-9, γ=1.1)", IPCMAS2, (L) -> get_IPCMAS2_params(L; γ = 1.1, μ0 = 0.5, λ0 = 1e-9)),
]

errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
dims = [2]
seed = 2025
num_of_instances = 4
maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "50000")))
verbose = any(x -> x in ("--verbose", "-v"), ARGS)
show_progress = !any(x -> x == "--no-progress", ARGS)
clearfolder = any(x -> x in ("--clear", "-c"), ARGS)



csv_file, solutions = startSolvingExample(title, algorithms, setup_example5, dims;
	errors = errors,
	seed = seed,
	num_of_instances = num_of_instances,
	maxiter = maxiter,
	verbose = verbose,
	show_progress = show_progress,
	clearfolder = clearfolder,
	plotit = false,
	plot_comparizon = false,
)



println(join(map(x -> "$(x.solver)-$(x.problem.name) : $(x.solution[1]), $(x.solution[2]), x0=$(round(x.problem.x0[1], digits=2)),x1=$(round(x.problem.x1[1], digits=2)), itrs = $(x.iterations)", solutions), "\n"))
