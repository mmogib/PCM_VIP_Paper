include("../includes.jl")


function setup_example4(n::Int; seed = 2025, num_of_instances = 1)
	rng = Xoshiro(seed)
	# Generate random matrix Z ∈ [1, 100]
	U = Uniform(-10, 10)
	x0s = vcat([1.8; -2.0; 2.0; 0.8; 0.5; 0.1; -0.8],
		rand(rng, U, num_of_instances),
	)
	x0s_len = length(x0s)
	problems = Vector{Problem}(undef, x0s_len)

	for (i, x0) in enumerate(x0s)

		# Lipschitz constant L = max eigenvalue of f
		L = 1.0

		# Upper triangular matrix with all entries one
		A_matrix = [1.0]

		# Define operator f
		B(x) =
			if x[1] < -1.0
				[2x[1] - 1]
			elseif -1.0 <= x[1] <= 1.0
				[x[1]^2]
			else
				[-2x[1] - 1]
			end
		Pc(x) = clamp.(x, -1, 1)
		function A_resolvent(x, λ)
			return Pc(x - λ * B(x))
		end

		# Generate initial points in [0,1]
		x1 = Pc([x0])
		problems[i] = Problem(; name = "Example 1 ($i)", Aλ = A_resolvent, A = x -> x, B = B, L = L, x0 = [x0], x1 = x1, n = n,
			stopping = (x, tol) -> abs(x[1]) < 1e-6,
		)
	end

	return problems
end


opts, pos = parse_args(ARGS)
title = "example 4"
algorithms = [
	# ("DeyHICPP", DeyHICPP, get_DeyHICPP_params),
	# ("IPCMAS1(λ1=1/2L, μ=0.5)", IPCMAS1, get_IPCMAS1_params),
	# ("IPCMAS1(λ1=1e-9, μ=0.5)", IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.5, λ0 = 1e-9)),
	# ("IPCMAS1(λ1=1, γ=1.1)", IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.5, λ0 = 1.0)),
	("IPCMAS1(λ1=1, γ=0.4)", IPCMAS1, (L) -> get_IPCMAS1_params(L; γ = 1.1, μ0 = 0.25, λ0 = 1.0)),
	# ("IPCMAS2(λ1=1e-9, γ=1.1)", IPCMAS2, (L) -> get_IPCMAS2_params(L; γ = 1.1, μ0 = 0.5, λ0 = 1e-9)),
]

errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
dims = [1]
seed = 2025
num_of_instances = 4
maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "50000")))
verbose = any(x -> x in ("--verbose", "-v"), ARGS)
show_progress = !any(x -> x == "--no-progress", ARGS)
clearfolder = any(x -> x in ("--clear", "-c"), ARGS)



csv_file, solutions = startSolvingExample(title, algorithms, setup_example4, dims;
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



println(join(map(x -> "$(x.solver)-$(x.problem.name) : $(x.solution[1]), x0=$(round(x.problem.x0[1], digits=2)),x1=$(round(x.problem.x1[1], digits=2)), itrs = $(x.iterations)", solutions), "\n"))
