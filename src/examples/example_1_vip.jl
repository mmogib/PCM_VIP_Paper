include("../includes.jl")


function setup_example1(n::Int; seed=2025, num_of_instances=1)
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
        problems[i] = Problem(; name="Example 1 ($i)", Aλ=A_resolvent, A=x -> A_matrix * x, B=B, L=L, x0=x0, x1=x1, n=n)
    end

    return problems
end


opts, pos = parse_args(ARGS)
title = "example 1"
# get_IPCMAS1_params:: L::Float64; γ = 1.1, μ0 = 0.5, α0 = 0.25, β0 = 0.0001, λ0::Union{Nothing, Float64} = nothing
# get_DongIPCA_params(L::Float64;	γ::Float64 = 1.5, τ0::Union{Nothing, Float64} = nothing,α::Float64 = 0.4,	α_seq::Function = n -> (n == 1 ? 0.0 : α))
algorithms = [
    ("DeyHICPP", DeyHICPP, L -> get_DeyHICPP_params(L; λ0=1 / (1.05 * L))),
    # ("DongIPCA(λ1=1/1.05L, μ=0.5)", DongIPCA, L -> get_DongIPCA_params(L; τ0 = 1 / (1.05 * L))),
    ("IPCMAS1", IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0=0.5, λ0=1 / (1.05 * L))),
    # ("IPCMAS1(λ1=1e-9, μ=0.5)", IPCMAS1, (L) -> get_IPCMAS1_params(L; μ0 = 0.5, λ0 = 1e-9)),
    ("IPCMAS2", IPCMAS2, (L) -> get_IPCMAS2_params(L; γ=1.1, λ0=1 / (1.05 * L))),
]

errors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# errors = [1e-4]
dims = [100, 150, 200, 250, 300]
# dims = [100, 150]
seed = 2025
num_of_instances = 5
maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "50000")))
verbose = any(x -> x in ("--verbose", "-v"), ARGS)
show_progress = !any(x -> x == "--no-progress", ARGS)
clearfolder = any(x -> x in ("--clear", "-c"), ARGS)



csv_file, solutions = startSolvingExample(title, algorithms, setup_example1, dims;
    errors=errors,
    seed=seed,
    num_of_instances=num_of_instances,
    maxiter=maxiter,
    verbose=verbose,
    show_progress=show_progress,
    clearfolder=clearfolder,
    plotit=true,
    plot_comparizon=false,
    # convergence_dims = nothing
)
