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
    h = (grid[end] - grid[1]) / (n - 1)
    function L2_inner_product(x::Vector, y::Vector)
        # Trapezoidal rule
        # result = 0.5 * (x[1] * y[1] + x[end] * y[end])
        # for i in 2:n-1
        #     result += x[i] * y[i]
        # end

        # return result * dx

        # Simpson's rule: (h/3)[f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]
        z = x .* y
        result = z[1] + z[end]  # First and last terms
        result += 4 * sum(z[2:2:end-1])  # Odd indices (coefficients of 4)
        result += 2 * sum(z[3:2:end-2])  # Even indices (coefficients of 2)

        return (h / 3) * result

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

    function setup_example2(n_points::Int; seed=2025, num_of_instances=1)
        rng = Xoshiro(seed)

        # Create uniform grid on [0, 1]
        if n_points % 2 != 0
            n_points += 1
        end
        grid = range(0.0, 1.0, length=n_points + 1) |> collect

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
            total_error = error_C + error_Q
            return total_error < tol, total_error
        end

        problems = Vector{Problem}(undef, num_of_instances * length(initial_points))
        # x0_func(t) = generate_smooth_random(t)
        # x1_func(t) = generate_smooth_random(t)
        # u_func(t) = generate_smooth_random(t)
        counter = 1

        for i in 1:num_of_instances
            for (init_name, x0_func, x1_func) in initial_points
                problems[counter] = Problem(;
                    name="$init_name",
                    Aλ=resolvent_indicator_C,
                    A=A_op,
                    B=grad_h,
                    L=L,
                    x0=discretize_L2_function(x0_func, grid),
                    x1=discretize_L2_function(x1_func, grid),
                    n=n_points,
                    stopping=(x, tol) -> compute_error_L2(x, 1e-3),
                    norm=l2norm,
                    dot=l2dot,
                )
                counter += 1
            end
        end
        return problems
    end
end






"""
	get_DeyHICPP_params_L2(L::Float64)

Get default parameters for DeyHICPP algorithm for Example 2.
"""
function get_DeyHICPP_params_L2(L::Float64)
    λ_constant = 0.01  # As specified in the paper (page 21)
    β_seq = n -> 1.0 / sqrt(n + 1)
    return (
        γ=0.01,  # As specified for Example 2
        λ_seq=n -> λ_constant,
        α=0.5,
        τ_seq=n -> 1.0 / n^2,  # Modified for Example 2
        β_seq,
        θ_seq=n -> 0.8 - β_seq(n),
    )
end


function get_IPCMAS1_params_L2(L::Float64; γ=1.1, μ0=0.5, α0=0.5, β0=0.3, λ0=0.5)
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
    θ_seq(n) = 0.9

    return (
        γ=γ,
        μ=μ,
        λ1=λ1,
        β=β,
        α=α_fixed,
        β_seq=β_seq,
        α_seq=α_seq,
        a_seq=a_seq,
        θ_seq=θ_seq,
    )
end


opts, pos = parse_args(ARGS)
title = "example 2"
initial_points =
    [
        ("Instance 1", t -> t^3 * exp(t) / 211 + 5 * t, t -> sin(t) + t^6),
        ("Instance 2", t -> exp(t), t -> t * exp(t^3), t -> cos(t)),
        ("Instance 3", t -> t + 1, t -> 3 * t^2 + t, t -> t^2 * exp(t)),
        ("Instance 4", t -> 11 * sin(t), t -> 5 * t^2, t -> exp(t / 2)),
        ("Instance 5", t -> 15 * t^3 + exp(t) / 22, t -> sin(t / 2)),
        ("Instance 6", t -> exp(t), t -> cos(2π * t)),
        ("Instance 7", t -> t + 1, t -> 3 * t^3 + 2 * t),
        ("Instance 8", t -> 11 * sin(t), t -> sqrt(t)),
    ]
algorithms = [
    ("DeyHICPP", DeyHICPP, get_DeyHICPP_params_L2),
    ("IPCMAS1", IPCMAS1, L -> get_IPCMAS1_params_L2(L; γ=1.1, μ0=0.5, α0=0.5, β0=0.3, λ0=0.05)),
]

errors = [1e-3]
dims = [100]
maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "50000")))
seed = 2025
num_of_instances = 1
verbose = any(x -> x in ("--verbose", "-v"), ARGS)
show_progress = !any(x -> x == "--no-progress", ARGS)
clearfolder = any(x -> x in ("--clear", "-c"), ARGS)

function plot_convergence(results)
    title = "example 2"
    title = replace(title, " " => "_")
    solutions = map(x -> x[:full_solution], results)
    filtered_solustions = solutions
    names = map(s -> s.solver, filtered_solustions) |> unique
    cases = map(x -> x.problem.name, filtered_solustions) |> unique
    plts = []
    for c in cases
        _c = replace(c, " " => "_")
        series = map(name -> begin
                hist = filter(s -> s.solver == name && s.problem.name == c, filtered_solustions) |> y -> map(x -> x.history[:err], y)
                (name, vcat(hist...))
            end, names)
        plt = make_convergence_plot(series, xlabel="Iterations", ylabel=L"E_n", ylimits=(1e-4, 1e1), left_margin=5Plots.mm)
        push!(plts, plt)
    end
    plt = plot(plts...,
        layout=(4, 2),
        # plot_title="Error plotting for the different choices",
        plot_titlefontsize=14,
        size=(900, 1200),
        dpi=400)
    savepath = prepare_filepath("results/$(title)/convergence_plot.png", dated=false)
    savefig(plt, savepath)
end

csv_file, solutions = startSolvingExample(title, algorithms, setup_example2_wrapper(initial_points), dims;
    errors=errors,
    seed=seed,
    num_of_instances=num_of_instances,
    maxiter=maxiter,
    verbose=verbose,
    show_progress=show_progress,
    clearfolder=clearfolder,
    plotit=false,
    plot_comparizon=false,
    plot_convergence=plot_convergence,
)

