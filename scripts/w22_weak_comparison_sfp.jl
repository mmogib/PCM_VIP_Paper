## w22_weak_comparison_sfp.jl — Example 2 (SFP) weak-convergence comparison
#
# Split feasibility problem in L²[0,1]. Compares IPCMAS1 against 3 weak-
# convergence competitors from the literature.
#
#   TanQin2024     — Tan & Qin (2024), Algorithm 3.1
#   ChenMiPCA      — Chen, Zhang, Dong (2020), Algorithm 3.1 (s=2)
#   PeeyadaIMFBSA  — Peeyada, Suparatulatorn, Cholamjiak (2022), Algorithm 3.1
#   IPCMAS1        — our Algorithm 1
#
# For SFP with A=I, the gradient operator F(x) = x − P_Q(x) is firmly
# non-expansive (1-cocoercive), so Peeyada's assumption holds. Chen and
# Tan-Qin require f monotone + L-Lipschitz (L=1), which also holds.
#
# Tolerances: {1e-1, 1e-2, 1e-3}. Per-iter data to results/example_2_weak/history/.
#
# Usage:
#   julia --project=. scripts/w22_weak_comparison_sfp.jl
#   julia --project=. scripts/w22_weak_comparison_sfp.jl --dim=100
#   julia --project=. scripts/w22_weak_comparison_sfp.jl --algo=IPCMAS1,TanQin2024

include("../src/includes.jl")

# ── L²[0,1] helpers and projections (mirrors s20) ──────────────────────────
function discretize_L2_function(f::Function, grid::Vector{Float64})
    return [f(t) for t in grid]
end

function L2InnerProduct(grid::Vector{Float64})
    n = length(grid)
    h = (grid[end] - grid[1]) / (n - 1)
    function L2_inner_product(x::Vector, y::Vector)
        z = x .* y
        result = z[1] + z[end]
        result += 4 * sum(z[2:2:end-1])
        result += 2 * sum(z[3:2:end-2])
        return (h / 3) * result
    end
end

L2Norm(grid::Vector{Float64}) = x -> sqrt(L2InnerProduct(grid)(x, x))

function ProjectOnC(grid::Vector{Float64})
    l2prod = L2InnerProduct(grid)
    g_C_vec = discretize_L2_function(t -> 3 * t^2, grid)
    norm_g_C_sq = l2prod(g_C_vec, g_C_vec)
    function P_C(x::Vector)
        inner = l2prod(x, g_C_vec)
        abs(inner) > 1e-12 ? (x - (inner / norm_g_C_sq) * g_C_vec) : x
    end
end

function ProjectOnQ(grid::Vector{Float64})
    l2prod = L2InnerProduct(grid)
    g_Q_vec = discretize_L2_function(t -> t / 3, grid)
    norm_g_Q_sq = l2prod(g_Q_vec, g_Q_vec)
    function P_Q(x::Vector)
        inner = l2prod(x, g_Q_vec)
        inner < -1 ? (x - ((inner + 1) / norm_g_Q_sq) * g_Q_vec) : x
    end
end

# ── Problem setup (mirrors s20's setup_example2) ───────────────────────────
function setup_example2_wrapper(initial_points)
    function setup_example2(n_points::Int; seed=2025, num_of_instances=1)
        rng = Xoshiro(seed)
        if n_points % 2 != 0
            n_points += 1
        end
        grid = range(0.0, 1.0, length=n_points + 1) |> collect

        A_op(x) = copy(x)
        A_star(x) = copy(x)
        function grad_h(x::Vector)
            Ax = A_op(x)
            return Ax - ProjectOnQ(grid)(Ax)
        end
        L = 1.0

        resolvent_indicator_C(x::Vector, λ::Float64) = ProjectOnC(grid)(x)
        l2norm = L2Norm(grid)
        l2dot = L2InnerProduct(grid)
        P_C = ProjectOnC(grid)
        P_Q = ProjectOnQ(grid)

        function compute_error_L2(x::Vector, tol::Float64)
            P_C_x = P_C(x)
            xPCx = x - P_C_x
            error_C = 0.5 * dot(xPCx, xPCx)
            Ax = A_op(x)
            P_Q_Ax = P_Q(Ax)
            AxPQAx = Ax - P_Q_Ax
            error_Q = 0.5 * dot(AxPQAx, AxPQAx)
            total_error = error_C + error_Q
            return total_error < tol, total_error
        end

        problems = Vector{Problem}(undef, num_of_instances * length(initial_points))
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
                    stopping=(x, tol) -> compute_error_L2(x, tol),
                    norm=l2norm,
                    dot=l2dot,
                )
                counter += 1
            end
        end
        return problems
    end
end

# ── IPCMAS1 parameters tuned by w11 sensitivity ─────────────────────────────
# α=0.30, β=0.25·β_max(0.30)=0.0119, θ̄=0.9, ξ_n=100/(n+1)^1.1.
# Satisfies A4/A5 (α<1/3, β<β_max(α)).
function get_IPCMAS1_params_L2(L::Float64; γ=1.1, μ0=0.5, α0=0.30, β0=0.0119, λ0=0.5)
    a_seq(n) = 100 / (n + 1)^(1.1)
    θ_seq(n) = 0.9
    return (
        γ=γ, μ=μ0, λ1=isnothing(λ0) ? 1.0 / (2 * L) : λ0,
        β=β0, α=α0, a_seq=a_seq, θ_seq=θ_seq,
    )
end

function main()
    opts, _ = parse_args(ARGS)
    force = any(x -> x == "--force", ARGS)

    title = "example 2 weak"

    initial_points = [
        ("Instance 1", t -> t^3 * exp(t) / 211 + 5 * t, t -> sin(t) + t^6),
        ("Instance 2", t -> exp(t), t -> t * exp(t^3)),
        ("Instance 3", t -> t + 1, t -> 3 * t^2 + t),
        ("Instance 4", t -> 11 * sin(t), t -> 5 * t^2),
        ("Instance 5", t -> 15 * t^3 + exp(t) / 22, t -> sin(t / 2)),
        ("Instance 6", t -> exp(t), t -> cos(2π * t)),
        ("Instance 7", t -> t + 1, t -> 3 * t^3 + 2 * t),
        ("Instance 8", t -> 11 * sin(t), t -> sqrt(t)),
    ]

    ALL_ALGORITHMS = Dict(
        "IPCMAS1"        => ("IPCMAS1",       IPCMAS1,       L -> get_IPCMAS1_params_L2(L; λ0=0.05)),
        "TanQin2024"     => ("TanQin2024",    TanQin2024,    L -> get_TanQin2024_params(L)),
        "ChenMiPCA"      => ("ChenMiPCA",     ChenMiPCA,     L -> get_ChenMiPCA_params(L; λ0=1 / (1.05 * L))),
        "PeeyadaIMFBSA"  => ("PeeyadaIMFBSA", PeeyadaIMFBSA, L -> get_PeeyadaIMFBSA_params(L)),
    )
    default_keys = ["IPCMAS1", "TanQin2024", "ChenMiPCA", "PeeyadaIMFBSA"]

    algo_filter = get(opts, "algo", "")
    if !isempty(algo_filter)
        selected_keys = split(algo_filter, ",") .|> strip .|> String
        for k in selected_keys
            haskey(ALL_ALGORITHMS, k) || error("Unknown algorithm '$k'. Available: $(join(keys(ALL_ALGORITHMS), ", "))")
        end
        title *= " ($(algo_filter))"
    else
        selected_keys = default_keys
    end
    algorithms = [ALL_ALGORITHMS[k] for k in selected_keys if haskey(ALL_ALGORITHMS, k)]

    errors = [1e-1, 1e-2, 1e-3]
    dims_str = get(opts, "dim", "100")
    dims = parse.(Int, split(dims_str, ","))
    maxiter = parse(Int, get(opts, "maxiter", get(opts, "itr", "20000")))
    seed = 2025
    num_of_instances = 1   # the 8 initial_points provide the instance variety
    verbose = any(x -> x in ("--verbose", "-v"), ARGS)
    show_progress = !any(x -> x == "--no-progress", ARGS)
    clearfolder = any(x -> x in ("--clear", "-c"), ARGS)

    logpath, tee, logfile = setup_logging("w22_weak_sfp"; logdir="results/example_2_weak")
    println(tee, "="^70)
    println(tee, "EXAMPLE 2 (SFP in L²[0,1]) — WEAK-CONVERGENCE COMPARISON")
    println(tee, "="^70)
    println(tee, "  Algorithms: ", join(first.(algorithms), ", "))
    println(tee, "  Tolerances: $errors")
    println(tee, "  n_points:   $dims")
    println(tee, "  Initial pairs: $(length(initial_points))")
    println(tee, "  MaxIter:    $maxiter")
    println(tee, "  save_history=true → per-iter CSVs enabled")
    force && println(tee, "  --force: re-running all configs")
    println(tee, "="^70)
    flush(tee)

    csv_file, solutions = startSolvingExample(title, algorithms,
        setup_example2_wrapper(initial_points), dims;
        errors=errors,
        seed=seed,
        num_of_instances=num_of_instances,
        maxiter=maxiter,
        verbose=verbose,
        show_progress=show_progress,
        clearfolder=clearfolder,
        plotit=false,
        plot_comparizon=false,
        io=tee,
        force=force,
        save_history=true,
    )

    teardown_logging(tee, logpath)
end

main()
