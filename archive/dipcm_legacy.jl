## dipcm_legacy.jl — Pre-2026-04-25 DIPCM implementation (archived)
#
# The new DIPCM matching paper §3.4 (3-term Halpern–Mann update with adaptive
# β'_n, θ_n caps) lives in ../src/algorithms.jl. This legacy implementation is
# preserved for historical reference and reproducibility of pre-2026-04-25
# results (e.g., the 2026-04-07 sensitivity table).
#
# To use:  include("../src/includes.jl"); include("dipcm_legacy.jl")
# Requires `Problem` and `Solution` types from ../src/types.jl.
#
# Update: x_{n+1} = α z_n + σ_n u_n   (constant α, σ_n = (1-α) - β_n)
# Constant inertial parameters (β_zn, θ_bar) — not strictly Assumption-A4''
# admissible for the new paper §3.4 but practically equivalent for the
# problems at hand.

function get_DIPCM_legacy_params(L::Float64;
	γ = 1.1, μ0 = 0.5, α0 = 0.2, β_zn = 0.0001, θ_bar = 0.9,
	β_decay = n -> 1.0 / (n + 1),
	λ0::Union{Nothing,Float64} = nothing)

	λ1 = isnothing(λ0) ? 1.0 / (2 * L) : λ0
	a_seq(n) = 100 / (n + 1)^(1.1)

	return (
		γ = γ,
		μ = μ0,
		λ1 = λ1,
		α = α0,
		β_zn = β_zn,
		θ_bar = θ_bar,
		β_decay = β_decay,
		a_seq = a_seq,
	)
end

"""
DIPCM_legacy — pre-2026-04-25 implementation, kept for reproducibility.

  z_n = x_n + β_zn (x_n - x_{n-1})        (constant β_zn)
  w_n = x_n + θ_bar (x_n - x_{n-1})        (constant θ_bar)
  y_n = J^A_λ(w_n - λ_n B(w_n))
  u_n = w_n - γ η_n d_n
  x_{n+1} = α z_n + σ_n u_n,  σ_n = (1 - α) - β_n,  β_n = β_decay(n)
"""
function DIPCM_legacy(problem::Problem;
	γ = 1.1, μ = 0.5, λ1 = 0.25,
	α = 0.2, β_zn = 0.0001, θ_bar = 0.9,
	β_decay = n -> 1.0 / (n + 1),
	a_seq = n -> 100 / (n + 1)^(1.1),
	tol = 1e-6, maxiter = 50000)::Solution

	Aresolvant, B, x0, x1, name = problem.Aλ, problem.B, problem.x0, problem.x1, problem.name
	dot, norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping

	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert 0 < μ < 1 "μ must be in (0,1)"
	@assert λ1 > 0 "λ₁ must be positive"
	@assert 0 < α < 1 "α must be in (0,1)"

	x_prev = copy(x0)
	x_curr = copy(x1)
	λ_curr = λ1
	n = 1
	converged = false
	J_A(x, λ) = Aresolvant(x, λ)

	history = Dict(
		:dk => Vector{Float64}(),
		:xk => Vector{Float64}(),
		:err => Vector{Float64}(),
	)

	while n <= maxiter
		normxk = norm(x_prev - x_curr)
		push!(history[:xk], normxk)

		# Halpern parameter
		β_n = β_decay(n)
		σ_n = (1 - α) - β_n
		if σ_n ≤ 0
			σ_n = eps()
			β_n = (1 - α) - σ_n
		end

		# θ_n: use fixed θ_bar in practice (adaptive control is for the proof only)
		θ_n = θ_bar

		# Step 1: z_n and w_n (double inertial)
		z_n = x_curr + β_zn * (x_curr - x_prev)
		w_n = x_curr + θ_n * (x_curr - x_prev)

		# y_n = J^A_λ(w_n - λ B(w_n))
		B_wn = B(w_n)
		y_n = J_A(w_n - λ_curr * B_wn, λ_curr)

		# Stopping criterion: y_n ≈ w_n
		stop, err = stopping_criterion(y_n - w_n, tol)
		push!(history[:err], err)
		if stop
			converged = true
			x_curr = y_n
			break
		end

		# Step 2: d_n, η_n, u_n (projection-contraction)
		B_yn = B(y_n)
		d_n = w_n - y_n - λ_curr * (B_wn - B_yn)

		normd = norm(d_n)
		push!(history[:dk], normd)
		η_n = if normd > eps()
			dot(w_n - y_n, d_n) / (normd^2)
		else
			0.0
		end

		u_n = w_n - γ * η_n * d_n

		# Step 3: x_{n+1} = α z_n + σ_n u_n  (two-term, weights < 1)
		x_next = α * z_n + σ_n * u_n

		# Step 4: Adaptive stepsize update
		B_diff_norm = norm(B_wn - B_yn)
		w_y_norm = norm(w_n - y_n)
		a_n = a_seq(n)

		λ_next = if B_diff_norm > eps()
			min(μ * w_y_norm / B_diff_norm, λ_curr + a_n)
		else
			λ_curr + a_n
		end

		x_prev = x_curr
		x_curr = x_next
		λ_curr = λ_next
		n += 1
	end

	return Solution{typeof(x_curr)}(;
		solver = "DIPCM_legacy",
		problem = problem,
		solution = x_curr,
		iterations = n - 1,
		converged = converged,
		parameters = Dict(
			:γ => γ, :μ => μ, :λ1 => λ1,
			:α => α, :β_zn => β_zn, :θ_bar => θ_bar,
			:β_decay => β_decay, :a_seq => a_seq,
			:tol => tol, :maxiter => maxiter,
		),
		history = history,
	)
end
