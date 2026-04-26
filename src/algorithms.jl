##############################################################################
# IPCMAS1 — Algorithm 1 (Paper's convex combination formula)
#
# Update: x_{n+1} = (1-α) z_n + α u_n   (constant α, weights sum to 1)
#
# Weak convergence under Assumption (A1)-(A5).
# NOTE: Converges very slowly in practice when L is large (λ_n ≈ μ/L).
#       Use DIPCM for practical computations.
##############################################################################

function get_IPCMAS1_params(L::Float64; γ = 1.1, μ0 = 0.5, α0 = 0.30, β0 = 0.0119, λ0::Union{Nothing, Float64} = nothing)
	# Defaults tuned by w11 sensitivity (2026-04-17): α=0.30, β=0.25·β_max(0.30)=0.0119,
	# θ̄=0.9, ξ_n=100/(n+1)^1.1. Valid under Assumption (A4)-(A5) since α<1/3 and
	# β_max(0.30) ≈ 0.0476 > 0.0119.

	μ = μ0
	λ1 = isnothing(λ0) ? 1.0 / (2 * L) : λ0
	a_seq(n) = 100 / (n + 1)^(1.1)
	θ_seq(n) = 0.9

	return (
		γ = γ,
		μ = μ,
		λ1 = λ1,
		β = β0,
		α = α0,
		a_seq = a_seq,
		θ_seq = θ_seq,
	)
end

function IPCMAS1(problem::Problem;
	γ = 1.8, μ = 0.5, λ1 = 0.25, α = 0.30, β = 0.0119,
	a_seq = n -> 0.0, θ_seq = n -> 0.9,
	tol = 1e-6, maxiter = 10000)::Solution

	Aresolvant, B, x0, x1, name = problem.Aλ, problem.B, problem.x0, problem.x1, problem.name
	dot, norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping
	# Validate parameters (paper Assumption A1–A5)
	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert 0 < μ < 1 "μ must be in (0,1)"
	@assert λ1 > 0 "λ₁ must be positive"
	@assert 0 < α < 1 / 3 "α must be in (0, 1/3) by Assumption (A5)"
	β_upper = (1 - 3α) / (3 * (1 - α))
	@assert 0 ≤ β < β_upper "β must be in [0, (1-3α)/(3(1-α)) = $(round(β_upper, digits=4))) by Assumption (A4)"

	# Initialize
	x_prev = copy(x0)
	x_curr = copy(x1)
	λ_curr = λ1
	n = 1
	converged = false
	# Resolvent operator J^A_λ
	# This should be provided based on the specific operator A
	J_A(x, λ) = Aresolvant(x, λ)  # This should be the resolvent of A
	history = Dict(
		:dk => Vector{Float64}(),
		:xk => Vector{Float64}(),
		:err => Vector{Float64}(),
	)
	while n <= maxiter
		normxk = norm(x_prev - x_curr)
		push!(history[:xk], normxk)
		# Get current parameters
		θ_n = θ_seq(n)
		a_n = a_seq(n)

		# Step 1: Compute zₙ, wₙ, and yₙ
		z_n = x_curr + β * (x_curr - x_prev)
		w_n = x_curr + θ_n * (x_curr - x_prev)

		# Compute yₙ = (I + λₙA)^(-1)(I - λₙB)(wₙ)
		B_wn = B(w_n)
		y_n = J_A(w_n - λ_curr * B_wn, λ_curr)

		# Check stopping criterion: if yₙ = wₙ
		stop, err = stopping_criterion(y_n - w_n, tol)
		push!(history[:err], err)
		if stop
			converged = true
			x_curr = y_n  # yₙ is a solution
			break
		end

		# Step 2: Compute uₙ = wₙ - γηₙdₙ
		B_yn = B(y_n)
		d_n = w_n - y_n - λ_curr * (B_wn - B_yn)

		# Compute ηₙ
		normd = norm(d_n)
		push!(history[:dk], normd)
		η_n = if normd > eps()
			dot(w_n - y_n, d_n) / (normd^2)
		else
			0.0
		end

		u_n = w_n - γ * η_n * d_n

		# Step 3: Compute xₙ₊₁ = (1-α)zₙ + α uₙ  (paper's convex combination)
		x_next = (1 - α) * z_n + α * u_n


		# Update λₙ₊₁
		B_diff_norm = norm(B_wn - B_yn)
		w_y_norm = norm(w_n - y_n)

		λ_next = if B_diff_norm > eps()
			min(μ * w_y_norm / B_diff_norm, λ_curr + a_n)
		else
			λ_curr + a_n
		end

		# Prepare for next iteration
		x_prev = x_curr
		x_curr = x_next
		λ_curr = λ_next
		n += 1
	end
	solution = Solution{typeof(x_curr)}(;
		solver = "IPCMAS1",
		problem = problem,
		solution = x_curr,
		iterations = n - 1,
		converged = converged,
		parameters = Dict(
			:γ => γ, :μ => μ, :λ1 => λ1, :α => α, :β => β,
			:a_seq => a_seq, :θ_seq => θ_seq,
			:tol => tol, :maxiter => maxiter,
		),
		history = history,
	)
	return solution
end



##############################################################################
# DIPCM — Double Inertial PCM with Implicit Contraction (paper §3.4)
#
# 3-term Halpern–Mann update with adaptive inertial parameters.
#   x_{n+1} = (1 - α_n - σ_n) z_n + σ_n u_n    (α_n is the Halpern weight on 0)
# Adaptive inertial parameters β'_n, θ_n controlled by sequences ε_n ≤ ε'_n
# with Σ ε'_n / α_n < ∞ (Assumption A5'').
#
# Strong convergence to P_Ω(0) without strong monotonicity, via Saejung–Yotkaew
# convergence lemma (Lemma 2.6 in Saejung & Yotkaew, Nonlinear Anal. 75 (2012)).
##############################################################################

function get_DIPCM_params(L::Float64;
	γ = 1.1, μ0 = 0.5,
	β_bar = 0.3, θ_bar = 0.9,
	α_seq = n -> 1.0 / (n + 1),
	σ_seq = nothing,                             # default below: σ_n = 0.8 - α_n
	ε_seq = n -> 5.0 / (n + 1)^2.1,              # caps β'_n
	εp_seq = n -> 10.0 / (n + 1)^2.1,            # caps θ_n;  must satisfy ε_n ≤ ε'_n
	ξ_seq = n -> 100.0 / (n + 1)^1.1,
	λ0::Union{Nothing,Float64} = nothing)

	λ1 = isnothing(λ0) ? 1.0 / (2 * L) : λ0
	σ_eff = isnothing(σ_seq) ? (n -> 0.8 - α_seq(n)) : σ_seq

	return (
		γ = γ,
		μ = μ0,
		λ1 = λ1,
		β_bar = β_bar,
		θ_bar = θ_bar,
		α_seq = α_seq,
		σ_seq = σ_eff,
		ε_seq = ε_seq,
		εp_seq = εp_seq,
		ξ_seq = ξ_seq,
	)
end

"""
DIPCM — Double Inertial PCM with Implicit Contraction (paper §3.4)

Step 1:
  β'_n = min{β̄, ε_n / ‖x_n - x_{n-1}‖, ε_n / ‖x_n - x_{n-1}‖²}    if x_n ≠ x_{n-1}
       = β̄                                                          otherwise
  θ_n  = min{θ̄, ε'_n / ‖x_n - x_{n-1}‖, ε'_n / ‖x_n - x_{n-1}‖²}  if x_n ≠ x_{n-1}
       = θ̄                                                          otherwise
  z_n = x_n + β'_n (x_n - x_{n-1})
  w_n = x_n + θ_n  (x_n - x_{n-1})
  y_n = J^A_{λ_n}(w_n - λ_n B(w_n))
Step 2: u_n = w_n - γ η_n d_n      (PCM contraction)
Step 3: x_{n+1} = (1 - α_n - σ_n) z_n + σ_n u_n
Step 4: λ_{n+1} = min{μ ‖w_n - y_n‖ / ‖B(w_n) - B(y_n)‖, λ_n + ξ_n}

Assumption halpern (paper line 1753):
  α_n → 0, Σ α_n = ∞;  σ_n ∈ (a, b) ⊂ (0, 1 - α_n);
  ε_n ≤ ε'_n with Σ ε'_n / α_n < ∞;  0 ≤ β̄ ≤ θ̄ ≤ 1 (A6'').

Strong convergence to P_Ω(0) by Theorem `thm:halpern-strong`.
"""
function DIPCM(problem::Problem;
	γ = 1.1, μ = 0.5, λ1 = 0.25,
	β_bar = 0.3, θ_bar = 0.9,
	α_seq = n -> 1.0 / (n + 1),
	σ_seq = n -> 0.8 - 1.0 / (n + 1),
	ε_seq = n -> 5.0 / (n + 1)^2.1,             # caps β'_n
	εp_seq = n -> 10.0 / (n + 1)^2.1,            # caps θ_n;  must satisfy ε_n ≤ ε'_n
	ξ_seq = n -> 100.0 / (n + 1)^1.1,
	tol = 1e-6, maxiter = 50000)::Solution

	Aresolvant, B, x0, x1, name = problem.Aλ, problem.B, problem.x0, problem.x1, problem.name
	dot, norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping

	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert 0 < μ < 1 "μ must be in (0,1)"
	@assert λ1 > 0 "λ₁ must be positive"
	@assert 0 ≤ β_bar ≤ θ_bar ≤ 1 "Need 0 ≤ β̄ ≤ θ̄ ≤ 1 (Assumption A6'')"

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
		diff = norm(x_curr - x_prev)
		push!(history[:xk], diff)

		ε_n  = ε_seq(n)
		εp_n = εp_seq(n)

		# Adaptive inertial parameters (paper eq:newbeta'n, eq:newthetan)
		if diff > eps()
			β_n = min(β_bar, ε_n / diff, ε_n / diff^2)
			θ_n = min(θ_bar, εp_n / diff, εp_n / diff^2)
		else
			β_n = β_bar
			θ_n = θ_bar
		end

		# Step 1: z_n, w_n, y_n
		z_n = x_curr + β_n * (x_curr - x_prev)
		w_n = x_curr + θ_n * (x_curr - x_prev)
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

		# Step 3: 3-term Halpern–Mann update
		α_n = α_seq(n)
		σ_n = σ_seq(n)
		@assert 0 < σ_n < 1 - α_n "σ_n must satisfy 0 < σ_n < 1 - α_n (got σ_$(n)=$σ_n, α_$(n)=$α_n)"
		x_next = (1 - α_n - σ_n) * z_n + σ_n * u_n

		# Step 4: λ_{n+1} update
		B_diff_norm = norm(B_wn - B_yn)
		w_y_norm = norm(w_n - y_n)
		ξ_n = ξ_seq(n)

		λ_next = if B_diff_norm > eps()
			min(μ * w_y_norm / B_diff_norm, λ_curr + ξ_n)
		else
			λ_curr + ξ_n
		end

		x_prev = x_curr
		x_curr = x_next
		λ_curr = λ_next
		n += 1
	end

	return Solution{typeof(x_curr)}(;
		solver = "DIPCM",
		problem = problem,
		solution = x_curr,
		iterations = n - 1,
		converged = converged,
		parameters = Dict(
			:γ => γ, :μ => μ, :λ1 => λ1,
			:β_bar => β_bar, :θ_bar => θ_bar,
			:α_seq => α_seq, :σ_seq => σ_seq,
			:ε_seq => ε_seq, :εp_seq => εp_seq,
			:ξ_seq => ξ_seq,
			:tol => tol, :maxiter => maxiter,
		),
		history = history,
	)
end


##############################################################################
# IPCMAS2 — Algorithm 2 (R-linear convergence under strong monotonicity)
##############################################################################

function get_IPCMAS2_params(L::Float64; γ = 1.8, μ0 = 0.5, α0 = 0.25, λ0::Union{Nothing, Float64} = nothing)
	# 	# Parameters from the paper for Example 1

	μ = μ0
	λ1 = isnothing(λ0) ? 1.0 / (2 * L) : λ0 #  # Constant step size
	α = α0 # ∈ (0, 1/3)
	θ = 0.5
	# β_seq(n) = 1.0 / (5 * n + 1)
	a_seq(n) = 100 / (n + 1)^(1.1) #aseq()
	aseq() = begin
		prefix = Float64[0.0]   # prefix[k+1] stores sum_{i=1}^k 1/i^2
		function (n::Int)
			n ≥ 1 || throw(ArgumentError("n must be ≥ 0"))
			while length(prefix) - 1 < n
				k = length(prefix)          # next i to add
				push!(prefix, prefix[end] + 1.0 / (2^k))
			end
			return prefix[n+1]
		end
	end
	return (
		γ = γ,
		μ = μ,
		λ1 = λ1,
		α = α,
		θ = θ,
		a_seq = a_seq,
	)

end

function IPCMAS2(problem::Problem;
	γ = 1.8, μ = 0.5, λ1 = 0.5, α = 0.3, θ = 0.5,
	a_seq = n -> 1 / n^2, tol = 1e-6, maxiter = 1000)

	Aresolvant, B, x0, x1, name = problem.Aλ, problem.B, problem.x0, problem.x1, problem.name
	dot, norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping
	# Validate parameters
	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert 0 < μ < 1 "μ must be in (0,1)"
	@assert λ1 > 0 "λ1 must be positive"

	# Initialize
	x_prev = copy(x0)
	x_curr = copy(x1)
	λ_curr = λ1
	n = 1
	converged = false

	# Resolvent operator J^A_λ
	# This should be provided based on the specific operator A
	J_A(x, λ) = Aresolvant(x, λ)  # This should be the resolvent of A
	history = Dict(
		:dk => Vector{Float64}(),
		:xk => Vector{Float64}(),
		:err => Vector{Float64}(),
	)
	while n <= maxiter
		normxk = norm(x_prev - x_curr)
		push!(history[:xk], normxk)

		# Step 1: Compute w_n and y_n
		w_n = x_curr + θ * (x_curr - x_prev)

		# Compute y_n = J^A_{λ_n}(w_n - λ_n*B(w_n))
		Bw_n = B(w_n)
		y_n = J_A(w_n - λ_curr * Bw_n, λ_curr)

		# Check stopping criterion
		stop, err = stopping_criterion(y_n - w_n, tol)
		push!(history[:err], err)
		if stop
			converged = true
			x_curr = y_n  # yₙ is a solution
			break
		end

		# Step 2: Compute u_n
		By_n = B(y_n)
		d_n = w_n - y_n - λ_curr * (Bw_n - By_n)

		# Compute η_n
		normd = norm(d_n)
		push!(history[:dk], normd)
		η_n = if normd > eps()
			dot(w_n - y_n, d_n) / (normd^2)
		else
			0.0
		end

		u_n = w_n - γ * η_n * d_n

		# Step 3: Update x_{n+1}
		x_next = (1 - α) * x_curr + α * u_n

		# Update λ_{n+1}
		B_diff_norm = norm(Bw_n - By_n)
		a_n = a_seq(n)

		λ_next = if B_diff_norm > eps()
			min(μ * norm(w_n - y_n) / B_diff_norm, λ_curr + a_n)
		else
			λ_curr + a_n
		end

		# Prepare for next iteration
		x_prev = x_curr
		x_curr = x_next
		λ_curr = λ_next
		n += 1
	end

	solution = Solution{typeof(x_curr)}(;
		solver = "IPCMAS2",
		problem = problem,
		solution = x_curr,
		iterations = n - 1,
		converged = converged,
		parameters = Dict(
			:γ => γ,
			:μ => μ,
			:λ1 => λ1,
			:α => α,
			:θ => θ,
			:a_seq => a_seq,
			:tol => tol,
			:maxiter => maxiter,
		),
		history = history,
	)
	return solution
end


function get_DeyHICPP_params(L::Float64; λ0 = 1 / 2L, β_seq = n -> 1.0 / (5 * n + 1))
	λ_constant = λ0
	return (
		γ = 1.5,
		λ_seq = n -> λ_constant,
		α = 0.5,
		τ_seq = n -> 1.0 / n^2,
		β_seq = β_seq,
		θ_seq = n -> 0.8 - β_seq(n),
	)
end


"""
Algorithm 1
Dey, S. (2023). A hybrid inertial and contraction proximal point algorithm for monotone variational inclusions. 
Numerical Algorithms, 93(1), 1–25. https://doi.org/10.1007/s11075-022-01400-0

"""

function DeyHICPP(problem::Problem;
	γ = 1.5, λ_seq = n -> 0.01, α = 0.5,
	τ_seq = n -> 1.0 / n^2, β_seq = n -> 1.0 / (5n + 1), θ_seq = n -> 0.8 - 1.0 / (5n + 1),
	tol = 1e-6, maxiter = 10000)::Solution
	Aresolvant, B, x0, x1, name = problem.Aλ, problem.B, problem.x0, problem.x1, problem.name
	_dot, _norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping

	# Validate parameters
	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert α > 0 "α must be positive"

	# Initialize
	x_prev = copy(x0)
	x_curr = copy(x1)
	n = 1
	converged = false
	history = Dict(
		:dk => Vector{Float64}(),
		:xk => Vector{Float64}(),
		:err => Vector{Float64}(),
	)
	while n <= maxiter

		# Get current parameters
		λₙ = λ_seq(n)
		τₙ = τ_seq(n)
		βₙ = β_seq(n)
		θₙ = θ_seq(n)

		# Step 1: Choose αₙ such that 0 ≤ αₙ ≤ ᾱₙ
		x_diff_norm = _norm(x_curr - x_prev)
		push!(history[:xk], x_diff_norm)
		ᾱₙ = if x_diff_norm > eps()
			min(α, τₙ / x_diff_norm)
		else
			α
		end
		αₙ = ᾱₙ  # We choose αₙ = ᾱₙ for best performance

		# Step 2: Compute wₙ and yₙ
		wₙ = x_curr + αₙ * (x_curr - x_prev)

		# Compute yₙ = J^A_{λₙ}(wₙ - λₙf(wₙ))
		B_wₙ = B(wₙ)
		yₙ = Aresolvant(wₙ - λₙ * B_wₙ, λₙ)

		# Check stopping criterion: if yₙ = wₙ
		stop, err = stopping_criterion(yₙ - wₙ, tol)
		push!(history[:err], err)
		if stop
			converged = true
			x_curr = yₙ  # yₙ is a solution
			break
		end

		# Step 3: Calculate zₙ = wₙ - γηₙdₙ
		B_yₙ = B(yₙ)
		dₙ = wₙ - yₙ - λₙ * (B_wₙ - B_yₙ)

		# Compute ηₙ
		normd = _norm(dₙ)
		push!(history[:dk], normd)
		ηₙ = if normd > eps()
			_dot(wₙ - yₙ, dₙ) / (normd^2)
		else
			0.0
		end

		zₙ = wₙ - γ * ηₙ * dₙ

		# Step 4: Calculate xₙ₊₁
		x_next = (1 - θₙ - βₙ) * x_curr + θₙ * zₙ

		# Prepare for next iteration
		x_prev = x_curr
		x_curr = x_next
		n += 1
	end
	solution = Solution{typeof(x_curr)}(;
		solver = "DeyHICPP",
		problem = problem,
		solution = x_curr,
		iterations = n - 1,
		converged = converged,
		parameters = Dict(
			:γ => γ, :λ_seq => λ_seq, :α => α,
			:τ_seq => τ_seq, :β_seq => β_seq, :θ_seq => θ_seq,
			:tol => 1e-6, :maxiter => 10000,
		),
		history = history,
	)
	return solution
end



##############################################################################
# Suantai, Cholamjiak, Inkrong, Kesornprom (2024)
# "A fast contraction algorithm using two inertial extrapolations for
#  variational inclusion problem and data classification"
# Carpathian J. Math. 40(3), 737-752.
# Algorithm 3.1
##############################################################################

"""
Parameters from Suantai et al. (2024), Section 4 (paper defaults):
  γ = 0.1, μ = 0.9, λ₀ = 0.01, η_k = 1/(k+1)², δ_k = 1/(5k+2)³
"""
function get_Suantai2024_params(L::Float64;
	γ = 0.1, μ0 = 0.9, λ0 = 0.01,
	η_seq = k -> 1.0 / (k + 1)^2,
	δ_seq = k -> 1.0 / (5 * k + 2)^3)

	return (
		γ = γ,
		μ = μ0,
		λ1 = λ0,
		η_seq = η_seq,
		δ_seq = δ_seq,
	)
end


"""
Algorithm 3.1 from Suantai et al. (2024), Carpathian J. Math. 40(3), 737-752.

Two inertial extrapolation terms combined in a single step using three
consecutive iterates (x_k, x_{k-1}, x_{k-2}):

    w_k = x_k + η_k(x_k - x_{k-1}) + δ_k(x_{k-1} - x_{k-2})
    y_k = J^A_{λ_k}(w_k - λ_k f(w_k))
    d(w_k, y_k) = (w_k - y_k) - λ_k(f(w_k) - f(y_k))
    x_{k+1} = w_k - γ β_k d(w_k, y_k)

with adaptive stepsize λ_{k+1} = min{μ‖w_k-y_k‖/‖f(w_k)-f(y_k)‖, λ_k}.
"""
function Suantai2024(problem::Problem;
	γ = 1.5, μ = 0.5, λ1 = 0.5,
	η_seq = k -> 1.0 / (k + 1)^2,
	δ_seq = k -> 1.0 / (k + 1)^2,
	tol = 1e-6, maxiter = 10000)::Solution

	Aresolvant, B, x0, x1, name = problem.Aλ, problem.B, problem.x0, problem.x1, problem.name
	_dot, _norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping

	# Validate parameters
	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert 0 < μ < 1 "μ must be in (0,1)"
	@assert λ1 > 0 "λ₁ must be positive"

	# Initialize: need x_{k-2}, x_{k-1}, x_k (three consecutive iterates)
	x_pp = copy(x0)   # x_{k-2}
	x_prev = copy(x0) # x_{k-1}
	x_curr = copy(x1) # x_k
	λ_curr = λ1
	k = 1
	converged = false

	J_A(x, λ) = Aresolvant(x, λ)
	history = Dict(
		:dk => Vector{Float64}(),
		:xk => Vector{Float64}(),
		:err => Vector{Float64}(),
	)

	while k <= maxiter
		normxk = _norm(x_curr - x_prev)
		push!(history[:xk], normxk)

		# Get current inertial parameters
		η_k = η_seq(k)
		δ_k = δ_seq(k)

		# Step 1: w_k = x_k + η_k(x_k - x_{k-1}) + δ_k(x_{k-1} - x_{k-2})
		w_k = x_curr + η_k * (x_curr - x_prev) + δ_k * (x_prev - x_pp)

		# Step 2: y_k = J^A_{λ_k}(w_k - λ_k f(w_k))
		B_wk = B(w_k)
		y_k = J_A(w_k - λ_curr * B_wk, λ_curr)

		# Check stopping criterion
		stop, err = stopping_criterion(y_k - w_k, tol)
		push!(history[:err], err)
		if stop
			converged = true
			x_curr = y_k
			break
		end

		# Step 3: d(w_k, y_k) and β_k
		B_yk = B(y_k)
		d_k = w_k - y_k - λ_curr * (B_wk - B_yk)

		normd = _norm(d_k)
		push!(history[:dk], normd)

		ϕ_k = _dot(w_k - y_k, d_k)
		β_k = if normd > eps()
			ϕ_k / (normd^2)
		else
			0.0
		end

		# Step 4: x_{k+1} = w_k - γ β_k d(w_k, y_k)
		x_next = w_k - γ * β_k * d_k

		# Update λ_{k+1} = min{μ‖w_k-y_k‖/‖f(w_k)-f(y_k)‖, λ_k}
		B_diff_norm = _norm(B_wk - B_yk)
		w_y_norm = _norm(w_k - y_k)

		λ_next = if B_diff_norm > eps()
			min(μ * w_y_norm / B_diff_norm, λ_curr)
		else
			λ_curr
		end

		# Shift iterates: x_{k-2} <- x_{k-1}, x_{k-1} <- x_k, x_k <- x_{k+1}
		x_pp = x_prev
		x_prev = x_curr
		x_curr = x_next
		λ_curr = λ_next
		k += 1
	end

	solution = Solution{typeof(x_curr)}(;
		solver = "SICIP",
		problem = problem,
		solution = x_curr,
		iterations = k - 1,
		converged = converged,
		parameters = Dict(
			:γ => γ, :μ => μ, :λ1 => λ1,
			:η_seq => η_seq, :δ_seq => δ_seq,
			:tol => tol, :maxiter => maxiter,
		),
		history = history,
	)
	return solution
end

##############################################################################

function get_DongIPCA_params(L::Float64;
	γ::Float64 = 1.5,
	τ0::Union{Nothing, Float64} = nothing,
	α::Float64 = 0.4,
	α_seq::Function = n -> (n == 1 ? 0.0 : α),
)
	# choose τ from Lipschitz constant if not provided
	τ = isnothing(τ0) ? 1.0 / (2L) : τ0

	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert τ > 0 "τ must be > 0"
	@assert 0 ≤ α < 1 "α must be in [0,1)"

	return (
		γ = γ,
		τ = τ,
		α_seq = α_seq,
	)
end


"""
Algorithm 3.1 (The inertial projection and contraction algorithm)
Dong, Q. L., Cho, Y. J., Zhong, L. L., & Rassias, Th. M. (2018). Inertial projection and contraction algorithms for variational inequalities. 
Journal of Global Optimization, 70(3), 687–704. https://doi.org/10.1007/s10898-017-0506-0

Inputs via `problem`:
  - f(x)       : single-valued mapping H→H (as `problem.f`; fallback to `problem.B`)
  - PC(x)      : projection onto C (as `problem.PC(x)`)
  - x0, x1     : initial guesses
  - dot, norm  : inner product & norm (defaults from `problem`)
  - stopping   : (x,tol) → Bool  (defaults from `problem`)

Keyword params:
  - γ ∈ (0,2), τ>0
  - α_seq(k) (nondecreasing, α₁=0, 0≤α_k≤α<1)
  - tol, maxiter
"""
function DongIPCA(problem::Problem;
	γ::Float64 = 1.8,           # relaxation in (0,2)
	τ::Float64 = 1.0,           # stepsize > 0
	α_seq::Function = k -> (k == 1 ? 0.0 : 0.3),  # inertial schedule
	tol::Float64 = 1e-6,
	maxiter::Int = 10_000,
)::Solution

	# ---- pull operators from `problem` (keep these two lines if your field names differ)
	f = problem.B
	PC = problem.Aλ
	# ---- spaces, io, and helpers
	dot = problem.dot
	norm = problem.norm
	stopping_criterion = problem.stopping

	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert τ > 0 "τ must be positive"

	x_prev = copy(problem.x0)
	x_curr = copy(problem.x1)

	niter = 0
	converged = false
	history = Dict(
		:dk => Vector{Float64}(),
		:xk => Vector{Float64}(),
		:err => Vector{Float64}(),
	)
	while niter < maxiter
		normxk = norm(x_prev - x_curr)
		push!(history[:xk], normxk)

		k  = niter + 1
		αk = α_seq(k)

		# w^k = x^k + α_k (x^k - x^{k-1})
		w = x_curr .+ αk .* (x_curr .- x_prev)

		# y^k = P_C( w^k − τ f(w^k) )
		fw = f(w)
		y  = PC(w .- τ .* fw, τ)

		wy = w .- y

		# d(w^k,y^k) = (w^k − y^k) − τ( f(w^k) − f(y^k) )
		fy = f(y)
		d  = wy .- τ .* (fw .- fy)

		nd2 = (norm(d))^2
		ϕ   = dot(wy, d)
		βk  = nd2 > eps() ? ϕ / nd2 : 0.0

		# stopping: y^k = w^k OR d = 0
		stop1, err = stopping_criterion(wy, tol)
		stop2, dk = stopping_criterion(d, tol)
		push!(history[:err], err)
		push!(history[:dk], dk)
		if stop1 || stop2
			converged = true
			x_next = y
			x_prev, x_curr = x_curr, x_next
			niter += 1
			break
		end

		# x^{k+1} = w^k − γ β_k d(w^k, y^k)
		x_next = w .- (γ * βk) .* d


		x_prev, x_curr = x_curr, x_next
		niter += 1
	end


	return Solution{typeof(x_curr)}(;
		solver = "DongIPCA",
		problem = problem,
		solution = x_curr,
		iterations = niter,
		converged = converged,
		parameters = Dict(
			:γ => γ, :τ => τ, :tol => tol, :maxiter => maxiter,
			:α_seq => α_seq,
		),
		history = history,
	)
end


##############################################################################
# TanQin2024 — Tan & Qin (2024), Algorithm 3.1
# "On relaxed inertial projection and contraction algorithms for solving
#  monotone inclusion problems"
#
# Problem: 0 ∈ (A + B)x, A single-valued monotone + L-Lipschitz, B maximal monotone.
# In our codebase: A ↔ problem.B (single-valued),  B ↔ problem.Aλ (resolvent).
#
# Update (using our notation, with s_n playing role of iterate):
#   u_n = s_n + ζ(s_n − s_{n−1})
#   t_n = J^B_{χ_n}(u_n − χ_n A(u_n))
#   g_n = u_n − t_n − χ_n(A(u_n) − A(t_n))
#   θ_n = ⟨u_n − t_n, g_n⟩ / ‖g_n‖²          (correction coefficient)
#   q_n = u_n − δ θ_n g_n
#   s_{n+1} = (1 − φ) s_n + φ q_n
#
# Non-monotonic adaptive stepsize:
#   χ_{n+1} = min{κ‖u_n − t_n‖/‖A(u_n) − A(t_n)‖, ξ_n χ_n + τ_n}   if A(u_n) ≠ A(t_n)
#           = ξ_n χ_n + τ_n                                         otherwise
#
# Feasibility: (1/φ) − 1 − φ ζ (1 + ζ) > 0  (paper eq. 3.1).
# Weak convergence; R-linear if B (set-valued) is strongly monotone.
#
# NOTE on τ_n: Paper's Condition (C3) requires Σ τ_n < ∞. The authors' own
# numerical experiments (Section 5) use τ_n = 1/(n+1), which DIVERGES and
# thus violates (C3). We default to 1/(n+1) to exactly reproduce their
# published experiments; pass `τ_seq = n -> 1/(n+1)^2` for the theoretically
# valid (summable) choice.
##############################################################################

function get_TanQin2024_params(L::Float64;
	κ = 0.5, δ = 1.5, ζ = 0.2, φ = 0.7,
	χ1::Union{Nothing,Float64} = nothing,
	ξ_seq = n -> 1.0 + 1.0 / (n + 1)^2,
	τ_seq = n -> 1.0 / (n + 1))

	χ1_val = isnothing(χ1) ? 1.0 : χ1
	return (
		κ = κ,
		δ = δ,
		ζ = ζ,
		φ = φ,
		χ1 = χ1_val,
		ξ_seq = ξ_seq,
		τ_seq = τ_seq,
	)
end

"""
Algorithm 3.1 of Tan & Qin (2024).
"""
function TanQin2024(problem::Problem;
	κ = 0.5, δ = 1.5, ζ = 0.2, φ = 0.7,
	χ1 = 1.0,
	ξ_seq = n -> 1.0 + 1.0 / (n + 1)^2,
	τ_seq = n -> 1.0 / (n + 1),
	tol = 1e-6, maxiter = 10000)::Solution

	Aresolvant, B_op, x0, x1 = problem.Aλ, problem.B, problem.x0, problem.x1
	_dot, _norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping

	@assert 0 < κ < 1 "κ must be in (0,1)"
	@assert 0 < δ < 2 "δ must be in (0,2)"
	@assert 0 < ζ < 1 "ζ must be in (0,1)"
	@assert 0 < φ < 1 "φ must be in (0,1)"
	@assert χ1 > 0 "χ₁ must be positive"
	feas = (1 / φ) * (1 - φ) - φ * ζ * (1 + ζ)
	@assert feas > 0 "Feasibility condition (1/φ)(1−φ) − φζ(1+ζ) > 0 violated (got $(round(feas,digits=4)))"

	s_prev = copy(x0)
	s_curr = copy(x1)
	χ_curr = χ1
	n = 1
	converged = false

	history = Dict{Symbol,Vector{<:Real}}(
		:dk     => Float64[],
		:xk     => Float64[],
		:err    => Float64[],
		:lambda => Float64[],
		:eta    => Float64[],
		:x_norm => Float64[],
		:wy_norm => Float64[],
		:t_iter => Float64[],
	)

	while n <= maxiter
		t0 = time_ns()

		push!(history[:xk], _norm(s_curr - s_prev))
		push!(history[:x_norm], _norm(s_curr))
		push!(history[:lambda], χ_curr)

		# Step 1: inertial extrapolation
		u_n = s_curr + ζ * (s_curr - s_prev)

		# Step 2: forward-backward (resolvent of B applied to forward step of A)
		Au_n = B_op(u_n)   # A in paper = B in our codebase
		t_n = Aresolvant(u_n - χ_curr * Au_n, χ_curr)

		wy = u_n - t_n
		push!(history[:wy_norm], _norm(wy))

		stop, err = stopping_criterion(wy, tol)
		push!(history[:err], err)
		if stop
			converged = true
			s_curr = t_n
			push!(history[:dk], 0.0)
			push!(history[:eta], 0.0)
			push!(history[:t_iter], (time_ns() - t0) / 1e9)
			break
		end

		# Step 3: correction direction g_n
		At_n = B_op(t_n)
		g_n = wy - χ_curr * (Au_n - At_n)

		normg = _norm(g_n)
		push!(history[:dk], normg)

		θ_n = normg > eps() ? _dot(wy, g_n) / (normg^2) : 0.0
		push!(history[:eta], θ_n)

		# Step 4: q_n and s_{n+1}
		q_n = u_n - δ * θ_n * g_n
		s_next = (1 - φ) * s_curr + φ * q_n

		# Step 5: adaptive χ update
		A_diff_norm = _norm(Au_n - At_n)
		ξ_n = ξ_seq(n)
		τ_n = τ_seq(n)
		χ_next = if A_diff_norm > eps()
			min(κ * _norm(wy) / A_diff_norm, ξ_n * χ_curr + τ_n)
		else
			ξ_n * χ_curr + τ_n
		end

		s_prev = s_curr
		s_curr = s_next
		χ_curr = χ_next
		push!(history[:t_iter], (time_ns() - t0) / 1e9)
		n += 1
	end

	return Solution{typeof(s_curr)}(;
		solver = "TanQin2024",
		problem = problem,
		solution = s_curr,
		iterations = n - 1,
		converged = converged,
		parameters = Dict(
			:κ => κ, :δ => δ, :ζ => ζ, :φ => φ, :χ1 => χ1,
			:ξ_seq => ξ_seq, :τ_seq => τ_seq,
			:tol => tol, :maxiter => maxiter,
		),
		history = history,
	)
end


##############################################################################
# ChenMiPCA — Chen, Zhang, Dong (2020), Algorithm 3.1 (Multi-step inertial PCM)
# "Multi-step inertial proximal contraction algorithms for monotone
#  variational inclusion problems"
#
# Problem: 0 ∈ A(x) + f(x), A maximal monotone, f monotone + L-Lipschitz.
# In our codebase: f ↔ problem.B,  A ↔ problem.Aλ (resolvent).
#
# Update (s-step inertial, here s = 2):
#   ω_k = x_k + α₁(x_k − x_{k−1}) + α₂(x_{k−1} − x_{k−2})
#   y_k = J^A_{λ}(ω_k − λ f(ω_k))
#   d_k = (ω_k − y_k) − λ(f(ω_k) − f(y_k))
#   β_k = ⟨ω_k − y_k, d_k⟩ / ‖d_k‖²
#   x_{k+1} = ω_k − γ β_k d_k
#
# Constant stepsize λ ∈ (0, 1/L). Weak convergence. Multi-step inertial.
##############################################################################

function get_ChenMiPCA_params(L::Float64;
	γ = 1.95, α1 = 0.9, α2 = -0.01,
	λ0::Union{Nothing,Float64} = nothing)

	λ = isnothing(λ0) ? 1.0 / (1.05 * L) : λ0
	return (
		γ = γ,
		α1 = α1,
		α2 = α2,
		λ1 = λ,
	)
end

"""
Algorithm 3.1 of Chen, Zhang, Dong (2020) — 2-step inertial PCM.
"""
function ChenMiPCA(problem::Problem;
	γ = 1.95, α1 = 0.9, α2 = -0.01,
	λ1 = 0.1,
	tol = 1e-6, maxiter = 10000)::Solution

	Aresolvant, B_op, x0, x1 = problem.Aλ, problem.B, problem.x0, problem.x1
	_dot, _norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping

	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert λ1 > 0 "λ must be positive"

	# Paper (Algorithm 3.1) initialization: "Choose x_0 ∈ H, x_{-i-1} = x_0 for
	# i ∈ S\{0}". This is a SINGLE-POINT initial convention — all prior iterates
	# equal x_0, so the first inertial extrapolation is zero. We use problem.x0
	# only (problem.x1 is ignored for Chen). Using x1 here would inject a
	# spurious α₁·(x1 − x0) term at iteration 1 with α₁=0.9, which blows up on
	# large-L problems (observed: 0/75 on Example 1 VIP before this fix).
	x_pp = copy(x0)   # x_{k−2} = x_0
	x_prev = copy(x0) # x_{k−1} = x_0
	x_curr = copy(x0) # x_k    = x_0
	λ = λ1
	k = 1
	converged = false

	history = Dict{Symbol,Vector{<:Real}}(
		:dk     => Float64[],
		:xk     => Float64[],
		:err    => Float64[],
		:lambda => Float64[],
		:eta    => Float64[],
		:x_norm => Float64[],
		:wy_norm => Float64[],
		:t_iter => Float64[],
	)

	while k <= maxiter
		t0 = time_ns()

		push!(history[:xk], _norm(x_curr - x_prev))
		push!(history[:x_norm], _norm(x_curr))
		push!(history[:lambda], λ)

		# Step 1: 2-step inertial extrapolation
		ω_k = x_curr + α1 * (x_curr - x_prev) + α2 * (x_prev - x_pp)

		# Step 2: resolvent + forward step
		f_ω = B_op(ω_k)
		y_k = Aresolvant(ω_k - λ * f_ω, λ)

		wy = ω_k - y_k
		push!(history[:wy_norm], _norm(wy))

		stop, err = stopping_criterion(wy, tol)
		push!(history[:err], err)
		if stop
			converged = true
			x_curr = y_k
			push!(history[:dk], 0.0)
			push!(history[:eta], 0.0)
			push!(history[:t_iter], (time_ns() - t0) / 1e9)
			break
		end

		# Step 3: contraction step
		f_y = B_op(y_k)
		d_k = wy - λ * (f_ω - f_y)

		normd = _norm(d_k)
		push!(history[:dk], normd)

		β_k = normd > eps() ? _dot(wy, d_k) / (normd^2) : 0.0
		push!(history[:eta], β_k)

		x_next = ω_k - γ * β_k * d_k

		# Shift iterates
		x_pp = x_prev
		x_prev = x_curr
		x_curr = x_next

		push!(history[:t_iter], (time_ns() - t0) / 1e9)
		k += 1
	end

	return Solution{typeof(x_curr)}(;
		solver = "ChenMiPCA",
		problem = problem,
		solution = x_curr,
		iterations = k - 1,
		converged = converged,
		parameters = Dict(
			:γ => γ, :α1 => α1, :α2 => α2, :λ1 => λ1,
			:tol => tol, :maxiter => maxiter,
		),
		history = history,
	)
end


##############################################################################
# PeeyadaIMFBSA — Peeyada, Suparatulatorn, Cholamjiak (2022), Algorithm 3.1
# "An inertial Mann forward-backward splitting algorithm of variational
#  inclusion problems and its application"
#
# Problem: 0 ∈ F(x) + G(x), F is β-COCOERCIVE, G maximal monotone.
# In our codebase: F ↔ problem.B,  G ↔ problem.Aλ (resolvent).
#
# IMPORTANT: Paper requires F β-cocoercive (stronger than monotone+Lipschitz).
# For our benchmarks where B = ∇((1/2) x^T M x) with M symmetric PSD, B is the
# gradient of a convex quadratic and thus (1/L)-cocoercive by Baillon–Haddad,
# so Peeyada's assumptions are satisfied. Document this in the paper's text.
#
# Update (inertial + Mann relaxation, simple forward-backward):
#   y_n = x_n + ξ_n(x_n − x_{n−1})            (inertial, adaptive ξ_n)
#   z_n = y_n + α_n(x_n − y_n)                 (Mann relaxation)
#   x_{n+1} = J^G_{γ_n}(z_n − γ_n F(z_n))
#
# Paper's adaptive ξ_n (data-classification choice, Section 4):
#   Δ = ‖x_n − x_{n−1}‖
#   ξ̃_n = 1/(Δ² + n²)
#   ξ_n = min(ξ̃_n/Δ, 0.5)  if Δ > 0
#        = 0.5               if Δ = 0
# This ensures Σ ξ_n ‖x_n − x_{n−1}‖ < ∞ (paper's summability condition).
#
# Paper's defaults: α_n = n/(2n+1), γ_n = 1.999/(2L) ≈ 1/L  (constant).
# Weak convergence.  Stopping: ‖x_{n+1} − z_n‖ (FB residual).
##############################################################################

"""
Default adaptive ξ rule of Peeyada 2022 (data classification, Sec. 4).
Takes iteration `n` and iterate gap Δ = ‖x_n − x_{n−1}‖.
"""
_peeyada_xi_rule(n::Int, Δ::Real) =
	Δ > eps() ? min(1.0 / (Δ^2 + n^2) / Δ, 0.5) : 0.5

function get_PeeyadaIMFBSA_params(L::Float64;
	γ_const::Union{Nothing,Float64} = nothing,
	ξ_rule::Function = _peeyada_xi_rule,
	α_seq = n -> n / (2n + 1))

	γ_val = isnothing(γ_const) ? 1.999 / (2.0 * L) : γ_const
	return (
		γ_const = γ_val,
		ξ_rule = ξ_rule,
		α_seq = α_seq,
	)
end

"""
Algorithm 3.1 of Peeyada, Suparatulatorn, Cholamjiak (2022) — inertial Mann
forward-backward splitting.

`ξ_rule(n, Δ)` is a function of iteration index and iterate gap; default is
the paper's adaptive rule. Pass e.g. `(n, Δ) -> 1/(n+1)^2` for a simple
iteration-only rule.
"""
function PeeyadaIMFBSA(problem::Problem;
	γ_const = 0.1,
	ξ_rule::Function = _peeyada_xi_rule,
	α_seq = n -> n / (2n + 1),
	tol = 1e-6, maxiter = 10000)::Solution

	Aresolvant, B_op, x0, x1 = problem.Aλ, problem.B, problem.x0, problem.x1
	_dot, _norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping

	@assert γ_const > 0 "γ must be positive"

	x_prev = copy(x0)
	x_curr = copy(x1)
	n = 1
	converged = false

	history = Dict{Symbol,Vector{<:Real}}(
		:dk     => Float64[],
		:xk     => Float64[],
		:err    => Float64[],
		:lambda => Float64[],
		:eta    => Float64[],
		:x_norm => Float64[],
		:wy_norm => Float64[],
		:t_iter => Float64[],
	)

	while n <= maxiter
		t0 = time_ns()

		Δ = _norm(x_curr - x_prev)
		push!(history[:xk], Δ)
		push!(history[:x_norm], _norm(x_curr))
		push!(history[:lambda], γ_const)

		ξ_n = ξ_rule(n, Δ)
		α_n = α_seq(n)
		push!(history[:eta], α_n)   # Mann relaxation coefficient

		# Step 1: inertial
		y_n = x_curr + ξ_n * (x_curr - x_prev)
		# Step 2: Mann relaxation
		z_n = y_n + α_n * (x_curr - y_n)
		# Step 3: forward-backward
		F_zn = B_op(z_n)
		x_next = Aresolvant(z_n - γ_const * F_zn, γ_const)

		# Stopping: x_{n+1} = z_n ⇒ z_n is a fixed point of J^G_γ(I − γF)
		residual = x_next - z_n
		push!(history[:wy_norm], _norm(residual))
		push!(history[:dk], _norm(residual))

		stop, err = stopping_criterion(residual, tol)
		push!(history[:err], err)
		if stop
			converged = true
			x_curr = x_next
			push!(history[:t_iter], (time_ns() - t0) / 1e9)
			break
		end

		x_prev = x_curr
		x_curr = x_next

		push!(history[:t_iter], (time_ns() - t0) / 1e9)
		n += 1
	end

	return Solution{typeof(x_curr)}(;
		solver = "PeeyadaIMFBSA",
		problem = problem,
		solution = x_curr,
		iterations = n - 1,
		converged = converged,
		parameters = Dict(
			:γ_const => γ_const, :ξ_rule => ξ_rule, :α_seq => α_seq,
			:tol => tol, :maxiter => maxiter,
		),
		history = history,
	)
end
