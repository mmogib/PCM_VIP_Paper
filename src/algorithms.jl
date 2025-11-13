function get_IPCMAS1_params(L::Float64; γ = 1.1, μ0 = 0.5, α0 = 0.25, β0 = 0.0001, λ0::Union{Nothing, Float64} = nothing)

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
				push!(prefix, prefix[end] + 1.0 / (2^k))
			end
			return prefix[n+1]
		end
	end

	β_seq(n) = 1.0 / (5 * n + 1)
	α_seq(n) = 0.8 - β_seq(n)
	# α_seq(n) = α_fixed #  0.8 - β_seq(n)
	a_seq(n) = 100 / (n + 1)^(1.1) #aseq()
	# ι_seq = n -> 1.0 / n^2
	θ_seq(n) = 0.9

	return (
		γ = γ,
		μ = μ,
		λ1 = λ1,
		β = β,
		α = α_fixed,
		β_seq = β_seq,
		α_seq = α_seq,
		a_seq = a_seq,
		θ_seq = θ_seq,
	)
end

function IPCMAS1(problem::Problem;
	γ = 1.8, μ = 0.5, λ1 = 0.25, α = 0.3, β = 0.1,
	β_seq = n -> 0.5, α_seq = n -> 0.3, a_seq = n -> 0.0, θ_seq = n -> 0.9,
	tol = 1e-6, maxiter = 10000)::Solution

	Aresolvant, B, x0, x1, name = problem.Aλ, problem.B, problem.x0, problem.x1, problem.name
	dot, norm = problem.dot, problem.norm
	stopping_criterion = problem.stopping
	# Validate parameters
	@assert 0 < γ < 2 "γ must be in (0,2)"
	@assert 0 < μ < 1 "μ must be in (0,1)"
	@assert λ1 > 0 "λ₁ must be positive"

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
		β_n = β_seq(n)
		θ_n = θ_seq(n)
		a_n = a_seq(n)
		α_n = α_seq(n)

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

		# Step 3: Compute xₙ₊₁
		x_next = (1 - α_n - β_n) * z_n + α_n * u_n


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
			:β_seq => β_seq, :α_seq => α_seq, :a_seq => a_seq,
			:tol => 1e-6, :maxiter => 10000,
		),
		history = history,
	)
	return solution
end



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
