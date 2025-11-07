function get_IPCMAS1_params(L::Float64; μ0 = 0.5, α0 = 0.25, β0 = 0.0001, λ0::Union{Nothing, Float64} = nothing)
	γ = 1.8
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

function IPCMAS1(problem::Problem;
	γ = 1.8, μ = 0.5, λ1 = 0.25, α = 0.3, β = 0.1,
	β_seq = n -> 0.5, α_seq = n -> 0.3, a_seq = n -> 0.0,
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
	messages = ["Solving problem ($name)"]
	while n <= maxiter
		# Get current parameters
		β_n = β_seq(n)
		message = "Iteration $(n) \n"
		message *= "\t lambda = $(λ_curr)"
		# ι_n = ι_seq(n)
		# x_prev_x_curr_norm = norm(x_prev - x_curr)
		θ_n = 0.9
		# ? 0.9 : if x_prev_x_curr_norm < eps()
		# 	α
		# else
		# 	min(ι_n / x_prev_x_curr_norm, α)
		# end

		a_n = a_seq(n)
		α_n = α_seq(n)

		# Step 1: Compute zₙ, wₙ, and yₙ
		z_n = x_curr + β * (x_curr - x_prev)
		w_n = x_curr + θ_n * (x_curr - x_prev)

		# Compute yₙ = (I + λₙA)^(-1)(I - λₙB)(wₙ)
		B_wn = B(w_n)
		y_n = J_A(w_n - λ_curr * B_wn, λ_curr)

		# Check stopping criterion: if yₙ = wₙ

		if stopping_criterion(y_n - w_n, tol)
			converged = true
			x_curr = y_n  # yₙ is a solution
			message *= "\n"
			message *= "="^50
			push!(messages, message)
			break
		end

		# Step 2: Compute uₙ = wₙ - γηₙdₙ
		B_yn = B(y_n)
		d_n = w_n - y_n - λ_curr * (B_wn - B_yn)

		# Compute ηₙ
		η_n = if norm(d_n) > eps()
			dot(w_n - y_n, d_n) / (norm(d_n)^2)
		else
			0.0
		end

		u_n = w_n - γ * η_n * d_n

		# Step 3: Compute xₙ₊₁
		x_next = (1 - α_n - β_n) * z_n + α_n * u_n

		# if stopping_criterion(x_next)
		# 	converged = true
		# 	x_curr = x_next
		# 	break
		# end

		# Update λₙ₊₁
		B_diff_norm = norm(B_wn - B_yn)
		w_y_norm = norm(w_n - y_n)

		λ_next = if B_diff_norm > eps()
			min(μ * w_y_norm / B_diff_norm, λ_curr + a_n)
		else
			λ_curr + a_n
		end
		message *= "\n"
		message *= "="^50
		push!(messages, message)

		# Prepare for next iteration
		x_prev = x_curr
		x_curr = x_next
		λ_curr = λ_next
		n += 1
	end
	push!(messages, "\n FINISHED $(name) \n")
	solution = Solution{typeof(x_curr)}(
		solution = x_curr,
		iterations = n - 1,
		converged = converged,
		parameters = Dict(
			:γ => γ, :μ => μ, :λ1 => λ1, :α => α, :β => β,
			:β_seq => β_seq, :α_seq => α_seq, :a_seq => a_seq,
			:tol => 1e-6, :maxiter => 10000,
		),
		messages = messages,
	)
	return solution
end



function get_IPCMAS2_params(L::Float64)
	# 	# Parameters from the paper for Example 1
	γ = 1.8
	μ = 0.5
	λ1 = 1.0 / (2 * L)
	α = 0.5 # ∈ (0, 1/3)
	θ = 0.5
	β_seq(n) = 1.0 / (5 * n + 1)
	a_seq(n) = 0.8 - β_seq(n)

	return (
		γ = γ,
		μ = μ,
		λ1 = λ1,
		α = α,
		θ = θ,
		a_seq = a_seq,
	)

end

function IPCMAS2(A, B, x0, x1;
	γ = 1.8, μ = 0.5, λ1 = 0.5, α = 0.3, θ = 0.5,
	a_seq = n -> 1 / n^2, tol = 1e-6, maxiter = 1000)

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
	J_A(x, λ) = A(x, λ)  # This should be the resolvent of A

	while n <= maxiter
		# Step 1: Compute w_n and y_n
		w_n = x_curr + θ * (x_curr - x_prev)

		# Compute y_n = J^A_{λ_n}(w_n - λ_n*B(w_n))
		Bw_n = B(w_n)
		y_n = J_A(w_n - λ_curr * Bw_n, λ_curr)

		# Check stopping criterion
		if norm(y_n - w_n) < tol
			converged = true
			break
		end

		# Step 2: Compute u_n
		By_n = B(y_n)
		d_n = w_n - y_n - λ_curr * (Bw_n - By_n)

		# Compute η_n
		η_n = if norm(d_n) > eps()
			dot(w_n - y_n, d_n) / (norm(d_n)^2)
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

	return x_curr, n - 1, converged
end


function get_DeyHICPP_params(L::Float64)
	λ_constant = 1.0 / (2 * L)
	β_seq(n) = 1.0 / (5 * n + 1)
	return (
		γ = 1.5,
		λ_seq = n -> λ_constant,
		α = 0.5,
		τ_seq = n -> 1.0 / n^2,
		β_seq = β_seq,
		θ_seq = n -> 0.8 - β_seq(n),
	)
end


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
	messages = ["Solving problem ($name)" * "=="^10]
	while n <= maxiter
		# Get current parameters
		message = "iteration: $n \n"
		λₙ = λ_seq(n)
		message *= "lambda = $(λₙ)"
		τₙ = τ_seq(n)
		βₙ = β_seq(n)
		θₙ = θ_seq(n)

		# Step 1: Choose αₙ such that 0 ≤ αₙ ≤ ᾱₙ
		x_diff_norm = _norm(x_curr - x_prev)
		ᾱₙ = if x_diff_norm > eps()
			min(α, τₙ / x_diff_norm)
		else
			α
		end
		αₙ = ᾱₙ  # We choose αₙ = ᾱₙ for best performance
		message *= "\t alpha_n = $(αₙ)"

		# Step 2: Compute wₙ and yₙ
		wₙ = x_curr + αₙ * (x_curr - x_prev)

		# Compute yₙ = J^A_{λₙ}(wₙ - λₙf(wₙ))
		B_wₙ = B(wₙ)
		yₙ = Aresolvant(wₙ - λₙ * B_wₙ, λₙ)

		# Check stopping criterion: if yₙ = wₙ
		if stopping_criterion(yₙ - wₙ, tol)
			converged = true
			x_curr = yₙ  # yₙ is a solution
			message *= " first check (true)\n"
			message *= "\n"
			message *= "="^50
			push!(messages, message)
			break
		end

		# Step 3: Calculate zₙ = wₙ - γηₙdₙ
		B_yₙ = B(yₙ)
		dₙ = wₙ - yₙ - λₙ * (B_wₙ - B_yₙ)

		# Compute ηₙ
		ηₙ = if _norm(dₙ) > eps()
			_dot(wₙ - yₙ, dₙ) / (_norm(dₙ)^2)
		else
			0.0
		end
		message *= "\t alpha_n = $(ηₙ)"

		zₙ = wₙ - γ * ηₙ * dₙ

		# Step 4: Calculate xₙ₊₁
		x_next = (1 - θₙ - βₙ) * x_curr + θₙ * zₙ

		# Prepare for next iteration
		x_prev = x_curr
		x_curr = x_next
		message *= "\n"
		message *= "="^50
		push!(messages, message)
		n += 1
	end
	push!(messages, "\n FINISHED SOLVING $(name)")
	solution = Solution{typeof(x_curr)}(
		solution = x_curr,
		iterations = n - 1,
		converged = converged,
		parameters = Dict(
			:γ => γ, :λ_seq => λ_seq, :α => α,
			:τ_seq => τ_seq, :β_seq => β_seq, :θ_seq => θ_seq,
			:tol => 1e-6, :maxiter => 10000,
		),
		messages = messages,
	)
	return solution
end
