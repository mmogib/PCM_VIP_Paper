# ============================================================================
# PROJECTION OPERATORS
# ============================================================================


# No projection needed
function proj_unconstrained(x)
	return x
end
function project_box(x, l, u)
	return clamp.(x, l, u)
end
# Example: proj_box([3.0, -2.0], 0.0, 1.0)  # returns [1.0, 0.0]
function proj_l2ball(x, r)
	normx = norm(x, 2)
	if normx <= r
		return x
	else
		return x * (r / normx)
	end
end
# Example: proj_l2ball([3.0, 4.0], 5.0)     # returns [3.0, 4.0]
#          proj_l2ball([3.0, 4.0], 4.0)     # returns [2.4, 3.2]
# Projects each vector x onto the probability simplex
function proj_simplex(x)
	d = length(x)
	u = sort(x, rev = true)
	cssv = cumsum(u) .- 1
	ind = findall(u .> (cssv ./ (1:d)))
	if isempty(ind)
		theta = 0.0
	else
		rho = maximum(ind)
		theta = cssv[rho] / rho
	end
	return max.(x .- theta, 0)
end
# Example: proj_simplex([.2, .8, -.5])  # returns a vector summing to 1 with non-negatives

function proj_affine(x, A, b)
	# Project x onto {x | A x = b}
	At = A'
	# P = I - A' * (A * A')^{-1} * A
	P = I - At * inv(A * At) * A
	return P * x + At * inv(A * At) * b
end
# Use with caution: efficient for small A only


function proj_nonnegative(x)
	T = eltype(x)
	return max.(x, T(0))
end




"""
	project_sphere(x, center, radius)

Project x onto sphere constraint Ω = {x ∈ Rⁿ | ‖x - center‖ ≤ radius}
"""
# function project_sphere(x, center, radius)
# 	diff = x - center
# 	dist = norm(diff)

# 	if dist <= radius
# 		return x  # Already inside sphere
# 	else
# 		return center + radius * diff / dist  # Project onto sphere surface
# 	end
# end

function project_sphere(x̂, center, radius)
	T = eltype(x̂)
	Y = similar(x̂)

	x1 = @view x̂[1, :]
	x2 = @view x̂[2, :]

	y1 = @view Y[1, :]
	y2 = @view Y[2, :]

	c1 = T(center[1])
	c2 = T(center[2])
	r = T(radius)

	dx1 = x1 .- c1
	dx2 = x2 .- c2
	dist = sqrt.(dx1 .^ 2 .+ dx2 .^ 2)                         # 1×B
	scale = min.(one(T), r ./ max.(dist, eps(T)))          # 1×B

	@. y1 = c1 + dx1 * scale
	@. y2 = c2 + dx2 * scale

	return Y
end



# projection of x onto H_{u,v} = { z : dot(u,z) ≤ v }

function proj_halfspace(x, u, v)
	T = eltype(x)
	u = T.(u)
	v = T(v)
	# dot(u,x) is GPU compatible in CUDA.jl and Metal.jl
	α = max((dot(u, x) - v) / (dot(u, u)), 0)
	return @. x - α * u
end



# φ(μ) = u' * P_Box[a,b]( x - μ u ) - v
function φ(μ, x, u, a, b, v)
	z = project_box(x .- μ .* u, a, b)
	return dot(u, z) - v
end

"""
projection onto { z : u' z = v } ∩ [a,b]
"""
function proj_hyperplane_box(x, u, a, b, v;
	μ_lo = -1e6, μ_hi = 1e6, tol = 1e-8, maxiter = 100)

	# bisection on μ
	f_lo = φ(μ_lo, x, u, a, b, v)
	f_hi = φ(μ_hi, x, u, a, b, v)
	# assume sign change
	for k in 1:maxiter
		μ_mid = (μ_lo + μ_hi) / 2
		f_mid = φ(μ_mid, x, u, a, b, v)
		if abs(f_mid) < tol
			μ = μ_mid
			break
		end
		if f_lo * f_mid < 0
			μ_hi = μ_mid
			f_hi = f_mid
		else
			μ_lo = μ_mid
			f_lo = f_mid
		end
		μ = μ_mid
	end

	# final projection
	return proj_box(x .- μ .* u, a, b)
end


# project x to ball centered at p, radius q
function proj_ball(x, p, q)
	d = x .- p
	nd = norm(d)
	s = q / max(nd, q)
	return p .+ s .* d
end

# column-wise projection
function proj_ball_batch(X, p, q)
	D  = X .- p        # n×B
	nd = sqrt.(sum(abs2, D; dims = 1))  # 1×B norms
	s  = q ./ max.(nd, q)
	return p .+ D .* s
end
