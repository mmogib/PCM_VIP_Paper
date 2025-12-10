struct AlgorithmIPCMAS1 end
struct AlgorithmIPCMAS2 end
struct AlgorithmDey end

@with_kw struct Problem
    name::String
    Aλ::Function          # Resolvent operator
    A::Function          # Resolvent operator
    B::Function          # Single-valued operator
    L::Float64          # Lipschitz constant
    x0::Vector{Float64} # Initial point 1
    x1::Vector{Float64} # Initial point 2
    n::Int              # Dimension
    dot::Function = dot
    norm::Function = norm
    stopping::Function = (x, tol) -> begin
        normx = norm(x)
        (normx < tol, normx)
    end
end

@with_kw struct Solution{T}
    solver::String = ""
    problem::Problem
    solution::T
    iterations::Int
    time::Float64 = 0.0
    converged::Bool = false
    parameters::Dict{Symbol,Any}
    history::Dict{Symbol,Vector{<:Real}}
end

# Solution(sol::Solution; time::T where {T <: Real}) = Solution(
# 	solution = sol.solution, iterations = sol.iterations, time = time, converged = sol.converged,
# 	parameters = sol.parameters,
# )


# struct L2Problem
# 	name::String
# 	Aλ::Function          # Resolvent operator
# 	A::Function          # Resolvent operator
# 	B::Function          # Single-valued operator
# 	L::Float64          # Lipschitz constant
# 	x0::Function # Initial point 1
# 	x1::Function # Initial point 2
# 	n::Int              # Dimension
# end
