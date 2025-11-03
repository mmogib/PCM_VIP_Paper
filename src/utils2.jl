
# Create results directory at project root (one level up from src)
results_dir = joinpath(@__DIR__, "..", "results")
if !isdir(results_dir)
	mkpath(results_dir)
	println("Created results directory: $results_dir")
end


function createFolder(folder::String; verbose::Bool = false)
	mkpath(folder)
	verbose && println("Created results directory: $folder")
	return folder
end


function getFilename(file::String; verbose::Bool = false, stamped::Bool = true)
	base = basename(file)
	base_parts = split(base, ".")
	first_part = join(base_parts[1:end-1], ".")
	ext = base_parts[end]
	dir = createFolder(dirname(file))
	filename = if stamped
		timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
		"$(dir)/$(first_part)_$(timestamp).$ext"
	else
		file
	end
	verbose && println("File Name: $filename \n")
	return filename

end

function getFilename(folder::String, file::String; kwarges...)
	folder = createFolder(folder)
	file = "$folder/$file"
	return getFilename(file; kwarges...)
end


function is_running_as_script(file::AbstractString = @__FILE__)
	pf = PROGRAM_FILE               # entry filename; "" at REPL
	return !isempty(pf) && abspath(pf) == abspath(file)
end

# ---------------------------
# Evaluation helper utilities
# ---------------------------

# Latest model path from results/models, filtered by example tag (e.g., "example1" or "example 1")
function latest_model_path(tag::AbstractString; models_dir::AbstractString = joinpath(@__DIR__, "..", "results", "models"))
	isdir(models_dir) || error("Model directory not found: $(models_dir)")
	files = filter(f -> endswith(lowercase(f), ".jld2"), readdir(models_dir; join = true))
	isempty(files) && error("No .jld2 models found in $(models_dir)")

	m = match(r"^\s*example\s*(\d+)\s*$"i, tag)
	prefix = nothing
	if m !== nothing
		exnum = m.captures[1]
		prefix = "example $(exnum)_"
	else
		prefix = string(tag)
	end

	function has_prefix(path)
		nm = basename(path)
		return startswith(nm, prefix)
	end
	candidates = filter(has_prefix, files)
	isempty(candidates) && error("No model files matching prefix '$(prefix)' in $(models_dir)")
	sort(candidates; by = f -> stat(f).mtime, rev = true)[1]
end

# Uniform time grid
time_grid(tspan::Tuple{<:Real, <:Real}, n::Int) = collect(range(first(tspan), last(tspan), length = n))

# Resolve projection and sampler fields from VIProblem (supports Ω/Ic naming)
function resolve_project_and_sampler(prob)
	project = nothing
	sampler = nothing
	for fn in (:project_Ω, :project_Omega, :project_Ic)
		if hasproperty(prob, fn)
			project = getfield(prob, fn)
			break
		end
	end
	for fn in (:sample_Ω, :sample_Omega, :sample_Ic)
		if hasproperty(prob, fn)
			sampler = getfield(prob, fn)
			break
		end
	end
	project === nothing && error("Could not resolve projection function from VIProblem")
	sampler === nothing && error("Could not resolve sampler function from VIProblem")
	return project, sampler
end

# ODE trajectory on a shared grid (requires vi_solve/ODESolver in scope by the caller)
function ode_trajectory(F, project, x0::AbstractVector{<:Real}, tspan;
	reltol::Real = 1e-6, abstol::Real = 1e-6, tgrid::AbstractVector{<:Real})
	solver = ODESolver(x0 = collect(Float64.(x0)), tspan = (float(tspan[1]), float(tspan[2])),
		reltol = float(reltol), abstol = float(abstol))
	result = vi_solve(F, project, solver; verbose = false)
	sol = result.extra["sol"]
	xs = [sol(t) for t in tgrid]
	reduce(hcat, xs)
end

# PINN trajectory (n×T)
pinn_trajectory(model, x0::AbstractVector{<:Real}, tgrid) = predict_trajectory(model, x0, tgrid)

# Trajectory error norms and per-time errors
function trajectory_errors(xs_ode::AbstractMatrix, xs_pinn::AbstractMatrix, tgrid::AbstractVector)
	@assert size(xs_ode) == size(xs_pinn)
	nT = size(xs_ode, 2)
	errs = Vector{Float64}(undef, nT)
	@views for i in 1:nT
		a = xs_ode[:, i] .- xs_pinn[:, i]
		errs[i] = norm(a)
	end
	linf = maximum(errs)
	l2 = 0.0
	@inbounds for i in 1:(nT-1)
		dt = tgrid[i+1] - tgrid[i]
		l2 += 0.5 * (errs[i] + errs[i+1]) * dt
	end
	return l2, linf, errs
end

# Build mixed initial sets (inside Ω and outside Ω)
function build_initial_sets(project::Function, sample::Function; n_inside::Int = 100, n_outside::Int = 100, eps_out::Real = 1e-4, seed::Int = 2025)
	Random.seed!(seed)
	rng = Xoshiro(seed)
	inside = [sample() for _ in 1:n_inside]
	outside = Vector{Vector{Float64}}()
	while length(outside) < n_outside
		x = sample()
		xpert = collect(Float64.(x))
		j = rand(rng, 1:length(xpert))
		s = rand(rng, (-1, 1))
		mag = 0.2 + rand(rng)  # in [0.2, 1.2)
		xpert[j] = xpert[j] + s * mag
		if norm(xpert - project(xpert)) > eps_out
			push!(outside, xpert)
		end
	end
	return inside, outside
end

# Minimal JSON writer using plain quotes and simple values
function write_json(path::AbstractString, meta::Dict)
	open(path, "w") do io
		write(io, "{\n")
		keys_list = collect(keys(meta))
		for (idx, k) in enumerate(keys_list)
			v = meta[k]
			print(io, "  \"$(k)\": ")
			if v isa AbstractString
				print(io, '"', v, '"')
			elseif v isa Bool
				print(io, v ? "true" : "false")
			elseif v isa Number
				print(io, string(v))
			elseif v isa Dict
				inner = join(["\"$(kk)\": " * (vv isa AbstractString ? '"' * string(vv) * '"' : string(vv)) for (kk, vv) in v], ", ")
				print(io, "{", inner, "}")
			else
				print(io, '"', string(v), '"')
			end
			if idx < length(keys_list)
				write(io, ",\n")
			else
				write(io, "\n")
			end
		end
		write(io, "}")
	end
end
