
"""
	csv_to_xlsx(csv_file::String, xlsx_file::String=""; 
				sheet_name::String="Sheet1")

Convert a CSV file to XLSX format.

# Arguments
- `csv_file`: Path to input CSV file
- `xlsx_file`: Path to output XLSX file (defaults to same name with .xlsx extension)
- `sheet_name`: Name of the sheet in Excel file (default: "Sheet1")

# Example
```julia
csv_to_xlsx("comparison_n100.csv")
csv_to_xlsx("comparison_n100.csv", "results.xlsx")
csv_to_xlsx("comparison_n100.csv", "results.xlsx", sheet_name="n=100")
```
"""
function csv_to_xlsx(csv_file::String, xlsx_file::String = "";
	sheet_name::String = "Sheet1", overwrite = false)

	# Check if CSV file exists
	if !isfile(csv_file)
		error("CSV file not found: $csv_file")
	end

	# Default output filename
	if isempty(xlsx_file)
		xlsx_file = replace(csv_file, ".csv" => ".xlsx")
	end

	# Read CSV file
	df = CSV.read(csv_file, DataFrame)

	# Write to XLSX
	XLSX.writetable(xlsx_file, sheet_name => df, overwrite = overwrite)

	println("Successfully converted $csv_file to $xlsx_file")
	println("Sheet name: $sheet_name")

	return xlsx_file
end

"""
	csv_to_xlsx_multiple(csv_files::Vector{String}, xlsx_file::String)

Convert multiple CSV files to a single XLSX file with multiple sheets.

# Arguments
- `csv_files`: Vector of CSV file paths
- `xlsx_file`: Path to output XLSX file

# Example
```julia
csv_to_xlsx_multiple(
	["comparison_n100.csv", "comparison_n200.csv"],
	"all_results.xlsx"
)
```
"""
function csv_to_xlsx_multiple(csv_files::Vector{String}, xlsx_file::String; sheet_names::Vector{String} = [], overwrite = false)

	# Check if all CSV files exist
	for csv_file in csv_files
		if !isfile(csv_file)
			error("CSV file not found: $csv_file")
		end
	end

	# Read all CSV files
	gotSheetNames = length(sheet_names) > 0
	sheets = Dict{String, DataFrame}()
	for (i, csv_file) in enumerate(csv_files)
		# Get sheet name from filename (without .csv extension)
		sheet_name = gotSheetNames ? sheet_names[i] : replace(basename(csv_file), ".csv" => "")

		# Read CSV
		df = CSV.read(csv_file, DataFrame)

		push!(sheets, sheet_name => df)
	end

	# Write to XLSX with multiple sheets
	XLSX.writetable(xlsx_file, sheets..., overwrite = overwrite)

	println("Successfully created $xlsx_file with $(length(csv_files)) sheets:")
	for (sheet_name, _) in sheets
		println("  - $sheet_name")
	end

	return xlsx_file
end



# function performance_profile_from_csv(path::AbstractString;
# 	tag::String = "Time",
# 	solvers::Union{Nothing, Vector{String}} = nothing,
# 	treat_nonconverged_as_inf::Bool = true,
# 	savepath::Union{Nothing, AbstractString} = nothing,
# 	return_data::Bool = false,
# )
# 	df = CSV.read(path, DataFrame)

# 	# Infer solver names from "<SOLVER>_<tag>" columns if not provided
# 	if solvers === nothing
# 		suffix = "_$(tag)"
# 		solvers = [String(replace(String(n), suffix => ""))
# 				   for n in names(df) if endswith(String(n), suffix)]
# 		isempty(solvers) && error("No columns found ending with '$suffix'. Check 'tag' or headers.")
# 	end

# 	# Build performance dictionary
# 	perf = Dict{String, Vector{Float64}}()
# 	for s in solvers
# 		metric_col = Symbol("$(s)_$(tag)")
# 		metric_col in unique(propertynames(df)) ||
# 			error("Missing column $(metric_col) for solver '$s'.")

# 		v = Float64.(df[!, metric_col])

# 		# Handle convergence if the column exists
# 		conv_col = Symbol("$(s)_Converged")
# 		if treat_nonconverged_as_inf && (conv_col in propertynames(df))
# 			conv = df[!, conv_col]
# 			# Replace non-converged with Inf
# 			@inbounds for i in eachindex(v)
# 				if !(conv[i] === true)
# 					v[i] = Inf
# 				end
# 			end
# 		end

# 		perf[s] = v
# 	end
# 	solvers = collect(keys(perf))
# 	data = hcat(values(perf)...)
# 	# Plot performance profile
# 	plt = performance_profile(PlotsBackend(), data, solvers;
# 		xlabel = "τ",
# 		ylabel = "Proportion of problems",
# 		title = "Performance Profile — $(tag)",
# 		legend = :bottomright,
# 	)
# 	display(plt)

# 	if savepath !== nothing
# 		savefig(plt, savepath)
# 	end

# 	return return_data ? (plt, perf) : plt
# end

function performance_profile_from_csv(
	path::AbstractString;
	tag::String = "Time",
	solvers::Union{Nothing, Vector{String}} = nothing,
	treat_nonconverged_as_inf::Bool = true,
	savepath::Union{Nothing, AbstractString} = nothing,
	return_data::Bool = false,
)
	df = CSV.read(path, DataFrame)
	if solvers === nothing
		solvers = unique(df.Algorithm)
	end

	perf = Dict{String, Vector{Float64}}()
	for s in solvers
		subdf = df[df.Algorithm.==s, :]
		v = Float64.(subdf[!, tag])
		if treat_nonconverged_as_inf && :Converged in propertynames(subdf)
			conv = subdf[!, :Converged]
			@inbounds for i in eachindex(v)
				if !(conv[i] === true)
					v[i] = NaN
				end
			end
		end
		perf[s] = v
	end

	solvers = collect(keys(perf))
	data = hcat(values(perf)...)

	plt = performance_profile(PlotsBackend(), data, solvers;
		xlabel = "τ",
		ylabel = "Proportion of problems",
		title = "Performance Profile — $(tag)",
		legend = :bottomright,
	)

	display(plt)
	if savepath !== nothing
		savefig(plt, savepath)
	end

	return return_data ? (plt, perf) : plt
end


function prepare_filepath(path::AbstractString; dated::Bool = false)::String
	dir = dirname(path)
	if !isempty(dir) && !isdir(dir)
		mkpath(dir)
	end
	if !dated
		return String(path)
	end
	name = basename(path)
	stem, ext = splitext(name)
	timestamp = Dates.format(Dates.now(), "yyyymmdd_HH_MM_SS")
	new_name = string(stem, "_", timestamp, ext)
	return isempty(dir) ? new_name : joinpath(dir, new_name)
end


function newest_csv_by_timestamp(dir::AbstractString, prefix::AbstractString; throw_on_empty::Bool = true)
	# Regex: ^prefix + yyyymmdd_HH_MM_SS.csv$
	rx = Regex("^" * prefix * "(\\d{8})_(\\d{2})_(\\d{2})_(\\d{2})\\.csv\$")

	candidates = Tuple{DateTime, String}[]
	for f in readdir(dir; join = true)
		name = basename(f)
		m = match(rx, name)
		m === nothing && continue
		ymd, HH, MM, SS = m.captures
		# Build DateTime from captured parts
		d = Date(ymd, dateformat"yyyymmdd")
		t = Time(parse(Int, HH), parse(Int, MM), parse(Int, SS))
		push!(candidates, (DateTime(d, t), f))
	end

	if isempty(candidates)
		if throw_on_empty
			error("No files matching pattern $(prefix)YYYYMMDD_HH_MM_SS.csv found in: $dir")
		else
			return nothing
		end
	end
	# Pick the file with the maximum timestamp; tie-break by filename for determinism
	path = last(candidates[end])
	return path
end


function find_newest_csv(dir::String, prefix::String)
	files = filter(f -> occursin("$prefix" * r"_\d{8}_\d{2}_\d{2}_\d{2}\.csv", f), readdir(dir))
	isempty(files) ? "" : joinpath(dir, maximum(files))
end



function clear_folder_recursive(path::AbstractString; clearSubfolders = false)
	for f in readdir(path; join = true)
		clearSubfolders ? rm(f; force = true, recursive = true) : (isfile(f) && rm(f; force = true))
	end
	return nothing
end
