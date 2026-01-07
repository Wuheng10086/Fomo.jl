# ==============================================================================
# io/model_loader.jl
#
# Unified model loader - handles multiple file formats
# Converts everything to a standard internal representation
# ==============================================================================

# JLD2 is already imported at module level

# Optional packages - try to load at module init
const _HAS_SEGYIO = Ref(false)
const _HAS_MAT = Ref(false)  
const _HAS_NPZ = Ref(false)
const _HAS_HDF5 = Ref(false)

# These will be set during module initialization
# Use Base.invokelatest when calling functions from dynamically loaded modules

# ==============================================================================
# Standard Model Structure
# ==============================================================================

"""
    VelocityModel

Standard internal representation for velocity models.
All loaders convert to this format.
"""
struct VelocityModel
    vp::Matrix{Float32}     # P-wave velocity
    vs::Matrix{Float32}     # S-wave velocity  
    rho::Matrix{Float32}    # Density
    dx::Float32             # Grid spacing in X
    dz::Float32             # Grid spacing in Z
    nx::Int                 # Grid points in X
    nz::Int                 # Grid points in Z
    x_origin::Float32       # X origin (default 0)
    z_origin::Float32       # Z origin (default 0)
    name::String            # Model name (optional)
end

# Constructor with auto-computed dimensions
# Note: Seismic convention - data is stored as vp[nz, nx] (depth first)
function VelocityModel(vp, vs, rho, dx, dz; 
                       x_origin=0.0f0, z_origin=0.0f0, name="unnamed")
    nz, nx = size(vp)  # Seismic convention: depth is first dimension
    @assert size(vs) == (nz, nx) "vs size mismatch"
    @assert size(rho) == (nz, nx) "rho size mismatch"
    VelocityModel(
        Float32.(vp), Float32.(vs), Float32.(rho),
        Float32(dx), Float32(dz), nx, nz,
        Float32(x_origin), Float32(z_origin), name
    )
end

# ==============================================================================
# Unified Load Function
# ==============================================================================

"""
    load_model(path; kwargs...) -> VelocityModel

Smart model loader - automatically detects format from extension.

# Supported Formats
- `.jld2` - Julia native (recommended, fastest)
- `.segy`, `.sgy` - SEG-Y format (requires SegyIO)
- `.bin` - Raw binary (requires nx, nz, dtype kwargs)
- `.mat` - MATLAB format (requires MAT.jl)
- `.npy` - NumPy format (requires NPZ.jl)
- `.h5`, `.hdf5` - HDF5 format

# Examples
```julia
# JLD2 (recommended)
model = load_model("marmousi.jld2")

# Binary with metadata
model = load_model("vp.bin"; nx=500, nz=200, dx=10.0, dz=10.0,
                   vs="vs.bin", rho="rho.bin")

# SEG-Y
model = load_model("model.segy"; dx=12.5, dz=12.5)

# Single field (returns just that array, not VelocityModel)
vp = load_model("vp.bin"; nx=500, nz=200, dtype=Float32, single=true)
```
"""
function load_model(path::String; kwargs...)
    ext = lowercase(splitext(path)[2])
    
    if ext == ".jld2"
        return _load_jld2(path; kwargs...)
    elseif ext in [".segy", ".sgy"]
        return _load_segy(path; kwargs...)
    elseif ext == ".bin"
        return _load_binary(path; kwargs...)
    elseif ext == ".mat"
        return _load_mat(path; kwargs...)
    elseif ext == ".npy"
        return _load_npy(path; kwargs...)
    elseif ext in [".h5", ".hdf5"]
        return _load_hdf5(path; kwargs...)
    else
        error("Unsupported format: $ext\nSupported: .jld2, .segy, .sgy, .bin, .mat, .npy, .h5, .hdf5")
    end
end

# ==============================================================================
# JLD2 Loader (Recommended Internal Format)
# ==============================================================================

function _load_jld2(path; kwargs...)
    data = JLD2.load(path)
    
    # Try standard field names
    vp = get(data, "vp", get(data, "Vp", get(data, "VP", nothing)))
    vs = get(data, "vs", get(data, "Vs", get(data, "VS", nothing)))
    rho = get(data, "rho", get(data, "Rho", get(data, "density", nothing)))
    dx = get(data, "dx", get(data, "dh", get(data, "spacing", 1.0f0)))
    dz = get(data, "dz", get(data, "dh", dx))
    name = get(data, "name", basename(path))
    
    if vp === nothing
        error("Cannot find vp/Vp/VP field in JLD2 file")
    end
    
    # If only vp exists, estimate vs and rho
    if vs === nothing
        @warn "vs not found, estimating vs = vp / 1.73"
        vs = vp ./ 1.73f0
    end
    if rho === nothing
        @warn "rho not found, using Gardner relation: rho = 310 * vp^0.25"
        rho = 310.0f0 .* (vp .^ 0.25f0)
    end
    
    return VelocityModel(vp, vs, rho, dx, dz; name=name)
end

# ==============================================================================
# Binary Loader
# ==============================================================================

function _load_binary(path; nx=nothing, nz=nothing, dx=1.0, dz=nothing,
                      dtype=Float32, order=:column_major,
                      vs=nothing, rho=nothing, single=false, kwargs...)
    
    if nx === nothing || nz === nothing
        error("Binary files require nx and nz parameters")
    end
    
    dz = dz === nothing ? dx : dz
    
    # Load single array
    function read_bin(file, nx, nz, dtype)
        data = zeros(dtype, nx * nz)
        open(file, "r") do io
            read!(io, data)
        end
        if order == :row_major
            return permutedims(reshape(data, nz, nx))
        else
            return reshape(data, nx, nz)
        end
    end
    
    vp = read_bin(path, nx, nz, dtype)
    
    if single
        return vp
    end
    
    # Load vs and rho if provided
    vs_data = if vs isa String
        read_bin(vs, nx, nz, dtype)
    elseif vs === nothing
        @warn "vs not provided, estimating vs = vp / 1.73"
        vp ./ 1.73f0
    else
        vs
    end
    
    rho_data = if rho isa String
        read_bin(rho, nx, nz, dtype)
    elseif rho === nothing
        @warn "rho not provided, using Gardner relation"
        310.0f0 .* (vp .^ 0.25f0)
    else
        rho
    end
    
    return VelocityModel(vp, vs_data, rho_data, dx, dz; name=basename(path))
end

# ==============================================================================
# SEG-Y Loader
# ==============================================================================

function _load_segy(path; dx=nothing, dz=nothing, 
                    vs=nothing, rho=nothing, kwargs...)
    
    # Check if SegyIO is available
    if !isdefined(@__MODULE__, :SegyIO) && !isdefined(Main, :SegyIO)
        error("""
        SegyIO.jl is required for SEG-Y files but not loaded.
        Please add to your script before calling load_model:
        
            using SegyIO
        
        Or install it: using Pkg; Pkg.add("SegyIO")
        """)
    end
    
    # Get the SegyIO module
    SegyIO_mod = isdefined(@__MODULE__, :SegyIO) ? (@__MODULE__).SegyIO : Main.SegyIO
    
    @info "Loading SEG-Y file: $path"
    block = Base.invokelatest(SegyIO_mod.segy_read, path)
    vp = Float32.(block.data)
    
    # Try to get spacing from header
    if dx === nothing
        dx = 1.0f0
        @warn "dx not specified, using dx=1.0"
    end
    dz = dz === nothing ? dx : dz
    
    # Handle vs and rho
    vs_data = if vs isa String && (endswith(vs, ".segy") || endswith(vs, ".sgy"))
        Float32.(Base.invokelatest(SegyIO_mod.segy_read, vs).data)
    elseif vs === nothing
        @warn "vs not provided, estimating vs = vp / 1.73"
        vp ./ 1.73f0
    else
        vs
    end
    
    rho_data = if rho isa String && (endswith(rho, ".segy") || endswith(rho, ".sgy"))
        Float32.(Base.invokelatest(SegyIO_mod.segy_read, rho).data)
    elseif rho === nothing
        @warn "rho not provided, using Gardner relation"
        310.0f0 .* (vp .^ 0.25f0)
    else
        rho
    end
    
    return VelocityModel(vp, vs_data, rho_data, dx, dz; name=basename(path))
end

# ==============================================================================
# MAT Loader
# ==============================================================================

function _load_mat(path; vp_key="vp", vs_key="vs", rho_key="rho",
                   dx=1.0, dz=nothing, kwargs...)
    
    if !isdefined(@__MODULE__, :MAT) && !isdefined(Main, :MAT)
        error("""
        MAT.jl is required for .mat files but not loaded.
        Please add to your script: using MAT
        Or install it: using Pkg; Pkg.add("MAT")
        """)
    end
    
    MAT_mod = isdefined(@__MODULE__, :MAT) ? (@__MODULE__).MAT : Main.MAT
    data = Base.invokelatest(MAT_mod.matread, path)
    
    vp = Float32.(get(data, vp_key, get(data, "Vp", get(data, "VP", nothing))))
    if vp === nothing
        error("Cannot find vp field. Available keys: $(keys(data))")
    end
    
    vs = get(data, vs_key, get(data, "Vs", nothing))
    rho = get(data, rho_key, get(data, "Rho", get(data, "density", nothing)))
    
    dz = dz === nothing ? dx : dz
    
    vs_data = vs === nothing ? vp ./ 1.73f0 : Float32.(vs)
    rho_data = rho === nothing ? 310.0f0 .* (vp .^ 0.25f0) : Float32.(rho)
    
    return VelocityModel(vp, vs_data, rho_data, dx, dz; name=basename(path))
end

# ==============================================================================
# NumPy Loader
# ==============================================================================

function _load_npy(path; dx=1.0, dz=nothing, vs=nothing, rho=nothing, kwargs...)
    
    if !isdefined(@__MODULE__, :NPZ) && !isdefined(Main, :NPZ)
        error("""
        NPZ.jl is required for .npy files but not loaded.
        Please add to your script: using NPZ
        Or install it: using Pkg; Pkg.add("NPZ")
        """)
    end
    
    NPZ_mod = isdefined(@__MODULE__, :NPZ) ? (@__MODULE__).NPZ : Main.NPZ
    
    vp = Float32.(Base.invokelatest(NPZ_mod.npzread, path))
    dz = dz === nothing ? dx : dz
    
    vs_data = if vs isa String
        Float32.(Base.invokelatest(NPZ_mod.npzread, vs))
    elseif vs === nothing
        vp ./ 1.73f0
    else
        Float32.(vs)
    end
    
    rho_data = if rho isa String
        Float32.(Base.invokelatest(NPZ_mod.npzread, rho))
    elseif rho === nothing
        310.0f0 .* (vp .^ 0.25f0)
    else
        Float32.(rho)
    end
    
    return VelocityModel(vp, vs_data, rho_data, dx, dz; name=basename(path))
end

# ==============================================================================
# HDF5 Loader
# ==============================================================================

function _load_hdf5(path; vp_key="vp", vs_key="vs", rho_key="rho",
                    dx=1.0, dz=nothing, kwargs...)
    
    if !isdefined(@__MODULE__, :HDF5) && !isdefined(Main, :HDF5)
        error("""
        HDF5.jl is required for .h5/.hdf5 files but not loaded.
        Please add to your script: using HDF5
        Or install it: using Pkg; Pkg.add("HDF5")
        """)
    end
    
    HDF5_mod = isdefined(@__MODULE__, :HDF5) ? (@__MODULE__).HDF5 : Main.HDF5
    
    Base.invokelatest(HDF5_mod.h5open, path, "r") do file
        vp = Float32.(read(file, vp_key))
        vs = haskey(file, vs_key) ? Float32.(read(file, vs_key)) : vp ./ 1.73f0
        rho = haskey(file, rho_key) ? Float32.(read(file, rho_key)) : 310.0f0 .* (vp .^ 0.25f0)
        
        dz = dz === nothing ? dx : dz
        return VelocityModel(vp, vs, rho, dx, dz; name=basename(path))
    end
end

# ==============================================================================
# Save Functions
# ==============================================================================

"""
    save_model(path, model::VelocityModel)
    save_model(path, vp, vs, rho, dx, dz; kwargs...)

Save model to JLD2 format (recommended for reuse).
"""
function save_model(path::String, model::VelocityModel)
    jldsave(path;
        vp = model.vp,
        vs = model.vs,
        rho = model.rho,
        dx = model.dx,
        dz = model.dz,
        nx = model.nx,
        nz = model.nz,
        name = model.name
    )
    @info "Model saved" path=path size=(model.nx, model.nz)
end

function save_model(path::String, vp, vs, rho, dx, dz; name="model")
    model = VelocityModel(vp, vs, rho, dx, dz; name=name)
    save_model(path, model)
end

# ==============================================================================
# Conversion Script Helper
# ==============================================================================

"""
    convert_model(input, output; kwargs...)

Convert model from any supported format to JLD2.

# Example
```julia
# SEG-Y to JLD2
convert_model("model.segy", "model.jld2"; dx=12.5)

# Binary to JLD2
convert_model("vp.bin", "model.jld2"; nx=500, nz=200, dx=10.0,
              vs="vs.bin", rho="rho.bin")
```
"""
function convert_model(input::String, output::String; kwargs...)
    model = load_model(input; kwargs...)
    save_model(output, model)
    @info "Conversion complete" input=input output=output
    return model
end

# ==============================================================================
# Quick Info
# ==============================================================================

"""
    model_info(model::VelocityModel)

Print model information.
"""
function model_info(model::VelocityModel)
    println("═" ^ 50)
    println("Model: $(model.name)")
    println("═" ^ 50)
    println("  Grid:     $(model.nx) × $(model.nz)")
    println("  Spacing:  dx=$(model.dx)m, dz=$(model.dz)m")
    println("  Size:     $(model.nx * model.dx)m × $(model.nz * model.dz)m")
    println("─" ^ 50)
    println("  Vp:   $(minimum(model.vp)) - $(maximum(model.vp)) m/s")
    println("  Vs:   $(minimum(model.vs)) - $(maximum(model.vs)) m/s")
    println("  Rho:  $(minimum(model.rho)) - $(maximum(model.rho)) kg/m³")
    println("═" ^ 50)
end

model_info(path::String; kwargs...) = model_info(load_model(path; kwargs...))

# ==============================================================================
# Load from Separate Files (Common Workflow)
# ==============================================================================

"""
    load_model_files(; vp, vs=nothing, rho=nothing, dx, dz=dx, nx=nothing, nz=nothing, kwargs...)

Load model from separate Vp, Vs, Rho files.

This is the most common workflow where each property is stored in a separate file.
Supports mixed formats (e.g., Vp as SEG-Y, Vs as binary).

# Arguments
- `vp`: Path to Vp file (required)
- `vs`: Path to Vs file (optional, defaults to Vp/1.73)
- `rho`: Path to density file (optional, defaults to Gardner relation)
- `dx`, `dz`: Grid spacing
- `nx`, `nz`: Grid dimensions (required for binary files)

# Examples
```julia
# Three SEG-Y files
model = load_model_files(vp="vp.segy", vs="vs.segy", rho="rho.segy", dx=12.5)

# Three binary files
model = load_model_files(vp="vp.bin", vs="vs.bin", rho="rho.bin", 
                         dx=10.0, nx=500, nz=200)

# Only Vp (Vs and Rho estimated)
model = load_model_files(vp="vp.segy", dx=10.0)

# Mixed formats
model = load_model_files(vp="vp.segy", vs="vs.bin", rho="rho.mat",
                         dx=10.0, nx=500, nz=200)
```
"""
function load_model_files(; vp::String, 
                           vs::Union{String,Nothing}=nothing, 
                           rho::Union{String,Nothing}=nothing,
                           dx::Real, dz::Union{Real,Nothing}=nothing,
                           nx::Union{Int,Nothing}=nothing, 
                           nz::Union{Int,Nothing}=nothing,
                           order::Symbol=:column_major,
                           name::String="model")
    
    dz = dz === nothing ? dx : dz
    
    # Helper to read a single file
    function read_file(path::String)
        ext = lowercase(splitext(path)[2])
        
        if ext in [".segy", ".sgy"]
            if !isdefined(@__MODULE__, :SegyIO) && !isdefined(Main, :SegyIO)
                error("SegyIO.jl required. Add 'using SegyIO' to your script.")
            end
            SegyIO_mod = isdefined(Main, :SegyIO) ? Main.SegyIO : (@__MODULE__).SegyIO
            return Float32.(Base.invokelatest(SegyIO_mod.segy_read, path).data)
            
        elseif ext == ".bin"
            if nx === nothing || nz === nothing
                error("Binary files require nx and nz parameters")
            end
            data = zeros(Float32, nx * nz)
            open(path, "r") do io
                read!(io, data)
            end
            if order == :row_major
                return permutedims(reshape(data, nz, nx))
            else
                return reshape(data, nx, nz)
            end
            
        elseif ext == ".jld2"
            d = JLD2.load(path)
            for key in ["vp", "Vp", "VP", "vs", "Vs", "VS", "rho", "Rho", "data"]
                if haskey(d, key)
                    return Float32.(d[key])
                end
            end
            error("No data found in JLD2: $path")
            
        elseif ext == ".npy"
            if !isdefined(@__MODULE__, :NPZ) && !isdefined(Main, :NPZ)
                error("NPZ.jl required. Add 'using NPZ' to your script.")
            end
            NPZ_mod = isdefined(Main, :NPZ) ? Main.NPZ : (@__MODULE__).NPZ
            return Float32.(Base.invokelatest(NPZ_mod.npzread, path))
            
        elseif ext == ".mat"
            if !isdefined(@__MODULE__, :MAT) && !isdefined(Main, :MAT)
                error("MAT.jl required. Add 'using MAT' to your script.")
            end
            MAT_mod = isdefined(Main, :MAT) ? Main.MAT : (@__MODULE__).MAT
            d = Base.invokelatest(MAT_mod.matread, path)
            for (k, v) in d
                if v isa AbstractMatrix
                    return Float32.(v)
                end
            end
            error("No matrix in MAT file: $path")
        else
            error("Unsupported format: $ext")
        end
    end
    
    # Load Vp
    @info "Loading Vp" path=vp
    vp_data = read_file(vp)
    
    # Load or estimate Vs
    vs_data = if vs !== nothing
        @info "Loading Vs" path=vs
        read_file(vs)
    else
        @info "Estimating Vs = Vp / 1.73"
        vp_data ./ 1.73f0
    end
    
    # Load or estimate Rho
    rho_data = if rho !== nothing
        @info "Loading Rho" path=rho
        read_file(rho)
    else
        @info "Estimating Rho (Gardner relation)"
        310.0f0 .* (vp_data .^ 0.25f0)
    end
    
    return VelocityModel(vp_data, vs_data, rho_data, Float32(dx), Float32(dz); name=name)
end
