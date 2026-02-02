# ==============================================================================
# core/simulation_configuration.jl
#
# Configuration structs for simulation and results.
# ★ MODIFIED: Added external wavelet support
# ==============================================================================

"""
    SimulationConfig

Configuration parameters for the simulation.

# Fields
- `nbc::Int`: Number of absorbing boundary layers (HABC)
- `fd_order::Int`: Finite difference order
- `dt::Union{Float32, Nothing}`: Time step in seconds
- `nt::Int`: Number of time steps
- `cfl::Float32`: CFL stability number
- `f0::Float32`: Source dominant frequency (Hz)
- `wavelet::Union{Vector{Float32}, Nothing}`: External source wavelet (★ NEW)
- `wavelet_t0::Union{Float32, Nothing}`: Time shift for internal wavelet (★ NEW)
- `free_surface::Bool`: Enable free surface (legacy)
- `source_type::Symbol`: Source mechanism type
  - `:explosion` (default): Pressure source, injects equally into σxx and σzz
  - `:force_z`: Vertical body force Fz, injects into vz with dt/ρ scaling
  - `:force_x`: Horizontal body force Fx, injects into vx with dt/ρ scaling
  - `:stress_txx`, `:stress_tzz`, `:stress_txz`: Single stress component source
- `output_dir::String`: Output directory
- `save_gather::Bool`: Save binary gather
- `show_progress::Bool`: Show progress bar
- `plot_gather::Bool`: Save gather plot
- `boundary_config::Union{BoundaryConfig, Nothing}`: Advanced boundary configuration
"""
struct SimulationConfig
    nbc::Int
    fd_order::Int
    dt::Union{Float32,Nothing}
    nt::Int
    cfl::Float32
    f0::Float32
    wavelet::Union{Vector{Float32},Nothing}      # ★ NEW: External wavelet
    wavelet_t0::Union{Float32,Nothing}           # ★ NEW: t0 for internal wavelet
    free_surface::Bool
    source_type::Symbol
    output_dir::String
    save_gather::Bool
    show_progress::Bool
    plot_gather::Bool
    boundary_config::Union{BoundaryConfig,Nothing}
end

"""
    SimulationConfig(; kwargs...)

Configuration for regular (flat) free surface simulation.

# Keyword Arguments
- `nbc::Int = 50`: Number of absorbing boundary layers (PML/HABC)
- `fd_order::Int = 8`: Finite difference order, must be even. Options: `2`, `4`, `6`, `8`, `10`
- `dt::Union{Float32, Nothing} = nothing`: Time step in seconds. If `nothing`, auto-computed from CFL condition
- `nt::Int = 3000`: Number of time steps
- `cfl::Float32 = 0.4f0`: CFL number for auto time step computation (typically 0.3-0.5)
- `f0::Float32 = 15.0f0`: Source dominant frequency in Hz
- `wavelet::Union{Vector{Float32}, Nothing} = nothing`: External source wavelet.
  ★ RECOMMENDED: Provide external wavelet for precise control and consistency with other software.
- `wavelet_t0::Union{Float32, Nothing} = nothing`: Time shift for internal Ricker wavelet.
  Only used when `wavelet=nothing`. Default: `1.2/f0` (SPECFEM2D/Deepwave convention).
- `free_surface::Bool = true`: Enable free surface at top boundary
- `source_type::Symbol = :explosion`: Source mechanism type.
  Options: `:explosion`, `:force_z`, `:force_x`, `:stress_txx`, `:stress_tzz`, `:stress_txz`
- `output_dir::String = "outputs"`: Directory for output files
- `save_gather::Bool = true`: Save seismic gather to binary file
- `show_progress::Bool = true`: Show progress bar during simulation
- `plot_gather::Bool = true`: Save gather plot as PNG

# Wavelet Handling Priority
1. If `wavelet` is provided → use external wavelet directly (★ RECOMMENDED)
2. If `wavelet_t0` is provided → generate Ricker with custom t0
3. Otherwise → generate Ricker with t0 = 1.2/f0 (SPECFEM2D compatible)

# Example
```julia
# ★ Method 1: External wavelet (RECOMMENDED)
dt, nt, f0 = 0.001f0, 2001, 15.0f0
wavelet = ricker_wavelet(f0, dt, nt)  # t0 = 1.2/f0 = 80ms

config = SimulationConfig(
    nt = nt,
    f0 = f0,
    wavelet = wavelet,       # ★ External wavelet
    source_type = :force_z
)

# Method 2: Internal wavelet with explicit t0
config = SimulationConfig(
    nt = 2001,
    f0 = 15.0f0,
    wavelet_t0 = 0.08f0,     # 80ms = 1.2/15
    source_type = :force_z
)

# Method 3: Internal wavelet with default t0 = 1.2/f0
config = SimulationConfig(nt=5000, f0=20.0f0)
```

See also: [`ricker_wavelet`](@ref), [`simulate!`](@ref), [`VideoConfig`](@ref)
"""
function SimulationConfig(;
    nbc::Int=50,
    fd_order::Int=8,
    dt::Union{Float32,Nothing}=nothing,
    nt::Int=3000,
    cfl::Float32=0.4f0,
    f0::Float32=15.0f0,
    wavelet::Union{Vector{Float32},Nothing}=nothing,     # ★ NEW
    wavelet_t0::Union{Float32,Nothing}=nothing,          # ★ NEW
    free_surface::Bool=true,
    source_type::Symbol=:explosion,
    output_dir::String="outputs",
    save_gather::Bool=true,
    show_progress::Bool=true,
    plot_gather::Bool=true,
    boundary_config::Union{BoundaryConfig,Nothing}=nothing
)
    # Validate source_type
    valid_source_types = (:explosion, :force_z, :force_x, :stress_txx, :stress_tzz, :stress_txz)
    if !(source_type in valid_source_types)
        error("Invalid source_type: $(source_type). Must be one of $(valid_source_types)")
    end

    # Validate external wavelet if provided
    if wavelet !== nothing
        if length(wavelet) != nt
            error("External wavelet length ($(length(wavelet))) must match nt ($nt)")
        end
        @info "Using external wavelet" length = length(wavelet) max_amp = maximum(abs, wavelet)
    end

    # If boundary_config is not provided, create from legacy free_surface
    actual_boundary_config = boundary_config
    if boundary_config === nothing
        top_boundary = free_surface ? :image : :absorbing
        actual_boundary_config = BoundaryConfig(top_boundary=top_boundary, nbc=nbc)
    end

    return SimulationConfig(
        nbc, fd_order, dt, nt, cfl, f0,
        wavelet, wavelet_t0,  # ★ NEW fields
        free_surface, source_type, output_dir,
        save_gather, show_progress, plot_gather, actual_boundary_config
    )
end

"""
    IrregularSurfaceConfig(; kwargs...)

Configuration for irregular free surface simulation using Immersed Boundary Method (IBM).

# Keyword Arguments
- `nbc::Int = 50`: Number of absorbing boundary layers
- `fd_order::Int = 8`: Finite difference order. Options: `2`, `4`, `6`, `8`, `10`
- `dt::Union{Float32, Nothing} = nothing`: Time step in seconds. If `nothing`, auto-computed
- `nt::Int = 3000`: Number of time steps
- `cfl::Float32 = 0.5f0`: CFL number (use smaller value ~0.4 for `:mirror` method)
- `f0::Float32 = 15.0f0`: Source dominant frequency in Hz
- `wavelet::Union{Vector{Float32}, Nothing} = nothing`: External source wavelet (★ NEW)
- `wavelet_t0::Union{Float32, Nothing} = nothing`: Time shift for internal wavelet (★ NEW)
- `ibm_method::Symbol = :direct_zero`: IBM boundary condition method
  - `:direct_zero` - Directly set ghost point values to zero. **Stable**, recommended for most cases
  - `:mirror` - Mirror/antisymmetric extrapolation. **Higher accuracy** but may need smaller time step
- `ibm_iterations::Int = 3`: Number of IBM iterations per time step (3-5 recommended)
- `src_depth::Float32 = 30.0f0`: Source depth below surface in meters
- `rec_depth::Float32 = 0.0f0`: Receiver depth below surface in meters (0 = on surface)
- `output_dir::String = "outputs"`: Directory for output files
- `save_gather::Bool = true`: Save seismic gather to binary file
- `show_progress::Bool = true`: Show progress bar during simulation
- `plot_gather::Bool = true`: Save gather plot as PNG
- `plot_model::Bool = true`: Save model setup plot as PNG

# Example
```julia
# With external wavelet
wavelet = ricker_wavelet(20.0f0, 0.001f0, 4000)

config = IrregularSurfaceConfig(
    nt = 4000,
    f0 = 20.0f0,
    wavelet = wavelet,           # ★ External wavelet
    ibm_method = :direct_zero,
    ibm_iterations = 3,
    src_depth = 50.0f0,
    rec_depth = 0.0f0,
    output_dir = "irregular_outputs"
)
```

See also: [`simulate_irregular!`](@ref), [`VideoConfig`](@ref)
"""
Base.@kwdef struct IrregularSurfaceConfig
    nbc::Int = 50
    fd_order::Int = 8
    dt::Union{Float32,Nothing} = nothing
    nt::Int = 3000
    cfl::Float32 = 0.5f0
    f0::Float32 = 15.0f0
    wavelet::Union{Vector{Float32},Nothing} = nothing    # ★ NEW
    wavelet_t0::Union{Float32,Nothing} = nothing         # ★ NEW
    ibm_method::Symbol = :direct_zero
    ibm_iterations::Int = 3
    src_depth::Float32 = 30.0f0
    rec_depth::Float32 = 0.0f0
    output_dir::String = "outputs"
    save_gather::Bool = true
    show_progress::Bool = true
    plot_gather::Bool = true
    plot_model::Bool = true
end

"""
    SimulationResult

Container for simulation results returned by `simulate!` and `simulate_irregular!`.

# Fields
- `gather::Matrix{Float32}`: Recorded seismogram, shape `[nt, n_receivers]`
- `dt::Float32`: Time step used in simulation (seconds)
- `nt::Int`: Number of time steps
- `video_file::Union{String, Nothing}`: Path to generated video file, or `nothing` if no video
- `gather_file::Union{String, Nothing}`: Path to saved gather binary file
- `gather_plot::Union{String, Nothing}`: Path to gather plot PNG file

# Example
```julia
result = simulate!(model, src_x, src_z, rec_x, rec_z; config=config)

# Access results
println("Gather size: ", size(result.gather))
println("Time step: ", result.dt, " s")
println("Video saved to: ", result.video_file)
```
"""
struct SimulationResult
    gather::Matrix{Float32}
    dt::Float32
    nt::Int
    video_file::Union{String,Nothing}
    gather_file::Union{String,Nothing}
    gather_plot::Union{String,Nothing}
end