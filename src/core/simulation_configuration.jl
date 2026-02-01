# ==============================================================================
# core/simulation_configuration.jl
#
# Configuration structs for simulation and results.
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
- `free_surface::Bool`: Enable free surface (legacy)
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
    free_surface::Bool
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
- `free_surface::Bool = true`: Enable free surface at top boundary
- `output_dir::String = "outputs"`: Directory for output files
- `save_gather::Bool = true`: Save seismic gather to binary file
- `show_progress::Bool = true`: Show progress bar during simulation
- `plot_gather::Bool = true`: Save gather plot as PNG

# Example
```julia
config = SimulationConfig(
    nt = 5000,
    f0 = 20.0f0,
    fd_order = 8,
    output_dir = "my_outputs"
)
```

See also: [`simulate!`](@ref), [`VideoConfig`](@ref)
"""
function SimulationConfig(;
    nbc::Int=50,
    fd_order::Int=8,
    dt::Union{Float32,Nothing}=nothing,
    nt::Int=3000,
    cfl::Float32=0.4f0,
    f0::Float32=15.0f0,
    free_surface::Bool=true,  # Legacy parameter
    output_dir::String="outputs",
    save_gather::Bool=true,
    show_progress::Bool=true,
    plot_gather::Bool=true,
    boundary_config::Union{BoundaryConfig,Nothing}=nothing  # New parameter
)
    # If boundary_config is not provided, create from legacy free_surface
    actual_boundary_config = boundary_config
    if boundary_config === nothing
        top_boundary = free_surface ? :image : :absorbing
        actual_boundary_config = BoundaryConfig(top_boundary=top_boundary, nbc=nbc)
    end

    return SimulationConfig(
        nbc, fd_order, dt, nt, cfl, f0, free_surface, output_dir,
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
config = IrregularSurfaceConfig(
    nt = 4000,
    f0 = 20.0f0,
    ibm_method = :direct_zero,  # or :mirror
    ibm_iterations = 3,
    src_depth = 50.0f0,         # source 50m below surface
    rec_depth = 0.0f0,          # receivers on surface
    output_dir = "irregular_outputs"
)
```

# Notes
- `:direct_zero` is more stable and works well for most topographies
- `:mirror` provides higher accuracy but may become unstable with sharp topography
- If simulation becomes unstable, try reducing `cfl` to 0.3-0.4

See also: [`simulate_irregular!`](@ref), [`VideoConfig`](@ref)
"""
Base.@kwdef struct IrregularSurfaceConfig
    nbc::Int = 50
    fd_order::Int = 8
    dt::Union{Float32,Nothing} = nothing
    nt::Int = 3000
    cfl::Float32 = 0.5f0
    f0::Float32 = 15.0f0
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
