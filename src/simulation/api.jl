# ==============================================================================
# simulation/api.jl
#
# High-level API for simplified simulation workflow
# ==============================================================================

# ==============================================================================
# Configuration Structs
# ==============================================================================

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
Base.@kwdef struct SimulationConfig
    nbc::Int = 50
    fd_order::Int = 8
    dt::Union{Float32,Nothing} = nothing
    nt::Int = 3000
    cfl::Float32 = 0.4f0
    f0::Float32 = 15.0f0
    free_surface::Bool = true
    output_dir::String = "outputs"
    save_gather::Bool = true
    show_progress::Bool = true
    plot_gather::Bool = true
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

# ==============================================================================
# Regular Surface API
# ==============================================================================

"""
    simulate!(model, src_x, src_z, rec_x, rec_z; config, video_config=nothing) -> SimulationResult

Run elastic wave simulation with regular (flat) free surface.

Automatically handles:
- Backend selection (CPU/GPU)
- Time step computation (if not specified)
- Wavefield initialization
- Source and receiver setup
- Progress display
- Output file generation

# Arguments
- `model::VelocityModel`: Velocity model containing vp, vs, rho arrays
- `src_x::Real`: Source x position in meters
- `src_z::Real`: Source z position in meters (depth from top)
- `rec_x::Vector{<:Real}`: Receiver x positions in meters
- `rec_z::Vector{<:Real}`: Receiver z positions in meters

# Keyword Arguments
- `config::SimulationConfig`: Simulation configuration (required)
- `video_config::Union{VideoConfig, Nothing} = nothing`: Video recording configuration. 
  If `nothing`, no video is recorded
- `be::Backend`: Backend to use Backend(:cpu) or Backend(:cuda). Default is Backend(:cuda).

# Returns
- `SimulationResult`: Contains gather data and paths to output files

# Output Files
All files are saved to `config.output_dir`:
- `gather.bin`: Binary file with seismic gather (if `save_gather=true`)
- `gather.png`: Plot of seismic gather (if `plot_gather=true`)
- `wavefield_<field>.mp4`: Wavefield video (if `video_config` provided)

# Example
```julia
using Fomo

# Create model
vp = fill(3000.0f0, 200, 400)
vs = fill(1800.0f0, 200, 400)
rho = fill(2200.0f0, 200, 400)
model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)

# Without video
result = simulate!(
    model,
    2000.0f0, 50.0f0,              # source at (2000m, 50m)
    Float32.(100:20:3900),         # receivers from 100m to 3900m
    fill(10.0f0, 190);             # all at 10m depth
    config = SimulationConfig(nt=3000, f0=15.0f0)
)

# With video
result = simulate!(
    model, 2000.0f0, 50.0f0, rec_x, rec_z;
    config = SimulationConfig(nt=3000),
    video_config = VideoConfig(fields=[:vz], skip=5, fps=30)
)
```

See also: [`SimulationConfig`](@ref), [`VideoConfig`](@ref), [`simulate_irregular!`](@ref)
"""
function simulate!(model::VelocityModel,
    src_x::Real, src_z::Real,
    rec_x::Vector{<:Real}, rec_z::Vector{<:Real};
    config::SimulationConfig=SimulationConfig(),
    video_config::Union{VideoConfig,Nothing}=nothing,
    be=backend(:cuda))

    mkpath(config.output_dir)

    #be = is_cuda_available() ? backend(:cuda) : backend(:cpu)
    @info "Simulation started" backend = typeof(be) model = model.name

    vp_max = maximum(model.vp)
    dt = config.dt === nothing ? config.cfl * min(model.dx, model.dz) / vp_max : config.dt
    dt = Float32(dt)

    @info "Parameters" dt_ms = round(dt * 1000, digits=3) nt = config.nt

    medium = init_medium(model, config.nbc, config.fd_order, be; free_surface=config.free_surface)
    habc = init_habc(medium.nx, medium.nz, config.nbc, medium.pad, dt, model.dx, model.dz, vp_max, be)
    fd_coeffs = to_device(get_fd_coefficients(config.fd_order), be)
    wavefield = Wavefield(medium.nx, medium.nz, be)
    params = SimParams(dt, config.nt, model.dx, model.dz, config.fd_order)

    src_i = round(Int, src_x / model.dx) + medium.pad + 1
    src_j = round(Int, src_z / model.dz) + medium.pad + 1
    wavelet = ricker_wavelet(config.f0, dt, config.nt)
    src = Source(src_i, src_j, to_device(wavelet, be))

    n_rec = length(rec_x)
    rec_i = [round(Int, x / model.dx) + medium.pad + 1 for x in rec_x]
    rec_j = [round(Int, z / model.dz) + medium.pad + 1 for z in rec_z]
    rec = Receivers(
        to_device(rec_i, be),
        to_device(rec_j, be),
        to_device(zeros(Float32, config.nt, n_rec), be),
        :vz
    )

    @info "Geometry" source = (src_x, src_z) n_receivers = n_rec

    recorder = nothing
    if video_config !== nothing
        recorder = MultiFieldRecorder(medium.nx, medium.nz, dt, video_config; pad=medium.pad)
    end

    reset!(be, wavefield)

    run_time_loop!(be, wavefield, medium, habc, fd_coeffs, src, rec, params;
        progress=config.show_progress,
        on_step=recorder === nothing ? nothing : (W, info) -> begin
            Fomo.record!(recorder.recorder, W, info.k, dt)
            return true
        end
    )

    @info "Simulation complete"

    gather = be isa CUDABackend ? Array(rec.data) : copy(rec.data)

    video_file, gather_file, gather_plot = _save_outputs(
        gather, dt, config, recorder, video_config
    )

    return SimulationResult(gather, dt, config.nt, video_file, gather_file, gather_plot)
end

# ==============================================================================
# Irregular Surface API
# ==============================================================================

"""
    simulate_irregular!(model, z_surface, src_x, rec_x; config, video_config=nothing) -> SimulationResult

Run elastic wave simulation with irregular free surface using Immersed Boundary Method (IBM).

The surface shape is defined by `z_surface`, which specifies the depth at each x grid point.
Source and receiver positions are specified relative to the surface (depth below surface).

# Arguments
- `model::VelocityModel`: Velocity model containing vp, vs, rho arrays
- `z_surface::Vector{Float32}`: Surface elevation at each x grid point in meters.
  Length must equal `model.nx`. Larger values = deeper surface.
- `src_x::Real`: Source x position in meters
- `rec_x::Vector{<:Real}`: Receiver x positions in meters

# Keyword Arguments
- `config::IrregularSurfaceConfig`: Simulation configuration (required).
  Includes `src_depth` and `rec_depth` for positioning relative to surface.
- `video_config::Union{VideoConfig, Nothing} = nothing`: Video recording configuration

# Returns
- `SimulationResult`: Contains gather data and paths to output files

# Output Files
All files are saved to `config.output_dir`:
- `gather.bin`: Binary file with seismic gather
- `gather.png`: Plot of seismic gather  
- `model_setup.png`: Model and geometry visualization (if `plot_model=true`)
- `surface_elevation.txt`: Surface elevation data
- `wavefield_<field>.mp4`: Wavefield video (if `video_config` provided)

# Example
```julia
using Fomo

# Create model
model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)

# Define surface using helper function
z_surface = sinusoidal_surface(nx, dx; base_depth=50, amplitude=30, wavelength=1000)

# Or combine multiple shapes
z_surface = combine_surfaces(
    sinusoidal_surface(nx, dx; amplitude=20),
    gaussian_valley(nx, dx; valley_depth=25, width=300)
)

# Or define custom shape
x = Float32.((0:nx-1) .* dx)
z_surface = Float32.(50.0 .+ 20.0 .* sin.(2π .* x ./ 1000.0))

# Run simulation
result = simulate_irregular!(
    model,
    z_surface,
    2000.0f0,                      # source x position
    Float32.(100:20:3900);         # receiver x positions
    config = IrregularSurfaceConfig(
        nt = 3000,
        ibm_method = :direct_zero,
        src_depth = 30.0f0,        # 30m below surface
        rec_depth = 0.0f0          # on surface
    ),
    video_config = VideoConfig(fields=[:vz], skip=10)
)
```

See also: [`IrregularSurfaceConfig`](@ref), [`sinusoidal_surface`](@ref), [`combine_surfaces`](@ref)
"""
function simulate_irregular!(model::VelocityModel,
    z_surface::Vector{Float32},
    src_x::Real,
    rec_x::Vector{<:Real};
    config::IrregularSurfaceConfig=IrregularSurfaceConfig(),
    video_config::Union{VideoConfig,Nothing}=nothing,
    be=backend(:cuda))

    mkpath(config.output_dir)

    #be = is_cuda_available() ? backend(:cuda) : backend(:cpu)
    @info "Irregular surface simulation" backend = typeof(be) ibm_method = config.ibm_method

    vp_max = maximum(model.vp)
    dt = config.dt === nothing ? config.cfl * min(model.dx, model.dz) / vp_max : config.dt
    dt = Float32(dt)

    @info "Parameters" dt_ms = round(dt * 1000, digits=3) nt = config.nt

    medium = init_medium(model, config.nbc, config.fd_order, be; free_surface=false)

    surface_cpu = init_irregular_surface(z_surface, medium;
        n_iter=config.ibm_iterations,
        method=config.ibm_method,
        backend=CPU_BACKEND)
    surface = be isa CUDABackend ? to_gpu(surface_cpu) : surface_cpu

    @info "Irregular surface" ghost_points = surface_cpu.n_ghost method = config.ibm_method

    habc = init_habc(medium.nx, medium.nz, config.nbc, medium.pad, dt, model.dx, model.dz, vp_max, be)
    fd_coeffs = to_device(get_fd_coefficients(config.fd_order), be)
    wavefield = Wavefield(medium.nx, medium.nz, be)
    params = SimParams(dt, config.nt, model.dx, model.dz, config.fd_order)

    wavelet = ricker_wavelet(config.f0, dt, config.nt)
    src = setup_irregular_source(Float32(src_x), config.src_depth, wavelet,
        surface_cpu, medium; backend=be)

    n_rec = length(rec_x)
    rec_depths = fill(config.rec_depth, n_rec)
    rec = setup_irregular_receivers(Float32.(rec_x), rec_depths, surface_cpu, medium, config.nt;
        type=:vz, backend=be)

    @info "Geometry" source_x = src_x src_depth = config.src_depth n_receivers = n_rec

    recorder = nothing
    if video_config !== nothing
        recorder = MultiFieldRecorder(medium.nx, medium.nz, dt, video_config)
    end

    if config.plot_model
        _plot_irregular_setup(model, z_surface, src_x, config.src_depth,
            Float32.(rec_x), fill(config.rec_depth, n_rec),
            config.ibm_method,
            joinpath(config.output_dir, "model_setup.png"))
    end

    reset!(be, wavefield)

    if config.show_progress
        prog = Progress(config.nt, desc="Simulating: ")
    end

    for k in 1:config.nt
        time_step_irregular!(be, wavefield, medium, habc, fd_coeffs,
            src, rec, k, params, surface)

        if recorder !== nothing
            Fomo.record!(recorder.recorder, wavefield, k, dt)
        end

        if config.show_progress
            next!(prog)
        end
    end

    synchronize(be)
    @info "Simulation complete"

    gather = be isa CUDABackend ? Array(rec.data) : copy(rec.data)

    # Save surface elevation
    open(joinpath(config.output_dir, "surface_elevation.txt"), "w") do io
        for (i, z) in enumerate(z_surface)
            println(io, "$((i-1) * model.dx) $z")
        end
    end

    video_file, gather_file, gather_plot = _save_outputs(
        gather, dt, config, recorder, video_config
    )

    return SimulationResult(gather, dt, config.nt, video_file, gather_file, gather_plot)
end

# ==============================================================================
# Surface Shape Helper Functions
# ==============================================================================

"""
    flat_surface(nx, dx, depth) -> Vector{Float32}

Create a flat (horizontal) surface at constant depth.

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters
- `depth::Real`: Constant depth of surface in meters

# Returns
- `Vector{Float32}`: Surface elevation array of length `nx`

# Example
```julia
z_surface = flat_surface(400, 10.0, 50.0)  # flat surface at 50m depth
```
"""
flat_surface(nx::Int, dx::Real, depth::Real) = fill(Float32(depth), nx)

"""
    sinusoidal_surface(nx, dx; base_depth=50, amplitude=20, wavelength=1000) -> Vector{Float32}

Create a sinusoidal (wavy) surface.

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `base_depth::Real = 50.0`: Mean depth of surface in meters
- `amplitude::Real = 20.0`: Amplitude of sine wave in meters
- `wavelength::Real = 1000.0`: Wavelength of sine wave in meters

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Gentle undulation
z_surface = sinusoidal_surface(400, 10.0; amplitude=15, wavelength=2000)

# Sharp ripples
z_surface = sinusoidal_surface(400, 10.0; amplitude=30, wavelength=500)
```
"""
function sinusoidal_surface(nx::Int, dx::Real;
    base_depth::Real=50.0,
    amplitude::Real=20.0,
    wavelength::Real=1000.0)
    x = Float32.((0:nx-1) .* dx)
    return Float32.(base_depth .+ amplitude .* sin.(2π .* x ./ wavelength))
end

"""
    gaussian_valley(nx, dx; base_depth=50, valley_depth=30, center=nothing, width=200) -> Vector{Float32}

Create a surface with a Gaussian valley (depression/canyon).

The valley goes **deeper** into the model (larger z values).

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `base_depth::Real = 50.0`: Depth of flat regions in meters
- `valley_depth::Real = 30.0`: Additional depth at valley center in meters
- `center::Union{Real, Nothing} = nothing`: Valley center x position. If `nothing`, centered in model
- `width::Real = 200.0`: Gaussian standard deviation (controls valley width) in meters

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Centered valley
z_surface = gaussian_valley(400, 10.0; valley_depth=40, width=300)

# Off-center valley
z_surface = gaussian_valley(400, 10.0; valley_depth=40, center=1000.0, width=200)
```
"""
function gaussian_valley(nx::Int, dx::Real;
    base_depth::Real=50.0,
    valley_depth::Real=30.0,
    center::Union{Real,Nothing}=nothing,
    width::Real=200.0)
    x = Float32.((0:nx-1) .* dx)
    center = center === nothing ? nx * dx / 2 : Float32(center)
    return Float32.(base_depth .+ valley_depth .* exp.(-(x .- center) .^ 2 ./ (2 * width^2)))
end

"""
    gaussian_hill(nx, dx; base_depth=80, hill_height=30, center=nothing, width=200) -> Vector{Float32}

Create a surface with a Gaussian hill (elevation).

The hill rises **up** from the base (smaller z values at peak).

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `base_depth::Real = 80.0`: Depth of flat regions in meters
- `hill_height::Real = 30.0`: Height of hill above base in meters
- `center::Union{Real, Nothing} = nothing`: Hill center x position. If `nothing`, centered in model
- `width::Real = 200.0`: Gaussian standard deviation (controls hill width) in meters

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
z_surface = gaussian_hill(400, 10.0; base_depth=100, hill_height=50, width=400)
```
"""
function gaussian_hill(nx::Int, dx::Real;
    base_depth::Real=80.0,
    hill_height::Real=30.0,
    center::Union{Real,Nothing}=nothing,
    width::Real=200.0)
    x = Float32.((0:nx-1) .* dx)
    center = center === nothing ? nx * dx / 2 : Float32(center)
    return Float32.(base_depth .- hill_height .* exp.(-(x .- center) .^ 2 ./ (2 * width^2)))
end

"""
    tilted_surface(nx, dx; depth_left=30, depth_right=70) -> Vector{Float32}

Create a linearly tilted (sloped) surface.

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `depth_left::Real = 30.0`: Depth at left edge (x=0) in meters
- `depth_right::Real = 70.0`: Depth at right edge in meters

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Dipping surface (deeper on right)
z_surface = tilted_surface(400, 10.0; depth_left=20, depth_right=80)

# Reverse dip
z_surface = tilted_surface(400, 10.0; depth_left=80, depth_right=20)
```
"""
function tilted_surface(nx::Int, dx::Real;
    depth_left::Real=30.0,
    depth_right::Real=70.0)
    return Float32.(range(depth_left, depth_right, length=nx))
end

"""
    step_surface(nx, dx; depth_left=30, depth_right=70, step_position=nothing) -> Vector{Float32}

Create a surface with a sharp step (cliff/escarpment/fault scarp).

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `depth_left::Real = 30.0`: Depth on left side of step in meters
- `depth_right::Real = 70.0`: Depth on right side of step in meters
- `step_position::Union{Real, Nothing} = nothing`: X position of step. If `nothing`, at center

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Step down (cliff)
z_surface = step_surface(400, 10.0; depth_left=30, depth_right=60)

# Step up (escarpment)
z_surface = step_surface(400, 10.0; depth_left=60, depth_right=30)

# Off-center step
z_surface = step_surface(400, 10.0; depth_left=30, depth_right=60, step_position=1500.0)
```
"""
function step_surface(nx::Int, dx::Real;
    depth_left::Real=30.0,
    depth_right::Real=70.0,
    step_position::Union{Real,Nothing}=nothing)
    step_idx = step_position === nothing ? nx ÷ 2 : round(Int, step_position / dx)
    z = zeros(Float32, nx)
    z[1:step_idx] .= Float32(depth_left)
    z[step_idx+1:end] .= Float32(depth_right)
    return z
end

"""
    random_surface(nx, dx; base_depth=50, amplitude=10, smoothness=5) -> Vector{Float32}

Create a random rough surface with controllable smoothness.

Uses Gaussian random noise with moving average smoothing.

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `base_depth::Real = 50.0`: Mean depth of surface in meters
- `amplitude::Real = 10.0`: Standard deviation of random roughness in meters
- `smoothness::Int = 5`: Smoothing window half-width in grid points. Larger = smoother

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Rough surface
z_surface = random_surface(400, 10.0; amplitude=20, smoothness=3)

# Gentle rolling terrain
z_surface = random_surface(400, 10.0; amplitude=15, smoothness=15)
```

# Note
Results are random and will differ each time. Use `Random.seed!()` for reproducibility.
"""
function random_surface(nx::Int, dx::Real;
    base_depth::Real=50.0,
    amplitude::Real=10.0,
    smoothness::Int=5)
    noise = randn(Float32, nx) .* Float32(amplitude)
    z = zeros(Float32, nx)
    for i in 1:nx
        i_start = max(1, i - smoothness)
        i_end = min(nx, i + smoothness)
        z[i] = sum(noise[i_start:i_end]) / (i_end - i_start + 1)
    end
    return Float32.(base_depth .+ z)
end

"""
    combine_surfaces(surfaces...; method=:add) -> Vector{Float32}

Combine multiple surface shapes into one.

# Arguments
- `surfaces...`: Two or more surface arrays (all must have same length)

# Keyword Arguments
- `method::Symbol = :add`: How to combine surfaces
  - `:add` - Add all surfaces element-wise (useful for superimposing perturbations)
  - `:min` - Take minimum depth at each point (highest elevation)
  - `:max` - Take maximum depth at each point (lowest elevation)

# Returns
- `Vector{Float32}`: Combined surface elevation array

# Example
```julia
# Sinusoidal base + valley
z_surface = combine_surfaces(
    sinusoidal_surface(400, 10.0; base_depth=50, amplitude=15),
    gaussian_valley(400, 10.0; base_depth=0, valley_depth=25)  # Note: base_depth=0 for perturbation
)

# Complex terrain
z_surface = combine_surfaces(
    flat_surface(400, 10.0, 60.0),
    sinusoidal_surface(400, 10.0; base_depth=0, amplitude=10, wavelength=500),
    gaussian_valley(400, 10.0; base_depth=0, valley_depth=20, center=1500.0)
)
```

# Tip
When combining, set `base_depth=0` for perturbation shapes so they add properly to the base.
"""
function combine_surfaces(surfaces...; method::Symbol=:add)
    nx = length(surfaces[1])
    if method == :add
        return Float32.(sum(surfaces))
    elseif method == :min
        result = copy(surfaces[1])
        for s in surfaces[2:end]
            result .= min.(result, s)
        end
        return Float32.(result)
    elseif method == :max
        result = copy(surfaces[1])
        for s in surfaces[2:end]
            result .= max.(result, s)
        end
        return Float32.(result)
    end
end

# ==============================================================================
# Internal Helper Functions
# ==============================================================================

function _save_outputs(gather, dt, config, recorder, video_config)
    video_file = nothing
    gather_file = nothing
    gather_plot = nothing

    n_rec = size(gather, 2)

    if config.save_gather
        gather_file = joinpath(config.output_dir, "gather.bin")
        open(gather_file, "w") do io
            write(io, Int32(config.nt))
            write(io, Int32(n_rec))
            write(io, gather)
        end
        @info "Saved" file = gather_file
    end

    if recorder !== nothing && video_config !== nothing
        # Generate video for each recorded field
        for field in video_config.fields
            video_file = joinpath(config.output_dir, "wavefield_$(field).mp4")
            generate_video(recorder.recorder, video_file;
                fps=video_config.fps, colormap=video_config.colormap)
            @info "Saved" file = video_file
        end
    end

    if config.plot_gather
        gather_plot = joinpath(config.output_dir, "gather.png")
        _plot_gather_simple(gather, dt, gather_plot)
        @info "Saved" file = gather_plot
    end

    return video_file, gather_file, gather_plot
end

function _plot_gather_simple(gather::Matrix{Float32}, dt::Float32, output::String)
    nt, n_rec = size(gather)
    t_axis = (0:nt-1) .* dt

    fig = CairoMakie.Figure(size=(900, 700))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel="Receiver", ylabel="Time (s)", title="Shot Gather")

    gmax = maximum(abs.(gather))
    data = gmax > 0 ? gather ./ gmax : gather

    hm = CairoMakie.heatmap!(ax, 1:n_rec, t_axis, data',
        colormap=:seismic, colorrange=(-0.3, 0.3))
    ax.yreversed = true
    CairoMakie.Colorbar(fig[1, 2], hm, label="Normalized Amplitude")
    CairoMakie.save(output, fig)
end

function _plot_irregular_setup(model::VelocityModel, z_surface::Vector{Float32},
    src_x::Real, src_depth::Real,
    rec_x::Vector{Float32}, rec_depth::Vector{Float32},
    ibm_method::Symbol, output::String)

    fig = CairoMakie.Figure(size=(1000, 700))

    x_axis = range(0, (model.nx - 1) * model.dx, length=model.nx)
    z_axis = range(0, (model.nz - 1) * model.dz, length=model.nz)

    ax = CairoMakie.Axis(fig[1, 1],
        xlabel="X (m)", ylabel="Z (m)",
        title="Model with Irregular Surface (IBM: $ibm_method)",
        aspect=CairoMakie.DataAspect())

    hm = CairoMakie.heatmap!(ax, x_axis, z_axis, model.vp', colormap=:viridis)

    surf_x = Float32.((0:length(z_surface)-1) .* model.dx)
    CairoMakie.lines!(ax, surf_x, z_surface, color=:white, linewidth=3, label="Free surface")

    src_idx = clamp(round(Int, src_x / model.dx) + 1, 1, length(z_surface))
    src_z = z_surface[src_idx] + src_depth
    CairoMakie.scatter!(ax, [Float32(src_x)], [Float32(src_z)],
        marker=:star5, markersize=20, color=:red, label="Source")

    rec_z = Float32[]
    for i in 1:length(rec_x)
        idx = clamp(round(Int, rec_x[i] / model.dx) + 1, 1, length(z_surface))
        push!(rec_z, z_surface[idx] + rec_depth[i])
    end
    step = max(1, length(rec_x) ÷ 20)
    CairoMakie.scatter!(ax, rec_x[1:step:end], rec_z[1:step:end],
        marker=:dtriangle, markersize=8, color=:cyan, label="Receivers")

    ax.yreversed = true
    CairoMakie.Colorbar(fig[1, 2], hm, label="Vp (m/s)")
    CairoMakie.axislegend(ax, position=:rb)
    CairoMakie.save(output, fig)
end