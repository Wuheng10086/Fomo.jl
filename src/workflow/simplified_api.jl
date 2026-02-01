# ==============================================================================
# simulation/simple_api.jl
#
# Simplified high-level API for easy simulation workflow
# ==============================================================================

# ==============================================================================
# Simplified Configuration Structs
# ==============================================================================

"""
    SimpleConfig(; kwargs...)

Simplified configuration for common simulation scenarios.

# Keyword Arguments
- `duration::Float64 = 2.0`: Simulation duration in seconds
- `dt::Union{Float64, Nothing} = nothing`: Time step in seconds (auto-computed if nothing)
- `dx::Float64 = 10.0`: Spatial grid spacing in meters
- `fd_order::Int = 8`: Finite difference order (2, 4, 6, 8, 10)
- `boundary_layers::Int = 50`: Number of absorbing boundary layers
- `source_freq::Float64 = 15.0`: Source dominant frequency in Hz
- `free_surface::Bool = true`: Enable free surface condition
- `output_dir::String = "outputs"`: Output directory
- `save_results::Bool = true`: Save results to files
- `show_progress::Bool = true`: Show progress bar
- `backend::Symbol = :auto`: Backend to use (:auto, :cpu, :cuda)

# Example
```julia
config = SimpleConfig(
    duration = 3.0,
    dx = 5.0,
    source_freq = 20.0,
    output_dir = "my_simulation"
)
```
"""
Base.@kwdef struct SimpleConfig
    duration::Float64 = 2.0
    dt::Union{Float64,Nothing} = nothing
    dx::Float64 = 10.0
    fd_order::Int = 8
    boundary_layers::Int = 50
    source_freq::Float64 = 15.0
    free_surface::Bool = true
    output_dir::String = "outputs"
    save_results::Bool = true
    show_progress::Bool = true
    backend::Symbol = :auto
end

"""
    TopographyConfig(; kwargs...)

Configuration for simulations with irregular topography (vacuum formulation).

# Keyword Arguments
- `duration::Float64 = 2.0`: Simulation duration in seconds
- `dt::Union{Float64, Nothing} = nothing`: Time step in seconds (auto-computed if nothing)
- `dx::Float64 = 10.0`: Spatial grid spacing in meters
- `fd_order::Int = 8`: Finite difference order (2, 4, 6, 8, 10)
- `boundary_layers::Int = 50`: Number of absorbing boundary layers
- `source_freq::Float64 = 15.0`: Source dominant frequency in Hz
- `src_depth::Float64 = 30.0`: Source depth below surface in meters
- `rec_depth::Float64 = 0.0`: Receiver depth below surface in meters
- `output_dir::String = "outputs"`: Output directory
- `save_results::Bool = true`: Save results to files
- `show_progress::Bool = true`: Show progress bar
- `backend::Symbol = :auto`: Backend to use (:auto, :cpu, :cuda)

# Example
```julia
config = TopographyConfig(
    duration = 3.0,
    dx = 5.0,
    src_depth = 50.0,
    output_dir = "topography_simulation"
)
```
"""
Base.@kwdef struct TopographyConfig
    duration::Float64 = 2.0
    dt::Union{Float64,Nothing} = nothing
    dx::Float64 = 10.0
    fd_order::Int = 8
    boundary_layers::Int = 50
    source_freq::Float64 = 15.0
    src_depth::Float64 = 30.0
    rec_depth::Float64 = 0.0
    output_dir::String = "outputs"
    save_results::Bool = true
    show_progress::Bool = true
    backend::Symbol = :auto
end

# ==============================================================================
# Model Construction Helpers
# ==============================================================================

"""
    create_homogeneous_model(vp, vs, rho, dims, dx; dz=nothing, name="homogeneous")

Create a homogeneous velocity model with uniform properties.

# Arguments
- `vp::Real`: P-wave velocity in m/s
- `vs::Real`: S-wave velocity in m/s
- `rho::Real`: Density in kg/m³
- `dims::Tuple{Int, Int}`: Dimensions as (nz, nx)
- `dx::Real`: Grid spacing in x-direction in meters

# Keyword Arguments
- `dz::Union{Real, Nothing} = nothing`: Grid spacing in z-direction in meters (same as dx if nothing)
- `name::String = "homogeneous"`: Model name

# Returns
- `VelocityModel`: Homogeneous velocity model

# Example
```julia
# Create a 200x400 model with 10m grid spacing
model = create_homogeneous_model(3000.0, 1800.0, 2200.0, (200, 400), 10.0)
```
"""
function create_homogeneous_model(vp::Real, vs::Real, rho::Real, dims::Tuple{Int,Int}, dx::Real;
    dz::Union{Real,Nothing}=nothing, name::String="homogeneous")
    nz, nx = dims
    dz = dz === nothing ? dx : dz

    vp_array = fill(Float32(vp), nz, nx)
    vs_array = fill(Float32(vs), nz, nx)
    rho_array = fill(Float32(rho), nz, nx)

    return VelocityModel(vp_array, vs_array, rho_array, Float32(dx), Float32(dz); name=name)
end

"""
    create_layered_model(layers, dx; dz=nothing, name="layered")

Create a layered velocity model.

# Arguments
- `layers::Vector{NamedTuple}`: Vector of layer specifications with fields:
  - `thickness::Real`: Layer thickness in meters
  - `vp::Real`: P-wave velocity in m/s
  - `vs::Real`: S-wave velocity in m/s
  - `rho::Real`: Density in kg/m³
- `dx::Real`: Grid spacing in x-direction in meters

# Keyword Arguments
- `dz::Union{Real, Nothing} = nothing`: Grid spacing in z-direction in meters (same as dx if nothing)
- `name::String = "layered"`: Model name

# Returns
- `VelocityModel`: Layered velocity model

# Example
```julia
# Create a 2-layer model
layers = [
    (thickness=100.0, vp=2500.0, vs=1500.0, rho=2000.0),  # Top layer
    (thickness=200.0, vp=3500.0, vs=2100.0, rho=2300.0)   # Bottom layer
]
model = create_layered_model(layers, 10.0)
```
"""
function create_layered_model(layers::Vector{<:NamedTuple}, dx::Real;
    dz::Union{Real,Nothing}=nothing, name::String="layered")
    dz = dz === nothing ? dx : dz

    # Calculate total depth
    total_depth = sum(layer.thickness for layer in layers)
    nz = round(Int, total_depth / dz) + 1
    nx = 400  # Default width, can be adjusted based on context

    # Initialize arrays
    vp_array = zeros(Float32, nz, nx)
    vs_array = zeros(Float32, nz, nx)
    rho_array = zeros(Float32, nz, nx)

    # Fill in layers
    current_depth = 0
    for (idx, layer) in enumerate(layers)
        layer_nz = round(Int, layer.thickness / dz)
        if idx == length(layers)
            # Last layer fills remaining space
            layer_nz = nz - current_depth
        end

        start_row = current_depth + 1
        end_row = min(current_depth + layer_nz, nz)

        if start_row <= nz
            vp_array[start_row:end_row, :] .= Float32(layer.vp)
            vs_array[start_row:end_row, :] .= Float32(layer.vs)
            rho_array[start_row:end_row, :] .= Float32(layer.rho)
        end

        current_depth += layer_nz
        if current_depth >= nz
            break
        end
    end

    return VelocityModel(vp_array, vs_array, rho_array, Float32(dx), Float32(dz); name=name)
end

"""
    create_gradient_model(vp_func, vs_func, rho_func, dims, dx; dz=nothing, name="gradient")

Create a velocity model with spatially varying properties using functions.

# Arguments
- `vp_func::Function`: Function f(z, x) returning P-wave velocity at (z, x) in m/s
- `vs_func::Function`: Function f(z, x) returning S-wave velocity at (z, x) in m/s
- `rho_func::Function`: Function f(z, x) returning density at (z, x) in kg/m³
- `dims::Tuple{Int, Int}`: Dimensions as (nz, nx)
- `dx::Real`: Grid spacing in x-direction in meters

# Keyword Arguments
- `dz::Union{Real, Nothing} = nothing`: Grid spacing in z-direction in meters (same as dx if nothing)
- `name::String = "gradient"`: Model name

# Returns
- `VelocityModel`: Gradient velocity model

# Example
```julia
# Create a model with velocity increasing with depth
vp_func(z, x) = 2000.0 + 1.5z  # 2000 + 1.5 m/s per meter depth
vs_func(z, x) = 1200.0 + 0.8z
rho_func(z, x) = 2000.0 + 0.3z

model = create_gradient_model(vp_func, vs_func, rho_func, (200, 400), 10.0)
```
"""
function create_gradient_model(vp_func::Function, vs_func::Function, rho_func::Function,
    dims::Tuple{Int,Int}, dx::Real;
    dz::Union{Real,Nothing}=nothing, name::String="gradient")
    nz, nx = dims
    dz = dz === nothing ? dx : dz

    vp_array = zeros(Float32, nz, nx)
    vs_array = zeros(Float32, nz, nx)
    rho_array = zeros(Float32, nz, nx)

    for i in 1:nx
        for j in 1:nz
            x = (i - 1) * dx
            z = (j - 1) * dz

            vp_array[j, i] = Float32(vp_func(z, x))
            vs_array[j, i] = Float32(vs_func(z, x))
            rho_array[j, i] = Float32(rho_func(z, x))
        end
    end

    return VelocityModel(vp_array, vs_array, rho_array, Float32(dx), Float32(dz); name=name)
end

# ==============================================================================
# Source and Receiver Configuration
# ==============================================================================

"""
    SourceConfig(position, wavelet_type=:ricker; amplitude=1.0, frequency=nothing)

Configure a source for simulation.

# Arguments
- `position::Tuple{Real, Real}`: Source position as (x, z) in meters
- `wavelet_type::Symbol`: Type of source wavelet (:ricker, :impulse, :sin)

# Keyword Arguments
- `amplitude::Real = 1.0`: Amplitude of the source
- `frequency::Union{Real, Nothing} = nothing`: Frequency of the source in Hz (uses config default if nothing)

# Returns
- Named tuple with source configuration

# Example
```julia
# Ricker wavelet source
src_config = SourceConfig((1000.0, 50.0), :ricker; frequency=15.0)

# Impulse source
src_config = SourceConfig((1500.0, 30.0), :impulse)
```
"""
function SourceConfig(position::Tuple{Real,Real}, wavelet_type::Symbol=:ricker;
    amplitude::Real=1.0, frequency::Union{Real,Nothing}=nothing)
    return (
        position=position,
        wavelet_type=wavelet_type,
        amplitude=amplitude,
        frequency=frequency
    )
end

"""
    ReceiverConfig(positions; spacing=nothing, count=nothing)

Configure receivers for simulation.

# Arguments
- `positions::Union{Vector{Tuple{Real, Real}}, Nothing} = nothing`: Explicit receiver positions as [(x₁,z₁), (x₂,z₂), ...]

# Keyword Arguments
- `spacing::Union{Real, Nothing} = nothing`: Spacing between receivers in meters (for evenly spaced receivers)
- `count::Union{Int, Nothing} = nothing`: Number of receivers (for evenly spaced receivers)
- `start_pos::Tuple{Real, Real} = (0.0, 0.0): Starting position for evenly spaced receivers
- `end_pos::Tuple{Real, Real} = (0.0, 0.0): Ending position for evenly spaced receivers

# Returns
- Named tuple with receiver configuration

# Example
```julia
# Explicit positions
rec_config = ReceiverConfig([(100.0, 10.0), (200.0, 10.0), (300.0, 10.0)])

# Evenly spaced receivers
rec_config = ReceiverConfig(nothing; spacing=20.0, count=100, start_pos=(50.0, 10.0), end_pos=(2050.0, 10.0))
```
"""
function ReceiverConfig(positions::Union{Vector{Tuple{Real,Real}},Nothing}=nothing;
    spacing::Union{Real,Nothing}=nothing, count::Union{Int,Nothing}=nothing,
    start_pos::Tuple{Real,Real}=(0.0, 0.0), end_pos::Tuple{Real,Real}=(0.0, 0.0))
    if positions !== nothing
        # Use explicit positions
        return (positions=positions,)
    elseif count !== nothing && start_pos !== nothing && end_pos !== nothing
        # Generate evenly spaced receivers based on count and positions
        x_start, z_start = start_pos
        x_end, z_end = end_pos

        x_positions = range(x_start, stop=x_end, length=count)
        z_positions = range(z_start, stop=z_end, length=count)

        positions = [(x_positions[i], z_positions[i]) for i in 1:count]
        return (positions=positions,)
    else
        error("Either provide explicit positions or count with start_pos and end_pos for evenly spaced receivers")
    end
end

# ==============================================================================
# High-Level Simulation Functions
# ==============================================================================

"""
    simulate(model, sources, receivers; config=SimpleConfig())

Run a simplified elastic wave simulation with automatic parameter calculation.

# Arguments
- `model::VelocityModel`: Velocity model for the simulation
- `sources::Union{SourceConfig, Vector{SourceConfig}}`: Source configuration(s)
- `receivers::ReceiverConfig`: Receiver configuration

# Keyword Arguments
- `config::Union{SimpleConfig, TopographyConfig} = SimpleConfig()`: Simulation configuration

# Returns
- `SimulationResult`: Results of the simulation

# Example
```julia
# Create a simple model
model = create_homogeneous_model(3000.0, 1800.0, 2200.0, (200, 400), 10.0)

# Configure source and receivers
src = SourceConfig((2000.0, 50.0), :ricker; frequency=15.0)
rec = ReceiverConfig(nothing; count=100, spacing=20.0, start_pos=(100.0, 10.0), end_pos=(2000.0, 10.0))

# Run simulation
result = simulate(model, src, rec; config=SimpleConfig(duration=2.0, dx=10.0))
```
"""
function simulate(model::VelocityModel,
    sources::Union{NamedTuple,Vector{<:NamedTuple}},
    receivers::NamedTuple;
    config::Union{SimpleConfig,TopographyConfig}=SimpleConfig())

    # Determine backend
    be = if config.backend == :auto
        is_cuda_available() ? backend(:cuda) : backend(:cpu)
    elseif config.backend == :cuda
        backend(:cuda)
    else
        backend(:cpu)
    end

    # Convert source configuration to appropriate format
    src_configs = sources isa NamedTuple ? [sources] : sources
    src_x, src_z = src_configs[1].position  # For now, just use first source

    # Extract receiver positions
    rec_positions = receivers.positions
    rec_x = [pos[1] for pos in rec_positions]
    rec_z = [pos[2] for pos in rec_positions]

    # Calculate number of time steps based on duration and dx
    vp_max = maximum(model.vp)
    dx = config.dx
    dt = config.dt === nothing ? 0.4 * dx / vp_max : config.dt  # CFL condition
    nt = round(Int, config.duration / dt)

    # Create appropriate configuration based on config type
    if config isa SimpleConfig
        sim_config = SimulationConfig(
            nt=nt,
            dt=Float32(dt),
            fd_order=config.fd_order,
            nbc=config.boundary_layers,
            f0=Float32(config.source_freq),
            free_surface=config.free_surface,
            output_dir=config.output_dir,
            save_gather=config.save_results,
            show_progress=config.show_progress
        )

        # Run the simulation
        return simulate!(model, src_x, src_z, rec_x, rec_z;
            config=sim_config, be=be)
    elseif config isa TopographyConfig
        error("TopographyConfig workflow is not available: IBM boundary method has been removed. Use init_medium_vacuum + run_time_loop! (vacuum formulation) to drive topography simulations.")
    end
end

"""
    batch_simulate(models, sources, receivers; config=SimpleConfig())

Run multiple simulations in batch mode.

# Arguments
- `models::Vector{VelocityModel}`: Vector of velocity models
- `sources::Vector{Union{SourceConfig, Vector{SourceConfig}}}`: Vector of source configurations
- `receivers::Vector{ReceiverConfig}`: Vector of receiver configurations

# Keyword Arguments
- `config::Union{SimpleConfig, TopographyConfig} = SimpleConfig()`: Simulation configuration

# Returns
- `Vector{SimulationResult}`: Vector of simulation results

# Example
```julia
# Create multiple models
models = [
    create_homogeneous_model(2500.0, 1500.0, 2000.0, (200, 400), 10.0),
    create_homogeneous_model(3000.0, 1800.0, 2200.0, (200, 400), 10.0)
]

# Configure sources and receivers
sources = [SourceConfig((2000.0, 50.0), :ricker; frequency=15.0) for _ in 1:2]
receivers = [ReceiverConfig(nothing; count=50, start_pos=(100.0, 10.0), end_pos=(2000.0, 10.0)) for _ in 1:2]

# Run batch simulation
results = batch_simulate(models, sources, receivers; config=SimpleConfig(duration=1.5))
```
"""
function batch_simulate(models::Vector{<:VelocityModel},
    sources::Vector{<:Union{NamedTuple,Vector{<:NamedTuple}}},
    receivers::Vector{<:NamedTuple};
    config::Union{SimpleConfig,TopographyConfig}=SimpleConfig())

    n_models = length(models)
    @assert length(sources) == n_models "Number of sources must match number of models"
    @assert length(receivers) == n_models "Number of receivers must match number of models"

    results = Vector{SimulationResult}(undef, n_models)

    for i in 1:n_models
        println("Running simulation $i/$n_models...")
        results[i] = simulate(models[i], sources[i], receivers[i]; config=config)
    end

    return results
end

"""
    param_scan(base_model, param_range, sources, receivers; config=SimpleConfig(), param_name=:vp)

Perform a parameter scan by varying a single parameter across a range of values.

# Arguments
- `base_model::VelocityModel`: Base velocity model to modify
- `param_range::Vector`: Range of parameter values to test
- `sources::Union{SourceConfig, Vector{SourceConfig}}`: Source configuration(s)
- `receivers::ReceiverConfig`: Receiver configuration
- `param_name::Symbol`: Parameter to vary (:vp, :vs, :rho)

# Keyword Arguments
- `config::Union{SimpleConfig, TopographyConfig} = SimpleConfig()`: Simulation configuration

# Returns
- `Vector{Tuple{Real, SimulationResult}}`: Vector of (parameter_value, result) tuples

# Example
```julia
# Create base model
base_model = create_homogeneous_model(3000.0, 1800.0, 2200.0, (200, 400), 10.0)

# Define parameter range
vp_range = [2800.0, 3000.0, 3200.0, 3400.0]

# Configure source and receivers
src = SourceConfig((2000.0, 50.0), :ricker; frequency=15.0)
rec = ReceiverConfig(nothing; count=50, start_pos=(100.0, 10.0), end_pos=(2000.0, 10.0))

# Perform parameter scan
scan_results = param_scan(base_model, vp_range, src, rec; param_name=:vp)
```
"""
function param_scan(base_model::VelocityModel,
    param_range::Vector,
    sources::Union{NamedTuple,Vector{<:NamedTuple}},
    receivers::NamedTuple;
    config::Union{SimpleConfig,TopographyConfig}=SimpleConfig(),
    param_name::Symbol=:vp)

    results = []

    for param_val in param_range
        # Create modified model with new parameter value
        new_model = _modify_model_param(base_model, param_val, param_name)

        # Run simulation
        result = simulate(new_model, sources, receivers; config=config)

        push!(results, (param_val, result))
        println("Completed simulation for $param_name = $param_val")
    end

    return results
end

# Helper function to modify a parameter in a model
function _modify_model_param(model::VelocityModel, new_value::Real, param_name::Symbol)
    # Create copies of the existing arrays
    new_vp = copy(model.vp)
    new_vs = copy(model.vs)
    new_rho = copy(model.rho)

    if param_name == :vp
        new_vp[:, :] .= Float32(new_value)
    elseif param_name == :vs
        new_vs[:, :] .= Float32(new_value)
    elseif param_name == :rho
        new_rho[:, :] .= Float32(new_value)
    else
        error("Unknown parameter name: $param_name. Use :vp, :vs, or :rho")
    end

    return VelocityModel(new_vp, new_vs, new_rho, model.dx, model.dz;
        x_origin=model.x_origin, z_origin=model.z_origin,
        name="$(model.name)_modified_$(param_name)")
end
