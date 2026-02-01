# ==============================================================================
# simulation/api.jl
#
# High-level API for simplified simulation workflow
# ==============================================================================



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
using ElasticWave2D

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

    # Use enhanced boundary configuration if available, otherwise use legacy
    if config.boundary_config !== nothing
        # Use enhanced boundary initialization based on boundary type
        if config.boundary_config.top_boundary == :vacuum
            # For vacuum boundary, we need to handle it specially
            # For now, we'll use the standard initialization but note that vacuum
            # boundaries are handled through material parameters
            medium = init_medium_with_boundaries(model.vp, model.vs, model.rho,
                model.dx, model.dz, config, be)
        else
            # Use standard initialization for other boundary types
            medium = init_medium_with_boundaries(model.vp, model.vs, model.rho,
                model.dx, model.dz, config, be)
        end
    else
        # Use legacy initialization
        medium = init_medium(model, config.nbc, config.fd_order, be; free_surface=config.free_surface)
    end

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

    # Use enhanced boundary conditions if available
    if config.boundary_config !== nothing
        # Use enhanced time loop with boundary configuration
        run_time_loop_with_boundaries!(be, wavefield, medium, habc, fd_coeffs, src, rec, params, config.boundary_config;
            progress=config.show_progress,
            on_step=recorder === nothing ? nothing : (W, info) -> begin
                ElasticWave2D.record!(recorder.recorder, W, info.k, dt)
                return true
            end
        )
    else
        # Use legacy time loop
        run_time_loop!(be, wavefield, medium, habc, fd_coeffs, src, rec, params;
            progress=config.show_progress,
            on_step=recorder === nothing ? nothing : (W, info) -> begin
                ElasticWave2D.record!(recorder.recorder, W, info.k, dt)
                return true
            end
        )
    end

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
using ElasticWave2D

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
# Enhanced boundary handling functions

"""
    apply_boundary_conditions!(backend, wavefield, medium, habc, params)

Apply appropriate boundary conditions based on the boundary configuration.
"""
function apply_boundary_conditions!(be::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig, params::SimParams)
    # Apply HABC to boundary strips
    backup_boundary!(be, W, H, M)
    apply_habc_velocity!(be, W, H, M)
    apply_habc_stress!(be, W, H, M)

    # Apply free surface condition if enabled
    if M.is_free_surface
        apply_free_surface!(be, W, M)
    end
end

"""
    init_medium_with_boundaries(vp, vs, rho, dx, dz, config, backend)

Initialize medium with appropriate boundary settings based on configuration.
"""
function init_medium_with_boundaries(vp::Matrix, vs::Matrix, rho::Matrix,
    dx::Real, dz::Real, config::SimulationConfig, backend::AbstractBackend)

    # Determine free surface setting from boundary config or legacy setting
    free_surface = if config.boundary_config !== nothing
        config.boundary_config.top_boundary in [:image, :vacuum]
    else
        # Use legacy free_surface parameter
        config.free_surface
    end

    # Initialize medium with determined free surface setting
    return init_medium(vp, vs, rho, dx, dz, config.nbc, config.fd_order, backend;
        free_surface=free_surface)
end



# ==============================================================================
# Internal Helper Functions
# ==============================================================================



#function _plot_irregular_setup(model::VelocityModel, z_surface::Vector{Float32},
#    src_x::Real, src_depth::Real,
#    rec_x::Vector{Float32}, rec_depth::Vector{Float32},
#    ibm_method::Symbol, output::String)
#
#    fig = CairoMakie.Figure(size=(1000, 700))
#
#    x_axis = range(0, (model.nx - 1) * model.dx, length=model.nx)
#    z_axis = range(0, (model.nz - 1) * model.dz, length=model.nz)
#
#    ax = CairoMakie.Axis(fig[1, 1],
#        xlabel="X (m)", ylabel="Z (m)",
#        title="Model with Irregular Surface (IBM: $ibm_method)",
#        aspect=CairoMakie.DataAspect())
#
#    hm = CairoMakie.heatmap!(ax, x_axis, z_axis, model.vp', colormap=:viridis)
#
#    surf_x = Float32.((0:length(z_surface)-1) .* model.dx)
#    CairoMakie.lines!(ax, surf_x, z_surface, color=:white, linewidth=3, label="Free surface")
#
#    src_idx = clamp(round(Int, src_x / model.dx) + 1, 1, length(z_surface))
#    src_z = z_surface[src_idx] + src_depth
#    CairoMakie.scatter!(ax, [Float32(src_x)], [Float32(src_z)],
#        marker=:star5, markersize=20, color=:red, label="Source")
#
#    rec_z = Float32[]
#    for i in 1:length(rec_x)
#        idx = clamp(round(Int, rec_x[i] / model.dx) + 1, 1, length(z_surface))
#        push!(rec_z, z_surface[idx] + rec_depth[i])
#    end
#    step = max(1, length(rec_x) ÷ 20)
#    CairoMakie.scatter!(ax, rec_x[1:step:end], rec_z[1:step:end],
#        marker=:dtriangle, markersize=8, color=:cyan, label="Receivers")
#
#    ax.yreversed = true
#    CairoMakie.Colorbar(fig[1, 2], hm, label="Vp (m/s)")
#    CairoMakie.axislegend(ax, position=:rb)
#    CairoMakie.save(output, fig)
#end

"""
    seismic_survey(model, sources, receivers; 
                   surface_method=:image,
                   vacuum_layers=10,
                   source_depth_margin=80.0,
                   config=SimulationConfig())

Simplified API for seismic survey simulation with flexible surface handling.

# Arguments
- `model`: Velocity model
- `sources`: Source positions as (x, z) tuple or array of tuples
- `receivers`: Receiver positions as (rec_x, rec_z) tuple

# Keyword Arguments
- `surface_method::Symbol`: How to handle the free surface
  - `:image` - Image Method free surface boundary condition (default)
  - `:absorbing` - Absorbing boundary (no surface waves)
  - `:vacuum` - Vacuum formulation (adds vacuum layers at top)
- `vacuum_layers::Int`: Number of vacuum layers when using `:vacuum` method (default: 10)
- `source_depth_margin::Float32`: Minimum distance from sources to top boundary in meters
  (only used for `:absorbing` method, default: 80.0m)
- `config::SimulationConfig`: Simulation configuration
- `video_config::Union{VideoConfig,Nothing}`: Video recording configuration (optional)

# Examples
```julia
# Method 1: Image Method free surface (classic approach)
result = seismic_survey(model, (src_x, src_z), (rec_x, rec_z);
    surface_method = :image
)

# Method 2: Vacuum formulation (recommended for consistency)
result = seismic_survey(model, (src_x, src_z), (rec_x, rec_z);
    surface_method = :vacuum,
    vacuum_layers = 10
)

# Method 3: No surface waves (absorbing top boundary)
result = seismic_survey(model, (src_x, src_z), (rec_x, rec_z);
    surface_method = :absorbing
)

# With video recording
video_config = VideoConfig(fields=[:vz], skip=20, fps=30)
result = seismic_survey(model, (src_x, src_z), (rec_x, rec_z);
    surface_method = :vacuum,
    config = config,
    video_config = video_config
)
```

See also: [`simulate!`](@ref), [`SimulationConfig`](@ref)
"""
function seismic_survey(model, sources, receivers;
    surface_method::Symbol=:image,
    vacuum_layers::Int=10,
    source_depth_margin=80.0,
    config=SimulationConfig(),
    video_config=nothing
)
    # Validate surface_method
    if !(surface_method in [:image, :absorbing, :vacuum])
        error("surface_method must be :image, :absorbing, or :vacuum")
    end

    # Parse sources
    if sources isa Tuple
        src_x, src_z = Float32(sources[1]), Float32(sources[2])
    elseif sources isa Vector
        src_x, src_z = Float32(sources[1][1]), Float32(sources[1][2])
    else
        error("Sources format not supported. Expected (x, z) tuple or array of tuples")
    end

    # Parse receivers
    rec_x = Float32.(receivers[1])
    rec_z = Float32.(receivers[2])

    # ===========================
    # Handle different surface methods
    # ===========================

    if surface_method == :vacuum
        # Add vacuum layers at top of model
        println("Using vacuum formulation for free surface ($(vacuum_layers) layers)")

        # Create expanded model with vacuum at top
        nz_new = model.nz + vacuum_layers

        vp_new = zeros(Float32, nz_new, model.nx)
        vs_new = zeros(Float32, nz_new, model.nx)
        rho_new = zeros(Float32, nz_new, model.nx)

        # Top vacuum_layers rows are vacuum (already zeros)
        # Copy original model below
        vp_new[vacuum_layers+1:end, :] = model.vp
        vs_new[vacuum_layers+1:end, :] = model.vs
        rho_new[vacuum_layers+1:end, :] = model.rho

        expanded_model = VelocityModel(
            vp_new, vs_new, rho_new,
            model.dx, model.dz;
            name=model.name * "_vacuum"
        )

        # Shift source and receiver z-coordinates
        vacuum_thickness = vacuum_layers * model.dz
        adjusted_src_z = src_z + vacuum_thickness
        adjusted_rec_z = rec_z .+ vacuum_thickness

        # Create config with free_surface disabled
        new_config = SimulationConfig(
            nbc=config.nbc,
            fd_order=config.fd_order,
            dt=config.dt,
            nt=config.nt,
            cfl=config.cfl,
            f0=config.f0,
            free_surface=false,  # Disabled! Using vacuum
            output_dir=config.output_dir,
            save_gather=config.save_gather,
            show_progress=config.show_progress,
            plot_gather=config.plot_gather,
            boundary_config=config.boundary_config
        )

        return simulate!(expanded_model, src_x, adjusted_src_z, rec_x, adjusted_rec_z;
            config=new_config, video_config=video_config)

    elseif surface_method == :absorbing
        # Expand model to keep sources away from absorbing boundary
        println("Using absorbing top boundary (no surface waves)")

        min_source_depth = src_z
        extra_layers = ceil(Int, source_depth_margin / model.dz)

        if extra_layers > 0
            # Expand model by replicating top row (no impedance contrast)
            top_vp_row = model.vp[1:1, :]
            top_vs_row = model.vs[1:1, :]
            top_rho_row = model.rho[1:1, :]

            expanded_vp = vcat(repeat(top_vp_row, extra_layers), model.vp)
            expanded_vs = vcat(repeat(top_vs_row, extra_layers), model.vs)
            expanded_rho = vcat(repeat(top_rho_row, extra_layers), model.rho)

            expanded_model = VelocityModel(
                expanded_vp, expanded_vs, expanded_rho,
                model.dx, model.dz;
                name=model.name * "_expanded"
            )

            # Shift coordinates
            shift = Float32(source_depth_margin)
            adjusted_src_z = src_z + shift
            adjusted_rec_z = rec_z .+ shift
        else
            expanded_model = model
            adjusted_src_z = src_z
            adjusted_rec_z = rec_z
        end

        new_config = SimulationConfig(
            nbc=config.nbc,
            fd_order=config.fd_order,
            dt=config.dt,
            nt=config.nt,
            cfl=config.cfl,
            f0=config.f0,
            free_surface=false,  # Absorbing
            output_dir=config.output_dir,
            save_gather=config.save_gather,
            show_progress=config.show_progress,
            plot_gather=config.plot_gather,
            boundary_config=config.boundary_config
        )

        return simulate!(expanded_model, src_x, adjusted_src_z, rec_x, adjusted_rec_z;
            config=new_config, video_config=video_config)

    elseif surface_method == :image
        # Use explicit free surface boundary condition (Image Method)
        println("Using explicit free surface boundary condition (Image Method)")

        new_config = SimulationConfig(
            nbc=config.nbc,
            fd_order=config.fd_order,
            dt=config.dt,
            nt=config.nt,
            cfl=config.cfl,
            f0=config.f0,
            free_surface=true,
            output_dir=config.output_dir,
            save_gather=config.save_gather,
            show_progress=config.show_progress,
            plot_gather=config.plot_gather,
            boundary_config=config.boundary_config
        )

        return simulate!(model, src_x, src_z, rec_x, rec_z;
            config=new_config, video_config=video_config)
    end
end