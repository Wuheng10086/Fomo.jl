# ==============================================================================
# simulation/time_stepper.jl
#
# Core time stepping logic - SINGLE IMPLEMENTATION for both CPU and GPU!
# The backend parameter determines which kernels are called.
#
# Visualization is handled through callbacks - core logic stays pure.
# ==============================================================================

# ProgressMeter is already imported in main module

"""
    TimeStepInfo

Information passed to callbacks at each time step.
"""
struct TimeStepInfo
    k::Int              # Current time step
    t::Float32          # Current time
    nt::Int             # Total time steps
end

"""
    time_step!(backend, W, M, H, a, src, rec, k, params)

Execute a single time step. This is the core simulation logic.

**This function is the same for CPU and GPU** - the `backend` parameter
dispatches to the appropriate kernel implementations.

# Time Step Sequence
1. Backup boundary values (for HABC temporal derivative)
2. Inject source wavelet
3. Update velocity + apply HABC
4. Update stress + apply HABC  
5. Apply free surface condition
6. Record at receivers
"""
function time_step!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
    a, src::AbstractSource, rec::Receivers, k::Int, params::SimParams)

    # 1. Backup boundary (for HABC extrapolation)
    backup_boundary!(backend, W, H, M)

    # 2. Source injection
    inject_source!(backend, W, src, k, params.dt)

    # 3. Velocity update + HABC
    update_velocity!(backend, W, M, a, params)
    apply_habc_velocity!(backend, W, H, M)
    apply_free_surface_velocity!(backend, W, M)

    # 4. Stress update + HABC
    update_stress!(backend, W, M, a, params)
    apply_habc_stress!(backend, W, H, M)
    apply_free_surface_stress!(backend, W, M)

    # 6. Record receivers
    record_receivers!(backend, W, rec, k)

    return nothing
end

"""
    time_step_with_boundaries!(backend, W, M, H, a, src, rec, k, params, boundary_config)

Execute a single time step with enhanced boundary configuration handling.
"""
function time_step_with_boundaries!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
    a, src::AbstractSource, rec::Receivers, k::Int, params::SimParams,
    boundary_config::SolverBoundaryConfig)

    # 1. Backup boundary (for HABC extrapolation)
    backup_boundary!(backend, W, H, M)

    # 2. Source injection
    inject_source!(backend, W, src, k, params.dt)

    # 3. Velocity update + HABC
    update_velocity!(backend, W, M, a, params)

    # Apply velocity boundary conditions based on configuration
    apply_boundary_conditions_velocity!(backend, W, M, H, boundary_config)

    # 4. Stress update + HABC
    update_stress!(backend, W, M, a, params)

    # Apply stress boundary conditions based on configuration
    apply_boundary_conditions_stress!(backend, W, M, H, boundary_config)

    # 5. Record receivers
    record_receivers!(backend, W, rec, k)

    return nothing
end

"""
    apply_boundary_conditions_velocity!(backend, W, M, H, boundary_config)

Apply appropriate boundary conditions to velocity fields based on configuration.
"""
function apply_boundary_conditions_velocity!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
    boundary_config::SolverBoundaryConfig)
    # Apply HABC to boundary strips (except where overridden by other conditions)
    apply_habc_velocity!(backend, W, H, M)
    if boundary_config.top_boundary == :image
        apply_free_surface_velocity!(backend, W, M)
    end
end

"""
    apply_boundary_conditions_stress!(backend, W, M, H, boundary_config)

Apply appropriate boundary conditions to stress fields based on configuration.
"""
function apply_boundary_conditions_stress!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
    boundary_config::SolverBoundaryConfig)
    # Apply HABC to boundary strips (except where overridden by other conditions)
    apply_habc_stress!(backend, W, H, M)

    # Apply top boundary condition based on configuration
    if boundary_config.top_boundary == :image
        apply_free_surface_stress!(backend, W, M)
    elseif boundary_config.top_boundary == :vacuum
        # Vacuum boundary condition is handled implicitly in the medium parameters
        # No explicit stress condition needed beyond HABC
    end

    # Apply other boundary conditions as needed
    # (Currently only top boundary is handled differently)
end

"""
    run_time_loop!(backend, W, M, H, a, src, rec, params; 
                   progress=true, on_step=nothing)

Run the complete time stepping loop.

# Arguments
- `backend`: CPU_BACKEND or CUDA_BACKEND
- `W`: Wavefield
- `M`: Medium  
- `H`: HABC configuration
- `a`: FD coefficients
- `src`: Source
- `rec`: Receivers
- `params`: Simulation parameters
- `progress`: Show progress bar
- `on_step`: Callback function `f(W, info::TimeStepInfo)` called after each step
             Return `false` to stop simulation early.

# Callback Example
```julia
# Save snapshot every 100 steps
function my_callback(W, info)
    if info.k % 100 == 0
        save_snapshot(W, "snap_\$(info.k).bin")
    end
    return true  # continue
end

run_time_loop!(backend, W, M, H, a, src, rec, params; on_step=my_callback)
```
"""
function run_time_loop!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
    a, src::AbstractSource, rec::Receivers, params::SimParams;
    progress::Bool=true, on_step=nothing)

    nt = params.nt
    dt = params.dt

    if progress
        p = Progress(nt; desc="Simulating: ", color=:cyan)
    end

    for k in 1:nt
        time_step!(backend, W, M, H, a, src, rec, k, params)

        # Callback
        if on_step !== nothing
            info = TimeStepInfo(k, k * dt, nt)
            cont = on_step(W, info)
            if cont === false
                @info "Simulation stopped early at step $k"
                break
            end
        end

        if progress
            next!(p)
        end
    end

    synchronize(backend)
    return nothing
end

"""
    run_time_loop_with_boundaries!(backend, W, M, H, a, src, rec, params, boundary_config;
                                   progress=true, on_step=nothing)

Run the complete time stepping loop with enhanced boundary configuration.

# Arguments
- `backend`: CPU_BACKEND or CUDA_BACKEND
- `W`: Wavefield
- `M`: Medium  
- `H`: HABC configuration
- `a`: FD coefficients
- `src`: Source
- `rec`: Receivers
- `params`: Simulation parameters
- `boundary_config`: Enhanced boundary configuration
- `progress`: Show progress bar
- `on_step`: Callback function `f(W, info::TimeStepInfo)` called after each step
             Return `false` to stop simulation early.
"""
function run_time_loop_with_boundaries!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
    a, src::AbstractSource, rec::Receivers, params::SimParams, boundary_config::SolverBoundaryConfig;
    progress::Bool=true, on_step=nothing)

    nt = params.nt
    dt = params.dt

    if progress
        p = Progress(nt; desc="Simulating: ", color=:cyan)
    end

    for k in 1:nt
        time_step_with_boundaries!(backend, W, M, H, a, src, rec, k, params, boundary_config)

        # Callback
        if on_step !== nothing
            info = TimeStepInfo(k, k * dt, nt)
            cont = on_step(W, info)
            if cont === false
                @info "Simulation stopped early at step $k"
                break
            end
        end

        if progress
            next!(p)
        end
    end

    synchronize(backend)
    return nothing
end
