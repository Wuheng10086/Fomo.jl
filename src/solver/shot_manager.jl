# ==============================================================================
# simulation/shots.jl
#
# Shot management - optimized for performance
# ==============================================================================

# ShotResult is defined in core/structures.jl

# ==============================================================================
# Single Shot Execution
# ==============================================================================

"""
    run_shot!(backend, W, M, H, a, src, rec, params; kwargs) -> ShotResult

Run a single shot and return the result.

This function:
1. Resets the wavefield
2. Clears receiver data
3. Runs the time loop
4. Returns gather data (copied to CPU if on GPU)

# Keyword Arguments
- `shot_id`: Shot identifier (default: 1)
- `progress`: Show progress bar (default: false for performance)
- `on_step`: Callback function for each time step (e.g., VideoRecorder)
"""
function run_shot!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
                   a, src::Source, rec::Receivers, params::SimParams;
                   shot_id::Int=1, progress::Bool=false, on_step=nothing)
    
    # Reset wavefield
    reset!(backend, W)
    
    # Clear receiver data
    fill!(rec.data, 0.0f0)
    
    # Run simulation
    run_time_loop!(backend, W, M, H, a, src, rec, params; 
                   progress=progress, on_step=on_step)
    
    # Get gather (copy to CPU if on GPU)
    gather = _get_gather(backend, rec)
    
    # Get receiver indices (ensure CPU arrays)
    rec_i = _to_cpu_vec(rec.i)
    rec_j = _to_cpu_vec(rec.j)
    
    return ShotResult(gather, shot_id, src.i, src.j, rec_i, rec_j)
end

# Helper to get gather data (always returns CPU array)
_get_gather(::CPUBackend, rec::Receivers) = copy(rec.data)
_get_gather(::CUDABackend, rec::Receivers) = Array(rec.data)

# Helper to ensure CPU vector
_to_cpu_vec(v::Vector) = Vector{Int}(v)
_to_cpu_vec(v::CuVector) = Vector{Int}(Array(v))
_to_cpu_vec(v) = Vector{Int}(collect(v))

# ==============================================================================
# Multi-Shot Configuration
# ==============================================================================

"""
    ShotConfig

Configuration for a single shot.
"""
struct ShotConfig
    source_x::Float32
    source_z::Float32
    shot_id::Int
end

"""
    MultiShotConfig

Configuration for multiple shots.
"""
struct MultiShotConfig
    shots::Vector{ShotConfig}
    wavelet::Vector{Float32}
    source_type::Symbol         # :pressure, :stress_txx, :stress_tzz, :stress_txz
end

"""
    MultiShotConfig(x_positions, z_positions, wavelet; source_type=:pressure)

Create multi-shot configuration from position arrays.
"""
function MultiShotConfig(x_positions::Vector{<:Real}, z_positions::Vector{<:Real}, 
                         wavelet::Vector{Float32}; source_type::Symbol=:pressure)
    n = length(x_positions)
    @assert length(z_positions) == n "x and z must have same length"
    
    shots = [ShotConfig(Float32(x_positions[i]), Float32(z_positions[i]), i) 
             for i in 1:n]
    
    return MultiShotConfig(shots, wavelet, source_type)
end

# ==============================================================================
# Multi-Shot Execution (OPTIMIZED - Memory Reuse)
# ==============================================================================

"""
    run_shots!(backend, W, M, H, a, rec_template, shot_config, params; kwargs) -> Vector{ShotResult}

Run multiple shots with MEMORY REUSE for maximum performance.

# Arguments
- `backend`: Compute backend
- `W`: Wavefield (will be reset between shots)
- `M`: Medium
- `H`: HABC configuration
- `a`: FD coefficients
- `rec_template`: Receivers template (indices only, data will be allocated once)
- `shot_config`: MultiShotConfig with shot positions
- `params`: Simulation parameters

# Keyword Arguments
- `on_shot_complete`: Callback `f(result::ShotResult)` called after each shot
- `on_step`: Callback for each time step (e.g., VideoRecorder)
- `progress`: Show progress for each shot (default: false)
- `verbose`: Print shot progress info (default: true)
"""
function run_shots!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
                    a, rec_template::Receivers, shot_config::MultiShotConfig, params::SimParams;
                    on_shot_complete=nothing, on_step=nothing, progress::Bool=false,
                    verbose::Bool=true)
    
    n_shots = length(shot_config.shots)
    results = Vector{ShotResult}(undef, n_shots)
    
    # Prepare wavelet on device ONCE
    wavelet_device = to_device(shot_config.wavelet, backend)
    
    # PRE-ALLOCATE receiver data buffer ONCE (will be reused)
    rec = _create_receivers(backend, rec_template, params.nt)
    
    if verbose
        @info "Running shots" n_shots=n_shots backend=typeof(backend)
    end
    
    t_start = time()
    
    for (i, shot) in enumerate(shot_config.shots)
        # Create source for this shot (lightweight - only indices change)
        src = _create_source(backend, M, shot, wavelet_device, shot_config.source_type)
        
        # REUSE receiver buffer - just clear data (no reallocation)
        fill!(rec.data, 0.0f0)
        
        # Reset wavefield
        reset!(backend, W)
        
        # Run time loop
        run_time_loop!(backend, W, M, H, a, src, rec, params;
                      progress=progress, on_step=on_step)
        
        # Get result
        gather = _get_gather(backend, rec)
        rec_i = _to_cpu_vec(rec.i)
        rec_j = _to_cpu_vec(rec.j)
        result = ShotResult(gather, shot.shot_id, src.i, src.j, rec_i, rec_j)
        
        results[i] = result
        
        # Callback
        if on_shot_complete !== nothing
            on_shot_complete(result)
        end
        
        # Minimal progress output (every 10% or at end)
        if verbose && (i % max(1, n_shots ÷ 10) == 0 || i == n_shots)
            elapsed = time() - t_start
            avg_time = elapsed / i
            eta = avg_time * (n_shots - i)
            @info "Shot progress" completed="$i/$n_shots" elapsed_s=round(elapsed, digits=1) eta_s=round(eta, digits=1)
        end
    end
    
    if verbose
        total_time = time() - t_start
        @info "All shots completed" total_s=round(total_time, digits=2) per_shot_s=round(total_time/n_shots, digits=3)
    end
    
    return results
end

# ==============================================================================
# FAST Multi-Shot (Minimal Overhead)
# ==============================================================================

"""
    run_shots_fast!(backend, W, M, H, a, rec_template, shot_config, params) -> Vector{Matrix{Float32}}

Ultra-fast multi-shot execution with minimal overhead.
Returns only gather matrices (no ShotResult wrapper).
No logging, no callbacks - pure computation.
"""
function run_shots_fast!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
                         a, rec_template::Receivers, shot_config::MultiShotConfig, params::SimParams)
    
    n_shots = length(shot_config.shots)
    gathers = Vector{Matrix{Float32}}(undef, n_shots)
    
    # Prepare wavelet on device ONCE
    wavelet_device = to_device(shot_config.wavelet, backend)
    
    # PRE-ALLOCATE receiver data buffer ONCE
    rec = _create_receivers(backend, rec_template, params.nt)
    
    for (i, shot) in enumerate(shot_config.shots)
        # Reset wavefield
        reset!(backend, W)
        
        # Create source
        src = _create_source(backend, M, shot, wavelet_device, shot_config.source_type)
        
        # Clear receiver data
        fill!(rec.data, 0.0f0)
        
        # Run simulation (no progress, no callbacks)
        run_time_loop!(backend, W, M, H, a, src, rec, params; 
                       progress=false, on_step=nothing)
        
        # Copy gather to CPU
        gathers[i] = _get_gather(backend, rec)
    end
    
    return gathers
end

# ==============================================================================
# Helper Functions
# ==============================================================================

"""
Convert physical coordinates to grid index.
"""
function _coord_to_index(x::Float32, dx::Float32, pad::Int)
    return round(Int32, x / dx) + pad + 1
end

"""
Create source on device from shot config.
"""
function _create_source(::CPUBackend, M::Medium, shot::ShotConfig, wavelet::Vector{Float32}, source_type::Symbol)
    i = _coord_to_index(shot.source_x, M.dx, M.pad)
    j = _coord_to_index(shot.source_z, M.dz, M.pad)
    if source_type == :pressure
        return Source(i, j, wavelet)
    elseif source_type == :stress_txx
        return StressSource(i, j, wavelet, :txx)
    elseif source_type == :stress_tzz
        return StressSource(i, j, wavelet, :tzz)
    elseif source_type == :stress_txz
        return StressSource(i, j, wavelet, :txz)
    else
        error("Unknown source_type: $source_type. Use :pressure or :stress_txx/:stress_tzz/:stress_txz")
    end
end

function _create_source(::CUDABackend, M::Medium, shot::ShotConfig, wavelet::CuVector{Float32}, source_type::Symbol)
    i = _coord_to_index(shot.source_x, M.dx, M.pad)
    j = _coord_to_index(shot.source_z, M.dz, M.pad)
    if source_type == :pressure
        return Source(Int32(i), Int32(j), wavelet)
    elseif source_type == :stress_txx
        return StressSource(Int32(i), Int32(j), wavelet, :txx)
    elseif source_type == :stress_tzz
        return StressSource(Int32(i), Int32(j), wavelet, :tzz)
    elseif source_type == :stress_txz
        return StressSource(Int32(i), Int32(j), wavelet, :txz)
    else
        error("Unknown source_type: $source_type. Use :pressure or :stress_txx/:stress_tzz/:stress_txz")
    end
end

"""
Create receivers on device.
"""
function _create_receivers(::CPUBackend, template::Receivers, nt::Int)
    data = zeros(Float32, nt, length(template.i))
    return Receivers(copy(template.i), copy(template.j), data, template.type)
end

function _create_receivers(::CUDABackend, template::Receivers, nt::Int)
    if !CUDA_AVAILABLE[]
        error("CUDA not functional (no GPU available)")
    end
    # Convert indices to GPU
    i_cpu = template.i isa Array ? template.i : Array(template.i)
    j_cpu = template.j isa Array ? template.j : Array(template.j)
    i_gpu = CuArray(Int32.(i_cpu))
    j_gpu = CuArray(Int32.(j_cpu))
    data = CUDA.zeros(Float32, nt, length(i_gpu))
    return Receivers(i_gpu, j_gpu, data, template.type)
end
