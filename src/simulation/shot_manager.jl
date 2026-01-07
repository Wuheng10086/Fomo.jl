# ==============================================================================
# simulation/shot_manager.jl
#
# Shot management - clean separation of single/multi-shot logic
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
- `progress`: Show progress bar (default: true)
- `on_step`: Callback function for each time step (e.g., VideoRecorder)
"""
function run_shot!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
                   a, src::Source, rec::Receivers, params::SimParams;
                   shot_id::Int=1, progress::Bool=true, on_step=nothing)
    
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
    source_type::Symbol         # :pressure, :force_x, :force_z
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
# Multi-Shot Execution
# ==============================================================================

"""
    run_shots!(backend, W, M, H, a, rec, shot_config, params; kwargs) -> Vector{ShotResult}

Run multiple shots and return all results.

# Arguments
- `backend`: Compute backend
- `W`: Wavefield (will be reset between shots)
- `M`: Medium
- `H`: HABC configuration
- `a`: FD coefficients
- `rec`: Receivers (indices only, data will be reset)
- `shot_config`: MultiShotConfig with shot positions
- `params`: Simulation parameters

# Keyword Arguments
- `on_shot_complete`: Callback `f(result::ShotResult)` called after each shot
- `on_step`: Callback for each time step (e.g., VideoRecorder)
- `progress`: Show progress for each shot
"""
function run_shots!(backend::AbstractBackend, W::Wavefield, M::Medium, H::HABCConfig,
                    a, rec_template::Receivers, shot_config::MultiShotConfig, params::SimParams;
                    on_shot_complete=nothing, on_step=nothing, progress::Bool=true)
    
    n_shots = length(shot_config.shots)
    results = Vector{ShotResult}(undef, n_shots)
    
    # Prepare wavelet on device
    wavelet_device = to_device(shot_config.wavelet, backend)
    
    @info "Running $n_shots shots on $(typeof(backend))"
    
    for (i, shot) in enumerate(shot_config.shots)
        @info "Shot $i/$n_shots at ($(shot.source_x), $(shot.source_z))"
        
        # Create source for this shot
        src = _create_source(backend, M, shot, wavelet_device)
        
        # Create fresh receiver data buffer
        rec = _create_receivers(backend, rec_template, params.nt)
        
        # Run shot with optional step callback
        result = run_shot!(backend, W, M, H, a, src, rec, params;
                          shot_id=shot.shot_id, progress=progress, on_step=on_step)
        
        results[i] = result
        
        # Callback
        if on_shot_complete !== nothing
            on_shot_complete(result)
        end
    end
    
    @info "All shots completed"
    return results
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
function _create_source(::CPUBackend, M::Medium, shot::ShotConfig, wavelet::Vector{Float32})
    i = _coord_to_index(shot.source_x, M.dx, M.pad)
    j = _coord_to_index(shot.source_z, M.dz, M.pad)
    return Source(i, j, wavelet)
end

function _create_source(::CUDABackend, M::Medium, shot::ShotConfig, wavelet::CuVector{Float32})
    i = _coord_to_index(shot.source_x, M.dx, M.pad)
    j = _coord_to_index(shot.source_z, M.dz, M.pad)
    return Source(Int32(i), Int32(j), wavelet)
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
