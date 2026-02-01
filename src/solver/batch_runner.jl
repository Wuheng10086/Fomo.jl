# ==============================================================================
# simulation/batch.jl
#
# High-performance batch simulation API
# Optimized for running many shots with minimal overhead
# ==============================================================================

# ==============================================================================
# BatchSimulator - Pre-initialized simulator for maximum performance
# ==============================================================================

"""
    BatchSimulator

Pre-initialized simulator for high-performance batch shot execution.
All heavy initialization is done once, then shots can be run rapidly.

# Example
```julia
# Initialize once (includes JIT compilation)
simulator = BatchSimulator(model, rec_x, rec_z; config=config)

# Run many shots rapidly
for (sx, sz) in shot_positions
    gather = simulate_shot!(simulator, sx, sz)
    # process gather...
end

# Or run all shots at once
gathers = simulate_shots!(simulator, src_x_vec, src_z_vec)
```
"""
mutable struct BatchSimulator
    # Backend
    backend::AbstractBackend

    # Pre-initialized structures (allocated once, reused)
    medium::Medium
    habc::HABCConfig
    fd_coeffs::Any  # CuArray or Array
    wavefield::Wavefield
    receivers::Receivers

    # Cached wavelet on device
    wavelet_device::Any

    # Parameters
    params::SimParams
    f0::Float32
    dx::Float32
    dz::Float32
    pad::Int

    # State
    is_initialized::Bool
end

"""
    BatchSimulator(model, rec_x, rec_z; config, be=backend(:cuda))

Create a pre-initialized batch simulator for high-performance multi-shot simulation.

# Arguments
- `model::VelocityModel`: Velocity model
- `rec_x::Vector{<:Real}`: Receiver x positions in meters
- `rec_z::Vector{<:Real}`: Receiver z positions in meters

# Keyword Arguments
- `config::SimulationConfig`: Simulation configuration
- `be::AbstractBackend`: Backend (default: CUDA if available)

# Returns
- `BatchSimulator`: Ready-to-use simulator

# Example
```julia
config = SimulationConfig(nt=4000, f0=15.0f0, nbc=50)
sim = BatchSimulator(model, rec_x, rec_z; config=config)

# Now run shots efficiently
gather1 = simulate_shot!(sim, 5000.0f0, 10.0f0)
gather2 = simulate_shot!(sim, 6000.0f0, 10.0f0)
```
"""
function BatchSimulator(
    model::VelocityModel,
    rec_x::Vector{<:Real},
    rec_z::Vector{<:Real};
    config::SimulationConfig=SimulationConfig(),
    be::AbstractBackend=is_cuda_available() ? backend(:cuda) : backend(:cpu)
)
    # Compute dt if not specified
    vp_max = maximum(model.vp)
    dt = if config.dt === nothing
        Float32(config.cfl * min(model.dx, model.dz) / vp_max)
    else
        config.dt
    end

    # Create SimParams with correct signature: (dt, nt, dx, dz, fd_order)
    params = SimParams(dt, config.nt, model.dx, model.dz, config.fd_order)

    # Initialize medium (ONCE)
    medium = init_medium(model, config.nbc, config.fd_order, be;
        free_surface=config.free_surface)

    # Initialize HABC (ONCE)
    habc = init_habc(medium.nx, medium.nz, config.nbc, medium.pad, dt,
        model.dx, model.dz, vp_max, be)

    # FD coefficients (ONCE)
    fd_coeffs = to_device(get_fd_coefficients(config.fd_order), be)

    # Wavefield (ONCE)
    wavefield = Wavefield(medium.nx, medium.nz, be)

    # Receivers (ONCE) - data buffer will be reused
    n_rec = length(rec_x)
    rec_i = [round(Int, x / model.dx) + medium.pad + 1 for x in rec_x]
    rec_j = [round(Int, z / model.dz) + medium.pad + 1 for z in rec_z]
    receivers = Receivers(
        to_device(rec_i, be),
        to_device(rec_j, be),
        to_device(zeros(Float32, config.nt, n_rec), be),
        :vz
    )

    # Pre-generate and cache wavelet on device
    wavelet = ricker_wavelet(config.f0, dt, config.nt)
    wavelet_device = to_device(wavelet, be)

    return BatchSimulator(
        be, medium, habc, fd_coeffs, wavefield, receivers,
        wavelet_device, params, config.f0, model.dx, model.dz, medium.pad, true
    )
end

# ==============================================================================
# Single Shot Execution
# ==============================================================================

"""
    simulate_shot!(sim::BatchSimulator, src_x, src_z; wavelet=nothing) -> Matrix{Float32}

Run a single shot using pre-initialized simulator. Returns gather matrix.

This is the fastest way to run individual shots as all initialization
has been done in the BatchSimulator constructor.

# Arguments
- `sim`: Pre-initialized BatchSimulator
- `src_x`: Source x position in meters
- `src_z`: Source z position in meters

# Keyword Arguments
- `wavelet`: Optional custom wavelet. If nothing, uses pre-cached Ricker wavelet.

# Returns
- `Matrix{Float32}`: Gather data of shape (nt, n_receivers)

# Example
```julia
sim = BatchSimulator(model, rec_x, rec_z; config=config)
gather = simulate_shot!(sim, 5000.0f0, 10.0f0)
```
"""
function simulate_shot!(sim::BatchSimulator, src_x::Real, src_z::Real;
    wavelet::Union{Vector{Float32},Nothing}=nothing)

    @assert sim.is_initialized "Simulator not initialized"

    # Use custom wavelet or cached one
    wavelet_dev = if wavelet === nothing
        sim.wavelet_device
    else
        to_device(wavelet, sim.backend)
    end

    # Create source (lightweight operation)
    src_i = round(Int, src_x / sim.dx) + sim.pad + 1
    src_j = round(Int, src_z / sim.dz) + sim.pad + 1
    source = Source(src_i, src_j, wavelet_dev)

    # Reset wavefield (in-place, fast)
    reset!(sim.backend, sim.wavefield)

    # Clear receiver data (in-place, fast)
    fill!(sim.receivers.data, 0.0f0)

    # Run time loop (no progress bar for speed)
    run_time_loop!(sim.backend, sim.wavefield, sim.medium, sim.habc,
        sim.fd_coeffs, source, sim.receivers, sim.params;
        progress=false, on_step=nothing)

    # Return gather (copy to CPU)
    return _get_gather_from_batch(sim.backend, sim.receivers)
end

_get_gather_from_batch(::CPUBackend, rec::Receivers) = copy(rec.data)
_get_gather_from_batch(::CUDABackend, rec::Receivers) = Array(rec.data)

# ==============================================================================
# Multi-Shot Execution
# ==============================================================================

"""
    simulate_shots!(sim::BatchSimulator, src_x, src_z; verbose=false) -> Vector{Matrix{Float32}}

Run multiple shots using pre-initialized simulator.

This is the most efficient way to run many shots as all initialization
is done once and memory is reused between shots.

# Arguments
- `sim`: Pre-initialized BatchSimulator
- `src_x`: Vector of source x positions in meters
- `src_z`: Vector of source z positions in meters

# Keyword Arguments
- `wavelet`: Optional custom wavelet (applied to all shots)
- `verbose`: Print progress (default: false for max speed)
- `on_shot_complete`: Optional callback function(gather, shot_id) called after each shot

# Returns
- `Vector{Matrix{Float32}}`: Vector of gather matrices

# Example
```julia
sim = BatchSimulator(model, rec_x, rec_z; config=config)

# Run 100 shots
src_x = Float32.(range(1000, 15000, length=100))
src_z = fill(10.0f0, 100)
gathers = simulate_shots!(sim, src_x, src_z; verbose=true)
```
"""
function simulate_shots!(sim::BatchSimulator,
    src_x::Vector{<:Real},
    src_z::Vector{<:Real};
    wavelet::Union{Vector{Float32},Nothing}=nothing,
    verbose::Bool=false,
    on_shot_complete::Union{Function,Nothing}=nothing)

    n_shots = length(src_x)
    @assert length(src_z) == n_shots "src_x and src_z must have same length"

    gathers = Vector{Matrix{Float32}}(undef, n_shots)

    # Use custom wavelet or cached one
    wavelet_dev = if wavelet === nothing
        sim.wavelet_device
    else
        to_device(wavelet, sim.backend)
    end

    t_start = time()

    for i in 1:n_shots
        # Create source
        src_i = round(Int, src_x[i] / sim.dx) + sim.pad + 1
        src_j = round(Int, src_z[i] / sim.dz) + sim.pad + 1
        source = Source(src_i, src_j, wavelet_dev)

        # Reset wavefield
        reset!(sim.backend, sim.wavefield)

        # Clear receiver data
        fill!(sim.receivers.data, 0.0f0)

        # Run time loop
        run_time_loop!(sim.backend, sim.wavefield, sim.medium, sim.habc,
            sim.fd_coeffs, source, sim.receivers, sim.params;
            progress=false, on_step=nothing)

        # Copy gather
        gathers[i] = _get_gather_from_batch(sim.backend, sim.receivers)

        # Callback
        if on_shot_complete !== nothing
            on_shot_complete(gathers[i], i)
        end

        # Progress (minimal overhead - every 10%)
        if verbose && (i % max(1, n_shots ÷ 10) == 0 || i == n_shots)
            elapsed = time() - t_start
            @info "Progress" completed = "$i/$n_shots" time_per_shot = "$(round(elapsed/i, digits=3))s"
        end
    end

    if verbose
        total = time() - t_start
        @info "Completed" n_shots = n_shots total_time = "$(round(total, digits=2))s" per_shot = "$(round(total/n_shots, digits=3))s"
    end

    return gathers
end

# ==============================================================================
# Benchmarking Utility
# ==============================================================================

"""
    benchmark_shots(model, rec_x, rec_z, src_x, src_z; config, n_warmup=2)

Benchmark shot simulation performance with proper warmup.

# Arguments
- `model`: Velocity model
- `rec_x, rec_z`: Receiver positions
- `src_x, src_z`: Source positions for benchmark

# Keyword Arguments
- `config`: Simulation configuration
- `n_warmup`: Number of warmup shots (default: 2)

# Returns
- NamedTuple with timing results

# Example
```julia
result = benchmark_shots(model, rec_x, rec_z, src_x, src_z; config=config)
println("Time per shot: \$(result.time_per_shot)s")
```
"""
function benchmark_shots(
    model::VelocityModel,
    rec_x::Vector{<:Real},
    rec_z::Vector{<:Real},
    src_x::Vector{<:Real},
    src_z::Vector{<:Real};
    config::SimulationConfig=SimulationConfig(),
    be::AbstractBackend=is_cuda_available() ? backend(:cuda) : backend(:cpu),
    n_warmup::Int=2
)
    println("="^60)
    println("  ElasticWave2D.jl Shot Benchmark")
    println("="^60)
    println()

    n_shots = length(src_x)

    # Initialize
    println("Initializing BatchSimulator...")
    t_init = time()
    sim = BatchSimulator(model, rec_x, rec_z; config=config, be=be)

    # Synchronize if CUDA
    if sim.backend isa CUDABackend
        CUDA.synchronize()
    end
    t_init = time() - t_init
    println("  Initialization: $(round(t_init, digits=3))s")

    # Warmup
    println("\nWarmup ($n_warmup shots)...")
    for i in 1:min(n_warmup, n_shots)
        _ = simulate_shot!(sim, src_x[i], src_z[i])
    end
    if sim.backend isa CUDABackend
        CUDA.synchronize()
    end
    println("  Warmup complete")

    # Benchmark
    println("\nBenchmarking $n_shots shots...")

    if sim.backend isa CUDABackend
        CUDA.synchronize()
    end

    t_start = time()
    gathers = simulate_shots!(sim, Float32.(src_x), Float32.(src_z); verbose=false)

    if sim.backend isa CUDABackend
        CUDA.synchronize()
    end

    total_time = time() - t_start
    time_per_shot = total_time / n_shots

    println("\n" * "="^60)
    println("  Results")
    println("="^60)
    println("  Backend:        $(typeof(sim.backend))")
    println("  Total shots:    $n_shots")
    println("  Total time:     $(round(total_time, digits=3))s")
    println("  Time per shot:  $(round(time_per_shot, digits=3))s")
    println("  Throughput:     $(round(1/time_per_shot, digits=2)) shots/s")
    println("="^60)

    return (
        time_per_shot=time_per_shot,
        total_time=total_time,
        n_shots=n_shots,
        init_time=t_init,
        gathers=gathers
    )
end