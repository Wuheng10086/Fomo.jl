# ==============================================================================
# simulation/parallel_shots.jl
#
# Intelligent parallel shot execution for maximum GPU utilization
# ==============================================================================

# ==============================================================================
# Memory Estimation
# ==============================================================================

"""
Estimate GPU memory required for simulation (in bytes).
"""
function estimate_memory_usage(nx::Int, nz::Int, nt::Int, n_rec::Int)
    sizeof_float = 4  # Float32
    
    # Wavefield: vx, vz, txx, tzz, txz (x2 for old values)
    wavefield = nx * nz * 10 * sizeof_float
    
    # Medium: lam, mu_txx, mu_txz, rho_vx, rho_vz
    medium = nx * nz * 5 * sizeof_float
    
    # HABC buffers
    habc = nx * nz * 4 * sizeof_float
    
    # Gather output
    gather = nt * n_rec * sizeof_float
    
    # FD coefficients (small)
    coeffs = 100 * sizeof_float
    
    total = wavefield + medium + habc + gather + coeffs
    
    return (
        total = total,
        wavefield = wavefield,
        medium = medium,
        habc = habc,
        gather = gather
    )
end

# ==============================================================================
# GPU Query Functions
# ==============================================================================

"""
Get information about all available CUDA devices.
"""
function get_gpu_info()
    if !CUDA_AVAILABLE[]
        return nothing
    end
    
    n_devices = length(CUDA.devices())
    gpus = []
    
    for i in 0:(n_devices-1)
        CUDA.device!(i)
        dev = CUDA.device()
        mem_total = CUDA.total_memory()
        mem_free = CUDA.available_memory()
        name = CUDA.name(dev)
        
        push!(gpus, (
            id = i,
            name = name,
            total_memory = mem_total,
            free_memory = mem_free
        ))
    end
    
    # Reset to device 0
    CUDA.device!(0)
    
    return gpus
end

"""
Print GPU information and memory estimate.
"""
function print_hardware_info(nx::Int, nz::Int, nt::Int, n_rec::Int)
    mem = estimate_memory_usage(nx, nz, nt, n_rec)
    
    println()
    println("=" ^ 70)
    println("  Hardware & Memory Analysis")
    println("=" ^ 70)
    
    println("\n  📊 Memory Requirements (per shot):")
    println("     Wavefield:  $(round(mem.wavefield / 1024^2, digits=1)) MB")
    println("     Medium:     $(round(mem.medium / 1024^2, digits=1)) MB")
    println("     HABC:       $(round(mem.habc / 1024^2, digits=1)) MB")
    println("     Gather:     $(round(mem.gather / 1024^2, digits=1)) MB")
    println("     ────────────────────────")
    println("     Total:      $(round(mem.total / 1024^2, digits=1)) MB")
    
    gpus = get_gpu_info()
    if gpus !== nothing && !isempty(gpus)
        println("\n  🎮 Available GPUs:")
        for gpu in gpus
            status = gpu.free_memory > mem.total ? "✓" : "⚠"
            println("     [$status] GPU $(gpu.id): $(gpu.name)")
            println("         Memory: $(round(gpu.free_memory / 1024^3, digits=2)) / $(round(gpu.total_memory / 1024^3, digits=2)) GB free")
        end
        
        # Recommendation
        usable_gpus = count(g -> g.free_memory > mem.total * 1.2, gpus)
        println("\n  💡 Recommendation:")
        if usable_gpus == 0
            println("     ⚠ Model may be too large for available GPU memory!")
            println("     Consider reducing grid size or using CPU mode.")
        elseif usable_gpus == 1
            println("     Single GPU mode - shots will run sequentially.")
            println("     Each shot uses $(round(mem.total / 1024^3, digits=2)) GB.")
        else
            println("     Multi-GPU available! Can run $usable_gpus shots in parallel.")
            println("     Use run_shots_multi_gpu!() for $(usable_gpus)x speedup.")
        end
    else
        println("\n  💻 CPU Mode (no CUDA)")
        println("     Use julia -t auto for multi-threaded kernels")
    end
    
    println("=" ^ 70)
    println()
end

# ==============================================================================
# Multi-GPU Parallel Execution
# ==============================================================================

"""
    run_shots_multi_gpu!(vp, vs, rho, dx, dz, nbc, fd_order,
                         rec_x, rec_z, src_x, src_z, wavelet, params;
                         free_surface=true, on_shot_complete=nothing, output_dir="")

Run shots distributed across multiple GPUs for maximum parallelism.
"""
function run_shots_multi_gpu!(
    vp::Matrix{Float32}, vs::Matrix{Float32}, rho::Matrix{Float32},
    dx::Float32, dz::Float32,
    nbc::Int, fd_order::Int,
    rec_x::Vector{Float32}, rec_z::Vector{Float32},
    src_x::Vector{Float32}, src_z::Vector{Float32},
    wavelet::Vector{Float32},
    params::SimParams;
    free_surface::Bool=true,
    on_shot_complete=nothing,
    output_dir::String="",
    rec_type::Symbol=:vz
)
    if !CUDA_AVAILABLE[]
        error("Multi-GPU mode requires CUDA. Use run_shots!() for CPU mode.")
    end
    
    n_gpus = length(CUDA.devices())
    n_shots = length(src_x)
    
    if n_gpus < 2
        @warn "Only 1 GPU available. Multi-GPU mode won't provide speedup."
    end
    
    # Print info
    nx, nz = size(vp)
    print_hardware_info(nx + 2*nbc, nz + 2*nbc, params.nt, length(rec_x))
    
    println("🚀 Starting Multi-GPU execution")
    println("   GPUs: $n_gpus")
    println("   Shots: $n_shots")
    println("   Shots per GPU: ~$(ceil(Int, n_shots / n_gpus))")
    println()
    
    # Distribute shots round-robin across GPUs
    gpu_assignments = [Int[] for _ in 1:n_gpus]
    for (i, shot_id) in enumerate(1:n_shots)
        gpu_id = mod(i - 1, n_gpus) + 1
        push!(gpu_assignments[gpu_id], shot_id)
    end
    
    # Results storage
    results = Vector{Union{ShotResult, Nothing}}(nothing, n_shots)
    results_lock = ReentrantLock()
    
    # Progress tracking
    completed = Threads.Atomic{Int}(0)
    
    # Launch tasks for each GPU
    tasks = []
    
    for gpu_id in 1:n_gpus
        shot_ids = gpu_assignments[gpu_id]
        if isempty(shot_ids)
            continue
        end
        
        t = Threads.@spawn begin
            # Set GPU for this task
            CUDA.device!(gpu_id - 1)  # 0-indexed
            
            gpu_name = CUDA.name(CUDA.device())
            @info "GPU $(gpu_id-1) ($gpu_name) starting $(length(shot_ids)) shots"
            
            # Initialize everything on this GPU
            be = CUDABackend()
            
            medium = init_medium(vp, vs, rho, dx, dz, nbc, fd_order, be;
                                free_surface=free_surface)
            
            avg_vp = sum(vp) / length(vp)
            habc = init_habc(medium.nx, medium.nz, nbc, params.dt, dx, dz,
                           avg_vp, be)
            
            fd_coeffs = to_device(get_fd_coefficients(fd_order), be)
            
            wavefield = Wavefield(medium.nx, medium.nz, be)
            
            # Setup receiver template
            rec_template = setup_receivers(rec_x, rec_z, medium; type=rec_type)
            
            # Process assigned shots
            for shot_id in shot_ids
                # Reset wavefield
                reset!(wavefield)
                
                # Create source
                src_i = round(Int, (src_x[shot_id] + nbc * dx) / dx) + 1
                src_j = round(Int, (src_z[shot_id] + nbc * dz) / dz) + 1
                source = Source(src_x[shot_id], src_z[shot_id], src_i, src_j,
                              to_device(wavelet, be))
                
                # Create receiver for this shot
                receivers = Receivers(
                    rec_template.x, rec_template.z,
                    rec_template.i_idx, rec_template.j_idx,
                    CUDA.zeros(Float32, params.nt, length(rec_x)),
                    rec_template.type
                )
                
                # Time loop
                for it in 1:params.nt
                    update_velocity!(be, wavefield, medium, fd_coeffs, params)
                    apply_habc_velocity!(be, wavefield, habc, medium)
                    update_stress!(be, wavefield, medium, fd_coeffs, params)
                    apply_habc_stress!(be, wavefield, habc, medium)
                    
                    if medium.free_surface
                        apply_free_surface!(be, wavefield, medium, params)
                    end
                    
                    inject_source!(be, wavefield, source, it, params)
                    record_receivers!(be, wavefield, receivers, it)
                end
                
                CUDA.synchronize()
                
                # Create result
                result = ShotResult(
                    Array(receivers.gather),
                    shot_id,
                    source.i, source.j,
                    Array(receivers.i_idx),
                    Array(receivers.j_idx)
                )
                
                # Store result (thread-safe)
                lock(results_lock) do
                    results[shot_id] = result
                end
                
                # Save if output_dir specified
                if !isempty(output_dir)
                    save_gather(result, joinpath(output_dir, "shot_$(shot_id).bin"))
                end
                
                # Callback
                if on_shot_complete !== nothing
                    on_shot_complete(result)
                end
                
                # Update progress
                done = Threads.atomic_add!(completed, 1) + 1
                @info "Shot $shot_id completed (GPU $(gpu_id-1)) - Progress: $done/$n_shots"
            end
            
            @info "GPU $(gpu_id-1) finished all assigned shots"
        end
        
        push!(tasks, t)
    end
    
    # Wait for all tasks
    for t in tasks
        wait(t)
    end
    
    # Convert to non-nothing vector
    final_results = [r for r in results if r !== nothing]
    
    println()
    println("✅ All $n_shots shots completed!")
    
    return final_results
end

# ==============================================================================
# Convenience Wrapper
# ==============================================================================

"""
    run_shots_auto!(model, rec_x, rec_z, src_x, src_z, wavelet, params;
                    nbc=50, fd_order=8, kwargs...)

Automatically choose the best execution strategy based on available hardware.
"""
function run_shots_auto!(
    model::VelocityModel,
    rec_x::Vector{Float32}, rec_z::Vector{Float32},
    src_x::Vector{Float32}, src_z::Vector{Float32},
    wavelet::Vector{Float32},
    params::SimParams;
    nbc::Int=50,
    fd_order::Int=8,
    free_surface::Bool=true,
    on_shot_complete=nothing,
    output_dir::String="",
    rec_type::Symbol=:vz
)
    n_shots = length(src_x)
    
    if CUDA_AVAILABLE[]
        n_gpus = length(CUDA.devices())
        
        if n_gpus >= 2 && n_shots >= 2
            @info "Auto-selected: Multi-GPU mode ($n_gpus GPUs)"
            return run_shots_multi_gpu!(
                model.vp, model.vs, model.rho,
                model.dx, model.dz,
                nbc, fd_order,
                rec_x, rec_z,
                src_x, src_z,
                wavelet, params;
                free_surface=free_surface,
                on_shot_complete=on_shot_complete,
                output_dir=output_dir,
                rec_type=rec_type
            )
        else
            @info "Auto-selected: Single GPU mode"
            be = CUDABackend()
            medium = init_medium(model, nbc, fd_order, be; free_surface=free_surface)
            habc = init_habc(medium.nx, medium.nz, nbc, params.dt, 
                           model.dx, model.dz, sum(model.vp)/length(model.vp), be)
            fd_coeffs = to_device(get_fd_coefficients(fd_order), be)
            wavefield = Wavefield(medium.nx, medium.nz, be)
            rec_template = setup_receivers(rec_x, rec_z, medium; type=rec_type)
            shot_config = MultiShotConfig(src_x, src_z, wavelet)
            
            return run_shots!(be, wavefield, medium, habc, fd_coeffs,
                            rec_template, shot_config, params;
                            on_shot_complete=on_shot_complete)
        end
    else
        @info "Auto-selected: CPU mode (use julia -t auto for threading)"
        be = CPUBackend()
        medium = init_medium(model, nbc, fd_order, be; free_surface=free_surface)
        habc = init_habc(medium.nx, medium.nz, nbc, params.dt,
                       model.dx, model.dz, sum(model.vp)/length(model.vp), be)
        fd_coeffs = to_device(get_fd_coefficients(fd_order), be)
        wavefield = Wavefield(medium.nx, medium.nz, be)
        rec_template = setup_receivers(rec_x, rec_z, medium; type=rec_type)
        shot_config = MultiShotConfig(src_x, src_z, wavelet)
        
        return run_shots!(be, wavefield, medium, habc, fd_coeffs,
                        rec_template, shot_config, params;
                        on_shot_complete=on_shot_complete)
    end
end
