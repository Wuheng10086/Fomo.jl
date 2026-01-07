# ==============================================================================
# run_parallel.jl
#
# Multi-GPU parallel shot execution example
# Automatically uses all available GPUs for maximum throughput
#
# Usage:
#   julia --project=.. run_parallel.jl model.jld2
#   julia --project=.. run_parallel.jl model.jld2 output_dir
# ==============================================================================

using Fomo
using Statistics

# ==============================================================================
# Configuration
# ==============================================================================

const NBC = 50
const FD_ORDER = 8
const F0 = 25.0f0
const TOTAL_TIME = 4.0f0
const CFL = 0.5f0

# ==============================================================================
# Main
# ==============================================================================

function main(model_path::String; output_dir::String="outputs/")
    # Get model name from path
    model_name = splitext(basename(model_path))[1]
    
    # Create output directory
    mkpath(output_dir)
    
    # Load model
    @info "Loading model: $model_path"
    model = load_model(model_path)
    model_info(model)
    
    # Calculate time stepping
    dt = CFL * min(model.dx, model.dz) / maximum(model.vp)
    nt = ceil(Int, TOTAL_TIME / dt)
    params = SimParams(dt, nt, model.dx, model.dz, FD_ORDER)
    
    @info "Time stepping" dt=dt nt=nt total_time=TOTAL_TIME
    
    # Define geometry
    x_max = (model.nx - 1) * model.dx
    z_max = (model.nz - 1) * model.dz
    
    # Sources: every 200m, 10m below surface
    src_x = Float32.(range(200, x_max - 200, step=200))
    src_z = fill(10.0f0, length(src_x))
    
    # Receivers: every 15m at 20m depth
    rec_x = Float32.(range(0, x_max, step=15))
    rec_z = fill(20.0f0, length(rec_x))
    
    # Wavelet
    wavelet = ricker_wavelet(F0, dt, nt)
    
    @info "Survey geometry" n_shots=length(src_x) n_receivers=length(rec_x)
    
    # Plot setup check (output filename includes model name)
    setup_plot_path = joinpath(output_dir, "$(model_name)_setup.png")
    plot_setup(model, src_x, src_z, rec_x, rec_z;
               output=setup_plot_path,
               title="$(model_name) - Survey Setup")
    
    println("\n" * "=" ^ 70)
    println("  Starting $(length(src_x)) shots")
    println("  Output directory: $output_dir")
    println("=" ^ 70 * "\n")
    
    # Run with automatic parallelization (no confirmation needed)
    start_time = time()
    
    results = run_shots_auto!(
        model, rec_x, rec_z, src_x, src_z, wavelet, params;
        nbc=NBC,
        fd_order=FD_ORDER,
        free_surface=true,
        output_dir=output_dir,
        on_shot_complete=r -> @info "Saved shot $(r.shot_id)"
    )
    
    elapsed = time() - start_time
    
    # Summary
    println("\n" * "=" ^ 70)
    println("  Execution Summary")
    println("=" ^ 70)
    println("  Total shots:     $(length(results))")
    println("  Total time:      $(round(elapsed / 60, digits=2)) minutes")
    println("  Time per shot:   $(round(elapsed / length(results), digits=2)) seconds")
    println("  Output dir:      $output_dir")
    println("=" ^ 70)
    
    # Save geometry
    be = is_cuda_available() ? backend(:cuda) : backend(:cpu)
    medium = init_medium(model, NBC, FD_ORDER, be; free_surface=true)
    geom = create_geometry(results, medium, params)
    save_geometry(joinpath(output_dir, "$(model_name)_geometry.jld2"), geom)
    save_geometry(joinpath(output_dir, "$(model_name)_geometry.json"), geom)
    
    @info "Done!" output_dir=output_dir
    
    return results
end

# ==============================================================================
# Entry Point
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("""
Usage: julia --project=.. run_parallel.jl <model.jld2> [output_dir]

Examples:
  julia --project=.. run_parallel.jl marmousi.jld2
  julia --project=.. run_parallel.jl marmousi.jld2 data/marm/

This script automatically:
  - Detects all available GPUs
  - Distributes shots across GPUs
  - Saves gathers and geometry
""")
        exit(1)
    end
    
    model_path = ARGS[1]
    output_dir = length(ARGS) >= 2 ? ARGS[2] : "outputs/"
    
    main(model_path; output_dir=output_dir)
end
