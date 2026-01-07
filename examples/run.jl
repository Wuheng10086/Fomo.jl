# ==============================================================================
# run.jl
#
# Fomo Framework Usage Example
#
# Usage:
#   julia --project=.. run.jl                    # Run with synthetic model
#   julia --project=.. run.jl model.jld2         # Run with model file
# ==============================================================================

using Fomo
using Statistics

# ==============================================================================
# Configuration
# ==============================================================================

const BACKEND_TYPE = :cuda  # :cpu or :cuda
const NBC = 50
const FD_ORDER = 8
const F0 = 15.0f0
const TOTAL_TIME = 1.0f0
const CFL = 0.5f0

# Video config (set to nothing to disable)
const VIDEO_CONFIG = nothing

# Output directory
const OUTPUT_DIR = "outputs/"

# ==============================================================================
# Main - Example with synthetic model
# ==============================================================================

function main_synthetic()
    @info "Running with synthetic model..."
    
    # Create simple layered model
    DX, DZ = 10.0f0, 10.0f0
    NX, NZ = 400, 200
    
    # Note: data is stored as [nz, nx] (depth first)
    vp = fill(3000.0f0, NZ, NX)
    vs = fill(1800.0f0, NZ, NX)
    rho = fill(2200.0f0, NZ, NX)
    
    # Add a layer (affects lower half of depth)
    vp[NZ÷2:end, :] .= 4000.0f0
    vs[NZ÷2:end, :] .= 2400.0f0
    
    # Create model struct
    model = VelocityModel(vp, vs, rho, DX, DZ; name="synthetic")
    
    # Define geometry
    x_src = Float32[model.nx * DX / 2]
    z_src = Float32[50.0]
    x_rec = Float32.(range(0, (model.nx-1)*DX, step=50))
    z_rec = fill(10.0f0, length(x_rec))
    
    # Check setup visually
    plot_setup(model, x_src, z_src, x_rec, z_rec; 
               output="$(OUTPUT_DIR)synthetic_setup.png")
    
    run_simulation_with_model(model, x_src, z_src, x_rec, z_rec)
end

# ==============================================================================
# Main - Example with model file
# ==============================================================================

function main_from_file(model_path::String)
    @info "Loading model from: $model_path"
    
    model_name = splitext(basename(model_path))[1]

    # Load model (auto-detects format)
    model = load_model(model_path)
    model_info(model)

    # Define geometry
    x_max = (model.nx - 1) * model.dx
    
    # Sources - every 200m
    x_src = Float32.(range(100, x_max - 100, step=200))
    z_src = fill(10.0f0, length(x_src))
    
    # Receivers - every 15m
    x_rec = Float32.(range(0, x_max, step=15))
    z_rec = fill(50.0f0, length(x_rec))
    
    # Check setup visually
    plot_setup(model, x_src, z_src, x_rec, z_rec; 
               output="$(OUTPUT_DIR)$(model_name)_setup.png",
               title="$(model_name) - Survey Setup")
    
    run_simulation_with_model(model, x_src, z_src, x_rec, z_rec)
end

# ==============================================================================
# Common simulation code
# ==============================================================================

function run_simulation_with_model(model::VelocityModel, 
                                   x_src::Vector{Float32}, z_src::Vector{Float32},
                                   x_rec::Vector{Float32}, z_rec::Vector{Float32})
    
    # Choose backend
    be = is_cuda_available() && BACKEND_TYPE == :cuda ? backend(:cuda) : backend(:cpu)
    
    # Time stepping
    dt = CFL * min(model.dx, model.dz) / maximum(model.vp)
    nt = ceil(Int, TOTAL_TIME / dt)
    
    # Initialize
    medium = init_medium(model, NBC, FD_ORDER, be; free_surface=true)
    habc = init_habc(medium.nx, medium.nz, nbc, params.dt, model.dx, model.dz, mean(model.vp), be)
    params = SimParams(dt, nt, model.dx, model.dz, FD_ORDER)
    
    @info "Simulation setup" backend=typeof(be) grid=(medium.nx, medium.nz) dt=dt nt=nt
    @info "Survey" n_shots=length(x_src) n_rec=length(x_rec)
    
    # Setup
    fd_coeffs = to_device(get_fd_coefficients(FD_ORDER), be)
    wavefield = Wavefield(medium.nx, medium.nz, be)
    wavelet = ricker_wavelet(F0, dt, nt)
    
    # Receivers
    rec_template = setup_receivers(x_rec, z_rec, medium; type=:vz)
    
    # Sources
    shot_config = MultiShotConfig(x_src, z_src, wavelet)
    
    # Video recorder
    video_callback = nothing
    if VIDEO_CONFIG !== nothing
        video_callback = MultiFieldRecorder(medium.nx, medium.nz, dt, VIDEO_CONFIG)
    end
    
    # Ensure output directory exists
    mkpath(OUTPUT_DIR)
    
    # Run
    results = run_shots!(be, wavefield, medium, habc, fd_coeffs,
                         rec_template, shot_config, params;
                         on_step=video_callback,
                         on_shot_complete=r -> save_gather(r, "$(OUTPUT_DIR)shot_$(r.shot_id).bin"))
    
    if video_callback !== nothing
        close(video_callback)
    end
    
    # Save geometry for migration
    geom = create_geometry(results, medium, params)
    save_geometry("$(OUTPUT_DIR)survey_geometry.jld2", geom)
    save_geometry("$(OUTPUT_DIR)survey_geometry.json", geom)
    
    @info "Done!" shots=length(results)
    return results
end

# ==============================================================================
# Entry point
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Ensure output directory exists
    mkpath(OUTPUT_DIR)
    
    if length(ARGS) >= 1
        # Load from file
        main_from_file(ARGS[1])
    else
        # Use synthetic model
        main_synthetic()
    end
end
