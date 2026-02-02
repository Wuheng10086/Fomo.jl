# ==============================================================================
# ElasticWave2D.jl
#
# High-Performance 2D Elastic Wave Simulation Framework
#
# Project Structure:
# ==================
# src/
# ├── compute/            # Hardware abstraction
# ├── core/               # Fundamental data structures
# ├── physics/            # Physical laws and numerical kernels
# ├── initialization/     # Setup routines
# ├── solver/             # Solvers and orchestration
# ├── workflow/           # High-level API
# ├── io/                 # Input/Output
# └── visualization/      # Plotting
#
# ==============================================================================

module ElasticWave2D

# ==============================================================================
# Dependencies
# ==============================================================================

using LoopVectorization
using ProgressMeter
using Printf
using Statistics
using CairoMakie
using JLD2
using JSON
using CUDA

# ==============================================================================
# CUDA Support
# ==============================================================================

const CUDA_AVAILABLE = Ref(false)

function __init__()
    if CUDA.functional()
        CUDA_AVAILABLE[] = true
        @info "ElasticWave2D: CUDA functional, GPU acceleration enabled"
    else
        @info "ElasticWave2D: CUDA not functional (no GPU), using CPU mode"
    end
end

is_cuda_available() = CUDA_AVAILABLE[]
is_cuda_functional() = CUDA_AVAILABLE[]

# ==============================================================================
# Exports
# ==============================================================================

# --- Compute ---
export AbstractBackend, CPUBackend, CUDABackend
export CPU_BACKEND, CUDA_BACKEND
export backend, to_device, synchronize
export is_cuda_available, is_cuda_functional

# --- Core ---
export Wavefield, Medium, HABCConfig
export Source, StressSource, Receivers, SimParams
export ShotConfig, MultiShotConfig, ShotResult
export VelocityModel
export BoundaryConfig

# --- Initialization ---
export init_medium, init_wavefield, init_habc, setup_receivers
export get_fd_coefficients
export init_medium_vacuum
export setup_vacuum_formulation!, compute_staggered_params_vacuum
export apply_vacuum_mask!, compute_surface_indices
export setup_receivers_on_surface, setup_source_on_surface
export flat_surface, sinusoidal_surface
export gaussian_valley, gaussian_hill, tilted_surface, step_surface
export random_surface, combine_surfaces
export validate_surface_elevation

# --- Wavelet (★ NEW) ---
export ricker_wavelet, gaussian_wavelet
export normalize_wavelet, wavelet_info, validate_external_wavelet

# --- Physics (Kernels) ---
export update_velocity!, update_stress!
export apply_habc!, apply_habc_velocity!, apply_habc_stress!
export backup_boundary!, apply_image_method!
export inject_source!, record_receivers!, reset!

# --- Solver ---
export TimeStepInfo
export time_step!, run_time_loop!
export run_shot!, run_shots!, run_shots_fast!
export BatchSimulator, simulate_shot!, simulate_shots!, benchmark_shots
export get_gpu_info, print_hardware_info
export run_shots_multi_gpu!, run_shots_auto!

# --- Workflow (API) ---
export SimulationConfig, SimulationResult, simulate!
export SimpleConfig, TopographyConfig
export create_homogeneous_model, create_layered_model, create_gradient_model
export SourceConfig, ReceiverConfig
export seismic_survey

# --- Visualization ---
export VideoConfig, FieldRecorder, MultiFieldRecorder
export generate_video
export plot_setup, plot_gather

# --- IO ---
export save_gather, load_gather
export load_model, load_model_files, save_model
export convert_model, model_info, resample_model, suggest_grid_spacing
export SurveyGeometry, MultiShotGeometry
export create_geometry, save_geometry, load_geometry

# ==============================================================================
# Include Files
# ==============================================================================

# 1. Compute (Hardware Abstraction)
include("compute/backend_types.jl")

# 2. Core (Data Structures)
include("core/simulation_types.jl")
include("core/velocity_model.jl")
include("core/boundary_configuration.jl")
include("core/simulation_configuration.jl")

# 3. Physics (Numerical Kernels)
include("physics/wave_propagation/velocity_kernel.jl")
include("physics/wave_propagation/stress_kernel.jl")
include("physics/boundaries/absorbing_boundary.jl")
include("physics/boundaries/vacuum_boundary.jl")
include("physics/interaction/source_receiver_kernel.jl")
include("physics/interaction/wavelet.jl")  # ★ NEW: Wavelet generation

# 4. Visualization (Must be defined before Solver/Workflow)
include("visualization/wavefield_video.jl")
include("visualization/static_plots.jl")

# 5. Initialization
include("initialization/medium_setup.jl")
include("initialization/vacuum_setup.jl")
include("initialization/surface_generator.jl")

# 6. Solver (Orchestration)
include("solver/time_stepper.jl")
include("solver/shot_manager.jl")
include("solver/parallel_executor.jl")
include("solver/batch_runner.jl")

# 7. Workflow (High-Level API)
include("workflow/simulation_api.jl")
include("workflow/simplified_api.jl")

# 8. IO (Input/Output)
include("io/model_io.jl")
include("io/seismic_data_io.jl")
include("io/geometry_io.jl")
include("io/output_manager.jl")

include("api/API.jl")
include("deprecations.jl")

end # module ElasticWave2D
