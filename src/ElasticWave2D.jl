# ==============================================================================
# ElasticWave2D.jl
#
# High-Performance 2D Elastic Wave Simulation Framework
#
# Project Structure:
# ==================
# src/
# ├── api/               # User API surface (single-shot simulate)
# ├── compute/           # Hardware abstraction (CPU/CUDA)
# ├── core/              # Fundamental data structures
# ├── physics/           # Physical laws and numerical kernels
# ├── initialization/    # Setup routines
# ├── solver/            # Solvers and orchestration
# ├── io/                # Input/Output
# └── visualization/     # Plotting and video
#
# Usage:
# ======
#   using ElasticWave2D
#   gather = simulate(model, source, receivers, boundary, outputs, simConfig)
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
# Exports - Core internals (for advanced users)
# ==============================================================================

# --- Compute ---
export AbstractBackend, CPUBackend, CUDABackend
export CPU_BACKEND, CUDA_BACKEND
export backend, to_device, synchronize
export is_cuda_available, is_cuda_functional

# --- Core Types ---
export Wavefield, Medium, HABCConfig
export Source, StressSource, ForceSource, Receivers, SimParams
export ShotResult, VelocityModel, BoundaryConfig

# --- User-Facing Types ---
export AbstractWavelet, RickerWavelet, CustomWavelet, Ricker, generate
export SourceMechanism, Explosion, ForceX, ForceZ, StressTxx, StressTzz, StressTxz
export SourceConfig
export RecordType, Vz, Vx, Pressure
export ReceiverConfig, line_receivers

# --- User API ---
export SimConfig, OutputConfig
export top_image, top_absorbing, top_vacuum
export simulate
export BatchSimulator, simulate_shot!, simulate_shots!

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

# --- Wavelet ---
export ricker_wavelet, gaussian_wavelet
export normalize_wavelet, wavelet_info, validate_external_wavelet

# --- Physics (Kernels) ---
export update_velocity!, update_stress!
export apply_habc!, apply_habc_velocity!, apply_habc_stress!
export backup_boundary!, apply_free_surface_velocity!, apply_free_surface_stress!
export inject_source!, record_receivers!, reset!

# --- Solver ---
export TimeStepInfo
export time_step!, run_time_loop!, run_time_loop_with_boundaries!
export compose_on_step

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

# --- Outputs (internal utilities; keep unexported to reduce API surface) ---

# ==============================================================================
# Include Files
# ==============================================================================

# 1. Compute (Hardware Abstraction)
include("compute/backend_types.jl")

# 2. Core (Data Structures)
include("core/simulation_types.jl")
include("core/velocity_model.jl")
include("core/boundary_configuration.jl")

# 2.5 Domain (User-facing types)
include("domain/api_types.jl")

# 3. Physics (Numerical Kernels)
include("physics/wave_propagation/velocity_kernel.jl")
include("physics/wave_propagation/stress_kernel.jl")
include("physics/boundaries/absorbing_boundary.jl")
include("physics/boundaries/vacuum_boundary.jl")
include("physics/boundaries/free_surface_boundary.jl")
include("physics/interaction/source_receiver_kernel.jl")
include("physics/interaction/wavelet.jl")

# 4. Visualization
include("visualization/wavefield_video.jl")
include("visualization/static_plots.jl")

# 5. Initialization
include("initialization/medium_setup.jl")
include("initialization/vacuum_setup.jl")
include("initialization/surface_generator.jl")

# 6. Solver (Orchestration)
include("solver/time_stepper.jl")
include("solver/observers.jl")
include("solver/shot_manager.jl")
include("solver/parallel_executor.jl")
include("solver/batch_runner.jl")

# 7. IO (Input/Output)
include("io/model_io.jl")
include("io/seismic_data_io.jl")
include("io/geometry_io.jl")

# 7.5 Outputs (directory structure & artifacts)
include("outputs/paths.jl")
include("outputs/artifacts.jl")

# 8. User API (single-shot)
include("api/single_shot_api.jl")

end # module ElasticWave2D
