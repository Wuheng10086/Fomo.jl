# ==============================================================================
# Fomo.jl - Forward Modeling
#
# High-Performance 2D Elastic Wave Simulation Framework
# ==============================================================================

module Fomo

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

# Check if CUDA is functional (has GPU)
const CUDA_AVAILABLE = Ref(false)

function __init__()
    if CUDA.functional()
        CUDA_AVAILABLE[] = true
        @info "Fomo: CUDA functional, GPU acceleration enabled"
    else
        @info "Fomo: CUDA not functional (no GPU), using CPU mode"
    end
end

is_cuda_available() = CUDA_AVAILABLE[]
is_cuda_functional() = CUDA_AVAILABLE[]

# ==============================================================================
# Exports
# ==============================================================================

# Backend system
export AbstractBackend, CPUBackend, CUDABackend
export CPU_BACKEND, CUDA_BACKEND
export backend, to_device, synchronize
export is_cuda_available, is_cuda_functional

# Data structures
export Wavefield, Medium, HABCConfig
export Source, Receivers, SimParams
export ShotConfig, MultiShotConfig, ShotResult

# Initialization
export init_medium, init_habc, setup_receivers
export get_fd_coefficients, ricker_wavelet

# Kernels (for advanced users)
export update_velocity!, update_stress!
export apply_habc!, apply_habc_velocity!, apply_habc_stress!
export backup_boundary!, apply_free_surface!
export inject_source!, record_receivers!, reset!

# Simulation interface
export TimeStepInfo
export time_step!, run_time_loop!
export run_shot!, run_shots!

# Parallel execution
export get_gpu_info, print_hardware_info
export run_shots_multi_gpu!, run_shots_auto!

# Visualization
export VideoConfig, VideoRecorder, MultiFieldRecorder
export plot_setup, generate_video

# IO
export save_gather, load_gather

# Model IO
export VelocityModel, load_model, load_model_files, save_model, convert_model, model_info

# Geometry IO (for migration)
export SurveyGeometry, MultiShotGeometry
export create_geometry, save_geometry, load_geometry

# ==============================================================================
# Include Files (order matters!)
# ==============================================================================

# Backend abstraction (must be first)
include("backends/backend.jl")

# Core structures
include("core/structures.jl")

# Kernels
include("kernels/velocity.jl")
include("kernels/stress.jl")
include("kernels/boundary.jl")
include("kernels/source_receiver.jl")

# IO (must be before utils/init.jl which uses VelocityModel)
include("io/output.jl")
include("io/model_loader.jl")

# Utilities (uses VelocityModel from model_loader.jl)
include("utils/init.jl")

# Geometry IO (ShotResult is now in structures.jl)
include("io/geometry_io.jl")

# Simulation
include("simulation/time_stepper.jl")
include("simulation/shot_manager.jl")
include("simulation/parallel_shots.jl")

# Visualization
include("visualization/video_recorder.jl")
include("visualization/setup_check.jl")

end # module Fomo
