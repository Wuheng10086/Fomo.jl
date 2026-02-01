# Migration Report: ElasticWave2D.jl Restructuring

This document details the systematic restructuring of the `ElasticWave2D.jl` project to improve modularity, semantic clarity, and adherence to standard naming conventions.

## 1. Directory Structure Changes

The flat `src/` structure has been reorganized into domain-driven categories:

| Domain | Description |
| :--- | :--- |
| **`compute`** | Hardware abstraction and backend management (CPU/GPU). |
| **`core`** | Fundamental data structures, types, and domain models. |
| **`physics`** | Numerical kernels implementing physical laws (wave equation, boundaries). |
| **`initialization`** | Setup routines for models, media, and geometry. |
| **`solver`** | Time-stepping logic, simulation orchestration, and parallel execution. |
| **`workflow`** | High-level user-facing APIs and configuration. |
| **`io`** | Input/Output operations for models and seismic data. |
| **`visualization`** | Plotting and video generation utilities. |

## 2. File Mapping Table

| Old Path | New Path | Description |
| :--- | :--- | :--- |
| `src/backends/backend.jl` | `src/compute/backend_types.jl` | Hardware backend definitions |
| `src/types/structures.jl` | `src/core/simulation_types.jl` | Core simulation structs |
| `src/types/model.jl` | `src/core/velocity_model.jl` | Velocity model definitions |
| `src/types/boundary_config.jl` | `src/core/boundary_configuration.jl` | Boundary configuration types |
| `src/kernels/velocity.jl` | `src/physics/wave_propagation/velocity_kernel.jl` | Velocity update kernel |
| `src/kernels/stress.jl` | `src/physics/wave_propagation/stress_kernel.jl` | Stress update kernel |
| `src/kernels/boundary.jl` | `src/physics/boundaries/absorbing_boundary.jl` | Absorbing boundary (HABC) kernel |
| `src/kernels/vacuum.jl` | `src/physics/boundaries/vacuum_boundary.jl` | Vacuum boundary kernel |
| `src/kernels/source_receiver.jl` | `src/physics/interaction/source_receiver_kernel.jl` | Source injection & receiver recording |
| `src/simulation/init.jl` | `src/initialization/medium_setup.jl` | Medium initialization (flat) |
| `src/simulation/init_vacuum.jl` | `src/initialization/vacuum_setup.jl` | Vacuum medium initialization |
| `src/simulation/surfaces.jl` | `src/initialization/surface_generator.jl` | Surface topography generators |
| `src/simulation/time_stepper.jl` | `src/solver/time_stepper.jl` | Core time loop |
| `src/simulation/shots.jl` | `src/solver/shot_manager.jl` | Single/Multi-shot management |
| `src/simulation/batch.jl` | `src/solver/batch_runner.jl` | Batch simulation runner |
| `src/simulation/parallel.jl` | `src/solver/parallel_executor.jl` | Parallel execution logic |
| `src/simulation/api.jl` | `src/workflow/simulation_api.jl` | Main Simulation API |
| `src/simulation/simple_api.jl` | `src/workflow/simplified_api.jl` | Simplified high-level API |
| `src/io/gather_io.jl` | `src/io/seismic_data_io.jl` | Seismic gather I/O |
| `src/io/geometry_io.jl` | `src/io/geometry_io.jl` | Survey geometry I/O |
| `src/io/model_io.jl` | `src/io/model_io.jl` | Velocity model I/O |
| `src/io/outputs.jl` | `src/io/output_manager.jl` | Simulation output helpers |
| `src/visualization/video.jl` | `src/visualization/wavefield_video.jl` | Wavefield video recording |
| `src/visualization/plots.jl` | `src/visualization/static_plots.jl` | Static plotting utilities |

## 3. Potential Risks & Mitigation

*   **Import Errors**: The main module `ElasticWave2D.jl` must be updated to strictly follow the dependency order (Compute -> Core -> Physics -> Initialization -> Solver -> Workflow -> IO/Vis).
*   **External Scripts**: Any user scripts directly `include()`-ing subfiles (bad practice but possible) will break. The official API exported by `ElasticWave2D` module remains unchanged, so standard usage `using ElasticWave2D` is safe.
*   **Git History**: Moving files might obscure git history. We use `git mv` equivalent operations where possible to preserve history.

## 4. Rollback Plan

If critical issues arise:
1.  Revert the commit/changes to `src/ElasticWave2D.jl`.
2.  Move files back to their original locations using the mapping table above in reverse.
