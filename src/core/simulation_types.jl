# ==============================================================================
# core/structures.jl (OPTIMIZED)
#
# Unified data structures using parametric types
# OPTIMIZATION: Added precomputed buoyancy (1/rho) and lam_2mu fields
# ==============================================================================

# ==============================================================================
# Wavefield - Parametric over array type
# ==============================================================================

"""
    Wavefield{T}

Velocity and stress components. Works for both CPU and GPU.
`T` is the array type (Array{Float32,2} or CuArray{Float32,2}).
"""
mutable struct Wavefield{T<:AbstractMatrix{Float32}}
    # Current time step
    vx::T
    vz::T
    txx::T
    tzz::T
    txz::T
    
    # Previous time step (for HABC)
    vx_old::T
    vz_old::T
    txx_old::T
    tzz_old::T
    txz_old::T
end

"""
    Wavefield(nx, nz, backend::AbstractBackend)

Create zero-initialized wavefield on specified backend.
"""
function Wavefield(nx::Int, nz::Int, b::CPUBackend)
    return Wavefield(
        zeros(Float32, nx, nz), zeros(Float32, nx, nz),
        zeros(Float32, nx, nz), zeros(Float32, nx, nz), zeros(Float32, nx, nz),
        zeros(Float32, nx, nz), zeros(Float32, nx, nz),
        zeros(Float32, nx, nz), zeros(Float32, nx, nz), zeros(Float32, nx, nz)
    )
end

function Wavefield(nx::Int, nz::Int, b::CUDABackend)
    if !CUDA_AVAILABLE[]
        error("CUDA not functional (no GPU available)")
    end
    return Wavefield(
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz),
        CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz), CUDA.zeros(Float32, nx, nz)
    )
end

# ==============================================================================
# Medium - OPTIMIZED with precomputed buoyancy and lam_2mu
# ==============================================================================

"""
    Medium{T}

Physical properties of the simulation domain.
OPTIMIZATION: Includes precomputed buoyancy (1/rho) and lam_2mu (lambda + 2*mu)
              to eliminate expensive divisions in the simulation loop.
"""
struct Medium{T<:AbstractMatrix{Float32}}
    nx::Int
    nz::Int
    dx::Float32
    dz::Float32
    x_max::Float32
    z_max::Float32
    M::Int              # FD half-stencil width
    pad::Int            # Boundary padding
    is_free_surface::Bool
    
    # Original material properties
    lam::T              # Lambda (Lamé's first parameter)
    mu_txx::T           # Mu at txx/tzz positions
    mu_txz::T           # Mu at txz positions (harmonic average)
    
    # OPTIMIZED: Precomputed values to eliminate divisions in hot loops
    buoy_vx::T          # 1/rho at vx positions (buoyancy)
    buoy_vz::T          # 1/rho at vz positions (buoyancy)
    lam_2mu::T          # lambda + 2*mu (precomputed)
end

# ==============================================================================
# HABC Configuration - Parametric
# ==============================================================================

"""
    HABCConfig{T}

Higdon Absorbing Boundary Condition parameters.
"""
struct HABCConfig{T<:AbstractMatrix{Float32}}
    nbc::Int
    qx::Float32
    qz::Float32
    qt_x::Float32
    qt_z::Float32
    qxt::Float32
    
    w_vx::T
    w_vz::T
    w_tau::T
end

# ==============================================================================
# Source & Receiver - Parametric
# ==============================================================================

abstract type AbstractSource end

"""
    Source{V<:AbstractVector}

Single source configuration.
"""
struct Source{V<:AbstractVector{Float32}, I<:Integer} <: AbstractSource
    i::I                # X grid index
    j::I                # Z grid index
    wavelet::V          # Source time function
end

struct StressSource{V<:AbstractVector{Float32}, I<:Integer} <: AbstractSource
    i::I
    j::I
    wavelet::V
    component::Symbol   # :txx, :tzz, :txz
end

"""
    Receivers{T,I}

Receiver configuration and data buffer.
"""
struct Receivers{T<:AbstractMatrix{Float32}, I<:AbstractVector{<:Integer}}
    i::I                    # X indices
    j::I                    # Z indices
    data::T                 # [nt × n_rec]
    type::Symbol            # :vz, :vx, :p
end

# ==============================================================================
# Simulation Parameters (immutable, no arrays)
# ==============================================================================

"""
    SimParams

Time stepping and grid parameters.
"""
struct SimParams
    dt::Float32
    nt::Int
    dtx::Float32        # dt/dx
    dtz::Float32        # dt/dz
    fd_order::Int
    M::Int              # FD half-stencil width
end

function SimParams(dt, nt, dx, dz, fd_order)
    M = fd_order ÷ 2
    SimParams(Float32(dt), nt, Float32(dt/dx), Float32(dt/dz), fd_order, M)
end

# ==============================================================================
# Shot Result
# ==============================================================================

"""
    ShotResult

Container for the results of a single shot simulation.
Typically returned by batch simulation functions or `run_shot!`.

# Fields
- `gather::Matrix{Float32}`: Recorded seismic gather data [nt × n_rec]. Always stored on CPU.
- `shot_id::Int`: Unique identifier for the shot.
- `src_i::Int`: Source X grid index.
- `src_j::Int`: Source Z grid index.
- `rec_i::Vector{Int}`: Receiver X grid indices.
- `rec_j::Vector{Int}`: Receiver Z grid indices.

# Example
```julia
# Access gather data from a shot result
data = result.gather
println("Shot ID: ", result.shot_id)
```
"""
struct ShotResult
    gather::Matrix{Float32}   # [nt × n_rec] - always on CPU
    shot_id::Int
    
    # Source position (grid indices)
    src_i::Int
    src_j::Int
    
    # Receiver positions (grid indices)
    rec_i::Vector{Int}
    rec_j::Vector{Int}
end
