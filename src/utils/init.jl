# ==============================================================================
# utils/init.jl (OPTIMIZED)
#
# Initialization utilities for Medium, HABC, etc.
# OPTIMIZATION: Precomputes buoyancy (1/rho) and lam_2mu to eliminate 
#               divisions in the hot loop
# ==============================================================================

# ==============================================================================
# FD Coefficients
# ==============================================================================

const FD_COEFFICIENTS = Dict(
    2  => Float32[1.0],
    4  => Float32[1.125, -0.041666667],
    6  => Float32[1.171875, -0.065104167, 0.0046875],
    8  => Float32[1.1962890625, -0.079752604167, 0.0095703125, -0.000697544643],
    10 => Float32[1.2115478515625, -0.089721679687, 0.0138427734375, -0.00176565987723, 0.0001186795166]
)

"""
    get_fd_coefficients(order::Int) -> Vector{Float32}

Get FD coefficients for given order (2, 4, 6, 8, or 10).
"""
function get_fd_coefficients(order::Int)
    haskey(FD_COEFFICIENTS, order) || error("Unsupported FD order: $order")
    return FD_COEFFICIENTS[order]
end

# ==============================================================================
# Ricker Wavelet
# ==============================================================================

"""
    ricker_wavelet(f0, dt, nt) -> Vector{Float32}

Generate Ricker (Mexican hat) wavelet.
"""
function ricker_wavelet(f0::Real, dt::Real, nt::Int)
    t0 = 1.0 / f0
    wavelet = zeros(Float32, nt)
    for i in 1:nt
        τ = (i-1) * dt - t0
        arg = (π * f0 * τ)^2
        wavelet[i] = Float32((1.0 - 2.0 * arg) * exp(-arg))
    end
    return wavelet
end

# ==============================================================================
# Medium Initialization - OPTIMIZED
# ==============================================================================

"""
    init_medium(vp, vs, rho, dx, dz, nbc, fd_order, backend; free_surface=true)

Initialize Medium with material properties.
OPTIMIZED: Precomputes buoyancy (1/rho) and lam_2mu (lambda + 2*mu)
          to eliminate expensive divisions in the simulation loop.

# Arguments
- `vp`, `vs`, `rho`: Velocity and density arrays [nz, nx] (seismic convention)
- `dx`, `dz`: Grid spacing
- `nbc`: Boundary layer thickness
- `fd_order`: Finite difference order
- `backend`: Target backend (CPU_BACKEND or CUDA_BACKEND)
- `free_surface`: Enable free surface at top
"""
function init_medium(vp::Matrix, vs::Matrix, rho::Matrix, 
                     dx::Real, dz::Real, nbc::Int, fd_order::Int, 
                     backend::AbstractBackend; free_surface::Bool=true)
    
    M = fd_order ÷ 2
    pad = nbc + M
    
    # Input data is [nz, nx] (seismic convention)
    # Transpose to [nx, nz] for simulation (better cache performance)
    vp_t = permutedims(vp)
    vs_t = permutedims(vs)
    rho_t = permutedims(rho)
    
    nx_inner, nz_inner = size(vp_t)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad
    
    x_max = Float32((nx_inner - 1) * dx)
    z_max = Float32((nz_inner - 1) * dz)
    
    # Pad and compute staggered parameters
    vp_pad = _pad_array(vp_t, pad)
    vs_pad = _pad_array(vs_t, pad)
    rho_pad = _pad_array(rho_t, pad)
    
    # Compute all material properties including precomputed values
    lam, mu_txx, mu_txz, buoy_vx, buoy_vz, lam_2mu = _compute_staggered_params_optimized(vp_pad, vs_pad, rho_pad)
    
    # Move to device
    return Medium(
        nx, nz, Float32(dx), Float32(dz), x_max, z_max,
        M, pad, free_surface,
        to_device(lam, backend),
        to_device(mu_txx, backend),
        to_device(mu_txz, backend),
        to_device(buoy_vx, backend),
        to_device(buoy_vz, backend),
        to_device(lam_2mu, backend)
    )
end

"""
    init_medium(model::VelocityModel, nbc, fd_order, backend; free_surface=true)

Initialize Medium from a VelocityModel struct (loaded via load_model).

# Example
```julia
model = load_model("marmousi.jld2")
medium = init_medium(model, 50, 8, backend(:cpu))
```
"""
function init_medium(model::VelocityModel, nbc::Int, fd_order::Int, 
                     backend::AbstractBackend; free_surface::Bool=true)
    return init_medium(model.vp, model.vs, model.rho, 
                       model.dx, model.dz, nbc, fd_order, backend;
                       free_surface=free_surface)
end

function _pad_array(data::Matrix, pad::Int)
    nx, nz = size(data)
    result = zeros(Float32, nx + 2*pad, nz + 2*pad)
    result[pad+1:pad+nx, pad+1:pad+nz] .= Float32.(data)
    
    # Extend boundaries
    for i in 1:pad
        result[i, :] .= result[pad+1, :]
        result[end-i+1, :] .= result[end-pad, :]
    end
    for j in 1:pad
        result[:, j] .= result[:, pad+1]
        result[:, end-j+1] .= result[:, end-pad]
    end
    return result
end

"""
    _compute_staggered_params_optimized(vp, vs, rho)

Compute staggered grid parameters with precomputed buoyancy and lam_2mu.
OPTIMIZATION: Division by rho is done once here, not in every time step!
"""
function _compute_staggered_params_optimized(vp, vs, rho)
    nx, nz = size(vp)
    
    # Basic Lamé parameters
    mu = Float32.(rho .* vs.^2)
    lam = Float32.(rho .* vp.^2 .- 2.0f0 .* mu)
    
    # OPTIMIZED: Precompute lambda + 2*mu
    lam_2mu = Float32.(lam .+ 2.0f0 .* mu)
    
    # OPTIMIZED: Precompute buoyancy (1/rho) instead of storing rho
    # This eliminates division in the velocity update kernel!
    buoy_vx = Float32.(1.0f0 ./ rho)
    
    # Staggered buoyancy for vz (average then invert)
    buoy_vz = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz-1
        for i in 1:nx-1
            # Average rho at staggered position, then invert
            rho_avg = 0.25f0 * (rho[i,j] + rho[i+1,j] + rho[i,j+1] + rho[i+1,j+1])
            buoy_vz[i, j] = 1.0f0 / rho_avg
        end
    end
    buoy_vz[nx, :] .= buoy_vz[nx-1, :]
    buoy_vz[:, nz] .= buoy_vz[:, nz-1]
    
    # Harmonic average for mu at txz positions
    mu_txz = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz-1
        for i in 1:nx
            mu_txz[i, j] = 2.0f0 / (1.0f0/mu[i,j] + 1.0f0/mu[i,j+1])
        end
    end
    mu_txz[:, nz] .= mu_txz[:, nz-1]
    
    return lam, mu, mu_txz, buoy_vx, buoy_vz, lam_2mu
end

# ==============================================================================
# HABC Initialization (unchanged)
# ==============================================================================

"""
    init_habc(nx, nz, nbc, dt, dx, dz, v_ref, backend)

Initializes the Higdon Absorbing Boundary Condition (HABC) configuration.
Computes extrapolation coefficients and spatial blending weight matrices.
"""
function init_habc(nx::Int, nz::Int, nbc::Int, dt::Real, dx::Real, dz::Real, 
                   v_ref::Real, backend::AbstractBackend)
    
    rx = Float32(v_ref * dt / dx)
    rz = Float32(v_ref * dt / dz)
    b_p = 0.45f0
    beta = 1.0f0
    
    # Precompute extrapolation coefficients
    qx = Float32((b_p * (beta + rx) - rx) / ((beta + rx) * (1 - b_p)))
    qz = Float32((b_p * (beta + rz) - rz) / ((beta + rz) * (1 - b_p)))
    qt_x = Float32((b_p * (beta + rx) - beta) / ((beta + rx) * (1 - b_p)))
    qt_z = Float32((b_p * (beta + rz) - beta) / ((beta + rz) * (1 - b_p)))
    qxt = Float32(b_p / (b_p - 1.0f0))
    
    # Distance function for blending
    dist(i, j) = min(i - 1, nx - i, j - 1, nz - j)
    
    # Generate weighting matrices
    w_vx = [Float32(clamp((dist(i, j) - 0.0) / nbc, 0.0, 1.0)) for j in 1:nz, i in 1:nx]
    w_vz = [Float32(clamp((dist(i, j) - 0.5) / nbc, 0.0, 1.0)) for j in 1:nz, i in 1:nx]
    w_tau = [Float32(clamp((dist(i, j) - 0.75) / nbc, 0.0, 1.0)) for j in 1:nz, i in 1:nx]
    
    return HABCConfig(
        nbc, qx, qz, qt_x, qt_z, qxt,
        to_device(w_vx, backend),
        to_device(w_vz, backend),
        to_device(w_tau, backend)
    )
end

# ==============================================================================
# Receiver Setup (unchanged)
# ==============================================================================

"""
    setup_receivers(x_positions, z_positions, medium; type=:vz)

Create receiver configuration (CPU indices, data allocated later).
"""
function setup_receivers(x::Vector{<:Real}, z::Vector{<:Real}, M::Medium; type::Symbol=:vz)
    n = length(x)
    i_rec = Vector{Int}(undef, n)
    j_rec = Vector{Int}(undef, n)
    
    for r in 1:n
        i_rec[r] = round(Int, x[r] / M.dx) + M.pad + 1
        j_rec[r] = round(Int, z[r] / M.dz) + M.pad + 1
    end
    
    # Dummy data - will be replaced per shot
    data = zeros(Float32, 1, n)
    return Receivers(i_rec, j_rec, data, type)
end
