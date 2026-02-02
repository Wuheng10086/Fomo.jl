# ==============================================================================
# initialization/medium_setup.jl (MODIFIED)
#
# Initialization utilities for Medium, HABC, etc.
# ★ MODIFIED: Removed ricker_wavelet (moved to physics/interaction/wavelet.jl)
# ==============================================================================

# ==============================================================================
# FD Coefficients
# ==============================================================================

const FD_COEFFICIENTS = Dict(
    2 => Float32[1.0],
    4 => Float32[1.125, -0.041666667],
    6 => Float32[1.171875, -0.065104167, 0.0046875],
    8 => Float32[1.1962890625, -0.079752604167, 0.0095703125, -0.000697544643],
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
# ★ ricker_wavelet has been MOVED to physics/interaction/wavelet.jl
# ==============================================================================

# ==============================================================================
# Medium Initialization - OPTIMIZED
# ==============================================================================

"""
    init_medium(vp, vs, rho, dx, dz, nbc, fd_order, backend; free_surface=true)

Initialize Medium with material properties.
"""
function init_medium(vp::Matrix, vs::Matrix, rho::Matrix,
    dx::Real, dz::Real, nbc::Int, fd_order::Int,
    backend::AbstractBackend; free_surface::Bool=true)

    M = fd_order ÷ 2
    pad = nbc + M

    vp_t = permutedims(vp)
    vs_t = permutedims(vs)
    rho_t = permutedims(rho)

    nx_inner, nz_inner = size(vp_t)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad

    x_max = Float32((nx_inner - 1) * dx)
    z_max = Float32((nz_inner - 1) * dz)

    vp_pad = _pad_array(vp_t, pad)
    vs_pad = _pad_array(vs_t, pad)
    rho_pad = _pad_array(rho_t, pad)

    lam, mu_txx, mu_txz, buoy_vx, buoy_vz, lam_2mu = _compute_staggered_params_optimized(vp_pad, vs_pad, rho_pad)

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

function init_medium_vacuum(vp::Matrix, vs::Matrix, rho::Matrix,
    dx::Real, dz::Real, nbc::Int, fd_order::Int,
    backend::AbstractBackend; surface_elevation=nothing)

    M = fd_order ÷ 2
    pad = nbc + M

    vp_t = permutedims(vp)
    vs_t = permutedims(vs)
    rho_t = permutedims(rho)

    nx_inner, nz_inner = size(vp_t)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad

    x_max = Float32((nx_inner - 1) * dx)
    z_max = Float32((nz_inner - 1) * dz)

    vp_pad = _pad_array(vp_t, pad)
    vs_pad = _pad_array(vs_t, pad)
    rho_pad = _pad_array(rho_t, pad)

    if surface_elevation !== nothing
        surface_j = setup_vacuum_formulation!(vp_pad, vs_pad, rho_pad, surface_elevation, dz, pad)
    end

    lam, mu_txx, mu_txz, buoy_vx, buoy_vz, lam_2mu = compute_staggered_params_vacuum(vp_pad, vs_pad, rho_pad)

    return Medium(
        nx, nz, Float32(dx), Float32(dz), x_max, z_max,
        M, pad, true,
        to_device(lam, backend),
        to_device(mu_txx, backend),
        to_device(mu_txz, backend),
        to_device(buoy_vx, backend),
        to_device(buoy_vz, backend),
        to_device(lam_2mu, backend)
    )
end

function init_medium(model::VelocityModel, nbc::Int, fd_order::Int,
    backend::AbstractBackend; free_surface::Bool=true)
    return init_medium(model.vp, model.vs, model.rho,
        model.dx, model.dz, nbc, fd_order, backend;
        free_surface=free_surface)
end

function _pad_array(data::Matrix, pad::Int)
    nx, nz = size(data)
    result = zeros(Float32, nx + 2 * pad, nz + 2 * pad)
    result[pad+1:pad+nx, pad+1:pad+nz] .= Float32.(data)

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

function _compute_staggered_params_optimized(vp, vs, rho)
    nx, nz = size(vp)

    mu = Float32.(rho .* vs .^ 2)
    lam = Float32.(rho .* vp .^ 2 .- 2.0f0 .* mu)
    lam_2mu = Float32.(lam .+ 2.0f0 .* mu)

    buoy_vx = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz
        for i in 1:nx-1
            rho1, rho2 = rho[i, j], rho[i+1, j]
            if rho1 == 0.0f0 && rho2 == 0.0f0
                buoy_vx[i, j] = 0.0f0
            elseif rho1 == 0.0f0
                buoy_vx[i, j] = 2.0f0 / rho2
            elseif rho2 == 0.0f0
                buoy_vx[i, j] = 2.0f0 / rho1
            else
                buoy_vx[i, j] = 2.0f0 / (rho1 + rho2)
            end
        end
    end
    buoy_vx[nx, :] .= buoy_vx[nx-1, :]

    buoy_vz = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz-1
        for i in 1:nx
            rho1, rho2 = rho[i, j], rho[i, j+1]
            if rho1 == 0.0f0 && rho2 == 0.0f0
                buoy_vz[i, j] = 0.0f0
            elseif rho1 == 0.0f0
                buoy_vz[i, j] = 2.0f0 / rho2
            elseif rho2 == 0.0f0
                buoy_vz[i, j] = 2.0f0 / rho1
            else
                buoy_vz[i, j] = 2.0f0 / (rho1 + rho2)
            end
        end
    end
    buoy_vz[:, nz] .= buoy_vz[:, nz-1]

    mu_txz = zeros(Float32, nx, nz)
    @inbounds for j in 1:nz-1
        for i in 1:nx-1
            m1, m2, m3, m4 = mu[i, j], mu[i+1, j], mu[i, j+1], mu[i+1, j+1]
            if m1 == 0.0f0 || m2 == 0.0f0 || m3 == 0.0f0 || m4 == 0.0f0
                mu_txz[i, j] = 0.0f0
            else
                mu_txz[i, j] = 4.0f0 / (1.0f0 / m1 + 1.0f0 / m2 + 1.0f0 / m3 + 1.0f0 / m4)
            end
        end
    end
    mu_txz[nx, :] .= mu_txz[nx-1, :]
    mu_txz[:, nz] .= mu_txz[:, nz-1]

    return lam, mu, mu_txz, buoy_vx, buoy_vz, lam_2mu
end

# ==============================================================================
# HABC Initialization
# ==============================================================================

function init_habc(nx::Int, nz::Int, nbc::Int, pad::Int, dt::Real, dx::Real, dz::Real,
    v_ref::Real, backend::AbstractBackend)

    rx = Float32(v_ref * dt / dx)
    rz = Float32(v_ref * dt / dz)
    b_p = 0.45f0
    beta = 1.0f0

    qx = Float32((b_p * (beta + rx) - rx) / ((beta + rx) * (1 - b_p)))
    qz = Float32((b_p * (beta + rz) - rz) / ((beta + rz) * (1 - b_p)))
    qt_x = Float32((b_p * (beta + rx) - beta) / ((beta + rx) * (1 - b_p)))
    qt_z = Float32((b_p * (beta + rz) - beta) / ((beta + rz) * (1 - b_p)))
    qxt = Float32(b_p / (b_p - 1.0f0))

    dist(i, j) = min(i - 1, nx - i, j - 1, nz - j)

    w_vx = [Float32(clamp((dist(i, j) - 0.0) / (pad - 1), 0.0, 1.0)) for j in 1:nz, i in 1:nx]
    w_vz = [Float32(clamp((dist(i, j) - 0.5) / (pad - 1), 0.0, 1.0)) for j in 1:nz, i in 1:nx]
    w_tau = [Float32(clamp((dist(i, j) - 0.75) / (pad - 1), 0.0, 1.0)) for j in 1:nz, i in 1:nx]

    return HABCConfig(
        pad - 1,
        qx, qz, qt_x, qt_z, qxt,
        to_device(w_vx, backend),
        to_device(w_vz, backend),
        to_device(w_tau, backend)
    )
end

function init_habc(nx::Int, nz::Int, nbc::Int, dt::Real, dx::Real, dz::Real,
    v_ref::Real, backend::AbstractBackend)
    pad = nbc + 4
    return init_habc(nx, nz, nbc, pad, dt, dx, dz, v_ref, backend)
end

# ==============================================================================
# Receiver Setup
# ==============================================================================

function setup_receivers(x::Vector{<:Real}, z::Vector{<:Real}, M::Medium; type::Symbol=:vz)
    n = length(x)
    i_rec = Vector{Int}(undef, n)
    j_rec = Vector{Int}(undef, n)

    for r in 1:n
        i_rec[r] = round(Int, x[r] / M.dx) + M.pad + 1
        j_rec[r] = round(Int, z[r] / M.dz) + M.pad + 1
    end

    data = zeros(Float32, 1, n)
    return Receivers(i_rec, j_rec, data, type)
end