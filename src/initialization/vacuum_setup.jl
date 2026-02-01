# ==============================================================================
# simulation/init_vacuum.jl
#
# Medium initialization with VACUUM FORMULATION for irregular free surfaces
# 
# This file extends init.jl with functions for topographic surfaces.
# Uses the improved vacuum formulation (Zeng et al., 2012).
#
# NOTE: This file depends on functions from init.jl (_pad_array, etc.)
#       Make sure init.jl is included before this file in ElasticWave2D.jl
#
# Reference:
#   Zeng et al. (2012). An improved vacuum formulation for 2D finite-difference
#   modeling of Rayleigh waves including surface topography and internal 
#   discontinuities. Geophysics, 77(1), T1-T9.
# ==============================================================================

# ==============================================================================
# Medium Initialization - VACUUM FORMULATION (Irregular Surface)
# ==============================================================================

"""
    init_medium_vacuum(vp, vs, rho, dx, dz, nbc, fd_order, backend, 
                       surface_elevation; free_surface=true)

Initialize Medium with VACUUM FORMULATION for irregular (topographic) free surface.

This uses the improved vacuum formulation (Zeng et al., 2012) which:
1. Sets material parameters to zero in the vacuum region (above surface)
2. Uses special parameter averaging that automatically satisfies τxz=0, τzz=0
   at the vacuum-solid interface without explicit boundary conditions

# Arguments
- `vp`, `vs`, `rho`: Velocity and density arrays [nz, nx] (seismic convention)
- `dx`, `dz`: Grid spacing
- `nbc`: Boundary layer thickness  
- `fd_order`: Finite difference order
- `backend`: Target backend (CPU_BACKEND or CUDA_BACKEND)
- `surface_elevation`: Surface elevation array [nx_inner] in physical units
  - Positive values: surface is ABOVE the original z=0 plane
  - Negative values: surface is BELOW the original z=0 plane
  - Zero: flat surface at z=0
- `free_surface`: Enable free surface (should be true for vacuum formulation)

# Returns
- `medium`: Medium struct with vacuum-aware parameters
- `surface_j`: Surface j-indices for each x position (for receiver placement etc.)

# Example
```julia
# Create a sinusoidal surface with 5m amplitude, 50m wavelength
nx_inner = 500
elevation = create_sinusoidal_surface(nx_inner, 5.0, 50.0, dx, dz, 0)

medium, surface_j = init_medium_vacuum(
    vp, vs, rho, dx, dz, 50, 8, CPU_BACKEND, elevation
)
```

# Note
With vacuum formulation, you do NOT need to call `apply_free_surface!()` or any
special boundary condition function in the time stepping loop. The free surface
boundary condition is automatically satisfied through the parameter averaging.

# Reference
Zeng et al. (2012). An improved vacuum formulation for 2D finite-difference
modeling of Rayleigh waves including surface topography and internal 
discontinuities. Geophysics, 77(1), T1-T9.
"""
function init_medium_vacuum(vp::Matrix, vs::Matrix, rho::Matrix,
    dx::Real, dz::Real, nbc::Int, fd_order::Int,
    backend::AbstractBackend, surface_elevation::Vector{<:Real};
    free_surface::Bool=true)

    M = fd_order ÷ 2
    pad = nbc + M

    # Input data is [nz, nx] (seismic convention)
    # Transpose to [nx, nz] for simulation
    vp_t = Float32.(permutedims(vp))
    vs_t = Float32.(permutedims(vs))
    rho_t = Float32.(permutedims(rho))

    nx_inner, nz_inner = size(vp_t)
    nx = nx_inner + 2 * pad
    nz = nz_inner + 2 * pad

    x_max = Float32((nx_inner - 1) * dx)
    z_max = Float32((nz_inner - 1) * dz)

    # Validate surface elevation
    validate_surface_elevation(surface_elevation, dz, pad, nz_inner)

    # Pad arrays (using function from init.jl)
    vp_pad = _pad_array(vp_t, pad)
    vs_pad = _pad_array(vs_t, pad)
    rho_pad = _pad_array(rho_t, pad)

    # Apply vacuum formulation: set parameters to 0 above surface
    surface_j = setup_vacuum_formulation!(vp_pad, vs_pad, rho_pad,
        Float32.(surface_elevation), dz, pad)

    # Compute staggered parameters with vacuum-aware averaging
    lam, mu_txx, mu_txz, buoy_vx, buoy_vz, lam_2mu = compute_staggered_params_vacuum(
        vp_pad, vs_pad, rho_pad)

    # Create Medium
    medium = Medium(
        nx, nz, Float32(dx), Float32(dz), x_max, z_max,
        M, pad, free_surface,
        to_device(lam, backend),
        to_device(mu_txx, backend),
        to_device(mu_txz, backend),
        to_device(buoy_vx, backend),
        to_device(buoy_vz, backend),
        to_device(lam_2mu, backend)
    )

    return medium, surface_j
end

"""
    init_medium_vacuum(model::VelocityModel, nbc, fd_order, backend, 
                       surface_elevation; free_surface=true)

Initialize Medium with vacuum formulation from a VelocityModel struct.
"""
function init_medium_vacuum(model::VelocityModel, nbc::Int, fd_order::Int,
    backend::AbstractBackend, surface_elevation::Vector{<:Real};
    free_surface::Bool=true)

    return init_medium_vacuum(model.vp, model.vs, model.rho,
        model.dx, model.dz, nbc, fd_order, backend, surface_elevation;
        free_surface=free_surface)
end

# ==============================================================================
# Receiver/Source Setup on Irregular Surface
# ==============================================================================

"""
    setup_receivers_on_surface(x_positions, surface_j, medium; type=:vz, z_offset=1)

Create receiver configuration with receivers placed ON the irregular surface.

# Arguments
- `x_positions`: X coordinates for receivers [n_rec]
- `surface_j`: Surface j-indices from init_medium_vacuum [nx]
- `medium`: Medium struct
- `type`: Recording type (:vx or :vz)
- `z_offset`: Offset from surface in grid points (positive = into ground)

# Example
```julia
medium, surface_j = init_medium_vacuum(...)
x_rec = collect(10.0:1.0:100.0)  # Receivers from x=10 to x=100
receivers = setup_receivers_on_surface(x_rec, surface_j, medium)
```
"""
function setup_receivers_on_surface(x::Vector{<:Real}, surface_j::Vector{Int},
    M::Medium; type::Symbol=:vz, z_offset::Int=1)
    n = length(x)
    i_rec = Vector{Int}(undef, n)
    j_rec = Vector{Int}(undef, n)

    for r in 1:n
        i_rec[r] = round(Int, x[r] / M.dx) + M.pad + 1

        # Clamp i_rec to valid range for surface_j lookup
        i_lookup = clamp(i_rec[r], 1, length(surface_j))

        # Place receiver on surface (or slightly below)
        j_rec[r] = surface_j[i_lookup] + z_offset
    end

    data = zeros(Float32, 1, n)
    return Receivers(i_rec, j_rec, data, type)
end

"""
    setup_source_on_surface(x_pos, surface_j, medium; z_offset=1)

Get source position indices with source placed on the irregular surface.

# Arguments
- `x_pos`: X coordinate of source
- `surface_j`: Surface j-indices from init_medium_vacuum
- `medium`: Medium struct
- `z_offset`: Offset from surface in grid points (positive = into ground)

# Returns
- `(i_src, j_src)`: Grid indices for source position
"""
function setup_source_on_surface(x_pos::Real, surface_j::Vector{Int}, M::Medium; z_offset::Int=1)
    i_src = round(Int, x_pos / M.dx) + M.pad + 1
    i_lookup = clamp(i_src, 1, length(surface_j))
    j_src = surface_j[i_lookup] + z_offset

    return i_src, j_src
end

# ==============================================================================
# Wavefield Initialization Helpers
# ==============================================================================

"""
    init_wavefield(nx, nz, backend)

Create zero-initialized wavefield. Convenience wrapper for Wavefield constructor.
"""
function init_wavefield(nx::Int, nz::Int, backend::AbstractBackend)
    return Wavefield(nx, nz, backend)
end

"""
    init_wavefield(medium, backend)

Create zero-initialized wavefield matching medium dimensions.
"""
function init_wavefield(M::Medium, backend::AbstractBackend)
    return Wavefield(M.nx, M.nz, backend)
end