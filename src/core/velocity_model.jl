# ==============================================================================
# types/model.jl
#
# VelocityModel structure and basic operations
# ==============================================================================

"""
    VelocityModel

Standard internal representation for velocity models.

# Fields
- `vp`: P-wave velocity matrix [nz, nx]
- `vs`: S-wave velocity matrix [nz, nx]
- `rho`: Density matrix [nz, nx]
- `dx`, `dz`: Grid spacing
- `nx`, `nz`: Grid dimensions
- `x_origin`, `z_origin`: Origin coordinates
- `name`: Model name

# Note
Seismic convention: data is stored as `field[nz, nx]` (depth first).
"""
struct VelocityModel
    vp::Matrix{Float32}     # P-wave velocity
    vs::Matrix{Float32}     # S-wave velocity  
    rho::Matrix{Float32}    # Density
    dx::Float32             # Grid spacing in X
    dz::Float32             # Grid spacing in Z
    nx::Int                 # Grid points in X
    nz::Int                 # Grid points in Z
    x_origin::Float32       # X origin (default 0)
    z_origin::Float32       # Z origin (default 0)
    name::String            # Model name
end

"""
    VelocityModel(vp, vs, rho, dx, dz; x_origin=0, z_origin=0, name="unnamed")

Construct a VelocityModel with auto-computed dimensions.

# Example
```julia
vp = fill(3000.0f0, 200, 400)  # [nz, nx]
vs = fill(1800.0f0, 200, 400)
rho = fill(2200.0f0, 200, 400)

model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0; name="simple_model")
```
"""
function VelocityModel(vp, vs, rho, dx, dz; 
                       x_origin=0.0f0, z_origin=0.0f0, name="unnamed")
    nz, nx = size(vp)  # Seismic convention: depth is first dimension
    @assert size(vs) == (nz, nx) "vs size mismatch: got $(size(vs)), expected ($nz, $nx)"
    @assert size(rho) == (nz, nx) "rho size mismatch: got $(size(rho)), expected ($nz, $nx)"
    
    VelocityModel(
        Float32.(vp), Float32.(vs), Float32.(rho),
        Float32(dx), Float32(dz), nx, nz,
        Float32(x_origin), Float32(z_origin), name
    )
end

"""
    model_info(model::VelocityModel)

Print model information.
"""
function model_info(model::VelocityModel)
    println("VelocityModel: $(model.name)")
    println("  Grid: $(model.nx) × $(model.nz)")
    println("  Spacing: dx=$(model.dx)m, dz=$(model.dz)m")
    println("  Physical size: $(model.nx * model.dx)m × $(model.nz * model.dz)m")
    println("  Vp range: $(minimum(model.vp)) - $(maximum(model.vp)) m/s")
    println("  Vs range: $(minimum(model.vs)) - $(maximum(model.vs)) m/s")
    println("  Rho range: $(minimum(model.rho)) - $(maximum(model.rho)) kg/m³")
end

"""
    suggest_grid_spacing(vp_min, freq_max; ppw=10)

Suggest grid spacing based on minimum velocity and maximum frequency.

# Arguments
- `vp_min`: Minimum P-wave velocity in model
- `freq_max`: Maximum frequency of source wavelet
- `ppw`: Points per wavelength (default: 10, recommended: 8-15)

# Returns
- Suggested grid spacing
"""
function suggest_grid_spacing(vp_min::Real, freq_max::Real; ppw::Int=10)
    wavelength_min = vp_min / freq_max
    dx_suggested = wavelength_min / ppw
    return dx_suggested
end

"""
    resample_model(model::VelocityModel, new_dx, new_dz) -> VelocityModel

Resample model to new grid spacing using bilinear interpolation.
"""
function resample_model(model::VelocityModel, new_dx::Real, new_dz::Real)
    # Compute new dimensions
    Lx = (model.nx - 1) * model.dx
    Lz = (model.nz - 1) * model.dz
    
    new_nx = round(Int, Lx / new_dx) + 1
    new_nz = round(Int, Lz / new_dz) + 1
    
    # Create interpolation grids
    old_x = range(0, Lx, length=model.nx)
    old_z = range(0, Lz, length=model.nz)
    new_x = range(0, Lx, length=new_nx)
    new_z = range(0, Lz, length=new_nz)
    
    # Resample each field
    vp_new = _bilinear_resample(model.vp, old_x, old_z, new_x, new_z)
    vs_new = _bilinear_resample(model.vs, old_x, old_z, new_x, new_z)
    rho_new = _bilinear_resample(model.rho, old_x, old_z, new_x, new_z)
    
    return VelocityModel(vp_new, vs_new, rho_new, Float32(new_dx), Float32(new_dz);
                         x_origin=model.x_origin, z_origin=model.z_origin,
                         name="$(model.name)_resampled")
end

"""
Simple bilinear interpolation for 2D arrays.
"""
function _bilinear_resample(data::Matrix{Float32}, old_x, old_z, new_x, new_z)
    nz_old, nx_old = size(data)
    nz_new, nx_new = length(new_z), length(new_x)
    
    result = zeros(Float32, nz_new, nx_new)
    
    for (j, x) in enumerate(new_x)
        for (i, z) in enumerate(new_z)
            # Find surrounding indices in old grid
            fx = (x - old_x[1]) / (old_x[2] - old_x[1]) + 1
            fz = (z - old_z[1]) / (old_z[2] - old_z[1]) + 1
            
            i0 = clamp(floor(Int, fz), 1, nz_old - 1)
            j0 = clamp(floor(Int, fx), 1, nx_old - 1)
            i1 = i0 + 1
            j1 = j0 + 1
            
            # Interpolation weights
            wz = fz - i0
            wx = fx - j0
            
            # Bilinear interpolation
            result[i, j] = (1 - wz) * (1 - wx) * data[i0, j0] +
                           (1 - wz) * wx * data[i0, j1] +
                           wz * (1 - wx) * data[i1, j0] +
                           wz * wx * data[i1, j1]
        end
    end
    
    return result
end
