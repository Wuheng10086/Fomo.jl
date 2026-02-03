# ==============================================================================
# free_surface_boundary.jl
#
# Free Surface Boundary Conditions using Image Method
# ==============================================================================

using LoopVectorization

"""
    apply_free_surface_velocity!(backend, W, M)

Apply symmetry/antisymmetry mirroring to velocity fields for Free Surface.
Mirrors the entire padding region to provide a consistent field for HABC.
"""
function apply_free_surface_velocity! end

"""
    apply_free_surface_stress!(backend, W, M)

Apply symmetry/antisymmetry mirroring to stress fields for Free Surface.
Sets tzz=0 at the surface.
"""
function apply_free_surface_stress! end

# ==============================================================================
# CPU Implementation
# ==============================================================================

function apply_free_surface_velocity!(::CPUBackend, W::Wavefield, M::Medium)
    if !M.is_free_surface
        return nothing
    end

    nx = M.nx
    j_fs = M.pad + 1
    mirror_depth = M.pad # 镜像整个 padding 区域，确保顶部 HABC 正常工作

    @inbounds for l in 1:mirror_depth
        j_air = j_fs - l
        # vz at j is j+1/2. j_fs is the symmetry center (z=0)
        j_rock_vz = j_fs + l - 1
        j_rock_vx = j_fs + l

        @simd for i in 1:nx
            W.vx[i, j_air] = W.vx[i, j_rock_vx]
            W.vz[i, j_air] = W.vz[i, j_rock_vz]
        end
    end
    return nothing
end

function apply_free_surface_stress!(::CPUBackend, W::Wavefield, M::Medium)
    if !M.is_free_surface
        return nothing
    end

    nx = M.nx
    j_fs = M.pad + 1
    mirror_depth = M.pad

    @inbounds for l in 1:mirror_depth
        j_air = j_fs - l
        # txz at j is j+1/2. j_fs is the antisymmetry center
        j_rock_txz = j_fs + l - 1
        j_rock_tau = j_fs + l

        @simd for i in 1:nx
            W.txx[i, j_air] = -W.txx[i, j_rock_tau]
            W.tzz[i, j_air] = -W.tzz[i, j_rock_tau]
            W.txz[i, j_air] = -W.txz[i, j_rock_txz]
        end
    end

    # Traction-free condition at z=0 (j_fs)
    @inbounds @simd for i in 1:nx
        W.tzz[i, j_fs] = 0.0f0
    end
    return nothing
end

# ==============================================================================
# GPU Implementation
# ==============================================================================

function apply_free_surface_velocity!(::CUDABackend, W::Wavefield, M::Medium)
    if !M.is_free_surface
        return nothing
    end

    nx = M.nx
    j_fs = M.pad + 1
    mirror_depth = M.pad

    threads = (256, 1)
    blocks = (cld(nx, 256), mirror_depth)

    @cuda threads=threads blocks=blocks _fs_velocity_kernel!(W.vx, W.vz, nx, j_fs)
    return nothing
end

function _fs_velocity_kernel!(vx, vz, nx, j_fs)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    l = blockIdx().y # 1 to M.pad

    if i <= nx
        j_air = j_fs - l
        j_rock_vz = j_fs + l - 1
        j_rock_vx = j_fs + l
        
        @inbounds vx[i, j_air] = vx[i, j_rock_vx]
        @inbounds vz[i, j_air] = vz[i, j_rock_vz]
    end
    return nothing
end

function apply_free_surface_stress!(::CUDABackend, W::Wavefield, M::Medium)
    if !M.is_free_surface
        return nothing
    end

    nx = M.nx
    j_fs = M.pad + 1
    mirror_depth = M.pad

    threads = (256, 1)
    blocks = (cld(nx, 256), mirror_depth + 1) # +1 for j_fs

    @cuda threads=threads blocks=blocks _fs_stress_kernel!(W.txx, W.tzz, W.txz, nx, j_fs)
    return nothing
end

function _fs_stress_kernel!(txx, tzz, txz, nx, j_fs)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    l = blockIdx().y - 1 # 0 to M.pad

    if i <= nx
        if l == 0
            @inbounds tzz[i, j_fs] = 0.0f0
        else
            j_air = j_fs - l
            j_rock_txz = j_fs + l - 1
            j_rock_tau = j_fs + l

            @inbounds txx[i, j_air] = -txx[i, j_rock_tau]
            @inbounds tzz[i, j_air] = -tzz[i, j_rock_tau]
            @inbounds txz[i, j_air] = -txz[i, j_rock_txz]
        end
    end
    return nothing
end
