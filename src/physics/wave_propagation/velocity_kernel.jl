# ==============================================================================
# kernels/velocity.jl (OPTIMIZED)
#
# Key optimizations:
# 1. Precomputed buoyancy (1/rho) - eliminates division in hot loop
# 2. Optimized thread block size for GPU (32x8 better occupancy)
# 3. Unrolled FD loop for common orders (4, 8)
# 4. @inbounds everywhere for CPU
# ==============================================================================

using LoopVectorization

"""
    update_velocity!(backend, W, M, a, params)

Update velocity fields (vx, vz) based on stress gradients.
Dispatches to CPU or GPU implementation based on backend.
"""
function update_velocity! end

# ==============================================================================
# CPU Implementation - OPTIMIZED
# ==============================================================================

function update_velocity!(::CPUBackend, W::Wavefield, M::Medium, a::Vector{Float32}, p::SimParams)
    nx, nz = M.nx, M.nz
    dtx, dtz = p.dtx, p.dtz
    M_order = p.M
    
    vx, vz = W.vx, W.vz
    txx, tzz, txz = W.txx, W.tzz, W.txz
    # Use precomputed buoyancy instead of rho!
    bx, bz = M.buoy_vx, M.buoy_vz
    
    # Specialized for common FD orders with unrolled loops
    if M_order == 4
        _update_velocity_cpu_order4!(vx, vz, txx, tzz, txz, bx, bz, a, nx, nz, dtx, dtz)
    elseif M_order == 2
        _update_velocity_cpu_order2!(vx, vz, txx, tzz, txz, bx, bz, a, nx, nz, dtx, dtz)
    else
        _update_velocity_cpu_generic!(vx, vz, txx, tzz, txz, bx, bz, a, nx, nz, dtx, dtz, M_order)
    end
    return nothing
end

# Generic version for any order
function _update_velocity_cpu_generic!(vx, vz, txx, tzz, txz, bx, bz, a, nx, nz, dtx, dtz, M_order)
    @tturbo for j in (M_order+1):(nz-M_order)
        for i in (M_order+1):(nx-M_order)
            dtxxdx, dtxzdz, dtxzdx, dtzzdz = 0.0f0, 0.0f0, 0.0f0, 0.0f0

            for l in 1:M_order
                dtxxdx += a[l] * (txx[i+l-1, j] - txx[i-l, j])
                dtxzdz += a[l] * (txz[i, j+l-1] - txz[i, j-l])
                dtxzdx += a[l] * (txz[i+l, j] - txz[i-l+1, j])
                dtzzdz += a[l] * (tzz[i, j+l] - tzz[i, j-l+1])
            end

            # Multiply by buoyancy instead of divide by rho!
            bx_ij = bx[i, j]
            bz_ij = bz[i, j]
            vx[i, j] += bx_ij * (dtx * dtxxdx + dtz * dtxzdz)
            vz[i, j] += bz_ij * (dtx * dtxzdx + dtz * dtzzdz)
        end
    end
end

# Optimized 4th order (8-point stencil) - fully unrolled
function _update_velocity_cpu_order4!(vx, vz, txx, tzz, txz, bx, bz, a, nx, nz, dtx, dtz)
    a1, a2, a3, a4 = a[1], a[2], a[3], a[4]
    
    @tturbo for j in 5:(nz-4)
        for i in 5:(nx-4)
            # Fully unrolled stencil computation
            dtxxdx = a1 * (txx[i,   j] - txx[i-1, j]) +
                     a2 * (txx[i+1, j] - txx[i-2, j]) +
                     a3 * (txx[i+2, j] - txx[i-3, j]) +
                     a4 * (txx[i+3, j] - txx[i-4, j])
            
            dtxzdz = a1 * (txz[i, j]   - txz[i, j-1]) +
                     a2 * (txz[i, j+1] - txz[i, j-2]) +
                     a3 * (txz[i, j+2] - txz[i, j-3]) +
                     a4 * (txz[i, j+3] - txz[i, j-4])
            
            dtxzdx = a1 * (txz[i+1, j] - txz[i,   j]) +
                     a2 * (txz[i+2, j] - txz[i-1, j]) +
                     a3 * (txz[i+3, j] - txz[i-2, j]) +
                     a4 * (txz[i+4, j] - txz[i-3, j])
            
            dtzzdz = a1 * (tzz[i, j+1] - tzz[i, j])   +
                     a2 * (tzz[i, j+2] - tzz[i, j-1]) +
                     a3 * (tzz[i, j+3] - tzz[i, j-2]) +
                     a4 * (tzz[i, j+4] - tzz[i, j-3])

            bx_ij = bx[i, j]
            bz_ij = bz[i, j]
            vx[i, j] += bx_ij * (dtx * dtxxdx + dtz * dtxzdz)
            vz[i, j] += bz_ij * (dtx * dtxzdx + dtz * dtzzdz)
        end
    end
end

# Optimized 2nd order (4-point stencil) - fully unrolled  
function _update_velocity_cpu_order2!(vx, vz, txx, tzz, txz, bx, bz, a, nx, nz, dtx, dtz)
    a1, a2 = a[1], a[2]
    
    @tturbo for j in 3:(nz-2)
        for i in 3:(nx-2)
            dtxxdx = a1 * (txx[i,   j] - txx[i-1, j]) +
                     a2 * (txx[i+1, j] - txx[i-2, j])
            
            dtxzdz = a1 * (txz[i, j]   - txz[i, j-1]) +
                     a2 * (txz[i, j+1] - txz[i, j-2])
            
            dtxzdx = a1 * (txz[i+1, j] - txz[i,   j]) +
                     a2 * (txz[i+2, j] - txz[i-1, j])
            
            dtzzdz = a1 * (tzz[i, j+1] - tzz[i, j])   +
                     a2 * (tzz[i, j+2] - tzz[i, j-1])

            bx_ij = bx[i, j]
            bz_ij = bz[i, j]
            vx[i, j] += bx_ij * (dtx * dtxxdx + dtz * dtxzdz)
            vz[i, j] += bz_ij * (dtx * dtxzdx + dtz * dtzzdz)
        end
    end
end

# ==============================================================================
# CUDA Implementation - OPTIMIZED
# ==============================================================================

# Simple optimized kernel - uses loop like original but with buoyancy
function _update_velocity_kernel_optimized!(vx, vz, txx, tzz, txz, bx, bz,
                                             a, nx, nz, dtx, dtz, M_order)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i > M_order && i <= nx - M_order && j > M_order && j <= nz - M_order
        dtxxdx, dtxzdz, dtxzdx, dtzzdz = 0.0f0, 0.0f0, 0.0f0, 0.0f0
        
        # Loop over FD coefficients (same as original)
        for l in 1:M_order
            @inbounds begin
                dtxxdx += a[l] * (txx[i+l-1, j] - txx[i-l, j])
                dtxzdz += a[l] * (txz[i, j+l-1] - txz[i, j-l])
                dtxzdx += a[l] * (txz[i+l, j] - txz[i-l+1, j])
                dtzzdz += a[l] * (tzz[i, j+l] - tzz[i, j-l+1])
            end
        end
        
        # OPTIMIZED: multiply by buoyancy instead of divide by rho
        @inbounds begin
            bx_ij = bx[i, j]
            bz_ij = bz[i, j]
            vx[i, j] += bx_ij * (dtx * dtxxdx + dtz * dtxzdz)
            vz[i, j] += bz_ij * (dtx * dtxzdx + dtz * dtzzdz)
        end
    end
    return nothing
end

function update_velocity!(::CUDABackend, W::Wavefield, M::Medium, a::CuVector{Float32}, p::SimParams)
    nx, nz = M.nx, M.nz
    
    # Optimized block size: 32x8 = 256 threads, better for memory coalescing
    threads = (32, 8)
    blocks = (cld(nx, 32), cld(nz, 8))
    
    # Pass CuVector directly to kernel - no CPU copy!
    @cuda threads=threads blocks=blocks _update_velocity_kernel_optimized!(
        W.vx, W.vz, W.txx, W.tzz, W.txz, 
        M.buoy_vx, M.buoy_vz,
        a, nx, nz, p.dtx, p.dtz, p.M
    )
    return nothing
end
