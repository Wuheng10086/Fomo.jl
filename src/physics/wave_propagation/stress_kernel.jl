# ==============================================================================
# kernels/stress.jl (OPTIMIZED)
#
# Stress update kernel - optimized version
# Key optimizations:
# 1. Unrolled FD loops for common orders
# 2. Better thread block configuration for GPU
# 3. Precomputed (lambda + 2*mu) factor
# ==============================================================================

"""
    update_stress!(backend, W, M, a, params)

Update stress fields (txx, tzz, txz) based on velocity gradients.
"""
function update_stress! end

# ==============================================================================
# CPU Implementation - OPTIMIZED
# ==============================================================================

function update_stress!(::CPUBackend, W::Wavefield, M::Medium, a::Vector{Float32}, p::SimParams)
    nx, nz = M.nx, M.nz
    dtx, dtz = p.dtx, p.dtz
    M_order = p.M
    
    vx, vz = W.vx, W.vz
    txx, tzz, txz = W.txx, W.tzz, W.txz
    lam, mu_txx, mu_txz = M.lam, M.mu_txx, M.mu_txz
    
    # Use precomputed lam_2mu
    lam_2mu = M.lam_2mu
    
    if M_order == 4
        _update_stress_cpu_order4!(txx, tzz, txz, vx, vz, lam, lam_2mu, mu_txz, a, nx, nz, dtx, dtz)
    elseif M_order == 2
        _update_stress_cpu_order2!(txx, tzz, txz, vx, vz, lam, lam_2mu, mu_txz, a, nx, nz, dtx, dtz)
    else
        _update_stress_cpu_generic!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
    end
    return nothing
end

# Generic version
function _update_stress_cpu_generic!(txx, tzz, txz, vx, vz, lam, mu_txx, mu_txz, a, nx, nz, dtx, dtz, M_order)
    @tturbo for j in (M_order+1):(nz-M_order)
        for i in (M_order+1):(nx-M_order)
            dvxdx, dvzdz, dvxdz, dvzdx = 0.0f0, 0.0f0, 0.0f0, 0.0f0

            for l in 1:M_order
                dvxdx += a[l] * (vx[i+l, j] - vx[i-l+1, j])
                dvzdz += a[l] * (vz[i, j+l-1] - vz[i, j-l])
                dvxdz += a[l] * (vx[i, j+l] - vx[i, j-l+1])
                dvzdx += a[l] * (vz[i+l-1, j] - vz[i-l, j])
            end

            l_val = lam[i, j]
            m_val = mu_txx[i, j]
            lam_2mu_val = l_val + 2.0f0 * m_val

            txx[i, j] += lam_2mu_val * (dvxdx * dtx) + l_val * (dvzdz * dtz)
            tzz[i, j] += l_val * (dvxdx * dtx) + lam_2mu_val * (dvzdz * dtz)
            txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
        end
    end
end

# Optimized 4th order - fully unrolled with precomputed lam+2mu
function _update_stress_cpu_order4!(txx, tzz, txz, vx, vz, lam, lam_2mu, mu_txz, a, nx, nz, dtx, dtz)
    a1, a2, a3, a4 = a[1], a[2], a[3], a[4]
    
    @tturbo for j in 5:(nz-4)
        for i in 5:(nx-4)
            # Fully unrolled velocity gradients
            dvxdx = a1 * (vx[i+1, j] - vx[i,   j]) +
                    a2 * (vx[i+2, j] - vx[i-1, j]) +
                    a3 * (vx[i+3, j] - vx[i-2, j]) +
                    a4 * (vx[i+4, j] - vx[i-3, j])
            
            dvzdz = a1 * (vz[i, j]   - vz[i, j-1]) +
                    a2 * (vz[i, j+1] - vz[i, j-2]) +
                    a3 * (vz[i, j+2] - vz[i, j-3]) +
                    a4 * (vz[i, j+3] - vz[i, j-4])
            
            dvxdz = a1 * (vx[i, j+1] - vx[i, j])   +
                    a2 * (vx[i, j+2] - vx[i, j-1]) +
                    a3 * (vx[i, j+3] - vx[i, j-2]) +
                    a4 * (vx[i, j+4] - vx[i, j-3])
            
            dvzdx = a1 * (vz[i,   j] - vz[i-1, j]) +
                    a2 * (vz[i+1, j] - vz[i-2, j]) +
                    a3 * (vz[i+2, j] - vz[i-3, j]) +
                    a4 * (vz[i+3, j] - vz[i-4, j])

            # Use precomputed lam_2mu
            l_val = lam[i, j]
            lam_2mu_val = lam_2mu[i, j]

            dvxdx_dtx = dvxdx * dtx
            dvzdz_dtz = dvzdz * dtz
            
            txx[i, j] += lam_2mu_val * dvxdx_dtx + l_val * dvzdz_dtz
            tzz[i, j] += l_val * dvxdx_dtx + lam_2mu_val * dvzdz_dtz
            txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
        end
    end
end

# Optimized 2nd order
function _update_stress_cpu_order2!(txx, tzz, txz, vx, vz, lam, lam_2mu, mu_txz, a, nx, nz, dtx, dtz)
    a1, a2 = a[1], a[2]
    
    @tturbo for j in 3:(nz-2)
        for i in 3:(nx-2)
            dvxdx = a1 * (vx[i+1, j] - vx[i,   j]) +
                    a2 * (vx[i+2, j] - vx[i-1, j])
            
            dvzdz = a1 * (vz[i, j]   - vz[i, j-1]) +
                    a2 * (vz[i, j+1] - vz[i, j-2])
            
            dvxdz = a1 * (vx[i, j+1] - vx[i, j])   +
                    a2 * (vx[i, j+2] - vx[i, j-1])
            
            dvzdx = a1 * (vz[i,   j] - vz[i-1, j]) +
                    a2 * (vz[i+1, j] - vz[i-2, j])

            l_val = lam[i, j]
            lam_2mu_val = lam_2mu[i, j]

            dvxdx_dtx = dvxdx * dtx
            dvzdz_dtz = dvzdz * dtz
            
            txx[i, j] += lam_2mu_val * dvxdx_dtx + l_val * dvzdz_dtz
            tzz[i, j] += l_val * dvxdx_dtx + lam_2mu_val * dvzdz_dtz
            txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
        end
    end
end

# ==============================================================================
# CUDA Implementation - OPTIMIZED
# ==============================================================================

function _update_stress_kernel_optimized!(txx, tzz, txz, vx, vz, lam, lam_2mu, mu_txz,
                                          a, nx, nz, dtx, dtz, M_order)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i > M_order && i <= nx - M_order && j > M_order && j <= nz - M_order
        dvxdx, dvzdz, dvxdz, dvzdx = 0.0f0, 0.0f0, 0.0f0, 0.0f0
        
        # Loop over FD coefficients (same as original)
        for l in 1:M_order
            @inbounds begin
                dvxdx += a[l] * (vx[i+l, j] - vx[i-l+1, j])
                dvzdz += a[l] * (vz[i, j+l-1] - vz[i, j-l])
                dvxdz += a[l] * (vx[i, j+l] - vx[i, j-l+1])
                dvzdx += a[l] * (vz[i+l-1, j] - vz[i-l, j])
            end
        end
        
        # OPTIMIZED: use precomputed lam_2mu
        @inbounds begin
            l_val = lam[i, j]
            lam_2mu_val = lam_2mu[i, j]
            
            dvxdx_dtx = dvxdx * dtx
            dvzdz_dtz = dvzdz * dtz
            
            txx[i, j] += lam_2mu_val * dvxdx_dtx + l_val * dvzdz_dtz
            tzz[i, j] += l_val * dvxdx_dtx + lam_2mu_val * dvzdz_dtz
            txz[i, j] += mu_txz[i, j] * (dvxdz * dtz + dvzdx * dtx)
        end
    end
    return nothing
end

function update_stress!(::CUDABackend, W::Wavefield, M::Medium, a::CuVector{Float32}, p::SimParams)
    nx, nz = M.nx, M.nz
    
    # Optimized block size
    threads = (32, 8)
    blocks = (cld(nx, 32), cld(nz, 8))
    
    # Pass CuVector directly - no CPU copy!
    @cuda threads=threads blocks=blocks _update_stress_kernel_optimized!(
        W.txx, W.tzz, W.txz, W.vx, W.vz, 
        M.lam, M.lam_2mu, M.mu_txz,
        a, nx, nz, p.dtx, p.dtz, p.M
    )
    return nothing
end
