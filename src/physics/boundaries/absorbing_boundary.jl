# ==============================================================================
# kernels/boundary.jl (OPTIMIZED)
#
# Boundary condition kernels - HABC and Free Surface
# OPTIMIZATIONS:
# 1. Better memory access patterns
# 2. Vectorized CPU operations with @simd
# 3. Optimized GPU thread configuration
# ==============================================================================

using LoopVectorization

# ==============================================================================
# Boundary Strip Backup - OPTIMIZED with vectorization
# ==============================================================================

"""
    copy_boundary_strip!(old, new, nbc, nx, nz, is_free_surface)

Backup boundary field values into `old` for HABC extrapolation.
OPTIMIZED with explicit @simd and @inbounds
"""
function copy_boundary_strip!(old, new, nbc, nx, nz, is_free_surface)
    j_top = 1 # 始终从顶部开始备份，确保即使有自由表面，顶部的 HABC 也能工作

    # Vertical strips (Left/Right) - vectorized
    @inbounds for j in j_top:nz
        @simd for i in 1:nbc+2
            old[i, j] = new[i, j]
        end
        @simd for i in (nx-nbc-1):nx
            old[i, j] = new[i, j]
        end
    end

    # Horizontal strips (Top/Bottom) - vectorized
    @inbounds for j in j_top:nbc+2
        @simd for i in nbc+3:nx-nbc-2
            old[i, j] = new[i, j]
        end
    end
    @inbounds for j in (nz-nbc-1):nz
        @simd for i in nbc+3:nx-nbc-2
            old[i, j] = new[i, j]
        end
    end
end

"""
    backup_boundary!(backend, W, H, M)

Backup all wavefield boundary values.
"""
function backup_boundary!(::CPUBackend, W::Wavefield, H::HABCConfig, M::Medium)
    nx, nz = M.nx, M.nz
    nbc = H.nbc
    is_fs = M.is_free_surface

    copy_boundary_strip!(W.vx_old, W.vx, nbc, nx, nz, is_fs)
    copy_boundary_strip!(W.vz_old, W.vz, nbc, nx, nz, is_fs)
    copy_boundary_strip!(W.txx_old, W.txx, nbc, nx, nz, is_fs)
    copy_boundary_strip!(W.tzz_old, W.tzz, nbc, nx, nz, is_fs)
    copy_boundary_strip!(W.txz_old, W.txz, nbc, nx, nz, is_fs)
    return nothing
end

# ==============================================================================
# HABC Application - OPTIMIZED
# ==============================================================================

"""
    apply_habc!(f, f_old, H, weights, nx, nz, is_free_surface)

Apply Higdon Absorbing Boundary Conditions (HABC) to field `f`.

## 物理原理：
1. 先处理四条边（1D HABC）：从内部向边界外推
2. 再处理四个角（2D HABC）：用相邻两条边的外推结果平均

## 边界区域划分（确保 Edges 不覆盖 Corners）：
- Left Edge:   i = 2:nbc+1,      j = nbc+2:nz-nbc-1  (不含Top/Bottom角)
- Right Edge:  i = nx-nbc:nx-1,  j = nbc+2:nz-nbc-1  (不含Top/Bottom角)
- Top Edge:    j = 2:nbc+1,      i = nbc+2:nx-nbc-1  (不含Left/Right角)
- Bottom Edge: j = nz-nbc:nz-1,  i = nbc+2:nx-nbc-1  (不含Left/Right角)
- Corners: 四个角区域
"""
function apply_habc!(f, f_old, H, weights, nx, nz, is_free_surface)
    nbc = H.nbc
    qx, qz, qt_x, qt_z, qxt = H.qx, H.qz, H.qt_x, H.qt_z, H.qxt

    # 边界索引定义
    i_left_start, i_left_end = 2, nbc + 1
    i_right_start, i_right_end = nx - nbc, nx - 1
    j_top_start, j_top_end = 2, nbc + 1
    j_bot_start, j_bot_end = nz - nbc, nz - 1

    # Edge 的非角落部分（完全对称）
    i_edge_start, i_edge_end = nbc + 2, nx - nbc - 1
    j_edge_start, j_edge_end = nbc + 2, nz - nbc - 1

    # =========================================================================
    # 1. 先处理四条边 (1D HABC) - 不含角落
    # =========================================================================

    # Left Edge (不含角落)
    @inbounds for i in i_left_start:i_left_end
        @simd for j in j_edge_start:j_edge_end
            sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
            w = weights[j, i]
            f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_x
        end
    end

    # Right Edge (不含角落) - 从外向内处理，确保读取的f[i-1,j]未被修改
    @inbounds for i in i_right_end:-1:i_right_start  # 反向循环！
        @simd for j in j_edge_start:j_edge_end
            sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
            w = weights[j, i]
            f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_x
        end
    end

    # Bottom Edge (不含角落) - 从外向内处理，确保读取的f[i,j-1]未被修改
    @inbounds for j in j_bot_end:-1:j_bot_start  # 反向循环！
        @simd for i in i_edge_start:i_edge_end
            sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
            w = weights[j, i]
            f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_z
        end
    end

    # Top Edge (不含角落)
    @inbounds for j in j_top_start:j_top_end
        @simd for i in i_edge_start:i_edge_end
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            w = weights[j, i]
            f[i, j] = w * f[i, j] + (1.0f0 - w) * sum_z
        end
    end

    # =========================================================================
    # 2. 再处理四个角落 (2D HABC) - 使用相邻两边的外推结果平均
    # =========================================================================

    # Left-Bottom Corner
    @inbounds for j in j_bot_start:j_bot_end
        @simd for i in i_left_start:i_left_end
            sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
            sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
            w = weights[j, i]
            f[i, j] = w * f[i, j] + (1.0f0 - w) * 0.5f0 * (sum_x + sum_z)
        end
    end

    # Right-Bottom Corner
    @inbounds for j in j_bot_start:j_bot_end
        @simd for i in i_right_start:i_right_end
            sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
            sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
            w = weights[j, i]
            f[i, j] = w * f[i, j] + (1.0f0 - w) * 0.5f0 * (sum_x + sum_z)
        end
    end

    # Left-Top Corner
    @inbounds for j in j_top_start:j_top_end
        @simd for i in i_left_start:i_left_end
            sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            w = weights[j, i]
            f[i, j] = w * f[i, j] + (1.0f0 - w) * 0.5f0 * (sum_x + sum_z)
        end
    end

    # Right-Top Corner
    @inbounds for j in j_top_start:j_top_end
        @simd for i in i_right_start:i_right_end
            sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
            sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
            w = weights[j, i]
            f[i, j] = w * f[i, j] + (1.0f0 - w) * 0.5f0 * (sum_x + sum_z)
        end
    end
end

"""
    apply_habc_velocity!(backend, W, H, M)

Apply HABC to velocity fields.
"""
function apply_habc_velocity!(::CPUBackend, W::Wavefield, H::HABCConfig, M::Medium)
    nx, nz = M.nx, M.nz
    apply_habc!(W.vx, W.vx_old, H, H.w_vx, nx, nz, M.is_free_surface)
    apply_habc!(W.vz, W.vz_old, H, H.w_vz, nx, nz, M.is_free_surface)
    return nothing
end

"""
    apply_habc_stress!(backend, W, H, M)

Apply HABC to stress fields.
"""
function apply_habc_stress!(::CPUBackend, W::Wavefield, H::HABCConfig, M::Medium)
    nx, nz = M.nx, M.nz
    apply_habc!(W.txx, W.txx_old, H, H.w_tau, nx, nz, M.is_free_surface)
    apply_habc!(W.tzz, W.tzz_old, H, H.w_tau, nx, nz, M.is_free_surface)
    apply_habc!(W.txz, W.txz_old, H, H.w_tau, nx, nz, M.is_free_surface)
    return nothing
end

# ==============================================================================
# GPU Implementations - OPTIMIZED
# ==============================================================================

function _backup_boundary_kernel_optimized!(vx, vx_old, vz, vz_old, txx, txx_old,
    tzz, tzz_old, txz, txz_old,
    nx, nz, nbc, is_free_surface)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= nx && j <= nz
        j_top = 1 # 始终备份顶部
        if j >= j_top
            # Check if we're in a boundary strip region
            is_strip = (i <= nbc + 2) || (i >= nx - nbc - 1) ||
                       (j <= nbc + 2) || (j >= nz - nbc - 1)
            if is_strip
                @inbounds begin
                    vx_old[i, j] = vx[i, j]
                    vz_old[i, j] = vz[i, j]
                    txx_old[i, j] = txx[i, j]
                    tzz_old[i, j] = tzz[i, j]
                    txz_old[i, j] = txz[i, j]
                end
            end
        end
    end
    return nothing
end

function backup_boundary!(::CUDABackend, W::Wavefield, H::HABCConfig, M::Medium)
    nx, nz = M.nx, M.nz
    nbc = H.nbc
    is_fs = M.is_free_surface

    # Optimized block size
    threads = (32, 8)
    blocks = (cld(nx, 32), cld(nz, 8))

    @cuda threads = threads blocks = blocks _backup_boundary_kernel_optimized!(
        W.vx, W.vx_old, W.vz, W.vz_old, W.txx, W.txx_old,
        W.tzz, W.tzz_old, W.txz, W.txz_old,
        nx, nz, nbc, is_fs
    )
    return nothing
end

function apply_habc_velocity!(::CUDABackend, W::Wavefield, H::HABCConfig, M::Medium)
    _apply_habc_gpu!(W.vx, W.vx_old, H, H.w_vx, M)
    _apply_habc_gpu!(W.vz, W.vz_old, H, H.w_vz, M)
    return nothing
end

function apply_habc_stress!(::CUDABackend, W::Wavefield, H::HABCConfig, M::Medium)
    _apply_habc_gpu!(W.txx, W.txx_old, H, H.w_tau, M)
    _apply_habc_gpu!(W.tzz, W.tzz_old, H, H.w_tau, M)
    _apply_habc_gpu!(W.txz, W.txz_old, H, H.w_tau, M)
    return nothing
end

function _apply_habc_gpu!(f, f_old, H, weights, M)
    nx, nz = M.nx, M.nz
    threads = (32, 8)
    blocks = (cld(nx, 32), cld(nz, 8))

    # GPU 需要分两次调用，按正确的物理顺序：
    # 1. 先处理 Edges (1D外推)
    # 2. 再处理 Corners (用相邻两边的结果平均)

    @cuda threads = threads blocks = blocks _habc_edges_kernel!(
        f, f_old, weights, H.qx, H.qz, H.qt_x, H.qt_z, H.qxt,
        H.nbc, nx, nz, M.is_free_surface
    )

    @cuda threads = threads blocks = blocks _habc_corners_kernel!(
        f, f_old, weights, H.qx, H.qz, H.qt_x, H.qt_z, H.qxt,
        H.nbc, nx, nz, M.is_free_surface
    )
end

"""
GPU Kernel: 处理四条边 (1D HABC) - 不含角落，先执行
"""
function _habc_edges_kernel!(f, f_old, w, qx, qz, qt_x, qt_z, qxt, nbc, nx, nz, is_free_surface)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i > 1 && i < nx && j > 1 && j < nz
        in_left = i <= nbc + 1
        in_right = i >= nx - nbc
        in_bottom = j >= nz - nbc
        in_top = j <= nbc + 1

        # 只处理边缘区域（不含角落）
        is_left_edge = in_left && !in_bottom && !in_top
        is_right_edge = in_right && !in_bottom && !in_top
        is_bottom_edge = in_bottom && !in_left && !in_right
        is_top_edge = in_top && !in_left && !in_right

        if is_left_edge || is_right_edge || is_bottom_edge || is_top_edge
            wt = w[j, i]
            one_minus_wt = 1.0f0 - wt

            @inbounds begin
                if is_left_edge
                    sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
                    f[i, j] = wt * f[i, j] + one_minus_wt * sum_x
                elseif is_right_edge
                    sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
                    f[i, j] = wt * f[i, j] + one_minus_wt * sum_x
                elseif is_bottom_edge
                    sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
                    f[i, j] = wt * f[i, j] + one_minus_wt * sum_z
                elseif is_top_edge
                    sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
                    f[i, j] = wt * f[i, j] + one_minus_wt * sum_z
                end
            end
        end
    end
    return nothing
end

"""
GPU Kernel: 处理四个角落 (2D HABC) - 用相邻两边的结果平均，后执行
"""
function _habc_corners_kernel!(f, f_old, w, qx, qz, qt_x, qt_z, qxt, nbc, nx, nz, is_free_surface)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i > 1 && i < nx && j > 1 && j < nz
        in_left = i <= nbc + 1
        in_right = i >= nx - nbc
        in_bottom = j >= nz - nbc
        in_top = j <= nbc + 1

        # 只处理角落区域
        is_corner = (in_left || in_right) && (in_top || in_bottom)

        if is_corner
            wt = w[j, i]
            one_minus_wt = 1.0f0 - wt

            @inbounds begin
                if in_left && in_bottom
                    sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
                    sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
                    f[i, j] = wt * f[i, j] + one_minus_wt * 0.5f0 * (sum_x + sum_z)
                elseif in_right && in_bottom
                    sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
                    sum_z = -qz * f[i, j-1] - qt_z * f_old[i, j] - qxt * f_old[i, j-1]
                    f[i, j] = wt * f[i, j] + one_minus_wt * 0.5f0 * (sum_x + sum_z)
                elseif in_left && in_top
                    sum_x = -qx * f[i+1, j] - qt_x * f_old[i, j] - qxt * f_old[i+1, j]
                    sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
                    f[i, j] = wt * f[i, j] + one_minus_wt * 0.5f0 * (sum_x + sum_z)
                elseif in_right && in_top
                    sum_x = -qx * f[i-1, j] - qt_x * f_old[i, j] - qxt * f_old[i-1, j]
                    sum_z = -qz * f[i, j+1] - qt_z * f_old[i, j] - qxt * f_old[i, j+1]
                    f[i, j] = wt * f[i, j] + one_minus_wt * 0.5f0 * (sum_x + sum_z)
                end
            end
        end
    end
    return nothing
end

