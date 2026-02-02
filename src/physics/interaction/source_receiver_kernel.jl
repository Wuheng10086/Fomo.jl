# ==============================================================================
# kernels/source_receiver.jl
#
# Source injection and receiver recording kernels
# ==============================================================================

# ==============================================================================
# Source Injection
# ==============================================================================

"""
    inject_source!(backend, W, source, k, dt)

Inject source wavelet at time step k.
"""
function inject_source! end

function inject_source!(::CPUBackend, W::Wavefield, src::Source, k::Int, dt::Float32)
    if k > length(src.wavelet)
        return nothing
    end

    wav = src.wavelet[k]
    i, j = src.i, src.j

    # Pressure source (explosion)
    W.txx[i, j] += wav
    W.tzz[i, j] += wav
    return nothing
end

function inject_source!(::CPUBackend, W::Wavefield, src::StressSource, k::Int, dt::Float32)
    if k > length(src.wavelet)
        return nothing
    end

    wav = src.wavelet[k]
    i, j = src.i, src.j

    if src.component == :txx
        W.txx[i, j] += wav
    elseif src.component == :tzz
        W.tzz[i, j] += wav
    elseif src.component == :txz
        W.txz[i, j] += wav
    else
        error("Unknown stress component: $(src.component). Use :txx, :tzz, or :txz")
    end

    return nothing
end

function _inject_source_kernel!(txx, tzz, wavelet, i0, j0, k)
    if threadIdx().x == 1 && blockIdx().x == 1
        wav = wavelet[k]
        @inbounds begin
            txx[i0, j0] += wav
            tzz[i0, j0] += wav
        end
    end
    return nothing
end

function _inject_field_at_kernel!(field, wavelet, i0, j0, k)
    if threadIdx().x == 1 && blockIdx().x == 1
        wav = wavelet[k]
        @inbounds field[i0, j0] += wav
    end
    return nothing
end

function inject_source!(::CUDABackend, W::Wavefield, src::Source, k::Int, dt::Float32)
    if k > length(src.wavelet)
        return nothing
    end

    @cuda threads = 1 blocks = 1 _inject_source_kernel!(
        W.txx, W.tzz, src.wavelet, src.i, src.j, k
    )
    return nothing
end

function inject_source!(::CUDABackend, W::Wavefield, src::StressSource, k::Int, dt::Float32)
    if k > length(src.wavelet)
        return nothing
    end

    if src.component == :txx
        @cuda threads = 1 blocks = 1 _inject_field_at_kernel!(W.txx, src.wavelet, src.i, src.j, k)
    elseif src.component == :tzz
        @cuda threads = 1 blocks = 1 _inject_field_at_kernel!(W.tzz, src.wavelet, src.i, src.j, k)
    elseif src.component == :txz
        @cuda threads = 1 blocks = 1 _inject_field_at_kernel!(W.txz, src.wavelet, src.i, src.j, k)
    else
        error("Unknown stress component: $(src.component). Use :txx, :tzz, or :txz")
    end

    return nothing
end

# ------------------------------------------------------------------------------
# ForceSource (body force)
#
# Physics: ρ ∂v/∂t = ∇·σ + f
# Discrete: v += dt * f(t) * buoyancy    (buoyancy = 1/ρ)
# ------------------------------------------------------------------------------

function inject_source!(::CPUBackend, W::Wavefield, src::ForceSource, k::Int, dt::Float32)
    if k > length(src.wavelet)
        return nothing
    end

    wav = src.wavelet[k]
    i, j = src.i, src.j
    scale = dt * src.buoyancy_at_src

    if src.component == :vz
        W.vz[i, j] += wav * scale
    elseif src.component == :vx
        W.vx[i, j] += wav * scale
    else
        error("Unknown velocity component: $(src.component). Use :vx or :vz")
    end

    return nothing
end

function _inject_force_kernel!(field, wavelet, i0, j0, k, scale)
    if threadIdx().x == 1 && blockIdx().x == 1
        wav = wavelet[k]
        @inbounds field[i0, j0] += wav * scale
    end
    return nothing
end

function inject_source!(::CUDABackend, W::Wavefield, src::ForceSource, k::Int, dt::Float32)
    if k > length(src.wavelet)
        return nothing
    end

    scale = dt * src.buoyancy_at_src

    if src.component == :vz
        @cuda threads = 1 blocks = 1 _inject_force_kernel!(
            W.vz, src.wavelet, src.i, src.j, k, scale
        )
    elseif src.component == :vx
        @cuda threads = 1 blocks = 1 _inject_force_kernel!(
            W.vx, src.wavelet, src.i, src.j, k, scale
        )
    else
        error("Unknown velocity component: $(src.component). Use :vx or :vz")
    end

    return nothing
end

# Additional method for direct injection to specific field (used in vacuum formulation)
function inject_source!(field::AbstractMatrix{Float32}, i::Int, j::Int, value::Float32, ::CPUBackend)
    field[i, j] += value
    return nothing
end

function _inject_field_kernel!(field, i0, j0, value)
    if threadIdx().x == 1 && blockIdx().x == 1
        @inbounds field[i0, j0] += value
    end
    return nothing
end

function inject_source!(field::AbstractMatrix{Float32}, i::Int, j::Int, value::Float32, ::CUDABackend)
    @cuda threads = 1 blocks = 1 _inject_field_kernel!(field, i, j, value)
    return nothing
end

# ==============================================================================
# Receiver Recording
# ==============================================================================

"""
    record_receivers!(backend, W, rec, k)

Record wavefield values at receiver locations for time step k.
"""
function record_receivers! end

function record_receivers!(::CPUBackend, W::Wavefield, rec::Receivers, k::Int)
    for r in 1:length(rec.i)
        ri, rj = rec.i[r], rec.j[r]
        if rec.type == :vz
            rec.data[k, r] = W.vz[ri, rj]
        elseif rec.type == :vx
            rec.data[k, r] = W.vx[ri, rj]
        elseif rec.type == :p
            rec.data[k, r] = (W.txx[ri, rj] + W.tzz[ri, rj]) * 0.5f0
        end
    end
    return nothing
end

function _record_kernel!(data, field, rec_i, rec_j, k, n_rec)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_rec
        ii, jj = rec_i[idx], rec_j[idx]
        @inbounds data[k, idx] = field[ii, jj]
    end
    return nothing
end

# FIXED: 专门的压力录制内核，避免创建临时数组
function _record_pressure_kernel!(data, txx, tzz, rec_i, rec_j, k, n_rec)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_rec
        ii, jj = rec_i[idx], rec_j[idx]
        @inbounds data[k, idx] = (txx[ii, jj] + tzz[ii, jj]) * 0.5f0
    end
    return nothing
end

function record_receivers!(::CUDABackend, W::Wavefield, rec::Receivers, k::Int)
    n_rec = length(rec.i)

    if rec.type == :vz
        @cuda threads = 256 blocks = cld(n_rec, 256) _record_kernel!(
            rec.data, W.vz, rec.i, rec.j, k, n_rec
        )
    elseif rec.type == :vx
        @cuda threads = 256 blocks = cld(n_rec, 256) _record_kernel!(
            rec.data, W.vx, rec.i, rec.j, k, n_rec
        )
    elseif rec.type == :p
        # FIXED: 使用专门的压力录制内核
        @cuda threads = 256 blocks = cld(n_rec, 256) _record_pressure_kernel!(
            rec.data, W.txx, W.tzz, rec.i, rec.j, k, n_rec
        )
    end
    return nothing
end

# Additional method for direct recording to gather matrix (used in vacuum formulation)
function record_receivers!(gather::AbstractMatrix{Float32}, W::Wavefield, rec::Receivers, k::Int, ::CPUBackend)
    for r in 1:length(rec.i)
        ri, rj = rec.i[r], rec.j[r]
        if rec.type == :vz
            gather[k, r] = W.vz[ri, rj]
        elseif rec.type == :vx
            gather[k, r] = W.vx[ri, rj]
        elseif rec.type == :p
            gather[k, r] = (W.txx[ri, rj] + W.tzz[ri, rj]) * 0.5f0
        end
    end
    return nothing
end

function _record_direct_kernel!(gather, field, rec_i, rec_j, k, n_rec)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_rec
        ii, jj = rec_i[idx], rec_j[idx]
        @inbounds gather[k, idx] = field[ii, jj]
    end
    return nothing
end

# FIXED: 专门的压力直接录制内核
function _record_direct_pressure_kernel!(gather, txx, tzz, rec_i, rec_j, k, n_rec)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_rec
        ii, jj = rec_i[idx], rec_j[idx]
        @inbounds gather[k, idx] = (txx[ii, jj] + tzz[ii, jj]) * 0.5f0
    end
    return nothing
end

function record_receivers!(gather::AbstractMatrix{Float32}, W::Wavefield, rec::Receivers, k::Int, ::CUDABackend)
    n_rec = length(rec.i)

    if rec.type == :vz
        @cuda threads = 256 blocks = cld(n_rec, 256) _record_direct_kernel!(
            gather, W.vz, rec.i, rec.j, k, n_rec
        )
    elseif rec.type == :vx
        @cuda threads = 256 blocks = cld(n_rec, 256) _record_direct_kernel!(
            gather, W.vx, rec.i, rec.j, k, n_rec
        )
    elseif rec.type == :p
        # FIXED: 使用专门的压力录制内核
        @cuda threads = 256 blocks = cld(n_rec, 256) _record_direct_pressure_kernel!(
            gather, W.txx, W.tzz, rec.i, rec.j, k, n_rec
        )
    end
    return nothing
end

# ==============================================================================
# Wavefield Reset
# ==============================================================================

"""
    reset!(backend, W)

Reset all wavefield components to zero.
"""
function reset!(::CPUBackend, W::Wavefield)
    for field in (W.vx, W.vz, W.txx, W.tzz, W.txz,
        W.vx_old, W.vz_old, W.txx_old, W.tzz_old, W.txz_old)
        fill!(field, 0.0f0)
    end
    return nothing
end

function reset!(::CUDABackend, W::Wavefield)
    for field in (W.vx, W.vz, W.txx, W.tzz, W.txz,
        W.vx_old, W.vz_old, W.txx_old, W.tzz_old, W.txz_old)
        fill!(field, 0.0f0)
    end
    return nothing
end

# Default dispatch (infer backend from wavefield type)
reset!(W::Wavefield{<:Array}) = reset!(CPU_BACKEND, W)
reset!(W::Wavefield{<:CuArray}) = reset!(CUDA_BACKEND, W)