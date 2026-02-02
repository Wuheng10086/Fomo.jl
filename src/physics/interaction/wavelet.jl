# ==============================================================================
# physics/interaction/wavelet.jl
#
# Source wavelet generation and utilities
# ==============================================================================

# ==============================================================================
# Ricker Wavelet
# ==============================================================================

"""
    ricker_wavelet(f0, dt, nt; t0=nothing) -> Vector{Float32}

Generate Ricker (Mexican hat) wavelet.

# Arguments
- `f0::Real`: Dominant frequency in Hz
- `dt::Real`: Time step in seconds  
- `nt::Int`: Number of time steps

# Keyword Arguments
- `t0::Union{Nothing, Real}=nothing`: Time shift (delay) in seconds.
  - If `nothing`: defaults to `1.2/f0` (SPECFEM2D/Deepwave convention)
  - Specify explicitly for precise control

# Returns
- `Vector{Float32}`: Wavelet samples of length `nt`

# Mathematical Definition
```
w(t) = (1 - 2π²f₀²τ²) × exp(-π²f₀²τ²)
```
where τ = t - t₀

# Time Convention
The default `t0 = 1.2/f0` ensures:
- Wavelet starts near zero at t=0 (numerical stability)
- Main peak occurs at t = t0
- Compatible with SPECFEM2D and Deepwave

# Example
```julia
# Standard (SPECFEM2D/Deepwave compatible)
dt, nt, f0 = 0.001f0, 2001, 15.0f0
wavelet = ricker_wavelet(f0, dt, nt)  # t0 = 1.2/15 = 80ms

# Custom t0
wavelet = ricker_wavelet(f0, dt, nt; t0=0.1)  # t0 = 100ms

# Legacy behavior (t0 = 1.0/f0)
wavelet = ricker_wavelet(f0, dt, nt; t0=1.0/f0)  # t0 = 66.7ms
```

See also: [`gaussian_wavelet`](@ref), [`simulate!`](@ref)
"""
function ricker_wavelet(f0::Real, dt::Real, nt::Int; t0::Union{Nothing,Real}=nothing)
    # Default t0 = 1.2/f0 (SPECFEM2D/Deepwave convention)
    t0_actual = t0 === nothing ? 1.2 / f0 : Float64(t0)

    wavelet = Vector{Float32}(undef, nt)
    a = π * f0

    @inbounds for i in 1:nt
        τ = (i - 1) * dt - t0_actual
        arg = (a * τ)^2
        wavelet[i] = Float32((1.0 - 2.0 * arg) * exp(-arg))
    end

    return wavelet
end

# ==============================================================================
# Gaussian Derivative Wavelet
# ==============================================================================

"""
    gaussian_wavelet(f0, dt, nt; t0=nothing, order=1) -> Vector{Float32}

Generate Gaussian derivative wavelet.

# Arguments
- `f0::Real`: Dominant frequency in Hz
- `dt::Real`: Time step in seconds
- `nt::Int`: Number of time steps

# Keyword Arguments  
- `t0::Union{Nothing, Real}=nothing`: Time shift (default: 1.2/f0)
- `order::Int=1`: Derivative order (1=first derivative, 2=second derivative=Ricker)

# Example
```julia
wavelet = gaussian_wavelet(15.0, 0.001, 2001; order=1)
```
"""
function gaussian_wavelet(f0::Real, dt::Real, nt::Int;
    t0::Union{Nothing,Real}=nothing,
    order::Int=1)
    t0_actual = t0 === nothing ? 1.2 / f0 : Float64(t0)

    wavelet = Vector{Float32}(undef, nt)
    a = π * f0

    @inbounds for i in 1:nt
        τ = (i - 1) * dt - t0_actual
        arg = (a * τ)^2

        if order == 1
            # First derivative of Gaussian
            wavelet[i] = Float32(-2.0 * a^2 * τ * exp(-arg))
        elseif order == 2
            # Second derivative (Ricker)
            wavelet[i] = Float32((1.0 - 2.0 * arg) * exp(-arg))
        else
            error("Unsupported order: $order. Use 1 or 2.")
        end
    end

    return wavelet
end

# ==============================================================================
# Utility Functions
# ==============================================================================

"""
    normalize_wavelet(wavelet; mode=:peak) -> Vector{Float32}

Normalize wavelet amplitude.

# Arguments
- `wavelet::Vector`: Input wavelet

# Keyword Arguments
- `mode::Symbol`: Normalization mode
  - `:peak` - Normalize peak amplitude to 1.0
  - `:rms` - Normalize RMS amplitude to 1.0
  - `:energy` - Normalize energy to 1.0

# Example
```julia
wavelet = ricker_wavelet(15.0, 0.001, 2001)
wavelet_norm = normalize_wavelet(wavelet; mode=:peak)
```
"""
function normalize_wavelet(wavelet::Vector; mode::Symbol=:peak)
    if mode == :peak
        scale = maximum(abs, wavelet)
    elseif mode == :rms
        scale = sqrt(sum(wavelet .^ 2) / length(wavelet))
    elseif mode == :energy
        scale = sqrt(sum(wavelet .^ 2))
    else
        error("Unknown normalization mode: $mode. Use :peak, :rms, or :energy.")
    end

    return Float32.(wavelet ./ max(scale, 1e-30))
end

"""
    wavelet_info(wavelet, dt; f0=nothing) -> NamedTuple

Analyze wavelet properties.

# Returns
NamedTuple with:
- `peak_sample::Int`: Sample index of peak amplitude
- `peak_time_ms::Float32`: Peak time in milliseconds
- `max_amplitude::Float32`: Maximum amplitude
- `dominant_freq::Float32`: Estimated dominant frequency (if determinable)
"""
function wavelet_info(wavelet::Vector, dt::Real; f0::Union{Nothing,Real}=nothing)
    peak_sample = argmax(abs.(wavelet))
    peak_time_ms = Float32((peak_sample - 1) * dt * 1000)
    max_amplitude = maximum(abs, wavelet)

    # Estimate dominant frequency from zero-crossings if f0 not provided
    dominant_freq = f0 === nothing ? NaN32 : Float32(f0)

    return (
        peak_sample=peak_sample,
        peak_time_ms=peak_time_ms,
        max_amplitude=Float32(max_amplitude),
        dominant_freq=dominant_freq,
        length=length(wavelet),
        dt_ms=Float32(dt * 1000)
    )
end

"""
    validate_external_wavelet(wavelet, nt, dt, f0) -> Bool

Validate an external wavelet and print diagnostics.

# Arguments
- `wavelet::Vector`: External wavelet to validate
- `nt::Int`: Expected number of time steps
- `dt::Real`: Time step
- `f0::Real`: Expected dominant frequency

# Returns
- `Bool`: Whether wavelet is valid

# Example
```julia
external_wavelet = load_wavelet_from_file("my_wavelet.npy")
if validate_external_wavelet(external_wavelet, 2001, 0.001, 15.0)
    # Use wavelet
end
```
"""
function validate_external_wavelet(wavelet::Vector, nt::Int, dt::Real, f0::Real)
    valid = true

    # Check length
    if length(wavelet) != nt
        @warn "Wavelet length mismatch" expected = nt actual = length(wavelet)
        valid = false
    end

    # Check peak timing
    info = wavelet_info(wavelet, dt; f0=f0)
    expected_peak_ms = 1.2f0 / Float32(f0) * 1000
    timing_diff = abs(info.peak_time_ms - expected_peak_ms)

    if timing_diff > 20  # More than 20ms difference
        @warn "Wavelet peak timing differs from SPECFEM2D convention" actual_peak_ms = info.peak_time_ms expected_peak_ms = expected_peak_ms difference_ms = timing_diff
    end

    # Check amplitude
    if info.max_amplitude < 1e-10
        @warn "Wavelet amplitude is very small" max_amplitude = info.max_amplitude
        valid = false
    end

    if valid
        @info "External wavelet validated" length = info.length peak_time_ms = info.peak_time_ms max_amplitude = info.max_amplitude
    end

    return valid
end