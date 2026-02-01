# ==============================================================================
# visualization/video_recorder.jl
#
# Wavefield recording and video generation
#
# IMPROVED: Better colorrange calculation using global max across all frames
# ==============================================================================

using Printf

# Import CairoMakie functions
using CairoMakie: Figure, Axis, heatmap!, Colorbar, DataAspect, Observable, record, lines!

"""
    VideoConfig(; fields, skip, downsample, colormap, fps, show_boundary)

Configuration for wavefield video recording.

# Keyword Arguments
- `fields::Vector{Symbol} = [:vz]`: Wavefield components to record. Options:
  - `:vx` - Horizontal velocity component
  - `:vz` - Vertical velocity component (recommended for surface waves)
  - `:vel` - Velocity magnitude √(vx² + vz²)
  - `:p` - Pressure field -(τxx + τzz)/2
  - `:txx` - Normal stress τxx
  - `:tzz` - Normal stress τzz
  - `:txz` - Shear stress τxz
- `skip::Int = 10`: Record every N time steps. Larger = smaller file, fewer frames
- `downsample::Int = 1`: Spatial downsampling factor. 2 = half resolution
- `colormap::Symbol = :seismic`: Color scheme for visualization. Options:
  - `:seismic` - Red-white-blue, good for signed fields (vx, vz, p)
  - `:inferno` - Yellow-red-black, good for magnitude fields (:vel)
  - `:viridis` - Green-blue-yellow
  - Any valid CairoMakie/Makie colormap
- `fps::Int = 30`: Video frame rate (frames per second)
- `show_boundary::Bool = true`: Show dashed line marking physical domain boundary
  (separates interior from absorbing boundary layers)

# Example
```julia
# Record vz component, every 5 steps, 30 fps
video_config = VideoConfig(fields=[:vz], skip=5, fps=30)

# Record multiple fields
video_config = VideoConfig(fields=[:vz, :vx], skip=10)

# Record velocity magnitude with inferno colormap
video_config = VideoConfig(fields=[:vel], skip=5, colormap=:inferno)

# High-quality video (more frames)
video_config = VideoConfig(fields=[:vz], skip=2, fps=60)

# Hide boundary marker
video_config = VideoConfig(fields=[:vz], skip=5, show_boundary=false)
```

# Notes
- Total frames = nt ÷ skip
- Video duration = (nt ÷ skip) / fps seconds
- Larger `skip` reduces file size but may miss fast wave propagation
- `:vel` uses asymmetric colormap (0 to max) since magnitude is non-negative
- The dashed boundary line helps distinguish physical domain from PML/HABC region

See also: [`simulate!`](@ref), [`simulate_irregular!`](@ref)
"""
struct VideoConfig
    fields::Vector{Symbol}
    skip::Int
    downsample::Int
    colormap::Symbol
    fps::Int
    show_boundary::Bool  # 显示物理区域边界框
end

function VideoConfig(; fields::Vector{Symbol}=[:vz], skip::Int=10,
    downsample::Int=1, colormap::Symbol=:seismic, fps::Int=30,
    show_boundary::Bool=true)
    VideoConfig(fields, skip, downsample, colormap, fps, show_boundary)
end

# ==============================================================================
# Field Recorder
# ==============================================================================

"""
    FieldRecorder

Record wavefield snapshots for later video generation.

# Fields
- `frames`: Dictionary of field name => vector of frames
- `times`: Vector of time values
- `config`: VideoConfig
"""
mutable struct FieldRecorder
    frames::Dict{Symbol,Vector{Matrix{Float32}}}
    times::Vector{Float32}
    config::VideoConfig
    nx::Int
    nz::Int
    pad::Int  # 边界层厚度 (nbc + M)
    frame_count::Int
end

function FieldRecorder(nx::Int, nz::Int, config::VideoConfig; pad::Int=0)
    frames = Dict{Symbol,Vector{Matrix{Float32}}}()
    for field in config.fields
        frames[field] = Matrix{Float32}[]
    end
    FieldRecorder(frames, Float32[], config, nx, nz, pad, 0)
end

"""
    record!(recorder, wavefield, it, dt)

Record current wavefield state if at recording interval.
"""
function record!(recorder::FieldRecorder, W::Wavefield, it::Int, dt::Float32)
    # Check if this time step should be recorded
    if it % recorder.config.skip != 0
        return nothing
    end

    ds = recorder.config.downsample

    for field in recorder.config.fields
        # Get field data (always copy to CPU)
        data = _extract_field(W, field)

        # Check for invalid data
        if any(!isfinite, data)
            @warn "Non-finite values in $field at step $it"
        end

        # Downsample if needed
        if ds > 1
            data = data[1:ds:end, 1:ds:end]
        end

        # Store as-is: data is [nx, nz], heatmap will handle display
        push!(recorder.frames[field], copy(data))
    end

    push!(recorder.times, it * dt)
    recorder.frame_count += 1

    return nothing
end

"""
Extract field data from wavefield.
"""
function _extract_field(W::Wavefield, field::Symbol)
    raw = if field == :p
        # Pressure (acoustic approximation): p ≈ -(txx + tzz) / 2
        -(Array(W.txx) .+ Array(W.tzz)) ./ 2
    elseif field == :vx
        Array(W.vx)
    elseif field == :vz
        Array(W.vz)
    elseif field == :vel
        # Velocity magnitude
        sqrt.(Array(W.vx) .^ 2 .+ Array(W.vz) .^ 2)
    elseif field == :txx
        Array(W.txx)
    elseif field == :tzz
        Array(W.tzz)
    elseif field == :txz
        Array(W.txz)
    else
        error("Unknown field: $field")
    end
    return Float32.(raw)
end

# Accessor functions
n_frames(recorder::FieldRecorder) = recorder.frame_count
get_times(recorder::FieldRecorder) = recorder.times
get_frames(recorder::FieldRecorder, field::Symbol) = recorder.frames[field]

function clear!(recorder::FieldRecorder)
    empty!(recorder.frames)
    empty!(recorder.times)
    recorder.frame_count = 0
end

# ==============================================================================
# Multi-Field Recorder (callback interface)
# ==============================================================================

"""
    MultiFieldRecorder

Callback-compatible recorder for use with run_shots!().
"""
mutable struct MultiFieldRecorder
    recorder::FieldRecorder
    dt::Float32
end

function MultiFieldRecorder(nx::Int, nz::Int, dt::Float32, config::VideoConfig; pad::Int=0)
    MultiFieldRecorder(FieldRecorder(nx, nz, config; pad=pad), dt)
end

# Make it callable as a callback
function (mfr::MultiFieldRecorder)(wavefield::Wavefield, it::Int)
    record!(mfr.recorder, wavefield, it, mfr.dt)
end

# Cleanup (no-op for memory recorder)
function Base.close(mfr::MultiFieldRecorder)
    @info "Recording complete" frames = mfr.recorder.frame_count
end

# ==============================================================================
# Video Generation - IMPROVED
# ==============================================================================

"""
    generate_video(recorder::FieldRecorder, output::String; 
                   fps=30, colormap=:seismic, clim=nothing, percentile=99.5)

Generate MP4 video from recorded frames.

# Arguments
- `recorder`: FieldRecorder with recorded frames
- `output`: Output filename (e.g., "wavefield.mp4")
- `fps`: Frames per second (default: 30)
- `colormap`: Colormap to use (default: :seismic)
- `clim`: Color limits (default: auto from data)
- `percentile`: Percentile for auto clim (default: 99.5, ignores outliers)

# Example
```julia
recorder = FieldRecorder(nx, nz, VideoConfig(fields=[:vz], skip=5))
# ... run simulation with recorder callback ...
generate_video(recorder, "output.mp4"; fps=30, colormap=:seismic)
```
"""
function generate_video(recorder::FieldRecorder, output::String;
    fps::Int=30, colormap::Symbol=:seismic,
    clim::Union{Nothing,Tuple{Float64,Float64}}=nothing,
    percentile::Float64=99.5)

    frames = recorder.frames
    if isempty(frames)
        @warn "No frames recorded!"
        return nothing
    end

    # Get field name and frames
    field_name = first(keys(frames))
    field_frames = frames[field_name]
    n_frames = length(field_frames)

    if n_frames == 0
        @warn "No frames for field $field_name!"
        return nothing
    end

    # Get dimensions from first frame (data is [nx, nz])
    first_frame = field_frames[1]
    nx_ds, nz_ds = size(first_frame)

    # Determine color limits
    if clim === nothing
        # Compute global max across ALL frames (FIXED!)
        clim_val = _compute_clim(field_frames, field_name, percentile)
    else
        clim_val = max(abs(clim[1]), abs(clim[2]))
    end

    @info "Video parameters" field = field_name frames = n_frames size = "$(nx_ds)×$(nz_ds)" clim = clim_val

    # Create figure (wider for horizontal layout)
    fig = Figure(size=(1000, 600))

    # Create observable for animation
    data = Observable(field_frames[1])
    time_text = Observable("t = 0.000 s")

    ax = Axis(fig[1, 1],
        title=time_text,
        xlabel="X (grid)",
        ylabel="Z (grid)",
        aspect=DataAspect())

    # heatmap!(ax, x_range, y_range, data) where data is [length(x), length(y)]
    # Data is [nx, nz], so x=1:nx, y=1:nz
    # Fix colorrange for non-negative fields like :vel
    crange = if field_name == :vel
        (0, clim_val)           # Velocity magnitude is non-negative
    else
        (-clim_val, clim_val)   # Other fields use symmetric range
    end

    hm = heatmap!(ax, 1:nx_ds, 1:nz_ds, data,
        colormap=colormap,
        colorrange=crange)
    ax.yreversed = true  # Z increases downward

    # 添加物理区域边界框（虚线）
    pad = recorder.pad
    ds = recorder.config.downsample
    if recorder.config.show_boundary && pad > 0
        # 计算降采样后的边界位置
        pad_ds = pad ÷ ds + 1
        x_min, x_max = pad_ds, nx_ds - pad_ds + 1
        z_min, z_max = pad_ds, nz_ds - pad_ds + 1

        # 画虚线矩形框
        boundary_x = [x_min, x_max, x_max, x_min, x_min]
        boundary_z = [z_min, z_min, z_max, z_max, z_min]
        lines!(ax, boundary_x, boundary_z,
            color=:black, linewidth=2, linestyle=:dash)
    end

    Colorbar(fig[1, 2], hm, label=string(field_name))

    # Record video
    @info "Generating video..." output = output

    record(fig, output, 1:n_frames; framerate=fps) do i
        data[] = field_frames[i]
        time_text[] = @sprintf("t = %.4f s", recorder.times[i])
    end

    @info "Video saved" path = output
    return output
end

"""
Compute color limit from frames using percentile to ignore outliers.
"""
function _compute_clim(frames::Vector{Matrix{Float32}}, field_name::Symbol, percentile::Float64)
    # Collect all absolute values
    all_abs = Float32[]

    # Sample frames to avoid memory issues (use at most 50 frames)
    n_frames = length(frames)
    sample_indices = if n_frames <= 50
        1:n_frames
    else
        round.(Int, range(1, n_frames, length=50))
    end

    for i in sample_indices
        frame = frames[i]
        # Only include finite values
        valid = filter(isfinite, vec(frame))
        append!(all_abs, abs.(valid))
    end

    if isempty(all_abs)
        @warn "No valid data in frames, using default clim=1.0"
        return 1.0f0
    end

    # Use percentile to avoid extreme outliers
    sort!(all_abs)
    idx = min(length(all_abs), ceil(Int, percentile / 100 * length(all_abs)))
    clim_val = all_abs[idx]

    # For velocity magnitude field, use non-symmetric range
    if field_name == :vel
        clim_val = clim_val * 1.2  # Add 20% headroom
    else
        clim_val = clim_val * 1.5  # Add 50% headroom for better contrast
    end

    # Ensure non-zero
    if clim_val <= 0 || !isfinite(clim_val)
        clim_val = 1.0f0
    end

    return clim_val
end

# Legacy compatibility
const VideoRecorder = FieldRecorder