# ==============================================================================
# src/api/API.jl
#
# 新 API 子模块
# ==============================================================================

module API

using ..ElasticWave2D: VelocityModel, Wavefield, Medium, SimParams,
    Source, StressSource, ForceSource, Receivers,
    init_medium, init_habc, get_fd_coefficients,
    to_device, run_time_loop!, run_time_loop_with_boundaries!,
    CUDA_AVAILABLE, CPU_BACKEND, CUDA_BACKEND,
    AbstractBackend, CUDABackend, BoundaryConfig

using Printf
using JLD2

# ==============================================================================
# Wavelet
# ==============================================================================

abstract type AbstractWavelet end

struct RickerWavelet <: AbstractWavelet
    f0::Float32
    delay::Float32
end

Ricker(f0::Real) = RickerWavelet(Float32(f0), 1.0f0 / Float32(f0))
Ricker(f0::Real, delay::Real) = RickerWavelet(Float32(f0), Float32(delay))

struct CustomWavelet <: AbstractWavelet
    data::Vector{Float32}
end
CustomWavelet(data::AbstractVector) = CustomWavelet(Float32.(data))

function generate(w::RickerWavelet, dt::Float32, nt::Int)
    wavelet = zeros(Float32, nt)
    for i in 1:nt
        t = (i - 1) * dt - w.delay
        arg = (π * w.f0 * t)^2
        wavelet[i] = Float32((1.0 - 2.0 * arg) * exp(-arg))
    end
    return wavelet
end

function generate(w::CustomWavelet, dt::Float32, nt::Int)
    n = length(w.data)
    n >= nt ? w.data[1:nt] : vcat(w.data, zeros(Float32, nt - n))
end

# ==============================================================================
# Source
# ==============================================================================

@enum SourceMechanism Explosion ForceX ForceZ StressTxx StressTzz StressTxz

struct SourceConfig
    x::Float32
    z::Float32
    wavelet::AbstractWavelet
    mechanism::SourceMechanism
end

SourceConfig(x::Real, z::Real, w::AbstractWavelet) = SourceConfig(Float32(x), Float32(z), w, Explosion)
SourceConfig(x::Real, z::Real, w::AbstractWavelet, m::SourceMechanism) = SourceConfig(Float32(x), Float32(z), w, m)
SourceConfig(x::Real, z::Real; f0::Real=15.0, type::SourceMechanism=Explosion) =
    SourceConfig(Float32(x), Float32(z), Ricker(f0), type)

# ==============================================================================
# Receivers
# ==============================================================================

@enum RecordType Vz Vx Pressure

struct ReceiverConfig
    x::Vector{Float32}
    z::Vector{Float32}
    record::RecordType
end

ReceiverConfig(x::AbstractVector, z::AbstractVector) = ReceiverConfig(Float32.(x), Float32.(z), Vz)
ReceiverConfig(x::AbstractVector, z::AbstractVector, r::RecordType) = ReceiverConfig(Float32.(x), Float32.(z), r)

function line_receivers(x0::Real, x1::Real, n::Int; z::Real=0.0, record::RecordType=Vz)
    ReceiverConfig(Float32.(range(x0, x1, n)), fill(Float32(z), n), record)
end

# ==============================================================================
# Boundary
# ==============================================================================

struct Boundary
    top::Symbol
    nbc::Int
    vacuum_layers::Int

    function Boundary(top::Symbol, nbc::Int=50, vac::Int=10)
        top in (:image, :habc, :vacuum) || error("top must be :image, :habc, or :vacuum")
        new(top, nbc, vac)
    end
end

FreeSurface(; nbc::Int=50) = Boundary(:image, nbc, 0)
Absorbing(; nbc::Int=50) = Boundary(:habc, nbc, 0)
Vacuum(layers::Int=10; nbc::Int=50) = Boundary(:vacuum, nbc, layers)

# ==============================================================================
# Config
# ==============================================================================

struct SimConfig
    nt::Int
    dt::Union{Float32,Nothing}
    cfl::Float32
    fd_order::Int
    boundary::Boundary
    output_dir::String
end

SimConfig(; nt=3000, dt=nothing, cfl=0.4, fd_order=8, boundary=FreeSurface(), output_dir="outputs") =
    SimConfig(nt, dt === nothing ? nothing : Float32(dt), Float32(cfl), fd_order, boundary, output_dir)

# ==============================================================================
# Video
# ==============================================================================

struct VideoSettings
    fields::Vector{Symbol}
    interval::Int
    fps::Int
    colormap::Symbol
    format::Symbol  # :gif 或 :mp4
end

Video(; fields=[:vz], interval=50, fps=20, colormap=:seismic, format=:mp4) =
    VideoSettings(fields, interval, fps, colormap, format)

# ==============================================================================
# Result
# ==============================================================================

struct SimResult
    gather::Matrix{Float32}
    dt::Float32
    nt::Int
    source::SourceConfig
    receivers::ReceiverConfig
    snapshots::Union{Dict{Symbol,Array{Float32,3}},Nothing}
    video_files::Union{Dict{Symbol,String},Nothing}
end

SimResult(g, dt, nt, s, r) = SimResult(g, dt, nt, s, r, nothing, nothing)
SimResult(g, dt, nt, s, r, snaps) = SimResult(g, dt, nt, s, r, snaps, nothing)

times(r::SimResult) = Float32.(0:r.nt-1) .* r.dt
trace(r::SimResult, i::Int) = r.gather[:, i]
n_receivers(r::SimResult) = size(r.gather, 2)

# ==============================================================================
# IO - save_result / load_result
# ==============================================================================

"""
    save_result(result, filename)
    save_result(filename, result)

保存模拟结果到 JLD2 文件。

# 示例
```julia
save_result(result, "shot_001.jld2")
save_result("shot_001.jld2", result)  # 也可以
```
"""
function save_result(result::SimResult, filename::String)
    # 确保扩展名
    endswith(filename, ".jld2") || (filename *= ".jld2")

    # 转换为可序列化的格式
    jldsave(filename;
        gather=result.gather,
        dt=result.dt,
        nt=result.nt,
        source_x=result.source.x,
        source_z=result.source.z,
        source_f0=result.source.wavelet isa RickerWavelet ? result.source.wavelet.f0 : 0f0,
        source_delay=result.source.wavelet isa RickerWavelet ? result.source.wavelet.delay : 0f0,
        source_mechanism=Int(result.source.mechanism),
        receivers_x=result.receivers.x,
        receivers_z=result.receivers.z,
        receivers_record=Int(result.receivers.record),
        snapshots=result.snapshots,
        video_files=result.video_files
    )
    @info "Saved result to $filename"
end

save_result(filename::String, result::SimResult) = save_result(result, filename)

"""
    load_result(filename) -> SimResult

从 JLD2 文件加载模拟结果。

# 示例
```julia
result = load_result("shot_001.jld2")
```
"""
function load_result(filename::String)
    data = load(filename)

    # 重建 wavelet
    wavelet = RickerWavelet(data["source_f0"], data["source_delay"])

    # 重建 source
    source = SourceConfig(
        data["source_x"],
        data["source_z"],
        wavelet,
        SourceMechanism(data["source_mechanism"])
    )

    # 重建 receivers
    receivers = ReceiverConfig(
        data["receivers_x"],
        data["receivers_z"],
        RecordType(data["receivers_record"])
    )

    SimResult(
        data["gather"],
        data["dt"],
        data["nt"],
        source,
        receivers,
        get(data, "snapshots", nothing),
        get(data, "video_files", nothing)
    )
end

"""
    save_gather(result, filename; format=:jld2)
    save_gather(gather, dt, filename; format=:jld2)

保存道集数据。支持格式: :jld2, :npy, :segy (TODO)

# 示例
```julia
save_gather(result, "gather.jld2")
save_gather(result.gather, result.dt, "gather.jld2")
```
"""
function save_gather(result::SimResult, filename::String; format=:jld2)
    save_gather(result.gather, result.dt, filename; format=format)
end

function save_gather(gather::Matrix{Float32}, dt::Real, filename::String; format=:jld2)
    if format == :jld2
        endswith(filename, ".jld2") || (filename *= ".jld2")
        jldsave(filename; gather=gather, dt=Float32(dt))
    else
        error("Format $format not supported yet. Use :jld2")
    end
    @info "Saved gather to $filename"
end

"""
    load_gather(filename) -> (gather, dt)

加载道集数据。

# 示例
```julia
gather, dt = load_gather("gather.jld2")
```
"""
function load_gather(filename::String)
    data = load(filename)
    return data["gather"], data["dt"]
end

# ==============================================================================
# simulate
# ==============================================================================

function simulate(
    model::VelocityModel,
    source::SourceConfig,
    receivers::ReceiverConfig;
    config::SimConfig=SimConfig(),
    video::Union{VideoSettings,Nothing}=nothing
)
    be = CUDA_AVAILABLE[] ? CUDA_BACKEND : CPU_BACKEND

    vp_max = maximum(model.vp)
    dt = config.dt === nothing ? config.cfl * min(model.dx, model.dz) / vp_max : config.dt

    working_model, z_off = _prepare_model(model, config.boundary)

    is_fs = config.boundary.top in (:image, :vacuum)
    M = init_medium(working_model, config.boundary.nbc, config.fd_order, be; free_surface=is_fs)
    H = init_habc(M.nx, M.nz, config.boundary.nbc, M.pad, dt, working_model.dx, working_model.dz, vp_max, be)
    a = to_device(get_fd_coefficients(config.fd_order), be)
    W = Wavefield(M.nx, M.nz, be)
    params = SimParams(dt, config.nt, working_model.dx, working_model.dz, config.fd_order)

    src = _make_source(source, M, working_model, z_off, dt, config.nt, be)
    rec = _make_receivers(receivers, M, working_model, z_off, config.nt, be)

    callback, snaps = _setup_callback(video, M, config.nt)

    if config.boundary.top == :image
        run_time_loop!(be, W, M, H, a, src, rec, params; progress=true, on_step=callback)
    else
        bc = BoundaryConfig(top_boundary=config.boundary.top == :habc ? :absorbing : :vacuum, nbc=config.boundary.nbc)
        run_time_loop_with_boundaries!(be, W, M, H, a, src, rec, params, bc; progress=true, on_step=callback)
    end

    gather = be isa CUDABackend ? Array(rec.data) : copy(rec.data)

    video_files = nothing
    if video !== nothing && snaps !== nothing
        video_files = _generate_videos(snaps, video, config.output_dir, M.pad)
    end

    return SimResult(gather, dt, config.nt, source, receivers, snaps, video_files)
end

function simulate(model::VelocityModel, sources::Vector{SourceConfig}, receivers::ReceiverConfig;
    config::SimConfig=SimConfig(), video::Union{VideoSettings,Nothing}=nothing)
    [simulate(model, s, receivers; config, video=i == 1 ? video : nothing) for (i, s) in enumerate(sources)]
end

# ==============================================================================
# Plotting
# ==============================================================================

function plot_gather(result::SimResult; kwargs...)
    plot_gather(result.gather, result.dt; kwargs...)
end

function plot_gather(gather::Matrix, dt::Real;
    cmap=:seismic, clim=nothing, title="Seismic Gather",
    xlabel="Trace", ylabel="Time (s)", save=nothing)

    _check_plots()

    nt, nrec = size(gather)
    t = (0:nt-1) .* dt

    if clim === nothing
        vmax = _pctl(abs.(gather), 99)
        clim = (-vmax, vmax)
    end

    plt = Main.Plots.heatmap(1:nrec, t, gather; c=cmap, clim=clim, yflip=true,
        xlabel=xlabel, ylabel=ylabel, title=title)

    save !== nothing && Main.Plots.savefig(plt, save)
    return plt
end

function plot_trace(result::SimResult, i::Int; title="Trace $i", xlabel="Time (s)", ylabel="Amplitude")
    _check_plots()
    t = times(result)
    Main.Plots.plot(t, trace(result, i); xlabel=xlabel, ylabel=ylabel, title=title, legend=false)
end

function plot_snapshot(result::SimResult, field::Symbol, frame::Int;
    cmap=:seismic, clim=nothing, title=nothing, save=nothing)

    _check_plots()
    result.snapshots === nothing && error("No snapshots. Use video=Video() in simulate()")
    haskey(result.snapshots, field) || error("Field :$field not found")

    data = result.snapshots[field][:, :, frame]

    if clim === nothing
        vmax = _pctl(abs.(data), 99)
        clim = (-vmax, vmax)
    end

    title = title === nothing ? "$field - Frame $frame" : title

    plt = Main.Plots.heatmap(data'; c=cmap, clim=clim, yflip=true, title=title, aspect_ratio=:equal)
    save !== nothing && Main.Plots.savefig(plt, save)
    return plt
end

function plot_model(model::VelocityModel; field::Symbol=:vp, cmap=:viridis, title=nothing, save=nothing)
    _check_plots()

    data = getfield(model, field)
    x = (0:model.nx-1) .* model.dx
    z = (0:model.nz-1) .* model.dz
    title = title === nothing ? "$(model.name) - $field" : title

    plt = Main.Plots.heatmap(x, z, data; c=cmap, yflip=true, xlabel="X (m)", ylabel="Z (m)",
        title=title, aspect_ratio=:equal)
    save !== nothing && Main.Plots.savefig(plt, save)
    return plt
end

# ==============================================================================
# Video Generation (GIF & MP4)
# ==============================================================================

function _generate_videos(snaps::Dict, video::VideoSettings, output_dir::String, pad::Int)
    if !isdefined(Main, :Plots)
        @warn "Plots.jl not loaded, skipping video generation. Run: using Plots"
        return nothing
    end

    mkpath(output_dir)
    video_files = Dict{Symbol,String}()

    for field in video.fields
        haskey(snaps, field) || continue

        data = snaps[field]
        nx, nz, nframes = size(data)

        # 裁剪 padding
        x1, x2 = pad + 1, nx - pad
        z1, z2 = pad + 1, nz - pad
        inner = data[x1:x2, z1:z2, :]

        vmax = _pctl(abs.(inner), 99)
        vmax = vmax == 0 ? 1.0f0 : vmax
        clim = (-vmax, vmax)

        # 根据格式选择扩展名
        ext = video.format == :mp4 ? "mp4" : "gif"
        filepath = joinpath(output_dir, "wavefield_$(field).$(ext)")

        # 生成动画
        anim = Main.Plots.Animation()
        for i in 1:nframes
            Main.Plots.heatmap(inner[:, :, i]'; c=video.colormap, clim=clim, yflip=true,
                title="$field - Frame $i/$nframes", aspect_ratio=:equal,
                size=(600, 500))
            Main.Plots.frame(anim)
        end

        # 保存为 GIF 或 MP4
        if video.format == :mp4
            Main.Plots.mp4(anim, filepath; fps=video.fps)
        else
            Main.Plots.gif(anim, filepath; fps=video.fps)
        end

        video_files[field] = filepath
        @info "Video saved" file = filepath format = video.format frames = nframes
    end

    return video_files
end

"""
    make_video(result, field; filename, fps, format, colormap)

从快照数据生成视频（在模拟完成后单独调用）。

# 参数
- `result::SimResult`: 模拟结果（需包含 snapshots）
- `field::Symbol`: 场名，如 :vz

# 关键字参数
- `filename`: 输出文件名
- `fps`: 帧率，默认 20
- `format`: :mp4 或 :gif，默认 :mp4
- `colormap`: 配色方案，默认 :seismic
- `clim`: 色标范围，默认自动

# 示例
```julia
make_video(result, :vz)
make_video(result, :vz; filename="my_video.mp4", fps=30)
make_video(result, :vz; format=:gif)
```
"""
function make_video(result::SimResult, field::Symbol;
    filename::Union{String,Nothing}=nothing,
    fps::Int=20,
    format::Symbol=:mp4,
    colormap::Symbol=:seismic,
    clim=nothing)

    _check_plots()
    result.snapshots === nothing && error("No snapshots in result")
    haskey(result.snapshots, field) || error("Field :$field not found")

    data = result.snapshots[field]
    nx, nz, nframes = size(data)

    # 自动文件名
    if filename === nothing
        ext = format == :mp4 ? "mp4" : "gif"
        filename = "wavefield_$(field).$(ext)"
    end

    # 色标
    if clim === nothing
        vmax = _pctl(abs.(data), 99)
        vmax = vmax == 0 ? 1.0f0 : vmax
        clim = (-vmax, vmax)
    end

    # 生成动画
    anim = Main.Plots.Animation()
    for i in 1:nframes
        Main.Plots.heatmap(data[:, :, i]'; c=colormap, clim=clim, yflip=true,
            title="$field - Frame $i/$nframes", aspect_ratio=:equal,
            size=(600, 500))
        Main.Plots.frame(anim)
    end

    # 保存
    if format == :mp4
        Main.Plots.mp4(anim, filename; fps=fps)
    else
        Main.Plots.gif(anim, filename; fps=fps)
    end

    @info "Video created" file = filename format = format frames = nframes fps = fps
    return filename
end

# ==============================================================================
# Internal helpers
# ==============================================================================

function _prepare_model(model::VelocityModel, bc::Boundary)
    bc.top != :vacuum && return model, 0.0f0

    n = bc.vacuum_layers
    nz_new = model.nz + n
    vp = zeros(Float32, nz_new, model.nx)
    vp[n+1:end, :] = model.vp
    vs = zeros(Float32, nz_new, model.nx)
    vs[n+1:end, :] = model.vs
    rho = fill(1f-10, nz_new, model.nx)
    rho[n+1:end, :] = model.rho

    VelocityModel(vp, vs, rho, model.dx, model.dz; name=model.name * "_vac"), Float32(n * model.dz)
end

function _make_source(s::SourceConfig, M, model, z_off, dt, nt, be)
    ix = round(Int, s.x / model.dx) + M.pad + 1
    iz = round(Int, (s.z + z_off) / model.dz) + M.pad + 1
    wav = to_device(generate(s.wavelet, dt, nt), be)

    if s.mechanism == Explosion
        Source(ix, iz, wav)
    elseif s.mechanism in (ForceX, ForceZ)
        jz = clamp(round(Int, (s.z + z_off) / model.dz) + 1, 1, model.nz)
        jx = clamp(round(Int, s.x / model.dx) + 1, 1, model.nx)
        buoy = 1.0f0 / max(model.rho[jz, jx], 1.0f0)
        ForceSource(ix, iz, wav, s.mechanism == ForceZ ? :vz : :vx, buoy)
    else
        comp = s.mechanism == StressTxx ? :txx : s.mechanism == StressTzz ? :tzz : :txz
        StressSource(ix, iz, wav, comp)
    end
end

function _make_receivers(r::ReceiverConfig, M, model, z_off, nt, be)
    ix = [round(Int, x / model.dx) + M.pad + 1 for x in r.x]
    iz = [round(Int, (z + z_off) / model.dz) + M.pad + 1 for z in r.z]
    rec_type = r.record == Vz ? :vz : r.record == Vx ? :vx : :p
    Receivers(to_device(ix, be), to_device(iz, be), to_device(zeros(Float32, nt, length(ix)), be), rec_type)
end

function _setup_callback(video, M, nt)
    video === nothing && return nothing, nothing

    nf = nt ÷ video.interval
    snaps = Dict(f => zeros(Float32, M.nx, M.nz, nf) for f in video.fields)
    frame = Ref(0)

    cb = (W, info) -> begin
        if info.k % video.interval == 0
            frame[] += 1
            frame[] <= nf && for f in video.fields
                snaps[f][:, :, frame[]] = Array(getfield(W, f))
            end
        end
        true
    end
    cb, snaps
end

_check_plots() = isdefined(Main, :Plots) || error("请先运行: using Plots")
_pctl(arr, p) = (s = sort(vec(arr)); s[max(1, round(Int, length(s) * p / 100))])

# ==============================================================================
# Exports
# ==============================================================================

export VelocityModel

# Wavelet
export AbstractWavelet, RickerWavelet, CustomWavelet, Ricker, generate

# Source
export SourceMechanism, Explosion, ForceX, ForceZ, StressTxx, StressTzz, StressTxz, SourceConfig

# Receivers
export RecordType, Vz, Vx, Pressure, ReceiverConfig, line_receivers

# Boundary
export Boundary, FreeSurface, Absorbing, Vacuum

# Config
export SimConfig, VideoSettings, Video

# Result
export SimResult, times, trace, n_receivers

# Core
export simulate

# IO
export save_result, load_result, save_gather, load_gather

# Plotting
export plot_gather, plot_trace, plot_snapshot, plot_model

# Video
export make_video

end # module API