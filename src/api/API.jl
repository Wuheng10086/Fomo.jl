# ==============================================================================
# src/api/API.jl
#
# 新 API 子模块
# ==============================================================================

module API

using ..ElasticWave2D: VelocityModel, Wavefield, Medium, SimParams,
    Source, StressSource, ForceSource, Receivers,
    init_medium, init_medium_vacuum, init_habc, get_fd_coefficients,
    to_device, run_time_loop!, run_time_loop_with_boundaries!,
    CUDA_AVAILABLE, CPU_BACKEND, CUDA_BACKEND,
    AbstractBackend, CUDABackend, BoundaryConfig,
    OutputConfig, ensure_output_dirs, resolve_output_path, default_video_filename, default_result_filename,
    generate_video, plot_setup, ArtifactsManifest, record_artifact!, write_manifest

using Printf
using JLD2
using CairoMakie: Figure, Axis, heatmap!, Colorbar, save

using ..ElasticWave2D: AbstractWavelet, RickerWavelet, CustomWavelet, Ricker, generate,
    SourceMechanism, Explosion, ForceX, ForceZ, StressTxx, StressTzz, StressTxz, SourceConfig,
    RecordType, Vz, Vx, Pressure, ReceiverConfig, line_receivers,
    Boundary, FreeSurface, Absorbing, Vacuum,
    SimConfig, VideoSettings, Video,
    SimResult, times, trace, n_receivers

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
    mkpath(dirname(filename))

    # 转换为可序列化的格式
    jldsave(filename;
        gather=result.gather,
        dt=result.dt,
        nt=result.nt,
        output_dir=result.output_dir,
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

function save_result(outputs::OutputConfig, result::SimResult; name::AbstractString="result")
    path = resolve_output_path(outputs, :results, "$(name).jld2")
    save_result(result, path)
    return path
end

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
        get(data, "video_files", nothing),
        get(data, "output_dir", "outputs")
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

function save_gather(outputs::OutputConfig, result::SimResult; name::AbstractString="gather", format=:jld2)
    path = resolve_output_path(outputs, :results, "$(name).jld2")
    save_gather(result.gather, result.dt, path; format=format)
    return path
end

function save_gather(gather::Matrix{Float32}, dt::Real, filename::String; format=:jld2)
    if format == :jld2
        endswith(filename, ".jld2") || (filename *= ".jld2")
        mkpath(dirname(filename))
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

"""
    simulate(model, source, receivers; config=SimConfig(), video=nothing) -> SimResult
    simulate(model, sources::Vector{SourceConfig}, receivers; config=SimConfig(), video=nothing) -> Vector{SimResult}

执行 2D 弹性波正演，返回道集与可选快照/视频信息。

若 `config.dt == nothing`，将按 `config.cfl` 与模型最大波速自动计算稳定时间步长。
"""
function simulate(
    model::VelocityModel,
    source::SourceConfig,
    receivers::ReceiverConfig;
    config::SimConfig=SimConfig(),
    outputs::Union{OutputConfig,Nothing}=nothing,
    video::Union{VideoSettings,Nothing}=nothing
)
    be = CUDA_AVAILABLE[] ? CUDA_BACKEND : CPU_BACKEND
    outputs = outputs === nothing ? OutputConfig(base_dir=config.output_dir) : outputs
    ensure_output_dirs(outputs)

    if outputs.plot_setup
        plot_setup(model, [source.x], [source.z], receivers.x, receivers.z;
            output=joinpath(outputs.base_dir, "setup.png"))
    end
    if video !== nothing && video.output_dir !== nothing
        Base.depwarn("Video(output_dir=...) is deprecated; use outputs=OutputConfig(base_dir=...) with fixed videos/ directory.", :simulate)
    end

    vp_max = maximum(model.vp)
    dt = config.dt === nothing ? config.cfl * min(model.dx, model.dz) / vp_max : config.dt
    working_model, z_off = _prepare_model(model, config.boundary)
    is_fs = config.boundary.top in (:image, :vacuum)
    M = config.boundary.top == :vacuum ?
        init_medium_vacuum(working_model.vp, working_model.vs, working_model.rho,
        working_model.dx, working_model.dz,
        config.boundary.nbc, config.fd_order, be) :
        init_medium(working_model, config.boundary.nbc, config.fd_order, be; free_surface=is_fs)
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
    any(.!isfinite.(gather)) && error("检测到非有限值 (NaN/Inf) 出现在采集数据中，可能是数值不稳定或边界设置问题。请检查模型与边界/时间步长。")

    video_files = nothing
    if video !== nothing && snaps !== nothing
        video_files = _generate_videos(snaps, video, outputs, M.pad, dt)
    end

    result = SimResult(gather, dt, config.nt, source, receivers, snaps, video_files, outputs.base_dir)
    manifest = ArtifactsManifest()
    record_artifact!(manifest, :result, resolve_output_path(outputs, :results, default_result_filename()))
    if video_files !== nothing
        for (field, path) in video_files
            record_artifact!(manifest, Symbol("video_$(field)"), path)
        end
    end
    write_manifest(outputs, manifest)
    return result
end

function simulate(model::VelocityModel, sources::Vector{SourceConfig}, receivers::ReceiverConfig;
    config::SimConfig=SimConfig(),
    outputs::Union{OutputConfig,Nothing}=nothing,
    video::Union{VideoSettings,Nothing}=nothing)
    [simulate(model, s, receivers; config, outputs, video=i == 1 ? video : nothing) for (i, s) in enumerate(sources)]
end

# ==============================================================================
# Plotting
# ==============================================================================

"""
    plot_gather(result; kwargs...) -> Figure
    plot_gather(gather, dt; kwargs...) -> Figure

绘制道集（Makie 后端）。可用 `output="xxx.png"` 保存图片。
"""
function plot_gather(result::SimResult; kwargs...)
    plot_gather(result.gather, result.dt; kwargs...)
end

function plot_gather(gather::Matrix, dt::Real;
    cmap=:seismic, clim=nothing, title="Seismic Gather",
    xlabel="Trace", ylabel="Time (s)", output=nothing,
    aspect=nothing)

    any(.!isfinite.(gather)) && error("采集数据包含 NaN/Inf，已停止绘图。")
    nt, nrec = size(gather)
    t = Float32.((0:nt-1) .* dt)

    if clim === nothing
        vmax = _pctl(abs.(gather), 99)
        vmax = vmax == 0 ? 1.0f0 : Float32(vmax)
        clim = (-vmax, vmax)
    end

    fig = Figure(size=(800, 600), fontsize=16)
    ax = Axis(fig[1, 1], 
        xlabel=xlabel, ylabel=ylabel, 
        title=title, yreversed=true,
        xlabelsize=18, ylabelsize=18,
        xtickalign=1, ytickalign=1,
        xminorticksvisible=true, yminorticksvisible=true,
        aspect=aspect)

    hm = heatmap!(ax, 1:nrec, t, gather'; colormap=cmap, colorrange=clim)
    Colorbar(fig[1, 2], hm, label="Amplitude", labelsize=18, ticklabelsize=14, width=15)

    if output !== nothing
        save(output, fig, px_per_unit=2)
    end
    return fig
end

"""
    plot_trace(result, i; title, xlabel, ylabel)

绘制单道曲线（Plots.jl 后端）。使用前需 `using Plots`。
"""
function plot_trace(result::SimResult, i::Int; title="Trace $i", xlabel="Time (s)", ylabel="Amplitude")
    _check_plots()
    t = times(result)
    Main.Plots.plot(t, trace(result, i); xlabel=xlabel, ylabel=ylabel, title=title, legend=false)
end

"""
    plot_snapshot(result, field, frame; cmap, clim, title, save)

绘制某个场在某一帧的快照（Plots.jl 后端）。使用前需在 `simulate(...; video=Video(...))` 产生快照。
"""
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

"""
    plot_model(model; field=:vp, cmap=:viridis, title=nothing, save=nothing)

绘制速度模型字段（Plots.jl 后端）。使用前需 `using Plots`。
"""
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

function _generate_videos(snaps::Dict, video::VideoSettings, outputs::OutputConfig, pad::Int, dt::Float32)
    ensure_output_dirs(outputs)
    video_files = Dict{Symbol,String}()
    ext = video.format == :gif ? "gif" : "mp4"

    for field in video.fields
        haskey(snaps, field) || continue
        data = snaps[field]
        nx, nz, nframes = size(data)

        x1, x2 = pad + 1, nx - pad
        z1, z2 = pad + 1, nz - pad
        inner = data[x1:x2, z1:z2, :]

        times = Float32.((1:nframes) .* (video.interval * dt))
        filepath = resolve_output_path(outputs, :videos, default_video_filename(field; ext=ext))
        generate_video(inner, filepath; field_name=field, fps=video.fps, colormap=video.colormap, times=times)

        video_files[field] = filepath
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
- `output_dir`: 输出目录；`nothing` 表示使用 `result.output_dir`
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
    outputs::Union{OutputConfig,Nothing}=nothing,
    fps::Int=20,
    format::Symbol=:mp4,
    colormap::Symbol=:seismic,
    clim=nothing)

    result.snapshots === nothing && error("No snapshots in result")
    haskey(result.snapshots, field) || error("Field :$field not found")

    data = result.snapshots[field]
    outputs = outputs === nothing ? OutputConfig(base_dir=result.output_dir) : outputs
    ensure_output_dirs(outputs)

    ext = format == :gif ? "gif" : "mp4"
    filepath = if filename === nothing
        resolve_output_path(outputs, :videos, default_video_filename(field; ext=ext))
    else
        isabspath(filename) ? filename : resolve_output_path(outputs, :videos, filename)
    end

    generate_video(data, filepath; field_name=field, fps=fps, colormap=colormap, clim=clim)
    return filepath
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
    rho = zeros(Float32, nz_new, model.nx)
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
export save_result, load_result, load_gather

# Plotting
export plot_gather, plot_trace, plot_snapshot, plot_model

# Video
export make_video

end # module API
