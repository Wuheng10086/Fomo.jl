"""
    SimConfig(; cfl=0.4, fd_order=8, dt=nothing, nt)

Simulation configuration for elastic wave propagation.
弹性波传播模拟配置。

# Fields / 字段
- `cfl::Float32`: CFL stability number (Courant-Friedrichs-Lewy). Default 0.4.
  CFL 稳定性系数，默认 0.4。
- `fd_order::Int`: Finite difference order (accuracy). Default 8.
  有限差分阶数（精度），默认 8 阶。
- `dt::Union{Float32,Nothing}`: Time step in seconds. `nothing` for auto-calculation based on CFL.
  时间步长（秒）。设为 `nothing` 时根据 CFL 自动计算。
- `nt::Int`: Total number of time steps.
  总时间步数。

# Example / 示例
```julia
config = SimConfig(cfl=0.4, fd_order=8, nt=2000)
config = SimConfig(dt=0.001f0, nt=2000)  # manual dt / 手动指定 dt
```
"""
struct SimConfig
    cfl::Float32
    fd_order::Int
    dt::Union{Float32,Nothing}
    nt::Int
end

SimConfig(; cfl=0.4, fd_order=8, dt=nothing, nt::Int) =
    SimConfig(Float32(cfl), fd_order, dt === nothing ? nothing : Float32(dt), nt)

"""
    BoundaryConfig(; top=:image, nbc=50, vacuum_layers=10)

Boundary condition configuration for the simulation domain.
模拟域边界条件配置。

# Fields / 字段
- `top::Symbol`: Top boundary type. Options: `:image` (free surface), `:absorbing` (HABC), `:vacuum`.
  顶部边界类型。选项: `:image`（自由表面）、`:absorbing`（吸收边界）、`:vacuum`（真空层）。
- `nbc::Int`: Number of absorbing boundary cells. Default 50.
  吸收边界层数，默认 50。
- `vacuum_layers::Int`: Number of vacuum layers (only for `:vacuum` top). Default 10.
  真空层数（仅当顶部为 `:vacuum` 时有效），默认 10。

# See Also / 参见
- [`top_image`](@ref): Convenience constructor for free surface. / 自由表面快捷构造。
- [`top_absorbing`](@ref): Convenience constructor for absorbing top. / 吸收边界快捷构造。
- [`top_vacuum`](@ref): Convenience constructor for vacuum top. / 真空层快捷构造。
"""
struct BoundaryConfig
    top::Symbol
    nbc::Int
    vacuum_layers::Int
end

BoundaryConfig(; top::Symbol=:image, nbc::Int=50, vacuum_layers::Int=10) =
    BoundaryConfig(top, nbc, vacuum_layers)

"""
    top_image(; nbc=50) -> BoundaryConfig

Create boundary config with free surface (image method) at top.
创建顶部为自由表面（镜像法）的边界配置。

# Arguments / 参数
- `nbc::Int`: Number of absorbing boundary cells on other sides. Default 50.
  其他边界的吸收层数，默认 50。
"""
top_image(; nbc::Int=50) = BoundaryConfig(top=:image, nbc=nbc, vacuum_layers=0)

"""
    top_absorbing(; nbc=50) -> BoundaryConfig

Create boundary config with absorbing boundary (HABC) at top.
创建顶部为吸收边界（HABC）的边界配置。

# Arguments / 参数
- `nbc::Int`: Number of absorbing boundary cells. Default 50.
  吸收边界层数，默认 50。
"""
top_absorbing(; nbc::Int=50) = BoundaryConfig(top=:absorbing, nbc=nbc, vacuum_layers=0)

"""
    top_vacuum(layers=10; nbc=50) -> BoundaryConfig

Create boundary config with vacuum layers at top for acoustic-elastic coupling.
创建顶部为真空层的边界配置，用于声-弹耦合模拟。

# Arguments / 参数
- `layers::Int`: Number of vacuum layers to add. Default 10.
  添加的真空层数，默认 10。
- `nbc::Int`: Number of absorbing boundary cells on other sides. Default 50.
  其他边界的吸收层数，默认 50。
"""
top_vacuum(layers::Int=10; nbc::Int=50) = BoundaryConfig(top=:vacuum, nbc=nbc, vacuum_layers=layers)

"""
    OutputConfig(; base_dir="outputs", plot_gather=false, plot_setup=false, video_config=nothing)

Output configuration controlling what files are generated during simulation.
输出配置，控制模拟过程中生成的文件。

# Fields / 字段
- `base_dir::String`: Base output directory. Default "outputs".
  输出根目录，默认 "outputs"。
- `plot_gather::Bool`: Whether to save gather plot as PNG. Default `false`.
  是否保存道集图为 PNG，默认 `false`。
- `plot_setup::Bool`: Whether to save setup plot (model + sources + receivers). Default `false`.
  是否保存设置图（模型 + 震源 + 检波器），默认 `false`。
- `video_config::Union{VideoConfig,Nothing}`: Video generation config, `nothing` to disable.
  视频生成配置，`nothing` 表示不生成视频。

# Example / 示例
```julia
outputs = OutputConfig(base_dir="my_results", plot_gather=true, plot_setup=true)
outputs = OutputConfig(video_config=VideoConfig(fields=[:vz], fps=20))
```
"""
struct OutputConfig
    base_dir::String
    plot_gather::Bool
    plot_setup::Bool
    video_config::Union{VideoConfig,Nothing}
end

OutputConfig(; base_dir::AbstractString="outputs", 
    plot_gather::Bool=false,
    plot_setup::Bool=false,
    video_config::Union{VideoConfig,Nothing}=nothing) =
    OutputConfig(String(base_dir), plot_gather, plot_setup, video_config)

"""
    simulate(model, source, receivers, boundary, outputs, simConfig; backend, progress) -> Matrix{Float32}

Execute a single-shot 2D elastic wave simulation and return the seismic gather.
执行单炮 2D 弹性波模拟，返回地震道集。

# Arguments / 参数
- `model::VelocityModel`: Velocity model containing vp, vs, rho.
  速度模型，包含纵波速度、横波速度和密度。
- `source::SourceConfig`: Source configuration (position, wavelet, mechanism).
  震源配置（位置、子波、震源机制）。
- `receivers::ReceiverConfig`: Receiver configuration (positions, record type).
  检波器配置（位置、记录类型）。
- `boundary::BoundaryConfig`: Boundary condition settings.
  边界条件设置。
- `outputs::OutputConfig`: Output file settings.
  输出文件设置。
- `simConfig::SimConfig`: Simulation parameters (dt, nt, fd_order, cfl).
  模拟参数（时间步长、步数、差分阶数、CFL 系数）。

# Keyword Arguments / 关键字参数
- `backend::AbstractBackend`: Computation backend. Auto-detects CUDA if available.
  计算后端，若 CUDA 可用则自动选择 GPU。
- `progress::Bool`: Show progress bar. Default `true`.
  是否显示进度条，默认 `true`。

# Returns / 返回
- `Matrix{Float32}`: Seismic gather of size `(nt, n_receivers)`.
  地震道集，大小为 `(nt, 检波器数)`。

# Example / 示例
```julia
model = VelocityModel(vp, vs, rho, dx, dz)
source = SourceConfig(500.0f0, 10.0f0, Ricker(20.0f0))
receivers = line_receivers(0.0f0, 1000.0f0, 10.0f0, 101)
boundary = top_image(nbc=50)
outputs = OutputConfig(base_dir="results", plot_gather=true)
config = SimConfig(nt=2000)

gather = simulate(model, source, receivers, boundary, outputs, config)
```
"""
function simulate(
    model::VelocityModel,
    source::SourceConfig,
    receivers::ReceiverConfig,
    boundary::BoundaryConfig,
    outputs::OutputConfig,
    simConfig::SimConfig;
    backend::AbstractBackend=(CUDA_AVAILABLE[] ? CUDA_BACKEND : CPU_BACKEND),
    progress::Bool=true,
)
    mkpath(outputs.base_dir)

    if outputs.plot_setup
        plot_setup(model, [source.x], [source.z], receivers.x, receivers.z;
            output=joinpath(outputs.base_dir, "setup.png"))
    end

    vp_max = maximum(model.vp)
    dt = simConfig.dt === nothing ? simConfig.cfl * min(model.dx, model.dz) / vp_max : simConfig.dt

    working_model, z_off = _prepare_model(model, boundary)
    free_surface = boundary.top in (:image, :vacuum)
    M = boundary.top == :vacuum ?
        init_medium_vacuum(working_model.vp, working_model.vs, working_model.rho,
        working_model.dx, working_model.dz,
        boundary.nbc, simConfig.fd_order, backend) :
        init_medium(working_model, boundary.nbc, simConfig.fd_order, backend; free_surface=free_surface)

    H = init_habc(M.nx, M.nz, boundary.nbc, M.pad, dt, working_model.dx, working_model.dz, vp_max, backend)
    a = to_device(get_fd_coefficients(simConfig.fd_order), backend)
    W = Wavefield(M.nx, M.nz, backend)
    params = SimParams(dt, simConfig.nt, working_model.dx, working_model.dz, simConfig.fd_order)

    src = _make_source(source, M, working_model, z_off, dt, simConfig.nt, backend)
    rec = _make_receivers(receivers, M, working_model, z_off, simConfig.nt, backend)

    recorder, on_step = _setup_video(outputs.video_config, M, dt)

    if boundary.top == :image
        run_time_loop!(backend, W, M, H, a, src, rec, params; progress=progress, on_step=on_step)
    else
        bc = SolverBoundaryConfig(top_boundary=boundary.top, nbc=boundary.nbc)
        run_time_loop_with_boundaries!(backend, W, M, H, a, src, rec, params, bc; progress=progress, on_step=on_step)
    end

    gather = backend isa CUDABackend ? Array(rec.data) : copy(rec.data)
    any(.!isfinite.(gather)) && error("Detected NaN/Inf in gather; check stability and boundary settings.")

    dt32 = Float32(dt)
    _save_result(outputs.base_dir, gather, dt32, simConfig.nt, source, receivers)

    if outputs.plot_gather
        plot_gather(gather, receivers.x, dt32; output_path=joinpath(outputs.base_dir, "gather.png"))
    end

    if recorder !== nothing
        _write_videos(recorder, outputs.base_dir)
    end

    return gather
end

function _prepare_model(model::VelocityModel, boundary::BoundaryConfig)
    boundary.top == :vacuum || return model, 0.0f0

    n = boundary.vacuum_layers
    nz_new = model.nz + n

    vp = zeros(Float32, nz_new, model.nx)
    vp[n+1:end, :] = model.vp
    vs = zeros(Float32, nz_new, model.nx)
    vs[n+1:end, :] = model.vs
    rho = zeros(Float32, nz_new, model.nx)
    rho[n+1:end, :] = model.rho

    return VelocityModel(vp, vs, rho, model.dx, model.dz; name=model.name * "_vac"), Float32(n * model.dz)
end

function _make_source(s::SourceConfig, M, model, z_off, dt, nt, be)
    ix = round(Int, s.x / model.dx) + M.pad + 1
    iz = round(Int, (s.z + z_off) / model.dz) + M.pad + 1
    wav = to_device(generate(s.wavelet, Float32(dt), nt), be)

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

function _setup_video(video_config::Union{VideoConfig,Nothing}, M, dt::Float32)
    video_config === nothing && return nothing, nothing
    recorder = FieldRecorder(M.nx, M.nz, video_config; pad=M.pad)
    on_step = (W, info) -> begin
        record!(recorder, W, info.k, dt)
        true
    end
    return recorder, on_step
end

function _write_videos(recorder::FieldRecorder, base_dir::String)
    for field in recorder.config.fields
        frames = get(recorder.frames, field, nothing)
        frames === nothing && continue
        isempty(frames) && continue
        out = joinpath(base_dir, "wavefield_$(field).mp4")
        tmp = FieldRecorder(recorder.nx, recorder.nz,
            VideoConfig(fields=[field], skip=recorder.config.skip,
                downsample=recorder.config.downsample,
                colormap=recorder.config.colormap,
                fps=recorder.config.fps,
                show_boundary=recorder.config.show_boundary);
            pad=recorder.pad)
        tmp.frames = Dict(field => frames)
        tmp.times = recorder.times
        tmp.frame_count = recorder.frame_count
        generate_video(tmp, out; fps=recorder.config.fps, colormap=recorder.config.colormap)
    end
    return nothing
end

function _save_result(base_dir::String, gather::Matrix{Float32}, dt::Float32, nt::Int,
    source::SourceConfig, receivers::ReceiverConfig)
    path = joinpath(base_dir, "result.jld2")

    wavelet_kind = source.wavelet isa RickerWavelet ? :ricker : :custom
    wavelet_f0 = source.wavelet isa RickerWavelet ? source.wavelet.f0 : 0.0f0
    wavelet_delay = source.wavelet isa RickerWavelet ? source.wavelet.delay : 0.0f0
    wavelet_data = source.wavelet isa CustomWavelet ? source.wavelet.data : Float32[]

    jldsave(path;
        gather=gather,
        dt=dt,
        nt=nt,
        source_x=source.x,
        source_z=source.z,
        source_mechanism=Int(source.mechanism),
        wavelet_kind=String(wavelet_kind),
        wavelet_f0=wavelet_f0,
        wavelet_delay=wavelet_delay,
        wavelet_data=wavelet_data,
        receivers_x=receivers.x,
        receivers_z=receivers.z,
        receivers_record=Int(receivers.record),
    )
    return path
end

"""
    BatchSimulator

Reusable simulator for efficient multi-shot simulations with shared medium and receivers.
可复用的模拟器，用于高效执行多炮模拟，共享介质和检波器配置。

Pre-allocates GPU/CPU resources once, then efficiently runs multiple shots by only
updating source positions. Ideal for seismic surveys with many shots.
预先分配 GPU/CPU 资源，然后通过仅更新震源位置高效执行多炮模拟。
适用于多炮地震勘探。

# Fields / 字段 (internal)
Internal fields are managed automatically. Use the constructor and `simulate_shot!`/`simulate_shots!`.
内部字段自动管理，请使用构造函数和 `simulate_shot!`/`simulate_shots!`。

# See Also / 参见
- [`BatchSimulator(model, source_template, receivers, boundary, simConfig)`](@ref): Constructor.
- [`simulate_shot!`](@ref): Run single shot. / 执行单炮。
- [`simulate_shots!`](@ref): Run multiple shots. / 执行多炮。
"""
mutable struct BatchSimulator
    backend::AbstractBackend
    medium::Medium
    habc::HABCConfig
    fd_coeffs::Any
    wavefield::Wavefield
    receivers::Receivers
    wavelet_device::Any
    params::SimParams
    source_template::SourceConfig
    receivers_cfg::ReceiverConfig
    working_model::VelocityModel
    z_off::Float32
    boundary::BoundaryConfig
    pad::Int
end

"""
    BatchSimulator(model, source_template, receivers, boundary, simConfig; backend) -> BatchSimulator

Create a reusable batch simulator for efficient multi-shot simulations.
创建可复用的批量模拟器，用于高效多炮模拟。

Pre-allocates all computational resources (medium, wavefield, receivers) once.
Only source position changes between shots, making multi-shot surveys very efficient.
预先分配所有计算资源（介质、波场、检波器），仅在不同炮之间改变震源位置，
使多炮采集非常高效。

# Arguments / 参数
- `model::VelocityModel`: Velocity model.
  速度模型。
- `source_template::SourceConfig`: Template source config (wavelet and mechanism are reused).
  震源模板配置（子波和机制将被复用）。
- `receivers::ReceiverConfig`: Receiver configuration (fixed for all shots).
  检波器配置（所有炮共用）。
- `boundary::BoundaryConfig`: Boundary condition settings.
  边界条件设置。
- `simConfig::SimConfig`: Simulation parameters.
  模拟参数。

# Keyword Arguments / 关键字参数
- `backend::AbstractBackend`: Computation backend. Auto-detects CUDA if available.
  计算后端，若 CUDA 可用则自动选择 GPU。

# Example / 示例
```julia
model = VelocityModel(vp, vs, rho, dx, dz)
source_template = SourceConfig(0f0, 10f0, Ricker(20f0))  # position will be overridden
receivers = line_receivers(0f0, 1000f0, 10f0, 101)
config = SimConfig(nt=2000)

sim = BatchSimulator(model, source_template, receivers, top_image(), config)

# Run shots at different positions / 在不同位置执行炮
gather1 = simulate_shot!(sim, 100f0, 10f0)
gather2 = simulate_shot!(sim, 200f0, 10f0)
```
"""
function BatchSimulator(
    model::VelocityModel,
    source_template::SourceConfig,
    receivers::ReceiverConfig,
    boundary::BoundaryConfig,
    simConfig::SimConfig;
    backend::AbstractBackend=(CUDA_AVAILABLE[] ? CUDA_BACKEND : CPU_BACKEND),
)
    vp_max = maximum(model.vp)
    dt = simConfig.dt === nothing ? simConfig.cfl * min(model.dx, model.dz) / vp_max : simConfig.dt

    working_model, z_off = _prepare_model(model, boundary)
    free_surface = boundary.top in (:image, :vacuum)
    M = boundary.top == :vacuum ?
        init_medium_vacuum(working_model.vp, working_model.vs, working_model.rho,
        working_model.dx, working_model.dz,
        boundary.nbc, simConfig.fd_order, backend) :
        init_medium(working_model, boundary.nbc, simConfig.fd_order, backend; free_surface=free_surface)

    H = init_habc(M.nx, M.nz, boundary.nbc, M.pad, dt, working_model.dx, working_model.dz, vp_max, backend)
    a = to_device(get_fd_coefficients(simConfig.fd_order), backend)
    W = Wavefield(M.nx, M.nz, backend)
    params = SimParams(dt, simConfig.nt, working_model.dx, working_model.dz, simConfig.fd_order)

    rec = _make_receivers(receivers, M, working_model, z_off, simConfig.nt, backend)
    wavelet_device = to_device(generate(source_template.wavelet, Float32(dt), simConfig.nt), backend)

    BatchSimulator(backend, M, H, a, W, rec, wavelet_device, params, source_template, receivers,
        working_model, z_off, boundary, M.pad)
end

"""
    simulate_shot!(sim, src_x, src_z; gather_out, progress, outputs, shot_id) -> Matrix{Float32}

Execute a single shot using a pre-configured BatchSimulator.
使用预配置的 BatchSimulator 执行单炮模拟。

Resets wavefield and receivers, then runs simulation at the specified source position.
This is much faster than creating a new simulation for each shot.
重置波场和检波器，然后在指定震源位置运行模拟。
这比每炮创建新模拟快得多。

# Arguments / 参数
- `sim::BatchSimulator`: Pre-configured simulator.
  预配置的模拟器。
- `src_x::Real`: Source X position in meters.
  震源 X 坐标（米）。
- `src_z::Real`: Source Z position in meters.
  震源 Z 坐标（米）。

# Keyword Arguments / 关键字参数
- `gather_out::Union{Matrix{Float32},Nothing}`: Pre-allocated output buffer. `nothing` allocates new.
  预分配的输出缓冲区，`nothing` 则新建。
- `progress::Bool`: Show progress bar. Default `false`.
  是否显示进度条，默认 `false`。
- `outputs::Union{OutputConfig,Nothing}`: Output settings. `nothing` disables file output.
  输出设置，`nothing` 表示不输出文件。
- `shot_id::Union{Int,Nothing}`: Shot identifier for naming output files.
  炮号，用于输出文件命名。

# Returns / 返回
- `Matrix{Float32}`: Seismic gather of size `(nt, n_receivers)`.
  地震道集，大小为 `(nt, 检波器数)`。

# Example / 示例
```julia
sim = BatchSimulator(model, source_template, receivers, boundary, config)
gather = simulate_shot!(sim, 500.0f0, 10.0f0)
gather = simulate_shot!(sim, 600.0f0, 10.0f0; progress=true)
```
"""
function simulate_shot!(
    sim::BatchSimulator,
    src_x::Real,
    src_z::Real;
    gather_out::Union{Matrix{Float32},Nothing}=nothing,
    progress::Bool=false,
    outputs::Union{OutputConfig,Nothing}=nothing,
    shot_id::Union{Int,Nothing}=nothing,
)
    src = _make_source_at(sim, src_x, src_z)

    reset!(sim.backend, sim.wavefield)
    fill!(sim.receivers.data, 0.0f0)

    if sim.boundary.top == :image
        run_time_loop!(sim.backend, sim.wavefield, sim.medium, sim.habc,
            sim.fd_coeffs, src, sim.receivers, sim.params;
            progress=progress, on_step=nothing)
    else
        bc = SolverBoundaryConfig(top_boundary=sim.boundary.top, nbc=sim.boundary.nbc)
        run_time_loop_with_boundaries!(sim.backend, sim.wavefield, sim.medium, sim.habc,
            sim.fd_coeffs, src, sim.receivers, sim.params, bc;
            progress=progress, on_step=nothing)
    end

    nt, nrec = size(sim.receivers.data)
    g = gather_out === nothing ? Matrix{Float32}(undef, nt, nrec) : gather_out
    size(g) == (nt, nrec) || error("gather_out size mismatch: got $(size(g)), expected ($(nt), $(nrec))")

    copyto!(g, sim.receivers.data)
    any(.!isfinite.(g)) && error("Detected NaN/Inf in gather; check stability and boundary settings.")

    if outputs !== nothing
        mkpath(outputs.base_dir)
        # 单炮 simulate 时才在这里画 setup
        if outputs.plot_setup && shot_id === nothing
            plot_setup(sim.working_model, [Float32(src_x)], [Float32(src_z)], sim.receivers_cfg.x, sim.receivers_cfg.z;
                output=joinpath(outputs.base_dir, "setup.png"))
        end
        src_cfg = SourceConfig(Float32(src_x), Float32(src_z), sim.source_template.wavelet, sim.source_template.mechanism)
        _save_shot_result(outputs.base_dir, g, sim.params.dt, sim.params.nt, src_cfg, sim.receivers_cfg; shot_id=shot_id)
        if outputs.plot_gather
            name = shot_id === nothing ? "gather.png" : "shot_$(lpad(string(shot_id), 4, '0'))_gather.png"
            plot_gather(g, sim.receivers_cfg.x, sim.params.dt; output_path=joinpath(outputs.base_dir, name))
        end
    end

    return g
end

"""
    simulate_shots!(sim, src_x, src_z; store, on_shot_complete, progress, outputs) -> Union{Vector{Matrix{Float32}}, Nothing}

Execute multiple shots at specified positions using a BatchSimulator.
使用 BatchSimulator 在指定位置执行多炮模拟。

Efficiently loops through all shot positions, optionally storing results and calling
a callback after each shot completes.
高效遍历所有炮点位置，可选择存储结果并在每炮完成后调用回调函数。

# Arguments / 参数
- `sim::BatchSimulator`: Pre-configured simulator.
  预配置的模拟器。
- `src_x::AbstractVector`: Vector of source X positions in meters.
  震源 X 坐标向量（米）。
- `src_z::AbstractVector`: Vector of source Z positions in meters.
  震源 Z 坐标向量（米）。

# Keyword Arguments / 关键字参数
- `store::Bool`: Whether to store all gathers in memory. Default `true`.
  是否将所有道集存储在内存中，默认 `true`。
- `on_shot_complete::Union{Function,Nothing}`: Callback `(gather, shot_index) -> nothing` after each shot.
  每炮完成后的回调函数 `(道集, 炮号) -> nothing`。
- `progress::Bool`: Show progress bar for first shot. Default `false`.
  是否为第一炮显示进度条，默认 `false`。
- `outputs::Union{OutputConfig,Nothing}`: Output settings. Saves each shot if provided.
  输出设置，若提供则保存每炮结果。

# Returns / 返回
- `Vector{Matrix{Float32}}`: Vector of gathers if `store=true`, otherwise `nothing`.
  若 `store=true` 返回道集向量，否则返回 `nothing`。

# Example / 示例
```julia
sim = BatchSimulator(model, source_template, receivers, boundary, config)

# Define shot positions / 定义炮点位置
src_x = collect(100f0:100f0:500f0)  # 5 shots
src_z = fill(10f0, 5)

# Run all shots / 执行所有炮
gathers = simulate_shots!(sim, src_x, src_z)

# With callback / 使用回调
simulate_shots!(sim, src_x, src_z; store=false) do gather, i
    @info "Shot \$i completed"
    # Process gather here / 在此处理道集
end
```
"""
function simulate_shots!(
    sim::BatchSimulator,
    src_x::AbstractVector,
    src_z::AbstractVector;
    store::Bool=true,
    on_shot_complete::Union{Function,Nothing}=nothing,
    progress::Bool=false,
    outputs::Union{OutputConfig,Nothing}=nothing,
)
    n = length(src_x)
    length(src_z) == n || error("src_x and src_z must have same length")

    # 如果开启了 plot_setup，在循环前画出包含所有震源和接收器的设置图
    if outputs !== nothing && outputs.plot_setup
        mkpath(outputs.base_dir)
        plot_setup(sim.working_model, Float32.(src_x), Float32.(src_z), sim.receivers_cfg.x, sim.receivers_cfg.z;
            output=joinpath(outputs.base_dir, "setup.png"))
    end

    nt, nrec = size(sim.receivers.data)
    buf = Matrix{Float32}(undef, nt, nrec)

    gathers = store ? Vector{Matrix{Float32}}(undef, n) : nothing
    for i in 1:n
        g = simulate_shot!(sim, src_x[i], src_z[i];
            gather_out=store ? nothing : buf,
            progress=progress && i == 1,
            outputs=outputs,
            shot_id=outputs === nothing ? nothing : i)

        if store
            gathers[i] = g
            on_shot_complete !== nothing && on_shot_complete(g, i)
        else
            on_shot_complete !== nothing && on_shot_complete(buf, i)
        end
    end

    return gathers
end

function _make_source_at(sim::BatchSimulator, x::Real, z::Real)
    ix = round(Int, x / sim.working_model.dx) + sim.pad + 1
    iz = round(Int, (z + sim.z_off) / sim.working_model.dz) + sim.pad + 1

    m = sim.working_model
    mech = sim.source_template.mechanism
    if mech == Explosion
        Source(ix, iz, sim.wavelet_device)
    elseif mech in (ForceX, ForceZ)
        jz = clamp(round(Int, (z + sim.z_off) / m.dz) + 1, 1, m.nz)
        jx = clamp(round(Int, x / m.dx) + 1, 1, m.nx)
        buoy = 1.0f0 / max(m.rho[jz, jx], 1.0f0)
        ForceSource(ix, iz, sim.wavelet_device, mech == ForceZ ? :vz : :vx, buoy)
    else
        comp = mech == StressTxx ? :txx : mech == StressTzz ? :tzz : :txz
        StressSource(ix, iz, sim.wavelet_device, comp)
    end
end

function _save_shot_result(base_dir::String, gather::Matrix{Float32}, dt::Float32, nt::Int,
    source::SourceConfig, receivers::ReceiverConfig; shot_id::Union{Int,Nothing}=nothing)
    name = shot_id === nothing ? "result.jld2" : "shot_$(lpad(string(shot_id), 4, '0')).jld2"
    path = joinpath(base_dir, name)

    wavelet_kind = source.wavelet isa RickerWavelet ? :ricker : :custom
    wavelet_f0 = source.wavelet isa RickerWavelet ? source.wavelet.f0 : 0.0f0
    wavelet_delay = source.wavelet isa RickerWavelet ? source.wavelet.delay : 0.0f0
    wavelet_data = source.wavelet isa CustomWavelet ? source.wavelet.data : Float32[]

    jldsave(path;
        gather=gather,
        dt=dt,
        nt=nt,
        source_x=source.x,
        source_z=source.z,
        source_mechanism=Int(source.mechanism),
        wavelet_kind=String(wavelet_kind),
        wavelet_f0=wavelet_f0,
        wavelet_delay=wavelet_delay,
        wavelet_data=wavelet_data,
        receivers_x=receivers.x,
        receivers_z=receivers.z,
        receivers_record=Int(receivers.record),
    )
    return path
end
