"""
    AbstractWavelet

震源子波的抽象类型。实现 `generate(w, dt, nt)` 即可自定义子波。
"""
abstract type AbstractWavelet end

"""
    RickerWavelet(f0, delay)

Ricker 子波参数化表示。
"""
struct RickerWavelet <: AbstractWavelet
    f0::Float32
    delay::Float32
end

"""
    Ricker(f0[, delay]) -> RickerWavelet

便捷构造 Ricker 子波。`delay` 默认取 `1/f0`。
"""
Ricker(f0::Real) = RickerWavelet(Float32(f0), 1.0f0 / Float32(f0))
Ricker(f0::Real, delay::Real) = RickerWavelet(Float32(f0), Float32(delay))

"""
    CustomWavelet(data) -> CustomWavelet

使用自定义采样点定义子波（将自动转换为 `Float32`）。
"""
struct CustomWavelet <: AbstractWavelet
    data::Vector{Float32}
end
CustomWavelet(data::AbstractVector) = CustomWavelet(Float32.(data))

"""
    generate(w, dt, nt) -> Vector{Float32}

根据子波 `w` 生成长度为 `nt`、采样间隔为 `dt` 的时间序列。
"""
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

"""
    SourceMechanism

震源机制枚举，用于指定注入的源类型（爆炸源、力源、应力源等）。
"""
@enum SourceMechanism Explosion ForceX ForceZ StressTxx StressTzz StressTxz

@doc "爆炸源（各向同性体积源）" Explosion
@doc "水平力源（Fx）" ForceX
@doc "垂直力源（Fz）" ForceZ
@doc "应力源：Txx 分量" StressTxx
@doc "应力源：Tzz 分量" StressTzz
@doc "应力源：Txz 分量" StressTxz

"""
    SourceConfig(x, z, wavelet[, mechanism])
    SourceConfig(x, z; f0=15.0, type=Explosion)

定义单个震源的位置、子波与机制。
"""
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

"""
    RecordType

检波器记录类型枚举。
"""
@enum RecordType Vz Vx Pressure

@doc "记录垂向速度 Vz" Vz
@doc "记录水平速度 Vx" Vx
@doc "记录压力/体应力（视求解器实现）" Pressure

"""
    ReceiverConfig(x, z[, record])

定义检波器阵列的位置与记录类型。
"""
struct ReceiverConfig
    x::Vector{Float32}
    z::Vector{Float32}
    record::RecordType
end

ReceiverConfig(x::AbstractVector, z::AbstractVector) = ReceiverConfig(Float32.(x), Float32.(z), Vz)
ReceiverConfig(x::AbstractVector, z::AbstractVector, r::RecordType) = ReceiverConfig(Float32.(x), Float32.(z), r)

"""
    line_receivers(x0, x1, n; z=0.0, record=Vz) -> ReceiverConfig

在深度 `z` 处生成一条从 `x0` 到 `x1` 的等间距检波器线（共 `n` 个）。
"""
function line_receivers(x0::Real, x1::Real, n::Int; z::Real=0.0, record::RecordType=Vz)
    ReceiverConfig(Float32.(range(x0, x1, n)), fill(Float32(z), n), record)
end

"""
    Boundary(top, nbc=50, vac=10)

边界配置：
- `top`: `:image`（自由表面镜像）、`:habc`（吸收边界）、`:vacuum`（真空层+吸收）
- `nbc`: 吸收层厚度（网格点数）
- `vac`: 真空层厚度（仅 `:vacuum` 生效）
"""
struct Boundary
    top::Symbol
    nbc::Int
    vacuum_layers::Int

    function Boundary(top::Symbol, nbc::Int=50, vac::Int=10)
        top in (:image, :habc, :vacuum) || error("top must be :image, :habc, or :vacuum")
        new(top, nbc, vac)
    end
end

"""
    FreeSurface(; nbc=50) -> Boundary

自由表面（顶边界镜像），其余边界为吸收层。
"""
FreeSurface(; nbc::Int=50) = Boundary(:image, nbc, 0)

"""
    Absorbing(; nbc=50) -> Boundary

顶边界也使用吸收边界。
"""
Absorbing(; nbc::Int=50) = Boundary(:habc, nbc, 0)

"""
    Vacuum(layers=10; nbc=50) -> Boundary

顶边界使用真空层（常用于更“硬”的自由表面近似），并带吸收层。
"""
Vacuum(layers::Int=10; nbc::Int=50) = Boundary(:vacuum, nbc, layers)

"""
    SimConfig(; nt=3000, dt=nothing, cfl=0.4, fd_order=8, boundary=FreeSurface(), output_dir="outputs")

模拟全局配置。
- `nt`: 时间步数
- `dt`: 时间步长；为 `nothing` 时按 CFL 自动计算
- `cfl`: CFL 系数（仅 `dt=nothing` 时生效）
- `fd_order`: 空间差分阶数
- `boundary`: 边界设置
- `output_dir`: 输出目录（用于视频/结果文件）
"""
struct SimConfig
    nt::Int
    dt::Union{Float32,Nothing}
    cfl::Float32
    fd_order::Int
    boundary::Boundary
    output_dir::String
end

SimConfig(; nt=3000, dt=nothing, cfl=0.4, fd_order=8, boundary=FreeSurface(), output_dir="outputs") =
    SimConfig(nt, dt === nothing ? nothing : Float32(dt), Float32(cfl), fd_order, boundary, String(output_dir))

"""
    VideoSettings(fields, interval, fps, colormap, format, output_dir)

视频/快照设置。
- `fields`: 要保存的场（如 `[:vz]`）
- `interval`: 每隔多少步保存一帧
- `fps`: 输出帧率
- `colormap`: 配色（传给 Plots.jl）
- `format`: `:gif` 或 `:mp4`
- `output_dir`: 视频输出目录；`nothing` 表示使用 `SimConfig.output_dir`
"""
struct VideoSettings
    fields::Vector{Symbol}
    interval::Int
    fps::Int
    colormap::Symbol
    format::Symbol
    output_dir::Union{Nothing,String}
end

Video(; fields=[:vz], interval=50, fps=20, colormap=:seismic, format=:mp4, output_dir=nothing) =
    VideoSettings(fields, interval, fps, colormap, format, output_dir === nothing ? nothing : String(output_dir))

"""
    SimResult

一次模拟的结果容器：
- `gather`: `nt × nrec` 道集
- `dt`, `nt`: 时间采样参数
- `source`, `receivers`: 本次模拟的源与检波器配置
- `snapshots`: 可选快照（用于绘图/视频）
- `video_files`: 可选生成的视频文件路径
- `output_dir`: 本次模拟的输出目录（默认同 `SimConfig.output_dir`）
"""
struct SimResult
    gather::Matrix{Float32}
    dt::Float32
    nt::Int
    source::SourceConfig
    receivers::ReceiverConfig
    snapshots::Union{Dict{Symbol,Array{Float32,3}},Nothing}
    video_files::Union{Dict{Symbol,String},Nothing}
    output_dir::String
end

SimResult(g, dt, nt, s, r; output_dir::AbstractString="outputs") =
    SimResult(g, dt, nt, s, r, nothing, nothing, String(output_dir))
SimResult(g, dt, nt, s, r, snaps; output_dir::AbstractString="outputs") =
    SimResult(g, dt, nt, s, r, snaps, nothing, String(output_dir))

times(r::SimResult) = Float32.(0:r.nt-1) .* r.dt
trace(r::SimResult, i::Int) = r.gather[:, i]
n_receivers(r::SimResult) = size(r.gather, 2)

