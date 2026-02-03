"""
    AbstractWavelet

Abstract type for source wavelets. Implement `generate(w, dt, nt)` for custom wavelets.
震源子波的抽象类型。实现 `generate(w, dt, nt)` 即可自定义子波。
"""
abstract type AbstractWavelet end

"""
    RickerWavelet(f0, delay)

Ricker wavelet (Mexican hat) parametric representation.
Ricker 子波（墨西哥帽）参数化表示。

# Fields / 字段
- `f0::Float32`: Peak frequency in Hz. 峰值频率（Hz）。
- `delay::Float32`: Time delay in seconds. 时间延迟（秒）。

# See Also / 参见
- [`Ricker`](@ref): Convenience constructor. / 便捷构造函数。
"""
struct RickerWavelet <: AbstractWavelet
    f0::Float32
    delay::Float32
end

"""
    Ricker(f0[, delay]) -> RickerWavelet

Convenience constructor for Ricker wavelet. Default `delay = 1/f0`.
便捷构造 Ricker 子波。`delay` 默认取 `1/f0`。

# Arguments / 参数
- `f0::Real`: Peak frequency in Hz. 峰值频率（Hz）。
- `delay::Real`: Time delay in seconds (optional). 时间延迟（秒，可选）。

# Example / 示例
```julia
wavelet = Ricker(20.0)        # 20 Hz, auto delay
wavelet = Ricker(20.0, 0.1)   # 20 Hz, 0.1s delay
```
"""
Ricker(f0::Real) = RickerWavelet(Float32(f0), 1.0f0 / Float32(f0))
Ricker(f0::Real, delay::Real) = RickerWavelet(Float32(f0), Float32(delay))

"""
    CustomWavelet(data) -> CustomWavelet

Create wavelet from custom sample data (auto-converts to Float32).
使用自定义采样点定义子波（将自动转换为 `Float32`）。

# Arguments / 参数
- `data::AbstractVector`: Wavelet samples. 子波采样点。

# Example / 示例
```julia
data = sin.(2π * 20 * (0:0.001:0.1))
wavelet = CustomWavelet(data)
```
"""
struct CustomWavelet <: AbstractWavelet
    data::Vector{Float32}
end
CustomWavelet(data::AbstractVector) = CustomWavelet(Float32.(data))

"""
    generate(w::AbstractWavelet, dt::Float32, nt::Int) -> Vector{Float32}

Generate time series of length `nt` with sampling interval `dt` from wavelet `w`.
根据子波 `w` 生成长度为 `nt`、采样间隔为 `dt` 的时间序列。

# Arguments / 参数
- `w`: Wavelet object. 子波对象。
- `dt`: Time step in seconds. 时间步长（秒）。
- `nt`: Number of time steps. 时间步数。

# Returns / 返回
- `Vector{Float32}`: Wavelet time series. 子波时间序列。
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

Source mechanism enumeration for specifying injection type.
震源机制枚举，用于指定注入的源类型。

# Values / 枚举值
- `Explosion`: Isotropic volume source (explosive). 爆炸源（各向同性体积源）。
- `ForceX`: Horizontal force source (Fx). 水平力源。
- `ForceZ`: Vertical force source (Fz). 垂直力源。
- `StressTxx`: Stress source, Txx component. 应力源 Txx 分量。
- `StressTzz`: Stress source, Tzz component. 应力源 Tzz 分量。
- `StressTxz`: Stress source, Txz component. 应力源 Txz 分量。
"""
@enum SourceMechanism Explosion ForceX ForceZ StressTxx StressTzz StressTxz

@doc "Explosive source (isotropic volume source). 爆炸源（各向同性体积源）" Explosion
@doc "Horizontal force source (Fx). 水平力源（Fx）" ForceX
@doc "Vertical force source (Fz). 垂直力源（Fz）" ForceZ
@doc "Stress source: Txx component. 应力源 Txx 分量" StressTxx
@doc "Stress source: Tzz component. 应力源 Tzz 分量" StressTzz
@doc "Stress source: Txz component. 应力源 Txz 分量" StressTxz

"""
    SourceConfig(x, z, wavelet[, mechanism])
    SourceConfig(x, z; f0=15.0, type=Explosion)

Define a single source with position, wavelet, and mechanism.
定义单个震源的位置、子波与机制。

# Arguments / 参数
- `x::Real`: Source X position in meters. 震源 X 坐标（米）。
- `z::Real`: Source Z position (depth) in meters. 震源 Z 坐标/深度（米）。
- `wavelet::AbstractWavelet`: Source wavelet. 震源子波。
- `mechanism::SourceMechanism`: Source type (default `Explosion`). 震源机制（默认爆炸源）。

# Keyword Arguments / 关键字参数
- `f0::Real`: Peak frequency for auto-created Ricker wavelet. 自动创建 Ricker 子波的峰值频率。
- `type::SourceMechanism`: Source mechanism. 震源机制。

# Example / 示例
```julia
# With explicit wavelet / 显式指定子波
source = SourceConfig(500.0, 10.0, Ricker(20.0))
source = SourceConfig(500.0, 10.0, Ricker(20.0), ForceZ)

# With keyword arguments / 使用关键字参数
source = SourceConfig(500.0, 10.0; f0=20.0)
source = SourceConfig(500.0, 10.0; f0=20.0, type=ForceZ)
```
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

Receiver recording type enumeration.
检波器记录类型枚举。

# Values / 枚举值
- `Vz`: Record vertical velocity component. 记录垂向速度分量。
- `Vx`: Record horizontal velocity component. 记录水平速度分量。
- `Pressure`: Record pressure/bulk stress. 记录压力/体应力。
"""
@enum RecordType Vz Vx Pressure

@doc "Record vertical velocity Vz. 记录垂向速度 Vz" Vz
@doc "Record horizontal velocity Vx. 记录水平速度 Vx" Vx
@doc "Record pressure/bulk stress. 记录压力/体应力" Pressure

"""
    ReceiverConfig(x, z[, record])

Define receiver array positions and recording type.
定义检波器阵列的位置与记录类型。

# Arguments / 参数
- `x::AbstractVector`: X positions of receivers in meters. 检波器 X 坐标向量（米）。
- `z::AbstractVector`: Z positions of receivers in meters. 检波器 Z 坐标向量（米）。
- `record::RecordType`: Recording type (default `Vz`). 记录类型（默认 `Vz`）。

# Example / 示例
```julia
# Custom receiver positions / 自定义检波器位置
x = [100.0, 200.0, 300.0]
z = [10.0, 10.0, 10.0]
receivers = ReceiverConfig(x, z)
receivers = ReceiverConfig(x, z, Vx)  # record Vx / 记录 Vx
```

# See Also / 参见
- [`line_receivers`](@ref): Create linear receiver array. / 创建线性检波器阵列。
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

Create a linear receiver array from `x0` to `x1` with `n` receivers at depth `z`.
在深度 `z` 处生成一条从 `x0` 到 `x1` 的等间距检波器线（共 `n` 个）。

# Arguments / 参数
- `x0::Real`: Starting X position in meters. 起始 X 坐标（米）。
- `x1::Real`: Ending X position in meters. 结束 X 坐标（米）。
- `n::Int`: Number of receivers. 检波器数量。

# Keyword Arguments / 关键字参数
- `z::Real`: Receiver depth in meters (default 0.0). 检波器深度（米），默认 0.0。
- `record::RecordType`: Recording type (default `Vz`). 记录类型，默认 `Vz`。

# Returns / 返回
- `ReceiverConfig`: Configured receiver array. 配置好的检波器阵列。

# Example / 示例
```julia
# 91 receivers from x=100m to x=1900m at z=10m
# 在 x=100m 到 x=1900m、深度 z=10m 处布置 91 个检波器
receivers = line_receivers(100.0, 1900.0, 91; z=10.0)

# Record horizontal velocity / 记录水平速度
receivers = line_receivers(0.0, 1000.0, 101; z=5.0, record=Vx)
```
"""
function line_receivers(x0::Real, x1::Real, n::Int; z::Real=0.0, record::RecordType=Vz)
    ReceiverConfig(Float32.(range(x0, x1, n)), fill(Float32(z), n), record)
end
