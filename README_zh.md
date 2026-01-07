# Fomo.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9%20|%201.10%20|%201.11-blue)](https://julialang.org/)

[中文文档](README_zh.md) | [English](README.md)

**Fomo** - **Fo**rward **Mo**deling：高性能二维各向同性弹性波数值模拟器。

## ✨ 特性

- 🚀 **后端调度架构** - 一套代码，CPU/GPU 自动切换
- 📐 **高阶交错网格有限差分** - 支持 2 至 8 阶空间精度
- 🛡️ **混合吸收边界 (HABC)** - 有效抑制边界反射
- 🌊 **自由地表建模** - 准确模拟 Rayleigh 面波
- ⚡ **多 GPU 并行** - 自动负载均衡，榨干显卡性能
- 📁 **多格式支持** - SEG-Y、Binary、MAT、NPY、HDF5、JLD2
- 🎬 **视频录制** - 实时波场可视化

## 📋 系统要求

- **Julia 1.9、1.10 或 1.11**（暂不支持 1.12，CairoMakie 兼容性问题）
- CUDA 显卡（可选，用于 GPU 加速）

## 🔧 安装

### 从 GitHub 安装

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/Fomo.jl")
```

### 本地开发

```bash
git clone https://github.com/Wuheng10086/Fomo.jl.git
cd Fomo.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### 可选依赖

读取不同格式的模型文件：

```julia
using Pkg
Pkg.add("SegyIO")  # SEG-Y 文件
Pkg.add("MAT")     # MATLAB 文件  
Pkg.add("NPZ")     # NumPy 文件
Pkg.add("HDF5")    # HDF5 文件
```

## 🚀 快速开始

```julia
using Fomo

# 创建速度模型
vp = fill(3000.0f0, 200, 100)
vs = fill(1800.0f0, 200, 100)
rho = fill(2200.0f0, 200, 100)

# 添加一个层
vp[:, 50:end] .= 4000.0f0
vs[:, 50:end] .= 2400.0f0

model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0; name="双层模型")

# 自动选择后端（有 GPU 就用 GPU）
be = is_cuda_available() ? backend(:cuda) : backend(:cpu)

# 初始化模拟
nbc, fd_order = 50, 8
medium = init_medium(model, nbc, fd_order, be; free_surface=true)

# 时间步进
dt = 0.5f0 * 10.0f0 / maximum(vp)
nt = 2000
habc = init_habc(medium.nx, medium.nz, nbc, dt, 10.0f0, 10.0f0, 3500.0f0, be)
params = SimParams(dt, nt, 10.0f0, 10.0f0, fd_order)

# 观测系统
rec_x = Float32.(0:20:1990)
rec_z = fill(10.0f0, length(rec_x))
rec = setup_receivers(rec_x, rec_z, medium; type=:vz)

src_x = Float32[1000.0]
src_z = Float32[20.0]
wavelet = ricker_wavelet(15.0f0, dt, nt)
shots = MultiShotConfig(src_x, src_z, wavelet)

# 运行模拟
fd_coeffs = to_device(get_fd_coefficients(fd_order), be)
wavefield = Wavefield(medium.nx, medium.nz, be)
results = run_shots!(be, wavefield, medium, habc, fd_coeffs, rec, shots, params)

# 保存结果
save_gather(results[1], "gather.bin")
```

## 📁 加载模型

```julia
using Fomo

# 从 JLD2 加载（推荐）
model = load_model("marmousi.jld2")

# 从分离的 SEG-Y 文件加载（需要 SegyIO）
using SegyIO
model = load_model_files(
    vp = "vp.segy",
    vs = "vs.segy", 
    rho = "rho.segy",
    dx = 12.5
)

# 保存为 JLD2 格式，下次加载更快
save_model("model.jld2", model)
```

## ⚡ 多 GPU 并行

```julia
using Fomo

model = load_model("marmousi.jld2")

# 定义观测系统
src_x = Float32.(100:200:16900)
src_z = fill(10.0f0, length(src_x))
rec_x = Float32.(0:15:17000)
rec_z = fill(20.0f0, length(rec_x))

wavelet = ricker_wavelet(25.0f0, dt, nt)
params = SimParams(dt, nt, model.dx, model.dz, 8)

# 自动使用所有可用 GPU！
results = run_shots_auto!(
    model, rec_x, rec_z, src_x, src_z, wavelet, params;
    nbc=50, fd_order=8, output_dir="outputs/"
)
```

## 🔍 设置验证

运行大规模模拟前，先检查观测系统设置：

```julia
using Fomo

model = load_model("model.jld2")

# 定义震源和检波器
src_x = Float32.(100:200:3000)
src_z = fill(10.0f0, length(src_x))
rec_x = Float32.(0:15:3500)
rec_z = fill(50.0f0, length(rec_x))

# 生成设置检查图
plot_setup(model, src_x, src_z, rec_x, rec_z; 
           output="setup_check.png",
           title="观测系统设置")
```

## 🎬 视频录制

```julia
using Fomo

# 配置视频录制
config = VideoConfig(
    fields = [:p],      # 录制压力场
    skip = 10,          # 每 10 步保存一帧
    downsample = 2      # 空间降采样
)

recorder = MultiFieldRecorder(medium.nx, medium.nz, dt, config)

# 带录制回调运行
results = run_shots!(be, wavefield, medium, habc, fd_coeffs,
                     rec, shots, params;
                     on_step = recorder)

# 生成 MP4 视频
generate_video(recorder.recorder, "wavefield.mp4"; fps=30)
```

## 🛠️ 命令行工具

```bash
# 转换模型格式
julia --project=. scripts/convert_model.jl \
    --vp=vp.segy --vs=vs.segy --rho=rho.segy \
    -o model.jld2 --dx=12.5 --transpose

# 检查模型维度
julia --project=. scripts/check_model.jl model.jld2 --fix

# 运行并行模拟
julia --project=. examples/run_parallel.jl model.jld2 outputs/
```

## 📂 项目结构

```
Fomo.jl/
├── src/
│   ├── Fomo.jl              # 主模块
│   ├── backends/            # CPU/CUDA 抽象层
│   ├── kernels/             # 有限差分核函数
│   ├── simulation/          # 炮管理
│   ├── io/                  # 模型/观测系统 I/O
│   └── visualization/       # 绑图 & 视频
├── examples/                # 使用示例
├── scripts/                 # 命令行工具
├── test/                    # 单元测试
└── docs/                    # 文档
```

## 🧪 运行测试

```bash
cd Fomo.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

## 📚 API 概览

### 核心类型
- `VelocityModel` - 速度模型容器
- `Medium` - 计算网格与材料属性
- `Wavefield` - 波场数组 (vx, vz, txx, tzz, txz)
- `SimParams` - 模拟参数

### 主要函数
- `init_medium()` - 初始化计算介质
- `init_habc()` - 初始化吸收边界
- `run_shots!()` - 顺序执行多炮
- `run_shots_auto!()` - 自动多 GPU 并行
- `load_model()` / `save_model()` - 模型读写
- `plot_setup()` - 可视化观测系统

## 📖 参考文献

1. Luo, Y., & Schuster, G. (1990). *Parsimonious staggered grid finite-differencing of the wave equation*. Geophysical Research Letters.

2. Liu, Y., & Sen, M. K. (2012). *A hybrid absorbing boundary condition for elastic staggered-grid modelling*. Geophysical Prospecting.

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 👤 作者

zswh - 2025
