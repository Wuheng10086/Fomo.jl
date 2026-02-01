# ElasticWave2D.jl

[English](README.md) | **🇨🇳 中文**

<p align="center">
  <b>基于Julia的GPU加速二维弹性波模拟</b><br>
  <i>在笔记本上运行地震正演模拟 - 无需集群，无需复杂配置</i>
</p>

<p align="center">
  <img src="docs/images/wavefield.gif" width="600" alt="波场动画">
</p>

## 为什么选择这个项目？

传统地震模拟代码安装困难、文档匮乏、依赖HPC集群。**ElasticWave2D.jl** 不一样：

- ✅ **一行安装** — 纯Julia，无需编译Fortran/C
- ✅ **游戏显卡可用** — GTX 1060、RTX 3060 等
- ✅ **CPU也优化** — 使用 `julia -t auto` 多线程加速
- ✅ **学生友好** — 清晰示例，可读代码
- ✅ **灵活边界** — HABC、自由表面、真空公式

## 功能特性

| 特性 | 描述 |
|------|------|
| **GPU加速** | CUDA.jl后端，比CPU快10-50倍 |
| **CPU优化** | 多线程内核，`julia -t auto` |
| **交错网格有限差分** | 2-10阶精度 (Virieux 1986) |
| **HABC边界** | Higdon吸收边界条件 (Ren & Liu 2014) |
| **真空公式** | 支持不规则地形、隧道、溶洞 (Zeng et al. 2012) |
| **视频录制** | 波场快照 → MP4 |
| **多种格式** | SEG-Y, Binary, HDF5, NPY, MAT, JLD2 |

## 安装

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/ElasticWave2D.jl")
```

**环境要求**：Julia 1.9+，GPU可选（自动检测CUDA）

## 快速开始

```julia
using ElasticWave2D

# 创建简单的双层模型
nx, nz = 200, 100
dx, dz = 10.0f0, 10.0f0

vp = fill(2000.0f0, nz, nx)
vs = fill(1200.0f0, nz, nx)
rho = fill(2000.0f0, nz, nx)
vp[50:end, :] .= 3500.0f0  # 下层速度更快

model = VelocityModel(vp, vs, rho, dx, dz)

# 观测系统
src_x, src_z = 1000.0f0, 20.0f0
rec_x = Float32.(collect(100:10:1900))
rec_z = fill(10.0f0, length(rec_x))

# 使用真空自由表面运行模拟
result = seismic_survey(
    model,
    (src_x, src_z),
    (rec_x, rec_z);
    surface_method = :vacuum,    # :vacuum, :free_surface, 或 :absorbing
    vacuum_layers = 5,
    config = SimulationConfig(nt=1000, f0=20.0f0)
)
```

## 示例

### 🎬 弹性波演示
双层介质中的高分辨率波传播，带视频输出。

```bash
julia -t auto examples/elastic_wave_demo.jl
```

<p align="center">
  <img src="docs/images/elastic_wave_setup.png" width="400" alt="模型设置">
  <img src="docs/images/elastic_wave_gather.png" width="400" alt="地震道集">
</p>

---

### 🏗️ 隧道检测（工程应用）
利用地震波绕射检测地下空洞。真空公式同时用于自由表面和隧道空腔。

```bash
julia -t auto examples/tunnel_detection_demo.jl
```

<p align="center">
  <img src="docs/images/tunnel_setup.png" width="400" alt="隧道模型">
  <img src="docs/images/tunnel_gather.png" width="400" alt="隧道道集">
</p>

**观察要点**：隧道边缘的绕射波、隧道后方的阴影区。

---

### 🛢️ 勘探地震（石油勘探）
对背斜构造成像——经典的油气圈闭。

```bash
julia -t auto examples/exploration_seismic_demo.jl
```

<p align="center">
  <img src="docs/images/exploration_setup.png" width="400" alt="勘探模型">
  <img src="docs/images/exploration_gather.png" width="400" alt="勘探道集">
</p>

**观察要点**：背斜顶部的反射"上拉"现象、多层反射。

---

### 🔬 边界条件对比
并排比较不同的地表处理方法。

```bash
julia -t auto examples/seismic_survey_demo.jl
```

| 方法 | 面波 | 适用场景 |
|------|------|----------|
| `:absorbing` | ❌ | 仅体波 |
| `:free_surface` | ✅ | 经典显式边界条件 |
| `:vacuum` | ✅ | 统一方法（推荐） |

**面波对比** — 两种方法都能产生Rayleigh波，结果几乎一致：

<p align="center">
  <img src="docs/images/freesurface_gather.png" width="400" alt="显式自由表面">
  <img src="docs/images/vacuum_gather.png" width="400" alt="真空公式">
</p>
<p align="center">
  <i>左：显式自由表面边界条件 | 右：真空公式</i>
</p>

真空方法提供更大灵活性（支持地形、内部空洞），精度相当。

## API 参考

### `seismic_survey` — 高级接口

```julia
seismic_survey(model, source, receivers;
    surface_method = :vacuum,     # :vacuum, :free_surface, :absorbing
    vacuum_layers = 10,           # 真空层数（仅用于 :vacuum）
    config = SimulationConfig(),
    video_config = nothing
)
```

### `simulate!` — 底层接口

```julia
result = simulate!(model, src_x, src_z, rec_x, rec_z;
    config = SimulationConfig(
        nt = 3000,           # 时间步数
        f0 = 15.0f0,         # 震源主频 (Hz)
        fd_order = 8,        # 有限差分精度阶数
        free_surface = true, # 显式自由表面边界条件
        output_dir = "outputs"
    ),
    video_config = VideoConfig(
        fields = [:vz],      # 记录垂直速度分量
        skip = 20,           # 帧间隔
        fps = 30
    )
)
```

### 地表方法对比

| 参数 | `free_surface=true` | `surface_method=:vacuum` |
|------|---------------------|--------------------------|
| 实现方式 | 显式边界条件 | 顶部ρ=0层 |
| 地形 | ❌ 仅平面 | ✅ 任意形状 |
| 内部空洞 | ❌ | ✅ 隧道、溶洞 |
| 一致性 | — | 全域物理一致 |

## 性能

**GPU** (RTX 3060, 12GB):

| 网格大小 | 时间步数 | 运行时间 |
|----------|----------|----------|
| 400×200 | 3000 | ~8 秒 |
| 800×400 | 5000 | ~45 秒 |
| 1200×600 | 8000 | ~3 分钟 |

**CPU** (8核，使用 `-t auto`)：比GPU慢约10-20倍，但中小规模模型仍然实用。

## 为什么写这个项目

作为一个地球物理专业的学生，我被现有的地震模拟工具折磨得够呛——SOFI2D、SPECFEM这些软件都要在Linux上跑，需要配置`make`，各种依赖问题让人头大。我只是想在自己的笔记本上跑个正演，却要花好几天配环境。

另外我发现PML（完美匹配层）边界计算量很大。而HABC（Higdon吸收边界条件）能达到相近的吸收效果，效率却高得多——当你用的是游戏显卡而不是超算集群时，这很重要。

所以我写了ElasticWave2D.jl——一个我希望自己刚入门时就有的工具。如果你也是一个只有笔记本电脑但充满好奇心的学生，这个项目就是为你准备的。

## 参考文献

1. Virieux, J. (1986). P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. *Geophysics*, 51(4), 889-901.

2. Zeng, C., Xia, J., Miller, R. D., & Tsoflias, G. P. (2012). An improved vacuum formulation for 2D finite-difference modeling of Rayleigh waves including surface topography and internal discontinuities. *Geophysics*, 77(1), T1-T9.

3. Ren, Z., & Liu, Y. (2014). A Higdon absorbing boundary condition. *Journal of Geophysics and Engineering*, 11(6), 065007.

## 引用

如果您在研究中使用了 ElasticWave2D.jl，请引用：

```bibtex
@software{elasticwave2d,
  author = {Wu Heng},
  title = {ElasticWave2D.jl: GPU-accelerated 2D Elastic Wave Simulation},
  url = {https://github.com/Wuheng10086/ElasticWave2D.jl},
  year = {2025}
}
```

## 为什么写这个项目

作为一名地球物理学生，我被现有工具折磨过：

- **SOFI3D、Specfem2D** — 需要Linux、`make`、MPI配置……我花在调试编译错误上的时间比做研究还多。
- **PML边界条件** — 虽然广泛使用，但计算量大。HABC用更少的层数和计算量就能达到类似的吸收效果。

所以我写了 ElasticWave2D.jl：一个**开箱即用**的工具 — `Pkg.add()` 就能跑。不用cmake，不用Fortran编译器，不用配MPI。

如果你是一个只想跑几个模拟、学习波动物理的学生，这个项目就是为你准备的。

## 目录结构

本项目采用领域驱动设计（Domain-Driven Design）重构了目录结构，以确保清晰度和可维护性：

```
ElasticWave2D.jl/
├── src/
│   ├── compute/            # 算力后端 (CPU/GPU 抽象层)
│   ├── core/               # 核心类型 (Wavefield, Medium, Configs)
│   ├── physics/            # 物理内核 (速度/应力更新, 边界条件)
│   ├── initialization/     # 初始化例程 (介质建模, 地形生成)
│   ├── solver/             # 求解器 (时间步进, 炮集管理, 并行计算)
│   ├── workflow/           # 工作流 (高级用户 API)
│   ├── io/                 # 数据读写 (模型加载, 地震数据 IO)
│   └── visualization/      # 可视化 (绘图与视频生成)
├── examples/               # 使用示例
├── tests/                  # 单元与集成测试
├── docs/                   # 文档
└── scripts/                # 实用脚本
```

## 贡献

## 许可证

MIT License