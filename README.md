# ElasticWave2D.jl

[🇨🇳 中文](README_zh.md) | **English**

<p align="center">
  <b>GPU-accelerated 2D elastic wave simulation in Julia</b><br>
  <i>Run seismic forward modeling on your laptop </i>
</p>

<p align="center">
  <img src="docs/images/wavefield.gif" width="600" alt="Wavefield Animation">
</p>

## Why This Project?

**ElasticWave2D.jl** provides a simple, Julia-native solution for 2D elastic wave simulation:

- ✅ **One-line install** — Pure Julia, no Fortran/C compilation
- ✅ **Runs on gaming GPUs** — GTX 1060, RTX 3060, etc.
- ✅ **CPU optimized too** — Multi-threaded with `julia -t auto`
- ✅ **Flexible boundaries** — HABC, Image Method, vacuum formulation

## Features

| Feature | Description |
|---------|-------------|
| **GPU Acceleration** | CUDA.jl backend, 10-50x faster than CPU |
| **CPU Optimized** | Multi-threaded kernels via `julia -t auto` |
| **Staggered Grid FD** | 2nd-10th order accuracy (Virieux 1986) |
| **HABC Boundaries** | Higdon Absorbing BC (Ren & Liu 2014) |
| **Image Method** | Accurate free surface BC (Robertsson 1996) |
| **Vacuum Formulation** | Irregular topography, tunnels, cavities (Zeng et al. 2012) |
| **Video Recording** | Wavefield snapshots → MP4/GIF |
| **Multiple Formats** | JLD2, Binary, SEG-Y (planned) |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/ElasticWave2D.jl")
```

**Requirements**: Julia 1.9+. GPU optional (auto-detects CUDA).

### Local Development (clone repo)
From the repo root (key is `--project=.` to activate this environment):

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate(); using ElasticWave2D; println(1)'
```

To `using ElasticWave2D` from anywhere, register the local path into your environment:

```julia
import Pkg
Pkg.develop(path="E:/dev/ElasticWave2D.jl")
```

### Modes & Optional Dependencies
- CPU mode: works without GPU; use `julia -t auto` for multithreading.
- GPU mode: CUDA.jl installed and device functional (auto-detected).
- Optional I/O formats (install on demand): `SegyIO` (SEG-Y), `MAT` (.mat), `NPZ` (.npy). Not in core deps:
  ```julia
  using Pkg
  Pkg.add(["SegyIO","MAT","NPZ"])  # pick any
  ```
  Example (SEG-Y):
  ```julia
  using SegyIO
  # segy = SegyIO.SegyFile("path.segy")
  ```

## Quick Start

```julia
using ElasticWave2D.API
using ElasticWave2D: OutputConfig, resolve_output_path

# Create a simple two-layer model
nx, nz = 200, 100
dx = 10.0f0

vp = fill(2000.0f0, nz, nx)
vs = fill(1200.0f0, nz, nx)
rho = fill(2000.0f0, nz, nx)
vp[50:end, :] .= 3500.0f0  # Faster layer below

model = VelocityModel(vp, vs, rho, dx, dx)

# Recommended: use an isolated output directory per run (flat layout)
outputs = OutputConfig(base_dir="outputs/quickstart")

# Run simulation
result = simulate(
    model,
    SourceConfig(1000.0, 20.0; f0=20.0),           # Source at (1000m, 20m depth)
    line_receivers(100.0, 1900.0, 181; z=10.0);    # 181 receivers
    config = SimConfig(nt=1000, boundary=Vacuum(10)),
    outputs = outputs
)

# Access results
println("Gather size: ", size(result.gather))
plot_gather(result; output=resolve_output_path(outputs, :figures, "gather.png"))
save_result(result, resolve_output_path(outputs, :results, "result.jld2"))
```

## Examples

### 🎬 Elastic Wave Demo
High-resolution wave propagation in a two-layer medium with video output.

```julia
using ElasticWave2D.API

model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)

result = simulate(
    model,
    SourceConfig(2000.0, 50.0, Ricker(15.0)),
    line_receivers(100, 3900, 191);
    config = SimConfig(nt=3000, boundary=FreeSurface()),
    outputs = OutputConfig(base_dir="outputs/elastic_wave_demo"),
    video = Video(fields=[:vz], interval=20, fps=30)
)
```

<p align="center">
  <img src="docs/images/elastic_wave_setup.png" width="400" alt="Elastic Wave Setup">
  <img src="docs/images/elastic_wave_gather.png" width="400" alt="Elastic Wave Gather">
</p>

---

### 🏗️ Tunnel Detection (Engineering)
Detect underground cavities using seismic diffraction. Uses vacuum formulation for both free surface and tunnel cavity.

```julia
using ElasticWave2D: OutputConfig

# Create model with tunnel (set ρ=0 for void)
rho[40:45, 95:105] .= 0.0f0  # Tunnel cavity

result = simulate(
    model,
    SourceConfig(500.0, 10.0; f0=50.0),
    line_receivers(100, 900, 81);
    config = SimConfig(nt=2000, boundary=Vacuum(10)),
    outputs = OutputConfig(base_dir="outputs/tunnel_demo")
)
```

<p align="center">
  <img src="docs/images/tunnel_setup.png" width="400" alt="Tunnel Setup">
  <img src="docs/images/tunnel_gather.png" width="400" alt="Tunnel Gather">
</p>

**What to look for**: Diffracted waves from tunnel edges, shadow zone behind tunnel.

---

### 🛢️ Exploration Seismic (Petroleum)
Image an anticlinal structure — a classic hydrocarbon trap.

<p align="center">
  <img src="docs/images/exploration_setup.png" width="400" alt="Exploration Setup">
  <img src="docs/images/exploration_gather.png" width="400" alt="Exploration Gather">
</p>

**What to look for**: Reflection "pull-up" at anticline crest, multiple layer reflections.

---

### 🔬 Boundary Comparison

| Method | Surface Waves | Use Case |
|--------|--------------|----------|
| `Absorbing()` | ❌ | Body waves only |
| `FreeSurface()` | ✅ | Accurate flat surface (Image Method) |
| `Vacuum(n)` | ✅ | Topography, cavities (recommended) |

```julia
# Compare different boundaries
for boundary in [Absorbing(), FreeSurface(), Vacuum(10)]
    outputs = OutputConfig(base_dir="outputs/boundary_$(boundary.top)")
    result = simulate(model, source, receivers;
        config = SimConfig(nt=2000, boundary=boundary),
        outputs = outputs)
end
```

<p align="center">
  <img src="docs/images/freesurface_gather.png" width="400" alt="Free Surface">
  <img src="docs/images/vacuum_gather.png" width="400" alt="Vacuum">
</p>
<p align="center">
  <i>Left: Image Method | Right: Vacuum formulation — nearly identical results</i>
</p>

## API Reference

### Outputs

Use `OutputConfig(base_dir=...)` to isolate outputs per run. Output layout is flat: results / figures / videos / manifests are written directly into `base_dir`.

### Core Types

```julia
# Wavelet
Ricker(f0)                    # Ricker wavelet with frequency f0
Ricker(f0, delay)             # With custom delay
CustomWavelet(data)           # User-provided wavelet

# Source
SourceConfig(x, z; f0=15.0)                    # Quick setup
SourceConfig(x, z, Ricker(15.0), ForceZ)       # Full control
# Mechanisms: Explosion, ForceX, ForceZ, StressTxx, StressTzz, StressTxz

# Receivers
line_receivers(x0, x1, n; z=0.0)              # Line of receivers
ReceiverConfig(x_vec, z_vec)                   # Custom positions
ReceiverConfig(x_vec, z_vec, Vx)              # Record Vx instead of Vz

# Boundary
FreeSurface()      # Image Method (flat surface)
Absorbing()        # HABC on all sides
Vacuum(n)          # n vacuum layers at top (recommended)

# Configuration
SimConfig(
    nt = 3000,           # Time steps
    dt = nothing,        # Auto-compute from CFL
    cfl = 0.4,           # CFL number
    fd_order = 8,        # FD accuracy (2,4,6,8,10)
    boundary = Vacuum(10),
    output_dir = "outputs"  # Legacy: prefer outputs=OutputConfig(base_dir=...) for output paths
)

# Video
Video(
    fields = [:vz],      # Fields to record
    interval = 50,       # Save every N steps
    fps = 20,
    format = :mp4        # :mp4 or :gif
)
```

### Main Functions

```julia
# Run simulation
result = simulate(model, source, receivers; config, video=nothing)

# Batch simulation (high performance)
using ElasticWave2D
sim = BatchSimulator(model, rec_x, rec_z; nt=3000, f0=15.0)
gathers = simulate_shots!(sim, src_x_vec, src_z_vec)

# I/O
save_result(result, "shot_001.jld2")
result = load_result("shot_001.jld2")

# Plotting
plot_gather(result)

# Requires `using Plots`
plot_trace(result, 50)
```

### Result Structure

```julia
result.gather      # [nt × n_receivers] seismogram
result.dt          # Time step used
result.nt          # Number of time steps
result.snapshots   # Dict of wavefield snapshots (if video enabled)
```

## Performance

**GPU** (RTX 3060, 12GB):

| Grid Size | Time Steps | Runtime |
|-----------|------------|---------|
| 400×200 | 3000 | ~8 sec |
| 800×400 | 5000 | ~45 sec |
| 1200×600 | 8000 | ~3 min |

**CPU** (8-core, with `-t auto`): ~10-20x slower than GPU, but still practical for small-medium models.

### Batch Performance

```julia
# Benchmark multi-shot performance
using ElasticWave2D
result = benchmark_shots(model, rec_x, rec_z, src_x, src_z; nt=3000, f0=15.0)
# Typical: 0.1-0.3 sec/shot on GPU for medium grids
```

## Why I Built This

As a geophysics student, I wanted a simple tool to run seismic forward modeling on my laptop without complex setup. Existing tools like SOFI2D and SPECFEM are powerful and well-established, but I needed something lightweight for quick experiments and learning.

I also found that HABC (Higdon Absorbing Boundary Conditions) offers good absorption quality with better computational efficiency than PML, which is helpful when running on consumer hardware.

So I built ElasticWave2D.jl — a simple, Julia-native tool for 2D elastic wave simulation. If you need a quick way to experiment with seismic modeling, this might be useful for you.

## Project Structure

```
ElasticWave2D.jl/
├── src/
│   ├── api/                # High-level API (recommended entry point)
│   ├── domain/             # User-facing types (wavelet/source/receiver/boundary/config/result)
│   ├── compute/            # Hardware abstraction (CPU/GPU)
│   ├── core/               # Fundamental types (Wavefield, Medium)
│   ├── physics/            # Numerical kernels (velocity, stress, boundaries)
│   ├── initialization/     # Setup routines (media, topography)
│   ├── solver/             # Time-stepping and batch execution
│   ├── io/                 # Input/Output
│   ├── outputs/            # Output paths & artifacts (flat output)
│   └── visualization/      # Plotting and video
├── examples/               # Usage examples
├── test/                   # Unit tests
└── docs/                   # Documentation
```

## References

1. Virieux, J. (1986). P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method. *Geophysics*, 51(4), 889-901.

2. Zeng, C., Xia, J., Miller, R. D., & Tsoflias, G. P. (2012). An improved vacuum formulation for 2D finite-difference modeling of Rayleigh waves including surface topography and internal discontinuities. *Geophysics*, 77(1), T1-T9.

3. Ren, Z., & Liu, Y. (2014). A Higdon absorbing boundary condition. *Journal of Geophysics and Engineering*, 11(6), 065007.

## Citation

If you use ElasticWave2D.jl in your research, please cite:

```bibtex
@software{elasticwave2d,
  author = {Wu Heng},
  title = {ElasticWave2D.jl: GPU-accelerated 2D Elastic Wave Simulation},
  url = {https://github.com/Wuheng10086/ElasticWave2D.jl},
  year = {2025}
}
```

## Contributing

Contributions welcome! Please open an issue or PR.

## License

MIT License
