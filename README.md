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
using ElasticWave2D

# Create a simple two-layer model
nx, nz = 200, 100
dx = 10.0f0

vp = fill(2000.0f0, nz, nx)
vs = fill(1200.0f0, nz, nx)
rho = fill(2000.0f0, nz, nx)
vp[50:end, :] .= 3500.0f0  # Faster layer below

model = VelocityModel(vp, vs, rho, dx, dx)

# Output settings (all artifacts written into this directory)
outputs = OutputConfig(base_dir="outputs/quickstart", plot_gather=true, video_config=nothing)
boundary = top_vacuum(10)  # or top_image(), top_absorbing()
simconf = SimConfig(nt=1000, cfl=0.4, fd_order=8, dt=nothing)

# Run simulation
gather = simulate(
    model,
    SourceConfig(1000.0, 20.0; f0=20.0),           # Source at (1000m, 20m depth)
    line_receivers(100.0, 1900.0, 181; z=10.0),    # 181 receivers
    boundary,
    outputs,
    simconf
)

println("Gather size: ", size(gather))
# Files created:
# - outputs/quickstart/result.jld2
# - outputs/quickstart/gather.png (if plot_gather=true)
```

## Batch Shots (Multi-Shot Simulation)

For efficient multi-shot surveys, use `BatchSimulator` to pre-allocate resources once:

```julia
using ElasticWave2D

# Setup model (same as single shot)
model = VelocityModel(vp, vs, rho, dx, dx)

# Create batch simulator (allocates GPU/CPU resources once)
src_template = SourceConfig(0.0, 0.0; f0=20.0)  # position will be overridden
receivers = line_receivers(100.0, 1900.0, 181; z=10.0)
boundary = top_image(nbc=50)
simconf = SimConfig(nt=2000)

sim = BatchSimulator(model, src_template, receivers, boundary, simconf)

# Define shot positions
src_x = Float32.(500:200:2500)  # 11 shots
src_z = fill(20.0f0, length(src_x))

# Run all shots (store results in memory)
gathers = simulate_shots!(sim, src_x, src_z)

# Or with output files and callback
outputs = OutputConfig(base_dir="outputs/batch", plot_gather=true, plot_setup=true)
simulate_shots!(sim, src_x, src_z; store=false, outputs=outputs) do gather, i
    @info "Shot $i completed" size=size(gather)
end
```

For single shots with a pre-allocated simulator:

```julia
gather1 = simulate_shot!(sim, 500.0f0, 20.0f0)
gather2 = simulate_shot!(sim, 600.0f0, 20.0f0; progress=true)
```

## Examples

### 🎬 Elastic Wave Demo
High-resolution wave propagation in a two-layer medium with video output.

```julia
using ElasticWave2D

model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)

outputs = OutputConfig(
    base_dir="outputs/elastic_wave_demo",
    plot_gather=true,
    video_config=VideoConfig(fields=[:vz], skip=20, fps=30),
)

gather = simulate(
    model,
    SourceConfig(2000.0, 50.0, Ricker(15.0)),
    line_receivers(100, 3900, 191),
    top_image(),
    outputs,
    SimConfig(nt=3000)
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
using ElasticWave2D

# Create model with tunnel (set ρ=0 for void)
nx, nz, dx = 200, 100, 5.0f0
vp = fill(2000.0f0, nz, nx)
vs = fill(1200.0f0, nz, nx)
rho = fill(2000.0f0, nz, nx)
rho[40:45, 95:105] .= 0.0f0  # Tunnel cavity
vp[40:45, 95:105] .= 0.0f0
vs[40:45, 95:105] .= 0.0f0

model = VelocityModel(vp, vs, rho, dx, dx)

gather = simulate(
    model,
    SourceConfig(250.0, 10.0; f0=60.0),
    line_receivers(50.0, 950.0, 200; z=10.0),
    top_vacuum(10),
    OutputConfig(base_dir="outputs/tunnel_demo", plot_gather=true),
    SimConfig(nt=1500)
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
| `top_absorbing()` | ❌ | Body waves only |
| `top_image()` | ✅ | Accurate flat surface (Image Method) |
| `top_vacuum(n)` | ✅ | Topography, cavities (recommended) |

```julia
# Compare different boundaries
outputs = OutputConfig(base_dir="outputs/boundary_compare", plot_gather=false, video_config=nothing)
for boundary in [top_absorbing(), top_image(), top_vacuum(10)]
    simulate(model, source, receivers, boundary, outputs, SimConfig(nt=2000); progress=false)
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
top_image()        # Image Method (flat surface)
top_absorbing()    # Absorbing at top
top_vacuum(n)      # n vacuum layers at top (recommended)

# Configuration
simconf = SimConfig(nt=3000, dt=nothing, cfl=0.4, fd_order=8)

# Video
video_config = VideoConfig(fields=[:vz], skip=50, fps=20)
```

### Main Functions

```julia
# Single-shot simulation (6 positional arguments)
gather = simulate(model, source, receivers, boundary, outputs, simconf)

# Batch simulation (efficient multi-shot)
sim = BatchSimulator(model, src_template, receivers, boundary, simconf)
gather = simulate_shot!(sim, src_x, src_z)                    # Single shot
gathers = simulate_shots!(sim, src_x_vec, src_z_vec)          # Multiple shots
simulate_shots!(sim, src_x_vec, src_z_vec; store=false) do gather, i
    # Process each shot with callback
end
```

### Output Files

When using `OutputConfig(base_dir="outputs/shot1", plot_gather=true, video_config=video_config)`:

| File | Description |
|------|-------------|
| `result.jld2` | Simulation result (gather, source, receivers) |
| `gather.png` | Seismic gather plot (if `plot_gather=true`) |
| `setup.png` | Model + sources + receivers (if `plot_setup=true`) |
| `wavefield_*.mp4` | Wavefield animation (if `video_config!=nothing`) |

## Performance

**GPU** (RTX 3060, 12GB):

| Grid Size | Time Steps | Runtime |
|-----------|------------|---------|
| 400×200 | 3000 | ~8 sec |
| 800×400 | 5000 | ~45 sec |
| 1200×600 | 8000 | ~3 min |

**CPU** (8-core, with `-t auto`): ~10-20x slower than GPU, but still practical for small-medium models.

## Why I Built This

As a geophysics student, I wanted a simple tool to run seismic forward modeling on my laptop without complex setup. Existing tools like SOFI2D and SPECFEM are powerful and well-established, but I needed something lightweight for quick experiments and learning.

I also found that HABC (Higdon Absorbing Boundary Conditions) offers good absorption quality with better computational efficiency than PML, which is helpful when running on consumer hardware.

So I built ElasticWave2D.jl — a simple, Julia-native tool for 2D elastic wave simulation. If you need a quick way to experiment with seismic modeling, this might be useful for you.

## Project Structure

```
ElasticWave2D.jl/
├── src/
│   ├── api/                # Single-shot user API (simulate)
│   ├── domain/             # User-facing basics (wavelet/source/receiver)
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
