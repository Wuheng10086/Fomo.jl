# ElasticWave2D.jl

[🇨🇳 中文](README_zh.md) | **English**

<p align="center">
  <b>GPU-accelerated 2D elastic wave simulation in Julia</b><br>
  <i>Run seismic forward modeling on your laptop — no cluster, no complex setup</i>
</p>

<p align="center">
  <img src="docs/images/wavefield.gif" width="600" alt="Wavefield Animation">
</p>

## Why This Project?

Traditional seismic simulation codes are hard to install, poorly documented, and require HPC clusters. **ElasticWave2D.jl** is different:

- ✅ **One-line install** — Pure Julia, no Fortran/C compilation
- ✅ **Runs on gaming GPUs** — GTX 1060, RTX 3060, etc.
- ✅ **CPU optimized too** — Multi-threaded with `julia -t auto`
- ✅ **Student-friendly** — Clear examples, readable code
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
| **Video Recording** | Wavefield snapshots → MP4 |
| **Multiple Formats** | SEG-Y, Binary, HDF5, NPY, MAT, JLD2 |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/ElasticWave2D.jl")
```

**Requirements**: Julia 1.9+. GPU optional (auto-detects CUDA).

## Quick Start

```julia
using ElasticWave2D

# Create a simple two-layer model
nx, nz = 200, 100
dx, dz = 10.0f0, 10.0f0

vp = fill(2000.0f0, nz, nx)
vs = fill(1200.0f0, nz, nx)
rho = fill(2000.0f0, nz, nx)
vp[50:end, :] .= 3500.0f0  # Faster layer below

model = VelocityModel(vp, vs, rho, dx, dz)

# Survey geometry
src_x, src_z = 1000.0f0, 20.0f0
rec_x = Float32.(collect(100:10:1900))
rec_z = fill(10.0f0, length(rec_x))

# Run simulation with vacuum free surface
result = seismic_survey(
    model,
    (src_x, src_z),
    (rec_x, rec_z);
    surface_method = :image,    # :vacuum, :image, or :absorbing
    vacuum_layers = 5,
    config = SimulationConfig(nt=1000, f0=20.0f0)
)
```

## Examples

### 🎬 Elastic Wave Demo
High-resolution wave propagation in a two-layer medium with video output.

```bash
julia -t auto examples/elastic_wave_demo.jl
```

<p align="center">
  <img src="docs/images/elastic_wave_setup.png" width="400" alt="Elastic Wave Setup">
  <img src="docs/images/elastic_wave_gather.png" width="400" alt="Elastic Wave Gather">
</p>

---

### 🏗️ Tunnel Detection (Engineering)
Detect underground cavities using seismic diffraction. Uses vacuum formulation for both free surface and tunnel cavity.

```bash
julia -t auto examples/tunnel_detection_demo.jl
```

<p align="center">
  <img src="docs/images/tunnel_setup.png" width="400" alt="Tunnel Setup">
  <img src="docs/images/tunnel_gather.png" width="400" alt="Tunnel Gather">
</p>

**What to look for**: Diffracted waves from tunnel edges, shadow zone behind tunnel.

---

### 🛢️ Exploration Seismic (Petroleum)
Image an anticlinal structure — a classic hydrocarbon trap.

```bash
julia -t auto examples/exploration_seismic_demo.jl
```

<p align="center">
  <img src="docs/images/exploration_setup.png" width="400" alt="Exploration Setup">
  <img src="docs/images/exploration_gather.png" width="400" alt="Exploration Gather">
</p>

**What to look for**: Reflection "pull-up" at anticline crest, multiple layer reflections.

---

### 🔬 Boundary Comparison Demo
Compare different surface handling methods side-by-side.

```bash
julia -t auto examples/seismic_survey_demo.jl
```

| Method | Surface Waves | Use Case |
|--------|--------------|----------|
| `:absorbing` | ❌ | Body waves only |
| `:image` | ✅ | Accurate flat surface BC (Image Method) |
| `:vacuum` | ✅ | Unified approach (recommended) |

**Surface wave comparison** — Both methods produce Rayleigh waves, with nearly identical results:

<p align="center">
  <img src="docs/images/freesurface_gather.png" width="400" alt="Explicit Free Surface">
  <img src="docs/images/vacuum_gather.png" width="400" alt="Vacuum Formulation">
</p>
<p align="center">
  <i>Left: Image Method BC | Right: Vacuum formulation</i>
</p>

The vacuum method offers more flexibility (topography, internal voids) with comparable accuracy.

## API Reference

### `seismic_survey` — High-level Interface

```julia
seismic_survey(model, source, receivers;
    surface_method = :vacuum,     # :vacuum, :image, :absorbing
    vacuum_layers = 10,           # Number of vacuum layers (for :vacuum)
    config = SimulationConfig(),
    video_config = nothing
)
```

### `simulate!` — Low-level Interface

```julia
result = simulate!(model, src_x, src_z, rec_x, rec_z;
    config = SimulationConfig(
        nt = 3000,           # Time steps
        f0 = 15.0f0,         # Source frequency (Hz)
        fd_order = 8,        # FD accuracy order
        free_surface = true, # Use Image Method BC
        output_dir = "outputs"
    ),
    video_config = VideoConfig(
        fields = [:vz],      # Record vertical velocity
        skip = 20,           # Frame interval
        fps = 30
    )
)
```

### Surface Method Comparison

| Parameter | `surface_method=:image` | `surface_method=:vacuum` |
|-----------|-------------------------|--------------------------|
| Implementation | Image Method BC | ρ=0 layers at top |
| Topography | ❌ Flat only | ✅ Any shape |
| Internal voids | ❌ | ✅ Tunnels, caves |
| Consistency | Accurate surface waves | Same physics everywhere |

## Performance

**GPU** (RTX 3060, 12GB):

| Grid Size | Time Steps | Runtime |
|-----------|------------|---------|
| 400×200 | 3000 | ~8 sec |
| 800×400 | 5000 | ~45 sec |
| 1200×600 | 8000 | ~3 min |

**CPU** (8-core, with `-t auto`): ~10-20x slower than GPU, but still practical for small-medium models.

## Why I Built This

As a geophysics student, I struggled with existing seismic simulation tools like SOFI2D and SPECFEM — they require Linux, complex `make` configurations, and endless dependency issues. I just wanted to run a simple forward model on my laptop without spending days on setup.

I also found that PML (Perfectly Matched Layer) boundaries are computationally expensive. HABC (Higdon Absorbing Boundary Conditions) offers similar absorption quality with much better efficiency, which matters when you're running on a gaming GPU instead of a cluster.

So I built ElasticWave2D.jl — a tool I wish I had when I started. If you're a student with a laptop and curiosity, this is for you.

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

## Project Structure

The project follows a domain-driven structure to ensure clarity and maintainability:

```
ElasticWave2D.jl/
├── src/
│   ├── compute/            # Hardware abstraction (CPU/GPU)
│   ├── core/               # Fundamental types (Wavefield, Medium, Configs)
│   ├── physics/            # Numerical kernels (Velocity, Stress, Boundaries)
│   ├── initialization/     # Setup routines (Media, Topography)
│   ├── solver/             # Time-stepping and orchestration logic
│   ├── workflow/           # High-level User APIs
│   ├── io/                 # Input/Output (Models, Seismic Data)
│   └── visualization/      # Plotting and Video generation
├── examples/               # Usage examples
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License
