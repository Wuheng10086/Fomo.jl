# Fomo.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9%20|%201.10%20|%201.11-blue)](https://julialang.org/)


https://github.com/user-attachments/assets/7218719c-b911-44ef-8ab5-153f242558f0


[中文文档](README_zh.md) | [English](README.md)

**Fomo** - **Fo**rward **Mo**deling: High-performance 2D isotropic elastic wave simulator in Julia.

## ✨ Features

- 🚀 **Backend-Dispatched Architecture** - Write once, run on CPU or GPU
- 📐 **High-Order Staggered-Grid FD** - 2nd to 8th order spatial accuracy
- 🛡️ **Hybrid Absorbing Boundary (HABC)** - Effective reflection suppression
- 🌊 **Free Surface Modeling** - Accurate Rayleigh wave simulation
- ⚡ **Multi-GPU Parallel** - Automatic load balancing across GPUs
- 📁 **Multiple Formats** - SEG-Y, Binary, MAT, NPY, HDF5, JLD2
- 🎬 **Video Recording** - Real-time wavefield visualization

## 📋 Requirements

- **Julia 1.9, 1.10, or 1.11** (1.12 not yet supported due to CairoMakie compatibility)
- CUDA-capable GPU (optional, for GPU acceleration)

## 🔧 Installation

### From GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/Fomo.jl")
```

### Local Development

```bash
git clone https://github.com/Wuheng10086/Fomo.jl.git
cd Fomo.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Optional Dependencies

For reading different file formats:

```julia
using Pkg
Pkg.add("SegyIO")  # SEG-Y files
Pkg.add("MAT")     # MATLAB files  
Pkg.add("NPZ")     # NumPy files
Pkg.add("HDF5")    # HDF5 files
```

## 🚀 Quick Start

```julia
using Fomo

# Create velocity model
vp = fill(3000.0f0, 200, 100)
vs = fill(1800.0f0, 200, 100)
rho = fill(2200.0f0, 200, 100)

# Add a layer
vp[:, 50:end] .= 4000.0f0
vs[:, 50:end] .= 2400.0f0

model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0; name="TwoLayer")

# Auto-select backend (GPU if available)
be = is_cuda_available() ? backend(:cuda) : backend(:cpu)

# Initialize simulation
nbc, fd_order = 50, 8
medium = init_medium(model, nbc, fd_order, be; free_surface=true)

# Time stepping
dt = 0.5f0 * 10.0f0 / maximum(vp)
nt = 2000
habc = init_habc(medium.nx, medium.nz, nbc, dt, 10.0f0, 10.0f0, 3500.0f0, be)
params = SimParams(dt, nt, 10.0f0, 10.0f0, fd_order)

# Acquisition geometry
rec_x = Float32.(0:20:1990)
rec_z = fill(10.0f0, length(rec_x))
rec = setup_receivers(rec_x, rec_z, medium; type=:vz)

src_x = Float32[1000.0]
src_z = Float32[20.0]
wavelet = ricker_wavelet(15.0f0, dt, nt)
shots = MultiShotConfig(src_x, src_z, wavelet)

# Run simulation
fd_coeffs = to_device(get_fd_coefficients(fd_order), be)
wavefield = Wavefield(medium.nx, medium.nz, be)
results = run_shots!(be, wavefield, medium, habc, fd_coeffs, rec, shots, params)

# Save results
save_gather(results[1], "gather.bin")
```

## 📁 Loading Models

```julia
using Fomo

# From JLD2 (recommended)
model = load_model("marmousi.jld2")

# From separate SEG-Y files (requires SegyIO)
using SegyIO
model = load_model_files(
    vp = "vp.segy",
    vs = "vs.segy", 
    rho = "rho.segy",
    dx = 12.5
)

# Save as JLD2 for faster loading
save_model("model.jld2", model)
```

## ⚡ Multi-GPU Execution

```julia
using Fomo

model = load_model("marmousi.jld2")

# Define geometry
src_x = Float32.(100:200:16900)
src_z = fill(10.0f0, length(src_x))
rec_x = Float32.(0:15:17000)
rec_z = fill(20.0f0, length(rec_x))

wavelet = ricker_wavelet(25.0f0, dt, nt)
params = SimParams(dt, nt, model.dx, model.dz, 8)

# Automatically uses all available GPUs!
results = run_shots_auto!(
    model, rec_x, rec_z, src_x, src_z, wavelet, params;
    nbc=50, fd_order=8, output_dir="outputs/"
)
```

## 🔍 Setup Verification

Before running a large simulation, verify your geometry:

```julia
using Fomo

model = load_model("model.jld2")

# Define sources and receivers
src_x = Float32.(100:200:3000)
src_z = fill(10.0f0, length(src_x))
rec_x = Float32.(0:15:3500)
rec_z = fill(50.0f0, length(rec_x))

# Generate setup check image
plot_setup(model, src_x, src_z, rec_x, rec_z; 
           output="setup_check.png",
           title="Survey Setup")
```

## 🎬 Video Recording

```julia
using Fomo

# Configure video recording
config = VideoConfig(
    fields = [:p],      # Record pressure field
    skip = 10,          # Save every 10 steps
    downsample = 2      # Spatial downsampling
)

recorder = MultiFieldRecorder(medium.nx, medium.nz, dt, config)

# Run with recording callback
results = run_shots!(be, wavefield, medium, habc, fd_coeffs,
                     rec, shots, params;
                     on_step = recorder)

# Generate MP4 video
generate_video(recorder.recorder, "wavefield.mp4"; fps=30)
```

## 🛠️ Command Line Tools

```bash
# Convert model formats
julia --project=. scripts/convert_model.jl \
    --vp=vp.segy --vs=vs.segy --rho=rho.segy \
    -o model.jld2 --dx=12.5 --transpose

# Check model dimensions
julia --project=. scripts/check_model.jl model.jld2 --fix

# Run parallel simulation
julia --project=. examples/run_parallel.jl model.jld2 outputs/
```

## 📂 Project Structure

```
Fomo.jl/
├── src/
│   ├── Fomo.jl              # Main module
│   ├── backends/            # CPU/CUDA abstraction
│   ├── kernels/             # FD kernels
│   ├── simulation/          # Shot management
│   ├── io/                  # Model/geometry I/O
│   └── visualization/       # Plotting & video
├── examples/                # Usage examples
├── scripts/                 # Command line tools
├── test/                    # Unit tests
└── docs/                    # Documentation
```

## 🧪 Running Tests

```bash
cd Fomo.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

## 📚 API Overview

### Core Types
- `VelocityModel` - Velocity model container
- `Medium` - Computational grid with material properties
- `Wavefield` - Wave field arrays (vx, vz, txx, tzz, txz)
- `SimParams` - Simulation parameters

### Main Functions
- `init_medium()` - Initialize computational medium
- `init_habc()` - Initialize absorbing boundaries
- `run_shots!()` - Run multiple shots sequentially
- `run_shots_auto!()` - Auto-parallel across GPUs
- `load_model()` / `save_model()` - Model I/O
- `plot_setup()` - Visualize acquisition geometry

## 📖 References

1. Luo, Y., & Schuster, G. (1990). *Parsimonious staggered grid finite-differencing of the wave equation*. Geophysical Research Letters.

2. Liu, Y., & Sen, M. K. (2012). *A hybrid absorbing boundary condition for elastic staggered-grid modelling*. Geophysical Prospecting.

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 👤 Author

zswh - 2025
