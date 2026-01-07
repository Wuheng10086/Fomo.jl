# Fomo.jl

[![Build Status](https://github.com/YOUR_USERNAME/Fomo.jl/workflows/CI/badge.svg)](https://github.com/YOUR_USERNAME/Fomo.jl/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


https://github.com/user-attachments/assets/7218719c-b911-44ef-8ab5-153f242558f0


[中文文档](README_zh.md) | [English](README.md)

**Fomo** - **Fo**rward **Mo**deling: High-performance 2D isotropic elastic wave numerical simulator in Julia.

## Features

- 🚀 **Backend-Dispatched Architecture**: Write once, run on CPU or GPU
- 📐 **High-Order Staggered-Grid (SGFD)**: Up to 8th order spatial accuracy
- 🛡️ **Hybrid Absorbing Boundary (HABC)**: Effective reflection suppression
- 🌊 **Free Surface Modeling**: Accurate Rayleigh wave simulation
- ⚡ **Multi-GPU Parallel Execution**: Automatic load balancing across GPUs
- 📁 **Multiple Model Formats**: SEG-Y, Binary, MAT, NPY, HDF5, JLD2

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/YOUR_USERNAME/Fomo.jl")
```

### Optional Dependencies

```julia
using Pkg

# For visualization
Pkg.add("CairoMakie")

# For file formats
Pkg.add("SegyIO")   # SEG-Y files
Pkg.add("MAT")      # MATLAB files
Pkg.add("NPZ")      # NumPy files
Pkg.add("HDF5")     # HDF5 files
```

## Quick Start

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

# Select backend (auto-detects CUDA)
be = is_cuda_available() ? backend(:cuda) : backend(:cpu)

# Initialize
nbc, fd_order = 50, 8
medium = init_medium(model, nbc, fd_order, be; free_surface=true)

# Time stepping
dt = 0.5f0 * 10.0f0 / maximum(vp)
nt = 2000

habc = init_habc(medium.nx, medium.nz, nbc, dt, 10.0f0, 10.0f0, 3500.0f0, be)
params = SimParams(dt, nt, 10.0f0, 10.0f0, fd_order)

# Setup acquisition
rec_x = Float32.(0:20:1990)
rec_z = fill(10.0f0, length(rec_x))
rec = setup_receivers(rec_x, rec_z, medium; type=:vz)

src_x = Float32[1000.0]
src_z = Float32[20.0]
wavelet = ricker_wavelet(15.0f0, dt, nt)
shots = MultiShotConfig(src_x, src_z, wavelet)

# Run
fd_coeffs = to_device(get_fd_coefficients(fd_order), be)
wavefield = Wavefield(medium.nx, medium.nz, be)
results = run_shots!(be, wavefield, medium, habc, fd_coeffs, rec, shots, params)

# Save
save_gather(results[1], "gather.bin")
```

## Visualization (requires CairoMakie)

```julia
using CairoMakie  # Load first!
using Fomo

model = load_model("model.jld2")
plot_setup(model, src_x, src_z, rec_x, rec_z; output="setup.png")
```

## Multi-GPU Execution

```julia
using Fomo

results = run_shots_auto!(
    model, rec_x, rec_z, src_x, src_z, wavelet, params;
    nbc=50, fd_order=8, output_dir="outputs/"
)
```

## Loading Models

```julia
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

# Convert and save
save_model("model.jld2", model)
```

## Documentation

See [examples/](examples/) for detailed examples.

## License

MIT License - see [LICENSE](LICENSE)

## Author

zswh - 2025
