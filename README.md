# Fomo.jl

[![CI](https://github.com/Wuheng10086/Fomo.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Wuheng10086/Fomo.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://Wuheng10086.github.io/Fomo.jl/)

**High-performance 2D elastic wave forward modeling in Julia**

Fomo.jl is a Julia package for 2D elastic wave simulation using the staggered-grid finite-difference method. It supports both CPU and GPU (CUDA) backends with optimized kernels for maximum performance.

## ✨ Features

- **Backend-dispatched architecture** - Same code runs on CPU or GPU
- **High-order staggered-grid FD** - 2nd to 10th order accuracy
- **HABC boundary conditions** - Higdon Absorbing Boundary Conditions
- **Free surface modeling** - Accurate surface wave simulation
- **Multi-GPU parallel execution** - Automatic workload distribution
- **Multiple model formats** - JLD2, SEG-Y, binary, and more

## 🚀 Performance Optimizations

This version includes significant performance optimizations:

| Optimization | Speedup | Description |
|-------------|---------|-------------|
| Precomputed buoyancy (1/ρ) | 15-25% | Eliminates division in velocity update |
| Precomputed λ+2μ | 5-10% | Reduces stress update computation |
| Unrolled FD stencils | 10-15% | Better SIMD vectorization |
| Optimized GPU blocks (32×8) | 10-20% | Better memory coalescing |

**Expected total speedup: 40-60%**

## 📦 Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/Fomo.jl")
```

## 🎯 Quick Start

```julia
using Fomo

# Create a simple velocity model
nx, nz = 200, 100
vp = fill(3000.0f0, nz, nx)
vs = fill(1800.0f0, nz, nx)
rho = fill(2200.0f0, nz, nx)

# Add a layer
vp[nz÷2:end, :] .= 4000.0f0
vs[nz÷2:end, :] .= 2400.0f0

model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0; name="TwoLayer")

# Choose backend
be = is_cuda_available() ? backend(:cuda) : backend(:cpu)

# Initialize simulation
nbc, fd_order = 50, 8
medium = init_medium(model, nbc, fd_order, be; free_surface=true)

# ... setup sources, receivers, and run simulation
# See examples/ for complete examples
```

## 📁 Project Structure

```
Fomo.jl/
├── src/
│   ├── Fomo.jl              # Main module
│   ├── backends/            # CPU/CUDA dispatch
│   ├── core/                # Data structures
│   ├── kernels/             # FD kernels (optimized)
│   ├── io/                  # Model and data I/O
│   ├── simulation/          # Time stepping
│   ├── utils/               # Initialization (optimized)
│   └── visualization/       # Plotting utilities
├── examples/                # Usage examples
├── scripts/                 # Utility scripts
├── test/                    # Unit tests
└── docs/                    # Documentation
```

## 📖 Examples

- `examples/basic_example.jl` - Simple two-layer model
- `examples/run.jl` - Full simulation with model file
- `examples/run_parallel.jl` - Multi-GPU parallel execution

## 🔧 Requirements

- Julia 1.9+
- CUDA.jl (optional, for GPU acceleration)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This package implements the staggered-grid finite-difference method for elastic wave propagation with Higdon Absorbing Boundary Conditions.
