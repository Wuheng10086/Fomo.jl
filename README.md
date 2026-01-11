# Fomo.jl

**Fo**rward **Mo**deling - High-Performance 2D Elastic Wave Simulation


https://github.com/user-attachments/assets/4cabc046-8a61-4dcc-9140-703bc50a7364


## Features

- 2D elastic wave propagation (P-SV)
- Staggered-grid finite difference method
- GPU acceleration (CUDA)
- Irregular free surface (IBM method)
- HABC absorbing boundary conditions
- Shot gather recording
- Wavefield video generation

## Project Structure

```
src/
├── Fomo.jl                   # Main module
├── backends/
│   └── backend.jl            # CPU/CUDA backend abstraction
├── types/
│   ├── structures.jl         # Core types (Wavefield, Medium, Source, etc.)
│   └── model.jl              # VelocityModel
├── kernels/
│   ├── velocity.jl           # Velocity update kernel
│   ├── stress.jl             # Stress update kernel
│   ├── boundary.jl           # HABC, free surface
│   ├── source_receiver.jl    # Source injection, receiver recording
│   └── ibm.jl                # Immersed Boundary Method
├── surface/
│   └── irregular.jl          # Irregular surface initialization
├── simulation/
│   ├── init.jl               # Medium/wavefield initialization
│   ├── time_stepper.jl       # Time stepping (regular surface)
│   ├── time_stepper_ibm.jl   # Time stepping (irregular surface)
│   ├── shots.jl              # Shot management
│   └── parallel.jl           # Multi-GPU parallel execution
├── io/
│   ├── model_io.jl           # Model load/save
│   ├── gather_io.jl          # Gather save/load
│   └── geometry_io.jl        # Survey geometry
└── visualization/
    ├── video.jl              # Wavefield video recording
    └── plots.jl              # Static plots
```

## Quick Start

```julia
using Fomo

# Create velocity model
vp = fill(3000.0f0, 200, 400)  # [nz, nx]
vs = fill(1800.0f0, 200, 400)
rho = fill(2200.0f0, 200, 400)
model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)

# Select backend
be = is_cuda_available() ? backend(:cuda) : backend(:cpu)

# Initialize simulation
medium = init_medium(model, 50, 8, be; free_surface=true)
wavefield = Wavefield(medium.nx, medium.nz, be)

# Setup source and receivers
wavelet = ricker_wavelet(15.0f0, dt, nt)
src = Source(src_i, src_j, to_device(wavelet, be))
rec = setup_receivers(rec_x, rec_z, nt, medium, be)

# Run simulation
run_shot!(be, wavefield, medium, habc, fd_coeffs, rec, src, params)
```

## Irregular Surface Example

```julia
# Create irregular surface
z_surface = 50.0f0 .+ 20.0f0 .* sin.(2π .* x ./ 1000.0f0)

# Initialize with IBM method
surface = init_irregular_surface(z_surface, medium; method=:direct_zero)
surface_gpu = to_gpu(surface)

# Setup source/receivers relative to surface
src = setup_irregular_source(src_x, src_depth, wavelet, surface, medium; backend=be)
rec = setup_surface_receivers(rec_x, surface, medium, nt; backend=be)

# Run with irregular surface
time_step_irregular!(be, wavefield, medium, habc, fd_coeffs, src, rec, k, params, surface_gpu)
```

## IBM Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `:direct_zero` | Direct stress zeroing | Complex terrain, stable |
| `:mirror` | Mirror interpolation | Gentle terrain, higher accuracy |

## Requirements

- Julia 1.9+
- CUDA.jl (for GPU support)
- CairoMakie.jl (for visualization)

## License

MIT
