# Fomo.jl

**High-performance 2D elastic wave simulation in Julia**

## Features

- Backend-dispatched architecture (CPU/CUDA)
- High-order staggered-grid finite-difference (2nd to 8th order)
- Hybrid Absorbing Boundary Condition (HABC)
- Free surface modeling
- Multi-GPU parallel execution
- Multiple model formats support

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/Fomo.jl")
```

## Quick Example

```julia
using Fomo

# Create model
vp = fill(3000.0f0, 200, 100)
vs = fill(1800.0f0, 200, 100)
rho = fill(2200.0f0, 200, 100)
model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)

# Initialize
be = backend(:cpu)
medium = init_medium(model, 50, 8, be)
# ... run simulation
```

See [Getting Started](@ref) for a complete tutorial.
