# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Wuheng10086/Fomo.jl")
```

## Basic Usage

### 1. Create a Velocity Model

```julia
using Fomo

# Create arrays
nx, nz = 200, 100
vp = fill(3000.0f0, nx, nz)
vs = fill(1800.0f0, nx, nz)
rho = fill(2200.0f0, nx, nz)

# Create model struct
model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0; name="MyModel")
```

### 2. Choose Backend

```julia
# CPU (always available)
be = backend(:cpu)

# GPU (if CUDA available)
if is_cuda_available()
    be = backend(:cuda)
end
```

### 3. Initialize Simulation

```julia
nbc = 50        # Absorbing boundary cells
fd_order = 8    # Finite difference order

medium = init_medium(model, nbc, fd_order, be; free_surface=true)
```

### 4. Run Simulation

See [examples/basic_example.jl](https://github.com/Wuheng10086/Fomo.jl/blob/main/examples/basic_example.jl) for a complete example.
