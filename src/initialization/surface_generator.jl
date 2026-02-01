# ==============================================================================
# simulation/surfaces.jl
#
# Helper functions for generating various surface topographies.
# ==============================================================================

"""
    flat_surface(nx, dx, depth) -> Vector{Float32}

Create a flat (horizontal) surface at constant depth.

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters
- `depth::Real`: Constant depth of surface in meters

# Returns
- `Vector{Float32}`: Surface elevation array of length `nx`

# Example
```julia
z_surface = flat_surface(400, 10.0, 50.0)  # flat surface at 50m depth
```
"""
flat_surface(nx::Int, dx::Real, depth::Real) = fill(Float32(depth), nx)

"""
    sinusoidal_surface(nx, dx; base_depth=50, amplitude=20, wavelength=1000) -> Vector{Float32}

Create a sinusoidal (wavy) surface.

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `base_depth::Real = 50.0`: Mean depth of surface in meters
- `amplitude::Real = 20.0`: Amplitude of sine wave in meters
- `wavelength::Real = 1000.0`: Wavelength of sine wave in meters

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Gentle undulation
z_surface = sinusoidal_surface(400, 10.0; amplitude=15, wavelength=2000)

# Sharp ripples
z_surface = sinusoidal_surface(400, 10.0; amplitude=30, wavelength=500)
```
"""
function sinusoidal_surface(nx::Int, dx::Real;
    base_depth::Real=50.0,
    amplitude::Real=20.0,
    wavelength::Real=1000.0)
    x = Float32.((0:nx-1) .* dx)
    return Float32.(base_depth .+ amplitude .* sin.(2π .* x ./ wavelength))
end

"""
    gaussian_valley(nx, dx; base_depth=50, valley_depth=30, center=nothing, width=200) -> Vector{Float32}

Create a surface with a Gaussian valley (depression/canyon).

The valley goes **deeper** into the model (larger z values).

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `base_depth::Real = 50.0`: Depth of flat regions in meters
- `valley_depth::Real = 30.0`: Additional depth at valley center in meters
- `center::Union{Real, Nothing} = nothing`: Valley center x position. If `nothing`, centered in model
- `width::Real = 200.0`: Gaussian standard deviation (controls valley width) in meters

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Centered valley
z_surface = gaussian_valley(400, 10.0; valley_depth=40, width=300)

# Off-center valley
z_surface = gaussian_valley(400, 10.0; valley_depth=40, center=1000.0, width=200)
```
"""
function gaussian_valley(nx::Int, dx::Real;
    base_depth::Real=50.0,
    valley_depth::Real=30.0,
    center::Union{Real,Nothing}=nothing,
    width::Real=200.0)
    x = Float32.((0:nx-1) .* dx)
    center = center === nothing ? nx * dx / 2 : Float32(center)
    return Float32.(base_depth .+ valley_depth .* exp.(-(x .- center) .^ 2 ./ (2 * width^2)))
end

"""
    gaussian_hill(nx, dx; base_depth=80, hill_height=30, center=nothing, width=200) -> Vector{Float32}

Create a surface with a Gaussian hill (elevation).

The hill rises **up** from the base (smaller z values at peak).

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `base_depth::Real = 80.0`: Depth of flat regions in meters
- `hill_height::Real = 30.0`: Height of hill above base in meters
- `center::Union{Real, Nothing} = nothing`: Hill center x position. If `nothing`, centered in model
- `width::Real = 200.0`: Gaussian standard deviation (controls hill width) in meters

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
z_surface = gaussian_hill(400, 10.0; base_depth=100, hill_height=50, width=400)
```
"""
function gaussian_hill(nx::Int, dx::Real;
    base_depth::Real=80.0,
    hill_height::Real=30.0,
    center::Union{Real,Nothing}=nothing,
    width::Real=200.0)
    x = Float32.((0:nx-1) .* dx)
    center = center === nothing ? nx * dx / 2 : Float32(center)
    return Float32.(base_depth .- hill_height .* exp.(-(x .- center) .^ 2 ./ (2 * width^2)))
end

"""
    tilted_surface(nx, dx; depth_left=30, depth_right=70) -> Vector{Float32}

Create a linearly tilted (sloped) surface.

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `depth_left::Real = 30.0`: Depth at left edge (x=0) in meters
- `depth_right::Real = 70.0`: Depth at right edge in meters

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Dipping surface (deeper on right)
z_surface = tilted_surface(400, 10.0; depth_left=20, depth_right=80)

# Reverse dip
z_surface = tilted_surface(400, 10.0; depth_left=80, depth_right=20)
```
"""
function tilted_surface(nx::Int, dx::Real;
    depth_left::Real=30.0,
    depth_right::Real=70.0)
    return Float32.(range(depth_left, depth_right, length=nx))
end

"""
    step_surface(nx, dx; depth_left=30, depth_right=70, step_position=nothing) -> Vector{Float32}

Create a surface with a sharp step (cliff/escarpment/fault scarp).

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `depth_left::Real = 30.0`: Depth on left side of step in meters
- `depth_right::Real = 70.0`: Depth on right side of step in meters
- `step_position::Union{Real, Nothing} = nothing`: X position of step. If `nothing`, at center

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Step down (cliff)
z_surface = step_surface(400, 10.0; depth_left=30, depth_right=60)

# Step up (escarpment)
z_surface = step_surface(400, 10.0; depth_left=60, depth_right=30)

# Off-center step
z_surface = step_surface(400, 10.0; depth_left=30, depth_right=60, step_position=1500.0)
```
"""
function step_surface(nx::Int, dx::Real;
    depth_left::Real=30.0,
    depth_right::Real=70.0,
    step_position::Union{Real,Nothing}=nothing)
    step_idx = step_position === nothing ? nx ÷ 2 : round(Int, step_position / dx)
    z = zeros(Float32, nx)
    z[1:step_idx] .= Float32(depth_left)
    z[step_idx+1:end] .= Float32(depth_right)
    return z
end

"""
    random_surface(nx, dx; base_depth=50, amplitude=10, smoothness=5) -> Vector{Float32}

Create a random rough surface with controllable smoothness.

Uses Gaussian random noise with moving average smoothing.

# Arguments
- `nx::Int`: Number of grid points in x direction
- `dx::Real`: Grid spacing in meters

# Keyword Arguments
- `base_depth::Real = 50.0`: Mean depth of surface in meters
- `amplitude::Real = 10.0`: Standard deviation of random roughness in meters
- `smoothness::Int = 5`: Smoothing window half-width in grid points. Larger = smoother

# Returns
- `Vector{Float32}`: Surface elevation array

# Example
```julia
# Rough surface
z_surface = random_surface(400, 10.0; amplitude=20, smoothness=3)

# Gentle rolling terrain
z_surface = random_surface(400, 10.0; amplitude=15, smoothness=15)
```

# Note
Results are random and will differ each time. Use `Random.seed!()` for reproducibility.
"""
function random_surface(nx::Int, dx::Real;
    base_depth::Real=50.0,
    amplitude::Real=10.0,
    smoothness::Int=5)
    noise = randn(Float32, nx) .* Float32(amplitude)
    z = zeros(Float32, nx)
    for i in 1:nx
        i_start = max(1, i - smoothness)
        i_end = min(nx, i + smoothness)
        z[i] = sum(noise[i_start:i_end]) / (i_end - i_start + 1)
    end
    return Float32.(base_depth .+ z)
end

"""
    combine_surfaces(surfaces...; method=:add) -> Vector{Float32}

Combine multiple surface shapes into one.

# Arguments
- `surfaces...`: Two or more surface arrays (all must have same length)

# Keyword Arguments
- `method::Symbol = :add`: How to combine surfaces
  - `:add` - Add all surfaces element-wise (useful for superimposing perturbations)
  - `:min` - Take minimum depth at each point (highest elevation)
  - `:max` - Take maximum depth at each point (lowest elevation)

# Returns
- `Vector{Float32}`: Combined surface elevation array

# Example
```julia
# Sinusoidal base + valley
z_surface = combine_surfaces(
    sinusoidal_surface(400, 10.0; base_depth=50, amplitude=15),
    gaussian_valley(400, 10.0; base_depth=0, valley_depth=25)  # Note: base_depth=0 for perturbation
)

# Complex terrain
z_surface = combine_surfaces(
    flat_surface(400, 10.0, 60.0),
    sinusoidal_surface(400, 10.0; base_depth=0, amplitude=10, wavelength=500),
    gaussian_valley(400, 10.0; base_depth=0, valley_depth=20, center=1500.0)
)
```

# Tip
When combining, set `base_depth=0` for perturbation shapes so they add properly to the base.
"""
function combine_surfaces(surfaces...; method::Symbol=:add)
    nx = length(surfaces[1])
    if method == :add
        return Float32.(sum(surfaces))
    elseif method == :min
        result = copy(surfaces[1])
        for s in surfaces[2:end]
            result .= min.(result, s)
        end
        return Float32.(result)
    elseif method == :max
        result = copy(surfaces[1])
        for s in surfaces[2:end]
            result .= max.(result, s)
        end
        return Float32.(result)
    end
end
