# ==============================================================================
# ForceSource type definition
# ==============================================================================

"""
    ForceSource{T<:AbstractVector{Float32}}

Body force source that injects into velocity fields (vx or vz).

Physics: In the velocity-stress formulation, a body force f appears in the
momentum equation as:
    ρ ∂v/∂t = ∇·σ + f
Discretized:
    v[i,j] += dt * f(t) / ρ[i,j]  =  dt * f(t) * buoyancy[i,j]

# Fields
- `i::Int`: Grid index in x-direction (including padding)
- `j::Int`: Grid index in z-direction (including padding)
- `wavelet::T`: Source time function (e.g., Ricker wavelet)
- `component::Symbol`: Velocity component to inject into, `:vx` or `:vz`
- `buoyancy_at_src::Float32`: 1/ρ at the source location (for correct scaling)

# Example
```julia
src = ForceSource(src_i, src_j, wavelet, :vz, 1.0f0 / rho[src_j, src_i])
```
"""
struct ForceSource{T<:AbstractVector{Float32}}
    i::Int
    j::Int
    wavelet::T
    component::Symbol      # :vx or :vz
    buoyancy_at_src::Float32  # 1/ρ at source point
end
