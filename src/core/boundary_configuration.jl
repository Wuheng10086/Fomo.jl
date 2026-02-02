# ==============================================================================
# types/boundary_config.jl
#
# Boundary configuration for ElasticWave2D.jl
# ==============================================================================

"""
    BoundaryConfig

Configuration for boundary conditions on all sides of the model.

# Fields
- `top_boundary::Symbol`: Top boundary condition (:absorbing, :image, :vacuum)
- `bottom_boundary::Symbol`: Bottom boundary condition (:absorbing, :image)
- `left_boundary::Symbol`: Left boundary condition (:absorbing, :periodic)
- `right_boundary::Symbol`: Right boundary condition (:absorbing, :periodic)
- `nbc::Int`: Number of absorbing boundary layers
- `top_padding::Int`: Additional padding layers at top (for sources near boundary)

# Supported Boundary Types
- `:image` - Image method free surface (stress-free, generates surface waves)
- `:absorbing` - HABC absorbing boundary (no reflections)
- `:vacuum` - Vacuum formulation (handled through material parameters)
- `:periodic` - Periodic boundary (for left/right only)

# Example
```julia
# Default: free surface at top, absorbing elsewhere
config = BoundaryConfig()

# All absorbing (no surface waves)
config = BoundaryConfig(top_boundary=:absorbing)

# Vacuum formulation
config = BoundaryConfig(top_boundary=:vacuum)
```
"""
struct BoundaryConfig
    top_boundary::Symbol    # :absorbing, :image, :vacuum
    bottom_boundary::Symbol # :absorbing, :image
    left_boundary::Symbol   # :absorbing, :periodic
    right_boundary::Symbol  # :absorbing, :periodic
    nbc::Int               # 吸收边界层数
    top_padding::Int       # 顶部额外填充层（用于震源靠近边界的情况）
end

# Default constructor
function BoundaryConfig(;
    top_boundary::Symbol=:image,  # FIXED: 与 SimulationConfig 保持一致
    bottom_boundary::Symbol=:absorbing,
    left_boundary::Symbol=:absorbing,
    right_boundary::Symbol=:absorbing,
    nbc::Int=50,
    top_padding::Int=0
)
    # Validate boundary types
    valid_top = (:absorbing, :image, :vacuum)
    valid_sides = (:absorbing, :periodic)
    valid_bottom = (:absorbing, :image)

    top_boundary in valid_top || error("Invalid top_boundary: $top_boundary. Must be one of $valid_top")
    bottom_boundary in valid_bottom || error("Invalid bottom_boundary: $bottom_boundary. Must be one of $valid_bottom")
    left_boundary in valid_sides || error("Invalid left_boundary: $left_boundary. Must be one of $valid_sides")
    right_boundary in valid_sides || error("Invalid right_boundary: $right_boundary. Must be one of $valid_sides")
    nbc > 0 || error("nbc must be positive, got $nbc")
    top_padding >= 0 || error("top_padding must be non-negative, got $top_padding")

    BoundaryConfig(top_boundary, bottom_boundary, left_boundary, right_boundary, nbc, top_padding)
end

"""
    is_free_surface(config::BoundaryConfig) -> Bool

Check if the top boundary is a free surface (image or vacuum method).
"""
is_free_surface(config::BoundaryConfig) = config.top_boundary in (:image, :vacuum)

"""
    needs_vacuum_layers(config::BoundaryConfig) -> Bool

Check if the configuration requires vacuum layers at the top.
"""
needs_vacuum_layers(config::BoundaryConfig) = config.top_boundary == :vacuum