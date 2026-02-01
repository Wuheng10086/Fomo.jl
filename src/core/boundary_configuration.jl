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
    top_boundary::Symbol=:free_surface,
    bottom_boundary::Symbol=:absorbing,
    left_boundary::Symbol=:absorbing,
    right_boundary::Symbol=:absorbing,
    nbc::Int=50,
    top_padding::Int=0
)
    BoundaryConfig(top_boundary, bottom_boundary, left_boundary, right_boundary, nbc, top_padding)
end