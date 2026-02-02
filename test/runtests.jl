# ==============================================================================
# ElasticWave2D.jl Test Suite
# ==============================================================================

using Test
using ElasticWave2D

# Include individual test files
@testset "ElasticWave2D.jl" begin
    include("test_types.jl")
    include("test_initialization.jl")
    include("test_kernels.jl")
    include("test_physics.jl")
end
