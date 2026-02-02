# ==============================================================================
# test/test_types.jl
#
# Tests for type hierarchy and basic structure functionality
# ==============================================================================

using Test
using ElasticWave2D

@testset "Type Hierarchy" begin
    @testset "Source Types" begin
        # All source types should be subtypes of AbstractSource
        @test Source <: ElasticWave2D.AbstractSource
        @test StressSource <: ElasticWave2D.AbstractSource
        @test ElasticWave2D.ForceSource <: ElasticWave2D.AbstractSource
        
        # Create instances and verify
        wavelet = zeros(Float32, 100)
        
        src = Source(10, 10, wavelet)
        @test src isa ElasticWave2D.AbstractSource
        @test src.i == 10
        @test src.j == 10
        
        stress_src = StressSource(10, 10, wavelet, :txx)
        @test stress_src isa ElasticWave2D.AbstractSource
        @test stress_src.component == :txx
        
        force_src = ElasticWave2D.ForceSource(10, 10, wavelet, :vz, 1.0f0)
        @test force_src isa ElasticWave2D.AbstractSource
        @test force_src.component == :vz
        @test force_src.buoyancy_at_src == 1.0f0
    end
    
    @testset "Backend Types" begin
        @test CPUBackend <: ElasticWave2D.AbstractBackend
        @test CUDABackend <: ElasticWave2D.AbstractBackend
        
        # Singleton instances
        @test ElasticWave2D.CPU_BACKEND isa CPUBackend
        @test ElasticWave2D.CUDA_BACKEND isa CUDABackend
        
        # Backend selection
        @test ElasticWave2D.backend(:cpu) isa CPUBackend
        if ElasticWave2D.is_cuda_available()
            @test ElasticWave2D.backend(:cuda) isa CUDABackend
        end
    end
    
    @testset "VelocityModel" begin
        nz, nx = 100, 200
        vp = fill(3000.0f0, nz, nx)
        vs = fill(1800.0f0, nz, nx)
        rho = fill(2200.0f0, nz, nx)
        
        model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0; name="test_model")
        
        @test model.nx == nx
        @test model.nz == nz
        @test model.dx == 10.0f0
        @test model.dz == 10.0f0
        @test model.name == "test_model"
        
        # Check that arrays are converted to Float32
        @test eltype(model.vp) == Float32
        @test eltype(model.vs) == Float32
        @test eltype(model.rho) == Float32
    end
    
    @testset "SimulationConfig" begin
        # Default config
        config = SimulationConfig()
        @test config.fd_order == 8
        @test config.nbc == 50
        @test config.free_surface == true
        @test config.source_type == :explosion
        
        # Custom config
        config = SimulationConfig(
            nt=5000,
            f0=20.0f0,
            source_type=:force_z,
            free_surface=false
        )
        @test config.nt == 5000
        @test config.f0 == 20.0f0
        @test config.source_type == :force_z
        @test config.free_surface == false
        
        # Invalid source type should error
        @test_throws ErrorException SimulationConfig(source_type=:invalid)
    end
    
    @testset "BoundaryConfig" begin
        # Default config
        bc = BoundaryConfig()
        @test bc.top_boundary == :image
        @test bc.bottom_boundary == :absorbing
        @test bc.nbc == 50
        
        # Custom config
        bc = BoundaryConfig(top_boundary=:vacuum, nbc=30)
        @test bc.top_boundary == :vacuum
        @test bc.nbc == 30
        
        # Invalid boundary type should error
        @test_throws ErrorException BoundaryConfig(top_boundary=:invalid)
        @test_throws ErrorException BoundaryConfig(nbc=-1)
    end
    
    @testset "SimParams" begin
        params = SimParams(0.001f0, 1000, 10.0f0, 10.0f0, 8)
        @test params.dt == 0.001f0
        @test params.nt == 1000
        @test params.dtx == 0.001f0 / 10.0f0
        @test params.dtz == 0.001f0 / 10.0f0
        @test params.fd_order == 8
        @test params.M == 4  # fd_order ÷ 2
    end
end

@testset "Wavefield" begin
    nx, nz = 50, 40
    
    @testset "CPU Wavefield" begin
        W = Wavefield(nx, nz, ElasticWave2D.CPU_BACKEND)
        
        @test size(W.vx) == (nx, nz)
        @test size(W.vz) == (nx, nz)
        @test size(W.txx) == (nx, nz)
        @test size(W.tzz) == (nx, nz)
        @test size(W.txz) == (nx, nz)
        
        # Should be initialized to zero
        @test all(W.vx .== 0)
        @test all(W.vz .== 0)
        
        # Test reset
        W.vx[10, 10] = 1.0f0
        ElasticWave2D.reset!(ElasticWave2D.CPU_BACKEND, W)
        @test all(W.vx .== 0)
    end
    
    if ElasticWave2D.is_cuda_available()
        @testset "GPU Wavefield" begin
            W = Wavefield(nx, nz, ElasticWave2D.CUDA_BACKEND)
            
            @test size(W.vx) == (nx, nz)
            
            # Test reset
            ElasticWave2D.reset!(ElasticWave2D.CUDA_BACKEND, W)
            @test all(Array(W.vx) .== 0)
        end
    end
end
