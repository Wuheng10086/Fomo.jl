# ==============================================================================
# test/test_initialization.jl
#
# Tests for initialization functions (medium, HABC, wavelet, etc.)
# ==============================================================================

using Test
using ElasticWave2D

@testset "Initialization" begin
    @testset "FD Coefficients" begin
        # Test supported orders
        for order in [2, 4, 6, 8, 10]
            coeffs = ElasticWave2D.get_fd_coefficients(order)
            @test length(coeffs) == order ÷ 2
            @test eltype(coeffs) == Float32
        end
        
        # Unsupported order should error
        @test_throws ErrorException ElasticWave2D.get_fd_coefficients(3)
        @test_throws ErrorException ElasticWave2D.get_fd_coefficients(12)
    end
    
    @testset "Ricker Wavelet" begin
        f0 = 15.0f0
        dt = 0.001f0
        nt = 500
        
        wavelet = ElasticWave2D.ricker_wavelet(f0, dt, nt)
        
        @test length(wavelet) == nt
        @test eltype(wavelet) == Float32
        
        # Ricker wavelet should have maximum amplitude around t0 = 1/f0
        t0_idx = round(Int, 1.0 / f0 / dt)
        max_idx = argmax(abs.(wavelet))
        @test abs(max_idx - t0_idx) <= 5  # Allow some tolerance
        
        # Should decay towards zero at edges
        @test abs(wavelet[1]) < abs(wavelet[t0_idx])
        @test abs(wavelet[end]) < 0.1f0
    end
    
    @testset "Medium Initialization" begin
        nz, nx = 100, 150
        vp = fill(3000.0f0, nz, nx)
        vs = fill(1800.0f0, nz, nx)
        rho = fill(2200.0f0, nz, nx)
        
        model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)
        
        nbc = 20
        fd_order = 8
        
        M = ElasticWave2D.init_medium(model, nbc, fd_order, ElasticWave2D.CPU_BACKEND; 
                                       free_surface=true)
        
        # Check dimensions (should be padded)
        pad = nbc + fd_order ÷ 2
        @test M.nx == nx + 2 * pad
        @test M.nz == nz + 2 * pad
        @test M.pad == pad
        @test M.M == fd_order ÷ 2
        @test M.is_free_surface == true
        
        # Check precomputed fields exist and have correct size
        @test size(M.lam) == (M.nx, M.nz)
        @test size(M.buoy_vx) == (M.nx, M.nz)
        @test size(M.buoy_vz) == (M.nx, M.nz)
        @test size(M.lam_2mu) == (M.nx, M.nz)
        
        # Buoyancy should be positive (1/rho)
        @test all(M.buoy_vx .>= 0)
        @test all(M.buoy_vz .>= 0)
    end
    
    @testset "HABC Initialization" begin
        nx, nz = 200, 150
        nbc = 30
        pad = 34
        dt = 0.001f0
        dx, dz = 10.0f0, 10.0f0
        vp_max = 4000.0f0
        
        H = ElasticWave2D.init_habc(nx, nz, nbc, pad, dt, dx, dz, vp_max, 
                                     ElasticWave2D.CPU_BACKEND)
        
        @test H.nbc == nbc
        
        # HABC coefficients should be real numbers
        @test isfinite(H.qx)
        @test isfinite(H.qz)
        @test isfinite(H.qt_x)
        @test isfinite(H.qt_z)
        @test isfinite(H.qxt)
        
        # Weight arrays should have correct size
        @test size(H.w_vx) == (nz, nx)
        @test size(H.w_vz) == (nz, nx)
        @test size(H.w_tau) == (nz, nx)
        
        # Weights should be in [0, 1]
        @test all(0 .<= H.w_vx .<= 1)
        @test all(0 .<= H.w_vz .<= 1)
        @test all(0 .<= H.w_tau .<= 1)
    end
    
    @testset "Device Transfer" begin
        x = rand(Float32, 10, 10)
        
        # CPU transfer should return Array
        x_cpu = ElasticWave2D.to_device(x, ElasticWave2D.CPU_BACKEND)
        @test x_cpu isa Array
        @test x_cpu == x
        
        # Scalar transfer
        @test ElasticWave2D.to_device(1.0f0, ElasticWave2D.CPU_BACKEND) == 1.0f0
        
        if ElasticWave2D.is_cuda_available()
            using CUDA
            x_gpu = ElasticWave2D.to_device(x, ElasticWave2D.CUDA_BACKEND)
            @test x_gpu isa CuArray
            @test Array(x_gpu) == x
        end
    end
    
    @testset "Grid Spacing Suggestion" begin
        vp_min = 1500.0f0  # Water velocity
        freq_max = 30.0f0
        
        dx = ElasticWave2D.suggest_grid_spacing(vp_min, freq_max; ppw=10)
        
        # dx should be wavelength / ppw
        wavelength = vp_min / freq_max
        @test dx ≈ wavelength / 10
    end
end
