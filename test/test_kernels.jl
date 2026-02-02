# ==============================================================================
# test/test_kernels.jl
#
# Tests for numerical kernels (velocity, stress, source, receiver)
# Includes CPU/GPU consistency tests
# ==============================================================================

using Test
using ElasticWave2D

@testset "Kernels" begin
    # Setup common test data
    nz, nx = 60, 80
    vp = fill(3000.0f0, nz, nx)
    vs = fill(1800.0f0, nz, nx)
    rho = fill(2200.0f0, nz, nx)
    
    model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)
    config = SimulationConfig(nbc=20, fd_order=8, nt=100, f0=15.0f0)
    
    @testset "Source Injection - CPU" begin
        be = ElasticWave2D.CPU_BACKEND
        M = ElasticWave2D.init_medium(model, config.nbc, config.fd_order, be)
        W = Wavefield(M.nx, M.nz, be)
        
        dt = 0.001f0
        wavelet = ElasticWave2D.ricker_wavelet(15.0f0, dt, 100)
        
        @testset "Explosion Source" begin
            src = Source(M.nx÷2, M.nz÷2, wavelet)
            
            # Inject at step 1
            ElasticWave2D.inject_source!(be, W, src, 1, dt)
            
            # txx and tzz should be modified at source location
            @test W.txx[src.i, src.j] == wavelet[1]
            @test W.tzz[src.i, src.j] == wavelet[1]
            
            # Other locations should be zero
            @test W.txx[src.i+5, src.j] == 0.0f0
        end
        
        @testset "Force Source" begin
            ElasticWave2D.reset!(be, W)
            
            buoyancy = 1.0f0 / 2200.0f0
            force_src = ElasticWave2D.ForceSource(M.nx÷2, M.nz÷2, wavelet, :vz, buoyancy)
            
            ElasticWave2D.inject_source!(be, W, force_src, 1, dt)
            
            # vz should be modified: vz += wavelet * dt * buoyancy
            expected = wavelet[1] * dt * buoyancy
            @test W.vz[force_src.i, force_src.j] ≈ expected
            
            # vx should be unchanged
            @test W.vx[force_src.i, force_src.j] == 0.0f0
        end
        
        @testset "Stress Source" begin
            ElasticWave2D.reset!(be, W)
            
            stress_src = StressSource(M.nx÷2, M.nz÷2, wavelet, :txx)
            
            ElasticWave2D.inject_source!(be, W, stress_src, 1, dt)
            
            @test W.txx[stress_src.i, stress_src.j] == wavelet[1]
            @test W.tzz[stress_src.i, stress_src.j] == 0.0f0  # Only txx
        end
    end
    
    @testset "Receiver Recording - CPU" begin
        be = ElasticWave2D.CPU_BACKEND
        M = ElasticWave2D.init_medium(model, config.nbc, config.fd_order, be)
        W = Wavefield(M.nx, M.nz, be)
        
        # Set some non-zero values
        W.vz[30, 25] = 1.5f0
        W.vz[40, 25] = 2.5f0
        W.txx[30, 25] = 100.0f0
        W.tzz[30, 25] = 200.0f0
        
        nt = 10
        rec_i = [30, 40]
        rec_j = [25, 25]
        
        @testset "vz Recording" begin
            rec = Receivers(rec_i, rec_j, zeros(Float32, nt, 2), :vz)
            
            ElasticWave2D.record_receivers!(be, W, rec, 1)
            
            @test rec.data[1, 1] == 1.5f0
            @test rec.data[1, 2] == 2.5f0
        end
        
        @testset "Pressure Recording" begin
            rec = Receivers(rec_i, rec_j, zeros(Float32, nt, 2), :p)
            
            ElasticWave2D.record_receivers!(be, W, rec, 1)
            
            # p = (txx + tzz) / 2
            @test rec.data[1, 1] == (100.0f0 + 200.0f0) / 2
        end
    end
    
    @testset "Wavefield Reset" begin
        be = ElasticWave2D.CPU_BACKEND
        W = Wavefield(50, 40, be)
        
        # Modify some values
        W.vx .= 1.0f0
        W.vz .= 2.0f0
        W.txx .= 3.0f0
        
        ElasticWave2D.reset!(be, W)
        
        @test all(W.vx .== 0)
        @test all(W.vz .== 0)
        @test all(W.txx .== 0)
        @test all(W.tzz .== 0)
        @test all(W.txz .== 0)
    end
    
    # GPU Consistency Tests
    if ElasticWave2D.is_cuda_available()
        using CUDA
        
        @testset "CPU/GPU Consistency" begin
            @testset "Source Injection Consistency" begin
                # Setup identical conditions on CPU and GPU
                be_cpu = ElasticWave2D.CPU_BACKEND
                be_gpu = ElasticWave2D.CUDA_BACKEND
                
                M_cpu = ElasticWave2D.init_medium(model, config.nbc, config.fd_order, be_cpu)
                M_gpu = ElasticWave2D.init_medium(model, config.nbc, config.fd_order, be_gpu)
                
                W_cpu = Wavefield(M_cpu.nx, M_cpu.nz, be_cpu)
                W_gpu = Wavefield(M_gpu.nx, M_gpu.nz, be_gpu)
                
                dt = 0.001f0
                wavelet = ElasticWave2D.ricker_wavelet(15.0f0, dt, 100)
                
                src_cpu = Source(M_cpu.nx÷2, M_cpu.nz÷2, wavelet)
                src_gpu = Source(M_gpu.nx÷2, M_gpu.nz÷2, CuArray(wavelet))
                
                ElasticWave2D.inject_source!(be_cpu, W_cpu, src_cpu, 1, dt)
                ElasticWave2D.inject_source!(be_gpu, W_gpu, src_gpu, 1, dt)
                
                @test Array(W_gpu.txx) ≈ W_cpu.txx
                @test Array(W_gpu.tzz) ≈ W_cpu.tzz
            end
            
            @testset "Receiver Recording Consistency" begin
                be_cpu = ElasticWave2D.CPU_BACKEND
                be_gpu = ElasticWave2D.CUDA_BACKEND
                
                nx, nz = 50, 40
                W_cpu = Wavefield(nx, nz, be_cpu)
                W_gpu = Wavefield(nx, nz, be_gpu)
                
                # Set identical values
                test_vals = rand(Float32, nx, nz)
                W_cpu.vz .= test_vals
                copyto!(W_gpu.vz, test_vals)
                
                nt = 10
                rec_i = [20, 30]
                rec_j = [15, 25]
                
                rec_cpu = Receivers(rec_i, rec_j, zeros(Float32, nt, 2), :vz)
                rec_gpu = Receivers(CuArray(rec_i), CuArray(rec_j), 
                                   CUDA.zeros(Float32, nt, 2), :vz)
                
                ElasticWave2D.record_receivers!(be_cpu, W_cpu, rec_cpu, 1)
                ElasticWave2D.record_receivers!(be_gpu, W_gpu, rec_gpu, 1)
                
                @test Array(rec_gpu.data) ≈ rec_cpu.data
            end
        end
    end
end
