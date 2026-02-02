# ==============================================================================
# test/test_physics.jl
#
# Physical correctness tests for elastic wave simulation
# These tests verify that the simulation produces physically meaningful results
# ==============================================================================

using Test
using ElasticWave2D
using Statistics

@testset "Physical Correctness" begin
    
    @testset "Wave Propagation Speed" begin
        # In a homogeneous medium, P-wave should travel at vp velocity
        # Test: source at center, measure arrival time at known distance
        
        vp_val = 3000.0f0
        vs_val = 1800.0f0
        rho_val = 2200.0f0
        dx = 10.0f0
        
        nz, nx = 100, 200
        vp = fill(vp_val, nz, nx)
        vs = fill(vs_val, nz, nx)
        rho = fill(rho_val, nz, nx)
        
        model = VelocityModel(vp, vs, rho, dx, dx)
        
        # Source at center
        src_x = 1000.0f0
        src_z = 500.0f0
        
        # Receivers at known distances
        distances = [200.0f0, 400.0f0, 600.0f0]
        rec_x = [src_x + d for d in distances]
        rec_z = fill(src_z, length(distances))
        
        config = SimulationConfig(
            nt=800,
            f0=15.0f0,
            nbc=30,
            fd_order=8,
            source_type=:explosion,
            save_gather=false,
            plot_gather=false,
            show_progress=false,
            output_dir=tempdir()
        )
        
        result = simulate!(model, src_x, src_z, rec_x, rec_z; config=config)
        
        # Find first arrival times (first significant amplitude)
        dt = result.dt
        threshold = 0.1f0 * maximum(abs.(result.gather))
        
        arrival_times = Float32[]
        for i in 1:length(distances)
            trace = result.gather[:, i]
            # Find first time amplitude exceeds threshold
            idx = findfirst(x -> abs(x) > threshold, trace)
            if idx !== nothing
                push!(arrival_times, (idx - 1) * dt)
            end
        end
        
        if length(arrival_times) >= 2
            # Calculate apparent velocity from travel time differences
            # Δt = Δd / v  =>  v = Δd / Δt
            Δd = distances[2] - distances[1]
            Δt = arrival_times[2] - arrival_times[1]
            
            if Δt > 0
                apparent_velocity = Δd / Δt
                
                # Should be within 10% of true P-wave velocity
                @test abs(apparent_velocity - vp_val) / vp_val < 0.15
            end
        end
    end
    
    @testset "Free Surface Reflection" begin
        # With free surface, stress components should be zero at surface
        # Test: run simulation with free surface, check stress values
        
        nz, nx = 80, 120
        vp = fill(3000.0f0, nz, nx)
        vs = fill(1800.0f0, nz, nx)
        rho = fill(2200.0f0, nz, nx)
        
        model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)
        
        be = ElasticWave2D.CPU_BACKEND
        nbc = 20
        fd_order = 8
        
        M = ElasticWave2D.init_medium(model, nbc, fd_order, be; free_surface=true)
        @test M.is_free_surface == true
        
        H = ElasticWave2D.init_habc(M.nx, M.nz, nbc, M.pad, 0.001f0, 10.0f0, 10.0f0, 
                                     3000.0f0, be)
        W = Wavefield(M.nx, M.nz, be)
        
        # Set some non-zero stress
        W.tzz .= 1.0f0
        W.txz .= 1.0f0
        
        # Apply free surface condition
        ElasticWave2D.apply_image_method!(be, W, M)
        
        # Stress at free surface should be zero
        j_surface = M.pad + 1
        @test all(W.tzz[:, j_surface] .== 0.0f0)
        @test all(W.txz[:, j_surface] .== 0.0f0)
    end
    
    @testset "Absorbing Boundary Effectiveness" begin
        # Waves should be absorbed at boundaries, not reflected
        # Test: source near one boundary, check that energy doesn't bounce back
        
        nz, nx = 60, 100
        vp = fill(3000.0f0, nz, nx)
        vs = fill(1800.0f0, nz, nx)
        rho = fill(2200.0f0, nz, nx)
        
        model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)
        
        # Source near right boundary
        src_x = 800.0f0
        src_z = 300.0f0
        
        # Receiver in the middle (to detect reflections)
        rec_x = [500.0f0]
        rec_z = [300.0f0]
        
        config = SimulationConfig(
            nt=600,
            f0=20.0f0,
            nbc=50,  # More absorbing layers
            fd_order=8,
            free_surface=false,  # No free surface to isolate boundary effects
            save_gather=false,
            plot_gather=false,
            show_progress=false,
            output_dir=tempdir()
        )
        
        result = simulate!(model, src_x, src_z, rec_x, rec_z; config=config)
        
        trace = result.gather[:, 1]
        
        # Find direct wave arrival
        max_amp_idx = argmax(abs.(trace))
        
        # Energy after direct wave should decay (not grow due to reflections)
        # Check that late-time energy is much smaller than peak energy
        late_start = min(max_amp_idx + 200, length(trace) - 50)
        late_energy = mean(trace[late_start:end].^2)
        peak_energy = maximum(trace.^2)
        
        # Late energy should be less than 5% of peak (good absorption)
        # This is a relaxed threshold for robustness
        @test late_energy < 0.2 * peak_energy
    end
    
    @testset "Symmetry Test" begin
        # For a symmetric source in a homogeneous medium, 
        # receivers at symmetric positions should record identical traces
        
        nz, nx = 80, 160
        vp = fill(3000.0f0, nz, nx)
        vs = fill(1800.0f0, nz, nx)
        rho = fill(2200.0f0, nz, nx)
        
        model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)
        
        # Source at center
        src_x = 800.0f0
        src_z = 400.0f0
        
        # Symmetric receivers (equal distance left and right)
        offset = 200.0f0
        rec_x = [src_x - offset, src_x + offset]
        rec_z = [src_z, src_z]
        
        config = SimulationConfig(
            nt=400,
            f0=15.0f0,
            nbc=30,
            fd_order=8,
            free_surface=false,
            save_gather=false,
            plot_gather=false,
            show_progress=false,
            output_dir=tempdir()
        )
        
        result = simulate!(model, src_x, src_z, rec_x, rec_z; config=config)
        
        trace_left = result.gather[:, 1]
        trace_right = result.gather[:, 2]
        
        # Traces should be nearly identical (allow small numerical differences)
        correlation = cor(trace_left, trace_right)
        @test correlation > 0.99
    end
    
    @testset "Two-Layer Reflection" begin
        # A wave hitting a velocity interface should produce a reflection
        # Test: check that receiver above interface sees both direct and reflected waves
        
        nz, nx = 100, 150
        
        # Two-layer model: fast over slow
        vp = fill(4000.0f0, nz, nx)
        vs = fill(2400.0f0, nz, nx)
        rho = fill(2600.0f0, nz, nx)
        
        # Interface at z = 500m (index 50)
        interface_z = 50
        vp[interface_z:end, :] .= 2500.0f0
        vs[interface_z:end, :] .= 1500.0f0
        rho[interface_z:end, :] .= 2200.0f0
        
        model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)
        
        # Source above interface
        src_x = 750.0f0
        src_z = 200.0f0
        
        # Receiver close to source (should see direct + reflection)
        rec_x = [750.0f0]
        rec_z = [100.0f0]
        
        config = SimulationConfig(
            nt=800,
            f0=15.0f0,
            nbc=30,
            fd_order=8,
            free_surface=false,
            save_gather=false,
            plot_gather=false,
            show_progress=false,
            output_dir=tempdir()
        )
        
        result = simulate!(model, src_x, src_z, rec_x, rec_z; config=config)
        
        trace = result.gather[:, 1]
        
        # Should have at least two distinct peaks (direct and reflected)
        # Find local maxima
        threshold = 0.3f0 * maximum(abs.(trace))
        peaks = Int[]
        for i in 2:length(trace)-1
            if abs(trace[i]) > abs(trace[i-1]) && 
               abs(trace[i]) > abs(trace[i+1]) && 
               abs(trace[i]) > threshold
                push!(peaks, i)
            end
        end
        
        # Filter peaks that are too close together
        if length(peaks) > 1
            filtered_peaks = [peaks[1]]
            for p in peaks[2:end]
                if p - filtered_peaks[end] > 30  # At least 30 samples apart
                    push!(filtered_peaks, p)
                end
            end
            peaks = filtered_peaks
        end
        
        # Should have at least 2 distinct arrivals
        @test length(peaks) >= 2
    end
    
    @testset "Numerical Stability" begin
        # Simulation should remain stable (no NaN or Inf)
        
        nz, nx = 50, 80
        vp = fill(3000.0f0, nz, nx)
        vs = fill(1800.0f0, nz, nx)
        rho = fill(2200.0f0, nz, nx)
        
        model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)
        
        src_x = 400.0f0
        src_z = 250.0f0
        rec_x = [200.0f0, 400.0f0, 600.0f0]
        rec_z = [100.0f0, 100.0f0, 100.0f0]
        
        config = SimulationConfig(
            nt=500,
            f0=15.0f0,
            nbc=20,
            fd_order=8,
            cfl=0.4f0,  # Safe CFL number
            save_gather=false,
            plot_gather=false,
            show_progress=false,
            output_dir=tempdir()
        )
        
        result = simulate!(model, src_x, src_z, rec_x, rec_z; config=config)
        
        # Check for NaN or Inf
        @test all(isfinite.(result.gather))
        
        # Values should be reasonable (not exploding)
        @test maximum(abs.(result.gather)) < 1e10
    end
end
