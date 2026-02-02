# =============================================================================
# Test Script: Source Types and Boundary Conditions
# =============================================================================

using ElasticWave2D
using ElasticWave2D.API: VelocityModel, SimConfig, SourceConfig, simulate, save_result, 
                         Explosion, ForceZ, ForceX, Absorbing, FreeSurface, Vacuum, line_receivers
using Printf

function run_tests()
    # 1. Setup Common Model
    # -------------------------------------------------------------------------
    println("Setting up model...")
    nx, nz, dx = 200, 150, 10.0f0
    
    # Homogeneous half-space to clearly see radiation patterns and reflections
    vp = fill(3000.0f0, nz, nx)
    vs = fill(1800.0f0, nz, nx)
    rho = fill(2200.0f0, nz, nx)
    
    model = VelocityModel(vp, vs, rho, dx, dx; name="Homogeneous")
    
    # Common Receiver Line
    receivers = line_receivers(100, 1900, 181; z=50.0) # z=50m
    
    # Output Directory
    out_dir = joinpath("outputs", "test_sources_boundaries")
    mkpath(out_dir)
    
    # Common Simulation Config
    nt = 1000
    
    # =========================================================================
    # Scenario A: Source Types Comparison (Absorbing Boundary)
    # =========================================================================
    println("\n" * "="^50)
    println("Scenario A: Testing Source Types (Absorbing Boundary)")
    println("="^50)
    
    # Use Absorbing boundary to simulate infinite medium (no reflections)
    config_abs = SimConfig(nt=nt, boundary=Absorbing(), output_dir=out_dir)
    
    sources = [
        (Explosion, "Explosion"),
        (ForceZ,    "ForceZ"),
        (ForceX,    "ForceX")
    ]
    
    for (mech, name) in sources
        println("  Running Source: $name...")
        
        # Source at center
        src = SourceConfig(1000.0, 750.0; f0=20.0, type=mech)
        
        result = simulate(model, src, receivers; config=config_abs)
        
        # Save and Plot
        fname = "source_$(lowercase(name))"
        save_result(result, joinpath(out_dir, "$fname.jld2"))
        
        ElasticWave2D.plot_gather(result.gather, result.receivers.x, result.dt; 
            title="Source: $name (Absorbing Boundary)", 
            output_path=joinpath(out_dir, "$fname.png"))
    end

    # =========================================================================
    # Scenario B: Boundary Conditions Comparison (Explosion Source)
    # =========================================================================
    println("\n" * "="^50)
    println("Scenario B: Testing Boundary Conditions (Explosion Source)")
    println("="^50)
    
    # Source close to surface to generate strong reflections/surface waves
    src_shallow = SourceConfig(1000.0, 100.0; f0=20.0, type=Explosion)
    
    boundaries = [
        (Absorbing(),      "Absorbing",    "boundary_absorbing"),
        (FreeSurface(),    "FreeSurface",  "boundary_freesurface"),
        (Vacuum(10),       "Vacuum(10)",   "boundary_vacuum")
    ]
    
    for (bc, name, fname) in boundaries
        println("  Running Boundary: $name...")
        
        config = SimConfig(nt=nt, boundary=bc, output_dir=out_dir)
        
        result = simulate(model, src_shallow, receivers; config=config)
        
        # Save and Plot
        save_result(result, joinpath(out_dir, "$fname.jld2"))
        
        ElasticWave2D.plot_gather(result.gather, result.receivers.x, result.dt; 
            title="Boundary: $name (Explosion Source)", 
            output_path=joinpath(out_dir, "$fname.png"))
    end
    
    println("\n" * "="^50)
    println("All tests completed. Results saved to $out_dir")
    println("="^50)
end

# Run the tests
run_tests()
