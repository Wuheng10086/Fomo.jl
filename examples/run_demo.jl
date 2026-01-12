# ==============================================================================
# run_demo.jl
#
# Demonstrates the high-level API
# ==============================================================================

using Fomo

# ==============================================================================
# Demo 1: Quick test with homogeneous model (no video)
# ==============================================================================

println("="^60)
println("  Demo 1: Quick test")
println("="^60)

nx, nz = 200, 100
dx = 10.0f0

vp = fill(3000.0f0, nz, nx)
vs = fill(1800.0f0, nz, nx)
rho = fill(2200.0f0, nz, nx)

model = VelocityModel(vp, vs, rho, dx, dx; name="Homogeneous")

result1 = simulate!(
    model,
    nx * dx / 2, 500.0f0,
    Float32.(range(dx * 5, dx * (nx - 5), length=50)),
    fill(10.0f0, 50);
    config=SimulationConfig(nt=1000, output_dir="outputs/demo1_quick", free_surface=false),
    video_config=VideoConfig(fields=[:vz], skip=10),
    be=backend(:cpu)
)

println("Outputs: outputs/demo1_quick/")


# ==============================================================================
# Demo 2: Surface waves with video
# ==============================================================================

println("\n" * "="^60)
println("  Demo 2: Surface waves (with video)")
println("="^60)

nx, nz = 600, 400
dx = 5.0f0

vp = zeros(Float32, nz, nx)
vs = zeros(Float32, nz, nx)
rho = zeros(Float32, nz, nx)

vp[1:160, :] .= 2500.0f0
vs[1:160, :] .= 1500.0f0
rho[1:160, :] .= 2200.0f0

vp[161:end, :] .= 4000.0f0
vs[161:end, :] .= 2400.0f0
rho[161:end, :] .= 2600.0f0

model2 = VelocityModel(vp, vs, rho, dx, dx; name="Two-layer")

println("Expected waves:")
println("  P-wave:   ~2500 m/s")
println("  S-wave:   ~1500 m/s")
println("  Rayleigh: ~1380 m/s")

# VideoConfig as separate parameter!
vc = VideoConfig(fields=[:vz], skip=5, fps=30, colormap=:seismic)

result2 = simulate!(
    model2,
    nx * dx / 2, 50.0f0,
    Float32.(range(100, nx * dx - 100, length=100)),
    fill(10.0f0, 100);
    config=SimulationConfig(nt=4000, f0=20.0f0, output_dir="outputs/demo2_surface_waves"),
    video_config=vc
)

println("Outputs: outputs/demo2_surface_waves/")


# ==============================================================================
# Demo 3: Irregular surface with video
# ==============================================================================

println("\n" * "="^60)
println("  Demo 3: Irregular surface (with video)")
println("="^60)

nx, nz = 400, 200
dx = 10.0f0

vp = fill(3000.0f0, nz, nx)
vs = fill(1800.0f0, nz, nx)
rho = fill(2200.0f0, nz, nx)
vp[100:end, :] .= 4000.0f0
vs[100:end, :] .= 2400.0f0

model3 = VelocityModel(vp, vs, rho, dx, dx; name="Irregular demo")

z_surface = combine_surfaces(
    sinusoidal_surface(nx, dx; base_depth=50, amplitude=25, wavelength=1500),
    gaussian_valley(nx, dx; base_depth=0, valley_depth=20, width=250)
)

println("Surface relief: $(round(maximum(z_surface) - minimum(z_surface), digits=1))m")

result3 = simulate_irregular!(
    model3,
    z_surface,
    nx * dx / 2,
    Float32.(50:20:(nx*dx-50));
    config=IrregularSurfaceConfig(
        nt=3000,
        ibm_method=:direct_zero,
        src_depth=30.0f0,
        output_dir="outputs/demo3_irregular"
    ),
    video_config=VideoConfig(fields=[:vz], skip=10)
)

println("Outputs: outputs/demo3_irregular/")


# ==============================================================================
# Summary
# ==============================================================================

println("\n" * "="^60)
println("  All demos complete!")
println("="^60)
println("\nOutput directories:")
println("  - outputs/demo1_quick/       (no video)")
println("  - outputs/demo2_surface_waves/ (with video)")
println("  - outputs/demo3_irregular/    (with video)")