# ==============================================================================
# basic_example.jl
#
# Basic usage example for Fomo.jl
#
# Usage:
#   julia --project=.. basic_example.jl
# ==============================================================================

using Fomo

# ==============================================================================
# 1. Create a simple layered velocity model
# ==============================================================================

println("Creating velocity model...")

nx, nz = 200, 100      # Grid size (nx: horizontal, nz: vertical)
dx, dz = 10.0f0, 10.0f0  # Grid spacing (m)

# Create velocity model (2 layers)
# Note: Data stored as [nz, nx] (depth first, seismic convention)
vp = fill(2000.0f0, nz, nx)
vs = fill(1200.0f0, nz, nx)
rho = fill(2000.0f0, nz, nx)

# Second layer (faster) - lower half of model
vp[nz÷2:end, :] .= 3500.0f0
vs[nz÷2:end, :] .= 2100.0f0
rho[nz÷2:end, :] .= 2400.0f0

model = VelocityModel(vp, vs, rho, dx, dz; name="TwoLayer")
model_info(model)

# ==============================================================================
# 2. Setup simulation parameters
# ==============================================================================

nbc = 50        # Absorbing boundary cells
fd_order = 8    # Finite difference order
f0 = 15.0f0     # Source frequency (Hz)
total_time = 1.0f0  # Total simulation time (s)

# Time stepping (CFL condition)
cfl = 0.5f0
dt = cfl * min(dx, dz) / maximum(vp)
nt = ceil(Int, total_time / dt)

println("\nTime stepping: dt=$dt s, nt=$nt steps")

# ==============================================================================
# 3. Define acquisition geometry
# ==============================================================================

# Source at center-top
src_x = Float32[model.nx * dx / 2]
src_z = Float32[20.0]

# Receivers across the surface
rec_x = Float32.(range(0, (model.nx-1)*dx, step=20))
rec_z = fill(10.0f0, length(rec_x))

println("Sources: $(length(src_x)), Receivers: $(length(rec_x))")

# ==============================================================================
# 4. Initialize and run simulation
# ==============================================================================

# Choose backend (CPU or CUDA)
be = is_cuda_available() ? backend(:cuda) : backend(:cpu)
println("\nUsing backend: $(typeof(be))")

# Initialize
medium = init_medium(model, nbc, fd_order, be; free_surface=true)
habc = init_habc(medium.nx, medium.nz, nbc, dt, dx, dz, 
                 sum(vp)/length(vp), be)
params = SimParams(dt, nt, dx, dz, fd_order)
fd_coeffs = to_device(get_fd_coefficients(fd_order), be)
wavefield = Wavefield(medium.nx, medium.nz, be)
wavelet = ricker_wavelet(f0, dt, nt)

# Setup receivers
rec = setup_receivers(rec_x, rec_z, medium; type=:vz)

# Setup source
shot_config = MultiShotConfig(src_x, src_z, wavelet)

# Run!
println("\nRunning simulation...")
results = run_shots!(be, wavefield, medium, habc, fd_coeffs,
                     rec, shot_config, params)

println("Done! Gather size: $(size(results[1].gather))")

# ==============================================================================
# 5. Save results
# ==============================================================================

mkpath("outputs")

# Save gather
save_gather(results[1], "outputs/shot_gather.bin")

# Save geometry
geom = create_geometry(results, medium, params)
save_geometry("outputs/geometry.jld2", geom)

println("\nResults saved!")
println("  - outputs/shot_gather.bin")
println("  - outputs/geometry.jld2")
