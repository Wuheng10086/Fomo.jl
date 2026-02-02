using ElasticWave2D

function elastic_wave_demo()
    println("ElasticWave2D.jl - Elastic Wave Phenomena Demo (Two-Layer Model)")
    println("======================================================")

    fast = get(ENV, "ELASTICWAVE_DEMO_FAST", "1") == "1"

    # 1. Define Model
    # Keep physical size same: 1600m x 1000m
    dx, dz = fast ? (10.0f0, 10.0f0) : (0.5f0, 0.5f0)
    nx = round(Int, 1600.0 / dx)
    nz = round(Int, 1000.0 / dz)

    println("1. Creating two-layer velocity model...")
    println("   Grid: $nx x $nz (dx=dz=$dx m)")

    # Initialize with top layer properties
    vp = fill(3000.0f0, nz, nx)
    vs = fill(1800.0f0, nz, nx)
    rho = fill(2200.0f0, nz, nx)

    # Define interface depth at 500m
    interface_z_idx = round(Int, 500.0 / dz)

    # Set bottom layer properties (High contrast)
    # Z range: interface to bottom
    vp[interface_z_idx:end, :] .= 4000.0f0
    vs[interface_z_idx:end, :] .= 2400.0f0
    rho[interface_z_idx:end, :] .= 2600.0f0

    model = VelocityModel(vp, vs, rho, dx, dz; name="two_layer_model")

    # 2. Define Source and Receivers
    println("2. Setting up source and receivers...")
    # Source near the top center
    src_x = 800.0f0
    src_z = 50.0f0   # Shallow source

    # Receivers on the surface
    rec_x = Float32.(collect(50:10:1550))
    rec_z = fill(2.0f0, length(rec_x)) # Near surface

    # Ensure output directory exists
    mkpath("outputs/elastic_wave_demo")

    # Plot Setup
    plot_setup(model, [src_x], [src_z], rec_x, rec_z;
        title="Two-Layer Model Setup",
        output="outputs/elastic_wave_demo/setup.png"
    )

    # 3. Run Simulation
    println("3. Running simulation...")
    # Higher frequency source (40Hz)

    config = SimulationConfig(
        nt=fast ? 1500 : 12000,
        f0=fast ? 25.0f0 : 40.0f0,
        cfl=0.3f0,
        free_surface=true,
        source_type=:force_z,
        output_dir="outputs/elastic_wave_demo",
        save_gather=true,
        plot_gather=true,
        show_progress=true
    )

    video_config = fast ? nothing : VideoConfig(
        fields=[:vz],
        skip=50,
        fps=20,
        colormap=:seismic
    )

    result = simulate!(model, src_x, src_z, rec_x, rec_z;
        config=config,
        video_config=video_config
    )

    println("\nSimulation complete!")
    println("Results saved to: $(config.output_dir)")
    if video_config !== nothing
        println(" - Check 'wavefield_vz.mp4' to see reflection/refraction")
    end
end

# Run the demo
elastic_wave_demo()
