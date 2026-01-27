using Fomo

function seismic_survey_demo()
    println("Fomo.jl - Seismic Survey Scenarios Demo (Optimized)")
    println("==================================================")

    # Common parameters
    nx, nz = 200, 100
    dx, dz = 10.0f0, 10.0f0
    nt = 1000
    f0 = 15.0f0

    # Create base layered model function
    # Correct dimension order: [nz, nx]
    function create_layered_model(name_suffix)
        vp = zeros(Float32, nz, nx)
        vs = zeros(Float32, nz, nx)
        rho = zeros(Float32, nz, nx)

        for ix in 1:nx
            for iz in 1:nz
                depth = (iz - 1) * dz
                if depth < 300
                    vp[iz, ix] = 2000.0
                    vs[iz, ix] = 1200.0
                    rho[iz, ix] = 2000.0
                elseif depth < 600
                    vp[iz, ix] = 3000.0
                    vs[iz, ix] = 1800.0
                    rho[iz, ix] = 2200.0
                else
                    vp[iz, ix] = 4000.0
                    vs[iz, ix] = 2400.0
                    rho[iz, ix] = 2500.0
                end
            end
        end
        return VelocityModel(vp, vs, rho, dx, dz; name="layer_model_$(name_suffix)")
    end

    # Define Geometry
    # Source at 1000m, 20m depth
    src_x_pos = 1000.0f0
    src_z_pos = 20.0f0
    sources = [(src_x_pos, src_z_pos)]

    # Receivers: Offset from source to avoid singularity at 0 offset
    # Source is at 1000m. Let's put receivers from 100m to 1900m.
    # The receiver at 1000m will be exactly on top of source.
    # Let's shift receivers slightly or accept the strong direct wave.
    # User requested: "Don't put rec and source at same position"
    # We can just ensure no receiver is exactly at 1000.0m
    # 100:20:1900 hits 1000.0. Let's shift by 10m: 110:20:1910
    # UPDATE: User requested denser spacing (5m)
    rec_x = Float32.(collect(102.5:5.0:1902.5)) # Offset by 2.5m to avoid exact 1000.0

    # Receivers at surface (z=0) or slightly buried?
    # Usually z=0 means free surface. 
    # For HABC (Scenario 1), z=0 is just the top boundary.
    # Let's put them at z=10m to be safe and consistent.
    rec_z = fill(10.0f0, length(rec_x))

    # =================================================================
    # Scenario 1: No Surface Waves (Absorbing Boundary)
    # =================================================================
    println("\n1. Scenario: No Surface Waves (Absorbing Boundary)")
    model1 = create_layered_model("no_sw")

    mkpath("outputs/survey_demo/1_no_surface_waves")
    plot_setup(model1, [sources[1][1]], [sources[1][2]], rec_x, rec_z;
        title="Scenario 1: No Surface Waves (Absorbing)",
        output="outputs/survey_demo/1_no_surface_waves/setup.png"
    )

    seismic_survey(model1, sources, (rec_x, rec_z);
        simulate_surface_waves=false,
        source_depth_margin=80.0,
        config=SimulationConfig(
            nt=nt, f0=f0,
            output_dir="outputs/survey_demo/1_no_surface_waves"
        )
    )

    # =================================================================
    # Scenario 2: Surface Waves (Explicit Free Surface)
    # =================================================================
    println("\n2. Scenario: Surface Waves (Explicit Free Surface)")
    model2 = create_layered_model("explicit_sw")

    mkpath("outputs/survey_demo/2_explicit_surface_waves")
    plot_setup(model2, [sources[1][1]], [sources[1][2]], rec_x, rec_z;
        title="Scenario 2: Surface Waves (Explicit Free Surface)",
        output="outputs/survey_demo/2_explicit_surface_waves/setup.png"
    )

    seismic_survey(model2, sources, (rec_x, rec_z);
        simulate_surface_waves=true,
        config=SimulationConfig(
            nt=nt, f0=f0,
            output_dir="outputs/survey_demo/2_explicit_surface_waves"
        )
    )

    # =================================================================
    # Scenario 3: Surface Waves (Vacuum Method - Flat)
    # =================================================================
    println("\n3. Scenario: Surface Waves (Vacuum Method - Flat)")
    model3 = create_layered_model("vacuum_flat")
    vacuum_layers = 5
    model3.vp[1:vacuum_layers, :] .= 0.0
    model3.vs[1:vacuum_layers, :] .= 0.0
    model3.rho[1:vacuum_layers, :] .= 0.0

    # Interface is at depth = vacuum_layers * dz = 50m
    # Source should be below interface. Put it at 70m (20m below interface).
    src_z_vac = 70.0f0
    sources_vac = [(src_x_pos, src_z_vac)]

    # Receivers should be ON the interface (the "surface")
    # Interface depth is 50m.
    rec_z_vac = fill(50.0f0, length(rec_x))

    mkpath("outputs/survey_demo/3_vacuum_flat")
    plot_setup(model3, [sources_vac[1][1]], [sources_vac[1][2]], rec_x, rec_z_vac;
        title="Scenario 3: Surface Waves (Vacuum Flat)",
        output="outputs/survey_demo/3_vacuum_flat/setup.png"
    )

    seismic_survey(model3, sources_vac, (rec_x, rec_z_vac);
        simulate_surface_waves=false,
        source_depth_margin=80.0,
        config=SimulationConfig(
            nt=nt, f0=f0,
            output_dir="outputs/survey_demo/3_vacuum_flat"
        )
    )

    # =================================================================
    # Scenario 4: Surface Waves (Vacuum Method - Irregular Topography)
    # =================================================================
    println("\n4. Scenario: Surface Waves (Vacuum Method - Irregular Topography)")
    model4 = create_layered_model("vacuum_topo")

    # Define topography function
    topo_z(x) = 50.0 + 30.0 * sin(2 * pi * x / 1000.0)

    # Create sinusoidal topography
    for ix in 1:nx
        x = (ix - 1) * dx
        surface_depth = topo_z(x)
        surface_depth_idx = round(Int, surface_depth / dz)
        surface_depth_idx = clamp(surface_depth_idx, 1, nz)

        # Set vacuum above surface
        if surface_depth_idx > 1
            model4.vp[1:surface_depth_idx-1, ix] .= 0.0
            model4.vs[1:surface_depth_idx-1, ix] .= 0.0
            model4.rho[1:surface_depth_idx-1, ix] .= 0.0
        end
    end

    # Dynamic Source Placement
    # Place source 20m below the local surface at src_x_pos
    src_surf_z = topo_z(src_x_pos)
    src_z_topo = src_surf_z + 20.0f0
    sources_topo = [(src_x_pos, Float32(src_z_topo))]

    # Dynamic Receiver Placement
    # Place receivers exactly on the local surface
    rec_z_topo = Vector{Float32}(undef, length(rec_x))
    for i in 1:length(rec_x)
        # Snap to nearest grid point to avoid being "inside" vacuum if interpolated
        # Ideally, should be exactly at the vacuum-solid interface index
        surf_z = topo_z(rec_x[i])
        grid_z_idx = round(Int, surf_z / dz)
        rec_z_topo[i] = Float32(grid_z_idx * dz)
    end

    mkpath("outputs/survey_demo/4_vacuum_topo")
    plot_setup(model4, [sources_topo[1][1]], [sources_topo[1][2]], rec_x, rec_z_topo;
        title="Scenario 4: Surface Waves (Vacuum Topo)",
        output="outputs/survey_demo/4_vacuum_topo/setup.png"
    )

    seismic_survey(model4, sources_topo, (rec_x, rec_z_topo);
        simulate_surface_waves=false,
        source_depth_margin=80.0,
        config=SimulationConfig(
            nt=nt, f0=f0,
            output_dir="outputs/survey_demo/4_vacuum_topo"
        )
    )

    println("\nSeismic survey scenarios complete!")
    println("Results saved to outputs/survey_demo/")
end

# Run the demo
seismic_survey_demo()
