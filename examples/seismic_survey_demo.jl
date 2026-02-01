using ElasticWave2D

"""
    seismic_survey_demo()

Compare different surface handling methods in ElasticWave2D.

This demo shows 4 scenarios:
1. Absorbing boundary (no surface waves)
2. Explicit free surface boundary condition
3. Vacuum formulation (flat surface)
4. Vacuum formulation (irregular topography)
"""
function seismic_survey_demo()
    println("ElasticWave2D.jl - Seismic Survey Scenarios Demo")
    println("================================================")
    println("Comparing different surface handling methods")
    println()

    # Common parameters
    nx, nz = 200, 100
    dx, dz = 10.0f0, 10.0f0
    nt = 1000
    f0 = 15.0f0

    # Create base layered model
    function create_layered_model(name_suffix)
        vp = zeros(Float32, nz, nx)
        vs = zeros(Float32, nz, nx)
        rho = zeros(Float32, nz, nx)

        for ix in 1:nx
            for iz in 1:nz
                depth = (iz - 1) * dz
                if depth < 300
                    vp[iz, ix] = 2000.0f0
                    vs[iz, ix] = 1200.0f0
                    rho[iz, ix] = 2000.0f0
                elseif depth < 600
                    vp[iz, ix] = 3000.0f0
                    vs[iz, ix] = 1800.0f0
                    rho[iz, ix] = 2200.0f0
                else
                    vp[iz, ix] = 4000.0f0
                    vs[iz, ix] = 2400.0f0
                    rho[iz, ix] = 2500.0f0
                end
            end
        end
        return VelocityModel(vp, vs, rho, dx, dz; name="layer_model_$(name_suffix)")
    end

    # Survey geometry
    src_x = 1000.0f0
    src_z = 20.0f0

    # Receivers (avoid exact source position)
    rec_x = Float32.(collect(102.5:5.0:1902.5))
    rec_z = fill(10.0f0, length(rec_x))

    # =================================================================
    # Scenario 1: Absorbing Boundary (No Surface Waves)
    # =================================================================
    println("1. Absorbing boundary (no surface waves)")

    model1 = create_layered_model("absorbing")
    output_dir1 = "outputs/survey_demo/1_absorbing"
    mkpath(output_dir1)

    plot_setup(model1, [src_x], [src_z], rec_x, rec_z;
        title="Scenario 1: Absorbing (No Surface Waves)",
        output=joinpath(output_dir1, "setup.png")
    )

    seismic_survey(model1, (src_x, src_z), (rec_x, rec_z);
        surface_method=:absorbing,
        config=SimulationConfig(
            nt=nt, f0=f0,
            output_dir=output_dir1
        )
    )

    # =================================================================
    # Scenario 2: Explicit Free Surface
    # =================================================================
    println("2. Explicit free surface boundary condition")

    model2 = create_layered_model("image_method")
    output_dir2 = "outputs/survey_demo/2_image_method"
    mkpath(output_dir2)

    plot_setup(model2, [src_x], [src_z], rec_x, rec_z;
        title="Scenario 2: Explicit Free Surface (Image Method)",
        output=joinpath(output_dir2, "setup.png")
    )

    # Use :image method (explicit BC)
    seismic_survey(model2, (src_x, src_z), (rec_x, rec_z);
        surface_method=:image,
        config=SimulationConfig(
            nt=nt, f0=f0,
            output_dir=output_dir2
        )
    )

    # =================================================================
    # Scenario 3: Vacuum Formulation (Flat Surface)
    # =================================================================
    println("3. Vacuum formulation (flat surface)")

    model3 = create_layered_model("vacuum_flat")
    output_dir3 = "outputs/survey_demo/3_vacuum_flat"
    mkpath(output_dir3)

    plot_setup(model3, [src_x], [src_z], rec_x, rec_z;
        title="Scenario 3: Vacuum (Flat)",
        output=joinpath(output_dir3, "setup.png")
    )

    # Use new vacuum API - automatically adds vacuum layers
    seismic_survey(model3, (src_x, src_z), (rec_x, rec_z);
        surface_method=:vacuum,
        vacuum_layers=5,
        config=SimulationConfig(
            nt=nt, f0=f0,
            output_dir=output_dir3
        )
    )

    # =================================================================
    # Scenario 4: Vacuum Formulation (Irregular Topography)
    # =================================================================
    println("4. Vacuum formulation (irregular topography)")

    model4 = create_layered_model("vacuum_topo")
    output_dir4 = "outputs/survey_demo/4_vacuum_topo"
    mkpath(output_dir4)

    # Define sinusoidal topography
    topo_z(x) = 50.0f0 + 30.0f0 * sin(2.0f0 * π * x / 1000.0f0)

    # Create topography by setting vacuum above surface
    for ix in 1:nx
        x = (ix - 1) * dx
        surface_depth = topo_z(x)
        surface_idx = round(Int, surface_depth / dz)
        surface_idx = clamp(surface_idx, 1, nz)

        if surface_idx > 1
            model4.vp[1:surface_idx-1, ix] .= 0.0f0
            model4.vs[1:surface_idx-1, ix] .= 0.0f0
            model4.rho[1:surface_idx-1, ix] .= 0.0f0
        end
    end

    # Place source 20m below local surface
    src_surf = topo_z(src_x)
    src_z_topo = src_surf + 20.0f0

    # Place receivers on the surface
    rec_z_topo = Float32[topo_z(x) for x in rec_x]

    plot_setup(model4, [src_x], [src_z_topo], rec_x, rec_z_topo;
        title="Scenario 4: Vacuum (Topography)",
        output=joinpath(output_dir4, "setup.png")
    )

    # For irregular topography, use simulate! directly
    # (seismic_survey's vacuum mode only supports flat surface)
    config4 = SimulationConfig(
        nt=nt, f0=f0,
        free_surface=false,  # Using vacuum
        output_dir=output_dir4
    )

    simulate!(model4, src_x, src_z_topo, rec_x, rec_z_topo; config=config4)

    # =================================================================
    # Summary
    # =================================================================
    println()
    println("="^50)
    println("Demo Complete!")
    println("="^50)
    println()
    println("Results saved to outputs/survey_demo/")
    println()
    println("Comparison:")
    println("  1_absorbing/    - No surface waves (body waves only)")
    println("  2_image_method/ - Explicit boundary condition (Image Method)")
    println("  3_vacuum_flat/  - Vacuum formulation (flat)")
    println("  4_vacuum_topo/  - Vacuum formulation (topography)")
    println()
    println("Surface method options in seismic_survey():")
    println("  :absorbing    - Absorbing top boundary")
    println("  :image        - Explicit free surface BC (Image Method)")
    println("  :vacuum       - Vacuum formulation (auto adds layers)")
end

# Run the demo
seismic_survey_demo()