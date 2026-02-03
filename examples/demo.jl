# =============================================================================
# ElasticWave2D.jl Examples
# =============================================================================
# 
# 运行方式:
#   julia --project=. examples/demo.jl              # 运行所有 demo
#   julia --project=. examples/demo.jl basic        # 只运行基础 demo
#   julia --project=. examples/demo.jl tunnel       # 只运行隧道 demo
#   julia --project=. examples/demo.jl batch        # 只运行批量 demo
#
# =============================================================================

using ElasticWave2D

# -----------------------------------------------------------------------------
# Demo 1: 基础双层模型
# -----------------------------------------------------------------------------
function demo_basic()
    println("\n" * "="^50)
    println("Demo 1: 双层模型 + 自由表面")
    println("="^50)

    # 创建双层模型
    nx, nz, dx = 200, 100, 10.0f0
    vp = fill(2500.0f0, nz, nx)
    vs = fill(1500.0f0, nz, nx)
    rho = fill(2200.0f0, nz, nx)
    vp[50:end, :] .= 3500.0f0  # 下层高速
    vs[50:end, :] .= 2100.0f0
    rho[50:end, :] .= 2500.0f0

    model = VelocityModel(vp, vs, rho, dx, dx)

    source = SourceConfig(1000.0, 30.0; f0=20.0)
    receivers = line_receivers(100.0, 1900.0, 91; z=10.0)

    video_config = VideoConfig(fields=[:vz], skip=10, fps=30)
    outputs = OutputConfig(base_dir="./outputs/demo_basic", plot_gather=true, plot_setup=true, video_config=video_config)
    boundary = top_image()
    simconf = SimConfig(nt=1500, dt=nothing, cfl=0.4, fd_order=8)

    gather = simulate(model, source, receivers, boundary, outputs, simconf; progress=true)

    println("✓ 完成! Gather size: $(size(gather))")
    return gather
end

# -----------------------------------------------------------------------------
# Demo 2: Vacuum 边界 (推荐方式)
# -----------------------------------------------------------------------------
function demo_vacuum()
    println("\n" * "="^50)
    println("Demo 2: Vacuum 自由表面 (推荐)")
    println("="^50)

    nx, nz, dx = 200, 100, 10.0f0
    vp = fill(3000.0f0, nz, nx)
    vs = fill(1800.0f0, nz, nx)
    rho = fill(2200.0f0, nz, nx)

    model = VelocityModel(vp, vs, rho, dx, dx)

    source = SourceConfig(1000.0, 50.0, Ricker(15.0), ForceZ)
    receivers = line_receivers(100.0, 1900.0, 91; z=10.0)

    outputs = OutputConfig(base_dir="./outputs/demo_vacuum", plot_gather=true, plot_setup=true, video_config=nothing)
    boundary = top_vacuum(10)
    simconf = SimConfig(nt=2000)
    gather = simulate(
        model,
        source,
        receivers,
        boundary,
        outputs,
        simconf;
        progress=true
    )

    println("✓ 完成! Gather size: $(size(gather))")
    return gather
end

# -----------------------------------------------------------------------------
# Demo 3: 隧道探测 (空腔)
# -----------------------------------------------------------------------------
function demo_tunnel()
    println("\n" * "="^50)
    println("Demo 3: 隧道/空腔探测")
    println("="^50)

    nx, nz, dx = 200, 100, 5.0f0
    vp = fill(2000.0f0, nz, nx)
    vs = fill(1200.0f0, nz, nx)
    rho = fill(2000.0f0, nz, nx)

    # 挖一个隧道 (ρ=0)
    rho[35:40, 90:110] .= 0.0f0
    vp[35:40, 90:110] .= 0.0f0
    vs[35:40, 90:110] .= 0.0f0

    model = VelocityModel(vp, vs, rho, dx, dx)

    source = SourceConfig(250.0, 10.0; f0=60.0)
    receivers = line_receivers(50.0, 950.0, 200; z=10.0)

    outputs = OutputConfig(base_dir="./outputs/demo_tunnel", plot_gather=true, video_config=nothing)
    boundary = top_vacuum(10)
    simconf = SimConfig(nt=1500)
    gather = simulate(
        model,
        source,
        receivers,
        boundary,
        outputs,
        simconf;
        progress=true
    )

    println("✓ 完成! 观察绕射波和阴影区")
    return gather
end

# -----------------------------------------------------------------------------
# Demo 4: 批量多炮模拟 (高性能)
# -----------------------------------------------------------------------------
function demo_batch()
    println("\n" * "="^50)
    println("Demo 4: 批量多炮模拟")
    println("="^50)

    nx, nz, dx = 300, 150, 10.0f0
    vp = fill(3000.0f0, nz, nx)
    vs = fill(1800.0f0, nz, nx)
    rho = fill(2200.0f0, nz, nx)

    model = VelocityModel(vp, vs, rho, dx, dx)

    receivers = line_receivers(100.0, 2900.0, 141; z=10.0)

    source_template = SourceConfig(0.0, 0.0; f0=15.0)
    boundary = top_image(nbc=50)
    simconf = SimConfig(nt=2000)
    sim = BatchSimulator(model, source_template, receivers, boundary, simconf)

    # 10 炮
    src_x = Float32.(500:200:2500)
    src_z = fill(20.0f0, length(src_x))

    println("运行 $(length(src_x)) 炮...")
    outputs = OutputConfig(base_dir="./outputs/demo_batch", plot_gather=false, plot_setup=true, video_config=nothing)
    t = @elapsed simulate_shots!(sim, src_x, src_z; store=false, outputs=outputs, progress=false)

    println("✓ 完成! 总耗时: $(round(t, digits=2))s, 平均: $(round(t/length(src_x), digits=3))s/炮")
    return nothing
end

# -----------------------------------------------------------------------------
# Demo 5: 带视频输出
# -----------------------------------------------------------------------------
function demo_video()
    println("\n" * "="^50)
    println("Demo 5: 波场视频录制")
    println("="^50)

    nx, nz, dx = 200, 100, 10.0f0
    vp = fill(3000.0f0, nz, nx)
    vs = fill(1800.0f0, nz, nx)
    rho = fill(2200.0f0, nz, nx)

    model = VelocityModel(vp, vs, rho, dx, dx)

    source = SourceConfig(1000.0, 50.0; f0=15.0)
    receivers = line_receivers(100.0, 1900.0, 91; z=10.0)

    outputs = OutputConfig(
        base_dir="./outputs/demo_video",
        plot_gather=true,
        video_config=VideoConfig(fields=[:vz], skip=10, fps=30),
    )
    boundary = top_image()
    simconf = SimConfig(nt=1000)
    gather = simulate(
        model,
        source,
        receivers,
        boundary,
        outputs,
        simconf;
        progress=true
    )

    println("✓ 完成! 视频保存到 $(outputs.base_dir)")
    return gather
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
function main()
    demos = Dict(
        "basic" => demo_basic,
        "vacuum" => demo_vacuum,
        "tunnel" => demo_tunnel,
        "batch" => demo_batch,
        "video" => demo_video,
    )

    if length(ARGS) == 0
        # 运行所有
        demo_basic()
        demo_vacuum()
        demo_tunnel()
        demo_batch()
        println("\n" * "="^50)
        println("所有 Demo 完成!")
        println("运行 `julia examples/demo.jl video` 生成波场视频")
        println("="^50)
    else
        name = ARGS[1]
        if haskey(demos, name)
            demos[name]()
        else
            println("未知 demo: $name")
            println("可选: $(join(keys(demos), ", "))")
        end
    end
end

main()
