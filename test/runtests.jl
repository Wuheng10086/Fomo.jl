# =============================================================================
# ElasticWave2D.jl 测试
# =============================================================================
#
# 运行: julia --project=. test/runtests.jl
#
# =============================================================================

using Test
using ElasticWave2D
using JLD2

@testset "ElasticWave2D.jl" begin

    include("outputs_tests.jl")

    # =========================================================================
    # 基础类型测试
    # =========================================================================
    @testset "VelocityModel" begin
        vp = fill(3000.0f0, 50, 100)
        vs = fill(1800.0f0, 50, 100)
        rho = fill(2200.0f0, 50, 100)

        model = VelocityModel(vp, vs, rho, 10.0f0, 10.0f0)

        @test model.nx == 100
        @test model.nz == 50
        @test model.dx == 10.0f0
        @test model.dz == 10.0f0
    end

    # =========================================================================
    # Wavelet 测试
    # =========================================================================
    @testset "Wavelet" begin
        # Ricker wavelet
        w = Ricker(15.0)
        @test w.f0 == 15.0f0

        wav = generate(w, 0.001f0, 1000)
        @test length(wav) == 1000
        @test maximum(abs, wav) > 0

        # CustomWavelet
        data = Float32.(sin.(0:0.1:10))
        cw = CustomWavelet(data)
        wav2 = generate(cw, 0.001f0, 50)
        @test length(wav2) == 50
    end

    # =========================================================================
    # Source/Receiver 配置测试
    # =========================================================================
    @testset "SourceConfig" begin
        src = SourceConfig(100.0, 50.0; f0=20.0)
        @test src.x == 100.0f0
        @test src.z == 50.0f0
        @test src.mechanism == Explosion

        src2 = SourceConfig(100.0, 50.0, Ricker(15.0), ForceZ)
        @test src2.mechanism == ForceZ
    end

    @testset "ReceiverConfig" begin
        rec = line_receivers(0.0, 1000.0, 51; z=10.0)
        @test length(rec.x) == 51
        @test rec.x[1] == 0.0f0
        @test rec.x[end] == 1000.0f0
        @test all(rec.z .== 10.0f0)
    end

    # =========================================================================
    # Boundary 配置测试
    # =========================================================================
    @testset "Boundary" begin
        b1 = top_image()
        @test b1.top == :image

        b2 = top_absorbing()
        @test b2.top == :absorbing

        b3 = top_vacuum(10)
        @test b3.top == :vacuum
        @test b3.vacuum_layers == 10
    end

    # =========================================================================
    # SimConfig 测试
    # =========================================================================
    @testset "SimConfig" begin
        config = SimConfig(nt=1000, fd_order=8)
        @test config.nt == 1000
        @test config.fd_order == 8
        @test config.dt === nothing
    end

    # =========================================================================
    # 模拟测试 (小网格，快速)
    # =========================================================================
    @testset "Simulate" begin
        # 创建小模型
        nx, nz = 60, 40
        dx = 10.0f0
        vp = fill(2000.0f0, nz, nx)
        vs = fill(1200.0f0, nz, nx)
        rho = fill(2000.0f0, nz, nx)
        model = VelocityModel(vp, vs, rho, dx, dx)

        src = SourceConfig(300.0, 100.0; f0=30.0)
        rec = line_receivers(50.0, 550.0, 11)

        out1 = OutputConfig(base_dir=mktempdir(), plot_gather=true, video_config=nothing)
        gather1 = simulate(model, src, rec, top_image(nbc=10), out1, SimConfig(nt=200);
            backend=CPU_BACKEND, progress=false)
        @test size(gather1) == (200, 11)
        @test !all(gather1 .== 0)
        @test isfile(joinpath(out1.base_dir, "result.jld2"))
        @test isfile(joinpath(out1.base_dir, "gather.png"))

        out2 = OutputConfig(base_dir=mktempdir(), plot_gather=false, video_config=nothing)
        gather2 = simulate(model, src, rec, top_vacuum(5; nbc=10), out2, SimConfig(nt=200);
            backend=CPU_BACKEND, progress=false)
        @test size(gather2) == (200, 11)
        @test !all(gather2 .== 0)

        out3 = OutputConfig(base_dir=mktempdir(), plot_gather=false, video_config=nothing)
        gather3 = simulate(model, src, rec, top_absorbing(nbc=10), out3, SimConfig(nt=200);
            backend=CPU_BACKEND, progress=false)
        @test size(gather3) == (200, 11)
    end

    # =========================================================================
    # Batch 测试
    # =========================================================================
    @testset "Batch" begin
        nx, nz = 60, 40
        dx = 10.0f0
        vp = fill(2000.0f0, nz, nx)
        vs = fill(1200.0f0, nz, nx)
        rho = fill(2000.0f0, nz, nx)
        model = VelocityModel(vp, vs, rho, dx, dx)

        src_template = SourceConfig(0.0, 0.0; f0=30.0)
        rec = line_receivers(50.0, 550.0, 11)

        sim = BatchSimulator(model, src_template, rec, top_image(nbc=10), SimConfig(nt=120); backend=CPU_BACKEND)

        g1 = simulate_shot!(sim, 300.0, 100.0; progress=false)
        @test size(g1) == (120, 11)
        @test !all(g1 .== 0)

        xs = [200.0, 300.0, 400.0]
        zs = [100.0, 100.0, 100.0]
        gathers = simulate_shots!(sim, xs, zs)
        @test length(gathers) == 3
        @test all(size(g) == (120, 11) for g in gathers)

        out = OutputConfig(base_dir=mktempdir(), plot_gather=false, video_config=nothing)
        simulate_shots!(sim, xs, zs; store=false, outputs=out, progress=false)
        @test isfile(joinpath(out.base_dir, "shot_0001.jld2"))
        @test isfile(joinpath(out.base_dir, "shot_0002.jld2"))
        @test isfile(joinpath(out.base_dir, "shot_0003.jld2"))
    end

    # =========================================================================
    # IO 测试
    # =========================================================================
    @testset "IO" begin
        # 创建测试结果
        nx, nz = 40, 30
        model = VelocityModel(
            fill(2000.0f0, nz, nx),
            fill(1200.0f0, nz, nx),
            fill(2000.0f0, nz, nx),
            10.0f0, 10.0f0
        )

        src = SourceConfig(200.0, 50.0; f0=30.0)
        rec = line_receivers(50.0, 350.0, 7)
        out = OutputConfig(base_dir=mktempdir(), plot_gather=false, video_config=nothing)

        gather = simulate(model, src, rec, top_image(nbc=10), out, SimConfig(nt=100);
            backend=CPU_BACKEND, progress=false)
        @test size(gather) == (100, 7)

        result_path = joinpath(out.base_dir, "result.jld2")
        @test isfile(result_path)
        data = JLD2.load(result_path)
        @test size(data["gather"]) == (100, 7)
        @test data["nt"] == 100
    end

end

println("\n✓ 所有测试通过")
