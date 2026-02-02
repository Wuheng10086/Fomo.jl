# =============================================================================
# ElasticWave2D.jl 测试
# =============================================================================
#
# 运行: julia --project=. test/runtests.jl
#
# =============================================================================

using Test
using ElasticWave2D
using ElasticWave2D.API

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
        b1 = FreeSurface()
        @test b1.top == :image

        b2 = Absorbing()
        @test b2.top == :habc

        b3 = Vacuum(10)
        @test b3.top == :vacuum
        @test b3.vacuum_layers == 10
    end

    # =========================================================================
    # SimConfig 测试
    # =========================================================================
    @testset "SimConfig" begin
        config = SimConfig(nt=1000, fd_order=8, boundary=Vacuum(5))
        @test config.nt == 1000
        @test config.fd_order == 8
        @test config.boundary.top == :vacuum
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

        # FreeSurface
        result1 = simulate(model, src, rec;
            config=SimConfig(nt=200, boundary=FreeSurface()))

        @test size(result1.gather) == (200, 11)
        @test result1.dt > 0
        @test !all(result1.gather .== 0)  # 有数据

        # Vacuum
        result2 = simulate(model, src, rec;
            config=SimConfig(nt=200, boundary=Vacuum(5)))

        @test size(result2.gather) == (200, 11)
        @test !all(result2.gather .== 0)

        # Absorbing
        result3 = simulate(model, src, rec;
            config=SimConfig(nt=200, boundary=Absorbing()))

        @test size(result3.gather) == (200, 11)
    end

    # =========================================================================
    # 多炮测试
    # =========================================================================
    @testset "MultiShot" begin
        nx, nz = 60, 40
        dx = 10.0f0
        vp = fill(2000.0f0, nz, nx)
        vs = fill(1200.0f0, nz, nx)
        rho = fill(2000.0f0, nz, nx)
        model = VelocityModel(vp, vs, rho, dx, dx)

        sources = [
            SourceConfig(200.0, 100.0; f0=30.0),
            SourceConfig(400.0, 100.0; f0=30.0),
        ]
        rec = line_receivers(50.0, 550.0, 11)

        results = simulate(model, sources, rec;
            config=SimConfig(nt=200, boundary=FreeSurface()))

        @test length(results) == 2
        @test size(results[1].gather) == (200, 11)
        @test size(results[2].gather) == (200, 11)
    end

    # =========================================================================
    # BatchSimulator 测试
    # =========================================================================
    @testset "BatchSimulator" begin
        nx, nz = 60, 40
        dx = 10.0f0
        vp = fill(2000.0f0, nz, nx)
        vs = fill(1200.0f0, nz, nx)
        rho = fill(2000.0f0, nz, nx)
        model = VelocityModel(vp, vs, rho, dx, dx)

        rec_x = Float32.(50:50:550)
        rec_z = fill(10.0f0, length(rec_x))

        sim = BatchSimulator(model, rec_x, rec_z; nt=200, f0=30.0f0)

        @test sim.is_initialized == true

        # 单炮
        gather1 = simulate_shot!(sim, 300.0f0, 100.0f0)
        @test size(gather1) == (200, length(rec_x))

        # 多炮
        src_x = Float32[200.0, 300.0, 400.0]
        src_z = Float32[100.0, 100.0, 100.0]
        gathers = simulate_shots!(sim, src_x, src_z)

        @test length(gathers) == 3
        @test all(size(g) == (200, length(rec_x)) for g in gathers)
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

        result = simulate(model,
            SourceConfig(200.0, 50.0; f0=30.0),
            line_receivers(50.0, 350.0, 7);
            config=SimConfig(nt=100, boundary=FreeSurface()))

        # 保存和加载
        tmpfile = tempname() * ".jld2"
        save_result(result, tmpfile)

        loaded = load_result(tmpfile)

        @test size(loaded.gather) == size(result.gather)
        @test loaded.dt == result.dt
        @test loaded.nt == result.nt

        rm(tmpfile)
    end

end

println("\n✓ 所有测试通过")
