# ==============================================================================
# test/test_new_api.jl
#
# 新 API 测试
# ==============================================================================

using Test
using ElasticWave2D

@testset "New API" begin

    @testset "Wavelet" begin
        w = Ricker(15.0)
        @test w.f0 == 15.0f0
        @test w.delay ≈ 1.0f0 / 15.0f0

        w = Ricker(20.0, 0.08)
        @test w.delay == 0.08f0

        data = generate(w, 0.001f0, 500)
        @test length(data) == 500
        @test eltype(data) == Float32

        cw = CustomWavelet([1.0, 2.0, 3.0])
        @test generate(cw, 0.001f0, 2) == Float32[1, 2]
        @test generate(cw, 0.001f0, 5) == Float32[1, 2, 3, 0, 0]
    end

    @testset "SourceConfig" begin
        src = SourceConfig(500.0, 50.0; f0=15.0)
        @test src.x == 500.0f0
        @test src.z == 50.0f0
        @test src.mechanism == Explosion

        src = SourceConfig(500.0, 50.0; f0=20.0, type=ForceZ)
        @test src.mechanism == ForceZ

        src = SourceConfig(500.0, 50.0, Ricker(25.0), StressTxx)
        @test src.wavelet.f0 == 25.0f0
        @test src.mechanism == StressTxx
    end

    @testset "ReceiverConfig" begin
        rec = ReceiverConfig([100, 200, 300], [10, 10, 10])
        @test length(rec.x) == 3
        @test rec.record == Vz

        rec = ReceiverConfig([100, 200], [10, 20], Pressure)
        @test rec.record == Pressure

        rec = line_receivers(0, 1000, 11)
        @test length(rec.x) == 11
        @test rec.x[1] == 0.0f0
        @test rec.x[end] == 1000.0f0

        rec = line_receivers(0, 1000, 11; z=20.0, record=Pressure)
        @test all(rec.z .== 20.0f0)
        @test rec.record == Pressure

        @test_throws Exception ReceiverConfig([1, 2, 3], [1, 2])
    end

    @testset "Boundary" begin
        bc = FreeSurface()
        @test bc.top == :image
        @test bc.nbc == 50

        bc = Absorbing()
        @test bc.top == :habc

        bc = Vacuum(15)
        @test bc.top == :vacuum
        @test bc.vacuum_layers == 15

        bc = FreeSurface(nbc=80)
        @test bc.nbc == 80

        @test_throws Exception Boundary(:invalid, 50, 0)

        bc = FreeSurface(nbc=50)
        pad = compute_padding(bc, 8)
        @test pad.top == 4
        @test pad.bottom == 54

        bc = Vacuum(10, nbc=50)
        pad = compute_padding(bc, 8)
        @test pad.top == 14
    end

    @testset "SimConfig" begin
        config = SimConfig()
        @test config.nt == 3000
        @test config.fd_order == 8
        @test config.boundary.top == :image

        config = SimConfig(nt=5000, boundary=Vacuum(10))
        @test config.nt == 5000
        @test config.boundary.top == :vacuum

        @test_throws Exception SimConfig(nt=-1)
        @test_throws Exception SimConfig(fd_order=7)
    end

    @testset "VideoSettings" begin
        video = Video()
        @test video.snapshots.fields == [:vz]
        @test video.snapshots.interval == 50

        video = Video(fields=[:vz, :vx], interval=20)
        @test :vx in video.snapshots.fields
        @test video.snapshots.interval == 20

        @test_throws Exception Video(fields=[:invalid])
    end

    @testset "SimResult" begin
        gather = rand(Float32, 100, 10)
        src = SourceConfig(500.0, 50.0; f0=15.0)
        rec = line_receivers(0, 1000, 10)

        result = SimResult(gather, 0.001f0, 100, src, rec)

        @test n_receivers(result) == 10
        @test length(times(result)) == 100
        @test trace(result, 1) == gather[:, 1]
    end

    @testset "Integration" begin
        nz, nx = 50, 80
        model = VelocityModel(
            fill(3000.0f0, nz, nx),
            fill(1800.0f0, nz, nx),
            fill(2200.0f0, nz, nx),
            10.0f0, 10.0f0
        )

        source = SourceConfig(400.0, 100.0; f0=15.0)
        receivers = line_receivers(100, 700, 7; z=50.0)
        config = SimConfig(nt=50)

        result = simulate(model, source, receivers; config=config)

        @test result isa SimResult
        @test size(result.gather) == (50, 7)
        @test all(isfinite.(result.gather))
    end
end