using Test
using ElasticWave2D
using ElasticWave2D: OutputConfig, ensure_output_dirs, resolve_output_path, write_manifest, ArtifactsManifest, record_artifact!

@testset "Outputs" begin
    tmp = mktempdir()
    outputs = OutputConfig(base_dir=tmp)
    ensure_output_dirs(outputs)

    @test isdir(tmp)

    p = resolve_output_path(outputs, :results, "x.jld2")
    @test p == joinpath(tmp, "x.jld2")

    m = ArtifactsManifest()
    record_artifact!(m, :result, p)
    manifest_path = write_manifest(outputs, m)
    @test isfile(manifest_path)
end
