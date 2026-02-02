using ElasticWave2D.API
using Plots

model = VelocityModel(fill(3000f0, 100, 200), fill(1800f0, 100, 200),
    fill(2200f0, 100, 200), 10.0, 10.0)
source = SourceConfig(1000.0, 200.0; f0=15.0)
receivers = line_receivers(100, 1900, 91; z=50.0)

# 带 MP4 视频
result = simulate(model, source, receivers;
    config=SimConfig(nt=500),
    video=Video(format=:mp4))

# 保存
save_result(result, "test.jld2")

# 加载验证
r2 = load_result("test.jld2")
println("加载成功: ", size(r2.gather))