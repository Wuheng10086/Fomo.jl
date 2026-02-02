using ElasticWave2D.API

# 模型
model = VelocityModel(
    fill(3000f0, 100, 200),
    fill(1800f0, 100, 200),
    fill(2200f0, 100, 200),
    10.0, 10.0
)

receivers = line_receivers(100, 1900, 91; z=50.0)

# ============ 测试 1：三种边界 ============
println("=== 测试边界条件 ===")

# 镜像法自由表面（默认）
r1 = simulate(model, SourceConfig(1000.0, 200.0; f0=15.0), receivers;
    config=SimConfig(nt=500, boundary=FreeSurface()))
println("FreeSurface: ✓")

# 全吸收
r2 = simulate(model, SourceConfig(1000.0, 200.0; f0=15.0), receivers;
    config=SimConfig(nt=500, boundary=Absorbing()))
println("Absorbing: ✓")

# 真空法
r3 = simulate(model, SourceConfig(1000.0, 500.0; f0=15.0), receivers;
    config=SimConfig(nt=500, boundary=Vacuum(10)))
println("Vacuum: ✓")

# ============ 测试 2：三种震源 ============
println("\n=== 测试震源类型 ===")

# 爆炸源
r4 = simulate(model, SourceConfig(1000.0, 200.0; f0=15.0, type=Explosion), receivers;
    config=SimConfig(nt=500))
println("Explosion: ✓")

# 垂直力
r5 = simulate(model, SourceConfig(1000.0, 200.0; f0=15.0, type=ForceZ), receivers;
    config=SimConfig(nt=500))
println("ForceZ: ✓")

# 水平力
r6 = simulate(model, SourceConfig(1000.0, 200.0; f0=15.0, type=ForceX), receivers;
    config=SimConfig(nt=500))
println("ForceX: ✓")

# ============ 测试 3：记录类型 ============
println("\n=== 测试记录类型 ===")

r7 = simulate(model, SourceConfig(1000.0, 200.0; f0=15.0),
    line_receivers(100, 1900, 91; z=50.0, record=Pressure);
    config=SimConfig(nt=500))
println("Pressure: ✓")

println("\n全部测试通过！")