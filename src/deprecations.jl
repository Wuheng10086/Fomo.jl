# ==============================================================================
# src/deprecations.jl
#
# 旧 API 废弃警告
# 在主模块 end 之前 include 此文件
# ==============================================================================

# ==============================================================================
# 废弃警告消息
# ==============================================================================

const DEPRECATION_MSG = """

╔══════════════════════════════════════════════════════════════════════════════╗
║                           API 升级通知                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  此函数已废弃，将在未来版本移除。请迁移到新 API:                               ║
║                                                                              ║
║      using ElasticWave2D.API                                                 ║
║      result = simulate(model, source, receivers; config)                     ║
║                                                                              ║
║  详见迁移指南: https://github.com/xxx/ElasticWave2D.jl#migration             ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""

const MIGRATION_GUIDE = """
旧 API → 新 API 对照:
─────────────────────────────────────────────────────────────────────────
旧: simulate!(model, src_x, src_z, rec_x, rec_z; config=SimulationConfig(...))
新: simulate(model, SourceConfig(src_x, src_z; f0=15.0), line_receivers(...); config=SimConfig(...))

旧: config.f0 = 15.0
新: SourceConfig(x, z, Ricker(15.0), type)

旧: config.source_type = :force_z
新: SourceConfig(x, z; type=ForceZ)

旧: config.free_surface = true
新: SimConfig(boundary=FreeSurface())

旧: seismic_survey(..., surface_method=:vacuum)
新: SimConfig(boundary=Vacuum(10))
─────────────────────────────────────────────────────────────────────────
"""

# ==============================================================================
# 警告触发器（只警告一次）
# ==============================================================================

const _WARNED = Ref(false)

function _deprecation_warning(func_name::String)
    if !_WARNED[]
        _WARNED[] = true
        @warn """
        $(func_name) 已废弃，请使用新 API:
        
            using ElasticWave2D.API
            result = simulate(model, SourceConfig(...), line_receivers(...))
        
        运行 ElasticWave2D.migration_guide() 查看完整迁移指南。
        """
    end
end

"""
    migration_guide()

显示从旧 API 迁移到新 API 的指南。
"""
function migration_guide()
    println(MIGRATION_GUIDE)
end

# ==============================================================================
# 包装旧函数（添加警告）
# ==============================================================================

# 保存原始函数引用
if isdefined(@__MODULE__, :simulate!)
    const _original_simulate! = simulate!
    
    function simulate!(args...; kwargs...)
        _deprecation_warning("simulate!")
        _original_simulate!(args...; kwargs...)
    end
end

if isdefined(@__MODULE__, :seismic_survey)
    const _original_seismic_survey = seismic_survey
    
    function seismic_survey(args...; kwargs...)
        _deprecation_warning("seismic_survey")
        _original_seismic_survey(args...; kwargs...)
    end
end

if isdefined(@__MODULE__, :simulate_irregular!)
    const _original_simulate_irregular! = simulate_irregular!
    
    function simulate_irregular!(args...; kwargs...)
        _deprecation_warning("simulate_irregular!")
        _original_simulate_irregular!(args...; kwargs...)
    end
end

# ==============================================================================
# 导出
# ==============================================================================

export migration_guide
