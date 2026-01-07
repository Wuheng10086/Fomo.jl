# ==============================================================================
# io/output.jl
#
# Output utilities
# ==============================================================================

"""
    save_gather(result::ShotResult, path::String; format=:bin)

Save shot gather to file.

# Formats
- `:bin`: Raw binary
- `:png`: PNG image
"""
function save_gather(result::ShotResult, path::String; format::Symbol=:bin)
    if format == :bin
        _save_bin(result.gather, path)
    elseif format == :png
        _save_png(result.gather, path)
    else
        error("Unknown format: $format")
    end
end

function _save_bin(data::Matrix{Float32}, path::String)
    open(path, "w") do io
        write(io, data)
    end
    @info "Saved gather to $path"
end

function _save_png(data::Matrix{Float32}, path::String)
    nt, n_rec = size(data)
    
    fig = CairoMakie.Figure(size=(800, 600))
    ax = CairoMakie.Axis(fig[1, 1], 
                         title="Shot Gather",
                         xlabel="Receiver",
                         ylabel="Time Sample",
                         yreversed=true)
    
    vmax = maximum(abs.(data)) * 0.5
    vmax = vmax > 0 ? vmax : 1.0f0
    
    CairoMakie.heatmap!(ax, 1:n_rec, 1:nt, data',
                        colormap=:seismic,
                        colorrange=(-vmax, vmax))
    
    CairoMakie.save(path, fig)
    @info "Saved gather image to $path"
end

"""
    load_gather(path::String, nt::Int, n_rec::Int) -> Matrix{Float32}

Load gather from binary file.
"""
function load_gather(path::String, nt::Int, n_rec::Int)
    data = zeros(Float32, nt, n_rec)
    open(path, "r") do io
        read!(io, data)
    end
    return data
end
