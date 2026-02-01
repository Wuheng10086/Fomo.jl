# ==============================================================================
# visualization/plots.jl
#
# Static plotting functions for ElasticWave2D.jl
# - Survey setup visualization
# - Irregular surface plots
# - Shot gather with surface
# ==============================================================================

# CairoMakie is imported at module level

# ==============================================================================
# Basic Survey Setup
# ==============================================================================

"""
    plot_setup(model::VelocityModel, x_src, z_src, x_rec, z_rec; 
               output=nothing, title="Survey Setup")

Plot Vp model with source and receiver positions for visual verification.

# Arguments
- `model`: VelocityModel struct  
- `x_src`, `z_src`: Source positions (meters)
- `x_rec`, `z_rec`: Receiver positions (meters)
- `output`: Output file path (default: "<model_name>_setup.png")
- `title`: Plot title
"""
function plot_setup(model::VelocityModel,
    x_src::AbstractVector, z_src::AbstractVector,
    x_rec::AbstractVector, z_rec::AbstractVector;
    output::Union{String,Nothing}=nothing,
    title::String="Survey Setup Check")

    nx, nz = model.nx, model.nz

    if output === nothing
        output = "$(model.name)_setup.png"
    end

    x_axis = range(0, (nx - 1) * model.dx, length=nx)
    z_axis = range(0, (nz - 1) * model.dz, length=nz)

    x_extent = (nx - 1) * model.dx
    z_extent = (nz - 1) * model.dz
    aspect = x_extent / z_extent

    fig_width = min(1600, max(800, round(Int, 600 * aspect)))
    fig_height = 700

    fig = Figure(size=(fig_width, fig_height), fontsize=14)

    ax1 = Axis(fig[1, 1],
        xlabel="X (m)",
        ylabel="Z (m)",
        title=title,
        aspect=DataAspect()
    )

    hm = heatmap!(ax1, x_axis, z_axis, model.vp', colormap=:viridis)

    scatter!(ax1, x_rec, z_rec,
        marker=:dtriangle, markersize=8, color=:blue,
        label="Receivers ($(length(x_rec)))")

    scatter!(ax1, x_src, z_src,
        marker=:star5, markersize=12, color=:red,
        strokewidth=1, strokecolor=:white,
        label="Sources ($(length(x_src)))")

    ax1.yreversed = true
    Colorbar(fig[1, 2], hm, label="Vp (m/s)")
    axislegend(ax1, position=:rb)

    info_text = """
    Model: $(model.name)
    Grid: $(nx) × $(nz) (nx × nz)
    Spacing: dx=$(model.dx)m, dz=$(model.dz)m
    Size: $(round(x_extent/1000, digits=2))km × $(round(z_extent/1000, digits=2))km
    Vp range: $(round(minimum(model.vp), digits=0)) - $(round(maximum(model.vp), digits=0)) m/s
    Sources: $(length(x_src)), Receivers: $(length(x_rec))
    """

    Label(fig[2, 1:2], info_text, fontsize=12, halign=:left, padding=(10, 10, 10, 10))

    save(output, fig)
    @info "Setup check saved" path = output

    _print_setup_summary(model, x_src, z_src, x_rec, z_rec, output)

    return fig
end

function plot_setup(model_path::String,
    x_src::AbstractVector, z_src::AbstractVector,
    x_rec::AbstractVector, z_rec::AbstractVector;
    kwargs...)
    model = load_model(model_path)
    return plot_setup(model, x_src, z_src, x_rec, z_rec; kwargs...)
end

function _print_setup_summary(model, x_src, z_src, x_rec, z_rec, output)
    x_extent = (model.nx - 1) * model.dx
    z_extent = (model.nz - 1) * model.dz

    println()
    println("="^60)
    println("  SETUP CHECK")
    println("="^60)
    println("  Model:     $(model.name)")
    println("  Grid:      $(model.nx) × $(model.nz) (nx × nz)")
    println("  X extent:  0 - $(x_extent) m")
    println("  Z extent:  0 - $(z_extent) m")
    println("-"^60)
    println("  Sources:   $(length(x_src))")
    println("  Receivers: $(length(x_rec))")
    println("="^60)
    println("  Plot saved to: $output")
    println("="^60)
end

# ==============================================================================
# Simple Shot Gather Plot
# ==============================================================================

"""
    plot_gather(gather, rec_x, dt; 
                title="Shot Gather", output_path=nothing, 
                clip_percentile=0.99, scale=1.0)

Plot seismic shot gather with proper time and distance axes.

# Arguments
- `gather::Matrix{Float32}`: [nt × nrec] seismic gather
- `rec_x::Vector{<:Real}`: Receiver x-coordinates (for x-axis labels)
- `dt::Real`: Time sampling interval [s]
- `title::String`: Plot title
- `output_path::Union{String,Nothing}`: Output file path (optional)
- `clip_percentile::Real`: Amplitude clipping percentile (0.99 = 99th percentile)
- `scale::Real`: Amplitude scaling factor
"""
function plot_gather(gather::Matrix{Float32},
    rec_x::Vector{<:Real},
    dt::Real;
    title::String="Shot Gather",
    output_path::Union{String,Nothing}=nothing,
    clip_percentile::Real=0.99,
    scale::Real=1.0)

    nt, n_rec = size(gather)
    @assert length(rec_x) == n_rec "Receiver coordinate length mismatch"

    # Time axis
    t_axis = Float32.((0:nt-1) .* dt)

    # Create figure
    fig = Figure(size=(1000, 600), fontsize=14)

    ax = Axis(fig[1, 1],
        xlabel="Distance (m)",
        ylabel="Time (s)",
        title=title,
        yreversed=true)

    # Amplitude scaling
    vmax = quantile(abs.(gather[:]), clip_percentile)
    if vmax ≈ 0.0
        scaled_gather = gather
    else
        scaled_gather = gather .* scale ./ vmax
    end

    # Plot gather
    hm = heatmap!(ax, Float32.(rec_x), t_axis, scaled_gather',
        colormap=:seismic)

    # Colorbar
    cb = Colorbar(fig[1, 2], hm, label="Amplitude")

    # Save if output path provided
    if output_path !== nothing
        save(output_path, fig)
        @info "Gather plot saved to $output_path"
    end

    return fig
end

# ==============================================================================
# Irregular Surface Plots
# ==============================================================================

"""
    plot_irregular_setup(model, z_surface, src_x, src_depth, rec_x, rec_depth;
                         output=nothing, title="Irregular Surface Setup")
"""
function plot_irregular_setup(model::VelocityModel,
    z_surface::Vector{<:Real},
    src_x::Vector{<:Real}, src_depth::Vector{<:Real},
    rec_x::Vector{<:Real}, rec_depth::Vector{<:Real};
    output::Union{String,Nothing}=nothing,
    title::String="Irregular Surface Setup",
    show_velocity::Bool=true)

    dx, dz = model.dx, model.dz
    nx, nz = model.nx, model.nz

    x_coords = Float32.((0:nx-1) .* dx)
    z_coords = Float32.((0:nz-1) .* dz)

    fig = Figure(size=(1200, 800))

    if show_velocity
        ax = Axis(fig[1, 1],
            xlabel="X (m)", ylabel="Z (m)",
            title=title,
            yreversed=true,
            aspect=DataAspect())

        hm = heatmap!(ax, x_coords, z_coords, model.vp',
            colormap=:viridis, alpha=0.7)
        Colorbar(fig[1, 2], hm, label="Vp (m/s)")
    else
        ax = Axis(fig[1, 1],
            xlabel="X (m)", ylabel="Z (m)",
            title=title,
            yreversed=true,
            aspect=DataAspect())
    end

    # Plot surface
    lines!(ax, x_coords[1:length(z_surface)], Float32.(z_surface),
        color=:brown, linewidth=3, label="Free Surface")

    # Compute absolute positions
    src_z_abs = _compute_abs_z(src_x, src_depth, z_surface, dx)
    rec_z_abs = _compute_abs_z(rec_x, rec_depth, z_surface, dx)

    # Plot sources and receivers
    scatter!(ax, Float32.(src_x), src_z_abs,
        marker=:star5, markersize=20, color=:red, label="Sources")
    scatter!(ax, Float32.(rec_x), rec_z_abs,
        marker=:dtriangle, markersize=8, color=:blue, label="Receivers")

    axislegend(ax, position=:rt)

    if output !== nothing
        save(output, fig, px_per_unit=2)
        @info "Setup plot saved to $output"
    end

    return fig
end

function _compute_abs_z(x_positions, depths, z_surface, dx)
    z_abs = Float32[]
    for i in 1:length(x_positions)
        idx = clamp(round(Int, x_positions[i] / dx) + 1, 1, length(z_surface))
        push!(z_abs, z_surface[idx] + depths[i])
    end
    return z_abs
end

"""
    plot_surface_comparison(z_flat, z_irregular, dx; output=nothing)
"""
function plot_surface_comparison(z_flat::Vector{<:Real},
    z_irregular::Vector{<:Real},
    dx::Real;
    output::Union{String,Nothing}=nothing)

    nx = length(z_irregular)
    x_coords = Float32.((0:nx-1) .* dx)

    fig = Figure(size=(1000, 400))
    ax = Axis(fig[1, 1],
        xlabel="X (m)", ylabel="Z (m)",
        title="Surface Comparison",
        yreversed=true)

    lines!(ax, x_coords, Float32.(z_flat),
        color=:blue, linewidth=2, label="Flat Surface", linestyle=:dash)
    lines!(ax, x_coords, Float32.(z_irregular),
        color=:red, linewidth=2, label="Irregular Surface")

    amp = maximum(z_irregular) - minimum(z_irregular)
    text!(ax, x_coords[end] * 0.7, minimum(z_irregular) * 0.5,
        text="Amplitude: $(round(amp, digits=1)) m",
        fontsize=14)

    axislegend(ax, position=:rb)

    if output !== nothing
        save(output, fig, px_per_unit=2)
    end

    return fig
end

# ==============================================================================
# Shot Gather with Surface
# ==============================================================================

"""
    plot_gather_with_surface(gather, dt, rec_x, z_surface, dx; 
                             output=nothing, title="Shot Gather", clip=0.99)
"""
function plot_gather_with_surface(gather::Matrix{Float32},
    dt::Real,
    rec_x::Vector{<:Real},
    z_surface::Vector{<:Real},
    dx::Real;
    output::Union{String,Nothing}=nothing,
    title::String="Shot Gather",
    clip::Real=0.99)

    nt, nrec = size(gather)
    t = Float32.((0:nt-1) .* dt)

    # Surface at receiver positions
    rec_z_surf = Float32[]
    for rx in rec_x
        idx = clamp(round(Int, rx / dx) + 1, 1, length(z_surface))
        push!(rec_z_surf, z_surface[idx])
    end

    fig = Figure(size=(1000, 800))

    # Top: Surface
    ax_top = Axis(fig[1, 1],
        ylabel="Z (m)",
        title="Surface Elevation",
        yreversed=true,
        height=100)

    x_full = Float32.((0:length(z_surface)-1) .* dx)
    lines!(ax_top, x_full, Float32.(z_surface), color=:brown, linewidth=2)
    scatter!(ax_top, Float32.(rec_x), rec_z_surf,
        marker=:dtriangle, markersize=6, color=:blue)

    # Bottom: Gather
    ax_bot = Axis(fig[2, 1],
        xlabel="Receiver X (m)",
        ylabel="Time (s)",
        title=title)

    vmax = quantile(abs.(gather[:]), clip)

    hm = heatmap!(ax_bot, Float32.(rec_x), t, gather,
        colormap=:seismic, colorrange=(-vmax, vmax))

    Colorbar(fig[2, 2], hm, label="Amplitude")
    linkxaxes!(ax_top, ax_bot)

    if output !== nothing
        save(output, fig, px_per_unit=2)
    end

    return fig
end