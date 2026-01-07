# ==============================================================================
# visualization/setup_check.jl
#
# Plot survey setup for visual verification before running simulation
# ==============================================================================

"""
    plot_setup(model::VelocityModel, x_src, z_src, x_rec, z_rec; 
               output=nothing, title="Survey Setup")

Plot Vp model with source and receiver positions for visual verification.

# Arguments
- `model`: VelocityModel struct  
- `x_src`, `z_src`: Source positions (meters)
- `x_rec`, `z_rec`: Receiver positions (meters)
- `output`: Output file path (default: "\$(model.name)_setup.png")
- `title`: Plot title
"""
function plot_setup(model::VelocityModel, 
                    x_src::AbstractVector, z_src::AbstractVector,
                    x_rec::AbstractVector, z_rec::AbstractVector;
                    output::Union{String, Nothing}=nothing,
                    title::String="Survey Setup Check")
    
    nx, nz = model.nx, model.nz
    
    # Default output filename includes model name
    if output === nothing
        output = "$(model.name)_setup.png"
    end
    
    x_axis = range(0, (nx-1) * model.dx, length=nx)
    z_axis = range(0, (nz-1) * model.dz, length=nz)
    
    # Figure size based on aspect ratio
    x_extent = (nx - 1) * model.dx
    z_extent = (nz - 1) * model.dz
    aspect = x_extent / z_extent
    
    # Reasonable figure size
    fig_width = min(1600, max(800, round(Int, 600 * aspect)))
    fig_height = 700
    
    # Create figure
    fig = Figure(size=(fig_width, fig_height), fontsize=14)
    
    # Main plot - Vp model with geometry
    ax1 = Axis(fig[1, 1],
        xlabel = "X (m)",
        ylabel = "Z (m)",
        title = title,
        aspect = DataAspect()
    )
    
    # Plot Vp model
    # vp is stored as [nz, nx], transpose to [nx, nz] for heatmap
    hm = heatmap!(ax1, x_axis, z_axis, model.vp',
                  colormap = :viridis)
    
    # Plot receivers (triangles)
    scatter!(ax1, x_rec, z_rec,
             marker = :dtriangle,
             markersize = 8,
             color = :blue,
             label = "Receivers ($(length(x_rec)))")
    
    # Plot sources (stars)
    scatter!(ax1, x_src, z_src,
             marker = :star5,
             markersize = 12,
             color = :red,
             strokewidth = 1,
             strokecolor = :white,
             label = "Sources ($(length(x_src)))")
    
    # Reverse y-axis (depth increases downward)
    ax1.yreversed = true
    
    # Colorbar
    Colorbar(fig[1, 2], hm, label = "Vp (m/s)")
    
    # Legend
    axislegend(ax1, position = :rb)
    
    # Info text
    info_text = """
    Model: $(model.name)
    Grid: $(nx) × $(nz) (nx × nz)
    Spacing: dx=$(model.dx)m, dz=$(model.dz)m
    Size: $(round(x_extent/1000, digits=2))km × $(round(z_extent/1000, digits=2))km
    Vp range: $(round(minimum(model.vp), digits=0)) - $(round(maximum(model.vp), digits=0)) m/s
    Sources: $(length(x_src)), Receivers: $(length(x_rec))
    """
    
    Label(fig[2, 1:2], info_text, fontsize=12, halign=:left, padding=(10, 10, 10, 10))
    
    # Save
    save(output, fig)
    @info "Setup check saved" path=output
    
    # Print summary
    println()
    println("=" ^ 60)
    println("  SETUP CHECK")
    println("=" ^ 60)
    println("  Model:     $(model.name)")
    println("  Grid:      $(nx) × $(nz) (nx × nz)")
    println("  X extent:  0 - $(x_extent) m ($(round(x_extent/1000, digits=2)) km)")
    println("  Z extent:  0 - $(z_extent) m ($(round(z_extent/1000, digits=2)) km)")
    println("  Vp range:  $(minimum(model.vp)) - $(maximum(model.vp)) m/s")
    println("-" ^ 60)
    println("  Sources:   $(length(x_src))")
    println("    X range: $(minimum(x_src)) - $(maximum(x_src)) m")
    println("    Z range: $(minimum(z_src)) - $(maximum(z_src)) m")
    println("  Receivers: $(length(x_rec))")
    println("    X range: $(minimum(x_rec)) - $(maximum(x_rec)) m")
    println("    Z range: $(minimum(z_rec)) - $(maximum(z_rec)) m")
    println("=" ^ 60)
    println("  Plot saved to: $output")
    println("=" ^ 60)
    
    return fig
end

"""
    plot_setup(model_path::String, x_src, z_src, x_rec, z_rec; kwargs...)

Load model from file and plot setup.
"""
function plot_setup(model_path::String, 
                    x_src::AbstractVector, z_src::AbstractVector,
                    x_rec::AbstractVector, z_rec::AbstractVector;
                    kwargs...)
    model = load_model(model_path)
    return plot_setup(model, x_src, z_src, x_rec, z_rec; kwargs...)
end
