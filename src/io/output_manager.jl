# ==============================================================================
# io/outputs.jl
#
# Helper functions for saving and plotting simulation outputs.
# ==============================================================================

function _save_outputs(gather, dt, config, recorder, video_config)
    video_file = nothing
    gather_file = nothing
    gather_plot = nothing

    n_rec = size(gather, 2)

    if config.save_gather
        gather_file = joinpath(config.output_dir, "gather.bin")
        open(gather_file, "w") do io
            write(io, Int32(config.nt))
            write(io, Int32(n_rec))
            write(io, gather)
        end
        @info "Saved" file = gather_file
    end

    if recorder !== nothing && video_config !== nothing
        # Generate video for each recorded field
        for field in video_config.fields
            video_file = joinpath(config.output_dir, "wavefield_$(field).mp4")
            generate_video(recorder.recorder, video_file;
                fps=video_config.fps, colormap=video_config.colormap)
            @info "Saved" file = video_file
        end
    end

    if config.plot_gather
        gather_plot = joinpath(config.output_dir, "gather.png")
        _plot_gather_simple(gather, dt, gather_plot)
        @info "Saved" file = gather_plot
    end

    return video_file, gather_file, gather_plot
end

function _plot_gather_simple(gather::Matrix{Float32}, dt::Float32, output::String)
    nt, n_rec = size(gather)
    t_axis = (0:nt-1) .* dt

    fig = CairoMakie.Figure(size=(900, 700))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel="Receiver", ylabel="Time (s)", title="Shot Gather")

    # 全局归一化
    gmax = maximum(abs.(gather))
    data = gmax > 1e-30 ? gather ./ gmax : gather

    hm = CairoMakie.heatmap!(ax, 1:n_rec, t_axis, data',
        colormap=:seismic, colorrange=(-0.5, 0.5))
    ax.yreversed = true
    CairoMakie.Colorbar(fig[1, 2], hm, label="Normalized Amplitude")
    CairoMakie.save(output, fig)
end
