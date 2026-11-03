using Pkg
Pkg.activate(@__DIR__)

using TethysFlow
using Tethys
using CairoMakie
using Statistics

"""
    _bottom_matrix(ds, arr3d::AbstractArray{<:Real,3}) -> Matrix{Float64}

Extract deepest wet cell value for each horizontal coordinate.
Immersed rock cells below the seabed hold 0.0 or NaN in hydrodynamic stores
and do not represent active water parcels.
The deepest wet fluid cell holding real water is identified and returned.
"""
function _bottom_matrix(ds, arr3d::AbstractArray{<:Real,3})
    depths = ds.depths
    bathy = hasproperty(ds, :bathymetry) ? ds.bathymetry : nothing
    nx, ny = size(arr3d, 1), size(arr3d, 2)
    nz = size(arr3d, 3)
    out = fill(NaN, nx, ny)

    # Use temperature as the canonical wet-fluid tracer reference when present
    ref_3d = hasproperty(ds, :temperature) && ds.temperature isa AbstractArray{<:Real,3} &&
             size(ds.temperature) == (nx, ny, nz) ? ds.temperature : arr3d

    for i in 1:nx, j in 1:ny
        k_start = 1
        if bathy isa AbstractMatrix && size(bathy, 1) >= i && size(bathy, 2) >= j
            bval = bathy[i, j]
            if isfinite(bval) && bval < 0.0
                kb = _find_bottom_level_bathy(depths, bval)
                if !isnothing(kb)
                    k_start = kb
                end
            end
        end

        # Scan upward from k_start to find the deepest wet fluid cell
        for k in k_start:nz
            v_ref = Float64(ref_3d[i, j, k])
            if isfinite(v_ref) && v_ref != 0.0
                out[i, j] = Float64(arr3d[i, j, k])
                break
            end
        end
    end
    return out
end

function _find_bottom_level_bathy(depths::AbstractVector{<:Real}, bval::Real)
    for (kk, z) in enumerate(depths)
        if isfinite(z) && z >= bval
            return kk
        end
    end
    return nothing
end

"""
    _load_coastline(store_path::AbstractString) -> Vector{NamedTuple}

Load coastline polygons from the run directory or default Scotian Shelf boundary.
"""
function _load_coastline(store_path::AbstractString)
    candidates = AbstractString[
        joinpath(dirname(store_path), "inputs", "coastline.parquet"),
        joinpath(dirname(store_path), "inputs", "coastline.dat")
    ]
    for c in candidates
        if isfile(c) && endswith(c, ".parquet")
            rings = Tethys.load_coastline_polygons(c)
            !isempty(rings) && return rings
        end
    end
    return Tethys.SCOTIAN_SHELF_COASTLINE
end

function main()
    store_path = normpath(joinpath(@__DIR__, "..", "lakehouse", "TethysFlow",
                                   "scotian_shelf", "hydrodynamics.zarr"))
    isdir(store_path) || error("Hydrodynamic store not found at $(store_path)")

    output_dir = normpath(joinpath(@__DIR__, "..", "lakehouse", "TethysFlow",
                                   "scotian_shelf", "outputs"))
    mkpath(output_dir)

    n_written = TethysFlow.store_interval_count(store_path)
    n_written > 0 || error("No written intervals found in store at $(store_path)")

    println("TethysFlow Scotian Shelf Temperature Assessment")
    println("Store path:        ", store_path)
    println("Output directory:  ", output_dir)
    println("Written intervals: ", n_written)

    coast = _load_coastline(store_path)
    println("Coastline rings:   ", length(coast))

    # Extract latest time slice (Day 245.0)
    latest_idx = n_written
    ds_latest = TethysFlow.extract_hydrodynamic_dataset(store_path; time_index = latest_idx)
    lons = ds_latest.lons
    lats = ds_latest.lats
    depths = ds_latest.depths
    k_surf = ds_latest.depth_index
    t_sec = ds_latest.time_seconds
    t_day = round(t_sec / 86400.0, digits = 1)
    t_hr = round(t_sec / 3600.0, digits = 1)

    println("Latest timestamp:  Day $(t_day) ($(t_hr) hours, index $(latest_idx))")
    println("Domain dimensions: ", length(lons), " x ", length(lats), " x ", length(depths))

    # Surface temperature (mask land cells holding 0.0 as NaN)
    T_surf = Float64.(ds_latest.temperature[:, :, k_surf])
    for idx in eachindex(T_surf)
        if T_surf[idx] == 0.0
            T_surf[idx] = NaN
        end
    end
    # Bottom temperature (deepest wet fluid layer)
    T_bot = _bottom_matrix(ds_latest, Float64.(ds_latest.temperature))
    # Stratification delta T
    delta_T = fill(NaN, size(T_surf))
    for i in 1:size(T_surf, 1), j in 1:size(T_surf, 2)
        if isfinite(T_surf[i, j]) && isfinite(T_bot[i, j])
            delta_T[i, j] = T_surf[i, j] - T_bot[i, j]
        end
    end

    # Determine thermal colormap ranges
    valid_surf = filter(x -> isfinite(x) && x != 0.0, T_surf)
    valid_bot = filter(x -> isfinite(x) && x != 0.0, T_bot)
    valid_dt = filter(isfinite, delta_T)

    cmin_t = min(minimum(valid_surf), minimum(valid_bot))
    cmax_t = max(maximum(valid_surf), maximum(valid_bot))
    dt_max = isempty(valid_dt) ? 10.0 : maximum(valid_dt)
    dt_min = isempty(valid_dt) ? 0.0 : min(0.0, minimum(valid_dt))

    println("Surface T range:   [", round(minimum(valid_surf), digits=2), ", ",
            round(maximum(valid_surf), digits=2), "] °C")
    println("Bottom T range:    [", round(minimum(valid_bot), digits=2), ", ",
            round(maximum(valid_bot), digits=2), "] °C")
    println("Stratification dT: [", round(dt_min, digits=2), ", ",
            round(dt_max, digits=2), "] °C")

    # 1. Surface Temperature Map
    fig_surf = Figure(size = (1200, 750), fontsize = 13)
    ax_s = Axis(fig_surf[1, 1],
                title = "Scotian Shelf Surface Temperature - Day $(t_day) ($(t_hr) h, Step $(latest_idx))",
                xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
                limits = (minimum(lons), maximum(lons), minimum(lats), maximum(lats)))
    hm_s = heatmap!(ax_s, lons, lats, T_surf;
                    colormap = :thermal, colorrange = (cmin_t, cmax_t), interpolate = false)
    for ring in coast
        lines!(ax_s, ring.lons, ring.lats; color = :white, linewidth = 1.3)
    end
    Colorbar(fig_surf[1, 2], hm_s; label = "Temperature (°C)")
    surf_path = joinpath(output_dir, "temperature_surface_day$(round(Int, t_day)).png")
    save(surf_path, fig_surf, px_per_unit = 2)
    println("Saved: ", surf_path)

    # 2. Bottom Temperature Map
    fig_bot = Figure(size = (1200, 750), fontsize = 13)
    ax_b = Axis(fig_bot[1, 1],
                title = "Scotian Shelf Bottom Temperature - Day $(t_day) ($(t_hr) h, Step $(latest_idx))",
                xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
                limits = (minimum(lons), maximum(lons), minimum(lats), maximum(lats)))
    hm_b = heatmap!(ax_b, lons, lats, T_bot;
                    colormap = :thermal, colorrange = (cmin_t, cmax_t), interpolate = false)
    for ring in coast
        lines!(ax_b, ring.lons, ring.lats; color = :white, linewidth = 1.3)
    end
    Colorbar(fig_bot[1, 2], hm_b; label = "Temperature (°C)")
    bot_path = joinpath(output_dir, "temperature_bottom_day$(round(Int, t_day)).png")
    save(bot_path, fig_bot, px_per_unit = 2)
    println("Saved: ", bot_path)

    # 3. Stratification Map (Delta T)
    fig_dt = Figure(size = (1200, 750), fontsize = 13)
    ax_d = Axis(fig_dt[1, 1],
                title = "Scotian Shelf Thermal Stratification (T_surf - T_bottom) - Day $(t_day)",
                xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
                limits = (minimum(lons), maximum(lons), minimum(lats), maximum(lats)))
    hm_d = heatmap!(ax_d, lons, lats, delta_T;
                    colormap = :inferno, colorrange = (0.0, max(1.0, dt_max)), interpolate = false)
    for ring in coast
        lines!(ax_d, ring.lons, ring.lats; color = :white, linewidth = 1.3)
    end
    Colorbar(fig_dt[1, 2], hm_d; label = "Thermal Stratification ΔT (°C)")
    dt_path = joinpath(output_dir, "temperature_stratification_day$(round(Int, t_day)).png")
    save(dt_path, fig_dt, px_per_unit = 2)
    println("Saved: ", dt_path)

    # 4. Multi-panel Synthesis Figure with 238-day Regional Timeseries
    println("Extracting seasonal timeseries across the $(n_written) intervals...")
    step_sample = max(1, round(Int, n_written / 48)) # ~48 temporal checkpoints (~every 5 days)
    sample_indices = collect(1:step_sample:n_written)
    if last(sample_indices) != n_written
        push!(sample_indices, n_written)
    end

    ts_days = Float64[]
    ts_surf_mean = Float64[]
    ts_surf_max = Float64[]
    ts_bot_mean = Float64[]

    for (k_i, s_idx) in enumerate(sample_indices)
        ds_s = TethysFlow.extract_hydrodynamic_dataset(store_path; time_index = s_idx)
        s_day = Float64(ds_s.time_seconds) / 86400.0
        T_s = Float64.(ds_s.temperature[:, :, ds_s.depth_index])
        T_b = _bottom_matrix(ds_s, Float64.(ds_s.temperature))

        fw_s = filter(x -> isfinite(x) && x != 0.0, T_s)
        fw_b = filter(x -> isfinite(x) && x != 0.0, T_b)

        if !isempty(fw_s) && !isempty(fw_b)
            push!(ts_days, s_day)
            push!(ts_surf_mean, mean(fw_s))
            push!(ts_surf_max, maximum(fw_s))
            push!(ts_bot_mean, mean(fw_b))
        end
    end

    fig_synth = Figure(size = (1800, 1100), fontsize = 13)

    # Panel A: Surface Temperature Map
    ax1 = Axis(fig_synth[1, 1],
               title = "A: Surface Temperature (Day $(t_day))",
               xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
               limits = (minimum(lons), maximum(lons), minimum(lats), maximum(lats)))
    hm1 = heatmap!(ax1, lons, lats, T_surf;
                   colormap = :thermal, colorrange = (cmin_t, cmax_t), interpolate = false)
    for ring in coast
        lines!(ax1, ring.lons, ring.lats; color = :white, linewidth = 1.1)
    end
    Colorbar(fig_synth[1, 2], hm1; label = "T (°C)")

    # Panel B: Bottom Temperature Map
    ax2 = Axis(fig_synth[1, 3],
               title = "B: Bottom Temperature (Day $(t_day))",
               xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
               limits = (minimum(lons), maximum(lons), minimum(lats), maximum(lats)))
    hm2 = heatmap!(ax2, lons, lats, T_bot;
                   colormap = :thermal, colorrange = (cmin_t, cmax_t), interpolate = false)
    for ring in coast
        lines!(ax2, ring.lons, ring.lats; color = :white, linewidth = 1.1)
    end
    Colorbar(fig_synth[1, 4], hm2; label = "T (°C)")

    # Panel C: Stratification Map
    ax3 = Axis(fig_synth[2, 1],
               title = "C: Stratification ΔT = T_surf - T_bot (Day $(t_day))",
               xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
               limits = (minimum(lons), maximum(lons), minimum(lats), maximum(lats)))
    hm3 = heatmap!(ax3, lons, lats, delta_T;
                   colormap = :inferno, colorrange = (0.0, max(1.0, dt_max)), interpolate = false)
    for ring in coast
        lines!(ax3, ring.lons, ring.lats; color = :white, linewidth = 1.1)
    end
    Colorbar(fig_synth[2, 2], hm3; label = "ΔT (°C)")

    # Panel D: 238-day Regional Temperature Timeseries
    ax4 = Axis(fig_synth[2, 3:4],
               title = "D: Regional Temperature Evolution (Day 7 to Day $(t_day))",
               xlabel = "Simulation Time (Days)",
               ylabel = "Regional Mean Temperature (°C)")
    lines!(ax4, ts_days, ts_surf_mean; color = :crimson, linewidth = 2.5, label = "Surface Mean T")
    lines!(ax4, ts_days, ts_surf_max; color = :orange, linewidth = 1.5, linestyle = :dash, label = "Surface Max T")
    lines!(ax4, ts_days, ts_bot_mean; color = :royalblue, linewidth = 2.5, label = "Bottom Mean T")
    lines!(ax4, ts_days, ts_surf_mean .- ts_bot_mean; color = :darkmagenta, linewidth = 2.0, linestyle = :dot, label = "Mean Stratification ΔT")
    axislegend(ax4, position = :lt, framevisible = true)

    Label(fig_synth[0, 1:4],
          "Scotian Shelf Hydrodynamic Simulation: Temperature and Stratification Assessment",
          fontsize = 18, font = :bold)

    synth_path = joinpath(output_dir, "scotian_shelf_temperature_assessment_day$(round(Int, t_day)).png")
    save(synth_path, fig_synth, px_per_unit = 2)
    println("Saved synthesis assessment figure: ", synth_path)
    println("Done.")
end

main()

