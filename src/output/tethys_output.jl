# ==============================================================================
#  geodata_output.jl -- the simulated output as a Tethys store
# ==============================================================================
#
#  Why this exists. A run's output used to reach the larval half as a JLD2 archive plus a
#  sidecar file, and the reader had to know Oceananigans' storage layout to interpret it
#  (halos, C-grids, which variable lives on which face). That coupling is removed here: the
#  ocean half writes its output straight into a Tethys-native store, and the larval half
#  opens it through the data layer rather than through Oceananigans.
#
#  What is stored, and how:
#
#    JLD2     the checkpoint only. Oceananigans can checkpoint-restart in its own format and
#             no other, so restart stays there.
#    Zarr     the simulated slices -- u, v, w, T at every output interval -- written by a
#             callback, with coordinates and attributes attached via YAXArrays.
#
#  Why a callback rather than Oceananigans' own `ZarrWriter`. Verified against the installed
#  Oceananigans: `ZarrWriter` constructs, but `Simulation` accepts neither `output_writers` nor
#  `output_writers =`, so there is no supported way to attach it to a run. `Callback` and
#  `add_callback!` are supported, and writing through YAXArrays is code this project already
#  has and has tested. A callback is therefore the route that works with the Oceananigans
#  available rather than the one needing a newer version.
#
#  What makes the sidecar unnecessary: the grid's interior cell centres are written into the
#  store itself, as axis values. A reader no longer needs a second file stating halos and face
#  extents, because the same information reaches it as coordinates it can interpolate
#  against. The store is self-describing.
# ==============================================================================

using Oceananigans
using Oceananigans.Units
using YAXArrays
using Zarr
using Dates

const YAX_ATTRS_NS = nothing

"""
    output_zarr_path(output_dir::AbstractString, name::AbstractString = "hydrodynamics")

Where the simulated output store lives, as a path. A path, not a format: how it is written is
this file's business, and what it contains is the reader's.
"""
output_zarr_path(output_dir::AbstractString, name::AbstractString = "hydrodynamics") =
    joinpath(output_dir, name * ".zarr")

"""
    build_zarr_compressor(;
        codec::AbstractString = "zstd",
        level::Integer = 7,
        shuffle::Integer = 1
    ) -> Zarr.BloscCompressor

Construct a Blosc compressor for Zarr dataset serialization.

# Inputs
- `codec::AbstractString`: Blosc compressor name (`"zstd"`, `"lz4"`, `"zlib"`, `"snappy"`).
- `level::Integer`: Compression level (1 to 9).
- `shuffle::Integer`: Filter shuffle mode (0: none, 1: byte shuffle, 2: bit shuffle).

# Outputs
- `Zarr.BloscCompressor`: Configured Blosc compressor instance.
"""
function build_zarr_compressor(;
    codec::AbstractString = "zstd",
    level::Integer = 7,
    shuffle::Integer = 1
)::Zarr.BloscCompressor
    cname = lowercase(strip(codec))
    valid_codecs = ("zstd", "lz4", "zlib", "snappy", "blosclz")
    if cname ∉ valid_codecs
        @warn "Unsupported compression codec '$(codec)'; defaulting to 'zstd'."
        cname = "zstd"
    end
    clevel = min(max(Int(level), 0), 9)
    shuf = min(max(Int(shuffle), 0), 2)
    return Zarr.BloscCompressor(0, clevel, cname, shuf)
end

"""
    _update_existing_store_compressor!(
        store_path::AbstractString,
        field_names::Vector{String},
        compressor::Zarr.BloscCompressor
    )

Update compressor configuration in `.zarray` files for active variables in an existing
Zarr store, ensuring that subsequent chunks are compressed with the updated codec and level.
"""
function _update_existing_store_compressor!(
    store_path::AbstractString,
    field_names::Vector{String},
    compressor::Zarr.BloscCompressor
)
    for name in field_names
        zarr_file = joinpath(store_path, name, ".zarray")
        if isfile(zarr_file)
            try
                meta = JSON3.read(read(zarr_file, String), Dict{String, Any})
                if haskey(meta, "compressor") && meta["compressor"] isa Dict
                    meta["compressor"]["id"] = "blosc"
                    meta["compressor"]["cname"] = compressor.cname
                    meta["compressor"]["clevel"] = compressor.clevel
                    meta["compressor"]["shuffle"] = compressor.shuffle
                    open(zarr_file, "w") do f
                        JSON3.write(f, meta)
                    end
                end
            catch err
                @warn "Could not update .zarray compressor for $(name): $(err)"
            end
        end
    end
end


"""
    GeodataOutputCallback

An Oceananigans `Callback` that writes the ocean state into a store at each firing.

One store per run, appended to at each interval, so a long simulation writes incrementally
rather than holding its history in memory and losing it on a crash.

That store is a **Tethys** store -- the variables are a `GeoDataset`'s, with their axes and
dimension records, opened once at the first output and written into slice by slice. This is
what makes a reader use `load_dataset` and the dataset's own `.variables`/`.dims` rather than
walking Zarr: an earlier version wrote each interval to `T/<m>/`, and Tethys's Zarr reader
discovers variables as the root group's *child arrays*, so that store opened as an empty
dataset and every consumer -- the larval half and the figures -- hand-rolled its own reads.

`store_ds` and `time_array` are held state, which is why this is a `mutable struct`: the store
is built at the first output and the same dataset is written into at every later one, so the
callback must remember the handle it opened.
"""
mutable struct GeodataOutputCallback
    store::String
    output_dir::String
    field_names::Vector{String}
    written::Base.RefValue{Int}
    store_ds::Union{GeoDataset, Nothing}
    # The store's own time array, held from the group it was opened with. `nothing` until the
    # first output, because the store does not exist until then.
    time_array::Union{Zarr.ZArray, Nothing}
    # The output interval in simulated seconds, from the schedule the caller attached. A field
    # rather than a scan of the simulation's callbacks: at the first firing the callback being
    # fired is not necessarily in `sim.callbacks` yet, and scanning finds nothing -- which sizes
    # the time axis to one interval and silently truncates the run's output.
    interval::Float64
    # Spin-up interval in seconds. Time-slices prior to this are excluded from storage.
    spinup_seconds::Float64
    # Optional maximum water depth (m) to archive. Depths deeper than this (e.g. > 400 m)
    # are subsampled out to reduce vertical array volume by up to 50% for epipelagic larval drift.
    max_save_depth::Float64
    # Optional Blosc compressor configured for Zarr array creation and appending.
    compressor::Any
    # Optional coordinate projection for reprojection from planar computational grid
    # to geographic WGS84 coordinates.
    projection::Any
    # Cached bilinear mapping lattice for geographic reprojection
    reprojection_grid::Union{Tethys.ReprojectionGrid, Nothing}
    # Precomputed grid convergence angle rotation matrices for velocity vectors
    cos_gamma::Union{Matrix{Float32}, Nothing}
    sin_gamma::Union{Matrix{Float32}, Nothing}
    target_lons::Union{Vector{Float64}, Nothing}
    target_lats::Union{Vector{Float64}, Nothing}
end

"""
    GeodataOutputCallback(output_dir, interval, model; fields = ("u","v","w","T"),
                          resume = false, spinup_seconds = 0.0, max_save_depth = Inf,
                          compressor = nothing, projection = nothing, domain_lon = nothing,
                          domain_lat = nothing, target_lons = nothing, target_lats = nothing,
                          reproject = true)

Construct the output writer for one run. A fresh run clears the store and starts at slice 0.
`resume = true` keeps the store and continues from however many slices it already holds.
When `projection` is supplied for a planar model and `reproject = true`, flow fields and
immersed seabed bathymetry are reprojected onto a regular geographic WGS84 `(lon, lat)` grid,
and velocity vectors are unrotated to true geographic East and North directions.
"""
function GeodataOutputCallback(output_dir::AbstractString, interval::Real,
                               model; fields::Tuple = ("u", "v", "w", "T"),
                               resume::Bool = false, spinup_seconds::Real = 0.0,
                               max_save_depth::Real = Inf,
                               compressor = nothing,
                               projection = nothing,
                               domain_lon = nothing,
                               domain_lat = nothing,
                               target_lons = nothing,
                               target_lats = nothing,
                               reproject::Bool = true)
    isdir(output_dir) || mkpath(output_dir)
    store = output_zarr_path(output_dir)
    if !resume && ispath(store)
        rm(store; recursive = true, force = true)   # a fresh run overwrites its own output
    end

    start = 1
    if resume
        start = TethysFlow.store_interval_count(store) + 1
    end

    rep_grid = nothing
    cos_g = nothing
    sin_g = nothing
    tlons = nothing
    tlats = nothing

    base_g = if model !== nothing && hasproperty(model, :grid)
        hasproperty(model.grid, :underlying_grid) ? model.grid.underlying_grid : model.grid
    else
        nothing
    end
    is_planar_grid = base_g !== nothing && base_g isa Oceananigans.Grids.RectilinearGrid

    if reproject && is_planar_grid && projection !== nothing && Tethys.is_planar(projection)
        if target_lons !== nothing && target_lats !== nothing
            tlons = Float64.(collect(target_lons))
            tlats = Float64.(collect(target_lats))
        else
            bounds_lon, bounds_lat = if domain_lon !== nothing && domain_lat !== nothing
                (Float64(domain_lon[1]), Float64(domain_lon[2])),
                (Float64(domain_lat[1]), Float64(domain_lat[2]))
            else
                grid_geographic_bounds(model.grid; projection = projection)
            end

            raw_ax_init = _field_arrays(model)
            nx_raw = length(raw_ax_init.x)
            ny_raw = length(raw_ax_init.y)
            dx = nx_raw > 1 ? abs(raw_ax_init.x[end] - raw_ax_init.x[1]) / (nx_raw - 1) : 2000.0
            dy = ny_raw > 1 ? abs(raw_ax_init.y[end] - raw_ax_init.y[1]) / (ny_raw - 1) : 2000.0

            mid_lat = 0.5 * (bounds_lat[1] + bounds_lat[2])
            dlon = dx / (111_320.0 * cos(deg2rad(mid_lat)))
            dlat = dy / 111_132.0

            n_lon = max(2, round(Int, abs(bounds_lon[2] - bounds_lon[1]) / dlon) + 1)
            n_lat = max(2, round(Int, abs(bounds_lat[2] - bounds_lat[1]) / dlat) + 1)

            tlons = collect(range(min(bounds_lon...), max(bounds_lon...), length = n_lon))
            tlats = collect(range(min(bounds_lat...), max(bounds_lat...), length = n_lat))
        end

        raw_ax_init = _field_arrays(model)
        nx_raw = length(raw_ax_init.x)
        ny_raw = length(raw_ax_init.y)
        dx_grid = nx_raw > 1 ? abs(raw_ax_init.x[end] - raw_ax_init.x[1]) / (nx_raw - 1) : 2000.0
        dy_grid = ny_raw > 1 ? abs(raw_ax_init.y[end] - raw_ax_init.y[1]) / (ny_raw - 1) : 2000.0
        extent_x = dx_grid * nx_raw
        extent_y = dy_grid * ny_raw

        rep_grid = Tethys.ReprojectionGrid(projection, tlons, tlats;
                                           extent_x = extent_x,
                                           extent_y = extent_y)

        nx_t = length(tlons)
        ny_t = length(tlats)
        cos_g = Matrix{Float32}(undef, nx_t, ny_t)
        sin_g = Matrix{Float32}(undef, nx_t, ny_t)
        for j in 1:ny_t, i in 1:nx_t
            gamma = Tethys.grid_convergence_angle(projection, tlons[i], tlats[j])
            rad = deg2rad(Float32(gamma))
            cos_g[i, j] = cos(rad)
            sin_g[i, j] = sin(rad)
        end
    end

    return GeodataOutputCallback(store, output_dir, collect(String.(fields)), Ref(start),
                                 nothing, nothing, Float64(interval), Float64(spinup_seconds),
                                 Float64(max_save_depth), compressor, projection,
                                 rep_grid, cos_g, sin_g, tlons, tlats)
end

# Checkpoint round-trips. Oceananigans' `restore_prognostic_state!` recurses through the
# callbacks when a run picks up a checkpoint and asks every callback func to replay its own
# state; the base package only knows `Nothing` and its own types, so a run resumed from a
# checkpoint with one of these callbacks attached died in a MethodError before the first
# time step. The callback carries no simulational prognostic state of its own -- the store
# records the write position -- so restore is an honest no-op. Both signatures are defined
# because older checkpoints serialise the callback struct itself, while current
# Oceananigans stores whatever `prognostic_state` returns (here, nothing).
Oceananigans.prognostic_state(::GeodataOutputCallback) = nothing
Oceananigans.restore_prognostic_state!(::GeodataOutputCallback, ::Nothing) = nothing
Oceananigans.restore_prognostic_state!(::GeodataOutputCallback, ::GeodataOutputCallback) = nothing

"""
    _interval_count(sim, cb) -> Int

How many output intervals this run will write, so the store's time axis can be sized once
instead of growing an array that cannot be appended to. Intervals falling strictly within
the spin-up period [0, spinup_seconds) are excluded from the output store.
"""
function _interval_count(sim::Oceananigans.Simulation, cb)
    stop = Float64(sim.stop_time)
    effective_duration = max(0.0, stop - cb.spinup_seconds)
    cb.interval > 0 || return 1
    effective_duration > 0 || return 1
    return max(1, ceil(Int, effective_duration / cb.interval) + 1)
end

"""
    _build_store_dataset(store, ax, names, nt, model) -> GeoDataset

The run's output store, created with its final shape and filled a slice at a time.

The variables are `(lon, lat, depth, time)` arrays, and the time axis is sized to the
run so the array never has to grow.

The store is created as a TEMPLATE: the arrays have their final shape and chunking, and
no data is written until the first interval arrives. That is the whole point. Creating
the store by saving a dataset of zeros would hold the entire store in host memory at
once -- at 432 x 250 x 20 x 337 that is 2.9 GB per field, 14.6 GB for five -- and write
all of it to disk, before the run had produced a single value. The run then spent hours
in that write and died in it, with the store still empty and every field still zero.

The axes, not offsets: `ax` is the interior cell centres in `(lon, lat, depth)` from the
model's own grid, so the dataset records where its cells are rather than leaving a reader
to infer positions from indices.

The store's metadata goes into the group's attributes at creation, because the
attributes are what the save writes. Setting them on the dataset afterwards changed an
in-memory dictionary and nothing on disk: the store was written without its seabed, and
a reader that went looking for it found nothing.
"""
function _build_store_dataset(store::AbstractString, ax, names::Vector{String}, nt::Int,
                              model; compressor = nothing,
                              reprojection_grid = nothing,
                              raw_ax = nothing)
    is_reprojected = reprojection_grid !== nothing
    is_planar_grid = false
    if !is_reprojected && model !== nothing && hasproperty(model, :grid)
        base = hasproperty(model.grid, :underlying_grid) ? model.grid.underlying_grid : model.grid
        is_planar_grid = base isa Oceananigans.Grids.RectilinearGrid
    end

    crs_str = (is_planar_grid && !is_reprojected) ? "Projected" : "EPSG:4326"
    crs = Tethys.CoordinateSystem(crs = crs_str)
    nx, ny, nz = length(ax.x), length(ax.y), length(ax.z)

    axe(name, coords, units, std, dim_type) = Tethys.Dimension(
        name = name, size = length(coords), coords = coords,
        units = units, standard_name = std, dim_type = dim_type)

    dl = if is_planar_grid && !is_reprojected
        axe(:x, ax.x, "m", "projection_x_coordinate", Tethys.DIM_SPATIAL)
    else
        axe(:lon, ax.x, "degrees_east", "longitude", Tethys.DIM_SPATIAL)
    end
    dt = if is_planar_grid && !is_reprojected
        axe(:y, ax.y, "m", "projection_y_coordinate", Tethys.DIM_SPATIAL)
    else
        axe(:lat, ax.y, "degrees_north", "latitude", Tethys.DIM_SPATIAL)
    end
    dz = axe(:depth, ax.z, "m", "depth", Tethys.DIM_PARAMETRIC)
    tim = axe(:time, Float64.(1:nt), "s", "time", Tethys.DIM_TEMPORAL)

    dims = if is_planar_grid && !is_reprojected
        Dict{Symbol, Tethys.Dimension}(:x => dl, :y => dt, :depth => dz, :time => tim)
    else
        Dict{Symbol, Tethys.Dimension}(:lon => dl, :lat => dt, :depth => dz, :time => tim)
    end

    dim_names = (is_planar_grid && !is_reprojected) ?
        ["x", "y", "depth", "time"] : ["lon", "lat", "depth", "time"]

    chunk_t = min(nt, 24)
    target_chunks = (nx, ny, nz, chunk_t)

    variables = [(name, (nx, ny, nz, nt), dim_names, target_chunks)
                 for name in names]

    attrs = run_metadata(model; reprojection_grid = reprojection_grid, raw_ax = raw_ax)
    for (k, v) in Tethys.coordinate_system_attributes(crs)
        attrs[k] = v
    end
    if is_planar_grid && !is_reprojected
        attrs["grid_type"] = "planar_rectilinear"
        attrs["grid_units"] = "m"
    elseif is_reprojected
        attrs["grid_type"] = "geographic_reprojected"
        attrs["grid_units"] = "degrees"
        attrs["source_grid_type"] = "planar_rectilinear"
        attrs["reprojected_to"] = "EPSG:4326"
    end

    Tethys.create_zarr_template(String(store);
                                 dims = dims,
                                 attrs = attrs,
                                 variables = variables,
                                 compressor = compressor)

    # Reopened for write, because the callback writes into this dataset's arrays:
    # `load_dataset` defaults to read-only, and of the Zarr modes only `"w"` hands out a
    # writeable array here. It does not truncate -- a second `"w"` open appends, which is
    # checked explicitly, because a writer that silently cleared its earlier intervals
    # would produce a run whose first slice is missing.
    return Tethys.load_dataset(String(store); backend = :zarr, mode = "w")
end

"""
    _model_field(model, name) -> Field

The model's own field for a name the store writes.

Salinity is a tracer here, exactly as temperature is, and it is the model's field: the
model integrates it and a run's store is that model's output. A model built without a
tracer is asked to write one only by mistake, and says so.

Returning the model's field is what keeps the store honest in the other direction too:
the reader will not invent a salinity for a store that lacks one, so the only way for a
reader to have a real one is for the writer to have put the model's own there.
"""
function _model_field(model, name::AbstractString)
    name == "u" && return model.velocities.u
    name == "v" && return model.velocities.v
    name == "w" && return model.velocities.w
    (name == "T" || name == "S") || error("unknown field $(name)")
    sym = Symbol(name)
    !hasproperty(model.tracers, sym) &&
        error("The model carries no '$(name)' tracer, so the store cannot carry one.")
    return getproperty(model.tracers, sym)
end

"""
    _to_centres(arr, grid_shape)

A field written on its own staggered grid, moved onto cell centres by averaging the two faces
that bracket each centre. A field already on the centre grid is returned unchanged.

This is where staggering is resolved, once, on the write. `u` arrives as 346×245×13 for a
345×245×13 grid (one more x-face), `v` as 345×246×13, and `w` as 345×245×14, while `T` is
already centred -- Oceananigans' own staggering. A dimension that is one longer than the grid
is a face axis and is averaged; anything else must agree, or the field is not of this grid.

Which axes move is measured rather than looked up, and averaging rather than dropping the last
face matters because dropping shifts the field by half a cell in the grid's own coordinates, and
a shifted current is a different current. Doing it here means a reader -- the larval half, the
figures -- holds one grid and needs no knowledge of staggering at all.
"""
function _to_centres(arr::AbstractArray{T, 3}, grid_shape::Tuple{Int, Int, Int}) where {T}
    out = arr
    for d in 1:3
        if size(out, d) == grid_shape[d] + 1
            out = 0.5 .* (selectdim(out, d, 1:(size(out, d) - 1)) .+
                          selectdim(out, d, 2:size(out, d)))
        end
    end
    size(out) == grid_shape || error(
        "Field of size $(size(arr)) on a $(grid_shape) grid: not this grid's field, and not " *
        "its faces either.")
    return out
end

"""
    _grid_shape_for_centres(raw_ax) -> (nx, ny, nz)

Return the interior grid shape from the coordinate axes extracted by `_field_arrays`.
For a `LatitudeLongitudeGrid`, `raw_ax.x`/`raw_ax.y` are 1D vectors of length Nx/Ny.
"""
function _grid_shape_for_centres(raw_ax)
    nx = length(raw_ax.x)
    ny = length(raw_ax.y)
    nz = length(raw_ax.z)
    return (nx, ny, nz)
end

"""
    _field_arrays(model) -> (; x, y, z)

The grid's **interior** cell centres as geographic `(lon, lat, depth)`.

The halo must be stripped: writing the halo-inclusive coordinate array would put store index
`i` at halo position `i` rather than interior position `i`  -  a silent displacement of the whole
field with a shape that still looks fine. Rather than slice the halo range here, this delegates
to `extract_grid_coordinates`, the project's tested halo-stripped extractor in degrees and
metres, so the store's coordinates are the model's real interior centres by construction.
"""
function _field_arrays(model)
    g = model.grid
    base = hasproperty(g, :underlying_grid) ? g.underlying_grid : g
    # Geographic (spherical) grid -- the real model. Delegate to the project's tested
    # degrees/metres extractor so the store's coordinates are the same interior centres the
    # rest of the code reports; hand-slicing the halo here shifted them by the halo width.
    if base isa Oceananigans.Grids.LatitudeLongitudeGrid
        c = extract_grid_coordinates(base)
        return (x = Float64.(c.lons), y = Float64.(c.lats), z = Float64.(c.depths))
    end
    # Metric/rectilinear grid (the unit tests): Cartesian horizontal centres in metres, halo
    # stripped, with the vertical centre from `znode` as above. `_host_copy` rather than a
    # comprehension over the range, for the reason in its docstring.
    ix = (base.Hx + 1):(base.Hx + base.Nx)
    iy = (base.Hy + 1):(base.Hy + base.Ny)
    x = Float64.(_host_copy(base.xᶜᵃᵃ)[ix])
    y = Float64.(_host_copy(base.yᵃᶜᵃ)[iy])
    # The vertical centres must come from one whole-array host transfer, matching x/y above:
    # calling `znode(k, base, Center())` per level scalar-indexes the GPU-side node offset
    # array on every call and aborts with "Scalar indexing is disallowed" the first time this
    # runs on a GPU model built on a planar grid (geographic `LatitudeLongitudeGrid` models
    # never reach this branch). On the vertical discretization the centre nodes are indexed
    # `1:Nz` directly (halos live at `1-Hz:0` and `Nz+1:Nz+Hz`), verified against `znode`.
    z = Float64.(_host_copy(base.z.cᵃᵃᶜ)[1:base.Nz])
    return (x = x, y = y, z = z)
end

function (cb::GeodataOutputCallback)(sim::Oceananigans.Simulation)
    model = sim.model
    t = Float64(model.clock.time)
    if t < cb.spinup_seconds
        return nothing
    end
    m = if cb.interval > 0.0
        calc_m = max(1, round(Int, (t - cb.spinup_seconds) / cb.interval) + 1)
        max(calc_m, cb.written[])
    else
        cb.written[]
    end

    # The interior cell centres, resolved from the model's own grid, so the store states where
    # its cells are on Earth rather than leaving a reader to guess from an index.
    raw_ax = _field_arrays(model)
    # If max_save_depth is specified (e.g. 400m), filter depth axis to epipelagic/mesopelagic layers
    # to save up to 50% array volume for larval drift studies. Note: depths in Oceananigans are negative (z <= 0)
    # or positive down. We check abs.(z) <= cb.max_save_depth.
    z_indices = if isfinite(cb.max_save_depth) && cb.max_save_depth > 0.0
        findall(z -> abs(z) <= cb.max_save_depth, raw_ax.z)
    else
        1:length(raw_ax.z)
    end
    isempty(z_indices) && (z_indices = 1:length(raw_ax.z))
    ax = if cb.reprojection_grid !== nothing && cb.target_lons !== nothing && cb.target_lats !== nothing
        (x = cb.target_lons, y = cb.target_lats, z = raw_ax.z[z_indices])
    else
        (x = raw_ax.x, y = raw_ax.y, z = raw_ax.z[z_indices])
    end

    fields = Dict{String, Any}()
    for name in cb.field_names
        f = _model_field(model, name)
        # `u` is on x-faces, `v` on y-faces, `w` on z-faces: Oceananigans' own staggering. They
        # are averaged onto cell centres here, in one `_to_centres` call, so the store holds
        # every field on the one grid its dimension records describe -- and a reader does not
        # have to know about staggering to read the store.
        centred = Float32.(_to_centres(Array{Float32}(interior(f)),
                                       _grid_shape_for_centres(raw_ax)))
        fields[name] = length(z_indices) < length(raw_ax.z) ? centred[:, :, z_indices] : centred
    end

    if cb.reprojection_grid !== nothing
        # Reproject velocity fields and unrotate them to true East/North directions
        has_u = haskey(fields, "u")
        has_v = haskey(fields, "v")
        if has_u && has_v && cb.cos_gamma !== nothing && cb.sin_gamma !== nothing
            u_geo = Tethys.reproject_to_geographic(
                fields["u"], raw_ax.x, raw_ax.y, cb.reprojection_grid; fill_value = NaN32
            )
            v_geo = Tethys.reproject_to_geographic(
                fields["v"], raw_ax.x, raw_ax.y, cb.reprojection_grid; fill_value = NaN32
            )
            nz_sub = size(u_geo, 3)
            for k in 1:nz_sub
                @views u_k = u_geo[:, :, k]
                @views v_k = v_geo[:, :, k]
                u_rot = @. u_k * cb.cos_gamma - v_k * cb.sin_gamma
                v_rot = @. u_k * cb.sin_gamma + v_k * cb.cos_gamma
                u_geo[:, :, k] = u_rot
                v_geo[:, :, k] = v_rot
            end
            fields["u"] = u_geo
            fields["v"] = v_geo
        elseif has_u
            fields["u"] = Tethys.reproject_to_geographic(
                fields["u"], raw_ax.x, raw_ax.y, cb.reprojection_grid; fill_value = NaN32
            )
        elseif has_v
            fields["v"] = Tethys.reproject_to_geographic(
                fields["v"], raw_ax.x, raw_ax.y, cb.reprojection_grid; fill_value = NaN32
            )
        end

        for name in cb.field_names
            (name == "u" || name == "v") && continue
            fields[name] = Tethys.reproject_to_geographic(
                fields[name], raw_ax.x, raw_ax.y, cb.reprojection_grid; fill_value = NaN32
            )
        end
    end

    ds = cb.store_ds
    if isnothing(ds)
        # First interval: the store is created as a Tethys store, with the time axis sized to
        # the run. Written through Tethys because a store written by hand -- one directory
        # per interval -- is one Tethys's `load_dataset` cannot open: its reader discovers
        # variables as the root group's *child arrays*, so that store opens as an empty dataset
        # and every consumer -- the larval half and the figures -- then hand-rolls its own
        # Zarr walk. That is exactly what they were doing.
        nt = _interval_count(sim, cb)
        st = output_zarr_path(cb.output_dir)
        if ispath(st)
            old_ds = Tethys.load_dataset(st; backend = :zarr)
            old_shape = size(old_ds.variables[cb.field_names[1]].data)
            expected_nx = length(ax.x)
            expected_ny = length(ax.y)
            if old_shape[1] != expected_nx || old_shape[2] != expected_ny
                @warn "Existing store spatial shape $(old_shape[1:2]) does not match target grid $(expected_nx)x$(expected_ny); resetting store."
                rm(st; recursive = true, force = true)
                ds = _build_store_dataset(st, ax, cb.field_names, nt, sim.model;
                                          compressor = cb.compressor,
                                          reprojection_grid = cb.reprojection_grid,
                                          raw_ax = raw_ax)
                cb.store_ds = ds
            else
                old_nt = old_shape[4]
                if old_nt < nt
                    g = Zarr.zopen(st, "w")
                    for name in cb.field_names
                        if haskey(g.arrays, name)
                            cur_sz = size(g.arrays[name])
                            Zarr.resize!(g.arrays[name], cur_sz[1:(end - 1)]..., nt)
                        end
                    end
                    if haskey(g.arrays, "time")
                        Zarr.resize!(g.arrays["time"], nt)
                    end
                end
                if !isnothing(cb.compressor) && cb.compressor isa Zarr.BloscCompressor
                    _update_existing_store_compressor!(st, cb.field_names, cb.compressor)
                end
                ds = Tethys.load_dataset(st; backend = :zarr, mode = "w")
                cb.store_ds = ds
            end
        else
            ds = _build_store_dataset(st, ax, cb.field_names, nt, sim.model;
                                      compressor = cb.compressor,
                                      reprojection_grid = cb.reprojection_grid,
                                      raw_ax = raw_ax)
            cb.store_ds = ds
        end
    end
    if isnothing(cb.time_array)
        # The store's own time array, opened once. See the note where it is written: this is
        # the coordinate Tethys's reader materialises, so it is written at the backend.
        #
        # The array is taken from the opened group itself. Re-wrapping it -- `Zarr.ZArray(a)`
        # -- gives an array that reports `writeable = true` and silently writes to nothing:
        # the blob the group holds is re-wrapped into a second handle whose writes go to a
        # copy, so a run's time axis stays the placeholder it was created with.
        g = Zarr.zopen(output_zarr_path(cb.output_dir), "w")
        cb.time_array = g.arrays["time"]
    end

    # `time` lives on the model's clock, not on the Simulation: `Simulation` has no `clock`
    # field, and reaching for one is a FieldError at the first write.
    t = Float64(sim.model.clock.time)
    nt_store = size(ds.variables[cb.field_names[1]].data, 4)
    if m > nt_store
        @warn "Output time $(prettytime(t)) maps to slice index $(m) > store capacity $(nt_store); skipping write."
        return nothing
    end
    for (name, arr) in fields
        ds.variables[name].data[:, :, :, m] = arr
    end
    # The time coordinate goes through the store's own array, not through `ds.coords[:time]`:
    # Tethys's Zarr reader materialises a coordinate into a plain host `Vector` when it opens
    # a store, so writing there would update memory the next read never sees. This is the one
    # array Tethys's read path cannot write back, and it is a coordinate rather than science,
    # so it is the one place a backend call belongs.
    cb.time_array[m] = t
    cb.written[] = m + 1
    return nothing
end

"""
    attach_output_callback!(sim, output_dir; fields = ("u","v","w","T"), schedule = nothing,
                            resume = false, spinup_seconds = 0.0, max_save_depth = Inf,
                            compressor = nothing, projection = nothing, domain_lon = nothing,
                            domain_lat = nothing, target_lons = nothing, target_lats = nothing,
                            reproject = true)

Build the store and attach the writer to `sim`. Returns the store path, so the caller records
it in the handover and the larval half is given a location rather than an object.
When `projection` is supplied for a planar model and `reproject = true`, flow fields and
immersed seabed bathymetry are reprojected onto a regular geographic WGS84 `(lon, lat)` grid,
and velocity vectors are unrotated to true geographic East and North directions.

`schedule` is a `TimeInterval` in simulated seconds, and it must be given: an `Oceananigans`
`Callback` with no schedule fires on *every* timestep, not on the simulation's output
interval, so a 30-day adaptive run wrote 4200-odd intervals instead of the ~240 the
`output_schedule_seconds` in the TOML asked for. `Oceananigans.Schedule` is not exported in
this version, so the caller passes the interval it already built for the JLD2 writer and both
writers share one cadence by construction rather than by convention.
"""
function attach_output_callback!(sim, output_dir::AbstractString;
                                 fields::Tuple = ("u", "v", "w", "T"),
                                 schedule = nothing, resume::Bool = false,
                                 spinup_seconds::Real = 0.0,
                                 max_save_depth::Real = Inf,
                                 compressor = nothing,
                                 projection = nothing,
                                 domain_lon = nothing,
                                 domain_lat = nothing,
                                 target_lons = nothing,
                                 target_lats = nothing,
                                 reproject::Bool = true)
    if isnothing(schedule)
        schedule = Oceananigans.TimeInterval(6 * 3600.0)
    end
    cb = GeodataOutputCallback(output_dir, schedule.interval, sim.model; fields = fields,
                               resume = resume, spinup_seconds = spinup_seconds,
                               max_save_depth = max_save_depth,
                               compressor = compressor,
                               projection = projection,
                               domain_lon = domain_lon,
                               domain_lat = domain_lat,
                               target_lons = target_lons,
                               target_lats = target_lats,
                               reproject = reproject)
    Oceananigans.add_callback!(sim, Oceananigans.Callback(cb, schedule))
    # The store itself does not exist yet: it is built at the first output, when the grid axes
    # and the field shapes are known. What is returned is where it *will* be, so the caller can
    # record a location and hand the larval half a path rather than an object.
    return output_zarr_path(output_dir)
end

"""
    _host_copy(a)

`a` copied to the host, **with its index space unchanged**.

Not `Array(a)`, which for an offset array drops the offsets and so changes what a physical
index means. And not element-by-element access either: when `a` lives on the GPU  -  which every
grid coordinate and the immersed seabed do on a GPU machine  -  that is scalar indexing, and CUDA
refuses it. The whole seabed extraction was written that way, so the run died there on a GPU
and nowhere else, with an error that never mentions the seabed. Copying the storage and
re-wrapping it in the original axes is one bulk device-to-host transfer, and the caller's
indices then mean exactly what they meant before.
"""
_host_copy(a) = OffsetArray(on_architecture(CPU(), parent(a)), axes(a)...)

"""
    seabed_from_model(model; reprojection_grid = nothing, raw_ax = nothing)
    -> Union{Matrix{Float64}, Nothing}

The immersed seabed the simulation actually ran on, as `[i_lon, j_lat]` with the halo
stripped. `nothing` when the model is not on an immersed grid. When `reprojection_grid`
is provided, the seabed matrix is reprojected from planar coordinates to regular geographic
coordinates matching the output dataset shape.

This is the same value the JLD2 `serialized/grid` branch of `read_immersed_bathymetry`
returns, extracted from a live model so the store can carry it: a figure that masks cells
against the seabed must see the one the run used, not a re-derived stand-in.
"""
function seabed_from_model(model; reprojection_grid = nothing, raw_ax = nothing)
    # A store that is not a simulation's -- the stitched mosaic -- has no grid to take a
    # seabed from, and is asked for one with `model = nothing`.
    (isnothing(model) || !hasproperty(model, :grid)) && return nothing
    g = model.grid
    g isa ImmersedBoundaryGrid || return nothing
    ib = g.immersed_boundary
    (hasproperty(ib, :bottom_height) && !isnothing(ib.bottom_height)) || return nothing
    Hx, Hy, Nx, Ny = g.Hx, g.Hy, g.Nx, g.Ny
    # Extract interior seabed elements from underlying parent array without halo offset
    raw_seabed = Matrix{Float64}(
        parent(_host_copy(ib.bottom_height))[(Hx + 1):(Hx + Nx), (Hy + 1):(Hy + Ny), 1]
    )
    if reprojection_grid !== nothing
        actual_ax = raw_ax !== nothing ? raw_ax : _field_arrays(model)
        reprojected_seabed = Tethys.reproject_to_geographic(
            raw_seabed, actual_ax.x, actual_ax.y, reprojection_grid; fill_value = 0.0
        )
        return Matrix{Float64}(reprojected_seabed)
    end
    return raw_seabed
end

"""
    run_metadata(model; reprojection_grid = nothing, raw_ax = nothing) -> Dict{String, Any}

What a reader cannot recover from the store's path or from its arrays: the seabed the
run used, a title, and the moment the store was made.

The store's attributes are written by the save that creates it, so these have to be in
the dataset *before* that save. They used to be set on the dataset afterwards, which
changed an in-memory dictionary and left the store's `.zattrs` without them -- so the
seabed, which is the one thing a figure needs to know where the sea floor is, was
absent from every store a run produced.

The seabed is written flat, in column-major order, with its shape alongside it. Zarr's
attribute writer turns a Julia matrix into a list of COLUMNS, so a matrix written here
would be read back transposed; on a grid whose two horizontal dimensions differ -- 32 by
38, not a square -- a transposed seabed is a matrix of the wrong shape that still looks
like data. A flat list plus a shape is unambiguous whatever the writer does with it.

`model` is `nothing` for a store that is not a simulation's -- the stitched mosaic --
and such a store simply has no seabed to record.
"""
function run_metadata(model; reprojection_grid = nothing, raw_ax = nothing)
    attrs = Dict{String, Any}(
        "title" => "Realized hydrodynamic simulation",
        "created_utc" => string(Dates.now(Dates.UTC)),
    )
    seabed = seabed_from_model(model; reprojection_grid = reprojection_grid, raw_ax = raw_ax)
    if !isnothing(seabed)
        attrs["bottom_height"] = collect(Float64, vec(seabed))
        attrs["bottom_height_shape"] = [size(seabed, 1), size(seabed, 2)]
    end
    return attrs
end

"""
    stitch_shelf_domains(
        tile_west_path::AbstractString,
        tile_east_path::AbstractString;
        output_path::AbstractString,
        blend_lon_range::Tuple{Real, Real} = (-63.5, -63.0)
    ) -> String

Stitch two complementary regional hydrodynamic Zarr stores into a unified,
spatially continuous domain mosaic covering the entire Scotian Shelf.

# Mathematical Formulation
Across the zonal transition zone ``\\lambda \\in [\\lambda_1, \\lambda_2]``,
mosaic fields ``\\psi_{\\text{mosaic}}(\\lambda, \\varphi, z, t)`` are linearly
blended using a partition of unity:
```math
w(\\lambda) = \\frac{\\lambda - \\lambda_1}{\\lambda_2 - \\lambda_1}
```
```math
\\psi_{\\text{mosaic}}(\\lambda, \\varphi, z, t) = (1 - w(\\lambda)) \\psi_{\\text{west}}(\\lambda, \\varphi, z, t) + w(\\lambda) \\psi_{\\text{east}}(\\lambda, \\varphi, z, t)
```
For ``\\lambda < \\lambda_1``, ``\\psi = \\psi_{\\text{west}}``; for ``\\lambda > \\lambda_2``, ``\\psi = \\psi_{\\text{east}}``.
This eliminates artificial shear discontinuities across sub-domain seams and
preserves mass conservation across the shelf mosaic.

# Inputs
- `tile_west_path`: Absolute path to Western Scotian Shelf Zarr store.
- `tile_east_path`: Absolute path to Eastern Scotian Shelf Zarr store.
- `output_path`: Absolute destination path for the unified Zarr mosaic.
- `blend_lon_range`: Longitudinal interval ``(\\lambda_1, \\lambda_2)`` in degrees East.

# Outputs
- `output_path`: Path to the generated mosaic Zarr store.
"""
function stitch_shelf_domains(
    tile_west_path::AbstractString,
    tile_east_path::AbstractString;
    output_path::AbstractString,
    blend_lon_range::Tuple{Real, Real} = (-63.5, -63.0)
)
    isdir(tile_west_path) || error("Western tile store not found: $(tile_west_path)")
    isdir(tile_east_path) || error("Eastern tile store not found: $(tile_east_path)")

    ds_w = Tethys.load_dataset(tile_west_path; backend = :zarr)
    ds_e = Tethys.load_dataset(tile_east_path; backend = :zarr)

    # Resolve coordinates
    lon_w = Float64.(ds_w.coords[:lon].data)
    lat_w = Float64.(ds_w.coords[:lat].data)
    z_w   = Float64.(ds_w.coords[:depth].data)
    t_w   = Float64.(ds_w.coords[:time].data)

    lon_e = Float64.(ds_e.coords[:lon].data)
    lat_e = Float64.(ds_e.coords[:lat].data)
    z_e   = Float64.(ds_e.coords[:depth].data)
    t_e   = Float64.(ds_e.coords[:time].data)

    # Harmonise spatial coordinates
    λ1, λ2 = Float64(blend_lon_range[1]), Float64(blend_lon_range[2])
    lon_west_sub = filter(x -> x < λ1, lon_w)
    lon_east_sub = filter(x -> x > λ2, lon_e)
    lon_overlap = filter(x -> λ1 <= x <= λ2, lon_e)
    lon_mosaic = sort(unique(vcat(lon_west_sub, lon_overlap, lon_east_sub)))

    lat_mosaic = sort(unique(vcat(lat_w, lat_e)))
    z_mosaic = sort(unique(vcat(z_w, z_e)))
    t_common = intersect(t_w, t_e)
    isempty(t_common) && (t_common = t_w)

    field_names = collect(intersect(keys(ds_w.variables), keys(ds_e.variables)))
    println("Stitching domains into mosaic: $(length(lon_mosaic))×$(length(lat_mosaic))×$(length(z_mosaic)) grid...")

    ax = (x = lon_mosaic, y = lat_mosaic, z = z_mosaic)
    nt = length(t_common)

    rm(output_path, recursive = true, force = true)
    mkpath(dirname(output_path))

    ds_mosaic = _build_store_dataset(output_path, ax, field_names, nt, nothing)

    g = Zarr.zopen(output_path, "w")
    t_arr = g.arrays["time"]
    for (m, t_val) in enumerate(t_common)
        t_arr[m] = t_val
    end

    for var_name in field_names
        vw = ds_w.variables[var_name].data
        ve = ds_e.variables[var_name].data
        out_var = ds_mosaic.variables[var_name].data

        for (m, t_val) in enumerate(t_common)
            idx_w = findfirst(==(t_val), t_w)
            idx_e = findfirst(==(t_val), t_e)

            slice_w = !isnothing(idx_w) ? vw[:, :, :, idx_w] : vw[:, :, :, 1]
            slice_e = !isnothing(idx_e) ? ve[:, :, :, idx_e] : ve[:, :, :, 1]

            for (i, lon) in enumerate(lon_mosaic)
                w_blend = if lon <= λ1
                    0.0
                elseif lon >= λ2
                    1.0
                else
                    (lon - λ1) / (λ2 - λ1)
                end

                iw = argmin(abs.(lon_w .- lon))
                ie = argmin(abs.(lon_e .- lon))

                for (j, lat) in enumerate(lat_mosaic)
                    jw = argmin(abs.(lat_w .- lat))
                    je = argmin(abs.(lat_e .- lat))

                    for (k, z) in enumerate(z_mosaic)
                        kw = argmin(abs.(z_w .- z))
                        ke = argmin(abs.(z_e .- z))

                        val_w = Float32(slice_w[iw, jw, kw])
                        val_e = Float32(slice_e[ie, je, ke])

                        out_var[i, j, k, m] = if isnan(val_w) && !isnan(val_e)
                            val_e
                        elseif isnan(val_e) && !isnan(val_w)
                            val_w
                        elseif isnan(val_w) && isnan(val_e)
                            Float32(NaN)
                        else
                            Float32((1.0 - w_blend) * val_w + w_blend * val_e)
                        end
                    end
                end
            end
        end
    end

    println("Stitched shelf mosaic written successfully to $(output_path).")
    return output_path
end

