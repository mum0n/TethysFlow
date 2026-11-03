"""
    grid_bathymetry.jl

Grid construction and immersed boundary setup for shelf hydrodynamic modeling.

TethysFlow-specific functions requiring Oceananigans grid types.
All generic geospatial operations (bathymetry I/O, regridding, coastline, marine cell
extraction, smoothing, buffering) are in Tethys.core.bathy. 
"""

using Oceananigans
using Oceananigans.Grids: Face, Center, znode

using Oceananigans.Architectures: architecture, on_architecture, CPU, GPU
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Tethys
using Random

"""
    build_shelf_grid(;
        architecture = :cpu,
        lon_range=(-68.0, -57.0),
        lat_range=(42.0, 47.0),
        z_range=(-1000.0, 0.0),
        z_faces = nothing,
        grid_size=(50, 50, 10),
        halo = (7, 7, 5),
        topology=(Bounded, Bounded, Bounded),
        fallback_to_cpu::Bool = false
    )

Construct a regional ocean model computational grid on the specified architecture.
The grid is a geographic spherical `LatitudeLongitudeGrid` aligned with lines of
longitude and latitude.

# Inputs
- `architecture`: Target device backend (`:cpu` or `:gpu`).
- `lon_range`: Zonal geographic bounds `(lon_min, lon_max)` in °E.
- `lat_range`: Meridional geographic bounds `(lat_min, lat_max)` in °N.
- `z_range`: Depth bounds `(z_min, z_max)` in meters.
- `z_faces`: Optional custom vector of vertical face coordinates.
- `grid_size`: Discretization tuple `(Nx, Ny, Nz)`.
- `halo`: Boundary ghost cell halo tuple `(Hx, Hy, Hz)`.
- `topology`: Spatial topology tuple (default `(Bounded, Bounded, Bounded)`).
- `fallback_to_cpu`: Whether to fall back to CPU if GPU initialization fails.

# Outputs
- `Oceananigans.Grids.AbstractGrid`: Constructed base computational grid.
"""
function build_shelf_grid(;
    architecture = :cpu,
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    z_range::Tuple{Real, Real} = (-3500.0, 0.0),
    z_faces::Union{Nothing, AbstractVector, Function} = nothing,
    grid_size::Tuple{Int, Int, Int} = (50, 50, 10),
    halo::Tuple{Int, Int, Int} = (7, 7, 5),
    topology::Tuple = (Bounded, Bounded, Bounded),
    fallback_to_cpu::Bool = false,
    projection = "geographic",
    projection_zone::Integer = 0,
    projection_lat1::Real = 0.0,
    projection_lat2::Real = 0.0,
    rotation_deg::Real = 0.0,
    lon_center::Union{Nothing, Real} = nothing,
    lat_center::Union{Nothing, Real} = nothing,
    extent_x_km::Union{Nothing, Real} = nothing,
    extent_y_km::Union{Nothing, Real} = nothing,
    extent_x::Union{Nothing, Real} = nothing,
    extent_y::Union{Nothing, Real} = nothing
)
    if lon_range[1] >= lon_range[2]
        error("Invalid longitude range: $(lon_range). lon_min must be < lon_max.")
    end
    if lat_range[1] >= lat_range[2]
        error("Invalid latitude range: $(lat_range). lat_min must be < lat_max.")
    end
    if any(s <= 0 for s in grid_size)
        error("Grid size dimensions must all be positive integers: $(grid_size)")
    end

    z_specification = if !isnothing(z_faces)
        if z_faces isa AbstractVector
            if length(z_faces) != grid_size[3] + 1
                error("Custom z_faces length $(length(z_faces)) must equal Nz + 1 = $(grid_size[3] + 1)")
            end
            if !issorted(z_faces)
                error("Custom z_faces must be strictly monotonically increasing.")
            end
        end
        z_faces
    else
        if z_range[1] >= z_range[2]
            error("Invalid vertical range: $(z_range). z_min must be < z_max.")
        end
        z_range
    end

    arch = resolve_architecture(architecture; fallback_to_cpu = fallback_to_cpu)

    eff_lx = if extent_x !== nothing && extent_x > 0.0
        Float64(extent_x)
    elseif extent_x_km !== nothing && extent_x_km > 0.0
        Float64(extent_x_km) * 1000.0
    else
        nothing
    end

    eff_ly = if extent_y !== nothing && extent_y > 0.0
        Float64(extent_y)
    elseif extent_y_km !== nothing && extent_y_km > 0.0
        Float64(extent_y_km) * 1000.0
    else
        nothing
    end

    proj_obj = if projection isa AbstractProjection
        projection
    else
        build_projection(
            projection = string(projection),
            zone = Int(projection_zone),
            lat_1 = projection_lat1 == 0.0 ? nothing : Float64(projection_lat1),
            lat_2 = projection_lat2 == 0.0 ? nothing : Float64(projection_lat2),
            lon_center = lon_center === nothing ? (lon_range[1] + lon_range[2]) / 2.0 : Float64(lon_center),
            lat_center = lat_center === nothing ? (lat_range[1] + lat_range[2]) / 2.0 : Float64(lat_center),
            rotation_deg = Float64(rotation_deg),
            extent_x = eff_lx === nothing ? 0.0 : eff_lx,
            extent_y = eff_ly === nothing ? 0.0 : eff_ly
        )
    end

    if !is_planar(proj_obj) && rotation_deg == 0.0
        return LatitudeLongitudeGrid(
            arch;
            size = grid_size,
            longitude = lon_range,
            latitude = lat_range,
            z = z_specification,
            topology = topology,
            halo = halo
        )
    end

    # Planar projected or rotated computational frame
    lx = if eff_lx !== nothing && eff_lx > 0.0
        eff_lx
    else
        c1 = project_forward(proj_obj, lon_range[1], lat_range[1])
        c2 = project_forward(proj_obj, lon_range[2], lat_range[1])
        c3 = project_forward(proj_obj, lon_range[2], lat_range[2])
        c4 = project_forward(proj_obj, lon_range[1], lat_range[2])
        maximum((c1[1], c2[1], c3[1], c4[1])) - minimum((c1[1], c2[1], c3[1], c4[1]))
    end

    ly = if eff_ly !== nothing && eff_ly > 0.0
        eff_ly
    else
        c1 = project_forward(proj_obj, lon_range[1], lat_range[1])
        c2 = project_forward(proj_obj, lon_range[2], lat_range[1])
        c3 = project_forward(proj_obj, lon_range[2], lat_range[2])
        c4 = project_forward(proj_obj, lon_range[1], lat_range[2])
        maximum((c1[2], c2[2], c3[2], c4[2])) - minimum((c1[2], c2[2], c3[2], c4[2]))
    end

    grid = RectilinearGrid(
        arch;
        size = grid_size,
        x = (-lx / 2.0, lx / 2.0),
        y = (-ly / 2.0, ly / 2.0),
        z = z_specification,
        topology = topology,
        halo = halo
    )
    return grid
end

"""
    grid_geographic_bounds(grid; projection = nothing) -> (Tuple{Float64, Float64}, Tuple{Float64, Float64})

Compute the geographic bounding box `((lon_min, lon_max), (lat_min, lat_max))` in
degrees East and North enclosing `grid`'s interior domain.
"""
function grid_geographic_bounds(grid; projection = nothing)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    base_g = architecture(base_g) isa GPU ? on_architecture(CPU(), base_g) : base_g
    if base_g isa LatitudeLongitudeGrid
        lon_range = (Float64(base_g.λᶠᵃᵃ[1]), Float64(base_g.λᶠᵃᵃ[base_g.Nx + 1]))
        lat_range = (Float64(base_g.φᵃᶠᵃ[1]), Float64(base_g.φᵃᶠᵃ[base_g.Ny + 1]))
        return lon_range, lat_range
    end
    if projection !== nothing
        x_min, x_max = Float64(base_g.xᶠᵃᵃ[1]), Float64(base_g.xᶠᵃᵃ[base_g.Nx + 1])
        y_min, y_max = Float64(base_g.yᵃᶠᵃ[1]), Float64(base_g.yᵃᶠᵃ[base_g.Ny + 1])
        c1 = project_inverse(projection, x_min, y_min)
        c2 = project_inverse(projection, x_max, y_min)
        c3 = project_inverse(projection, x_max, y_max)
        c4 = project_inverse(projection, x_min, y_max)
        lons = (c1[1], c2[1], c3[1], c4[1])
        lats = (c1[2], c2[2], c3[2], c4[2])
        return (minimum(lons) - 0.2, maximum(lons) + 0.2), (minimum(lats) - 0.2, maximum(lats) + 0.2)
    end
    error("RectilinearGrid requires projection to determine geographic bounds.")
end

"""
    load_vertical_grid_csv(filepath::AbstractString) -> Vector{Float64}

Load vertical layer face coordinates from a CSV file (e.g. `scotian_shelf_vertical_grid.csv`).
"""
function load_vertical_grid_csv(filepath::AbstractString)::Vector{Float64}
    if !isfile(filepath)
        error("Vertical grid file does not exist: $(filepath)")
    end
    lines = readlines(filepath)
    if length(lines) < 2
        error("Vertical grid file $(filepath) is empty or missing data.")
    end
    faces = Float64[]
    for l in lines[2:end]
        parts = split(strip(l), ',')
        if length(parts) >= 3
            bot_val = parse(Float64, parts[2])
            top_val = parse(Float64, parts[3])
            if isempty(faces)
                push!(faces, bot_val)
            end
            push!(faces, top_val)
        end
    end
    if !issorted(faces)
        if issorted(faces; rev = true)
            error(
                "Vertical grid file $(filepath) is written positive-downward (first face " *
                "$(first(faces)) m, last $(last(faces)) m). Oceananigans requires z faces " *
                "NEGATIVE and ascending, ending at 0 m at the surface: rewrite so the first " *
                "row is the deepest layer and the last row's top is 0.0.")
        end
        error("Parsed vertical grid faces from $(filepath) are not monotonically sorted.")
    end
    abs(last(faces)) <= 1e-6 || error(
        "Vertical grid faces from $(filepath) must end at 0.0 m (the surface); got " *
        "$(last(faces)) m.")
    first(faces) < 0 || error(
        "Vertical grid faces from $(filepath) must be negative below the surface; got " *
        "$(first(faces)) m as the first (deepest) face.")
    return faces
end

"""
    build_immersed_grid(
        grid::LatitudeLongitudeGrid,
        bathymetry::Union{AbstractMatrix, GeoDataset};
        varname::AbstractString = "elevation"
    )

Wrap a base `LatitudeLongitudeGrid` with an `ImmersedBoundaryGrid` using
`GridFittedBottom` representing ocean seafloor topography.

`bathymetry` is either the `(Nx, Ny)` topography matrix itself, or a `GeoDataset` it is
read from through `Tethys.bathymetry_arrays`, which selects the elevation variable
and resolves the `(lon, lat)` orientation from the dimensions the dataset declared. A file
path is deliberately not accepted: which file a source resolves to, and in what format, is
decided inside `Tethys`, and a caller that passes a path is making that decision
itself.
"""
function build_immersed_grid(
    grid,
    bathymetry::Union{AbstractMatrix, GeoDataset};
    varname::AbstractString = "elevation"
)
    topo_matrix::Matrix{Float64} = if bathymetry isa GeoDataset
        elev_var = haskey(bathymetry.variables, varname) ? varname :
                   first(Tethys.variables_like(bathymetry, ["elevation", "altitude", "z", "topo"]))
        elev_ga = bathymetry.variables[elev_var]
        perm = Tethys.dim_permutation(elev_ga.dims, :lon, :lat)
        raw = Array{Float64}(elev_ga.data)
        ndims(raw) == 1 ? reshape(raw, :, 1) : permutedims(raw, perm)
    else
        Matrix{Float64}(bathymetry)
    end

    nx, ny, _ = size(grid)
    t_nx, t_ny = size(topo_matrix)

    if (nx != t_nx) || (ny != t_ny)
        error(
            "Bathymetry dimensions ($(t_nx), $(t_ny)) do not match grid horizontal " *
            "dimensions ($(nx), $(ny))."
        )
    end

    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    arch = architecture(grid)
    z_max = arch isa GPU ? 0.0 : znode(base_g.Nz + 1, base_g, Face())
    max_topo = maximum(topo_matrix)
    if max_topo > z_max
        @warn "Maximum bathymetry elevation ($(max_topo) m) exceeds surface " *
              "height ($(z_max) m). Emerged land points present."
    end

    arch = architecture(grid)
    arch_topo = on_architecture(arch, topo_matrix)
    immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(arch_topo))
    return immersed_grid
end

"""
    is_in_bay_of_fundy(lon::Real, lat::Real) -> Bool

Determine whether geographic coordinates `(lon, lat)` in degrees East and North fall
within the hypertidal Bay of Fundy, Minas Basin, or Chignecto Bay exclusion zone.
"""
@inline function is_in_bay_of_fundy(lon::Real, lat::Real)::Bool
    if lat < 44.4 || lat > 46.0
        return false
    end
    if lon < -67.3 || lon > -64.0
        return false
    end
    if lat <= 44.8 && lon > -65.6
        return false
    end
    if lat <= 45.1 && lon > -65.2
        return false
    end
    if lat >= 45.7 && lon > -64.2
        return false
    end
    return true
end

"""
    _grid_interior_centres(grid; projection = nothing) -> (Vector/Matrix{Float64}, Vector/Matrix{Float64})

Longitude and latitude of `grid`'s interior cell centres.
For a `LatitudeLongitudeGrid`, returns 1D vectors `(lons, lats)`.
For a `RectilinearGrid` with `projection !== nothing`, returns 2D matrices `(lon_mat, lat_mat)`.
"""
function _grid_interior_centres(grid; projection = nothing)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    cpu_g = architecture(base_g) isa GPU ? on_architecture(CPU(), base_g) : base_g
    if cpu_g isa LatitudeLongitudeGrid
        return ([Float64(cpu_g.λᶜᵃᵃ[i]) for i in 1:cpu_g.Nx],
                [Float64(cpu_g.φᵃᶜᵃ[j]) for j in 1:cpu_g.Ny])
    end
    if projection !== nothing
        nx, ny = cpu_g.Nx, cpu_g.Ny
        xs = [Float64(cpu_g.xᶜᵃᵃ[i]) for i in 1:nx]
        ys = [Float64(cpu_g.yᵃᶜᵃ[j]) for j in 1:ny]
        lon_mat = Matrix{Float64}(undef, nx, ny)
        lat_mat = Matrix{Float64}(undef, nx, ny)
        for j in 1:ny, i in 1:nx
            lon_mat[i, j], lat_mat[i, j] = project_inverse(projection, xs[i], ys[j])
        end
        return (lon_mat, lat_mat)
    end
    return ([Float64(cpu_g.xᶜᵃᵃ[i]) for i in 1:cpu_g.Nx],
            [Float64(cpu_g.yᵃᶜᵃ[j]) for j in 1:cpu_g.Ny])
end

"""
    _fit_bathymetry_to_grid(
        grid,
        elevation::AbstractMatrix,
        lon::AbstractVector,
        lat::AbstractVector;
        projection = nothing,
        min_water_depth::Real = 10.0,
        max_water_depth::Real = 0.0,
        mask_bay_of_fundy::Bool = false
    ) -> Matrix{Float64}

Regrid `(elevation, lon, lat)` onto `grid`'s interior cell centres and condition the result
for use as a `GridFittedBottom`.
"""
function _fit_bathymetry_to_grid(
    grid,
    elevation::AbstractMatrix,
    lon::AbstractVector,
    lat::AbstractVector;
    projection = nothing,
    min_water_depth::Real = 10.0,
    max_water_depth::Real = 0.0,
    mask_bay_of_fundy::Bool = false,
    smoothing_passes::Integer = 4
)
    nx, ny, _ = size(grid)
    target_lons, target_lats = _grid_interior_centres(grid; projection = projection)

    regridded_topo = if target_lons isa AbstractMatrix && target_lats isa AbstractMatrix
        out = Matrix{Float64}(undef, nx, ny)
        lons_clean = Float64.(collect(lon))
        lats_clean = Float64.(collect(lat))
        elev_clean = Array{Float64}(elevation)
        if size(elev_clean) == (length(lats_clean), length(lons_clean))
            elev_clean = permutedims(elev_clean, (2, 1))
        end
        if !issorted(lons_clean)
            p = sortperm(lons_clean)
            lons_clean = lons_clean[p]
            elev_clean = elev_clean[p, :]
        end
        if !issorted(lats_clean)
            p = sortperm(lats_clean)
            lats_clean = lats_clean[p]
            elev_clean = elev_clean[:, p]
        end
        itp = Interpolations.interpolate(
            (lons_clean, lats_clean),
            elev_clean,
            Interpolations.Gridded(Interpolations.Linear())
        )
        itp_flat = Interpolations.extrapolate(itp, Interpolations.Flat())
        for j in 1:ny, i in 1:nx
            out[i, j] = itp_flat(target_lons[i, j], target_lats[i, j])
        end
        out
    else
        regrid_2d_field(lon, lat, elevation, target_lons, target_lats)
    end

    if mask_bay_of_fundy
        n_masked = 0
        for j in 1:ny, i in 1:nx
            cell_lon = target_lons isa AbstractMatrix ? target_lons[i, j] : target_lons[i]
            cell_lat = target_lats isa AbstractMatrix ? target_lats[i, j] : target_lats[j]
            if is_in_bay_of_fundy(cell_lon, cell_lat)
                regridded_topo[i, j] = 0.0
                n_masked += 1
            end
        end
        @info "Bay of Fundy masking active: set $(n_masked) cells to solid land (0.0 m)."
    end

    # Condition bathymetry
    h_floor = Float64(min_water_depth)
    h_ceiling = Float64(max_water_depth)
    conditioned_topo = smooth_bathymetry(
        regridded_topo,
        passes = Int(smoothing_passes),
        alpha = 0.5,
        h_min = h_floor,
        h_max = h_ceiling
    )

    if mask_bay_of_fundy
        for j in 1:ny, i in 1:nx
            cell_lon = target_lons isa AbstractMatrix ? target_lons[i, j] : target_lons[i]
            cell_lat = target_lats isa AbstractMatrix ? target_lats[i, j] : target_lats[j]
            if is_in_bay_of_fundy(cell_lon, cell_lat)
                conditioned_topo[i, j] = 0.0
            end
        end
    end

    return conditioned_topo
end

"""
    build_immersed_grid_from_real_data(
        grid;
        projection = nothing,
        source = nothing,
        tier = nothing,
        varname::AbstractString = "elevation",
        output_path::Union{Nothing, AbstractString} = nothing,
        force::Bool = false,
        min_water_depth::Real = 10.0,
        max_water_depth::Real = 0.0,
        mask_bay_of_fundy::Bool = false,
        kwargs...
    ) -> ImmersedBoundaryGrid

Construct an `ImmersedBoundaryGrid` by interpolating real-world bathymetry onto the target
model grid coordinates.
"""
function build_immersed_grid_from_real_data(
    grid;
    projection = nothing,
    source = nothing,
    tier = nothing,
    varname::AbstractString = "elevation",
    output_path::Union{Nothing, AbstractString} = nothing,
    force::Bool = false,
    min_water_depth::Real = 10.0,
    max_water_depth::Real = 0.0,
    mask_bay_of_fundy::Bool = false,
    smoothing_passes::Integer = 4,
    kwargs...
)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    lon_range, lat_range = grid_geographic_bounds(base_g; projection = projection)

    bathy_elevation, bathy_lon, bathy_lat = if !force && !isnothing(output_path) && (isfile(output_path) || isdir(output_path))
        ds = Tethys.load_dataset(output_path)
        elev_var = haskey(ds.variables, varname) ? varname :
                   first(Tethys.variables_like(ds, ["elevation", "altitude", "z", "topo"]))
        elev_ga = ds.variables[elev_var]
        perm = Tethys.dim_permutation(elev_ga.dims, :lon, :lat)
        lon_c, _ = Tethys.axis(ds, :lon)
        lat_c, _ = Tethys.axis(ds, :lat)
        raw = Array{Float64}(elev_ga.data)
        elev = ndims(raw) == 1 ? reshape(raw, :, 1) : permutedims(raw, perm)
        (elev, lon_c, lat_c)
    else
        bathy = Tethys.bathymetry_field(;
            source = source,
            lon_range = lon_range,
            lat_range = lat_range,
            tier = tier,
            varname = varname,
            output_path = output_path,
            force = force,
            kwargs...
        )
        (bathy.elevation, bathy.lon, bathy.lat)
    end

    conditioned_topo = _fit_bathymetry_to_grid(
        base_g, bathy_elevation, bathy_lon, bathy_lat;
        projection = projection,
        min_water_depth = min_water_depth,
        max_water_depth = max_water_depth,
        mask_bay_of_fundy = mask_bay_of_fundy,
        smoothing_passes = smoothing_passes
    )

    return build_immersed_grid(grid, conditioned_topo)
end

"""
    extract_grid_coordinates(grid; projection = nothing) -> NamedTuple

Extract cell-center spatial coordinates `(lons, lats, depths)` from an
Oceananigans computational grid.
"""
function extract_grid_coordinates(grid; projection = nothing)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    cpu_g = architecture(base_g) isa GPU ? on_architecture(CPU(), base_g) : base_g
    nx, ny, nz = cpu_g.Nx, cpu_g.Ny, cpu_g.Nz

    lons, lats = _grid_interior_centres(cpu_g; projection = projection)

    depths = [Float64(Oceananigans.Grids.znode(k, cpu_g, Oceananigans.Grids.Center()))
              for k in 1:nz]

    x_coords = if cpu_g isa RectilinearGrid
        ix = (cpu_g.Hx + 1):(cpu_g.Hx + cpu_g.Nx)
        Float64.(_host_copy(cpu_g.xᶜᵃᵃ)[ix])
    else
        lons
    end
    y_coords = if cpu_g isa RectilinearGrid
        iy = (cpu_g.Hy + 1):(cpu_g.Hy + cpu_g.Ny)
        Float64.(_host_copy(cpu_g.yᵃᶜᵃ)[iy])
    else
        lats
    end

    return (lons = lons, lats = lats, depths = depths, x = x_coords, y = y_coords)
end

"""
    get_strata_buffered_envelope(
        polygons::AbstractVector{<:NamedTuple};
        buffer_km::Real = 100.0
    ) -> NamedTuple

Compute the collective bounding envelope across a set of administrative stratum polygons
(e.g., loaded CFAs) expanded by a user-defined buffer distance (default 100.0 km).
"""
function get_strata_buffered_envelope(
    polygons::AbstractVector{<:NamedTuple};
    buffer_km::Real = 100.0
)
    if isempty(polygons)
        error("Cannot compute buffered envelope for empty polygon list.")
    end

    all_lons = Float64[]
    all_lats = Float64[]

    for poly in polygons
        append!(all_lons, poly.lons)
        append!(all_lats, poly.lats)
    end

    raw_lon = extrema(all_lons)
    raw_lat = extrema(all_lats)

    buf_lon, buf_lat = expand_domain_with_buffer(raw_lon, raw_lat, buffer_km = buffer_km)
    dlon, dlat = buffer_distance_to_degrees(buffer_km, 0.5 * (raw_lat[1] + raw_lat[2]))

    return (
        lon_range = buf_lon,
        lat_range = buf_lat,
        raw_lon_range = raw_lon,
        raw_lat_range = raw_lat,
        buffer_km = Float64(buffer_km),
        dlon = dlon,
        dlat = dlat
    )
end
