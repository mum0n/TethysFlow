"""
    projected_vector_dataset(input_path, output_path, projection; pairs)

Create a Lakehouse-sidecar dataset whose vector pairs are expressed in the model grid
frame. The source file is never modified. Coordinates remain geographic; only vector
components are rotated, which lets downstream interpolation use the original lon/lat axes.

Each pair is `(u_name, v_name, output_u_name, output_v_name)`. If the output names match
the input names, the derived dataset can be consumed by existing readers without changing
the source dataset's variable names.
"""
function projected_vector_dataset(
    input_path::AbstractString,
    output_path::AbstractString,
    projection;
    pairs::Tuple
)
    input = Tethys.load_dataset(input_path)
    output = Tethys.transform_vector_dataset(input, projection; pairs = pairs)
    attrs = copy(output.attrs)
    attrs["derived_from"] = abspath(input_path)
    output = Tethys.GeoDataset(
        output.variables, output.coords, output.dims, output.crs, attrs, nothing,
        String(output_path))
    mkpath(dirname(output_path))
    Tethys.save_dataset(output_path, output; backend = :ncdatasets, overwrite = true)
    return String(output_path)
end

"""Ensure a source or already-projected sidecar is in the requested model frame."""
function ensure_projected_vector_dataset(
    input_path::AbstractString,
    output_path::AbstractString,
    projection;
    pairs::Tuple
)
    if isfile(output_path) || isdir(output_path)
        try
            ds = Tethys.load_dataset(output_path)
            frame = get(ds.attrs, "vector_frame", get(ds.attrs, "TethysFlow_vector_frame", ""))
            signature = get(ds.attrs, "projection_signature",
                            get(ds.attrs, "TethysFlow_projection", ""))
            if frame == "model_grid" && signature == projection_signature(projection) &&
               mtime(output_path) >= mtime(input_path)
                return String(output_path)
            end
        catch
        end
    end
    if normpath(abspath(input_path)) == normpath(abspath(output_path))
        ds = Tethys.load_dataset(input_path)
        frame = get(ds.attrs, "vector_frame", get(ds.attrs, "TethysFlow_vector_frame", ""))
        signature = get(ds.attrs, "projection_signature",
                        get(ds.attrs, "TethysFlow_projection", ""))
        frame == "model_grid" && signature == projection_signature(projection) || error(
            "Projected forcing $(input_path) does not match the configured model projection. " *
            "Re-run data ingestion from the original geographic input; refusing to rotate " *
            "an already-projected file a second time.")
        return String(input_path)
    end
    return projected_vector_dataset(input_path, output_path, projection; pairs = pairs)
end

"""A stable label used to distinguish projected-forcing products by grid frame."""
projection_signature(projection) = Tethys.projection_signature(projection)

"""Resolve the run-local, projected wind sidecar in the Lakehouse inputs directory."""
projected_wind_path(input_dir::AbstractString) = joinpath(input_dir, "wind_active_grid.nc")

"""Resolve the run-local, projected boundary-current sidecar."""
projected_boundary_velocity_path(input_dir::AbstractString) =
    joinpath(input_dir, "boundary_velocities_grid.nc")

"""Coordinate bounds on which a geographic or planar model grid is defined."""
function model_coordinate_ranges(grid)
    base = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    base = architecture(base) isa GPU ? on_architecture(CPU(), base) : base
    if hasproperty(base, :xᶠᵃᵃ) && hasproperty(base, :yᵃᶠᵃ)
        return ((Float64(base.xᶠᵃᵃ[1]), Float64(base.xᶠᵃᵃ[base.Nx + 1])),
                (Float64(base.yᵃᶠᵃ[1]), Float64(base.yᵃᶠᵃ[base.Ny + 1])))
    end
    return ((Float64(base.λᶠᵃᵃ[1]), Float64(base.λᶠᵃᵃ[base.Nx + 1])),
            (Float64(base.φᵃᶠᵃ[1]), Float64(base.φᵃᶠᵃ[base.Ny + 1])))
end
