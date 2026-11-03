"""
    TethysFlow.jl

Hydrodynamic simulation engine for coastal and shelf seas, integrated with Tethys.jl.
"""
module TethysFlow
 
using
    Random,
    CairoMakie,
    CUDA,
    Adapt,
    NCDatasets,
    Downloads,
    DataFrames,
    Dates,
    Statistics,
    LinearAlgebra,
    TOML,
    JLD2,
    TaylorSeries,
    Interpolations,
    Distributions,
    Tethys,
    OffsetArrays,
    Zarr

# The provenance registry, as a submodule so its names do not have to be re-exported one by
# one. This is the registry the run reports through.
include("data/manifest.jl")
using .Manifest

using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans
using JSON3

# ============================================================================
# Source layout, in dependency order. The figure families are included by
# `output/visualization.jl` itself, so they are NOT listed here: a double include
# redefines every method on the same line, which Julia refuses during precompilation.
# ============================================================================
include("config/configuration.jl")
include("data/open_data.jl")
include("utils/architecture.jl")
include("utils/tides.jl")
include("geometry/projection.jl")
include("data/projected_forcing.jl")
include("data/grid_bathymetry.jl")
include("data/vertical_grid.jl")
include("model/hydrodynamic_model.jl")
include("model/tidal_forcing.jl")
include("model/simulation.jl")
include("model/simulation_result.jl")
include("output/tethys_output.jl")
include("output/visualization.jl")
include("data/tethys_integration.jl")
# The ocean-half run segments, as library code. Included last so every name its
# bodies reach for is already defined; it adds the run driver the CLI dispatches to.
include("driver.jl")
include("assimilation.jl")
include("data_assimilation.jl")

# Coordinate Reference Systems & Projections (re-exported from Tethys / Tethys)
export
    AbstractProjection,
    GeographicProjection,
    CoordRefSystemsProjection,
    UTMProjection,
    LambertProjection,
    MercatorProjection,
    RotatedFrame,
    ReprojectionGrid,
    is_planar,
    resolve_crs,
    build_projection,
    project_forward,
    project_inverse,
    grid_convergence_angle,
    geographic_to_rotated,
    rotated_to_geographic,
    rotate_vector,
    unrotate_vector,
    rotate_vector_to_grid,
    rotate_vector_to_geo,
    reproject_to_geographic,
    optimal_shelf_rotation_angle

# Ocean Grid & Bathymetry
export
    build_shelf_grid,
    build_immersed_grid_from_real_data,
    build_hydrodynamic_model,
    extract_grid_coordinates

# Coriolis & Planar Dynamics
export
    ProjectedBetaPlane

# Boundary & Sponge Layer Dynamics
export
    CartesianSpongeMask,
    stitch_shelf_domains

# Observation Nudging & Data Assimilation
export
    optimal_interpolation,
    build_observation_nudging,
    nudging_tendency,
    load_observation_csv,
    obs_indices,
    qc_observations

# Configuration & Driver API
export
    HydrodynamicOptions,
    HydrodynamicConfig,
    load_configuration,
    configuration_to_options,
    build_zarr_compressor,
    TethysFlow

end # module
