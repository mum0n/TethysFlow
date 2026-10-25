"""
    defaults.jl

Configuration defaults and study definitions.

Where a configuration file is found, what a bare configuration name resolves to,
the domain each named study covers, and the fully populated default dictionary.
These are all *values*, so they live together; nothing here reads a file.
"""


"""
    find_default_config_path() -> String

Locate the configuration file in the project workspace, or return `""` when there is none.

Only project-root locations are searched. A configuration is an input the operator supplies;
it is never discovered inside a previous run's containerised `inputs/` directory, and it is
never guessed at. In particular there is no fallback to `configs/default.toml`: a run that
silently adopts a different configuration than the one intended produces output that looks
complete and answers a question nobody asked. Callers must fail loudly instead, which is what
`load_configuration` does with an empty path.

To run, pass one explicitly:

    julia --project=. TethysFlowRun.jl --config=configs/scotian_shelf.toml

or place `TethysFlow.toml` at the project root.
"""
function find_default_config_path()::String
    for p in ("TethysFlow.toml", "TethysFlow.config")
        isfile(p) && return p
    end
    ""
end

const NO_CONFIG_ADVICE = """
No configuration file was found.

A configuration is required, not optional. It names the datasets to ingest, the domain and
grid, the forcing sources, and the model parameters, and those choices are what a run is.
Without one the package would have to invent all of it, and the resulting output would carry
no statement of what it actually simulated -- it would be indistinguishable from a run that
was configured deliberately.

Available configurations in this project:
  configs/default.toml       Cape Breton : small box, low resolution, 3 days,
                             keyless data (etopo2022, WOA23, open_meteo). For demos.
  configs/scotian_shelf.toml  The full snow-crab Scotian Shelf domain, nwa_bathymetry
                             and GLORYS12V1 (Copernicus Marine account required).

Run with one of:

    julia --project=. TethysFlowRun.jl --config=configs/scotian_shelf.toml

or copy one to TethysFlow.toml at the project root, which the loader picks up
automatically. For a study of your own, copy the closest config and edit the sections the
run depends on: [data] for datasets, [domain] and [grid] for the box and resolution,
[tides] for tidal forcing, and [atmosphere] for wind and heat flux."""

"""
    study_domain_ranges(cfg) -> NamedTuple

Geographic bounds of the *study domain* — the computational/analysis grid — read from a
parsed configuration.

Authoritative source is the TOML `[domain]` section (`lon_min`, `lon_max`, `lat_min`,
`lat_max`). No domain is compiled into the package: the configuration file is the single
source of truth, so a run against a different shelf only requires editing TOML.

# Inputs
- `cfg`: Parsed configuration `AbstractDict` (e.g. from [`load_configuration`](@ref)).

# Outputs
- `NamedTuple`: `(lon = (lon_min, lon_max), lat = (lat_min, lat_max))`, in degrees.
"""
function study_domain_ranges(cfg)::NamedTuple
    d = get(cfg, "domain", Dict{String, Any}())
    return (
        lon = (Float64(get(d, "lon_min", -68.0)), Float64(get(d, "lon_max", -57.0))),
        lat = (Float64(get(d, "lat_min", 42.0)), Float64(get(d, "lat_max", 47.5)))
    )
end

"""
    embedding_domain_ranges(cfg) -> NamedTuple

Geographic bounds of the *embedding domain* — the larger region from which open boundary
and atmospheric forcing data are extracted — read from a parsed configuration.

Authoritative source is the TOML `[boundaries]` section (`embedding_lon_min`,
`embedding_lon_max`, `embedding_lat_min`, `embedding_lat_max`). The study domain nests
inside this region.

# Inputs
- `cfg`: Parsed configuration `AbstractDict`.

# Outputs
- `NamedTuple`: `(lon = (lon_min, lon_max), lat = (lat_min, lat_max))`, in degrees.
"""
function embedding_domain_ranges(cfg)::NamedTuple
    b = get(cfg, "boundaries", Dict{String, Any}())
    return (
        lon = (Float64(get(b, "embedding_lon_min", -71.0)), Float64(get(b, "embedding_lon_max", -53.0))),
        lat = (Float64(get(b, "embedding_lat_min", 40.0)), Float64(get(b, "embedding_lat_max", 48.5)))
    )
end

"""
    resolve_config_name(config_file::AbstractString = "") -> String

Extract the base configuration name from a config file path, or return
the default configuration name (`"TethysFlow"`) if unspecified or empty.

# Inputs
- `config_file::AbstractString`: Path to a `.toml` file or empty string.

# Outputs
- `String`: Clean configuration identifier without directory or extension.
"""
function resolve_config_name(config_file::AbstractString = "")::String
    cfg_path = isempty(strip(config_file)) ? find_default_config_path() : String(config_file)
    base = basename(cfg_path)
    stem = splitext(base)[1]
    return isempty(stem) ? "TethysFlow" : stem
end


"""
    get_default_configuration() -> Dict{String, Any}

Generate a comprehensive dictionary of all default parameters across all modeling domains.
"""
function get_default_configuration()::Dict{String, Any}
    return Dict{String, Any}(
        "domain" => Dict{String, Any}(
            "lon_min" => -71.0,
            "lon_max" => -53.0,
            "lat_min" => 40.0,
            "lat_max" => 48.5,
            "z_min" => -3500.0,
            "z_max" => 0.0,
            "buffer_km" => 100.0
        ),
        "grid" => Dict{String, Any}(
            "nx" => 50,
            "ny" => 50,
            "nz" => 10,
            "vertical_stretching_mode" => "oceananigans",
            "vertical_grid_file" => ""
        ),
        "data" => Dict{String, Any}(
            "bathy_source" => "etopo2022",
            "coastline_source" => "natural_earth_10m",
            "wind_source" => "open_meteo",
            "hydrography_source" => "woa23",
            "wind_time_iso" => "2023-06-01T00:00:00Z"
        ),
        "bathymetry" => Dict{String, Any}(
            "source" => "etopo2022",
            "resolution_arcsec" => 15,
            "inshore_depth" => 20.0,
            "shelf_slope" => 180.0
        ),
        "boundaries" => Dict{String, Any}(
            "method" => "relaxation",
            "ocean_boundary_source" => "glorys12v1",
            "obc_type" => "flather_chapman",
            # The sponge band's width is not a setting here: it is [domain] buffer_km,
            # which is the margin the band lives in. It is derived when the options are
            # built, and recorded in the resolved configuration.
            "sponge_timescale" => 3600.0,
            "u_inflow" => -0.15,
            "v_inflow" => 0.05
        ),
        "atmosphere" => Dict{String, Any}(
            "source" => "era5",
            "drag_formulation" => "garratt_1977",
            "bulk_heat_flux" => true
        ),
        "tides" => Dict{String, Any}(
            "enable_tides" => true,
            "constituents" => ["M2"],
            "tidal_u_amp" => 0.25,
            "tidal_v_amp" => 0.12,
            "tidal_period" => 44712.0,
            "tidal_phase" => 0.0
        ),
        "hydrodynamics" => Dict{String, Any}(
            "sim_duration_hours" => 12.0,
            "sim_dt_seconds" => 120.0,
            "adaptive_cfl" => true,
            "target_cfl" => 0.2,
            "divergence_velocity_limit" => 20.0,
            "coriolis_latitude" => 44.5,
            "surface_heat_flux" => 50.0,
            # The quadratic seabed friction coefficient. It is applied to the model as a
            # `BulkDrag` bottom boundary condition and is the drag the Courant estimator
            # protects, so the two are read from this one value.
            "cd_drag" => 2.5e-3,
            "max_flow_snapshots" => 0,
            "allow_analytical_fallback" => false,
            "turbulence_closure" => "nemotke",
            "enable_sea_ice" => true,
            "enable_oxygen" => true
        ),
        "storage" => Dict{String, Any}(
            "output_path" => joinpath("outputs", "tethys_flow.jld2"),
            "enable_checkpoint" => true,
            "checkpoint_prefix" => "checkpoint_TethysFlow",
            "checkpoint_schedule_seconds" => 21600.0,
            "checkpoint_dir" => "",
            "checkpoint_cleanup" => true
        ),
        "visualization" => Dict{String, Any}(
            "interactive_map" => true,
            "animate_hydro" => false,
            "anim_variable" => "dashboard",
            "anim_fps" => 10,
            "anim_format" => "mp4",
            "anim_depth" => -2.5,
            "anim_overlay_particles" => false,
            "anim_output_path" => "",
            "title" => "Regional Coastal and Shelf Hydrodynamic Simulation"
        ),
        "runtype" => Dict{String, Any}(
            "use_gpu" => false,
            "fallback_to_cpu" => false,
            "seed" => 42
        )
    )
end