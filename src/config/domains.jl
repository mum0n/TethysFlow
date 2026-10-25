"""
    domains.jl

Which configuration section belongs to which domain.

The one place that knows this. `conversion.jl` reads a section, and `options.jl` holds a field
for it; before this file the partition existed only implicitly, built by hand in both the
reader and the writer and nowhere else.

Two domains in TethysFlow: `:shared` and `:ocean`.
`[domain]`, `[grid]`, `[metadata]`, `[storage]`, and `[runtype]` are `:shared`.
The physical forcing and model configuration sections belong to `:ocean`.
"""

"""
    OCEAN, SHARED

The configuration domains, as `Symbol`s.

`OCEAN` is the physical model: what is forced, what is bounded, what the sea floor is.
`SHARED` is infrastructure: metadata, domain, grid, storage, and runtype.
"""
const OCEAN   = :ocean
const SHARED  = :shared

"""
    CONFIG_SECTION_DOMAIN

Section name → domain. The single mapping every check reads.
"""
const CONFIG_SECTION_DOMAIN = Dict{String, Symbol}(
    # Shared: infrastructure parameters.
    "metadata"              => SHARED,
    "domain"                => SHARED,
    "grid"                  => SHARED,
    "storage"               => SHARED,
    "runtype"               => SHARED,
    # Ocean: what the physical model is forced and bounded by.
    "data"                  => OCEAN,
    "bathymetry"            => OCEAN,
    "atmosphere"            => OCEAN,
    "boundaries"            => OCEAN,
    "tides"                 => OCEAN,
    "bottom_boundary_layer" => OCEAN,
    "hydrodynamics"         => OCEAN,
    "observations"          => OCEAN,
    "visualization"         => OCEAN,
)

"""
    CONFIG_SECTIONS

Every configuration section this package knows, sorted.

An unknown section is an error rather than an ignored one: `conversion.jl` resolves each key
with a default, so a section that is misspelled — `[doman]` instead of `[domain]` — reads as
absent and every parameter inside it silently becomes its default. The run then reports a
domain it was never given, and nothing in the output distinguishes that from having asked for it.
"""
const CONFIG_SECTIONS = sort!(collect(keys(CONFIG_SECTION_DOMAIN)))

"""
    DOMAIN_SECTIONS

Domain → the sections it owns, sorted. Derived from `CONFIG_SECTION_DOMAIN` rather than written
out again, because a second list is a second thing to keep right.
"""
const DOMAIN_SECTIONS = Dict{Symbol, Vector{String}}(
    d => sort!(collect(k for (k, v) in CONFIG_SECTION_DOMAIN if v == d))
    for d in (OCEAN, SHARED)
)

"""
    section_domain(section::AbstractString) -> Symbol

Which domain a configuration section belongs to, erroring when the section is not known.

The error names the sections that are known, so a misspelling is corrected from the message
rather than guessed at. That is the point: an unrecognised section would otherwise be ignored,
and a run would proceed on defaults it never chose.
"""
function section_domain(section::AbstractString)::Symbol
    d = get(CONFIG_SECTION_DOMAIN, section, nothing)
    d === nothing && error(
        "Unknown configuration section [$(section)]. Known sections: " *
        "$(join(CONFIG_SECTIONS, ", ")). Check the spelling; a section this package does not " *
        "know is ignored, and every parameter inside it then falls back to its default.")
    return d
end

"""
    validate_config_sections(cfg::AbstractDict, origin::AbstractString = "") -> Nothing

Check every section of a parsed configuration is one this package knows.

Called by `load_configuration`, so the check happens once on the file the run actually named
rather than at each read site. An unrecognised section is an error, not a warning: every
reader skips a key it does not have a default for only if the section is missing, and a
misspelled section looks exactly like a missing one.

`origin` is only used in the message, so the error says which file was wrong.
"""
function validate_config_sections(cfg::AbstractDict, origin::AbstractString = "")::Nothing
    for k in sort!(collect(keys(cfg)))
        k in CONFIG_SECTIONS && continue
        where = isempty(origin) ? "" : " in $(origin)"
        error("Unknown configuration section [$(k)]$(where). Known sections: " *
              "$(join(CONFIG_SECTIONS, ", ")).")
    end
    return nothing
end
