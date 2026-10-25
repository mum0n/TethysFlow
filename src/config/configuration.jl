"""
    configuration.jl

Centralized configuration for `TethysFlow.jl`.

Assembled from five files, one per kind of thing rather than one per stage:

- `defaults.jl`    paths, study definitions, and default values
- `domains.jl`     which configuration section belongs to which domain
- `options.jl`     the types: runtime options and configuration structs
- `io.jl`          reading and writing TOML
- `conversion.jl`  dictionary <-> options <-> config structs

`domains` is included before `io` because `load_configuration` calls
`validate_config_sections`, and `defaults` is included first because everything falls back on it.
"""

include(joinpath(@__DIR__, "defaults.jl"))
include(joinpath(@__DIR__, "domains.jl"))
include(joinpath(@__DIR__, "options.jl"))
include(joinpath(@__DIR__, "io.jl"))
include(joinpath(@__DIR__, "conversion.jl"))

