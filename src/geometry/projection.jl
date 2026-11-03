"""
    projection.jl

Map projections, coordinate reference systems, conformal transformations,
vector rotations, and Coriolis dynamics for regional computational ocean models.

General CRS and projection support (UTM, Lambert Conformal Conic, Mercator,
RotatedFrame, EPSG/ESRI/WKT resolution, vector rotations, and reprojection lattices)
is provided by `Tethys`. This file re-exports those primitives and implements
Oceananigans-specific integration (`ProjectedBetaPlane` Coriolis dynamics).

# References
- Snyder, J. P. (1987). Map projections--A working manual (Vol. 1395). USGPO.
- Karney, C. F. (2011). Transverse Mercator with an accuracy of a few nanometers.
  Journal of Geodesy, 85(8), 475-485.
"""

using Oceananigans
using Oceananigans.Coriolis: AbstractRotation, EnstrophyConserving
import Oceananigans.Coriolis: fᶠᶠᵃ, fᶜᶜᵃ
using Oceananigans.Grids: xnode, ynode, Face, Center
import Adapt
using Tethys

# Re-export general CRS and projection definitions from Tethys
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

# ==============================================================================
# Geodetic Constants
# ==============================================================================
const EARTH_MEAN_RADIUS = 6371000.0
const EARTH_ROTATION_RATE = 7.292115e-5

# ==============================================================================
# Projected Coriolis Parameter (Oceananigans Integration)
# ==============================================================================

"""
    ProjectedBetaPlane{S, FT} <: Oceananigans.Coriolis.AbstractRotation{S}

Coriolis parameter on a projected or rotated Cartesian plane where planetary vorticity
and its linear gradients are evaluated:
```math
f(x, y) = f_0 + \\beta_x x + \\beta_y y
```
with:
```math
f_0 = 2 \\Omega \\sin\\phi_0, \\quad \\beta = \\frac{2\\Omega\\cos\\phi_0}{R}
```
```math
\\beta_x = \\beta \\sin\\alpha, \\quad \\beta_y = \\beta \\cos\\alpha
```
where ``\\alpha`` is the effective alignment angle relative to true East.

# Fields
- `scheme::S`: Numerical discretization scheme (e.g. `EnstrophyConserving`).
- `f₀::FT`: Central Coriolis parameter at domain origin (s⁻¹).
- `βx::FT`: Planetary vorticity gradient along rotated x-axis (m⁻¹ s⁻¹).
- `βy::FT`: Planetary vorticity gradient along rotated y-axis (m⁻¹ s⁻¹).
"""
struct ProjectedBetaPlane{S, FT} <: AbstractRotation{S}
    scheme :: S
    f₀     :: FT
    βx     :: FT
    βy     :: FT
end


"""
    ProjectedBetaPlane(FT = Float64; latitude, angle_deg = 0.0, scheme = EnstrophyConserving())

Construct a `ProjectedBetaPlane` centered at geographic `latitude` with orientation
angle `angle_deg` in degrees counter-clockwise from East.
"""
function ProjectedBetaPlane(
    FT = Float64;
    latitude::Real,
    angle_deg::Real = 0.0,
    scheme = EnstrophyConserving(),
    rotation_rate::Real = EARTH_ROTATION_RATE,
    radius::Real = EARTH_MEAN_RADIUS
)
    phi0_rad = deg2rad(Float64(latitude))
    theta_rad = deg2rad(Float64(angle_deg))

    f0 = 2.0 * rotation_rate * sin(phi0_rad)
    beta = 2.0 * rotation_rate * cos(phi0_rad) / radius

    bx = beta * sin(theta_rad)
    by = beta * cos(theta_rad)

    return ProjectedBetaPlane(scheme, convert(FT, f0), convert(FT, bx), convert(FT, by))
end

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::ProjectedBetaPlane) =
    coriolis.f₀ +
    coriolis.βx * xnode(i, j, k, grid, Face(), Face(), Center()) +
    coriolis.βy * ynode(i, j, k, grid, Face(), Face(), Center())

@inline fᶜᶜᵃ(i, j, k, grid, coriolis::ProjectedBetaPlane) =
    coriolis.f₀ +
    coriolis.βx * xnode(i, j, k, grid, Center(), Center(), Center()) +
    coriolis.βy * ynode(i, j, k, grid, Center(), Center(), Center())

Adapt.adapt_structure(to, pbp::ProjectedBetaPlane) =
    ProjectedBetaPlane(
        Adapt.adapt(to, pbp.scheme),
        Adapt.adapt(to, pbp.f₀),
        Adapt.adapt(to, pbp.βx),
        Adapt.adapt(to, pbp.βy)
    )

function Base.summary(pbp::ProjectedBetaPlane{S, FT}) where {S, FT}
    return "ProjectedBetaPlane{$FT}(f₀=$(pbp.f₀), βx=$(pbp.βx), βy=$(pbp.βy))"
end

Base.show(io::IO, pbp::ProjectedBetaPlane) = print(io, summary(pbp))
