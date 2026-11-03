# ==============================================================================
# Map Projections, CoordRefSystems Integration & Beta Plane Tests
# ==============================================================================

using Test
using Oceananigans
using TethysFlow
const TethysFlow = TethysFlow

@testset "Map Projections and CoordRefSystems Integration" begin
    # Reference coordinates on Scotian Shelf
    lon_c = -63.0
    lat_c = 44.0

    @testset "UTM Zone 20N Projection" begin
        utm = UTMProjection(zone = 20, is_north = true, lon_center = lon_c, lat_center = lat_c)
        @test is_planar(utm) == true
        @test utm isa CoordRefSystemsProjection

        # Center point projects to (0, 0) relative to grid center
        x0, y0 = project_forward(utm, lon_c, lat_c)
        @test abs(x0) < 1e-4
        @test abs(y0) < 1e-4

        # Roundtrip test
        lon_test = -62.5
        lat_test = 44.3
        x, y = project_forward(utm, lon_test, lat_test)
        lon_rt, lat_rt = project_inverse(utm, x, y)
        @test isapprox(lon_rt, lon_test, atol = 1e-7)
        @test isapprox(lat_rt, lat_test, atol = 1e-7)

        # Vector rotation preserves magnitude
        u_geo, v_geo = -0.15, 0.05
        u_rot, v_rot = rotate_vector_to_grid(utm, u_geo, v_geo, lon_test, lat_test)
        @test isapprox(hypot(u_rot, v_rot), hypot(u_geo, v_geo), atol = 1e-10)

        # Reverse rotation
        u_back, v_back = rotate_vector_to_geo(utm, u_rot, v_rot, lon_test, lat_test)
        @test isapprox(u_back, u_geo, atol = 1e-10)
        @test isapprox(v_back, v_geo, atol = 1e-10)
    end

    @testset "Lambert Conformal Conic Projection" begin
        lambert = LambertProjection(
            lat_1 = 43.0,
            lat_2 = 47.0,
            lon_0 = lon_c,
            lat_0 = lat_c,
            lon_center = lon_c,
            lat_center = lat_c
        )
        @test is_planar(lambert) == true

        # Center point projects to near (0, 0)
        x0, y0 = project_forward(lambert, lon_c, lat_c)
        @test abs(x0) < 1e-4
        @test abs(y0) < 1e-4

        # Roundtrip test across Scotian Shelf
        test_pts = [(-64.0, 43.5), (-62.0, 45.0), (-60.0, 44.2)]
        for (lon_t, lat_t) in test_pts
            x, y = project_forward(lambert, lon_t, lat_t)
            lon_rt, lat_rt = project_inverse(lambert, x, y)
            @test isapprox(lon_rt, lon_t, atol = 1e-6)
            @test isapprox(lat_rt, lat_t, atol = 1e-6)
        end
    end

    @testset "Mercator Projection" begin
        merc = MercatorProjection(lon_0 = lon_c, lat_ts = lat_c, lon_center = lon_c, lat_center = lat_c)
        @test is_planar(merc) == true

        x, y = project_forward(merc, -62.0, 44.5)
        lon_rt, lat_rt = project_inverse(merc, x, y)
        @test isapprox(lon_rt, -62.0, atol = 1e-6)
        @test isapprox(lat_rt, 44.5, atol = 1e-6)
    end

    @testset "Rotated Frame (Conformal Tangent)" begin
        rot_frame = RotatedFrame(
            lon_center = lon_c,
            lat_center = lat_c,
            angle_deg = 50.0
        )
        @test is_planar(rot_frame) == true
        @test rot_frame.angle_deg == 50.0

        # Center projects to (0, 0)
        x0, y0 = project_forward(rot_frame, lon_c, lat_c)
        @test abs(x0) < 1e-6
        @test abs(y0) < 1e-6

        # Roundtrip test
        lon_t = -61.5
        lat_t = 44.8
        x, y = project_forward(rot_frame, lon_t, lat_t)
        lon_rt, lat_rt = project_inverse(rot_frame, x, y)
        @test isapprox(lon_rt, lon_t, atol = 1e-8)
        @test isapprox(lat_rt, lat_t, atol = 1e-8)

        # Vector rotation
        u_geo, v_geo = 1.0, 0.0
        u_grid, v_grid = rotate_vector_to_grid(rot_frame, u_geo, v_geo, lon_t, lat_t)
        @test isapprox(hypot(u_grid, v_grid), 1.0, atol = 1e-10)
        u_rev, v_rev = rotate_vector_to_geo(rot_frame, u_grid, v_grid, lon_t, lat_t)
        @test isapprox(u_rev, u_geo, atol = 1e-10)
        @test isapprox(v_rev, v_geo, atol = 1e-10)
    end

    @testset "build_projection Factory Function" begin
        p_utm = build_projection(projection = "utm", zone = 20, lon_center = lon_c, lat_center = lat_c)
        @test p_utm isa CoordRefSystemsProjection

        p_lcc = build_projection(projection = "lambert", lon_center = lon_c, lat_center = lat_c)
        @test p_lcc isa CoordRefSystemsProjection

        p_rot = build_projection(projection = "rotated", rotation_deg = 50.0, lon_center = lon_c, lat_center = lat_c)
        @test p_rot isa RotatedFrame

        p_geo = build_projection(projection = "geographic")
        @test p_geo isa GeographicProjection
    end

    @testset "ProjectedBetaPlane & HydrostaticFreeSurfaceModel Execution" begin
        coriolis = ProjectedBetaPlane(latitude = 44.5, angle_deg = 50.0)
        @test coriolis.f₀ > 0.0
        @test coriolis.βx != 0.0 || coriolis.βy != 0.0

        # Construct small rectilinear grid
        grid = RectilinearGrid(
            size = (10, 10, 5),
            x = (-50_000, 50_000),
            y = (-50_000, 50_000),
            z = (-200, 0),
            topology = (Bounded, Bounded, Bounded)
        )

        model = HydrostaticFreeSurfaceModel(
            grid;
            coriolis = coriolis,
            momentum_advection = WENO(),
            tracer_advection = WENO(),
            tracers = (:T, :S),
            buoyancy = SeawaterBuoyancy()
        )

        set!(model, T = 10.0, S = 35.0)
        time_step!(model, 60.0)
        @test model.clock.time == 60.0
        @test all(isfinite, interior(model.velocities.u))
        @test all(isfinite, interior(model.velocities.v))
    end

    @testset "build_shelf_grid with Projection" begin
        proj = build_projection(
            projection = "utm",
            zone = 20,
            lon_center = -63.0,
            lat_center = 44.0,
            extent_x = 100_000.0,
            extent_y = 60_000.0
        )

        grid = build_shelf_grid(
            architecture = :cpu,
            grid_size = (16, 12, 5),
            z_range = (-200.0, 0.0),
            projection = proj,
            extent_x = 100_000.0,
            extent_y = 60_000.0
        )

        @test grid isa RectilinearGrid
        @test grid.Nx == 16
        @test grid.Ny == 12
        @test grid.Nz == 5
        @test isapprox(grid.xᶠᵃᵃ[grid.Nx + 1] - grid.xᶠᵃᵃ[1], 100_000.0, atol = 1.0)
        @test isapprox(grid.yᵃᶠᵃ[grid.Ny + 1] - grid.yᵃᶠᵃ[1], 60_000.0, atol = 1.0)

        coords = extract_grid_coordinates(grid; projection = proj)
        @test haskey(coords, :x)
        @test haskey(coords, :y)
        @test length(coords.x) == 16
        @test length(coords.y) == 12
    end

    @testset "CoordRefSystems Direct CRS & WKT Resolution" begin
        # 1. Integer EPSG code
        p_epsg_int = build_projection(crs = 32620, lon_center = lon_c, lat_center = lat_c)
        @test p_epsg_int isa CoordRefSystemsProjection
        @test is_planar(p_epsg_int) == true

        # 2. String EPSG code
        p_epsg_str = build_projection(projection = "EPSG:32620", lon_center = lon_c, lat_center = lat_c)
        @test p_epsg_str isa CoordRefSystemsProjection

        # 3. Numeric string
        p_epsg_num = build_projection(projection = "32620", lon_center = lon_c, lat_center = lat_c)
        @test p_epsg_num isa CoordRefSystemsProjection

        # 4. ESRI code
        p_esri = build_projection(crs = "ESRI:102035", lon_center = 0.0, lat_center = 80.0)
        @test p_esri isa CoordRefSystemsProjection

        # 5. WKT string
        wkt = "PROJCS[\"WGS 84 / UTM zone 20N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",-63],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1],AUTHORITY[\"EPSG\",\"32620\"]]"
        p_wkt = build_projection(crs = wkt, lon_center = lon_c, lat_center = lat_c)
        @test p_wkt isa CoordRefSystemsProjection

        # 6. omerc alias
        p_omerc = build_projection(projection = "omerc", lon_center = lon_c, lat_center = lat_c, rotation_deg = 45.0)
        @test p_omerc isa RotatedFrame
        @test p_omerc.angle_deg == 45.0

        # Verify roundtrip on resolved CRS projection
        x, y = project_forward(p_epsg_int, lon_c + 0.1, lat_c + 0.1)
        lon_back, lat_back = project_inverse(p_epsg_int, x, y)
        @test isapprox(lon_back, lon_c + 0.1, atol = 1e-6)
        @test isapprox(lat_back, lat_c + 0.1, atol = 1e-6)
    end

    @testset "Projected forcing vectors use model-grid coordinates" begin
        projection = RotatedFrame(lon_center = -62.5, lat_center = 43.5, angle_deg = 50.0)
        T = TethysFlow.Tethys
        lon = [-63.0, -62.0]
        lat = [43.0, 44.0]
        dims = (
            T.Dimension(name = :lon, size = 2, coords = lon, units = "degrees_east"),
            T.Dimension(name = :lat, size = 2, coords = lat, units = "degrees_north"),
            T.Dimension(name = :time, size = 1, coords = [0.0]),
        )
        crs = T.CoordinateSystem(crs = "EPSG:4326")
        coords = Dict(
            :lon => T.GeoArray(lon, (dims[1],), crs),
            :lat => T.GeoArray(lat, (dims[2],), crs),
        )
        source = T.GeoDataset(
            Dict("u" => T.GeoArray(ones(2, 2, 1), dims, crs),
                 "v" => T.GeoArray(zeros(2, 2, 1), dims, crs)),
            coords, Dict(:lon => dims[1], :lat => dims[2], :time => dims[3]),
            crs, Dict{String, Any}(), nothing, "projection_fixture")
        rotated = T.transform_vector_dataset(
            source, projection; pairs = ((:u, :v, :u_grid, :v_grid),))
        u_grid = rotated.variables["u_grid"].data
        v_grid = rotated.variables["v_grid"].data
        expected_u, expected_v = rotate_vector_to_grid(projection, 1.0, 0.0, -63.0, 43.0)
        @test isapprox(u_grid[1, 1, 1], expected_u; atol = 1e-12)
        @test isapprox(v_grid[1, 1, 1], expected_v; atol = 1e-12)
        @test isapprox(hypot(u_grid[1, 1, 1], v_grid[1, 1, 1]), 1.0; atol = 1e-12)

        harmonics = TethysFlow.TidalVelocityHarmonics(
            [:M2], [-63.0, -62.0], [43.0, 44.0],
            ones(1, 2, 2), zeros(1, 2, 2), zeros(1, 2, 2), zeros(1, 2, 2), [1.0])
        x0, y0 = project_forward(projection, -63.0, 43.0)
        x1, y1 = project_forward(projection, -62.0, 44.0)
        coefficients = TethysFlow.TidalCoefficientGrid(
            harmonics, (x0, x1), (y0, y1); projection = projection, nx = 2, ny = 2)
        tidal = TethysFlow.tidal_velocity_coefficients(coefficients)
        @test isapprox(tidal.u(x0, y0, 0.0, 0.0), expected_u; atol = 1e-12)
        @test isapprox(tidal.v(x0, y0, 0.0, 0.0), expected_v; atol = 1e-12)
    end
end
