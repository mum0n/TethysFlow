using Test
using TethysFlow
const TethysFlow = TethysFlow
using Tethys
using Oceananigans

const OCEAN_TMP = joinpath(@__DIR__, "outputs", "_test_geoceanmodel")

@testset "Output Store" begin
    if ispath(OCEAN_TMP)
        rm(OCEAN_TMP; recursive = true, force = true)
    end
    mkpath(OCEAN_TMP)

    local mk = Oceananigans.RectilinearGrid(size = (4, 4, 2), extent = (4, 4, 2))
    local model = Oceananigans.HydrostaticFreeSurfaceModel(mk; tracers = (:T,))
    set!(model, T = 13.5)

    local store = TethysFlow.attach_output_callback!(
        Oceananigans.Simulation(model; Δt = 1.0, stop_time = 2.0), OCEAN_TMP;
        fields = ("u", "T"))
    @test store == TethysFlow.output_zarr_path(OCEAN_TMP)

    local sim = Oceananigans.Simulation(model; Δt = 1.0, stop_time = 2.0)
    TethysFlow.attach_output_callback!(sim, OCEAN_TMP; fields = ("u", "T"))
    Oceananigans.run!(sim)
    @test isdir(store)

    local ds = Tethys.load_dataset(store; backend = :zarr)
    @test "T" in keys(ds.variables) && "u" in keys(ds.variables)
    @test length(ds.dims[:lon].coords) == 4
    @test length(ds.dims[:lat].coords) == 4
    @test length(ds.dims[:depth].coords) == 2
    @test !isempty(ds.coords[:time].data)
    @test size(ds.variables["T"].data) == size(ds.variables["u"].data)

    local T0 = ds.variables["T"].data[:, :, :, 1]
    @test size(T0) == (4, 4, 2)
    @test T0[1, 1, 1] == 13.5

    # Re-run replaces rather than appends
    local d = joinpath(OCEAN_TMP, "rerun")
    local function one_run(value)
        local mk_r = Oceananigans.RectilinearGrid(size = (4, 4, 2), extent = (4, 4, 2))
        local model_r = Oceananigans.HydrostaticFreeSurfaceModel(mk_r; tracers = (:T,))
        set!(model_r, T = value)
        local sim_r = Oceananigans.Simulation(model_r; Δt = 1.0, stop_time = 1.0)
        TethysFlow.attach_output_callback!(sim_r, d; fields = ("T",))
        Oceananigans.run!(sim_r)
        return sum(length(Tethys.load_dataset(
            TethysFlow.output_zarr_path(d); backend = :zarr).coords[:time].data))
    end
    local first_n = one_run(10.0)
    @test first_n > 0
    @test one_run(20.0) == first_n

    # Interval detection and store metadata inspection
    @test TethysFlow._count_written_intervals(Float64[]) == 0
    @test TethysFlow._count_written_intervals(Float64.(1:50)) == 0
    @test TethysFlow._count_written_intervals([1800.0, 3600.0, 5400.0, 4.0, 5.0]) == 3
    @test TethysFlow._count_written_intervals([1800.0, 3600.0, NaN, NaN]) == 2
    @test TethysFlow._count_written_intervals([1800.0, 3600.0, 5400.0]) == 3
    @test TethysFlow._count_written_intervals([1.0]) == 0
    @test TethysFlow._count_written_intervals([1800.0]) == 1

    local store_path = TethysFlow.output_zarr_path(OCEAN_TMP)
    @test TethysFlow.store_interval_count(store_path) > 0
    local info = TethysFlow._store_sim_info(store_path; stop_time = 2.0)
    @test info.exists
    @test info.is_complete
    @test info.n_timesteps > 0

    # Checkpoint retention pruning
    local cp_test_dir = joinpath(OCEAN_TMP, "test_cp_prune")
    mkpath(cp_test_dir)
    for iter in [100, 200, 300, 400, 500]
        touch(joinpath(cp_test_dir, "checkpoint_test_iteration$(iter).jld2"))
    end
    @test length(TethysFlow.list_candidate_checkpoints(cp_test_dir; prefix = "checkpoint_test")) == 5
    local deleted = TethysFlow.prune_checkpoints!(cp_test_dir; prefix = "checkpoint_test", keep = 3)
    @test length(deleted) == 2
    local remaining = TethysFlow.list_candidate_checkpoints(cp_test_dir; prefix = "checkpoint_test")
    @test length(remaining) == 3
    @test occursin("500", remaining[1])
    @test occursin("400", remaining[2])
    @test occursin("300", remaining[3])
    rm(cp_test_dir; recursive = true, force = true)

    # Periodic checkpoint generation with independent schedules
    local cp_run_dir = joinpath(OCEAN_TMP, "test_cp_run")
    mkpath(cp_run_dir)
    local m_cp = Oceananigans.HydrostaticFreeSurfaceModel(
        Oceananigans.RectilinearGrid(size = (4, 4, 2), extent = (4, 4, 2));
        tracers = (:T,)
    )
    local sim_cp = TethysFlow.setup_hydrodynamic_simulation(
        m_cp,
        Δt = 1.0,
        stop_time = 10.0,
        enable_checkpoint = true,
        checkpoint_dir = cp_run_dir,
        checkpoint_prefix = "checkpoint_test_periodic",
        checkpoint_schedule = 5.0,
        checkpoint_keep = 3
    )
    Oceananigans.run!(sim_cp)
    local cp_files = TethysFlow.list_candidate_checkpoints(
        cp_run_dir; prefix = "checkpoint_test_periodic"
    )
    # Both iteration 0, 5, and 10 checkpoints must be written
    @test length(cp_files) >= 2
    @test any(f -> occursin("iteration10", f), cp_files)
    @test any(f -> occursin("iteration5", f), cp_files)

    # find_latest_checkpoint correctly locates iteration 10
    local latest_valid = TethysFlow.find_latest_checkpoint(
        cp_run_dir; prefix = "checkpoint_test_periodic"
    )
    @test !isnothing(latest_valid)
    @test occursin("iteration10", latest_valid)

    # A corrupted higher iteration file is gracefully skipped in favor of the latest intact checkpoint
    local corrupt_path = joinpath(cp_run_dir, "checkpoint_test_periodic_iteration15.jld2")
    write(corrupt_path, "TRUNCATED_SUPERBLOCK_GARBAGE_BYTES")
    local recovered_latest = TethysFlow.find_latest_checkpoint(
        cp_run_dir; prefix = "checkpoint_test_periodic"
    )
    @test !isnothing(recovered_latest)
    @test occursin("iteration10", recovered_latest)

    rm(cp_run_dir; recursive = true, force = true)

    # Zarr Blosc compressor construction and configuration
    local comp = TethysFlow.build_zarr_compressor(codec = "zstd", level = 7, shuffle = 1)
    @test comp.cname == "zstd"
    @test comp.clevel == 7
    @test comp.shuffle == 1
    local fallback_comp = TethysFlow.build_zarr_compressor(codec = "unsupported_codec", level = 12, shuffle = 5)
    @test fallback_comp.cname == "zstd"
    @test fallback_comp.clevel == 9
    @test fallback_comp.shuffle == 2

    # Verification of Zstd compression metadata on created store
    local comp_store_dir = joinpath(OCEAN_TMP, "test_zstd_store")
    mkpath(comp_store_dir)
    local m_zstd = Oceananigans.HydrostaticFreeSurfaceModel(
        Oceananigans.RectilinearGrid(size = (4, 4, 2), extent = (4, 4, 2));
        tracers = (:T,)
    )
    local sim_zstd = Oceananigans.Simulation(m_zstd; Δt = 1.0, stop_time = 1.0)
    TethysFlow.attach_output_callback!(
        sim_zstd,
        comp_store_dir;
        fields = ("T",),
        schedule = Oceananigans.TimeInterval(1.0),
        compressor = comp
    )
    Oceananigans.run!(sim_zstd)
    local zarr_meta_file = joinpath(comp_store_dir, "hydrodynamics.zarr", "T", ".zarray")
    @test isfile(zarr_meta_file)
    local meta_text = read(zarr_meta_file, String)
    @test occursin("\"cname\":\"zstd\"", meta_text)
    @test occursin("\"clevel\":7", meta_text)
    rm(comp_store_dir; recursive = true, force = true)

    # Geographic reprojection and vector unrotation during output saving
    local rep_store_dir = joinpath(OCEAN_TMP, "test_rep_store")
    mkpath(rep_store_dir)
    local proj_test = build_projection(
        projection = "lambert",
        lat_1 = 42.0, lat_2 = 47.0,
        lon_center = -62.0, lat_center = 44.5,
        rotation_deg = 55.0,
        extent_x = 1056.0 * 1e3, extent_y = 512.0 * 1e3
    )
    local grid_rep = Oceananigans.RectilinearGrid(
        size = (8, 6, 2),
        x = (-528e3, 528e3),
        y = (-256e3, 256e3),
        z = (-200.0, 0.0)
    )
    local m_rep = Oceananigans.HydrostaticFreeSurfaceModel(
        grid_rep;
        tracers = (:T, :S)
    )
    set!(m_rep, T = 12.5, S = 34.8, u = 0.5, v = -0.2)
    local t_lons = collect(range(-65.0, -60.0, length = 10))
    local t_lats = collect(range(42.0, 46.0, length = 8))
    local sim_rep = Oceananigans.Simulation(m_rep; Δt = 1.0, stop_time = 2.0)
    TethysFlow.attach_output_callback!(
        sim_rep,
        rep_store_dir;
        fields = ("u", "v", "T", "S"),
        schedule = Oceananigans.TimeInterval(1.0),
        projection = proj_test,
        target_lons = t_lons,
        target_lats = t_lats,
        reproject = true
    )
    Oceananigans.run!(sim_rep)
    local ds_rep = Tethys.load_dataset(
        TethysFlow.output_zarr_path(rep_store_dir); backend = :zarr
    )
    @test ds_rep.crs.crs == "EPSG:4326"
    @test length(ds_rep.dims[:lon].coords) == 10
    @test length(ds_rep.dims[:lat].coords) == 8
    @test length(ds_rep.dims[:depth].coords) == 2
    @test size(ds_rep.variables["T"].data) == (10, 8, 2, 3)
    @test size(ds_rep.variables["u"].data) == (10, 8, 2, 3)
    @test ds_rep.coords[:lon].data ≈ t_lons
    @test ds_rep.coords[:lat].data ≈ t_lats
    @test ds_rep.attrs["grid_type"] == "geographic_reprojected"
    @test ds_rep.attrs["reprojected_to"] == "EPSG:4326"
    rm(rep_store_dir; recursive = true, force = true)
end
