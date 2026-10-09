#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/kt_models.hh>
#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/assembler/cg.hh>
#include <DiFfRG/discretization/FEM/assembler/dg.hh>
#include <DiFfRG/discretization/FEM/assembler/ldg.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/common/snapshot_state.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/data/snapshot.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace DiFfRG;

namespace
{
  ConfigTree make_config(const unsigned int fe_order = 2)
  {
    return json::value(
        {{"physical", {{"Lambda", 1.}}},
         {"discretization",
          {{"fe_order", fe_order},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", "0:0.125:1"}, {"y_grid", "0:0.25:1"}, {"z_grid", "0:0.25:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1e-1},
             {"level", 0},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 5e-2}}}}},
         {"timestepping", {{"final_time", 1.}, {"output_dt", 1e-1}}},
         {"output", {{"verbosity", 0}, {"vtk", false}}}});
  }

  Testing::PhysicalParameters linear_profile()
  {
    Testing::PhysicalParameters prm;
    prm.initial_x0[0] = 0.25;
    prm.initial_x1[0] = 1.5;
    return prm;
  }

  /// Refine the cells whose centre is left of x_cut, @p levels times, then coarsen part of it back.
  template <int dim> void adapt_irregularly(dealii::Triangulation<dim> &triangulation)
  {
    for (int level = 0; level < 2; ++level) {
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->center()[0] < 0.5 - 0.2 * level) cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
    }
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->level() == 2 && cell->center()[0] < 0.125) cell->set_coarsen_flag();
    triangulation.execute_coarsening_and_refinement();
  }

  /// Cell id -> that cell's local dof values. A rebuilt mesh may enumerate its cells in a different
  /// order than the one it was saved from, so states are compared per cell, not per position.
  std::map<std::string, std::vector<double>> by_cell(const SnapshotSpatialState &state)
  {
    std::map<std::string, std::vector<double>> cells;
    for (std::size_t c = 0; c < state.active_cells.size(); ++c)
      cells[state.active_cells[c]] = std::vector<double>(state.values.begin() + c * state.dofs_per_cell,
                                                         state.values.begin() + (c + 1) * state.dofs_per_cell);
    return cells;
  }

  double max_difference(const std::vector<double> &a, const std::vector<double> &b)
  {
    REQUIRE(a.size() == b.size());
    double worst = 0.;
    for (std::size_t i = 0; i < a.size(); ++i)
      worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
  }

  /// Capture through a replicated view: capture_cellwise_state reads every dof, which a distributed vector
  /// cannot serve. Serially the view is a passthrough.
  template <typename Discretization, typename Assembler>
  SnapshotSpatialState capture_replicated(const Discretization &discretization, const Assembler &assembler,
                                          const typename Discretization::VectorType &spatial)
  {
    SolutionView<typename Discretization::VectorType> view;
    assembler.reinit_solution_view(view);
    view.refresh(spatial);
    return DiFfRG::internal::capture_cellwise_state(discretization.get_dof_handler(), view.get());
  }

  /// Interpolate on one discretization, capture, restore into a fresh one, capture again.
  template <typename Discretization, typename Assembler, typename FlowingVariables, typename Model>
  void require_roundtrip(const ConfigTree &config, Model &model, const bool adapt_source)
  {
    constexpr uint dim = Discretization::dim;
    using Mesh = typename Discretization::Mesh;

    Mesh source_mesh{Config::ConfigurationMesh<dim>(config)};
    Discretization source(source_mesh, config);
    Assembler source_assembler(source, model, config);
    if (adapt_source) {
      adapt_irregularly(source.get_triangulation());
      source.reinit();
      source_assembler.reinit();
    }
    FlowingVariables source_state(source);
    source_state.interpolate(model);
    const auto captured = capture_replicated(source, source_assembler, source_state.spatial_data());

    REQUIRE(captured.active_cells.size() == source.get_triangulation().n_active_cells());
    REQUIRE(captured.values.size() == captured.active_cells.size() * captured.dofs_per_cell);
    // Guard against a degenerate field, which would make every comparison below pass.
    REQUIRE(*std::max_element(captured.values.begin(), captured.values.end()) -
                *std::min_element(captured.values.begin(), captured.values.end()) >
            0.5);

    Mesh target_mesh{Config::ConfigurationMesh<dim>(config)};
    Discretization target(target_mesh, config);
    Assembler target_assembler(target, model, config);
    FlowingVariables target_state(target);
    target_state.interpolate(model);
    target_state.spatial_data() = 0.;
    target_assembler.restore_snapshot_state(captured, target_state.spatial_data());

    REQUIRE(target.get_triangulation().n_active_cells() == source.get_triangulation().n_active_cells());
    const auto recaptured = capture_replicated(target, target_assembler, target_state.spatial_data());
    REQUIRE(by_cell(recaptured) == by_cell(captured));
  }
} // namespace

TEST_CASE("ModelState stores named values", "[snapshot]")
{
  DiFfRG::Init();
  ModelState state;
  state.set("lock", true);
  state.set("last", 0.125);
  state.set("count", 7);
  state.set("profile", std::vector<double>{1., 2., 3.});

  REQUIRE(state.get<bool>("lock"));
  REQUIRE(state.get<double>("last") == 0.125);
  REQUIRE(state.get<int>("count") == 7);
  REQUIRE(state.get_vector("profile") == std::vector<double>{1., 2., 3.});
  REQUIRE_THROWS(state.get<double>("profile"));
  REQUIRE_THROWS(state.get<double>("missing"));
  REQUIRE_THROWS(state.set("a/b", 1.));
}

TEST_CASE("config_diff lists changed, added and removed leaves", "[snapshot]")
{
  DiFfRG::Init();
  const json::value before = json::parse(R"({"physical": {"T": 0.1, "Nc": 3, "mu": 0.0},
                                             "output": {"name": "a"}, "grid": "0:1:2"})");
  const json::value after = json::parse(R"({"physical": {"T": 0.2, "Nc": 3.0, "extra": true},
                                            "output": {"name": "b"}, "grid": "0:1:2"})");

  const auto diff = config_diff(before, after, {"/output"});
  const std::set<std::string> lines(diff.begin(), diff.end());
  // Nc: 3 vs 3.0 is the same number and must not be reported.
  REQUIRE(lines == std::set<std::string>{"/physical/T: 0.1 -> 0.2", "/physical/mu: removed (0)",
                                         "/physical/extra: added (true)"});
  REQUIRE(config_diff(before, before).empty());
}

TEST_CASE("SnapshotSchedule converts, snaps and bounds snapshot times", "[snapshot]")
{
  DiFfRG::Init();
  ConfigTree config = make_config();
  config().as_object()["timestepping"].as_object()["snapshots"] =
      json::parse(R"({"k": [0.5, 0.25], "t": [0.33, 5.0, 0.0]})");
  const SnapshotSchedule schedule(config, /*Lambda=*/1., /*output_dt=*/0.1);

  // t = ln 2 = 0.693 -> 0.7, ln 4 = 1.386 -> 1.4, 0.33 -> 0.3; 0 is not after the start and 5 is
  // past the end.
  const auto times = schedule.times_in(0., 2.);
  REQUIRE(times.size() == 3);
  REQUIRE_THAT(times[0], Catch::Matchers::WithinAbs(0.3, 1e-12));
  REQUIRE_THAT(times[1], Catch::Matchers::WithinAbs(0.7, 1e-12));
  REQUIRE_THAT(times[2], Catch::Matchers::WithinAbs(1.4, 1e-12));

  // Snapping is relative to the start of the run.
  const auto shifted = schedule.times_in(0.05, 2.);
  REQUIRE_THAT(shifted[0], Catch::Matchers::WithinAbs(0.35, 1e-12));

  // A restart at a snapshot's time does not write that snapshot again.
  const auto after_restart = schedule.times_in(0.7, 2.);
  REQUIRE(after_restart.size() == 1);

  config().as_object()["timestepping"].as_object()["snapshots"].as_object()["snap_to_output_grid"] = false;
  const SnapshotSchedule exact(config, 1., 0.1);
  REQUIRE_THAT(exact.times_in(0., 2.)[1], Catch::Matchers::WithinAbs(std::log(2.), 1e-14));

  REQUIRE_THROWS(SnapshotSchedule(config, /*Lambda=*/-1., 0.1));
}

TEST_CASE("Snapshot files round-trip", "[snapshot]")
{
  DiFfRG::Init();
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "snapshot_io", "snapshot_io");
  const auto path = root.run_file("_snapshot_000", ".h5");

  SnapshotData data;
  data.t = 0.75;
  data.Lambda = 2.;
  data.k = 2. * std::exp(-0.75);
  data.dim = 2;
  data.spatial.fe_name = "FE_Q<2>(2)";
  data.spatial.dofs_per_cell = 2;
  data.spatial.n_coarse_cells = 1;
  data.spatial.n_dofs = 3;
  data.spatial.active_cells = {"0_1:0", "0_1:1"};
  data.spatial.values = {1., 2., 2., 3.};
  data.variables = {0.5, -0.5};
  data.model.set("lock", true);
  data.model.set("history", std::vector<double>{4., 5.});
  data.config_json = R"({"physical":{"T":0.1}})";

  write_snapshot(path, data);
  REQUIRE(std::filesystem::exists(path));
  REQUIRE_FALSE(std::filesystem::exists(path.string() + ".tmp"));

  const auto read = read_snapshot(path);
  REQUIRE(read.t == data.t);
  REQUIRE(read.k == data.k);
  REQUIRE(read.Lambda == data.Lambda);
  REQUIRE(read.dim == data.dim);
  REQUIRE(std::isnan(read.last_adaptation_time));
  REQUIRE(read.spatial.fe_name == data.spatial.fe_name);
  REQUIRE(read.spatial.dofs_per_cell == data.spatial.dofs_per_cell);
  REQUIRE(read.spatial.n_dofs == data.spatial.n_dofs);
  REQUIRE(read.spatial.active_cells == data.spatial.active_cells);
  REQUIRE(read.spatial.values == data.spatial.values);
  REQUIRE(read.variables == data.variables);
  REQUIRE(read.model.get<bool>("lock"));
  REQUIRE(read.model.get_vector("history") == std::vector<double>{4., 5.});
  REQUIRE(json::parse(read.config_json) == json::parse(data.config_json));

  REQUIRE_THROWS(read_snapshot(root.run_file("_does_not_exist", ".h5")));
}

TEST_CASE("Cell-wise state restores into a fresh discretization", "[snapshot]")
{
  DiFfRG::Init();
  const auto config = make_config();
  SECTION("CG")
  {
    using Model = Testing::ModelExp<2>;
    Model model(linear_profile());
    using Discretization = CG::Discretization<Model, RectangularMesh<2>>;
    require_roundtrip<Discretization, CG::Assembler<Discretization>, FE::FlowingVariables<Discretization>>(
        config, model, false);
  }
  SECTION("DG")
  {
    using Model = Testing::ModelExp<2>;
    Model model(linear_profile());
    using Discretization = DG::Discretization<Model, RectangularMesh<2>>;
    require_roundtrip<Discretization, DG::Assembler<Discretization>, FE::FlowingVariables<Discretization>>(
        config, model, false);
  }
  SECTION("LDG")
  {
    using Model = Testing::LDGModelConstant<1>;
    Model model(linear_profile());
    // LDG is serial-only; a plain RectangularMesh is partitioned in an MPI build.
    using Discretization = LDG::Discretization<Model, RectangularMeshSerial<1>>;
    require_roundtrip<Discretization, LDG::Assembler<Discretization>, FE::FlowingVariables<Discretization>>(
        config, model, false);
  }
  SECTION("FV")
  {
    using Model = Testing::ModelBurgersKT<1>;
    Model model(linear_profile());
    using Discretization = FV::Discretization<Model, RectangularMesh<1>>;
    require_roundtrip<Discretization, FV::KurganovTadmor::Assembler<Discretization, Model>,
                      FV::FlowingVariables<Discretization>>(config, model, false);
  }
}

TEST_CASE("An adapted mesh is rebuilt from its cell ids", "[snapshot]")
{
  DiFfRG::Init();
  const auto config = make_config();
  SECTION("CG, with hanging nodes")
  {
    using Model = Testing::ModelExp<2>;
    Model model(linear_profile());
    using Discretization = CG::Discretization<Model, RectangularMesh<2>>;
    require_roundtrip<Discretization, CG::Assembler<Discretization>, FE::FlowingVariables<Discretization>>(config,
                                                                                                           model, true);
  }
  SECTION("DG")
  {
    using Model = Testing::ModelExp<2>;
    Model model(linear_profile());
    using Discretization = DG::Discretization<Model, RectangularMesh<2>>;
    require_roundtrip<Discretization, DG::Assembler<Discretization>, FE::FlowingVariables<Discretization>>(config,
                                                                                                           model, true);
  }
  SECTION("A finer current mesh is coarsened back")
  {
    RectangularMesh<2> mesh{Config::ConfigurationMesh<2>(config)};
    auto &triangulation = mesh.get_triangulation();
    std::vector<std::string> coarse_cells;
    for (const auto &cell : triangulation.active_cell_iterators())
      coarse_cells.push_back(cell->id().to_string());
    triangulation.refine_global(2);
    REQUIRE(DiFfRG::internal::restore_active_cells(triangulation, triangulation.n_cells(0), coarse_cells));
    REQUIRE(triangulation.n_active_cells() == coarse_cells.size());
    REQUIRE_FALSE(DiFfRG::internal::restore_active_cells(triangulation, triangulation.n_cells(0), coarse_cells));
  }
  SECTION("A different coarse mesh is rejected")
  {
    RectangularMesh<2> mesh{Config::ConfigurationMesh<2>(config)};
    REQUIRE_THROWS(DiFfRG::internal::restore_active_cells(mesh.get_triangulation(), 3, {"0_0:"}));
  }
}

TEST_CASE("A snapshot of a different finite element is rejected", "[snapshot]")
{
  DiFfRG::Init();
  using Model = Testing::ModelExp<2>;
  Model model(linear_profile());
  using Discretization = CG::Discretization<Model, RectangularMesh<2>>;

  const auto config = make_config(2);
  RectangularMesh<2> mesh{Config::ConfigurationMesh<2>(config)};
  Discretization source(mesh, config);
  CG::Assembler<Discretization> source_assembler(source, model, config);
  FE::FlowingVariables<Discretization> state(source);
  state.interpolate(model);
  const auto captured = capture_replicated(source, source_assembler, state.spatial_data());

  const auto other_config = make_config(1);
  RectangularMesh<2> other_mesh{Config::ConfigurationMesh<2>(other_config)};
  Discretization other(other_mesh, other_config);
  CG::Assembler<Discretization> other_assembler(other, model, other_config);
  FE::FlowingVariables<Discretization> other_state(other);
  other_state.interpolate(model);
  REQUIRE_THROWS_WITH(other_assembler.restore_snapshot_state(captured, other_state.spatial_data()),
                      Catch::Matchers::ContainsSubstring("finite element"));
}

#if defined(DEAL_II_WITH_MPI) && defined(DEAL_II_WITH_PETSC)

TEST_CASE("The cell-wise layout does not depend on the rank count", "[snapshot][mpi]")
{
  DiFfRG::Init();
  using namespace dealii;
  using Model = Testing::ModelExp<2>;
  Model model(linear_profile());
  using SerialDisc = CG::Discretization<Model, RectangularMeshSerial<2>>;
  using ParallelDisc = CG::Discretization<Model, RectangularMeshParallel<2>>;
  using ParVector = typename ParallelDisc::VectorType;
  const auto config = make_config();

  // Serial reference on every rank.
  RectangularMeshSerial<2> serial_mesh{Config::ConfigurationMesh<2>(config)};
  SerialDisc serial(serial_mesh, config);
  CG::Assembler<SerialDisc> serial_assembler(serial, model, config);
  FE::FlowingVariables<SerialDisc> serial_state(serial);
  serial_state.interpolate(model);
  const auto serial_capture =
      DiFfRG::internal::capture_cellwise_state(serial.get_dof_handler(), serial_state.spatial_data());

  // The same field on the partitioned mesh, whose dof numbering depends on the rank count.
  RectangularMeshParallel<2> parallel_mesh{Config::ConfigurationMesh<2>(config)};
  ParallelDisc parallel(parallel_mesh, config);
  CG::Assembler<ParallelDisc> parallel_assembler(parallel, model, config);
  FE::FlowingVariables<ParallelDisc> parallel_state(parallel);
  parallel_state.interpolate(model);

  SolutionView<ParVector> view;
  parallel_assembler.reinit_solution_view(view);
  view.refresh(parallel_state.spatial_data());
  const auto parallel_capture = DiFfRG::internal::capture_cellwise_state(parallel.get_dof_handler(), view.get());

  // Written with N ranks, it is the serial layout.
  REQUIRE(parallel_capture.active_cells == serial_capture.active_cells);
  REQUIRE(max_difference(parallel_capture.values, serial_capture.values) < 1e-14);

  // And a serial snapshot restores into the partitioned state.
  parallel_state.spatial_data() = 0.;
  parallel_assembler.restore_snapshot_state(serial_capture, parallel_state.spatial_data());
  view.refresh(parallel_state.spatial_data());
  const auto restored = DiFfRG::internal::capture_cellwise_state(parallel.get_dof_handler(), view.get());
  REQUIRE(max_difference(restored.values, serial_capture.values) == 0.);
}

#endif
