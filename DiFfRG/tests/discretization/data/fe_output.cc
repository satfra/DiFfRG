#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>

#include <filesystem>
#include <fstream>
#include <iterator>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test FE output on Constant model", "[output][cg]")
{
  DiFfRG::Init();

  using namespace dealii;
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = CG::Assembler<Discretization>;

  ConfigTree json = json::value(
      {{"physical", {}},
       {"integration",
        {{"x_quadrature_order", 32},
         {"angle_quadrature_order", 8},
         {"x0_quadrature_order", 16},
         {"x0_summands", 8},
         {"q0_quadrature_order", 16},
         {"q0_summands", 8},
         {"x_extent_tolerance", 1e-3},
         {"x0_extent_tolerance", 1e-3},
         {"q0_extent_tolerance", 1e-3},
         {"jacobian_quadrature_factor", 0.5}}},
       {"discretization",
        {{"fe_order", 3},
         {"overintegration", 0},
         {"output_subdivisions", 2},

         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},

         {"grid", {{"x_grid", "0:0.0001:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
         {"adaptivity",
          {{"start_adapt_at", 0.},
           {"adapt_dt", 1e-1},
           {"level", 0},
           {"refine_percent", 1e-1},
           {"coarsen_percent", 5e-2}}}}},
       {"timestepping",
        {{"final_time", 1.},
         {"output_dt", 1e-1},
         {"explicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}},
         {"implicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}}}},
       {"output", {{"live_plot", false}, {"verbosity", 0}}}});

  Testing::PhysicalParameters p_prm = {/*x0_initial = */ 0., /*x1_initial = */ GENERATE(take(1, random(1., 10.)))};

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
    log->info("DiFfRG Application started");
  } catch (const spdlog::spdlog_ex &e) {
    // nothing, the logger is already set up
  }

  // Define the objects needed to run the simulation
  Model model(p_prm);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  // Set up the initial condition
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  const VectorType &src = initial_condition.spatial_data();

  Timer timer;
  {
    FEOutput<dim, VectorType> fe_output("./testing", "output_name", "other_folder", Config::OutputSettings(json));

    HDF5Output hdf5_output("./testing/", "output_name.h5", Config::OutputSettings(json).configuration_json);
    auto root_group = hdf5_output.get_file().root();
    root_group.create_group("FE");
    fe_output.set_hdf5_output(&hdf5_output);

    constexpr uint output_num = 20;
    for (uint i = 0; i < output_num; ++i) {
      fe_output.attach(discretization.get_dof_handler(), src, "solution");
      fe_output.flush(i);
    }
    std::cout << "FEOutput flushed " << output_num << " times in " << timer.wall_time() << " seconds." << std::endl;
  }
  std::cout << "FEOutput finished after " << timer.wall_time() << " seconds." << std::endl;
}

// With /output/vtk off the worker thread has nothing to do -- its whole body is guarded on
// save_vtk -- so spawning and joining one per frame is pure overhead. That is exactly the
// configuration an HDF5-only cluster run uses, which is why it is worth pinning.
TEST_CASE("FEOutput spawns no writer thread when VTK is disabled", "[output][cg][lifecycle]")
{
  DiFfRG::Init();

  using namespace dealii;
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;

  ConfigTree json = json::value({{"physical", {}},
                                 {"discretization",
                                  {{"fe_order", 1},
                                   {"overintegration", 0},
                                   {"output_subdivisions", 1},
                                   {"EoM_abs_tol", 1e-10},
                                   {"EoM_max_iter", 0},
                                   {"grid", {{"x_grid", "0:0.25:1"}, {"y_grid", "0:1:1"}, {"z_grid", "0:1:1"}}}}},
                                 {"output", {{"verbosity", 0}, {"vtk", false}}}});

  Testing::PhysicalParameters p_prm = {0., 1.};
  Model model(p_prm);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  auto output_path = OutputPath::temporary();
  FEOutput<dim, VectorType> fe_output(output_path.root().string(), "novtk", "fields", Config::OutputSettings(json));
  REQUIRE(fe_output.will_discard()); // no VTK and no HDF5 sink attached

  for (uint i = 0; i < 5; ++i) {
    fe_output.attach(discretization.get_dof_handler(), initial_condition.spatial_data(), "u");
    fe_output.flush(static_cast<double>(i));
    CHECK(fe_output.pending_workers() == 0);
  }
  CHECK_NOTHROW(fe_output.finish());
  CHECK_FALSE(std::filesystem::exists(output_path.root() / "novtk.pvd"));
}

// Regression test for the nondeterministic timestepper-test crashes (SIGSEGV / "Could not
// write pvd file."). Pre-fix, multiple writer threads spawned by flush() race on
// time_series and on the shared .pvd path, which manifests as undefined behaviour and as
// std::terminate via an uncaught exception escaping the worker thread. This test rapidly
// flushes many times at the default async buffer size so the race is reliably triggered.
TEST_CASE("FEOutput async drain preserves reuse and publication order", "[output][cg][regression][lifecycle]")
{
  DiFfRG::Init();

  using namespace dealii;
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;

  ConfigTree json =
      json::value({{"physical", {}},
                   {"discretization",
                    {{"fe_order", 3},
                     {"overintegration", 0},
                     {"output_subdivisions", 2},
                     {"EoM_abs_tol", 1e-10},
                     {"EoM_max_iter", 0},
                     {"grid", {{"x_grid", "0:0.05:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
                     {"adaptivity",
                      {{"start_adapt_at", 0.},
                       {"adapt_dt", 1e-1},
                       {"level", 0},
                       {"refine_percent", 1e-1},
                       {"coarsen_percent", 5e-2}}}}},
                   {"output", {{"live_plot", false}, {"verbosity", 0}}}});

  Testing::PhysicalParameters p_prm = {/*x0_initial = */ 0., /*x1_initial = */ 1.};

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
  } catch (const spdlog::spdlog_ex &) {
    // logger already set up
  }

  Model model(p_prm);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  const VectorType &src = initial_condition.spatial_data();

  constexpr uint output_num = 200;
  auto output_path = OutputPath::temporary();
  const auto output_root = output_path.root();

  REQUIRE_NOTHROW([&]() {
    FEOutput<dim, VectorType> fe_output(output_root.string(), "stress_output", "vtu", Config::OutputSettings(json));
    HDF5Output hdf5_output(output_root.string(), "stress_output.h5", Config::OutputSettings(json).configuration_json);
    auto root_group = hdf5_output.get_file().root();
    root_group.create_group("FE");
    fe_output.set_hdf5_output(&hdf5_output);

    for (uint i = 0; i < output_num / 2; ++i) {
      fe_output.attach(discretization.get_dof_handler(), src, "solution");
      fe_output.flush(static_cast<double>(i));
    }
    fe_output.drain();

    for (uint i = output_num / 2; i < output_num; ++i) {
      fe_output.attach(discretization.get_dof_handler(), src, "solution");
      fe_output.flush(static_cast<double>(i));
    }
    fe_output.drain();
  }());

  CHECK(std::filesystem::exists(output_root / "vtu/stress_output_000000.vtu"));
  CHECK(std::filesystem::exists(output_root / "vtu/stress_output_000199.vtu"));

  std::ifstream pvd(output_root / "stress_output.pvd");
  REQUIRE(pvd.good());
  const std::string pvd_contents((std::istreambuf_iterator<char>(pvd)), std::istreambuf_iterator<char>());
  const auto first_frame = pvd_contents.find("stress_output_000000.vtu");
  const auto resumed_frame = pvd_contents.find("stress_output_000100.vtu");
  const auto last_frame = pvd_contents.find("stress_output_000199.vtu");
  REQUIRE(first_frame != std::string::npos);
  REQUIRE(resumed_frame != std::string::npos);
  REQUIRE(last_frame != std::string::npos);
  CHECK(first_frame < resumed_frame);
  CHECK(resumed_frame < last_frame);
}
