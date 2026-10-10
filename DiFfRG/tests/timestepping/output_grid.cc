#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

#include <DiFfRG/discretization/data/hdf5_output.hh>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace DiFfRG;

// Steppers whose own steps do not land on the output grid interpolate their frames onto it.

namespace
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>>;
  using Assembler = DG::Assembler<Discretization>;

  ConfigTree make_config(const double final_time)
  {
    return json::value(
        {{"physical", {{"Lambda", 1.}}},
         {"discretization",
          {{"fe_order", 1},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1e-12},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1e-1},
             {"level", 0},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 0.}}}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", 0.1},
           {"explicit",
            {{"dt", 7e-3}, {"minimal_dt", 1e-7}, {"maximal_dt", 1e-2}, {"abs_tol", 1e-8}, {"rel_tol", 1e-8}}},
           {"implicit",
            {{"dt", 7e-3}, {"minimal_dt", 1e-7}, {"maximal_dt", 3e-2}, {"abs_tol", 1e-9}, {"rel_tol", 1e-6}}}}},
         {"output", {{"verbosity", 0}, {"vtk", false}}}});
  }

  template <template <typename> typename TimeStepperFor>
  std::vector<double> frame_times(const std::string &name, const double final_time)
  {
    const auto config = make_config(final_time);
    const auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, name);
    {
      Testing::PhysicalParameters p_prm{};
      Model model(p_prm);
      RectangularMeshSerial<dim> mesh{Config::ConfigurationMesh<dim>(config)};
      Discretization discretization(mesh, config);
      Assembler assembler(discretization, model, config);
      OutputSession<Assembler> data_out(path, config);
      TimeStepperFor<Assembler> stepper(config, assembler, data_out);
      FE::FlowingVariables<Discretization> state(discretization);
      state.interpolate(model);
      stepper.run(state, 0., config.get_double("/timestepping/final_time"));
    }
    // Each frame is a group /FE/<n> with its time as attribute.
    auto file = hdf5::File::open(path.run_file(".h5").string(), hdf5::Access::ReadOnly);
    auto fe = file.root().open_group("FE");
    auto names = fe.child_names();
    std::sort(names.begin(), names.end());
    std::vector<double> times;
    for (const auto &frame : names)
      times.push_back(fe.open_group(frame).read_attribute<double>("time"));
    return times;
  }

  template <template <typename> typename TimeStepperFor> void require_frames_on_grid(const std::string &name)
  {
    const auto times = frame_times<TimeStepperFor>(name, 0.3);
    REQUIRE(times.size() == 4);
    for (unsigned i = 0; i < times.size(); ++i)
      REQUIRE_THAT(times[i], Catch::Matchers::WithinAbs(0.1 * i, 1e-12));

    // A final time off the grid gets a frame of its own.
    const auto off_grid = frame_times<TimeStepperFor>(name + "_off", 0.35);
    REQUIRE(off_grid.size() == 5);
    REQUIRE_THAT(off_grid.back(), Catch::Matchers::WithinAbs(0.35, 1e-12));
  }

  template <typename Assembler> using ImplicitEuler = TimeStepperImplicitEuler<Assembler>;
  template <typename Assembler> using TRBDF2 = TimeStepperTRBDF2<Assembler>;
  template <typename Assembler> using ExplicitEuler = TimeStepperExplicitEuler<Assembler>;
} // namespace

TEST_CASE("Frames are written on the output grid", "[timestepping][output]")
{
  SECTION("ImplicitEuler") { require_frames_on_grid<ImplicitEuler>("grid_ie"); }
  SECTION("TRBDF2") { require_frames_on_grid<TRBDF2>("grid_trbdf2"); }
  SECTION("ExplicitEuler") { require_frames_on_grid<ExplicitEuler>("grid_ee"); }
}
