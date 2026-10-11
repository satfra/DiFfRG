#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

#include <cmath>
#include <string>

using namespace DiFfRG;

// The IDA hybrids keep spatial states of their own (the bracket and checkpoints the explicit part interpolates in);
// a mesh adaptation has to move them to the new mesh together with IDA's state.

namespace
{
  constexpr uint dim = 1;

  /// du/dt = -K u, dv/dt = -u (see ModelHybridSmooth), with a refinement indicator that grows with x, so that
  /// HAdaptivity really refines.
  class Model : public Testing::ModelHybridSmooth<dim>
  {
  public:
    using Testing::ModelHybridSmooth<dim>::ModelHybridSmooth;

    template <int mdim, typename NumberType, typename Solution>
    void cell_indicator(NumberType &indicator, const Point<mdim> &p, const Solution &) const
    {
      indicator = p[0];
    }
  };

  using Discretization = CG::Discretization<Model, RectangularMeshSerial<dim>>;
  using Assembler = CG::Assembler<Discretization>;

  ConfigTree make_config()
  {
    return json::value(
        {{"physical", {{"Lambda", 1.}}},
         {"discretization",
          {{"fe_order", 1},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1e-12},
           {"EoM_max_iter", 100},
           {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 0.1},
             {"level", 2},
             {"refine_percent", 0.3},
             {"coarsen_percent", 0.}}}}},
         {"timestepping",
          {{"final_time", 0.5},
           {"output_dt", 0.05},
           {"explicit",
            {{"dt", 1e-3}, {"minimal_dt", 1e-8}, {"maximal_dt", 1e-2}, {"abs_tol", 1e-10}, {"rel_tol", 1e-8}}},
           // IDA continues at its step size across adaptations, so this alone bounds the coupling error of the
           // hybrids (IDA + RK54 misses 1e-5 in v at 5e-2, with or without adaptation).
           {"implicit",
            {{"dt", 1e-3}, {"minimal_dt", 1e-8}, {"maximal_dt", 5e-3}, {"abs_tol", 1e-10}, {"rel_tol", 1e-8}}}}},
         {"output", {{"verbosity", 0}, {"vtk", false}}}});
  }

  template <template <typename> typename TimeStepperFor> void require_adapted_run_is_accurate(const std::string &name)
  {
    const auto config = make_config();
    const auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, name);
    Testing::PhysicalParameters p_prm{};
    Model model(p_prm);
    RectangularMeshSerial<dim> mesh{Config::ConfigurationMesh<dim>(config)};
    Discretization discretization(mesh, config);
    Assembler assembler(discretization, model, config);
    OutputSession<Assembler> data_out(path, config);
    HAdaptivity adaptor(assembler, config);
    TimeStepperFor<Assembler> stepper(config, assembler, data_out, adaptor);
    FE::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    const auto cells_before = discretization.get_triangulation().n_active_cells();

    const double t_end = config.get_double("/timestepping/final_time");
    stepper.run(state, 0., t_end);

    REQUIRE(discretization.get_triangulation().n_active_cells() > cells_before);
    const double u_exact = Model::u_exact(t_end);
    const double v_exact = Model::v0 + Model::u0 / Model::K * (std::exp(-Model::K * t_end) - 1.);
    for (const double u : state.spatial_data())
      REQUIRE_THAT(u, Catch::Matchers::WithinRel(u_exact, 1e-5));
    REQUIRE_THAT(state.variable_data()[0], Catch::Matchers::WithinAbs(v_exact, 1e-5));
  }

  template <typename Assembler> using IDA_ABM = TimeStepperSUNDIALS_IDA_BoostABM<Assembler>;
  template <typename Assembler> using IDA_RK = TimeStepperSUNDIALS_IDA_BoostRK54<Assembler>;
} // namespace

TEST_CASE("IDA hybrids survive mesh adaptation", "[timestepping][sundials_ida_boost][adaptivity]")
{
  SECTION("IDA, variables implicit") { require_adapted_run_is_accurate<TimeStepperSUNDIALS_IDA>("hybrid_adapt_ida"); }
  SECTION("IDA + ABM") { require_adapted_run_is_accurate<IDA_ABM>("hybrid_adapt_abm"); }
  SECTION("IDA + RK54") { require_adapted_run_is_accurate<IDA_RK>("hybrid_adapt_rk"); }
}
