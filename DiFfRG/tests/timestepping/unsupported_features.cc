#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

#include <string>

using namespace DiFfRG;

// Steppers that cannot integrate variables, or cannot adapt the mesh, must refuse instead of
// silently ignoring them.

namespace
{
  constexpr uint dim = 1;
  using Discretization = CG::Discretization<Testing::ModelHybridSmooth<dim>, RectangularMeshSerial<dim>>;
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
             {"adapt_dt", 1e-1},
             {"level", 0},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 0.}}}}},
         {"timestepping",
          {{"final_time", 0.1},
           {"output_dt", 5e-2},
           {"explicit",
            {{"dt", 1e-3}, {"minimal_dt", 1e-7}, {"maximal_dt", 1e-2}, {"abs_tol", 1e-8}, {"rel_tol", 1e-8}}},
           {"implicit",
            {{"dt", 1e-3}, {"minimal_dt", 1e-7}, {"maximal_dt", 5e-2}, {"abs_tol", 1e-8}, {"rel_tol", 1e-8}}}}},
         {"output", {{"verbosity", 0}, {"vtk", false}}}});
  }

  struct Setup {
    ConfigTree config = make_config();
    Testing::PhysicalParameters p_prm{};
    Testing::ModelHybridSmooth<dim> model{p_prm};
    RectangularMeshSerial<dim> mesh{Config::ConfigurationMesh<dim>(config)};
    Discretization discretization{mesh, config};
    Assembler assembler{discretization, model, config};
    OutputPath path;
    OutputSession<Assembler> data_out;
    FE::FlowingVariables<Discretization> state{discretization};

    explicit Setup(const std::string &name)
        : path(OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, name)), data_out(path, config)
    {
      state.interpolate(model);
    }
  };

  template <template <typename> typename TimeStepperFor> void require_refuses_variables(const std::string &name)
  {
    Setup s(name);
    REQUIRE(s.state.variable_data().size() > 0);
    TimeStepperFor<Assembler> stepper(s.config, s.assembler, s.data_out);
    REQUIRE_THROWS_AS(stepper.run(s.state, 0., 0.1), std::invalid_argument);
  }

  template <typename Assembler> using ImplicitEuler = TimeStepperImplicitEuler<Assembler>;
  template <typename Assembler> using TRBDF2 = TimeStepperTRBDF2<Assembler>;
  template <typename Assembler> using ExplicitEuler = TimeStepperExplicitEuler<Assembler>;
} // namespace

TEST_CASE("Steppers without variable support refuse a flow with variables", "[timestepping]")
{
  SECTION("ImplicitEuler") { require_refuses_variables<ImplicitEuler>("refuse_ie"); }
  SECTION("TRBDF2") { require_refuses_variables<TRBDF2>("refuse_trbdf2"); }
  SECTION("ExplicitEuler") { require_refuses_variables<ExplicitEuler>("refuse_ee"); }
}

TEST_CASE("Steppers without mesh adaptation refuse an adaptor", "[timestepping]")
{
  Setup s("refuse_adaptor");
  HAdaptivity adaptor(s.assembler, s.config);
  REQUIRE_THROWS_AS(TimeStepperBoostRK54<Assembler>(s.config, s.assembler, s.data_out, adaptor), std::invalid_argument);
  REQUIRE_THROWS_AS(TimeStepperBoostABM<Assembler>(s.config, s.assembler, s.data_out, adaptor), std::invalid_argument);

  NoAdaptivity<typename Assembler::VectorType> no_adaptor;
  REQUIRE_NOTHROW(TimeStepperBoostRK54<Assembler>(s.config, s.assembler, s.data_out, no_adaptor));
}
