#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include <algorithm>
#include <mutex>
#include <cmath>
#include <utility>
#include <vector>

using namespace DiFfRG;

// In the stagger coupling (/timestepping/explicit/coupling_mode = 1) IDA is served the explicit variables by
// extrapolation until the next explicit step is taken, and by interpolation after. Where v bends strongly the two
// differ, and switching from one to the other is a jump in the served v(t) that IDA's step history cannot follow. The
// served v(t) must stay continuous: here v relaxes fast, v_dot = lambda (1 - v), with explicit steps of lambda^-1, and
// every value the flow of u is served is recorded.
class ModelRelax
    : public def::AbstractModel<ModelRelax, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>,
                                                                VariableDescriptor<Scalar<"v">>,
                                                                ExtractorDescriptor<Scalar<"v_eom">>>>,
      public def::Time,
      public def::NoNumFlux<ModelRelax>,
      public def::FlowBoundaries<ModelRelax>,
      public def::AD<ModelRelax>
{
public:
  static constexpr uint dim = 1;
  static constexpr bool batch_reads_derivatives = false;
  static constexpr double lambda = 10.;

  ModelRelax(Testing::PhysicalParameters) {}

  template <typename Vector> void initial_condition(const Point<dim> &, Vector &values) const { values[0] = 0.; }
  template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = 0.; }

  template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector &) const
  {
    return {{x[0] - 0.5}};
  }

  template <typename NT, typename Solution>
  void extract(std::array<NT, 1> &extractors, const Point<dim> &, const Solution &sol) const
  {
    extractors[0] = get<"variables">(sol)[0];
  }

  // u_dot = v
  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<dim> &, const Solution &sol) const
  {
    const double v = double(get<"extractors">(sol)[0]);
    const std::lock_guard<std::mutex> lock(served_mutex);
    served.emplace_back(t, v);
    s_i[0] = -v;
  }

  // v_dot = lambda (1 - v)
  template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &sol) const
  {
    r_a[0] = -lambda * (1. - get<"variables">(sol)[0]);
  }

  mutable std::vector<std::pair<double, double>> served;
  mutable std::mutex served_mutex; // source() runs on several threads
};

TEST_CASE("The variables served to IDA in the stagger coupling are continuous in time",
          "[timestepping][sundials_ida_boost][abm]")
{
  DiFfRG::Init();
  using Discretization = CG::Discretization<ModelRelax, RectangularMesh<1>>;
  using Assembler = CG::Assembler<Discretization>;
  constexpr double final_time = 1., explicit_dt = 0.1;

  ConfigTree json = json::value(
      {{"physical", {{"Lambda", 1.}}},
       {"discretization",
        {{"fe_order", 1},
         {"overintegration", 0},
         {"output_subdivisions", 1},
         {"EoM_abs_tol", 1e-12},
         {"EoM_max_iter", 100},
         {"grid", {{"x_grid", "0:0.25:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}}}},
       {"timestepping",
        {{"final_time", final_time},
         {"output_dt", 0.5},
         {"explicit",
          {{"dt", explicit_dt},
           {"minimal_dt", explicit_dt},
           {"maximal_dt", explicit_dt},
           {"abs_tol", 1e-2},
           {"rel_tol", 1e-2},
           {"coupling_mode", 1}}},
         {"implicit",
          {{"dt", 1e-3}, {"minimal_dt", 1e-9}, {"maximal_dt", 2e-2}, {"abs_tol", 1e-10}, {"rel_tol", 1e-8}}}}},
       {"output", {{"verbosity", 0}, {"vtk", false}}}});

  ModelRelax model(Testing::PhysicalParameters{});
  Discretization::Mesh mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);
  auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "test_served_continuity",
                                    "test_served_continuity");
  OutputSession<Assembler> data_out(path, json);
  TimeStepperSUNDIALS_IDA_BoostABM<Assembler> stepper(json, assembler, data_out);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  REQUIRE_NOTHROW(stepper.run(initial_condition, 0., final_time));

  // No jumps: between neighbouring served times, v changes at most as fast as the exact v(t) = 1 - exp(-lambda t)
  // does at t = 0, with room for the extrapolation error.
  auto served = model.served;
  std::sort(served.begin(), served.end());
  REQUIRE(served.size() > 100);
  double worst = 0.;
  for (size_t i = 1; i < served.size(); ++i) {
    const double dt = served[i].first - served[i - 1].first, dv = std::abs(served[i].second - served[i - 1].second);
    worst = std::max(worst, dv - 2. * ModelRelax::lambda * dt);
  }
  CHECK(worst < 1e-6);
}
