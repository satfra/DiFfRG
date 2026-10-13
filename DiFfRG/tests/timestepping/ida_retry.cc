#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include <stdexcept>

using namespace DiFfRG;

// IDA restarts once, cold, at its last accepted state when a step fails (IDADriver::solve_dae), and throws on a second
// failure before the next stop time. The model here changes its flow history-dependently: u_dot = a until it is first
// evaluated at t >= t_c, b from then on, also for earlier t. IDA's step history then no longer fits the residual, and
// with a minimal step size its error test cannot recover -- the situation of a hybrid stepper whose explicit variables
// were updated behind IDA's back. A cold restart computes u_dot afresh and continues.
class ModelFlip
    : public def::AbstractModel<ModelFlip, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::NoNumFlux<ModelFlip>,
      public def::FlowBoundaries<ModelFlip>,
      public def::AD<ModelFlip>
{
public:
  static constexpr uint dim = 1;
  static constexpr double a = 1., b = -2., t_c = 0.3;
  /// alternate: after t_c, switch the flow at every evaluation, which no restart can follow.
  bool alternate = false;

  ModelFlip(Testing::PhysicalParameters) {}

  template <typename Vector> void initial_condition(const Point<dim> &, Vector &values) const { values[0] = 0.; }

  // u_dot = -source
  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<dim> &, const Solution &) const
  {
    if (t >= t_c) switched = alternate ? !switched : true;
    s_i[0] = switched ? -b : -a;
  }

  mutable bool switched = false;
};

static void run(const bool alternate, const std::string &name)
{
  DiFfRG::Init();
  using Discretization = CG::Discretization<ModelFlip, RectangularMesh<1>>;
  using Assembler = CG::Assembler<Discretization>;
  constexpr double final_time = 1.;

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
         {"implicit",
          {{"dt", 1e-3}, {"minimal_dt", 1e-4}, {"maximal_dt", 5e-2}, {"abs_tol", 1e-12}, {"rel_tol", 1e-10}}}}},
       {"output", {{"verbosity", 0}, {"vtk", false}}}});

  ModelFlip model(Testing::PhysicalParameters{});
  model.alternate = alternate;
  Discretization::Mesh mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);
  auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, name);
  OutputSession<Assembler> data_out(path, json);
  TimeStepperSUNDIALS_IDA<Assembler> stepper(json, assembler, data_out);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  if (alternate) {
    CHECK_THROWS_AS(stepper.run(initial_condition, 0., final_time), std::runtime_error);
    return;
  }
  REQUIRE_NOTHROW(stepper.run(initial_condition, 0., final_time));
  // From the restart at some t_r in [t_c - 2 maximal_dt, t_c] on, u_dot = b: u(1) = a t_r + b (1 - t_r), which grows
  // with t_r.
  const auto u_of = [](const double t_r) { return ModelFlip::a * t_r + ModelFlip::b * (1. - t_r); };
  const double u_end = initial_condition.data().block(0)[0];
  CHECK(u_end >= u_of(ModelFlip::t_c - 1e-1) - 1e-8);
  CHECK(u_end <= u_of(ModelFlip::t_c) + 1e-8);
}

TEST_CASE("IDA retries a failed step once from a cold restart", "[timestepping][sundials_ida][retry]")
{
  run(false, "test_ida_retry");
}

TEST_CASE("IDA gives up when the retry fails too", "[timestepping][sundials_ida][retry]")
{
  run(true, "test_ida_retry_fail");
}
