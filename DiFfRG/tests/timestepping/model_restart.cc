#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/solution_view.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/discretization/mesh/h_adaptivity.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/timestepping/timestepping.hh>

using namespace DiFfRG;

// A flow that is discontinuous in time at a point the model chooses (see HasNeedsRestart): u and v grow linearly,
// and once an accepted state has u >= u_c the model switches both rates and asks for a restart. With the switch at
// t_s the exact solution is piecewise linear,
//     u(t) = u0 + a t_s + b (t - t_s),   v(t) = v0 + alpha t_s + beta (t - t_s),
// which both IDA and the Adams steps integrate exactly -- unless a step history reaches across the switch.
class ModelSwitch
    : public def::AbstractModel<ModelSwitch,
                                ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<Scalar<"v">>>>,
      public def::Time,
      public def::NoNumFlux<ModelSwitch>,
      public def::FlowBoundaries<ModelSwitch>,
      public def::AD<ModelSwitch>
{
public:
  static constexpr uint dim = 1;
  static constexpr bool batch_reads_derivatives = false;
  static constexpr double u0 = 0., v0 = 1., u_c = 0.37;
  static constexpr double a = 1., b = -3., alpha = 2., beta = 0.5;

  ModelSwitch(Testing::PhysicalParameters) {}

  template <typename Vector> void initial_condition(const Point<dim> &, Vector &values) const { values[0] = u0; }
  template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = v0; }

  // u_dot = -source
  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<dim> &, const Solution &) const
  {
    s_i[0] = switched ? -b : -a;
  }
  // v_dot = -r_a
  template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &) const
  {
    r_a[0] = switched ? -beta : -alpha;
  }

  template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector &) const
  {
    return {{x[0] - 0.5}};
  }

  template <int d, typename NT>
  bool needs_restart(const double t_accepted, const Point<d> &, const SolutionSample<d, NT> &sample) const
  {
    monotone &= t_accepted >= last_t;
    last_t = t_accepted;
    if (switched || double(sample[0].values[0]) < u_c) return false;
    switched = true;
    t_switch = t_accepted;
    ++restarts;
    return true;
  }

  double u_exact(const double tt) const { return u0 + a * t_switch + b * (tt - t_switch); }
  double v_exact(const double tt) const { return v0 + alpha * t_switch + beta * (tt - t_switch); }

  mutable bool switched = false, monotone = true;
  mutable double t_switch = -1., last_t = -1.;
  mutable uint restarts = 0;
};

template <template <typename...> typename Stepper> void run_switch(const std::string &name)
{
  DiFfRG::Init();
  using Discretization = CG::Discretization<ModelSwitch, RectangularMesh<1>>;
  using Assembler = CG::Assembler<Discretization>;
  constexpr double final_time = 1., implicit_max_dt = 2e-2;

  ConfigTree json = json::value(
      {{"physical", {{"Lambda", 1.}}},
       {"discretization",
        {{"fe_order", 1},
         {"overintegration", 0},
         {"output_subdivisions", 1},
         {"EoM_abs_tol", 1e-12},
         {"EoM_max_iter", 100},
         {"grid", {{"x_grid", "0:0.25:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
         {"adaptivity", {{"start_adapt_at", 0.}, {"adapt_dt", 1e-1}, {"level", 0}}}}},
       {"timestepping",
        {{"final_time", final_time},
         {"output_dt", 0.1},
         {"explicit", {{"dt", 1e-2}, {"minimal_dt", 1e-7}, {"maximal_dt", 1e-2}, {"abs_tol", 1e-12}, {"rel_tol", 1e-10}}},
         {"implicit",
          {{"dt", 1e-3}, {"minimal_dt", 1e-9}, {"maximal_dt", implicit_max_dt}, {"abs_tol", 1e-12}, {"rel_tol", 1e-10}}}}},
       {"output", {{"verbosity", 0}, {"vtk", false}}}});

  ModelSwitch model(Testing::PhysicalParameters{});
  Discretization::Mesh mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);
  auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, name);
  OutputSession<Assembler> data_out(path, json);
  HAdaptivity mesh_adaptor(assembler, json);
  Stepper<Assembler> stepper(json, assembler, data_out, mesh_adaptor);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  REQUIRE_NOTHROW(stepper.run(initial_condition, 0., final_time));

  // One switch, at the first accepted step past the crossing, seen only at accepted (non-decreasing) times.
  CHECK(model.restarts == 1);
  CHECK(model.monotone);
  const double t_crossing = (ModelSwitch::u_c - ModelSwitch::u0) / ModelSwitch::a;
  CHECK(model.t_switch >= t_crossing - 1e-10);
  CHECK(model.t_switch <= t_crossing + implicit_max_dt);

  SolutionView<Discretization::VectorType> u_view, v_view;
  assembler.reinit_solution_view(u_view);
  u_view.refresh(initial_condition.data().block(0));
  DiFfRG::internal::reinit_variables_view(v_view, 1, assembler.get_communicator());
  v_view.refresh(initial_condition.data().block(1));
  for (uint i = 0; i < u_view.get().size(); ++i)
    CHECK(is_close(u_view[i], model.u_exact(final_time), 1e-8));
  CHECK(is_close(v_view[0], model.v_exact(final_time), 1e-8));
}

template <typename Assembler> using IDA = TimeStepperSUNDIALS_IDA<Assembler>;
template <typename Assembler> using IDA_ABM = TimeStepperSUNDIALS_IDA_BoostABM<Assembler>;

TEST_CASE("A model's needs_restart restarts IDA at the discontinuity", "[timestepping][restart][sundials_ida]")
{
  run_switch<IDA>("test_model_restart_ida");
}

TEST_CASE("A model's needs_restart restarts IDA and the Adams steps at the discontinuity",
          "[timestepping][restart][sundials_ida_boost][abm]")
{
  run_switch<IDA_ABM>("test_model_restart_ida_abm");
}
