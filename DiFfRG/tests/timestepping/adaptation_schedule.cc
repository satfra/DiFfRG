#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace DiFfRG;

// The IDA steppers ask the mesh adaptor at multiples of its adaptation_period(), not only at output times: with
// output_dt much longer than adapt_dt, adaptations would otherwise be delayed to the next output.

class ModelLinear
    : public def::AbstractModel<ModelLinear,
                                ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<Scalar<"v">>>>,
      public def::Time,
      public def::NoNumFlux<ModelLinear>,
      public def::FlowBoundaries<ModelLinear>,
      public def::AD<ModelLinear>
{
public:
  static constexpr uint dim = 1;
  static constexpr bool batch_reads_derivatives = false;

  ModelLinear(Testing::PhysicalParameters) {}

  template <typename Vector> void initial_condition(const Point<dim> &, Vector &values) const { values[0] = 0.; }
  template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = 1.; }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<dim> &, const Solution &) const
  {
    s_i[0] = -1.;
  }
  template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &) const
  {
    r_a[0] = -0.5;
  }
};

/// Never changes the mesh; records the times at which it is asked.
template <typename VectorType> class RecordingAdaptor : public AbstractAdaptor<VectorType>
{
public:
  explicit RecordingAdaptor(const double period) : period(period) {}
  virtual bool operator()(const double t, VectorType &) override
  {
    times.push_back(t);
    return false;
  }
  virtual bool adapt(VectorType &) override { return false; }
  virtual double adaptation_period() const override { return period; }

  const double period;
  std::vector<double> times;
};

template <template <typename...> typename Stepper> void check_schedule(const std::string &name)
{
  DiFfRG::Init();
  using Discretization = CG::Discretization<ModelLinear, RectangularMesh<1>>;
  using Assembler = CG::Assembler<Discretization>;
  constexpr double final_time = 1., output_dt = 0.5, period = 0.1;

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
         {"output_dt", output_dt},
         {"explicit", {{"dt", 1e-2}, {"minimal_dt", 1e-7}, {"maximal_dt", 1e-2}, {"abs_tol", 1e-12}, {"rel_tol", 1e-10}}},
         {"implicit",
          {{"dt", 1e-3}, {"minimal_dt", 1e-9}, {"maximal_dt", 5e-2}, {"abs_tol", 1e-12}, {"rel_tol", 1e-10}}}}},
       {"output", {{"verbosity", 0}, {"vtk", false}}}});

  ModelLinear model(Testing::PhysicalParameters{});
  Discretization::Mesh mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);
  auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, name);
  OutputSession<Assembler> data_out(path, json);
  RecordingAdaptor<typename Discretization::VectorType> adaptor(period);
  Stepper<Assembler> stepper(json, assembler, data_out, adaptor);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  REQUIRE_NOTHROW(stepper.run(initial_condition, 0., final_time));

  // Asked at every multiple of the period, once each, the output times included.
  std::vector<double> expected;
  for (int n = 1; n <= 10; ++n)
    expected.push_back(n * period);
  REQUIRE(adaptor.times.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
    CHECK(is_close(adaptor.times[i], expected[i], 1e-10));
}

template <typename Assembler> using IDA = TimeStepperSUNDIALS_IDA<Assembler>;
template <typename Assembler> using IDA_ABM = TimeStepperSUNDIALS_IDA_BoostABM<Assembler>;

TEST_CASE("IDA asks the mesh adaptor at every adaptation period, between output times",
          "[timestepping][adaptivity][sundials_ida]")
{
  check_schedule<IDA>("test_adaptation_schedule_ida");
}

TEST_CASE("IDA with Adams variables asks the mesh adaptor at every adaptation period, between output times",
          "[timestepping][adaptivity][sundials_ida_boost][abm]")
{
  check_schedule<IDA_ABM>("test_adaptation_schedule_ida_abm");
}
