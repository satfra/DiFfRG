#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace DiFfRG;

// Timesteppers on a pure-variables system (no FE space): dv/dt = -lambda v.

namespace
{
  struct Record {
    std::vector<double> frame_times;
    unsigned residual_calls = 0;
  };

  /// dv/dt = -lambda v; records the time of every output frame and counts dt_variables calls.
  class ModelDecay
      : public def::AbstractModel<
            ModelDecay,
            ComponentDescriptor<FEFunctionDescriptor<>, VariableDescriptor<Scalar<"v">>, ExtractorDescriptor<>>>,
        public def::Time,
        public def::AD<ModelDecay>
  {
  public:
    ModelDecay(const double lambda, std::shared_ptr<Record> record) : lambda(lambda), record(std::move(record)) {}

    template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = 1.; }

    // The timesteppers solve dv/dt = -dt_variables.
    template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &data) const
    {
      if constexpr (std::is_same_v<std::decay_t<decltype(residual[0])>, double>) record->residual_calls++;
      residual[0] = lambda * get<"variables">(data)[0];
    }

    template <int dim, typename DataOut, typename Solutions>
    void readouts(DataOut &, const Point<dim> &, const Solutions &) const
    {
      record->frame_times.push_back(t);
    }

  private:
    double lambda;
    std::shared_ptr<Record> record;
  };

  ConfigTree make_config(const double final_time, const double output_dt)
  {
    return json::value(
        {{"physical", {{"Lambda", 1.}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", output_dt},
           {"explicit",
            {{"dt", 1e-2}, {"minimal_dt", 1e-10}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-10}, {"rel_tol", 1e-10}}},
           {"implicit",
            {{"dt", 1e-3}, {"minimal_dt", 1e-10}, {"maximal_dt", 1.}, {"abs_tol", 1e-10}, {"rel_tol", 1e-8}}}}},
         {"output", {{"verbosity", 0}, {"vtk", false}}}});
  }

  template <template <typename> typename TimeStepperFor>
  double run_decay(const std::string &name, const double lambda, const ConfigTree &config,
                   std::shared_ptr<Record> record)
  {
    using Assembler = Variables::Assembler<ModelDecay>;
    ModelDecay model(lambda, record);
    Assembler assembler(model, config);
    const auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, name);
    OutputSession<Assembler> data_out(path, config);
    TimeStepperFor<Assembler> time_stepper(config, assembler, data_out);
    FlowingVariables state;
    state.interpolate(model);
    time_stepper.run(state, 0., config.get_double("/timestepping/final_time"));
    return state.variable_data()[0];
  }

  template <typename Assembler> using IDA = TimeStepperSUNDIALS_IDA<Assembler>;
  template <typename Assembler> using RK54 = TimeStepperBoostRK54<Assembler>;
  template <typename Assembler> using ABM = TimeStepperBoostABM<Assembler>;
  template <typename Assembler> using RK4 = TimeStepperRK<Assembler>;

  /// 0.3 / 0.1 rounds down to 2 in floating point, the case that used to lose the last frame.
  template <template <typename> typename TimeStepperFor> void require_all_frames(const std::string &name)
  {
    auto record = std::make_shared<Record>();
    const double v = run_decay<TimeStepperFor>(name, 1., make_config(0.3, 0.1), record);
    REQUIRE_THAT(v, Catch::Matchers::WithinRel(std::exp(-0.3), 1e-6));
    REQUIRE(record->frame_times.size() == 4);
    for (unsigned i = 0; i < 4; ++i)
      REQUIRE_THAT(record->frame_times[i], Catch::Matchers::WithinAbs(0.1 * i, 1e-12));
  }
} // namespace

TEST_CASE("Explicit variable steppers write the frame at the final time", "[timestepping][variables]")
{
  SECTION("BoostRK54") { require_all_frames<RK54>("frames_rk54"); }
  SECTION("BoostABM") { require_all_frames<ABM>("frames_abm"); }
  SECTION("RK4") { require_all_frames<RK4>("frames_rk4"); }
  SECTION("IDA") { require_all_frames<IDA>("frames_ida"); }
}

TEST_CASE("IDA solves a stiff variable equation with few residual evaluations",
          "[timestepping][variables][sundials_ida]")
{
  // With lambda = 1e4 the step size is limited by accuracy, not stability, only if the Newton
  // matrix alpha + d(dt_variables)/dv has the right sign.
  auto record = std::make_shared<Record>();
  const double v = run_decay<IDA>("stiff_ida", 1e4, make_config(1., 0.5), record);
  INFO("residual calls: " << record->residual_calls);
  REQUIRE(std::abs(v) < 1e-6);
  REQUIRE(record->residual_calls < 2000);
}
