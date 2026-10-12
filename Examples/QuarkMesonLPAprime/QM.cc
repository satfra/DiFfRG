#include <DiFfRG/DiFfRG.hh>

#include "model.hh"

using namespace DiFfRG;

int main(int argc, char *argv[])
{
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  const auto config = config_helper.get_config();

  using Model = QuarkMesonLPAprime;
  constexpr uint dim = Model::dim;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;
  using Assembler = CG::Assembler<Discretization>;
  // IDA for the potential; the couplings ZPhi and hPhi are stepped explicitly, so their momentum integrals are
  // evaluated once per explicit step instead of in every Newton iteration.
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<Assembler>;

  Model model(config);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(config)};
  OutputSession<Assembler> data_out(config);
  const auto log = data_out.report_port();
  Discretization discretization(mesh, config, log);
  Assembler assembler(discretization, model, config);
  TimeStepper time_stepper(config, assembler, data_out);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  Timer timer;
  try {
    time_stepper.run(initial_condition, 0., config.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Simulation finished with exception {}", e.what());
    return -1;
  }
  log.info("Simulation finished after " + time_format(timer.wall_time()));
  return 0;
}
