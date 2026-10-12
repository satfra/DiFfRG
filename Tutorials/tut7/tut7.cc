#include "model.hh"

/**
 * Runs the O(N) flow of model.hh with the discretization chosen by /tutorial/discretization (cg, dg or kt), with
 * h-adaptivity controlled by /discretization/adaptivity.
 */
template <typename Model, typename Discretization, typename Assembler> int run(const ConfigTree &config)
{
  constexpr uint dim = Model::dim;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

  Model model(config);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(config)};
  OutputSession<Assembler> data_out(config);
  const auto log = data_out.report_port();
  Discretization discretization(mesh, config, log);
  Assembler assembler(discretization, model, config);
  HAdaptivity mesh_adaptor(assembler, config);
  TimeStepper time_stepper(config, assembler, data_out, mesh_adaptor);

  // FE and FV discretizations store their solution differently.
  auto initial_condition = [&] {
    if constexpr (std::is_same_v<Model, ON_KT>)
      return FV::FlowingVariables(discretization);
    else
      return FE::FlowingVariables(discretization);
  }();
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

int main(int argc, char *argv[])
{
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  const auto config = config_helper.get_config();

  using CGDiscretization = CG::Discretization<ON_FEM, RectangularMesh<1>>;
  using DGDiscretization = DG::Discretization<ON_FEM, RectangularMesh<1>>;
  using KTDiscretization = FV::Discretization<ON_KT, RectangularMesh<1>>;
  using Reconstructor = def::TVDReconstructor<1, def::MinModLimiter, double>;

  const std::string discretization = config.get_string("/tutorial/discretization");
  if (discretization == "cg") return run<ON_FEM, CGDiscretization, CG::Assembler<CGDiscretization>>(config);
  if (discretization == "dg") return run<ON_FEM, DGDiscretization, DG::Assembler<DGDiscretization>>(config);
  if (discretization == "kt")
    return run<ON_KT, KTDiscretization, FV::KurganovTadmor::Assembler<KTDiscretization, ON_KT, Reconstructor>>(config);
  std::cerr << "Unknown /tutorial/discretization: " << discretization << std::endl;
  return 1;
}
