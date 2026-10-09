// FV / Kurganov-Tadmor headers must be included before any header that pulls in
// DiFfRG::Quadrature<NT> (e.g. DiFfRG/physics/integration.hh via DiFfRG/DiFfRG.hh),
// otherwise unqualified Quadrature<dim> inside KurganovTadmor.hh resolves to the
// DiFfRG class template instead of dealii::Quadrature<int>.
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>
#include <DiFfRG/model/fv_boundaries.hh>

#include "model_KT_sigma.hh"

// Choices for types
using Model = ON_finiteT_KT_sigma;
constexpr uint dim = Model::dim;
using Discretization = FV::Discretization<Model, RectangularMesh<dim>>;
using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
using WaveSpeed = FV::KurganovTadmor::MaxEigenvalueWaveSpeed;
using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor, WaveSpeed>;
using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

int main(int argc, char *argv[])
{
  // Initialize DiFfRG and thus the MPI and Kokkos environments.
  const auto config_helper = DiFfRG::Init(argc, argv, "parameter_KT_sigma.toml").get_configuration_helper();
  const auto config = config_helper.get_config();

  // Define the objects needed to run the simulation
  Model model(config);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(config)};
  OutputSession<Assembler> data_out(config);
  const auto log = data_out.report_port();
  Discretization discretization(mesh, config, log);
  Assembler assembler(discretization, model, config);
  NoAdaptivity mesh_adaptor(assembler);
  TimeStepper time_stepper(config, assembler, data_out, mesh_adaptor);

  // Set up the initial condition
  FV::FlowingVariables<Discretization> initial_condition(discretization);
  initial_condition.interpolate(model);

  Timer timer;
  try {
    time_stepper.run(initial_condition, 0., config.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Simulation finished with exception {}", e.what());
    return -1;
  }
  auto time = timer.wall_time();
  log.info("Simulation finished after " + time_format(time));
  return 0;
}
