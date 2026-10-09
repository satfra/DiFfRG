#include "model.hh"

/**
 * Runs the flow of model.hh with the CG or the KT assembler (/tutorial/assembler = cg or kt) and evaluates the
 * momentum integrals per point, or batched on the CPU or the GPU (/tutorial/backend = per_point, cpu or gpu).
 */
template <typename Model, typename Discretization, typename Assembler> int run(const ConfigTree &config)
{
  constexpr uint dim = Model::dim;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

  Model model(config);
  typename Discretization::Mesh mesh{Config::ConfigurationMesh<dim>(config)};
  OutputSession<Assembler> data_out(config);
  const auto log = data_out.report_port();
  Discretization discretization(mesh, config, log);
  Assembler assembler(discretization, model, config);
  NoAdaptivity mesh_adaptor(assembler);
  TimeStepper time_stepper(config, assembler, data_out, mesh_adaptor);

  constexpr bool is_kt = requires { typename Assembler::Reconstructor; };
  auto initial_condition = [&] {
    if constexpr (is_kt)
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

  // Where the time of one residual and one jacobian went, averaged over the run.
  for (const auto &[name, times] : {std::pair{"residual", assembler.residual_phase_times()},
                                    std::pair{"jacobian", assembler.jacobian_phase_times()}})
    if (times.calls > 0)
      log.info("{} ({} calls), ms per call: extract {:.3f}, gather {:.3f}, evaluate {:.3f}, scatter {:.3f}", name,
               times.calls, 1e3 * times.extract / times.calls, 1e3 * times.gather / times.calls,
               1e3 * times.evaluate / times.calls, 1e3 * times.scatter / times.calls);
  return 0;
}

int main(int argc, char *argv[])
{
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  const auto config = config_helper.get_config();

  const auto with_backend = [&]<Backend backend>() {
    using CGModel = ON_CG<backend>;
    using CGDiscretization = CG::Discretization<CGModel, RectangularMesh<1>>;
    using KTModel = ON_KT<backend>;
    using KTDiscretization = FV::Discretization<KTModel, RectangularMesh<1>>;
    using Reconstructor = def::TVDReconstructor<1, def::MinModLimiter, double>;
    const std::string assembler = config.get_string("/tutorial/assembler");
    if (assembler == "cg") return run<CGModel, CGDiscretization, CG::Assembler<CGDiscretization>>(config);
    if (assembler == "kt")
      return run<KTModel, KTDiscretization, FV::KurganovTadmor::Assembler<KTDiscretization, KTModel, Reconstructor>>(
          config);
    std::cerr << "Unknown /tutorial/assembler: " << assembler << std::endl;
    return 1;
  };

  const std::string backend = config.get_string("/tutorial/backend");
  if (backend == "per_point") return with_backend.template operator()<Backend::per_point>();
  if (backend == "cpu") return with_backend.template operator()<Backend::cpu>();
  if (backend == "gpu") return with_backend.template operator()<Backend::gpu>();
  std::cerr << "Unknown /tutorial/backend: " << backend << std::endl;
  return 1;
}
