#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "model_batched.hh"

/**
 * CG.cc / DG.cc / LDG.cc / KT.cc with a batched flux. /batched/assembler is cg, dg, ldg or kt (the latter with
 * parameter_KT.toml); /batched/backend selects how the model evaluates the flux:
 *   per_point -- per-point flux with the TBB integrator (AbstractModel's default evaluate_batch)
 *   tbb       -- one map_points call with the TBB integrator
 *   gpu       -- one map_points call with the GPU integrator in double precision
 *   gpu_float -- one map_points call with the GPU integrator in single precision
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
  constexpr bool is_kt = requires { typename Assembler::Reconstructor; };
  // KT requires a fixed rectangular mesh; no h-adaptivity.
  auto mesh_adaptor = [&] {
    if constexpr (is_kt)
      return NoAdaptivity(assembler);
    else
      return HAdaptivity(assembler, config);
  }();
  TimeStepper time_stepper(config, assembler, data_out, mesh_adaptor);

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
  return 0;
}

int main(int argc, char *argv[])
{
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  const auto config = config_helper.get_config();

  using namespace ON_batched;
  const std::string assembler = config.get_string("/batched/assembler", "cg");
  // b = backend: Backend and whether the model evaluates its flux batched.
  const auto with_assembler = [&]<Backend b, bool batched>() {
    using M = Model<b, batched>;
    using ML = ModelLDG<b, batched>;
    using CGD = CG::Discretization<M, RectangularMesh<M::dim>>;
    using DGD = DG::Discretization<M, RectangularMesh<M::dim>>;
    using LDGD = LDG::Discretization<ML, RectangularMeshSerial<ML::dim>>;
    using MK = ModelKT<b, batched>;
    using KTD = FV::Discretization<MK, RectangularMesh<MK::dim>>;
    using KTA = FV::KurganovTadmor::Assembler<KTD, MK, def::TVDReconstructor<1, def::MinModLimiter, double>>;
    if (assembler == "cg") return run<M, CGD, CG::Assembler<CGD>>(config);
    if (assembler == "dg") return run<M, DGD, DG::Assembler<DGD>>(config);
    if (assembler == "ldg") return run<ML, LDGD, LDG::Assembler<LDGD>>(config);
    if (assembler == "kt") return run<MK, KTD, KTA>(config);
    std::cerr << "Unknown /batched/assembler: " << assembler << std::endl;
    return 1;
  };
  const std::string backend = config.get_string("/batched/backend", "tbb");
  if (backend == "per_point") return with_assembler.template operator()<Backend::TBB, false>();
  if (backend == "tbb") return with_assembler.template operator()<Backend::TBB, true>();
  if (backend == "gpu") return with_assembler.template operator()<Backend::GPU, true>();
  if (backend == "gpu_float") return with_assembler.template operator()<Backend::GPU_float, true>();
  std::cerr << "Unknown /batched/backend: " << backend << std::endl;
  return 1;
}
