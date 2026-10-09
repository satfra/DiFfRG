#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "model_batched.hh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <optional>
#include <sstream>

/**
 * Times residual() and jacobian() of the ONfiniteT assembly (--assembler cg, dg, ldg or kt) for one configuration
 * of the flux:
 *   B0  -- per-point flux with the TBB integrator (AbstractModel's default evaluate_batch)
 *   B1  -- evaluate_batch with one map_points call on TBB
 *   G64 -- evaluate_batch with one map_points call on the GPU, double precision
 *   G32 -- evaluate_batch with one map_points call on the GPU, single precision
 *
 * Usage: bench_batched --config B1 --cells 1024 --xorder 32 [--assembler cg] [--reps 10] [--threads 0]
 *                      [--policy auto] [--parameters parameter.toml] [--time 1]
 * Prints one line "RESULT,<csv>" with the header given by --header: the median and mean wall time of a call, then
 * per call the assembler's own stages (extract, gather, evaluate, scatter; see AssemblyPhaseTimes) and what the
 * stages leave of the mean ("other"), each for the residual and the jacobian.
 */
namespace
{
  struct Options {
    std::string config = "B1";
    std::string assembler = "cg";
    uint cells = 64;
    uint xorder = 32;
    uint reps = 10;
    uint threads = 0;
    std::string policy = "auto";
    std::string parameters = "parameter.toml";
    double time = 1.;
  };

  template <typename Model> using CGDiscretization = CG::Discretization<Model, RectangularMesh<Model::dim>>;
  template <typename Model> using DGDiscretization = DG::Discretization<Model, RectangularMesh<Model::dim>>;
  template <typename Model> using LDGDiscretization = LDG::Discretization<Model, RectangularMeshSerial<Model::dim>>;
  template <typename Model> using KTDiscretization = FV::Discretization<Model, RectangularMesh<Model::dim>>;
  template <typename Discretization>
  using KTAssembler = FV::KurganovTadmor::Assembler<Discretization, typename Discretization::Model, def::TVDReconstructor<1, def::MinModLimiter, double>>;

  double median(std::vector<double> v)
  {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  }

  double mean(const std::vector<double> &v) { return std::accumulate(v.begin(), v.end(), 0.) / v.size(); }

  MapPointsPolicy parse_policy(const std::string &p)
  {
    if (p == "thread") return MapPointsPolicy::thread_per_point;
    if (p == "team") return MapPointsPolicy::team_per_point;
    return MapPointsPolicy::automatic;
  }

  template <typename Model, template <typename> typename Discretization_, template <typename> typename Assembler_> void run(const Options &opt, const ConfigTree &config)
  {
    constexpr uint dim = Model::dim;
    using Discretization = Discretization_<Model>;
    using Assembler = Assembler_<Discretization>;
    using VectorType = typename Discretization::VectorType;
    using SparseMatrixType = typename Discretization::SparseMatrixType;
    using clock = std::chrono::steady_clock;

    Model model(config);
    model.set_time(opt.time);
    // Every number type of every integral the model evaluates (the KT model has two, and its flux jacobian runs
    // through the second-order AD integrator).
    const auto set_policy = [&](auto &flow) {
      flow.integrator.set_map_points_policy(parse_policy(opt.policy));
      flow.integrator_AD.set_map_points_policy(parse_policy(opt.policy));
      if constexpr (requires { flow.integrator_AD2; }) flow.integrator_AD2.set_map_points_policy(parse_policy(opt.policy));
    };
    if constexpr (requires { model.sigma_integrator(); }) {
      set_policy(model.pion_integrator());
      set_policy(model.sigma_integrator());
    } else
      set_policy(model.integrator());

    typename Discretization::Mesh mesh{Config::ConfigurationMesh<dim>(config)};
    Discretization discretization(mesh, config);
    Assembler assembler(discretization, model, config);

    auto state = [&] {
      if constexpr (requires { typename Assembler::Reconstructor; })
        return FV::FlowingVariables(discretization);
      else
        return FE::FlowingVariables(discretization);
    }();
    state.interpolate(model);
    const VectorType &u = state.spatial_data();
    const VectorType u_dot(u);
    VectorType residual(u);
    SparseMatrixType jacobian(assembler.get_sparsity_pattern_jacobian());

    const auto time_residual = [&] {
      residual = 0;
      const auto t0 = clock::now();
      assembler.residual(residual, u, 1., u_dot, 1.);
      return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };
    const auto time_jacobian = [&] {
      jacobian = 0;
      const auto t0 = clock::now();
      assembler.jacobian(jacobian, u, 1., u_dot, 1., 1.);
      return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };

    for (int i = 0; i < 3; ++i) {
      time_residual();
      time_jacobian();
    }
    assembler.reset_phase_times();

    std::vector<double> t_res, t_jac;
    for (uint i = 0; i < opt.reps; ++i)
      t_res.push_back(time_residual());
    for (uint i = 0; i < opt.reps; ++i)
      t_jac.push_back(time_jacobian());

    // Per call, in ms: the four stages and what they leave of the mean call time.
    const auto stages = [](const AssemblyPhaseTimes &t, const double mean_ms) {
      const double per_call = 1e3 / t.calls;
      return std::array<double, 5>{t.extract * per_call, t.gather * per_call, t.evaluate * per_call, t.scatter * per_call, mean_ms - t.total() * per_call};
    };
    const auto res_stages = stages(assembler.residual_phase_times(), mean(t_res));
    const auto jac_stages = stages(assembler.jacobian_phase_times(), mean(t_jac));

    const size_t n_points = discretization.get_triangulation().n_active_cells() * (discretization.get_fe().degree + 1);
    std::ostringstream line;
    line << std::setprecision(6) << "RESULT," << opt.assembler << "," << opt.config << "," << opt.cells << "," << opt.xorder << "," << n_points << ","
         << (opt.threads == 0 ? DiFfRG::n_threads() : opt.threads) << "," << opt.policy << "," << median(t_res) << "," << median(t_jac) << "," << mean(t_res) << "," << mean(t_jac);
    for (const double p : res_stages)
      line << "," << p;
    for (const double p : jac_stages)
      line << "," << p;
    line << std::setprecision(15) << "," << residual.l2_norm() << "," << jacobian.frobenius_norm();
    std::cout << line.str() << std::endl;
  }
} // namespace

int main(int argc, char *argv[])
{
  Options opt;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i], value = argv[i + 1];
    if (key == "--config")
      opt.config = value;
    else if (key == "--cells")
      opt.cells = std::stoul(value);
    else if (key == "--xorder")
      opt.xorder = std::stoul(value);
    else if (key == "--reps")
      opt.reps = std::stoul(value);
    else if (key == "--threads")
      opt.threads = std::stoul(value);
    else if (key == "--policy")
      opt.policy = value;
    else if (key == "--time")
      opt.time = std::stod(value);
    else if (key == "--assembler")
      opt.assembler = value;
    else if (key == "--parameters")
      opt.parameters = value;
    else {
      std::cerr << "Unknown option " << key << std::endl;
      return 1;
    }
  }
  if (argc == 2 && std::strcmp(argv[1], "--header") == 0) {
    std::cout << "assembler,config,cells,xorder,points,threads,policy,residual_ms,jacobian_ms,residual_mean_ms,jacobian_mean_ms,"
                 "res_extract_ms,res_gather_ms,res_evaluate_ms,res_scatter_ms,res_other_ms,jac_extract_ms,jac_gather_ms,"
                 "jac_evaluate_ms,jac_scatter_ms,jac_other_ms,residual_norm,jacobian_norm"
              << std::endl;
    return 0;
  }

  // Must outlive Init and the assembler: TBB and Kokkos pick up the active parallelism at Init.
  std::optional<tbb::global_control> thread_limit;
  if (opt.threads > 0) thread_limit.emplace(tbb::global_control::max_allowed_parallelism, opt.threads);
  char *init_argv[] = {argv[0]};
  DiFfRG::Init(1, init_argv);

  ConfigTree config(opt.parameters);
  std::ostringstream step;
  step << std::setprecision(17) << 0.015 / opt.cells;
  config.set_string("/discretization/grid/x_grid", "0:" + step.str() + ":0.015");
  config.set_uint("/discretization/grid/refine", 0);
  config.set_uint("/integration/x_order", opt.xorder);
  config.set_uint("/output/verbosity", 0);

  using namespace ON_batched;
  const auto run_config = [&]<template <Backend, bool> typename M, template <typename> typename D, template <typename> typename A>() {
    if (opt.config == "B0")
      run<M<Backend::TBB, false>, D, A>(opt, config);
    else if (opt.config == "B1")
      run<M<Backend::TBB, true>, D, A>(opt, config);
    else if (opt.config == "G64")
      run<M<Backend::GPU, true>, D, A>(opt, config);
    else if (opt.config == "G32")
      run<M<Backend::GPU_float, true>, D, A>(opt, config);
    else
      return false;
    return true;
  };
  bool known = false;
  if (opt.assembler == "cg")
    known = run_config.template operator()<Model, CGDiscretization, CG::Assembler>();
  else if (opt.assembler == "dg")
    known = run_config.template operator()<Model, DGDiscretization, DG::Assembler>();
  else if (opt.assembler == "ldg")
    known = run_config.template operator()<ModelLDG, LDGDiscretization, LDG::Assembler>();
  else if (opt.assembler == "kt")
    known = run_config.template operator()<ModelKT, KTDiscretization, KTAssembler>();
  if (!known) {
    std::cerr << "Unknown --assembler " << opt.assembler << " or --config " << opt.config << std::endl;
    return 1;
  }
  return 0;
}
