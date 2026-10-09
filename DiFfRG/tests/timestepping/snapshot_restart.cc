#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

#include <DiFfRG/discretization/data/snapshot.hh>
#include <DiFfRG/timestepping/timestep_control/pi.hh>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace DiFfRG;

// These tests pin down what a restart from a flow snapshot promises:
//
//   * A run that writes a snapshot at t_s and a second run restarted from that snapshot end in
//     the same state. Both restart the solver at t_s -- the first because the flow is split into
//     segments there, the second because it starts there -- so this holds to rounding, not merely
//     to the time-stepping tolerance. That is only true if everything the continued flow depends
//     on is in the snapshot: the state, the mesh, the model's history-dependent state and the
//     adaptation schedule.
//   * Splitting a flow into segments does not change its result beyond the solver tolerance.
//
// Each run builds its own mesh, discretization, assembler and output session, exactly like an
// application does, so no state can leak from the seed run into the restarted one.

namespace
{
  /**
   * @brief du/dt = r u, where the rate r doubles once the flow has been evaluated at a time in
   * (0.3, 0.4). That latch is history: a restart at t = 0.5 can not recompute it from the state
   * or from t, so it only continues the right flow if the model state travels with the snapshot.
   */
  template <uint dim, bool restorable = true>
  class ModelLatch
      : public def::AbstractModel<ModelLatch<dim, restorable>, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
        public def::Time,
        public def::NoNumFlux<ModelLatch<dim, restorable>>,
        public def::FlowBoundaries<ModelLatch<dim, restorable>>,
        public def::AD<ModelLatch<dim, restorable>>
  {
  public:
    ModelLatch(const Testing::PhysicalParameters prm) : prm(prm) {}

    void set_time(const double t_)
    {
      def::Time::set_time(t_);
      if (t_ > 0.3 && t_ < 0.4) latched = true;
    }

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, 1> &s_i, const Point<dim> & /*p*/, const Solution &sol) const
    {
      s_i[0] = -(latched ? 2. : 1.) * get<0>(sol)[0];
    }

    /// Steers h-adaptivity towards large x, so that an adaptive run really changes its mesh.
    template <int d, typename NumberType, typename Solution>
    void cell_indicator(NumberType &indicator, const Point<d> &p, const Solution & /*sol*/) const
    {
      indicator = p[0] * p[0];
    }

    void save_state(ModelState &state) const { state.set("latched", latched); }
    void load_state(const ModelState &state)
      requires restorable
    {
      latched = state.get<bool>("latched");
    }

    bool latched = false;

  private:
    const Testing::PhysicalParameters prm;
  };

  Testing::PhysicalParameters linear_profile()
  {
    Testing::PhysicalParameters prm;
    prm.initial_x0[0] = 0.5;
    prm.initial_x1[0] = 1.;
    return prm;
  }

  ConfigTree base_config(const double final_time = 1.)
  {
    return json::value(
        {{"physical", {{"Lambda", 1.}}},
         {"discretization",
          {{"fe_order", 1},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1e-12},
           {"EoM_max_iter", 100},
           {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1e-1},
             {"level", 0},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 0.}}}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", 5e-2},
           {"explicit",
            {{"dt", 1e-3},
             {"minimal_dt", 1e-7},
             {"maximal_dt", 1e-2},
             {"abs_tol", 1e-12},
             {"rel_tol", 1e-10},
             {"coupling_mode", 1}}},
           {"implicit",
            {{"dt", 1e-3}, {"minimal_dt", 1e-7}, {"maximal_dt", 5e-2}, {"abs_tol", 1e-12}, {"rel_tol", 1e-10}}}}},
         {"output", {{"verbosity", 0}, {"vtk", false}}}});
  }

  ConfigTree with_snapshots(ConfigTree config, const std::string &snapshots_json)
  {
    config().as_object()["timestepping"].as_object()["snapshots"] = json::parse(snapshots_json);
    return config;
  }

  ConfigTree with_restart(ConfigTree config, const std::filesystem::path &file)
  {
    config().as_object()["restart"] = json::value({{"file", file.string()}});
    return config;
  }

  struct FlowState {
    SnapshotSpatialState initial;
    SnapshotSpatialState spatial;
    std::vector<double> variables;
  };

  void ensure_logger()
  {
    try {
      auto log = spdlog::stdout_color_mt("log");
      log->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
      // already set up
    }
  }

  /// One complete application run: build everything, interpolate, run(0, final_time).
  template <typename Model, typename Discretization, typename Assembler, template <typename> typename TimeStepperFor,
            bool adapt = false>
  FlowState run_flow(const ConfigTree &config, const OutputPath &path, const double t_start = 0.)
  {
    ensure_logger();
    constexpr uint dim = Discretization::dim;
    using VectorType = typename Discretization::VectorType;

    Model model(linear_profile());
    // Serial on purpose: this tests serial-only steppers among others;
    // a plain RectangularMesh is partitioned in an MPI build.
    RectangularMeshSerial<dim> mesh{Config::ConfigurationMesh<dim>(config)};
    Discretization discretization(mesh, config);
    Assembler assembler(discretization, model, config);
    OutputSession<Assembler> data_out(path, config);
    std::unique_ptr<AbstractAdaptor<VectorType>> adaptor;
    if constexpr (adapt)
      adaptor = std::make_unique<HAdaptivity<Assembler>>(assembler, config);
    else
      adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    TimeStepperFor<Assembler> time_stepper(config, assembler, data_out, *adaptor);

    FlowingVariablesFor<Discretization> state(discretization);
    state.interpolate(model);

    FlowState result;
    result.initial = DiFfRG::internal::capture_cellwise_state(discretization.get_dof_handler(), state.spatial_data());
    time_stepper.run(state, t_start, config.get_double("/timestepping/final_time"));
    result.spatial = DiFfRG::internal::capture_cellwise_state(discretization.get_dof_handler(), state.spatial_data());
    if (state.data().n_blocks() > 1)
      for (std::size_t i = 0; i < state.variable_data().size(); ++i)
        result.variables.push_back(state.variable_data()[i]);
    return result;
  }

  /// Largest difference between two states, matched cell by cell (a rebuilt mesh may order its
  /// cells differently); fails if they do not live on the same mesh.
  double max_difference(const SnapshotSpatialState &a, const SnapshotSpatialState &b)
  {
    REQUIRE(a.active_cells.size() == b.active_cells.size());
    REQUIRE(a.dofs_per_cell == b.dofs_per_cell);
    std::map<std::string, std::size_t> position;
    for (std::size_t c = 0; c < b.active_cells.size(); ++c)
      position[b.active_cells[c]] = c;
    double worst = 0.;
    for (std::size_t c = 0; c < a.active_cells.size(); ++c) {
      const auto it = position.find(a.active_cells[c]);
      REQUIRE(it != position.end());
      for (unsigned int i = 0; i < a.dofs_per_cell; ++i)
        worst =
            std::max(worst, std::abs(a.values[c * a.dofs_per_cell + i] - b.values[it->second * a.dofs_per_cell + i]));
    }
    return worst;
  }

  double max_abs(const SnapshotSpatialState &a)
  {
    double m = 0.;
    for (const double v : a.values)
      m = std::max(m, std::abs(v));
    return m;
  }

  /// For du/dt = u every dof grows like e^t, so the exact final state is the initial one scaled.
  double max_error_exponential(const FlowState &state, const double elapsed)
  {
    double worst = 0.;
    for (std::size_t i = 0; i < state.spatial.values.size(); ++i)
      worst = std::max(worst, std::abs(state.spatial.values[i] - std::exp(elapsed) * state.initial.values[i]));
    return worst;
  }

  /// dv/dt = v without any FE space (dim = 0).
  class ModelVariableExp
      : public def::AbstractModel<ModelVariableExp, ComponentDescriptor<FEFunctionDescriptor<>,
                                                                        VariableDescriptor<Scalar<"v">>,
                                                                        ExtractorDescriptor<>>>,
        public def::Time,
        public def::NoJacobians
  {
  public:
    template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = 1.; }
    template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &data) const
    {
      residual[0] = -get<"variables">(data)[0];
    }
  };

  template <template <typename> typename TimeStepperFor>
  std::vector<double> run_variables_flow(const ConfigTree &config, const OutputPath &path)
  {
    ensure_logger();
    using Assembler = Variables::Assembler<ModelVariableExp>;
    ModelVariableExp model;
    Assembler assembler(model, config);
    OutputSession<Assembler> data_out(path, config);
    TimeStepperFor<Assembler> time_stepper(config, assembler, data_out);
    FlowingVariables state;
    state.interpolate(model);
    time_stepper.run(state, 0., config.get_double("/timestepping/final_time"));
    return {state.variable_data().begin(), state.variable_data().end()};
  }

  /// Seed with a snapshot at t = 0.5, restart from it, and require both runs to end in the same state.
  template <typename Model, typename Disc, typename Asm, template <typename> typename TimeStepperFor>
  void require_restart_reproduces(const std::string &name, const double tolerance = 1e-12)
  {
    const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, name);
    const auto config = base_config();
    const auto seed = run_flow<Model, Disc, Asm, TimeStepperFor>(with_snapshots(config, R"({"t": [0.5]})"), root);
    const auto restarted = run_flow<Model, Disc, Asm, TimeStepperFor>(
        with_restart(config, root.run_file("_snapshot_000", ".h5")), root.child("restart", "restart"));
    REQUIRE(max_difference(seed.spatial, restarted.spatial) <= tolerance * max_abs(seed.spatial));
    REQUIRE(seed.variables.size() == restarted.variables.size());
    for (std::size_t i = 0; i < seed.variables.size(); ++i)
      REQUIRE_THAT(restarted.variables[i], Catch::Matchers::WithinAbs(seed.variables[i], tolerance));
  }

  template <typename Model> using DGDisc = DG::Discretization<Model, RectangularMeshSerial<1>>;
  template <typename Model> using DGAsm = DG::Assembler<DGDisc<Model>>;
  template <typename Assembler> using IDA = TimeStepperSUNDIALS_IDA<Assembler>;
  template <typename Assembler> using IDA_ABM = TimeStepperSUNDIALS_IDA_BoostABM<Assembler>;
  template <typename Assembler> using RK54 = TimeStepperBoostRK54<Assembler>;
  template <typename Assembler> using IDA_RK = TimeStepperSUNDIALS_IDA_BoostRK54<Assembler>;
  template <typename Assembler> using ImplicitEuler = TimeStepperImplicitEuler<Assembler>;
  template <typename Assembler> using TRBDF2 = TimeStepperTRBDF2<Assembler>;
  template <typename Assembler> using ExplicitEuler = TimeStepperExplicitEuler<Assembler>;
  template <typename Assembler> using ABM = TimeStepperBoostABM<Assembler>;
  template <typename Assembler> using RK4 = TimeStepperRK<Assembler>;
} // namespace

TEST_CASE("A restarted flow ends where the snapshotting flow ends", "[snapshot][timestepping]")
{
  using Model = ModelLatch<1>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "seed", "seed");
  const auto config = base_config();

  const auto seed = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(with_snapshots(config, R"({"t": [0.5]})"), root);

  const auto snapshot_file = root.run_file("_snapshot_000", ".h5");
  REQUIRE(std::filesystem::exists(snapshot_file));
  const auto snapshot = read_snapshot(snapshot_file);
  REQUIRE_THAT(snapshot.t, Catch::Matchers::WithinAbs(0.5, 1e-14));
  REQUIRE_THAT(snapshot.k, Catch::Matchers::WithinRel(std::exp(-0.5), 1e-14));
  // The latch fired before the snapshot, so it has to be in it.
  REQUIRE(snapshot.model.get<bool>("latched"));

  const auto restarted = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(with_restart(config, snapshot_file),
                                                                           root.child("restart", "restart"));
  INFO("max |seed - restarted| = " << max_difference(seed.spatial, restarted.spatial));
  REQUIRE(max_difference(seed.spatial, restarted.spatial) <= 1e-12 * max_abs(seed.spatial));

  SECTION("Without load_state the latch is lost and the flow differs")
  {
    using Forgetful = ModelLatch<1, false>;
    const auto forgetful = run_flow<Forgetful, DGDisc<Forgetful>, DGAsm<Forgetful>, IDA>(
        with_restart(config, snapshot_file), root.child("forgetful", "forgetful"));
    // Rate 1 instead of 2 over the remaining half: off by a factor of about e^{-1/2}.
    REQUIRE(max_difference(seed.spatial, forgetful.spatial) > 0.1 * max_abs(seed.spatial));
  }
}

TEST_CASE("Segmenting a flow at snapshots does not change it", "[snapshot][timestepping]")
{
  using Model = Testing::ModelExp<1>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "segments", "segments");
  const auto config = base_config();

  const auto plain = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(config, root.child("plain", "plain"));
  const auto segmented = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(
      with_snapshots(config, R"({"t": [0.25, 0.5, 0.75]})"), root.child("segmented", "segmented"));

  REQUIRE(std::filesystem::exists(root.child("segmented", "segmented").run_file("_snapshot_002", ".h5")));
  const double scale = max_abs(plain.spatial);
  REQUIRE(max_error_exponential(plain, 1.) <= 1e-7 * scale);
  REQUIRE(max_error_exponential(segmented, 1.) <= 1e-7 * scale);
  REQUIRE(max_difference(plain.spatial, segmented.spatial) <= 1e-7 * scale);
}

TEST_CASE("A flow can start at t_start > 0", "[snapshot][timestepping]")
{
  using Model = Testing::ModelExp<1>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "offset", "offset");
  const auto state = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(base_config(), root, /*t_start=*/0.3);
  REQUIRE(max_error_exponential(state, 0.7) <= 1e-7 * max_abs(state.spatial));
}

TEST_CASE("stop_after_last ends the seed run at its snapshot", "[snapshot][timestepping]")
{
  using Model = Testing::ModelExp<1>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "stop", "stop");
  const auto seed = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(
      with_snapshots(base_config(), R"({"k": [0.6065306597126334], "stop_after_last": true})"), root);

  // k = e^{-1/2} with Lambda = 1 is t = 0.5.
  const auto snapshot = read_snapshot(root.run_file("_snapshot_000", ".h5"));
  REQUIRE_THAT(snapshot.t, Catch::Matchers::WithinAbs(0.5, 1e-12));
  REQUIRE(max_difference(seed.spatial, snapshot.spatial) == 0.);
  REQUIRE(max_error_exponential(seed, 0.5) <= 1e-7 * max_abs(seed.spatial));
}

TEST_CASE("Explicit stepper: restart reproduces the flow", "[snapshot][timestepping]")
{
  using Model = Testing::ModelExp<1>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "rk", "rk");
  const auto config = base_config();

  const auto seed = run_flow<Model, DGDisc<Model>, DGAsm<Model>, RK54>(with_snapshots(config, R"({"t": [0.5]})"), root);
  const auto restarted = run_flow<Model, DGDisc<Model>, DGAsm<Model>, RK54>(
      with_restart(config, root.run_file("_snapshot_000", ".h5")), root.child("restart", "restart"));

  const double scale = max_abs(seed.spatial);
  REQUIRE(max_difference(seed.spatial, restarted.spatial) <= 1e-12 * scale);
  REQUIRE(max_error_exponential(restarted, 1.) <= 1e-7 * scale);
}

TEST_CASE("Hybrid stepper: restart reproduces spatial state and variables", "[snapshot][timestepping]")
{
  using Model = Testing::ModelHybridTwoWay<1>;
  using Disc = CG::Discretization<Model, RectangularMeshSerial<1>>;
  using Asm = CG::Assembler<Disc>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "hybrid", "hybrid");
  const auto config = base_config();

  const auto seed = run_flow<Model, Disc, Asm, IDA_ABM>(with_snapshots(config, R"({"t": [0.5]})"), root);
  const auto snapshot = read_snapshot(root.run_file("_snapshot_000", ".h5"));
  REQUIRE(snapshot.variables.size() == 1);
  // The snapshot is taken at a segment end, where even the hybrid holds its exact variables.
  REQUIRE_THAT(snapshot.variables[0], Catch::Matchers::WithinAbs(Model::v_exact(0.5), 1e-5));

  const auto restarted = run_flow<Model, Disc, Asm, IDA_ABM>(
      with_restart(config, root.run_file("_snapshot_000", ".h5")), root.child("restart", "restart"));
  REQUIRE(restarted.variables.size() == 1);
  REQUIRE(max_difference(seed.spatial, restarted.spatial) <= 1e-10);
  REQUIRE_THAT(restarted.variables[0], Catch::Matchers::WithinAbs(seed.variables[0], 1e-10));
  REQUIRE_THAT(restarted.variables[0], Catch::Matchers::WithinAbs(Model::v_exact(1.), 1e-5));
  for (const double u : restarted.spatial.values)
    REQUIRE_THAT(u, Catch::Matchers::WithinAbs(Model::u_exact(1.), 1e-5));
}

TEST_CASE("Adaptive flow: the restart rebuilds the adapted mesh", "[snapshot][timestepping]")
{
  using Model = ModelLatch<1>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "adaptive", "adaptive");
  auto config = base_config();
  auto &adaptivity = config().as_object()["discretization"].as_object()["adaptivity"].as_object();
  adaptivity["level"] = 2;
  adaptivity["adapt_dt"] = 0.2;
  adaptivity["refine_percent"] = 0.3;

  const auto seed =
      run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA, true>(with_snapshots(config, R"({"t": [0.5]})"), root);
  const auto snapshot = read_snapshot(root.run_file("_snapshot_000", ".h5"));
  // Guard: the test only means something if the mesh at the snapshot is not the initial one.
  REQUIRE(snapshot.spatial.active_cells.size() > seed.initial.active_cells.size());
  REQUIRE(std::isfinite(snapshot.last_adaptation_time));

  const auto restarted = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA, true>(
      with_restart(config, root.run_file("_snapshot_000", ".h5")), root.child("restart", "restart"));
  // Same final mesh (the adaptation schedule was restored too) and the same state on it.
  REQUIRE(max_difference(seed.spatial, restarted.spatial) <= 1e-12 * max_abs(seed.spatial));
}

TEST_CASE("Every stepper restarts exactly where the seed run continues", "[snapshot][timestepping]")
{
  // ModelExp grows like e^t, so a restart that ignored the snapshot would end a factor e^{1/2} off.
  using Model = Testing::ModelExp<1>;
  SECTION("Implicit Euler") { require_restart_reproduces<Model, DGDisc<Model>, DGAsm<Model>, ImplicitEuler>("ie"); }
  SECTION("TRBDF2") { require_restart_reproduces<Model, DGDisc<Model>, DGAsm<Model>, TRBDF2>("trbdf2"); }
  SECTION("Explicit Euler") { require_restart_reproduces<Model, DGDisc<Model>, DGAsm<Model>, ExplicitEuler>("ee"); }
  SECTION("Boost ABM") { require_restart_reproduces<Model, DGDisc<Model>, DGAsm<Model>, ABM>("abm"); }
  SECTION("IDA + Boost RK")
  {
    using Hybrid = Testing::ModelHybridTwoWay<1>;
    using Disc = CG::Discretization<Hybrid, RectangularMeshSerial<1>>;
    require_restart_reproduces<Hybrid, Disc, CG::Assembler<Disc>, IDA_RK>("ida_rk", 1e-10);
  }
}

TEST_CASE("Variables-only flow: restart reproduces the flow", "[snapshot][timestepping]")
{
  const auto config = base_config();
  const auto check = [&]<template <typename> typename TimeStepperFor>(const std::string &name) {
    const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, name);
    const auto seed = run_variables_flow<TimeStepperFor>(with_snapshots(config, R"({"t": [0.5]})"), root);
    const auto restarted = run_variables_flow<TimeStepperFor>(
        with_restart(config, root.run_file("_snapshot_000", ".h5")), root.child("restart", "restart"));
    REQUIRE(restarted.size() == 1);
    REQUIRE_THAT(restarted[0], Catch::Matchers::WithinAbs(seed[0], 1e-12));
    REQUIRE_THAT(restarted[0], Catch::Matchers::WithinRel(std::exp(1.), 1e-6));
  };
  SECTION("deal.II RK") { check.template operator()<RK4>("vars_rk"); }
  SECTION("Boost ABM") { check.template operator()<ABM>("vars_abm"); }
}

TEST_CASE("A stuck timestep controller throws instead of ending early", "[snapshot][timestepping]")
{
  // Ending early would hand run() a state short of the segment end, which it would then write as
  // the snapshot at the segment end.
  struct FailingSolver {
    double get_error() const { return 1.; }
    void set_ignore_nonconv(bool) {}
  } solver;
  TC_PI<FailingSolver> tc(solver, 1, 0., 1., 1e-2, 1e-7, 1e-1, 1e-1);
  auto never_converges = [](double, double) { throw std::runtime_error("no convergence"); };
  auto no_output = [](double) {};
  REQUIRE_THROWS_WITH(
      [&]() {
        while (!tc.finished())
          tc.advance(never_converges, no_output);
      }(),
      Catch::Matchers::ContainsSubstring("stuck"));
}

TEST_CASE("Snapshots are never overwritten", "[snapshot][timestepping]")
{
  using Model = Testing::ModelExp<1>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "guard", "guard");
  const auto config = with_snapshots(base_config(), R"({"t": [0.5]})");
  run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(config, root);

  // A second run into the same output stops before stepping -- also a restart from another run's
  // snapshot that writes snapshots there.
  REQUIRE_THROWS_WITH((run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(config, root)),
                      Catch::Matchers::ContainsSubstring("already exists"));
  const auto other = root.child("other", "other");
  run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(config, other);
  REQUIRE_THROWS_WITH((run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(
                          with_snapshots(with_restart(base_config(), other.run_file("_snapshot_000", ".h5")),
                                         R"({"t": [0.75]})"),
                          root)),
                      Catch::Matchers::ContainsSubstring("already exists"));
}

TEST_CASE("Restarts can be chained", "[snapshot][timestepping]")
{
  // seed -> snapshot at 0.25 -> restart writes one at 0.5 -> restart from that. Same segment
  // boundaries as one run with snapshots at 0.25 and 0.5, so the same final state.
  using Model = ModelLatch<1>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "chain", "chain");
  const auto config = base_config();

  const auto reference = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(
      with_snapshots(config, R"({"t": [0.25, 0.5]})"), root.child("reference", "reference"));
  run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(with_snapshots(config, R"({"t": [0.25]})"), root);
  run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(
      with_snapshots(with_restart(config, root.run_file("_snapshot_000", ".h5")), R"({"t": [0.5]})"),
      root.child("first", "first"));
  const auto chained = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(
      with_restart(config, root.child("first", "first").run_file("_snapshot_000", ".h5")),
      root.child("second", "second"));

  REQUIRE(max_difference(reference.spatial, chained.spatial) <= 1e-12 * max_abs(reference.spatial));
}

TEST_CASE("Restarting from its own snapshot continues an interrupted run", "[snapshot][timestepping]")
{
  // A run with snapshots at 0.25 and 0.5 that died in between: its snapshot_001 is missing.
  using Model = ModelLatch<1>;
  const auto root = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "cont", "cont");
  const auto config = with_snapshots(base_config(), R"({"t": [0.25, 0.5]})");
  const auto uninterrupted = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(config, root);
  const auto second = root.run_file("_snapshot_001", ".h5");
  const auto second_snapshot = read_snapshot(second);
  std::filesystem::remove(second);

  const auto continued = run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(
      with_restart(config, root.run_file("_snapshot_000", ".h5")), root);
  REQUIRE(max_difference(uninterrupted.spatial, continued.spatial) <= 1e-12 * max_abs(uninterrupted.spatial));
  // The continuation wrote the missing snapshot under its original number.
  REQUIRE(std::filesystem::exists(second));
  REQUIRE(max_difference(read_snapshot(second).spatial, second_snapshot.spatial) <=
          1e-12 * max_abs(second_snapshot.spatial));

  // Continuing again from the first snapshot would overwrite the second.
  REQUIRE_THROWS_WITH((run_flow<Model, DGDisc<Model>, DGAsm<Model>, IDA>(
                          with_restart(config, root.run_file("_snapshot_000", ".h5")), root)),
                      Catch::Matchers::ContainsSubstring("already continued past"));
}
