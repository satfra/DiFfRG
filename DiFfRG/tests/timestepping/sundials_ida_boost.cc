#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>

#include "DiFfRG/discretization/mesh/h_adaptivity.hh"
#include <algorithm>
#include <fstream>
#include <limits>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sstream>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/discretization/common/solution_view.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>
#include <DiFfRG/timestepping/timestepping.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------
//
// These tests exercise the hybrid timesteppers, in which the implicit IDA controller
// drives the FEM block and a separate explicit stepper integrates the variable block
// "on demand". Three model families are exercised:
//
//   * ModelHybridRollback : sharp transition at t = 0.5 -> IDA rejects steps; this is the
//     regression test for the rejection-rollback bug (a rejected IDA step leaking into the
//     explicit buffer would corrupt v).
//
//   * ModelHybridSmooth   : du/dt = -K u (no rejections) with explicit.dt set so that each
//     IDA trial step contains only one or two explicit substeps. This is the regime in
//     which a typical FRG production run operates and the regime in which a per-trial-step
//     reset() of the multistep stepper would be most visible.
//
//   * ModelHybridTwoWay   : the FEM source reads the explicit variable v through an
//     extractor, closing the coupling loop (production FRG models do exactly this through
//     wave-function-renormalisation extractors). If serving v back to IDA across Newton
//     iterations is not consistent with the spatial trial state, Newton's convergence
//     breaks down and IDA reports recoverable residual errors.

// The discretization and assembler that run_hybrid builds internally. The TEST_CASEs below have
// to name the assembler in order to spell their stepper type, and it must be the *same* type:
// run_hybrid constructs the stepper with a pointer to its own assembler, so a spelling that drifts
// from this one would not compile. Defining it once here is what keeps the two in step.
constexpr uint hybrid_dim = 1;
template <typename Model> using HybridDiscretization = CG::Discretization<Model, RectangularMesh<hybrid_dim>>;
template <typename Model> using HybridAssembler = CG::Assembler<HybridDiscretization<Model>>;

template <typename Model, typename TimeStepper>
bool run_hybrid(const std::string &test_name, double expected_precision, double cur_dt = 1e-3, double final_time = 1.0,
                double output_dt = 5e-2, double impl_max_dt = 1e-1, double impl_rel_tol = 1e-6, int coupling_mode = 0,
                double expl_max_dt = -1., double expected_stepper_kind = std::numeric_limits<double>::quiet_NaN())
{
  using namespace dealii;
  DiFfRG::Init();
  // -1 means "pick a cap consistent with cur_dt". The diagnostics below deliberately set the
  // explicit substep to the output interval (5e-2, 2e-2), which is above the old fixed 1e-2
  // default -- an explicit.dt above explicit.maximal_dt that AbstractTimestepper now rejects at
  // construction. Callers that pass a cap explicitly are unaffected.
  if (expl_max_dt < 0.) expl_max_dt = std::max(1e-2, cur_dt);
  constexpr uint dim = hybrid_dim;
  using Discretization = HybridDiscretization<Model>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = HybridAssembler<Model>;

  ConfigTree json =
      json::value({{"physical", {{"Lambda", 1.}}},
                   {"discretization",
                    {{"fe_order", 3},
                     {"overintegration", 0},
                     {"output_subdivisions", 2},
                     {"EoM_abs_tol", 1e-12},
                     {"EoM_max_iter", 100},
                     {"grid", {{"x_grid", "0:0.05:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
                     {"adaptivity",
                      {{"start_adapt_at", 0.},
                       {"adapt_dt", 1e-1},
                       {"level", 0},
                       {"refine_percent", 1e-1},
                       {"coarsen_percent", 5e-2}}}}},
                   {"timestepping",
                    {{"final_time", final_time},
                     {"output_dt", output_dt},
                     {"explicit",
                      {{"dt", cur_dt},
                       {"minimal_dt", 1e-7},
                       {"maximal_dt", expl_max_dt},
                       {"abs_tol", 1e-13},
                       {"rel_tol", 1e-10},
                       {"coupling_mode", coupling_mode}}},
                     {"implicit",
                      {{"dt", 1e-3},
                       {"minimal_dt", 1e-7},
                       {"maximal_dt", impl_max_dt},
                       {"abs_tol", 1e-10},
                       {"rel_tol", impl_rel_tol},
                       {"jacobian_diagnostics", true}}}}},
                   {"output", {{"verbosity", 0}, {"vtk", false}}}});

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
  } catch (const spdlog::spdlog_ex &) {
    // logger already set up
  }

  Testing::PhysicalParameters p_prm = {};
  Model model(p_prm);
  typename Discretization::Mesh mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);
  auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, test_name, test_name);
  OutputSession<Assembler> data_out(data_out_path, json);
  HAdaptivity mesh_adaptor(assembler, json);
  TimeStepper time_stepper(json, assembler, data_out, mesh_adaptor);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  try {
    time_stepper.run(initial_condition, 0., final_time);
  } catch (std::exception &e) {
    std::cout << test_name << ": simulation finished with exception " << e.what() << std::endl;
    return false;
  }

  model.set_time(final_time);
  bool valid = true;

  // Under MPI only rank 0 writes the diagnostics table, and the state is distributed: the table is
  // checked there, the state through replicated views on every rank, and the verdict agreed below.
  const MPI_Comm comm = assembler.get_communicator();
  if (std::isfinite(expected_stepper_kind) && MPI::rank(comm) == 0) {
    const auto table_path = data_out_path.root() / (data_out_path.run_name() + "_jacobian_diagnostics.csv");
    std::ifstream table(table_path);
    std::string header_line, data_line;
    valid &= static_cast<bool>(std::getline(table, header_line));
    valid &= static_cast<bool>(std::getline(table, data_line));

    std::vector<std::string> columns;
    std::istringstream header(header_line);
    for (std::string column; std::getline(header, column, ',');)
      columns.push_back(column);
    const auto kind = std::find(columns.begin(), columns.end(), "stepper_kind");
    const auto stage = std::find(columns.begin(), columns.end(), "stage");
    valid &= kind != columns.end() && stage != columns.end();
    if (kind != columns.end() && stage != columns.end()) {
      const auto kind_index = static_cast<std::size_t>(std::distance(columns.begin(), kind));
      const auto stage_index = static_cast<std::size_t>(std::distance(columns.begin(), stage));
      do {
        std::vector<double> values;
        std::istringstream row(data_line);
        for (std::string value; std::getline(row, value, ',');)
          values.push_back(std::stod(value));
        valid &= values.size() == columns.size();
        if (values.size() == columns.size()) {
          valid &= values[kind_index] == expected_stepper_kind;
          valid &= values[stage_index] == static_cast<double>(ImplicitTimestepperStage::main);
        }
      } while (std::getline(table, data_line));
    }
  }

  SolutionView<VectorType> fem_view, variables_view;
  assembler.reinit_solution_view(fem_view);
  fem_view.refresh(initial_condition.data().block(0));
  reinit_variables_view(variables_view, initial_condition.data().block(1).size(), comm);
  variables_view.refresh(initial_condition.data().block(1));

  // block 0: FEM solution
  const auto &support_points = discretization.get_support_points();
  for (uint i = 0; i < support_points.size(); ++i) {
    const double is = fem_view[i];
    const double should = model.solution(support_points[i]);
    if (!is_close(is, should, expected_precision))
      std::cout << test_name << " FEM u: is " << is << " should be " << should << " (rel. error "
                << std::abs(is - should) / std::max(std::abs(should), 1e-30) << ")" << std::endl;
    valid &= is_close(is, should, expected_precision);
  }

  // block 1: explicit variable
  const double v_is = variables_view[0];
  const double v_should = model.variable_solution();
  if (!is_close(v_is, v_should, expected_precision))
    std::cout << test_name << " explicit variable v: is " << v_is << " should be " << v_should << " (rel. error "
              << std::abs(v_is - v_should) / std::max(std::abs(v_should), 1e-30) << ")" << std::endl;
  valid &= is_close(v_is, v_should, expected_precision);

  valid = !MPI::any_of(comm, !valid);
  if (!valid) std::cerr << "Failed " << test_name << std::endl;
  return valid;
}

//--------------------------------------------
// Original regression tests: rejection-rollback on a sharp logistic FEM
//--------------------------------------------

TEST_CASE("Test SUNDIALS IDA + Boost ABM hybrid stepper", "[timestepping][sundials_ida_boost][abm]")
{
  using Model = Testing::ModelHybridRollback<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  REQUIRE(run_hybrid<Model, TimeStepper>("test_sundials_ida_boost_abm", 1e-3, 1e-3, 1., 5e-2, 1e-1, 1e-6, 0, 1e-2,
                                         static_cast<double>(ImplicitTimestepperKind::ida_boost_abm)));
}

TEST_CASE("Test SUNDIALS IDA + Boost RK hybrid stepper", "[timestepping][sundials_ida_boost][rk]")
{
  using Model = Testing::ModelHybridRollback<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostRK<HybridAssembler<Model>, DefaultLinearSolver, 0>;
  REQUIRE(run_hybrid<Model, TimeStepper>("test_sundials_ida_boost_rk", 1e-3, 1e-3, 1., 5e-2, 1e-1, 1e-6, 0, 1e-2,
                                         static_cast<double>(ImplicitTimestepperKind::ida_boost_rk)));
}

//--------------------------------------------
// Diagnostic: smooth problem, ~1 explicit substep per IDA trial step.
//
// FRG production runs typically have explicit.dt of the same order as IDA's accepted
// step, so each request_variables call appends ~1 ABM substep. The integration must
// still match the analytic solution to a sensible precision -- if the per-trial-step
// machinery (rebuild on commit, multistep history reset, spatial interpolation, ...)
// degrades the explicit accuracy on accepted steps, this is where it shows up.
//--------------------------------------------

TEST_CASE("Hybrid ABM stepper retains accuracy on smooth problem at ~1 substep per IDA step",
          "[timestepping][sundials_ida_boost][abm][diag]")
{
  using Model = Testing::ModelHybridSmooth<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  // explicit.dt = output_dt so each IDA trial step usually triggers exactly one ABM substep
  REQUIRE(run_hybrid<Model, TimeStepper>("test_diag_abm_smooth", /*tol=*/1e-3, /*cur_dt=*/5e-2,
                                         /*final_time=*/1.0, /*output_dt=*/5e-2,
                                         /*impl_max_dt=*/5e-2));
}

TEST_CASE("Hybrid ABM cur_dt scan on smooth problem", "[timestepping][sundials_ida_boost][abm][scan]")
{
  using Model = Testing::ModelHybridSmooth<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  // Sweep cur_dt while holding output_dt and IDA max_dt fixed; the test prints the
  // observed v error so the scaling can be read off the log. We do not REQUIRE pass --
  // this is a diagnostic scan, not a regression gate.
  for (double cur_dt : {5e-2, 2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3}) {
    const std::string tag = "test_scan_abm_dt_smooth_" + std::to_string(cur_dt);
    run_hybrid<Model, TimeStepper>(tag, /*tol=*/1e-30, /*cur_dt=*/cur_dt,
                                   /*final_time=*/1.0, /*output_dt=*/5e-2,
                                   /*impl_max_dt=*/5e-2);
  }
}

TEST_CASE("Hybrid ABM cur_dt scan on two-way coupled problem", "[timestepping][sundials_ida_boost][abm][scan]")
{
  using Model = Testing::ModelHybridTwoWay<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  for (double cur_dt : {2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3}) {
    const std::string tag = "test_scan_abm_dt_twoway_" + std::to_string(cur_dt);
    run_hybrid<Model, TimeStepper>(tag, /*tol=*/1e-30, /*cur_dt=*/cur_dt,
                                   /*final_time=*/1.0, /*output_dt=*/5e-2,
                                   /*impl_max_dt=*/5e-2);
  }
}

//--------------------------------------------
// Diagnostic scans comparing the three coupling modes (0 lag / predictor, 1 stagger,
// 2 predict / lookahead) in the regime they are designed for: dt_implicit << dt_explicit.
// The explicit step is held large (maximal_dt = 2e-1) and IDA's step small
// (impl_max_dt = 1e-3), so IDA subsamples each explicit step ~100x. The v-error vs cur_dt
// is printed for each mode so the modes can be compared directly from the log. Non-gating.
//--------------------------------------------

namespace
{
  const char *coupling_mode_name(int m) { return m == 0 ? "lag" : (m == 1 ? "stagger" : "predict"); }
} // namespace

TEST_CASE("Hybrid ABM coupling-mode scan, dt_impl<<dt_expl, two-way",
          "[timestepping][sundials_ida_boost][abm][scan][modes]")
{
  using Model = Testing::ModelHybridTwoWay<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  for (int mode : {0, 1, 2})
    for (double cur_dt : {1e-1, 5e-2, 2.5e-2}) {
      const std::string tag =
          "test_modes_twoway_" + std::string(coupling_mode_name(mode)) + "_" + std::to_string(cur_dt);
      run_hybrid<Model, TimeStepper>(tag, /*tol=*/1e-30, /*cur_dt=*/cur_dt,
                                     /*final_time=*/1.0, /*output_dt=*/5e-2,
                                     /*impl_max_dt=*/1e-3, /*impl_rel_tol=*/1e-6,
                                     /*coupling_mode=*/mode, /*expl_max_dt=*/2e-1);
    }
}

TEST_CASE("Hybrid ABM coupling-mode scan, dt_impl<<dt_expl, smooth",
          "[timestepping][sundials_ida_boost][abm][scan][modes]")
{
  using Model = Testing::ModelHybridSmooth<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  for (int mode : {0, 1, 2})
    for (double cur_dt : {1e-1, 5e-2, 2.5e-2}) {
      const std::string tag =
          "test_modes_smooth_" + std::string(coupling_mode_name(mode)) + "_" + std::to_string(cur_dt);
      run_hybrid<Model, TimeStepper>(tag, /*tol=*/1e-30, /*cur_dt=*/cur_dt,
                                     /*final_time=*/1.0, /*output_dt=*/5e-2,
                                     /*impl_max_dt=*/1e-3, /*impl_rel_tol=*/1e-6,
                                     /*coupling_mode=*/mode, /*expl_max_dt=*/2e-1);
    }
}

// Weak explicit<-implicit coupling: the explicit variable is nearly autonomous, so the
// arrow-1 extrapolation that destabilises mode 1 on the stiff two-way model should be
// harmless here. This isolates how the v-serving choice affects the *implicit* (u)
// accuracy in the dt_impl<<dt_expl regime, and checks mode 2 specifically.
TEST_CASE("Hybrid ABM coupling-mode scan, dt_impl<<dt_expl, weak coupling",
          "[timestepping][sundials_ida_boost][abm][scan][modes]")
{
  using Model = Testing::ModelHybridWeakCoupling<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  for (int mode : {0, 1, 2})
    for (double cur_dt : {1e-1, 5e-2, 2.5e-2}) {
      const std::string tag = "test_modes_weak_" + std::string(coupling_mode_name(mode)) + "_" + std::to_string(cur_dt);
      run_hybrid<Model, TimeStepper>(tag, /*tol=*/1e-30, /*cur_dt=*/cur_dt,
                                     /*final_time=*/1.0, /*output_dt=*/5e-2,
                                     /*impl_max_dt=*/1e-3, /*impl_rel_tol=*/1e-6,
                                     /*coupling_mode=*/mode, /*expl_max_dt=*/2e-1);
    }
}

// Stress test: severely under-resolved one-way coupling (smooth model, K=2) at very
// coarse explicit steps so K*cur_dt = O(1). This is the regime where STAGGER's output
// riding on half-step-EXTRAPOLATED coupling should overshoot, while LAG's interpolated
// coupling stays bounded -- i.e. where LAG can beat STAGGER. Non-gating diagnostic.
TEST_CASE("Hybrid ABM coupling-mode scan, stiff under-resolved one-way",
          "[timestepping][sundials_ida_boost][abm][scan][stiff]")
{
  using Model = Testing::ModelHybridSmooth<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  for (int mode : {0, 1, 2})
    for (double cur_dt : {5e-1, 2.5e-1, 1.25e-1}) {
      const std::string tag = "test_stiff_" + std::string(coupling_mode_name(mode)) + "_" + std::to_string(cur_dt);
      run_hybrid<Model, TimeStepper>(tag, /*tol=*/1e-30, /*cur_dt=*/cur_dt,
                                     /*final_time=*/1.0, /*output_dt=*/5e-1,
                                     /*impl_max_dt=*/1e-3, /*impl_rel_tol=*/1e-6,
                                     /*coupling_mode=*/mode, /*expl_max_dt=*/1.0);
    }
}

TEST_CASE("Hybrid ABM mode 2 (lookahead) handles weak explicit<-implicit coupling",
          "[timestepping][sundials_ida_boost][abm][diag]")
{
  using Model = Testing::ModelHybridWeakCoupling<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  REQUIRE(run_hybrid<Model, TimeStepper>("test_diag_abm_weak_predict", /*tol=*/1e-3, /*cur_dt=*/5e-2,
                                         /*final_time=*/1.0, /*output_dt=*/5e-2, /*impl_max_dt=*/1e-3,
                                         /*impl_rel_tol=*/1e-6, /*coupling_mode=*/2, /*expl_max_dt=*/2e-1));
}

TEST_CASE("Hybrid RK stepper retains accuracy on smooth problem at ~1 substep per IDA step",
          "[timestepping][sundials_ida_boost][rk][diag]")
{
  using Model = Testing::ModelHybridSmooth<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostRK<HybridAssembler<Model>, DefaultLinearSolver, 0>;
  REQUIRE(run_hybrid<Model, TimeStepper>("test_diag_rk_smooth", /*tol=*/1e-3, /*cur_dt=*/5e-2,
                                         /*final_time=*/1.0, /*output_dt=*/5e-2,
                                         /*impl_max_dt=*/5e-2));
}

//--------------------------------------------
// Diagnostic: two-way coupled FEM <-> variable.
//
// The FEM source reads v through an extractor and v's residual reads u, mirroring how a
// production FRG model couples back to wave-function renormalisations. If the variable
// served back to IDA across Newton iterations is built from speculative, unconverged
// spatial trial values, Newton convergence will fail and IDA will throw "recoverable
// residual errors" -- which is exactly what the QM_LPAp run does against the current fix.
//--------------------------------------------

TEST_CASE("Hybrid ABM stepper handles two-way FEM <-> variable coupling",
          "[timestepping][sundials_ida_boost][abm][diag]")
{
  using Model = Testing::ModelHybridTwoWay<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<HybridAssembler<Model>>;
  // cur_dt is the substep this diagnostic actually exercised all along: it used to request 2e-2
  // against an explicit maximal_dt of 1e-2, which clamped it. Now that AbstractTimestepper rejects
  // dt > maximal_dt instead of silently capping, the request has to be the value that ran -- and the
  // 1e-3 gate is calibrated for it (cur_dt scan: 1e-2 -> 5.7e-4, 2e-2 -> 2.3e-3).
  REQUIRE(run_hybrid<Model, TimeStepper>("test_diag_abm_twoway", /*tol=*/1e-3, /*cur_dt=*/1e-2,
                                         /*final_time=*/1.0, /*output_dt=*/5e-2,
                                         /*impl_max_dt=*/5e-2));
}

TEST_CASE("Hybrid RK stepper handles two-way FEM <-> variable coupling", "[timestepping][sundials_ida_boost][rk][diag]")
{
  using Model = Testing::ModelHybridTwoWay<1>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostRK<HybridAssembler<Model>, DefaultLinearSolver, 0>;
  // cur_dt is the substep this diagnostic actually exercised all along: it used to request 2e-2
  // against an explicit maximal_dt of 1e-2, which clamped it. Now that AbstractTimestepper rejects
  // dt > maximal_dt instead of silently capping, the request has to be the value that ran -- and the
  // 1e-3 gate is calibrated for it (cur_dt scan: 1e-2 -> 5.7e-4, 2e-2 -> 2.3e-3).
  REQUIRE(run_hybrid<Model, TimeStepper>("test_diag_rk_twoway", /*tol=*/1e-3, /*cur_dt=*/1e-2,
                                         /*final_time=*/1.0, /*output_dt=*/5e-2,
                                         /*impl_max_dt=*/5e-2));
}
