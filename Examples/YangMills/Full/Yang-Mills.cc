#include <DiFfRG/DiFfRG.hh>

#include <optional>
using namespace DiFfRG;

#include "model.hh"
#include "tuning.hh"

// Choices for types
using Model = YangMills;
using Assembler = Variables::Assembler<Model>;
// Pure glue, variables only -> a purely explicit stepper (Boost Adams-Bashforth-Moulton).
using TimeStepper = TimeStepperBoostABM<Assembler>;

/**
 * @brief Run one flow and classify its endpoint.
 *
 * The output location is passed in rather than read from /output/folder, because the tuner gives
 * every probe its own child directory; external_log routes this run's messages into the tuner's
 * log instead of the run's own session log.
 *
 * obs_out and residual_out report the tuning observable. This function answers only "did this
 * flow produce a usable IR endpoint"; whether that endpoint lies above or below the tuning target
 * is the root finder's business. A flow converging to Zc(0) below the target is therefore
 * reported as false -- it does not qualify as an answer -- but still writes its Zc(0) into
 * obs_out, because it is a perfectly good sample of the critical power law. Both pointers may be
 * null, and both outputs are NaN-initialised, so "wrote nothing" needs no separate flag.
 */
bool run(const ConfigTree &config, const OutputPath &output_path,
         std::optional<ReportPort> external_log = std::nullopt, double *obs_out = nullptr,
         double *residual_out = nullptr)
{
  if (obs_out) *obs_out = std::numeric_limits<double>::quiet_NaN();
  if (residual_out) *residual_out = std::numeric_limits<double>::quiet_NaN();
  const double final_time = config.get_double("/timestepping/final_time");

  // Define the objects needed to run the simulation
  OutputSession<Assembler> data_out(output_path, config);
  const auto log = external_log.value_or(data_out.report_port());
  Model model(config);
  Assembler assembler(model, config);
  TimeStepper time_stepper(config, assembler, data_out);

  // Set up the initial condition
  FlowingVariables initial_condition;
  initial_condition.interpolate(model);

  // Start the timestepping
  try {
    time_stepper.run(initial_condition, 0., final_time);
  } catch (const FlowAbort &e) {
    // Aborted deliberately: the RG time travels with the exception, so there is no trajectory
    // scan and no dependence on whatever partial output made it to disk.
    log.info("Flow aborted early at t = {:.4f} ({:.0f}% of the flow skipped)", e.t, 100. * (1. - e.t / final_time));
    if (residual_out) *residual_out = e.t;
    log.flush();
    return false;
  } catch (std::exception &e) {
    log.error("Timestepping finished with exception {}", e.what());
    if (residual_out) *residual_out = time_of_divergence(output_path);
    log.flush();
    return false;
  }

  HDF5Input hdf5_input(output_path.run_file(".h5").string());
  const auto m2A = hdf5_input.load_scalar<double>("m2A").back();
  const auto m2A_fit = hdf5_input.load_scalar<double>("m2A_fit").back();
  std::vector<double> Zc(p_grid_size);
  hdf5_input.load_map("Zc", Zc.data());

  // Predicate for the m2A search. Two modes, selected by /tuning/m2A/zc_target:
  //   target = 0: reject only the diverged (past-critical) side, so the search boundary is the
  //            true scaling separatrix Zc(0) -> 0.
  //   target > 0: also reject the near-scaling sliver Zc(0) < target, moving the boundary onto
  //            the safe side at Zc(0) = target. That trades a razor-thin separatrix for a wide
  //            basin of fast, non-divergent flows.
  const double zc_target = config.get_double("/tuning/m2A/zc_target", 0.0);

  // Ran away: no usable endpoint and no observable, but it does carry how far it got. The runaway
  // half is flow_has_run_away(), the same criterion the in-flow abort applies to the same
  // quantity. m2A < 0 is NOT a runaway test and stays separate: it is the phase classifier that
  // puts the search boundary on the scaling separatrix, where m2A tips negative long before it
  // blows up.
  const double zc_min = *std::min_element(Zc.begin(), Zc.end());
  if (!std::isfinite(m2A) || !std::isfinite(Zc[0]) || m2A < 0 || Zc[0] <= 0 || Zc[0] > 1.1 ||
      flow_has_run_away(m2A, config.get_double("/physical/m2A"), zc_min)) {
    log.error("Diverging result: m2A = {}, Zc(0) = {}", m2A, Zc[0]);
    if (residual_out) *residual_out = time_of_divergence(output_path);
    return false;
  }

  // Converged. The observable is meaningful whichever side of the target it falls on. The
  // residual stays NaN: this flow never diverged, so it has no divergence time to report.
  if (obs_out) *obs_out = Zc[0];

  if (Zc[0] < zc_target) {
    log.info("Below target: Zc(0) = {} < {} (converged; reported as a scaling sample)", Zc[0], zc_target);
    return false;
  }

  log.info("Timestepping finished successfully, m2A = {} (extrapolated {}), Zc(0) = {}", m2A, m2A_fit, Zc[0]);
  return true;
}

int main(int argc, char *argv[])
{
  Timer timer;

  // get all needed parameters and parse from the CLI
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  auto config = config_helper.get_config();
  OutputPath output_path(config);

  // The Yang-Mills flow must be tuned in the initial gluon mass parameter m2A: too negative lands
  // on the Higgs branch (the solution diverges), too small on the massive branch, and the scaling
  // regime lies in between.
  if (config.get_bool("/tuning/m2A/tune")) {
    tune_m2A(config, output_path, run);
  } else {
    // A standalone run has no search to satisfy, so it is judged only on whether it reached a
    // usable IR endpoint -- which is exactly what a finite observable means. run()'s own return
    // additionally applies the zc_target acceptance, and a flow tuned onto the scaling side
    // deliberately lands below that target: reporting it as a failure here would be wrong.
    double obs = std::numeric_limits<double>::quiet_NaN();
    run(config, output_path, std::nullopt, &obs);
    const auto time = timer.wall_time();
    std::cout << "Program finished after " << time_format(time) << std::endl;
    return std::isfinite(obs) ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  // We print a bit of exit information.
  const auto time = timer.wall_time();
  std::cout << "Program finished after " << time_format(time) << std::endl;
  return 0;
}
