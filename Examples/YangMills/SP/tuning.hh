#pragma once

#include <DiFfRG/common/minimization.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/root_finding.hh>
#include <DiFfRG/common/run_reporter.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/hdf5_input.hh>
#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/data/output_settings.hh>

#include <flow_abort.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <sstream>

/**
 * @file tuning.hh
 *
 * Tuning the Yang-Mills flow. Two nested searches:
 *   - tune_m2A drives the initial gluon mass onto the scaling separatrix, using the critical
 *     scaling of Zc(0) as the observable.
 *   - tune_STI varies the gauge couplings so that the three coupling avatars (A^3, A^4, Acbc)
 *     agree, i.e. so the truncation respects the Slavnov-Taylor identities. Every objective
 *     evaluation nests a full m2A tune, which is why the cost of the inner search dominates.
 *
 * The model side of the contract is one function,
 *   bool run(const ConfigTree &, const OutputPath &, std::optional<ReportPort>, double *obs,
 *            double *residual)
 * writing Zc(0) into @c obs for any flow that reaches a usable IR endpoint (whether or not it
 * clears the tuning target), and the RG time it died at into @c residual otherwise.
 */

namespace tune_m2A_defaults
{
  /// Bracket widening when the hypothesised bounds do not straddle, and its safety cap.
  /// 2^24 covers seven decades, far past any cutoff scale this model is run at.
  constexpr double bracket_expansion = 2.0;
  constexpr uint bracket_max_iter = 24;

  /// The Zc(0) power law is asymptotic; it sets in around Zc(0) ~ 0.2. Fitted outside that
  /// window it does not degrade gracefully -- it returns a confident critical point that is
  /// simply wrong. Expressed relative to the target, so the window moves with it.
  constexpr double obs_window = 20.0;

  /// Minimum residual admitted to the divergent fit, as a fraction of final_time. Zero: with the
  /// |m2A|-based runaway detector the divergence-time law holds from the very first failed probe,
  /// so there is nothing to discard.
  constexpr double residual_window = 0.0;

  /// Budget cap. The search halves the bracket at least every other step, so it always
  /// terminates; this only bounds a pathological model.
  constexpr uint max_flows = 40;

  /// Pre-locate stage: all four integration orders are halved, cutting the quadrature cost by
  /// 16x. Far too coarse to trust as an answer -- which is why the STI search does not do it --
  /// but fine here, because the stage's ONLY output is a bracket that the full-fidelity stage
  /// then probes and widens if it does not straddle. A wrong bracket costs a few flows; it
  /// cannot move the result.
  constexpr double prelocate_quadrature_factor = 0.5;

  /// How far the separatrix moves between the two fidelities, relative. One number sets both
  /// halves of the handover, because they are the same statement: the pad added to the coarse
  /// bracket before the full stage probes it, and the resolution at which the pre-locate stops
  /// refining. Bracketing finer than the shift is wasted -- the pad washes it out again -- and
  /// padding by less than the shift hands the expensive stage a bracket that does not contain
  /// the answer. Keeping them one constant makes the second failure unreachable.
  constexpr double prelocate_shift_rel = 3e-2;

  /// The pre-locate aims at a much looser target than the real one. It only has to pin the
  /// separatrix to within the pad, and a loose target makes its acceptance interval vastly wider
  /// while leaving the fitted critical point just as good.
  constexpr double prelocate_target_factor = 30.;
  constexpr double prelocate_target_max = 0.5;

  /// Warm-start half-width, relative in u. The couplings move by ~1% per STI simplex step, so
  /// the critical point moves by a comparable fraction whatever Lambda is.
  constexpr double warm_start_rel_delta = 0.05;
} // namespace tune_m2A_defaults

/**
 * @brief RG time at which a flow ran away, read back from its recorded trajectory.
 *
 * The failing side of the m2A search is not a wasted bit. Near the separatrix a flow runs away at
 * t_fail = a - ln(m2A_c - m2A)/theta with theta approaching 2, the tree-level mass dimension of
 * m2A, so three failed flows locate m2A_c far better than three bisections do. That is what turns
 * the approach phase from a dozen flows into a handful.
 *
 * Crossings are interpolated in log|m2A| against t, recovering sub-step resolution: the samples
 * are quantised at /timestepping/output_dt and that quantisation is what scatters the fitted
 * exponent. A flow whose recording simply stops (the stepper threw) never crosses anything, so
 * the last recorded time is returned -- that is when it died.
 */
inline double time_of_divergence(const OutputPath &output_path)
{
  constexpr double runaway_factor = 100.;
  try {
    HDF5Input hdf5_input(output_path.run_file(".h5").string());
    const auto t = hdf5_input.load_scalar<double>("time");
    const auto m2A = hdf5_input.load_scalar<double>("m2A");
    const auto zc = hdf5_input.load_scalar<double>("Zc_pmin");
    const std::size_t n = std::min({t.size(), m2A.size(), zc.size()});
    if (n == 0) return 0.;

    const double threshold = runaway_factor * std::max(std::abs(m2A[0]), 1.);
    for (std::size_t i = 1; i < n; ++i) {
      if (!flow_has_run_away(m2A[i], m2A[0], zc[i])) continue;
      const bool blown = !std::isfinite(m2A[i]) || std::abs(m2A[i]) > threshold;
      if (!blown) return t[i]; // triggered on Zc, so there is no magnitude to interpolate against
      // |m2A| grows geometrically through the runaway, so interpolate its logarithm.
      const double a = std::abs(m2A[i - 1]), b = std::abs(m2A[i]);
      if (!(std::isfinite(b) && a > 0. && b > a)) return t[i - 1];
      const double f = (std::log(threshold) - std::log(a)) / (std::log(b) - std::log(a));
      return t[i - 1] + (t[i] - t[i - 1]) * std::clamp(f, 0., 1.);
    }
    return t[n - 1]; // never crossed: healthy, or a recording that stops where it died
  } catch (const std::exception &) {
    return 0.; // no readable output at all
  }
}

/**
 * @brief What a tune_m2A call did, for a caller that has to decide whether to trust it.
 *
 * The flow count is here because it is the only honest measure of what an inner tune costs, and
 * a warm-start policy that is not measured against it is a guess.
 */
struct M2AResult {
  double m2A = std::numeric_limits<double>::quiet_NaN();
  bool converged = false;
  uint flows = 0;
};

/**
 * @brief Tune m2A onto the gluon-mass separatrix using the critical scaling of Zc(0).
 *
 * Three things make this cheap where plain bisection is not.
 *
 * 1. The search runs in the dimensionless u = m2A / Lambda^2. The initial condition is
 *    ZA(p) = (p^2 + m2A)/p^2 at k = Lambda, so u is the natural variable and the bracket becomes
 *    transferable across cutoff scales instead of going stale by orders of magnitude.
 *
 * 2. Convergent probes obey Zc(0) = C (m2A - m2A_c)^beta, so three of them fix m2A_c far more
 *    sharply than the bracket they came from. The law only holds for small Zc(0), which is what
 *    the observable window guards.
 *
 * 3. Divergent probes feed the companion law t_fail = a - ln(m2A_c - m2A)/theta, which works far
 *    outside the window where the convergent one does -- so it is what accelerates the approach
 *    phase, while the convergent fit sharpens the endgame.
 *
 * Both are bracket-safeguarded inside ScalingRootFinder, so a wrong fit costs at most a factor
 * two over bisection and can never move the answer.
 *
 * The tolerance is relative and on the OBSERVABLE rather than on m2A, which is the other half of
 * making one setting work at any Lambda.
 *
 * @param hint_m2A       previous answer to warm-start from; 0 disables
 * @param hint_rel_delta relative half-width of the warm-start bracket in u; <= 0 disables
 * @param tol_override   relative observable tolerance, overriding /tuning/m2A/rel_tol
 * @param result         optional out-parameter; see M2AResult
 */
template <typename FUN>
double tune_m2A(ConfigTree &config, const OutputPath &output_path, const FUN &run, const double hint_m2A = 0.0,
                const double hint_rel_delta = -1.0, const double tol_override = -1.0, M2AResult *result = nullptr)
{
  const double Lambda = config.get_double("/physical/Lambda");
  const double L2 = powr<2>(Lambda);
  const double zc_target = config.get_double("/tuning/m2A/zc_target", 0.0);
  const double rel_tol = tol_override > 0 ? tol_override : config.get_double("/tuning/m2A/rel_tol", 1e-3);

  const auto tuning_path = output_path.child(output_path.run_name() + "_m2A_tuning", "tuning");
  RunReporter tuning_logger(tuning_path, OutputSettings(config), MPI::rank(MPI_COMM_WORLD) == 0);
  const auto log = tuning_logger.port();

  // Folder of every probe that reached the IR, keyed by u, so that the tuned result copied out is
  // the flow the finder actually returned rather than whichever probe happened to run last.
  std::map<double, std::filesystem::path> success_folders;

  const auto run_at = [&](const double u, double *obs, double *residual) {
    const double m2A = u * L2;
    config.set_double("/physical/m2A", m2A);
    std::stringstream name;
    name << std::scientific << std::setprecision(12) << "m2A_" << m2A;
    const auto trial_path = tuning_path.child(name.str(), output_path.run_name(), output_path.field_directory());
    const bool ok = run(config, trial_path, log, obs, residual);
    // File every flow that reached a usable IR endpoint, not only one that cleared the target:
    // with a positive Zc(0) target the accepted answer is by construction a below-target probe,
    // for which run() returns false.
    if (ok || (obs != nullptr && std::isfinite(*obs))) success_folders[u] = trial_path.root();
    return ok;
  };

  double u_lo, u_hi;
  const double u_hint = hint_m2A / L2;
  // The half-width is taken on |u_hint| rather than by scaling u_hint itself, so that the bracket
  // is built the same way whatever the sign of the critical point.
  const bool have_warm_start = hint_rel_delta > 0. && std::isfinite(u_hint) && u_hint != 0.;
  if (have_warm_start) {
    const double delta = hint_rel_delta * std::abs(u_hint);
    u_lo = u_hint - delta;
    u_hi = u_hint + delta;
    log.info("Warm-start bracket in u = m2A/Lambda^2: [{:.8e}, {:.8e}] around {:.8e}", u_lo, u_hi, u_hint);
  } else {
    u_lo = config.get_double("/tuning/m2A/u_lower");
    u_hi = config.get_double("/tuning/m2A/u_upper");
    log.info("Cold bracket in u = m2A/Lambda^2: [{:.8e}, {:.8e}]  (m2A: [{:.6e}, {:.6e}])", u_lo, u_hi, u_lo * L2,
             u_hi * L2);
  }
  if (u_lo > u_hi) std::swap(u_lo, u_hi);

  /// What one search at one fidelity produced. The fitted separatrix and the hard bracket are
  /// reported separately because a later stage is seeded with the bracket -- two flows that
  /// actually ran, one on each side -- and only hinted at by the fit.
  struct StageResult {
    double u_star = std::numeric_limits<double>::quiet_NaN();
    double u_c = std::numeric_limits<double>::quiet_NaN();
    double beta = std::numeric_limits<double>::quiet_NaN();
    double u_div = -std::numeric_limits<double>::infinity();
    double u_ok = std::numeric_limits<double>::infinity();
    bool converged = false;
    uint flows = 0;
  };

  /// Thrown out of the probe callback to end a stage whose bracket is already finer than the
  /// caller can use. ScalingRootFinder has no abort hook, and its x_rel_floor is no substitute:
  /// it measures the bracket against max(|x|, 1), which for u ~ 1e-3 is an absolute floor rather
  /// than a relative one. Thrown BEFORE the flow runs, so it costs nothing.
  struct BracketResolved {
  };

  const auto run_stage = [&](const double lo, const double hi, const char *tag, const double beta_seed,
                             const double stage_target, const double bracket_floor_rel) {
    // The hard bracket, tracked here because the finder exposes only its width. A probe is
    // evidence about the separatrix only when it produced no observable at all: a convergent flow
    // that merely came in under the target returns false too, and treating that as a runaway
    // would bracket the wrong thing.
    double u_div = -std::numeric_limits<double>::infinity();
    double u_ok = std::numeric_limits<double>::infinity();

    ScalingRootFinder search(
        [&](const double u, double &obs, double &residual) -> bool {
          if (bracket_floor_rel > 0. && std::isfinite(u_div) && std::isfinite(u_ok)) {
            const double mid = 0.5 * (u_div + u_ok);
            if (std::abs(mid) > 0. && u_ok - u_div <= bracket_floor_rel * std::abs(mid)) {
              log.info("{} bracket resolved to {:.6e} ({:.2f}% of u), at the limit of what this fidelity can be "
                       "trusted to; stopping",
                       tag, u_ok - u_div, 100. * (u_ok - u_div) / std::abs(mid));
              throw BracketResolved{};
            }
          }
          const bool ok = run_at(u, &obs, &residual);
          if (!std::isfinite(obs))
            u_div = std::max(u_div, u);
          else if (obs >= stage_target)
            u_ok = std::min(u_ok, u);
          // One line per flow, carrying everything needed to replay the search offline: which of
          // the two branches this probe fed, and where the fits stood when it was taken.
          log.info("{} probe {:2d}  u = {:.12e}  m2A = {:.12e}  {}  Zc(0) = {:.6e}  t_fail = {:.4f}  "
                   "| u_c = {:.12e}  beta = {:.4f}  theta = {:.4f}  bracket = {:.6e}",
                   tag, search.get_iter(), u, u * L2, ok ? "OK  " : (std::isfinite(obs) ? "LOW " : "DIV "), obs,
                   residual, search.get_x_critical(), search.get_exponent(), search.get_theta(),
                   search.bracket_width());
          return ok;
        },
        stage_target, rel_tol, tune_m2A_defaults::max_flows);

    // With a positive target the goal is not to land ON Zc(0) = zc_target but anywhere convergent
    // below it: the whole point of the target is to stay off the razor separatrix, and the gauge
    // avatars plateau across the entire near-critical basin. That turns the acceptance set from a
    // window of width rel_tol into the whole interval between the separatrix and the target
    // crossing, so a merely decent fit succeeds on its first attempt.
    if (stage_target > 0.) search.set_acceptance(ScalingRootFinder::Acceptance::BelowTarget);

    search.set_bounds(lo, hi);
    search.set_expansion_factor(tune_m2A_defaults::bracket_expansion);
    search.set_expansion_max_iter(tune_m2A_defaults::bracket_max_iter);
    search.set_obs_window(tune_m2A_defaults::obs_window);
    search.set_residual_window(tune_m2A_defaults::residual_window * config.get_double("/timestepping/final_time"));
    if (std::isfinite(beta_seed)) search.seed_exponent(beta_seed);

    StageResult r;
    try {
      r.u_star = search.search();
      r.converged = search.converged();
    } catch (const BracketResolved &) {
      // Not a failure: the stage did the whole of its job, which was to bracket the separatrix to
      // the requested resolution. Its best point is the tightest convergent probe it took.
      r.u_star = u_ok;
      r.converged = true;
    }
    r.u_c = search.get_x_critical();
    r.beta = search.get_exponent();
    r.u_div = u_div;
    r.u_ok = u_ok;
    r.flows = search.get_counts().total;

    const auto c = search.get_counts();
    log.info("{} stage: {} flows, u = {:.12e}, Zc(0) = {:.6e}, converged = {}", tag, c.total, r.u_star,
             search.get_obs(), r.converged);
    log.info("{}   budget: expand {}, power-law {}, mixed {}, approach {}, divergent {}, secant {}, bisect {}, "
             "forced {}",
             tag, c.expand, c.model_obs, c.model_mixed, c.approach, c.model_residual, c.secant, c.bisect,
             c.forced_bisect);
    if (std::isfinite(r.u_c))
      log.info("{}   fitted separatrix: u_c = {:.12e}, m2A_c = {:.12e}, beta = {:.4f}, theta = {:.4f}", tag, r.u_c,
               r.u_c * L2, r.beta, search.get_theta());
    return r;
  };

  // Pre-locate at reduced fidelity, then hand the full-fidelity stage a bracket around it. The
  // coarse answer is never used as an answer -- only as a bracket the expensive stage validates
  // and, if it does not straddle, widens geometrically. That is what makes a 16x cheaper
  // quadrature safe here and unsafe inside the STI objective. Skipped on a warm start: a previous
  // answer is a better bracket than a coarse re-derivation of one.
  double beta_seed = std::numeric_limits<double>::quiet_NaN();
  double pre_u_c = std::numeric_limits<double>::quiet_NaN();
  uint prelocate_flows = 0;
  if (!have_warm_start) {
    const uint x0 = config.get_uint("/integration/x_order");
    const uint c10 = config.get_uint("/integration/cos1_order");
    const uint c20 = config.get_uint("/integration/cos2_order");
    const uint p0 = config.get_uint("/integration/phi_order");
    const double f = tune_m2A_defaults::prelocate_quadrature_factor;
    config.set_uint("/integration/x_order", std::max(4u, (uint)(x0 * f)));
    config.set_uint("/integration/cos1_order", std::max(2u, (uint)(c10 * f)));
    config.set_uint("/integration/cos2_order", std::max(2u, (uint)(c20 * f)));
    config.set_uint("/integration/phi_order", std::max(2u, (uint)(p0 * f)));
    log.info("Pre-locate at reduced quadrature: x {} -> {}, cos1 {} -> {}, cos2 {} -> {}, phi {} -> {}", x0,
             config.get_uint("/integration/x_order"), c10, config.get_uint("/integration/cos1_order"), c20,
             config.get_uint("/integration/cos2_order"), p0, config.get_uint("/integration/phi_order"));

    const double pre_target = (zc_target > 0.) ? std::min(zc_target * tune_m2A_defaults::prelocate_target_factor,
                                                          tune_m2A_defaults::prelocate_target_max)
                                               : zc_target;
    const auto pre = run_stage(u_lo, u_hi, "[pre]", std::numeric_limits<double>::quiet_NaN(), pre_target,
                               tune_m2A_defaults::prelocate_shift_rel);
    prelocate_flows = pre.flows;

    config.set_uint("/integration/x_order", x0);
    config.set_uint("/integration/cos1_order", c10);
    config.set_uint("/integration/cos2_order", c20);
    config.set_uint("/integration/phi_order", p0);

    // Hand on the coarse BRACKET, not the coarse critical point: the bracket contains the coarse
    // separatrix by construction, whereas the fitted u_c can sit anywhere inside it. Padding both
    // ends by the fidelity shift covers the only thing the coarse stage cannot know.
    const double centre = std::isfinite(pre.u_c) ? pre.u_c : pre.u_star;
    pre_u_c = centre;
    const bool bracketed = std::isfinite(pre.u_div) && std::isfinite(pre.u_ok);
    if (pre.converged && (bracketed || std::isfinite(centre))) {
      const double scale = bracketed ? std::max(std::abs(pre.u_div), std::abs(pre.u_ok)) : std::abs(centre);
      const double pad = tune_m2A_defaults::prelocate_shift_rel * scale;
      u_lo = (bracketed ? pre.u_div : centre) - pad;
      u_hi = (bracketed ? pre.u_ok : centre) + pad;
      // Only carry the exponent if it looks like it came from inside the scaling window: the
      // pre-locate stops at a deliberately loose target, and seeding a fit taken outside the
      // asymptote is worse than seeding nothing.
      if (pre.beta > 0.3 && pre.beta < 1.5) beta_seed = pre.beta;
      log.info("Pre-located the separatrix in {} coarse flows: u_c = {:.12e}, coarse bracket [{:.12e}, {:.12e}]; "
               "full-fidelity bracket [{:.12e}, {:.12e}] (pad +-{:.2f}%)",
               pre.flows, centre, pre.u_div, pre.u_ok, u_lo, u_hi, 100. * tune_m2A_defaults::prelocate_shift_rel);
    } else
      log.warn("Pre-locate did not converge; falling back to the original bracket.");
  }

  // No bracket floor at full fidelity: here the bracket IS the answer, so there is no coarser
  // stage downstream to wash out the last bisections.
  const auto full = run_stage(u_lo, u_hi, "[full]", beta_seed, zc_target, 0.);
  const double m2A = full.u_star * L2;

  // What the pad was actually up against, so prelocate_shift_rel stays calibrated against
  // evidence. A shift approaching the pad means the next truncation over will need a widening.
  if (std::isfinite(pre_u_c) && std::isfinite(full.u_c) && pre_u_c != 0.) {
    const double shift = std::abs(full.u_c - pre_u_c) / std::abs(pre_u_c);
    log.info("Coarse-to-full shift of the separatrix: {:.3e} relative ({:.0f}% of the {:.3e} pad)", shift,
             100. * shift / tune_m2A_defaults::prelocate_shift_rel, tune_m2A_defaults::prelocate_shift_rel);
  }

  log.info("m2A tuning finished: m2A = {:.12e}, u = {:.12e}, target Zc(0) < {:.6e}, converged = {}", m2A, full.u_star,
           zc_target, full.converged);
  if (!full.converged)
    log.warn("m2A tuning did NOT reach the requested tolerance; the result is the best probe taken.");

  if (result != nullptr) {
    result->m2A = m2A;
    result->converged = full.converged;
    result->flows = prelocate_flows + full.flows;
  }

  // The returned point was actually run, so its own output IS the tuned result -- no confirmation
  // flow needed. Only fall back to a re-run if its folder is missing, which means the flow left
  // no readable output at all.
  const auto it = success_folders.find(full.u_star);
  if (it != success_folders.end())
    output_path.copy_tree_from(it->second);
  else {
    std::stringstream fname;
    fname << std::scientific << std::setprecision(12) << "final_" << m2A;
    config.set_double("/physical/m2A", m2A);
    log.info("Re-running the converged point at m2A = {:.12e}", m2A);
    const auto final_path = tuning_path.child(fname.str(), output_path.run_name(), output_path.field_directory());
    run(config, final_path, log, nullptr, nullptr);
    output_path.copy_tree_from(final_path.root());
  }

  return m2A;
}

/**
 * @brief Drive the three gauge-coupling avatars together by varying alphaA3 and alphaA4.
 *
 * At a Slavnov-Taylor respecting truncation the couplings extracted from the A^3, A^4 and Acbc
 * vertices coincide; the truncation error shows up as a spread between them. Minimising that
 * spread over the two initial couplings (alphaAcbc is held fixed as the overall scale) is what
 * fixes the initial condition.
 *
 * The objective is expensive -- each evaluation is a full m2A tune -- so two things reduce it:
 * the flow is shortened and the quadrature coarsened during the search, both restored for the
 * final refinement, and every tune after the first is warm-started from the previous answer,
 * which is what makes the nested search affordable.
 */
template <typename FUN> void tune_STI(ConfigTree &config, const OutputPath &output_path, const FUN &run)
{
  const auto tuning_path = output_path.child(output_path.run_name() + "_STI_tuning", "tuning");
  RunReporter tuning_logger(tuning_path, OutputSettings(config), MPI::rank(MPI_COMM_WORLD) == 0);
  const auto log = tuning_logger.port();

  const double original_final_time = config.get_double("/timestepping/final_time");
  const uint original_x_order = config.get_uint("/integration/x_order");
  const uint original_cos1_order = config.get_uint("/integration/cos1_order");
  const uint original_cos2_order = config.get_uint("/integration/cos2_order");
  const uint original_phi_order = config.get_uint("/integration/phi_order");

  const double quadrature_factor = config.get_double("/tuning/STI/quadrature_factor", 1.0);
  config.set_uint("/integration/x_order", std::max(1u, (uint)(original_x_order * quadrature_factor)));
  config.set_uint("/integration/cos1_order", std::max(1u, (uint)(original_cos1_order * quadrature_factor)));
  config.set_uint("/integration/cos2_order", std::max(1u, (uint)(original_cos2_order * quadrature_factor)));
  config.set_uint("/integration/phi_order", std::max(1u, (uint)(original_phi_order * quadrature_factor)));
  config.set_double("/timestepping/final_time", config.get_double("/tuning/STI/flow_final_time"));

  log.info("Search fidelity: quadrature x{:.2f} (x {} -> {}, cos1 {} -> {}, cos2 {} -> {}, phi {} -> {}), "
           "flow to t = {:.2f} instead of {:.2f}",
           quadrature_factor, original_x_order, config.get_uint("/integration/x_order"), original_cos1_order,
           config.get_uint("/integration/cos1_order"), original_cos2_order, config.get_uint("/integration/cos2_order"),
           original_phi_order, config.get_uint("/integration/phi_order"),
           config.get_double("/timestepping/final_time"), original_final_time);

  uint counter = 0;
  uint inner_flows = 0;
  double last_m2A = 0.0;
  double warm_delta = -1.0;
  const double coarse_tol = config.get_double("/tuning/m2A/rel_tol") * config.get_double("/tuning/STI/coarse_factor");

  GSLSimplexMinimizer<2> minimizer(
      [&](const std::array<double, 2> &x) -> double {
        const auto step_path =
            tuning_path.child("step_" + std::to_string(counter), output_path.run_name(), output_path.field_directory());

        config.set_double("/physical/alphaA3", x[0]);
        config.set_double("/physical/alphaA4", x[1]);

        log.info("STI tuning step {}: alphaAcbc = {:.8e}, alphaA3 = {:.8e}, alphaA4 = {:.8e}", counter,
                 config.get_double("/physical/alphaAcbc"), x[0], x[1]);

        M2AResult inner;
        last_m2A = tune_m2A(config, step_path, run, last_m2A, warm_delta, coarse_tol, &inner);
        // Every tune after the first is warm-started from the previous answer.
        warm_delta = tune_m2A_defaults::warm_start_rel_delta;
        inner_flows += inner.flows;
        counter++;
        if (!inner.converged)
          log.warn("STI step {}: the nested m2A tune did not converge; this objective value is unreliable", counter);

        HDF5Input hdf5_input(step_path.run_file(".h5").string());
        const auto ZA3 = hdf5_input.load_map("ZA3");
        const auto ZA4 = hdf5_input.load_map("ZA4");
        const auto ZAcbc = hdf5_input.load_map("ZAcbc");
        const auto ZA = hdf5_input.load_map("ZA");
        const auto Zc = hdf5_input.load_map("Zc");
        const auto pGeV = hdf5_input.load_map_coord<1>("ZA");

        // The avatars are compared at two momenta, p_low and 2 p_low, both well inside the
        // perturbative window where the identity is supposed to hold exactly.
        const double p_low = config.get_double("/tuning/STI/scale");
        const auto nearest = [&](const double target) {
          uint best = 0;
          double best_dist = std::numeric_limits<double>::max();
          for (uint i = 0; i < pGeV.size(); ++i)
            if (std::abs(pGeV[i] - target) < best_dist) {
              best_dist = std::abs(pGeV[i] - target);
              best = i;
            }
          return best;
        };
        const uint idx_low = nearest(p_low);
        const uint idx_high = nearest(2. * p_low);

        // Invariant couplings alpha = g^2/(4 pi), each built from its own vertex dressing and the
        // propagator dressings of its legs.
        const auto avatar_spread = [&](const uint i) {
          const double a3 = powr<2>(ZA3[i]) / (powr<3>(ZA[i]) * 4 * M_PI);
          const double a4 = ZA4[i] / (powr<2>(ZA[i]) * 4 * M_PI);
          const double acbc = powr<2>(ZAcbc[i]) / (ZA[i] * powr<2>(Zc[i]) * 4 * M_PI);
          return std::max({std::abs(a3 - acbc), std::abs(a3 - a4), std::abs(a4 - acbc)});
        };
        const double diff = 0.7 * powr<2>(avatar_spread(idx_low)) + 0.3 * powr<2>(avatar_spread(idx_high));

        log.info("STI step {}: avatar spread at p = {:.3f} / {:.3f} GeV, objective = {:.8e} ({} inner flows so far)",
                 counter, pGeV[idx_low], pGeV[idx_high], diff, inner_flows);
        return diff;
      },
      config.get_double("/tuning/STI/tol"), 1000);

  minimizer.set_step_size(config.get_double("/tuning/STI/step_size"));
  minimizer.set_x0({{config.get_double("/physical/alphaA3"), config.get_double("/physical/alphaA4")}});
  const std::array<double, 2> minimum = minimizer.minimize();

  log.info("STI search finished after {} objective evaluations ({} flows): alphaA3 = {:.8e}, alphaA4 = {:.8e}",
           counter, inner_flows, minimum[0], minimum[1]);

  // Final refinement: the optimal couplings, at full quadrature and over the full flow.
  config.set_double("/physical/alphaA3", minimum[0]);
  config.set_double("/physical/alphaA4", minimum[1]);
  config.set_double("/timestepping/final_time", original_final_time);
  config.set_uint("/integration/x_order", original_x_order);
  config.set_uint("/integration/cos1_order", original_cos1_order);
  config.set_uint("/integration/cos2_order", original_cos2_order);
  config.set_uint("/integration/phi_order", original_phi_order);

  const auto final_path = tuning_path.child("final", output_path.run_name(), output_path.field_directory());
  // The fidelity change moves the separatrix, so the final tune gets a widened warm-start bracket
  // rather than the search-fidelity one.
  tune_m2A(config, final_path, run, last_m2A, 4. * tune_m2A_defaults::warm_start_rel_delta);

  output_path.copy_tree_from(final_path.root());
}
