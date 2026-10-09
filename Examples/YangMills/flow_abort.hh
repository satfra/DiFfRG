#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

/**
 * @file flow_abort.hh
 *
 * The runaway criterion and the exception carrying it, shared by the two Yang-Mills apps in this
 * directory. It lives apart from tuning.hh so that a model.hh can include it without pulling in
 * the whole m2A/STI machinery.
 */

/**
 * @brief Thrown from a model's dt_variables() the moment its flow is demonstrably running away.
 *
 * Carries the RG time at which that happened, which is exactly the residual the tuner's
 * divergent-branch fit wants -- so aborting costs no information and saves the rest of the
 * integration. On the Higgs side of the separatrix that is most of it: the flow blows up at
 * intermediate k and the stepper then crawls, without ever throwing, for the remaining RG time.
 */
struct FlowAbort : public std::runtime_error {
  explicit FlowAbort(const double t) : std::runtime_error("flow ran away at t = " + std::to_string(t)), t(t) {}
  double t;
};

/**
 * @brief The runaway criterion, shared by the in-flow abort and the post-hoc trajectory scan.
 *
 * One definition and two call sites, so that a probe cannot be classified differently depending
 * on which of the two stopped it. Detection is on |m2A(t)| leaving a wide band around its own
 * initial value:
 *   - Zc(p_min) <= 0 does not suffice on its own; it can stay positive through a genuine runaway.
 *   - ZA(p_min) <= 0 is useless: ZA = (p^2 + m2A)/p^2 is already negative at t = 0 for every
 *     probe below the separatrix, healthy ones included.
 *   - |m2A(t)| leaving its own initial scale by 100x fires on every diverging probe and on no
 *     converged one, and does so monotonically in the distance to criticality -- which is the
 *     property the divergence-time fit rests on.
 * Zc going non-positive or non-finite is kept as an additional trigger, whichever comes first.
 *
 * The exact multiplier only shifts the abort time by a roughly constant offset, and the fit
 * depends on differences of that time alone, so it does not need to be a parameter.
 *
 * @param m2A_t   IR gluon mass along the flow, ZA(p_min) p_min^2 - p_min^2
 * @param m2A_0   its initial value, which sets the scale
 * @param zc_min  smallest ghost dressing on the grid
 */
inline bool flow_has_run_away(const double m2A_t, const double m2A_0, const double zc_min)
{
  constexpr double runaway_factor = 100.;
  if (!std::isfinite(m2A_t) || std::abs(m2A_t) > runaway_factor * std::max(std::abs(m2A_0), 1.)) return true;
  return !std::isfinite(zc_min) || zc_min <= 0.;
}
