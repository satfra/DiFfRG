#pragma once

#include <DiFfRG/common/quadrature/quadrature_provider.hh>

#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>

#include <DiFfRG/physics/integration/vacuum/integrator_p2.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_1ang.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_4d_2ang.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_4d_3ang.hh>

#include <DiFfRG/physics/integration/finiteT/integrator_fT.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_1ang.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_4d_2ang.hh>

#include <DiFfRG/physics/integration/lattice/integrator_lat.hh>

/**
 * @file
 * @brief All momentum integrators, plus helpers that forward the RG scale and the temperature to them.
 *
 * Generated flow classes hold one integrator per precision in members `integrator`, and optionally `integrator_AD`
 * and `integrator_AD2` (for automatic-differentiation types). The `all_set_*` functions call the corresponding setter
 * on each of these members that has it, and skip those that do not, e.g.
 * @code
 * DiFfRG::all_set_k(flow, k); // call every RG step: the radial cutoff of most integrators scales with k
 * DiFfRG::all_set_T(flow, T); // finite-temperature integrators only
 * @endcode
 */

namespace DiFfRG
{
  /// `T` has a member `integrator_AD`.
  template <typename T>
  concept has_integrator_AD = requires(T t) { t.integrator_AD; };

  /// `T` has a member `integrator_AD2`.
  template <typename T>
  concept has_integrator_AD2 = requires(T t) { t.integrator_AD2; };

  // ----------------------------------------------------
  // Setting scale k
  // ----------------------------------------------------

  /// `T` has a member function `set_k(double)`.
  template <typename T>
  concept has_set_k = requires(T t, double k) { t.set_k(k); };

  /// Calls `integrator.set_k(k)`.
  template <typename Int>
    requires DiFfRG::has_set_k<Int>
  void invoke_set_k(Int &integrator, const double k)
  {
    integrator.set_k(k);
  }
  /// Does nothing: the integrator has no `set_k`.
  template <typename Int>
    requires(!DiFfRG::has_set_k<Int>)
  void invoke_set_k(Int &, const double)
  {
    // do nothing
  }
  /**
   * @brief Sets the RG scale \f$k\f$ on every integrator of `flow`.
   *
   * Calls `set_k` on `flow.integrator` and, if present, on `flow.integrator_AD` and `flow.integrator_AD2`.
   * Members without a `set_k` are skipped silently.
   */
  template <typename Int> void all_set_k(Int &flow, const double k)
  {
    invoke_set_k(flow.integrator, k);
    if constexpr (DiFfRG::has_integrator_AD<Int>) invoke_set_k(flow.integrator_AD, k);
    if constexpr (DiFfRG::has_integrator_AD2<Int>) invoke_set_k(flow.integrator_AD2, k);
  }

  // ----------------------------------------------------
  // Setting temperature T
  // ----------------------------------------------------

  /// `T` has a member function `set_T(double)`.
  template <typename T>
  concept has_set_T = requires(T t, double mT) { t.set_T(mT); };

  /// Calls `integrator.set_T(T)`.
  template <typename Int>
    requires DiFfRG::has_set_T<Int>
  void invoke_set_T(Int &integrator, const double T)
  {
    integrator.set_T(T);
  }
  /// Does nothing: the integrator has no `set_T`.
  template <typename Int>
    requires(!DiFfRG::has_set_T<Int>)
  void invoke_set_T(Int &, const double)
  {
    // do nothing
  }
  /**
   * @brief Sets the temperature \f$T\f$ on every integrator of `flow`.
   *
   * Calls `set_T` on `flow.integrator` and, if present, on `flow.integrator_AD` and `flow.integrator_AD2`.
   * Members without a `set_T` are skipped silently.
   */
  template <typename Int> void all_set_T(Int &flow, const double T)
  {
    invoke_set_T(flow.integrator, T);
    if constexpr (DiFfRG::has_integrator_AD<Int>) invoke_set_T(flow.integrator_AD, T);
    if constexpr (DiFfRG::has_integrator_AD2<Int>) invoke_set_T(flow.integrator_AD2, T);
  }

  // ----------------------------------------------------
  // Setting the typical energy scale
  // ----------------------------------------------------

  /// `T` has a member function `set_typical_E(double)`.
  template <typename T>
  concept has_set_typical_E = requires(T t, double typical_E) { t.set_typical_E(typical_E); };

  /// Calls `integrator.set_typical_E(typical_E)`.
  template <typename Int>
    requires DiFfRG::has_set_typical_E<Int>
  void invoke_set_typical_E(Int &integrator, const double typical_E)
  {
    integrator.set_typical_E(typical_E);
  }
  /// Does nothing: the integrator has no `set_typical_E`.
  template <typename Int>
    requires(!DiFfRG::has_set_typical_E<Int>)
  void invoke_set_typical_E(Int &, const double)
  {
    // do nothing
  }
  /**
   * @brief Sets the typical energy scale (see QuadratureIntegrator_fT::set_typical_E) on every integrator of `flow`.
   *
   * Calls `set_typical_E` on `flow.integrator` and, if present, on `flow.integrator_AD` and `flow.integrator_AD2`.
   * Members without a `set_typical_E` are skipped silently.
   */
  template <typename Int> void all_set_typical_E(Int &flow, const double typical_E)
  {
    invoke_set_typical_E(flow.integrator, typical_E);
    if constexpr (DiFfRG::has_integrator_AD<Int>) invoke_set_typical_E(flow.integrator_AD, typical_E);
    if constexpr (DiFfRG::has_integrator_AD2<Int>) invoke_set_typical_E(flow.integrator_AD2, typical_E);
  }

  // ----------------------------------------------------
  // Setting the x-extent
  // ----------------------------------------------------

  /// `T` has a member function `set_x_extent(double)`.
  template <typename T>
  concept has_set_x_extent = requires(T t, double x_extent) { t.set_x_extent(x_extent); };

  /// Calls `integrator.set_x_extent(x_extent)`.
  template <typename Int>
    requires DiFfRG::has_set_x_extent<Int>
  void invoke_set_x_extent(Int &integrator, const double x_extent)
  {
    integrator.set_x_extent(x_extent);
  }
  /// Does nothing: the integrator has no `set_x_extent`.
  template <typename Int>
    requires(!DiFfRG::has_set_x_extent<Int>)
  void invoke_set_x_extent(Int &, const double)
  {
    // do nothing
  }
  /**
   * @brief Sets the radial cutoff \f$x_\text{extent}\f$ (in units of \f$k^2\f$) on every integrator of `flow`.
   *
   * Calls `set_x_extent` on `flow.integrator` and, if present, on `flow.integrator_AD` and `flow.integrator_AD2`.
   * Members without a `set_x_extent` are skipped silently.
   */
  template <typename Int> void all_set_x_extent(Int &flow, const double x_extent)
  {
    invoke_set_x_extent(flow.integrator, x_extent);
    if constexpr (DiFfRG::has_integrator_AD<Int>) invoke_set_x_extent(flow.integrator_AD, x_extent);
    if constexpr (DiFfRG::has_integrator_AD2<Int>) invoke_set_x_extent(flow.integrator_AD2, x_extent);
  }
} // namespace DiFfRG
