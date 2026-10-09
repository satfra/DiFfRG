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

namespace DiFfRG
{
  template <typename T>
  concept has_integrator_AD = requires(T t) { t.integrator_AD; };

  template <typename T>
  concept has_integrator_AD2 = requires(T t) { t.integrator_AD2; };

  // ----------------------------------------------------
  // Setting scale k
  // ----------------------------------------------------

  template <typename T>
  concept has_set_k = requires(T t, double k) { t.set_k(k); };

  template <typename Int>
    requires DiFfRG::has_set_k<Int>
  void invoke_set_k(Int &integrator, const double k)
  {
    integrator.set_k(k);
  }
  template <typename Int>
    requires(!DiFfRG::has_set_k<Int>)
  void invoke_set_k(Int &, const double)
  {
    // do nothing
  }
  template <typename Int> void all_set_k(Int &integrator, const double k)
  {
    invoke_set_k(integrator.integrator, k);
    if constexpr (DiFfRG::has_integrator_AD<Int>) invoke_set_k(integrator.integrator_AD, k);
    if constexpr (DiFfRG::has_integrator_AD2<Int>) invoke_set_k(integrator.integrator_AD2, k);
  }

  // ----------------------------------------------------
  // Setting temperature T
  // ----------------------------------------------------

  template <typename T>
  concept has_set_T = requires(T t, double mT) { t.set_T(mT); };

  template <typename Int>
    requires DiFfRG::has_set_T<Int>
  void invoke_set_T(Int &integrator, const double T)
  {
    integrator.set_T(T);
  }
  template <typename Int>
    requires(!DiFfRG::has_set_T<Int>)
  void invoke_set_T(Int &, const double)
  {
    // do nothing
  }
  template <typename Int> void all_set_T(Int &integrator, const double T)
  {
    invoke_set_T(integrator.integrator, T);
    if constexpr (DiFfRG::has_integrator_AD<Int>) invoke_set_T(integrator.integrator_AD, T);
    if constexpr (DiFfRG::has_integrator_AD2<Int>) invoke_set_T(integrator.integrator_AD2, T);
  }

  // ----------------------------------------------------
  // Setting the typical energy scale
  // ----------------------------------------------------

  template <typename T>
  concept has_set_typical_E = requires(T t, double typical_E) { t.set_typical_E(typical_E); };

  template <typename Int>
    requires DiFfRG::has_set_typical_E<Int>
  void invoke_set_typical_E(Int &integrator, const double typical_E)
  {
    integrator.set_typical_E(typical_E);
  }
  template <typename Int>
    requires(!DiFfRG::has_set_typical_E<Int>)
  void invoke_set_typical_E(Int &, const double)
  {
    // do nothing
  }
  template <typename Int> void all_set_typical_E(Int &integrator, const double typical_E)
  {
    invoke_set_typical_E(integrator.integrator, typical_E);
    if constexpr (DiFfRG::has_integrator_AD<Int>) invoke_set_typical_E(integrator.integrator_AD, typical_E);
    if constexpr (DiFfRG::has_integrator_AD2<Int>) invoke_set_typical_E(integrator.integrator_AD2, typical_E);
  }

  // ----------------------------------------------------
  // Setting the x-extent
  // ----------------------------------------------------

  template <typename T>
  concept has_set_x_extent = requires(T t, double x_extent) { t.set_x_extent(x_extent); };

  template <typename Int>
    requires DiFfRG::has_set_x_extent<Int>
  void invoke_set_x_extent(Int &integrator, const double x_extent)
  {
    integrator.set_x_extent(x_extent);
  }
  template <typename Int>
    requires(!DiFfRG::has_set_x_extent<Int>)
  void invoke_set_x_extent(Int &, const double)
  {
    // do nothing
  }
  template <typename Int> void all_set_x_extent(Int &integrator, const double x_extent)
  {
    invoke_set_x_extent(integrator.integrator, x_extent);
    if constexpr (DiFfRG::has_integrator_AD<Int>) invoke_set_x_extent(integrator.integrator_AD, x_extent);
    if constexpr (DiFfRG::has_integrator_AD2<Int>) invoke_set_x_extent(integrator.integrator_AD2, x_extent);
  }
} // namespace DiFfRG
