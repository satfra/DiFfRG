#pragma once

// Per-mode flow-equation kernels for the integrator-based LSM tests, plus the
// IntegratorWrapper that owns the three NumberType specialisations (double,
// Real<1>, Real<2>) needed by the KT assembler.
//
// Kernel form (standard d=3+1 finite-T LPA, as in
// Examples/ONfiniteT/flows/V_pion/kernel.hh):
//   0.25 · CothFiniteT(E/2T) · RBdot · 1/E,   E = sqrt(p² + m² + RB).
// The combined `coth(E/2T)/E` factor encodes the Matsubara-summed bosonic
// propagator. T → 0 reduces to 1/E (vacuum); T → ∞ reduces to 2T/E² (the
// dimensional-reduction form, i.e. 1/(p²+m²+RB) up to the 2T prefactor).
//
// Litim T=0 variants drop CothFiniteT (which is +1 at T=0). For Litim with
// RB = (k²-p²)·θ and RBdot = 2k²·θ, integrated over d³p/(2π)³ on p<k support,
// the closed form is k⁵ / (12π² · sqrt(k² + m²)); finite-T Litim closed form
// is k⁵ · coth(sqrt(k²+m²)/(2T)) / (12π² · sqrt(k² + m²)).

#include <DiFfRG/physics/integration.hh>
#include <DiFfRG/physics/physics.hh>
#include <DiFfRG/physics/regulators.hh>
#include <DiFfRG/physics/thermodynamics.hh>

#include <autodiff/forward/real/real.hpp>

namespace on_kt_3D
{
  // --- Finite-T (PolyExp) per-mode kernels -----------------------------------

  // Note: subexpression ordering and (commutative) operand order is reproduced
  // byte-for-byte from Examples/ONfiniteT/flows/V_pion/kernel.hh so that the
  // integrator output is bit-identical between the double and autodiff::real
  // specialisations (any reordering of `a+b+c` changes the AD ε-coefficient
  // accumulation by ulps, which can sum to O(1e-6) over the sum over
  // quadrature points — see Phase N diagnostic).
  template <typename _Regulator> class V_pion_kernel_polyexp
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const auto &k, const auto & /*N*/,
                                                   const auto &T, const auto &m2Pi)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using Kokkos::sqrt;
      const auto _interp1 = Regulator::RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = CothFiniteT(sqrt(powr<2>(l1) + m2Pi + _interp1), T);
      const auto _interp3 = Regulator::RBdot(powr<2>(k), powr<2>(l1));
      const auto _cse1 = powr<2>(l1);
      return (0.25) * (_interp2) * (_interp3) * (sqrt(powr<-1>(_cse1 + _interp1 + m2Pi)));
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto & /*k*/, const auto & /*N*/, const auto & /*T*/,
                                                     const auto & /*m2Pi*/)
    {
      return 0.0;
    }
  };

  template <typename _Regulator> class V_sigma_kernel_polyexp
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const auto &k, const auto & /*N*/,
                                                   const auto &T, const auto &m2Sigma)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using Kokkos::sqrt;
      const auto _interp1 = Regulator::RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = Regulator::RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = CothFiniteT(sqrt(powr<2>(l1) + m2Sigma + _interp1), T);
      const auto _cse1 = powr<2>(l1);
      return (0.25) * (_interp3) * (_interp4) * (sqrt(powr<-1>(_cse1 + _interp1 + m2Sigma)));
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto & /*k*/, const auto & /*N*/, const auto & /*T*/,
                                                     const auto & /*m2Sigma*/)
    {
      return 0.0;
    }
  };

  // --- T=0 / Litim variants (no CothFiniteT, which is +1 at T=0) --------------

  template <typename _Regulator> class V_pion_kernel_litim_T0
  {
  public:
    using Regulator = _Regulator;
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const auto &k, const auto & /*N*/,
                                                   const auto & /*T*/, const auto &m2Pi)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using Kokkos::sqrt;
      const auto _interp1 = Regulator::RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = Regulator::RBdot(powr<2>(k), powr<2>(l1));
      const auto _cse1 = powr<2>(l1);
      return (0.25) * (_interp3) * (sqrt(powr<-1>(_cse1 + _interp1 + m2Pi)));
    }
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto & /*k*/, const auto & /*N*/, const auto & /*T*/,
                                                     const auto & /*m2Pi*/)
    {
      return 0.0;
    }
  };

  template <typename _Regulator> class V_sigma_kernel_litim_T0
  {
  public:
    using Regulator = _Regulator;
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const auto &k, const auto & /*N*/,
                                                   const auto & /*T*/, const auto &m2Sigma)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using Kokkos::sqrt;
      const auto _interp1 = Regulator::RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = Regulator::RBdot(powr<2>(k), powr<2>(l1));
      const auto _cse1 = powr<2>(l1);
      return (0.25) * (_interp3) * (sqrt(powr<-1>(_cse1 + _interp1 + m2Sigma)));
    }
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto & /*k*/, const auto & /*N*/, const auto & /*T*/,
                                                     const auto & /*m2Sigma*/)
    {
      return 0.0;
    }
  };

  // --- Integrator wrapper ----------------------------------------------------
  //
  // Owns the three NT specialisations of Integrator_p2<3>. The Real<2, double>
  // variant is required because KT's AD flux-derivative extraction seeds
  // second-order AD for the wave-speed Hessian.

  template <typename Kernel> class IntegratorWrapper
  {
  public:
    using Regulator = typename Kernel::Regulator;

    IntegratorWrapper(DiFfRG::QuadratureProvider &qp, const DiFfRG::ConfigTree &json)
        : integrator(qp, json), integrator_AD(qp, json), integrator_AD2(qp, json)
    {}

    DiFfRG::Integrator_p2<3, double, Kernel, DiFfRG::TBB_exec> integrator;
    DiFfRG::Integrator_p2<3, autodiff::real, Kernel, DiFfRG::TBB_exec> integrator_AD;
    DiFfRG::Integrator_p2<3, autodiff::Real<2, double>, Kernel, DiFfRG::TBB_exec> integrator_AD2;

    void get(double &dest, const double &k, const double &N, const double &T, const double &m2)
    {
      integrator.get(dest, k, N, T, m2);
    }
    void get(autodiff::real &dest, const double &k, const double &N, const double &T, const autodiff::real &m2)
    {
      integrator_AD.get(dest, k, N, T, m2);
    }
    void get(autodiff::Real<2, double> &dest, const double &k, const double &N, const double &T,
             const autodiff::Real<2, double> &m2)
    {
      integrator_AD2.get(dest, k, N, T, m2);
    }
  };

  using V_pion_polyexp_t = IntegratorWrapper<V_pion_kernel_polyexp<DiFfRG::PolynomialExpRegulator<>>>;
  using V_sigma_polyexp_t = IntegratorWrapper<V_sigma_kernel_polyexp<DiFfRG::PolynomialExpRegulator<>>>;
  using V_pion_litim_T0_t = IntegratorWrapper<V_pion_kernel_litim_T0<DiFfRG::LitimRegulator<>>>;
  using V_sigma_litim_T0_t = IntegratorWrapper<V_sigma_kernel_litim_T0<DiFfRG::LitimRegulator<>>>;

  // --- Flow containers (own QuadratureProvider + integrator pair) -----------

  class PolyExpFlows
  {
  public:
    PolyExpFlows(const DiFfRG::ConfigTree &json)
        : quadrature_provider(json), V_pion(quadrature_provider, json), V_sigma(quadrature_provider, json)
    {}
    void set_k(double k)
    {
      DiFfRG::all_set_k(V_pion, k);
      DiFfRG::all_set_k(V_sigma, k);
    }
    void set_T(double T)
    {
      DiFfRG::all_set_T(V_pion, T);
      DiFfRG::all_set_T(V_sigma, T);
    }

    DiFfRG::QuadratureProvider quadrature_provider;
    V_pion_polyexp_t V_pion;
    V_sigma_polyexp_t V_sigma;
  };

  class LitimT0Flows
  {
  public:
    LitimT0Flows(const DiFfRG::ConfigTree &json)
        : quadrature_provider(json), V_pion(quadrature_provider, json), V_sigma(quadrature_provider, json)
    {}
    void set_k(double k)
    {
      DiFfRG::all_set_k(V_pion, k);
      DiFfRG::all_set_k(V_sigma, k);
    }
    void set_T(double T)
    {
      DiFfRG::all_set_T(V_pion, T);
      DiFfRG::all_set_T(V_sigma, T);
    }

    DiFfRG::QuadratureProvider quadrature_provider;
    V_pion_litim_T0_t V_pion;
    V_sigma_litim_T0_t V_sigma;
  };

} // namespace on_kt_3D
