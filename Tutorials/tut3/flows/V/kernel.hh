#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class V_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const auto &k, const auto &N, const auto &T, const auto &m2Pi,
                                                   const auto &m2Sigma)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = CothFiniteT(sqrt(powr<2>(l1) + m2Pi + RB(powr<2>(k), powr<2>(l1))), T);
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = CothFiniteT(sqrt(powr<2>(l1) + m2Sigma + RB(powr<2>(k), powr<2>(l1))), T);
      const auto _cse1 = powr<2>(l1);
      return 0.25 *
        fma(-1., _interp2 * _interp3 * sqrt(powr<-1>(_cse1 + _interp1 + m2Pi)),
            fma(_interp3, _interp4 * sqrt(powr<-1>(_cse1 + _interp1 + m2Sigma)),
                fma(_interp2, _interp3 * sqrt(powr<-1>(_cse1 + _interp1 + m2Pi)) * N, 0.)));
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto &k, const auto &N, const auto &T, const auto &m2Pi, const auto &m2Sigma)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      return 0.;
    }

  private:
    static KOKKOS_FORCEINLINE_FUNCTION auto RB(const auto &k2, const auto &p2) { return Regulator::RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RF(const auto &k2, const auto &p2) { return Regulator::RF(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RBdot(const auto &k2, const auto &p2) { return Regulator::RBdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RFdot(const auto &k2, const auto &p2) { return Regulator::RFdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RB(const auto &k2, const auto &p2) { return Regulator::dq2RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RF(const auto &k2, const auto &p2) { return Regulator::dq2RF(k2, p2); }
  };
} // namespace DiFfRG
using DiFfRG::V_kernel;