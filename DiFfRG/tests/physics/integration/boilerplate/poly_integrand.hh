#pragma once

#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>

template <typename T> struct expected_precision {
  static constexpr T value = 1e-9;
};
template <> struct expected_precision<float> {
  static constexpr float value = 2e-3;
};
template <> struct expected_precision<double> {
  static constexpr double value = 1e-10;
};

template <int dim, typename T, int inv = 1> class PolyIntegrand
{
};

template <typename T, int inv> class PolyIntegrand<1, T, inv>
{
public:
  using ctype = typename DiFfRG::get_type::ctype<T>;
  static KOKKOS_INLINE_FUNCTION auto kernel(const ctype q, const T /*c*/, const T x0, const T x1, const T x2,
                                            const T x3)
  {
    using DiFfRG::powr;
    return powr<inv>(x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q));
  }

  static KOKKOS_INLINE_FUNCTION auto constant(const T c, const T /*x0*/, const T /*x1*/, const T /*x2*/, const T /*x3*/)
  {
    return c;
  }
};

template <typename T> class ParameterizedPolynomialIntegrand
{
public:
  using ctype = typename DiFfRG::get_type::ctype<T>;

  static KOKKOS_INLINE_FUNCTION T kernel(const ctype q, const T m)
  {
    using DiFfRG::powr;
    return m + m * m * q + m * m * m * powr<2>(q);
  }

  static KOKKOS_INLINE_FUNCTION T constant(const T /*m*/) { return {}; }
};

template <typename T, int inv> class PolyIntegrand<2, T, inv>
{
public:
  using ctype = typename DiFfRG::get_type::ctype<T>;
  static KOKKOS_INLINE_FUNCTION auto kernel(const ctype qx, const ctype qy, const T /*c*/, const T x0, const T x1,
                                            const T x2, const T x3, const T y0, const T y1, const T y2, const T y3)
  {
    using DiFfRG::powr;
    return (x0 + x1 * powr<1>(qx) + x2 * powr<2>(qx) + x3 * powr<3>(qx)) *
           powr<inv>(y0 + y1 * powr<1>(qy) + y2 * powr<2>(qy) + y3 * powr<3>(qy));
  }

  static KOKKOS_INLINE_FUNCTION auto constant(const T c, const T /*x0*/, const T /*x1*/, const T /*x2*/, const T /*x3*/,
                                              const T /*y0*/, const T /*y1*/, const T /*y2*/, const T /*y3*/)
  {
    return c;
  }
};

template <typename T, int inv> class PolyIntegrand<3, T, inv>
{
public:
  using ctype = typename DiFfRG::get_type::ctype<T>;
  static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const ctype qx, const ctype qy, const ctype qz, const T /*c*/,
                                                 const T x0, const T x1, const T x2, const T x3, const T y0, const T y1,
                                                 const T y2, const T y3, const T z0, const T z1, const T z2, const T z3)
  {
    using DiFfRG::powr;
    return (x0 + x1 * powr<1>(qx) + x2 * powr<2>(qx) + x3 * powr<3>(qx)) *
           (y0 + y1 * powr<1>(qy) + y2 * powr<2>(qy) + y3 * powr<3>(qy)) *
           powr<inv>(z0 + z1 * powr<1>(qz) + z2 * powr<2>(qz) + z3 * powr<3>(qz));
  }

  static KOKKOS_FORCEINLINE_FUNCTION auto constant(const T c, const T /*x0*/, const T /*x1*/, const T /*x2*/,
                                                   const T /*x3*/, const T /*y0*/, const T /*y1*/, const T /*y2*/,
                                                   const T /*y3*/, const T /*z0*/, const T /*z1*/, const T /*z2*/,
                                                   const T /*z3*/)
  {
    return c;
  }
};

template <typename T, int inv> class PolyIntegrand<4, T, inv>
{
public:
  using ctype = typename DiFfRG::get_type::ctype<T>;
  static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const ctype qx, const ctype qy, const ctype qz, const ctype qw,
                                                 const T /*c*/, const T x0, const T x1, const T x2, const T x3,
                                                 const T y0, const T y1, const T y2, const T y3, const T z0, const T z1,
                                                 const T z2, const T z3, const T w0, const T w1, const T w2, const T w3)
  {
    using DiFfRG::powr;
    return (x0 + x1 * powr<1>(qx) + x2 * powr<2>(qx) + x3 * powr<3>(qx)) *
           (y0 + y1 * powr<1>(qy) + y2 * powr<2>(qy) + y3 * powr<3>(qy)) *
           (z0 + z1 * powr<1>(qz) + z2 * powr<2>(qz) + z3 * powr<3>(qz)) *
           powr<inv>(w0 + w1 * powr<1>(qw) + w2 * powr<2>(qw) + w3 * powr<3>(qw));
  }

  static KOKKOS_FORCEINLINE_FUNCTION auto constant(const T c, const T /*x0*/, const T /*x1*/, const T /*x2*/,
                                                   const T /*x3*/, const T /*y0*/, const T /*y1*/, const T /*y2*/,
                                                   const T /*y3*/, const T /*z0*/, const T /*z1*/, const T /*z2*/,
                                                   const T /*z3*/, const T /*w0*/, const T /*w1*/, const T /*w2*/,
                                                   const T /*w3*/)
  {
    return c;
  }
};

template <typename T, int inv> class PolyIntegrand<5, T, inv>
{
public:
  using ctype = typename DiFfRG::get_type::ctype<T>;
  static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const ctype qx, const ctype qy, const ctype qz, const ctype qw,
                                                 const ctype qv, const T /*c*/, const T x0, const T x1, const T x2,
                                                 const T x3, const T y0, const T y1, const T y2, const T y3, const T z0,
                                                 const T z1, const T z2, const T z3, const T w0, const T w1, const T w2,
                                                 const T w3, const T v0, const T v1, const T v2, const T v3)
  {
    using DiFfRG::powr;
    return (x0 + x1 * powr<1>(qx) + x2 * powr<2>(qx) + x3 * powr<3>(qx)) *
           (y0 + y1 * powr<1>(qy) + y2 * powr<2>(qy) + y3 * powr<3>(qy)) *
           (z0 + z1 * powr<1>(qz) + z2 * powr<2>(qz) + z3 * powr<3>(qz)) *
           (w0 + w1 * powr<1>(qw) + w2 * powr<2>(qw) + w3 * powr<3>(qw)) *
           powr<inv>(v0 + v1 * powr<1>(qv) + v2 * powr<2>(qv) + v3 * powr<3>(qv));
  }
  static KOKKOS_FORCEINLINE_FUNCTION auto constant(const T c, const T /*x0*/, const T /*x1*/, const T /*x2*/,
                                                   const T /*x3*/, const T /*y0*/, const T /*y1*/, const T /*y2*/,
                                                   const T /*y3*/, const T /*z0*/, const T /*z1*/, const T /*z2*/,
                                                   const T /*z3*/, const T /*w0*/, const T /*w1*/, const T /*w2*/,
                                                   const T /*w3*/, const T /*v0*/, const T /*v1*/, const T /*v2*/,
                                                   const T /*v3*/)
  {
    return c;
  }
};

template <typename _Regulator> class quark_kernel
{
public:
  using Regulator = _Regulator;

  static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double &lf1, const double &p0, const double &k,
                                                   const double &T, const double &mq2)
  {
    using namespace DiFfRG;
    using namespace DiFfRG::compute;
    const auto _repl1 = RF(powr<2>(k), powr<2>(lf1));
    return (-8.) * (_repl1 + lf1) *
           ((powr<-1>(powr<2>(_repl1) + (2.) * ((_repl1) * (lf1)) + powr<2>(lf1) + mq2 +
                      powr<2>(p0 + (3.141592653589793) * (T)))) *
            (RFdot(powr<2>(k), powr<2>(lf1))));
  }

  static KOKKOS_FORCEINLINE_FUNCTION double constant(const double & /* k */, const double & /* T */,
                                                     const double & /* mq2 */)
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

template <typename _Regulator> class quarkIntegrated_kernel
{
public:
  using Regulator = _Regulator;

  static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double &lf1, const double &k, const double &T,
                                                   const double &mq2)
  {
    using namespace DiFfRG;
    using namespace DiFfRG::compute;
    const auto _repl1 = RF(powr<2>(k), powr<2>(lf1));
    const auto _repl2 = RFdot(powr<2>(k), powr<2>(lf1));
    // Analytic fermionic Matsubara sum: T Σ_n 1/(E² + ω_n²) = tanh(E/(2T)) / (2E)
    // with E² = (RF + q)² + mq2
    const auto E2 = powr<2>(_repl1) + (2.) * (_repl1 * lf1) + powr<2>(lf1) + mq2;
    const auto E = std::sqrt(E2);
    return (-4.) * (_repl1 + lf1) * _repl2 * std::tanh((0.5) * E / T) / E;
  }

  static KOKKOS_FORCEINLINE_FUNCTION double constant(const double & /* k */, const double & /* T */,
                                                     const double & /* mq2 */)
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
