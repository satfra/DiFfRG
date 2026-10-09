#pragma once

#include <DiFfRG/common/kokkos.hh>

#include <cmath>
#include <cstddef>

/**
 * @brief A separable lattice kernel whose full-zone sum is cheap to compute by brute force.
 *
 *     f(q) = g_0(q_0) * prod_{i>0} g_s(q_i),   g_0(q) = 1 + q^2 + c q^3,   g_s(q) = 2 - cos(a_s q) + q^2
 *
 * g_s is even but not lattice-periodic, so a halved spatial sum is only right with the endpoint
 * weights 1, 2, ..., 2, 1. The odd part c q^3 of g_0 only survives a q0 sum over the whole zone
 * n = -N/2 .. N/2 - 1, so c must be 0 when q0 is halved.
 *
 * get() form: kernel(q_0, ..., q_{d-1}, c, a_s), constant(c, a_s) = 0.
 * map() form: kernel(q_0, ..., q_{d-1}, p_0, p_1, c, a_s) = f(q) (1 + p_0) + p_1, constant(p_0, p_1, c, a_s) = p_0 p_1.
 */
template <int d, typename NT> struct LatticeTestKernel {
  template <typename... A> static KOKKOS_INLINE_FUNCTION NT kernel(const A &...a)
  {
    static_assert(sizeof...(A) == d + 2 || sizeof...(A) == d + 4);
    const double v[] = {double(a)...};
    constexpr size_t n = sizeof...(A);
    const double c = v[n - 2], a_s = v[n - 1];
    double f = 1. + v[0] * v[0] + c * v[0] * v[0] * v[0];
    for (int i = 1; i < d; ++i)
      f *= 2. - Kokkos::cos(a_s * v[i]) + v[i] * v[i];
    if constexpr (n == d + 4) f = f * (1. + v[d]) + v[d + 1];
    return NT(f);
  }

  template <typename... A> static KOKKOS_INLINE_FUNCTION NT constant(const A &...a)
  {
    static_assert(sizeof...(A) == 2 || sizeof...(A) == 4);
    const double v[] = {double(a)...};
    if constexpr (sizeof...(A) == 4)
      return NT(v[0] * v[1]);
    else
      return NT(0);
  }
};

/// The kernel summed over the whole Brillouin zone, n_i = -N_i/2 .. N_i/2 - 1, with unit weights.
inline double lattice_brute_force_sum(const int d, const unsigned N_t, const unsigned N_s, const double a_t,
                                      const double a_s, const double c)
{
  const double pi2 = 2. * M_PI;
  double s0 = 0.;
  for (int n = -int(N_t) / 2; n < int(N_t) / 2; ++n) {
    const double q = pi2 * n / (N_t * a_t);
    s0 += 1. + q * q + c * q * q * q;
  }
  double ss = 0.;
  for (int n = -int(N_s) / 2; n < int(N_s) / 2; ++n) {
    const double q = pi2 * n / (N_s * a_s);
    ss += 2. - std::cos(a_s * q) + q * q;
  }
  double result = s0 / (N_t * a_t);
  for (int i = 1; i < d; ++i)
    result *= ss / (N_s * a_s);
  return result;
}
