#ifndef HOISTED_FT_NO_MAIN
#define CATCH_CONFIG_MAIN
#endif
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>
#include <DiFfRG/physics/interpolation/spline_interpolator_1d.hh>

#include <cmath>
#include <vector>

using namespace DiFfRG;

namespace
{
  /**
   * A kernel on two box variables (x, c), the frequency q0, an external point p and the arguments (a, b, Z). The
   * part Z(x) exp(-x^2) / (1 + q0^2 + a x^2) depends on x, q0, a and Z only; the rest also on c, p and b. `odd` adds
   * a part odd in q0, `even` declares the kernel even instead.
   */
  template <bool even> struct PlainKernel {
    static constexpr bool matsubara_even = even;

    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto loop_part(const double x, const double q0, const A &a, const Z &z)
    {
      return z(x) * Kokkos::exp(-x * x) / (1. + q0 * q0 + a * x * x);
    }
    template <typename P, typename A, typename B>
    static KOKKOS_FORCEINLINE_FUNCTION auto rest(const P &loop, const double c, const double q0, const double p,
                                                 const A &a, const B &b)
    {
      if constexpr (even)
        return loop * (1. + b * c * p);
      else
        return loop * (1. + b * c * p) + 0.3 * c * q0 * a;
    }

    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double x, const double c, const double q0, const double p,
                                                   const A &a, const B &b, const Z &z)
    {
      return rest(loop_part<A, B>(x, q0, a, z), c, q0, p, a, b);
    }
    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const double p, const A &a, const B &b, const Z &)
    {
      return a * b * p;
    }
  };

  /// PlainKernel in hoisted form: pre() holds the loop part.
  template <bool even> struct HoistedKernel : PlainKernel<even> {
    using Base = PlainKernel<even>;
    static constexpr bool has_pre = true;
    // x, c, q0, p, a, b, Z
    static constexpr std::array<bool, 7> pre_reads{true, false, true, false, true, false, true};

    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto pre(const double x, const double, const double q0, const double, const A &a,
                                                const B &, const Z &z)
    {
      return device::make_tuple(Base::template loop_part<A, B>(x, q0, a, z));
    }
    template <typename P, typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto post(const P &pre, const double, const double c, const double q0,
                                                 const double p, const A &a, const B &b, const Z &)
    {
      return Base::rest(device::get<0>(pre), c, q0, p, a, b);
    }
  };

  using Coordinates = LinearCoordinates1D<double>;
  using Interpolator = SplineInterpolator1D<double, Coordinates>;

  template <typename K, typename ExecutionSpace>
  using Integrator = QuadratureIntegrator_fT<3, double, K, ExecutionSpace>;

  template <typename K, typename ExecutionSpace>
  Integrator<K, ExecutionSpace> make(QuadratureProvider &provider, const double T)
  {
    Integrator<K, ExecutionSpace> integrator(provider, {{16, 8}}, {{0., -1.}}, {{2., 1.}},
                                             {{QuadratureType::legendre, QuadratureType::legendre}}, T);
    integrator.set_k(1.);
    return integrator;
  }

  void fill(Interpolator &z, const double shift)
  {
    std::vector<double> v(z.get_coordinates().size());
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = 1. + 0.5 * std::sin(0.7 * double(i) + shift);
    z.update(v.data());
  }

  bool close(const double a, const double b) { return std::abs(a - b) <= 1e-13 * (1. + std::abs(b)); }

  template <bool even, typename ExecutionSpace> void check_all(QuadratureProvider &provider, const double T)
  {
    auto plain = make<PlainKernel<even>, ExecutionSpace>(provider, T);
    auto hoisted = make<HoistedKernel<even>, ExecutionSpace>(provider, T);
    STATIC_REQUIRE(Integrator<HoistedKernel<even>, ExecutionSpace>::can_hoist);
    STATIC_REQUIRE(!Integrator<PlainKernel<even>, ExecutionSpace>::can_hoist);

    const Coordinates zgrid(16, 0., 2.);
    Interpolator z(zgrid);
    fill(z, 0.);

    SECTION("get")
    {
      for (const double a : {0.5, 0.7})
        for (const double b : {-0.3, 0.4}) {
          double r_plain{}, r_hoisted{};
          plain.get(r_plain, 0.8, a, b, z);
          hoisted.get(r_hoisted, 0.8, a, b, z);
          CAPTURE(a, b, r_plain, r_hoisted);
          REQUIRE(close(r_hoisted, r_plain));
        }
    }

    SECTION("map")
    {
      const Coordinates pgrid(12, 0.1, 1.5);
      std::vector<double> m_plain(pgrid.size()), m_hoisted(pgrid.size());
      plain.map(m_plain.data(), pgrid, 0.6, 0.2, z);
      hoisted.map(m_hoisted.data(), pgrid, 0.6, 0.2, z);
      Kokkos::fence();
      for (size_t i = 0; i < pgrid.size(); ++i) {
        CAPTURE(i, m_plain[i], m_hoisted[i]);
        REQUIRE(close(m_hoisted[i], m_plain[i]));
      }
    }

    SECTION("map_points")
    {
      const size_t n = 37;
      std::vector<double> p(n), a(n), b(n), r_plain(n), r_hoisted(n);
      for (size_t i = 0; i < n; ++i) {
        p[i] = 0.1 + 0.03 * double(i);
        a[i] = 0.4 + 0.01 * double(i);
        b[i] = -0.5 + 0.02 * double(i);
      }
      // b per point is not read by pre(): hoisted. a per point is: the plain path, still correct.
      for (const bool a_per_point : {false, true}) {
        const PointArg<double> a_arg = a_per_point ? PointArg<double>(a) : PointArg<double>(0.55);
        plain.map_points(PointSpan<double>(r_plain), p, a_arg, b, z);
        hoisted.map_points(PointSpan<double>(r_hoisted), p, a_arg, b, z);
        for (size_t i = 0; i < n; ++i) {
          CAPTURE(a_per_point, i, r_plain[i], r_hoisted[i]);
          REQUIRE(close(r_hoisted[i], r_plain[i]));
        }
      }
    }

    SECTION("reused table follows its inputs")
    {
      auto both = [&](const double a) {
        double r_plain{}, r_hoisted{};
        plain.get(r_plain, 0.8, a, 0.3, z);
        hoisted.get(r_hoisted, 0.8, a, 0.3, z);
        CAPTURE(a, r_plain, r_hoisted);
        REQUIRE(close(r_hoisted, r_plain));
        return r_hoisted;
      };
      const double r0 = both(0.5);
      REQUIRE(both(0.5) == r0); // reused
      REQUIRE(both(0.9) != r0); // an argument pre() reads changed
      fill(z, 1.);              // the interpolator data changed
      const double r1 = both(0.5);
      REQUIRE(r1 != r0);
      const auto epoch = internal::current_interpolator_data_epoch();
      fill(z, 1.); // identical data: no change recorded
      REQUIRE(internal::current_interpolator_data_epoch() == epoch);
      REQUIRE(both(0.5) == r1);
      plain.set_k(1.3); // a different frequency rule
      hoisted.set_k(1.3);
      both(0.5);
      hoisted.set_pre_table_reuse(false);
      both(0.5);
    }
  }
} // namespace

TEST_CASE("Hoisted finite-T kernels match their plain form", "[integration][quadrature][finiteT][hoisting]")
{
  DiFfRG::Init();
  QuadratureProvider provider;
  // Monien rule, and the T = 0 integral.
  const double T = GENERATE(0.15, 0.);

  SECTION("GPU") { check_all<false, GPU_exec>(provider, T); }
  SECTION("GPU, even kernel") { check_all<true, GPU_exec>(provider, T); }
  SECTION("Kokkos host") { check_all<false, KokkosHost_exec>(provider, T); }
}
