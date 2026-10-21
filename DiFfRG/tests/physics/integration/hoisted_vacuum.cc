#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_1ang.hh>
#include <DiFfRG/physics/interpolation/spline_interpolator_1d.hh>
#include <DiFfRG/physics/regulators.hh>

#include <autodiff/forward/real.hpp>

#include <cmath>
#include <vector>

using namespace DiFfRG;

namespace
{
  /**
   * A kernel on two integration variables (x, c), an external point p and the arguments (a, b, Z). The part
   * Z(x) exp(-x^2) / (1 + a x^2) depends on x, a and Z only; the rest also on c, p and b.
   */
  struct PlainKernel {
    template <typename A, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto loop_part(const double x, const A &a, const Z &z)
    {
      return z(x) * Kokkos::exp(-x * x) / (1. + a * x * x);
    }
    template <typename P, typename B> static KOKKOS_FORCEINLINE_FUNCTION auto rest(const P &loop, const double c,
                                                                                    const double p, const B &b)
    {
      return loop * (1. + b * c * p) + b * b * c;
    }

    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double x, const double c, const double p, const A &a,
                                                   const B &b, const Z &z)
    {
      return rest(loop_part(x, a, z), c, p, b);
    }
    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const double p, const A &a, const B &b, const Z &)
    {
      return a * b * p;
    }
  };

  /// PlainKernel in hoisted form: pre() holds the loop part.
  struct HoistedKernel : PlainKernel {
    static constexpr bool has_pre = true;
    // x, c, p, a, b, Z
    static constexpr std::array<bool, 6> pre_reads{true, false, false, true, false, true};

    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto pre(const double x, const double, const double, const A &a, const B &,
                                                const Z &z)
    {
      return device::make_tuple(loop_part(x, a, z));
    }
    template <typename P, typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto post(const P &pre, const double, const double c, const double p,
                                                 const A &, const B &b, const Z &)
    {
      return rest(device::get<0>(pre), c, p, b);
    }
  };

  /// A hoisted kernel whose pre() reads the second integration variable: never hoisted.
  struct BadHoistedKernel : HoistedKernel {
    static constexpr std::array<bool, 6> pre_reads{true, true, false, true, false, true};
  };

  using Coordinates = LinearCoordinates1D<double>;
  using Interpolator = SplineInterpolator1D<double, Coordinates>;

  template <typename K, typename ExecutionSpace, typename NT = double>
  using Integrator = QuadratureIntegrator<2, NT, K, ExecutionSpace>;

  template <typename K, typename ExecutionSpace, typename NT = double>
  Integrator<K, ExecutionSpace, NT> make(QuadratureProvider &provider)
  {
    return Integrator<K, ExecutionSpace, NT>(provider, {{16, 8}}, {{0., -1.}}, {{2., 1.}},
                                             {{QuadratureType::legendre, QuadratureType::legendre}});
  }

  void fill(Interpolator &z, const double shift)
  {
    std::vector<double> v(z.get_coordinates().size());
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = 1. + 0.5 * std::sin(0.7 * double(i) + shift);
    z.update(v.data());
  }

  bool close(const double a, const double b) { return std::abs(a - b) <= 1e-13 * (1. + std::abs(b)); }

  template <typename ExecutionSpace> void check_all(QuadratureProvider &provider)
  {
    auto plain = make<PlainKernel, ExecutionSpace>(provider);
    auto hoisted = make<HoistedKernel, ExecutionSpace>(provider);
    STATIC_REQUIRE(Integrator<HoistedKernel, ExecutionSpace>::can_hoist);
    STATIC_REQUIRE(!Integrator<PlainKernel, ExecutionSpace>::can_hoist);
    STATIC_REQUIRE(!Integrator<BadHoistedKernel, ExecutionSpace>::can_hoist);

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

    SECTION("map_points with an AD argument per point")
    {
      // The AD type is what KT's evaluate_batch feeds in; b, the AD argument, is post()-only.
      auto plain_ad = make<PlainKernel, ExecutionSpace, autodiff::real>(provider);
      auto hoisted_ad = make<HoistedKernel, ExecutionSpace, autodiff::real>(provider);
      const size_t n = 23;
      std::vector<double> p(n);
      std::vector<autodiff::real> b(n), r_plain(n), r_hoisted(n);
      for (size_t i = 0; i < n; ++i) {
        p[i] = 0.1 + 0.05 * double(i);
        b[i] = -0.5 + 0.04 * double(i);
        b[i][1] = 1.;
      }
      plain_ad.map_points(PointSpan<autodiff::real>(r_plain), p, 0.55, b, z);
      hoisted_ad.map_points(PointSpan<autodiff::real>(r_hoisted), p, 0.55, b, z);
      for (size_t i = 0; i < n; ++i)
        for (size_t o = 0; o < 2; ++o) {
          CAPTURE(i, o, r_plain[i][o], r_hoisted[i][o]);
          REQUIRE(close(r_hoisted[i][o], r_plain[i][o]));
        }
    }

    SECTION("set_hoist(false) evaluates kernel() only")
    {
      double r_on{}, r_off{};
      hoisted.get(r_on, 0.8, 0.5, 0.3, z);
      hoisted.set_hoist(false);
      hoisted.get(r_off, 0.8, 0.5, 0.3, z);
      REQUIRE(close(r_off, r_on));
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
      REQUIRE(both(0.5) == r1);
      plain.set_grid_extents({{0., -1.}}, {{2.5, 1.}}); // the first axis moved
      hoisted.set_grid_extents({{0., -1.}}, {{2.5, 1.}});
      REQUIRE(both(0.5) != r1);
      hoisted.set_pre_table_reuse(false);
      both(0.5);
    }
  }

  /// HoistedKernel on (q, cos) of Integrator_p2_1ang, with the regulator it needs.
  template <typename K> struct WithRegulator : K {
    using Regulator = PolynomialExpRegulator<>;
  };
} // namespace

TEST_CASE("Hoisted vacuum kernels match their plain form", "[integration][quadrature][hoisting]")
{
  DiFfRG::Init();
  QuadratureProvider provider;

  SECTION("GPU") { check_all<GPU_exec>(provider); }
  SECTION("Kokkos host") { check_all<KokkosHost_exec>(provider); }
}

TEST_CASE("Integrator_p2_1ang forwards the hoisted form", "[integration][quadrature][hoisting]")
{
  DiFfRG::Init();
  QuadratureProvider provider;
  using Plain = Integrator_p2_1ang<4, double, WithRegulator<PlainKernel>, GPU_exec>;
  using Hoisted = Integrator_p2_1ang<4, double, WithRegulator<HoistedKernel>, GPU_exec>;
  STATIC_REQUIRE(Hoisted::can_hoist);
  Plain plain(provider, {{24, 8}}, 3.);
  Hoisted hoisted(provider, {{24, 8}}, 3.);
  plain.set_k(0.7);
  hoisted.set_k(0.7);

  const Coordinates zgrid(16, 0., 2.);
  Interpolator z(zgrid);
  fill(z, 0.3);

  const Coordinates pgrid(9, 0.1, 1.5);
  std::vector<double> m_plain(pgrid.size()), m_hoisted(pgrid.size());
  plain.map(m_plain.data(), pgrid, 0.6, 0.2, z);
  hoisted.map(m_hoisted.data(), pgrid, 0.6, 0.2, z);
  Kokkos::fence();
  for (size_t i = 0; i < pgrid.size(); ++i) {
    CAPTURE(i, m_plain[i], m_hoisted[i]);
    REQUIRE(close(m_hoisted[i], m_plain[i]));
  }
}
