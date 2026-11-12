#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_4d_2ang.hh>
#include <DiFfRG/physics/interpolation/spline_interpolator_1d.hh>
#include <DiFfRG/physics/regulators.hh>

#include <cmath>
#include <vector>

using namespace DiFfRG;

namespace
{
  /**
   * A kernel on two integration variables (x, c) and an external point (S, u), with arguments (a, b, Z), shaped like a
   * NumTracer trace: a numerator N = Σ_n a_n(c, u) x^n S^(2-n), homogeneous in the scales x and S, over a
   * denominator Q = x² + S² - x S c u, times a loop part Z(x) exp(-x²) and a rest reading the arguments.
   * The table holds a_0, a_1, a_2 at every (c, u) node: tab_box_mask {false, true}, tab_ext_mask {false, true}.
   */
  struct PlainKernel {
    static KOKKOS_FORCEINLINE_FUNCTION double coeff(const int n, const double c, const double u)
    {
      return n == 0 ? 1. + c * u : (n == 1 ? c - 2. * u : c * c * u + 0.5);
    }
    static KOKKOS_FORCEINLINE_FUNCTION double trace(const double a0, const double a1, const double a2, const double x,
                                                    const double c, const double S, const double u)
    {
      return (a0 * S * S + a1 * x * S + a2 * x * x) / (x * x + S * S - x * S * c * u);
    }
    template <typename Z> static KOKKOS_FORCEINLINE_FUNCTION auto loop_part(const double x, const Z &z)
    {
      return z(x) * Kokkos::exp(-x * x);
    }
    template <typename L, typename A, typename B>
    static KOKKOS_FORCEINLINE_FUNCTION auto rest(const L &loop, const double tr, const double c, const double u,
                                                 const A &a, const B &b)
    {
      return loop * (a + b * c) * tr + b * u;
    }

    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double x, const double c, const double S, const double u,
                                                   const A &a, const B &b, const Z &z)
    {
      return rest(loop_part(x, z), trace(coeff(0, c, u), coeff(1, c, u), coeff(2, c, u), x, c, S, u), c, u, a, b);
    }
    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const double, const double, const A &, const B &, const Z &)
    {
      return 0.;
    }
  };

  /// PlainKernel with its trace read from the table.
  struct TabKernel : PlainKernel {
    static constexpr bool has_tab = true;
    static constexpr std::size_t tab_size = 3;
    static constexpr std::array<bool, 2> tab_box_mask{false, true};
    static constexpr std::array<bool, 2> tab_ext_mask{false, true};

    template <typename... R>
    static KOKKOS_FORCEINLINE_FUNCTION void tab(double *out, const std::size_t stride, const double, const double c,
                                                const double, const double u, const R &...)
    {
      for (int n = 0; n < 3; ++n)
        out[n * stride] = coeff(n, c, u);
    }
    template <typename T> static KOKKOS_FORCEINLINE_FUNCTION double read(const T *row, const std::size_t stride, int n)
    {
      return double(row[n * stride]);
    }
    template <typename T, typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel_tab(const T *row, const std::size_t st, const double x,
                                                       const double c, const double S, const double u, const A &a,
                                                       const B &b, const Z &z)
    {
      return rest(loop_part(x, z), trace(read(row, st, 0), read(row, st, 1), read(row, st, 2), x, c, S, u), c, u, a,
                  b);
    }
  };

  /// TabKernel in hoisted form as well: pre() holds the loop part, post_tab() the rest.
  struct HoistedTabKernel : TabKernel {
    static constexpr bool has_pre = true;
    // x, c, S, u, a, b, Z
    static constexpr std::array<bool, 7> pre_reads{true, false, false, false, false, false, true};

    template <typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto pre(const double x, const double, const double, const double, const A &,
                                                const B &, const Z &z)
    {
      return device::make_tuple(loop_part(x, z));
    }
    template <typename P, typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto post(const P &pre, const double x, const double c, const double S,
                                                 const double u, const A &a, const B &b, const Z &)
    {
      return rest(device::get<0>(pre), trace(coeff(0, c, u), coeff(1, c, u), coeff(2, c, u), x, c, S, u), c, u, a, b);
    }
    template <typename P, typename T, typename A, typename B, typename Z>
    static KOKKOS_FORCEINLINE_FUNCTION auto post_tab(const P &pre, const T *row, const std::size_t st, const double x,
                                                     const double c, const double S, const double u, const A &a,
                                                     const B &b, const Z &)
    {
      return rest(device::get<0>(pre), trace(read(row, st, 0), read(row, st, 1), read(row, st, 2), x, c, S, u), c,
                  u, a, b);
    }
  };

  using Coordinates = LinearCoordinates1D<double>;
  using ExtCoordinates = CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>>;
  using Interpolator = SplineInterpolator1D<double, Coordinates>;

  template <typename K, typename ExecutionSpace> using Integrator = QuadratureIntegrator<2, double, K, ExecutionSpace>;

  template <typename K, typename ExecutionSpace> Integrator<K, ExecutionSpace> make(QuadratureProvider &provider)
  {
    return Integrator<K, ExecutionSpace>(provider, {{16, 8}}, {{0., -1.}}, {{2., 1.}},
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

  template <typename K, typename ExecutionSpace>
  void check_map(QuadratureProvider &provider, const ExtCoordinates &ext, const double a, const double b,
                 const Interpolator &z, Integrator<K, ExecutionSpace> &tabbed)
  {
    auto plain = make<PlainKernel, ExecutionSpace>(provider);
    std::vector<double> m_plain(ext.size()), m_tab(ext.size());
    plain.map(m_plain.data(), ext, a, b, z);
    tabbed.map(m_tab.data(), ext, a, b, z);
    Kokkos::fence();
    for (size_t i = 0; i < ext.size(); ++i) {
      CAPTURE(i, m_plain[i], m_tab[i]);
      REQUIRE(close(m_tab[i], m_plain[i]));
    }
  }

  template <typename ExecutionSpace> void check_all(QuadratureProvider &provider)
  {
    STATIC_REQUIRE(Integrator<TabKernel, ExecutionSpace>::can_tab);
    STATIC_REQUIRE(Integrator<HoistedTabKernel, ExecutionSpace>::can_tab);
    STATIC_REQUIRE(Integrator<HoistedTabKernel, ExecutionSpace>::can_hoist);
    STATIC_REQUIRE(!Integrator<PlainKernel, ExecutionSpace>::can_tab);

    const Coordinates zgrid(16, 0., 2.);
    Interpolator z(zgrid);
    fill(z, 0.);
    // S from 0.05 to 3 (below and above the loop range), u in [-0.9, 0.9]
    const ExtCoordinates ext(LinearCoordinates1D<double>(7, 0.05, 3.), LinearCoordinates1D<double>(5, -0.9, 0.9));

    SECTION("map, plain table form")
    {
      auto tabbed = make<TabKernel, ExecutionSpace>(provider);
      check_map(provider, ext, 0.6, 0.2, z, tabbed);
    }

    SECTION("map, hoisted table form")
    {
      auto tabbed = make<HoistedTabKernel, ExecutionSpace>(provider);
      check_map(provider, ext, 0.6, 0.2, z, tabbed);
    }

    SECTION("the table follows the grids, not the arguments")
    {
      auto tabbed = make<HoistedTabKernel, ExecutionSpace>(provider);
      check_map(provider, ext, 0.6, 0.2, z, tabbed);
      check_map(provider, ext, 0.9, -0.4, z, tabbed); // other arguments: same table
      fill(z, 1.);
      check_map(provider, ext, 0.9, -0.4, z, tabbed); // other interpolator data: same table
      const ExtCoordinates ext2(LinearCoordinates1D<double>(4, 0.1, 2.), LinearCoordinates1D<double>(6, -0.5, 0.8));
      check_map(provider, ext2, 0.6, 0.2, z, tabbed); // other shapes: rebuilt
      tabbed.set_grid_extents({{0., -0.5}}, {{2.5, 1.}}); // the masked axis moved: rebuilt
      auto plain = make<PlainKernel, ExecutionSpace>(provider);
      plain.set_grid_extents({{0., -0.5}}, {{2.5, 1.}});
      std::vector<double> m_plain(ext.size()), m_tab(ext.size());
      plain.map(m_plain.data(), ext, 0.6, 0.2, z);
      tabbed.map(m_tab.data(), ext, 0.6, 0.2, z);
      Kokkos::fence();
      for (size_t i = 0; i < ext.size(); ++i) {
        CAPTURE(i, m_plain[i], m_tab[i]);
        REQUIRE(close(m_tab[i], m_plain[i]));
      }
    }
  }

  /// A kernel on (q, cos1, cos2) of Integrator_p2_4D_2ang: the trace kernel with cos2 unused.
  template <typename K> struct On4D : K {
    using Regulator = PolynomialExpRegulator<>;
    template <typename... R>
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double q, const double c, const double, const R &...r)
    {
      return K::kernel(q, c, r...);
    }
    template <typename... R>
    static KOKKOS_FORCEINLINE_FUNCTION void tab(double *out, const std::size_t st, const double q, const double c,
                                                const double, const R &...r)
      requires K::has_tab
    {
      K::tab(out, st, q, c, r...);
    }
    template <typename T, typename... R>
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel_tab(const T *row, const std::size_t st, const double q,
                                                       const double c, const double, const R &...r)
      requires K::has_tab
    {
      return K::kernel_tab(row, st, q, c, r...);
    }
    static constexpr std::array<bool, 3> tab_box_mask{false, true, false};
  };
} // namespace

TEST_CASE("Tabulated vacuum kernels match their plain form", "[integration][quadrature][tables]")
{
  DiFfRG::Init();
  QuadratureProvider provider;

  SECTION("GPU") { check_all<GPU_exec>(provider); }
  SECTION("Kokkos host") { check_all<KokkosHost_exec>(provider); }
}

TEST_CASE("Integrator_p2_4D_2ang forwards the table form", "[integration][quadrature][tables]")
{
  DiFfRG::Init();
  QuadratureProvider provider;
  using Plain = Integrator_p2_4D_2ang<4, double, On4D<PlainKernel>, GPU_exec>;
  using Tabbed = Integrator_p2_4D_2ang<4, double, On4D<TabKernel>, GPU_exec>;
  STATIC_REQUIRE(Tabbed::can_tab);
  STATIC_REQUIRE(!Plain::can_tab);
  Plain plain(provider, {{24, 6, 6}}, 3.);
  Tabbed tabbed(provider, {{24, 6, 6}}, 3.);
  for (const double k : {0.7, 2.}) { // the table survives a change of k (the radial axis is not masked)
    plain.set_k(k);
    tabbed.set_k(k);

    const Coordinates zgrid(16, 0., 2.);
    Interpolator z(zgrid);
    fill(z, 0.3);
    const ExtCoordinates ext(LinearCoordinates1D<double>(5, 0.1, 2.), LinearCoordinates1D<double>(3, -0.8, 0.8));
    std::vector<double> m_plain(ext.size()), m_tab(ext.size());
    plain.map(m_plain.data(), ext, 0.6, 0.2, z);
    tabbed.map(m_tab.data(), ext, 0.6, 0.2, z);
    Kokkos::fence();
    for (size_t i = 0; i < ext.size(); ++i) {
      CAPTURE(k, i, m_plain[i], m_tab[i]);
      REQUIRE(close(m_tab[i], m_plain[i]));
    }
  }
}
