#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/multi_value.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_4d_3ang.hh>

#include <vector>

using namespace DiFfRG;

namespace
{
  // Three projections of one integrand, as for one loop projected onto several vertex structures: a shared factor
  // times a projection-specific one. `p` is the external point in map(), `a` an argument.
  template <int j> KOKKOS_FORCEINLINE_FUNCTION auto projector(const auto u, const auto v, const auto p)
  {
    if constexpr (j == 0)
      return decltype(u)(1);
    else if constexpr (j == 1)
      return u * v;
    else
      return p - v;
  }
  template <typename NT> KOKKOS_FORCEINLINE_FUNCTION NT shared(const auto u, const auto v, const auto p, const NT a)
  {
    return a * Kokkos::exp(-u * u - v) / (decltype(u)(1) + p * p + u * v);
  }

  template <int j, typename NT> struct Projection {
    static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const auto x, const auto y, const auto p, const NT a)
    {
      return shared(x, y, p, a) * projector<j>(x, y, p);
    }
    static KOKKOS_FORCEINLINE_FUNCTION NT constant(const auto p, const NT a) { return a * p * double(j + 1); }
  };

  template <typename NT> struct AllProjections {
    using MV = MultiValue<NT, 3>;
    static KOKKOS_FORCEINLINE_FUNCTION MV kernel(const auto x, const auto y, const auto p, const NT a)
    {
      const NT s = shared(x, y, p, a);
      return MV(s * projector<0>(x, y, p), s * projector<1>(x, y, p), s * projector<2>(x, y, p));
    }
    static KOKKOS_FORCEINLINE_FUNCTION MV constant(const auto p, const NT a)
    {
      return MV(a * p * 1., a * p * 2., a * p * 3.);
    }
  };

  // The same for the 4D momentum integral with three angles: kernel(q, cos1, cos2, phi, p, a).
  template <int j, typename NT> struct Projection4D {
    static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const auto q, const auto c1, const auto c2, const auto phi,
                                                 const auto p, const NT a)
    {
      return shared(q, c1 * c2, p, a) * projector<j>(q, Kokkos::cos(phi), p);
    }
    static KOKKOS_FORCEINLINE_FUNCTION NT constant(const auto, const NT) { return NT(0); }
  };
  template <typename NT> struct AllProjections4D {
    using MV = MultiValue<NT, 3>;
    static KOKKOS_FORCEINLINE_FUNCTION MV kernel(const auto q, const auto c1, const auto c2, const auto phi,
                                                 const auto p, const NT a)
    {
      const NT s = shared(q, c1 * c2, p, a);
      const auto cp = Kokkos::cos(phi);
      return MV(s * projector<0>(q, cp, p), s * projector<1>(q, cp, p), s * projector<2>(q, cp, p));
    }
    static KOKKOS_FORCEINLINE_FUNCTION MV constant(const auto, const NT) { return MV(NT(0)); }
  };

  template <typename NT> NT argument()
  {
    if constexpr (is_complex<NT>::value)
      return NT(0.7, -0.3);
    else
      return NT(0.7);
  }

  template <typename OT> void check_close(const std::array<std::vector<OT>, 3> &ref, const std::array<OT *, 3> &got,
                                          const size_t n, const double tolerance)
  {
    for (size_t j = 0; j < 3; ++j)
      for (size_t i = 0; i < n; ++i) {
        CAPTURE(j, i);
        CHECK(is_close(got[j][i], ref[j][i], tolerance));
      }
  }

  /// The MultiValue integrator against three scalar integrators, through every entry point.
  template <typename NT, typename ExecutionSpace> void check(const double tolerance)
  {
    using OT = get_type::double_precision<NT>;
    using MultiIntegrator = QuadratureIntegrator<2, MultiValue<NT, 3>, AllProjections<NT>, ExecutionSpace>;
    QuadratureProvider provider;
    const std::array<size_t, 2> grid{{24, 16}};
    const std::array<get_type::ctype<NT>, 2> lo{{0., 0.}}, hi{{1.5, 2.}};
    const std::array<QuadratureType, 2> rule{{QuadratureType::legendre, QuadratureType::legendre}};

    MultiIntegrator multi(provider, grid, lo, hi, rule);
    QuadratureIntegrator<2, NT, Projection<0, NT>, ExecutionSpace> s0(provider, grid, lo, hi, rule);
    QuadratureIntegrator<2, NT, Projection<1, NT>, ExecutionSpace> s1(provider, grid, lo, hi, rule);
    QuadratureIntegrator<2, NT, Projection<2, NT>, ExecutionSpace> s2(provider, grid, lo, hi, rule);

    const NT a = argument<NT>();
    const size_t n = 37;
    const LinearCoordinates1D<double> coordinates(n, 0., 1.);

    std::array<std::vector<OT>, 3> ref;
    for (auto &r : ref)
      r.resize(n);
    s0.map(ref[0].data(), coordinates, a);
    s1.map(ref[1].data(), coordinates, a);
    s2.map(ref[2].data(), coordinates, a);

    {
      INFO("map into non-contiguous parts of one array");
      std::vector<OT> buffer(3 * n + 5);
      const std::array<OT *, 3> dest{{buffer.data(), buffer.data() + n + 2, buffer.data() + 2 * n + 5}};
      multi.map(dest, coordinates, a);
      check_close(ref, dest, n, tolerance);
    }
    {
      INFO("map with a braced list of destinations");
      std::vector<OT> d0(n), d1(n), d2(n);
      multi.map({d0.data(), d1.data(), d2.data()}, coordinates, a);
      check_close(ref, {{d0.data(), d1.data(), d2.data()}}, n, tolerance);
    }
    {
      INFO("map inside DeferredMaps lands when the scope ends");
      const OT sentinel = OT(42.);
      std::vector<OT> d0(n, sentinel), d1(n, sentinel), d2(n, sentinel);
      {
        DeferredMaps defer;
        multi.map({d0.data(), d1.data(), d2.data()}, coordinates, a);
        // Without a device backend every map runs immediately, so there is nothing to defer.
        if constexpr (DiFfRG::internal::has_device_backend) {
          CHECK(d0[0] == sentinel);
          CHECK(d2[n - 1] == sentinel);
        }
      }
      check_close(ref, {{d0.data(), d1.data(), d2.data()}}, n, tolerance);
    }
    {
      INFO("get");
      MultiValue<OT, 3> v;
      multi.get(v, 0.3, a);
      std::array<OT, 3> r;
      s0.get(r[0], 0.3, a);
      s1.get(r[1], 0.3, a);
      s2.get(r[2], 0.3, a);
      for (size_t j = 0; j < 3; ++j)
        CHECK(is_close(v[j], r[j], tolerance));
    }
    {
      INFO("map_points with a per-point argument");
      std::vector<double> p(n);
      for (size_t i = 0; i < n; ++i)
        p[i] = 0.05 * double(i);
      std::array<std::vector<OT>, 3> r;
      std::vector<OT> d0(n), d1(n), d2(n);
      for (auto &x : r)
        x.resize(n);
      s0.map_points(PointSpan<OT>(r[0]), p, a);
      s1.map_points(PointSpan<OT>(r[1]), p, a);
      s2.map_points(PointSpan<OT>(r[2]), p, a);
      multi.map_points(std::array<PointSpan<OT>, 3>{{PointSpan<OT>(d0), PointSpan<OT>(d1), PointSpan<OT>(d2)}}, p, a);
      check_close(r, {{d0.data(), d1.data(), d2.data()}}, n, tolerance);
    }
  }
} // namespace

TEST_CASE("MultiValue arithmetic", "[integration][multi_output]")
{
  using MV = MultiValue<double, 3>;
  const MV a(1., 2., 3.), b(0.5);
  const MV c = 2. * a + b - a * 0.5 - MV(a / 2.);
  CHECK(c[0] == 2. + 0.5 - 0.5 - 0.5);
  CHECK(c[2] == 6. + 0.5 - 1.5 - 1.5);
  CHECK((-a)[1] == -2.);
  CHECK(MV()[1] == 0.);
  const MultiValue<float, 3> f(1.f, 2.f, 3.f);
  const MV widened = f;
  CHECK(widened[2] == 3.);
  STATIC_REQUIRE(std::is_same_v<get_type::ctype<MultiValue<complex<float>, 2>>, float>);
  STATIC_REQUIRE(std::is_same_v<get_type::double_precision<MultiValue<float, 2>>, MultiValue<double, 2>>);
  STATIC_REQUIRE(!std::is_convertible_v<double, MV>);
}

TEST_CASE("MultiValue kernels equal separate kernels", "[integration][multi_output]")
{
  DiFfRG::Init();

  SECTION("GPU")
  {
    check<double, GPU_exec>(1e-14);
    check<complex<double>, GPU_exec>(1e-14);
    check<float, GPU_exec>(1e-6);
  }
  SECTION("Kokkos host")
  {
    check<double, KokkosHost_exec>(1e-14);
    check<float, KokkosHost_exec>(1e-6);
  }
  SECTION("TBB")
  {
    check<double, TBB_exec>(1e-14);
    check<complex<double>, TBB_exec>(1e-14);
    check<float, TBB_exec>(1e-6);
  }
}

TEST_CASE("MultiValue kernel through Integrator_p2_4D_3ang", "[integration][multi_output]")
{
  DiFfRG::Init();

  auto run = [](auto execution_space) {
    using ExecutionSpace = decltype(execution_space);
    QuadratureProvider provider;
    const std::array<size_t, 4> grid{{24, 6, 6, 6}};
    Integrator_p2_4D_3ang<4, MultiValue<double, 3>, AllProjections4D<double>, ExecutionSpace> multi(provider, grid,
                                                                                                    2.);
    Integrator_p2_4D_3ang<4, double, Projection4D<0, double>, ExecutionSpace> s0(provider, grid, 2.);
    Integrator_p2_4D_3ang<4, double, Projection4D<1, double>, ExecutionSpace> s1(provider, grid, 2.);
    Integrator_p2_4D_3ang<4, double, Projection4D<2, double>, ExecutionSpace> s2(provider, grid, 2.);
    s0.set_k(0.8);
    s1.set_k(0.8);
    s2.set_k(0.8);
    multi.set_k(0.8);

    const size_t n = 20;
    const LinearCoordinates1D<double> coordinates(n, 0., 1.);
    std::array<std::vector<double>, 3> ref;
    for (auto &r : ref)
      r.resize(n);
    s0.map(ref[0].data(), coordinates, 0.7);
    s1.map(ref[1].data(), coordinates, 0.7);
    s2.map(ref[2].data(), coordinates, 0.7);

    std::vector<double> d0(n), d1(n), d2(n);
    multi.map({d0.data(), d1.data(), d2.data()}, coordinates, 0.7);
    check_close(ref, {{d0.data(), d1.data(), d2.data()}}, n, 1e-14);
  };
  SECTION("GPU") { run(GPU_exec()); }
  SECTION("TBB") { run(TBB_exec()); }
}
