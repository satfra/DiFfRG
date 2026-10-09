#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <vector>

using namespace DiFfRG;

namespace
{
  using C1 = FocusedLogCoordinates1D<double>;
  using C3 = CoordinatePackND<FocusedLogCoordinates1D<double>, LogarithmicCoordinates1D<double>,
                              LinearPeriodicCoordinates1D<double>>;

  const C1 c1(64, 0.005, 250., 1., 2.);
  const C3 c3(C1(24, 0.005, 250., 1., 2.), LogarithmicCoordinates1D<double>(5, 0., 1. - 1e-4, 2.),
              LinearPeriodicCoordinates1D<double>(12, -5. * M_PI / 6., 7. * M_PI / 6.));

  std::vector<double> data1D(const double shift)
  {
    std::vector<double> d(c1.size());
    for (size_t i = 0; i < d.size(); ++i)
      d[i] = shift + std::log1p(c1.forward(i));
    return d;
  }

  std::vector<double> data3D(const double shift)
  {
    std::vector<double> d(c3.size());
    for (size_t i = 0; i < d.size(); ++i)
      d[i] = shift + 0.1 * std::sin(0.37 * i);
    return d;
  }

  // must be a handle, not the interpolator: catches an integrator that silently falls back
  template <typename... T> constexpr bool all_handles = (!has_kernel_handle<std::remove_cvref_t<T>> && ...);

  // a kernel as MakeKernel/NumTracer emit it: interpolators are `auto`, so it takes their handles
  template <typename RT> struct HandleKernel {
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const RT &q, const RT &p, const auto &Z, const auto &V)
    {
      static_assert(all_handles<decltype(Z), decltype(V)>);
      return Z(q) * V(p, RT(0.3), RT(0.5)) * Kokkos::exp(-q * q);
    }
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const RT &p, const auto &Z, const auto &V)
    {
      static_assert(all_handles<decltype(Z), decltype(V)>);
      return Z(p) * V(p, RT(0.3), RT(0.5)) * RT(1e-3);
    }
  };

  // a kernel naming the interpolator types, as emitted before handles: it is passed the interpolators
  template <typename RT> struct NamedKernel {
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const RT &q, const RT &p, const SplineInterpolator1D<double, C1> &Z,
                                                   const LinearInterpolatorND<double, C3> &V)
    {
      return Z(q) * V(p, RT(0.3), RT(0.5)) * Kokkos::exp(-q * q);
    }
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const RT &p, const SplineInterpolator1D<double, C1> &Z,
                                                     const LinearInterpolatorND<double, C3> &V)
    {
      return Z(p) * V(p, RT(0.3), RT(0.5)) * RT(1e-3);
    }
  };

  // whether an Integrator_p2 computing in RT hands KERNEL the interpolators' handles
  template <typename RT, typename KERNEL>
  constexpr bool takes_handles =
      DiFfRG::internal::takes_handles<RT, DiFfRG::internal::Transform_p2<4, RT, KERNEL>, RT, 2, double,
                                      SplineInterpolator1D<double, C1>, LinearInterpolatorND<double, C3>>;

  // finite T: a stack over Matsubara frequencies, plus the 1D and 2D linear interpolators
  using CB = BosonicCoordinates1DFiniteT<int, double, FocusedLogCoordinates1D<double>>;
  using CL = LogarithmicCoordinates1D<double>;
  using C2 = CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>;
  const CB cb(FocusedLogCoordinates1D<double>(32, 0.01, 10., 1., 1.), -6, 7, 0.15);
  const CL cl(32, 0.01, 10., 2.);
  const C2 c2(LogarithmicCoordinates1D<double>(16, 0.01, 10., 2.), LinearCoordinates1D<double>(8, -1., 1.));

  template <typename RT> struct FTKernel {
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const RT &q, const RT &q0, const RT &p, const auto &B, const auto &L1,
                                                   const auto &L2)
    {
      static_assert(all_handles<decltype(B), decltype(L1), decltype(L2)>);
      return B(q0, q) * L1(q) * L2(p, RT(0.2)) * Kokkos::exp(-q * q - q0 * q0);
    }
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const RT &p, const auto &B, const auto &L1, const auto &L2)
    {
      static_assert(all_handles<decltype(B), decltype(L1), decltype(L2)>);
      return L1(p) * RT(1e-3);
    }
  };

  template <typename Integrator, typename... Args> std::vector<double> integrate_fT(const Args &...args)
  {
    QuadratureProvider qp;
    Integrator integrator(qp, {48}, 4.);
    integrator.set_T(0.15);
    integrator.set_k(1.);
    integrator.set_typical_E(1.);
    std::vector<double> out(cl.size());
    integrator.map(out.data(), cl, args...);
    double g = 0;
    integrator.get(g, 0.7, args...);
    out.push_back(g);
    return out;
  }

  template <typename Integrator, typename... Args> std::vector<double> integrate(const Args &...args)
  {
    QuadratureProvider qp;
    Integrator integrator(qp, {64}, 4.);
    std::vector<double> out(c1.size());
    integrator.map(out.data(), c1, args...);
    double g = 0;
    integrator.get(g, 0.7, args...);
    out.push_back(g);
    return out;
  }
} // namespace

TEST_CASE("Interpolator handles", "[interpolation][handle]")
{
  DiFfRG::Init();

  SplineInterpolator1D<double, C1> Z(c1);
  LinearInterpolatorND<double, C3> V(c3);
  STATIC_REQUIRE(has_kernel_handle<decltype(Z)>);
  STATIC_REQUIRE(has_kernel_handle<decltype(V)>);
  STATIC_REQUIRE(!has_kernel_handle<double>);
  STATIC_REQUIRE(sizeof(Z.handle<float>()) < sizeof(Z));
  STATIC_REQUIRE(sizeof(V.handle<float>()) < sizeof(V));

  // the single-precision copy follows every update
  for (const double shift : {1., 2.5}) {
    const auto d1 = data1D(shift);
    const auto d3 = data3D(shift);
    Z.update(d1.data());
    V.update(d3.data());

    for (const double p : {0.004, 0.01, 0.3, 1.0, 1.7, 40., 300.}) {
      // A handle evaluates the same formula as the interpolator, but in a different inlining context: under
      // -ffast-math the two can round differently in the last bit.
      CHECK(Z.handle<double>()(p) == Catch::Approx(Z(p)).epsilon(1e-14));
      CHECK(V.handle<double>()(p, 0.3, 0.5) == Catch::Approx(V(p, 0.3, 0.5)).epsilon(1e-14));
      CHECK(Z.handle<float>()(float(p)) == Catch::Approx(Z(p)).epsilon(2e-6));
      CHECK(V.handle<float>()(float(p), 0.3f, 0.5f) == Catch::Approx(V(p, 0.3, 0.5)).epsilon(2e-6));
    }
  }

  const auto reference = integrate<Integrator_p2<4, double, HandleKernel<double>, GPU_exec>>(Z, V);

  SECTION("A kernel naming the interpolator types gets the interpolators")
  {
    STATIC_REQUIRE(!takes_handles<double, NamedKernel<double>>);
    const auto named = integrate<Integrator_p2<4, double, NamedKernel<double>, GPU_exec>>(Z, V);
    for (size_t i = 0; i < named.size(); ++i)
      CHECK(named[i] == Catch::Approx(reference[i]).epsilon(1e-13)); // same integral, other inlining: rounding
  }

  SECTION("A float kernel reads the single-precision copy")
  {
    STATIC_REQUIRE(takes_handles<float, HandleKernel<float>>);
    const auto gpu = integrate<Integrator_p2<4, float, HandleKernel<float>, GPU_exec>>(Z, V);
    const auto tbb = integrate<Integrator_p2<4, float, HandleKernel<float>, TBB_exec>>(Z, V);
    for (size_t i = 0; i < gpu.size(); ++i) {
      CHECK(gpu[i] == Catch::Approx(reference[i]).epsilon(1e-5));
      CHECK(tbb[i] == Catch::Approx(reference[i]).epsilon(1e-5));
    }
  }
}

TEST_CASE("Interpolator handles at finite temperature", "[interpolation][handle][finiteT]")
{
  DiFfRG::Init();

  SplineInterpolator1DStack<double, CB> B(cb);
  LinearInterpolator1D<double, CL> L1(cl);
  LinearInterpolator2D<double, C2> L2(c2);
  STATIC_REQUIRE(has_kernel_handle<decltype(B)> && has_kernel_handle<decltype(L1)> && has_kernel_handle<decltype(L2)>);

  std::vector<double> dB(cb.size()), dL1(cl.size()), dL2(c2.size());
  for (size_t i = 0; i < dB.size(); ++i)
    dB[i] = 1. + 0.05 * std::cos(0.3 * i);
  for (size_t i = 0; i < dL1.size(); ++i)
    dL1[i] = 1. + std::log1p(cl.forward(i));
  for (size_t i = 0; i < dL2.size(); ++i)
    dL2[i] = 1. + 0.1 * std::sin(0.37 * i);
  B.update(dB.data());
  L1.update(dL1.data());
  L2.update(dL2.data());

  for (const double p : {0.004, 0.3, 1.7, 40.}) {
    const double q0 = 2. * M_PI * 0.15;
    CHECK(B.handle<double>()(q0, p) == Catch::Approx(B(q0, p)).epsilon(1e-14));
    CHECK(L1.handle<double>()(p) == Catch::Approx(L1(p)).epsilon(1e-14));
    CHECK(L2.handle<double>()(p, 0.2) == Catch::Approx(L2(p, 0.2)).epsilon(1e-14));
    CHECK(B.handle<float>()(float(q0), float(p)) == Catch::Approx(B(q0, p)).epsilon(2e-6));
    CHECK(L1.handle<float>()(float(p)) == Catch::Approx(L1(p)).epsilon(2e-6));
    CHECK(L2.handle<float>()(float(p), 0.2f) == Catch::Approx(L2(p, 0.2)).epsilon(2e-6));
  }

  const auto reference = integrate_fT<Integrator_fT_p2<4, double, FTKernel<double>, GPU_exec>>(B, L1, L2);
  const auto gpu = integrate_fT<Integrator_fT_p2<4, float, FTKernel<float>, GPU_exec>>(B, L1, L2);
  const auto tbb = integrate_fT<Integrator_fT_p2<4, float, FTKernel<float>, TBB_exec>>(B, L1, L2);
  const auto tbb64 = integrate_fT<Integrator_fT_p2<4, double, FTKernel<double>, TBB_exec>>(B, L1, L2);
  for (size_t i = 0; i < reference.size(); ++i) {
    CHECK(gpu[i] == Catch::Approx(reference[i]).epsilon(1e-5));
    CHECK(tbb[i] == Catch::Approx(reference[i]).epsilon(1e-5));
    CHECK(tbb64[i] == Catch::Approx(reference[i]).epsilon(1e-12));
  }
}
