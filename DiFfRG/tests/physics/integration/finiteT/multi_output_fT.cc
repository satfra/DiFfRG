#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/multi_value.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_1ang.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

#include <vector>

using namespace DiFfRG;

namespace
{
  enum class Kind { plain, even, split };

  // A summand with an algebraic tail in the frequency and one that vanishes for |q0| >= 1.
  KOKKOS_FORCEINLINE_FUNCTION double tail(const double x, const double q0, const double p, const double a)
  {
    return a / (1. + p * p + x * x + q0 * q0);
  }
  KOKKOS_FORCEINLINE_FUNCTION double bump(const double x, const double q0, const double a)
  {
    const double s = 1. - q0 * q0;
    return s > 0. ? a * x * s * s : 0.;
  }
  template <int j> KOKKOS_FORCEINLINE_FUNCTION double projector(const double x, const double q0, const double p)
  {
    if constexpr (j == 0)
      return 1.;
    else if constexpr (j == 1)
      return x * x + q0 * q0;
    else
      return p + x;
  }

  /// Projection j of one integrand: (tail [+ bump]) * projector, plus an odd part for `plain`.
  template <Kind kind, int j> struct ProjectionT {
    static constexpr bool matsubara_even = kind == Kind::even;
    static constexpr bool matsubara_split = kind == Kind::split;

    static KOKKOS_FORCEINLINE_FUNCTION double kernel_tail(const double x, const double q0, const double p,
                                                          const double a)
    {
      const double odd = kind == Kind::plain ? 0.1 * a * x * q0 : 0.;
      return tail(x, q0, p, a) * projector<j>(x, q0, p) + odd;
    }
    static KOKKOS_FORCEINLINE_FUNCTION double kernel_finite_extent(const double x, const double q0, const double p,
                                                                   const double a)
    {
      return bump(x, q0, a) * projector<j>(x, q0, p);
    }
    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double x, const double q0, const double p, const double a)
    {
      if constexpr (kind == Kind::split)
        return kernel_tail(x, q0, p, a) + kernel_finite_extent(x, q0, p, a);
      else
        return kernel_tail(x, q0, p, a);
    }
    static KOKKOS_FORCEINLINE_FUNCTION double constant(const double p, const double a) { return a * p * double(j); }
  };

  template <Kind kind> struct AllProjectionsT {
    using MV = MultiValue<double, 3>;
    static constexpr bool matsubara_even = kind == Kind::even;
    static constexpr bool matsubara_split = kind == Kind::split;

    template <int j> using P = ProjectionT<kind, j>;

    static KOKKOS_FORCEINLINE_FUNCTION MV kernel_tail(const double x, const double q0, const double p, const double a)
    {
      return MV(P<0>::kernel_tail(x, q0, p, a), P<1>::kernel_tail(x, q0, p, a), P<2>::kernel_tail(x, q0, p, a));
    }
    static KOKKOS_FORCEINLINE_FUNCTION MV kernel_finite_extent(const double x, const double q0, const double p,
                                                               const double a)
    {
      return MV(P<0>::kernel_finite_extent(x, q0, p, a), P<1>::kernel_finite_extent(x, q0, p, a),
                P<2>::kernel_finite_extent(x, q0, p, a));
    }
    static KOKKOS_FORCEINLINE_FUNCTION MV kernel(const double x, const double q0, const double p, const double a)
    {
      return MV(P<0>::kernel(x, q0, p, a), P<1>::kernel(x, q0, p, a), P<2>::kernel(x, q0, p, a));
    }
    static KOKKOS_FORCEINLINE_FUNCTION MV constant(const double p, const double a)
    {
      return MV(P<0>::constant(p, a), P<1>::constant(p, a), P<2>::constant(p, a));
    }
  };

  void check_close(const std::array<std::vector<double>, 3> &ref, const std::array<std::vector<double>, 3> &got,
                   const double tolerance)
  {
    for (size_t j = 0; j < 3; ++j)
      for (size_t i = 0; i < ref[j].size(); ++i) {
        CAPTURE(j, i);
        CHECK(is_close(got[j][i], ref[j][i], tolerance));
      }
  }

  template <Kind kind, typename ExecutionSpace>
  void check(QuadratureProvider &provider, const double T, const double tolerance)
  {
    using Multi = QuadratureIntegrator_fT<2, MultiValue<double, 3>, AllProjectionsT<kind>, ExecutionSpace>;
    const std::array<size_t, 1> grid{{24}};
    const std::array<double, 1> lo{{0.}}, hi{{1.5}};
    const std::array<QuadratureType, 1> rule{{QuadratureType::legendre}};
    Multi multi(provider, grid, lo, hi, rule, T);
    QuadratureIntegrator_fT<2, double, ProjectionT<kind, 0>, ExecutionSpace> s0(provider, grid, lo, hi, rule, T);
    QuadratureIntegrator_fT<2, double, ProjectionT<kind, 1>, ExecutionSpace> s1(provider, grid, lo, hi, rule, T);
    QuadratureIntegrator_fT<2, double, ProjectionT<kind, 2>, ExecutionSpace> s2(provider, grid, lo, hi, rule, T);
    multi.set_k(1.);
    s0.set_k(1.);
    s1.set_k(1.);
    s2.set_k(1.);
    CAPTURE(int(kind), T, multi.get_matsubara_size());

    const size_t n = 29;
    const LinearCoordinates1D<double> coordinates(n, 0., 1.);
    std::array<std::vector<double>, 3> ref, got;
    for (size_t j = 0; j < 3; ++j) {
      ref[j].resize(n);
      got[j].resize(n);
    }
    s0.map(ref[0].data(), coordinates, 0.7);
    s1.map(ref[1].data(), coordinates, 0.7);
    s2.map(ref[2].data(), coordinates, 0.7);
    multi.map({got[0].data(), got[1].data(), got[2].data()}, coordinates, 0.7);
    check_close(ref, got, tolerance);

    MultiValue<double, 3> v;
    multi.get(v, 0.4, 0.7);
    std::array<double, 3> r;
    s0.get(r[0], 0.4, 0.7);
    s1.get(r[1], 0.4, 0.7);
    s2.get(r[2], 0.4, 0.7);
    for (size_t j = 0; j < 3; ++j)
      CHECK(is_close(v[j], r[j], tolerance));
  }

  template <typename ExecutionSpace> void check_all(QuadratureProvider &provider, const double T, const double tol)
  {
    check<Kind::plain, ExecutionSpace>(provider, T, tol);
    check<Kind::even, ExecutionSpace>(provider, T, tol);
    check<Kind::split, ExecutionSpace>(provider, T, tol);
  }

  // Integrator_fT_p2_1ang: kernel(q, cos, q0, p, a).
  template <int j> struct Projection3D {
    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double q, const double c, const double q0, const double p,
                                                     const double a)
    {
      return tail(q, q0, p * c, a) * projector<j>(q, q0, p);
    }
    static KOKKOS_FORCEINLINE_FUNCTION double constant(const double, const double) { return 0.; }
  };
  struct AllProjections3D {
    using MV = MultiValue<double, 3>;
    static KOKKOS_FORCEINLINE_FUNCTION MV kernel(const double q, const double c, const double q0, const double p,
                                                 const double a)
    {
      return MV(Projection3D<0>::kernel(q, c, q0, p, a), Projection3D<1>::kernel(q, c, q0, p, a),
                Projection3D<2>::kernel(q, c, q0, p, a));
    }
    static KOKKOS_FORCEINLINE_FUNCTION MV constant(const double, const double) { return MV(0.); }
  };
} // namespace

TEST_CASE("Finite-T MultiValue kernels equal separate kernels", "[integration][multi_output][finiteT]")
{
  DiFfRG::Init();
  QuadratureProvider provider;

  // T = 0.02 and 0.3: Matsubara sums; T = 0: the vacuum rule.
  const double T = GENERATE(0.02, 0.3, 0.);
  SECTION("GPU") { check_all<GPU_exec>(provider, T, 1e-14); }
  SECTION("TBB") { check_all<TBB_exec>(provider, T, 1e-14); }
}

TEST_CASE("Finite-T MultiValue kernel through Integrator_fT_p2_1ang", "[integration][multi_output][finiteT]")
{
  DiFfRG::Init();
  QuadratureProvider provider;

  Integrator_fT_p2_1ang<4, MultiValue<double, 3>, AllProjections3D, GPU_exec> multi(provider, {32, 8}, 2.);
  Integrator_fT_p2_1ang<4, double, Projection3D<0>, GPU_exec> s0(provider, {32, 8}, 2.);
  Integrator_fT_p2_1ang<4, double, Projection3D<1>, GPU_exec> s1(provider, {32, 8}, 2.);
  Integrator_fT_p2_1ang<4, double, Projection3D<2>, GPU_exec> s2(provider, {32, 8}, 2.);
  multi.set_T(0.1);
  s0.set_T(0.1);
  s1.set_T(0.1);
  s2.set_T(0.1);
  multi.set_k(0.8);
  s0.set_k(0.8);
  s1.set_k(0.8);
  s2.set_k(0.8);

  const size_t n = 20;
  const LinearCoordinates1D<double> coordinates(n, 0., 1.);
  std::array<std::vector<double>, 3> ref, got;
  for (size_t j = 0; j < 3; ++j) {
    ref[j].resize(n);
    got[j].resize(n);
  }
  s0.map(ref[0].data(), coordinates, 0.7);
  s1.map(ref[1].data(), coordinates, 0.7);
  s2.map(ref[2].data(), coordinates, 0.7);
  multi.map({got[0].data(), got[1].data(), got[2].data()}, coordinates, 0.7);
  check_close(ref, got, 1e-14);
}
