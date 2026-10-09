#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

// (S0, S1, SPhi) as in the YangMills Full example: bounded S0 and S1, periodic shape angle
using Coordinates3D = LogLinLinPeriodicCoordinates;

namespace
{
  struct Grid {
    int n0, n1, n2;
    Coordinates3D coords;
    Grid(int n0, int n1, int n2)
        : n0(n0), n1(n1), n2(n2),
          coords(LogarithmicCoordinates1D<double>(n0, 1e-3, 10., 4.), LinearCoordinates1D<double>(n1, 0., 1.),
                 LinearPeriodicCoordinates1D<double>(n2, -M_PI, M_PI))
    {
    }

    template <typename F> PeriodicCubicInterpolator3D<double, Coordinates3D> fill(const F &f) const
    {
      std::vector<double> in(n0 * n1 * n2);
      for (int i = 0; i < n0; ++i)
        for (int j = 0; j < n1; ++j)
          for (int l = 0; l < n2; ++l) {
            const auto pt = coords.forward(i, j, l);
            in[(i * n1 + j) * n2 + l] = f(i, j, pt[2]);
          }
      PeriodicCubicInterpolator3D<double, Coordinates3D> ip(coords);
      ip.update(in.data());
      return ip;
    }
  };
} // namespace

TEST_CASE("PeriodicCubicInterpolator3D is an interpolator", "[interpolator][periodic][cubic]")
{
  STATIC_REQUIRE(is_interpolator<PeriodicCubicInterpolator3D<double, Coordinates3D>>);
  STATIC_REQUIRE(has_kernel_handle<PeriodicCubicInterpolator3D<double, Coordinates3D>>);
}

TEST_CASE("PeriodicCubicInterpolator3D: nodes, seam, bounded axes", "[3D][interpolation][periodic][cubic]")
{
  DiFfRG::Init();

  const Grid g(8, 5, 12);
  // linear in the indices of the bounded axes, which those reproduce exactly; smooth in the angle
  const auto f = [](int i, int j, double phi) { return (1. + 0.5 * i) * (2. + j) * (1. + 0.3 * std::sin(phi)); };
  const auto ip = g.fill(f);

  for (int i = 0; i < g.n0; ++i)
    for (int j = 0; j < g.n1; ++j)
      for (int l = 0; l < g.n2; ++l) {
        const auto pt = g.coords.forward(i, j, l);
        CHECK(is_close(ip(pt[0], pt[1], pt[2]), f(i, j, pt[2]), 1e-12));
      }

  // between nodes of the bounded axes, at an angle node, the result is the bilinear blend
  const auto a = g.coords.forward(size_t(2), size_t(1), size_t(4));
  const auto b = g.coords.forward(size_t(3), size_t(2), size_t(4));
  const double x = 0.5 * (a[0] + b[0]), y = 0.25 * a[1] + 0.75 * b[1];
  const auto [ix, iy, iz] = g.coords.backward(x, y, a[2]);
  CHECK(is_close(ip(x, y, a[2]), (1. + 0.5 * ix) * (2. + iy) * (1. + 0.3 * std::sin(a[2])), 1e-10));

  // periodic: continuous across the seam and invariant under winding
  CHECK(is_close(ip(a[0], a[1], M_PI - 1e-9), ip(a[0], a[1], -M_PI + 1e-9), 1e-7));
  for (const double phi : {-3.0, -1.1, 0.0, 0.7, 2.9, 3.1})
    CHECK(is_close(ip(a[0], a[1], phi), ip(a[0], a[1], phi + 6 * M_PI), 1e-10));
}

TEST_CASE("PeriodicCubicInterpolator3D: no kink at a node", "[3D][interpolation][periodic][cubic]")
{
  DiFfRG::Init();

  const Grid g(4, 3, 12);
  const auto node = g.coords.forward(size_t(1), size_t(1), size_t(9));
  const double s0 = node[0], s1 = node[1], phi0 = node[2];

  // Even about phi0: the interpolant must be even too, with zero slope, i.e. deviate like delta^2.
  // A linear stencil would deviate like |delta| here.
  const auto even =
      g.fill([&](int, int, double phi) { return std::cos(phi - phi0) + 0.2 * std::cos(2 * (phi - phi0)); });
  const double z0 = even(s0, s1, phi0);
  for (const double d : {1e-1, 1e-2, 1e-3}) {
    CHECK(is_close(even(s0, s1, phi0 + d), even(s0, s1, phi0 - d), 1e-12));
    const double curvature = (even(s0, s1, phi0 + d) - z0) / (d * d);
    CHECK(std::abs(curvature) < 2.);
  }
  // the second difference stays finite as delta -> 0, so the deviation is genuinely quadratic
  const double c2 = (even(s0, s1, phi0 + 1e-2) - z0) / 1e-4, c3 = (even(s0, s1, phi0 + 1e-3) - z0) / 1e-6;
  CHECK(is_close(c2, c3, 0.05 * std::abs(c2)));

  // Odd about phi0: zero at the node, slope the central difference of the neighbouring nodes
  const auto odd = g.fill([&](int, int, double phi) { return std::sin(phi - phi0); });
  const double h = 2 * M_PI / g.n2;
  CHECK(is_close(odd(s0, s1, phi0), 0., 1e-12));
  const double slope = (odd(s0, s1, phi0 + 1e-6) - odd(s0, s1, phi0 - 1e-6)) / 2e-6;
  CHECK(is_close(slope, std::sin(h) / h, 1e-6));
}

TEST_CASE("PeriodicCubicInterpolator3D: third-order convergence in the angle", "[3D][interpolation][periodic][cubic]")
{
  DiFfRG::Init();

  const auto f = [](int, int, double phi) { return std::exp(std::cos(phi)); };
  const auto max_error = [&](int n2) {
    const Grid g(3, 3, n2);
    const auto ip = g.fill(f);
    const auto pt = g.coords.forward(size_t(1), size_t(1), size_t(0));
    double err = 0.;
    for (int s = 0; s < 1000; ++s) {
      const double phi = -M_PI + 2 * M_PI * (s + 0.5) / 1000;
      err = std::max(err, std::abs(ip(pt[0], pt[1], phi) - f(0, 0, phi)));
    }
    return err;
  };
  const double e12 = max_error(12), e24 = max_error(24), e48 = max_error(48);
  // h^3 would be a ratio of 8 per halving
  CHECK(e12 / e24 > 6.);
  CHECK(e24 / e48 > 6.);
}

TEST_CASE("PeriodicCubicInterpolator3D: host, device and handles agree", "[3D][interpolation][periodic][cubic][gpu]")
{
  DiFfRG::Init();

  const Grid g(8, 4, 12);
  const auto ip =
      g.fill([](int i, int j, double phi) { return std::cos(phi) + 0.1 * j + 0.01 * i * std::sin(2 * phi); });

  const auto ref = g.coords.forward(size_t(3), size_t(1), size_t(0));
  const double s0 = 1.3 * ref[0], s1 = 0.4, phi = 2.9;

  const double res_host = ip(s0, s1, phi);
  double res_gpu = 0.;
  Kokkos::parallel_reduce(
      "periodic cubic 3D", Kokkos::RangePolicy(0, 1),
      KOKKOS_LAMBDA(const uint, double &update) { update += ip(s0, s1, phi); }, res_gpu);
  CHECK(is_close(res_host, res_gpu, 1e-12));

  CHECK(ip.handle<double>()(s0, s1, phi) == Catch::Approx(res_host).epsilon(1e-14));
  CHECK(ip.handle<float>()(float(s0), float(s1), float(phi)) == Catch::Approx(res_host).epsilon(1e-5));
}
