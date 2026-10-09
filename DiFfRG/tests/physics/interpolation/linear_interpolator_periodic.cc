#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

TEST_CASE("Test template constraints for periodic interpolators", "[interpolator][periodic]")
{
  STATIC_REQUIRE(is_interpolator<LinearInterpolator1D<double, LinPeriodicCoordinates>>);
  STATIC_REQUIRE(is_interpolator<LinearInterpolatorND<double, LogLinPeriodicCoordinates>>);
  STATIC_REQUIRE(is_interpolator<LinearInterpolatorND<double, LogLinLinPeriodicCoordinates>>);
}

//--------------------------------------------
// 1D
//--------------------------------------------

TEMPLATE_TEST_CASE("Test 1D periodic interpolation", "[float][double][interpolation][periodic]", float, double)
{
  using T = TestType;
  DiFfRG::Init();

  using ctype = T;
  using Coordinates = LinearPeriodicCoordinates1D<ctype>;

  const ctype start = GENERATE(take(2, random(-10., 10.)));
  const ctype period = GENERATE(take(2, random(1., 20.)));
  const int size = GENERATE(take(3, random(3, 64)));
  const ctype stop = start + period;
  const ctype dx = period / size;

  Coordinates coords(size, start, stop);

  // a smooth function with exactly the periodicity of the grid, so that the seam carries no jump
  const auto f = [&](const ctype x) { return std::cos(2 * ctype(M_PI) * (x - start) / period); };

  std::vector<T> in_data(size);
  for (int i = 0; i < size; ++i)
    in_data[i] = f(coords.forward(i));

  LinearInterpolator1D<T, Coordinates> interpolator(coords);
  interpolator.update(in_data.data());
  const auto &ip = interpolator;

  const ctype eps = std::numeric_limits<ctype>::epsilon() * 1e2;

  // The grid points are reproduced exactly, including the first one when approached from either side
  for (int i = 0; i < size; ++i)
    CHECK(is_close(ip(coords.forward(i)), in_data[i], eps));
  CHECK(is_close(ip(stop), in_data[0], eps));
  CHECK(is_close(ip(start - period), in_data[0], eps));
  CHECK(is_close(ip(start + 3 * period), in_data[0], eps));

  // The seam cell [stop - dx, stop) interpolates between the last and the first grid point, i.e. the
  // interpolant is continuous across it - this is what a clamped, non-periodic axis gets wrong.
  for (const ctype s : {ctype(0.25), ctype(0.5), ctype(0.75)}) {
    const ctype x = stop - dx + s * dx;
    const T expected = (1 - s) * in_data[size - 1] + s * in_data[0];
    if (!is_close(ip(x), expected, 1e-5))
      std::cout << "x: " << x << " got: " << ip(x) << " expected: " << expected << std::endl;
    CHECK(is_close(ip(x), expected, 1e-5));
  }

  // periodicity of the interpolant itself, everywhere and for arbitrary winding
  const ctype x_pt = start + GENERATE(take(5, random(0., 1.))) * period;
  CHECK(is_close(ip(x_pt), ip(x_pt + period), 1e-5));
  CHECK(is_close(ip(x_pt), ip(x_pt - 4 * period), 1e-5));
}

TEST_CASE("Test that a periodic axis is closed while a bounded one is clamped", "[interpolation][periodic]")
{
  DiFfRG::Init();

  constexpr int size = 4;
  const std::vector<double> data{0., 1., 2., 3.};

  // Bounded: outside the grid the boundary value is held, and the last grid point is reproduced
  LinearCoordinates1D<double> bounded(size, 0., 3.);
  LinearInterpolator1D<double, LinearCoordinates1D<double>> bip(bounded);
  bip.update(data.data());
  CHECK(is_close(bip(-10.), 0., 1e-12));
  CHECK(is_close(bip(1.5), 1.5, 1e-12));
  CHECK(is_close(bip(3.), 3., 1e-12));
  CHECK(is_close(bip(10.), 3., 1e-12));

  // Periodic over [0, 4): the cell [3, 4) wraps from data[3] back to data[0]
  LinearPeriodicCoordinates1D<double> periodic(size, 0., 4.);
  LinearInterpolator1D<double, LinearPeriodicCoordinates1D<double>> pip(periodic);
  pip.update(data.data());
  CHECK(is_close(pip(1.5), 1.5, 1e-12));
  CHECK(is_close(pip(3.), 3., 1e-12));
  CHECK(is_close(pip(3.5), 1.5, 1e-12));
  CHECK(is_close(pip(4.), 0., 1e-12));
  CHECK(is_close(pip(-0.5), 1.5, 1e-12));
  CHECK(is_close(pip(9.5), 1.5, 1e-12));
}

//--------------------------------------------
// 3D, as used for (S0, S1, SPhi) in the YangMills Full example
//--------------------------------------------

TEST_CASE("Test 3D interpolation with a periodic last axis", "[3D][interpolation][periodic]")
{
  DiFfRG::Init();

  using Coordinates3D = LogLinLinPeriodicCoordinates;

  const int p1_size = 12, p2_size = 5, p3_size = 6;
  Coordinates3D coords(LogarithmicCoordinates1D<double>(p1_size, 1e-3, 10., 4.),
                       LinearCoordinates1D<double>(p2_size, 0., 1.),
                       LinearPeriodicCoordinates1D<double>(p3_size, -M_PI, M_PI));

  // separable, periodic in the last argument
  const auto f = [](const double s0, const double s1, const double phi) {
    return (1. + s0) * (2. + s1) * std::cos(phi);
  };

  std::vector<double> in_data(p1_size * p2_size * p3_size);
  for (int i = 0; i < p1_size; ++i)
    for (int j = 0; j < p2_size; ++j)
      for (int l = 0; l < p3_size; ++l) {
        const auto pt = coords.forward(i, j, l);
        in_data[(i * p2_size + j) * p3_size + l] = f(pt[0], pt[1], pt[2]);
      }

  LinearInterpolatorND<double, Coordinates3D> interpolator(coords);
  interpolator.update(in_data.data());
  const auto &ip = interpolator;

  // nodes are reproduced
  for (int i = 0; i < p1_size; ++i)
    for (int j = 0; j < p2_size; ++j)
      for (int l = 0; l < p3_size; ++l) {
        const auto pt = coords.forward(i, j, l);
        CHECK(is_close(ip(pt[0], pt[1], pt[2]), in_data[(i * p2_size + j) * p3_size + l], 1e-10));
      }

  // the seam at +-pi carries no jump, and phi is a genuine angle for the interpolant
  const auto pt = coords.forward(size_t(5), size_t(2), size_t(0));
  const double s0 = pt[0], s1 = pt[1];
  CHECK(is_close(ip(s0, s1, M_PI - 1e-9), ip(s0, s1, -M_PI + 1e-9), 1e-6));
  for (const double phi : {-3.0, -1.1, 0.0, 0.7, 2.9, 3.1}) {
    CHECK(is_close(ip(s0, s1, phi), ip(s0, s1, phi + 2 * M_PI), 1e-10));
    CHECK(is_close(ip(s0, s1, phi), ip(s0, s1, phi - 6 * M_PI), 1e-10));
  }

  // in the seam cell the interpolant blends the last node into the first one
  const double dphi = 2 * M_PI / p3_size;
  const double mid = ip(s0, s1, M_PI - 0.5 * dphi);
  const double expected = 0.5 * (f(s0, s1, M_PI - dphi) + f(s0, s1, -M_PI));
  if (!is_close(mid, expected, 1e-10)) std::cout << "mid: " << mid << " expected: " << expected << std::endl;
  CHECK(is_close(mid, expected, 1e-10));
}

TEST_CASE("Test 3D periodic interpolation on GPU", "[3D][interpolation][periodic][gpu]")
{
  DiFfRG::Init();

  using Coordinates3D = LogLinLinPeriodicCoordinates;

  const int p1_size = 8, p2_size = 4, p3_size = 6;
  Coordinates3D coords(LogarithmicCoordinates1D<double>(p1_size, 1e-3, 10., 4.),
                       LinearCoordinates1D<double>(p2_size, 0., 1.),
                       LinearPeriodicCoordinates1D<double>(p3_size, -M_PI, M_PI));

  std::vector<double> in_data(p1_size * p2_size * p3_size);
  for (int i = 0; i < p1_size; ++i)
    for (int j = 0; j < p2_size; ++j)
      for (int l = 0; l < p3_size; ++l) {
        const auto pt = coords.forward(i, j, l);
        in_data[(i * p2_size + j) * p3_size + l] = std::cos(pt[2]) + 0.1 * pt[1];
      }

  LinearInterpolatorND<double, Coordinates3D> interpolator(coords);
  interpolator.update(in_data.data());

  const auto ref = coords.forward(size_t(3), size_t(1), size_t(0));
  const double s0 = ref[0], s1 = ref[1];
  // sample inside the seam cell, where a clamped axis would read unrelated data
  const double phi = M_PI - 0.5 * (2 * M_PI / p3_size);

  const double res_host = interpolator(s0, s1, phi);

  double res_gpu = 0.;
  Kokkos::parallel_reduce(
      "periodic 3D", Kokkos::RangePolicy(0, 1),
      KOKKOS_LAMBDA(const uint, double &update) { update += interpolator(s0, s1, phi); }, res_gpu);

  CHECK(is_close(res_host, res_gpu, 1e-12));
  // the seam value is the average of the last and the first node
  const double expected = 0.5 * (in_data[(3 * p2_size + 1) * p3_size + (p3_size - 1)] +
                                 in_data[(3 * p2_size + 1) * p3_size + 0]);
  CHECK(is_close(res_host, expected, 1e-10));
}
