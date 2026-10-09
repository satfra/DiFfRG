#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

TEST_CASE("Test template constraints for LinearInterpolator2D", "[interpolator]")
{
  STATIC_REQUIRE(is_interpolator<LinearInterpolator2D<double, LinLinCoordinates>>);
  STATIC_REQUIRE(is_interpolator<LinearInterpolator2D<complex<double>, LinLinCoordinates>>);
  STATIC_REQUIRE(is_interpolator<LinearInterpolator2D<autodiff::real, LinLinCoordinates>>);
  STATIC_REQUIRE(is_interpolator<LinearInterpolator2D<cxreal, LinLinCoordinates>>);

  STATIC_REQUIRE(!is_interpolator<int>);
  STATIC_REQUIRE(!is_interpolator<std::array<double, 3>>);
}

TEMPLATE_TEST_CASE("Test 2D interpolation", "[float][double][complex][autodiff][interpolation][2D]", float, double,
                   complex<double>, complex<float>, autodiff::real, cxreal)
{
  using T = TestType;

  DiFfRG::Init();

  using ctype = typename get_type::ctype<T>;

  using Coordinates1D = LinearCoordinates1D<ctype>;
  using Coordinates2D = CoordinatePackND<Coordinates1D, Coordinates1D>;

  const ctype p1_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p1_stop = GENERATE(take(2, random(1, 100))) + p1_start;
  const int p1_size = GENERATE(take(2, random(10, 100)));

  const ctype p2_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p2_stop = GENERATE(take(2, random(1, 100))) + p1_start;
  const int p2_size = GENERATE(take(2, random(10, 100)));

  std::vector<T> in_data(p1_size * p2_size);
  for (int i = 0; i < p1_size; ++i)
    for (int j = 0; j < p2_size; ++j)
      in_data[i * p2_size + j] = j;

  Coordinates2D coords(Coordinates1D(p1_size, p1_start, p1_stop), Coordinates1D(p2_size, p2_start, p2_stop));
  LinearInterpolatorND<T, Coordinates2D> interpolator(coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(2, random(2, 200)));
  const ctype p1_pt = (p1_start + GENERATE(take(3, random(0., 1.))) * (p1_stop - p1_start));
  const ctype p2_pt = (p2_start + GENERATE(take(3, random(0., 1.))) * (p2_stop - p2_start));

  const auto res_host = interpolator(p1_pt, p2_pt) * ctype(n_el);

  auto [p1_idx, p2_idx] = coords.backward(p1_pt, p2_pt);
  p1_idx = std::max((ctype)0, std::min(p1_idx, ctype(p1_size)));
  p2_idx = std::max((ctype)0, std::min(p2_idx, ctype(p2_size)));
  const auto res_local = p2_idx * ctype(n_el);

  if (!is_close(res_host, res_local, 1e-6 * n_el))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, 1e-6 * n_el));
}

TEMPLATE_TEST_CASE("Test 2D interpolation GPU", "[float][double][complex][autodiff][interpolation][2D]", float, double,
                   complex<double>, complex<float>, autodiff::real, cxreal)
{
  using T = TestType;

  DiFfRG::Init();

  using ctype = typename get_type::ctype<T>;

  using Coordinates1D = LinearCoordinates1D<ctype>;
  using Coordinates2D = CoordinatePackND<Coordinates1D, Coordinates1D>;

  const ctype p1_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p1_stop = GENERATE(take(2, random(1, 100))) + p1_start;
  const int p1_size = GENERATE(take(2, random(10, 100)));

  const ctype p2_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p2_stop = GENERATE(take(2, random(1, 100))) + p1_start;
  const int p2_size = GENERATE(take(2, random(10, 100)));

  std::vector<T> in_data(p1_size * p2_size);
  for (int i = 0; i < p1_size; ++i)
    for (int j = 0; j < p2_size; ++j)
      in_data[i * p2_size + j] = j;

  Coordinates2D coords(Coordinates1D(p1_size, p1_start, p1_stop), Coordinates1D(p2_size, p2_start, p2_stop));
  LinearInterpolatorND<T, Coordinates2D> interpolator(coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(2, random(2, 200)));
  const ctype p1_pt = (p1_start + GENERATE(take(3, random(0., 1.))) * (p1_stop - p1_start));
  const ctype p2_pt = (p2_start + GENERATE(take(3, random(0., 1.))) * (p2_stop - p2_start));

  const auto res_host = interpolator(p1_pt, p2_pt) * ctype(n_el);

  auto [p1_idx, p2_idx] = coords.backward(p1_pt, p2_pt);
  p1_idx = std::max((ctype)0, std::min(p1_idx, ctype(p1_size)));
  p2_idx = std::max((ctype)0, std::min(p2_idx, ctype(p2_size)));
  const auto res_local = p2_idx * ctype(n_el);

  T res_gpu;
  Kokkos::parallel_reduce(
      "Get one point", Kokkos::RangePolicy(0, n_el),
      KOKKOS_LAMBDA(const uint, T &update) { update += interpolator(p1_pt, p2_pt); }, res_gpu);

  if (!is_close(res_host, res_local, 1e-6 * n_el))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  if (!is_close(res_gpu, res_local, 1e-6 * n_el))
    std::cout << "gpu: " << res_gpu << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, 1e-6 * n_el));
  CHECK(is_close(res_gpu, res_local, 1e-6 * n_el));
}
TEST_CASE("Test 2D row-major element access", "[interpolator]")
{
  DiFfRG::Init();

  // update() takes row-major input, and operator[] hands it back under the same linear index. The
  // two are not the same traversal of memory: the device view is pinned to GPU_memory, hence
  // LayoutLeft, so a flat read of the mirror would come back transposed. Extents are deliberately
  // unequal and the payload is the linear index itself, so a transpose cannot go unnoticed.
  using Coordinates1D = LinearCoordinates1D<double>;
  using Coordinates2D = CoordinatePackND<Coordinates1D, Coordinates1D>;

  constexpr size_t n0 = 5, n1 = 7;
  std::vector<double> in_data(n0 * n1);
  for (size_t i = 0; i < in_data.size(); ++i)
    in_data[i] = double(i);

  Coordinates2D coords(Coordinates1D(n0, 0., 1.), Coordinates1D(n1, 0., 1.));
  LinearInterpolatorND<double, Coordinates2D> interpolator(coords);
  interpolator.update(in_data.data());

  for (size_t i = 0; i < in_data.size(); ++i)
    CHECK(interpolator[i] == in_data[i]);

  // and the interpolant agrees with the input on the grid nodes it was built from
  for (size_t i = 0; i < n0; ++i)
    for (size_t j = 0; j < n1; ++j) {
      const auto x = coords.forward(i, j);
      CHECK(is_close(interpolator(x[0], x[1]), in_data[i * n1 + j], 1e-12));
    }
}
