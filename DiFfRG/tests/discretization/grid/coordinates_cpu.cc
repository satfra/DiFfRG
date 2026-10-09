#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/combined_coordinates.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/discretization/coordinates/stack_coordinates.hh>

#include <cmath>
#include <set>
#include <string>

using namespace DiFfRG;

TEST_CASE("Test 1D Logarithmic coordinates", "[1D][coordinates]")
{
  const float p_start = GENERATE(take(3, random(1e-6, 1e-1)));
  const float p_stop = GENERATE(take(3, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(3, random(10, 100)));
  const float p_bias = GENERATE(take(3, random(1., 10.)));

  LogarithmicCoordinates1D<float> coords(p_size, p_start, p_stop, p_bias);

  const auto grid = make_grid(coords);
  CHECK(is_close(grid[0][0], p_start, 1e-8f));
  for (uint i = 1; i < grid.size() - 1; ++i)
    CHECK(grid[i][0] > grid[i - 1][0]);
  CHECK(is_close(grid[grid.size() - 1][0], p_stop, 1e-5));
}

TEST_CASE("Test 1D focused logarithmic coordinates", "[1D][coordinates]")
{
  const double p_start = GENERATE(take(2, random(1e-4, 1e-2)));
  const double p_stop = GENERATE(take(2, random(10., 300.)));
  const int p_size = GENERATE(take(2, random(16, 128)));
  // center both inside and outside [p_start, p_stop] -- the latter is a legal one-sided stretch
  const double p_center = GENERATE(take(2, random(0.05, 5.)), take(1, random(400., 1000.)));
  const double p_focus = GENERATE(0., take(3, random(0.1, 6.)));

  FocusedLogCoordinates1D<double> coords(p_size, p_start, p_stop, p_center, p_focus);

  const auto grid = make_grid(coords);
  REQUIRE(grid.size() == size_t(p_size));

  // Code all over the models pairs grid element 0 with p_start directly instead of going through
  // forward(), so the lower endpoint has to come back bit-for-bit. The upper one is only required
  // to a few ulp -- nothing reads it that way.
  CHECK(grid[0][0] == p_start);
  CHECK(is_close(grid[p_size - 1][0], p_stop, 1e-12));

  for (int i = 1; i < p_size; ++i)
    CHECK(grid[i][0] > grid[i - 1][0]);

  // backward inverts forward, both on grid points and between them
  for (int i = 0; i < p_size; ++i)
    CHECK(is_close(coords.backward(coords.forward(i)), double(i), 1e-8));
  for (const double x : {0.3, 1.5, 0.5 * (p_size - 1), p_size - 1.7})
    CHECK(is_close(coords.backward(coords.forward(x)), x, 1e-8));

  // backward_derivative is the derivative of backward
  for (const double y : {p_start * 2., p_center, p_stop / 2.}) {
    const double h = 1e-6 * y;
    const double fd = (coords.backward(y + h) - coords.backward(y - h)) / (2. * h);
    CHECK(is_close(coords.backward_derivative(y), fd, 1e-5));
  }
}

TEST_CASE("Focused logarithmic coordinates: focus reduces to a geometric grid at 0", "[1D][coordinates]")
{
  const double p_start = 5e-3, p_stop = 250.;
  const int p_size = 64;

  FocusedLogCoordinates1D<double> coords(p_size, p_start, p_stop, 1.0, 0.0);
  const auto grid = make_grid(coords);

  // at focus == 0 the spacing is constant in log(p), independent of the center
  const double ratio = std::log(grid[1][0] / grid[0][0]);
  for (int i = 1; i < p_size; ++i)
    CHECK(is_close(std::log(grid[i][0] / grid[i - 1][0]), ratio, 1e-10));
  CHECK(is_close(ratio, std::log(p_stop / p_start) / (p_size - 1), 1e-12));

  // the center is irrelevant in this limit
  FocusedLogCoordinates1D<double> other(p_size, p_start, p_stop, 100.0, 0.0);
  for (int i = 0; i < p_size; ++i)
    CHECK(is_close(other.forward(i), coords.forward(i), 1e-12));
}

TEST_CASE("Focused logarithmic coordinates: clustering behaviour", "[1D][coordinates]")
{
  const double p_start = 5e-3, p_stop = 250., p_center = 1.0;
  const int p_size = 64;

  const auto cell_at_center = [&](const double focus) {
    FocusedLogCoordinates1D<double> coords(p_size, p_start, p_stop, p_center, focus);
    const auto grid = make_grid(coords);
    // width in decades of the cell straddling the center
    for (int i = 1; i < p_size; ++i)
      if (grid[i][0] >= p_center) return std::log10(grid[i][0] / grid[i - 1][0]);
    return 0.;
  };

  // more focus means a finer cell at the center
  const double w0 = cell_at_center(0.), w1 = cell_at_center(1.), w2 = cell_at_center(2.), w4 = cell_at_center(4.);
  CHECK(w1 < w0);
  CHECK(w2 < w1);
  CHECK(w4 < w2);
  // the calibration point quoted in the class documentation
  CHECK(is_close(w2, 0.0212, 1e-3));

  // the map is symmetric in log(p) about the center, so the points split evenly around it
  FocusedLogCoordinates1D<double> coords(p_size, p_start, p_stop, p_center, 2.0);
  const auto grid = make_grid(coords);
  int below = 0;
  for (int i = 0; i < p_size; ++i)
    if (grid[i][0] < p_center) ++below;
  CHECK(std::abs(below - (p_size - below)) <= 1);
}

TEST_CASE("Focused logarithmic coordinates: inverse is stable far from the center", "[1D][coordinates]")
{
  // backward() is asinh(focus * (log(y) - log(center))) written out with a branch, because the
  // naive log(v + sqrt(v*v + 1)) cancels for v << 0. Check it against the library asinh.
  const double p_start = 1e-8, p_stop = 1e8, p_center = 1.0, p_focus = 3.0;
  const int p_size = 128;

  FocusedLogCoordinates1D<double> coords(p_size, p_start, p_stop, p_center, p_focus);

  const double s_min = std::asinh(p_focus * std::log(p_start / p_center));
  const double s_max = std::asinh(p_focus * std::log(p_stop / p_center));
  const double a = (s_max - s_min) / (p_size - 1);

  for (const double y : {1e-8, 1e-6, 1e-3, 0.1, 1.0, 10., 1e3, 1e6, 1e8}) {
    const double expected = (std::asinh(p_focus * std::log(y / p_center)) - s_min) / a;
    CHECK(is_close(coords.backward(y), expected, 1e-12));
  }
}

TEST_CASE("Focused logarithmic coordinates: to_string separates distinct grids", "[1D][coordinates]")
{
  // to_string() is the identity key of a coordinate dataset in HDF5, so grids that differ in any
  // parameter must render differently -- including IR cutoffs well below the six decimals that
  // std::to_string would emit.
  using C = FocusedLogCoordinates1D<double>;
  std::set<std::string> keys;
  for (const double start : {1e-3, 1e-4, 1e-6, 1e-7, 1e-8, 5e-9})
    keys.insert(C(64, start, 250., 1.0, 2.0).to_string());
  CHECK(keys.size() == 6);

  keys.clear();
  for (const double focus : {0.0, 1.0, 2.0, 2.5})
    keys.insert(C(64, 5e-3, 250., 1.0, focus).to_string());
  for (const double center : {0.5, 0.7, 1.0})
    keys.insert(C(64, 5e-3, 250., center, 2.0).to_string());
  for (const int size : {32, 64, 128})
    keys.insert(C(size, 5e-3, 250., 1.0, 2.0).to_string());
  // 4 focus + 3 centers + 3 sizes, with (center=1, focus=2, size=64) shared between the groups
  CHECK(keys.size() == 8);

  // and equality agrees with the key
  CHECK(C(64, 5e-3, 250., 1.0, 2.0) == C(64, 5e-3, 250., 1.0, 2.0));
  CHECK(!(C(64, 5e-3, 250., 1.0, 2.0) == C(64, 5e-3, 250., 1.0, 2.5)));
  CHECK(!(C(64, 5e-3, 250., 1.0, 2.0) == C(64, 5e-3, 250., 0.7, 2.0)));
}

TEST_CASE("Focused logarithmic coordinates: invalid arguments throw", "[1D][coordinates]")
{
  using C = FocusedLogCoordinates1D<double>;
  CHECK_THROWS(C(1, 1e-3, 10., 1., 2.));    // grid_extent must be > 1
  CHECK_THROWS(C(32, 0., 10., 1., 2.));     // start must be > 0
  CHECK_THROWS(C(32, -1e-3, 10., 1., 2.));  // start must be > 0
  CHECK_THROWS(C(32, 10., 10., 1., 2.));    // stop must be > start
  CHECK_THROWS(C(32, 1e-3, 1e-4, 1., 2.));  // stop must be > start
  CHECK_THROWS(C(32, 1e-3, 10., 0., 2.));   // center must be > 0
  CHECK_THROWS(C(32, 1e-3, 10., 1., -1.));  // focus must be >= 0
  CHECK_NOTHROW(C(2, 1e-3, 10., 1., 2.));
  CHECK_NOTHROW(C(32, 1e-3, 10., 1e6, 2.)); // center outside [start, stop] is legal
}

// prefix(n): the first n points, bit for bit, as the same type -- shared checks for the 1D types.
template <typename C> void check_prefix(const C &coords)
{
  const size_t N = coords.size();
  const std::string parent_key = coords.to_string();
  CHECK(coords.parent_size() == N);
  CHECK(parent_key.find(".prefix(") == std::string::npos);

  const std::set<size_t> ns{size_t(1), size_t(2), N / 3, N / 2, N - 1, N};
  std::set<std::string> keys{parent_key};
  for (const size_t n : ns) {
    const C pre = coords.prefix(n);
    REQUIRE(pre.size() == n);
    CHECK(pre.parent_size() == N);
    for (size_t i = 0; i < n; ++i)
      CHECK(pre.forward(i) == coords.forward(i)); // exact, not is_close: kernels map over the prefix
    for (const double x : {0.3, 0.5 * (N - 1)})
      CHECK(pre.backward(pre.forward(x)) == coords.backward(coords.forward(x)));
    // prefix of a prefix is a prefix of the parent
    const C pre2 = pre.prefix((n + 1) / 2);
    CHECK(pre2 == coords.prefix((n + 1) / 2));
    CHECK(pre2.to_string() == coords.prefix((n + 1) / 2).to_string());
    if (n < N) {
      CHECK(!(pre == coords));
      CHECK(pre.to_string() == parent_key + ".prefix(" + std::to_string(n) + ")");
    } else {
      // the full prefix is the grid itself
      CHECK(pre == coords);
      CHECK(pre.to_string() == parent_key);
    }
    keys.insert(pre.to_string());
  }
  // one key per distinct n, the n = N one being the parent's
  CHECK(keys.size() == ns.size());

  CHECK_THROWS(coords.prefix(0));
  CHECK_THROWS(coords.prefix(N + 1));
}

TEST_CASE("Coordinates prefix(): first n points bit for bit", "[1D][coordinates][prefix]")
{
  check_prefix(LinearCoordinates1D<double>(37, -2.5, 11.));
  check_prefix(LogarithmicCoordinates1D<double>(41, 1e-3, 30., 5.));
  check_prefix(FocusedLogCoordinates1D<double>(64, 5e-3, 250., 1., 2.));
  check_prefix(FocusedLogCoordinates1D<double>(50, 1e-4, 100., 0.3, 0.)); // pure-log branch
  check_prefix(LinearCoordinates1D<float>(20, 0.f, 1.f));
  check_prefix(LogarithmicCoordinates1D<float>(25, 1e-3f, 10.f, 3.f));
  check_prefix(FocusedLogCoordinates1D<float>(30, 1e-3f, 10.f, 1.f, 1.f));
}

TEST_CASE("Coordinates prefix(): identity keys", "[1D][coordinates][prefix]")
{
  // A grid that is not a prefix renders exactly as before -- this string is the key of the
  // momentum-grid datasets in existing HDF5 outputs.
  CHECK(FocusedLogCoordinates1D<double>(64, 0.005, 250., 1., 2.).to_string() ==
        "FocusedLogCoordinates1D(64, 0.005, 250, 1, 2)");

  // A prefix is neither its parent nor a genuine grid with its point count. For
  // LogarithmicCoordinates1D the latter even has different points (forward() divides the index by
  // the point count), so a shared key would hand one of them the other's cached positions.
  const LogarithmicCoordinates1D<double> L(40, 1e-3, 30., 5.);
  const auto Lp = L.prefix(20);
  const LogarithmicCoordinates1D<double> L20(20, 1e-3, 30., 5.);
  CHECK(Lp.to_string() != L20.to_string());
  CHECK(!(Lp == L20));
  CHECK(Lp.forward(10) != L20.forward(10));

  const FocusedLogCoordinates1D<double> F(64, 0.005, 250., 1., 2.);
  const auto Fp = F.prefix(40);
  const FocusedLogCoordinates1D<double> F40(40, 0.005, Fp.forward(39), 1., 2.);
  CHECK(Fp.to_string() != F40.to_string());
  CHECK(!(Fp == F40));
}

TEST_CASE("Coordinates prefix(): conversion between number types keeps the prefix", "[1D][coordinates][prefix]")
{
  const FocusedLogCoordinates1D<double> F(64, 0.005, 250., 1., 2.);
  const FocusedLogCoordinates1D<float> Ff(F.prefix(25));
  CHECK(Ff.size() == 25);
  CHECK(Ff.parent_size() == 64);
  CHECK(Ff == FocusedLogCoordinates1D<float>(F).prefix(25));
  for (size_t i = 0; i < 25; ++i)
    CHECK(is_close(double(Ff.forward(i)), F.forward(i), 1e-5 * F.forward(i)));

  const LogarithmicCoordinates1D<double> L(40, 1e-3, 30., 5.);
  const LogarithmicCoordinates1D<float> Lf(L.prefix(17));
  CHECK(Lf.size() == 17);
  CHECK(Lf.parent_size() == 40);
  for (size_t i = 0; i < 17; ++i)
    CHECK(is_close(double(Lf.forward(i)), L.forward(i), 1e-5 * (1. + L.forward(i))));

  const LinearCoordinates1D<double> Li(30, -1., 2.);
  const LinearCoordinates1D<float> Lif(Li.prefix(9));
  CHECK(Lif.size() == 9);
  CHECK(Lif.parent_size() == 30);
  CHECK(is_close(double(Lif.forward(8)), Li.forward(8), 1e-6));
}

TEST_CASE("Test 1D Linear coordinates", "[1D][coordinates]")
{
  const float p_start = GENERATE(take(3, random(1e-6, 1e-1)));
  const float p_stop = GENERATE(take(3, random(1., 100.))) + p_start;
  const int p_size = GENERATE(take(3, random(10, 100)));

  LinearCoordinates1D<float> coords(p_size, p_start, p_stop);

  const auto grid = make_grid(coords);
  CHECK(is_close(grid[0][0], p_start, 1e-8f));
  for (uint i = 1; i < grid.size() - 1; ++i)
    CHECK(grid[i][0] > grid[i - 1][0]);
  CHECK(is_close(grid[grid.size() - 1][0], p_stop, 1e-5f));
}

TEST_CASE("Test 1D Linear periodic coordinates", "[1D][coordinates][periodic]")
{
  const double p_start = GENERATE(take(3, random(-10., 10.)));
  const double p_period = GENERATE(take(3, random(1., 100.)));
  const int p_size = GENERATE(take(3, random(2, 100)));
  const double p_stop = p_start + p_period;
  const double dx = p_period / p_size;

  LinearPeriodicCoordinates1D<double> coords(p_size, p_start, p_stop);

  const auto grid = make_grid(coords);
  REQUIRE(grid.size() == size_t(p_size));

  // The endpoint is excluded: the grid runs start, start + dx, ..., stop - dx
  CHECK(is_close(grid[0][0], p_start, 1e-10 * p_period));
  for (uint i = 1; i < grid.size(); ++i)
    CHECK(grid[i][0] > grid[i - 1][0]);
  CHECK(is_close(grid[grid.size() - 1][0], p_stop - dx, 1e-10 * p_period));

  // backward is the inverse of forward on the grid points
  for (int i = 0; i < p_size; ++i)
    CHECK(is_close(coords.backward(coords.forward(i)), double(i), 1e-8));

  // and it folds everything else back into [0, p_size). Note that right at the seam the fold may land on either
  // side of it, so compare indices as points on a circle of circumference p_size.
  const auto circular_distance = [&](const double a, const double b) {
    const double d = std::abs(a - b);
    return std::min(d, double(p_size) - d);
  };
  CHECK(circular_distance(coords.backward(p_stop), 0.) < 1e-8);
  CHECK(circular_distance(coords.backward(p_start - dx), double(p_size - 1)) < 1e-8);
  CHECK(circular_distance(coords.backward(p_start + 3 * p_period + 2 * dx), 2.) < 1e-8);
  for (const double y : {p_start - 7 * p_period, p_start + 11 * p_period, p_stop - 1e-12 * p_period}) {
    const double idx = coords.backward(y);
    CHECK(idx >= 0.);
    CHECK(idx < double(p_size));
  }
}

TEST_CASE("Test 2D CoordinatePackND with two LogarithmicCoordinates1D", "[2D][coordinates]")
{
  const float p_start_1 = GENERATE(take(2, random(1e-6, 1e-1)));
  const float p_stop_1 = GENERATE(take(2, random(1., 100.))) + p_start_1;
  const int p_size_1 = GENERATE(take(2, random(10, 100)));
  const float p_bias_1 = GENERATE(take(2, random(1., 10.)));
  LogarithmicCoordinates1D<float> coords1(p_size_1, p_start_1, p_stop_1, p_bias_1);

  const float p_start_2 = GENERATE(take(2, random(1e-6, 1e-1)));
  const float p_stop_2 = GENERATE(take(2, random(1., 100.))) + p_start_2;
  const int p_size_2 = GENERATE(take(2, random(10, 100)));
  const float p_bias_2 = GENERATE(take(2, random(1., 10.)));
  LogarithmicCoordinates1D<float> coords2(p_size_2, p_start_2, p_stop_2, p_bias_2);

  CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>> coords(coords1, coords2);

  const auto grid = make_grid(coords);

  // Check the boundaries
  for (uint i = 0; i < coords2.size(); ++i) {
    // lower boundary
    if (!is_close(grid[i][1], coords2.forward(i), 1e-5))
      std::cout << "grid[i][1]: " << grid[i][1] << " coords2.forward(i): " << coords2.forward(i) << std::endl;
    CHECK(is_close(grid[i][1], coords2.forward(i), 1e-5));

    if (!is_close(grid[i][0], coords1.forward(0), 1e-5))
      std::cout << "grid[i][0]: " << grid[i][0] << " coords1.forward(0): " << coords1.forward(0) << std::endl;
    CHECK(is_close(grid[i][0], coords1.forward(0), 1e-5));

    if (!is_close(p_start_1, coords1.forward(0), 1e-5))
      std::cout << "p_start_1: " << p_start_1 << " coords1.forward(0): " << coords1.forward(0) << std::endl;
    CHECK(is_close(p_start_1, coords1.forward(0), 1e-5));

    // upper boundary
    if (!is_close(grid[(coords1.size() - 1) * coords2.size() + i][1], coords2.forward(i), 1e-5))
      std::cout << "grid[(coords1.size() - 1) * coords2.size() + i][1]: "
                << grid[(coords1.size() - 1) * coords2.size() + i][1] << " coords2.forward(i): " << coords2.forward(i)
                << std::endl;
    CHECK(is_close(grid[(coords1.size() - 1) * coords2.size() + i][1], coords2.forward(i), 1e-5));

    if (!is_close(grid[(coords1.size() - 1) * coords2.size() + i][0], coords1.forward(coords1.size() - 1), 1e-5))
      std::cout << "grid[(coords1.size() - 1) * coords2.size() + i][0]: "
                << grid[(coords1.size() - 1) * coords2.size() + i][0]
                << " coords1.forward(coords1.size() - 1): " << coords1.forward(coords1.size() - 1) << std::endl;
    CHECK(is_close(grid[(coords1.size() - 1) * coords2.size() + i][0], coords1.forward(coords1.size() - 1), 1e-5));

    if (!is_close(p_stop_1, coords1.forward(p_size_1 - 1), 1e-5))
      std::cout << "p_stop_1: " << p_stop_1 << " coords1.forward(p_size_1 - 1): " << coords1.forward(p_size_1 - 1)
                << std::endl;
    CHECK(is_close(p_stop_1, coords1.forward(p_size_1 - 1), 1e-5));
  }
}