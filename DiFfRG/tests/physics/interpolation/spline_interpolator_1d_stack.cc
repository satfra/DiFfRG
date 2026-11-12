#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

TEST_CASE("Test template constraints for SplineInterpolator1DStack", "[interpolator]")
{
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<>>>);
  STATIC_REQUIRE(
      is_interpolator<SplineInterpolator1DStack<complex<double>, BosonicCoordinates1DFiniteT<>>>);
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1DStack<autodiff::real, BosonicCoordinates1DFiniteT<>>>);
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1DStack<cxreal, BosonicCoordinates1DFiniteT<>>>);

  STATIC_REQUIRE(is_interpolator<SplineInterpolator1DStack<double, FermionicCoordinates1DFiniteT<>>>);
  STATIC_REQUIRE(
      is_interpolator<SplineInterpolator1DStack<complex<double>, FermionicCoordinates1DFiniteT<>>>);
  STATIC_REQUIRE(
      is_interpolator<SplineInterpolator1DStack<autodiff::real, FermionicCoordinates1DFiniteT<>>>);
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1DStack<cxreal, FermionicCoordinates1DFiniteT<>>>);

  STATIC_REQUIRE(!is_interpolator<int>);
  STATIC_REQUIRE(!is_interpolator<std::array<double, 3>>);
}

TEMPLATE_TEST_CASE("Test 1D spline interpolation stack", "[float][double][complex][autodiff]", double, complex<double>,
                   autodiff::real, cxreal)
{
  using Type = TestType;
  DiFfRG::Init();

  using ctype = typename get_type::ctype<Type>;

  const int m_start = GENERATE(take(2, random(-50, 5)));
  const int m_size = GENERATE(take(2, random(2, 64)));
  const int m_stop = m_start + m_size;

  const ctype T = GENERATE(take(2, random(1e-6, 1e-1)));

  const ctype p_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p_stop = GENERATE(take(2, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(2, random(10, 100)));
  const ctype p_bias = GENERATE(take(2, random(1., 10.)));

  std::vector<Type> in_data(m_size * p_size);
  for (int i = 0; i < m_size; ++i)
    for (int j = 0; j < p_size; ++j)
      in_data[i * p_size + j] = i * p_size + j;
  BosonicCoordinates1DFiniteT<int, ctype> coords(m_start, m_stop, T, p_size, p_start, p_stop, p_bias);
  SplineInterpolator1DStack<Type, BosonicCoordinates1DFiniteT<int, ctype>> interpolator(coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(2, random(2, 200)));
  const ctype m_pt = (m_start + int(GENERATE(take(2, random(0., 1.))) * (m_stop - m_start - 0.5))) * 2. * M_PI * T;
  const ctype p_pt = (p_start + GENERATE(take(2, random(0., 1.))) * (p_stop - p_start));

  const auto res_host = interpolator(m_pt, p_pt) * ctype(n_el);

  auto [m_frac, p_idx] = coords.backward(m_pt, p_pt);
  const size_t m_idx = size_t(std::lround(m_frac)); // m_pt sits on a Matsubara value

  p_idx = std::max(ctype(0), std::min(p_idx, ctype(p_size)));
  const auto res_local = (in_data[m_idx * p_size + std::floor(p_idx)] +
                          (p_idx - std::floor(p_idx)) * (in_data[m_idx * p_size + std::ceil(p_idx)] -
                                                         in_data[m_idx * p_size + std::floor(p_idx)])) *
                         ctype(n_el);

  if (!is_close(res_host, res_local, 1e-6 * n_el))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, 1e-6 * n_el));
}

TEMPLATE_TEST_CASE("Test 1D spline interpolation stack GPU", "[float][double][complex][autodiff]", double,
                   complex<double>, autodiff::real, cxreal)
{
  using Type = TestType;
  DiFfRG::Init();

  using ctype = typename get_type::ctype<Type>;

  const int m_start = GENERATE(take(2, random(-50, 5)));
  const int m_size = GENERATE(take(2, random(2, 64)));
  const int m_stop = m_start + m_size;

  const ctype T = GENERATE(take(2, random(1e-6, 1e-1)));

  const ctype p_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p_stop = GENERATE(take(2, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(2, random(10, 100)));
  const ctype p_bias = GENERATE(take(2, random(1., 10.)));

  std::vector<Type> in_data(m_size * p_size);
  for (int i = 0; i < m_size; ++i)
    for (int j = 0; j < p_size; ++j)
      in_data[i * p_size + j] = i * p_size + j;
  BosonicCoordinates1DFiniteT<int, ctype> coords(m_start, m_stop, T, p_size, p_start, p_stop, p_bias);
  SplineInterpolator1DStack<Type, BosonicCoordinates1DFiniteT<int, ctype>> interpolator(coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(2, random(2, 200)));
  const ctype m_pt = (m_start + int(GENERATE(take(2, random(0., 1.))) * (m_stop - m_start - 0.5))) * 2. * M_PI * T;
  const ctype p_pt = (p_start + GENERATE(take(2, random(0., 1.))) * (p_stop - p_start));

  const auto res_host = interpolator(m_pt, p_pt) * ctype(n_el);

  auto [m_frac, p_idx] = coords.backward(m_pt, p_pt);
  const size_t m_idx = size_t(std::lround(m_frac)); // m_pt sits on a Matsubara value

  p_idx = std::max(ctype(0), std::min(p_idx, ctype(p_size)));
  const auto res_local = (in_data[m_idx * p_size + std::floor(p_idx)] +
                          (p_idx - std::floor(p_idx)) * (in_data[m_idx * p_size + std::ceil(p_idx)] -
                                                         in_data[m_idx * p_size + std::floor(p_idx)])) *
                         ctype(n_el);

  Type res_gpu;
  Kokkos::parallel_reduce(
      "Get one point", Kokkos::RangePolicy(0, n_el),
      KOKKOS_LAMBDA(const uint, Type &update) { update += interpolator(m_pt, p_pt); }, res_gpu);

  if (!is_close(res_host, res_local, 1e-6 * n_el))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  if (!is_close(res_gpu, res_local, 1e-6 * n_el))
    std::cout << "gpu: " << res_gpu << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, 1e-6 * n_el));
  CHECK(is_close(res_gpu, res_local, 1e-6 * n_el));
}
TEST_CASE("Test 1D spline interpolation stack between Matsubara rows", "[interpolator]")
{
  DiFfRG::Init();

  // Rows constant in p, so the frequency dependence is the only thing interpolated.
  const int m_size = 4, p_size = 16;
  const double T = 0.01;
  const std::vector<double> rows{1., 3., 4., 8.};
  std::vector<double> in_data(m_size * p_size);
  for (int i = 0; i < m_size; ++i)
    for (int j = 0; j < p_size; ++j)
      in_data[i * p_size + j] = rows[i];
  BosonicCoordinates1DFiniteT<int, double> coords(0, m_size, T, p_size, 1e-2, 6., 2.);
  SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double>> interpolator(coords);
  interpolator.update(in_data.data());

  const double w = 2. * M_PI * T;
  const double p = 0.1;
  // exact on the rows
  for (int i = 0; i < m_size; ++i)
    CHECK(is_close(interpolator(i * w, p), rows[i], 1e-12));
  // cubic in between, not linear: the linear join would give 1.5 and 3.5
  CHECK(!is_close(interpolator(0.25 * w, p), 1.5, 1e-3));
  CHECK(!is_close(interpolator(1.5 * w, p), 3.5, 1e-3));
  // rows constant in p stay constant in p between the rows
  CHECK(is_close(interpolator(1.5 * w, 0.02), interpolator(1.5 * w, 3.), 1e-12));
  // a stack starting at n = 0 is even in the frequency, and smooth through zero: zero slope there
  CHECK(is_close(interpolator(-1.5 * w, p), interpolator(1.5 * w, p), 1e-12));
  CHECK(is_close(interpolator(-2. * w, p), rows[2], 1e-12));
  const double eps = 1e-4;
  CHECK(std::abs(interpolator(eps * w, p) - rows[0]) < 10. * eps * eps * (rows[1] - rows[0]));
}

/**
 * @brief A frequency sum evaluated BETWEEN the Matsubara rows must reproduce the sum over the rows.
 *
 * This is what a finite-temperature flow does: its loops read dressings stored on the Matsubara rows at the nodes
 * of a Gauss rule for sums (Monien), which lie between the rows. The rows here sample an O(4)-symmetric dressing,
 * g = 1 / (p0^2 + p^2 + E^2)^2, so beyond the top row the stack's folding of the frequency into the momentum is
 * exact, and the reference is the sum of g over the Matsubara frequencies itself. With the rows joined linearly the
 * summand had a kink at every row, and the Gauss rule, built for smooth summands, missed the sum by an amount that
 * more nodes did not reduce; the cubic join makes the summand smooth.
 */
TEST_CASE("Gauss sum over an interpolated stack reproduces the Matsubara sum", "[interpolator]")
{
  DiFfRG::Init();

  const double T = 0.01, E = GENERATE(0.05, 0.3), p = GENERATE(0.02, 0.4);
  // The momentum grid reaches far enough (60 GeV) that the clamped value beyond it, ~1e-7, does not matter, and is
  // dense where g varies (bias 10 clusters it at small p).
  const int m_size = 24, p_size = 128;
  BosonicCoordinates1DFiniteT<int, double> coords(0, m_size, T, p_size, 1e-3, 60., 10.);
  const auto g = [&](const double p0, const double q) { return 1. / powr<2>(powr<2>(p0) + powr<2>(q) + powr<2>(E)); };

  std::vector<double> in_data(m_size * p_size);
  for (int i = 0; i < m_size; ++i)
    for (int j = 0; j < p_size; ++j) {
      const auto pt = coords.forward(size_t(i), size_t(j));
      in_data[i * p_size + j] = g(pt[0], pt[1]);
    }
  SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double>> interpolator(coords);
  interpolator.update(in_data.data());

  // The reference: the sum of g itself over the Matsubara frequencies.
  double ref = T * g(0., p);
  for (int n = 1; n < 200000; ++n)
    ref += 2. * T * g(2. * M_PI * T * n, p);

  // The Gauss rule for sums as the integrators size it: reaching 200 times the summand's scale.
  MatsubaraQuadrature<double> rule(T, std::sqrt(powr<2>(p) + powr<2>(E)), 2, 8, 1024, 64, 1.);
  const double got = rule.sum([&](const double p0) { return interpolator(p0, p); });

  const double rel_err = std::abs(got / ref - 1.);
  INFO("E = " << E << ", p = " << p << ": relative error " << rel_err << " with " << rule.size() << " nodes");
  std::cout << "E = " << E << ", p = " << p << ": relative error " << rel_err << " with " << rule.size() << " nodes"
            << std::endl;
  CHECK(rel_err < 1e-5);
}

TEST_CASE("Test stack row-major element access", "[interpolator]")
{
  DiFfRG::Init();

  // See LinearInterpolator2D: row-major in, row-major out, over a LayoutLeft mirror.
  constexpr int m_start = -2, m_stop = 3; // 5 stack entries
  constexpr size_t n_p = 7;
  const double T = 1e-2;

  std::vector<double> in_data((m_stop - m_start) * n_p);
  for (size_t i = 0; i < in_data.size(); ++i)
    in_data[i] = double(i);

  BosonicCoordinates1DFiniteT<int, double> coords(m_start, m_stop, T, n_p, 1e-3, 10., 2.);
  SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double>> interpolator(coords);
  interpolator.update(in_data.data());

  for (size_t i = 0; i < in_data.size(); ++i)
    CHECK(interpolator[i] == in_data[i]);
}

TEST_CASE("Clamped spline stack reproduces a cubic on each row", "[interpolator]")
{
  DiFfRG::Init();

  // Rows differ by a constant, so all share the end slopes passed to update().
  const auto f = [](double x) { return x * x * x - 2. * x * x; };
  const int m_size = 3;
  const size_t n = 11;
  const double T = 0.1;
  std::vector<double> in_data(m_size * n);
  for (int i = 0; i < m_size; ++i)
    for (size_t j = 0; j < n; ++j)
      in_data[i * n + j] = f(double(j)) + i;
  LinearCoordinates1D<double> radial(n, 0., double(n - 1));
  BosonicCoordinates1DFiniteT<int, double, LinearCoordinates1D<double>> coords(radial, 0, m_size, T);
  SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LinearCoordinates1D<double>>>
      interpolator(coords);
  interpolator.update(in_data.data(), 0., 3. * 100. - 4. * 10.);

  const double w = 2. * M_PI * T;
  for (int i = 0; i < m_size; ++i)
    for (const double x : {0.25, 3.25, 6.7, 9.9})
      CHECK(is_close(interpolator(i * w, x), f(x) + i, 1e-10));
}
