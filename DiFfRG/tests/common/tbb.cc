#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/tbb.hh>
#include <DiFfRG/common/types.hh>

using namespace DiFfRG;

namespace
{
  void check_complex_approx(const complex<double> actual, const complex<double> expected)
  {
    CHECK(real(actual) == Catch::Approx(real(expected)));
    CHECK(imag(actual) == Catch::Approx(imag(expected)));
  }
} // namespace

static_assert(std::is_same_v<get_type::ctype<autodiff::Real<2, float>>, float>);
static_assert(std::is_same_v<get_type::ctype<autodiff::Real<2, double>>, double>);
static_assert(std::is_same_v<get_type::ctype<autodiff::Real<2, complex<float>>>, float>);
static_assert(std::is_same_v<get_type::ctype<autodiff::Real<2, complex<double>>>, double>);

// Test TBBReduction up to 9 dimensions with random values
TEMPLATE_TEST_CASE_SIG("Test TBB reducer", "[integration][tbb]", ((int dim), dim), (1), (2), (3), (4), (5), (6), (7),
                       (8), (9))
{
  const auto val = GENERATE(take(10, random(-1000., 1000.)));
  auto functor = [val](const auto &) { return val; };
  device::array<size_t, dim> grid_size;
  for (size_t i = 0; i < dim; ++i) {
    grid_size[i] = GENERATE(take(1, random(1, 10)));
  }
  double res = TBBReduction<dim, double, decltype(functor)>(grid_size, functor);
  double expected = val;
  for (size_t i = 0; i < dim; ++i) {
    expected *= grid_size[i];
  }
  CHECK(res == Catch::Approx(expected).epsilon(1e-6));
}

TEST_CASE("Test TBB reducer with second-order complex autodiff", "[integration][tbb][autodiff]")
{
  using NT = cxReal<2, double>;

  constexpr int dim = 3;
  device::array<size_t, dim> grid_size{{2, 3, 4}};

  const NT value(std::array<complex<double>, 3>{
      complex<double>(1.25, -0.5), complex<double>(2.0, 0.25), complex<double>(-1.5, 0.75)});

  auto functor = [value](const auto &) { return value; };
  const NT res = TBBReduction<dim, NT, decltype(functor)>(grid_size, functor);

  double entries = 1.;
  for (const auto size : grid_size)
    entries *= size;

  check_complex_approx(autodiff::val(res), entries * autodiff::val(value));
  check_complex_approx(autodiff::derivative<1>(res), entries * autodiff::derivative<1>(value));
  check_complex_approx(autodiff::derivative<2>(res), entries * autodiff::derivative<2>(value));
}
