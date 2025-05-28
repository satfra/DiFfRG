#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/math.hh>

using namespace DiFfRG;

TEST_CASE("Test autodiff on CPU", "[autodiff][cpu]")
{
  auto validate_no_ad = [](const auto &x, const auto &value, std::string name = "") {
    if (!is_close(real(x), real(value), std::numeric_limits<double>::epsilon())) {
      std::cout << "is: real(" << name << ") = " << real(x) << std::endl;
      std::cout << "should: real(value) = " << real(value) << std::endl;
    }
    CHECK(is_close(real(x), real(value), std::numeric_limits<double>::epsilon()));

    if (!is_close(imag(x), imag(value), std::numeric_limits<double>::epsilon())) {
      std::cout << "is: imag(" << name << ") = " << imag(x) << std::endl;
      std::cout << "should: imag(value) = " << imag(value) << std::endl;
    }
    CHECK(is_close(imag(x), imag(value), std::numeric_limits<double>::epsilon()));
  };
  auto validate = [](const auto &x, const auto &value, const auto &derivative, std::string name = "") {
    if (!is_close(real(autodiff::val(x)), real(value), std::numeric_limits<double>::epsilon())) {
      std::cout << "is: real(autodiff::val(" << name << ")) = " << real(autodiff::val(x)) << std::endl;
      std::cout << "should: real(value) = " << real(value) << std::endl;
    }
    CHECK(is_close(real(autodiff::val(x)), real(value), std::numeric_limits<double>::epsilon()));

    if (!is_close(imag(autodiff::val(x)), imag(value), std::numeric_limits<double>::epsilon())) {
      std::cout << "is: imag(autodiff::val(" << name << ")) = " << imag(autodiff::val(x)) << std::endl;
      std::cout << "should: imag(value) = " << imag(value) << std::endl;
    }
    CHECK(is_close(imag(autodiff::val(x)), imag(value), std::numeric_limits<double>::epsilon()));

    if (!is_close(real(autodiff::derivative(x)), real(derivative), 10. * std::numeric_limits<double>::epsilon())) {
      std::cout << "is: real(autodiff::derivative(" << name << ")) = " << real(autodiff::derivative(x)) << std::endl;
      std::cout << "should: real(derivative) = " << real(derivative) << std::endl;
    }
    CHECK(is_close(real(autodiff::derivative(x)), real(derivative), 10. * std::numeric_limits<double>::epsilon()));

    if (!is_close(imag(autodiff::derivative(x)), imag(derivative), std::numeric_limits<double>::epsilon())) {
      std::cout << "is: imag(autodiff::derivative(" << name << ")) = " << imag(autodiff::derivative(x)) << std::endl;
      std::cout << "should: imag(derivative) = " << imag(derivative) << std::endl;
    }
    CHECK(is_close(imag(autodiff::derivative(x)), imag(derivative), std::numeric_limits<double>::epsilon()));
  };

  const double x = 2.0;
  const autodiff::real ad_x(std::array<double, 2>{{3.0, 5.0}});
  const complex<double> c_x(2.0, 3.0);
  const cxReal ad_c_x(std::array<complex<double>, 2>{{complex<double>(3.0, 2.0), complex<double>(5.0, -5.0)}});

  SECTION("Multiplication")
  {
    // x, c_x
    validate_no_ad(c_x * x, complex<double>(4.0, 6.0), "c_x * x");
    validate_no_ad(x * c_x, complex<double>(4.0, 6.0), "x * c_x");

    // x, ad_x
    validate(ad_x * x, 6.0, 10.0, "ad_x * x");
    validate(x * ad_x, 6.0, 10.0, "x * ad_x");

    // x, ad_c_x
    validate(ad_c_x * x, complex<double>(6.0, 4.0), complex<double>(10.0, -10.0), "ad_c_x * x");
    validate(x * ad_c_x, complex<double>(6.0, 4.0), complex<double>(10.0, -10.0), "x * ad_c_x");

    // c_x, ad_x
    validate(ad_x * c_x, complex<double>(6.0, 9.0), complex<double>(10.0, 15.0), "ad_x * c_x");
    validate(c_x * ad_x, complex<double>(6.0, 9.0), complex<double>(10.0, 15.0), "c_x * ad_x");

    // c_x, ad_c_x
    validate(ad_c_x * c_x, complex<double>(0.0, 13.0), complex<double>(25.0, 5.0), "ad_c_x * c_x");
    validate(c_x * ad_c_x, complex<double>(0.0, 13.0), complex<double>(25.0, 5.0), "c_x * ad_c_x");

    // ad_x, ad_c_x
    validate(ad_x * ad_c_x, complex<double>(9.0, 6.0), complex<double>(30.0, -5.0), "ad_x * ad_c_x");
    validate(ad_c_x * ad_x, complex<double>(9.0, 6.0), complex<double>(30.0, -5.0), "ad_c_x * ad_x");

    // ad_c_x, ad_c_x
    validate(ad_c_x * ad_c_x, complex<double>(5.0, 12.0), complex<double>(50.0, -10.0), "ad_c_x * ad_c_x");
  }

  SECTION("Addition")
  {
    // x, c_x
    validate_no_ad(c_x + x, complex<double>(4.0, 3.0), "c_x + x");
    validate_no_ad(x + c_x, complex<double>(4.0, 3.0), "x + c_x");

    // x, ad_x
    validate(ad_x + x, 5.0, 5.0, "ad_x + x");
    validate(x + ad_x, 5.0, 5.0, "x + ad_x");

    // x, ad_c_x
    validate(ad_c_x + x, complex<double>(5.0, 2.0), complex<double>(5.0, -5.0), "ad_c_x + x");
    validate(x + ad_c_x, complex<double>(5.0, 2.0), complex<double>(5.0, -5.0), "x + ad_c_x");

    // c_x, ad_x
    validate(ad_x + c_x, complex<double>(5.0, 3.0), complex<double>(5.0, 0.0), "ad_x + c_x");
    validate(c_x + ad_x, complex<double>(5.0, 3.0), complex<double>(5.0, 0.0), "c_x + ad_x");

    // c_x, ad_c_x
    validate(ad_c_x + c_x, complex<double>(5.0, 5.0), complex<double>(5.0, -5.0), "ad_c_x + c_x");
    validate(c_x + ad_c_x, complex<double>(5.0, 5.0), complex<double>(5.0, -5.0), "c_x + ad_c_x");

    // ad_x, ad_c_x
    validate(ad_x + ad_c_x, complex<double>(6.0, 2.0), complex<double>(10.0, -5.0), "ad_x + ad_c_x");
    validate(ad_c_x + ad_x, complex<double>(6.0, 2.0), complex<double>(10.0, -5.0), "ad_c_x + ad_x");

    // ad_c_x, ad_c_x
    validate(ad_c_x + ad_c_x, complex<double>(6.0, 4.0), complex<double>(10.0, -10.0), "ad_c_x + ad_c_x");
  }

  SECTION("Subtraction")
  {
    // x, c_x
    validate_no_ad(c_x - x, complex<double>(0.0, 3.0), "c_x - x");
    validate_no_ad(x - c_x, complex<double>(0.0, -3.0), "x - c_x");

    // x, ad_x
    validate(ad_x - x, 1.0, 5.0, "ad_x - x");
    validate(x - ad_x, -1.0, -5.0, "x - ad_x");

    // x, ad_c_x
    validate(ad_c_x - x, complex<double>(1.0, 2.0), complex<double>(5.0, -5.0), "ad_c_x - x");
    validate(x - ad_c_x, complex<double>(-1.0, -2.0), complex<double>(-5.0, 5.0), "x - ad_c_x");

    // c_x, ad_x
    validate(ad_x - c_x, complex<double>(1.0, -3.0), complex<double>(5.0, 0.0), "ad_x - c_x");
    validate(c_x - ad_x, complex<double>(-1.0, 3.0), complex<double>(-5.0, 0.0), "c_x - ad_x");

    // c_x, ad_c_x
    validate(ad_c_x - c_x, complex<double>(1.0, -1.0), complex<double>(5.0, -5.0), "ad_c_x - c_x");
    validate(c_x - ad_c_x, complex<double>(-1.0, 1.0), complex<double>(-5.0, 5.0), "c_x - ad_c_x");

    // ad_x, ad_c_x
    validate(ad_x - ad_c_x, complex<double>(0.0, -2.0), complex<double>(0.0, 5.0), "ad_x - ad_c_x");
    validate(ad_c_x - ad_x, complex<double>(0.0, 2.0), complex<double>(0.0, -5.0), "ad_c_x - ad_x");

    // ad_c_x, ad_c_x
    validate(ad_c_x - ad_c_x, complex<double>(0., 0.), complex<double>(0., 0.), "ad_c_x - ad_c_x");
  }

  SECTION("Division")
  {
    // x, c_x
    validate_no_ad(c_x / x, complex<double>(1.0, 1.5), "c_x / x");
    validate_no_ad(x / c_x, complex<double>(4. / 13., -6. / 13.), "x / c_x");

    // x, ad_x
    validate(ad_x / x, 1.5, 5. / 2., "ad_x / x");
    validate(x / ad_x, 2. / 3., -10. / 9., "x / ad_x");

    // x, ad_c_x
    validate(ad_c_x / x, complex<double>(3. / 2., 1.), complex<double>(5. / 2., -5. / 2.), "ad_c_x / x");
    validate(x / ad_c_x, complex<double>(6. / 13., -4.0 / 13.), complex<double>(70. / 169., 170. / 169.), "x / ad_c_x");

    // c_x, ad_x
    validate(ad_x / c_x, complex<double>(6. / 13., -9. / 13.), complex<double>(10. / 13., -15. / 13.), "ad_x / c_x");
    validate(c_x / ad_x, complex<double>(2. / 3., 1.), complex<double>(-10. / 9., -5. / 3.), "c_x / ad_x");

    // c_x, ad_c_x
    validate(ad_c_x / c_x, complex<double>(12. / 13., -5. / 13.), complex<double>(-5. / 13., -25. / 13.),
             "ad_c_x / c_x");
    validate(c_x / ad_c_x, complex<double>(12. / 13., 5. / 13.), complex<double>(-185. / 169., 275. / 169.),
             "c_x / adc_x");

    // ad_x, ad_c_x
    validate(ad_x / ad_c_x, complex<double>(9. / 13., -6. / 13.), complex<double>(300. / 169., 125. / 169.),
             "ad_x / ad_c_x");
    validate(ad_c_x / ad_x, complex<double>(1., 2. / 3.), complex<double>(0.0, -25. / 9.), "ad_c_x / ad_x");

    // ad_c_x, ad_c_x
    validate(ad_c_x / ad_c_x, complex<double>(1., 0.), complex<double>(0., 0.), "ad_c_x / ad_c_x");
  }
}