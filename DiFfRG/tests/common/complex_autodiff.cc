#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>

#include <Kokkos_Random.hpp>

using namespace DiFfRG;

TEST_CASE("Test autodiff for complex numbers", "[autodiff][complex]")
{
  constexpr double eps = 10 * std::numeric_limits<double>::epsilon();
  auto validate_no_ad = [](const auto &x, const auto &value, std::string name = "") {
    // check the real part of the value
    if (!is_close(real(x), real(value), eps)) {
      std::cout << "is: real(" << name << ") = " << real(x) << std::endl;
      std::cout << "should: real(value) = " << real(value) << std::endl;
    }
    CHECK(is_close(real(x), real(value), eps));

    // check the imaginary part of the value
    if (!is_close(imag(x), imag(value), eps)) {
      std::cout << "is: imag(" << name << ") = " << imag(x) << std::endl;
      std::cout << "should: imag(value) = " << imag(value) << std::endl;
    }
    CHECK(is_close(imag(x), imag(value), eps));
  };
  auto validate = [](const auto &x, const auto &value, const auto &derivative, std::string name = "") {
    // check the real part of the value
    if (!is_close(real(autodiff::val(x)), real(value), eps)) {
      std::cout << "is: real(autodiff::val(" << name << ")) = " << real(autodiff::val(x)) << std::endl;
      std::cout << "should: real(value) = " << real(value) << std::endl;
    }
    CHECK(is_close(real(autodiff::val(x)), real(value), eps));

    // check the imaginary part of the value
    if (!is_close(imag(autodiff::val(x)), imag(value), eps)) {
      std::cout << "is: imag(autodiff::val(" << name << ")) = " << imag(autodiff::val(x)) << std::endl;
      std::cout << "should: imag(value) = " << imag(value) << std::endl;
    }
    CHECK(is_close(imag(autodiff::val(x)), imag(value), eps));

    // check the real part of the derivative
    if (!is_close(real(autodiff::derivative(x)), real(derivative), eps)) {
      std::cout << "is: real(autodiff::derivative(" << name << ")) = " << real(autodiff::derivative(x)) << std::endl;
      std::cout << "should: real(derivative) = " << real(derivative) << std::endl;
    }
    CHECK(is_close(real(autodiff::derivative(x)), real(derivative), 10 * eps));

    // check the imaginary part of the derivative
    if (!is_close(imag(autodiff::derivative(x)), imag(derivative), 10 * eps)) {
      std::cout << "is: imag(autodiff::derivative(" << name << ")) = " << imag(autodiff::derivative(x)) << std::endl;
      std::cout << "should: imag(derivative) = " << imag(derivative) << std::endl;
    }
    CHECK(is_close(imag(autodiff::derivative(x)), imag(derivative), 10 * eps));
  };

  const double x = 2.0;
  const autodiff::real ad_x(std::array<double, 2>{{3.0, 5.0}});
  const complex<double> c_x(2.0, 3.0);
  const cxreal ad_c_x(std::array<complex<double>, 2>{{complex<double>(3.0, 2.0), complex<double>(5.0, -5.0)}});

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

TEST_CASE("Test autodiff with Kokkos", "[autodiff][kokkos]")
{
  constexpr double eps = 2000 * std::numeric_limits<double>::epsilon();
  DiFfRG::Init();

  auto validate = [](const auto &x, const auto &value, const auto &derivative, std::string name = "") {
    // check the real part of the value
    if (!is_close(real(autodiff::val(x)), real(value), eps)) {
      std::cout << "is: real(autodiff::val(" << name << ")) = " << real(autodiff::val(x)) << std::endl;
      std::cout << "should: real(value) = " << real(value) << std::endl;
    }
    CHECK(is_close(real(autodiff::val(x)), real(value), eps));

    // check the imaginary part of the value
    if (!is_close(imag(autodiff::val(x)), imag(value), eps)) {
      std::cout << "is: imag(autodiff::val(" << name << ")) = " << imag(autodiff::val(x)) << std::endl;
      std::cout << "should: imag(value) = " << imag(value) << std::endl;
    }
    CHECK(is_close(imag(autodiff::val(x)), imag(value), eps));

    // check the real part of the derivative
    if (!is_close(real(autodiff::derivative(x)), real(derivative), eps)) {
      std::cout << "is: real(autodiff::derivative(" << name << ")) = " << real(autodiff::derivative(x)) << std::endl;
      std::cout << "should: real(derivative) = " << real(derivative) << std::endl;
    }
    CHECK(is_close(real(autodiff::derivative(x)), real(derivative), eps));

    // check the imaginary part of the derivative
    if (!is_close(imag(autodiff::derivative(x)), imag(derivative), eps)) {
      std::cout << "is: imag(autodiff::derivative(" << name << ")) = " << imag(autodiff::derivative(x)) << std::endl;
      std::cout << "should: imag(derivative) = " << imag(derivative) << std::endl;
    }
    CHECK(is_close(imag(autodiff::derivative(x)), imag(derivative), eps));
  };

  // Take a derivative of a matrix multiplication
  // A: N x M, v: M x 1, result: N x 1
  const long N = GENERATE(take(4, random(powr<4>(2), powr<9>(2))));
  const long M = GENERATE(take(4, random(powr<4>(2), powr<9>(2))));

  SECTION("GPU")
  {
    Kokkos::View<complex<double> **, GPU_memory> A("A", N, M);
    Kokkos::View<cxreal *, GPU_memory> v("v", M);
    Kokkos::View<cxreal *, GPU_memory> result("result", N);

    Kokkos::Random_XorShift64_Pool<GPU_exec> random_pool(/*seed=*/12345);

    // Fill A and v with random values
    Kokkos::parallel_for(
        "Fill A", Kokkos::MDRangePolicy<GPU_exec, Kokkos::Rank<2>>({0, 0}, {N, M}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          A(i, j) = complex<double>(generator.drand(0., 1.), generator.drand(0., 1.));

          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });

    Kokkos::parallel_for(
        "Fill v", Kokkos::RangePolicy<GPU_exec>(0, M), KOKKOS_LAMBDA(const size_t i) {
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          v(i) = cxreal({complex<double>(generator.drand(0., 1.), generator.drand(0., 1.)), complex<double>(1., 0.)});

          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });

    Kokkos::fence();

    // Compute the matrix-vector product
    Kokkos::parallel_for(
        "Matrix-vector multiplication", Kokkos::RangePolicy<GPU_exec>(0, N), KOKKOS_LAMBDA(const size_t i) {
          cxreal temp{};
          for (long j = 0; j < M; ++j) {
            temp += A(i, j) * v(j);
          }
          result(i) = temp;
        });

    Kokkos::fence();

    // make host mirrors
    auto A_h = Kokkos::create_mirror_view(A);
    auto v_h = Kokkos::create_mirror_view(v);
    auto result_h = Kokkos::create_mirror_view(result);

    // copy data to host
    Kokkos::deep_copy(A_h, A);
    Kokkos::deep_copy(v_h, v);
    Kokkos::deep_copy(result_h, result);

    // Check the result
    for (long i = 0; i < N; ++i) {
      complex<double> expected_value = 0;
      complex<double> expected_derivative = 0;
      for (long j = 0; j < M; ++j) {
        expected_value += A_h(i, j) * autodiff::val(v_h(j));
        expected_derivative += A_h(i, j);
      }
      validate(result_h(i), expected_value, expected_derivative, "result(" + std::to_string(i) + ")");
    }
  }

  SECTION("Threads")
  {
    Kokkos::View<complex<double> **, KokkosHost_memory> A("A", N, M);
    Kokkos::View<cxreal *, KokkosHost_memory> v("v", M);
    Kokkos::View<cxreal *, KokkosHost_memory> result("result", N);

    Kokkos::Random_XorShift64_Pool<KokkosHost_exec> random_pool(/*seed=*/12345);

    // Fill A and v with random values
    Kokkos::parallel_for(
        "Fill A", Kokkos::MDRangePolicy<KokkosHost_exec, Kokkos::Rank<2>>({0, 0}, {N, M}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          A(i, j) = complex<double>(generator.drand(0., 1.), generator.drand(0., 1.));

          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });

    Kokkos::parallel_for(
        "Fill v", Kokkos::RangePolicy<KokkosHost_exec>(0, M), KOKKOS_LAMBDA(const size_t i) {
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          v(i) = cxreal({complex<double>(generator.drand(0., 1.), generator.drand(0., 1.)), complex<double>(1., 0.)});

          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });

    Kokkos::fence();

    // Compute the matrix-vector product
    Kokkos::parallel_for(
        "Matrix-vector multiplication", Kokkos::RangePolicy<KokkosHost_exec>(0, N), KOKKOS_LAMBDA(const size_t i) {
          cxreal temp{};
          for (long j = 0; j < M; ++j) {
            temp += A(i, j) * v(j);
          }
          result(i) = temp;
        });

    Kokkos::fence();

    // Check the result
    for (long i = 0; i < N; ++i) {
      complex<double> expected_value = 0;
      complex<double> expected_derivative = 0;
      for (long j = 0; j < M; ++j) {
        expected_value += A(i, j) * autodiff::val(v(j));
        expected_derivative += A(i, j);
      }
      validate(result(i), expected_value, expected_derivative, "result(" + std::to_string(i) + ")");
    }
  }
}