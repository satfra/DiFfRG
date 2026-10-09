#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>

#include <Kokkos_Random.hpp>

using namespace DiFfRG;

constexpr double eps = 100 * std::numeric_limits<double>::epsilon();

TEST_CASE("Test autodiff with Kokkos", "[autodiff][kokkos]")
{
  DiFfRG::Init();

  auto validate = [](const auto &x, const auto &value, const auto &derivative, std::string name = "") {
    // check the value
    if (!is_close(autodiff::val(x), value, eps)) {
      std::cout << "is: autodiff::val(" << name << ") = " << autodiff::val(x) << std::endl;
      std::cout << "should: value = " << value << std::endl;
    }
    CHECK(is_close(autodiff::val(x), value, eps));

    // check the derivative
    if (!is_close(autodiff::derivative(x), derivative, eps)) {
      std::cout << "is: autodiff::derivative(" << name << ") = " << autodiff::derivative(x) << std::endl;
      std::cout << "should: derivative = " << derivative << std::endl;
    }
    CHECK(is_close(autodiff::derivative(x), derivative, 10 * eps));
  };

  // Take a derivative of a matrix multiplication
  // A: N x M, v: M x 1, result: N x 1
  const long N = GENERATE(take(4, random(powr<4>(2), powr<9>(2))));
  const long M = GENERATE(take(4, random(powr<4>(2), powr<9>(2))));

  SECTION("GPU")
  {
    Kokkos::View<double **, GPU_memory> A("A", N, M);
    Kokkos::View<autodiff::real *, GPU_memory> v("v", M);
    Kokkos::View<autodiff::real *, GPU_memory> result("result", N);

    Kokkos::Random_XorShift64_Pool<GPU_exec> random_pool(/*seed=*/12345);

    // Fill A and v with random values
    Kokkos::parallel_for(
        "Fill A", Kokkos::MDRangePolicy<GPU_exec, Kokkos::Rank<2>>({0, 0}, {N, M}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          A(i, j) = generator.drand(0., 1.);

          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });

    Kokkos::parallel_for(
        "Fill v", Kokkos::RangePolicy<GPU_exec>(0, M), KOKKOS_LAMBDA(const size_t i) {
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          v(i) = autodiff::real({generator.drand(0., 1.), 1.0});

          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });

    Kokkos::fence();

    // Compute the matrix-vector product
    Kokkos::parallel_for(
        "Matrix-vector multiplication", Kokkos::RangePolicy<GPU_exec>(0, N), KOKKOS_LAMBDA(const size_t i) {
          autodiff::real temp{};
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
      double expected_value = 0;
      double expected_derivative = 0;
      for (long j = 0; j < M; ++j) {
        expected_value += A_h(i, j) * autodiff::val(v_h(j));
        expected_derivative += A_h(i, j);
      }
      validate(result_h(i), expected_value, expected_derivative, "result(" + std::to_string(i) + ")");
    }
  }

  SECTION("Threads")
  {
    Kokkos::View<double **, KokkosHost_memory> A("A", N, M);
    Kokkos::View<autodiff::real *, KokkosHost_memory> v("v", M);
    Kokkos::View<autodiff::real *, KokkosHost_memory> result("result", N);

    Kokkos::Random_XorShift64_Pool<KokkosHost_exec> random_pool(/*seed=*/12345);

    // Fill A and v with random values
    Kokkos::parallel_for(
        "Fill A", Kokkos::MDRangePolicy<KokkosHost_exec, Kokkos::Rank<2>>({0, 0}, {N, M}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          A(i, j) = generator.drand(0., 1.);

          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });

    Kokkos::parallel_for(
        "Fill v", Kokkos::RangePolicy<KokkosHost_exec>(0, M), KOKKOS_LAMBDA(const size_t i) {
          // acquire the state of the random number generator engine
          auto generator = random_pool.get_state();

          v(i) = autodiff::real({generator.drand(0., 1.), 1.0});

          // do not forget to release the state of the engine
          random_pool.free_state(generator);
        });

    Kokkos::fence();

    // Compute the matrix-vector product
    Kokkos::parallel_for(
        "Matrix-vector multiplication", Kokkos::RangePolicy<KokkosHost_exec>(0, N), KOKKOS_LAMBDA(const size_t i) {
          autodiff::real temp{};
          for (long j = 0; j < M; ++j) {
            temp += A(i, j) * v(j);
          }
          result(i) = temp;
        });

    Kokkos::fence();

    // Check the result
    for (long i = 0; i < N; ++i) {
      double expected_value = 0;
      double expected_derivative = 0;
      for (long j = 0; j < M; ++j) {
        expected_value += A(i, j) * autodiff::val(v(j));
        expected_derivative += A(i, j);
      }
      validate(result(i), expected_value, expected_derivative, "result(" + std::to_string(i) + ")");
    }
  }
}
