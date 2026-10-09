#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/abstract_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <autodiff/forward/real.hpp>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <cstddef>

using NumberType = double;
using namespace dealii;

template <int dim>
using TVDReconstructorImpl = DiFfRG::def::TVDReconstructor<dim, DiFfRG::def::MinModLimiter, NumberType>;

// ──────────────────────────────────────────────────────────────────────
// Concept check
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVDReconstructor<dim, MinModLimiter> satisfies HasReconstructor concept", "[FV][reconstructor]")
{
  static_assert(DiFfRG::def::HasReconstructor<TVDReconstructorImpl<1>>);
  static_assert(DiFfRG::def::HasReconstructor<TVDReconstructorImpl<2>>);
}

// ──────────────────────────────────────────────────────────────────────
// 1D, one component (moved from KurganovTadmor_tests.cc)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation in 1D, one component", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 1;
  constexpr int n_faces = 2 * dim;
  using Reconstructor = TVDReconstructorImpl<dim>;
  const Point<dim> x_center(1.0);
  const std::array<NumberType, n_components> u_center = {2.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  std::array<std::array<NumberType, n_components>, n_faces> u_n;

  SECTION("Symmetric neighbours - full gradient")
  {
    u_n = {{{1.0}, {3.0}}};
    NumberType reference = 1.0;
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
  }

  SECTION("Asymmetric neighbours - smaller slope selected")
  {
    u_n = {{{1.0}, {2.5}}};
    NumberType reference = 0.5;
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
  }

  SECTION("Opposite-sign slopes - clipped to zero")
  {
    u_n = {{{1.0}, {1.0}}};
    NumberType reference = 0.0;
    auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));

    u_n = {{{5.0}, {4.0}}};
    u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
  }
}

TEST_CASE("TVD gradient at point in 1D on a monotone stencil", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 1;
  constexpr int n_faces = 2 * dim;
  using Reconstructor = TVDReconstructorImpl<dim>;

  const Point<dim> x_center(1.0);
  const std::array<NumberType, n_components> u_center = {2.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  const std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0}, {3.0}}};

  SECTION("Returns the forward slope")
  {
    const Point<dim> x_q(1.5);
    const auto u_grad =
        (Reconstructor::compute_gradient_at_point<n_components>)(x_center, x_q, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx((u_n[1][0] - u_center[0]) / (x_n[1][0] - x_center[0])));
  }

  SECTION("Returns the backward slope")
  {
    const Point<dim> x_q(0.5);
    const auto u_grad =
        (Reconstructor::compute_gradient_at_point<n_components>)(x_center, x_q, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx((u_center[0] - u_n[0][0]) / (x_center[0] - x_n[0][0])));
  }
}

TEST_CASE("TVD gradient at point in 1D on a cusp", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 1;
  constexpr int n_faces = 2 * dim;
  using Reconstructor = TVDReconstructorImpl<dim>;

  const Point<dim> x_center(1.0);
  const std::array<NumberType, n_components> u_center = {0.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  const std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0}, {3.0}}};

  SECTION("Returns the forward slope")
  {
    const Point<dim> x_q(1.5);
    const auto u_grad =
        (Reconstructor::compute_gradient_at_point<n_components>)(x_center, x_q, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx((u_n[1][0] - u_center[0]) / (x_n[1][0] - x_center[0])));
  }

  SECTION("Returns the backward slope")
  {
    const Point<dim> x_q(0.5);
    const auto u_grad =
        (Reconstructor::compute_gradient_at_point<n_components>)(x_center, x_q, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx((u_center[0] - u_n[0][0]) / (x_center[0] - x_n[0][0])));
  }
}

// ──────────────────────────────────────────────────────────────────────
// 1D, two components (moved from KurganovTadmor_tests.cc)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation in 1D, two components", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 2;
  constexpr int n_faces = 2 * dim;
  using Reconstructor = TVDReconstructorImpl<dim>;
  const Point<dim> x_center(1.0);
  const std::array<NumberType, n_components> u_center = {2.0, 3.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  std::array<std::array<NumberType, n_components>, n_faces> u_n;

  SECTION("Symmetric neighbours - both components")
  {
    u_n = {{{1.0, 2.0}, {3.0, 4.0}}};
    NumberType reference = 1.0;
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));
  }

  SECTION("Asymmetric neighbours - smaller slope for both components")
  {
    u_n = {{{1.0, 2.0}, {2.5, 3.5}}};
    NumberType reference = 0.5;
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));
  }

  SECTION("Opposite-sign slopes - clipped to zero for both components")
  {
    u_n = {{{1.0, 2.0}, {1.0, 2.0}}};
    NumberType reference = 0.0;
    auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));

    u_n = {{{5.0, 1.0}, {4.0, 0.0}}};
    u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));
  }
}

TEST_CASE("TVD gradient at point in 1D, two components", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 2;
  constexpr int n_faces = 2 * dim;
  using Reconstructor = TVDReconstructorImpl<dim>;

  const Point<dim> x_center(1.0);
  const std::array<NumberType, n_components> u_center = {2.0, 5.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  const std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0, 4.0}, {3.0, 6.0}}};

  SECTION("Returns the forward slope for both components")
  {
    const Point<dim> x_q(1.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);

    CHECK(u_grad[0][0] == Catch::Approx((u_n[1][0] - u_center[0]) / (x_n[1][0] - x_center[0])));
    CHECK(u_grad[1][0] == Catch::Approx((u_n[1][1] - u_center[1]) / (x_n[1][0] - x_center[0])));
  }

  SECTION("Returns the backward slope for both components")
  {
    const Point<dim> x_q(0.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);

    CHECK(u_grad[0][0] == Catch::Approx((u_center[0] - u_n[0][0]) / (x_center[0] - x_n[0][0])));
    CHECK(u_grad[1][0] == Catch::Approx((u_center[1] - u_n[0][1]) / (x_center[0] - x_n[0][0])));
  }
}

// ──────────────────────────────────────────────────────────────────────
// 2D, one component (moved from KurganovTadmor_tests.cc)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation in 2D, one component", "[FV][reconstructor]")
{
  constexpr int dim = 2;
  constexpr int n_components = 1;
  constexpr int n_faces = 2 * dim;
  using Reconstructor = TVDReconstructorImpl<dim>;
  const Point<dim> x_center(1.0, 1.0);
  const std::array<NumberType, n_components> u_center = {2.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0, 1.0), Point<dim>(2.0, 1.0), Point<dim>(1.0, 0.0),
                                               Point<dim>(1.0, 2.0)};
  std::array<std::array<NumberType, n_components>, n_faces> u_n;

  SECTION("Full gradient in both dimensions")
  {
    u_n = {{{1.0}, {3.0}, {0.0}, {4.0}}};
    NumberType reference_1 = 1.0;
    NumberType reference_2 = 2.0;
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference_1));
    CHECK(u_grad[0][1] == Catch::Approx(reference_2));
  }

  SECTION("Smaller slope selected per dimension")
  {
    u_n = {{{1.0}, {2.5}, {2.5}, {-0.5}}};
    NumberType reference = 0.5;
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[0][1] == Catch::Approx(-reference));
  }

  SECTION("Clipping in one dimension only")
  {
    u_n = {{{1.0}, {1.0}, {0.0}, {4.0}}};
    NumberType reference_1 = 0.0;
    NumberType reference_2 = 2.0;
    auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference_1));
    CHECK(u_grad[0][1] == Catch::Approx(reference_2));
  }
}

TEST_CASE("TVD gradient at point in 2D, one component", "[FV][reconstructor]")
{
  constexpr int dim = 2;
  constexpr int n_components = 1;
  constexpr int n_faces = 2 * dim;
  using Reconstructor = TVDReconstructorImpl<dim>;

  const Point<dim> x_center(1.0, 1.0);
  const std::array<NumberType, n_components> u_center = {2.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0, 1.0), Point<dim>(2.0, 1.0), Point<dim>(1.0, 0.0),
                                               Point<dim>(1.0, 2.0)};
  const std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0}, {3.0}, {0.0}, {4.0}}};

  SECTION("Returns the forward x slope and the backward y slope")
  {
    const Point<dim> x_q(1.5, 0.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);

    CHECK(u_grad[0][0] == Catch::Approx((u_n[1][0] - u_center[0]) / (x_n[1][0] - x_center[0])));
    CHECK(u_grad[0][1] == Catch::Approx((u_center[0] - u_n[2][0]) / (x_center[1] - x_n[2][1])));
  }

  SECTION("Returns the backward x slope and the forward y slope")
  {
    const Point<dim> x_q(0.5, 1.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);

    CHECK(u_grad[0][0] == Catch::Approx((u_center[0] - u_n[0][0]) / (x_center[0] - x_n[0][0])));
    CHECK(u_grad[0][1] == Catch::Approx((u_n[3][0] - u_center[0]) / (x_n[3][1] - x_center[1])));
  }

  SECTION("Returns the backward x slope and the backward y slope")
  {
    const Point<dim> x_q(0.5, 0.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx((u_center[0] - u_n[0][0]) / (x_center[0] - x_n[0][0])));
    CHECK(u_grad[0][1] == Catch::Approx((u_center[0] - u_n[2][0]) / (x_center[1] - x_n[2][1])));
  }
  SECTION("Returns the forward x slope and the forward y slope")
  {
    const Point<dim> x_q(1.5, 1.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx((u_n[1][0] - u_center[0]) / (x_n[1][0] - x_center[0])));
    CHECK(u_grad[0][1] == Catch::Approx((u_n[3][0] - u_center[0]) / (x_n[3][1] - x_center[1])));
  }
}

// ──────────────────────────────────────────────────────────────────────
// 2D, two components (new test)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation in 2D, two components", "[FV][reconstructor]")
{
  constexpr int dim = 2;
  constexpr int n_components = 2;
  constexpr int n_faces = 2 * dim;
  using Reconstructor = TVDReconstructorImpl<dim>;
  const Point<dim> x_center(1.0, 1.0);
  const std::array<NumberType, n_components> u_center = {2.0, 5.0};
  // neighbours: left, right, bottom, top
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0, 1.0), Point<dim>(2.0, 1.0), Point<dim>(1.0, 0.0),
                                               Point<dim>(1.0, 2.0)};

  SECTION("Independent gradients per component")
  {
    // component 0: left=1, right=3 => slopes +1,+1 => grad_x=1
    //              bottom=0, top=4 => slopes +2,+2 => grad_y=2
    // component 1: left=4, right=6 => slopes +1,+1 => grad_x=1
    //              bottom=3, top=7 => slopes +2,+2 => grad_y=2
    std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0, 4.0}, {3.0, 6.0}, {0.0, 3.0}, {4.0, 7.0}}};
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(1.0));
    CHECK(u_grad[0][1] == Catch::Approx(2.0));
    CHECK(u_grad[1][0] == Catch::Approx(1.0));
    CHECK(u_grad[1][1] == Catch::Approx(2.0));
  }

  SECTION("One component clipped, other has full gradient")
  {
    // component 0: left=3, right=3 => slopes -1,+1 => clipped to 0
    //              bottom=2, top=2 => slopes  0, 0 => 0
    // component 1: left=4, right=6 => slopes +1,+1 => grad_x=1
    //              bottom=4, top=6 => slopes +1,+1 => grad_y=1
    std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{3.0, 4.0}, {3.0, 6.0}, {2.0, 4.0}, {2.0, 6.0}}};
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(0.0));
    CHECK(u_grad[0][1] == Catch::Approx(0.0));
    CHECK(u_grad[1][0] == Catch::Approx(1.0));
    CHECK(u_grad[1][1] == Catch::Approx(1.0));
  }
}

TEST_CASE("TVD gradient at point in 2D, two components", "[FV][reconstructor]")
{
  constexpr int dim = 2;
  constexpr int n_components = 2;
  constexpr int n_faces = 2 * dim;
  using Reconstructor = TVDReconstructorImpl<dim>;

  const Point<dim> x_center(1.0, 1.0);
  const std::array<NumberType, n_components> u_center = {2.0, 5.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0, 1.0), Point<dim>(2.0, 1.0), Point<dim>(1.0, 0.0),
                                               Point<dim>(1.0, 2.0)};
  const std::array<std::array<NumberType, n_components>, n_faces> u_n = {
      {{1.0, 4.0}, {3.0, 6.0}, {0.0, 3.0}, {4.0, 7.0}}};

  SECTION("Returns the forward x slope and the backward y slope for both components")
  {
    const Point<dim> x_q(1.5, 0.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);

    CHECK(u_grad[0][0] == Catch::Approx((u_n[1][0] - u_center[0]) / (x_n[1][0] - x_center[0])));
    CHECK(u_grad[0][1] == Catch::Approx((u_center[0] - u_n[2][0]) / (x_center[1] - x_n[2][1])));
    CHECK(u_grad[1][0] == Catch::Approx((u_n[1][1] - u_center[1]) / (x_n[1][0] - x_center[0])));
    CHECK(u_grad[1][1] == Catch::Approx((u_center[1] - u_n[2][1]) / (x_center[1] - x_n[2][1])));
  }

  SECTION("Returns the backward x slope and the forward y slope for both components")
  {
    const Point<dim> x_q(0.5, 1.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);

    CHECK(u_grad[0][0] == Catch::Approx((u_center[0] - u_n[0][0]) / (x_center[0] - x_n[0][0])));
    CHECK(u_grad[0][1] == Catch::Approx((u_n[3][0] - u_center[0]) / (x_n[3][1] - x_center[1])));
    CHECK(u_grad[1][0] == Catch::Approx((u_center[1] - u_n[0][1]) / (x_center[0] - x_n[0][0])));
    CHECK(u_grad[1][1] == Catch::Approx((u_n[3][1] - u_center[1]) / (x_n[3][1] - x_center[1])));
  }

  SECTION("Returns the backward x slope and the backward y slope for both components")
  {
    const Point<dim> x_q(0.5, 0.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);

    CHECK(u_grad[0][0] == Catch::Approx((u_center[0] - u_n[0][0]) / (x_center[0] - x_n[0][0])));
    CHECK(u_grad[0][1] == Catch::Approx((u_center[0] - u_n[2][0]) / (x_center[1] - x_n[2][1])));
    CHECK(u_grad[1][0] == Catch::Approx((u_center[1] - u_n[0][1]) / (x_center[0] - x_n[0][0])));
    CHECK(u_grad[1][1] == Catch::Approx((u_center[1] - u_n[2][1]) / (x_center[1] - x_n[2][1])));
  }

  SECTION("Returns the forward x slope and the forward y slope for both components")
  {
    const Point<dim> x_q(1.5, 1.5);
    const auto u_grad = Reconstructor::compute_gradient_at_point<n_components>(x_center, x_q, u_center, x_n, u_n);

    CHECK(u_grad[0][0] == Catch::Approx((u_n[1][0] - u_center[0]) / (x_n[1][0] - x_center[0])));
    CHECK(u_grad[0][1] == Catch::Approx((u_n[3][0] - u_center[0]) / (x_n[3][1] - x_center[1])));
    CHECK(u_grad[1][0] == Catch::Approx((u_n[1][1] - u_center[1]) / (x_n[1][0] - x_center[0])));
    CHECK(u_grad[1][1] == Catch::Approx((u_n[3][1] - u_center[1]) / (x_n[3][1] - x_center[1])));
  }
}

// ──────────────────────────────────────────────────────────────────────
// Non-unit spacing (new test)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation with non-unit spacing", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 1;
  constexpr int n_faces = 2;
  using Reconstructor = TVDReconstructorImpl<dim>;

  SECTION("dx = 0.5 on both sides, symmetric neighbours")
  {
    const Point<dim> x_center(1.0);
    const std::array<NumberType, n_components> u_center = {2.0};
    // neighbours at distance 0.5 on each side
    const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.5), Point<dim>(1.5)};
    // left slope = (1-2)/(0.5-1) = -1/(-0.5) = 2
    // right slope = (3-2)/(1.5-1) = 1/0.5 = 2
    // minmod(2,2) = 2
    std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0}, {3.0}}};
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(2.0));
  }

  SECTION("Asymmetric spacing - different slopes")
  {
    const Point<dim> x_center(1.0);
    const std::array<NumberType, n_components> u_center = {2.0};
    // left neighbour at 0.0 (dx=1), right neighbour at 1.25 (dx=0.25)
    const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(1.25)};
    // left slope = (1-2)/(0-1) = 1
    // right slope = (2.5-2)/(1.25-1) = 0.5/0.25 = 2
    // minmod(1,2) = 1
    std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0}, {2.5}}};
    const auto u_grad = Reconstructor::compute_gradient<n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(1.0));
  }
}

// ──────────────────────────────────────────────────────────────────────
// compute_gradient_derivative — 1D, one component
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient derivative in 1D, one component", "[FV][reconstructor]")
{
  using AD = autodiff::Real<1, NumberType>;
  constexpr int dim = 1;
  constexpr int n_components = 1;
  using Reconstructor = TVDReconstructorImpl<dim>;
  const Point<dim> x_center(1.0);
  // Asymmetric setup: left slope = 1, right slope = 0.5
  // MinMod(1, 0.5) = 0.5 (right slope active)
  const std::array<Point<dim>, 2> x_n = {Point<dim>(0.0), Point<dim>(2.0)};

  SECTION("w.r.t. cell centre")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)}};
    seed(u_cp[0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_cp[0]);
    CHECK(result[0][0] == Catch::Approx(-1.0));
  }

  SECTION("w.r.t. left neighbour (inactive slope)")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)}};
    seed(u_np[0][0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_np[0][0]);
    CHECK(result[0][0] == Catch::Approx(0.0));
  }

  SECTION("w.r.t. right neighbour (active slope)")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)}};
    seed(u_np[1][0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_np[1][0]);
    CHECK(result[0][0] == Catch::Approx(1.0));
  }

  SECTION("opposite-sign slopes — all derivatives zero")
  {
    // left slope = (3-2)/(-1) = -1, right slope = (3-2)/1 = 1, MinMod = 0
    // Seed centre
    {
      std::array<AD, 1> u_cp = {AD(2.0)};
      std::array<std::array<AD, 1>, 2> u_np = {std::array<AD, 1>{AD(3.0)}, std::array<AD, 1>{AD(3.0)}};
      seed(u_cp[0]);
      const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
      unseed(u_cp[0]);
      CHECK(result[0][0] == Catch::Approx(0.0));
    }
    // Seed left
    {
      std::array<AD, 1> u_cp = {AD(2.0)};
      std::array<std::array<AD, 1>, 2> u_np = {std::array<AD, 1>{AD(3.0)}, std::array<AD, 1>{AD(3.0)}};
      seed(u_np[0][0]);
      const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
      unseed(u_np[0][0]);
      CHECK(result[0][0] == Catch::Approx(0.0));
    }
    // Seed right
    {
      std::array<AD, 1> u_cp = {AD(2.0)};
      std::array<std::array<AD, 1>, 2> u_np = {std::array<AD, 1>{AD(3.0)}, std::array<AD, 1>{AD(3.0)}};
      seed(u_np[1][0]);
      const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
      unseed(u_np[1][0]);
      CHECK(result[0][0] == Catch::Approx(0.0));
    }
  }
}

TEST_CASE("TVD third derivative at a 1D face uses non-uniform divided differences", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 1;
  using Reconstructor = TVDReconstructorImpl<dim>;

  const std::array<Point<dim>, 4> x = {Point<dim>(-0.7), Point<dim>(0.0), Point<dim>(0.4), Point<dim>(1.3)};
  std::array<std::array<NumberType, n_components>, 4> u{};
  for (size_t i = 0; i < x.size(); ++i)
    u[i][0] = 2.0 + 0.5 * x[i][0] - 0.25 * x[i][0] * x[i][0] + 3.0 * x[i][0] * x[i][0] * x[i][0];

  const auto third_derivatives = Reconstructor::compute_third_derivatives_at_face<n_components>(x, u);
  CHECK(third_derivatives[0][0][0][0] == Catch::Approx(18.0).margin(1e-12));
}

TEST_CASE("TVD third derivative AD trace matches finite differences", "[FV][reconstructor]")
{
  using AD = autodiff::Real<1, NumberType>;
  constexpr int dim = 1;
  constexpr int n_components = 1;
  using Reconstructor = TVDReconstructorImpl<dim>;

  const std::array<Point<dim>, 4> x = {Point<dim>(-0.3), Point<dim>(0.2), Point<dim>(0.9), Point<dim>(1.7)};
  std::array<std::array<AD, n_components>, 4> u{};
  for (size_t i = 0; i < x.size(); ++i)
    u[i][0] = AD(1.0 + x[i][0] + 0.4 * x[i][0] * x[i][0]);

  seed(u[2][0]);
  const auto derivative = Reconstructor::compute_third_derivatives_at_face_derivative<n_components>(x, u);
  unseed(u[2][0]);

  constexpr double eps = 1e-7;
  std::array<std::array<NumberType, n_components>, 4> u_plus{};
  std::array<std::array<NumberType, n_components>, 4> u_minus{};
  for (size_t i = 0; i < x.size(); ++i) {
    u_plus[i][0] = u[i][0].val();
    u_minus[i][0] = u[i][0].val();
  }
  u_plus[2][0] += eps;
  u_minus[2][0] -= eps;

  const auto third_plus = Reconstructor::compute_third_derivatives_at_face<n_components>(x, u_plus);
  const auto third_minus = Reconstructor::compute_third_derivatives_at_face<n_components>(x, u_minus);
  const auto fd = (third_plus[0][0][0][0] - third_minus[0][0][0][0]) / (2.0 * eps);

  CHECK(derivative[0][0][0][0] == Catch::Approx(fd).epsilon(1e-8));
}

// ──────────────────────────────────────────────────────────────────────
// compute_gradient_derivative — 1D, two components
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient derivative in 1D, two components", "[FV][reconstructor]")
{
  using AD = autodiff::Real<1, NumberType>;
  constexpr int dim = 1;
  constexpr int n_components = 2;
  using Reconstructor = TVDReconstructorImpl<dim>;
  const Point<dim> x_center(1.0);
  const std::array<Point<dim>, 2> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  // Component 0: u_c=2, left=1, right=2.5 → slopes (1, 0.5), MinMod=0.5, right active
  // Component 1: u_c=5, left=4.5, right=7 → slopes (0.5, 2), MinMod=0.5, left active

  SECTION("seed centre component 0 — cross-component derivative is zero")
  {
    std::array<AD, 2> u_cp = {AD(2.0), AD(5.0)};
    std::array<std::array<AD, 2>, 2> u_np = {std::array<AD, 2>{AD(1.0), AD(4.5)}, std::array<AD, 2>{AD(2.5), AD(7.0)}};
    seed(u_cp[0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_cp[0]);
    CHECK(result[0][0] == Catch::Approx(-1.0)); // right slope active for comp 0
    CHECK(result[1][0] == Catch::Approx(0.0));  // comp 1 is independent of comp 0's DOF
  }

  SECTION("seed centre component 1 — cross-component derivative is zero")
  {
    std::array<AD, 2> u_cp = {AD(2.0), AD(5.0)};
    std::array<std::array<AD, 2>, 2> u_np = {std::array<AD, 2>{AD(1.0), AD(4.5)}, std::array<AD, 2>{AD(2.5), AD(7.0)}};
    seed(u_cp[1]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_cp[1]);
    CHECK(result[0][0] == Catch::Approx(0.0)); // comp 0 is independent of comp 1's DOF
    CHECK(result[1][0] == Catch::Approx(1.0)); // left slope active for comp 1
  }

  SECTION("seed left neighbour component 1 — active slope")
  {
    std::array<AD, 2> u_cp = {AD(2.0), AD(5.0)};
    std::array<std::array<AD, 2>, 2> u_np = {std::array<AD, 2>{AD(1.0), AD(4.5)}, std::array<AD, 2>{AD(2.5), AD(7.0)}};
    seed(u_np[0][1]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_np[0][1]);
    CHECK(result[0][0] == Catch::Approx(0.0)); // comp 0 unaffected
    // comp 1 left slope: du_1 = (u_L1 - u_c1)/(x_L - x_c) = (u_L1 - 5)/(-1)
    // d(du_1)/d(u_L1) = 1/(-1) = -1
    CHECK(result[1][0] == Catch::Approx(-1.0));
  }
}

// ──────────────────────────────────────────────────────────────────────
// compute_gradient_derivative — 2D, one component
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient derivative in 2D, one component", "[FV][reconstructor]")
{
  using AD = autodiff::Real<1, NumberType>;
  constexpr int dim = 2;
  constexpr int n_components = 1;
  using Reconstructor = TVDReconstructorImpl<dim>;
  const Point<dim> x_center(1.0, 1.0);
  // left, right, bottom, top
  const std::array<Point<dim>, 4> x_n = {Point<dim>(0.0, 1.0), Point<dim>(2.0, 1.0), Point<dim>(1.0, 0.0),
                                         Point<dim>(1.0, 2.0)};
  // u_center = 2, u_n = {1, 2.5, 0.5, 3}
  // Dim 0: slopes (1, 0.5), MinMod = 0.5 (right active)
  // Dim 1: slopes (1.5, 1), MinMod = 1 (top/right active)

  SECTION("w.r.t. cell centre")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)},
                                             std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(3.0)}};
    seed(u_cp[0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_cp[0]);
    CHECK(result[0][0] == Catch::Approx(-1.0)); // dim 0: right active, d/du_c = -1
    CHECK(result[0][1] == Catch::Approx(-1.0)); // dim 1: top active, d/du_c = -1
  }

  SECTION("w.r.t. left neighbour (dim 0 inactive)")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)},
                                             std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(3.0)}};
    seed(u_np[0][0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_np[0][0]);
    CHECK(result[0][0] == Catch::Approx(0.0)); // left inactive in dim 0
    CHECK(result[0][1] == Catch::Approx(0.0)); // left doesn't participate in dim 1
  }

  SECTION("w.r.t. right neighbour (dim 0 active)")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)},
                                             std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(3.0)}};
    seed(u_np[1][0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_np[1][0]);
    CHECK(result[0][0] == Catch::Approx(1.0)); // right active in dim 0
    CHECK(result[0][1] == Catch::Approx(0.0)); // right doesn't participate in dim 1
  }

  SECTION("w.r.t. bottom neighbour (dim 1 inactive)")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)},
                                             std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(3.0)}};
    seed(u_np[2][0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_np[2][0]);
    CHECK(result[0][0] == Catch::Approx(0.0)); // bottom doesn't participate in dim 0
    CHECK(result[0][1] == Catch::Approx(0.0)); // bottom inactive in dim 1
  }

  SECTION("w.r.t. top neighbour (dim 1 active)")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)},
                                             std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(3.0)}};
    seed(u_np[3][0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_np[3][0]);
    CHECK(result[0][0] == Catch::Approx(0.0)); // top doesn't participate in dim 0
    CHECK(result[0][1] == Catch::Approx(1.0)); // top active in dim 1
  }
}

// ──────────────────────────────────────────────────────────────────────
// compute_gradient_derivative — non-unit spacing
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient derivative with non-unit spacing", "[FV][reconstructor]")
{
  using AD = autodiff::Real<1, NumberType>;
  constexpr int dim = 1;
  constexpr int n_components = 1;
  using Reconstructor = TVDReconstructorImpl<dim>;
  const Point<dim> x_center(1.0);
  // dx = 0.5 on each side
  const std::array<Point<dim>, 2> x_n = {Point<dim>(0.5), Point<dim>(1.5)};
  // left slope = (1-2)/(-0.5) = 2
  // right slope = (2.5-2)/0.5 = 1
  // MinMod(2, 1) = 1 (right active)

  SECTION("w.r.t. cell centre")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)}};
    seed(u_cp[0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_cp[0]);
    // d(du_right)/d(u_c) = -1/0.5 = -2
    CHECK(result[0][0] == Catch::Approx(-2.0));
  }

  SECTION("w.r.t. right neighbour (active)")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)}};
    seed(u_np[1][0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_np[1][0]);
    // d(du_right)/d(u_R) = 1/0.5 = 2
    CHECK(result[0][0] == Catch::Approx(2.0));
  }

  SECTION("w.r.t. left neighbour (inactive)")
  {
    std::array<AD, 1> u_cp = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_np = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(2.5)}};
    seed(u_np[0][0]);
    const auto result = Reconstructor::compute_gradient_derivative<n_components>(x_center, u_cp, x_n, u_np);
    unseed(u_np[0][0]);
    CHECK(result[0][0] == Catch::Approx(0.0));
  }
}
