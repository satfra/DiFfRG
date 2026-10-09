#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/assembler/flux_jacobian_hessian.hh"
#include "DiFfRG/model/model.hh"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include <boilerplate/kt_trace_evaluation.hh>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

using NumberType = double;
using namespace dealii;
namespace KT = DiFfRG::FV::KurganovTadmor;

/// Cell width handed to the flux helpers below. None of the test models reads it; it is here so the
/// calls match the signature that carries the local grid spacing through to a model's flux.
static constexpr double test_cell_width = 1.0;

/// Empty extractor / variable data for the direct helper calls below. These models declare neither,
/// so the flux never reads them; they are here because the helpers take them unconditionally.
static const std::array<double, 0> no_extractors{};
static const dealii::Vector<double> no_variables{};

// ---------------------------------------------------------------------------
// Test models
// ---------------------------------------------------------------------------

// 1-component Burgers: F(u) = u^2/2.  J = u, H = 1.
using FEFunctionDesc1 = DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u">>;
using Components1 = DiFfRG::ComponentDescriptor<FEFunctionDesc1>;

class BurgersModel : public DiFfRG::def::AbstractModel<BurgersModel, Components1>
{
public:
  template <int dim, typename NT, typename Solutions, size_t n>
  static void flux(std::array<Tensor<1, dim, NT>, n> &F, const Point<dim> & /*x*/, const Solutions &sol)
  {
    const auto u = get<"fe_functions">(sol);
    F[0][0] = u[0] * u[0] / NT(2);
  }
};

// 1-component cubic: F(u) = u^3/3.  J = u^2, H = 2u.
class CubicModel : public DiFfRG::def::AbstractModel<CubicModel, Components1>
{
public:
  template <int dim, typename NT, typename Solutions, size_t n>
  static void flux(std::array<Tensor<1, dim, NT>, n> &F, const Point<dim> & /*x*/, const Solutions &sol)
  {
    const auto u = get<"fe_functions">(sol);
    F[0][0] = u[0] * u[0] * u[0] / NT(3);
  }
};

// 2-component coupled: F_1 = u1*u2, F_2 = u2^2/2.
// J = [[u2, u1], [0, u2]]
// H_1[j][c]: d²F_1/du_j du_c = [[0,1],[1,0]]  (constant)
// H_2[j][c]: d²F_2/du_j du_c = [[0,0],[0,1]]  (constant)
using FEFunctionDesc2 = DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u1">, DiFfRG::Scalar<"u2">>;
using Components2 = DiFfRG::ComponentDescriptor<FEFunctionDesc2>;

class CoupledModel : public DiFfRG::def::AbstractModel<CoupledModel, Components2>
{
public:
  template <int dim, typename NT, typename Solutions, size_t n>
  static void flux(std::array<Tensor<1, dim, NT>, n> &F, const Point<dim> & /*x*/, const Solutions &sol)
  {
    const auto u = get<"fe_functions">(sol);
    F[0][0] = u[0] * u[1];         // F_1 = u1*u2
    F[1][0] = u[1] * u[1] / NT(2); // F_2 = u2^2/2
  }
};

// ---------------------------------------------------------------------------
// Burgers (1D, 1 component)
// ---------------------------------------------------------------------------

TEST_CASE("Burgers 1D: Jacobian values", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;
  BurgersModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u_plus = {3.0};
  const std::array<NumberType, nc> u_minus = {-2.0};

  const auto [F_plus, J_plus, H_plus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<BurgersModel, NumberType, dim, nc>(
          u_plus, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_minus, J_minus, H_minus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<BurgersModel, NumberType, dim, nc>(
          u_minus, x_q, test_cell_width, no_extractors, no_variables, model);

  // F(u) = u^2/2
  CHECK(F_plus[0][0] == Catch::Approx(4.5));  // 3^2/2
  CHECK(F_minus[0][0] == Catch::Approx(2.0)); // (-2)^2/2

  // dF/du = u
  CHECK(J_plus[0][0][0] == Catch::Approx(3.0));
  CHECK(J_minus[0][0][0] == Catch::Approx(-2.0));
}

TEST_CASE("Burgers 1D: diagonal Hessian", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;
  BurgersModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u_plus = {3.0};
  const std::array<NumberType, nc> u_minus = {-2.0};

  const auto [F_plus, J_plus, H_plus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<BurgersModel, NumberType, dim, nc>(
          u_plus, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_minus, J_minus, H_minus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<BurgersModel, NumberType, dim, nc>(
          u_minus, x_q, test_cell_width, no_extractors, no_variables, model);

  // d²F/du² = 1 (constant)
  CHECK(H_plus[0][0][0][0] == Catch::Approx(1.0));
  CHECK(H_minus[0][0][0][0] == Catch::Approx(1.0));
}

// ---------------------------------------------------------------------------
// Cubic (1D, 1 component)
// ---------------------------------------------------------------------------

TEST_CASE("Cubic 1D: Jacobian values", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;
  CubicModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u_plus = {2.0};
  const std::array<NumberType, nc> u_minus = {-3.0};

  const auto [F_plus, J_plus, H_plus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CubicModel, NumberType, dim, nc>(
          u_plus, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_minus, J_minus, H_minus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CubicModel, NumberType, dim, nc>(
          u_minus, x_q, test_cell_width, no_extractors, no_variables, model);

  // F(u) = u^3/3
  CHECK(F_plus[0][0] == Catch::Approx(8.0 / 3.0)); // 2^3/3
  CHECK(F_minus[0][0] == Catch::Approx(-9.0));     // (-3)^3/3

  // dF/du = u^2
  CHECK(J_plus[0][0][0] == Catch::Approx(4.0));
  CHECK(J_minus[0][0][0] == Catch::Approx(9.0));
}

TEST_CASE("Cubic 1D: Hessian values", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;
  CubicModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u_plus = {2.0};
  const std::array<NumberType, nc> u_minus = {-3.0};

  const auto [F_plus, J_plus, H_plus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CubicModel, NumberType, dim, nc>(
          u_plus, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_minus, J_minus, H_minus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CubicModel, NumberType, dim, nc>(
          u_minus, x_q, test_cell_width, no_extractors, no_variables, model);

  // d²F/du² = 2u
  CHECK(H_plus[0][0][0][0] == Catch::Approx(4.0));   // 2*2
  CHECK(H_minus[0][0][0][0] == Catch::Approx(-6.0)); // 2*(-3)
}

// ---------------------------------------------------------------------------
// Coupled 2-component (1D)
// ---------------------------------------------------------------------------

TEST_CASE("Coupled 2-component 1D: Jacobian", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;
  CoupledModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u_plus = {2.0, 3.0};
  const std::array<NumberType, nc> u_minus = {1.0, 4.0};

  const auto [F_plus, J_plus, H_plus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CoupledModel, NumberType, dim, nc>(
          u_plus, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_minus, J_minus, H_minus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CoupledModel, NumberType, dim, nc>(
          u_minus, x_q, test_cell_width, no_extractors, no_variables, model);

  // F_1 = u1*u2, F_2 = u2^2/2
  // F_plus at (u1=2, u2=3): {6.0, 4.5}
  CHECK(F_plus[0][0] == Catch::Approx(6.0));
  CHECK(F_plus[1][0] == Catch::Approx(4.5));
  // F_minus at (u1=1, u2=4): {4.0, 8.0}
  CHECK(F_minus[0][0] == Catch::Approx(4.0));
  CHECK(F_minus[1][0] == Catch::Approx(8.0));

  // J = [[u2, u1], [0, u2]]
  // J_plus at (u1=2, u2=3): [[3, 2], [0, 3]]
  CHECK(J_plus[0][0][0] == Catch::Approx(3.0)); // dF1/du1 = u2
  CHECK(J_plus[0][0][1] == Catch::Approx(2.0)); // dF1/du2 = u1
  CHECK(J_plus[0][1][0] == Catch::Approx(0.0)); // dF2/du1 = 0
  CHECK(J_plus[0][1][1] == Catch::Approx(3.0)); // dF2/du2 = u2

  // J_minus at (u1=1, u2=4): [[4, 1], [0, 4]]
  CHECK(J_minus[0][0][0] == Catch::Approx(4.0));
  CHECK(J_minus[0][0][1] == Catch::Approx(1.0));
  CHECK(J_minus[0][1][0] == Catch::Approx(0.0));
  CHECK(J_minus[0][1][1] == Catch::Approx(4.0));
}

TEST_CASE("Coupled 2-component 1D: off-diagonal Hessian", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;
  CoupledModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u_plus = {2.0, 3.0};
  const std::array<NumberType, nc> u_minus = {1.0, 4.0};

  const auto [F_plus, J_plus, H_plus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CoupledModel, NumberType, dim, nc>(
          u_plus, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_minus, J_minus, H_minus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CoupledModel, NumberType, dim, nc>(
          u_minus, x_q, test_cell_width, no_extractors, no_variables, model);

  // H_1[j][c] = d²F_1/du_j du_c = [[0,1],[1,0]] (constant, independent of u)
  CHECK(H_plus[0][0][0][0] == Catch::Approx(0.0)); // d²F1/du1²
  CHECK(H_plus[0][0][0][1] == Catch::Approx(1.0)); // d²F1/(du1 du2)
  CHECK(H_plus[0][0][1][0] == Catch::Approx(1.0)); // d²F1/(du2 du1) — symmetry
  CHECK(H_plus[0][0][1][1] == Catch::Approx(0.0)); // d²F1/du2²

  // H_2[j][c] = d²F_2/du_j du_c = [[0,0],[0,1]]
  CHECK(H_plus[0][1][0][0] == Catch::Approx(0.0));
  CHECK(H_plus[0][1][0][1] == Catch::Approx(0.0));
  CHECK(H_plus[0][1][1][0] == Catch::Approx(0.0));
  CHECK(H_plus[0][1][1][1] == Catch::Approx(1.0)); // d²F2/du2²

  // Same for minus side (constant Hessian, same values)
  CHECK(H_minus[0][0][0][1] == Catch::Approx(1.0));
  CHECK(H_minus[0][0][1][0] == Catch::Approx(1.0));
  CHECK(H_minus[0][1][1][1] == Catch::Approx(1.0));
}

TEST_CASE("Coupled 2-component 1D: Hessian symmetry", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;
  CoupledModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u_plus = {2.0, 3.0};
  const std::array<NumberType, nc> u_minus = {1.0, 4.0};

  const auto [F_plus, J_plus, H_plus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CoupledModel, NumberType, dim, nc>(
          u_plus, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_minus, J_minus, H_minus] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<CoupledModel, NumberType, dim, nc>(
          u_minus, x_q, test_cell_width, no_extractors, no_variables, model);

  for (size_t d = 0; d < dim; ++d)
    for (size_t i = 0; i < nc; ++i)
      for (size_t j = 0; j < nc; ++j)
        for (size_t c = 0; c < nc; ++c) {
          CHECK(H_plus[d][i][j][c] == Catch::Approx(H_plus[d][i][c][j]));
          CHECK(H_minus[d][i][j][c] == Catch::Approx(H_minus[d][i][c][j]));
        }
}

// ---------------------------------------------------------------------------
// Finite-difference validation
// ---------------------------------------------------------------------------

TEST_CASE("Burgers 1D: Hessian matches finite differences", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;
  BurgersModel model;
  const Point<dim> x_q(0.0);
  const NumberType u0 = 2.5;
  const NumberType eps = 1e-5;

  // Compute Hessian via AD
  const std::array<NumberType, nc> u = {u0};
  const auto [F, J, H] = DiFfRG::Testing::compute_flux_jacobian_and_hessian<BurgersModel, NumberType, dim, nc>(
      u, x_q, test_cell_width, no_extractors, no_variables, model);

  // Compute Jacobian at u+eps and u-eps, then finite-difference the Hessian
  const std::array<NumberType, nc> u_fwd = {u0 + eps};
  const std::array<NumberType, nc> u_bwd = {u0 - eps};
  const auto [F_fwd, J_fwd, H_fwd] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<BurgersModel, NumberType, dim, nc>(
          u_fwd, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_bwd, J_bwd, H_bwd] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<BurgersModel, NumberType, dim, nc>(
          u_bwd, x_q, test_cell_width, no_extractors, no_variables, model);

  const NumberType H_fd = (J_fwd[0][0][0] - J_bwd[0][0][0]) / (2 * eps);
  CHECK(H[0][0][0][0] == Catch::Approx(H_fd).epsilon(1e-4));
}

TEST_CASE("Coupled 2-component 1D: Hessian matches finite differences", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;
  CoupledModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u = {2.0, 3.0};
  const NumberType eps = 1e-5;

  const auto [F0, J0, H0] = DiFfRG::Testing::compute_flux_jacobian_and_hessian<CoupledModel, NumberType, dim, nc>(
      u, x_q, test_cell_width, no_extractors, no_variables, model);

  // Finite-difference d²F_i/(du_j du_c) ≈ (J_i_c(u + eps*e_j) - J_i_c(u - eps*e_j)) / (2*eps)
  for (size_t j = 0; j < nc; ++j) {
    std::array<NumberType, nc> u_fwd = u, u_bwd = u;
    u_fwd[j] += eps;
    u_bwd[j] -= eps;

    const auto [F_fwd, J_fwd, H_fwd] =
        DiFfRG::Testing::compute_flux_jacobian_and_hessian<CoupledModel, NumberType, dim, nc>(
            u_fwd, x_q, test_cell_width, no_extractors, no_variables, model);
    const auto [F_bwd, J_bwd, H_bwd] =
        DiFfRG::Testing::compute_flux_jacobian_and_hessian<CoupledModel, NumberType, dim, nc>(
            u_bwd, x_q, test_cell_width, no_extractors, no_variables, model);

    for (size_t i = 0; i < nc; ++i)
      for (size_t c = 0; c < nc; ++c) {
        const NumberType H_fd = (J_fwd[0][i][c] - J_bwd[0][i][c]) / (2 * eps);
        CHECK(H0[0][i][j][c] == Catch::Approx(H_fd).epsilon(1e-4));
      }
  }
}

// Gradient-dependent scalar flux:
// F(u,g) = u^2/2 + 3*u*g + 5*g^2/2.
class GradientFluxModel : public DiFfRG::def::AbstractModel<GradientFluxModel, Components1>
{
public:
  template <int dim, typename NT, typename Solutions, size_t n>
  static void flux(std::array<Tensor<1, dim, NT>, n> &F, const Point<dim> & /*x*/, const Solutions &sol)
  {
    const auto u = get<"fe_functions">(sol);
    const auto grad_u = get<"fe_derivatives">(sol);
    // Model callbacks commonly accumulate several diagrams.
    F[0][0] += u[0] * u[0] / NT(2) + NT(3) * u[0] * grad_u[0][0] + NT(5) * grad_u[0][0] * grad_u[0][0] / NT(2);
  }
};

TEST_CASE("Flux AD extracts state, gradient, and mixed derivatives", "[FV][flux_jacobian_hessian]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;
  const std::array<NumberType, nc> u = {2.0};
  std::array<Tensor<1, dim, NumberType>, nc> grad_u{};
  grad_u[0][0] = 4.0;

  const auto result = DiFfRG::Testing::compute_flux_derivatives_ad<GradientFluxModel, NumberType, dim, nc>(
      u, grad_u, Point<dim>(), test_cell_width, no_extractors, no_variables, GradientFluxModel{});

  CHECK(result.F[0][0] == Catch::Approx(66.0));
  CHECK(result.J[0][0][0] == Catch::Approx(14.0));
  CHECK(result.H[0][0][0][0] == Catch::Approx(1.0));
  CHECK(result.grad_J[0][0][0][0] == Catch::Approx(26.0));
  CHECK(result.mixed_H[0][0][0][0][0] == Catch::Approx(3.0));
}
