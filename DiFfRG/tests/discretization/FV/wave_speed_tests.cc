#include <DiFfRG/discretization/FV/wave_speed/abstract_wave_speed.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>

#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <boilerplate/kt_models.hh>
#include <boilerplate/kt_trace_evaluation.hh>

using NumberType = double;
using namespace dealii;
namespace KT = DiFfRG::FV::KurganovTadmor;

/// Cell width handed to the flux helpers below. None of these models reads it.
static constexpr double test_cell_width = 1.0;

/// Empty extractor / variable data for the direct helper calls below. These models declare neither,
/// so the flux never reads them; they are here because the helpers take them unconditionally.
static const std::array<double, 0> no_extractors{};
static const dealii::Vector<double> no_variables{};

using KT::MaxEigenvalueWaveSpeed;

template <typename NT, size_t nc> using JacobianMatrix = KT::internal::JacobianMatrix<NT, nc>;
template <typename NT, int dim, size_t nc> using HessianTensor = KT::internal::HessianTensor<NT, dim, nc>;

// nvcc's cudafe emits the first use of a specialization symbolically but constant-folds every
// later one, rendering a size_t argument as the ill-formed functional cast `unsigned long((1))`.
// Naming each specialization once here keeps the use sites below off that path.
using HessianTensor_1_1 = HessianTensor<NumberType, 1, 1>; // <dim, n_components>
using HessianTensor_2_1 = HessianTensor<NumberType, 2, 1>;

// ---------------------------------------------------------------------------
// Concept satisfaction (compile-time)
// ---------------------------------------------------------------------------

static_assert(DiFfRG::def::HasWaveSpeed<MaxEigenvalueWaveSpeed>);

// ---------------------------------------------------------------------------
// Test models from boilerplate/models.hh
// ---------------------------------------------------------------------------

// SymCoupledModel is defined locally as it doesn't exist in models.hh
// 2-component symmetric coupled: F_1 = u1^2/2 + u2, F_2 = u1 + u2^2/2.
// J = [[u1, 1], [1, u2]] — symmetric, always diagonalizable with distinct eigenvalues
// H_1[j][c] = d²F_1/(du_j du_c) = [[1, 0], [0, 0]]
// H_2[j][c] = d²F_2/(du_j du_c) = [[0, 0], [0, 1]]
using FEFunctionDesc2 = DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u1">, DiFfRG::Scalar<"u2">>;
using Components2 = DiFfRG::ComponentDescriptor<FEFunctionDesc2>;

class SymCoupledModel : public DiFfRG::def::AbstractModel<SymCoupledModel, Components2>
{
public:
  template <int dim, typename NT, typename Solutions, size_t n>
  static void flux(std::array<Tensor<1, dim, NT>, n> &F, const Point<dim> & /*x*/, const Solutions &sol)
  {
    const auto u = get<"fe_functions">(sol);
    F[0][0] = u[0] * u[0] / NT(2) + u[1]; // F_1 = u1^2/2 + u2
    F[1][0] = u[0] + u[1] * u[1] / NT(2); // F_2 = u1 + u2^2/2
  }
};

// ===========================================================================
// MaxEigenvalueWaveSpeed::compute_speeds
// ===========================================================================

TEST_CASE("compute_speeds 1D scalar — positive Jacobians", "[FV][wave_speed]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0][0][0] = 3.0;
  J_minus[0][0][0] = 1.0;

  const auto a = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_plus, J_minus);
  CHECK(a[0] == Catch::Approx(3.0));
}

TEST_CASE("compute_speeds 1D scalar — negative Jacobians", "[FV][wave_speed]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0][0][0] = -2.0;
  J_minus[0][0][0] = -5.0;

  const auto a = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_plus, J_minus);
  CHECK(a[0] == Catch::Approx(5.0));
}

TEST_CASE("compute_speeds 1D scalar — mixed sign", "[FV][wave_speed]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0][0][0] = 2.0;
  J_minus[0][0][0] = -4.0;

  const auto a = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_plus, J_minus);
  CHECK(a[0] == Catch::Approx(4.0));
}

TEST_CASE("compute_speeds 1D scalar — zero Jacobians", "[FV][wave_speed]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0][0][0] = 0.0;
  J_minus[0][0][0] = 0.0;

  const auto a = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_plus, J_minus);
  CHECK(a[0] == Catch::Approx(0.0));
}

TEST_CASE("compute_speeds 1D 2-component — diagonal Jacobians", "[FV][wave_speed]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;

  // J_plus = diag(3, 1), eigenvalues {3, 1}, spectral_radius = 3
  // J_minus = diag(2, 4), eigenvalues {2, 4}, spectral_radius = 4
  // a = max(3, 4) = 4
  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0] = {{{3.0, 0.0}, {0.0, 1.0}}};
  J_minus[0] = {{{2.0, 0.0}, {0.0, 4.0}}};

  const auto a = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_plus, J_minus);
  CHECK(a[0] == Catch::Approx(4.0));
}

TEST_CASE("compute_speeds 1D 2-component — non-diagonal (coupled)", "[FV][wave_speed]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;

  // J = [[0, 1], [1, 0]], eigenvalues = ±1, spectral_radius = 1
  // Both sides identical => a = 1
  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0] = {{{0.0, 1.0}, {1.0, 0.0}}};
  J_minus[0] = {{{0.0, 1.0}, {1.0, 0.0}}};

  const auto a = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_plus, J_minus);
  CHECK(a[0] == Catch::Approx(1.0));
}

TEST_CASE("compute_speeds 2D 1-component — independent per dimension", "[FV][wave_speed]")
{
  constexpr int dim = 2;
  constexpr size_t nc = 1;

  // dim 0: J_plus = 2, J_minus = 3 => a[0] = 3
  // dim 1: J_plus = 5, J_minus = 1 => a[1] = 5
  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0][0][0] = 2.0;
  J_plus[1][0][0] = 5.0;
  J_minus[0][0][0] = 3.0;
  J_minus[1][0][0] = 1.0;

  const auto a = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_plus, J_minus);
  CHECK(a[0] == Catch::Approx(3.0));
  CHECK(a[1] == Catch::Approx(5.0));
}

TEST_CASE("selected speed derivatives average equal max-speed branches", "[FV][wave_speed]")
{
  constexpr int dim = 2;
  constexpr size_t nc = 1;

  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0][0][0] = 2.0;
  J_minus[0][0][0] = -3.0;
  J_plus[1][0][0] = -4.0;
  J_minus[1][0][0] = 4.0;

  HessianTensor_2_1 H_plus{}, H_minus{};
  H_plus[0][0][0][0] = 2.0;
  H_minus[0][0][0][0] = 6.0;
  H_plus[1][0][0][0] = 2.0;
  H_minus[1][0][0][0] = 6.0;

  const auto branches = MaxEigenvalueWaveSpeed::select_speed_branches<NumberType, dim, nc>(J_plus, J_minus);
  const auto [da_plus, da_minus] =
      MaxEigenvalueWaveSpeed::compute_selected_speed_derivatives<NumberType, dim, nc>(J_plus, J_minus, H_plus, H_minus);

  CHECK(branches[0] == KT::WaveSpeedBranch::minus);
  CHECK(branches[1] == KT::WaveSpeedBranch::average);
  CHECK(da_plus[0][0] == Catch::Approx(0.0));
  CHECK(da_minus[0][0] == Catch::Approx(-6.0));
  CHECK(da_plus[1][0] == Catch::Approx(-1.0));
  CHECK(da_minus[1][0] == Catch::Approx(3.0));
}

// ===========================================================================
// MaxEigenvalueWaveSpeed::compute_speed_derivatives
// ===========================================================================

TEST_CASE("compute_speed_derivatives 1D 1-component — positive J", "[FV][wave_speed]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  // J_plus = [[3]], J_minus = [[1]], H = [[1]] for both (Burgers-like)
  // da_plus/du = sign(3) * 1 = 1
  // da_minus/du = sign(1) * 1 = 1
  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0][0][0] = 3.0;
  J_minus[0][0][0] = 1.0;

  HessianTensor_1_1 H_plus{}, H_minus{};
  H_plus[0][0][0][0] = 1.0;
  H_minus[0][0][0][0] = 1.0;

  const auto [da_plus, da_minus] =
      MaxEigenvalueWaveSpeed::compute_speed_derivatives<NumberType, dim, nc>(J_plus, J_minus, H_plus, H_minus);

  CHECK(da_plus[0][0] == Catch::Approx(1.0));
  CHECK(da_minus[0][0] == Catch::Approx(1.0));
}

TEST_CASE("compute_speed_derivatives 1D 1-component — negative J", "[FV][wave_speed]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  // J_plus = [[-2]], H = [[1]]
  // da_plus/du = sign(-2) * 1 = -1
  std::array<JacobianMatrix<NumberType, nc>, dim> J_plus{}, J_minus{};
  J_plus[0][0][0] = -2.0;
  J_minus[0][0][0] = -5.0;

  HessianTensor_1_1 H_plus{}, H_minus{};
  H_plus[0][0][0][0] = 1.0;
  H_minus[0][0][0][0] = 1.0;

  const auto [da_plus, da_minus] =
      MaxEigenvalueWaveSpeed::compute_speed_derivatives<NumberType, dim, nc>(J_plus, J_minus, H_plus, H_minus);

  CHECK(da_plus[0][0] == Catch::Approx(-1.0));
  CHECK(da_minus[0][0] == Catch::Approx(-1.0));
}

TEST_CASE("compute_speed_derivatives 1D 1-component — Burgers FD validation", "[FV][wave_speed]")
{
  // Validate da/du against finite differences using actual Burgers model data
  constexpr int dim = 1;
  constexpr size_t nc = 1;
  DiFfRG::Testing::PhysicalParameters prm;
  DiFfRG::Testing::ModelBurgersKT<dim> model(prm);
  const Point<dim> x_q(0.0);
  const NumberType u0 = 2.5;
  const NumberType eps = 1e-6;

  // Compute J and H at u0
  const std::array<NumberType, nc> u = {u0};
  const auto [F, J, H] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<DiFfRG::Testing::ModelBurgersKT<dim>, NumberType, dim, nc>(
          u, x_q, test_cell_width, no_extractors, no_variables, model);

  // Use J for both plus and minus (self-consistent test)
  const auto [da_plus, da_minus] = MaxEigenvalueWaveSpeed::compute_speed_derivatives<NumberType, dim, nc>(J, J, H, H);

  // Finite-difference: perturb u, recompute J, recompute speed
  const std::array<NumberType, nc> u_fwd = {u0 + eps};
  const std::array<NumberType, nc> u_bwd = {u0 - eps};
  const auto [F_fwd, J_fwd, H_fwd] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<DiFfRG::Testing::ModelBurgersKT<dim>, NumberType, dim, nc>(
          u_fwd, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_bwd, J_bwd, H_bwd] =
      DiFfRG::Testing::compute_flux_jacobian_and_hessian<DiFfRG::Testing::ModelBurgersKT<dim>, NumberType, dim, nc>(
          u_bwd, x_q, test_cell_width, no_extractors, no_variables, model);

  const auto a_fwd = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_fwd, J_fwd);
  const auto a_bwd = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_bwd, J_bwd);

  const NumberType da_fd = (a_fwd[0] - a_bwd[0]) / (2.0 * eps);
  CHECK(da_plus[0][0] == Catch::Approx(da_fd).epsilon(1e-4));
  CHECK(da_plus[0][0] == da_minus[0][0]); // symmetric for this test since both sides are identical
}

TEST_CASE("compute_speed_derivatives 1D 2-component — SymCoupledModel FD validation", "[FV][wave_speed]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;
  SymCoupledModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u = {2.0, 3.0};
  const NumberType eps = 1e-6;

  // Compute J and H at u
  const auto [F, J, H] = DiFfRG::Testing::compute_flux_jacobian_and_hessian<SymCoupledModel, NumberType, dim, nc>(
      u, x_q, test_cell_width, no_extractors, no_variables, model);

  const auto [da_plus, da_minus] = MaxEigenvalueWaveSpeed::compute_speed_derivatives<NumberType, dim, nc>(J, J, H, H);

  // Finite-difference for each component
  for (size_t c = 0; c < nc; ++c) {
    std::array<NumberType, nc> u_fwd = u, u_bwd = u;
    u_fwd[c] += eps;
    u_bwd[c] -= eps;

    const auto [F_fwd, J_fwd, H_fwd] =
        DiFfRG::Testing::compute_flux_jacobian_and_hessian<SymCoupledModel, NumberType, dim, nc>(
            u_fwd, x_q, test_cell_width, no_extractors, no_variables, model);
    const auto [F_bwd, J_bwd, H_bwd] =
        DiFfRG::Testing::compute_flux_jacobian_and_hessian<SymCoupledModel, NumberType, dim, nc>(
            u_bwd, x_q, test_cell_width, no_extractors, no_variables, model);

    const auto a_fwd = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_fwd, J_fwd);
    const auto a_bwd = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(J_bwd, J_bwd);

    const NumberType da_fd = (a_fwd[0] - a_bwd[0]) / (2.0 * eps);
    CHECK(da_plus[0][c] == Catch::Approx(da_fd).epsilon(1e-4));
    CHECK(da_plus[0][c] == da_minus[0][c]); // symmetric for this test since both sides are identical
  }
}

TEST_CASE("compute_speed_derivatives 1D 2-component — asymmetric plus/minus FD validation", "[FV][wave_speed]")
{
  // Test with different plus and minus states to exercise both sides independently
  constexpr int dim = 1;
  constexpr size_t nc = 2;
  SymCoupledModel model;
  const Point<dim> x_q(0.0);
  const std::array<NumberType, nc> u_plus_val = {2.0, 3.0};
  const std::array<NumberType, nc> u_minus_val = {1.0, 4.0};
  const NumberType eps = 1e-6;

  const auto [F_p, J_p, H_p] = DiFfRG::Testing::compute_flux_jacobian_and_hessian<SymCoupledModel, NumberType, dim, nc>(
      u_plus_val, x_q, test_cell_width, no_extractors, no_variables, model);
  const auto [F_m, J_m, H_m] = DiFfRG::Testing::compute_flux_jacobian_and_hessian<SymCoupledModel, NumberType, dim, nc>(
      u_minus_val, x_q, test_cell_width, no_extractors, no_variables, model);

  const auto [da_plus, da_minus] =
      MaxEigenvalueWaveSpeed::compute_speed_derivatives<NumberType, dim, nc>(J_p, J_m, H_p, H_m);

  // FD for da_plus (perturb plus side)
  for (size_t c = 0; c < nc; ++c) {
    std::array<NumberType, nc> u_fwd = u_plus_val, u_bwd = u_plus_val;
    u_fwd[c] += eps;
    u_bwd[c] -= eps;

    const auto [Ff, Jf, Hf] = DiFfRG::Testing::compute_flux_jacobian_and_hessian<SymCoupledModel, NumberType, dim, nc>(
        u_fwd, x_q, test_cell_width, no_extractors, no_variables, model);
    const auto [Fb, Jb, Hb] = DiFfRG::Testing::compute_flux_jacobian_and_hessian<SymCoupledModel, NumberType, dim, nc>(
        u_bwd, x_q, test_cell_width, no_extractors, no_variables, model);

    // da_plus is the derivative of rho(J_plus) w.r.t. u_plus components,
    // regardless of which side is dominant in the max. The assembler uses it accordingly.
    const NumberType a_plus_fwd = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(Jf, Jf)[0];
    const NumberType a_plus_bwd = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(Jb, Jb)[0];
    const NumberType da_fd = (a_plus_fwd - a_plus_bwd) / (2.0 * eps);
    CHECK(da_plus[0][c] == Catch::Approx(da_fd).epsilon(1e-4));
  }

  // FD for da_minus (perturb minus side)
  for (size_t c = 0; c < nc; ++c) {
    std::array<NumberType, nc> u_fwd = u_minus_val, u_bwd = u_minus_val;
    u_fwd[c] += eps;
    u_bwd[c] -= eps;

    const auto [Ff, Jf, Hf] = DiFfRG::Testing::compute_flux_jacobian_and_hessian<SymCoupledModel, NumberType, dim, nc>(
        u_fwd, x_q, test_cell_width, no_extractors, no_variables, model);
    const auto [Fb, Jb, Hb] = DiFfRG::Testing::compute_flux_jacobian_and_hessian<SymCoupledModel, NumberType, dim, nc>(
        u_bwd, x_q, test_cell_width, no_extractors, no_variables, model);

    const NumberType a_minus_fwd = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(Jf, Jf)[0];
    const NumberType a_minus_bwd = MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, nc>(Jb, Jb)[0];
    const NumberType da_fd = (a_minus_fwd - a_minus_bwd) / (2.0 * eps);
    CHECK(da_minus[0][c] == Catch::Approx(da_fd).epsilon(1e-4));
  }
}
