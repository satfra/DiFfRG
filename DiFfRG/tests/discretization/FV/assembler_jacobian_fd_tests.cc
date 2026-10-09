#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <boilerplate/kt_models.hh>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace DiFfRG;
using namespace dealii;

// Helper: set up a minimal JSON config for the FV assembler and mesh
static ConfigTree make_json()
{
  return json::value(
      {{"physical", {{"Lambda", 1.}}},
       {"discretization",
        {{"fe_order", 0},
         {"overintegration", 0},
         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},
         {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}}}},
       {"output", {{"verbosity", 0}}}});
}

static ConfigTree make_json_2d()
{
  return json::value(
      {{"physical", {{"Lambda", 1.}}},
       {"discretization",
        {{"fe_order", 0},
         {"overintegration", 0},
         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},
         {"grid", {{"x_grid", "0:0.25:1"}, {"y_grid", "0:0.25:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
       {"output", {{"verbosity", 0}}}});
}

// Helper: ensure the spdlog "log" sink exists (required by FV Discretization)
static void ensure_logger()
{
  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
  } catch (const spdlog::spdlog_ex &) {
  }
}

class SourceOnlyModel
    : public def::AbstractModel<SourceOnlyModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<SourceOnlyModel>,
      public def::FlowBoundaries<SourceOnlyModel>,
      public def::FVDefaultBoundaries<SourceOnlyModel>,
      public def::AD<SourceOnlyModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.25 * pos[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const { return {1.0 + 0.25 * pos[0]}; }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = 0.0;
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = 0.0;
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> &pos, const Solution & /*sol*/) const
  {
    s_i[0] = 0.5 + pos[0];
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                       [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }
};

class ThirdDerivativeDiffusionModel
    : public def::AbstractModel<ThirdDerivativeDiffusionModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<ThirdDerivativeDiffusionModel>,
      public def::FlowBoundaries<ThirdDerivativeDiffusionModel>,
      public def::FVDefaultBoundaries<ThirdDerivativeDiffusionModel>,
      public def::AD<ThirdDerivativeDiffusionModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = solution(pos)[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const
  {
    const double x = pos[0];
    return {1.0 + 0.2 * x - 0.1 * x * x + 0.5 * x * x * x};
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = 0.0;
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &third_derivatives = get<"fe_third_derivatives">(sol);
    F_i[0][0] = NT(0.05) * third_derivatives[0][0][0][0];
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    s_i[0] = 0.0;
  }

  template <typename NT, typename Vector, typename VectorDot>
  void mass(std::array<NT, 1> &m_i, const Point<1> & /*pos*/, const Vector & /*u*/, const VectorDot & /*dt_u*/) const
  {
    m_i[0] = 0.0;
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                       [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }
};

class GradientDependentKTModel
    : public def::AbstractModel<GradientDependentKTModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<GradientDependentKTModel>,
      public def::FlowBoundaries<GradientDependentKTModel>,
      public def::FVDefaultBoundaries<GradientDependentKTModel>,
      public def::AD<GradientDependentKTModel>
{
public:
  static constexpr double kt_diffusion = 0.4;
  static constexpr double separate_diffusion = 0.7;

  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = solution(pos)[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const { return {1.0 + 0.2 * pos[0] + 0.03 * pos[0] * pos[0]}; }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    const auto &grad_u = get<"fe_derivatives">(sol);
    F_i[0][0] += NT(0.5) * u[0] * u[0] - NT(kt_diffusion) * u[0] * grad_u[0][0];
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &grad_u = get<"fe_derivatives">(sol);
    F_i[0][0] += -NT(separate_diffusion) * grad_u[0][0];
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    s_i[0] = NT(0);
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                       [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }
};

class DiffusiveAffineBoundary2DModel
    : public def::AbstractModel<DiffusiveAffineBoundary2DModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<DiffusiveAffineBoundary2DModel>,
      public def::FlowBoundaries<DiffusiveAffineBoundary2DModel>,
      public def::FVDefaultBoundaries<DiffusiveAffineBoundary2DModel>,
      public def::AD<DiffusiveAffineBoundary2DModel>
{
public:
  template <typename Vector> void initial_condition(const Point<2> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.2 * pos[0] + 0.35 * pos[1];
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution &sol) const
  {
    const auto &fe_derivatives = get<1>(sol);
    F_i[0] = -2.5 * fe_derivatives[0];
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    s_i[0] = NT(0.0);
  }
};

class DiffusiveNonAffineTangentialBoundary2DModel
    : public def::AbstractModel<DiffusiveNonAffineTangentialBoundary2DModel,
                                ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<DiffusiveNonAffineTangentialBoundary2DModel>,
      public def::FlowBoundaries<DiffusiveNonAffineTangentialBoundary2DModel>,
      public def::AD<DiffusiveNonAffineTangentialBoundary2DModel>
{
public:
  template <typename Vector> void initial_condition(const Point<2> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.2 * pos[0] + 0.35 * pos[1];
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution &sol) const
  {
    const auto &fe_derivatives = get<1>(sol);
    F_i[0] = -2.5 * fe_derivatives[0];
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    s_i[0] = NT(0.0);
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 2);
    static_assert(n_components == 1);
    using namespace def::BoundaryStencilIndex;

    unsigned int axis = 0;
    if (std::abs(x_face[1] - x_stencil[physical_cell][1]) > std::abs(x_face[0] - x_stencil[physical_cell][0])) axis = 1;

    const bool lower_boundary = x_face[axis] <= x_stencil[physical_cell][axis];
    const double delta = lower_boundary ? (x_stencil[upper_inner][axis] - x_stencil[physical_cell][axis])
                                        : (x_stencil[physical_cell][axis] - x_stencil[lower_inner][axis]);
    const unsigned int tangential_axis = 1U - axis;
    const NT shift = (lower_boundary ? NT(3.0) : NT(-2.0)) + NT(0.75) * NT(x_stencil[physical_cell][tangential_axis]);

    if (lower_boundary) {
      x_stencil[lower_inner] = x_stencil[physical_cell];
      x_stencil[lower_outer] = x_stencil[physical_cell];
      x_stencil[lower_inner][axis] = x_stencil[physical_cell][axis] - delta;
      x_stencil[lower_outer][axis] = x_stencil[physical_cell][axis] - 2.0 * delta;
      u_stencil[lower_inner][0] = NT(2.0) * u_stencil[physical_cell][0] - u_stencil[upper_inner][0] + shift;
      u_stencil[lower_outer][0] =
          NT(3.0) * u_stencil[physical_cell][0] - NT(2.0) * u_stencil[upper_inner][0] + NT(2.0) * shift;
      return true;
    }

    x_stencil[upper_inner] = x_stencil[physical_cell];
    x_stencil[upper_outer] = x_stencil[physical_cell];
    x_stencil[upper_inner][axis] = x_stencil[physical_cell][axis] + delta;
    x_stencil[upper_outer][axis] = x_stencil[physical_cell][axis] + 2.0 * delta;
    u_stencil[upper_inner][0] = NT(2.0) * u_stencil[physical_cell][0] - u_stencil[lower_inner][0] + shift;
    u_stencil[upper_outer][0] =
        NT(3.0) * u_stencil[physical_cell][0] - NT(2.0) * u_stencil[lower_inner][0] + NT(2.0) * shift;
    return true;
  }
};

/**
 * Compares the analytic Jacobian from the KT assembler against a central-difference
 * finite-difference Jacobian of residual(). Detects missing terms in jacobian() —
 * in particular the diffusion flux Jacobian which is currently not implemented.
 *
 * With ModelBurgersTravelingWaveKT (nu=100, dx=0.1): missing diffusion entries differ
 * by O(nu/dx) = O(1000), far above the relative tolerance of 1e-4.
 */
TEST_CASE("KT Jacobian matches FD Jacobian for traveling wave model (detects missing diffusion)", "[FV][KT]")
{
  using Model = Testing::ModelBurgersTravelingWaveKT<1>;
  using NumberType = double;
  // Serial on purpose: this tests serial linear algebra, and a plain RectangularMesh is partitioned in an MPI build.
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, NumberType>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  Testing::PhysicalParameters p_prm;
  const ConfigTree json = make_json();

  Model model(p_prm);
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());

  VectorType sol_dot(n_dofs); // zero — excluded via weight_mass=0 / alpha=0

  // --- Analytic Jacobian (spatial part only: weight=1, alpha=0, beta=0) ---
  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> J_analytic(sp);
  assembler.jacobian(J_analytic, sol, 1.0, sol_dot, /*alpha=*/0.0, /*beta=*/0.0);

  // --- Finite-difference Jacobian via central differences of residual() ---
  // residual(r, u, weight=1, sol_dot, weight_mass=0) gives just the spatial flux/source residual
  const double eps = 1e-8;
  std::vector<std::vector<double>> J_fd(n_dofs, std::vector<double>(n_dofs, 0.0));

  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_p = sol, u_m = sol;
    u_p[j] += eps;
    u_m[j] -= eps;

    VectorType r_p(n_dofs), r_m(n_dofs);
    assembler.residual(r_p, u_p, 1.0, sol_dot, /*weight_mass=*/0.0);
    assembler.residual(r_m, u_m, 1.0, sol_dot, /*weight_mass=*/0.0);

    for (int i = 0; i < n_dofs; ++i)
      J_fd[i][j] = (r_p[i] - r_m[i]) / (2.0 * eps);
  }

  const int boundary_layer = 3;
  const double tol = 1e-4;
  bool pass = true;
  for (int i = 0; i < n_dofs; ++i) {
    for (int j = 0; j < n_dofs; ++j) {
      // if (j < boundary_layer || j >= n_dofs - boundary_layer) continue;

      const double analytic = J_analytic.el(i, j);
      const double fd = J_fd[i][j];
      const double err = std::abs(analytic - fd);
      const double scale = std::max(1.0, std::abs(fd));
      if (err > tol * scale) {
        std::cout << "Jacobian mismatch at [" << i << "," << j << "]: "
                  << "analytic=" << analytic << "  fd=" << fd << "  rel_err=" << err / scale << "\n";
        pass = false;
      }
    }
  }
  REQUIRE(pass);
}

TEST_CASE("KT Jacobian matches FD Jacobian for third-derivative diffusion model on interior rows", "[FV][KT]")
{
  using Model = ThirdDerivativeDiffusionModel;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, NumberType>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  const ConfigTree json = make_json();

  Model model;
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 8);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 0.7 + 0.11 * static_cast<double>(i) - 0.015 * static_cast<double>(i * i) +
             0.001 * static_cast<double>(i * i * i);

  VectorType sol_dot(n_dofs);

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> J_analytic(sp);
  assembler.jacobian(J_analytic, sol, 1.0, sol_dot, 0.0, 0.0);

  const double eps = 1e-8;
  std::vector<std::vector<double>> J_fd(n_dofs, std::vector<double>(n_dofs, 0.0));
  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_p = sol, u_m = sol;
    u_p[j] += eps;
    u_m[j] -= eps;
    VectorType r_p(n_dofs), r_m(n_dofs);
    assembler.residual(r_p, u_p, 1.0, sol_dot, 0.0);
    assembler.residual(r_m, u_m, 1.0, sol_dot, 0.0);
    for (int i = 0; i < n_dofs; ++i)
      J_fd[i][j] = (r_p[i] - r_m[i]) / (2.0 * eps);
  }

  const double tol = 2e-4;
  bool pass = true;
  for (int i = 3; i < n_dofs - 3; ++i) {
    for (int j = 0; j < n_dofs; ++j) {
      const double analytic = J_analytic.el(i, j);
      const double fd = J_fd[i][j];
      const double err = std::abs(analytic - fd);
      const double scale = std::max(1.0, std::abs(fd));
      if (err > tol * scale) {
        std::cout << "Third-derivative Jacobian mismatch at [" << i << "," << j << "]: "
                  << "analytic=" << analytic << "  fd=" << fd << "  rel_err=" << err / scale << "\n";
        pass = false;
      }
    }
  }
  REQUIRE(pass);
}

/**
 * Sanity check: the same FD consistency test for the pure-advection Burgers model
 * (no diffusion flux). The current Jacobian should already pass this.
 */
TEST_CASE("KT Jacobian matches FD Jacobian for pure advection Burgers model", "[FV][KT]")
{
  using Model = Testing::ModelBurgersKT<1>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, NumberType>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = 0.0;
  p_prm.initial_x1[0] = 1.0;
  const ConfigTree json = make_json();

  Model model(p_prm);
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());

  VectorType sol_dot(n_dofs);

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> J_analytic(sp);
  assembler.jacobian(J_analytic, sol, 1.0, sol_dot, 0.0, 0.0);

  const double eps = 1e-8;
  std::vector<std::vector<double>> J_fd(n_dofs, std::vector<double>(n_dofs, 0.0));
  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_p = sol, u_m = sol;
    u_p[j] += eps;
    u_m[j] -= eps;
    VectorType r_p(n_dofs), r_m(n_dofs);
    assembler.residual(r_p, u_p, 1.0, sol_dot, 0.0);
    assembler.residual(r_m, u_m, 1.0, sol_dot, 0.0);
    for (int i = 0; i < n_dofs; ++i)
      J_fd[i][j] = (r_p[i] - r_m[i]) / (2.0 * eps);
  }

  const double tol = 1e-4;
  bool pass = true;
  for (int i = 0; i < n_dofs; ++i) {
    for (int j = 0; j < n_dofs; ++j) {

      const double analytic = J_analytic.el(i, j);
      const double fd = J_fd[i][j];
      const double err = std::abs(analytic - fd);
      const double scale = std::max(1.0, std::abs(fd));
      if (err > tol * scale) {
        std::cout << "Jacobian mismatch at [" << i << "," << j << "]: "
                  << "analytic=" << analytic << "  fd=" << fd << "  rel_err=" << err / scale << "\n";
        pass = false;
      }
    }
  }
  REQUIRE(pass);
}

TEST_CASE("KT gradient-dependent flux and separate diffusion Jacobian match FD", "[FV][KT][gradient]")
{
  using Model = GradientDependentKTModel;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, NumberType>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  const ConfigTree json = make_json();
  Model model;
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 5);

  // Exercise nonconstant reconstructed states and gradients while keeping dF/du away from the |lambda| kink.
  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 1.1 + 0.025 * static_cast<double>(i) + 0.001 * static_cast<double>(i * i);

  VectorType sol_dot(n_dofs);
  sol_dot = 0.0;

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> analytic(sp);
  assembler.jacobian(analytic, sol, 1.0, sol_dot, 0.0, 0.0);

  const double eps = 1.0e-7;
  const double tolerance = 3.0e-4;
  bool pass = true;
  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_plus = sol;
    VectorType u_minus = sol;
    u_plus[j] += eps;
    u_minus[j] -= eps;

    VectorType residual_plus(n_dofs);
    VectorType residual_minus(n_dofs);
    assembler.residual(residual_plus, u_plus, 1.0, sol_dot, 0.0);
    assembler.residual(residual_minus, u_minus, 1.0, sol_dot, 0.0);

    for (int i = 0; i < n_dofs; ++i) {
      const double fd = (residual_plus[i] - residual_minus[i]) / (2.0 * eps);
      const double actual = sp.exists(i, j) ? analytic.el(i, j) : 0.0;
      const double scale = std::max(1.0, std::abs(fd));
      if (std::abs(actual - fd) > tolerance * scale) {
        std::cout << "Gradient-dependent KT Jacobian mismatch at [" << i << "," << j << "]: analytic=" << actual
                  << " fd=" << fd << " rel_err=" << std::abs(actual - fd) / scale << "\n";
        pass = false;
      }
    }
  }
  REQUIRE(pass);
}

TEST_CASE("KT Jacobian for x-dependent source-only model is diagonal", "[FV][KT]")
{
  using Model = SourceOnlyModel;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, NumberType>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  const ConfigTree json = make_json();

  Model model;
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());

  VectorType sol_dot(n_dofs);
  VectorType residual(n_dofs);
  assembler.residual(residual, sol, 1.0, sol_dot, 0.0);

  bool has_nonzero_source_residual = false;
  for (int i = 0; i < n_dofs; ++i)
    has_nonzero_source_residual = has_nonzero_source_residual || std::abs(residual[i]) > 1e-12;
  REQUIRE(has_nonzero_source_residual);

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> jacobian(sp);
  assembler.jacobian(jacobian, sol, 1.0, sol_dot, 0.0, 0.0);

  for (int i = 0; i < n_dofs; ++i)
    for (int j = 0; j < n_dofs; ++j)
      CHECK(jacobian.el(i, j) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("KT first-order Jacobian strategy does not use reconstruction-neighbor columns", "[FV][KT]")
{
  using Model = Testing::ModelBurgersKT<1>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, NumberType>;
  using ResidualReconstructor = def::TVDReconstructor<Discretization::dim, def::MinModLimiter, NumberType>;
  using JacobianReconstructor = def::FirstOrderReconstructor<Discretization::dim, NumberType>;
  using ExactAssembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using ApproxJacobianAssembler =
      FV::KurganovTadmor::Assembler<Discretization, Model, ResidualReconstructor,
                                    FV::KurganovTadmor::MaxEigenvalueWaveSpeed, JacobianReconstructor>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = 0.0;
  p_prm.initial_x1[0] = 1.0;
  const ConfigTree json = make_json();

  Model model(p_prm);
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  ExactAssembler exact_assembler(discretization, model, json);
  ApproxJacobianAssembler approx_assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 5);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 1.0 + 0.1 * static_cast<double>(i) + 0.03 * static_cast<double>(i * i);

  VectorType sol_dot(n_dofs);
  VectorType residual_exact(n_dofs);
  VectorType residual_approx(n_dofs);
  exact_assembler.residual(residual_exact, sol, 1.0, sol_dot, 0.0);
  approx_assembler.residual(residual_approx, sol, 1.0, sol_dot, 0.0);

  for (int i = 0; i < n_dofs; ++i)
    CHECK(residual_approx[i] == Catch::Approx(residual_exact[i]).margin(1e-12));

  const SparsityPattern &sp = approx_assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> jacobian(sp);
  approx_assembler.jacobian(jacobian, sol, 1.0, sol_dot, 0.0, 0.0);

  const int row = n_dofs / 2;
  REQUIRE(row >= 2);
  REQUIRE(row + 2 < n_dofs);

  CHECK_FALSE(sp.exists(row, row - 2));
  CHECK_FALSE(sp.exists(row, row + 2));
  CHECK(jacobian.el(row, row - 2) == Catch::Approx(0.0).margin(1e-12));
  CHECK(jacobian.el(row, row + 2) == Catch::Approx(0.0).margin(1e-12));

  const bool has_immediate_neighbor_contribution =
      (sp.exists(row, row - 1) && std::abs(jacobian.el(row, row - 1)) > 1e-12) ||
      (sp.exists(row, row + 1) && std::abs(jacobian.el(row, row + 1)) > 1e-12);
  CHECK(has_immediate_neighbor_contribution);
}

TEST_CASE("KT TVD Jacobian strategy keeps reconstruction-neighbor columns", "[FV][KT]")
{
  using Model = Testing::ModelBurgersKT<1>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, NumberType>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = 0.0;
  p_prm.initial_x1[0] = 1.0;
  const ConfigTree json = make_json();

  Model model(p_prm);
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 5);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 1.0 + 0.2 * static_cast<double>(i) - 0.01 * static_cast<double>(i * i);

  VectorType sol_dot(n_dofs);
  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> jacobian(sp);
  assembler.jacobian(jacobian, sol, 1.0, sol_dot, 0.0, 0.0);

  const int row = n_dofs / 2;
  REQUIRE(row >= 2);
  REQUIRE(row + 2 < n_dofs);

  CHECK(sp.exists(row, row - 2));
  CHECK(sp.exists(row, row + 2));

  const bool has_reconstruction_neighbor_contribution =
      std::abs(jacobian.el(row, row - 2)) > 1e-12 || std::abs(jacobian.el(row, row + 2)) > 1e-12;
  CHECK(has_reconstruction_neighbor_contribution);
}

TEST_CASE("KT 2D Jacobian matches FD Jacobian for diagonal Burgers model", "[FV][KT][2d]")
{
  using Model = Testing::ModelBurgers2DKT;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<2>, NumberType>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = 1.0;
  p_prm.initial_x1[0] = 0.2;
  p_prm.initial_x2[0] = 0.35;
  const ConfigTree json = make_json_2d();

  Model model(p_prm);
  RectangularMeshSerial<2> mesh{Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 4);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 1.0 + 0.03 * static_cast<double>(i) + 0.002 * static_cast<double>(i * i);

  VectorType sol_dot(n_dofs);

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> J_analytic(sp);
  assembler.jacobian(J_analytic, sol, 1.0, sol_dot, 0.0, 0.0);

  const double eps = 1e-7;
  std::vector<std::vector<double>> J_fd(n_dofs, std::vector<double>(n_dofs, 0.0));
  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_p = sol, u_m = sol;
    u_p[j] += eps;
    u_m[j] -= eps;

    VectorType r_p(n_dofs), r_m(n_dofs);
    assembler.residual(r_p, u_p, 1.0, sol_dot, 0.0);
    assembler.residual(r_m, u_m, 1.0, sol_dot, 0.0);

    for (int i = 0; i < n_dofs; ++i)
      J_fd[i][j] = (r_p[i] - r_m[i]) / (2.0 * eps);
  }

  const double tol = 2e-4;
  bool pass = true;
  for (int i = 0; i < n_dofs; ++i) {
    for (int j = 0; j < n_dofs; ++j) {
      const double analytic = J_analytic.el(i, j);
      const double fd = J_fd[i][j];
      const double err = std::abs(analytic - fd);
      const double scale = std::max(1.0, std::abs(fd));
      if (err > tol * scale) {
        std::cout << "2D Jacobian mismatch at [" << i << "," << j << "]: "
                  << "analytic=" << analytic << "  fd=" << fd << "  rel_err=" << err / scale << "\n";
        pass = false;
      }
    }
  }
  REQUIRE(pass);
}

TEST_CASE("KT 2D boundary Jacobian matches FD for affine ghost diffusion", "[FV][KT][2d]")
{
  using Model = DiffusiveAffineBoundary2DModel;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<2>, NumberType>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  const ConfigTree json = make_json_2d();
  Model model;
  RectangularMeshSerial<2> mesh{Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 4);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 0.7 + 0.19 * static_cast<double>(i) + 0.003 * static_cast<double>(i * i);

  VectorType sol_dot(n_dofs);
  sol_dot = 0.0;

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> J_analytic(sp);
  assembler.jacobian(J_analytic, sol, 1.0, sol_dot, 0.0, 0.0);

  const auto &support_points = discretization.get_support_points();
  REQUIRE(static_cast<int>(support_points.size()) == n_dofs);
  double x_min = std::numeric_limits<double>::infinity();
  double x_max = -std::numeric_limits<double>::infinity();
  double y_min = std::numeric_limits<double>::infinity();
  double y_max = -std::numeric_limits<double>::infinity();
  for (const auto &point : support_points) {
    x_min = std::min(x_min, point[0]);
    x_max = std::max(x_max, point[0]);
    y_min = std::min(y_min, point[1]);
    y_max = std::max(y_max, point[1]);
  }
  const auto is_boundary_dof = [&](const int dof) {
    const auto &point = support_points[static_cast<std::size_t>(dof)];
    return std::abs(point[0] - x_min) < 1.0e-12 || std::abs(point[0] - x_max) < 1.0e-12 ||
           std::abs(point[1] - y_min) < 1.0e-12 || std::abs(point[1] - y_max) < 1.0e-12;
  };

  const double eps = 1e-7;
  bool pass = true;
  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_p = sol, u_m = sol;
    u_p[j] += eps;
    u_m[j] -= eps;

    VectorType r_p(n_dofs), r_m(n_dofs);
    assembler.residual(r_p, u_p, 1.0, sol_dot, 0.0);
    assembler.residual(r_m, u_m, 1.0, sol_dot, 0.0);

    for (int i = 0; i < n_dofs; ++i) {
      if (!is_boundary_dof(i) && !is_boundary_dof(j)) continue;

      const double analytic = sp.exists(i, j) ? J_analytic.el(i, j) : 0.0;
      const double fd = (r_p[i] - r_m[i]) / (2.0 * eps);
      const double err = std::abs(analytic - fd);
      const double scale = std::max(1.0, std::abs(fd));
      if (err > 2.0e-4 * scale) {
        std::cout << "2D boundary Jacobian mismatch at [" << i << "," << j << "]: "
                  << "analytic=" << analytic << "  fd=" << fd << "  rel_err=" << err / scale << "\n";
        pass = false;
      }
    }
  }
  REQUIRE(pass);
}

TEST_CASE("KT 2D boundary Jacobian uses model-owned tangential ghost derivatives", "[FV][KT][2d]")
{
  using Model = DiffusiveNonAffineTangentialBoundary2DModel;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<2>, NumberType>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  const ConfigTree json = make_json_2d();
  Model model;
  RectangularMeshSerial<2> mesh{Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 4);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 0.7 + 0.19 * static_cast<double>(i) + 0.003 * static_cast<double>(i * i);

  VectorType sol_dot(n_dofs);
  sol_dot = 0.0;

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> J_analytic(sp);
  assembler.jacobian(J_analytic, sol, 1.0, sol_dot, 0.0, 0.0);

  const auto &support_points = discretization.get_support_points();
  REQUIRE(static_cast<int>(support_points.size()) == n_dofs);
  double x_min = std::numeric_limits<double>::infinity();
  double x_max = -std::numeric_limits<double>::infinity();
  double y_min = std::numeric_limits<double>::infinity();
  double y_max = -std::numeric_limits<double>::infinity();
  for (const auto &point : support_points) {
    x_min = std::min(x_min, point[0]);
    x_max = std::max(x_max, point[0]);
    y_min = std::min(y_min, point[1]);
    y_max = std::max(y_max, point[1]);
  }
  const auto is_boundary_dof = [&](const int dof) {
    const auto &point = support_points[static_cast<std::size_t>(dof)];
    return std::abs(point[0] - x_min) < 1.0e-12 || std::abs(point[0] - x_max) < 1.0e-12 ||
           std::abs(point[1] - y_min) < 1.0e-12 || std::abs(point[1] - y_max) < 1.0e-12;
  };

  const double eps = 1e-7;
  bool pass = true;
  int missing_sparsity_mismatches = 0;
  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_p = sol, u_m = sol;
    u_p[j] += eps;
    u_m[j] -= eps;

    VectorType r_p(n_dofs), r_m(n_dofs);
    assembler.residual(r_p, u_p, 1.0, sol_dot, 0.0);
    assembler.residual(r_m, u_m, 1.0, sol_dot, 0.0);

    for (int i = 0; i < n_dofs; ++i) {
      if (!is_boundary_dof(i) && !is_boundary_dof(j)) continue;

      const bool in_sparsity = sp.exists(i, j);
      const double analytic = in_sparsity ? J_analytic.el(i, j) : 0.0;
      const double fd = (r_p[i] - r_m[i]) / (2.0 * eps);
      const double err = std::abs(analytic - fd);
      const double scale = std::max(1.0, std::abs(fd));
      if (err > 2.0e-4 * scale) {
        if (!in_sparsity) ++missing_sparsity_mismatches;
        std::cout << "2D non-affine boundary Jacobian mismatch at [" << i << "," << j << "]: "
                  << "row=(" << support_points[static_cast<std::size_t>(i)][0] << ", "
                  << support_points[static_cast<std::size_t>(i)][1] << ") "
                  << "column=(" << support_points[static_cast<std::size_t>(j)][0] << ", "
                  << support_points[static_cast<std::size_t>(j)][1] << ") "
                  << "in_sparsity=" << in_sparsity << " analytic=" << analytic << "  fd=" << fd
                  << "  rel_err=" << err / scale << "\n";
        pass = false;
      }
    }
  }
  CHECK(missing_sparsity_mismatches == 0);
  REQUIRE(pass);
}

/**
 * A source that reads "fe_derivatives" is nonlocal: it reaches the whole 2*dim stencil through the
 * reconstruction. Flux and diffusion flux are identically zero here, so the source is the ONLY
 * contribution to the Jacobian and a missing chain-rule term cannot hide behind the face blocks.
 *
 * The boundary rows are checked too: FVDefaultBoundaries extrapolates the ghosts from interior dofs,
 * so the gradient in the first and last cell depends on them.
 */
class GradientSourceKTModel
    : public def::AbstractModel<GradientSourceKTModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<GradientSourceKTModel>,
      public def::FlowBoundaries<GradientSourceKTModel>,
      public def::FVDefaultBoundaries<GradientSourceKTModel>,
      public def::AD<GradientSourceKTModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.2 * pos[0] + 0.4 * pos[0] * pos[0];
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = NT(0);
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = NT(0);
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> &pos, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    const auto &grad_u = get<"fe_derivatives">(sol);
    s_i[0] = NT(0.3) * u[0] * u[0] + NT(0.8) * grad_u[0][0] + NT(0.5) * u[0] * grad_u[0][0] + NT(0.25 * pos[0]);
  }
};

class GradientSource2DModel
    : public def::AbstractModel<GradientSource2DModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<GradientSource2DModel>,
      public def::FlowBoundaries<GradientSource2DModel>,
      public def::FVDefaultBoundaries<GradientSource2DModel>,
      public def::AD<GradientSource2DModel>
{
public:
  template <typename Vector> void initial_condition(const Point<2> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.2 * pos[0] + 0.35 * pos[1] + 0.3 * pos[0] * pos[0] + 0.15 * pos[1] * pos[1];
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<2> & /*pos*/, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    const auto &grad_u = get<"fe_derivatives">(sol);
    s_i[0] = NT(0.3) * u[0] * u[0] + NT(0.8) * grad_u[0][0] - NT(0.6) * grad_u[0][1] +
             NT(0.5) * u[0] * grad_u[0][1];
  }
};

/**
 * A source that reads "fe_hessians" (opted in via source_uses_hessians), nonlinear in the curvature so the
 * chain-rule term dS/du'' * du''/du_j is not a constant. As for the gradient source, flux and diffusion flux
 * vanish, so the source is the only contribution to the Jacobian.
 */
class HessianSourceKTModel
    : public def::AbstractModel<HessianSourceKTModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<HessianSourceKTModel>,
      public def::FlowBoundaries<HessianSourceKTModel>,
      public def::FVDefaultBoundaries<HessianSourceKTModel>,
      public def::AD<HessianSourceKTModel>
{
public:
  static constexpr bool source_uses_hessians = true;

  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.2 * pos[0] + 0.4 * pos[0] * pos[0];
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = NT(0);
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = NT(0);
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    const auto &hess_u = get<"fe_hessians">(sol);
    s_i[0] = NT(0.3) * u[0] * u[0] + NT(0.7) * hess_u[0][0][0] + NT(0.5) * u[0] * hess_u[0][0][0] +
             NT(0.2) * hess_u[0][0][0] * hess_u[0][0][0];
  }
};

class HessianSource2DModel
    : public def::AbstractModel<HessianSource2DModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<HessianSource2DModel>,
      public def::FlowBoundaries<HessianSource2DModel>,
      public def::FVDefaultBoundaries<HessianSource2DModel>,
      public def::AD<HessianSource2DModel>
{
public:
  static constexpr bool source_uses_hessians = true;

  template <typename Vector> void initial_condition(const Point<2> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.2 * pos[0] + 0.35 * pos[1] + 0.3 * pos[0] * pos[0] + 0.15 * pos[1] * pos[1];
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 2, NT>, 1> &F_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0] = Tensor<1, 2, NT>();
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<2> & /*pos*/, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    const auto &hess_u = get<"fe_hessians">(sol);
    s_i[0] = NT(0.3) * u[0] * u[0] + NT(0.7) * hess_u[0][0][0] - NT(0.4) * hess_u[0][1][1] +
             NT(0.5) * u[0] * hess_u[0][1][1] + NT(0.1) * hess_u[0][0][0] * hess_u[0][1][1];
  }
};

namespace
{
  /**
   * @brief Compare the assembled Jacobian against a central difference of residual() on every row.
   */
  template <typename Assembler, typename VectorType>
  bool jacobian_matches_fd(Assembler &assembler, const VectorType &sol, double eps, double tol, const char *label)
  {
    const int n_dofs = static_cast<int>(sol.size());
    VectorType sol_dot(n_dofs);

    const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
    SparseMatrix<double> J_analytic(sp);
    assembler.jacobian(J_analytic, sol, 1.0, sol_dot, 0.0, 0.0);

    bool pass = true;
    for (int j = 0; j < n_dofs; ++j) {
      VectorType u_p = sol, u_m = sol;
      u_p[j] += eps;
      u_m[j] -= eps;

      VectorType r_p(n_dofs), r_m(n_dofs);
      assembler.residual(r_p, u_p, 1.0, sol_dot, 0.0);
      assembler.residual(r_m, u_m, 1.0, sol_dot, 0.0);

      for (int i = 0; i < n_dofs; ++i) {
        const double fd = (r_p[i] - r_m[i]) / (2.0 * eps);
        const double analytic = J_analytic.el(i, j);
        const double err = std::abs(analytic - fd);
        const double scale = std::max(1.0, std::abs(fd));
        if (err > tol * scale) {
          std::cout << label << " Jacobian mismatch at [" << i << "," << j << "]: analytic=" << analytic
                    << "  fd=" << fd << "  rel_err=" << err / scale << "\n";
          pass = false;
        }
      }
    }
    return pass;
  }
} // namespace

TEST_CASE("KT gradient-dependent source Jacobian matches FD", "[FV][KT][gradient][source]")
{
  using Model = GradientSourceKTModel;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();
  const ConfigTree json = make_json();

  Model model;
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 8);

  // Strictly convex and increasing, so the minmod branch is the same on both sides of the FD
  // perturbation and the comparison is not sitting on the limiter's kink.
  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 1.0 + 0.05 * static_cast<double>(i) + 0.004 * static_cast<double>(i * i);

  REQUIRE(jacobian_matches_fd(assembler, sol, 1e-7, 2e-4, "gradient source"));
}

TEST_CASE("KT 2D gradient-dependent source Jacobian matches FD", "[FV][KT][gradient][source][2d]")
{
  using Model = GradientSource2DModel;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<2>, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();
  const ConfigTree json = make_json_2d();

  Model model;
  RectangularMeshSerial<2> mesh{Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 8);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 1.0 + 0.05 * static_cast<double>(i) + 0.004 * static_cast<double>(i * i);

  REQUIRE(jacobian_matches_fd(assembler, sol, 1e-7, 2e-4, "2D gradient source"));
}

TEST_CASE("KT hessian-dependent source Jacobian matches FD", "[FV][KT][hessian][source]")
{
  using Model = HessianSourceKTModel;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();
  const ConfigTree json = make_json();

  Model model;
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 8);

  // A cubic term on top of the convex profile, so the curvature varies from cell to cell.
  for (int i = 0; i < n_dofs; ++i) {
    const double x = static_cast<double>(i);
    sol[i] = 1.0 + 0.05 * x + 0.004 * x * x + 0.0007 * x * x * x;
  }

  REQUIRE(jacobian_matches_fd(assembler, sol, 1e-7, 2e-4, "hessian source"));
}

TEST_CASE("KT 2D hessian-dependent source Jacobian matches FD", "[FV][KT][hessian][source][2d]")
{
  using Model = HessianSource2DModel;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<2>, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();
  const ConfigTree json = make_json_2d();

  Model model;
  RectangularMeshSerial<2> mesh{Config::ConfigurationMesh<2>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 8);

  for (int i = 0; i < n_dofs; ++i) {
    const double x = static_cast<double>(i);
    sol[i] = 1.0 + 0.05 * x + 0.004 * x * x + 0.0007 * x * x * x;
  }

  REQUIRE(jacobian_matches_fd(assembler, sol, 1e-7, 2e-4, "2D hessian source"));
}

// A potential-flow-like model on the rho-symmetric boundary: a gradient-free advective flux and a diffusion
// flux that depends on rho and on the gradient, as for a pion loop 1/(1 + u) and a sigma loop
// 1/(1 + u + 2 rho u'). The flat profile has neighbouring one-sided slopes that differ by ~6e-9, i.e. it sits
// close to, but not on, a minmod kink -- as a potential does in its convexity-restored region.
template <bool with_advection, bool with_diffusion, bool flat = false>
class RhoSymmetricPotentialModel
    : public def::AbstractModel<RhoSymmetricPotentialModel<with_advection, with_diffusion, flat>,
                                ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::RhoSymmetricLinearExtrapolationBoundaries<
          RhoSymmetricPotentialModel<with_advection, with_diffusion, flat>>,
      public def::AD<RhoSymmetricPotentialModel<with_advection, with_diffusion, flat>>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    if constexpr (flat)
      values[0] = 0.5 + 1e-7 * pos[0] + 3e-8 * pos[0] * pos[0];
    else
      values[0] = 0.1 + 0.5 * pos[0] * pos[0];
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    if constexpr (with_advection) F_i[0][0] = NT(1.) / (NT(1.) + u[0]);
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> &pos, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    const auto &grad_u = get<"fe_derivatives">(sol);
    if constexpr (with_diffusion) F_i[0][0] = NT(1.) / (NT(1.) + u[0] + NT(2. * pos[0]) * grad_u[0][0]);
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    s_i[0] = NT(0);
  }
};

template <typename Model> static bool rho_symmetric_jacobian_matches_fd(const char *label, const double eps = 1.0e-7)
{
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<1>, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();
  const ConfigTree json = make_json();
  Model model;
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  VectorType sol_dot(n_dofs);

  SparseMatrix<double> analytic(assembler.get_sparsity_pattern_jacobian());
  assembler.jacobian(analytic, sol, 1.0, sol_dot, 0.0, 0.0);

  const double tolerance = 1.0e-5;
  bool pass = true;
  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_plus = sol, u_minus = sol;
    u_plus[j] += eps;
    u_minus[j] -= eps;
    VectorType r_plus(n_dofs), r_minus(n_dofs);
    assembler.residual(r_plus, u_plus, 1.0, sol_dot, 0.0);
    assembler.residual(r_minus, u_minus, 1.0, sol_dot, 0.0);
    for (int i = 0; i < n_dofs; ++i) {
      const double fd = (r_plus[i] - r_minus[i]) / (2.0 * eps);
      const double actual = assembler.get_sparsity_pattern_jacobian().exists(i, j) ? analytic.el(i, j) : 0.0;
      if (std::abs(actual - fd) > tolerance * std::max(1.0, std::abs(fd))) {
        std::cout << label << " Jacobian mismatch at [" << i << "," << j << "]: analytic=" << actual << " fd=" << fd
                  << "\n";
        pass = false;
      }
    }
  }
  return pass;
}

TEST_CASE("KT Jacobian matches FD on the rho-symmetric boundary", "[FV][KT][boundary]")
{
  SECTION("advection") { REQUIRE(rho_symmetric_jacobian_matches_fd<RhoSymmetricPotentialModel<true, false>>("adv")); }
  SECTION("diffusion") { REQUIRE(rho_symmetric_jacobian_matches_fd<RhoSymmetricPotentialModel<false, true>>("diff")); }
  SECTION("both") { REQUIRE(rho_symmetric_jacobian_matches_fd<RhoSymmetricPotentialModel<true, true>>("both")); }
  // The difference step must stay below the slope gap, or it straddles the kink itself.
  SECTION("near a limiter kink")
  {
    REQUIRE(rho_symmetric_jacobian_matches_fd<RhoSymmetricPotentialModel<true, true, true>>("flat", 1.0e-10));
  }
}
