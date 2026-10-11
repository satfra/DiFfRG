#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/timestepping/jacobian_alpha_update.hh>
#include <DiFfRG/timestepping/sundials_ida.hh>

#include <cmath>
#include <string>

using namespace DiFfRG;
using namespace dealii;

// When IDA only changes its step size or order, the Jacobian F_u + alpha F_udot is changed to the new alpha instead of
// reassembled (IDADriver::update_jacobian_alpha, JacobianAlphaUpdate). That is exact only if the assembled Jacobian is
// linear in alpha with slope jacobian_mass, which is checked here for each assembler, with a state-dependent mass.

namespace
{
  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>;

  /// (1 + u^2 / 2) du/dt + d/dx (u^2 / 2) = 0
  class Model : public def::AbstractModel<Model, Components>,
                public def::Time,
                public def::LLFFlux<Model>,
                public def::FlowBoundaries<Model>,
                public def::AD<Model>
  {
  public:
    template <typename Vector> void initial_condition(const Point<1> &x, Vector &values) const
    {
      values[0] = 1. + 0.5 * std::sin(3. * x[0]);
    }

    template <int dim, typename NT, typename Vector, typename Vector_dot, size_t n>
    void mass(std::array<NT, n> &m, const Point<dim> &, const Vector &u, const Vector_dot &u_dot) const
    {
      m[0] = (1. + 0.5 * u[0] * u[0]) * u_dot[0];
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, 1, NT>, 1> &F, const Point<1> &, const Solution &sol) const
    {
      F[0][0] = 0.5 * powr<2>(get<"fe_functions">(sol)[0]);
    }

    /// KT: constant extrapolation into the ghost cells.
    template <int mdim, typename NT, size_t n>
    bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n> &u, def::BoundaryStencilPoints<mdim> &x,
                                const Point<mdim> &x_face) const
    {
      using namespace def::BoundaryStencilIndex;
      const Tensor<1, mdim> to_ghost = 2. * (x_face - x[physical_cell]);
      const bool lower = to_ghost[0] < 0.;
      const auto [inner, outer] = lower ? std::pair{lower_inner, lower_outer} : std::pair{upper_inner, upper_outer};
      x[inner] = x[physical_cell] + to_ghost;
      x[outer] = x[physical_cell] + 2. * to_ghost;
      u[inner][0] = u[outer][0] = u[physical_cell][0];
      return true;
    }
  };

  ConfigTree make_config(const uint fe_order, const bool reuse_jacobian = true)
  {
    return json::value({{"physical", {{"Lambda", 1.}}},
                        {"discretization",
                         {{"fe_order", fe_order},
                          {"overintegration", 0},
                          {"EoM_abs_tol", 1e-10},
                          {"EoM_max_iter", 0},
                          {"grid", {{"x_grid", "0:0.05:1"}, {"refine", 0}}}}},
                        {"timestepping",
                         {{"final_time", 0.5},
                          {"output_dt", 0.1},
                          {"implicit",
                           {{"dt", 1e-3},
                            {"minimal_dt", 1e-10},
                            {"maximal_dt", 0.1},
                            {"abs_tol", 1e-10},
                            {"rel_tol", 1e-8},
                            {"reuse_jacobian", reuse_jacobian}}}}},
                        {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", false}}}});
  }

  template <typename Discretization, typename Assembler, typename FlowingVariables>
  void require_jacobian_linear_in_alpha(const uint fe_order)
  {
    using VectorType = typename Discretization::VectorType;
    using SparseMatrixType = typename Discretization::SparseMatrixType;
    const auto config = make_config(fe_order);
    Model model;
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
    Discretization discretization(mesh, config);
    Assembler assembler(discretization, model, config);
    FlowingVariables state(discretization);
    state.interpolate(model);
    const VectorType u = state.spatial_data();
    VectorType u_dot(u);
    for (std::size_t i = 0; i < u_dot.size(); ++i)
      u_dot[i] = std::cos(0.3 * i);

    const auto jacobian = [&](const double alpha) {
      SparseMatrixType J;
      assembler.reinit_matrix(J);
      J = 0;
      assembler.jacobian(J, u, 1., u_dot, alpha, 1.);
      return J;
    };
    const double alpha_0 = 1.3, alpha_1 = 41.;
    auto updated = jacobian(alpha_0);
    const auto expected = jacobian(alpha_1);

    JacobianAlphaUpdate<VectorType, SparseMatrixType, 1> update(assembler);
    REQUIRE(!update.update(updated, alpha_1)); // nothing remembered yet
    update.remember(0., u, u_dot, alpha_0);
    REQUIRE(update.update(updated, alpha_1));

    updated.add(-1., expected);
    CHECK(updated.frobenius_norm() <= 1e-12 * expected.frobenius_norm());
  }
} // namespace

TEST_CASE("The Jacobian is linear in alpha, with slope jacobian_mass", "[timestepping][sundials_ida]")
{
  DiFfRG::Init();
  SECTION("CG")
  {
    using D = CG::Discretization<Model, RectangularMeshSerial<1>>;
    require_jacobian_linear_in_alpha<D, CG::Assembler<D>, FE::FlowingVariables<D>>(1);
  }
  SECTION("DG")
  {
    using D = DG::Discretization<Model, RectangularMeshSerial<1>>;
    require_jacobian_linear_in_alpha<D, DG::Assembler<D>, FE::FlowingVariables<D>>(1);
  }
  SECTION("KT")
  {
    using D = FV::Discretization<Model, RectangularMeshSerial<1>, double>;
    require_jacobian_linear_in_alpha<D, FV::KurganovTadmor::Assembler<D, Model>, FV::FlowingVariables<D>>(0);
  }
}

namespace
{
  using Discretization = CG::Discretization<Model, RectangularMeshSerial<1>>;
  using Assembler = CG::Assembler<Discretization>;

  struct Result {
    typename Discretization::VectorType u;
    uint jacobians;
  };

  Result run(const bool reuse_jacobian)
  {
    const auto config = make_config(1, reuse_jacobian);
    const auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "ida_reuse", "ida_reuse");
    Model model;
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
    Discretization discretization(mesh, config);
    Assembler assembler(discretization, model, config);
    OutputSession<Assembler> data_out(path, config);
    NoAdaptivity<typename Discretization::VectorType> adaptor;
    TimeStepperSUNDIALS_IDA<Assembler> stepper(config, assembler, data_out, adaptor);
    FE::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    stepper.run(state, 0., 0.5);
    return {state.spatial_data(), assembler.num_jacobians()};
  }
} // namespace

TEST_CASE("IDA with Jacobian reuse gives the same result with fewer assemblies", "[timestepping][sundials_ida]")
{
  DiFfRG::Init();
  const auto reused = run(true);
  const auto assembled = run(false);
  CAPTURE(reused.jacobians, assembled.jacobians);
  CHECK(reused.jacobians < assembled.jacobians);
  auto deviation = reused.u;
  deviation -= assembled.u;
  CHECK(deviation.linfty_norm() <= 1e-6 * assembled.u.linfty_norm());
}
