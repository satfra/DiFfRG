#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <boilerplate/models.hh>

#include <cmath>

using namespace DiFfRG;
using namespace dealii;

// On a locally refined mesh the hanging-node entries of a solution vector are not unknowns: they follow from their
// masters. The timesteppers never set them, so the CG assembler must not read them.

TEST_CASE("CG assembly ignores the hanging-node entries of the solution", "[discretization][cg]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  using Model = Testing::ModelBurgers<dim>;
  using Discretization = CG::Discretization<Model, RectangularMeshSerial<dim>>;
  using Assembler = CG::Assembler<Discretization>;

  const ConfigTree config(json::value({{"physical", {}},
                                       {"discretization",
                                        {{"fe_order", 2},
                                         {"overintegration", 0},
                                         {"EoM_abs_tol", 1e-10},
                                         {"EoM_max_iter", 0},
                                         {"grid", {{"x_grid", "0:0.25:1"}, {"y_grid", "0:0.25:1"}, {"refine", 0}}}}},
                                       {"output", {{"verbosity", 0}}}}));
  Testing::PhysicalParameters prm;
  prm.initial_x0 = {{0.2, 0., 0.}};
  prm.initial_x1 = {{1., 0., 0.}};
  prm.initial_x2 = {{0.5, 0., 0.}};
  Model model(prm);
  RectangularMeshSerial<dim> mesh{Config::ConfigurationMesh<dim>(config)};
  Discretization discretization(mesh, config);
  Assembler assembler(discretization, model, config);

  // Refine one cell: its neighbours get hanging nodes.
  discretization.get_triangulation().begin_active()->set_refine_flag();
  discretization.get_triangulation().execute_coarsening_and_refinement();
  discretization.reinit();
  assembler.reinit();
  const auto &constraints = discretization.get_constraints();
  REQUIRE(constraints.n_constraints() > 0);

  FE::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  typename Discretization::VectorType u(state.spatial_data()), u_dot(u.size());
  constraints.distribute(u);
  for (size_t i = 0; i < u.size(); ++i)
    u_dot[i] = std::sin(0.7 * i);
  constraints.distribute(u_dot);

  // The same state, with garbage in the hanging-node entries.
  auto u_stale = u, u_dot_stale = u_dot;
  for (const auto &line : constraints.get_lines()) {
    u_stale[line.index] += 0.3;
    u_dot_stale[line.index] -= 0.7;
  }

  const auto residual = [&](const auto &v, const auto &v_dot) {
    typename Discretization::VectorType r(v.size());
    assembler.residual(r, v, 1., v_dot, 1.);
    return r;
  };
  auto deviation = residual(u_stale, u_dot_stale);
  const auto reference = residual(u, u_dot);
  deviation -= reference;
  REQUIRE(reference.linfty_norm() > 0.);
  CHECK(deviation.linfty_norm() <= 1e-12 * reference.linfty_norm());

  SparseMatrix<double> J(assembler.get_sparsity_pattern_jacobian()), J_stale(assembler.get_sparsity_pattern_jacobian());
  assembler.jacobian(J, u, 1., u_dot, 1., 1.);
  assembler.jacobian(J_stale, u_stale, 1., u_dot_stale, 1., 1.);
  J_stale.add(-1., J);
  CHECK(J_stale.linfty_norm() <= 1e-12 * J.linfty_norm());

  // The constrained dofs are algebraic for IDA.
  const auto differential = assembler.get_differential_indices();
  for (const auto &line : constraints.get_lines())
    CHECK_FALSE(differential.is_element(line.index));
}
