#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/kt_models.hh>
#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/mesh/h_adaptivity.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>

#include <type_traits>
#include <vector>

// Distributed assembly: does an n-rank residual equal the serial one?
//
// This is the gate that catches the failure mode with no error message. On a partitioned mesh
// an interior face is visited only when *both* neighbours are locally owned unless
// assemble_ghost_faces_once is set, so a face between an owned and a ghost cell is assembled by
// neither rank and every partition-boundary flux silently vanishes. The answer stays finite,
// smooth and plausible -- it is just wrong.
//
// The comparison is made inside one process: every rank builds a serial-policy discretization
// alongside the distributed one and assembles both. That keeps the test to a couple of seconds
// (no timestepping, no output, no cross-run file comparison) and removes the serial reference
// as a source of drift.
//
// The file must contribute at least one test case in a serial build: catch_discover_tests
// registers a <target>_NOT_BUILT placeholder that is only replaced once cases are discovered.
// The DG refinement-indicator case below is that, and is a regression test in its own right.

namespace
{
  using namespace DiFfRG;

  /// 16 cells on [0, 1] at fe_order 2 -- small enough to stay fast at 4 ranks, big enough that
  /// every rank owns several cells and there are real partition boundaries to get wrong.
  ///
  /// The integer literals here matter: boost::json stores an `unsigned` in a different arm than
  /// an `int`, and ConfigTree::get_uint used to throw on the unsigned one, so passing fe_order
  /// as an `unsigned` made the key silently unreadable and the discretization fell back to its
  /// default order. That is fixed in config_tree.cc now, but write plain ints regardless.
  ConfigTree make_json(const int adapt_level = 0)
  {
    return json::value(
        {{"physical", {}},
         {"integration", {{"x_quadrature_order", 8}, {"x_extent_tolerance", 1e-3}}},
         {"discretization",
          {{"fe_order", 2},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", "0:0.0625:1"}, {"y_grid", "0:0.5:1"}, {"z_grid", "0:0.5:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1e-1},
             {"level", adapt_level},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 5e-2}}}}},
         {"timestepping", {{"final_time", 1.}, {"output_dt", 1e-1}}},
         {"output", {{"live_plot", false}, {"verbosity", 0}}}});
  }

  /// u(x) = 0.2 + x, i.e. a state that is neither zero nor constant.
  ///
  /// PhysicalParameters holds four std::array<double, 3>, so the obvious
  /// `PhysicalParameters{0., 1.}` does NOT mean "x0 = 0, x1 = 1": brace elision fills
  /// initial_x0 with {0, 1, 0} and leaves initial_x1 at zero, giving u == 0 everywhere. With a
  /// zero state ModelBurgers' flux is zero, the residual is zero, and a residual comparison
  /// compares zero against zero and passes no matter what the assembler does. Set the fields by
  /// name instead.
  Testing::PhysicalParameters nontrivial_parameters()
  {
    Testing::PhysicalParameters prm;
    prm.initial_x0 = {{0.2, 0., 0.}};
    prm.initial_x1 = {{1., 0., 0.}};
    return prm;
  }

  /// A model whose face_indicator is non-zero and whose cell_indicator is left at the default
  /// (zero), so refinement_indicator is non-zero if and only if faces were actually visited.
  template <uint dim> class FaceOnlyIndicatorModel : public Testing::ModelBurgers<dim>
  {
  public:
    using Testing::ModelBurgers<dim>::ModelBurgers;

    template <int mdim, typename NumberType, typename Solutions_s, typename Solutions_n>
    void face_indicator(std::array<NumberType, 2> &indicator, const Tensor<1, mdim> & /*normal*/,
                        const Point<mdim> & /*p*/, const Solutions_s & /*sol_s*/, const Solutions_n & /*sol_n*/) const
    {
      indicator[0] = 1.;
      indicator[1] = 1.;
    }
  };

  /// A model whose cell_indicator grows with x, so refine_and_coarsen_fixed_fraction selects a
  /// proper subset of the cells. ModelBurgers leaves cell_indicator at the default (zero), and a
  /// uniformly zero indicator refines nothing at all -- which would make the adaptivity test below
  /// pass without ever transferring a solution.
  template <uint dim> class CellIndicatorModel : public Testing::ModelBurgers<dim>
  {
  public:
    using Testing::ModelBurgers<dim>::ModelBurgers;

    template <int mdim, typename NumberType, typename Solution>
    void cell_indicator(NumberType &indicator, const Point<mdim> &p, const Solution & /*sol*/) const
    {
      indicator = p[0];
    }
  };
} // namespace

// A Discretization defaults its vector and matrix from the *mesh*, not from the build
// configuration. That is what makes RectangularMeshSerial<dim> sufficient on its own: a test
// compared against a serial reference has to get serial linear algebra in an MPI build tree too.
// When the pair defaulted to the build configuration, this pin was silently insufficient -- the
// mesh went serial while the vectors stayed PETSc, and the resulting static_assert named the
// mesh, not the vectors the caller had never mentioned.
static_assert(
    std::is_same_v<typename CG::Discretization<Testing::ModelBurgers<1>, RectangularMeshSerial<1>>::VectorType,
                   dealii::Vector<double>>,
    "A serial mesh must default to serial linear algebra.");
static_assert(
    std::is_same_v<typename FV::Discretization<Testing::ModelBurgersKT<1>, RectangularMeshSerial<1>>::SparseMatrixType,
                   dealii::SparseMatrix<double>>,
    "A serial mesh must default to a serial matrix.");

TEST_CASE("DG refinement indicator sees face contributions", "[discretization][dg]")
{
  DiFfRG::Init();
  using namespace dealii;
  constexpr uint dim = 1;
  using Model = FaceOnlyIndicatorModel<dim>;
  using DGDiscretization = DG::Discretization<Model, RectangularMeshSerial<dim>>;
  using Assembler = DG::Assembler<DGDiscretization>;

  const auto json = make_json();
  Model model(nontrivial_parameters());
  RectangularMeshSerial<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  DGDiscretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FE::FlowingVariables<DGDiscretization> initial_condition(discretization);
  initial_condition.interpolate(model);

  Vector<double> indicator(mesh.get_triangulation().n_active_cells());
  assembler.refinement_indicator(indicator, initial_condition.spatial_data());

  // Regression: DG's indicator once dropped the face term (a mesh_loop flag was missing), so
  // DG refined on the cell term alone. With a model whose only indicator contribution is the
  // face one, that bug makes every entry zero.
  REQUIRE(indicator.linfty_norm() > 0.);
}

#if defined(DEAL_II_WITH_MPI) && defined(DEAL_II_WITH_PETSC)

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>

#include <DiFfRG/discretization/common/solution_view.hh>

namespace
{
  template <uint dim> using ParallelMesh = RectangularMeshParallel<dim>;

  // The converse half of the rule above: a partitioned mesh must bring the distributed pair with
  // it, so ParallelMesh<dim> alone is a complete spelling.
  static_assert(std::is_same_v<typename CG::Discretization<Testing::ModelBurgers<1>, ParallelMesh<1>>::VectorType,
                               dealii::PETScWrappers::MPI::Vector>,
                "A partitioned mesh must default to distributed linear algebra.");

  /**
   * @brief Assemble one residual serially and one distributed, and require them to agree.
   *
   * Matching is done cell by cell rather than by global index: a DoFHandler on a
   * parallel::shared::Triangulation renumbers so that each rank owns a contiguous range, so
   * global index i denotes different nodes in the two handlers. The meshes are structurally
   * identical, so cells correspond by (level, index).
   */
  template <typename SerialAssembler, typename SerialDisc, typename ParallelAssembler, typename ParallelDisc,
            typename SerialVars, typename ParallelVars, typename Model>
  void require_distributed_residual_matches_serial(Model &model, const ConfigTree &json)
  {
    using namespace dealii;
    constexpr uint dim = SerialDisc::dim;

    RectangularMeshSerial<dim> serial_mesh{Config::ConfigurationMesh<dim>(json)};
    SerialDisc serial_disc(serial_mesh, json);
    SerialAssembler serial_assembler(serial_disc, model, json);
    SerialVars serial_ic(serial_disc);
    serial_ic.interpolate(model);
    typename SerialDisc::VectorType serial_residual(serial_ic.spatial_data());
    serial_residual = 0.;
    serial_assembler.residual(serial_residual, serial_ic.spatial_data(), 1., serial_ic.spatial_data(), 1.);

    ParallelMesh<dim> parallel_mesh{Config::ConfigurationMesh<dim>(json)};
    ParallelDisc parallel_disc(parallel_mesh, json);
    ParallelAssembler parallel_assembler(parallel_disc, model, json);
    ParallelVars parallel_ic(parallel_disc);
    parallel_ic.interpolate(model);

    using ParVector = typename ParallelDisc::VectorType;
    // The assemblers read every dof of a cell they own, including ones another rank owns, so the
    // state has to be handed over as a fully replicated view -- exactly what the timestepper does.
    SolutionView<ParVector> state;
    state.reinit(parallel_disc.get_locally_owned_dofs(), parallel_disc.get_locally_relevant_dofs(),
                 parallel_disc.get_communicator());
    state.refresh(parallel_ic.spatial_data());

    ParVector parallel_residual(parallel_ic.spatial_data());
    parallel_residual = 0.;
    parallel_assembler.residual(parallel_residual, state.get(), 1., state.get(), 1.);

    SolutionView<ParVector> gathered;
    gathered.reinit(parallel_disc.get_locally_owned_dofs(), parallel_disc.get_locally_relevant_dofs(),
                    parallel_disc.get_communicator());
    gathered.refresh(parallel_residual);

    const auto &serial_dh = serial_disc.get_dof_handler();
    const auto &parallel_dh = parallel_disc.get_dof_handler();
    REQUIRE(serial_dh.n_dofs() == parallel_dh.n_dofs());

    const double scale = std::max(serial_residual.linfty_norm(), 1e-300);
    std::vector<types::global_dof_index> serial_indices, parallel_indices;
    double worst = 0.;
    for (const auto &parallel_cell : parallel_dh.active_cell_iterators()) {
      const typename DoFHandler<dim>::active_cell_iterator serial_cell(
          &serial_dh.get_triangulation(), parallel_cell->level(), parallel_cell->index(), &serial_dh);
      const auto n = parallel_cell->get_fe().n_dofs_per_cell();
      serial_indices.resize(n);
      parallel_indices.resize(n);
      parallel_cell->get_dof_indices(parallel_indices);
      serial_cell->get_dof_indices(serial_indices);
      for (unsigned int k = 0; k < n; ++k)
        worst = std::max(worst, std::abs(gathered[parallel_indices[k]] - serial_residual[serial_indices[k]]));
    }

    // Guard against the test quietly becoming a no-op. A comparison against a zero reference
    // passes for every possible assembler bug, and that is exactly what happened here once
    // already -- the state was degenerate, so both residuals were identically zero and stripping
    // assemble_ghost_faces_once did not make the test fail. Require a reference with something
    // in it before believing the agreement below.
    REQUIRE(scale > 1e-8);

    // Distributed assembly sums off-process row contributions in MPI arrival order, so this is
    // not bitwise -- but a dropped partition-boundary flux is an O(1) error, not an O(eps) one.
    INFO("worst |distributed - serial| = " << worst << ", residual scale = " << scale);
    REQUIRE(worst < 1e-10 * scale);
  }

  /**
   * @brief Assemble one jacobian serially and one distributed, and require their action on the state,
   * J u, to agree. Matched cell by cell, as in require_distributed_residual_matches_serial.
   */
  template <typename SerialAssembler, typename SerialDisc, typename ParallelAssembler, typename ParallelDisc,
            typename SerialVars, typename ParallelVars, typename Model>
  void require_distributed_jacobian_matches_serial(Model &model, const ConfigTree &json)
  {
    using namespace dealii;
    constexpr uint dim = SerialDisc::dim;

    RectangularMeshSerial<dim> serial_mesh{Config::ConfigurationMesh<dim>(json)};
    SerialDisc serial_disc(serial_mesh, json);
    SerialAssembler serial_assembler(serial_disc, model, json);
    SerialVars serial_ic(serial_disc);
    serial_ic.interpolate(model);
    const auto &serial_u = serial_ic.spatial_data();
    typename SerialDisc::SparseMatrixType serial_jacobian(serial_assembler.get_sparsity_pattern_jacobian());
    serial_jacobian = 0.;
    serial_assembler.jacobian(serial_jacobian, serial_u, 1., serial_u, 1., 1.);
    typename SerialDisc::VectorType serial_action(serial_u);
    serial_jacobian.vmult(serial_action, serial_u);

    ParallelMesh<dim> parallel_mesh{Config::ConfigurationMesh<dim>(json)};
    ParallelDisc parallel_disc(parallel_mesh, json);
    ParallelAssembler parallel_assembler(parallel_disc, model, json);
    ParallelVars parallel_ic(parallel_disc);
    parallel_ic.interpolate(model);

    using ParVector = typename ParallelDisc::VectorType;
    SolutionView<ParVector> state;
    state.reinit(parallel_disc.get_locally_owned_dofs(), parallel_disc.get_locally_relevant_dofs(),
                 parallel_disc.get_communicator());
    state.refresh(parallel_ic.spatial_data());

    typename ParallelDisc::SparseMatrixType parallel_jacobian;
    parallel_assembler.reinit_matrix(parallel_jacobian);
    parallel_assembler.jacobian(parallel_jacobian, state.get(), 1., state.get(), 1., 1.);
    ParVector parallel_action(parallel_ic.spatial_data());
    parallel_jacobian.vmult(parallel_action, parallel_ic.spatial_data());

    SolutionView<ParVector> gathered;
    gathered.reinit(parallel_disc.get_locally_owned_dofs(), parallel_disc.get_locally_relevant_dofs(),
                    parallel_disc.get_communicator());
    gathered.refresh(parallel_action);

    const auto &serial_dh = serial_disc.get_dof_handler();
    const auto &parallel_dh = parallel_disc.get_dof_handler();
    REQUIRE(serial_dh.n_dofs() == parallel_dh.n_dofs());

    const double scale = std::max(serial_action.linfty_norm(), 1e-300);
    std::vector<types::global_dof_index> serial_indices, parallel_indices;
    double worst = 0.;
    for (const auto &parallel_cell : parallel_dh.active_cell_iterators()) {
      const typename DoFHandler<dim>::active_cell_iterator serial_cell(
          &serial_dh.get_triangulation(), parallel_cell->level(), parallel_cell->index(), &serial_dh);
      const auto n = parallel_cell->get_fe().n_dofs_per_cell();
      serial_indices.resize(n);
      parallel_indices.resize(n);
      parallel_cell->get_dof_indices(parallel_indices);
      serial_cell->get_dof_indices(serial_indices);
      for (unsigned int k = 0; k < n; ++k)
        worst = std::max(worst, std::abs(gathered[parallel_indices[k]] - serial_action[serial_indices[k]]));
    }

    REQUIRE(scale > 1e-8);
    INFO("worst |distributed - serial| = " << worst << ", J u scale = " << scale);
    REQUIRE(worst < 1e-10 * scale);
  }
} // namespace

TEST_CASE("CG residual is unchanged by distribution", "[discretization][cg][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using SerialDisc = CG::Discretization<Model, RectangularMeshSerial<dim>>;
  using ParallelDisc = CG::Discretization<Model, ParallelMesh<dim>>;
  Model model(nontrivial_parameters());
  require_distributed_residual_matches_serial<CG::Assembler<SerialDisc>, SerialDisc, CG::Assembler<ParallelDisc>,
                                              ParallelDisc, FE::FlowingVariables<SerialDisc>,
                                              FE::FlowingVariables<ParallelDisc>>(model, make_json());
}

TEST_CASE("CG jacobian is unchanged by distribution", "[discretization][cg][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using SerialDisc = CG::Discretization<Model, RectangularMeshSerial<dim>>;
  using ParallelDisc = CG::Discretization<Model, ParallelMesh<dim>>;
  Model model(nontrivial_parameters());
  require_distributed_jacobian_matches_serial<CG::Assembler<SerialDisc>, SerialDisc, CG::Assembler<ParallelDisc>,
                                              ParallelDisc, FE::FlowingVariables<SerialDisc>,
                                              FE::FlowingVariables<ParallelDisc>>(model, make_json());
}

TEST_CASE("DG residual is unchanged by distribution", "[discretization][dg][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using SerialDisc = DG::Discretization<Model, RectangularMeshSerial<dim>>;
  using ParallelDisc = DG::Discretization<Model, ParallelMesh<dim>>;
  Model model(nontrivial_parameters());
  require_distributed_residual_matches_serial<DG::Assembler<SerialDisc>, SerialDisc, DG::Assembler<ParallelDisc>,
                                              ParallelDisc, FE::FlowingVariables<SerialDisc>,
                                              FE::FlowingVariables<ParallelDisc>>(model, make_json());
}

/// Burgers with derivatives and hessians in its batches, so the DG gather and scatter of those run distributed.
template <uint dim> struct BurgersWithDerivatives : Testing::ModelBurgers<dim> {
  using Testing::ModelBurgers<dim>::ModelBurgers;
  static constexpr bool batch_reads_derivatives = true;
};

TEST_CASE("DG residual with derivatives is unchanged by distribution", "[discretization][dg][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = BurgersWithDerivatives<dim>;
  using SerialDisc = DG::Discretization<Model, RectangularMeshSerial<dim>>;
  using ParallelDisc = DG::Discretization<Model, ParallelMesh<dim>>;
  Model model(nontrivial_parameters());
  require_distributed_residual_matches_serial<DG::Assembler<SerialDisc>, SerialDisc, DG::Assembler<ParallelDisc>,
                                              ParallelDisc, FE::FlowingVariables<SerialDisc>,
                                              FE::FlowingVariables<ParallelDisc>>(model, make_json());
}

TEST_CASE("KT residual is unchanged by distribution", "[discretization][kt][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersKT<dim>;
  using SerialDisc = FV::Discretization<Model, RectangularMeshSerial<dim>>;
  using ParallelDisc = FV::Discretization<Model, ParallelMesh<dim>>;
  Model model(nontrivial_parameters());
  require_distributed_residual_matches_serial<FV::KurganovTadmor::Assembler<SerialDisc, Model>, SerialDisc,
                                              FV::KurganovTadmor::Assembler<ParallelDisc, Model>, ParallelDisc,
                                              FV::FlowingVariables<SerialDisc>, FV::FlowingVariables<ParallelDisc>>(
      model, make_json());
}

TEST_CASE("KT jacobian is unchanged by distribution", "[discretization][kt][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersKT<dim>;
  using SerialDisc = FV::Discretization<Model, RectangularMeshSerial<dim>>;
  using ParallelDisc = FV::Discretization<Model, ParallelMesh<dim>>;
  Model model(nontrivial_parameters());
  require_distributed_jacobian_matches_serial<FV::KurganovTadmor::Assembler<SerialDisc, Model>, SerialDisc,
                                              FV::KurganovTadmor::Assembler<ParallelDisc, Model>, ParallelDisc,
                                              FV::FlowingVariables<SerialDisc>, FV::FlowingVariables<ParallelDisc>>(
      model, make_json());
}

// In 2D the partition boundaries run along many faces, some of them described from a cell another rank owns.
TEST_CASE("KT residual and jacobian are unchanged by distribution in 2D", "[discretization][kt][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  using Model = Testing::ModelBurgers2DKT;
  using SerialDisc = FV::Discretization<Model, RectangularMeshSerial<dim>>;
  using ParallelDisc = FV::Discretization<Model, ParallelMesh<dim>>;
  using SerialAssembler = FV::KurganovTadmor::Assembler<SerialDisc, Model>;
  using ParallelAssembler = FV::KurganovTadmor::Assembler<ParallelDisc, Model>;
  auto prm = nontrivial_parameters();
  prm.initial_x2 = {{0.3, 0., 0.}};
  Model model(prm);
  ConfigTree json = make_json();
  json.set_string("/discretization/grid/x_grid", "0:0.125:1");
  json.set_string("/discretization/grid/y_grid", "0:0.125:1");
  require_distributed_residual_matches_serial<SerialAssembler, SerialDisc, ParallelAssembler, ParallelDisc,
                                              FV::FlowingVariables<SerialDisc>, FV::FlowingVariables<ParallelDisc>>(
      model, json);
  require_distributed_jacobian_matches_serial<SerialAssembler, SerialDisc, ParallelAssembler, ParallelDisc,
                                              FV::FlowingVariables<SerialDisc>, FV::FlowingVariables<ParallelDisc>>(
      model, json);
}

TEST_CASE("HAdaptivity transfers the solution identically under distribution", "[discretization][adaptivity][mpi]")
{
  DiFfRG::Init();
  using namespace dealii;
  constexpr uint dim = 1;
  using Model = CellIndicatorModel<dim>;
  using SerialDisc = CG::Discretization<Model, RectangularMeshSerial<dim>>;
  using ParallelDisc = CG::Discretization<Model, ParallelMesh<dim>>;
  using ParVector = typename ParallelDisc::VectorType;

  const auto json = make_json(/*adapt_level=*/2);
  Model model(nontrivial_parameters());

  RectangularMeshSerial<dim> serial_mesh{Config::ConfigurationMesh<dim>(json)};
  SerialDisc serial_disc(serial_mesh, json);
  CG::Assembler<SerialDisc> serial_assembler(serial_disc, model, json);
  FE::FlowingVariables<SerialDisc> serial_ic(serial_disc);
  serial_ic.interpolate(model);
  typename SerialDisc::VectorType serial_sol(serial_ic.spatial_data());
  const auto n_dofs_before = serial_disc.get_dof_handler().n_dofs();

  HAdaptivity<CG::Assembler<SerialDisc>> serial_adaptor(serial_assembler, json);
  REQUIRE(serial_adaptor.adapt(serial_sol));

  ParallelMesh<dim> parallel_mesh{Config::ConfigurationMesh<dim>(json)};
  ParallelDisc parallel_disc(parallel_mesh, json);
  CG::Assembler<ParallelDisc> parallel_assembler(parallel_disc, model, json);
  FE::FlowingVariables<ParallelDisc> parallel_ic(parallel_disc);
  parallel_ic.interpolate(model);
  ParVector parallel_sol(parallel_ic.spatial_data());

  HAdaptivity<CG::Assembler<ParallelDisc>> parallel_adaptor(parallel_assembler, json);
  REQUIRE(parallel_adaptor.adapt(parallel_sol));

  const auto &serial_dh = serial_disc.get_dof_handler();
  const auto &parallel_dh = parallel_disc.get_dof_handler();

  // Vacuity guards. If nothing was refined, or the mesh came out different, the value comparison
  // below is meaningless -- and a zero-size or zero-valued reference passes for every bug.
  REQUIRE(serial_dh.n_dofs() > n_dofs_before);
  REQUIRE(serial_dh.n_dofs() == parallel_dh.n_dofs());

  SolutionView<ParVector> gathered;
  gathered.reinit(parallel_disc.get_locally_owned_dofs(), parallel_disc.get_locally_relevant_dofs(),
                  parallel_disc.get_communicator());
  gathered.refresh(parallel_sol);

  const double scale = std::max(serial_sol.linfty_norm(), 1e-300);
  REQUIRE(scale > 1e-8);

  std::vector<types::global_dof_index> serial_indices, parallel_indices;
  double worst = 0.;
  for (const auto &parallel_cell : parallel_dh.active_cell_iterators()) {
    const typename DoFHandler<dim>::active_cell_iterator serial_cell(
        &serial_dh.get_triangulation(), parallel_cell->level(), parallel_cell->index(), &serial_dh);
    const auto n = parallel_cell->get_fe().n_dofs_per_cell();
    serial_indices.resize(n);
    parallel_indices.resize(n);
    parallel_cell->get_dof_indices(parallel_indices);
    serial_cell->get_dof_indices(serial_indices);
    for (unsigned int k = 0; k < n; ++k)
      worst = std::max(worst, std::abs(gathered[parallel_indices[k]] - serial_sol[serial_indices[k]]));
  }

  INFO("worst |distributed - serial| = " << worst << ", solution scale = " << scale);
  REQUIRE(worst < 1e-10 * scale);
}

#endif // DEAL_II_WITH_MPI && DEAL_II_WITH_PETSC
