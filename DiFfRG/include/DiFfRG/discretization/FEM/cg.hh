#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
// #include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/run_reporter.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/assembler/cg.hh>
#include <DiFfRG/discretization/common/parallel_dofs.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

// std
#include <type_traits>

namespace DiFfRG
{
  namespace CG
  {
    using namespace dealii;

    /**
     * @brief The continuous Galerkin space: every FE function is continuous across cells and piecewise polynomial of
     * degree \f$p\f$ = `/discretization/fe_order` (FE_Q, nodal at Gauss-Lobatto points) in each direction.
     *
     * The unknowns are the nodal values of all FE functions. Holds the dof numbering, the constraints and the support
     * points; the equation itself is assembled by CG::Assembler. Keys: `/discretization/fe_order` (uint, default 3,
     * warns if missing). See discretization.hh for how to choose a scheme.
     *
     * @tparam ModelOrComponents_ The Model, or a bare ComponentDescriptor. With a descriptor, Model is void and the
     *         assembler has to be given the model explicitly.
     * @tparam Mesh_ The mesh, e.g. RectangularMesh<dim>. A partitioned mesh requires distributed linear algebra and
     *         vice versa (checked at compile time).
     * @tparam NumberType_ The scalar type; must be that of VectorType_.
     * @tparam VectorType_ The vector type; by default serial or distributed, following Mesh_ (LAVectorFor).
     * @tparam SparseMatrixType_ The matrix type, matching VectorType_ (LASparseMatrixFor).
     */
    template <typename ModelOrComponents_, typename Mesh_, typename NumberType_ = double,
              typename VectorType_ = LAVectorFor<Mesh_, NumberType_>,
              typename SparseMatrixType_ = LASparseMatrixFor<Mesh_, NumberType_>>
    class Discretization
    {
    public:
      /// The model this discretization belongs to, or void if it was built from a bare descriptor.
      using Model = typename DiFfRG::internal::model_of_descriptor<ModelOrComponents_>::type;
      /// The ComponentDescriptor: names and counts of FE functions, extractors and variables.
      using Components = typename DiFfRG::internal::components_of<ModelOrComponents_>::type;
      using NumberType = NumberType_;
      using VectorType = VectorType_;
      using SparseMatrixType = SparseMatrixType_;
      using Mesh = Mesh_;
      /// Dimension of the field-space domain.
      static constexpr uint dim = Mesh::dim;

      /// Whether the linear algebra is distributed over MPI ranks.
      static constexpr bool is_distributed = is_distributed_la<VectorType>;

      static_assert(SupportedVectorType<VectorType>,
                    "The VectorType is not one DiFfRG knows: see the get_type trait table in "
                    "DiFfRG/common/linear_algebra.hh.");

      // The scalar type has to agree with the vector's own. PETSc fixes PetscScalar at configure
      // time and does not template on it, so `Discretization<Model, Mesh, float>` next to a PETSc
      // vector would otherwise silently run the model in float against a double vector.
      static_assert(std::is_same_v<NumberType, get_type::NumberType<VectorType>>,
                    "NumberType does not match the number type of VectorType.");

      // A distributed vector partitions its rows by rank ownership, which only exists once the
      // mesh has been partitioned. Pairing one with a serial triangulation compiles cleanly and
      // then hands every rank the same complete index set, so every rank owns every row and the
      // assembled residual is summed n_ranks times -- a wrong answer with no error anywhere.
      static_assert(!is_distributed || Mesh::is_parallel, "A distributed VectorType requires a partitioned mesh, i.e. "
                                                          "RectangularMeshParallel<dim>.");

      // The converse, and the more dangerous direction. A serial vector on a partitioned mesh
      // compiles and runs: every rank assembles only the cells it owns into a full-size vector
      // that nothing ever sums, so the answer is silently wrong rather than absent.
      static_assert(is_distributed || !Mesh::is_parallel,
                    "A partitioned mesh requires a distributed VectorType. Either let both follow "
                    "the build configuration, or pin the mesh serial with RectangularMeshSerial<dim>.");

      /// Distribute the dofs on @p mesh, which must outlive this object.
      Discretization(Mesh &mesh, const ConfigTree &config, ReportPort report_port = {})
          : mesh(mesh), config(config), log(std::move(report_port)),
            fe(std::make_shared<FESystem<dim>>(FE_Q<dim>(config.get_uint_or_warn("/discretization/fe_order", 3)),
                                               Components::count_fe_functions(0))),
            dof_handler(mesh.get_triangulation())
      {
        setup_dofs();
      };

      /**
       * @brief The dof constraints: hanging nodes, plus the model's affine constraints once an assembler has been
       * built (its reinit adds them). @p i is ignored.
       */
      const auto &get_constraints(const uint i = 0) const
      {
        (void)i;
        return constraints;
      }
      auto &get_constraints(const uint i = 0)
      {
        (void)i;
        return constraints;
      }
      /// The DoFHandler; @p i is ignored.
      const auto &get_dof_handler(const uint i = 0) const
      {
        (void)i;
        return dof_handler;
      }
      auto &get_dof_handler(const uint i = 0)
      {
        (void)i;
        return dof_handler;
      }
      /// The finite element (one FE_Q per FE function); @p i must be 0.
      const auto &get_fe(const uint i = 0) const
      {
        if (i != 0) throw std::runtime_error("Wrong FE index");
        return *fe;
      }
      /// The (bi)linear mapping from the reference cell.
      const auto &get_mapping() const { return mapping; }
      /// The triangulation of the mesh.
      const auto &get_triangulation() const { return mesh.get_triangulation(); }
      auto &get_triangulation() { return mesh.get_triangulation(); }
      /// The parameters this discretization was built with.
      const auto &get_config() const { return config; }
      /// Where this discretization reports to (log messages).
      ReportPort report_port() const { return log; }

      /// The support point of @p dof, i.e. where its shape function is nodal.
      const Point<dim> &get_support_point(const uint dof) const { return support_points[dof]; }
      /// The support points of all dofs, indexed by dof.
      const auto &get_support_points() const { return support_points; }

      /// The dofs this rank owns; all of them on a serial mesh.
      const dealii::IndexSet &get_locally_owned_dofs() const { return locally_owned_dofs; }
      /// The dofs this rank can read (owned plus ghosts), see ParallelDoFs::make_ghost_set.
      const dealii::IndexSet &get_locally_relevant_dofs() const { return locally_relevant_dofs; }

      /// The communicator of the dofs; MPI_COMM_SELF on a serial mesh.
      MPI_Comm get_communicator() const { return dof_handler.get_mpi_communicator(); }

      /// Redistribute the dofs on the current mesh, e.g. after refinement.
      void reinit() { setup_dofs(); }

      /// Block sizes of the full state: the number of dofs, then the number of variables if there are any.
      std::vector<uint> get_block_structure() const
      {
        std::vector<uint> block_structure{dof_handler.n_dofs()};
        if (Components::count_variables() > 0) block_structure.push_back(Components::count_variables());
        return block_structure;
      }

    protected:
      void setup_dofs()
      {
        dof_handler.distribute_dofs(*fe);
        // DoFRenumbering::component_wise(dof_handler);

        log.info("FEM: Number of active cells: {}", mesh.get_triangulation().n_active_cells());
        log.info("FEM: Number of degrees of freedom: {}", dof_handler.n_dofs());

        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        constraints.close();

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        locally_relevant_dofs = DiFfRG::internal::ParallelDoFs::make_ghost_set(dof_handler);

        // NOT DoFTools::map_dofs_to_support_points(..., std::vector<Point>&): that overload is
        // documented to *error out* on a DoFHandler built on a parallel::TriangulationBase,
        // because its precondition is an array sized to the global dof count. See
        // ParallelDoFs::build_support_points.
        DiFfRG::internal::ParallelDoFs::build_support_points(mapping, dof_handler, support_points);
      }

      Mesh &mesh;
      ConfigTree config;
      ReportPort log;

      std::shared_ptr<FESystem<dim>> fe;
      DoFHandler<dim> dof_handler;
      AffineConstraints<NumberType> constraints;
      MappingQ1<dim> mapping;
      std::vector<Point<dim>> support_points;
      dealii::IndexSet locally_owned_dofs;
      dealii::IndexSet locally_relevant_dofs;
    };
  } // namespace CG
} // namespace DiFfRG
