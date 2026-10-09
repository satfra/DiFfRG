#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/run_reporter.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/common/parallel_dofs.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

#include <iostream>
#include <string>

// std
#include <type_traits>

namespace DiFfRG
{
  namespace FV
  {
    using namespace dealii;

    /**
     * @brief Class to manage the system on which we solve, i.e. fe spaces, grids, etc.
     * This class is a System for FV systems.
     *
     * @tparam Model_ The Model class used for the Simulation
     */
    template <typename ModelOrComponents_, typename Mesh_, typename NumberType_ = double,
              typename VectorType_ = LAVectorFor<Mesh_, NumberType_>,
              typename SparseMatrixType_ = LASparseMatrixFor<Mesh_, NumberType_>>
    class Discretization
    {
    public:
      /// The model this discretization belongs to, or void if it was built from a bare descriptor.
      using Model = typename DiFfRG::internal::model_of_descriptor<ModelOrComponents_>::type;
      using Components = typename DiFfRG::internal::components_of<ModelOrComponents_>::type;
      using NumberType = NumberType_;
      using VectorType = VectorType_;
      using SparseMatrixType = SparseMatrixType_;
      using Mesh = Mesh_;
      static constexpr uint dim = Mesh::dim;

      /**
       * @brief Whether the linear algebra is distributed over MPI ranks.
       */
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
      static constexpr bool is_fv_discretization = true;
      Discretization(Mesh &mesh, const ConfigTree &config, ReportPort report_port = {})
          : mesh(mesh), config(config), log(std::move(report_port)),
            fe(std::make_shared<FESystem<dim>>(FE_DGQ<dim>(0), Components::count_fe_functions(0))),
            dof_handler(mesh.get_triangulation())
      {
        warn_on_ignored_fe_order();
        setup_dofs();
      }

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
      const auto &get_fe(uint i = 0) const
      {
        if (i != 0) throw std::runtime_error("Wrong FE index");
        return *fe;
      }
      const auto &get_mapping() const { return mapping; }
      const auto &get_triangulation() const { return mesh.get_triangulation(); }
      auto &get_triangulation() { return mesh.get_triangulation(); }

      /**
       * @brief Get the support point for a given dof.
       *
       * @param dof The dof index.
       * @return const dealii::Point<dim>& The support point for the given dof.
       */
      const Point<dim> &get_support_point(const uint dof) const { return support_points[dof]; }
      /**
       * @brief Get the support points for all dofs.
       *
       * @return const std::vector<Point<dim>>& The support points for all dofs.
       */
      const auto &get_support_points() const { return support_points; }

      const auto &get_config() const { return config; }
      ReportPort report_port() const { return log; }

      /**
       * @brief The dof rows this rank owns. Complete on a serial mesh.
       */
      const dealii::IndexSet &get_locally_owned_dofs() const { return locally_owned_dofs; }
      /**
       * @brief The dof rows this rank can read. See ParallelDoFs::make_ghost_set for why this is
       * everything at the replicated-mesh rung rather than the one-ring relevant set.
       */
      const dealii::IndexSet &get_locally_relevant_dofs() const { return locally_relevant_dofs; }
      /**
       * @brief MPI_COMM_SELF on a serial mesh, so callers never have to branch on the build type.
       */
      MPI_Comm get_communicator() const { return dof_handler.get_mpi_communicator(); }

      void reinit() { setup_dofs(); }

      uint get_closest_dof(const Point<dim> &p) const
      {
        uint dof = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (uint i = 0; i < support_points.size(); ++i) {
          const auto dist = p.distance(support_points[i]);
          if (dist < min_dist) {
            min_dist = dist;
            dof = i;
          }
        }
        return dof;
      }

      std::vector<uint> get_block_structure() const
      {
        std::vector<uint> block_structure{dof_handler.n_dofs()};
        if (Components::count_variables() > 0) block_structure.push_back(Components::count_variables());
        return block_structure;
      }

    protected:
      /**
       * @brief The finite volume discretization always works on cell averages, i.e. FE_DGQ(0), and never reads
       * /discretization/fe_order. Warn the user if they set it to something nonzero, as it has no effect.
       */
      void warn_on_ignored_fe_order() const
      {
        const uint fe_order = config.get_uint("/discretization/fe_order", 0);
        if (fe_order == 0) return;

        const std::string message =
            "FV: /discretization/fe_order = " + std::to_string(fe_order) +
            " is ignored. The finite volume discretization always uses piecewise constants (fe_order = 0); "
            "reconstruction order is set by the choice of reconstructor and limiter instead.";
        log.warn("{}", message);
      }

      void setup_dofs()
      {
        dof_handler.distribute_dofs(*fe);

        log.info("FV: Number of active cells: {}", mesh.get_triangulation().n_active_cells());
        log.info("FV: Number of degrees of freedom: {}", dof_handler.n_dofs());

        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        constraints.close();

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        locally_relevant_dofs = ParallelDoFs::make_ghost_set(dof_handler);

        // NOT DoFTools::map_dofs_to_support_points(..., std::vector<Point>&): that overload is
        // documented to *error out* on a DoFHandler built on a parallel::TriangulationBase,
        // because its precondition is an array sized to the global dof count. See
        // ParallelDoFs::build_support_points.
        ParallelDoFs::build_support_points(mapping, dof_handler, support_points);
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
  } // namespace FV
} // namespace DiFfRG
