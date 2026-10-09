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

// DiFfRG
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/run_reporter.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/assembler/ldg.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

namespace DiFfRG
{
  namespace LDG
  {
    using namespace dealii;

    /**
     * @brief Class to manage the system on which we solve, i.e. fe spaces, grids, etc.
     * This class is a System for LDG systems, i.e. DG with additional projections (e.g. derivatives).
     *
     * @tparam Model_ The Model class used for the Simulation
     */
    template <typename ModelOrComponents_, typename Mesh_, typename NumberType_ = double> class Discretization
    {
    public:
      /// The model this discretization belongs to, or void if it was built from a bare descriptor.
      using Model = typename DiFfRG::internal::model_of_descriptor<ModelOrComponents_>::type;
      using Components = typename DiFfRG::internal::components_of<ModelOrComponents_>::type;
      using NumberType = NumberType_;
      using VectorType = Vector<NumberType>;
      using SparseMatrixType = BlockSparseMatrix<NumberType>;
      using Mesh = Mesh_;
      static constexpr uint dim = Mesh::dim;

      // LDG is deliberately excluded from the distributed policy, so it fixes serial linear
      // algebra above rather than taking VectorType_/SparseMatrixType_ parameters. Three things
      // block it, all structural: DoFRenumbering::component_wise assumes globally contiguous
      // per-component blocks, which a parallel DoFHandler destroys (it renumbers per rank), and
      // the BlockSparsityPattern sizing depends on that; the LDG jacobian does sparse
      // matrix-matrix products, for which dealii::SparseMatrix::mmult has no PETSc analogue in
      // the same API; and the component mass matrix is factorised with SparseDirectUMFPACK.
      //
      // Serial vectors on a *partitioned* mesh would still compile, and would only fail later
      // and obscurely -- setup_dofs calls the std::vector overload of map_dofs_to_support_points,
      // which deal.II documents as unsuitable for a parallel::TriangulationBase. Reject the
      // combination here instead, where the message can name the cause.
      static_assert(!Mesh::is_parallel,
                    "LDG does not support a partitioned mesh. Use RectangularMeshSerial<dim>, or pick "
                    "CG/DG/KT for a distributed run. Note that a plain RectangularMesh<dim> is "
                    "partitioned in an MPI build, so LDG has to name the serial mesh explicitly.");

      Discretization(Mesh &mesh, const ConfigTree &config, ReportPort report_port = {})
          : mesh(mesh), config(config), log(std::move(report_port))
      {
        static_assert(Components::count_fe_subsystems() > 1,
                      "LDG must have a defined submodel of the Model with index 1.");
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i) {
          fe.emplace_back(std::make_shared<FESystem<dim>>(
              FE_DGQ<dim>(config.get_uint_or_warn("/discretization/fe_order", 3)), Components::count_fe_functions(i)));
          dof_handler.emplace_back(std::make_shared<DoFHandler<dim>>(mesh.get_triangulation()));
          constraints.emplace_back();
        }
        setup_dofs();
      };

      const auto &get_constraints(const uint i = 0) const { return constraints[i]; }
      auto &get_constraints(const uint i = 0)
      {
        (void)i;
        return constraints[i];
      }
      const auto &get_dof_handler(const uint i = 0) const { return *(dof_handler[i]); }
      auto &get_dof_handler(const uint i = 0) { return *(dof_handler[i]); }
      auto get_dof_handler_list() const
      {
        std::vector<const DoFHandler<dim> *> ret;
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i)
          ret.push_back(&(get_dof_handler(i)));
        return ret;
      }
      const auto &get_fe(uint i = 0) const { return *(fe[i]); }
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

      // LDG is deliberately excluded from the distributed policy: DoFRenumbering::component_wise
      // assumes globally contiguous per-component blocks, which a parallel DoFHandler destroys, and
      // the assembler does sparse matrix-matrix products and a UMFPACK factorisation with no PETSc
      // analogue. These three accessors exist only because LDG shares FE::FlowingVariables with
      // CG/DG, which now asks every discretization for its layout. On the serial triangulation LDG
      // is restricted to, they are exactly the trivial answers.
      const dealii::IndexSet &get_locally_owned_dofs() const { return locally_owned_dofs; }
      const dealii::IndexSet &get_locally_relevant_dofs() const { return locally_relevant_dofs; }
      MPI_Comm get_communicator() const { return dof_handler[0]->get_mpi_communicator(); }

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
        std::vector<uint> block_structure{dof_handler[0]->n_dofs()};
        if (Components::count_variables() > 0) block_structure.push_back(Components::count_variables());
        return block_structure;
      }

    protected:
      void setup_dofs()
      {
        log.info("FEM: Number of active cells: {}", mesh.get_triangulation().n_active_cells());

        for (uint i = 0; i < Components::count_fe_subsystems(); ++i) {
          dof_handler[i]->distribute_dofs(*(fe[i]));
          DoFRenumbering::component_wise(*(dof_handler[i]));
          constraints[i].clear();
          DoFTools::make_hanging_node_constraints(*(dof_handler[i]), constraints[i]);
          constraints[i].close();

          if (i == 0) log.info("FEM: Number of degrees of freedom: {}", dof_handler[0]->n_dofs());
        }

        locally_owned_dofs = dof_handler[0]->locally_owned_dofs();
        locally_relevant_dofs = dealii::complete_index_set(dof_handler[0]->n_dofs());

        support_points.resize(dof_handler[0]->n_dofs());
        DoFTools::map_dofs_to_support_points(mapping, *dof_handler[0], support_points);
      }

      Mesh &mesh;
      ConfigTree config;
      ReportPort log;

      std::vector<std::shared_ptr<FESystem<dim>>> fe;
      std::vector<std::shared_ptr<DoFHandler<dim>>> dof_handler;
      std::vector<AffineConstraints<NumberType>> constraints;
      MappingQ1<dim> mapping;
      std::vector<Point<dim>> support_points;
      dealii::IndexSet locally_owned_dofs;
      dealii::IndexSet locally_relevant_dofs;
    };
  } // namespace LDG
} // namespace DiFfRG
