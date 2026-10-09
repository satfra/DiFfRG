#pragma once

// external libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

// DiFfRG
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/solution_view.hh>

namespace DiFfRG
{
  using namespace dealii;
  /**
   * @brief Implement a simple interface to do all adaptivity tasks, i.e. solution transfer, reinit of dofHandlers, etc.
   *
   * @tparam Assembler The used assembler should implement the methods refinement_indicator and reinit
   */
  template <typename Assembler>
  class HAdaptivity : public AbstractAdaptor<typename Assembler::Discretization::VectorType>
  {
    using Discretization = typename Assembler::Discretization;
    using VectorType = typename Discretization::VectorType;
    static constexpr uint dim = Discretization::dim;

  public:
    HAdaptivity(Assembler &assembler, const ConfigTree &config)
        : assembler(assembler), discretization(assembler.get_discretization())
    {
      adapt_t = config.get_double("/discretization/adaptivity/start_adapt_at", 0.0);
      adapt_dt = config.get_double("/discretization/adaptivity/adapt_dt", 1e-1);
      adapt_level = config.get_uint("/discretization/adaptivity/level", 0);
      adapt_upper = config.get_double("/discretization/adaptivity/refine_percent", 1e-1);
      adapt_lower = config.get_double("/discretization/adaptivity/coarsen_percent", 5e-2);
    }

    virtual double last_adaptation_time() const override { return last_adapt; }
    virtual void set_last_adaptation_time(const double t) override { last_adapt = t; }

    virtual ~HAdaptivity() = default;

    /**
     * @brief Check if an adaptation step should be done and tranfer the given solution to the new mesh.
     *
     * @param t Current time; at adapt_dt time distances we perform an adaptation
     * @param sol Current solution
     * @return true if adapation has happened, false otherwise
     */
    virtual bool operator()(const double t, VectorType &sol) override
    {
      if (adapt_level > 0 && t >= adapt_t - 1e-12 * adapt_dt && (t - last_adapt + 1e-12 * adapt_dt) >= adapt_dt) {
        if (!adapt(sol)) return false;
        last_adapt = t;
        return true;
      }
      return false;
    }

    /**
     * @brief Force an adaptation and transfer the solution sol to the new mes

     * @param solution to be transferred
     */
    virtual bool adapt(VectorType &solution) override
    {
      auto &triangulation = discretization.get_triangulation();
      auto &dof_handler = discretization.get_dof_handler(0);
      auto &constraints = discretization.get_constraints(0);

      // Both readers below -- refinement_indicator() here and SolutionTransfer further down -- take
      // arbitrary global dof indices, which the vector we are handed cannot answer: it holds only
      // this rank's rows, and deal.II's non-ghosted read path is `ptr[index - local_begin]` with no
      // bounds check outside debug builds. A partition-boundary read therefore does not fail, it
      // returns whatever is next in memory. Read both through a fully-replicated view instead.
      SolutionView<VectorType> view;
      assembler.reinit_solution_view(view);
      view.refresh(solution);

      Vector<double> indicator(triangulation.n_active_cells());
      indicator = 0;
      assembler.refinement_indicator(indicator, view.get());

      if constexpr (Discretization::Mesh::is_parallel) {
        // Each rank fills only the entries of the cells it owns, so on a partitioned mesh every
        // other entry is still zero. Refining on
        // that unreduced vector is not a crash -- it silently refines a different set of cells on
        // every rank, which then diverge. Sum first; the mesh is replicated, so after this every
        // rank holds the identical, complete indicator.
        MPI::sum_reduce(triangulation.get_mpi_communicator(), indicator.data(), static_cast<int>(indicator.size()));
      }

      GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, indicator, adapt_upper, adapt_lower);

      if (triangulation.n_levels() > adapt_level)
        for (const auto &cell : triangulation.active_cell_iterators_on_level(adapt_level))
          cell->clear_refine_flag();
      for (const auto &cell : triangulation.active_cell_iterators_on_level(0))
        cell->clear_coarsen_flag();

      bool any_refined = false;
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->refine_flag_set()) {
          any_refined = true;
          break;
        }
      // Agreed: this early return decides whether execute_coarsening_and_refinement() -- a
      // collective on a parallel triangulation -- is reached at all. The reduced indicator above
      // should already make every rank decide identically; this makes a hang impossible rather
      // than unlikely.
      if (!MPI::any_of(triangulation.get_mpi_communicator(), any_refined)) return false;

      SolutionTransfer<dim, VectorType> solution_trans(dof_handler);

      // A copy, not the view itself: reinit_vector() below resizes `solution`, and the serial view
      // aliases its source rather than snapshotting it. SolutionTransfer also has to keep reading
      // this right through execute_coarsening_and_refinement(), which is where the packing happens.
      VectorType previous_solution = view.get();
      triangulation.prepare_coarsening_and_refinement();
      solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
      triangulation.execute_coarsening_and_refinement();

      discretization.reinit();
      assembler.reinit();
      assembler.reinit_vector(solution);

      // The write side needs no distributed handling. SolutionTransfer is built with
      // average_values = false, so interpolate() ends in compress(VectorOperation::insert), and on
      // a replicated mesh every rank unpacks every cell to bit-identical values -- identical
      // replica, identical mesh, deterministic interpolation -- so it does not matter which rank's
      // insert the owner keeps. (With average_values = true this would instead be a sum over
      // n_ranks redundant contributions, correct only because the valence is inflated by the same
      // factor, and no longer exact.)
      solution_trans.interpolate(solution);
      constraints.distribute(solution);

      return true;
    }

  protected:
    Assembler &assembler;
    Discretization &discretization;

    /// -inf until the first adaptation, so that the first one happens as soon as start_adapt_at is reached.
    double last_adapt = -std::numeric_limits<double>::infinity();
    double adapt_t, adapt_dt, adapt_upper, adapt_lower;
    uint adapt_level;
  };
} // namespace DiFfRG