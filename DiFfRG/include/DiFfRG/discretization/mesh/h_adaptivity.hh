#pragma once

// external libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

// standard library
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <vector>

// DiFfRG
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/solution_view.hh>

namespace DiFfRG
{
  using namespace dealii;
  // clang-format off
  /**
   * @brief Adaptive mesh refinement and coarsening (h-adaptivity) driven by the assembler's refinement indicator.
   *
   * At each adaptation, the assembler computes one non-negative indicator \f$\eta_K\f$ per cell \f$K\f$
   * (Assembler::refinement_indicator, from the model's cell_indicator and face_indicator).
   * With \f$\eta = \sum_K \eta_K\f$, the cells with the largest \f$\eta_K\f$ whose sum reaches `refine_percent` \f$\cdot\,\eta\f$ are refined,
   * and the cells with the smallest \f$\eta_K\f$ whose sum stays below `coarsen_percent` \f$\cdot\,\eta\f$ are
   * coarsened (deal.II's GridRefinement::refine_and_coarsen_fixed_fraction). Then
   * - no cell is refined beyond `level`, and no cell is coarsened below the level of the initial mesh;
   * - no cell is coarsened within `coarsen_buffer` layers of neighbours of a cell flagged for refinement, nor if it
   *   was created by one of the last `coarsen_age` adaptations (protects the region right behind a moving front);
   * - if no cell remains flagged for refinement, nothing is done (an adaptation never only coarsens).
   *
   * The hanging-node constraints are applied to the solution before it is interpolated onto the new mesh, and again
   * afterwards. Discretization and assembler are reinitialized.
   *
   * ### Parameters
   * | Key                                           | Type   | Default | Meaning |
   * |-----------------------------------------------|--------|---------|---------|
   * | `/discretization/adaptivity/start_adapt_at`   | double | 0       | RG time of the first adaptation |
   * | `/discretization/adaptivity/adapt_dt`         | double | 0.1     | minimum RG time between two adaptations |
   * | `/discretization/adaptivity/level`            | uint   | 0       | maximum refinement level of a cell (absolute, counted from the coarse grid); 0 disables adaptivity |
   * | `/discretization/adaptivity/refine_percent`   | double | 0.1     | fraction of the total indicator to refine |
   * | `/discretization/adaptivity/coarsen_percent`  | double | 0.05    | fraction of the total indicator to coarsen |
   * | `/discretization/adaptivity/coarsen_buffer`   | uint   | 2       | cell layers around refined cells that are not coarsened |
   * | `/discretization/adaptivity/coarsen_age`      | uint   | 2       | cells created by the last that many adaptations are not coarsened |
   *
   * Since `level` is absolute, a mesh created with `/discretization/grid/refine` \f$= r\f$ is only refined further if
   * `level` \f$> r\f$.
   *
   * @tparam Assembler The assembler; must provide get_discretization(), refinement_indicator(), reinit(),
   * reinit_vector() and reinit_solution_view().
   */
  // clang-format on
  template <typename Assembler>
  class HAdaptivity : public AbstractAdaptor<typename Assembler::Discretization::VectorType>
  {
    using Discretization = typename Assembler::Discretization;
    using VectorType = typename Discretization::VectorType;
    static constexpr uint dim = Discretization::dim;

  public:
    /// Read the parameters (see the class description) and record the level of the initial mesh.
    HAdaptivity(Assembler &assembler, const ConfigTree &config)
        : assembler(assembler), discretization(assembler.get_discretization())
    {
      adapt_t = config.get_double("/discretization/adaptivity/start_adapt_at", 0.0);
      adapt_dt = config.get_double("/discretization/adaptivity/adapt_dt", 1e-1);
      adapt_level = config.get_uint("/discretization/adaptivity/level", 0);
      adapt_upper = config.get_double("/discretization/adaptivity/refine_percent", 1e-1);
      adapt_lower = config.get_double("/discretization/adaptivity/coarsen_percent", 5e-2);
      coarsen_buffer = config.get_uint("/discretization/adaptivity/coarsen_buffer", 2);
      coarsen_age = config.get_uint("/discretization/adaptivity/coarsen_age", 2);
      min_level = std::numeric_limits<uint>::max();
      for (const auto &cell : discretization.get_triangulation().active_cell_iterators())
        min_level = std::min(min_level, uint(cell->level()));
    }

    /// @copydoc AbstractAdaptor::last_adaptation_time
    virtual double last_adaptation_time() const override { return last_adapt; }
    /// @copydoc AbstractAdaptor::set_last_adaptation_time
    virtual void set_last_adaptation_time(const double t) override { last_adapt = t; }

    virtual ~HAdaptivity() = default;

    /**
     * @brief Adapt if `level` > 0, @p t ≥ `start_adapt_at`, and at least `adapt_dt` has passed since the last
     * adaptation; transfer @p sol to the new mesh.
     *
     * @return true if the mesh changed. A skipped adaptation (nothing to refine) does not reset the schedule.
     */
    virtual bool operator()(const double t, VectorType &sol) override { return adapt_at(t, sol, {}); }

    /// As operator()(t, sol), also transferring @p also_transfer to the new mesh.
    virtual bool adapt_at(const double t, VectorType &sol, const std::vector<VectorType *> &also_transfer) override
    {
      if (adapt_level > 0 && t >= adapt_t - 1e-12 * adapt_dt && (t - last_adapt + 1e-12 * adapt_dt) >= adapt_dt) {
        if (!adapt(sol, also_transfer)) return false;
        last_adapt = t;
        return true;
      }
      return false;
    }

    /**
     * @brief Adapt now, regardless of the schedule, and transfer @p solution to the new mesh.
     *
     * @return true if the mesh changed.
     */
    virtual bool adapt(VectorType &solution) override { return adapt(solution, {}); }

    /// As adapt(solution), also transferring @p also_transfer to the new mesh.
    bool adapt(VectorType &solution, const std::vector<VectorType *> &also_transfer)
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

      // Never refine beyond adapt_level, never coarsen below the mesh we started from.
      for (const auto &cell : triangulation.active_cell_iterators()) {
        if (uint(cell->level()) >= adapt_level) cell->clear_refine_flag();
        if (uint(cell->level()) <= min_level) cell->clear_coarsen_flag();
      }
      protect_from_coarsening(triangulation);

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
      std::vector<VectorType> previous{view.get()};
      for (const auto *v : also_transfer) {
        view.refresh(*v);
        previous.push_back(view.get());
      }
      // The timesteppers leave the constrained entries (hanging nodes) stale; interpolating them onto the new mesh
      // would carry that into the unconstrained dofs. For a time derivative this applies the inhomogeneities too,
      // which only touches dofs that are constrained again after the transfer.
      for (auto &v : previous)
        constraints.distribute(v);
      triangulation.prepare_coarsening_and_refinement();
      solution_trans.prepare_for_coarsening_and_refinement(previous);
      std::set<CellId> active_before;
      for (const auto &cell : triangulation.active_cell_iterators())
        active_before.insert(cell->id());
      triangulation.execute_coarsening_and_refinement();
      ++n_adaptations;
      std::map<CellId, uint> still_active;
      for (const auto &cell : triangulation.active_cell_iterators()) {
        const auto id = cell->id();
        if (!active_before.contains(id))
          still_active[id] = n_adaptations;
        else if (const auto found = created_at.find(id); found != created_at.end())
          still_active.insert(*found);
      }
      created_at = std::move(still_active);

      discretization.reinit();
      assembler.reinit();
      std::vector<VectorType *> targets{&solution};
      targets.insert(targets.end(), also_transfer.begin(), also_transfer.end());
      for (auto *v : targets)
        assembler.reinit_vector(*v);

      // The write side needs no distributed handling. SolutionTransfer is built with
      // average_values = false, so interpolate() ends in compress(VectorOperation::insert), and on
      // a replicated mesh every rank unpacks every cell to bit-identical values -- identical
      // replica, identical mesh, deterministic interpolation -- so it does not matter which rank's
      // insert the owner keeps. (With average_values = true this would instead be a sum over
      // n_ranks redundant contributions, correct only because the valence is inflated by the same
      // factor, and no longer exact.)
      solution_trans.interpolate(targets);
      for (auto *v : targets)
        constraints.distribute(*v);

      return true;
    }

  protected:
    /**
     * @brief Clear the coarsening flags within coarsen_buffer layers around cells flagged for refinement and on cells
     * created by one of the last coarsen_age adaptations.
     */
    template <typename Tria> void protect_from_coarsening(Tria &triangulation) const
    {
      using Iterator = typename Tria::active_cell_iterator;
      std::vector<Iterator> frontier;
      std::set<CellId> seen;
      for (const auto &cell : triangulation.active_cell_iterators()) {
        if (cell->refine_flag_set()) {
          frontier.push_back(cell);
          seen.insert(cell->id());
        }
        if (const auto found = created_at.find(cell->id());
            found != created_at.end() && n_adaptations - found->second < coarsen_age)
          cell->clear_coarsen_flag();
      }
      for (uint layer = 0; layer < coarsen_buffer && !frontier.empty(); ++layer) {
        std::vector<Iterator> next;
        for (const auto &cell : frontier) {
          std::vector<Iterator> neighbors;
          GridTools::get_active_neighbors<Tria>(cell, neighbors);
          for (const auto &neighbor : neighbors)
            if (seen.insert(neighbor->id()).second) {
              neighbor->clear_coarsen_flag();
              next.push_back(neighbor);
            }
        }
        frontier = std::move(next);
      }
    }

    Assembler &assembler;
    Discretization &discretization;

    /// -inf until the first adaptation, so that the first one happens as soon as start_adapt_at is reached.
    double last_adapt = -std::numeric_limits<double>::infinity();
    double adapt_t, adapt_dt, adapt_upper, adapt_lower;
    uint adapt_level;
    /// Level of the coarsest active cell at construction.
    uint min_level;
    uint coarsen_buffer, coarsen_age;
    /// Number of adaptations so far, and for each active cell the adaptation that created it (if any did).
    uint n_adaptations = 0;
    std::map<CellId, uint> created_at;
  };
} // namespace DiFfRG