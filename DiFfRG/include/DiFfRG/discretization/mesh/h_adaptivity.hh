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
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>

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
    HAdaptivity(Assembler &assembler, const JSONValue &json)
        : assembler(assembler), discretization(assembler.get_discretization())
    {
      adapt_t = json.get_double("/discretization/adaptivity/start_adapt_at");
      adapt_dt = json.get_double("/discretization/adaptivity/adapt_dt");
      adapt_level = json.get_uint("/discretization/adaptivity/level");
      adapt_upper = json.get_double("/discretization/adaptivity/refine_percent");
      adapt_lower = json.get_double("/discretization/adaptivity/coarsen_percent");
    }

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

      Vector<double> indicator(triangulation.n_active_cells());
      indicator = 0;
      assembler.refinement_indicator(indicator, solution);

      GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, indicator, adapt_upper, adapt_lower);

      std::vector<typename Triangulation<dim>::active_cell_iterator> refined_cells;
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->refine_flag_set()) refined_cells.push_back(cell);
      if (triangulation.n_levels() > adapt_level)
        for (const auto &cell : triangulation.active_cell_iterators_on_level(adapt_level))
          cell->clear_refine_flag();
      for (const auto &cell : triangulation.active_cell_iterators_on_level(0))
        cell->clear_coarsen_flag();

      refined_cells.clear();
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->refine_flag_set()) refined_cells.push_back(cell);
      if (refined_cells.size() == 0) return false;

      SolutionTransfer<dim, VectorType> solution_trans(dof_handler);

      VectorType previous_solution = solution;
      triangulation.prepare_coarsening_and_refinement();
      solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
      triangulation.execute_coarsening_and_refinement();

      discretization.reinit();
      assembler.reinit();
      assembler.reinit_vector(solution);

      solution_trans.interpolate(solution);
      constraints.distribute(solution);

      return true;
    }

  protected:
    Assembler &assembler;
    Discretization &discretization;

    VectorType indicator;

    double last_adapt, adapt_t, adapt_dt, adapt_upper, adapt_lower;
    uint adapt_level;
  };
} // namespace DiFfRG