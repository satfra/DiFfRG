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

  template <typename VectorType> class NoAdaptivity : public AbstractAdaptor<VectorType>
  {
  public:
    NoAdaptivity() {}

    /**
     * @brief Check if an adaptation step should be done and tranfer the given solution to the new mesh.
     *
     * @param t Current time; at adapt_dt time distances we perform an adaptation
     * @param sol Current solution
     * @return true if adapation has happened, false otherwise
     */
    virtual bool operator()(const double, VectorType &) override { return false; }

    /**
     * @brief Force an adaptation and transfer the solution sol to the new mes

     * @param solution to be transferred
     */
    virtual bool adapt(VectorType &) override { return false; }
  };
} // namespace DiFfRG