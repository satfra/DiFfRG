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

#include <concepts>
#include <type_traits>

namespace DiFfRG
{
  using namespace dealii;

  template <typename VectorType> class NoAdaptivity : public AbstractAdaptor<VectorType>
  {
  public:
    NoAdaptivity() {}

    /**
     * @brief Deduction-only constructor.
     *
     * NoAdaptivity needs nothing from the assembler; taking one lets a driver write
     * `NoAdaptivity mesh_adaptor(assembler);` next to `HAdaptivity mesh_adaptor(assembler, json);`
     * instead of respelling VectorType. The first constraint keeps this template from hijacking
     * the copy constructor, the second turns a mismatched argument into a clean substitution
     * failure rather than a silently wrong adaptor.
     */
    template <typename AssemblerOrDiscretization>
      requires(!std::same_as<std::remove_cvref_t<AssemblerOrDiscretization>, NoAdaptivity>) &&
              std::same_as<typename std::remove_cvref_t<AssemblerOrDiscretization>::VectorType, VectorType>
    explicit NoAdaptivity(const AssemblerOrDiscretization &)
    {
    }

    virtual ~NoAdaptivity() = default;

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

  template <typename AssemblerOrDiscretization>
  NoAdaptivity(const AssemblerOrDiscretization &) -> NoAdaptivity<typename AssemblerOrDiscretization::VectorType>;
} // namespace DiFfRG