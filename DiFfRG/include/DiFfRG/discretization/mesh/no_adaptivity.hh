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

  /**
   * @brief Mesh adaptor that never changes the mesh. Use it with timesteppers that do not support adaptivity.
   *
   * Construct it either with the vector type, `NoAdaptivity<VectorType>`, or from the assembler (or discretization),
   * `NoAdaptivity mesh_adaptor(assembler);`, which deduces the vector type.
   *
   * @tparam VectorType Vector type of the spatial solution.
   */
  template <typename VectorType> class NoAdaptivity : public AbstractAdaptor<VectorType>
  {
  public:
    NoAdaptivity() {}

    /**
     * @brief Deduce VectorType from an assembler or discretization; the argument is not used otherwise.
     */
    template <typename AssemblerOrDiscretization>
      requires(!std::same_as<std::remove_cvref_t<AssemblerOrDiscretization>, NoAdaptivity>) &&
              std::same_as<typename std::remove_cvref_t<AssemblerOrDiscretization>::VectorType, VectorType>
    explicit NoAdaptivity(const AssemblerOrDiscretization &)
    {
    }

    virtual ~NoAdaptivity() = default;

    /**
     * @brief Does nothing. @return false.
     */
    virtual bool operator()(const double, VectorType &) override { return false; }

    /**
     * @brief Does nothing. @return false.
     */
    virtual bool adapt(VectorType &) override { return false; }
  };

  template <typename AssemblerOrDiscretization>
  NoAdaptivity(const AssemblerOrDiscretization &) -> NoAdaptivity<typename AssemblerOrDiscretization::VectorType>;
} // namespace DiFfRG