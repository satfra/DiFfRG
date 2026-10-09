#pragma once

// external libraries
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <spdlog/spdlog.h>
#include <tbb/tbb.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/assembler_core.hh>

namespace DiFfRG
{
  using namespace dealii;
  using std::array;

  /**
   * @brief What CG and DG share: the solution at a single point, from the FE space (see internal::AssemblerCore).
   */
  template <typename Discretization_, typename Model_>
  class FEMAssembler : public internal::AssemblerCore<FEMAssembler<Discretization_, Model_>, Discretization_, Model_>
  {
    using Core = internal::AssemblerCore<FEMAssembler<Discretization_, Model_>, Discretization_, Model_>;
    friend Core;

  public:
    using typename Core::Components;
    using typename Core::NumberType;
    using typename Core::VectorType;
    static constexpr uint dim = Core::dim;

    using Core::Core;

    virtual void rebuild_jacobian_sparsity() = 0;

    /// The FE solution and the reconstructed raw potential at one point.
    template <typename PotentialEvaluation = RawPotentialEvaluation<dim, NumberType>> struct PointEvaluation {
      std::vector<Vector<NumberType>> values{Vector<NumberType>(Components::count_fe_functions())};
      std::vector<std::vector<Tensor<1, dim, NumberType>>> gradients{
          std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
      std::vector<std::vector<Tensor<2, dim, NumberType>>> hessians{
          std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
      PotentialEvaluation potential;
      /// Kept for the shape values the extractor jacobian needs.
      std::shared_ptr<FEValues<dim>> fe_values;
    };

    /// @brief Evaluate the FE solution and the raw potential at @p x, which lies in @p cell.
    template <typename RawPotential>
    auto evaluate_at(const Point<dim> &x, const typename DoFHandler<dim>::cell_iterator &cell,
                     const VectorType &solution_global, const RawPotential &raw_potential) const
    {
      PointEvaluation<decltype(evaluate_raw_potential(raw_potential, this->mapping, x))> evaluation;
      evaluation.fe_values = std::make_shared<FEValues<dim>>(
          this->mapping, this->fe, this->mapping.transform_real_to_unit_cell(cell, x),
          update_values | update_gradients | update_quadrature_points | update_JxW_values | update_hessians);
      evaluation.fe_values->reinit(cell);
      evaluation.fe_values->get_function_values(solution_global, evaluation.values);
      evaluation.fe_values->get_function_gradients(solution_global, evaluation.gradients);
      evaluation.fe_values->get_function_hessians(solution_global, evaluation.hessians);
      evaluation.potential = evaluate_raw_potential(raw_potential, this->mapping, x);
      return evaluation;
    }

  protected:
    template <typename Evaluation, typename Extractors>
    static auto solution_tie(const Evaluation &e, const Extractors &extractors, const VectorType &variables)
    {
      return Core::e_tie(e.values[0], e.gradients[0], e.hessians[0], extractors, variables, e.potential.value,
                         e.potential.gradient, e.potential.mass_hessian);
    }
  };
} // namespace DiFfRG
