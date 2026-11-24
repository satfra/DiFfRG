#pragma once

#include <DiFfRG/discretization/common/eom.hh>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace DiFfRG::Testing
{
  /// Differentiate the vector readout used by the direct search, without reconstructing a scalar potential.
  template <int dim, typename VectorType>
  dealii::Tensor<2, dim> direct_eom_state_derivative(const VectorType &state, const dealii::DoFHandler<dim> &dofs,
                                                     const dealii::Mapping<dim> &mapping,
                                                     const dealii::Point<dim> &point)
  {
    const auto evaluate = [&](const auto &values, const auto &state_dofs) {
      const auto cell = dealii::GridTools::find_active_cell_around_point(mapping, state_dofs, point).first;
      dealii::FEValues<dim> fe_values(mapping, state_dofs.get_fe(), mapping.transform_real_to_unit_cell(cell, point),
                                      dealii::update_gradients);
      fe_values.reinit(cell);
      std::vector<std::vector<dealii::Tensor<1, dim>>> gradients(
          1, std::vector<dealii::Tensor<1, dim>>(state_dofs.get_fe().n_components()));
      fe_values.get_function_gradients(values, gradients);
      dealii::Tensor<2, dim> jacobian;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          jacobian[i][j] = gradients[0][i][j];
      return jacobian;
    };
    if (dofs.get_fe().conforms(dealii::FiniteElementData<dim>::H1)) return evaluate(state, dofs);
    const auto recovered = DiFfRG::internal::reconstruct_eom_state(state, dofs, mapping);
    return evaluate(recovered.values, *recovered.dof_handler);
  }

  template <int dim>
  void print_direct_eom_accuracy(const std::string_view label, const dealii::Point<dim> &point,
                                 const dealii::Point<dim> &exact_point, const dealii::Tensor<2, dim> &du,
                                 const dealii::Tensor<2, dim> &exact_du)
  {
    double root_error = 0., derivative_error = 0., symmetric_error = 0.;
    for (unsigned int i = 0; i < dim; ++i) {
      root_error = std::max(root_error, std::abs(point[i] - exact_point[i]));
      for (unsigned int j = 0; j < dim; ++j) {
        derivative_error = std::max(derivative_error, std::abs(du[i][j] - exact_du[i][j]));
        symmetric_error =
            std::max(symmetric_error, std::abs(.5 * (du[i][j] + du[j][i] - exact_du[i][j] - exact_du[j][i])));
      }
    }
    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();
    std::cout << std::setprecision(17) << std::scientific << "direct EoM accuracy | " << label
              << " | root_max_abs=" << root_error << " | Du_max_abs=" << derivative_error
              << " | sym_Du_max_abs=" << symmetric_error << " | point=" << point << " | Du=" << du << '\n';
    std::cout.flags(flags);
    std::cout.precision(precision);
  }
} // namespace DiFfRG::Testing
