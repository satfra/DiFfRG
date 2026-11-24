#pragma once

#include <DiFfRG/discretization/common/eom_config.hh>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <deal.II/base/polynomials_p.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace DiFfRG
{
  namespace internal
  {
    template <int dim> const dealii::PolynomialsP<dim> &eom_polynomial_basis(const unsigned int degree)
    {
      static const std::array<dealii::PolynomialsP<dim>, 4> bases{
          dealii::PolynomialsP<dim>(0), dealii::PolynomialsP<dim>(1), dealii::PolynomialsP<dim>(2),
          dealii::PolynomialsP<dim>(3)};
      return bases.at(degree);
    }
  } // namespace internal

  /** Local vector polynomial fitted to FV data. Its components are independent: curl is retained.
   * Parity, when explicitly configured, is about the coordinate planes x_d=0, before EoM source terms.
   */
  template <int dim> struct EoMVectorPolynomial {
    dealii::Point<dim> anchor, origin, scale;
    std::vector<std::vector<unsigned int>> basis_indices;
    std::vector<Eigen::VectorXd> coefficients;
    std::vector<unsigned int> orders;
    std::vector<double> relative_residuals;
    unsigned int requested_order = 0;
    std::size_t sample_count = 0;

    template <typename NumberType = double> dealii::Vector<NumberType> value(const dealii::Point<dim> &point) const
    {
      const auto &basis = internal::eom_polynomial_basis<dim>(requested_order);
      const auto p = scaled_point(point);
      dealii::Vector<NumberType> out(coefficients.size());
      for (unsigned int c = 0; c < coefficients.size(); ++c)
        for (unsigned int k = 0; k < basis_indices[c].size(); ++k)
          out[c] += coefficients[c][k] * basis.compute_value(basis_indices[c][k], p);
      return out;
    }

    std::vector<dealii::Tensor<1, dim>> gradient(const dealii::Point<dim> &point) const
    {
      const auto &basis = internal::eom_polynomial_basis<dim>(requested_order);
      const auto p = scaled_point(point);
      std::vector<dealii::Tensor<1, dim>> out(coefficients.size());
      for (unsigned int c = 0; c < coefficients.size(); ++c)
        for (unsigned int k = 0; k < basis_indices[c].size(); ++k) {
          const auto derivative = basis.compute_grad(basis_indices[c][k], p);
          for (unsigned int d = 0; d < dim; ++d)
            out[c][d] += coefficients[c][k] * derivative[d] / scale[d];
        }
      return out;
    }

    /// Jacobian of the first dim state components; this is not the Jacobian of source-modified Model::EoM.
    dealii::Tensor<2, dim> state_jacobian(const dealii::Point<dim> &point) const
    {
      const auto gradients = gradient(point);
      dealii::Tensor<2, dim> out;
      for (unsigned int i = 0; i < std::min<std::size_t>(dim, gradients.size()); ++i)
        out[i] = gradients[i];
      return out;
    }

  private:
    dealii::Point<dim> scaled_point(const dealii::Point<dim> &point) const
    {
      dealii::Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = (point[d] - origin[d]) / scale[d];
      return p;
    }
  };

  namespace internal
  {
    template <int dim> struct EoMPatchSample {
      dealii::Point<dim> lower, upper;
      Eigen::VectorXd values;
      double weight;
    };

    template <int dim>
    std::pair<EoMVectorPolynomial<dim>, dealii::Point<dim>>
    initialize_eom_polynomial(const dealii::DoFHandler<dim> &dofs, const dealii::Mapping<dim> &mapping,
                              const dealii::Point<dim> &anchor, const Config::EoMConfig &config)
    {
      const auto components = dofs.get_fe().n_components();
      if ((!config.patch_direction.empty() && config.patch_direction.size() != dim) ||
          (!config.patch_parity.empty() && config.patch_parity.size() != components))
        throw std::invalid_argument("EoM patch direction/parity has the wrong dimension or component count.");
      for (const auto &row : config.patch_parity)
        if (row.size() != dim) throw std::invalid_argument("Each EoM parity row needs one entry per coordinate.");
      EoMVectorPolynomial<dim> fit;
      fit.anchor = fit.origin = anchor;
      fit.requested_order = config.patch_order;
      fit.basis_indices.resize(components);
      fit.coefficients.resize(components);
      fit.orders.resize(components);
      fit.relative_residuals.resize(components);
      const auto cell = dealii::GridTools::find_active_cell_around_point(mapping, dofs, anchor).first;
      dealii::Point<dim> width;
      for (unsigned int d = 0; d < dim; ++d) {
        width[d] = cell->extent_in_direction(d);
        fit.scale[d] = config.patch_radius * width[d];
        for (const auto &row : config.patch_parity)
          if (row[d] != 0) fit.origin[d] = 0.;
      }
      return {std::move(fit), width};
    }

    template <int dim, typename VectorType>
    std::optional<std::vector<EoMPatchSample<dim>>>
    collect_eom_patch_samples(const VectorType &state, const dealii::DoFHandler<dim> &dofs,
                              const dealii::Mapping<dim> &mapping, const dealii::Point<dim> &anchor,
                              const dealii::Point<dim> &width, const Config::EoMConfig &config)
    {
      dealii::Tensor<1, dim> normal;
      double norm = 0., exclusion = 0.;
      for (const auto entry : config.patch_direction)
        norm = std::hypot(norm, entry);
      if (norm > 0.)
        for (unsigned int d = 0; d < dim; ++d) {
          normal[d] = config.patch_direction[d] / norm;
          exclusion += config.patch_skip_layers * std::abs(normal[d]) * width[d];
        }
      std::vector<EoMPatchSample<dim>> samples;
      std::vector<dealii::types::global_dof_index> indices(dofs.get_fe().n_dofs_per_cell());
      dealii::Point<dim> midpoint;
      for (unsigned int d = 0; d < dim; ++d)
        midpoint[d] = .5;
      for (const auto &cell : dofs.active_cell_iterators()) {
        const auto point = mapping.transform_unit_to_real_cell(cell, midpoint);
        double distance_squared = 0., side = 0., geometry_scale = 1.;
        auto [lower, upper] = cell->bounding_box().get_boundary_points();
        bool allowed = true;
        for (unsigned int d = 0; d < dim; ++d) {
          distance_squared += std::pow((point[d] - anchor[d]) / width[d], 2);
          geometry_scale = std::max(geometry_scale, std::abs(anchor[d]));
          side += normal[d] * ((normal[d] >= 0. ? lower[d] : upper[d]) - anchor[d]);
          // Whole cells must lie beyond the buffer in every selected coordinate.
          if (config.patch_direction.empty() &&
              lower[d] - anchor[d] < config.patch_skip_layers * width[d] - 1000. *
                                                                               std::numeric_limits<double>::epsilon() *
                                                                               std::max(1., std::abs(anchor[d])))
            allowed = false;
        }
        if (!allowed || distance_squared > config.patch_radius * config.patch_radius ||
            side < exclusion - 1000. * std::numeric_limits<double>::epsilon() * geometry_scale)
          continue;
        Eigen::VectorXd values = Eigen::VectorXd::Zero(dofs.get_fe().n_components());
        cell->get_dof_indices(indices);
        for (unsigned int i = 0; i < indices.size(); ++i)
          values[dofs.get_fe().system_to_component_index(i).first] = state[indices[i]];
        if (!values.allFinite()) return std::nullopt;
        samples.push_back({lower, upper, std::move(values),
                           std::exp(-2. * distance_squared / (config.patch_radius * config.patch_radius))});
      }
      return samples;
    }

    template <int dim>
    std::vector<unsigned int> eom_polynomial_terms(const unsigned int degree, const Config::EoMConfig &config,
                                                   const unsigned int component)
    {
      const auto &basis = eom_polynomial_basis<dim>(degree);
      std::vector<unsigned int> terms;
      for (unsigned int i = 0; i < basis.n(); ++i) {
        const auto powers = basis.directional_degrees(i);
        bool allowed = true;
        for (unsigned int d = 0; d < dim; ++d) {
          const int parity = config.patch_parity.empty() ? 0 : config.patch_parity[component][d];
          allowed = allowed && !((parity == -1 && powers[d] % 2 == 0) || (parity == 1 && powers[d] % 2 != 0));
        }
        if (allowed) terms.push_back(i);
      }
      // Preserve the previous column ordering and hence the pivoted QR's numerical behaviour.
      std::sort(terms.begin(), terms.end(),
                [&](auto a, auto b) { return basis.directional_degrees(a) < basis.directional_degrees(b); });
      return terms;
    }

    template <int dim>
    std::pair<Eigen::MatrixXd, Eigen::VectorXd>
    assemble_eom_fit_system(const std::vector<EoMPatchSample<dim>> &samples, const EoMVectorPolynomial<dim> &fit,
                            const std::vector<unsigned int> &terms, const unsigned int degree,
                            const unsigned int component)
    {
      const auto &basis = eom_polynomial_basis<dim>(degree);
      Eigen::MatrixXd design(samples.size(), terms.size());
      Eigen::VectorXd rhs(samples.size());
      for (unsigned int row = 0; row < samples.size(); ++row) {
        const auto &sample = samples[row];
        rhs[row] = sample.weight * sample.values[component];
        for (unsigned int column = 0; column < terms.size(); ++column) {
          const auto powers = basis.directional_degrees(terms[column]);
          double moment = 1.;
          for (unsigned int d = 0; d < dim; ++d) {
            const auto exponent = powers[d];
            const double a = (sample.lower[d] - fit.origin[d]) / fit.scale[d];
            const double b = (sample.upper[d] - fit.origin[d]) / fit.scale[d];
            moment *= (std::pow(b, exponent + 1) - std::pow(a, exponent + 1)) / ((exponent + 1) * (b - a));
          }
          design(row, column) = sample.weight * moment;
        }
      }
      return {std::move(design), std::move(rhs)};
    }

    struct EoMFitSolution {
      Eigen::VectorXd coefficients;
      double relative_residual;
    };

    inline std::optional<EoMFitSolution> solve_eom_fit_system(Eigen::MatrixXd design, const Eigen::VectorXd &rhs)
    {
      // Column scaling separates rank detection from component units and monomial magnitude.
      Eigen::VectorXd scales(design.cols());
      for (int j = 0; j < design.cols(); ++j) {
        scales[j] = design.col(j).norm();
        if (scales[j] > 0.) design.col(j) /= scales[j];
      }
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
      qr.setThreshold(1e-10);
      if (qr.rank() != design.cols()) return std::nullopt;
      const Eigen::VectorXd coefficients = qr.solve(rhs);
      if (!coefficients.allFinite()) return std::nullopt;
      return EoMFitSolution{coefficients.cwiseQuotient(scales),
                            (design * coefficients - rhs).norm() /
                                std::max(rhs.norm(), 100. * std::numeric_limits<double>::epsilon())};
    }

    template <int dim, typename VectorType>
    std::optional<EoMVectorPolynomial<dim>>
    fit_eom_polynomial(const VectorType &state, const dealii::DoFHandler<dim> &dofs,
                       const dealii::Mapping<dim> &mapping, const dealii::Point<dim> &anchor,
                       const Config::EoMConfig &config)
    {
      if (config.patch_order == 0 || dofs.get_fe().degree != 0) return std::nullopt;
      auto [fit, width] = initialize_eom_polynomial(dofs, mapping, anchor, config);
      const auto samples = collect_eom_patch_samples(state, dofs, mapping, anchor, width, config);
      if (!samples || samples->empty()) return std::nullopt;
      fit.sample_count = samples->size();
      for (unsigned int c = 0; c < fit.coefficients.size(); ++c) {
        for (unsigned int degree = config.patch_order; degree > 0; --degree) {
          auto terms = eom_polynomial_terms<dim>(degree, config, c);
          if (terms.empty() || samples->size() < terms.size()) continue;
          auto [design, rhs] = assemble_eom_fit_system(*samples, fit, terms, degree, c);
          auto solution = solve_eom_fit_system(std::move(design), rhs);
          if (!solution) continue;
          fit.basis_indices[c] = std::move(terms);
          fit.coefficients[c] = std::move(solution->coefficients);
          fit.relative_residuals[c] = solution->relative_residual;
          fit.orders[c] = degree;
          break;
        }
        if (fit.orders[c] == 0) return std::nullopt;
      }
      return fit;
    }
  } // namespace internal
} // namespace DiFfRG
