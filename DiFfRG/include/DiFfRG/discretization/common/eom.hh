#pragma once

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/eom_config.hh>
#include <DiFfRG/discretization/common/eom_polynomial.hh>
#include <DiFfRG/discretization/common/serial_mirror.hh>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <boost/signals2/connection.hpp>
#include <cmath>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/numerics/fe_field_function.h>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tbb/tbb.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace DiFfRG
{
  /// Source discretization determines whether FV cell-average polynomial refinement is applicable.
  enum class EoMDataKind { finite_element, finite_volume };

  /// A locally admissible solution of the box-constrained EoM, before postprocessing.
  template <int dim> struct EoMCandidate {
    dealii::Point<dim> point;
    /// -1: physical lower bound, +1: physical upper bound, 0: interior.
    std::array<int, dim> active_bounds{};
    /// Infinity norm of the EoM components violating constrained stationarity.
    double residual = 0.;
    /// Frobenius norm of (J-J^T)/2, in physical field-space coordinates.
    double jacobian_skew_norm = 0.;
    /// Accepted local FV refinement, if enabled and supported by the selected patch.
    std::shared_ptr<const EoMVectorPolynomial<dim>> local_fit;
  };

  /// Result of get_EoM_result.
  template <int dim, typename NumberType> struct EoMResult {
    /// The EoM point.
    dealii::Point<dim> point;
    /// The selected point before EoM_postprocess, also used as the next search's warm start.
    dealii::Point<dim> search_point;
    /// Locally admissible candidates found by the sampled search; not a certified exhaustive isolation.
    std::vector<EoMCandidate<dim>> candidates;
    /// False if a FV patch had insufficient independent samples for an affine state reconstruction.
    bool reconstruction_full_rank = true;
    /// Fit at search_point; null when disabled, unavailable, or rejected by the constrained root checks.
    std::shared_ptr<const EoMVectorPolynomial<dim>> local_fit;
    /// Symmetric derivative of the first dim state components at search_point, from the same fit/vector interpolation.
    /// Model coordinate/source factors and wave-function normalizations are not applied.
    std::optional<dealii::Tensor<2, dim>> hessian;
    /// Absolute maximum entry difference from a quadratic fit on the same patch; a consistency indicator only.
    std::optional<double> hessian_fit_difference;
  };

  namespace internal
  {
    using namespace dealii;
    template <int dim>
    dealii::Point<dim> get_origin(const dealii::DoFHandler<dim> &dof_handler,
                                  typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell)
    {
      using CellIterator = typename dealii::DoFHandler<dim>::cell_iterator;

      auto l1_norm = [](const Point<dim> &p) {
        double norm = 0.;
        for (uint d = 0; d < dim; ++d)
          norm += std::abs(p[d]);
        return norm;
      };

      std::vector<Point<dim>> candidates;

      auto iterate_cell = [&](const CellIterator &cell) {
        std::array<Point<dim>, GeometryInfo<dim>::vertices_per_cell> vertices;
        for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
          vertices[i] = cell->vertex(i);

        auto min_it = std::min_element(vertices.begin(), vertices.end(),
                                       [&](const auto &p1, const auto &p2) { return l1_norm(p1) < l1_norm(p2); });
        candidates.push_back(*min_it);
      };

      for (const auto &cell : dof_handler.active_cell_iterators())
        iterate_cell(cell);

      auto min_it = std::min_element(candidates.begin(), candidates.end(),
                                     [&](const auto &p1, const auto &p2) { return l1_norm(p1) < l1_norm(p2); });

      Point<dim> origin = *min_it;
      EoM_cell = GridTools::find_active_cell_around_point(dof_handler, origin);

      return origin;
    }

    /// The cell with the same (level, index) in another DoFHandler on a structurally identical mesh.
    template <int dim>
    typename dealii::DoFHandler<dim>::active_cell_iterator
    matching_dof_cell(const dealii::DoFHandler<dim> &target_dof_handler,
                      const typename dealii::DoFHandler<dim>::active_cell_iterator &source_cell)
    {
      return typename dealii::DoFHandler<dim>::active_cell_iterator(
          &target_dof_handler.get_triangulation(), source_cell->level(), source_cell->index(), &target_dof_handler);
    }

    template <int dim, typename NumberType> struct ContinuousEoMGeometry;
    /// Per-assembler vector interpolation geometry. Contains no scalar field or factorization.
    template <int dim, typename NumberType> struct EoMStateCache {
      std::shared_ptr<ContinuousEoMGeometry<dim, NumberType>> eom_geometry;
    };
    template <int dim, typename VectorType> class LocalSolution
    {
    public:
      using NumberType = typename VectorType::value_type;
      using Iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

      struct NativeReadoutScratch {
        std::vector<dealii::types::global_dof_index> indices;
        std::vector<NumberType> native_values, tensor_values;
      };

      /// Element-only tables shared by all cell searches in one call.
      struct TensorReadoutData {
        bool supported = false, power = false, standard_power = false;
        std::vector<double> nodes;
        std::vector<std::array<uint, dim>> node_indices;
        std::vector<std::pair<uint, uint>> system_indices;
        std::array<std::array<double, 3>, 3> power_basis{};

        TensorReadoutData(const dealii::FiniteElement<dim> &fe, const dealii::Mapping<dim> &mapping)
        {
          if (!fe.is_primitive() || fe.n_base_elements() != 1 || fe.degree == 0 || fe.degree > 5 ||
              dynamic_cast<const dealii::FE_Q<dim> *>(&fe.base_element(0)) == nullptr ||
              dynamic_cast<const dealii::MappingQ1<dim> *>(&mapping) == nullptr)
            return;
          const auto &support = fe.base_element(0).get_unit_support_points();
          nodes.reserve(fe.degree + 1);
          for (const auto &p : support)
            if (std::find(nodes.begin(), nodes.end(), p[0]) == nodes.end()) nodes.push_back(p[0]);
          std::sort(nodes.begin(), nodes.end());
          node_indices.resize(support.size());
          std::vector<uint> lexicographic(support.size());
          for (uint i = 0; i < support.size(); ++i) {
            uint stride = 1;
            for (uint d = 0; d < dim; ++d, stride *= nodes.size()) {
              node_indices[i][d] = std::find(nodes.begin(), nodes.end(), support[i][d]) - nodes.begin();
              lexicographic[i] += stride * node_indices[i][d];
            }
          }
          power = fe.degree <= 2;
          standard_power = power && nodes.front() == 0. && nodes.back() == 1. && (nodes.size() == 2 || nodes[1] == .5);
          for (uint i = 0; i < fe.n_dofs_per_cell(); ++i) {
            auto [component, node] = fe.system_to_component_index(i);
            system_indices.emplace_back(component, power ? lexicographic[node] : node);
          }
          if (power)
            for (uint i = 0; i < nodes.size(); ++i) {
              power_basis[i][0] = 1.;
              uint degree = 0;
              for (uint j = 0; j < nodes.size(); ++j) {
                if (i == j) continue;
                const double inverse_distance = 1. / (nodes[i] - nodes[j]);
                ++degree;
                for (uint k = degree; k > 0; --k)
                  power_basis[i][k] = (power_basis[i][k - 1] - nodes[j] * power_basis[i][k]) * inverse_distance;
                power_basis[i][0] *= -nodes[j] * inverse_distance;
              }
            }
          supported = true;
        }
      };

      LocalSolution(const VectorType &sol, const dealii::DoFHandler<dim> &dof_handler,
                    const dealii::Mapping<dim> &mapping, const Iterator &cell,
                    const TensorReadoutData *readout = nullptr, NativeReadoutScratch *scratch = nullptr)
          : native_data(scratch ? *scratch : owned_native_data), native_values(native_data.native_values),
            tensor_values(native_data.tensor_values), n_components(dof_handler.get_fe().n_components()),
            native_cell(cell), native_mapping(mapping), native_fe(dof_handler.get_fe()), tensor_element(readout)
      {
        if (dof_handler.get_fe().degree != 0) {
          if (native_fe.is_primitive()) {
            auto &indices = native_data.indices;
            indices.resize(native_fe.n_dofs_per_cell());
            cell->get_dof_indices(indices);
            if (prepare_tensor_readout(cell)) {
              tensor_values.resize(indices.size());
              for (uint i = 0; i < indices.size(); ++i) {
                const auto [component, node] = tensor_element->system_indices[i];
                tensor_values[node * n_components + component] = sol[indices[i]];
              }
              if (tensor_element->standard_power) {
                if (tensor_element->nodes.size() == 2)
                  prepare_standard_power<2>();
                else
                  prepare_standard_power<3>();
              } else if (tensor_element->power)
                prepare_power_readout();
            } else {
              native_values.clear();
              native_values.reserve(indices.size());
              for (const auto index : indices)
                native_values.push_back(sol[index]);
            }
          } else {
            fe_function =
                std::make_unique<dealii::Functions::FEFieldFunction<dim, VectorType>>(dof_handler, sol, mapping);
            fe_function->set_active_cell(cell);
          }
          dg0 = false;
          return;
        }
        dg0 = true;
        full_rank = false;
        centre_values.reinit(n_components);
        slopes.resize(n_components);

        std::vector<Iterator> patch{cell};
        for (uint layer = 0; layer < 2 && patch.size() < 2 * (dim + 1); ++layer) {
          const std::vector<Iterator> frontier = patch;
          for (const auto &member : frontier) {
            std::vector<Iterator> neighbors;
            dealii::GridTools::get_active_neighbors<dealii::DoFHandler<dim>>(member, neighbors);
            for (const auto &neighbor : neighbors)
              if (std::find(patch.begin(), patch.end(), neighbor) == patch.end()) patch.push_back(neighbor);
          }
        }

        dealii::QMidpoint<dim> midpoint;
        dealii::FEValues<dim> fe_values(mapping, dof_handler.get_fe(), midpoint,
                                        dealii::update_values | dealii::update_quadrature_points);
        std::vector<dealii::Vector<NumberType>> point_values(1, dealii::Vector<NumberType>(n_components));
        std::vector<dealii::Point<dim>> centres;
        std::vector<dealii::Vector<NumberType>> values;
        for (const auto &member : patch) {
          fe_values.reinit(member);
          fe_values.get_function_values(sol, point_values);
          centres.push_back(fe_values.quadrature_point(0));
          values.push_back(point_values[0]);
        }
        centre = centres[0];
        centre_values = values[0];
        if (patch.size() < dim + 1) return;

        const double scale = std::max(cell->diameter(), std::numeric_limits<double>::epsilon());
        Eigen::MatrixXd design(patch.size(), dim + 1);
        for (uint row = 0; row < patch.size(); ++row) {
          design(row, 0) = 1.;
          for (uint d = 0; d < dim; ++d)
            design(row, d + 1) = (centres[row][d] - centre[d]) / scale;
        }
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
        if (qr.rank() < dim + 1) return;
        full_rank = true;
        for (uint c = 0; c < n_components; ++c) {
          Eigen::VectorXd samples(patch.size());
          for (uint row = 0; row < patch.size(); ++row)
            samples[row] = (double)values[row][c];
          const Eigen::VectorXd coefficients = qr.solve(samples);
          // The fit's own intercept, not the centre value: a least-squares line through the patch.
          centre_values[c] = coefficients[0];
          for (uint d = 0; d < dim; ++d)
            slopes[c][d] = coefficients[d + 1] / scale;
        }
      }

      dealii::Vector<NumberType> operator()(const dealii::Point<dim> &point) const
      {
        dealii::Vector<NumberType> out(n_components);
        vector_value(point, out);
        return out;
      }

      void vector_value(const dealii::Point<dim> &point, dealii::Vector<NumberType> &out) const
      {
        out = 0.;
        if (!dg0) {
          const auto *element = tensor_element;
          if (tensor_power_readout) {
            // Q1/Q2 coefficients in tensor power order: Horner avoids rebuilding Lagrange weights on every trial.
            if (element->nodes.size() == 2)
              evaluate_power<2>(point, out);
            else
              evaluate_power<3>(point, out);
          } else if (tensor_readout) {
            // The same FE_Q tensor polynomial, evaluated without virtual shape calls or an inverse mapping
            // for each finite-difference/Newton trial. Only Cartesian MappingQ1 cells use this path.
            std::array<std::array<double, 6>, dim> basis{};
            for (uint d = 0; d < dim; ++d) {
              const double x = (point[d] - tensor_origin[d]) * tensor_inverse_width[d];
              for (uint i = 0; i < element->nodes.size(); ++i) {
                basis[d][i] = 1.;
                for (uint j = 0; j < element->nodes.size(); ++j)
                  if (i != j) basis[d][i] *= (x - element->nodes[j]) / (element->nodes[i] - element->nodes[j]);
              }
            }
            for (uint node = 0; node < element->node_indices.size(); ++node) {
              double weight = 1.;
              for (uint d = 0; d < dim; ++d)
                weight *= basis[d][element->node_indices[node][d]];
              for (uint c = 0; c < n_components; ++c)
                out[c] += weight * tensor_values[node * n_components + c];
            }
          } else if (native_values.empty()) {
            fe_function->vector_value(point, out);
          } else {
            const auto unit_point = native_mapping.transform_real_to_unit_cell(native_cell, point);
            for (uint i = 0; i < native_values.size(); ++i)
              out[native_fe.system_to_component_index(i).first] +=
                  native_values[i] * native_fe.shape_value(i, unit_point);
          }
          return;
        }
        for (uint c = 0; c < n_components; ++c) {
          out[c] = centre_values[c];
          for (uint d = 0; d < dim; ++d)
            out[c] += slopes[c][d] * (point[d] - centre[d]);
        }
      }

      bool has_full_rank() const { return full_rank; }

    private:
      template <uint count> void evaluate_power(const dealii::Point<dim> &point, dealii::Vector<NumberType> &out) const
      {
        constexpr uint capacity = [] {
          uint n = 1;
          for (uint d = 0; d < dim; ++d)
            n *= count;
          return n;
        }();
        std::array<NumberType, capacity> work;
        for (uint c = 0; c < n_components; ++c) {
          uint remaining = capacity;
          for (uint i = 0; i < capacity; ++i)
            work[i] = tensor_values[i * n_components + c];
          for (uint d = 0; d < dim; ++d) {
            const double x = (point[d] - tensor_origin[d]) * tensor_inverse_width[d];
            remaining /= count;
            for (uint i = 0; i < remaining; ++i) {
              NumberType value = work[(i + 1) * count - 1];
              for (uint k = count - 1; k > 0; --k)
                value = work[i * count + k - 1] + x * value;
              work[i] = value;
            }
          }
          out[c] = work[0];
        }
      }

      bool prepare_tensor_readout(const Iterator &cell)
      {
        if (!tensor_element) {
          owned_tensor_element = std::make_unique<TensorReadoutData>(native_fe, native_mapping);
          tensor_element = owned_tensor_element.get();
        }
        if (!tensor_element->supported) return false;
        tensor_origin = cell->vertex(0);
        for (uint d = 0; d < dim; ++d) {
          const double width = cell->vertex(1u << d)[d] - tensor_origin[d];
          if (!(width > 0.)) return false;
          tensor_inverse_width[d] = 1. / width;
        }
        for (uint v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
          for (uint d = 0; d < dim; ++d) {
            const double expected = (v & (1u << d)) ? cell->vertex(1u << d)[d] : tensor_origin[d];
            if (cell->vertex(v)[d] != expected) return false;
          }
        tensor_readout = true;
        return true;
      }

      void prepare_power_readout()
      {
        const uint count = tensor_element->nodes.size();
        const auto &basis = tensor_element->power_basis;
        const uint nodes = tensor_element->node_indices.size();
        for (uint d = 0, stride = 1; d < dim; ++d, stride *= count)
          for (uint block = 0; block < nodes; block += stride * count)
            for (uint offset = 0; offset < stride; ++offset)
              for (uint c = 0; c < n_components; ++c) {
                std::array<NumberType, 3> values{};
                for (uint i = 0; i < count; ++i)
                  values[i] = tensor_values[(block + offset + i * stride) * n_components + c];
                for (uint k = 0; k < count; ++k) {
                  NumberType coefficient = 0.;
                  for (uint i = 0; i < count; ++i)
                    coefficient += basis[i][k] * values[i];
                  tensor_values[(block + offset + k * stride) * n_components + c] = coefficient;
                }
              }
        tensor_power_readout = true;
      }

      template <uint count> void prepare_standard_power()
      {
        constexpr uint nodes = [] {
          uint n = 1;
          for (uint d = 0; d < dim; ++d)
            n *= count;
          return n;
        }();
        for (uint d = 0, stride = 1; d < dim; ++d, stride *= count)
          for (uint block = 0; block < nodes; block += stride * count)
            for (uint offset = 0; offset < stride; ++offset)
              for (uint c = 0; c < n_components; ++c) {
                const uint i = (block + offset) * n_components + c;
                const uint step = stride * n_components;
                const auto first = tensor_values[i + step] - tensor_values[i];
                if constexpr (count == 2)
                  tensor_values[i + step] = first;
                else {
                  const auto last = tensor_values[i + 2 * step] - tensor_values[i];
                  tensor_values[i + step] = 4. * first - last;
                  tensor_values[i + 2 * step] = 2. * (last - 2. * first);
                }
              }
        tensor_power_readout = true;
      }

      NativeReadoutScratch owned_native_data;
      NativeReadoutScratch &native_data;
      std::vector<NumberType> &native_values, &tensor_values;
      uint n_components;
      bool dg0 = false;
      bool full_rank = true;
      dealii::Point<dim> centre;
      dealii::Vector<NumberType> centre_values;
      std::vector<dealii::Tensor<1, dim>> slopes;
      std::unique_ptr<dealii::Functions::FEFieldFunction<dim, VectorType>> fe_function;
      Iterator native_cell;
      const dealii::Mapping<dim> &native_mapping;
      const dealii::FiniteElement<dim> &native_fe;
      bool tensor_readout = false;
      bool tensor_power_readout = false;
      dealii::Point<dim> tensor_origin;
      dealii::Tensor<1, dim> tensor_inverse_width;
      const TensorReadoutData *tensor_element;
      std::unique_ptr<TensorReadoutData> owned_tensor_element;
    };

    /// Continuous vector readout of a discontinuous state. No scalar-gradient projection or global solve.
    template <int dim, typename NumberType> struct ContinuousEoMState {
      std::shared_ptr<dealii::FiniteElement<dim>> finite_element;
      std::shared_ptr<dealii::DoFHandler<dim>> dof_handler;
      dealii::Vector<NumberType> values;
      bool full_rank = true;
    };

    /** Mesh-only map from FV cell values to the averaged continuous Q1 readout. Its weights include the
     * least-squares intercept and slopes, exactly as LocalSolution. Columns refer to cells/components, so
     * renumbering source DoFs does not invalidate the map. No solution or EoM callback is cached.
     */
    template <int dim, typename NumberType> struct ContinuousEoMGeometry {
      using Iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
      std::shared_ptr<dealii::FiniteElement<dim>> finite_element;
      std::shared_ptr<dealii::DoFHandler<dim>> dof_handler;
      dealii::AffineConstraints<NumberType> constraints;
      dealii::SparsityPattern sparsity;
      dealii::SparseMatrix<NumberType> interpolation;
      std::vector<Iterator> cells;
      bool full_rank = true;

      ContinuousEoMGeometry(const dealii::DoFHandler<dim> &source, const dealii::Mapping<dim> &mapping)
          : source_address(&source), triangulation_address(&source.get_triangulation()), mapping_address(&mapping),
            source_fe(source.get_fe().get_name())
      {
        const uint components = source.get_fe().n_components();
        const dealii::FE_Q<dim> scalar_fe(1);
        finite_element = std::make_shared<dealii::FESystem<dim>>(scalar_fe, components);
        dof_handler = std::make_shared<dealii::DoFHandler<dim>>(serial_mirror(source.get_triangulation()));
        dof_handler->distribute_dofs(*finite_element);
        cells.resize(source.get_triangulation().n_active_cells());
        std::vector<dealii::Point<dim>> centres(cells.size());
        const dealii::QMidpoint<dim> midpoint;
        dealii::FEValues<dim> values(mapping, source.get_fe(), midpoint, dealii::update_quadrature_points);
        for (const auto &cell : source.active_cell_iterators()) {
          const uint index = cell->active_cell_index();
          cells[index] = cell;
          values.reinit(cell);
          centres[index] = values.quadrature_point(0);
        }
        std::vector<std::map<uint, double>> rows(dof_handler->n_dofs());
        std::vector<uint> contributions(rows.size(), 0);
        std::vector<dealii::types::global_dof_index> target_dofs(finite_element->n_dofs_per_cell());
        for (const auto &cell : cells) {
          std::vector<Iterator> patch{cell};
          for (uint layer = 0; layer < 2 && patch.size() < 2 * (dim + 1); ++layer) {
            const auto frontier = patch;
            for (const auto &member : frontier) {
              std::vector<Iterator> neighbors;
              dealii::GridTools::get_active_neighbors<dealii::DoFHandler<dim>>(member, neighbors);
              for (const auto &neighbor : neighbors)
                if (std::find(patch.begin(), patch.end(), neighbor) == patch.end()) patch.push_back(neighbor);
            }
          }
          const auto centre = centres[cell->active_cell_index()];
          const double scale = std::max(cell->diameter(), std::numeric_limits<double>::epsilon());
          Eigen::MatrixXd design(patch.size(), dim + 1);
          for (uint row = 0; row < patch.size(); ++row) {
            design(row, 0) = 1.;
            for (uint d = 0; d < dim; ++d)
              design(row, d + 1) = (centres[patch[row]->active_cell_index()][d] - centre[d]) / scale;
          }
          const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
          const bool fitted = patch.size() >= dim + 1 && qr.rank() == dim + 1;
          full_rank = full_rank && fitted;
          Eigen::MatrixXd fit_map;
          if (fitted) {
            fit_map.resize(dim + 1, patch.size());
            for (uint column = 0; column < patch.size(); ++column)
              fit_map.col(column) = qr.solve(Eigen::VectorXd::Unit(patch.size(), column));
          }
          matching_dof_cell(*dof_handler, cell)->get_dof_indices(target_dofs);
          for (uint node = 0; node < scalar_fe.n_dofs_per_cell(); ++node) {
            const auto point = mapping.transform_unit_to_real_cell(cell, scalar_fe.get_unit_support_points()[node]);
            for (uint sample = 0; sample < (fitted ? patch.size() : 1); ++sample) {
              double weight = 1.;
              if (fitted) {
                weight = fit_map(0, sample);
                for (uint d = 0; d < dim; ++d)
                  weight += (point[d] - centre[d]) / scale * fit_map(d + 1, sample);
              }
              for (uint c = 0; c < components; ++c) {
                const auto row = target_dofs[finite_element->component_to_system_index(c, node)];
                rows[row][patch[sample]->active_cell_index() * components + c] += weight;
              }
            }
            for (uint c = 0; c < components; ++c)
              ++contributions[target_dofs[finite_element->component_to_system_index(c, node)]];
          }
        }
        dealii::DynamicSparsityPattern dsp(rows.size(), cells.size() * components);
        for (uint row = 0; row < rows.size(); ++row)
          for (const auto &[column, weight] : rows[row])
            dsp.add(row, column);
        sparsity.copy_from(dsp);
        interpolation.reinit(sparsity);
        for (uint row = 0; row < rows.size(); ++row)
          for (const auto &[column, weight] : rows[row])
            interpolation.set(row, column, weight / contributions[row]);
        dealii::DoFTools::make_hanging_node_constraints(*dof_handler, constraints);
        constraints.close();
        connection = source.get_triangulation().signals.any_change.connect([this]() { stale = true; });
      }

      bool describes(const dealii::DoFHandler<dim> &source, const dealii::Mapping<dim> &mapping) const
      {
        return !stale && connection.connected() && source_address == &source && mapping_address == &mapping &&
               triangulation_address == &source.get_triangulation() &&
               cells.size() == source.get_triangulation().n_active_cells() && source_fe == source.get_fe().get_name();
      }

      template <typename VectorType> ContinuousEoMState<dim, NumberType> apply(const VectorType &state) const
      {
        const auto &source_fe = cells.front()->get_dof_handler().get_fe();
        const uint components = source_fe.n_components();
        dealii::Vector<NumberType> samples(cells.size() * components);
        std::vector<dealii::types::global_dof_index> indices(source_fe.n_dofs_per_cell());
        for (uint cell = 0; cell < cells.size(); ++cell) {
          cells[cell]->get_dof_indices(indices);
          for (uint i = 0; i < indices.size(); ++i)
            samples[cell * components + source_fe.system_to_component_index(i).first] = state[indices[i]];
        }
        ContinuousEoMState<dim, NumberType> result{finite_element, dof_handler};
        result.full_rank = full_rank;
        result.values.reinit(dof_handler->n_dofs());
        interpolation.vmult(result.values, samples);
        constraints.distribute(result.values);
        return result;
      }

    private:
      const void *source_address;
      const void *triangulation_address;
      const void *mapping_address;
      std::string source_fe;
      bool stale = false;
      boost::signals2::scoped_connection connection;
    };

    template <int dim, typename VectorType>
    ContinuousEoMState<dim, typename VectorType::value_type>
    reconstruct_eom_state(const VectorType &sol, const dealii::DoFHandler<dim> &source,
                          const dealii::Mapping<dim> &mapping,
                          EoMStateCache<dim, typename VectorType::value_type> *cache = nullptr)
    {
      using NumberType = typename VectorType::value_type;
      if (cache != nullptr && source.get_fe().degree == 0 && source.get_fe().is_primitive() &&
          dynamic_cast<const dealii::MappingQ1<dim> *>(&mapping) != nullptr) {
        if (!cache->eom_geometry || !cache->eom_geometry->describes(source, mapping))
          cache->eom_geometry = std::make_shared<ContinuousEoMGeometry<dim, NumberType>>(source, mapping);
        return cache->eom_geometry->apply(sol);
      }
      ContinuousEoMState<dim, NumberType> state;
      const uint components = source.get_fe().n_components();
      const dealii::FE_Q<dim> scalar_fe(std::max(1u, source.get_fe().degree));
      state.finite_element = std::make_unique<dealii::FESystem<dim>>(scalar_fe, components);
      // Never distribute new DoFs on a parallel mesh: this search can run on rank zero alone.
      state.dof_handler = std::make_unique<dealii::DoFHandler<dim>>(serial_mirror(source.get_triangulation()));
      state.dof_handler->distribute_dofs(*state.finite_element);
      state.values.reinit(state.dof_handler->n_dofs());
      std::vector<uint> contributions(state.values.size(), 0);
      std::vector<dealii::types::global_dof_index> dofs(state.finite_element->n_dofs_per_cell());
      for (const auto &cell : source.active_cell_iterators()) {
        LocalSolution<dim, VectorType> local(sol, source, mapping, cell);
        state.full_rank = state.full_rank && local.has_full_rank();
        std::vector<dealii::Vector<NumberType>> samples;
        for (const auto &p : scalar_fe.get_unit_support_points())
          samples.push_back(local(mapping.transform_unit_to_real_cell(cell, p)));
        const auto target_cell = matching_dof_cell(*state.dof_handler, cell);
        target_cell->get_dof_indices(dofs);
        for (uint i = 0; i < dofs.size(); ++i) {
          const auto [component, node] = state.finite_element->system_to_component_index(i);
          state.values[dofs[i]] += samples[node][component];
          ++contributions[dofs[i]];
        }
      }
      for (uint i = 0; i < state.values.size(); ++i)
        state.values[i] /= contributions[i];
      dealii::AffineConstraints<NumberType> constraints;
      dealii::DoFTools::make_hanging_node_constraints(*state.dof_handler, constraints);
      constraints.close();
      constraints.distribute(state.values);
      return state;
    }

    template <int dim> using EoMSearchBox = std::pair<dealii::Point<dim>, dealii::Point<dim>>;

    /// Fixed geometric tolerance: relative to domain width, with a guard for coordinate roundoff.
    inline double eom_boundary_distance_tolerance(const double lower, const double upper)
    {
      return std::max(1e-12 * (upper - lower),
                      16. * std::numeric_limits<double>::epsilon() * std::max(std::abs(lower), std::abs(upper)));
    }

    template <int dim>
    EoMSearchBox<dim> eom_search_box(const dealii::DoFHandler<dim> &dof_handler,
                                     const EoMSearchBox<dim> *physical_bounds)
    {
      const auto mesh_bounds =
          dealii::GridTools::compute_bounding_box(dof_handler.get_triangulation()).get_boundary_points();
      auto [lower, upper] = physical_bounds ? *physical_bounds : mesh_bounds;
      for (uint d = 0; d < dim; ++d) {
        const double tolerance = eom_boundary_distance_tolerance(mesh_bounds.first[d], mesh_bounds.second[d]);
        if (!std::isfinite(lower[d]) || !std::isfinite(upper[d]) || !(lower[d] < upper[d]) ||
            lower[d] < mesh_bounds.first[d] - tolerance || upper[d] > mesh_bounds.second[d] + tolerance)
          throw std::invalid_argument("The physical EoM search box must lie inside the mesh and have positive widths.");
        // Mesh generation and saved coordinates can differ by a few ulps at configured end points.
        lower[d] = std::max(lower[d], mesh_bounds.first[d]);
        upper[d] = std::min(upper[d], mesh_bounds.second[d]);
        if (!(lower[d] < upper[d]))
          throw std::invalid_argument("The physical EoM search box must intersect the mesh with positive widths.");
      }
      return {lower, upper};
    }

    /// Physical rectangular bounds of a cell, clipped to the physical search domain.
    template <int dim, typename Cell>
    std::pair<dealii::Point<dim>, dealii::Point<dim>> eom_cell_box(const Cell &cell, const dealii::Point<dim> &lower,
                                                                   const dealii::Point<dim> &upper)
    {
      auto lo = cell->vertex(0), hi = lo;
      for (const auto v : cell->vertex_indices())
        for (uint d = 0; d < dim; ++d) {
          lo[d] = std::min(lo[d], cell->vertex(v)[d]);
          hi[d] = std::max(hi[d], cell->vertex(v)[d]);
        }
      for (uint d = 0; d < dim; ++d) {
        lo[d] = std::max(lo[d], lower[d]);
        hi[d] = std::min(hi[d], upper[d]);
      }
      return {lo, hi};
    }

    template <int dim, typename Field>
    bool eom_jacobian(const Field &field, const dealii::Point<dim> &point, const dealii::Point<dim> &lo,
                      const dealii::Point<dim> &hi, Eigen::Matrix<double, dim, dim> &jacobian)
    {
      for (uint d = 0; d < dim; ++d) {
        double h = std::cbrt(std::numeric_limits<double>::epsilon()) * (hi[d] - lo[d]);
        const double interior_distance = std::min(point[d] - lo[d], hi[d] - point[d]);
        if (interior_distance > 16. * std::numeric_limits<double>::epsilon() * std::abs(point[d]))
          h = std::min(h, 0.25 * interior_distance);
        auto a = point, b = point;
        a[d] = std::max(lo[d], point[d] - h);
        b[d] = std::min(hi[d], point[d] + h);
        if (!(b[d] > a[d])) return false;
        const auto fa = field(a), fb = field(b);
        for (uint i = 0; i < dim; ++i)
          jacobian(i, d) = (fb[i] - fa[i]) / (b[d] - a[d]);
      }
      return jacobian.allFinite();
    }

    template <int dim>
    bool eom_locally_admissible(const Eigen::Matrix<double, dim, dim> &jacobian,
                                const Eigen::Matrix<double, dim, 1> &value, const std::array<int, dim> &active,
                                const double tolerance)
    {
      std::vector<uint> critical;
      for (uint d = 0; d < dim; ++d)
        if (active[d] == 0 || std::abs(value[d]) <= tolerance) critical.push_back(d);
      if (critical.empty()) return true; // Strict complementarity at a corner.
      Eigen::MatrixXd symmetric(critical.size(), critical.size());
      for (uint i = 0; i < critical.size(); ++i)
        for (uint j = 0; j < critical.size(); ++j)
          symmetric(i, j) = 0.5 * (jacobian(critical[i], critical[j]) + jacobian(critical[j], critical[i]));
      const double curvature_tolerance = 100. * std::numeric_limits<double>::epsilon() * std::max(1., symmetric.norm());
      if constexpr (dim == 2) {
        if (critical.size() == 2 && active[0] != 0 && active[1] != 0) {
          // Exact copositivity check on the one-sided critical cone at a weak corner.
          const double a = symmetric(0, 0), b = symmetric(1, 1);
          return a >= -curvature_tolerance && b >= -curvature_tolerance &&
                 active[0] * active[1] * symmetric(0, 1) + std::sqrt(std::max(0., a) * std::max(0., b)) >=
                     -curvature_tolerance;
        }
      }
      // PSD on the full critical subspace is sufficient. For 3D weak corners this is conservative.
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(symmetric);
      return eigen.info() == Eigen::Success && eigen.eigenvalues().minCoeff() >= -curvature_tolerance;
    }

    template <int dim, typename Field>
    std::optional<EoMCandidate<dim>>
    refine_direct_eom_candidate(const Field &field, dealii::Point<dim> point, const dealii::Point<dim> &lo,
                                const dealii::Point<dim> &hi, const dealii::Point<dim> &lower,
                                const dealii::Point<dim> &upper, const std::array<int, dim> &fixed,
                                const Config::EoMConfig &config)
    {
      using Matrix = Eigen::Matrix<double, dim, dim>;
      using Value = Eigen::Matrix<double, dim, 1>;
      const double tolerance = std::max(config.abs_tol, 100. * std::numeric_limits<double>::epsilon());
      std::array<unsigned int, dim> free_storage{};
      unsigned int n_free = 0;
      for (unsigned int d = 0; d < dim; ++d)
        if (fixed[d] == 0) free_storage[n_free++] = d;
      const std::span<const unsigned int> free(free_storage.data(), n_free);
      for (uint d = 0; d < dim; ++d) {
        if (fixed[d] == -1) point[d] = lower[d];
        if (fixed[d] == 1) point[d] = upper[d];
      }
      auto value = field(point);
      const auto merit = [&](const Value &f) {
        double norm = 0.;
        for (const auto d : free)
          norm += f[d] * f[d];
        return norm;
      };
      for (uint iteration = 0; iteration < config.max_iter && value.allFinite(); ++iteration) {
        double residual = 0.;
        for (const auto d : free)
          residual = std::max(residual, std::abs(value[d]));
        if (residual <= tolerance) break;
        Matrix jacobian;
        if (!eom_jacobian<dim>(field, point, lo, hi, jacobian)) break;
        using ReducedMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, dim, dim>;
        using ReducedVector = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, dim, 1>;
        ReducedMatrix reduced(free.size(), free.size());
        ReducedVector rhs(free.size());
        for (uint i = 0; i < free.size(); ++i) {
          rhs[i] = -value[free[i]];
          for (uint j = 0; j < free.size(); ++j)
            reduced(i, j) = jacobian(free[i], free[j]);
        }
        ReducedVector direction(free.size());
        bool solved = false;
        if (free.size() == 1 && reduced(0, 0) != 0.) {
          direction[0] = rhs[0] / reduced(0, 0);
          solved = true;
        } else if (free.size() == 2) {
          const double scale = reduced.cwiseAbs().maxCoeff();
          if (scale > 0.) {
            const double a = reduced(0, 0) / scale, b = reduced(0, 1) / scale;
            const double c = reduced(1, 0) / scale, d = reduced(1, 1) / scale;
            const double determinant = a * d - b * c;
            if (std::abs(determinant) > 64. * std::numeric_limits<double>::epsilon()) {
              const double r0 = rhs[0] / scale, r1 = rhs[1] / scale;
              direction[0] = (d * r0 - b * r1) / determinant;
              direction[1] = (a * r1 - c * r0) / determinant;
              solved = true;
            }
          }
        }
        if (!solved) {
          const Eigen::ColPivHouseholderQR<ReducedMatrix> qr(reduced);
          direction = qr.solve(rhs);
          if (qr.rank() < static_cast<int>(free.size())) direction = reduced.transpose() * rhs;
        }
        if (!direction.allFinite() || direction.squaredNorm() == 0.) break;
        double alpha = 1.;
        for (uint i = 0; i < free.size(); ++i) {
          const uint d = free[i];
          if (direction[i] > 0.) alpha = std::min(alpha, (hi[d] - point[d]) / direction[i]);
          if (direction[i] < 0.) alpha = std::min(alpha, (lo[d] - point[d]) / direction[i]);
        }
        bool accepted = false;
        for (uint backtrack = 0; alpha > 0. && backtrack < config.max_backtracks; ++backtrack, alpha *= 0.5) {
          auto next = point;
          for (uint i = 0; i < free.size(); ++i)
            next[free[i]] = std::clamp(point[free[i]] + alpha * direction[i], lo[free[i]], hi[free[i]]);
          const auto next_value = field(next);
          if (next_value.allFinite() && merit(next_value) < (1. - config.armijo_coefficient * alpha) * merit(value)) {
            point = next;
            value = next_value;
            accepted = true;
            break;
          }
        }
        if (!accepted) break;
      }
      if (!value.allFinite()) return std::nullopt;
      EoMCandidate<dim> candidate;
      candidate.point = point;
      for (uint d = 0; d < dim; ++d) {
        const double distance_tolerance = eom_boundary_distance_tolerance(lower[d], upper[d]);
        if (point[d] - lower[d] <= distance_tolerance)
          candidate.active_bounds[d] = -1;
        else if (upper[d] - point[d] <= distance_tolerance)
          candidate.active_bounds[d] = 1;
        const double violation = candidate.active_bounds[d] == -1  ? std::max(0., -value[d])
                                 : candidate.active_bounds[d] == 1 ? std::max(0., value[d])
                                                                   : std::abs(value[d]);
        candidate.residual = std::max(candidate.residual, violation);
      }
      if (candidate.residual > tolerance) return std::nullopt;
      Matrix jacobian;
      if (!eom_jacobian<dim>(field, point, lo, hi, jacobian) ||
          !eom_locally_admissible<dim>(jacobian, value, candidate.active_bounds, tolerance))
        return std::nullopt;
      candidate.jacobian_skew_norm = (0.5 * (jacobian - jacobian.transpose())).norm();
      return candidate;
    }

    template <int dim, typename VectorType, typename EoMFUN>
    std::optional<EoMCandidate<dim>>
    refine_eom_on_polynomial(const EoMCandidate<dim> &coarse, const VectorType &state,
                             const dealii::DoFHandler<dim> &dofs, const dealii::Mapping<dim> &mapping,
                             const EoMFUN &get_EoM, const Config::EoMConfig &config, const EoMSearchBox<dim> &bounds)
    {
      auto polynomial = fit_eom_polynomial(state, dofs, mapping, coarse.point, config);
      if (!polynomial) return std::nullopt;
      using Value = Eigen::Matrix<double, dim, 1>;
      using NumberType = typename VectorType::value_type;
      const auto &[lower, upper] = bounds;
      auto lo = lower, hi = upper;
      // Extrapolation may move through cell interfaces, but may not jump to a distant branch.
      for (unsigned int d = 0; d < dim; ++d) {
        const double width = polynomial->scale[d] / config.patch_radius;
        lo[d] = std::max(lower[d], coarse.point[d] - width);
        hi[d] = std::min(upper[d], coarse.point[d] + width);
      }
      const auto field = [&](const dealii::Point<dim> &point) {
        const auto f = get_EoM(point, polynomial->template value<NumberType>(point));
        Value out;
        for (unsigned int d = 0; d < dim; ++d)
          out[d] = f[d];
        return out;
      };
      unsigned int patterns = 1;
      for (unsigned int d = 0; d < dim; ++d)
        patterns *= 3;
      std::optional<EoMCandidate<dim>> best;
      double distance = std::numeric_limits<double>::infinity();
      for (unsigned int pattern = 0; pattern < patterns; ++pattern) {
        std::array<int, dim> fixed{};
        unsigned int code = pattern;
        bool applicable = true;
        for (unsigned int d = 0; d < dim; ++d, code /= 3) {
          fixed[d] = code % 3 == 0 ? 0 : (code % 3 == 1 ? -1 : 1);
          if (fixed[d] == -1 && lo[d] != lower[d]) applicable = false;
          if (fixed[d] == 1 && hi[d] != upper[d]) applicable = false;
        }
        if (!applicable) continue;
        const auto candidate =
            refine_direct_eom_candidate<dim>(field, coarse.point, lo, hi, lower, upper, fixed, config);
        if (!candidate) continue;
        double candidate_distance = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          candidate_distance += std::pow((candidate->point[d] - coarse.point[d]) / polynomial->scale[d], 2);
        if (candidate_distance < distance) best = candidate, distance = candidate_distance;
      }
      if (best) best->local_fit = std::make_shared<EoMVectorPolynomial<dim>>(std::move(*polynomial));
      return best;
    }

    /**
     * @brief Sampled constrained root search on rectangular cells. Internal faces only delimit Newton trials;
     * they never replace an EoM equation by a constraint. All candidates satisfy physical-box stationarity.
     */
    template <int dim, typename VectorType, typename EoMFUN>
    std::vector<EoMCandidate<dim>>
    find_direct_eom_candidates(const VectorType &sol, const dealii::DoFHandler<dim> &dof_handler,
                               const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
                               const Config::EoMConfig &config, const std::optional<dealii::Point<dim>> &initial_guess,
                               const EoMSearchBox<dim> &bounds)
    {
      using Matrix = Eigen::Matrix<double, dim, dim>;
      using Value = Eigen::Matrix<double, dim, 1>;
      const auto &[lower, upper] = bounds;
      const double tolerance = std::max(config.abs_tol, 100. * std::numeric_limits<double>::epsilon());
      const typename LocalSolution<dim, VectorType>::TensorReadoutData readout(dof_handler.get_fe(), mapping);
      using Cell = typename dealii::DoFHandler<dim>::active_cell_iterator;
      struct LocatedCandidate {
        EoMCandidate<dim> candidate;
        double position_tolerance;
      };
      std::vector<Cell> cells;
      for (const auto &cell : dof_handler.active_cell_iterators())
        cells.push_back(cell);
      std::vector<std::vector<LocatedCandidate>> cell_candidates(cells.size());
      const auto same_root = [&](const auto &point, double position_tolerance, const auto &other) {
        for (uint d = 0; d < dim; ++d)
          if (std::abs(other.point[d] - point[d]) >
              position_tolerance + eom_boundary_distance_tolerance(lower[d], upper[d]))
            return false;
        return true;
      };
      tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const auto &range) {
        typename LocalSolution<dim, VectorType>::NativeReadoutScratch scratch;
        dealii::Vector<typename VectorType::value_type> point_values(dof_handler.get_fe().n_components());
        for (size_t cell_index = range.begin(); cell_index < range.end(); ++cell_index) {
          const auto &cell = cells[cell_index];
          auto &candidates = cell_candidates[cell_index];
          const auto [lo, hi] = eom_cell_box<dim>(cell, lower, upper);
          bool intersects = true;
          for (uint d = 0; d < dim; ++d)
            intersects = intersects && hi[d] > lo[d];
          if (!intersects) continue;
          LocalSolution<dim, VectorType> local(sol, dof_handler, mapping, cell, &readout, &scratch);
          const auto field = [&](const dealii::Point<dim> &x) {
            local.vector_value(x, point_values);
            const auto f = get_EoM(x, point_values);
            Value value;
            for (uint d = 0; d < dim; ++d)
              value[d] = f[d];
            return value;
          };
          std::vector<dealii::Point<dim>> seeds;
          seeds.reserve(dealii::GeometryInfo<dim>::vertices_per_cell + 2 + (dim == 1 ? 2 : 0));
          if (initial_guess) {
            bool inside = true;
            for (uint d = 0; d < dim; ++d)
              inside = inside && std::isfinite((*initial_guess)[d]) && (*initial_guess)[d] >= lo[d] &&
                       (*initial_guess)[d] <= hi[d];
            if (inside) seeds.push_back(*initial_guess);
          }
          seeds.push_back(lo + 0.5 * (hi - lo));
          for (uint v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {
            auto p = lo;
            for (uint d = 0; d < dim; ++d)
              if (v & (1u << d)) p[d] = hi[d];
            seeds.push_back(p);
          }
          if constexpr (dim == 1) {
            seeds.push_back(lo + 0.25 * (hi - lo));
            seeds.push_back(lo + 0.75 * (hi - lo));
          }
          uint patterns = 1;
          for (uint d = 0; d < dim; ++d)
            patterns *= 3;
          for (uint pattern = 0; pattern < patterns; ++pattern) {
            std::array<int, dim> fixed{};
            bool applicable = true;
            uint code = pattern;
            for (uint d = 0; d < dim; ++d, code /= 3) {
              fixed[d] = code % 3 == 0 ? 0 : (code % 3 == 1 ? -1 : 1);
              if (fixed[d] == -1 && lo[d] != lower[d]) applicable = false;
              if (fixed[d] == 1 && hi[d] != upper[d]) applicable = false;
            }
            if (!applicable) continue;
            for (auto point : seeds) {
              const auto solved = refine_direct_eom_candidate<dim>(field, point, lo, hi, lower, upper, fixed, config);
              if (!solved) continue;
              auto candidate = *solved;
              point = candidate.point;
              Matrix jacobian;
              if (!eom_jacobian<dim>(field, point, lo, hi, jacobian)) continue;
              const Eigen::ColPivHouseholderQR<Matrix> full_qr(jacobian);
              const double position_tolerance =
                  full_qr.rank() == dim ? 4. * tolerance * full_qr.solve(Matrix::Identity()).norm() : 0.;
              const bool duplicate = std::any_of(candidates.begin(), candidates.end(), [&](const auto &other) {
                return same_root(point, position_tolerance, other.candidate);
              });
              if (!duplicate) candidates.push_back({candidate, position_tolerance});
            }
          }
        }
      });
      // Merge in cell/seed order, independently of task scheduling.
      std::vector<EoMCandidate<dim>> candidates;
      for (const auto &group : cell_candidates)
        for (const auto &entry : group)
          if (std::none_of(candidates.begin(), candidates.end(), [&](const auto &other) {
                return same_root(entry.candidate.point, entry.position_tolerance, other);
              }))
            candidates.push_back(entry.candidate);
      // Check both sides of an internal interface. A positive one-sided derivative alone can accept a false
      // minimum; the interface is not a physical constraint.
      candidates.erase(
          std::remove_if(candidates.begin(), candidates.end(),
                         [&](auto &candidate) {
                           for (const auto &[cell, unit_point] : dealii::GridTools::find_all_active_cells_around_point(
                                    mapping, dof_handler, candidate.point)) {
                             (void)unit_point;
                             const auto [lo, hi] = eom_cell_box<dim>(cell, lower, upper);
                             bool intersects = true;
                             for (uint d = 0; d < dim; ++d)
                               intersects = intersects && hi[d] > lo[d];
                             if (!intersects) continue;
                             LocalSolution<dim, VectorType> local(sol, dof_handler, mapping, cell, &readout);
                             const auto field = [&](const dealii::Point<dim> &x) {
                               const auto f = get_EoM(x, local(x));
                               Value value;
                               for (uint d = 0; d < dim; ++d)
                                 value[d] = f[d];
                               return value;
                             };
                             Matrix jacobian;
                             if (!eom_jacobian<dim>(field, candidate.point, lo, hi, jacobian) ||
                                 !eom_locally_admissible<dim>(jacobian, field(candidate.point), candidate.active_bounds,
                                                              tolerance))
                               return true;
                             candidate.jacobian_skew_norm = std::max(candidate.jacobian_skew_norm,
                                                                     (0.5 * (jacobian - jacobian.transpose())).norm());
                           }
                           return false;
                         }),
          candidates.end());
      if (candidates.empty())
        throw std::runtime_error("Direct EoM search found no locally admissible constrained root; refine the "
                                 "field-space mesh or increase EoM_max_iter.");
      return candidates;
    }

    /// Direct line work for candidate ordering; never builds or projects onto a scalar FE space.
    template <int dim, typename VectorType, typename EoMFUN>
    double eom_path_work(const VectorType &state, const dealii::DoFHandler<dim> &dofs,
                         const dealii::Mapping<dim> &mapping, const EoMFUN &eom, const EoMSearchBox<dim> &bounds,
                         const dealii::Point<dim> &point)
    {
      const auto &[origin, upper] = bounds;
      const auto direction = point - origin;
      if (direction.norm_square() == 0.) return 0.;
      std::vector<double> cuts{0., 1.};
      for (const auto &cell : dofs.active_cell_iterators()) {
        const auto [lo, hi] = eom_cell_box<dim>(cell, origin, upper);
        for (unsigned int d = 0; d < dim; ++d)
          if (direction[d] != 0.)
            for (const double edge : {lo[d], hi[d]}) {
              const double t = (edge - origin[d]) / direction[d];
              if (t > 0. && t < 1.) cuts.push_back(t);
            }
      }
      std::sort(cuts.begin(), cuts.end());
      cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
      const dealii::QGauss<1> quadrature(std::max(8u, dofs.get_fe().degree + 2));
      double work = 0.;
      for (unsigned int i = 1; i < cuts.size(); ++i) {
        const double a = cuts[i - 1], b = cuts[i];
        if (b - a <= 16. * std::numeric_limits<double>::epsilon()) continue;
        const auto middle = origin + (.5 * (a + b)) * direction;
        const auto cell = dealii::GridTools::find_active_cell_around_point(mapping, dofs, middle).first;
        LocalSolution<dim, VectorType> local(state, dofs, mapping, cell);
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
          const auto x = origin + (a + (b - a) * quadrature.point(q)[0]) * direction;
          const auto f = eom(x, local(x));
          for (unsigned int d = 0; d < dim; ++d)
            work += (b - a) * quadrature.weight(q) * f[d] * direction[d];
        }
      }
      return work;
    }

  } // namespace internal

  /**
   * @brief Find a locally stable constrained zero of the EoM field, including physical edges and corners.
   *
   * In the interior solve `get_EoM(x, u(x)) = 0`; at a physical lower/upper bound require the corresponding
   * component to be nonnegative/nonpositive. The full, possibly nonsymmetric Jacobian is used in Newton steps.
   * A sampled search over all rectangular cells and physical-bound activity patterns discovers candidates;
   * negative curvature in feasible critical directions is rejected. Discovery is not certified exhaustive.
   *
   * Continuous FE states are evaluated directly. Discontinuous states are averaged into a continuous vector
   * readout (FV/DG0 first uses local affine state fits). Explicit coordinate dependence is evaluated afterwards,
   * so singular source terms are never fitted from centre samples. This readout retains numerical curl.
   *
   * Competing candidates are ranked by integrating the direct EoM along straight paths from the physical
   * lower corner. Cell interfaces split the quadrature. This recovers potential differences for conservative
   * fields; with curl it is an explicit path convention, not a unique thermodynamic ordering. No scalar field
   * is reconstructed. Postprocessing receives the same state readout as the search.
   *
   * The parameters are described in Config::EoMConfig.
   *
   * @param EoM_cell Output: the cell containing the returned point.
   * @param sol The solution, with every global dof readable.
   * @param dof_handler, mapping Its dofs and mapping.
   * @param get_EoM Callable `(Point<dim> x, Vector u(x)) -> array<double, dim>`, the EoM field.
   * @param EoM_postprocess Callable `(Point<dim> x, Vector u(x)) -> Point<dim>` applied to the result.
   * @param config The parameters.
   * @param initial_guess Optional extra root-search seed (typically the previous EoM point).
   * @param cache Optional cache of vector interpolation geometry; nullptr builds it for this call only.
   * @param physical_bounds Physical domain from the mesh; assemblers supply it automatically. Standalone calls
   * without mesh metadata use the triangulation bounds when this is nullptr.
   * @param data_kind Source discretization; assemblers set this automatically. Only FV data uses polynomial
   * refinement, which always fits cell averages. FE states retain their existing readout.
   */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  EoMResult<dim, typename VectorType::value_type>
  get_EoM_result(typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
                 const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
                 const EoMPFUN &EoM_postprocess, const Config::EoMConfig &config,
                 const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt,
                 internal::EoMStateCache<dim, typename VectorType::value_type> *cache = nullptr,
                 const internal::EoMSearchBox<dim> *physical_bounds = nullptr,
                 const EoMDataKind data_kind = EoMDataKind::finite_element)
  {
    config.validate();
    const auto bounds = internal::eom_search_box(dof_handler, physical_bounds);
    using NumberType = typename VectorType::value_type;
    if (config.max_iter == 0) {
      const auto origin = internal::get_origin(dof_handler, EoM_cell);
      return {.point = origin, .search_point = origin};
    }
    const auto locate = [&](const auto &state, const auto &state_dofs) {
      using StateVector = std::decay_t<decltype(state)>;
      EoMResult<dim, NumberType> result;
      result.candidates =
          internal::find_direct_eom_candidates(state, state_dofs, mapping, get_EoM, config, initial_guess, bounds);
      if (data_kind == EoMDataKind::finite_volume && config.patch_order != 0 && dof_handler.get_fe().degree == 0)
        for (auto &candidate : result.candidates)
          if (const auto refined =
                  internal::refine_eom_on_polynomial(candidate, sol, dof_handler, mapping, get_EoM, config, bounds))
            candidate = *refined;
      std::sort(result.candidates.begin(), result.candidates.end(), [](const auto &a, const auto &b) {
        for (uint d = 0; d < dim; ++d) {
          if (a.point[d] < b.point[d]) return true;
          if (a.point[d] > b.point[d]) return false;
        }
        return false;
      });
      const auto energy = [&](const dealii::Point<dim> &point) {
        return internal::eom_path_work(state, state_dofs, mapping, get_EoM, bounds, point);
      };
      std::size_t selected = 0;
      if (result.candidates.size() > 1) {
        double best_energy = energy(result.candidates[0].point);
        if (!std::isfinite(best_energy)) throw std::runtime_error("Nonfinite EoM candidate ranking.");
        for (std::size_t i = 1; i < result.candidates.size(); ++i) {
          const double candidate_energy = energy(result.candidates[i].point);
          if (!std::isfinite(candidate_energy)) throw std::runtime_error("Nonfinite EoM candidate ranking.");
          if (candidate_energy < best_energy) selected = i, best_energy = candidate_energy;
        }
      }
      result.search_point = result.candidates[selected].point;
      result.local_fit = result.candidates[selected].local_fit;
      const auto state_cell =
          dealii::GridTools::find_active_cell_around_point(mapping, state_dofs, result.search_point).first;
      internal::LocalSolution<dim, StateVector> local(state, state_dofs, mapping, state_cell);
      dealii::Tensor<2, dim> state_jacobian;
      if (result.local_fit) {
        state_jacobian = result.local_fit->state_jacobian(result.search_point);
        if (config.patch_order == 3 && std::all_of(result.local_fit->orders.begin(), result.local_fit->orders.end(),
                                                   [](const auto order) { return order == 3; })) {
          auto quadratic_config = config;
          quadratic_config.patch_order = 2;
          if (const auto quadratic =
                  internal::fit_eom_polynomial(sol, dof_handler, mapping, result.local_fit->anchor, quadratic_config)) {
            const auto difference = state_jacobian - quadratic->state_jacobian(result.search_point);
            double maximum = 0.;
            for (unsigned int i = 0; i < dim; ++i)
              for (unsigned int j = 0; j < dim; ++j)
                maximum = std::max(maximum, std::abs(.5 * (difference[i][j] + difference[j][i])));
            result.hessian_fit_difference = maximum;
          }
        }
        result.point =
            EoM_postprocess(result.search_point, result.local_fit->template value<NumberType>(result.search_point));
      } else {
        dealii::FEValues<dim> values(mapping, state_dofs.get_fe(),
                                     mapping.transform_real_to_unit_cell(state_cell, result.search_point),
                                     dealii::update_gradients);
        values.reinit(state_cell);
        std::vector<std::vector<dealii::Tensor<1, dim>>> gradients(
            1, std::vector<dealii::Tensor<1, dim>>(state_dofs.get_fe().n_components()));
        values.get_function_gradients(state, gradients);
        for (unsigned int i = 0; i < std::min<std::size_t>(dim, gradients[0].size()); ++i)
          state_jacobian[i] = gradients[0][i];
        result.point = EoM_postprocess(result.search_point, local(result.search_point));
      }
      result.hessian = .5 * (state_jacobian + transpose(state_jacobian));
      EoM_cell = dealii::GridTools::find_active_cell_around_point(mapping, dof_handler, result.point).first;
      return result;
    };
    if (dof_handler.get_fe().conforms(dealii::FiniteElementData<dim>::H1)) return locate(sol, dof_handler);
    auto state = internal::reconstruct_eom_state(sol, dof_handler, mapping, cache);
    auto result = locate(state.values, *state.dof_handler);
    result.reconstruction_full_rank = state.full_rank;
    return result;
  }

  /// As above, with tolerance and iteration limit given directly.
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  EoMResult<dim, typename VectorType::value_type> get_EoM_result(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, [[maybe_unused]] const auto &values) { return p; },
      const double EoM_abs_tol = Config::EoMConfig::default_abs_tol,
      const uint max_iter = Config::EoMConfig::default_max_iter,
      const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt,
      internal::EoMStateCache<dim, typename VectorType::value_type> *cache = nullptr,
      const internal::EoMSearchBox<dim> *physical_bounds = nullptr,
      const EoMDataKind data_kind = EoMDataKind::finite_element)
  {
    return get_EoM_result(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess,
                          Config::EoMConfig(EoM_abs_tol, max_iter), initial_guess, cache, physical_bounds, data_kind);
  }

  /// Only the located point.
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<dim> get_EoM_point(typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
                                   const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping,
                                   const EoMFUN &get_EoM, const EoMPFUN &EoM_postprocess,
                                   const Config::EoMConfig &config,
                                   const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt,
                                   internal::EoMStateCache<dim, typename VectorType::value_type> *cache = nullptr,
                                   const internal::EoMSearchBox<dim> *physical_bounds = nullptr,
                                   const EoMDataKind data_kind = EoMDataKind::finite_element)
  {
    return get_EoM_result(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess, config, initial_guess, cache,
                          physical_bounds, data_kind)
        .point;
  }

  /// As above, with tolerance and iteration limit given directly.
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<dim> get_EoM_point(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, [[maybe_unused]] const auto &values) { return p; },
      const double EoM_abs_tol = Config::EoMConfig::default_abs_tol,
      const uint max_iter = Config::EoMConfig::default_max_iter,
      const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt,
      const internal::EoMSearchBox<dim> *physical_bounds = nullptr,
      const EoMDataKind data_kind = EoMDataKind::finite_element)
  {
    return get_EoM_point<dim>(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess,
                              Config::EoMConfig(EoM_abs_tol, max_iter), initial_guess, nullptr, physical_bounds,
                              data_kind);
  }
} // namespace DiFfRG
