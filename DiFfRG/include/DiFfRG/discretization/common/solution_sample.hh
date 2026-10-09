#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/discretization/common/cell_geometry.hh>

// standard library
#include <algorithm>
#include <concepts>
#include <vector>

namespace DiFfRG
{
  /// @brief The centre of the reference cell.
  template <int dim> dealii::Point<dim> unit_cell_centre()
  {
    dealii::Point<dim> centre;
    for (uint d = 0; d < dim; ++d)
      centre[d] = 0.5;
    return centre;
  }

  /// @brief One cell's worth of a SolutionSample.
  template <int dim, typename NumberType> struct SolutionSampleEntry {
    /// Cell centre.
    dealii::Point<dim> point;
    /// Smallest face-normal width of the cell, i.e. the local grid spacing.
    double cell_width;
    /// Solution values at `point`, one per component.
    std::vector<NumberType> values;
    /// Solution gradients at `point`, one per component.
    std::vector<dealii::Tensor<1, dim, NumberType>> gradients;
  };

  /**
   * @brief A read-only snapshot of the discrete solution, one sample per active cell.
   *
   * Exists so a model can answer questions about the solution that are *not* pointwise. The EoM
   * machinery in common/eom.hh drives everything from a pointwise field `EoM(x, u)`, which is
   * enough to locate a minimum but cannot express a feature defined by comparing a cell with its
   * neighbours -- a shock, a kink, the edge of a plateau. A model that needs one of those overrides
   * `extractor_point` (see model/model.hh) and is handed one of these.
   *
   * Sampling is at cell centres, one entry per active cell, which is the resolution such features
   * are defined at in the first place. Entries are sorted lexicographically by coordinate, so in
   * one dimension they simply run left to right and neighbouring entries are neighbouring cells.
   * In higher dimensions the ordering is well-defined but adjacency is the model's problem.
   *
   * Like the rest of discretization/common, this assumes the replicated-mesh rung: the mesh is
   * whole on every rank and the solution reaches it through a SolutionView that holds every index,
   * so the sample is built rank-locally and is identical on every rank without communicating.
   */
  template <int dim, typename NumberType> class SolutionSample
  {
  public:
    using Entry = SolutionSampleEntry<dim, NumberType>;

    SolutionSample() = default;
    SolutionSample(std::vector<Entry> entries, const uint n_components)
        : entries(std::move(entries)), m_n_components(n_components)
    {
      std::sort(this->entries.begin(), this->entries.end(), [](const Entry &a, const Entry &b) {
        for (uint d = 0; d < dim; ++d)
          if (a.point[d] != b.point[d]) return a.point[d] < b.point[d];
        return false;
      });
    }

    size_t size() const { return entries.size(); }
    bool empty() const { return entries.empty(); }
    uint n_components() const { return m_n_components; }

    const Entry &operator[](const size_t i) const { return entries[i]; }
    auto begin() const { return entries.begin(); }
    auto end() const { return entries.end(); }

    /**
     * @brief Fill the gradients by central differences between neighbouring samples.
     *
     * For a builder that can supply values cheaply but not gradients -- a finite-volume assembler,
     * whose piecewise-constant shape functions have none -- this recovers a usable gradient in O(n)
     * without touching the scheme's reconstruction machinery. It is a plain difference, not the
     * limited slope the scheme itself uses, so it is deliberately NOT what the flux path sees.
     *
     * Only meaningful in one dimension, where the sorted order is the mesh order; a no-op otherwise.
     */
    void compute_central_difference_gradients()
    {
      if constexpr (dim != 1) return;
      if (entries.size() < 2) return;
      for (size_t i = 0; i < entries.size(); ++i) {
        const size_t lo = (i == 0) ? 0 : i - 1;
        const size_t hi = std::min(i + 1, entries.size() - 1);
        const double dx = entries[hi].point[0] - entries[lo].point[0];
        for (uint c = 0; c < m_n_components; ++c)
          entries[i].gradients[c][0] = dx != 0. ? (entries[hi].values[c] - entries[lo].values[c]) / dx : NumberType(0.);
      }
    }

  private:
    std::vector<Entry> entries;
    uint m_n_components = 0;
  };

  /**
   * @brief Whether @p Model chooses its own point at which the extractors are evaluated.
   *
   * By default the extractors are evaluated at the EoM, i.e. the minimum of the potential
   * reconstructed from `Model::EoM` (see discretization/common/eom.hh). A model that needs them
   * somewhere else declares
   *
   * ```cpp
   * template <int dim, typename NT>
   * Point<dim> extractor_point(const Point<dim> &EoM, const SolutionSample<dim, NT> &sample) const;
   * ```
   *
   * and returns whatever point it likes -- typically after inspecting @p sample for a feature that
   * is not expressible pointwise, such as a shock or a kink. Returning @p EoM reproduces the
   * default exactly.
   *
   * This affects `Model::extract` only. `Model::readouts` keeps being evaluated at the EoM supplied
   * by `readouts_multiple`, so a model can report at the minimum while feeding its Variables from
   * somewhere else entirely.
   *
   * Note that DiFfRG::def::AbstractModel deliberately provides no default implementation: the
   * absence of the method is what this concept detects, and it is what lets an assembler skip
   * building the sample for the models -- almost all of them -- that do not need one.
   */
  template <typename Model, int dim, typename NumberType>
  concept HasExtractorPoint = requires(const Model &model, const dealii::Point<dim> &EoM,
                                       const SolutionSample<dim, NumberType> &sample) {
    { model.template extractor_point<dim, NumberType>(EoM, sample) } -> std::convertible_to<dealii::Point<dim>>;
  };

  /**
   * @brief Build a SolutionSample, taking values and gradients from a callback.
   *
   * The geometry -- which cells, where their centres are, how wide they are, and the ordering --
   * is fixed here; @p fill only has to say what the solution is at a given cell centre. This is
   * what lets a finite-volume assembler contribute reconstructed gradients (its DG0 shape
   * functions have none) without duplicating any of the traversal.
   *
   * @param fill `void(cell, point, std::vector<NumberType> &values, std::vector<Tensor<1,dim,NumberType>> &gradients)`,
   *             with both output vectors pre-sized to @p n_components.
   */
  template <int dim, typename NumberType, typename FillFUN>
  SolutionSample<dim, NumberType> make_solution_sample(const dealii::DoFHandler<dim> &dof_handler,
                                                       const dealii::Mapping<dim> &mapping, const uint n_components,
                                                       const FillFUN &fill)
  {
    std::vector<SolutionSampleEntry<dim, NumberType>> entries;
    entries.reserve(dof_handler.get_triangulation().n_active_cells());

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      SolutionSampleEntry<dim, NumberType> entry;
      entry.point = mapping.transform_unit_to_real_cell(cell, unit_cell_centre<dim>());
      entry.cell_width = internal::cell_width(cell);
      entry.values.resize(n_components);
      entry.gradients.resize(n_components);
      fill(cell, entry.point, entry.values, entry.gradients);
      entries.push_back(std::move(entry));
    }

    return SolutionSample<dim, NumberType>(std::move(entries), n_components);
  }

  /**
   * @brief Build a SolutionSample by evaluating the finite-element solution at each cell centre.
   *
   * The general path, correct for any continuous or discontinuous element. Note that for a DG0
   * (finite-volume) solution the gradients come out identically zero, because the shape functions
   * are constant -- such an assembler should reconstruct them and use the callback overload above.
   */
  template <int dim, typename VectorType>
  SolutionSample<dim, typename VectorType::value_type>
  make_solution_sample(const VectorType &solution, const dealii::DoFHandler<dim> &dof_handler,
                       const dealii::Mapping<dim> &mapping)
  {
    using NumberType = typename VectorType::value_type;
    const uint n_components = dof_handler.get_fe().n_components();

    const dealii::QMidpoint<dim> midpoint;
    dealii::FEValues<dim> fe_v(mapping, dof_handler.get_fe(), midpoint,
                               dealii::update_values | dealii::update_gradients);
    std::vector<dealii::Vector<NumberType>> cell_values(1, dealii::Vector<NumberType>(n_components));
    std::vector<std::vector<dealii::Tensor<1, dim, NumberType>>> cell_gradients(
        1, std::vector<dealii::Tensor<1, dim, NumberType>>(n_components));

    return make_solution_sample<dim, NumberType>(
        dof_handler, mapping, n_components,
        [&](const auto &cell, const dealii::Point<dim> &, std::vector<NumberType> &values,
            std::vector<dealii::Tensor<1, dim, NumberType>> &gradients) {
          fe_v.reinit(cell);
          fe_v.get_function_values(solution, cell_values);
          fe_v.get_function_gradients(solution, cell_gradients);
          for (uint c = 0; c < values.size(); ++c) {
            values[c] = cell_values[0][c];
            gradients[c] = cell_gradients[0][c];
          }
        });
  }
} // namespace DiFfRG
