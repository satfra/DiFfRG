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
   * @brief The solution sampled at the centre of every active cell; handed to a model's `extractor_point`.
   *
   * Lets a model locate features defined by comparing neighbouring cells (a shock, a kink, the edge of a plateau),
   * which the pointwise EoM field cannot express; see HasExtractorPoint.
   *
   * Entries are sorted lexicographically by their centre (first coordinate first). In one dimension they run left to
   * right and neighbouring entries are neighbouring cells; in higher dimensions neighbouring entries need not be
   * adjacent. The sample is identical on every MPI rank.
   */
  template <int dim, typename NumberType> class SolutionSample
  {
  public:
    using Entry = SolutionSampleEntry<dim, NumberType>;

    /// An empty sample.
    SolutionSample() = default;
    /// Take @p entries (each with @p n_components values and gradients) and sort them by centre.
    SolutionSample(std::vector<Entry> entries, const uint n_components)
        : entries(std::move(entries)), m_n_components(n_components)
    {
      std::sort(this->entries.begin(), this->entries.end(), [](const Entry &a, const Entry &b) {
        for (uint d = 0; d < dim; ++d)
          if (a.point[d] != b.point[d]) return a.point[d] < b.point[d];
        return false;
      });
    }

    /// Number of entries, i.e. of active cells.
    size_t size() const { return entries.size(); }
    /// Whether there are no entries.
    bool empty() const { return entries.empty(); }
    /// Number of solution components per entry (the length of Entry::values and Entry::gradients).
    uint n_components() const { return m_n_components; }

    /// The @p i-th entry in sorted order.
    const Entry &operator[](const size_t i) const { return entries[i]; }
    /// Iterate over the entries in sorted order.
    auto begin() const { return entries.begin(); }
    /// End of the entries.
    auto end() const { return entries.end(); }

    /**
     * @brief Replace the gradients by differences of neighbouring entries: central inside, one-sided at the ends.
     *
     * One dimension only (a no-op otherwise). Used for FV solutions, whose cell-wise constant shape functions have
     * no gradient. Unrelated to the limited slopes of the FV scheme.
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
   * @brief Whether @p Model chooses the point at which its extractors are evaluated.
   *
   * By default, Model::extract is evaluated at the EoM point (get_EoM_point_with_potential). A model that wants
   * another point implements
   *
   * ```cpp
   * template <int dim, typename NT>
   * Point<dim> extractor_point(const Point<dim> &EoM, const SolutionSample<dim, NT> &sample) const;
   * ```
   *
   * and returns it; returning @p EoM gives the default. Only extract() is affected; the readouts stay at their EoM.
   * def::AbstractModel has no default implementation, so the sample is built only for models that declare one.
   */
  template <typename Model, int dim, typename NumberType>
  concept HasExtractorPoint = requires(const Model &model, const dealii::Point<dim> &EoM,
                                       const SolutionSample<dim, NumberType> &sample) {
    { model.template extractor_point<dim, NumberType>(EoM, sample) } -> std::convertible_to<dealii::Point<dim>>;
  };

  /**
   * @brief Whether @p Model can ask the timestepper for a restart.
   *
   * A model whose flow has a discontinuity in time (e.g. its extractors switching between two minima at a
   * first-order transition) implements
   *
   * ```cpp
   * template <int dim, typename NT>
   * bool needs_restart(double t, const Point<dim> &EoM, const SolutionSample<dim, NT> &sample) const;
   * ```
   *
   * The implicit timesteppers call it once per accepted step, with the accepted state, and on true restart the
   * integration at @p t without its step history (see AbstractAssembler::needs_restart). Since it only ever sees
   * accepted states, it is the place to make the decision that causes the discontinuity: the model may change its
   * state there, and every evaluation after the restart sees the new state.
   */
  template <typename Model, int dim, typename NumberType>
  concept HasNeedsRestart = requires(const Model &model, const double t, const dealii::Point<dim> &EoM,
                                     const SolutionSample<dim, NumberType> &sample) {
    { model.template needs_restart<dim, NumberType>(t, EoM, sample) } -> std::convertible_to<bool>;
  };

  /**
   * @brief Build a SolutionSample over the active cells of @p dof_handler, with values and gradients from @p fill.
   *
   * @param dof_handler Its active cells are the cells of the sample.
   * @param mapping Maps the reference cell to the cells, for the sample points.
   * @param n_components Number of components sampled per point.
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
   * @brief Build a SolutionSample by evaluating the FE solution at each cell centre. For a cell-wise constant (DG0/FV)
   * solution the gradients are zero.
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
