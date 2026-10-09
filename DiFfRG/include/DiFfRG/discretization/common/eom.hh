#pragma once

// external libraries
#include <boost/signals2/connection.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/fe_field_function.h>

#include <DiFfRG/discretization/common/batched_scatter.hh>
#include <DiFfRG/discretization/common/cell_geometry.hh>
#include <DiFfRG/discretization/common/eom_config.hh>
#include <DiFfRG/discretization/common/serial_mirror.hh>
#include <DiFfRG/discretization/data/output_timings.hh>

// standard library
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief How much of an output frame went into reconstructing potentials.
     *
     * `solve_potential` is the single most expensive thing an output frame does on larger
     * meshes -- it distributes a fresh DoFHandler, assembles a flux system and runs a direct
     * sparse factorisation, once for the raw potential plus once per readout id. The counter
     * lets `OutputSession` attribute that cost without threading a timing object through the
     * whole EoM call chain.
     *
     * `thread_local` rather than atomic: readouts() is called from the single thread that runs
     * the contributor callback, so this stays lock-free and correct.
     */
    struct PotentialSolveStats {
      std::size_t solves = 0;
      double seconds = 0.;
    };

    inline PotentialSolveStats &potential_solve_stats()
    {
      static thread_local PotentialSolveStats stats;
      return stats;
    }
  } // namespace internal

  /**
   * @brief An owning scalar potential reconstructed from a model EoM vector field.
   *
   * The finite element and DoFHandler are retained alongside the potential vector so the
   * reconstruction can be attached to delayed finite-element output safely.
   */
  template <int dim, typename NumberType> struct ReconstructedEoMPotential {
    dealii::Point<dim> minimum;
    std::shared_ptr<dealii::FiniteElement<dim>> finite_element;
    std::shared_ptr<dealii::DoFHandler<dim>> dof_handler;
    dealii::Vector<NumberType> values;
  };

  /**
   * @brief A scalar potential reconstructed from a model-provided raw gradient.
   *
   * The element and DoF handler are shared with DiFfRG::internal::PotentialSystemCache, which retains them
   * across calls; only the coefficient vector belongs to this result alone.
   *
   * Its additive gauge is fixed at the mesh origin. Consequently, value differences and all derivatives are
   * gauge-independent, while the absolute value uses the convention U(origin) = 0.
   */
  /** @brief Optional auxiliary field used only for mass extraction from FV data. */
  template <int dim, typename NumberType> struct RecoveredMassHessian {
    std::unique_ptr<dealii::FiniteElement<dim>> finite_element;
    std::unique_ptr<dealii::DoFHandler<dim>> dof_handler;
    std::array<dealii::Vector<NumberType>, dim * dim> values;
    std::vector<dealii::Point<dim>> sample_centers;
    std::vector<dealii::Tensor<2, dim, NumberType>> samples;
    std::vector<std::vector<uint>> neighbors;
    std::vector<std::vector<double>> face_weights;
    std::vector<std::vector<uint>> kink_faces_by_cell;
    std::vector<dealii::Point<dim>> kink_face_centers;
    std::vector<dealii::Tensor<1, dim>> kink_face_normals;
    std::vector<double> kink_face_widths;
    std::vector<double> kink_face_strengths;
  };

  template <int dim, typename NumberType> struct ReconstructedRawPotential {
    std::shared_ptr<dealii::FiniteElement<dim>> finite_element;
    std::shared_ptr<dealii::DoFHandler<dim>> dof_handler;
    dealii::Vector<NumberType> values;
    std::unique_ptr<RecoveredMassHessian<dim, NumberType>> recovered_mass_hessian;
  };

  template <typename NumberType> struct ReconstructedRawPotential<0, NumberType> {
    dealii::Vector<NumberType> values;
  };

  /** @brief Reconstructed potential data and the explicitly separate Hessian used for mass extraction. */
  template <int dim, typename NumberType> struct RawPotentialEvaluation {
    NumberType value{};
    dealii::Tensor<1, dim, NumberType> gradient;
    dealii::Tensor<2, dim, NumberType> potential_hessian;
    dealii::Tensor<2, dim, NumberType> mass_hessian;
  };

  /**
   * @brief Stand-in for a raw potential that a model has declared it does not read.
   *
   * Deliberately without operations, so a model that sets extract_uses_potential to false and then reads one
   * of the potential slots fails to compile instead of quietly receiving a zero potential.
   */
  struct UnusedPotential {
  };

  /** @brief Evaluating an unread potential: the same three slots, all inert. */
  struct UnusedPotentialEvaluation {
    UnusedPotential value, gradient, hessian;
  };

  template <int dim>
  UnusedPotentialEvaluation evaluate_raw_potential(const UnusedPotential &, const dealii::Mapping<dim> &,
                                                   const dealii::Point<dim> &)
  {
    return {};
  }

  template <typename NumberType> struct ReconstructedEoMPotential<0, NumberType> {
    dealii::Point<0> minimum;
    dealii::Vector<NumberType> values;
  };

  /**
   * @brief Result of finding an EoM point, optionally including its reconstructed potential.
   */
  template <int dim, typename NumberType> struct EoMResult {
    dealii::Point<dim> point;
    std::optional<ReconstructedEoMPotential<dim, NumberType>> potential;
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

    /**
     * @brief The cell at the same position in another DoFHandler.
     *
     * Addressed by (level, index) rather than by copying the source accessor.
     * TriaAccessorBase::copy_from also copies the *triangulation pointer*, which is silently wrong
     * as soon as the two DoFHandlers sit on different -- though structurally identical --
     * triangulations, which is exactly what the distributed policy does in solve_potential.
     */
    template <int dim>
    typename dealii::DoFHandler<dim>::active_cell_iterator
    matching_dof_cell(const dealii::DoFHandler<dim> &target_dof_handler,
                      const typename dealii::DoFHandler<dim>::active_cell_iterator &source_cell)
    {
      return typename dealii::DoFHandler<dim>::active_cell_iterator(
          &target_dof_handler.get_triangulation(), source_cell->level(), source_cell->index(), &target_dof_handler);
    }

    template <int dim> double l1_distance(const dealii::Point<dim> &a, const dealii::Point<dim> &b)
    {
      double distance = 0.;
      for (uint d = 0; d < dim; ++d)
        distance += std::abs(a[d] - b[d]);
      return distance;
    }

    template <int dim>
    double resolve_potential_smoothing_length(const dealii::DoFHandler<dim> &dof_handler,
                                              const double configured_length)
    {
      if (!std::isfinite(configured_length) || configured_length < -1. ||
          (configured_length < 0. && configured_length != -1.))
        throw std::invalid_argument(
            "EoM_smoothing_length must be -1 (automatic), zero, or a positive physical length.");

      if (configured_length == -1.) return 2. * minimum_face_normal_cell_width(dof_handler);
      return configured_length;
    }

    template <int dim>
    Config::EoMConfig resolve_eom_config(const dealii::DoFHandler<dim> &dof_handler, Config::EoMConfig config)
    {
      config.validate();
      config.smoothing_length = resolve_potential_smoothing_length(dof_handler, config.smoothing_length);
      return config;
    }

    template <int dim, typename NumberType>
    dealii::types::global_dof_index
    select_gauge_dof(const dealii::AffineConstraints<NumberType> &constraints, const dealii::Point<dim> &origin,
                     const std::map<dealii::types::global_dof_index, dealii::Point<dim>> &support_points)
    {
      dealii::types::global_dof_index best_dof = numbers::invalid_dof_index;
      double best_distance = std::numeric_limits<double>::max();
      for (const auto &dof_and_point : support_points) {
        const auto dof = dof_and_point.first;
        if (constraints.is_constrained(dof)) continue;

        const double distance = l1_distance(dof_and_point.second, origin);
        if (distance < best_distance) {
          best_distance = distance;
          best_dof = dof;
        }
      }

      if (best_dof == numbers::invalid_dof_index)
        throw std::runtime_error("EoM potential reconstruction could not find an unconstrained gauge DoF.");

      return best_dof;
    }

    template <int dim> std::unique_ptr<dealii::FiniteElement<dim>> make_eom_potential_fe()
    {
      return std::make_unique<FE_Q<dim>>(2);
    }

    template <int dim> std::unique_ptr<dealii::FiniteElement<dim>> make_raw_potential_fe(const uint order = 2)
    {
      return std::make_unique<FE_Q<dim>>(order);
    }

    /** @brief Which of the two potentials a reconstruction is for; they are cached separately. */
    enum class PotentialKind : uint { eom = 0, raw = 1 };
    inline constexpr uint n_potential_kinds = 2;

    template <int dim>
    std::unique_ptr<dealii::FiniteElement<dim>> make_potential_fe(const PotentialKind kind, const uint raw_order)
    {
      return kind == PotentialKind::eom ? make_eom_potential_fe<dim>() : make_raw_potential_fe<dim>(raw_order);
    }

    /**
     * @brief The mesh-dependent half of solve_potential, retained across calls.
     *
     * The potential system is a least-squares projection of a gradient field onto a continuous element, and
     * its matrix depends only on the mesh, that element and the smoothing length -- the solution enters
     * through the right-hand side alone. Rebuilt per call, as it was, every reconstruction distributes a
     * fresh DoF handler, builds a flux sparsity pattern, constructs quadratures and runs a direct sparse
     * factorisation. Every residual and jacobian evaluation of a model carrying extractors pays that twice.
     * Cached, all of it happens once per mesh and a reconstruction costs one right-hand side assembly plus a
     * back-substitution.
     *
     * A cache is owned by the assembler that reconstructs, so it cannot outlive the mesh it describes, and
     * is used from whichever thread drives assembly or readouts. It is not internally synchronised and must
     * not be shared between threads that run concurrently.
     */
    template <int dim, typename NumberType> struct PotentialAssemblyData;

    template <int dim, typename NumberType> class PotentialSystemCache
    {
      static constexpr int face_dim = dim > 0 ? dim - 1 : 0;

    public:
      struct Entry {
        std::shared_ptr<dealii::FiniteElement<dim>> finite_element;
        std::shared_ptr<dealii::DoFHandler<dim>> dof_handler;
        dealii::AffineConstraints<NumberType> constraints;
        dealii::SparsityPattern sparsity_pattern;
        dealii::SparseMatrix<NumberType> matrix;
        dealii::SparseDirectUMFPACK factorization;
        dealii::Quadrature<dim> quadrature;
        dealii::Quadrature<face_dim> face_quadrature;
        /// Where each potential DoF sits; needed by the gauge choice and by every minimum search.
        std::map<dealii::types::global_dof_index, dealii::Point<dim>> support_points;
        /// What the right-hand side assembly reuses: its cells, faces and per-thread FE scratch.
        std::unique_ptr<PotentialAssemblyData<dim, NumberType>> assembly;

        /**
         * @brief Whether this entry still describes the system that would be built now.
         *
         * `stale` is set from the triangulation's change signal, which also covers the refinement that
         * leaves the cell count untouched; the counters catch a mesh that was swapped wholesale. A
         * disconnected signal means the triangulation this entry was built for is gone -- destroying one
         * disconnects its slots -- so the address may now belong to a different mesh entirely.
         */
        bool describes(const dealii::Triangulation<dim> &tria, const dealii::DoFHandler<dim> &solution_dof_handler,
                       const double smoothing, const uint potential_fe_degree) const
        {
          return !stale && connection.connected() && triangulation == &tria && solution_dofs == &solution_dof_handler &&
                 n_active_cells == tria.n_active_cells() && n_levels == tria.n_levels() &&
                 n_vertices == tria.n_vertices() && solution_fe_degree == solution_dof_handler.get_fe().degree &&
                 smoothing_length == smoothing && finite_element != nullptr &&
                 finite_element->degree == potential_fe_degree;
        }

        void note_built_for(const dealii::Triangulation<dim> &tria, const dealii::DoFHandler<dim> &solution_dof_handler,
                            const double smoothing)
        {
          triangulation = &tria;
          solution_dofs = &solution_dof_handler;
          n_active_cells = tria.n_active_cells();
          n_levels = tria.n_levels();
          n_vertices = tria.n_vertices();
          solution_fe_degree = solution_dof_handler.get_fe().degree;
          smoothing_length = smoothing;
          connection = tria.signals.any_change.connect([this]() { stale = true; });
          stale = false;
        }

      private:
        const void *triangulation = nullptr;
        const void *solution_dofs = nullptr;
        std::size_t n_active_cells = 0;
        uint n_levels = 0;
        std::size_t n_vertices = 0;
        uint solution_fe_degree = 0;
        double smoothing_length = 0.;
        boost::signals2::scoped_connection connection;
        bool stale = true;
      };

      /**
       * @brief The entry for one kind of potential, created empty on first use.
       *
       * Held indirectly because a SparseMatrix points at its own SparsityPattern: the entries must not move
       * once a matrix has been attached to one.
       */
      Entry &entry(const PotentialKind kind)
      {
        auto &slot = entries[static_cast<uint>(kind)];
        if (!slot) slot = std::make_unique<Entry>();
        return *slot;
      }

    private:
      std::array<std::unique_ptr<Entry>, n_potential_kinds> entries;
    };

    template <int dim, typename EoMValue> dealii::Tensor<1, dim> eom_to_tensor(const EoMValue &eom)
    {
      dealii::Tensor<1, dim> out;
      for (uint d = 0; d < dim; ++d)
        out[d] = eom[d];
      return out;
    }

    template <int dim, typename NumberType> struct DG0GradientModel {
      dealii::Point<dim> center;
      dealii::Tensor<1, dim, NumberType> value;
      dealii::Tensor<2, dim, NumberType> jacobian;
    };

    template <int dim, typename VectorType, typename GradientFUN>
    std::vector<DG0GradientModel<dim, typename VectorType::value_type>>
    recover_dg0_gradient_models(const VectorType &sol, const dealii::DoFHandler<dim> &solution_dof_handler,
                                const dealii::Mapping<dim> &mapping, const GradientFUN &get_gradient)
    {
      using NumberType = typename VectorType::value_type;
      using Iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

      const uint n_cells = solution_dof_handler.get_triangulation().n_active_cells();
      std::vector<Iterator> cells(n_cells);
      std::vector<DG0GradientModel<dim, NumberType>> models(n_cells);

      dealii::QMidpoint<dim> midpoint;
      dealii::FEValues<dim> values(mapping, solution_dof_handler.get_fe(), midpoint,
                                   dealii::update_values | dealii::update_quadrature_points);
      std::vector<dealii::Vector<NumberType>> solution_values(
          1, dealii::Vector<NumberType>(solution_dof_handler.get_fe().n_components()));
      for (const auto &cell : solution_dof_handler.active_cell_iterators()) {
        const uint index = cell->active_cell_index();
        cells[index] = cell;
        values.reinit(cell);
        values.get_function_values(sol, solution_values);
        models[index].center = values.quadrature_point(0);
        models[index].value = eom_to_tensor<dim>(get_gradient(models[index].center, solution_values[0]));
      }

      const uint target_patch_size = std::min<uint>(n_cells, 2 * (dim + 1));
      for (uint cell_index = 0; cell_index < n_cells; ++cell_index) {
        std::vector<uint> patch{cell_index};
        std::vector<uint> frontier{cell_index};
        std::vector<bool> seen(n_cells, false);
        seen[cell_index] = true;

        while (patch.size() < target_patch_size && !frontier.empty()) {
          std::vector<uint> next_frontier;
          for (const uint index : frontier) {
            std::vector<Iterator> neighbors;
            dealii::GridTools::get_active_neighbors<dealii::DoFHandler<dim>>(cells[index], neighbors);
            for (const auto &neighbor : neighbors) {
              const uint neighbor_index = neighbor->active_cell_index();
              if (seen[neighbor_index]) continue;
              seen[neighbor_index] = true;
              patch.push_back(neighbor_index);
              next_frontier.push_back(neighbor_index);
            }
          }
          frontier = std::move(next_frontier);
        }

        if (patch.size() < dim + 1) continue;
        const double scale = std::max(cells[cell_index]->diameter(), std::numeric_limits<double>::epsilon());
        Eigen::MatrixXd design(patch.size(), dim + 1);
        for (uint row = 0; row < patch.size(); ++row) {
          design(row, 0) = 1.;
          for (uint d = 0; d < dim; ++d)
            design(row, d + 1) = (models[patch[row]].center[d] - models[cell_index].center[d]) / scale;
        }

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
        if (qr.rank() < dim + 1) continue;
        for (uint component = 0; component < dim; ++component) {
          Eigen::VectorXd samples(patch.size());
          for (uint row = 0; row < patch.size(); ++row)
            samples[row] = models[patch[row]].value[component];
          const Eigen::VectorXd coefficients = qr.solve(samples);
          for (uint d = 0; d < dim; ++d)
            models[cell_index].jacobian[component][d] = coefficients[d + 1] / scale;
        }
      }

      return models;
    }

    template <int dim, typename NumberType>
    dealii::Tensor<1, dim, NumberType> evaluate_dg0_gradient_model(const DG0GradientModel<dim, NumberType> &model,
                                                                   const dealii::Point<dim> &point)
    {
      auto value = model.value;
      for (uint component = 0; component < dim; ++component)
        for (uint d = 0; d < dim; ++d)
          value[component] += model.jacobian[component][d] * (point[d] - model.center[d]);
      return value;
    }

    template <int dim, typename NumberType>
    dealii::Tensor<2, dim, NumberType> symmetric_jacobian(const DG0GradientModel<dim, NumberType> &model)
    {
      dealii::Tensor<2, dim, NumberType> hessian;
      for (uint d = 0; d < dim; ++d)
        for (uint e = 0; e < dim; ++e)
          hessian[d][e] = 0.5 * (model.jacobian[d][e] + model.jacobian[e][d]);
      return hessian;
    }

    template <int dim> struct HessianJumpCompatibility {
      double relative_jump = 0.;
      double rank_one_confidence = 0.;
      bool is_rank_one = false;
      dealii::Tensor<1, dim> normal;
    };

    inline constexpr double minimum_rank_one_confidence = 0.5;

    /** Measure whether a Hessian jump is compatible with a C1 scalar potential across a smooth interface. */
    template <int dim, typename NumberType>
    HessianJumpCompatibility<dim> hessian_jump_compatibility(const dealii::Tensor<2, dim, NumberType> &left,
                                                             const dealii::Tensor<2, dim, NumberType> &right,
                                                             const dealii::Tensor<1, dim> &coordinate_scale)
    {
      Eigen::Matrix<double, dim, dim> jump;
      Eigen::Matrix<double, dim, dim> average;
      for (uint d = 0; d < dim; ++d)
        for (uint e = 0; e < dim; ++e) {
          jump(d, e) = coordinate_scale[d] * static_cast<double>(right[d][e] - left[d][e]) * coordinate_scale[e];
          average(d, e) =
              0.5 * coordinate_scale[d] * static_cast<double>(right[d][e] + left[d][e]) * coordinate_scale[e];
        }

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, dim, dim>> eigensolver(jump);
      if (eigensolver.info() != Eigen::Success) return {};

      uint leading = 0;
      double leading_value = 0.;
      double remainder_square = 0.;
      for (uint i = 0; i < dim; ++i) {
        const double value = std::abs(eigensolver.eigenvalues()[i]);
        if (value > leading_value) {
          remainder_square += leading_value * leading_value;
          leading_value = value;
          leading = i;
        } else {
          remainder_square += value * value;
        }
      }

      HessianJumpCompatibility<dim> result;
      const double epsilon = std::numeric_limits<double>::epsilon();
      result.relative_jump = jump.norm() / std::max(average.norm(), epsilon);
      result.rank_one_confidence =
          std::clamp(1. - std::sqrt(remainder_square) / std::max(leading_value, epsilon), 0., 1.);
      result.is_rank_one = leading_value > epsilon && result.rank_one_confidence >= minimum_rank_one_confidence;
      for (uint d = 0; d < dim; ++d)
        result.normal[d] = eigensolver.eigenvectors()(d, leading) / std::max(coordinate_scale[d], epsilon);
      const double normal_norm = result.normal.norm();
      if (normal_norm > 0.) result.normal /= normal_norm;
      return result;
    }

    template <int dim, typename NumberType>
    void prepare_eom_side_hessian_recovery(ReconstructedRawPotential<dim, NumberType> &potential,
                                           const std::vector<DG0GradientModel<dim, NumberType>> &models,
                                           const dealii::DoFHandler<dim> &solution_dof_handler,
                                           const double jump_threshold)
    {
      using Iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
      const uint n_cells = models.size();
      std::vector<Iterator> cells(n_cells);
      std::vector<dealii::Tensor<1, dim>> coordinate_scales(n_cells);
      auto &recovery = *potential.recovered_mass_hessian;

      recovery.sample_centers.resize(n_cells);
      recovery.samples.resize(n_cells);
      recovery.neighbors.resize(n_cells);
      recovery.face_weights.resize(n_cells);
      recovery.kink_faces_by_cell.resize(n_cells);

      for (const auto &cell : solution_dof_handler.active_cell_iterators()) {
        const uint index = cell->active_cell_index();
        cells[index] = cell;
        recovery.sample_centers[index] = models[index].center;
        recovery.samples[index] = symmetric_jacobian(models[index]);
        for (uint d = 0; d < dim; ++d) {
          double lower = std::numeric_limits<double>::max();
          double upper = -std::numeric_limits<double>::max();
          for (uint vertex = 0; vertex < cell->n_vertices(); ++vertex) {
            lower = std::min(lower, cell->vertex(vertex)[d]);
            upper = std::max(upper, cell->vertex(vertex)[d]);
          }
          coordinate_scales[index][d] = std::max(upper - lower, std::numeric_limits<double>::epsilon());
        }
      }

      for (uint index = 0; index < n_cells; ++index) {
        std::vector<Iterator> neighbors;
        dealii::GridTools::get_active_neighbors<dealii::DoFHandler<dim>>(cells[index], neighbors);
        for (const auto &neighbor : neighbors) {
          const uint neighbor_index = neighbor->active_cell_index();
          dealii::Tensor<1, dim> scale;
          for (uint d = 0; d < dim; ++d)
            scale[d] = 0.5 * (coordinate_scales[index][d] + coordinate_scales[neighbor_index][d]);
          const auto compatibility =
              hessian_jump_compatibility(recovery.samples[index], recovery.samples[neighbor_index], scale);
          const double strength =
              compatibility.is_rank_one ? compatibility.relative_jump * compatibility.rank_one_confidence : 0.;
          const double ratio = strength / std::max(jump_threshold, std::numeric_limits<double>::epsilon());
          const double face_weight = jump_threshold <= 0. ? (strength > 0. ? 0. : 1.) : 1. / (1. + std::pow(ratio, 6.));
          recovery.neighbors[index].push_back(neighbor_index);
          recovery.face_weights[index].push_back(face_weight);
          if (index < neighbor_index && face_weight < 1. - 1e-8) {
            const auto separation = recovery.sample_centers[neighbor_index] - recovery.sample_centers[index];
            const double width = separation.norm();
            if (width > 0.) {
              const uint face_index = recovery.kink_face_centers.size();
              recovery.kink_face_centers.push_back(
                  0.5 * (recovery.sample_centers[index] + recovery.sample_centers[neighbor_index]));
              recovery.kink_face_normals.push_back(compatibility.normal);
              recovery.kink_face_widths.push_back(width);
              recovery.kink_face_strengths.push_back(1. - face_weight);
              recovery.kink_faces_by_cell[index].push_back(face_index);
              recovery.kink_faces_by_cell[neighbor_index].push_back(face_index);
            }
          }
        }
      }
    }

    template <int dim> struct EoMSideHessianBlend {
      double weight = 0.;
      dealii::Tensor<1, dim> normal;
    };

    /** Smoothly localize the one-sided recovery to the detected non-analytic faces. */
    template <int dim, typename NumberType>
    EoMSideHessianBlend<dim> eom_side_hessian_blend(const ReconstructedRawPotential<dim, NumberType> &potential,
                                                    const uint root_cell, const dealii::Point<dim> &point)
    {
      EoMSideHessianBlend<dim> result;
      if (!potential.recovered_mass_hessian) return result;
      const auto &recovery = *potential.recovered_mass_hessian;
      if (root_cell >= recovery.kink_faces_by_cell.size()) return result;
      for (const uint face : recovery.kink_faces_by_cell[root_cell]) {
        const double width = recovery.kink_face_widths[face];
        const double distance = point.distance(recovery.kink_face_centers[face]);
        const double radius = 1.5 * width;
        const double weight =
            recovery.kink_face_strengths[face] * std::exp(-0.5 * distance * distance / (radius * radius));
        if (weight > result.weight) {
          result.weight = weight;
          result.normal = recovery.kink_face_normals[face];
        }
      }
      result.weight = std::clamp(result.weight, 0., 1.);
      return result;
    }

    /**
     * Recover the cellwise FV Hessians into a continuous Q1 tensor field.
     *
     * Each vertex receives the unbiased mean of all incident-cell Hessians. This removes
     * cell-face resets without introducing a preferred coordinate direction.
     */
    template <int dim, typename NumberType>
    void recover_continuous_hessian_field(ReconstructedRawPotential<dim, NumberType> &potential,
                                          const std::vector<DG0GradientModel<dim, NumberType>> &models,
                                          const dealii::DoFHandler<dim> &solution_dof_handler)
    {
      potential.recovered_mass_hessian = std::make_unique<RecoveredMassHessian<dim, NumberType>>();
      auto &recovery = *potential.recovered_mass_hessian;
      recovery.finite_element = std::make_unique<dealii::FE_Q<dim>>(1);
      recovery.dof_handler = std::make_unique<dealii::DoFHandler<dim>>(solution_dof_handler.get_triangulation());
      recovery.dof_handler->distribute_dofs(*recovery.finite_element);

      const auto n_dofs = recovery.dof_handler->n_dofs();
      std::vector<std::vector<uint>> incident_cells(n_dofs);
      std::vector<dealii::types::global_dof_index> local_dof_indices(recovery.finite_element->n_dofs_per_cell());
      for (const auto &cell : recovery.dof_handler->active_cell_iterators()) {
        cell->get_dof_indices(local_dof_indices);
        for (const auto dof : local_dof_indices)
          incident_cells[dof].push_back(cell->active_cell_index());
      }

      for (auto &values : recovery.values)
        values.reinit(n_dofs);

      for (dealii::types::global_dof_index dof = 0; dof < n_dofs; ++dof) {
        const auto &cells = incident_cells[dof];
        if (cells.empty()) continue;

        for (uint derivative_direction = 0; derivative_direction < dim; ++derivative_direction) {
          dealii::Tensor<1, dim, NumberType> mean_trace;

          for (const auto cell_index : cells) {
            const auto hessian = symmetric_jacobian(models[cell_index]);
            dealii::Tensor<1, dim, NumberType> trace;
            for (uint component = 0; component < dim; ++component)
              trace[component] = hessian[component][derivative_direction];
            mean_trace += trace;
          }
          mean_trace /= static_cast<double>(cells.size());
          for (uint component = 0; component < dim; ++component)
            recovery.values[component * dim + derivative_direction][dof] = mean_trace[component];
        }
      }

      dealii::AffineConstraints<NumberType> constraints;
      dealii::DoFTools::make_hanging_node_constraints(*recovery.dof_handler, constraints);
      constraints.close();
      for (auto &values : recovery.values)
        constraints.distribute(values);
    }

    /** Average a symmetric Hessian over the face-connected smooth component containing the evaluation point. */
    template <int dim, typename NumberType>
    std::optional<dealii::Tensor<2, dim, NumberType>>
    evaluate_eom_side_hessian(const ReconstructedRawPotential<dim, NumberType> &potential, const uint root_cell,
                              const dealii::Point<dim> &point)
    {
      if (!potential.recovered_mass_hessian) return std::nullopt;
      const auto &recovery = *potential.recovered_mass_hessian;
      if (root_cell >= recovery.neighbors.size()) return std::nullopt;

      struct ConnectedCell {
        uint index;
        uint depth;
        double connectivity;
      };
      std::vector<ConnectedCell> connected_cells{{root_cell, 0, 1.}};
      std::deque<uint> frontier{0};
      constexpr uint max_depth = 3;
      while (!frontier.empty()) {
        const uint connected_index = frontier.front();
        frontier.pop_front();
        const auto current = connected_cells[connected_index];
        if (current.depth == max_depth) continue;
        for (uint edge = 0; edge < recovery.neighbors[current.index].size(); ++edge) {
          const uint neighbor = recovery.neighbors[current.index][edge];
          const double candidate = current.connectivity * recovery.face_weights[current.index][edge];
          const auto existing = std::find_if(connected_cells.begin(), connected_cells.end(),
                                             [neighbor](const auto &cell) { return cell.index == neighbor; });
          if (existing != connected_cells.end()) {
            if (candidate <= existing->connectivity + 1e-12) continue;
            existing->depth = current.depth + 1;
            existing->connectivity = candidate;
            frontier.push_back(std::distance(connected_cells.begin(), existing));
          } else if (candidate >= 1e-6) {
            connected_cells.push_back({neighbor, current.depth + 1, candidate});
            frontier.push_back(connected_cells.size() - 1);
          }
        }
      }

      double nearest_distance = std::numeric_limits<double>::max();
      for (const uint neighbor : recovery.neighbors[root_cell])
        nearest_distance =
            std::min(nearest_distance, recovery.sample_centers[root_cell].distance(recovery.sample_centers[neighbor]));
      if (!std::isfinite(nearest_distance)) return std::nullopt;
      const double fit_radius = 1.5 * nearest_distance;
      dealii::Tensor<2, dim, NumberType> hessian;
      double weight_sum = 0.;
      for (const auto &cell : connected_cells) {
        const double distance = point.distance(recovery.sample_centers[cell.index]);
        const double spatial_weight = std::exp(-0.5 * distance * distance / (fit_radius * fit_radius));
        const double weight = cell.connectivity * spatial_weight;
        if (weight < 1e-8) continue;
        hessian += weight * recovery.samples[cell.index];
        weight_sum += weight;
      }
      if (!(weight_sum > 0.)) return std::nullopt;
      hessian /= weight_sum;
      return hessian;
    }

    template <int dim, typename NumberType> struct PotentialAssemblyScratch {
      PotentialAssemblyScratch(const dealii::Mapping<dim> &mapping, const dealii::FiniteElement<dim> &solution_fe,
                               const dealii::FiniteElement<dim> &potential_fe,
                               const dealii::Quadrature<dim> &quadrature,
                               const dealii::Quadrature<dim - 1> &face_quadrature)
          : solution_fe_values(mapping, solution_fe, quadrature, update_values | update_quadrature_points),
            potential_fe_values(mapping, potential_fe, quadrature,
                                update_gradients | update_JxW_values | update_quadrature_points),
            solution_fe_interface_values(mapping, solution_fe, face_quadrature,
                                         update_values | update_quadrature_points),
            potential_fe_interface_values(mapping, potential_fe, face_quadrature,
                                          update_values | update_gradients | update_JxW_values |
                                              update_quadrature_points | update_normal_vectors)
      {
        solution_values.resize(quadrature.size(), Vector<NumberType>(solution_fe.n_components()));
        solution_interface_values[0].resize(face_quadrature.size(), Vector<NumberType>(solution_fe.n_components()));
        solution_interface_values[1].resize(face_quadrature.size(), Vector<NumberType>(solution_fe.n_components()));
      }

      PotentialAssemblyScratch(const PotentialAssemblyScratch &scratch)
          : solution_fe_values(scratch.solution_fe_values.get_mapping(), scratch.solution_fe_values.get_fe(),
                               scratch.solution_fe_values.get_quadrature(),
                               scratch.solution_fe_values.get_update_flags()),
            potential_fe_values(scratch.potential_fe_values.get_mapping(), scratch.potential_fe_values.get_fe(),
                                scratch.potential_fe_values.get_quadrature(),
                                scratch.potential_fe_values.get_update_flags()),
            solution_fe_interface_values(scratch.solution_fe_interface_values.get_mapping(),
                                         scratch.solution_fe_interface_values.get_fe(),
                                         scratch.solution_fe_interface_values.get_quadrature(),
                                         scratch.solution_fe_interface_values.get_update_flags()),
            potential_fe_interface_values(scratch.potential_fe_interface_values.get_mapping(),
                                          scratch.potential_fe_interface_values.get_fe(),
                                          scratch.potential_fe_interface_values.get_quadrature(),
                                          scratch.potential_fe_interface_values.get_update_flags())
      {
        const uint n_components = scratch.solution_fe_values.get_fe().n_components();
        solution_values.resize(scratch.solution_fe_values.get_quadrature().size(), Vector<NumberType>(n_components));
        solution_interface_values[0].resize(scratch.solution_fe_interface_values.get_quadrature().size(),
                                            Vector<NumberType>(n_components));
        solution_interface_values[1].resize(scratch.solution_fe_interface_values.get_quadrature().size(),
                                            Vector<NumberType>(n_components));
      }

      FEValues<dim> solution_fe_values;
      FEValues<dim> potential_fe_values;
      FEInterfaceValues<dim> solution_fe_interface_values;
      FEInterfaceValues<dim> potential_fe_interface_values;
      std::vector<Vector<NumberType>> solution_values;
      std::array<std::vector<Vector<NumberType>>, 2> solution_interface_values;
    };

    /// The local system of one cell or one interior face of the potential reconstruction.
    template <typename NumberType> struct PotentialLocalSystem {
      void reinit(const uint n_dofs)
      {
        matrix.reinit(n_dofs, n_dofs);
        rhs.reinit(n_dofs);
        dof_indices.resize(n_dofs);
      }

      FullMatrix<NumberType> matrix;
      Vector<NumberType> rhs;
      std::vector<types::global_dof_index> dof_indices;
    };

    /**
     * @brief The mesh-dependent part of assemble_potential_system, kept with the cached system: the cells and
     * interior faces it visits, their local systems, and every thread's FE scratch, whose construction costs more
     * than assembling a small system.
     */
    template <int dim, typename NumberType> struct PotentialAssemblyData {
      PotentialAssemblyData(const dealii::DoFHandler<dim> &potential_dof_handler,
                            const dealii::AffineConstraints<NumberType> &constraints,
                            const dealii::Mapping<dim> &mapping, const dealii::FiniteElement<dim> &solution_fe,
                            const dealii::FiniteElement<dim> &potential_fe, const dealii::Quadrature<dim> &quadrature,
                            const dealii::Quadrature<dim - 1> &face_quadrature)
          : scratches(PotentialAssemblyScratch<dim, NumberType>(mapping, solution_fe, potential_fe, quadrature,
                                                                face_quadrature))
      {
        cells.reinit(potential_dof_handler, constraints);
        topology.reinit(cells);
        cell_systems.resize(cells.size());
        face_systems.resize(topology.faces.size());
      }

      /// The cells; their coloring goes unused, FaceTopology is built from them.
      ColoredCells<dim> cells;
      FaceTopology<dim> topology;
      tbb::enumerable_thread_specific<PotentialAssemblyScratch<dim, NumberType>> scratches;
      std::vector<PotentialLocalSystem<NumberType>> cell_systems, face_systems;
    };

    template <int dim, typename VectorType, typename EoMFUN>
    void assemble_potential_system(const VectorType &sol, const dealii::DoFHandler<dim> &solution_dof_handler,
                                   const dealii::FiniteElement<dim> &potential_fe, const dealii::Mapping<dim> &mapping,
                                   const EoMFUN &get_EoM,
                                   const dealii::AffineConstraints<typename VectorType::value_type> &constraints,
                                   PotentialAssemblyData<dim, typename VectorType::value_type> &data,
                                   dealii::SparseMatrix<typename VectorType::value_type> &matrix,
                                   dealii::Vector<typename VectorType::value_type> &rhs, const double smoothing_length,
                                   const bool assemble_matrix)
    {
      using NumberType = typename VectorType::value_type;
      using Iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
      using Scratch = PotentialAssemblyScratch<dim, NumberType>;
      using LocalSystem = PotentialLocalSystem<NumberType>;

      const bool recover_dg0_gradient = solution_dof_handler.get_fe().degree == 0;
      const auto dg0_gradient_models = recover_dg0_gradient
                                           ? recover_dg0_gradient_models(sol, solution_dof_handler, mapping, get_EoM)
                                           : std::vector<DG0GradientModel<dim, NumberType>>{};

      const auto evaluate_gradient = [&](const Iterator &cell, const dealii::Point<dim> &point,
                                         const dealii::Vector<NumberType> &values) {
        if (recover_dg0_gradient)
          return evaluate_dg0_gradient_model(dg0_gradient_models[cell->active_cell_index()], point);
        return eom_to_tensor<dim>(get_EoM(point, values));
      };

      const auto assemble_cell = [&](const Iterator &potential_cell, Scratch &scratch, LocalSystem &local) {
        const auto solution_cell = matching_dof_cell(solution_dof_handler, potential_cell);
        const uint dofs_per_cell = potential_fe.n_dofs_per_cell();

        scratch.solution_fe_values.reinit(solution_cell);
        scratch.potential_fe_values.reinit(potential_cell);
        scratch.solution_fe_values.get_function_values(sol, scratch.solution_values);

        local.reinit(dofs_per_cell);
        potential_cell->get_dof_indices(local.dof_indices);

        for (const auto q : scratch.potential_fe_values.quadrature_point_indices()) {
          const auto eom = evaluate_gradient(solution_cell, scratch.potential_fe_values.quadrature_point(q),
                                             scratch.solution_values[q]);

          for (uint i = 0; i < dofs_per_cell; ++i) {
            const auto grad_i = scratch.potential_fe_values.shape_grad(i, q);
            local.rhs(i) += scratch.potential_fe_values.JxW(q) * scalar_product(eom, grad_i);

            if (!assemble_matrix) continue;
            for (uint j = 0; j < dofs_per_cell; ++j)
              local.matrix(i, j) += scratch.potential_fe_values.JxW(q) *
                                    scalar_product(scratch.potential_fe_values.shape_grad(j, q), grad_i);
          }
        }
      };
      // Every constraint on the potential is homogeneous -- hanging nodes, plus the gauge DoF pinned to
      // zero -- so distributing the right-hand side alone needs no local matrix to carry an inhomogeneity.
      const auto insert = [&](const FullMatrix<NumberType> &local_matrix, const Vector<NumberType> &local_rhs,
                              const std::vector<types::global_dof_index> &dofs) {
        if (assemble_matrix)
          constraints.distribute_local_to_global(local_matrix, local_rhs, dofs, matrix, rhs);
        else
          constraints.distribute_local_to_global(local_rhs, dofs, rhs);
      };
      const auto assemble_face = [&](const Iterator &potential_cell, const uint face_no, const uint subface_no,
                                     const Iterator &potential_neighbor, const uint neighbor_face_no,
                                     const uint neighbor_subface_no, Scratch &scratch, LocalSystem &local) {
        const auto solution_cell = matching_dof_cell(solution_dof_handler, potential_cell);
        const auto solution_neighbor = matching_dof_cell(solution_dof_handler, potential_neighbor);

        scratch.solution_fe_interface_values.reinit(solution_cell, face_no, subface_no, solution_neighbor,
                                                    neighbor_face_no, neighbor_subface_no);
        scratch.potential_fe_interface_values.reinit(potential_cell, face_no, subface_no, potential_neighbor,
                                                     neighbor_face_no, neighbor_subface_no);

        const auto &solution_fe_interface_values = scratch.solution_fe_interface_values;
        const auto &potential_fe_interface_values = scratch.potential_fe_interface_values;

        // The potential is scalar, so the plain shape functions serve; FEValues views (operator[]) are built lazily
        // through a deal.II task the caller waits for, which deadlocks when every TBB worker does it at once.
        const uint n_interface_dofs = potential_fe_interface_values.n_current_interface_dofs();
        local.reinit(n_interface_dofs);
        local.dof_indices = potential_fe_interface_values.get_interface_dof_indices();
        const auto &solution_fe_values_s = solution_fe_interface_values.get_fe_face_values(0);
        const auto &solution_fe_values_n = solution_fe_interface_values.get_fe_face_values(1);
        const auto &q_points = potential_fe_interface_values.get_quadrature_points();

        auto &solution_values_s = scratch.solution_interface_values[0];
        auto &solution_values_n = scratch.solution_interface_values[1];
        solution_fe_values_s.get_function_values(sol, solution_values_s);
        solution_fe_values_n.get_function_values(sol, solution_values_n);

        const double h_face = std::min(solution_cell->diameter(), solution_neighbor->diameter());
        const double h_normal = std::min(face_normal_cell_width(solution_cell, face_no),
                                         face_normal_cell_width(solution_neighbor, neighbor_face_no));
        const uint degree = potential_fe.degree;
        const double tau = 10. * (degree + 1.) * (degree + 1.) / h_face;
        const double gradient_jump_weight = smoothing_length * smoothing_length / h_normal;

        for (const auto q : potential_fe_interface_values.quadrature_point_indices()) {
          const auto normal = potential_fe_interface_values.normal_vector(q);
          const auto eom_s = evaluate_gradient(solution_cell, q_points[q], solution_values_s[q]);
          const auto eom_n = evaluate_gradient(solution_neighbor, q_points[q], solution_values_n[q]);
          const auto average_eom = 0.5 * (eom_s + eom_n);
          const double rhs_flux = scalar_product(average_eom, normal);

          for (uint i = 0; i < n_interface_dofs; ++i) {
            const double jump_i = potential_fe_interface_values.jump_in_shape_values(i, q);
            const Tensor<1, dim> average_grad_i = potential_fe_interface_values.average_of_shape_gradients(i, q);
            const double normal_gradient_jump_i =
                scalar_product(potential_fe_interface_values.jump_in_shape_gradients(i, q), normal);

            local.rhs(i) += -potential_fe_interface_values.JxW(q) * rhs_flux * jump_i;

            if (!assemble_matrix) continue;
            for (uint j = 0; j < n_interface_dofs; ++j) {
              const double jump_j = potential_fe_interface_values.jump_in_shape_values(j, q);
              const Tensor<1, dim> average_grad_j = potential_fe_interface_values.average_of_shape_gradients(j, q);
              const double normal_gradient_jump_j =
                  scalar_product(potential_fe_interface_values.jump_in_shape_gradients(j, q), normal);

              local.matrix(i, j) +=
                  potential_fe_interface_values.JxW(q) *
                  (-scalar_product(average_grad_j, normal) * jump_i - scalar_product(average_grad_i, normal) * jump_j +
                   tau * jump_j * jump_i + gradient_jump_weight * normal_gradient_jump_i * normal_gradient_jump_j);
            }
          }
        }
      };

      // Driven by the POTENTIAL DoFHandler, not the solution one: solve_potential gives the potential its own
      // serial copy of the mesh, so every rank visits every cell and every face, and the traversal is the same on
      // every rank and in the serial case. The cells and the interior faces (each once, as FaceTopology lists them)
      // are assembled in parallel into their own local systems, then inserted in a fixed order.
      const auto &cells = data.cells;
      const auto &topology = data.topology;
      auto &scratches = data.scratches;
      auto &cell_systems = data.cell_systems;
      auto &face_systems = data.face_systems;
      tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const auto &r) {
        for (size_t k = r.begin(); k != r.end(); ++k)
          assemble_cell(cells[k], scratches.local(), cell_systems[k]);
      });
      tbb::parallel_for(tbb::blocked_range<size_t>(0, topology.faces.size()), [&](const auto &r) {
        for (size_t f = r.begin(); f != r.end(); ++f) {
          const auto &face = topology.faces[f];
          assemble_face(Iterator(face.cell[0]), face.face_no[0], face.subface_no[0], Iterator(face.cell[1]),
                        face.face_no[1], face.subface_no[1], scratches.local(), face_systems[f]);
        }
      });
      for (const auto &local : cell_systems)
        insert(local.matrix, local.rhs, local.dof_indices);
      for (const auto &local : face_systems)
        insert(local.matrix, local.rhs, local.dof_indices);
    }

    template <int dim, typename NumberType> struct PotentialMinimum {
      dealii::Point<dim> point;
      double value = std::numeric_limits<double>::max();
    };

    template <int dim> struct LocalPotentialEvaluation {
      double value = 0.;
      dealii::Tensor<1, dim> gradient;
      dealii::Tensor<2, dim> hessian;
    };

    template <int dim>
    LocalPotentialEvaluation<dim> evaluate_local_potential(const dealii::FiniteElement<dim> &potential_fe,
                                                           const std::vector<double> &local_values,
                                                           const dealii::Point<dim> &point)
    {
      LocalPotentialEvaluation<dim> evaluation;
      for (uint i = 0; i < potential_fe.n_dofs_per_cell(); ++i) {
        evaluation.value += local_values[i] * potential_fe.shape_value(i, point);
        evaluation.gradient += local_values[i] * potential_fe.shape_grad(i, point);
        evaluation.hessian += local_values[i] * potential_fe.shape_grad_grad(i, point);
      }
      return evaluation;
    }

    template <int dim>
    bool projected_newton_direction(const dealii::Tensor<2, dim> &hessian,
                                    const dealii::Tensor<1, dim> &projected_gradient,
                                    const std::array<bool, dim> &fixed, dealii::Tensor<1, dim> &direction)
    {
      std::array<uint, dim> free_indices{};
      uint n_free = 0;
      for (uint d = 0; d < dim; ++d)
        if (!fixed[d]) free_indices[n_free++] = d;

      if (n_free == 0) return false;

      Eigen::MatrixXd reduced_hessian(n_free, n_free);
      Eigen::VectorXd reduced_gradient(n_free);
      double hessian_norm = 0.;
      for (uint i = 0; i < n_free; ++i) {
        reduced_gradient[i] = projected_gradient[free_indices[i]];
        for (uint j = 0; j < n_free; ++j) {
          reduced_hessian(i, j) = hessian[free_indices[i]][free_indices[j]];
          hessian_norm = std::max(hessian_norm, std::abs(reduced_hessian(i, j)));
        }
      }
      if (!reduced_hessian.allFinite() || !reduced_gradient.allFinite()) return false;

      const double positivity_threshold = 100. * std::numeric_limits<double>::epsilon() * std::max(1., hessian_norm);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(reduced_hessian);
      if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
          !(solver.eigenvalues().minCoeff() > positivity_threshold))
        return false;

      const Eigen::VectorXd reduced_direction =
          -solver.eigenvectors() *
          (solver.eigenvectors().transpose() * reduced_gradient).cwiseQuotient(solver.eigenvalues());
      if (!reduced_direction.allFinite()) return false;

      direction = {};
      for (uint i = 0; i < n_free; ++i)
        direction[free_indices[i]] = reduced_direction[i];
      return true;
    }

    template <int dim, typename NumberType>
    PotentialMinimum<dim, NumberType>
    refine_cell_minimum(const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                        const dealii::FiniteElement<dim> &potential_fe, const dealii::Mapping<dim> &mapping,
                        const dealii::Vector<NumberType> &potential, const dealii::Quadrature<dim> &quadrature,
                        const Config::EoMConfig &config,
                        const std::optional<dealii::Point<dim>> &initial_unit_point = std::nullopt)
    {
      std::vector<dealii::types::global_dof_index> local_dof_indices(potential_fe.n_dofs_per_cell());
      cell->get_dof_indices(local_dof_indices);

      std::vector<double> local_values(potential_fe.n_dofs_per_cell());
      for (uint i = 0; i < potential_fe.n_dofs_per_cell(); ++i)
        local_values[i] = (double)potential[local_dof_indices[i]];

      dealii::Point<dim> sampled_point = quadrature.point(0);
      auto sampled_evaluation = evaluate_local_potential(potential_fe, local_values, sampled_point);
      const auto consider_seed = [&](const dealii::Point<dim> &candidate) {
        const auto candidate_evaluation = evaluate_local_potential(potential_fe, local_values, candidate);
        if (candidate_evaluation.value < sampled_evaluation.value) {
          sampled_point = candidate;
          sampled_evaluation = candidate_evaluation;
        }
      };

      for (const auto &support_point : potential_fe.get_unit_support_points())
        consider_seed(support_point);
      for (const auto &quadrature_point : quadrature.get_points())
        consider_seed(quadrature_point);

      const double gradient_tolerance = std::max(config.abs_tol, 100. * std::numeric_limits<double>::epsilon());

      const auto refine_from_seed = [&](dealii::Point<dim> point) {
        auto evaluation = evaluate_local_potential(potential_fe, local_values, point);
        for (uint iteration = 0; iteration < config.max_iter; ++iteration) {
          std::array<bool, dim> fixed{};
          dealii::Tensor<1, dim> projected_gradient;
          double projected_gradient_norm = 0.;
          for (uint d = 0; d < dim; ++d) {
            const bool fixed_at_lower = point[d] <= config.bound_tolerance && evaluation.gradient[d] > 0.;
            const bool fixed_at_upper = point[d] >= 1. - config.bound_tolerance && evaluation.gradient[d] < 0.;
            fixed[d] = fixed_at_lower || fixed_at_upper;
            projected_gradient[d] = fixed[d] ? 0. : evaluation.gradient[d];
            const double extent = std::max(cell->extent_in_direction(d), std::numeric_limits<double>::epsilon());
            projected_gradient_norm = std::max(projected_gradient_norm, std::abs(projected_gradient[d]) / extent);
          }
          if (projected_gradient_norm <= gradient_tolerance) break;

          dealii::Tensor<1, dim> direction;
          bool used_newton = projected_newton_direction<dim>(evaluation.hessian, projected_gradient, fixed, direction);
          double directional_derivative = dealii::scalar_product(evaluation.gradient, direction);
          if (!used_newton || !(directional_derivative < 0.) || !std::isfinite(directional_derivative)) {
            direction = -projected_gradient;
            directional_derivative = dealii::scalar_product(evaluation.gradient, direction);
            used_newton = false;
          }
          if (!(directional_derivative < 0.) || !std::isfinite(directional_derivative)) break;

          const auto try_step = [&](const dealii::Tensor<1, dim> &search_direction, const double search_derivative,
                                    dealii::Point<dim> &accepted_point,
                                    LocalPotentialEvaluation<dim> &accepted_evaluation) {
            double alpha = 1.;
            for (uint d = 0; d < dim; ++d) {
              if (search_direction[d] > 0.)
                alpha = std::min(alpha, (1. - point[d]) / search_direction[d]);
              else if (search_direction[d] < 0.)
                alpha = std::min(alpha, -point[d] / search_direction[d]);
            }
            if (!(alpha > 0.) || !std::isfinite(alpha)) return false;

            for (uint backtrack = 0; backtrack < config.max_backtracks; ++backtrack) {
              auto candidate = point;
              for (uint d = 0; d < dim; ++d)
                candidate[d] = std::clamp(point[d] + alpha * search_direction[d], 0., 1.);
              const auto candidate_evaluation = evaluate_local_potential(potential_fe, local_values, candidate);
              if (candidate_evaluation.value <=
                  evaluation.value + config.armijo_coefficient * alpha * search_derivative) {
                accepted_point = candidate;
                accepted_evaluation = candidate_evaluation;
                return true;
              }
              alpha *= 0.5;
            }
            return false;
          };

          dealii::Point<dim> next_point;
          LocalPotentialEvaluation<dim> next_evaluation;
          bool accepted = try_step(direction, directional_derivative, next_point, next_evaluation);
          if (!accepted && used_newton) {
            direction = -projected_gradient;
            directional_derivative = dealii::scalar_product(evaluation.gradient, direction);
            if (directional_derivative < 0.)
              accepted = try_step(direction, directional_derivative, next_point, next_evaluation);
          }
          if (!accepted || !(next_evaluation.value < evaluation.value)) break;

          point = next_point;
          evaluation = next_evaluation;
        }
        return std::pair{point, evaluation};
      };

      auto [point, evaluation] = refine_from_seed(sampled_point);
      if (initial_unit_point) {
        auto warm_start = *initial_unit_point;
        bool valid = true;
        for (uint d = 0; d < dim; ++d) {
          valid = valid && std::isfinite(warm_start[d]) && warm_start[d] >= -config.bound_tolerance &&
                  warm_start[d] <= 1. + config.bound_tolerance;
          warm_start[d] = std::clamp(warm_start[d], 0., 1.);
        }
        if (valid) {
          auto [warm_point, warm_evaluation] = refine_from_seed(warm_start);
          if (warm_evaluation.value < evaluation.value) {
            point = warm_point;
            evaluation = warm_evaluation;
          }
        }
      }

      return {.point = mapping.transform_unit_to_real_cell(cell, point), .value = evaluation.value};
    }

    template <int dim, typename NumberType>
    PotentialMinimum<dim, NumberType> find_potential_minimum(
        const dealii::DoFHandler<dim> &potential_dof_handler, const dealii::FiniteElement<dim> &potential_fe,
        const dealii::Mapping<dim> &mapping, const dealii::Vector<NumberType> &potential,
        const dealii::Quadrature<dim> &quadrature, const Config::EoMConfig &config,
        const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt,
        const std::map<dealii::types::global_dof_index, dealii::Point<dim>> *cached_support_points = nullptr)
    {
      PotentialMinimum<dim, NumberType> minimum;

      const auto own_support_points = cached_support_points != nullptr
                                          ? std::map<dealii::types::global_dof_index, dealii::Point<dim>>{}
                                          : DoFTools::map_dofs_to_support_points(mapping, potential_dof_handler);
      const auto &support_points = cached_support_points != nullptr ? *cached_support_points : own_support_points;
      for (const auto &dof_and_point : support_points) {
        const auto dof = dof_and_point.first;
        if ((double)potential[dof] < minimum.value) {
          minimum.value = (double)potential[dof];
          minimum.point = dof_and_point.second;
        }
      }

      FEValues<dim> potential_fe_values(mapping, potential_fe, quadrature, update_values | update_quadrature_points);
      std::vector<NumberType> potential_values(quadrature.size());
      for (const auto &cell : potential_dof_handler.active_cell_iterators()) {
        potential_fe_values.reinit(cell);
        potential_fe_values.get_function_values(potential, potential_values);

        for (const auto q : potential_fe_values.quadrature_point_indices()) {
          if ((double)potential_values[q] < minimum.value) {
            minimum.value = (double)potential_values[q];
            minimum.point = potential_fe_values.quadrature_point(q);
          }
        }
      }

      const auto minimum_cell =
          GridTools::find_active_cell_around_point(mapping, potential_dof_handler, minimum.point).first;
      using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
      struct CandidateCell {
        CellIterator cell;
        std::optional<dealii::Point<dim>> initial_unit_point;
      };
      std::vector<CandidateCell> candidate_cells;
      const auto add_candidate = [&](const CellIterator &cell,
                                     const std::optional<dealii::Point<dim>> &initial_unit_point = std::nullopt) {
        const auto candidate = std::find_if(candidate_cells.begin(), candidate_cells.end(),
                                            [&](const auto &entry) { return entry.cell == cell; });
        if (candidate == candidate_cells.end())
          candidate_cells.push_back({cell, initial_unit_point});
        else if (initial_unit_point)
          candidate->initial_unit_point = initial_unit_point;
      };
      const auto add_cell_and_neighbors = [&](const CellIterator &cell) {
        add_candidate(cell);
        std::vector<CellIterator> active_neighbors;
        GridTools::get_active_neighbors<dealii::DoFHandler<dim>>(cell, active_neighbors);
        for (const auto &neighbor : active_neighbors)
          add_candidate(neighbor);
      };

      add_cell_and_neighbors(minimum_cell);
      if (initial_guess) {
        bool finite = true;
        for (uint d = 0; d < dim; ++d)
          finite = finite && std::isfinite((*initial_guess)[d]);
        const auto bounding_box = GridTools::compute_bounding_box(potential_dof_handler.get_triangulation());
        if (finite && bounding_box.point_inside(*initial_guess)) {
          try {
            const auto initial_cell =
                GridTools::find_active_cell_around_point(mapping, potential_dof_handler, *initial_guess).first;
            if (initial_cell != potential_dof_handler.end()) {
              add_cell_and_neighbors(initial_cell);
              add_candidate(initial_cell, mapping.transform_real_to_unit_cell(initial_cell, *initial_guess));
            }
          } catch (const dealii::ExceptionBase &) {
          }
        }
      }

      for (const auto &candidate : candidate_cells) {
        const auto &cell = candidate.cell;
        const auto refined = refine_cell_minimum<dim, NumberType>(cell, potential_fe, mapping, potential, quadrature,
                                                                  config, candidate.initial_unit_point);
        if (refined.value < minimum.value) minimum = refined;
      }

      return minimum;
    }

    template <int dim, typename VectorType, typename GradientFUN>
    ReconstructedRawPotential<dim, typename VectorType::value_type>
    solve_potential(const VectorType &sol, const dealii::DoFHandler<dim> &solution_dof_handler,
                    const dealii::Mapping<dim> &mapping, const GradientFUN &get_gradient,
                    const Config::EoMConfig &config, const PotentialKind kind,
                    PotentialSystemCache<dim, typename VectorType::value_type> *cache)
    {
      using NumberType = typename VectorType::value_type;

      auto &solve_stats = potential_solve_stats();
      ++solve_stats.solves;
      ScopedTimer solve_timer(solve_stats.seconds);

      auto origin_cell = solution_dof_handler.begin_active();
      const auto origin = get_origin(solution_dof_handler, origin_cell);
      const double smoothing_length = resolve_potential_smoothing_length(solution_dof_handler, config.smoothing_length);

      // The potential gets its own SERIAL mesh whenever the solution lives on a partitioned one.
      //
      // Two things force this, and either alone is fatal:
      //  * distribute_dofs() on a parallel triangulation is collective (MPI_Allgather inside
      //    ParallelShared::distribute_dofs), while this whole call chain runs inside readouts(),
      //    which OutputSession invokes on rank 0 only. Rank 0 would enter a collective nobody
      //    else reaches, and the run hangs with no diagnostic.
      //  * even if it did not hang, the assembly over a partitioned mesh visits only the calling
      //    rank's (owned) cells, so the potential system would be assembled with entire rows left
      //    at zero -- a singular matrix handed to UMFPACK, or silently wrong readouts.
      //
      // The mesh is replicated at this rung and the solution is a full replica, so a serial copy is
      // complete and every rank would compute the identical potential from it.
      const auto &potential_triangulation = serial_mirror(solution_dof_handler.get_triangulation());

      // Callers that pass no cache get one that lives for this call only, which is the behaviour this
      // function had throughout: build everything, factorize, solve, throw it all away.
      PotentialSystemCache<dim, NumberType> uncached;
      auto &system = (cache != nullptr ? *cache : uncached).entry(kind);

      // The eom potential is always quadratic; the raw potential's order is a config choice, so a
      // change of /discretization/EoM/raw_potential_order must invalidate the cached system.
      const uint potential_order = kind == PotentialKind::eom ? 2 : config.raw_potential_order;
      const bool rebuild =
          !system.describes(potential_triangulation, solution_dof_handler, smoothing_length, potential_order);
      if (rebuild) {
        // The old assembly data refers to the FE and DoF handler replaced below.
        system.assembly.reset();
        system.finite_element = make_potential_fe<dim>(kind, potential_order);
        system.dof_handler = std::make_shared<DoFHandler<dim>>(potential_triangulation);
        system.dof_handler->distribute_dofs(*system.finite_element);

        system.support_points = DoFTools::map_dofs_to_support_points(mapping, *system.dof_handler);

        system.constraints.clear();
        DoFTools::make_hanging_node_constraints(*system.dof_handler, system.constraints);
        const auto gauge_dof = select_gauge_dof(system.constraints, origin, system.support_points);
        system.constraints.add_line(gauge_dof);
        system.constraints.set_inhomogeneity(gauge_dof, 0.);
        system.constraints.close();

        DynamicSparsityPattern dsp(system.dof_handler->n_dofs());
        DoFTools::make_flux_sparsity_pattern(*system.dof_handler, dsp, system.constraints,
                                             /*keep_constrained_dofs = */ true);
        system.matrix.clear();
        system.sparsity_pattern.copy_from(dsp);
        system.matrix.reinit(system.sparsity_pattern);

        const uint quadrature_order =
            std::max<uint>(std::max<uint>(solution_dof_handler.get_fe().degree, system.finite_element->degree) + 2, 2);
        system.quadrature = QGauss<dim>(quadrature_order);
        system.face_quadrature = QGauss<dim - 1>(quadrature_order);
        system.assembly = std::make_unique<PotentialAssemblyData<dim, NumberType>>(
            *system.dof_handler, system.constraints, mapping, solution_dof_handler.get_fe(), *system.finite_element,
            system.quadrature, system.face_quadrature);
      }

      Vector<NumberType> rhs(system.dof_handler->n_dofs());
      assemble_potential_system(sol, solution_dof_handler, *system.finite_element, mapping, get_gradient,
                                system.constraints, *system.assembly, system.matrix, rhs, smoothing_length,
                                /*assemble_matrix = */ rebuild);

      if (rebuild) {
        system.factorization.initialize(system.matrix);
        system.note_built_for(potential_triangulation, solution_dof_handler, smoothing_length);
      }

      Vector<NumberType> potential(system.dof_handler->n_dofs());
      system.factorization.vmult(potential, rhs);
      system.constraints.distribute(potential);

      return {
          .finite_element = system.finite_element, .dof_handler = system.dof_handler, .values = std::move(potential)};
    }

    template <int dim, typename VectorType, typename EoMFUN>
    ReconstructedEoMPotential<dim, typename VectorType::value_type>
    reconstruct_potential(typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
                          const dealii::DoFHandler<dim> &solution_dof_handler, const dealii::Mapping<dim> &mapping,
                          const EoMFUN &get_EoM, const Config::EoMConfig &config,
                          const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt,
                          PotentialSystemCache<dim, typename VectorType::value_type> *cache = nullptr)
    {
      auto potential = solve_potential(sol, solution_dof_handler, mapping, get_EoM, config, PotentialKind::eom, cache);

      // The minimum search wants the same quadrature the system was assembled with, and the same support
      // points the gauge choice used. Both come from the cache when there is one: deal.II builds Gauss
      // points by Newton iteration in long double, and the support point map is one node per DoF.
      const uint quadrature_order =
          std::max<uint>(std::max<uint>(solution_dof_handler.get_fe().degree, potential.finite_element->degree) + 2, 2);
      const std::optional<QGauss<dim>> own_quadrature =
          cache != nullptr ? std::nullopt : std::optional<QGauss<dim>>(quadrature_order);
      const auto *system = cache != nullptr ? &cache->entry(PotentialKind::eom) : nullptr;

      const auto minimum =
          find_potential_minimum(*potential.dof_handler, *potential.finite_element, mapping, potential.values,
                                 system != nullptr ? system->quadrature : *own_quadrature, config, initial_guess,
                                 system != nullptr ? &system->support_points : nullptr);
      EoM_cell = GridTools::find_active_cell_around_point(solution_dof_handler, minimum.point);
      return {.minimum = minimum.point,
              .finite_element = std::move(potential.finite_element),
              .dof_handler = std::move(potential.dof_handler),
              .values = std::move(potential.values)};
    }

  } // namespace internal

  /**
   * @brief Reconstruct a scalar raw potential without locating its minimum.
   *
   * The supplied callback must return the unmodified gradient of the desired scalar potential. Unlike an EoM callback,
   * it must not include explicit-breaking or other terms which should be absent from readouts and extractors.
   */
  template <int dim, typename VectorType, typename GradientFUN>
  ReconstructedRawPotential<dim, typename VectorType::value_type>
  reconstruct_raw_potential(const VectorType &sol, const dealii::DoFHandler<dim> &dof_handler,
                            const dealii::Mapping<dim> &mapping, const GradientFUN &get_gradient,
                            const Config::EoMConfig &config,
                            internal::PotentialSystemCache<dim, typename VectorType::value_type> *cache = nullptr)
  {
    static_assert(dim > 0, "A raw spatial potential cannot be reconstructed in zero dimensions.");
    config.validate();
    auto potential =
        internal::solve_potential(sol, dof_handler, mapping, get_gradient, config, internal::PotentialKind::raw, cache);
    if (config.raw_potential_recover_mass_hessian && dof_handler.get_fe().degree == 0) {
      const auto models = internal::recover_dg0_gradient_models(sol, dof_handler, mapping, get_gradient);
      internal::recover_continuous_hessian_field(potential, models, dof_handler);
      if (config.raw_potential_mass_hessian_jump_threshold >= 0.)
        internal::prepare_eom_side_hessian_recovery(potential, models, dof_handler,
                                                    config.raw_potential_mass_hessian_jump_threshold);
    }
    return potential;
  }

  /**
   * @brief Evaluate a reconstructed raw potential and its mass-extraction Hessian at a real-space point.
   *
   * `potential_hessian` is exactly the second derivative of `value`. `mass_hessian` equals it unless FV mass-Hessian
   * recovery was requested, in which case it contains the independently recovered tensor intended for observables.
   */
  template <int dim, typename NumberType>
  RawPotentialEvaluation<dim, NumberType>
  evaluate_raw_potential(const ReconstructedRawPotential<dim, NumberType> &potential,
                         const dealii::Mapping<dim> &mapping, const dealii::Point<dim> &point)
  {
    const auto cell = dealii::GridTools::find_active_cell_around_point(mapping, *potential.dof_handler, point).first;
    const auto unit_point = mapping.transform_real_to_unit_cell(cell, point);
    dealii::FEValues<dim> fe_values(mapping, *potential.finite_element, unit_point,
                                    dealii::update_values | dealii::update_gradients | dealii::update_hessians);
    fe_values.reinit(cell);

    std::vector<NumberType> values(1);
    std::vector<dealii::Tensor<1, dim, NumberType>> gradients(1);
    std::vector<dealii::Tensor<2, dim, NumberType>> potential_hessians(1);
    fe_values.get_function_values(potential.values, values);
    fe_values.get_function_gradients(potential.values, gradients);
    fe_values.get_function_hessians(potential.values, potential_hessians);
    auto mass_hessian = potential_hessians[0];
    if (potential.recovered_mass_hessian) {
      const auto &recovery = *potential.recovered_mass_hessian;
      const auto hessian_cell = internal::matching_dof_cell(*recovery.dof_handler, cell);
      dealii::FEValues<dim> hessian_fe_values(mapping, *recovery.finite_element, unit_point, dealii::update_values);
      hessian_fe_values.reinit(hessian_cell);
      for (uint d = 0; d < dim; ++d)
        for (uint e = 0; e < dim; ++e) {
          std::vector<NumberType> component_values(1);
          hessian_fe_values.get_function_values(recovery.values[d * dim + e], component_values);
          mass_hessian[d][e] = component_values[0];
        }
      mass_hessian = 0.5 * (mass_hessian + transpose(mass_hessian));
      if (const auto eom_side_hessian =
              internal::evaluate_eom_side_hessian(potential, cell->active_cell_index(), point)) {
        const auto blend = internal::eom_side_hessian_blend(potential, cell->active_cell_index(), point);
        const auto correction = *eom_side_hessian - mass_hessian;
        const double normal_correction = blend.normal * correction * blend.normal;
        mass_hessian += blend.weight * normal_correction * outer_product(blend.normal, blend.normal);
      }
    }
    return {.value = values[0],
            .gradient = gradients[0],
            .potential_hessian = potential_hessians[0],
            .mass_hessian = mass_hessian};
  }

  /**
   * @brief Reconstruct a potential whose gradient approximates the model EoM vector field and return a sampled and
   * locally refined minimum of that potential.
   */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  EoMResult<dim, typename VectorType::value_type>
  get_EoM_point_with_potential(typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
                               const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping,
                               const EoMFUN &get_EoM, const EoMPFUN &EoM_postprocess, const Config::EoMConfig &config,
                               const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt,
                               internal::PotentialSystemCache<dim, typename VectorType::value_type> *cache = nullptr)
  {
    config.validate();
    if (config.max_iter == 0) return {.point = internal::get_origin(dof_handler, EoM_cell), .potential = std::nullopt};

    auto potential =
        internal::reconstruct_potential(EoM_cell, sol, dof_handler, mapping, get_EoM, config, initial_guess, cache);
    auto EoM = potential.minimum;

    dealii::Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
    dealii::Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);
    fe_function.set_active_cell(EoM_cell);
    fe_function.vector_value(EoM, values);

    EoM = EoM_postprocess(EoM, values);
    EoM_cell = dealii::GridTools::find_active_cell_around_point(dof_handler, EoM);

    return {.point = EoM, .potential = std::move(potential)};
  }

  /** Compatibility overload accepting the historical scalar EoM settings. */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  EoMResult<dim, typename VectorType::value_type> get_EoM_point_with_potential(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, [[maybe_unused]] const auto &values) { return p; },
      const double EoM_abs_tol = Config::EoMConfig::default_abs_tol,
      const uint max_iter = Config::EoMConfig::default_max_iter,
      const double EoM_smoothing_length = Config::EoMConfig::default_smoothing_length,
      const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt,
      internal::PotentialSystemCache<dim, typename VectorType::value_type> *cache = nullptr)
  {
    return get_EoM_point_with_potential(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess,
                                        Config::EoMConfig(EoM_abs_tol, max_iter, EoM_smoothing_length), initial_guess,
                                        cache);
  }

  /**
   * @brief Reconstruct a potential whose gradient approximates the model EoM vector field and return a sampled and
   * locally refined minimum of that potential.
   */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<dim>
  get_EoM_point(typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
                const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
                const EoMPFUN &EoM_postprocess, const Config::EoMConfig &config,
                const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt,
                internal::PotentialSystemCache<dim, typename VectorType::value_type> *cache = nullptr)
  {
    return get_EoM_point_with_potential(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess, config,
                                        initial_guess, cache)
        .point;
  }

  /** Compatibility overload accepting the historical scalar EoM settings. */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<dim> get_EoM_point(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, [[maybe_unused]] const auto &values) { return p; },
      const double EoM_abs_tol = Config::EoMConfig::default_abs_tol,
      const uint max_iter = Config::EoMConfig::default_max_iter,
      const double EoM_smoothing_length = Config::EoMConfig::default_smoothing_length,
      const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt)
  {
    return get_EoM_point(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess,
                         Config::EoMConfig(EoM_abs_tol, max_iter, EoM_smoothing_length), initial_guess);
  }
} // namespace DiFfRG
