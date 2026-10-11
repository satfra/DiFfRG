#pragma once

// external libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

// standard library
#include <algorithm>
#include <array>
#include <concepts>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// DiFfRG
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/solution_view.hh>
#include <DiFfRG/model/model.hh>

namespace DiFfRG
{
  using namespace dealii;

  namespace internal
  {
    /// A stand-in solution type to call the indicator hooks with.
    struct IndicatorProbe {
    };

    /// Whether @p Model defines its own cell_indicator or face_indicator instead of inheriting AbstractModel's.
    template <typename Model> consteval bool defines_refinement_indicator()
    {
      // Hooks that cannot be called like AbstractModel's hide them, so they are the model's own as well.
      return !requires(const Model &model, double &value, std::array<double, 2> &values, const Point<1> &x,
                       const Tensor<1, 1> &normal, const IndicatorProbe &sol) {
        { model.cell_indicator(value, x, sol) } -> std::same_as<def::DefaultIndicator>;
        { model.face_indicator(values, normal, x, sol, sol) } -> std::same_as<def::DefaultIndicator>;
      };
    }
  } // namespace internal

  // clang-format off
  /**
   * @brief Adaptive mesh refinement and coarsening (h-adaptivity) driven by a refinement indicator.
   *
   * At each adaptation, one non-negative indicator \f$\eta_K\f$ is computed per cell \f$K\f$, chosen by `indicator`:
   * - `"model"`: Assembler::refinement_indicator, from the model's cell_indicator and face_indicator.
   * - `"curvature"`: \f$\eta_K = \max_c h_K^2\, |\partial^2 u_c|_K / s_c\f$, with \f$h_K\f$ the cell diameter,
   *   \f$|\partial^2 u_c|_K\f$ the size of the second derivative of component \f$c\f$ (FV: of the parabola through
   *   neighbouring averages, Assembler::cell_curvatures; FEM: deal.II's DerivativeApproximation), and \f$s_c\f$ the
   *   range \f$\max u_c - \min u_c\f$ over the domain. \f$h_K^2 |\partial^2 u|/8\f$ is the deviation of \f$u\f$ from
   *   its linear interpolant on the cell, so \f$\eta_K\f$ is about 8 times the relative resolution error. It grows
   *   wherever a kink, a shock or a narrow extremum forms and drops by 4 with each refinement.
   * - `"auto"` (default): `"model"` if the model defines cell_indicator or face_indicator, `"curvature"` otherwise.
   *
   * With `predict` (default), each cell's indicator is extrapolated to the next adaptation: its growth since the
   * previous one is added, \f$\eta_K + \max(0, \eta_K - \eta_K^{\rm prev})\f$. A forming kink is then refined one
   * adaptation earlier, while the solution there is still smooth enough for the transfer to the new mesh to be cheap
   * for the timestepper; with longer `adapt_dt` it also keeps the kink resolved in between. (A cell refined since
   * the previous adaptation compares with a quarter of its parent's value, one coarsened with four times its largest
   * child's, as \f$\eta_K \propto h_K^2\f$.)
   *
   * The cells are then marked by `marking`:
   * - `"tolerance"` (default): cells with \f$\eta_K >\f$ `tolerance` are refined, cells with \f$\eta_K <\f$
   *   `coarsen_tolerance` coarsened (the equidistribution strategy of time-dependent adaptive FEM). A smooth solution
   *   keeps the initial mesh; refinement starts where a feature becomes unresolved, and a mesh that resolves the
   *   solution is kept, with no IDA restart. The tolerance is absolute, so it needs an indicator with a known scale,
   *   like `"curvature"`. `coarsen_tolerance` must lie below `tolerance` divided by the factor by which a refinement
   *   lowers \f$\eta_K\f$ (4 for `"curvature"`), or a coarsened cell is refined again at once.
   * - `"fraction"`: with \f$\eta = \sum_K \eta_K\f$, the cells with the largest \f$\eta_K\f$ whose sum
   *   reaches `refine_percent` \f$\cdot\,\eta\f$ are refined, and the cells with the smallest \f$\eta_K\f$ whose sum
   *   stays below `coarsen_percent` \f$\cdot\,\eta\f$ are coarsened (deal.II's
   *   GridRefinement::refine_and_coarsen_fixed_fraction). This changes the mesh at every adaptation, also where the
   *   solution is resolved.
   *
   * Then
   * - no cell is refined beyond `level`, and no cell is coarsened below the level of the initial mesh;
   * - no cell is coarsened within `coarsen_buffer` layers of neighbours of a cell flagged for refinement, nor if it
   *   was created by one of the last `coarsen_age` adaptations that changed the mesh. Coarsening a region that is
   *   refined again soon after is expensive: in fRG flows, e.g., the flat region of a potential has \f$\eta_K
   *   \approx 0\f$, while its edge moves through it. With `coarsen_age` = 10, only regions that no recent change
   *   needed are coarsened;
   * - if no cell remains flagged for refinement (`"fraction"`), or none for any change (`"tolerance"`), nothing is
   *   done.
   *
   * The hanging-node constraints are applied to the solution before it is interpolated onto the new mesh, and again
   * afterwards. For an assembler that provides `prolongation_slopes` (FV), the children of a refined cell get the
   * cell average plus its slope times the offset of their centre, instead of the bare average. Discretization and
   * assembler are reinitialized.
   *
   * ### Parameters
   * | Key                                           | Type   | Default | Meaning |
   * |-----------------------------------------------|--------|---------|---------|
   * | `/discretization/adaptivity/start_adapt_at`   | double | 0       | RG time of the first adaptation |
   * | `/discretization/adaptivity/adapt_dt`         | double | 0.2     | minimum RG time between two adaptations |
   * | `/discretization/adaptivity/level`            | uint   | 0       | maximum refinement level of a cell (absolute, counted from the coarse grid); 0 disables adaptivity |
   * | `/discretization/adaptivity/indicator`        | string | "auto"  | "auto", "model" or "curvature", see above |
   * | `/discretization/adaptivity/marking`          | string | "tolerance" | "tolerance" or "fraction", see above |
   * | `/discretization/adaptivity/refine_percent`   | double | 0.1     | "fraction": fraction of the total indicator to refine |
   * | `/discretization/adaptivity/coarsen_percent`  | double | 0.05    | "fraction": fraction of the total indicator to coarsen |
   * | `/discretization/adaptivity/tolerance`        | double | 3e-4    | "tolerance": refine cells with a larger indicator |
   * | `/discretization/adaptivity/predict`          | bool   | true    | extrapolate the indicator to the next adaptation, see above |
   * | `/discretization/adaptivity/coarsen_tolerance`| double | tolerance / 8 | "tolerance": coarsen cells with a smaller indicator |
   * | `/discretization/adaptivity/coarsen_buffer`   | uint   | 2       | cell layers around refined cells that are not coarsened |
   * | `/discretization/adaptivity/coarsen_age`      | uint   | 10      | cells created by the last that many mesh changes are not coarsened |
   *
   * Since `level` is absolute, a mesh created with `/discretization/grid/refine` \f$= r\f$ is only refined further if
   * `level` \f$> r\f$.
   *
   * @tparam Assembler The assembler; must provide get_discretization(), refinement_indicator(), reinit(),
   * reinit_vector() and reinit_solution_view(); optionally cell_curvatures() (FV) and prolongation_slopes().
   */
  // clang-format on
  template <typename Assembler>
  class HAdaptivity : public AbstractAdaptor<typename Assembler::Discretization::VectorType>
  {
    using Discretization = typename Assembler::Discretization;
    using VectorType = typename Discretization::VectorType;
    static constexpr uint dim = Discretization::dim;

    static consteval bool assembler_supports_adaptation()
    {
      if constexpr (requires { Assembler::supports_mesh_adaptation; })
        return Assembler::supports_mesh_adaptation;
      else
        return true;
    }
    static_assert(assembler_supports_adaptation(),
                  "This assembler does not support mesh adaptation in this dimension; use NoAdaptivity.");

  public:
    /// Read the parameters (see the class description) and record the level of the initial mesh.
    HAdaptivity(Assembler &assembler, const ConfigTree &config)
        : assembler(assembler), discretization(assembler.get_discretization())
    {
      adapt_t = config.get_double("/discretization/adaptivity/start_adapt_at", 0.0);
      adapt_dt = config.get_double("/discretization/adaptivity/adapt_dt", 0.2);
      adapt_level = config.get_uint("/discretization/adaptivity/level", 0);
      adapt_upper = config.get_double("/discretization/adaptivity/refine_percent", 1e-1);
      adapt_lower = config.get_double("/discretization/adaptivity/coarsen_percent", 5e-2);
      const auto choice = [&](const std::string &key, const std::string &fallback,
                              const std::vector<std::string> &options) {
        const auto value = config.get_string("/discretization/adaptivity/" + key, fallback);
        if (std::find(options.begin(), options.end(), value) == options.end())
          throw std::invalid_argument("HAdaptivity: /discretization/adaptivity/" + key + " must be one of \"" +
                                      options[0] + "\", \"" + options[1] +
                                      (options.size() > 2 ? "\", \"" + options[2] : "") + "\", not \"" + value + "\".");
        return value;
      };
      const auto indicator = choice("indicator", "auto", {"auto", "model", "curvature"});
      curvature_indicator =
          indicator == "curvature" ||
          (indicator == "auto" && !internal::defines_refinement_indicator<typename Discretization::Model>());
      tolerance_marking = choice("marking", "tolerance", {"tolerance", "fraction"}) == "tolerance";
      tolerance = config.get_double("/discretization/adaptivity/tolerance", 3e-4);
      coarsen_tolerance = config.get_double("/discretization/adaptivity/coarsen_tolerance", tolerance / 8.);
      predict = config.get_bool("/discretization/adaptivity/predict", true);
      coarsen_buffer = config.get_uint("/discretization/adaptivity/coarsen_buffer", 2);
      coarsen_age = config.get_uint("/discretization/adaptivity/coarsen_age", 10);
      min_level = std::numeric_limits<uint>::max();
      for (const auto &cell : discretization.get_triangulation().active_cell_iterators())
        min_level = std::min(min_level, uint(cell->level()));
    }

    /// @copydoc AbstractAdaptor::last_adaptation_time
    virtual double last_adaptation_time() const override { return last_adapt; }
    /// @copydoc AbstractAdaptor::set_last_adaptation_time
    virtual void set_last_adaptation_time(const double t) override { last_adapt = t; }

    /// Stores the number of mesh changes, the creation of each cell (coarsen_age) and the indicators of the last
    /// adaptation (predict).
    virtual void save_state(ModelState &state) const override
    {
      state.set("n_adaptations", n_adaptations);
      state.set("created_at", encode(created_at));
      state.set("previous_indicator", encode(previous_indicator));
    }

    /// @copydoc AbstractAdaptor::load_state
    virtual void load_state(const ModelState &state) override
    {
      if (!state.contains("n_adaptations")) return;
      n_adaptations = state.get<uint>("n_adaptations");
      created_at.clear();
      for (const auto &[id, value] : decode(state.get_vector("created_at")))
        created_at[id] = static_cast<uint>(value);
      previous_indicator = decode(state.get_vector("previous_indicator"));
    }

    virtual ~HAdaptivity() = default;

    /**
     * @brief Adapt if `level` > 0, @p t ≥ `start_adapt_at`, and at least `adapt_dt` has passed since the last
     * adaptation; transfer @p sol to the new mesh.
     *
     * @return true if the mesh changed. A skipped adaptation (nothing to refine) does not reset the schedule.
     */
    virtual bool operator()(const double t, VectorType &sol) override { return adapt_at(t, sol, {}); }

    /// As operator()(t, sol), also transferring @p also_transfer to the new mesh.
    virtual bool adapt_at(const double t, VectorType &sol, const std::vector<VectorType *> &also_transfer) override
    {
      if (adapt_level > 0 && t >= adapt_t - 1e-12 * adapt_dt && (t - last_adapt + 1e-12 * adapt_dt) >= adapt_dt) {
        if (!adapt(sol, also_transfer)) return false;
        last_adapt = t;
        return true;
      }
      return false;
    }

    /**
     * @brief Adapt now, regardless of the schedule, and transfer @p solution to the new mesh.
     *
     * @return true if the mesh changed.
     */
    virtual bool adapt(VectorType &solution) override { return adapt(solution, {}); }

    /// As adapt(solution), also transferring @p also_transfer to the new mesh.
    bool adapt(VectorType &solution, const std::vector<VectorType *> &also_transfer)
    {
      auto &triangulation = discretization.get_triangulation();
      auto &dof_handler = discretization.get_dof_handler(0);
      auto &constraints = discretization.get_constraints(0);

      // Both readers below -- refinement_indicator() here and SolutionTransfer further down -- take
      // arbitrary global dof indices, which the vector we are handed cannot answer: it holds only
      // this rank's rows, and deal.II's non-ghosted read path is `ptr[index - local_begin]` with no
      // bounds check outside debug builds. A partition-boundary read therefore does not fail, it
      // returns whatever is next in memory. Read both through a fully-replicated view instead.
      SolutionView<VectorType> view;
      assembler.reinit_solution_view(view);
      view.refresh(solution);

      Vector<double> indicator(triangulation.n_active_cells());
      indicator = 0;
      if (curvature_indicator)
        compute_curvature_indicator(indicator, view.get());
      else
        assembler.refinement_indicator(indicator, view.get());

      if constexpr (Discretization::Mesh::is_parallel) {
        // Each rank fills only the entries of the cells it owns, so on a partitioned mesh every
        // other entry is still zero. Refining on
        // that unreduced vector is not a crash -- it silently refines a different set of cells on
        // every rank, which then diverge. Sum first; the mesh is replicated, so after this every
        // rank holds the identical, complete indicator.
        MPI::sum_reduce(triangulation.get_mpi_communicator(), indicator.data(), static_cast<int>(indicator.size()));
      }
      if (predict) extrapolate_indicator(indicator);

      if (tolerance_marking) {
        GridRefinement::refine(triangulation, indicator, tolerance);
        GridRefinement::coarsen(triangulation, indicator, coarsen_tolerance);
      } else
        GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, indicator, adapt_upper, adapt_lower);

      // Never refine beyond adapt_level, never coarsen below the mesh we started from.
      for (const auto &cell : triangulation.active_cell_iterators()) {
        if (uint(cell->level()) >= adapt_level) cell->clear_refine_flag();
        if (uint(cell->level()) <= min_level) cell->clear_coarsen_flag();
      }
      protect_from_coarsening(triangulation);
      // Drops coarsening flags that cannot be honoured (e.g. not all siblings flagged), so that they do not count.
      if (tolerance_marking) triangulation.prepare_coarsening_and_refinement();

      bool any_change = false;
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->refine_flag_set() || (tolerance_marking && cell->coarsen_flag_set())) {
          any_change = true;
          break;
        }
      // Agreed: this early return decides whether execute_coarsening_and_refinement() -- a
      // collective on a parallel triangulation -- is reached at all. The reduced indicator above
      // should already make every rank decide identically; this makes a hang impossible rather
      // than unlikely.
      if (!MPI::any_of(triangulation.get_mpi_communicator(), any_change)) return false;

      SolutionTransfer<dim, VectorType> solution_trans(dof_handler);

      // A copy, not the view itself: reinit_vector() below resizes `solution`, and the serial view
      // aliases its source rather than snapshotting it. SolutionTransfer also has to keep reading
      // this right through execute_coarsening_and_refinement(), which is where the packing happens.
      std::vector<VectorType> previous{view.get()};
      for (const auto *v : also_transfer) {
        view.refresh(*v);
        previous.push_back(view.get());
      }
      // The timesteppers leave the constrained entries (hanging nodes) stale; interpolating them onto the new mesh
      // would carry that into the unconstrained dofs. For a time derivative this applies the inhomogeneities too;
      // harmless, as the IDA steppers only use a transferred derivative as the starting guess of their restart.
      for (auto &v : previous)
        constraints.distribute(v);
      triangulation.prepare_coarsening_and_refinement();
      const auto slopes = refined_cell_slopes(previous);
      solution_trans.prepare_for_coarsening_and_refinement(previous);
      std::set<CellId> active_before;
      for (const auto &cell : triangulation.active_cell_iterators())
        active_before.insert(cell->id());
      triangulation.execute_coarsening_and_refinement();
      ++n_adaptations;
      std::map<CellId, uint> next_created_at;
      for (const auto &cell : triangulation.active_cell_iterators()) {
        const auto id = cell->id();
        if (!active_before.contains(id))
          next_created_at[id] = n_adaptations;
        else if (const auto found = created_at.find(id); found != created_at.end())
          next_created_at.insert(*found);
      }
      created_at = std::move(next_created_at);

      discretization.reinit();
      assembler.reinit();
      std::vector<VectorType *> targets{&solution};
      targets.insert(targets.end(), also_transfer.begin(), also_transfer.end());
      for (auto *v : targets)
        assembler.reinit_vector(*v);

      // The write side needs no distributed handling. SolutionTransfer is built with
      // average_values = false, so interpolate() ends in compress(VectorOperation::insert), and on
      // a replicated mesh every rank unpacks every cell to bit-identical values -- identical
      // replica, identical mesh, deterministic interpolation -- so it does not matter which rank's
      // insert the owner keeps. (With average_values = true this would instead be a sum over
      // n_ranks redundant contributions, correct only because the valence is inflated by the same
      // factor, and no longer exact.)
      solution_trans.interpolate(targets);
      add_prolongation_slopes(slopes, targets);
      for (auto *v : targets)
        constraints.distribute(*v);

      return true;
    }

  protected:
    /// A map from cells to numbers as a flat list: per entry the binary CellId, then the value.
    template <typename T> static std::vector<double> encode(const std::map<CellId, T> &map)
    {
      std::vector<double> list;
      for (const auto &[id, value] : map) {
        for (const auto word : id.template to_binary<dim>())
          list.push_back(word);
        list.push_back(static_cast<double>(value));
      }
      return list;
    }
    static std::map<CellId, double> decode(const std::vector<double> &list)
    {
      using Binary = CellId::binary_type;
      constexpr std::size_t n = std::tuple_size_v<Binary> + 1;
      if (list.size() % n != 0) throw std::runtime_error("HAdaptivity::load_state: corrupt cell list.");
      std::map<CellId, double> map;
      for (std::size_t i = 0; i < list.size(); i += n) {
        Binary binary;
        for (std::size_t j = 0; j + 1 < n; ++j)
          binary[j] = static_cast<typename Binary::value_type>(list[i + j]);
        map[CellId(binary)] = list[i + n - 1];
      }
      return map;
    }

    /// The `"curvature"` indicator of the class documentation, for the cells owned by this rank.
    void compute_curvature_indicator(Vector<double> &indicator, const VectorType &u) const
    {
      const auto &dof_handler = discretization.get_dof_handler(0);
      const uint n_components = dof_handler.get_fe().n_components();
      Vector<float> curvature(indicator.size());
      for (uint c = 0; c < n_components; ++c) {
        ComponentMask mask(n_components, false);
        mask.set(c, true);
        double lowest = std::numeric_limits<double>::max(), highest = std::numeric_limits<double>::lowest();
        for (const auto i : DoFTools::extract_dofs(dof_handler, mask)) {
          lowest = std::min(lowest, double(u[i]));
          highest = std::max(highest, double(u[i]));
        }
        // extract_dofs gives only this rank's dofs.
        const auto comm = discretization.get_triangulation().get_mpi_communicator();
        const double range = Utilities::MPI::max(highest, comm) - Utilities::MPI::min(lowest, comm);
        if (!(range > 0.)) continue; // a constant component has no features
        if constexpr (requires { assembler.cell_curvatures(u, c); }) {
          const auto values = assembler.cell_curvatures(u, c);
          std::copy(values.begin(), values.end(), curvature.begin());
        } else
          DerivativeApproximation::approximate_second_derivative(dof_handler, u, curvature, c);
        for (const auto &cell : discretization.get_triangulation().active_cell_iterators())
          if (cell->is_locally_owned()) {
            const auto k = cell->active_cell_index();
            indicator[k] = std::max(indicator[k], std::pow(cell->diameter(), 2) * curvature[k] / range);
          }
      }
    }

    /**
     * @brief predict: add to each cell's indicator its growth since the previous adaptation, i.e. extrapolate it to
     * the next one. A cell refined since then compares with a quarter of its parent's value, one coarsened with four
     * times the largest of its children's (the indicator scales like \f$h^2\f$).
     */
    void extrapolate_indicator(Vector<double> &indicator)
    {
      std::map<CellId, double> current;
      for (const auto &cell : discretization.get_triangulation().active_cell_iterators()) {
        const auto k = cell->active_cell_index();
        const auto id = cell->id();
        current[id] = indicator[k];
        if (previous_indicator.empty()) continue;
        double before = -1.;
        if (const auto found = previous_indicator.find(id); found != previous_indicator.end())
          before = found->second;
        else if (cell->level() > 0) {
          if (const auto parent = previous_indicator.find(cell->parent()->id()); parent != previous_indicator.end())
            before = parent->second / 4.;
        }
        if (before < 0.)
          for (const auto &[old_id, value] : previous_indicator)
            if (id.is_ancestor_of(old_id)) before = std::max(before, 4. * value);
        if (before >= 0.) indicator[k] += std::max(0., indicator[k] - before);
      }
      previous_indicator = std::move(current);
    }

    /// Centre of a cell flagged for refinement and the slopes of the vectors being transferred there.
    struct RefinedCellSlopes {
      Point<dim> center;
      std::vector<std::vector<Tensor<1, dim>>> slopes; // [vector][component]
    };

    /**
     * @brief For an assembler that provides `prolongation_slopes(v)` (the FV assembler): per cell flagged for
     * refinement, the slopes of each of @p vectors. Empty otherwise.
     */
    std::map<CellId, RefinedCellSlopes> refined_cell_slopes(const std::vector<VectorType> &vectors) const
    {
      std::map<CellId, RefinedCellSlopes> result;
      if constexpr (requires { assembler.prolongation_slopes(vectors[0]); }) {
        std::vector<decltype(assembler.prolongation_slopes(vectors[0]))> per_vector;
        for (const auto &v : vectors)
          per_vector.push_back(assembler.prolongation_slopes(v));
        for (const auto &cell : discretization.get_triangulation().active_cell_iterators()) {
          if (!cell->refine_flag_set()) continue;
          auto &entry = result[cell->id()];
          entry.center = cell->center();
          for (const auto &slopes : per_vector) {
            const auto &cell_slopes = slopes[cell->active_cell_index()];
            entry.slopes.emplace_back(cell_slopes.begin(), cell_slopes.end());
          }
        }
      }
      return result;
    }

    /// Children of the cells in @p slopes: add slope * (child centre - parent centre) to their transferred value.
    void add_prolongation_slopes(const std::map<CellId, RefinedCellSlopes> &slopes,
                                 const std::vector<VectorType *> &targets) const
    {
      if (slopes.empty()) return;
      const auto &dof_handler = discretization.get_dof_handler(0);
      const auto &fe = dof_handler.get_fe();
      std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
      for (size_t k = 0; k < targets.size(); ++k) {
        auto &v = *targets[k];
        const auto owned = v.locally_owned_elements();
        for (const auto &cell : dof_handler.active_cell_iterators()) {
          if (cell->level() == 0) continue;
          const auto found = slopes.find(cell->parent()->id());
          if (found == slopes.end()) continue;
          cell->get_dof_indices(dofs);
          const auto shift = cell->center() - found->second.center;
          for (uint i = 0; i < dofs.size(); ++i)
            if (owned.is_element(dofs[i]))
              // Read and insert: compress(insert) below must not follow an add on distributed vectors.
              v(dofs[i]) = v(dofs[i]) + found->second.slopes[k][fe.system_to_component_index(i).first] * shift;
        }
        v.compress(VectorOperation::insert);
      }
    }

    /**
     * @brief Clear the coarsening flags within coarsen_buffer layers around cells flagged for refinement and on cells
     * created by one of the last coarsen_age adaptations.
     */
    template <typename Tria> void protect_from_coarsening(Tria &triangulation) const
    {
      using Iterator = typename Tria::active_cell_iterator;
      std::vector<Iterator> frontier;
      std::set<CellId> seen;
      for (const auto &cell : triangulation.active_cell_iterators()) {
        if (cell->refine_flag_set()) {
          frontier.push_back(cell);
          seen.insert(cell->id());
        }
        if (const auto found = created_at.find(cell->id());
            found != created_at.end() && n_adaptations - found->second < coarsen_age)
          cell->clear_coarsen_flag();
      }
      for (uint layer = 0; layer < coarsen_buffer && !frontier.empty(); ++layer) {
        std::vector<Iterator> next;
        for (const auto &cell : frontier) {
          std::vector<Iterator> neighbors;
          GridTools::get_active_neighbors<Tria>(cell, neighbors);
          for (const auto &neighbor : neighbors)
            if (seen.insert(neighbor->id()).second) {
              neighbor->clear_coarsen_flag();
              next.push_back(neighbor);
            }
        }
        frontier = std::move(next);
      }
    }

    Assembler &assembler;
    Discretization &discretization;

    /// -inf until the first adaptation, so that the first one happens as soon as start_adapt_at is reached.
    double last_adapt = -std::numeric_limits<double>::infinity();
    double adapt_t, adapt_dt, adapt_upper, adapt_lower;
    uint adapt_level;
    /// Level of the coarsest active cell at construction.
    uint min_level;
    uint coarsen_buffer, coarsen_age;
    bool curvature_indicator, tolerance_marking;
    double tolerance, coarsen_tolerance;
    /// Number of adaptations so far, and for each active cell the adaptation that created it (if any did).
    uint n_adaptations = 0;
    std::map<CellId, uint> created_at;
    /// predict: the indicator of each active cell at the previous adaptation.
    bool predict;
    std::map<CellId, double> previous_indicator;
  };
} // namespace DiFfRG