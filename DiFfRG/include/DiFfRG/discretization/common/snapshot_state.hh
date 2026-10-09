#pragma once

// external libraries
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector_operation.h>

// standard library
#include <concepts>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Named, history-dependent model state carried by a flow snapshot.
   *
   * Most models are a pure function of the solution and the RG time, and a snapshot of the
   * discrete state is then enough to continue a flow. Some models are not: they latch a value the
   * first time something happens (an EoM lock, say) and keep acting on it. A restart that does not
   * know about such a latch silently continues a *different* flow. Models with that kind of state
   * implement
   *
   * @code
   * void save_state(DiFfRG::ModelState &state) const;
   * void load_state(const DiFfRG::ModelState &state);
   * @endcode
   *
   * and the assemblers call them when a snapshot is written or restored. Both are optional and
   * detected at compile time (HasSaveState, HasLoadState); there is deliberately no default in
   * def::AbstractModel, because the absence of the method is exactly what is detected.
   *
   * Values are stored as doubles. Integers are exact up to 2^53 and booleans map to 0/1.
   */
  class ModelState
  {
  public:
    template <typename T>
      requires std::is_arithmetic_v<T>
    void set(const std::string &name, const T value)
    {
      set(name, std::vector<double>{static_cast<double>(value)});
    }

    void set(const std::string &name, std::vector<double> values)
    {
      if (name.empty() || name.find('/') != std::string::npos)
        throw std::invalid_argument("ModelState::set: the name '" + name +
                                    "' is not valid; names must be non-empty and must not contain '/'.");
      entries[name] = std::move(values);
    }

    template <typename T>
      requires std::is_arithmetic_v<T>
    T get(const std::string &name) const
    {
      const auto &values = get_vector(name);
      if (values.size() != 1)
        throw std::runtime_error("ModelState::get: the entry '" + name + "' holds " + std::to_string(values.size()) +
                                 " values, not a single one.");
      if constexpr (std::is_same_v<T, bool>)
        return values[0] != 0.;
      else
        return static_cast<T>(values[0]);
    }

    const std::vector<double> &get_vector(const std::string &name) const
    {
      const auto it = entries.find(name);
      if (it == entries.end()) throw std::runtime_error("ModelState::get: no entry named '" + name + "'.");
      return it->second;
    }

    bool contains(const std::string &name) const { return entries.contains(name); }
    bool empty() const { return entries.empty(); }
    const std::map<std::string, std::vector<double>> &data() const { return entries; }

  private:
    std::map<std::string, std::vector<double>> entries;
  };

  template <typename Model>
  concept HasSaveState = requires(const Model &model, ModelState &state) { model.save_state(state); };

  template <typename Model>
  concept HasLoadState = requires(Model &model, const ModelState &state) { model.load_state(state); };

  /**
   * @brief The spatial part of a flow snapshot, in a layout that does not depend on dof numbering.
   *
   * Every active cell is recorded by its CellId together with its local dof values, in the order
   * of the element's local dofs. That is independent of how the dofs happen to be numbered, which
   * matters in three places: a parallel::shared::Triangulation renumbers dofs by subdomain, so the
   * global numbering depends on the rank count; LDG renumbers component-wise; and an adapted mesh
   * has to be rebuilt before any numbering exists at all. Dofs shared between cells (CG) are stored
   * once per cell with identical values.
   */
  struct SnapshotSpatialState {
    std::string fe_name;
    unsigned int dofs_per_cell = 0;
    unsigned int n_coarse_cells = 0;
    unsigned long long n_dofs = 0;
    /// CellId::to_string() of every active cell.
    std::vector<std::string> active_cells;
    /// active_cells.size() * dofs_per_cell values, cell by cell.
    std::vector<double> values;

    bool empty() const { return active_cells.empty(); }
  };

  namespace internal
  {
    template <typename Model> void save_model_state(const Model &model, ModelState &state)
    {
      if constexpr (HasSaveState<Model>) model.save_state(state);
    }

    /// @return whether the model consumed the state, i.e. whether it implements load_state.
    template <typename Model> bool load_model_state(Model &model, const ModelState &state)
    {
      if constexpr (HasLoadState<Model>) {
        model.load_state(state);
        return true;
      } else
        return false;
    }

    /**
     * @brief Record the active cells and their local dof values.
     *
     * @param replica A vector from which every global dof index can be read, i.e. a serial vector
     * or a refreshed SolutionView. Reads only, so this is safe inside a rank-0-only region.
     */
    template <int dim, typename VectorType>
    SnapshotSpatialState capture_cellwise_state(const dealii::DoFHandler<dim> &dof_handler, const VectorType &replica)
    {
      SnapshotSpatialState state;
      const auto &fe = dof_handler.get_fe();
      const auto &triangulation = dof_handler.get_triangulation();
      state.fe_name = fe.get_name();
      state.dofs_per_cell = fe.n_dofs_per_cell();
      state.n_coarse_cells = triangulation.n_cells(0);
      state.n_dofs = dof_handler.n_dofs();
      state.active_cells.reserve(triangulation.n_active_cells());
      state.values.reserve(triangulation.n_active_cells() * state.dofs_per_cell);

      std::vector<dealii::types::global_dof_index> dof_indices(state.dofs_per_cell);
      for (const auto &cell : dof_handler.active_cell_iterators()) {
        state.active_cells.push_back(cell->id().to_string());
        cell->get_dof_indices(dof_indices);
        for (const auto index : dof_indices)
          state.values.push_back(replica(index));
      }
      return state;
    }

    /**
     * @brief Refine and coarsen @p triangulation until its active cells are exactly @p active_cells.
     *
     * Every active cell of the current mesh is either one of the targets, a proper ancestor of one
     * (so it is refined), or a descendant of one (so it is coarsened). Refinement and coarsening
     * proceed one level per pass, which keeps every intermediate mesh at least as balanced as the
     * target, so the triangulation's own smoothing never has to add or drop flags.
     *
     * Deterministic in its input, so on a replicated (parallel::shared) mesh every rank performs the
     * identical sequence of collective refinements.
     *
     * @return whether the mesh was changed.
     * @throws std::runtime_error if the coarse mesh differs or the target mesh cannot be reached.
     */
    template <int dim>
    bool restore_active_cells(dealii::Triangulation<dim> &triangulation, const unsigned int n_coarse_cells,
                              const std::vector<std::string> &active_cells)
    {
      if (triangulation.n_cells(0) != n_coarse_cells)
        throw std::runtime_error("Snapshot restore: the snapshot was written on a coarse mesh with " +
                                 std::to_string(n_coarse_cells) + " cells, but the current coarse mesh has " +
                                 std::to_string(triangulation.n_cells(0)) +
                                 ". The grid configuration must match the snapshot.");

      std::set<dealii::CellId> targets;
      for (const auto &id : active_cells)
        targets.emplace(id);
      if (targets.size() != active_cells.size())
        throw std::runtime_error("Snapshot restore: the snapshot lists an active cell more than once.");

      // Every proper ancestor of a target cell has to end up refined.
      std::set<dealii::CellId> ancestors;
      for (const auto &target : targets) {
        const auto &child_indices = target.get_child_indices();
        for (std::size_t level = 0; level < child_indices.size(); ++level)
          ancestors.emplace(target.get_coarse_cell_id(),
                            std::vector<std::uint8_t>(child_indices.begin(), child_indices.begin() + level));
      }

      const auto matches = [&]() {
        if (triangulation.n_active_cells() != targets.size()) return false;
        for (const auto &cell : triangulation.active_cell_iterators())
          if (!targets.contains(cell->id())) return false;
        return true;
      };
      if (matches()) return false;

      // Each pass moves every mismatched cell one level towards its target. The level difference
      // is bounded by the deeper of the two meshes, so this many passes always suffice.
      std::size_t max_target_level = 0;
      for (const auto &target : targets)
        max_target_level = std::max<std::size_t>(max_target_level, target.get_child_indices().size());
      const std::size_t max_passes = std::max<std::size_t>(max_target_level, triangulation.n_levels()) + 2;

      for (std::size_t pass = 0; pass < max_passes; ++pass) {
        bool flagged = false;
        for (const auto &cell : triangulation.active_cell_iterators()) {
          const auto id = cell->id();
          if (targets.contains(id)) continue;
          if (ancestors.contains(id))
            cell->set_refine_flag();
          else
            cell->set_coarsen_flag();
          flagged = true;
        }
        if (!flagged) break;
        triangulation.execute_coarsening_and_refinement();
        if (matches()) return true;
      }
      throw std::runtime_error("Snapshot restore: could not rebuild the snapshot's mesh (" +
                               std::to_string(targets.size()) + " active cells) from the current mesh (" +
                               std::to_string(triangulation.n_active_cells()) + " active cells).");
    }

    /**
     * @brief Write the snapshot's cell-wise values into @p vector, which must already have the
     * layout of @p dof_handler. Each rank writes only the dofs it owns.
     */
    template <int dim, typename VectorType>
    void restore_cellwise_values(const SnapshotSpatialState &state, const dealii::DoFHandler<dim> &dof_handler,
                                 const dealii::IndexSet &locally_owned_dofs, VectorType &vector)
    {
      const auto &fe = dof_handler.get_fe();
      if (fe.get_name() != state.fe_name || fe.n_dofs_per_cell() != state.dofs_per_cell)
        throw std::runtime_error("Snapshot restore: the snapshot was written with the finite element '" +
                                 state.fe_name + "', but the current discretization uses '" + fe.get_name() + "'.");
      const auto &triangulation = dof_handler.get_triangulation();
      if (triangulation.n_active_cells() != state.active_cells.size() || dof_handler.n_dofs() != state.n_dofs ||
          state.values.size() != state.active_cells.size() * std::size_t(state.dofs_per_cell))
        throw std::runtime_error("Snapshot restore: the snapshot's mesh or dof count does not match the "
                                 "current discretization.");

      std::vector<dealii::types::global_dof_index> dof_indices(state.dofs_per_cell);
      for (std::size_t c = 0; c < state.active_cells.size(); ++c) {
        const auto tria_cell = triangulation.create_cell_iterator(dealii::CellId(state.active_cells[c]));
        const typename dealii::DoFHandler<dim>::active_cell_iterator cell(&triangulation, tria_cell->level(),
                                                                          tria_cell->index(), &dof_handler);
        cell->get_dof_indices(dof_indices);
        for (unsigned int i = 0; i < state.dofs_per_cell; ++i)
          if (locally_owned_dofs.is_element(dof_indices[i]))
            vector(dof_indices[i]) = state.values[c * state.dofs_per_cell + i];
      }
      // insert, not add: CG dofs shared between cells are written once per cell, all with the same
      // value. A no-op for the serial vector types.
      vector.compress(dealii::VectorOperation::insert);
    }

    /**
     * @brief The restore sequence shared by all mesh-based assemblers.
     *
     * Mirrors HAdaptivity::adapt: change the mesh, then discretization.reinit() and
     * assembler.reinit(), then size the vector. No constraints are distributed: every dof,
     * constrained or not, is restored from the snapshot, which already satisfied them.
     */
    template <typename Discretization, typename Assembler, typename VectorType>
    void restore_spatial_state(const SnapshotSpatialState &state, Discretization &discretization, Assembler &assembler,
                               VectorType &spatial)
    {
      if (restore_active_cells(discretization.get_triangulation(), state.n_coarse_cells, state.active_cells)) {
        discretization.reinit();
        assembler.reinit();
      }
      assembler.reinit_vector(spatial);
      restore_cellwise_values(state, discretization.get_dof_handler(), discretization.get_locally_owned_dofs(),
                              spatial);
    }
  } // namespace internal
} // namespace DiFfRG
