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
   * @brief Named model state stored in a flow snapshot, for models whose behaviour depends on their history.
   *
   * A snapshot stores the solution and the RG time. A model that keeps additional state (e.g. a flag set once some
   * condition was met) must store it too, or a restarted flow differs from the uninterrupted one. Such a model
   * implements
   *
   * @code
   * void save_state(DiFfRG::ModelState &state) const { state.set("locked", locked); }
   * void load_state(const DiFfRG::ModelState &state) { locked = state.get<bool>("locked"); }
   * @endcode
   *
   * Both are optional and detected at compile time (HasSaveState, HasLoadState); def::AbstractModel has no default.
   *
   * Values are stored as doubles: integers are exact up to \f$2^{53}\f$, booleans are 0/1.
   */
  class ModelState
  {
  public:
    /// Store a single number under @p name (non-empty, without '/'); overwrites an existing entry.
    template <typename T>
      requires std::is_arithmetic_v<T>
    void set(const std::string &name, const T value)
    {
      set(name, std::vector<double>{static_cast<double>(value)});
    }

    /// Store a list of numbers under @p name (non-empty, without '/'); overwrites an existing entry.
    void set(const std::string &name, std::vector<double> values)
    {
      if (name.empty() || name.find('/') != std::string::npos)
        throw std::invalid_argument("ModelState::set: the name '" + name +
                                    "' is not valid; names must be non-empty and must not contain '/'.");
      entries[name] = std::move(values);
    }

    /// The single number stored under @p name; throws if it is missing or holds a list of another length.
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

    /// The list stored under @p name; throws if it is missing.
    const std::vector<double> &get_vector(const std::string &name) const
    {
      const auto it = entries.find(name);
      if (it == entries.end()) throw std::runtime_error("ModelState::get: no entry named '" + name + "'.");
      return it->second;
    }

    /// Whether an entry @p name exists.
    bool contains(const std::string &name) const { return entries.contains(name); }
    /// Whether nothing is stored.
    bool empty() const { return entries.empty(); }
    /// All entries.
    const std::map<std::string, std::vector<double>> &data() const { return entries; }

  private:
    std::map<std::string, std::vector<double>> entries;
  };

  /// The model implements `save_state(ModelState &) const`.
  template <typename Model>
  concept HasSaveState = requires(const Model &model, ModelState &state) { model.save_state(state); };

  /// The model implements `load_state(const ModelState &)`.
  template <typename Model>
  concept HasLoadState = requires(Model &model, const ModelState &state) { model.load_state(state); };

  /**
   * @brief Spatial part of a flow snapshot: every active cell (by CellId) with its local dof values, independent of
   * the global dof numbering and the number of MPI ranks.
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
    /// Call model.save_state if it exists.
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

    /// Record the active cells and their local dof values from a vector in which every global dof is readable.
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

    /// Refine and coarsen @p triangulation until its active cells are @p active_cells. @return whether it changed.
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

    /// Write the snapshot's values into @p vector (already laid out for @p dof_handler); each rank writes its own dofs.
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

    /// Rebuild the mesh, reinitialize discretization and assembler, and fill @p spatial from the snapshot.
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
