#pragma once

// DiFfRG
#include <DiFfRG/physics/integration/map_scheduler.hh>

// external libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/tbb.h>

// standard library
#include <array>
#include <bit>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief The locally owned cells of a batched assembler, colored for its scatter.
     *
     * In the scatter every cell adds into the rows of its own dofs and, through the constraints, into
     * those of their constraint masters. No two cells of a color share such a row, so the cells of one color
     * can add into a global matrix or vector concurrently. Columns do not matter: concurrent additions to
     * different rows never touch the same entry.
     */
    template <int dim> class ColoredCells
    {
    public:
      using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

      template <typename NumberType>
      void reinit(const dealii::DoFHandler<dim> &dof_handler, const dealii::AffineConstraints<NumberType> &constraints)
      {
        cells.clear();
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned()) cells.push_back(cell);

        // Greedy coloring, one bit per color and row.
        std::vector<std::uint64_t> row_colors(dof_handler.n_dofs(), 0);
        std::vector<dealii::types::global_dof_index> rows;
        colors.clear();
        constrained.assign(cells.size(), false);
        for (size_t k = 0; k < cells.size(); ++k) {
          rows.resize(cells[k]->get_fe().n_dofs_per_cell());
          cells[k]->get_dof_indices(rows);
          const size_t n_own = rows.size();
          for (size_t i = 0; i < n_own; ++i)
            if (constraints.is_constrained(rows[i])) {
              constrained[k] = true;
              if (const auto *entries = constraints.get_constraint_entries(rows[i]))
                for (const auto &entry : *entries)
                  rows.push_back(entry.first);
            }
          std::uint64_t taken = 0;
          for (const auto row : rows)
            taken |= row_colors[row];
          if (~taken == 0) throw std::runtime_error("ColoredCells: more than 64 cell colors needed.");
          const uint color = std::countr_one(taken);
          for (const auto row : rows)
            row_colors[row] |= std::uint64_t(1) << color;
          if (color >= colors.size()) colors.resize(color + 1);
          colors[color].push_back(k);
        }
      }

      size_t size() const { return cells.size(); }
      const CellIterator &operator[](const size_t k) const { return cells[k]; }
      const std::vector<CellIterator> &all() const { return cells; }
      /// Whether a dof of cell k is constrained, i.e. its insertion has to go through the constraints.
      bool is_constrained(const size_t k) const { return constrained[k]; }

      /**
       * @brief assemble(k, scratch, local) for every cell k, and insert(local) the result into @p global.
       *
       * deal.II's serial (block) vectors and matrices are written concurrently, one parallel loop per color. Other
       * types (PETSc) are assembled in parallel into @p buffer and inserted serially in the same order.
       * Either way the result does not depend on the thread count. Ends with global.compress(add).
       *
       * @param local the member of Scratch that holds one cell's result
       */
      template <typename Global, typename Scratch, typename Local, typename Assemble, typename Insert>
      void scatter(Global &global, tbb::enumerable_thread_specific<Scratch> &scratch, Local Scratch::*local,
                   std::vector<Local> &buffer, const Assemble &assemble, const Insert &insert) const
      {
        using NT = typename Global::value_type;
        constexpr bool concurrent =
            std::is_same_v<Global, dealii::Vector<NT>> || std::is_same_v<Global, dealii::SparseMatrix<NT>> ||
            std::is_same_v<Global, dealii::BlockVector<NT>> || std::is_same_v<Global, dealii::BlockSparseMatrix<NT>>;
        // map() is collective and each rank visits only its own cells; see NoMapsHere.
        const NoMapsHere no_maps_during_assembly;
        if constexpr (concurrent) {
          for (const auto &color : colors)
            tbb::parallel_for(tbb::blocked_range<size_t>(0, color.size()), [&](const tbb::blocked_range<size_t> &r) {
              auto &s = scratch.local();
              for (size_t i = r.begin(); i != r.end(); ++i) {
                assemble(color[i], s, s.*local);
                insert(s.*local);
              }
            });
        } else {
          buffer.resize(cells.size());
          tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const tbb::blocked_range<size_t> &r) {
            auto &s = scratch.local();
            for (size_t k = r.begin(); k != r.end(); ++k)
              assemble(k, s, buffer[k]);
          });
          for (const auto &color : colors)
            for (const size_t k : color)
              insert(buffer[k]);
        }
        global.compress(dealii::VectorOperation::add);
      }

    private:
      std::vector<CellIterator> cells;
      /// Cell indices k by color.
      std::vector<std::vector<size_t>> colors;
      std::vector<bool> constrained;
    };

    /**
     * @brief matrix(rows[i], cols[j]) += M(i, j), straight into the blocks of @p matrix.
     *
     * BlockSparseMatrix::add itself serializes all callers on one mutex (it shares a scratch buffer), which turns
     * a concurrent scatter into a contended serial one. Adding entry by entry into the blocks is safe for distinct
     * rows.
     */
    template <typename NT>
    void add_to_blocks(dealii::BlockSparseMatrix<NT> &matrix, const std::vector<dealii::types::global_dof_index> &rows,
                       const std::vector<dealii::types::global_dof_index> &cols, const dealii::FullMatrix<NT> &M)
    {
      const auto &row_blocks = matrix.get_row_indices();
      const auto &col_blocks = matrix.get_column_indices();
      for (unsigned int i = 0; i < rows.size(); ++i) {
        const auto [block_row, local_row] = row_blocks.global_to_local(rows[i]);
        for (unsigned int j = 0; j < cols.size(); ++j) {
          if (M(i, j) == NT(0)) continue;
          const auto [block_col, local_col] = col_blocks.global_to_local(cols[j]);
          matrix.block(block_row, block_col).add(local_row, local_col, M(i, j));
        }
      }
    }

    /// @p cell of another DoFHandler on the same triangulation, e.g. the same cell of another LDG level.
    template <int dim, typename Iterator>
    typename dealii::DoFHandler<dim>::active_cell_iterator on(const dealii::DoFHandler<dim> &dof_handler,
                                                              const Iterator &cell)
    {
      return {&dof_handler.get_triangulation(), cell->level(), cell->index(), &dof_handler};
    }

    /**
     * @brief The boundary faces and interior faces of the cells of a ColoredCells, for a batched assembler whose
     * scatter adds each face into the rows of its own cell.
     *
     * An interior face is listed once, seen from its finer cell (between cells of one level: from the smaller one),
     * with that cell pair's (sub)face numbers, and referenced by each of its cells that this rank owns. Faces to
     * ghost cells are listed on every rank that owns one of their cells: each rank adds only into its own rows, so
     * it needs every face of its cells.
     */
    template <int dim> struct FaceTopology {
      /// An interior face: cell[0] is the finer of the two cells (between cells of one level, the smaller one), the
      /// normal points out of it, and the numerical flux takes cell[0] as trace s.
      struct InteriorFace {
        std::array<typename dealii::DoFHandler<dim>::cell_iterator, 2> cell;
        std::array<uint, 2> face_no, subface_no;
      };
      /// Interior face f seen from the owned cell on its side `side`.
      struct FaceRef {
        size_t face;
        uint side;
      };

      /// Boundary faces in cell order; those of cell k are [boundary_begin[k], boundary_begin[k + 1]).
      std::vector<std::pair<typename dealii::DoFHandler<dim>::active_cell_iterator, uint>> boundary_faces;
      std::vector<size_t> boundary_begin;
      /// Interior faces; those of owned cell k are face_refs[face_ref_begin[k] .. face_ref_begin[k + 1]).
      std::vector<InteriorFace> faces;
      std::vector<FaceRef> face_refs;
      std::vector<size_t> face_ref_begin;

      void reinit(const ColoredCells<dim> &cells)
      {
        boundary_faces.clear();
        boundary_begin.assign(1, 0);
        faces.clear();
        constexpr uint none = dealii::numbers::invalid_unsigned_int;
        const auto add = [&](const auto &a, const uint fa, const uint sfa, const auto &b, const uint fb,
                             const uint sfb) { faces.push_back({{a, b}, {fa, fb}, {sfa, sfb}}); };

        for (const auto &cell : cells.all()) {
          for (const uint f : cell->face_indices()) {
            const bool periodic = cell->has_periodic_neighbor(f);
            if (cell->at_boundary(f) && !periodic) {
              boundary_faces.emplace_back(cell, f);
              continue;
            }
            const auto neighbor = cell->neighbor_or_periodic_neighbor(f);
            if (neighbor->has_children()) {
              // The finer cells list this face, unless they are not ours.
              if constexpr (dim == 1) {
                auto child = neighbor;
                while (child->has_children())
                  child = child->child(1 - f);
                if (!child->is_locally_owned()) add(child, 1 - f, none, cell, f, none);
              } else {
                const uint nf =
                    periodic ? cell->periodic_neighbor_of_periodic_neighbor(f) : cell->neighbor_of_neighbor(f);
                for (uint sf = 0; sf < cell->face(f)->n_children(); ++sf) {
                  const auto child = periodic ? cell->periodic_neighbor_child_on_subface(f, sf)
                                              : cell->neighbor_child_on_subface(f, sf);
                  if (!child->is_locally_owned()) add(child, nf, none, cell, f, sf);
                }
              }
              continue;
            }
            bool coarser;
            if constexpr (dim == 1)
              coarser = cell->level() > neighbor->level();
            else
              coarser = periodic ? cell->periodic_neighbor_is_coarser(f) : cell->neighbor_is_coarser(f);
            if (coarser) {
              if constexpr (dim == 1)
                add(cell, f, none, neighbor, periodic ? cell->periodic_neighbor_face_no(f) : cell->neighbor_face_no(f),
                    none);
              else {
                const auto [nf, nsf] = periodic ? cell->periodic_neighbor_of_coarser_periodic_neighbor(f)
                                                : cell->neighbor_of_coarser_neighbor(f);
                add(cell, f, none, neighbor, nf, nsf);
              }
              continue;
            }
            // Same level: listed from the smaller of two owned cells.
            if (neighbor->is_locally_owned() && neighbor < cell) continue;
            add(cell, f, none, neighbor, periodic ? cell->periodic_neighbor_face_no(f) : cell->neighbor_face_no(f),
                none);
          }
          boundary_begin.push_back(boundary_faces.size());
        }

        // Each face is referenced by its owned cells, in face order.
        const auto n_active = cells.size() > 0 ? cells[0]->get_triangulation().n_active_cells() : 0;
        std::vector<int> owned_index(n_active, -1);
        for (size_t k = 0; k < cells.size(); ++k)
          owned_index[cells[k]->active_cell_index()] = k;
        face_ref_begin.assign(cells.size() + 1, 0);
        for (const auto &face : faces)
          for (uint side = 0; side < 2; ++side)
            if (face.cell[side]->is_locally_owned())
              ++face_ref_begin[owned_index[face.cell[side]->active_cell_index()] + 1];
        std::partial_sum(face_ref_begin.begin(), face_ref_begin.end(), face_ref_begin.begin());
        face_refs.resize(face_ref_begin.back());
        std::vector<size_t> next(face_ref_begin.begin(), face_ref_begin.end() - 1);
        for (size_t f = 0; f < faces.size(); ++f)
          for (uint side = 0; side < 2; ++side)
            if (faces[f].cell[side]->is_locally_owned())
              face_refs[next[owned_index[faces[f].cell[side]->active_cell_index()]]++] = {f, side};
      }

      /// Reinit @p fe_iv on interior face f, with the cells of @p dof_handler (any DoFHandler on the triangulation).
      const dealii::FEInterfaceValues<dim> &reinit(dealii::FEInterfaceValues<dim> &fe_iv, const size_t f,
                                                   const dealii::DoFHandler<dim> &dof_handler) const
      {
        const auto &face = faces[f];
        fe_iv.reinit(on(dof_handler, face.cell[0]), face.face_no[0], face.subface_no[0], on(dof_handler, face.cell[1]),
                     face.face_no[1], face.subface_no[1]);
        return fe_iv;
      }

      /// The number of interior faces of owned cell k.
      size_t n_interior_faces(const size_t k) const { return face_ref_begin[k + 1] - face_ref_begin[k]; }
    };
  } // namespace internal
} // namespace DiFfRG
