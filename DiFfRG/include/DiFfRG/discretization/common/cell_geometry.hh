#pragma once

// external libraries
#include <deal.II/dofs/dof_handler.h>

// standard library
#include <cmath>
#include <limits>
#include <stdexcept>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief The extent of a cell along the normal of one of its faces.
     *
     * Volume over face area, which is the width of the cell in the direction that face looks in.
     * Cheaper and better behaved than the diameter for anisotropic cells: on a long thin cell the
     * diameter is dominated by the long direction, while this reports the short one when asked
     * about the face that spans it.
     */
    template <typename CellIterator> double face_normal_cell_width(const CellIterator &cell, const uint face_no)
    {
      const double face_measure = cell->face(face_no)->measure();
      if (!(face_measure > 0.) || !std::isfinite(face_measure))
        throw std::runtime_error("face_normal_cell_width: invalid face measure.");

      const double width = cell->measure() / face_measure;
      if (!(width > 0.) || !std::isfinite(width))
        throw std::runtime_error("face_normal_cell_width: invalid face-normal cell width.");
      return width;
    }

    /**
     * @brief The smallest face-normal width over all faces of @p cell.
     *
     * Templated on the iterator rather than on the dimension: the only thing needed of a cell is
     * that it can measure itself and its faces, and the assemblers hand out several different
     * accessor types.
     */
    template <typename CellIterator> double cell_width(const CellIterator &cell)
    {
      double width = std::numeric_limits<double>::max();
      for (uint face_no = 0; face_no < cell->n_faces(); ++face_no)
        width = std::min(width, face_normal_cell_width(cell, face_no));
      return width;
    }

    /// @brief The smallest face-normal width anywhere on the mesh.
    template <int dim> double minimum_face_normal_cell_width(const dealii::DoFHandler<dim> &dof_handler)
    {
      double minimum_width = std::numeric_limits<double>::max();
      for (const auto &cell : dof_handler.active_cell_iterators())
        minimum_width = std::min(minimum_width, cell_width(cell));

      if (!(minimum_width < std::numeric_limits<double>::max()))
        throw std::runtime_error("minimum_face_normal_cell_width: the mesh has no active cells.");
      return minimum_width;
    }
  } // namespace internal
} // namespace DiFfRG
