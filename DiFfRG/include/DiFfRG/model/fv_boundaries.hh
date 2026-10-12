#pragma once

// standard library
#include <array>
#include <cmath>

// external libraries
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

// clang-format off
/**
 * @file fv_boundaries.hh
 * @brief Boundary conditions of the Kurganov-Tadmor (KT) finite-volume assembler: ghost-cell mixins.
 *
 * KT reconstructs the solution at a face from the cell averages of two cells on either side. At a boundary face the
 * missing cells are ghost cells, and the model fills them through
 * @code
 * template <int dim, typename NumberType, size_t n_components>
 * bool apply_boundary_stencil(BoundaryStencilValues<dim, NumberType, n_components> &u_stencil,
 *                             BoundaryStencilPoints<dim> &x_stencil, const Point<dim> &x_face) const;
 * @endcode
 * Return false if the boundary is not supported; KT then throws. A model usually inherits one of the mixins below.
 *
 * ### The stencil
 * Five cells along the axis normal to the face, indexed by def::BoundaryStencilIndex:
 * | Index | Name            | Lower boundary            | Upper boundary            |
 * |-------|-----------------|---------------------------|---------------------------|
 * | 0     | `lower_outer`   | ghost, to be filled       | second interior cell      |
 * | 1     | `lower_inner`   | ghost, to be filled       | first interior cell       |
 * | 2     | `physical_cell` | the boundary cell \f$P\f$ | the boundary cell \f$P\f$ |
 * | 3     | `upper_inner`   | first interior cell       | ghost, to be filled       |
 * | 4     | `upper_outer`   | second interior cell      | ghost, to be filled       |
 *
 * `u_stencil[i][c]` is the cell average of component c, `x_stencil[i]` the cell centre; a mixin sets both for the
 * ghosts and may also overwrite the value used for \f$P\f$. The face is "lower" if it lies below the centre of
 * \f$P\f$ along that axis. The arrays have \f$2d + 3\f$ entries; in 2D, entries 5 and 6 are not filled by the
 * assembler and not read by the mixins.
 *
 * ### Mixins
 * With \f$u_P\f$, \f$u_1\f$, \f$u_2\f$ the values of the boundary cell and the first and second interior cell
 * ("odd about \f$P\f$" = about the centre of the boundary cell; "affine" = as FVDefaultBoundaries, mirrored):
 * | Mixin | dim | Lower boundary: ghosts (inner, outer); boundary cell | Upper |
 * |-------|-----|------------------------------------------------------|-------|
 * | FVDefaultBoundaries | 1, 2 | \f$(2u_P - u_1,\ 3u_P - 2u_1)\f$: affine | affine |
 * | OriginOddLinearExtrapolationBoundaries | 1, 2 | \f$(-u_1,\ -u_2)\f$, \f$u_P \to 0\f$: odd about \f$P\f$ | affine |
 * | OriginShiftedOddLinearExtrapolationBoundaries | 1 | \f$(2b - u_1,\ 2b - u_2)\f$, \f$u_P \to b\f$ | affine |
 */
// clang-format on

namespace DiFfRG
{
  namespace def
  {
    using namespace dealii;
    /// Cell averages of the boundary stencil: `[stencil index][component]`, see fv_boundaries.hh.
    template <int dim, typename NumberType, std::size_t n_components>
    using BoundaryStencilValues = std::array<std::array<NumberType, n_components>, 2 * dim + 3>;

    /// Cell centres of the boundary stencil, see fv_boundaries.hh.
    template <int dim> using BoundaryStencilPoints = std::array<Point<dim>, 2 * dim + 3>;

    /// Positions in the boundary stencil, from lower to upper along the axis normal to the boundary face.
    namespace BoundaryStencilIndex
    {
      constexpr size_t lower_outer = 0;
      constexpr size_t lower_inner = 1;
      constexpr size_t physical_cell = 2;
      constexpr size_t upper_inner = 3;
      constexpr size_t upper_outer = 4;
    } // namespace BoundaryStencilIndex

    namespace internal
    {
      enum class BoundarySide { lower, upper };

      template <int dim> struct BoundaryStencilGeometry;

      template <> struct BoundaryStencilGeometry<1>
      {
        static unsigned int active_axis(const BoundaryStencilPoints<1> &, const Point<1> &) { return 0; }
      };

      template <> struct BoundaryStencilGeometry<2>
      {
        static unsigned int active_axis(const BoundaryStencilPoints<2> &x_stencil, const Point<2> &x_face)
        {
          using namespace BoundaryStencilIndex;
          const double distance_0 = std::abs(x_face[0] - x_stencil[physical_cell][0]);
          const double distance_1 = std::abs(x_face[1] - x_stencil[physical_cell][1]);
          return distance_1 > distance_0 ? 1U : 0U;
        }
      };

      template <int dim>
      BoundarySide boundary_side(const unsigned int axis, const BoundaryStencilPoints<dim> &x_stencil,
                                 const Point<dim> &x_face)
      {
        return x_face[axis] <= x_stencil[BoundaryStencilIndex::physical_cell][axis] ? BoundarySide::lower
                                                                                    : BoundarySide::upper;
      }

      template <int dim>
      double boundary_delta(const unsigned int axis, const BoundarySide side,
                            const BoundaryStencilPoints<dim> &x_stencil)
      {
        using namespace BoundaryStencilIndex;
        return side == BoundarySide::lower ? (x_stencil[upper_inner][axis] - x_stencil[physical_cell][axis])
                                           : (x_stencil[physical_cell][axis] - x_stencil[lower_inner][axis]);
      }

      template <int dim, typename NumberType, size_t n_components>
      void apply_affine_two_ghost_extrapolation(const unsigned int axis, const BoundarySide side,
                                                BoundaryStencilValues<dim, NumberType, n_components> &u_stencil,
                                                BoundaryStencilPoints<dim> &x_stencil)
      {
        using namespace BoundaryStencilIndex;
        const double delta = boundary_delta(axis, side, x_stencil);

        if (side == BoundarySide::lower) {
          x_stencil[lower_inner] = x_stencil[physical_cell];
          x_stencil[lower_outer] = x_stencil[physical_cell];
          x_stencil[lower_inner][axis] = x_stencil[physical_cell][axis] - delta;
          x_stencil[lower_outer][axis] = x_stencil[physical_cell][axis] - 2.0 * delta;
          for (size_t c = 0; c < n_components; ++c) {
            u_stencil[lower_inner][c] = NumberType(2.0) * u_stencil[physical_cell][c] - u_stencil[upper_inner][c];
            u_stencil[lower_outer][c] =
                NumberType(3.0) * u_stencil[physical_cell][c] - NumberType(2.0) * u_stencil[upper_inner][c];
          }
          return;
        }

        x_stencil[upper_inner] = x_stencil[physical_cell];
        x_stencil[upper_outer] = x_stencil[physical_cell];
        x_stencil[upper_inner][axis] = x_stencil[physical_cell][axis] + delta;
        x_stencil[upper_outer][axis] = x_stencil[physical_cell][axis] + 2.0 * delta;
        for (size_t c = 0; c < n_components; ++c) {
          u_stencil[upper_inner][c] = NumberType(2.0) * u_stencil[physical_cell][c] - u_stencil[lower_inner][c];
          u_stencil[upper_outer][c] =
              NumberType(3.0) * u_stencil[physical_cell][c] - NumberType(2.0) * u_stencil[lower_inner][c];
        }
      }
    } // namespace internal

    /**
     * @brief Affine extrapolation on both boundaries: the ghosts continue the straight line through the boundary
     * cell and the first interior cell. 1D and 2D.
     */
    template <typename Model> class FVDefaultBoundaries
    {
    public:
      /// Fill the ghost cells of @p u_stencil and @p x_stencil for the boundary face at @p x_face; always true.
      template <int dim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(BoundaryStencilValues<dim, NumberType, n_components> &u_stencil,
                                  BoundaryStencilPoints<dim> &x_stencil, const Point<dim> &x_face) const
      {
        static_assert(dim == 1 || dim == 2,
                      "FV KT default boundary stencils currently support one- and two-dimensional FV domains.");

        const unsigned int axis = internal::BoundaryStencilGeometry<dim>::active_axis(x_stencil, x_face);
        const auto side = internal::boundary_side(axis, x_stencil, x_face);
        internal::apply_affine_two_ghost_extrapolation(axis, side, u_stencil, x_stencil);
        return true;
      }
    };

    /**
     * @brief Deprecated: FVDefaultBoundaries. This used to reflect the lower ghosts evenly about the face at
     * \f$\rho = 0\f$, which imposes \f$u'(0) = 0\f$; a function of \f$\rho\f$ has no such symmetry (e.g.
     * \f$u = V'(\rho)\f$ has \f$u'(0) = V''(0)\f$), so the ghosts continue the derivative instead.
     */
    template <typename Model>
    using RhoSymmetricLinearExtrapolationBoundaries [[deprecated(
        "RhoSymmetricLinearExtrapolationBoundaries is FVDefaultBoundaries now: rho = 0 is no symmetry plane in rho, "
        "the ghosts continue the derivative. Use FVDefaultBoundaries.")]] = FVDefaultBoundaries<Model>;

    /**
     * @brief Lower boundary: odd reflection about the centre of the boundary cell, which is meant to sit at the
     * origin \f$\sigma = 0\f$; that cell's value is set to 0 in the stencil. Upper boundary: affine extrapolation.
     *
     * 1D: all components are odd. 2D: requires two components, a vector \f$(u_0, u_1)\f$; at a lower face normal to
     * axis a, component a is odd as above and the other one is reflected evenly about the face.
     */
    template <typename Model> class OriginOddLinearExtrapolationBoundaries
    {
    public:
      /// Fill the ghost cells of @p u_stencil and @p x_stencil for the boundary face at @p x_face; always true.
      template <int dim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(BoundaryStencilValues<dim, NumberType, n_components> &u_stencil,
                                  BoundaryStencilPoints<dim> &x_stencil, const Point<dim> &x_face) const
      {
        static_assert(dim == 1 || dim == 2,
                      "OriginOddLinearExtrapolationBoundaries supports one- and two-dimensional FV domains.");
        if constexpr (dim == 2) {
          static_assert(n_components == dim,
                        "OriginOddLinearExtrapolationBoundaries requires one flow component per dimension in 2D.");
        }

        using namespace BoundaryStencilIndex;
        const unsigned int axis = internal::BoundaryStencilGeometry<dim>::active_axis(x_stencil, x_face);
        const auto side = internal::boundary_side(axis, x_stencil, x_face);
        if (side == internal::BoundarySide::upper) {
          internal::apply_affine_two_ghost_extrapolation(axis, side, u_stencil, x_stencil);
          return true;
        }

        const double delta = internal::boundary_delta(axis, side, x_stencil);
        x_stencil[lower_inner] = x_stencil[physical_cell];
        x_stencil[lower_outer] = x_stencil[physical_cell];
        x_stencil[lower_inner][axis] = x_stencil[physical_cell][axis] - delta;
        x_stencil[lower_outer][axis] = x_stencil[physical_cell][axis] - 2.0 * delta;

        if constexpr (dim == 1) {
          for (size_t c = 0; c < n_components; ++c) {
            u_stencil[lower_inner][c] = -u_stencil[upper_inner][c];
            u_stencil[lower_outer][c] = -u_stencil[upper_outer][c];
            u_stencil[physical_cell][c] = NumberType(0.0);
          }
        } else {
          const size_t odd_component = axis;
          const size_t even_component = 1U - axis;
          u_stencil[lower_inner][odd_component] = -u_stencil[upper_inner][odd_component];
          u_stencil[lower_outer][odd_component] = -u_stencil[upper_outer][odd_component];
          u_stencil[physical_cell][odd_component] = NumberType(0.0);
          u_stencil[lower_inner][even_component] = u_stencil[physical_cell][even_component];
          u_stencil[lower_outer][even_component] = u_stencil[upper_inner][even_component];
        }

        return true;
      }
    };

    /**
     * @brief As OriginOddLinearExtrapolationBoundaries, but \f$u_c - b_c\f$ is odd: the boundary cell takes the value
     * \f$b_c\f$. 1D only.
     *
     * The model provides the values \f$b_c\f$, one per component:
     * @code
     * template <class NT, size_t n> std::array<NT, n> origin_odd_reflection_values() const;
     * @endcode
     */
    template <typename Model> class OriginShiftedOddLinearExtrapolationBoundaries
    {
      static bool is_lower_boundary_1d(const Point<1> &x_face, const BoundaryStencilPoints<1> &x_stencil)
      {
        return x_face[0] <= x_stencil[BoundaryStencilIndex::physical_cell][0];
      }

    public:
      /// Fill the ghost cells of @p u_stencil and @p x_stencil for the boundary face at @p x_face; always true.
      template <int dim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(BoundaryStencilValues<dim, NumberType, n_components> &u_stencil,
                                  BoundaryStencilPoints<dim> &x_stencil, const Point<dim> &x_face) const
      {
        static_assert(dim == 1, "OriginShiftedOddLinearExtrapolationBoundaries currently supports only 1D FV domains.");

        const bool lower_boundary = is_lower_boundary_1d(x_face, x_stencil);
        using namespace BoundaryStencilIndex;
        const double delta = lower_boundary ? (x_stencil[upper_inner][0] - x_stencil[physical_cell][0])
                                            : (x_stencil[physical_cell][0] - x_stencil[lower_inner][0]);

        if (lower_boundary) {
          const auto origin_values =
              static_cast<const Model &>(*this).template origin_odd_reflection_values<NumberType, n_components>();

          x_stencil[lower_inner][0] = x_stencil[physical_cell][0] - delta;
          x_stencil[lower_outer][0] = x_stencil[physical_cell][0] - 2.0 * delta;
          for (size_t c = 0; c < n_components; ++c) {
            const NumberType b = origin_values[c];
            u_stencil[lower_inner][c] = NumberType(2.0) * b - u_stencil[upper_inner][c];
            u_stencil[lower_outer][c] = NumberType(2.0) * b - u_stencil[upper_outer][c];
            u_stencil[physical_cell][c] = b;
          }
          return true;
        }

        x_stencil[upper_inner][0] = x_stencil[physical_cell][0] + delta;
        x_stencil[upper_outer][0] = x_stencil[physical_cell][0] + 2.0 * delta;
        for (size_t c = 0; c < n_components; ++c) {
          u_stencil[upper_inner][c] = NumberType(2.0) * u_stencil[physical_cell][c] - u_stencil[lower_inner][c];
          u_stencil[upper_outer][c] =
              NumberType(3.0) * u_stencil[physical_cell][c] - NumberType(2.0) * u_stencil[lower_inner][c];
        }
        return true;
      }
    };
  } // namespace def
} // namespace DiFfRG
