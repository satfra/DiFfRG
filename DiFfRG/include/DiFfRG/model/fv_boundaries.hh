#pragma once

// standard library
#include <array>
#include <cmath>

// external libraries
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

namespace DiFfRG
{
  namespace def
  {
    using namespace dealii;
    template <int dim, typename NumberType, std::size_t n_components>
    using BoundaryStencilValues = std::array<std::array<NumberType, n_components>, 2 * dim + 3>;

    template <int dim> using BoundaryStencilPoints = std::array<Point<dim>, 2 * dim + 3>;

    namespace BoundaryStencilIndex
    {
      constexpr size_t lower_outer = 0;
      constexpr size_t lower_inner = 1;
      constexpr size_t physical_cell = 2;
      constexpr size_t upper_inner = 3;
      constexpr size_t upper_outer = 4;
    } // namespace BoundaryStencilIndex

    namespace BoundaryStencilDetail
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
    } // namespace BoundaryStencilDetail

    /**
     * @brief Default FV boundary strategy used by the Kurganov-Tadmor assembler.
     *
     * In the stencil-only KT implementation this means affine extrapolation from the
     * nearest physical cells on either boundary.
     */
    template <typename Model> class FVDefaultBoundaries
    {
    public:
      template <int dim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(BoundaryStencilValues<dim, NumberType, n_components> &u_stencil,
                                  BoundaryStencilPoints<dim> &x_stencil, const Point<dim> &x_face) const
      {
        static_assert(dim == 1 || dim == 2,
                      "FV KT default boundary stencils currently support one- and two-dimensional FV domains.");

        const unsigned int axis = BoundaryStencilDetail::BoundaryStencilGeometry<dim>::active_axis(x_stencil, x_face);
        const auto side = BoundaryStencilDetail::boundary_side(axis, x_stencil, x_face);
        BoundaryStencilDetail::apply_affine_two_ghost_extrapolation(axis, side, u_stencil, x_stencil);
        return true;
      }
    };

    /**
     * @brief FV boundary strategy for rho-coordinate models with an even lower-boundary symmetry.
     *
     * The lower boundary face is the symmetry plane at rho = 0. The first physical cell remains a free
     * cell-centered unknown; only the two ghost cells are filled by even reflection. The upper boundary
     * keeps the affine extrapolation used by FVDefaultBoundaries.
     */
    template <typename Model> class RhoSymmetricLinearExtrapolationBoundaries
    {
    public:
      template <int dim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(BoundaryStencilValues<dim, NumberType, n_components> &u_stencil,
                                  BoundaryStencilPoints<dim> &x_stencil, const Point<dim> &x_face) const
      {
        static_assert(dim == 1,
                      "RhoSymmetricLinearExtrapolationBoundaries currently supports only 1D FV domains.");

        using namespace BoundaryStencilIndex;
        const bool lower_boundary = x_face[0] <= x_stencil[physical_cell][0];
        const double delta = lower_boundary ? (x_stencil[upper_inner][0] - x_stencil[physical_cell][0])
                                            : (x_stencil[physical_cell][0] - x_stencil[lower_inner][0]);

        if (lower_boundary) {
          x_stencil[lower_inner][0] = x_stencil[physical_cell][0] - delta;
          x_stencil[lower_outer][0] = x_stencil[physical_cell][0] - 2.0 * delta;
          for (size_t c = 0; c < n_components; ++c) {
            u_stencil[lower_inner][c] = u_stencil[physical_cell][c];
            u_stencil[lower_outer][c] = u_stencil[upper_inner][c];
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

    /**
     * @brief FV boundary strategy using odd reflection at the origin and linear extrapolation at the outer boundary.
     *
     * Within the current face-based FV implementation this uses:
     * - odd reflection of the conserved quantity at the lower boundary (sigma = 0),
     * - linear extrapolation at the upper boundary (sigma = sigma_max),
     * - smooth-extension ghost gradients.
     *
     * The strategy targets one-dimensional FV domains and paper-style two-dimensional vector parity.
     */
    template <typename Model> class OriginOddLinearExtrapolationBoundaries
    {
    public:
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
        const unsigned int axis = BoundaryStencilDetail::BoundaryStencilGeometry<dim>::active_axis(x_stencil, x_face);
        const auto side = BoundaryStencilDetail::boundary_side(axis, x_stencil, x_face);
        if (side == BoundaryStencilDetail::BoundarySide::upper) {
          BoundaryStencilDetail::apply_affine_two_ghost_extrapolation(axis, side, u_stencil, x_stencil);
          return true;
        }

        const double delta = BoundaryStencilDetail::boundary_delta(axis, side, x_stencil);
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
     * @brief FV boundary strategy using odd reflection around a model-provided origin value.
     *
     * This mirrors `u - b` oddly at the lower boundary and keeps affine extrapolation at the upper boundary.
     */
    template <typename Model> class OriginShiftedOddLinearExtrapolationBoundaries
    {
      static bool is_lower_boundary_1d(const Point<1> &x_face, const BoundaryStencilPoints<1> &x_stencil)
      {
        return x_face[0] <= x_stencil[BoundaryStencilIndex::physical_cell][0];
      }

    public:
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
