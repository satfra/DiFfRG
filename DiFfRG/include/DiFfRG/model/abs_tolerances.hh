#pragma once

// external libraries
#include <deal.II/base/point.h>

// standard library
#include <array>
#include <cstddef>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Whether a model provides per-component tolerances for the IDA timesteppers.
     *
     * Optional hook, for components that live on very different scales, or whose error should be measured against
     * another scale than their own size. The values are the whole tolerance: IDA then runs with relative tolerance 0,
     * so a model that wants a relative tolerance builds it in, e.g. \f$\max(\text{abs\_tol},\, \text{rel\_tol}\,|s|)\f$
     * with \f$s\f$ the scale of the component.
     * @code
     * template <int dim, typename Solution, size_t n>
     * void abs_tolerances(std::array<double, n> &atol, const Point<dim> &x, const Solution &sol, double abs_tol,
     *                     double rel_tol) const;
     * @endcode
     * It is called once per dof location (LDG: support point; KT: cell centre) and fills one absolute tolerance per
     * FE function; @p atol is pre-filled with the uniform @p abs_tol, @p rel_tol is the stepper's relative tolerance.
     * `sol` is a named tuple with `"fe_functions"`, `"fe_derivatives"`, `"fe_hessians"` (LDG: of the FE functions,
     * not the LDG levels; KT: reconstructed gradient and curvature). Used by the LDG and KT assemblers with serial
     * vectors only; elsewhere the uniform `abs_tol` applies. @see AbstractAssembler::local_abs_tolerances
     */
    template <typename Model, int dim, typename Solution, size_t n>
    concept HasAbsTolerances =
        requires(const Model &m, std::array<double, n> &atol, const dealii::Point<dim> &x, const Solution &sol) {
          m.abs_tolerances(atol, x, sol, 1., 1.);
        };
  } // namespace def
} // namespace DiFfRG
