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
     * @brief A model that sets per-component absolute tolerances for the implicit time stepper.
     *
     * The hook is called once per dof location (cell centre for FV, support point for LDG) with the solution
     * there, and fills one absolute tolerance per FE function:
     *
     *     template <int dim, typename Solution, size_t n>
     *     void abs_tolerances(std::array<double, n> &atol, const Point<dim> &x, const Solution &sol,
     *                         double abs_tol, double rel_tol) const;
     *
     * `sol` is a named tuple with "fe_functions" (values), "fe_derivatives" (Tensor<1, dim> per component) and
     * "fe_hessians" (Tensor<2, dim> per component). abs_tol and rel_tol are the stepper's uniform tolerances.
     * @see AbstractAssembler::local_abs_tolerances
     */
    template <typename Model, int dim, typename Solution, size_t n>
    concept HasAbsTolerances =
        requires(const Model &m, std::array<double, n> &atol, const dealii::Point<dim> &x, const Solution &sol) {
          m.abs_tolerances(atol, x, sol, 1., 1.);
        };
  } // namespace def
} // namespace DiFfRG
