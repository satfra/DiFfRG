#pragma once

#include <DiFfRG/common/tuples.hh>
#include <tuple>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      namespace internal
      {
        /**
         * @brief The named tuple handed to model.flux().
         *
         * "extractors" and "variables" carry the same data the source() tuple does, filled once per
         * residual and once per jacobian by the assembler's extract() before any flux is evaluated.
         * They are plain NumberType even inside the AD-seeded flux loops, i.e. the flux jacobian does
         * not contain the dependence of an extractor on the FE solution -- the extractors are frozen
         * within a Newton step, exactly as they are for source(). A model that solves for something
         * at the EoM in extract() and reads it here therefore gets an approximate jacobian, which
         * costs Newton iterations rather than accuracy: the extraction is redone for every residual.
         *
         * The slot order matches the FEM assemblers' batch_tie: the discretization-specific solution
         * entries first, then "extractors", "variables".
         */
        template <typename... T> auto flux_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>,
                             StringSet<"fe_functions", "fe_derivatives", "extractors", "variables">>(std::tie(t...));
        }

        /// @brief The named tuple handed to model.diffusion_flux(). @see flux_tie for the entries.
        template <typename... T> auto diffusion_flux_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives", "fe_third_derivatives",
                                                           "extractors", "variables">>(std::tie(t...));
        }
      } // namespace internal
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
