#pragma once

// external libraries
#include <deal.II/sundials/ida.h>
#include <ida/ida.h>

// standard library
#include <iostream>
#include <stdexcept>
#include <string>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Bound IDA's step size to [minimal_dt, maximal_dt].
     *
     * deal.II's IDA wrapper does not set step-size bounds. DiFfRG's deal.II patch
     * (dependencies/dealii_ida_max_num_steps.patch) adds a custom_setup hook through which they are set; with an
     * unpatched deal.II the bounds are skipped with a one-time warning.
     */
    template <typename VectorType>
    void set_ida_step_bounds(dealii::SUNDIALS::IDA<VectorType> &ida, const double minimal_dt, const double maximal_dt)
    {
      if constexpr (requires { ida.custom_setup; }) {
        ida.custom_setup = [minimal_dt, maximal_dt](void *ida_mem) {
          if (IDASetMinStep(ida_mem, minimal_dt) < 0 || IDASetMaxStep(ida_mem, maximal_dt) < 0)
            throw std::runtime_error("IDA: could not set the step size bounds [" + std::to_string(minimal_dt) + ", " +
                                     std::to_string(maximal_dt) + "]");
        };
      } else {
        static bool warned = false;
        if (!warned)
          std::cerr << "WARNING: this deal.II lacks DiFfRG's IDA patch; /timestepping/implicit/minimal_dt and "
                       "maximal_dt do not bound IDA's step size. Rebuild the dependency bundle to enable them."
                    << std::endl;
        warned = true;
      }
    }
  } // namespace internal
} // namespace DiFfRG
