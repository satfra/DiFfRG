#pragma once

#include <atomic>
#include <cstdint>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Counts changes of interpolator data in the process.
     *
     * Every interpolator update() that changes its data increments it. Caches of quantities computed from
     * interpolators, such as the hoisted pre() tables of the finite-temperature integrators, store the value they
     * were built at and are rebuilt when it moved.
     */
    inline std::atomic<std::uint64_t> interpolator_data_epoch{0};

    inline void bump_interpolator_data_epoch() { interpolator_data_epoch.fetch_add(1, std::memory_order_relaxed); }

    inline std::uint64_t current_interpolator_data_epoch()
    {
      return interpolator_data_epoch.load(std::memory_order_relaxed);
    }
  } // namespace internal
} // namespace DiFfRG
