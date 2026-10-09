#pragma once

#include <DiFfRG/common/configuration_helper.hh>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

/**
 * Environment-driven overrides for the assembly benchmarks, so that a sweep needs no rebuild. The
 * benchmarks are linked against Catch2WithMain, which rejects unknown command-line flags, so the
 * environment is the remaining channel.
 *
 *   DIFFRG_BENCH_CELLS  -> cells per axis; rewrites the grid spec
 *   DIFFRG_BENCH_FE     -> restricts the fe_order GENERATE to a single value
 *
 * Unset variables leave the config alone.
 */
namespace DiFfRG
{
  namespace Testing
  {
    inline std::optional<unsigned int> env_uint(const char *name)
    {
      const char *raw = std::getenv(name);
      if (raw == nullptr || *raw == '\0') return std::nullopt;
      return static_cast<unsigned int>(std::strtoul(raw, nullptr, 10));
    }

    /// Apply the sweep overrides in place.
    inline void apply_benchmark_sweep(DiFfRG::ConfigTree &json)
    {
      if (const auto requested = env_uint("DIFFRG_BENCH_CELLS")) {
        const unsigned int cells = std::max(1u, *requested);
        // Full precision: six decimals is already off by a cell at 2048.
        std::ostringstream step;
        step << std::setprecision(17) << 1.0 / static_cast<double>(cells);
        const std::string spec = "0:" + step.str() + ":1";
        for (const auto *axis : {"x_grid", "y_grid", "z_grid"})
          json.set_string(std::string("/discretization/grid/") + axis, spec);
      }
    }

    /// The benchmark's own fe_order set, or the single order pinned by DIFFRG_BENCH_FE.
    inline std::vector<int> benchmark_fe_orders(std::vector<int> orders)
    {
      if (const auto pinned = env_uint("DIFFRG_BENCH_FE")) return {static_cast<int>(*pinned)};
      return orders;
    }
  } // namespace Testing
} // namespace DiFfRG
