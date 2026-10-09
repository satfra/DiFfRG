#pragma once

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/config_tree.hh>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kt_regression
{
  inline void ensure_logger()
  {
    try {
      auto log = spdlog::stdout_color_mt("log");
      log->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
  }

  // Ensure DiFfRG (MPI/Kokkos/TBB) is initialized exactly once. Required by
  // tests that construct a QuadratureProvider directly (i.e. tests that use
  // the integrator-based fluxes rather than a closed-form expression).
  inline void ensure_diffrg_initialized() { static const DiFfRG::Init diffrg_init; }

  inline double json_number_to_double(const DiFfRG::json::value &value)
  {
    if (value.is_double()) return value.as_double();
    if (value.is_int64()) return static_cast<double>(value.as_int64());
    if (value.is_uint64()) return static_cast<double>(value.as_uint64());
    throw std::runtime_error("Expected JSON number.");
  }

  inline std::vector<double> json_array_to_doubles(const DiFfRG::json::array &values)
  {
    std::vector<double> result;
    result.reserve(values.size());
    for (const auto &value : values)
      result.push_back(json_number_to_double(value));
    return result;
  }

  inline std::optional<std::string> compare_profiles(const std::string &name, const std::vector<double> &x_values,
                                                     const std::vector<double> &simulated,
                                                     const std::vector<double> &reference, const double abs_tolerance,
                                                     const double rel_tolerance)
  {
    if (x_values.size() != simulated.size() || x_values.size() != reference.size())
      return std::string("Size mismatch while comparing ") + name + ".";

    std::ostringstream mismatch;
    mismatch << std::setprecision(17);

    std::size_t mismatch_count = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    for (std::size_t i = 0; i < simulated.size(); ++i) {
      const double abs_error = std::abs(simulated[i] - reference[i]);
      const double reference_magnitude = std::abs(reference[i]);
      const double rel_error = reference_magnitude == 0.0
                                   ? (abs_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                                   : abs_error / reference_magnitude;
      max_abs_error = std::max(max_abs_error, abs_error);
      max_rel_error = std::max(max_rel_error, rel_error);

      if (abs_error <= abs_tolerance || rel_error <= rel_tolerance) continue;

      if (mismatch_count == 0) mismatch << name << " comparison failed.\n";
      if (mismatch_count < 8) {
        mismatch << "  i=" << i << ", sigma=" << x_values[i] << ", simulated=" << simulated[i]
                 << ", reference=" << reference[i] << ", abs_error=" << abs_error << ", rel_error=" << rel_error
                 << '\n';
      }
      ++mismatch_count;
    }

    if (mismatch_count == 0) return std::nullopt;

    mismatch << "  total mismatches=" << mismatch_count << ", max_abs_error=" << max_abs_error
             << ", max_rel_error=" << max_rel_error;
    return mismatch.str();
  }

  struct SampledProfile {
    std::vector<double> x;
    std::vector<double> y;
  };

  template <typename FlowingVariablesType, typename DiscretizationType>
  SampledProfile sample_sorted_profile(const FlowingVariablesType &state, const DiscretizationType &discretization,
                                       const double origin_tol)
  {
    const auto &support_points = discretization.get_support_points();
    const auto &spatial_data = state.spatial_data();

    std::vector<std::pair<double, double>> sampled_values;
    sampled_values.reserve(spatial_data.size());
    for (unsigned int i = 0; i < spatial_data.size(); ++i)
      sampled_values.emplace_back(support_points[i][0], spatial_data[i]);
    std::sort(sampled_values.begin(), sampled_values.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    SampledProfile result;
    result.x.reserve(sampled_values.size());
    result.y.reserve(sampled_values.size());
    for (const auto &[position, value] : sampled_values) {
      result.x.push_back(position);
      result.y.push_back(value);
    }
    if (!result.x.empty() && std::abs(result.x.front()) < origin_tol) result.y.front() = 0.0;
    return result;
  }
} // namespace kt_regression
