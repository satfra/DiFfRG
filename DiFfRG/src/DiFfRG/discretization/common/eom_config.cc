#include <DiFfRG/discretization/common/eom_config.hh>

#include <cmath>
#include <stdexcept>

// fmt::format instead of std::format: libstdc++ ships <format> only from GCC
// 13, and spdlog's fmt is bundled anyway -- this keeps GCC 12 viable.
#include <spdlog/fmt/fmt.h>

namespace DiFfRG::Config
{
  EoMConfig::EoMConfig(const double abs_tol, const unsigned int max_iter, const double smoothing_length,
                       const double bound_tolerance, const double armijo_coefficient, const unsigned int max_backtracks)
      : abs_tol(abs_tol), max_iter(max_iter), smoothing_length(smoothing_length), bound_tolerance(bound_tolerance),
        armijo_coefficient(armijo_coefficient), max_backtracks(max_backtracks)
  {
    validate();
  }

  EoMConfig::EoMConfig(const ConfigTree &config)
      : EoMConfig(config.get_double("/discretization/EoM_abs_tol", default_abs_tol),
                  config.get_uint("/discretization/EoM_max_iter", default_max_iter),
                  config.get_double("/discretization/EoM_smoothing_length", default_smoothing_length),
                  config.get_double("/discretization/EoM_bound_tolerance", default_bound_tolerance),
                  config.get_double("/discretization/EoM_armijo_coefficient", default_armijo_coefficient),
                  config.get_uint("/discretization/EoM_max_backtracks", default_max_backtracks))
  {
    raw_potential_order = config.get_uint("/discretization/raw_potential_order", default_raw_potential_order);
    raw_potential_recover_mass_hessian = config.get_bool("/discretization/raw_potential_recover_mass_hessian",
                                                         default_raw_potential_recover_mass_hessian);
    raw_potential_mass_hessian_jump_threshold = config.get_double(
        "/discretization/raw_potential_mass_hessian_jump_threshold", default_raw_potential_mass_hessian_jump_threshold);
    validate();
  }

  std::string EoMConfig::get_defaults()
  {
    return fmt::format(R"({{
  "discretization": {{
    "EoM_abs_tol": {},
    "EoM_max_iter": {},
    "EoM_smoothing_length": {},
    "EoM_bound_tolerance": {},
    "EoM_armijo_coefficient": {},
    "EoM_max_backtracks": {},
    "raw_potential_order": {},
    "raw_potential_recover_mass_hessian": {},
    "raw_potential_mass_hessian_jump_threshold": {}
  }}
}})",
                       default_abs_tol, default_max_iter, default_smoothing_length, default_bound_tolerance,
                       default_armijo_coefficient, default_max_backtracks, default_raw_potential_order,
                       default_raw_potential_recover_mass_hessian, default_raw_potential_mass_hessian_jump_threshold);
  }

  void EoMConfig::validate() const
  {
    if (!std::isfinite(abs_tol) || abs_tol < 0.)
      throw std::invalid_argument("EoM_abs_tol must be finite and non-negative.");
    if (!std::isfinite(smoothing_length) || (smoothing_length < 0. && smoothing_length != -1.))
      throw std::invalid_argument("EoM_smoothing_length must be -1 (automatic), zero, or a positive physical length.");
    if (!std::isfinite(bound_tolerance) || bound_tolerance < 0.)
      throw std::invalid_argument("EoM_bound_tolerance must be finite and non-negative.");
    if (!std::isfinite(armijo_coefficient) || !(armijo_coefficient > 0. && armijo_coefficient < 1.))
      throw std::invalid_argument("EoM_armijo_coefficient must be finite and strictly between zero and one.");
    if (max_backtracks == 0) throw std::invalid_argument("EoM_max_backtracks must be positive.");
    if (raw_potential_order != 2 && raw_potential_order != 3)
      throw std::invalid_argument("raw_potential_order must be either 2 or 3.");
    const auto valid_jump_threshold = [](const double value) {
      return std::isfinite(value) && (value == -1. || value >= 0.);
    };
    if (!valid_jump_threshold(raw_potential_mass_hessian_jump_threshold))
      throw std::invalid_argument(
          "raw_potential_mass_hessian_jump_threshold must be -1 (disabled), zero, or positive.");
  }
} // namespace DiFfRG::Config
