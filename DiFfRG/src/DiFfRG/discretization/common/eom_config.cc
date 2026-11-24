#include <DiFfRG/discretization/common/eom_config.hh>

#include <cmath>
#include <stdexcept>
#include <utility>

// fmt::format instead of std::format: libstdc++ ships <format> only from GCC
// 13, and spdlog's fmt is bundled anyway -- this keeps GCC 12 viable.
#include <spdlog/fmt/fmt.h>

namespace DiFfRG::Config
{
  EoMConfig::EoMConfig(const double abs_tol, const unsigned int max_iter, const double armijo_coefficient,
                       const unsigned int max_backtracks)
      : abs_tol(abs_tol), max_iter(max_iter), armijo_coefficient(armijo_coefficient), max_backtracks(max_backtracks)
  {
    validate();
  }

  EoMConfig::EoMConfig(const ConfigTree &config)
      : EoMConfig(config.get_double("/discretization/EoM_abs_tol", default_abs_tol),
                  config.get_uint("/discretization/EoM_max_iter", default_max_iter),
                  config.get_double("/discretization/EoM_armijo_coefficient", default_armijo_coefficient),
                  config.get_uint("/discretization/EoM_max_backtracks", default_max_backtracks))
  {
    const auto read_array = [&](const std::string &key) {
      std::vector<double> bounds;
      if (config.contains(key)) {
        const auto tree = static_cast<json::value>(config);
        const auto &array = tree.at_pointer(key).as_array();
        for (std::size_t i = 0; i < array.size(); ++i)
          bounds.push_back(config.get_double(key + "/" + std::to_string(i)));
      }
      return bounds;
    };
    patch_order = config.get_uint("/discretization/EoM_patch_order", 0);
    patch_radius = config.get_double("/discretization/EoM_patch_radius", 7.);
    patch_direction = read_array("/discretization/EoM_patch_direction");
    patch_skip_layers = config.get_uint("/discretization/EoM_patch_skip_layers", 1);
    if (config.contains("/discretization/EoM_patch_parity")) {
      const auto tree = static_cast<json::value>(config);
      const auto &rows = tree.at_pointer("/discretization/EoM_patch_parity").as_array();
      for (std::size_t c = 0; c < rows.size(); ++c) {
        std::vector<int> parity;
        for (std::size_t d = 0; d < rows[c].as_array().size(); ++d)
          parity.push_back(
              config.get_int("/discretization/EoM_patch_parity/" + std::to_string(c) + "/" + std::to_string(d)));
        patch_parity.push_back(std::move(parity));
      }
    }
    validate();
  }

  std::string EoMConfig::get_defaults()
  {
    return fmt::format(R"({{
  "discretization": {{
    "EoM_abs_tol": {},
    "EoM_max_iter": {},
    "EoM_armijo_coefficient": {},
    "EoM_max_backtracks": {},
    "EoM_patch_order": 0,
    "EoM_patch_radius": 7.0,
    "EoM_patch_direction": [],
    "EoM_patch_skip_layers": 1,
    "EoM_patch_parity": []
  }}
}})",
                       default_abs_tol, default_max_iter, default_armijo_coefficient, default_max_backtracks);
  }

  void EoMConfig::validate() const
  {
    if (!std::isfinite(abs_tol) || abs_tol < 0.)
      throw std::invalid_argument("EoM_abs_tol must be finite and non-negative.");
    if (!std::isfinite(armijo_coefficient) || !(armijo_coefficient > 0. && armijo_coefficient < 1.))
      throw std::invalid_argument("EoM_armijo_coefficient must be finite and strictly between zero and one.");
    if (max_backtracks == 0) throw std::invalid_argument("EoM_max_backtracks must be positive.");
    if (patch_order != 0 && patch_order != 2 && patch_order != 3)
      throw std::invalid_argument("EoM_patch_order must be 0 (off), 2, or 3.");
    if (!std::isfinite(patch_radius) || patch_radius <= 0.)
      throw std::invalid_argument("EoM_patch_radius must be finite and positive.");
    double direction_norm = 0.;
    for (const double entry : patch_direction) {
      if (!std::isfinite(entry)) throw std::invalid_argument("EoM_patch_direction must be finite.");
      direction_norm = std::hypot(direction_norm, entry);
    }
    if (!std::isfinite(direction_norm)) throw std::invalid_argument("EoM_patch_direction norm must be finite.");
    if (!patch_direction.empty() && direction_norm == 0. && patch_skip_layers != 0)
      throw std::invalid_argument("A centred EoM patch requires EoM_patch_skip_layers = 0.");
    for (const auto &parity : patch_parity)
      for (const int entry : parity)
        if (entry < -1 || entry > 1) throw std::invalid_argument("EoM_patch_parity entries must be -1, 0, or 1.");
  }
} // namespace DiFfRG::Config
