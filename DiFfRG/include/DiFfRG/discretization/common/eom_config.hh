#pragma once

#include <DiFfRG/common/config_tree.hh>

#include <string>

namespace DiFfRG::Config
{
  // clang-format off
  /**
   * @brief Parameters of the EoM search (get_EoM_point_with_potential) and of the raw-potential reconstruction
   * (reconstruct_raw_potential).
   *
   * | Key                                                        | Type   | Default | Meaning |
   * |------------------------------------------------------------|--------|---------|---------|
   * | `/discretization/EoM_abs_tol`                              | double | 1e-12   | gradient tolerance of the minimum search; at least \f$100\,\epsilon_{\rm machine}\f$ is used |
   * | `/discretization/EoM_max_iter`                             | uint   | 100     | Newton steps per seed of the minimum search; 0: no search, the EoM point is the mesh vertex with the smallest \f$\ell^1\f$ norm |
   * | `/discretization/EoM_smoothing_length`                     | double | -1      | smoothing length \f$\ell\f$ of the potential reconstructions; -1: twice the smallest cell width normal to a face; otherwise a field-space length \f$\ge 0\f$ |
   * | `/discretization/EoM_bound_tolerance`                      | double | 1e-12   | distance (unit-cell coordinates) from a cell face below which a search coordinate is held at the face |
   * | `/discretization/EoM_armijo_coefficient`                   | double | 1e-4    | sufficient-decrease coefficient of the line search; in (0, 1) |
   * | `/discretization/EoM_max_backtracks`                       | uint   | 20      | maximum step halvings of the line search; > 0 |
   * | `/discretization/raw_potential_order`                      | uint   | 2       | polynomial order of the raw potential, 2 or 3 |
   * | `/discretization/raw_potential_recover_mass_hessian`       | bool   | false   | DG0/FV only: recover a continuous Hessian field for the masses |
   * | `/discretization/raw_potential_mass_hessian_jump_threshold`| double | -1      | with the above: relative Hessian jump across a face above which the face is treated as a kink and one-sided values are used near it; -1 disables |
   *
   * The constructors call validate(), which throws std::invalid_argument for values outside these ranges.
   */
  // clang-format on
  struct EoMConfig {
    inline static constexpr double default_abs_tol = 1e-12;
    inline static constexpr unsigned int default_max_iter = 100;
    inline static constexpr double default_smoothing_length = -1.;
    inline static constexpr double default_bound_tolerance = 1e-12;
    inline static constexpr double default_armijo_coefficient = 1e-4;
    inline static constexpr unsigned int default_max_backtracks = 20;
    inline static constexpr unsigned int default_raw_potential_order = 2;
    inline static constexpr bool default_raw_potential_recover_mass_hessian = false;
    inline static constexpr double default_raw_potential_mass_hessian_jump_threshold = -1.;

    /// All defaults.
    EoMConfig() = default;
    /// The search parameters given directly; the raw-potential parameters at their defaults.
    EoMConfig(double abs_tol, unsigned int max_iter, double smoothing_length,
              double bound_tolerance = default_bound_tolerance, double armijo_coefficient = default_armijo_coefficient,
              unsigned int max_backtracks = default_max_backtracks);
    /// Read all keys of the table above.
    explicit EoMConfig(const ConfigTree &config);

    /// The defaults as a JSON parameter-file fragment.
    static std::string get_defaults();
    /// Throw std::invalid_argument if a value is out of range.
    void validate() const;

    double abs_tol = default_abs_tol;
    unsigned int max_iter = default_max_iter;
    double smoothing_length = default_smoothing_length;
    double bound_tolerance = default_bound_tolerance;
    double armijo_coefficient = default_armijo_coefficient;
    unsigned int max_backtracks = default_max_backtracks;
    unsigned int raw_potential_order = default_raw_potential_order;
    bool raw_potential_recover_mass_hessian = default_raw_potential_recover_mass_hessian;
    double raw_potential_mass_hessian_jump_threshold = default_raw_potential_mass_hessian_jump_threshold;
  };
} // namespace DiFfRG::Config
