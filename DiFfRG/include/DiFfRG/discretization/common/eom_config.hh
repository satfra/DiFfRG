#pragma once

#include <DiFfRG/common/config_tree.hh>

#include <string>
#include <vector>

namespace DiFfRG::Config
{
  // clang-format off
  /**
   * @brief Parameters of the EoM search (get_EoM_result) and its root-local vector-field Hessian
   *
   * | Key                                                        | Type   | Default | Meaning |
   * |------------------------------------------------------------|--------|---------|---------|
   * | `/discretization/EoM_abs_tol`                              | double | 1e-12   | componentwise EoM tolerance of the constrained root search; at least \f$100\,\epsilon_{\rm machine}\f$ is used |
   * | `/discretization/EoM_max_iter`                             | uint   | 100     | Newton steps per seed; 0: no search, return the mesh vertex with the smallest \f$\ell^1\f$ norm |
   * | `/discretization/EoM_patch_order`                          | uint   | 0       | FV local refinement: 0 off, 2 quadratic, 3 cubic; lowers degree on rank-deficient patches |
   * | `/discretization/EoM_patch_radius`                         | double | 7       | positive patch radius in local cell widths |
   * | `/discretization/EoM_patch_direction`                      | array  | []      | empty: upper side in every coordinate; nonzero vector: projected half-plane; zero vector: centred |
   * | `/discretization/EoM_patch_skip_layers`                    | uint   | 1       | exclusion buffer in each coordinate for the default upper patch, or projected widths for an explicit direction |
   * | `/discretization/EoM_patch_parity`                         | array  | []      | one row per state component, one entry per coordinate: -1 odd, +1 even about x_d=0, 0 unrestricted |
   * | `/discretization/EoM_armijo_coefficient`                   | double | 1e-4    | sufficient-decrease coefficient of the line search; in (0, 1) |
   * | `/discretization/EoM_max_backtracks`                       | uint   | 20      | maximum step halvings of the line search; > 0 |
   *
   * The constructors call validate(), which throws std::invalid_argument for values outside these ranges.
   */
  // clang-format on
  struct EoMConfig {
    inline static constexpr double default_abs_tol = 1e-12;
    inline static constexpr unsigned int default_max_iter = 100;
    inline static constexpr double default_armijo_coefficient = 1e-4;
    inline static constexpr unsigned int default_max_backtracks = 20;

    /// All defaults.
    EoMConfig() = default;
    /// Search parameters given directly.
    EoMConfig(double abs_tol, unsigned int max_iter, double armijo_coefficient = default_armijo_coefficient,
              unsigned int max_backtracks = default_max_backtracks);
    /// Read all keys of the table above.
    explicit EoMConfig(const ConfigTree &config);

    /// The defaults as a JSON parameter-file fragment.
    static std::string get_defaults();
    /// Throw std::invalid_argument if a value is out of range.
    void validate() const;

    double abs_tol = default_abs_tol;
    unsigned int max_iter = default_max_iter;
    double armijo_coefficient = default_armijo_coefficient;
    unsigned int max_backtracks = default_max_backtracks;
    /// Optional FV polynomial refinement: 0 off, 2 quadratic, 3 cubic, with rank-based order reduction.
    unsigned int patch_order = 0;
    /// Patch radius in local cell widths, chosen for upper-side cubic fits in 1D-3D away from domain edges.
    double patch_radius = 7.;
    /// Empty: upper side in every coordinate. Nonzero vector: projected half-plane; zero vector: centred.
    std::vector<double> patch_direction;
    /// Buffer in each coordinate for the default patch, or projected widths for an explicit direction.
    unsigned int patch_skip_layers = 1;
    /// Per state component and coordinate about x_d=0: -1 odd, +1 even, 0 unrestricted. Empty: no parity.
    std::vector<std::vector<int>> patch_parity;
  };
} // namespace DiFfRG::Config
