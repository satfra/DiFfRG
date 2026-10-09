#pragma once

#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "flows/flows.hh"

struct ParametersSigma {
  ParametersSigma(const ConfigTree &config)
  {
    try {
      Lambda = config.get_double("/physical/Lambda");
      N = config.get_double("/physical/N");
      T = config.get_double("/physical/T");
      lambda = config.get_double("/physical/lambda");
      m2 = config.get_double("/physical/m2");

      std::cout << "Parameters (sigma): " << Lambda << " " << N << " " << T << " " << lambda << " " << m2 << std::endl;
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }
  double Lambda, N, T, lambda, m2;
};

// One FE function u = dV/dsigma. No extractors, no variables.
using FEFunctionDesc_sigma = FEFunctionDescriptor<Scalar<"u">>;
using Components_sigma = ComponentDescriptor<FEFunctionDesc_sigma>;
constexpr auto idxf_sigma = FEFunctionDesc_sigma{};

/**
 * @brief O(N) finite-temperature LPA flow in sigma-coordinates, driven through the KT FV assembler.
 *
 * Dependent variable: u(sigma) = dV/dsigma.
 * Spatial coordinate: sigma in [0, sigma_max].
 *
 * KT advection / diffusion split (each mode separately):
 *   F_adv  = (N-1) * V_pion(m^2_Pi = u/sigma)     pion / Goldstone loop, depends on u only
 *   F_diff = V_sigma(m^2_Sigma = du/dsigma)       sigma / radial loop, depends on the gradient
 *
 * Unlike the rho-form, m^2_Sigma here does NOT pick up a geometric "2*rho" factor in front
 * of the gradient, so a per-cell checkerboard in u does not get geometrically amplified into
 * a large negative m^2_Sigma at the outer boundary. This formulation matches the in-tree KT
 * regression test (DiFfRG/tests/discretization/FV/regression/on_model_kt_regression.cc).
 */
class ON_finiteT_KT_sigma : public def::AbstractModel<ON_finiteT_KT_sigma, Components_sigma>,
                            public def::fRG,
                            public def::OriginOddLinearExtrapolationBoundaries<ON_finiteT_KT_sigma>,
                            public def::ConstrainOriginSupportPointToZero<"u", ON_finiteT_KT_sigma>,
                            public def::AD<ON_finiteT_KT_sigma>
{
public:
  using Components = Components_sigma;
  static constexpr uint dim = 1;
  // tolerance for the u/sigma origin guard, matching the regression test convention
  static constexpr double origin_tol = 1.0e-12;

protected:
  const ParametersSigma prm;
  mutable ONFiniteTFlows flow_equations;
  double cell_width;

  // Initial potential V_init(sigma) = (m2/2) * sigma^2 + (lambda/8) * sigma^4
  // ==> u_init(sigma) = dV/dsigma = m2 * sigma + (lambda/2) * sigma^3
  // The cell-averaged u over a cell of width dx centred at sigma is
  // (V(sigma+dx/2) - V(sigma-dx/2)) / dx.
  double V_init(double sigma) const
  {
    const double s2 = sigma * sigma;
    return 0.5 * prm.m2 * s2 + (prm.lambda / 8.0) * s2 * s2;
  }

public:
  ON_finiteT_KT_sigma(const ConfigTree &config) : def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config)
  {
    flow_equations.set_k(Lambda);
    flow_equations.set_T(prm.T);
    // Parse cell width from the x_grid string "lo:dx:hi"; assume uniform spacing.
    // Fallback: leave at zero and let initial_condition fall back to point evaluation.
    const std::string grid = config.get_string("/discretization/grid/x_grid");
    const auto first_colon = grid.find(':');
    const auto second_colon = grid.find(':', first_colon + 1);
    if (first_colon != std::string::npos && second_colon != std::string::npos) {
      cell_width = std::stod(grid.substr(first_colon + 1, second_colon - first_colon - 1));
    } else {
      cell_width = 0.0;
    }
  }

  template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
  {
    const double sigma = pos[0];
    if (cell_width > 0.0) {
      // exact cell average of u_init over [sigma - dx/2, sigma + dx/2]
      const double V_right = V_init(sigma + 0.5 * cell_width);
      const double V_left = V_init(sigma - 0.5 * cell_width);
      values[idxf_sigma("u")] = (V_right - V_left) / cell_width;
    } else {
      // point evaluation: u_init(sigma) = m2 * sigma + (lambda/2) * sigma^3
      values[idxf_sigma("u")] = prm.m2 * sigma + 0.5 * prm.lambda * sigma * sigma * sigma;
    }
  }

  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * prm.Lambda;
    flow_equations.set_k(k);
  }

  /**
   * @brief KT hyperbolic advection flux: (N-1) * V_pion(u/sigma).
   *
   * Depends on u only (no gradient). The u/sigma factor is guarded near sigma = 0
   * to avoid the 0/0 indeterminacy at the origin (mirrors on_model_kt_regression.cc:267-268).
   */
  template <typename NT, typename Solution> void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F_i, const Point<dim> &x, const Solution &sol) const
  {
    const auto &fe_functions = get<0>(sol);
    const NT u = fe_functions[idxf_sigma("u")];
    const double sigma = x[0];
    const NT m2Pi = (std::abs(sigma) < origin_tol) ? NT(0.0) : (u / NT(sigma));

    NT pion_loop;
    flow_equations.V_pion.get(pion_loop, k, prm.N, prm.T, m2Pi);
    F_i[idxf_sigma("u")][0] = (prm.N - 1.) * pion_loop;
  }

  /**
   * @brief KT parabolic / diffusion flux: V_sigma(du/dsigma).
   *
   * Depends only on the gradient — no geometric weighting, in contrast to the rho-form.
   */
  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F_i, const Point<dim> & /*x*/, const Solution &sol) const
  {
    const auto &fe_derivatives = get<1>(sol);
    const NT m2Sigma = fe_derivatives[idxf_sigma("u")][0];

    NT sigma_loop;
    flow_equations.V_sigma.get(sigma_loop, k, prm.N, prm.T, m2Sigma);
    F_i[idxf_sigma("u")][0] = sigma_loop;
  }
};
