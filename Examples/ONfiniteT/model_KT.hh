#pragma once

#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "flows/flows.hh"

struct Parameters {
  Parameters(const ConfigTree &config)
  {
    try {
      Lambda = config.get_double("/physical/Lambda");
      N = config.get_double("/physical/N");
      T = config.get_double("/physical/T");
      lambda = config.get_double("/physical/lambda");
      m2 = config.get_double("/physical/m2");

      std::cout << "Parameters: " << Lambda << " " << N << " " << T << " " << lambda << " " << m2 << std::endl;
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }
  double Lambda, N, T, lambda, m2;
};

// One FE function m2 = dV/drho, no extractors, no variables.
using FEFunctionDesc = FEFunctionDescriptor<Scalar<"m2">>;
using Components = ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

/**
 * @brief O(N) finite-temperature LPA flow, driven through the Kurganov-Tadmor FV assembler.
 *
 * The dependent variable is m^2 = dV/drho (same as in Examples/ONfiniteT). The KT assembler
 * needs the flux split into a hyperbolic advection piece (depends on the field only) and a
 * parabolic diffusion piece (allowed to use the gradient). For the O(N) effective potential
 * the natural split is:
 *
 *   F_adv  = (N-1) * V_pion(m^2_Pi = m^2)                  pion (Goldstone) loop
 *   F_diff = V_sigma(m^2_Sigma = m^2 + 2*rho*dm^2/drho)    sigma (radial) loop
 *
 * Field output (m^2 as a function of rho) is written via the standard FE/VTK output.
 */
class ON_finiteT_KT : public def::AbstractModel<ON_finiteT_KT, Components>,
                      public def::fRG,
                      public def::RhoSymmetricLinearExtrapolationBoundaries<ON_finiteT_KT>,
                      public def::AD<ON_finiteT_KT>
{
public:
  static constexpr uint dim = 1;

protected:
  const Parameters prm;
  mutable ONFiniteTFlows flow_equations;

public:
  ON_finiteT_KT(const ConfigTree &config) : def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config)
  {
    flow_equations.set_k(Lambda);
    flow_equations.set_T(prm.T);
  }

  template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
  {
    const auto &rho = pos[0];
    values[idxf("m2")] = prm.m2 + prm.lambda / 2. * rho;
  }

  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * prm.Lambda;
    flow_equations.set_k(k);
  }

  /**
   * @brief Hyperbolic advection flux: (N-1) * pion-loop. Depends on m^2 only.
   */
  template <typename NT, typename Solution> void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F_i, const Point<dim> & /*x*/, const Solution &sol) const
  {
    const auto &fe_functions = get<0>(sol);
    const auto m2Pi = fe_functions[idxf("m2")];

    NT pion_loop;
    flow_equations.V_pion.get(pion_loop, k, prm.N, prm.T, m2Pi);
    F_i[idxf("m2")][0] = (prm.N - 1.) * pion_loop;
  }

  /**
   * @brief Parabolic/diffusion flux: sigma-loop. Uses m^2 and dm^2/drho via m^2_Sigma.
   */
  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F_i, const Point<dim> &x, const Solution &sol) const
  {
    const auto rho = x[0];
    const auto &fe_functions = get<0>(sol);
    const auto &fe_derivatives = get<1>(sol);

    const auto m2Sigma = fe_functions[idxf("m2")] + 2. * rho * fe_derivatives[idxf("m2")][0];

    NT sigma_loop;
    flow_equations.V_sigma.get(sigma_loop, k, prm.N, prm.T, m2Sigma);
    F_i[idxf("m2")][0] = sigma_loop;
  }
};
