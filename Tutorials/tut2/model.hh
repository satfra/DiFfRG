#pragma once

#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>

using namespace DiFfRG;

struct Parameters {
  Parameters(const ConfigTree &value)
  {
    try {
      Nf = value.get_double("/physical/Nf");
      Nc = value.get_double("/physical/Nc");

      T = value.get_double("/physical/T");
      muq = value.get_double("/physical/muq");

      m2Phi = value.get_double("/physical/m2Phi");
      lambdaPhi = value.get_double("/physical/lambdaPhi");
      hPhi = value.get_double("/physical/hPhi");
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }
  double Nf, Nc, T, muq, m2Phi, lambdaPhi, hPhi;
};

using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
using Components = ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

namespace tF = fRG::TFLitimSpatial; // use Litim threshold functions

/**
 * @brief This class implements the numerical model for a Quark-Meson model.
 */
class Tut2 : public def::AbstractModel<Tut2, Components>,
             public def::fRG,                  // this handles the fRG time
             public def::FlowBoundaries<Tut2>, // use Inflow/Outflow boundaries
             public def::AD<Tut2>              // define all jacobians per AD
{
private:
  const Parameters prm;

public:
  static constexpr uint dim = 1;

  Tut2(const ConfigTree &json) : def::fRG(json.get_double("/physical/Lambda")), prm(json) {}

  template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
  {
    const auto rhoPhi = pos[0];
    values[idxf("u")] = prm.m2Phi + prm.lambdaPhi / 2. * rhoPhi;
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> &pos,
            const Solution &sol) const
  {
    const auto rhoPhi = pos[0];
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);

    const auto m2Quark = powr<2>(prm.hPhi) * rhoPhi / (prm.Nf);
    const auto m2Pion = fe_functions[idxf("u")];
    const auto m2Sigma = m2Pion + 2. * rhoPhi * fe_derivatives[idxf("u")][0];

    flux[idxf("u")][0] = fluxPion(m2Pion) + fluxSigma(m2Sigma) + fluxQuark(m2Quark);
  }

private:
  template <typename NT> NT fluxSigma(const NT m2Sigma) const
  {
    return (k4 * tF::lB<0, NT>(m2Sigma / k2, k, prm.T, 4)) / (4. * powr<2>(M_PI));
  }
  template <typename NT> NT fluxPion(const NT m2Pion) const
  {
    return (k4 * (powr<2>(prm.Nf) - 1.) * tF::lB<0, NT>(m2Pion / k2, k, prm.T, 4)) / (4. * powr<2>(M_PI));
  }
  template <typename NT> NT fluxQuark(const NT m2Quark) const
  {
    return -((k4 * prm.Nc * prm.Nf * tF::lF<0, NT>(m2Quark / k2, k, prm.T, prm.muq, 4)) / powr<2>(M_PI));
  }
};
