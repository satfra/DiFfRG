#pragma once

#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "flows/flows.hh"
struct Parameters {
  Parameters(const ConfigTree &value)
  {
    try {
      Lambda = value.get_double("/physical/Lambda");
      N = value.get_double("/physical/N");
      T = value.get_double("/physical/T");
      lambda2 = value.get_double("/physical/lambda2");
      lambda4 = value.get_double("/physical/lambda4");
      lambda6 = value.get_double("/physical/lambda6");
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }
  double Lambda, N, T, lambda2, lambda4, lambda6;
};

// As for components, we have one FE function (u) and no extractors or variables.
using FEFunctionDesc = FEFunctionDescriptor<Scalar<"m2">>;
using Components = ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

/**
 * @brief This class implements the numerical model for the quark-meson model at finite temperature and chemical potential.
 */
class ON_finiteT : public def::AbstractModel<ON_finiteT, Components>,
                   public def::fRG,                        // this handles the fRG time
                   public def::LLFFlux<ON_finiteT>,        // use a LLF numflux
                   public def::FlowBoundaries<ON_finiteT>, // use Inflow/Outflow boundaries
                   public def::AD<ON_finiteT>              // define all jacobians per AD
{
  // ----------------------------------------------------------------------------------------------------
  // variables and typedefs
  // ----------------------------------------------------------------------------------------------------
public:
  static constexpr uint dim = 1;

protected:
  const Parameters prm;
  mutable ONFiniteTFlows flow_equations;

  // ----------------------------------------------------------------------------------------------------
  // initialization
  // ----------------------------------------------------------------------------------------------------
public:
  ON_finiteT(const ConfigTree &json) : def::fRG(json.get_double("/physical/Lambda")), prm(json), flow_equations(json)
  {
    flow_equations.set_k(Lambda);
    flow_equations.set_T(prm.T);
  }

  template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
  {
    const auto &rho = pos[0];
    values[idxf("m2")] = prm.lambda2 + prm.lambda4 * rho + prm.lambda6 * powr<2>(rho);
  }

  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * prm.Lambda;
    flow_equations.set_k(k);
  }

  // ----------------------------------------------------------------------------------------------------
  // instationary equations
  // ----------------------------------------------------------------------------------------------------
public:
  /**
   * @brief The flux function is integrated against derivatives of test functions and also gives a boundary contribution
   */
  template <typename NT, typename Solution> void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> &x, const Solution &sol) const
  {
    const auto rho = x[0];
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);

    const auto m2Pi = fe_functions[idxf("m2")];
    const auto m2Sigma = fe_functions[idxf("m2")] + 2. * rho * fe_derivatives[idxf("m2")][0];

    flow_equations.V.get(flux[idxf("m2")][0], k, prm.N, prm.T, m2Pi, m2Sigma);
  }

  template <int dim, typename NumberType, typename Solution> void cell_indicator(NumberType &indicator, const Point<dim> & /*p*/, const Solution &sol) const
  {
    const auto &fe_hessians = get<"fe_hessians">(sol);

    indicator = fe_hessians[idxf("m2")][0][0];
  }

  template <int dim, typename DataOut, typename Solutions> void readouts(DataOut &output, const Point<dim> &x, const Solutions &sol) const
  {
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);

    const double rho = x[0];
    const double sigma = std::sqrt(2. * rho);

    const double m2Pi = fe_functions[idxf("m2")];
    const double m2Sigma = fe_functions[idxf("m2")] + 2. * rho * fe_derivatives[idxf("m2")][0];

    const double mPi = m2Pi > 0. ? std::sqrt(m2Pi) : 0.;
    const double mSigma = m2Sigma > 0. ? std::sqrt(m2Sigma) : 0.;

    auto out_file = output.table("data.csv");
    out_file.set_Lambda(Lambda);

    out_file.value("sigma [GeV]", sigma);
    out_file.value("m^2_{pi} [GeV^2]", m2Pi);
    out_file.value("m^2_{sigma} [GeV^2]", m2Sigma);
    out_file.value("m_{pi} [GeV]", mPi);
    out_file.value("m_{sigma} [GeV]", mSigma);
  }
};
