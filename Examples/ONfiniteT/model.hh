#pragma once

#include "flows/flows.hh"
#include <DiFfRG/model/model.hh>

using namespace dealii;
using namespace DiFfRG;

struct Parameters {
  Parameters(const JSONValue &value)
  {
    try {
      Lambda = value.get_double("/physical/Lambda");
      N = value.get_double("/physical/N");
      T = value.get_double("/physical/T");
      lambda = value.get_double("/physical/lambda");
      m2 = value.get_double("/physical/m2");

      std::cout << "Parameters: " << Lambda << " " << N << " " << T << " " << lambda << " " << m2 << std::endl;
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }
  double Lambda, N, T, lambda, m2;
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
  mutable ON_finiteTFlowEquations flow_equations;

  // ----------------------------------------------------------------------------------------------------
  // initialization
  // ----------------------------------------------------------------------------------------------------
public:
  ON_finiteT(const JSONValue &json) : def::fRG(json.get_double("/physical/Lambda")), prm(json), flow_equations(json)
  {
    flow_equations.set_k(Lambda);
    flow_equations.print_parameters("log");
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

  // ----------------------------------------------------------------------------------------------------
  // instationary equations
  // ----------------------------------------------------------------------------------------------------
public:
  /**
   * @brief The flux function is integrated against derivatives of test functions and also gives a boundary contribution
   */
  template <typename NT> 
  void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const auto &x, const auto &sol) const
  {
    const auto rho = x[0];
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);

    const auto m2Pi = fe_functions[idxf("m2")];
    const auto m2Sigma = fe_functions[idxf("m2")] + 2. * rho * fe_derivatives[idxf("m2")][0];

    flux[idxf("m2")][0] = flow_equations.V_integrator.get<NT>(k, prm.N, prm.T, rho, m2Pi, m2Sigma);
  }

  template <int dim, typename NumberType, typename Solution>
  void cell_indicator(NumberType & indicator, const Point<dim> & /*p*/, const Solution & sol) const
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

    auto &out_file = output.csv_file("data.csv");
    out_file.set_Lambda(Lambda);

    out_file.value("sigma [GeV]", sigma);
    out_file.value("m^2_{pi} [GeV^2]", m2Pi);
    out_file.value("m^2_{sigma} [GeV^2]", m2Sigma);
    out_file.value("m_{pi} [GeV]", mPi);
    out_file.value("m_{sigma} [GeV]", mSigma);
  }
};
