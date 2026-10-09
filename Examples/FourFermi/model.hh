#pragma once

#include "flows/flows.hh"
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/utils.hh>

#include <DiFfRG/common/external_data_interpolator.hh>

using namespace dealii;
using namespace DiFfRG;

struct Parameters {
  Parameters(const ConfigTree &json)
  {
    try {
      Lambda = json.get_double("/physical/Lambda");
      T = json.get_double("/physical/T");
      muq = json.get_double("/physical/muq");
      feedback = json.get_bool("/physical/feedback");
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }

  double Lambda, T, muq;
  bool feedback;
};

using VariableDesc = VariableDescriptor<FunctionND<"lambdas", 10>>;
using Components = ComponentDescriptor<FEFunctionDescriptor<>, VariableDesc, ExtractorDescriptor<>>;
constexpr auto idxv = VariableDesc{};

/**
 * @brief This class implements the numerical model for the quark-meson model at finite temperature and chemical
 * potential.
 */
class FourFermi : public def::AbstractModel<FourFermi, Components>,
                  public def::fRG,        // this handles the fRG time
                  public def::NoJacobians // define all jacobians per AD
{
  const Parameters prm;

  using Coordinates1D = LogarithmicCoordinates1D<float>;
  const Coordinates1D coordinates1D;

  const std::vector<float> grid1D;
  mutable TexLinearInterpolator1D<double, Coordinates1D> MQ;

  mutable FourFermiFlowEquations flow_equations;

  ExternalDataInterpolator external_data;

public:
  FourFermi(const ConfigTree &json, ReportPort log = {})
      : def::fRG(json.get_double("/physical/Lambda")), prm(json), coordinates1D(8, 0., 20., 5.), MQ(coordinates1D),
        flow_equations(json)
  {
    flow_equations.set_k(prm.Lambda);
    flow_equations.print_parameters(log);

    external_data =
        ExternalDataInterpolator({"external_data/kGeV.txt", "external_data/etaA.txt", "external_data/etaQ.txt",
                                  "external_data/gAqbq1.txt"},                          // k-scale, eta_A, etaQ, gAqbq1
                                 {[&](double x) { return -std::log(x / prm.Lambda); }}, // transform to RG time t
                                 ',', false, 0, 1);
  }

  static constexpr double input_scale = 3. / 4.;
  double external_etaA() const
  {
    const double ext_t = -std::log(input_scale * k / prm.Lambda);
    return external_data.value(ext_t, 1);
  }
  double external_etaQ() const
  {
    const double ext_t = -std::log(input_scale * k / prm.Lambda);
    return external_data.value(ext_t, 2);
  }
  double external_gAqbq1() const
  {
    const double ext_t = -std::log(input_scale * k / prm.Lambda);
    return external_data.value(ext_t, 3);
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const
  {
    for (uint i = 0; i < 10; ++i)
      values[i] = 0.;
  }

  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * Lambda;
    flow_equations.set_k(k);
  }

  template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &data) const
  {
    const auto &variables = get<"variables">(data);

    std::vector<double> lambdas(10, 0.);
    if (prm.feedback)
      for (uint i = 0; i < 10; ++i)
        lambdas[i] = variables[i];

    const auto gAqbq1 = external_gAqbq1();
    const auto etaA = external_etaA();
    const auto etaQ = external_etaQ();
    const auto p0f = M_PI * prm.T * std::exp(-k / (M_PI * prm.T));

    // set up arguments for the integrators
    const auto arguments = std::tie(p0f, prm.T, prm.muq, gAqbq1, etaA, etaQ, MQ, lambdas[0], lambdas[1], lambdas[2],
                                    lambdas[3], lambdas[4], lambdas[5], lambdas[6], lambdas[7], lambdas[8], lambdas[9]);

    std::apply(
        [&](const auto &...args) {
          residual[0] = flow_equations.lambda1_integrator.get<double>(k, args...);
          residual[1] = flow_equations.lambda2_integrator.get<double>(k, args...);
          residual[2] = flow_equations.lambda3_integrator.get<double>(k, args...);
          residual[3] = flow_equations.lambda4_integrator.get<double>(k, args...);
          residual[4] = flow_equations.lambda5_integrator.get<double>(k, args...);
          residual[5] = flow_equations.lambda6_integrator.get<double>(k, args...);
          residual[6] = flow_equations.lambda7_integrator.get<double>(k, args...);
          residual[7] = flow_equations.lambda8_integrator.get<double>(k, args...);
          residual[8] = flow_equations.lambda9_integrator.get<double>(k, args...);
          residual[9] = flow_equations.lambda10_integrator.get<double>(k, args...);
        },
        arguments);

    if (std::isnan(residual[0] + residual[1] + residual[2] + residual[3] + residual[4] + residual[5] + residual[6] +
                   residual[7] + residual[8] + residual[9])) {
      for (uint i = 0; i < 10; ++i)
        residual[i] = 0.;
      throw std::runtime_error("NaN in residual");
    }
    if (std::isnan(lambdas[0] + lambdas[1] + lambdas[2] + lambdas[3] + lambdas[4] + lambdas[5] + lambdas[6] +
                   lambdas[7] + lambdas[8] + lambdas[9])) {
      for (uint i = 0; i < 10; ++i)
        residual[i] = 0.;
      throw std::runtime_error("NaN in lambdas");
    }
    if ((lambdas[0] + lambdas[1] + lambdas[2] + lambdas[3] + lambdas[4] + lambdas[5] + lambdas[6] +
                   lambdas[7] + lambdas[8] + lambdas[9]) > 1e5) {
      for (uint i = 0; i < 10; ++i)
        residual[i] = 0.;
      throw std::runtime_error("NaN in lambdas");
    }
    if ((residual[0] + residual[1] + residual[2] + residual[3] + residual[4] + residual[5] + residual[6] + residual[7] +
         residual[8] + residual[9]) > 1e5) {
      for (uint i = 0; i < 10; ++i)
        residual[i] = 0.;
      throw std::runtime_error("residual too large");
    }
  }

  template <int dim, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim> &, const Solutions &sol) const
  {
    const auto &variables = get<"variables">(sol);

    auto &out_file = output.csv_file("data.csv");
    out_file.set_Lambda(Lambda);

    out_file.value("lambda1", variables[0]);
    out_file.value("lambda2", variables[1]);
    out_file.value("lambda3", variables[2]);
    out_file.value("lambda4", variables[3]);
    out_file.value("lambda5", variables[4]);
    out_file.value("lambda6", variables[5]);
    out_file.value("lambda7", variables[6]);
    out_file.value("lambda8", variables[7]);
    out_file.value("lambda9", variables[8]);
    out_file.value("lambda10", variables[9]);

    const auto lambda3 = variables[2];
    const auto lambda6 = variables[5];
    const auto lambda8 = variables[7];
    const auto lambda_csc = 29. * ((11 * lambda3) / 58. + (7 * lambda6) / 116. - (11 * lambda8) / 116.);
    out_file.value("lambda_csc", lambda_csc);
  }
};
