#pragma once

#include "flows/flows.hh"
#include <DiFfRG/common/interpolation.hh>
#include <DiFfRG/common/minimization.hh>
#include <DiFfRG/model/model.hh>
#include <cmath>
#include <spdlog/spdlog.h>

using namespace dealii;
using namespace DiFfRG;

struct Parameters {
  Parameters(const JSONValue &value)
  {
    try {
      Lambda = value.get_double("/physical/Lambda");
      Nc = value.get_double("/physical/Nc");
      Nf = value.get_double("/physical/Nf");
      T = value.get_double("/physical/T");
      muq = value.get_double("/physical/muq");

      m2Phi = value.get_double("/physical/m2Phi");
      lambdaPhi = value.get_double("/physical/lambdaPhi");
      hPhi = value.get_double("/physical/hPhi");

      cSigma = value.get_double("/physical/cSigma");
    } catch (std::exception &e) {
      std::cout << "Error while reading parameters: " << e.what() << std::endl;
      throw;
    }
  }
  double Lambda, Nc, Nf, T, muq;
  double m2Phi, lambdaPhi, hPhi, cSigma;
};

using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
using VariableDesc = VariableDescriptor<Scalar<"ZQ">, Scalar<"ZPhi">, Scalar<"hPhi">>;
using ExtractorDesc = ExtractorDescriptor<Scalar<"etaQ">, Scalar<"etaPhi">, Scalar<"rhoPhi">, Scalar<"d1V">, Scalar<"d2V">, Scalar<"d3V">>;
using LDG1FunctionDesc = FEFunctionDescriptor<Scalar<"du_r">, Scalar<"du_l">>;
using LDG2FunctionDesc = FEFunctionDescriptor<Scalar<"ddu_rr">, Scalar<"ddu_ll">, Scalar<"ddu_rl">, Scalar<"ddu_lr">>;
using Components = ComponentDescriptor<FEFunctionDesc, VariableDesc, ExtractorDesc>;

constexpr auto idxf = FEFunctionDesc{};
constexpr auto idxv = VariableDesc{};
constexpr auto idxe = ExtractorDesc{};
constexpr auto idxl = LDG1FunctionDesc{};

template <typename Model> using LDGFluxes = typename def::LDGUpDownFluxes<Model, def::UpDownFlux<def::FlowDirections<0, 0>, def::UpDown<def::from_right, def::from_left>>>;

/**
 * @brief This class implements the numerical model for the quark-meson model at finite temperature and chemical potential.
 */
class QuarkMesonLPAprime : public def::AbstractModel<QuarkMesonLPAprime, Components>,
                           public def::fRG,                                // this handles the fRG time
                           public def::LLFFlux<QuarkMesonLPAprime>,        // use a LLF numflux
                                                                           //   public LDGFluxes<QuarkMesonLPAprime>,           // use LDG fluxes
                           public def::FlowBoundaries<QuarkMesonLPAprime>, // use Inflow/Outflow boundaries
                           public def::FE_AD<QuarkMesonLPAprime>           // define all jacobians per AD
{
  // ----------------------------------------------------------------------------------------------------
  // variables and typedefs
  // ----------------------------------------------------------------------------------------------------
public:
  static constexpr uint dim = 1;
  enum { mesonic };

protected:
  const Parameters prm;

  mutable QuarkMesonFlowEquations flow_equations;

  mutable double cSigma;

  // ----------------------------------------------------------------------------------------------------
  // initialization
  // ----------------------------------------------------------------------------------------------------
public:
  QuarkMesonLPAprime(const JSONValue &json) : def::fRG(json.get_double("/physical/Lambda")), prm(json), flow_equations(json)
  {
    flow_equations.set_k(Lambda);
    flow_equations.print_parameters("log");

    // this->components().add_dependency(1, 0, 0, 0);
    // this->components().add_dependency(1, 1, 0, 0);
    // this->components().set_jacobian_constant(1, 0);
  }

  template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
  {
    const auto rhoPhi = pos[mesonic];
    values[idxf("u")] = prm.m2Phi + prm.lambdaPhi / 2. * rhoPhi;
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const
  {
    values[idxv("ZQ")] = 1.;
    values[idxv("ZPhi")] = 1.;

    values[idxv("hPhi")] = (prm.hPhi);
    cSigma = prm.cSigma;
  }

  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * prm.Lambda;
    flow_equations.set_k(k);
  }

  template <uint submodel, typename NT, typename Variables>
  void ldg_flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(submodel)> &flux, const Point<dim> & /*pos*/, const Variables &variables) const
  {
    if constexpr (submodel == 1) {
      flux[idxl("du_r")][mesonic] = variables[idxf("u")]; // du_r = d(u)/d(mesonic)
      flux[idxl("du_l")][mesonic] = variables[idxf("u")]; // du_r = d(u)/d(mesonic)
    } else if constexpr (submodel == 2) {
      flux[idxl("du_r")][mesonic] = variables[idxf("u")]; // du_r = d(u)/d(mesonic)
      flux[idxl("du_l")][mesonic] = variables[idxf("u")]; // du_r = d(u)/d(mesonic)
    }
  }

  // ----------------------------------------------------------------------------------------------------
  // instationary equations
  // ----------------------------------------------------------------------------------------------------

  template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &sol) const
  {
    const auto &variables = get<"variables">(sol);
    const auto &extractors = get<"extractors">(sol);

    const auto rhoPhi = extractors[idxe("rhoPhi")];

    const auto ZQ = variables[idxv("ZQ")];
    const auto ZPhi = variables[idxv("ZPhi")];

    const auto etaQ = extractors[idxe("etaQ")];
    const auto etaPhi = extractors[idxe("etaPhi")];

    const auto hPhi = (variables[idxv("hPhi")]);
    const auto d1V = extractors[idxe("d1V")];
    const auto d2V = extractors[idxe("d2V")];
    const auto d3V = extractors[idxe("d3V")];

    const double p0f = std::max(1e-3, M_PI * prm.T * std::exp(-k / (M_PI * prm.T)));

    const auto arguments = std::tie(prm.Nc, prm.Nf, prm.T, prm.muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);

    residual[idxv("ZQ")] = -etaQ * ZQ;
    residual[idxv("ZPhi")] = -etaPhi * ZPhi;

    std::apply(
        [&](auto &&...args) {
          std::vector<std::future<double>> hPhi_futures;

          hPhi_futures.emplace_back(std::move(flow_equations.hPhi0_integrator.request<double>(k, p0f, 1e-5, args...)));

          residual[idxv("hPhi")] = hPhi_futures[0].get();
        },
        arguments);

    cSigma = std::pow(ZPhi, -0.5) * prm.cSigma;
  }

  template <typename NT, typename Solution> void extract(std::array<NT, Components::count_extractors()> &extractors, const Point<dim> &x, const Solution &sol) const
  {
    const auto rhoPhi = x[mesonic];

    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);
    const auto &fe_hessians = get<"fe_hessians">(sol);
    const auto &variables = get<"variables">(sol);

    auto etaQ = extractors[idxe("etaQ")];
    auto etaPhi = extractors[idxe("etaPhi")];

    const double hPhi = variables[idxv("hPhi")];
    const double d1V = fe_functions[idxf("u")] < 0 ? 0. : fe_functions[idxf("u")];
    const double d2V = fe_derivatives[idxf("u")][mesonic];
    const double d3V = fe_hessians[idxf("u")][mesonic][mesonic];

    extractors[idxe("rhoPhi")] = rhoPhi;
    extractors[idxe("d1V")] = d1V;
    extractors[idxe("d2V")] = d2V;
    extractors[idxe("d3V")] = d3V;

    const double p0f = std::max(1e-3, M_PI * prm.T * std::exp(-k / (M_PI * prm.T)));

    const auto arguments = std::tie(prm.Nc, prm.Nf, prm.T, prm.muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);

    // iterate the etas until convergence
    {
      double eps = 0.;
      uint it = 0;
      do {
        // etaA
        double new_etaQ = 0.;
        double new_etaPhi = 0.;

        std::apply(
            [&](auto &&...args) {
              new_etaQ = flow_equations.etaQ_integrator.get<double>(k, p0f, 1e-4, args...);
              new_etaPhi = flow_equations.etaPhi_integrator.get<double>(k, p0f, 1e-4, args...);
            },
            arguments);

        eps = 0.0;
        eps = std::max(eps, std::abs(double((new_etaQ - etaQ) / new_etaQ)));
        eps = std::max(eps, std::abs(double((new_etaPhi - etaPhi) / new_etaPhi)));

        etaQ = new_etaQ;
        etaPhi = new_etaPhi;

        it++;
      } while (eps > 1e-5 && it < 8);

      if (etaPhi > 4.)
        etaPhi = 4.;

      extractors[idxe("etaQ")] = etaQ;
      extractors[idxe("etaPhi")] = etaPhi;
    }
  }

  template <typename NT, typename Solution> void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> &x, const Solution &sol) const
  {
    const auto rhoPhi = x[mesonic];

    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);
    const auto &variables = get<"variables">(sol);
    const auto &extractors = get<"extractors">(sol);

    const auto etaQ = extractors[idxe("etaQ")];
    const auto etaPhi = extractors[idxe("etaPhi")];

    const auto hPhi = (variables[idxv("hPhi")]);
    const auto d1V = fe_functions[idxf("u")];
    const auto d2V = fe_derivatives[idxf("u")][0];
    const auto d3V = extractors[idxe("d3V")];

    const double p0f = std::max(1e-3, M_PI * prm.T * std::exp(-k / (M_PI * prm.T)));

    const auto arguments = std::tie(prm.Nc, prm.Nf, prm.T, prm.muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);

    std::apply([&](auto &&...args) { flux[idxf("u")][mesonic] = flow_equations.V_integrator.get<NT>(k, p0f, 0., args...); }, arguments);

    if (!std::isfinite((double)flux[idxf("u")][mesonic])) {
      spdlog::error("flux is not finite");
      spdlog::error("k = {}", k);
      spdlog::error("p0f = {}", p0f);
      spdlog::error("etaQ = {}", etaQ);
      spdlog::error("etaPhi = {}", etaPhi);
      spdlog::error("hPhi = {}", hPhi);
      spdlog::error("d1V = {}",(double) d1V);
      spdlog::error("d2V = {}",(double) d2V);
      spdlog::error("d3V = {}",(double) d3V);
      spdlog::error("rhoPhi = {}", rhoPhi);
    }
  }

  template <int dim, typename NT, typename Solutions, size_t n_fe_functions> void source(std::array<NT, n_fe_functions> &s_i, const Point<dim> &x, const Solutions &sol) const
  {
    const auto rhoPhi = x[mesonic];

    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);
    const auto &extractors = get<"extractors">(sol);

    const auto etaPhi = extractors[idxe("etaPhi")];
    const auto d1V = fe_functions[idxf("u")];
    const auto d2V = fe_derivatives[idxf("u")][0];

    s_i[idxf("u")] = etaPhi * (d1V + d2V * rhoPhi);
  }

  template <int dim, typename NumberType, typename Vector, typename Vector_dot>
  void mass(std::array<NumberType, Components::count_fe_functions(0)> &m_i, const Point<dim> &, const Vector &u_i, const Vector_dot &dt_u_i) const
  {
    m_i[idxf("u")] = dt_u_i[idxf("u")];
  }

  template <int dim, typename Vector> std::array<double, 1> EoM(const Point<dim> &x, const Vector &u_i) const
  {
    const double sigma = std::sqrt(2. * x[mesonic]);
    const auto d1V = u_i[idxf("u")];
    return {{d1V - cSigma / (sigma + 1e-14)}};
  }

  mutable double last_EoM = 0.;
  mutable bool lock_EoM = false;

  template <int dim, typename Vector> Point<dim> EoM_postprocess(const Point<dim> &EoM, const Vector &) const
  {
    //std::cout << "EoM change: " << last_EoM - EoM[0] << std::endl;
    if (lock_EoM || last_EoM - EoM[0] > 5e-4) {
      lock_EoM = true;
      return Point<dim>(last_EoM);
    }
    else
      last_EoM = EoM[0];
    return EoM;
  }

  template <int dim, typename DataOut, typename Solutions> void readouts(DataOut &output, const Point<dim> &x, const Solutions &sol, const std::string filename, const int) const
  {
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);
    const auto &variables = get<"variables">(sol);
    const auto &extractors = get<"extractors">(sol);

    const double rhoPhi = x[mesonic];
    const double sigma = std::sqrt(2. * rhoPhi);

    const double ZQ = variables[idxv("ZQ")];
    const double ZPhi = variables[idxv("ZPhi")];

    const double etaQ = extractors[idxe("etaQ")];
    const double etaPhi = extractors[idxe("etaPhi")];

    const auto hPhi = (variables[idxv("hPhi")]);
    const auto d2V = fe_derivatives[idxf("u")][mesonic];
    const auto m2Pion = fe_functions[idxf("u")];
    const auto m2Sigma = m2Pion + 2. * rhoPhi * d2V;

    const double mQuark = (sigma * hPhi / std::sqrt(2. * 2.));
    const double mPion = m2Pion > 0. ? std::sqrt(m2Pion) : 0.;
    const double mSigma = m2Sigma > 0. ? std::sqrt(m2Sigma) : 0.;

    auto &out_file = output.csv_file(filename);
    out_file.set_Lambda(Lambda);

    out_file.value("sigma [GeV]", sigma);

    out_file.value("ZQ", ZQ);
    out_file.value("ZPhi", ZPhi);

    out_file.value("etaQ", etaQ);
    out_file.value("etaPhi", etaPhi);

    out_file.value("hPhi", hPhi);

    out_file.value("m_{pi} [GeV]", mPion);
    out_file.value("m_{sigma} [GeV]", mSigma);
    out_file.value("m_q [GeV]", mQuark);
  }

  template <typename FUN, typename DataOut> void readouts_multiple(FUN &helper, DataOut &) const
  {
    // chiral EoM
    helper(
        [&](const auto &, const auto &u_i) {
          const auto d1V = u_i[idxf("u")];
          return std::array<double, 1>{{d1V}};
        },
        [&](auto &output, const auto &x, const auto &sol) { this->readouts(output, x, sol, "data_chiral.csv", 0); });
    // physical EoM
    helper(
        [&](const auto &x, const auto &u_i) {
          const auto sigma = std::sqrt(2. * x[mesonic]);
          const auto d1V = u_i[idxf("u")];
          return std::array<double, 1>{{d1V - cSigma / (sigma + 1e-14)}};
        },
        [&](auto &output, const auto &x, const auto &sol) { this->readouts(output, x, sol, "data_running_EoM.csv", 1); });
  }
};
