#pragma once

#include <DiFfRG/model/model.hh>

#include "flows/flows.hh"

#include <cmath>

using namespace DiFfRG;

struct Parameters {
  Parameters(const ConfigTree &config)
      : Lambda(config.get_double("/physical/Lambda")), Nc(config.get_double("/physical/Nc")),
        Nf(config.get_double("/physical/Nf")), T(config.get_double("/physical/T")),
        muq(config.get_double("/physical/muq")), m2Phi(config.get_double("/physical/m2Phi")),
        lambdaPhi(config.get_double("/physical/lambdaPhi")), hPhi(config.get_double("/physical/hPhi")),
        cSigma(config.get_double("/physical/cSigma"))
  {
  }
  double Lambda, Nc, Nf, T, muq;
  double m2Phi, lambdaPhi, hPhi, cSigma;
};

// The potential lives on the field-space grid; the meson wave-function renormalization and the Yukawa coupling are
// plain numbers flowing alongside it; the anomalous dimensions and the potential derivatives at the minimum are
// read off the FE solution once per evaluation.
using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
using VariableDesc = VariableDescriptor<Scalar<"ZPhi">, Scalar<"hPhi">>;
using ExtractorDesc = ExtractorDescriptor<Scalar<"etaQ">, Scalar<"etaPhi">, Scalar<"rhoPhi">, Scalar<"d1V">,
                                          Scalar<"d2V">, Scalar<"d3V">>;
using Components = ComponentDescriptor<FEFunctionDesc, VariableDesc, ExtractorDesc>;

constexpr auto idxf = FEFunctionDesc{};
constexpr auto idxv = VariableDesc{};
constexpr auto idxe = ExtractorDesc{};

/**
 * @brief Quark-meson model in LPA' at finite temperature and quark chemical potential.
 *
 * All fields are renormalized, Z = 1, so the wave-function renormalizations enter only through the anomalous
 * dimensions eta = -d_t Z / Z: in the regulator insertions d_t R - eta R and in the rescaling of u and hPhi. The
 * only exception is the explicit symmetry breaking, which is fixed for the bare field; ZPhi is carried for it.
 *
 * u = dV/drho is an FE function; ZPhi and hPhi are variables; etaQ, etaPhi and the derivatives of V at the minimum
 * are extractors. The flows come from QM.wl.
 */
class QuarkMesonLPAprime : public def::AbstractModel<QuarkMesonLPAprime, Components>,
                           public def::fRG,                                // this handles the fRG time
                           public def::FlowBoundaries<QuarkMesonLPAprime>, // use Inflow/Outflow boundaries
                           public def::FE_AD<QuarkMesonLPAprime> // jacobians per AD, extractors held fixed
{
public:
  static constexpr uint dim = 1;

  QuarkMesonLPAprime(const ConfigTree &config)
      : def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config), cSigma(prm.cSigma)
  {
    flow_equations.set_k(Lambda);
  }

  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * prm.Lambda;
    flow_equations.set_k(k);
  }

  template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
  {
    values[idxf("u")] = prm.m2Phi + prm.lambdaPhi / 2. * pos[0];
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const
  {
    values[idxv("ZPhi")] = 1.;
    values[idxv("hPhi")] = prm.hPhi;
  }

  /**
   * @brief Read off the potential at the EoM and solve for the anomalous dimensions there.
   *
   * etaQ and etaPhi appear on both sides of their flows (through the regulator insertion d_t R - eta R), so they
   * are found by fixed-point iteration, starting from the values of the previous evaluation.
   */
  template <typename NT, typename Solution>
  void extract(std::array<NT, Components::count_extractors()> &extractors, const Point<dim> &x,
               const Solution &sol) const
  {
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);
    const auto &fe_hessians = get<"fe_hessians">(sol);
    const auto &variables = get<"variables">(sol);

    const double rhoPhi = x[0];
    const double hPhi = variables[idxv("hPhi")];
    // Clipped at zero: a negative pion mass would put a pole into the meson propagators of the eta loops.
    const double d1V = std::max(0., double(fe_functions[idxf("u")]));
    const double d2V = fe_derivatives[idxf("u")][0];
    const double d3V = fe_hessians[idxf("u")][0][0];

    extractors[idxe("rhoPhi")] = rhoPhi;
    extractors[idxe("d1V")] = d1V;
    extractors[idxe("d2V")] = d2V;
    extractors[idxe("d3V")] = d3V;

    const double p0f = fermionic_p0();
    double etaQ = extractors[idxe("etaQ")];
    double etaPhi = extractors[idxe("etaPhi")];
    const auto arguments = device::tie(k, p_ext, p0f, prm.Nf, prm.Nc, prm.T, prm.muq, etaQ, etaPhi, hPhi, d1V,
                                       d2V, d3V, rhoPhi);
    double eps = 0.;
    uint it = 0;
    do {
      double new_etaQ = 0., new_etaPhi = 0.;
      flow_equations.etaQ.get(new_etaQ, arguments);
      flow_equations.etaPhi.get(new_etaPhi, arguments);

      eps = std::max(std::abs((new_etaQ - etaQ) / new_etaQ), std::abs((new_etaPhi - etaPhi) / new_etaPhi));
      etaQ = new_etaQ;
      etaPhi = new_etaPhi;
    } while (eps > 1e-5 && ++it < 8);

    extractors[idxe("etaQ")] = etaQ;
    // A safety cap: keeps a runaway of the fixed-point iteration from propagating into the flow.
    extractors[idxe("etaPhi")] = std::min(etaPhi, 4.);
  }

  /// The flows of the variables: d_t ZPhi = -etaPhi ZPhi, and the Yukawa coupling, including its rescaling.
  template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &sol) const
  {
    const auto &variables = get<"variables">(sol);
    const auto &extractors = get<"extractors">(sol);

    const auto ZPhi = variables[idxv("ZPhi")];
    const auto etaQ = extractors[idxe("etaQ")];
    const auto etaPhi = extractors[idxe("etaPhi")];

    const double hPhi = variables[idxv("hPhi")];
    const double d1V = extractors[idxe("d1V")];
    const double d2V = extractors[idxe("d2V")];
    const double d3V = extractors[idxe("d3V")];
    const double rhoPhi = extractors[idxe("rhoPhi")];
    const double p0f = fermionic_p0();
    const auto arguments = device::tie(k, p_ext, p0f, prm.Nf, prm.Nc, prm.T, prm.muq, etaQ, etaPhi, hPhi, d1V, d2V,
                                       d3V, rhoPhi);

    // Residual convention: d_t v = -residual.
    residual[idxv("ZPhi")] = -etaPhi * ZPhi;
    flow_equations.hPhi0.get(residual[idxv("hPhi")], arguments);

    // The explicit symmetry breaking c sigma is fixed for the bare field; for the renormalized field used here
    // it scales with ZPhi^(-1/2).
    cSigma = std::pow(ZPhi, -0.5) * prm.cSigma;
  }

  /// The flow of u = dV/drho; the anomalous dimensions enter through the regulator insertions.
  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> &x,
            const Solution &sol) const
  {
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);
    const auto &fe_hessians = get<"fe_hessians">(sol);
    const auto &variables = get<"variables">(sol);
    const auto &extractors = get<"extractors">(sol);

    const auto etaQ = extractors[idxe("etaQ")];
    const auto etaPhi = extractors[idxe("etaPhi")];
    const auto hPhi = variables[idxv("hPhi")];
    const auto d1V = fe_functions[idxf("u")];
    const auto d2V = fe_derivatives[idxf("u")][0];
    const auto d3V = fe_hessians[idxf("u")][0][0];
    const double rhoPhi = x[0];
    const double p0f = fermionic_p0();
    const auto arguments =
        device::tie(k, p_ext, p0f, prm.Nf, prm.Nc, prm.T, prm.muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    flow_equations.V.get(flux[idxf("u")][0], arguments);
  }

  /// The rescaling of u = dV/drho with the renormalized field: etaPhi (u + rho du/drho).
  template <typename NT, typename Solution>
  void source(std::array<NT, Components::count_fe_functions(0)> &s_i, const Point<dim> &x, const Solution &sol) const
  {
    const auto u = get<"fe_functions">(sol)[idxf("u")];
    const auto du = get<"fe_derivatives">(sol)[idxf("u")][0];
    const auto etaPhi = get<"extractors">(sol)[idxe("etaPhi")];
    s_i[idxf("u")] = etaPhi * (u + du * x[0]);
  }

  /// The EoM with explicit symmetry breaking: the minimum of V(rho) - cSigma sigma, i.e. dV/drho = cSigma / sigma.
  template <int dim_, typename Vector> std::array<double, 1> EoM(const Point<dim_> &x, const Vector &u_i) const
  {
    const double sigma = std::sqrt(2. * x[0]);
    return {{u_i[idxf("u")] - cSigma / (sigma + 1e-14)}};
  }

  /// The potential is reconstructed from u alone, without the symmetry-breaking term.
  template <int dim_, typename Vector>
  std::array<double, 1> raw_potential_gradient(const Point<dim_> &, const Vector &u_i) const
  {
    return {{u_i[idxf("u")]}};
  }

  /// cSigma follows ZPhi, but the EoM cannot see the variables: carry it across a restart from a snapshot.
  void save_state(DiFfRG::ModelState &state) const
  {
    state.set("cSigma", cSigma);
  }

  void load_state(const DiFfRG::ModelState &state)
  {
    cSigma = state.get<double>("cSigma");
  }

  template <int dim_, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim_> &x, const Solutions &sol, const std::string &filename) const
  {
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);
    const auto &variables = get<"variables">(sol);
    const auto &extractors = get<"extractors">(sol);

    const double rhoPhi = x[0];
    const double sigma = std::sqrt(2. * rhoPhi);
    const double hPhi = variables[idxv("hPhi")];
    const double m2Pion = fe_functions[idxf("u")];
    const double m2Sigma = m2Pion + 2. * rhoPhi * fe_derivatives[idxf("u")][0];

    auto table = output.table(filename);
    table.value("sigma [GeV]", sigma);
    table.value("ZPhi", variables[idxv("ZPhi")]);
    table.value("etaQ", extractors[idxe("etaQ")]);
    table.value("etaPhi", extractors[idxe("etaPhi")]);
    table.value("hPhi", hPhi);
    table.value("m_pi [GeV]", std::sqrt(std::max(0., m2Pion)));
    table.value("m_sigma [GeV]", std::sqrt(std::max(0., m2Sigma)));
    table.value("m_q [GeV]", sigma * hPhi / std::sqrt(2. * prm.Nf));
  }

  /// Two readouts: at the minimum of V itself (the chiral limit) and at the EoM with explicit symmetry breaking.
  template <typename FUN, typename DataOut> void readouts_multiple(FUN &helper, DataOut &) const
  {
    helper(
        "chiral", [&](const auto &, const auto &u_i) { return std::array<double, 1>{{u_i[idxf("u")]}}; },
        [&](auto &output, const auto &x, const auto &sol) { readouts(output, x, sol, "data_chiral.csv"); });
    helper(
        "physical", [&](const auto &x, const auto &u_i) { return EoM(x, u_i); },
        [&](auto &output, const auto &x, const auto &sol) { readouts(output, x, sol, "data.csv"); });
  }

private:
  const Parameters prm;
  // mutable: the integrators' get() is not const, while the model's callbacks are.
  mutable LPA_QM_flows flow_equations;
  mutable double cSigma;

  /// External spatial momentum of the eta projections, (g(p) - g(0)) / p^2: small but finite.
  static constexpr double p_ext = 1e-4;

  /// Lowest fermionic Matsubara frequency pi T, smoothly cut off at small T where the sum becomes an integral.
  double fermionic_p0() const { return std::max(1e-3, M_PI * prm.T * std::exp(-k / (M_PI * prm.T))); }
};
