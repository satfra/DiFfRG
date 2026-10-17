#pragma once

#include <DiFfRG/model/model.hh>

#include "flows/flows.hh"

#include <algorithm>
#include <cmath>

using namespace DiFfRG;

struct Parameters {
  Parameters(const ConfigTree &config)
      : Lambda(config.get_double("/physical/Lambda")), T(config.get_double("/physical/T")),
        m2Phi(config.get_double("/physical/m2Phi")), lambdaPhi(config.get_double("/physical/lambdaPhi")),
        hPhi(config.get_double("/physical/hPhi")), cSigma(config.get_double("/physical/cSigma")),
        p_grid_min(config.get_double("/discretization/p_grid_min")),
        p_grid_max(config.get_double("/discretization/p_grid_max")),
        p_grid_bias(config.get_double("/discretization/p_grid_bias"))
  {
  }
  double Lambda, T;
  double m2Phi, lambdaPhi, hPhi, cSigma;
  double p_grid_min, p_grid_max, p_grid_bias;
};

// The meson wave functions live on a stack of n_matsubara bosonic Matsubara frequencies 2 pi T n, n = 0, 1, ...,
// each with the same logarithmic grid of p_grid_size spatial momenta. Entry m * p_grid_size + i is (2 pi T m, p_i).
static constexpr uint p_grid_size = 24;
static constexpr uint n_matsubara = 4;
static constexpr uint n_stack = n_matsubara * p_grid_size;

// The potential u = V' is an FE function. ZPi and ZSig are variables on the stack; Zbar is the wave function absorbed
// into the renormalised meson field so far. The extractors are the potential at the EoM.
using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
using VariableDesc = VariableDescriptor<FunctionND<"ZPi", n_stack>, FunctionND<"ZSig", n_stack>, Scalar<"Zbar">>;
using ExtractorDesc = ExtractorDescriptor<Scalar<"rhoPhi">, Scalar<"d1V">, Scalar<"d2V">, Scalar<"d3V">>;
using Components = ComponentDescriptor<FEFunctionDesc, VariableDesc, ExtractorDesc>;

constexpr auto idxf = FEFunctionDesc{};
constexpr auto idxv = VariableDesc{};
constexpr auto idxe = ExtractorDesc{};

/**
 * @brief Quark-meson model at finite temperature with momentum-dependent meson wave functions ZPi(p0, p) and
 * ZSig(p0, p).
 *
 * The meson field is renormalised such that ZPi(0, p_min) = 1 at every scale. The wave function absorbed into it,
 * Zbar, enters through the anomalous dimension etaPi = -d_t Zbar / Zbar (in the regulator insertions and the
 * rescaling of u, ZPi and ZSig), and through the renormalised Yukawa coupling hPhi / sqrt(Zbar) and explicit
 * breaking cSigma / sqrt(Zbar). The quarks keep their classical wave function. The flows come from QM.wl.
 */
class QuarkMeson : public def::AbstractModel<QuarkMeson, Components>,
                   public def::fRG,                        // this handles the fRG time
                   public def::FlowBoundaries<QuarkMeson>, // use Inflow/Outflow boundaries
                   public def::FE_AD<QuarkMeson>           // jacobians per AD, extractors held fixed
{
public:
  static constexpr uint dim = 1;
  // evaluate_batch reads values and first derivatives only.
  static constexpr bool batch_reads_hessians = false;

  using Coordinates = BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>;
  using Interpolator = SplineInterpolator1DStack<double, Coordinates>;

  QuarkMeson(const ConfigTree &config)
      : def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config),
        coordinates(0, n_matsubara, prm.T, p_grid_size, prm.p_grid_min, prm.p_grid_max, prm.p_grid_bias),
        ZPi(coordinates), ZSig(coordinates), cSigma(prm.cSigma)
  {
    flow_equations.set_T(prm.T);
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
    for (uint i = 0; i < n_stack; ++i) {
      values[idxv("ZPi") + i] = 1.;
      values[idxv("ZSig") + i] = 1.;
    }
    values[idxv("Zbar")] = 1.;
  }

  /**
   * @brief Read off the potential at the EoM, and the inputs of the potential flow that depend on the variables.
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
    // Clipped at zero: a negative pion mass would put a pole into the meson propagators of the two-point flows.
    const double d1V = std::max(0., double(fe_functions[idxf("u")]));
    const double d2V = fe_derivatives[idxf("u")][0];
    const double d3V = fe_hessians[idxf("u")][0][0];
    extractors[idxe("rhoPhi")] = rhoPhi;
    extractors[idxe("d1V")] = d1V;
    extractors[idxe("d2V")] = d2V;
    extractors[idxe("d3V")] = d3V;

    // The rest carries no derivative with respect to the FE solution: compute it in the plain pass only.
    if constexpr (std::is_same_v<NT, double>) {
      update_dressings(variables.data());
      hPhi = prm.hPhi / std::sqrt(variables[idxv("Zbar")]);

      // The Matsubara quadrature is sized from the largest mass in the problem.
      const double m2Quark = powr<2>(hPhi) * rhoPhi / 2.;
      const double m2Sigma = d1V + 2. * rhoPhi * d2V;
      flow_equations.set_typical_E(std::sqrt(std::max({m2Quark, d1V, m2Sigma, 0.}) + powr<2>(k)));

      etaPi = solve_eta(rhoPhi, d1V, d2V, d3V);
    }
  }

  /// The flows of ZPi and ZSig on the stack, and of Zbar.
  template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &sol) const
  {
    const auto &variables = get<"variables">(sol);
    const auto &extractors = get<"extractors">(sol);

    const double rhoPhi = extractors[idxe("rhoPhi")];
    const double d1V = extractors[idxe("d1V")];
    const double d2V = extractors[idxe("d2V")];
    const double d3V = extractors[idxe("d3V")];
    const double Zbar = variables[idxv("Zbar")];

    update_dressings(variables.data());
    hPhi = prm.hPhi / std::sqrt(Zbar);
    double eta = solve_eta(rhoPhi, d1V, d2V, d3V);

    // The order of the arguments is that of kernelParameterList in QM.wl.
    const auto arguments = device::tie(k, prm.T, hPhi, eta, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
    // One integral with two outputs: the group writes both flows in one pass.
    flow_equations.Mesons.map({&residual[idxv("ZPi")], &residual[idxv("ZSig")]}, coordinates, arguments);

    // Residual convention: d_t v = -residual, with eta = -d_t Zbar / Zbar.
    residual[idxv("Zbar")] = -eta * Zbar;
    // The explicit symmetry breaking is fixed for the bare field.
    cSigma = prm.cSigma / std::sqrt(Zbar);
  }

  /**
   * @brief The flow of u = V' at all points of a batch. The flux is the loop, one map_points() call on the GPU; the
   * source is the rescaling with the renormalised field, etaPi (u + rho u').
   */
  template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
  {
    using NT = typename Batch::number_type;
    const auto rho = batch.coordinates(0);
    const auto u = batch.values(idxf("u"));
    const auto du = batch.derivatives(idxf("u"), 0);
    if (out.requested(Term::flux)) {
      // rho, u and u' differ per point; everything else is shared by all points.
      const NT d3V(0.);
      flow_equations.V.map_points(out.flux(idxf("u"), 0), k, prm.T, hPhi, etaPi, rho, u, du, d3V, ZPi, ZSig);
    }
    if (out.requested(Term::source)) {
      const auto s = out.source(idxf("u"));
      for (size_t i = 0; i < batch.size(); ++i)
        s[i] = etaPi * (u[i] + rho[i] * du[i]);
    }
  }

  /// The EoM with explicit symmetry breaking: the minimum of V(rho) - cSigma sigma, i.e. V' = cSigma / sigma.
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

  /// cSigma follows Zbar, but the EoM cannot see the variables: carry it across a restart from a snapshot.
  void save_state(DiFfRG::ModelState &state) const { state.set("cSigma", cSigma); }
  void load_state(const DiFfRG::ModelState &state) { cSigma = state.get<double>("cSigma"); }

  template <int dim_, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim_> &x, const Solutions &sol) const
  {
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);
    const auto &variables = get<"variables">(sol);

    const double rhoPhi = x[0];
    const double sigma = std::sqrt(2. * rhoPhi);
    const double m2Pion = fe_functions[idxf("u")];
    const double m2Sigma = m2Pion + 2. * rhoPhi * fe_derivatives[idxf("u")][0];
    // ZPi(0, p_min) = 1; ZSig(0, p_min) normalises the sigma mass. ZPi(2 pi T, p_min) / ZPi(0, p_min) measures the
    // splitting of temporal and spatial momenta in the heat bath.
    const double ZSig0 = variables[idxv("ZSig")];
    const double ZPiPar = variables[idxv("ZPi") + p_grid_size];
    const double h = prm.hPhi / std::sqrt(variables[idxv("Zbar")]);

    auto table = output.table("data.csv");
    table.value("sigma [GeV]", sigma);
    table.value("m_pi [GeV]", std::sqrt(std::max(0., m2Pion)));
    table.value("m_sigma [GeV]", std::sqrt(std::max(0., m2Sigma / ZSig0)));
    table.value("m_q [GeV]", sigma * h / 2.);
    table.value("etaPi", etaPi);
    table.value("Zbar", variables[idxv("Zbar")]);
    table.value("ZSig(0,p_min)", ZSig0);
    table.value("ZPi(2piT,p_min)", ZPiPar);
  }

private:
  const Parameters prm;
  // mutable: the integrators' get(), map() and map_points() are not const, while the model's callbacks are.
  mutable QMFlows flow_equations;
  const Coordinates coordinates;
  mutable Interpolator ZPi, ZSig;
  // Inputs of the potential flow, set in extract(): the renormalised Yukawa coupling and etaPi at the current
  // state. cSigma is set in dt_variables().
  mutable double hPhi = 0., etaPi = 0., cSigma;

  void update_dressings(const double *variables) const
  {
    ZPi.update(&variables[idxv("ZPi")]);
    ZSig.update(&variables[idxv("ZSig")]);
  }

  /**
   * @brief etaPi such that ZPi(0, p_min) stays one, i.e. its flow vanishes there.
   *
   * etaPi enters the flow linearly, through the regulator insertion d_t R - etaPi R and the rescaling + etaPi ZPi,
   * so two evaluations of the flow at that point fix it.
   */
  double solve_eta(const double &rhoPhi, const double &d1V, const double &d2V, const double &d3V) const
  {
    double eta = 0.;
    const auto arguments = device::tie(k, prm.T, hPhi, eta, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
    MultiValue<double, 2> g0, g1;
    flow_equations.Mesons.get(g0, 0., prm.p_grid_min, arguments);
    eta = 1.;
    flow_equations.Mesons.get(g1, 0., prm.p_grid_min, arguments);
    return -g0[0] / (g1[0] - g0[0]);
  }
};
