#pragma once

#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "flows/flows.hh"

struct Parameters {
  Parameters(const ConfigTree &json)
  {
    try {
      Lambda = json.get_double("/physical/Lambda");

      alphaA3 = json.get_double("/physical/alphaA3");
      alphaA4 = json.get_double("/physical/alphaA4");
      alphaAcbc = json.get_double("/physical/alphaAcbc");

      tilt_A3 = json.get_double("/physical/tilt_A3");
      tilt_A4 = json.get_double("/physical/tilt_A4");
      tilt_Acbc = json.get_double("/physical/tilt_Acbc");

      m2A = json.get_double("/physical/m2A");

      p_grid_min = json.get_double("/discretization/p_grid_min");
      p_grid_max = json.get_double("/discretization/p_grid_max");
      p_grid_bias = json.get_double("/discretization/p_grid_bias");

      eta_iter_max = json.get_int("/physical/eta_iter_max");
      eta_tol = json.get_double("/physical/eta_tol");

    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }

  double Lambda;
  double alphaA3, alphaA4, alphaAcbc;
  double tilt_A3, tilt_A4, tilt_Acbc;
  double m2A;

  int eta_iter_max;
  double eta_tol;

  double p_grid_min, p_grid_max, p_grid_bias;
};

// Size of the momentum grid
static constexpr uint p_grid_size = 96;
// As for components, we have one FE function (u) and several extractors.
using VariableDesc =
    VariableDescriptor<FunctionND<"ZA3", p_grid_size>, FunctionND<"ZAcbc", p_grid_size>, FunctionND<"ZA4", p_grid_size>,

                       FunctionND<"ZA", p_grid_size>, FunctionND<"Zc", p_grid_size>>;
using Components = ComponentDescriptor<FEFunctionDescriptor<>, VariableDesc, ExtractorDescriptor<>>;

constexpr auto idxv = VariableDesc{};

/**
 * @brief This class implements the numerical model for the quark-meson model at finite temperature and chemical
 * potential.
 */
class YangMills : public def::AbstractModel<YangMills, Components>,
                  public def::fRG,        // this handles the fRG time
                  public def::NoJacobians // define all jacobians per AD
{
  const Parameters prm;

  using Coordinates1D = LogarithmicCoordinates1D<double>;
  const Coordinates1D coordinates1D;

  mutable YangMillsFlows flow_equations;

  mutable SplineInterpolator1D<double, Coordinates1D> dtZc, dtZA, ZA, Zc;
  mutable SplineInterpolator1D<double, Coordinates1D> ZA4, ZAcbc, ZA3;

public:
  YangMills(const ConfigTree &json)
      : def::fRG(json.get_double("/physical/Lambda")), prm(json),
        coordinates1D(p_grid_size, prm.p_grid_min, prm.p_grid_max, prm.p_grid_bias), flow_equations(json),
        dtZc(coordinates1D), dtZA(coordinates1D), ZA(coordinates1D), Zc(coordinates1D), // propagators
        ZA4(coordinates1D), ZAcbc(coordinates1D), ZA3(coordinates1D)                    // couplings
  {
    flow_equations.set_k(prm.Lambda);
    k = std::exp(-t) * Lambda;
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const
  {
    for (uint i = 0; i < p_grid_size; ++i) {
      const double p = coordinates1D.forward(i);
      values[idxv("ZA4") + i] = 4. * M_PI * prm.alphaA4 + prm.tilt_A4 * std::log(p / prm.p_grid_max);
      values[idxv("ZA3") + i] = std::sqrt(4. * M_PI * prm.alphaA3) + prm.tilt_A3 * std::log(p / prm.p_grid_max);
      values[idxv("ZAcbc") + i] = std::sqrt(4. * M_PI * prm.alphaAcbc) + prm.tilt_Acbc * std::log(p / prm.p_grid_max);

      values[idxv("ZA") + i] = (powr<2>(p) + prm.m2A) / powr<2>(p);
      values[idxv("Zc") + i] = 1.;
    }
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

    ZA3.update(&variables.data()[idxv("ZA3")]);
    ZAcbc.update(&variables.data()[idxv("ZAcbc")]);
    ZA4.update(&variables.data()[idxv("ZA4")]);

    ZA.update(&variables.data()[idxv("ZA")]);
    Zc.update(&variables.data()[idxv("Zc")]);

    // set up arguments for the integrators
    const auto arguments = device::tie(k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);

    // copy the propagators for comparison
    std::vector<double> old_dtZA(p_grid_size);
    std::vector<double> old_dtZc(p_grid_size);

    // start by solving the equations for propagators
    bool eta_converged = false;
    int n_iter = 0;
    while (!eta_converged) {
      // copy
      for (uint i = 0; i < p_grid_size; ++i) {
        old_dtZA[i] = dtZA[i];
        old_dtZc[i] = dtZc[i];
      }

      const auto ZA_exec = flow_equations.ZA.map(&residual[idxv("ZA")], coordinates1D, arguments);
      const auto Zc_exec = flow_equations.Zc.map(&residual[idxv("Zc")], coordinates1D, arguments);

      dtZA.update(&residual[idxv("ZA")]);
      dtZc.update(&residual[idxv("Zc")]);

      // check distance
      double dist = 0.;
      for (uint i = 0; i < p_grid_size; ++i) {
        dist = std::max(dist, std::abs(dtZA[i] - old_dtZA[i]) / std::abs(dtZA[i]));
        dist = std::max(dist, std::abs(dtZc[i] - old_dtZc[i]) / std::abs(dtZc[i]));
      }
      if (dist < prm.eta_tol || n_iter > prm.eta_iter_max) eta_converged = true;
      n_iter++;
    }
    std::cout << "Converged after " << n_iter << " iterations." << std::endl;

    const auto exec_ZA4 = flow_equations.ZA4.map(&residual[idxv("ZA4")], coordinates1D, arguments);
    const auto exec_ZAcbc = flow_equations.ZAcbc.map(&residual[idxv("ZAcbc")], coordinates1D, arguments);
    const auto exec_ZA3 = flow_equations.ZA3.map(&residual[idxv("ZA3")], coordinates1D, arguments);
  }

  template <int dim, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim> &, const Solutions &sol) const
  {
    const auto &variables = get<"variables">(sol);

    // sanity check: make sure 0 < Zc[0] < 1
    if (Zc[0] < 0 || Zc[0] > 1) {
      throw std::runtime_error("Diverging result: Zc(0) = " + std::to_string(Zc[0]));
    }

    auto hdf = output.hdf5();
    hdf.map("ZA", coordinates1D, &(variables.data()[idxv("ZA")]));
    hdf.map("Zc", coordinates1D, &(variables.data()[idxv("Zc")]));

    hdf.map("dtZA", dtZA);
    hdf.map("dtZc", dtZc);

    hdf.map("ZAcbc", coordinates1D, &(variables.data()[idxv("ZAcbc")]));
    hdf.map("ZA3", coordinates1D, &(variables.data()[idxv("ZA3")]));
    hdf.map("ZA4", coordinates1D, &(variables.data()[idxv("ZA4")]));

    hdf.scalar("k", k);
    hdf.scalar("m2A",
               variables[idxv("ZA") + 0] * powr<2>(coordinates1D.forward(0)) - powr<2>(coordinates1D.forward(0)));
  }
};
