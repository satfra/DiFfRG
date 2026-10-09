#pragma once

#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "flows/flows.hh"

#include <flow_abort.hh>

struct Parameters {
  Parameters(const ConfigTree &config)
  {
    try {
      Lambda = config.get_double("/physical/Lambda");

      alphaA3 = config.get_double("/physical/alphaA3");
      alphaA4 = config.get_double("/physical/alphaA4");
      alphaAcbc = config.get_double("/physical/alphaAcbc");

      tilt_A3 = config.get_double("/physical/tilt_A3");
      tilt_A4 = config.get_double("/physical/tilt_A4");
      tilt_Acbc = config.get_double("/physical/tilt_Acbc");

      m2A = config.get_double("/physical/m2A");

      p_grid_min = config.get_double("/discretization/p_grid_min");
      p_grid_max = config.get_double("/discretization/p_grid_max");
      p_grid_center = config.get_double("/discretization/p_grid_center");
      p_grid_focus = config.get_double("/discretization/p_grid_focus");

      eta_iter_max = config.get_int("/physical/eta_iter_max");
      eta_tol = config.get_double("/physical/eta_tol");

    } catch (std::exception &e) {
      std::cerr << "Error in reading parameters: " << e.what() << std::endl;
      throw;
    }
  }

  double Lambda;
  double alphaA3, alphaA4, alphaAcbc;
  double tilt_A3, tilt_A4, tilt_Acbc;
  double m2A;

  int eta_iter_max;
  double eta_tol;

  // Momentum grid: logarithmic, but with the points clustered around p_grid_center so that the
  // ~1 GeV structure of the dressings is resolved rather than smeared over two or three cells.
  // p_grid_focus = 0 recovers a plain logarithmic grid.
  double p_grid_min, p_grid_max, p_grid_center, p_grid_focus;
};

// Size of the momentum grid
static constexpr uint p_grid_size = 96;
// Everything is a Variable on the momentum grid: the gauge vertex dressings and the propagator
// dressings. No FE functions, no extractors.
using VariableDesc =
    VariableDescriptor<FunctionND<"ZA3", p_grid_size>, FunctionND<"ZAcbc", p_grid_size>, FunctionND<"ZA4", p_grid_size>,

                       FunctionND<"ZA", p_grid_size>, FunctionND<"Zc", p_grid_size>>;
using Components = ComponentDescriptor<FEFunctionDescriptor<>, VariableDesc, ExtractorDescriptor<>>;

constexpr auto idxv = VariableDesc{};

/**
 * @brief SU(3) Yang-Mills in a vertex expansion, with every vertex evaluated on the symmetric
 * point so that each dressing lives on one momentum grid.
 */
class YangMills : public def::AbstractModel<YangMills, Components>,
                  public def::fRG,        // this handles the fRG time
                  public def::NoJacobians // define all jacobians per AD
{
  const Parameters prm;

  using Coordinates1D = FocusedLogCoordinates1D<double>;
  const Coordinates1D coordinates1D;

  mutable YangMillsFlows flow_equations;

  mutable SplineInterpolator1D<double, Coordinates1D> dtZc, dtZA, ZA, Zc;
  mutable SplineInterpolator1D<double, Coordinates1D> ZA4, ZAcbc, ZA3;

public:
  YangMills(const ConfigTree &config)
      : def::fRG(config.get_double("/physical/Lambda")), prm(config),
        coordinates1D(p_grid_size, prm.p_grid_min, prm.p_grid_max, prm.p_grid_center, prm.p_grid_focus),
        flow_equations(config),
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

    // Early abort, on the INCOMING state and before any kernel is launched. The placement is the
    // whole safety argument, not a convenience: Variables::Assembler::residual_variables calls
    // dt_variables() and only then fences, so throwing after a .map() would unwind past buffers
    // with kernels still in flight. Here nothing has been launched in this call and the previous
    // call already fenced.
    //
    // The criterion is shared with the tuner's post-hoc trajectory scan, so a probe is classified
    // the same way however it was stopped, and the RG time travels out inside the exception --
    // which is precisely the residual the divergent-branch fit consumes.
    {
      const double m2A_t = variables.data()[idxv("ZA")] * powr<2>(prm.p_grid_min) - powr<2>(prm.p_grid_min);
      double zc_min = std::numeric_limits<double>::infinity();
      for (uint i = 0; i < p_grid_size; ++i)
        zc_min = std::min(zc_min, variables.data()[idxv("Zc") + i]);
      if (flow_has_run_away(m2A_t, prm.m2A, zc_min)) throw FlowAbort(t);
    }

    ZA3.update(&variables.data()[idxv("ZA3")]);
    ZAcbc.update(&variables.data()[idxv("ZAcbc")]);
    ZA4.update(&variables.data()[idxv("ZA4")]);

    ZA.update(&variables.data()[idxv("ZA")]);
    Zc.update(&variables.data()[idxv("Zc")]);

    // set up arguments for the integrators
    const auto arguments = device::tie(k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);

    // Self-consistent solve for the propagator anomalous dimensions: dtZA and dtZc appear inside
    // their own regulator insertions, so the previous iterate is kept to measure convergence.
    std::vector<double> old_dtZA(p_grid_size), old_dtZc(p_grid_size);
    bool eta_converged = false;
    int n_iter = 0;
    while (!eta_converged) {
      for (uint i = 0; i < p_grid_size; ++i) {
        old_dtZA[i] = dtZA[i];
        old_dtZc[i] = dtZc[i];
      }

      // ZA and Zc do not read each other's result, so issue both before waiting for either.
      {
        DeferredMaps defer;
        flow_equations.ZA.map(&residual[idxv("ZA")], coordinates1D, arguments);
        flow_equations.Zc.map(&residual[idxv("Zc")], coordinates1D, arguments);
      } // both land here

      dtZA.update(&residual[idxv("ZA")]);
      dtZc.update(&residual[idxv("Zc")]);

      double dist = 0.;
      for (uint i = 0; i < p_grid_size; ++i) {
        dist = std::max(dist, std::abs(dtZA[i] - old_dtZA[i]) / std::abs(dtZA[i]));
        dist = std::max(dist, std::abs(dtZc[i] - old_dtZc[i]) / std::abs(dtZc[i]));
      }
      n_iter++;
      if (dist < prm.eta_tol || n_iter >= prm.eta_iter_max) eta_converged = true;
    }

    // The vertices are mutually independent and nothing here reads them back, so the host issues
    // all three and only then waits.
    {
      DeferredMaps defer;
      flow_equations.ZA4.map(&residual[idxv("ZA4")], coordinates1D, arguments);
      flow_equations.ZAcbc.map(&residual[idxv("ZAcbc")], coordinates1D, arguments);
      flow_equations.ZA3.map(&residual[idxv("ZA3")], coordinates1D, arguments);
    }
  }

  template <int dim, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim> &, const Solutions &sol) const
  {
    const auto &variables = get<"variables">(sol);
    auto hdf = output.hdf5();

    // No health check here: whether a flow is usable is the driver's decision, taken once on the
    // finished trajectory. Throwing from a readout would abort the run at an arbitrary output
    // step and leave the tuner with a truncated file instead of a classified probe.
    hdf.map("ZA", coordinates1D, &(variables.data()[idxv("ZA")]));
    hdf.map("Zc", coordinates1D, &(variables.data()[idxv("Zc")]));

    hdf.map("dtZA", dtZA);
    hdf.map("dtZc", dtZc);

    hdf.map("ZAcbc", coordinates1D, &(variables.data()[idxv("ZAcbc")]));
    hdf.map("ZA3", coordinates1D, &(variables.data()[idxv("ZA3")]));
    hdf.map("ZA4", coordinates1D, &(variables.data()[idxv("ZA4")]));

    // The scalar trajectories the tuner reads back: m2A is the observable it tunes, and the two
    // p_min values are what the runaway criterion is evaluated on, sample by sample.
    hdf.scalar("k", k);
    hdf.scalar("m2A", variables[idxv("ZA")] * powr<2>(prm.p_grid_min) - powr<2>(prm.p_grid_min));
    hdf.scalar("Zc_pmin", variables[idxv("Zc")]);
    hdf.scalar("ZA_pmin", variables[idxv("ZA")]);
  }
};
