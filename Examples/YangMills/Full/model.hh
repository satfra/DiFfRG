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
  // The same clustering is applied to the S0 (overall scale) axis of the vertex grid.
  // p_grid_focus = 0 recovers a plain logarithmic grid.
  double p_grid_min, p_grid_max, p_grid_center, p_grid_focus;
};

// Size of the 1D (propagator / symmetric-point) momentum grid.
static constexpr uint p_grid_size = 96;
// Size of the angle-resolved vertex grid. The 3-point dressings are tabulated over the symmetric
// triangle variables (S0, S1, SPhi): S0 the overall scale (logarithmic), S1 the shape (linear,
// [0,1)), SPhi an angle (linear). This is the legacy parametrisation; gridding this way keeps the
// transverse projector's collinear singularity confined to a single grid corner (S1->1, SPhi->0),
// instead of a whole face as with (|p1|,|p2|,cos), which is what makes the flow tunable.
static constexpr uint S0_grid_size = 96; // match the legacy momentum resolution on the vertex scale axis
static constexpr uint S1_grid_size = 8;
// SPhi is a genuine angle: everything downstream depends only on sin(SPhi), cos(SPhi), cos(2 SPhi), so the axis is
// gridded periodically over [-pi, pi) with the endpoint excluded. That drops one full 2D plane of variables compared
// to the naive [-pi, pi] grid (where -pi and +pi were duplicated, unconstrained degrees of freedom) at unchanged
// pi/3 spacing, and makes the interpolation wrap across the seam instead of clamping.
static constexpr uint SPhi_grid_size = 6;
static constexpr uint vertex_grid_size = S0_grid_size * S1_grid_size * SPhi_grid_size;

// The 3-gluon (ZA3), ghost-gluon (ZAcbc) and 4-gluon tadpole (ZA4tadpole) are angle-resolved 3D
// grids over (S0,S1,SPhi); the 4-gluon symmetric point (ZA4SP) is 1D; ZA and Zc are the
// gluon/ghost dressings (1D). The tadpole feeds the gluon self-energy with loop-dependent
// (external,loop) kinematics, as in the legacy code.
using VariableDesc =
    VariableDescriptor<FunctionND<"ZA3", vertex_grid_size>, FunctionND<"ZAcbc", vertex_grid_size>,
                       FunctionND<"ZA4SP", p_grid_size>, FunctionND<"ZA4tadpole", vertex_grid_size>,

                       FunctionND<"ZA", p_grid_size>, FunctionND<"Zc", p_grid_size>>;
using Components = ComponentDescriptor<FEFunctionDescriptor<>, VariableDesc, ExtractorDescriptor<>>;

constexpr auto idxv = VariableDesc{};

/**
 * @brief Fully momentum- and angle-dependent Yang-Mills truncation. The gluon mass enters only
 * through the initial condition of the gluon dressing ZA, and is read off from ZA at the lowest
 * grid point in the readouts (no separate m2A variable).
 */
class YangMills : public def::AbstractModel<YangMills, Components>,
                  public def::fRG,        // this handles the fRG time
                  public def::NoJacobians // define all jacobians per AD
{
  const Parameters prm;

  using Coordinates1D = FocusedLogCoordinates1D<double>;
  // CoordinatePackND<FocusedLog, Lin, LinPeriodic> over (S0, S1, SPhi)
  using Coordinates3D = FocusedLogLinLinPeriodicCoordinates;

  const Coordinates1D coordinates1D;
  const Coordinates1D S0_coordinates;
  const LinearCoordinates1D<double> S1_coordinates;
  const LinearPeriodicCoordinates1D<double> SPhi_coordinates;
  const Coordinates3D coordinates3D;

  mutable YangMillsFlows flow_equations;

  mutable SplineInterpolator1D<double, Coordinates1D> dtZc, dtZA, ZA, Zc, ZA4SP;
  mutable LinearInterpolatorND<double, Coordinates3D> ZA3, ZAcbc, ZA4tadpole;

public:
  YangMills(const ConfigTree &config)
      : def::fRG(config.get_double("/physical/Lambda")), prm(config),
        coordinates1D(p_grid_size, prm.p_grid_min, prm.p_grid_max, prm.p_grid_center, prm.p_grid_focus),
        S0_coordinates(S0_grid_size, prm.p_grid_min, prm.p_grid_max, prm.p_grid_center, prm.p_grid_focus),
        S1_coordinates(S1_grid_size, 0.0, 0.9999),       // shape variable, [0,1)
        SPhi_coordinates(SPhi_grid_size, -M_PI, M_PI),   // periodic angle, matches the atan2 feed range (-pi, pi]
        coordinates3D(S0_coordinates, S1_coordinates, SPhi_coordinates), flow_equations(config),
        dtZc(coordinates1D), dtZA(coordinates1D), ZA(coordinates1D), Zc(coordinates1D), ZA4SP(coordinates1D),
        ZA3(coordinates3D), ZAcbc(coordinates3D), ZA4tadpole(coordinates3D)
  {
    flow_equations.set_k(prm.Lambda);
    k = std::exp(-t) * Lambda;
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const
  {
    // angle-resolved 3-point dressings (3D grids over S0,S1,SPhi)
    for (uint i = 0; i < S0_grid_size; ++i)
      for (uint j = 0; j < S1_grid_size; ++j)
        for (uint l = 0; l < SPhi_grid_size; ++l) {
          const auto point = coordinates3D.forward(i, j, l);
          const double S0 = point[0]; // overall scale, used only for the (small) initial tilt
          const size_t idx = (i * S1_grid_size + j) * SPhi_grid_size + l;
          values[idxv("ZA3") + idx] =
              std::sqrt(4. * M_PI * prm.alphaA3) + prm.tilt_A3 * std::log(S0 / prm.p_grid_max);
          values[idxv("ZAcbc") + idx] =
              std::sqrt(4. * M_PI * prm.alphaAcbc) + prm.tilt_Acbc * std::log(S0 / prm.p_grid_max);
          values[idxv("ZA4tadpole") + idx] =
              4. * M_PI * prm.alphaA4 + prm.tilt_A4 * std::log(S0 / prm.p_grid_max);
        }

    // 1D dressings
    for (uint i = 0; i < p_grid_size; ++i) {
      const double p = coordinates1D.forward(i);
      values[idxv("ZA4SP") + i] = 4. * M_PI * prm.alphaA4 + prm.tilt_A4 * std::log(p / prm.p_grid_max);
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
    ZA4SP.update(&variables.data()[idxv("ZA4SP")]);
    ZA4tadpole.update(&variables.data()[idxv("ZA4tadpole")]);

    ZA.update(&variables.data()[idxv("ZA")]);
    Zc.update(&variables.data()[idxv("Zc")]);

    // set up arguments for the integrators (order matches the generated map() signature)
    const auto arguments = device::tie(k, ZA3, ZAcbc, ZA4SP, ZA4tadpole, dtZc, Zc, dtZA, ZA);

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
    // all four and only then waits.
    {
      DeferredMaps defer;
      flow_equations.ZA3.map(&residual[idxv("ZA3")], coordinates3D, arguments);
      flow_equations.ZAcbc.map(&residual[idxv("ZAcbc")], coordinates3D, arguments);
      flow_equations.ZA4SP.map(&residual[idxv("ZA4SP")], coordinates1D, arguments);
      flow_equations.ZA4tadpole.map(&residual[idxv("ZA4tadpole")], coordinates3D, arguments);
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
    hdf.map("ZA4SP", coordinates1D, &(variables.data()[idxv("ZA4SP")]));

    hdf.map("dtZA", dtZA);
    hdf.map("dtZc", dtZc);

    hdf.map("ZA3", coordinates3D, &(variables.data()[idxv("ZA3")]));
    hdf.map("ZAcbc", coordinates3D, &(variables.data()[idxv("ZAcbc")]));

    // The scalar trajectories the tuner reads back: m2A is the observable it tunes, and the two
    // p_min values are what the runaway criterion is evaluated on, sample by sample. m2A is taken
    // at the lowest grid point, the same definition the in-flow abort uses -- the least-squares
    // extrapolation below is the better physical estimate but a worse classifier, since it can
    // stay finite while the flow is already running away.
    hdf.scalar("k", k);
    hdf.scalar("m2A", variables[idxv("ZA")] * powr<2>(prm.p_grid_min) - powr<2>(prm.p_grid_min));
    hdf.scalar("Zc_pmin", variables[idxv("Zc")]);
    hdf.scalar("ZA_pmin", variables[idxv("ZA")]);

    // The gluon mass gap extrapolated to p = 0. In the IR the gluon acquires a mass gap,
    // ZA(p) = 1 + m2A/p^2 + (corrections), so the per-point estimate m2A_i := (ZA(p_i) - 1) p_i^2
    // obeys m2A_i = m2A + b p_i^2 + ... A single deepest point is fragile to grid noise; instead
    // least-squares fit a straight line in p^2 over the n_fit deepest-IR grid points and report
    // the intercept -- the genuine p -> 0 limit, not a window average.
    const uint n_fit = std::min<uint>(6u, p_grid_size);
    const double *ZA_data = &(variables.data()[idxv("ZA")]);
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (uint i = 0; i < n_fit; ++i) {
      const double p2 = powr<2>(coordinates1D.forward(i));
      const double m2A_i = (ZA_data[i] - 1.0) * p2;
      sx += p2;
      sy += m2A_i;
      sxx += p2 * p2;
      sxy += p2 * m2A_i;
    }
    // Intercept of the least-squares line; fall back to the mean of the deep-IR points if
    // they are degenerate in p² (denom == 0).
    const double denom = n_fit * sxx - sx * sx;
    hdf.scalar("m2A_fit", std::abs(denom) > 0.0 ? (sy * sxx - sx * sxy) / denom : sy / n_fit);
  }
};
