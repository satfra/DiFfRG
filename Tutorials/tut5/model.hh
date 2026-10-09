#pragma once

#include <DiFfRG/model/model.hh>

#include <algorithm>
#include <optional>
#include <vector>

using namespace DiFfRG;

struct Parameters {
  Parameters(const ConfigTree &value)
  {
    try {
      Lambda = value.get_double("/physical/Lambda");
      N = value.get_double("/physical/N");
      lambda2 = value.get_double("/physical/lambda2");
      lambda4 = value.get_double("/physical/lambda4");
      lambda6 = value.get_double("/physical/lambda6");

      contrast = value.get_double("/shock/contrast");
      locality = value.get_double("/shock/locality");
      gap = value.get_uint("/shock/gap");
      width = value.get_uint("/shock/width");
      window = value.get_uint("/shock/window");
      edge_cells = value.get_uint("/shock/edge_cells");
      offset_cells = value.get_uint("/shock/offset_cells");
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }
  double Lambda, N, lambda2, lambda4, lambda6;
  double contrast, locality;
  uint gap, width, window, edge_cells, offset_cells;
};

using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
using VariableDesc = VariableDescriptor<Scalar<"A">>;
using ExtractorDesc = ExtractorDescriptor<Scalar<"m2Sigma_x">, Scalar<"rho_x">>;
using Components = ComponentDescriptor<FEFunctionDesc, VariableDesc, ExtractorDesc>;

constexpr auto idxf = FEFunctionDesc{};
constexpr auto idxv = VariableDesc{};
constexpr auto idxe = ExtractorDesc{};

/**
 * @brief O(N) effective potential in d = 3, with the extractors read off above the shock.
 *
 * The flow of u = d_rho V is a conservation law in field space, solved here with the Kurganov-Tadmor
 * finite-volume assembler. Started from a potential with a negative quartic and a stabilising sextic
 * coupling, u develops a *kink* -- a jump in d_rho u at finite rho -- separating the symmetric
 * region around the origin from the broken branch beyond it.
 *
 * That kink is what makes this model interesting for extractors. The equation of motion sits at the
 * global minimum of the potential, which in this parameter range is at rho = 0, so anything read off
 * at the EoM is a statement about the symmetric phase. The physics of the broken branch lives on the
 * far side of the kink. `extractor_point` below is how the model says so.
 */
class Tut5 : public def::AbstractModel<Tut5, Components>,
             public def::fRG,                                       // this handles the fRG time
             public def::RhoSymmetricLinearExtrapolationBoundaries<Tut5>, // even at rho = 0
             public def::AD<Tut5>                                    // define all jacobians per AD
{
private:
  const Parameters prm;

public:
  static constexpr uint dim = 1;

  Tut5(const ConfigTree &json) : def::fRG(json.get_double("/physical/Lambda")), prm(json) {}

  template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
  {
    const auto rho = pos[0];
    // V = 1/2 lambda2 phi^2 + 1/4 lambda4 phi^4 + 1/6 lambda6 phi^6, so u = d_rho V is
    values[idxf("u")] = prm.lambda2 + 2. * prm.lambda4 * rho + 4. * prm.lambda6 * powr<2>(rho);
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const { values[idxv("A")] = 0.; }

  /**
   * @brief Advective flux: the N-1 Goldstone modes, which depend on u alone.
   */
  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> & /*pos*/,
            const Solution &sol) const
  {
    const auto m2Pi = get<0>(sol)[idxf("u")];
    flux[idxf("u")][0] = (prm.N - 1.) * loop(m2Pi);
  }

  /**
   * @brief Diffusive flux: the radial mode, whose mass carries d_rho u.
   */
  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> &pos,
                      const Solution &sol) const
  {
    const auto rho = pos[0];
    const auto m2Sigma = get<0>(sol)[idxf("u")] + 2. * rho * get<1>(sol)[idxf("u")][0];
    flux[idxf("u")][0] = loop(m2Sigma);
  }

  /**
   * @brief Read the solution at whatever point `extractor_point` chose.
   *
   * The values stored here are the only thing `dt_variables` gets to see of the field-space
   * solution. Storing rho_x as well is what lets the readouts below report *where* they were taken.
   */
  template <typename NT, typename Solution>
  void extract(std::array<NT, Components::count_extractors()> &extractors, const Point<dim> &x,
               const Solution &sol) const
  {
    const auto rho = x[0];
    const auto u = get<"fe_functions">(sol)[idxf("u")];
    const auto du = get<"fe_derivatives">(sol)[idxf("u")][0];

    extractors[idxe("m2Sigma_x")] = u + 2. * rho * du;
    extractors[idxe("rho_x")] = rho;
  }

  /**
   * @brief A diagnostic accumulated along the flow: A(t) = int_0^t m^2_sigma(rho_x) dt'.
   *
   * Deliberately a plain observable rather than a truncation. It exists to close the loop -- an
   * extractor read off the field-space solution driving an ODE that is not discretized in field
   * space at all -- with nothing else going on. A real LPA' would flow a wave function Z_phi here
   * instead, from an anomalous dimension computed in `extract`; see Examples/QuarkMesonLPAprime.
   * The residual convention is A_dot = -r_a.
   */
  template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &sol) const
  {
    r_a[idxv("A")] = -get<"extractors">(sol)[idxe("m2Sigma_x")];
  }

  /**
   * @brief Evaluate the extractors just above the lowest shock, or at the EoM if u is smooth.
   *
   * The shock is a *weak* discontinuity: u stays continuous and only its slope jumps. So the thing
   * to measure is the step between the slope on either side, never a jump in u itself -- the latter
   * is O(drho) at a kink, which would make any threshold on it a threshold on the grid spacing.
   *
   * Two tests separate a real kink from the smooth curvature u has everywhere:
   *   * the step must stand out from its neighbourhood by a factor `contrast`, and
   *   * the slope variation actually realised across the transition zone must account for at least
   *     `locality` of the step. The solver occasionally puts a small spike into d_rho u with a
   *     smooth shoulder leading up to it; the shoulder alone tilts the two sides against each other
   *     and fakes a step, but only a fraction of it is realised locally. This second test is the one
   *     that does the work.
   *
   * The full detector this is cut down from carries a few more guards; see the LPA' model in the
   * shock-scaling project for it.
   */
  template <int d, typename NT> Point<d> extractor_point(const Point<d> &EoM, const SolutionSample<d, NT> &sample) const
  {
    const auto shock = find_lowest_shock(sample);
    if (!shock) return EoM;
    return sample[std::min(*shock + prm.offset_cells, sample.size() - 1)].point;
  }

  template <int dim_, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim_> &x, const Solutions &sol) const
  {
    const auto &fe_functions = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);
    const auto &extractors = get<"extractors">(sol);
    const auto &variables = get<"variables">(sol);

    const double rho = x[0];
    const double m2Pi = fe_functions[idxf("u")];
    const double m2Sigma = m2Pi + 2. * rho * fe_derivatives[idxf("u")][0];

    auto hdf5 = output.hdf5();
    hdf5.scalar("k", k);
    // Where the readout is: the EoM.
    hdf5.scalar("rho_EoM", rho);
    hdf5.scalar("sigma_EoM", std::sqrt(2. * rho));
    hdf5.scalar("m^2_pi(EoM)", m2Pi);
    hdf5.scalar("m^2_sigma(EoM)", m2Sigma);
    // Where the extractors were: above the shock, when there is one.
    hdf5.scalar("rho_extract", extractors[idxe("rho_x")]);
    hdf5.scalar("m^2_sigma(extract)", extractors[idxe("m2Sigma_x")]);
    hdf5.scalar("A", variables[idxv("A")]);
  }

  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * prm.Lambda;
    k2 = powr<2>(k);
  }

private:
  /**
   * @brief One mode's contribution to the flux, in d = 3 with a Litim regulator.
   *
   * With R_k(q) = (k^2 - q^2) theta(k^2 - q^2) the loop integral is elementary: d_t R = 2 k^2 inside
   * the shell, the propagator is the constant 1/(k^2 + m^2) there, and the shell has volume
   * k^3 / (6 pi^2) in three dimensions. So the potential flows as
   *
   *   d_t V = -(k^5 / (6 pi^2)) sum_modes 1 / (k^2 + m_i^2) ,
   *
   * and u = d_rho V obeys the conservation law d_t u = d_rho(flux) with the flux below. No
   * threshold-function library and no generated kernel needed -- which is the point of doing this
   * tutorial in d = 3 with this regulator.
   */
  template <typename NT> NT loop(const NT m2) const { return powr<5>(k) / (6. * powr<2>(M_PI) * (k2 + m2)); }

  /// Median of the slopes in [first, last), leaving `values` reordered.
  static double median_of(std::vector<double> &values)
  {
    if (values.empty()) return 0.;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
  }

  /// Index of the sample just above the lowest kink of u, or nothing if u is smooth.
  template <int d, typename NT> std::optional<size_t> find_lowest_shock(const SolutionSample<d, NT> &sample) const
  {
    const size_t n = sample.size();
    if (n < 2 * (prm.width + prm.gap) + 3) return std::nullopt;

    // Slopes between neighbouring cell centres. slope[i] lives between samples i and i+1.
    std::vector<double> slope(n - 1);
    for (size_t i = 0; i + 1 < n; ++i)
      slope[i] = (sample[i + 1].values[0] - sample[i].values[0]) / (sample[i + 1].point[0] - sample[i].point[0]);

    // Reused across cells: median_of reorders its argument, so it needs a scratch copy, but this
    // runs on every residual evaluation and there is no reason to reallocate one per cell.
    std::vector<double> left, right, neighbourhood;
    std::vector<double> step(n, 0.), variation(n, 0.);
    for (size_t i = prm.width + prm.gap; i + prm.width + prm.gap < n; ++i) {
      left.assign(slope.begin() + (i - prm.width - prm.gap), slope.begin() + (i - prm.gap));
      right.assign(slope.begin() + (i + prm.gap), slope.begin() + (i + prm.gap + prm.width));
      step[i] = std::abs(median_of(right) - median_of(left));
      // Slope variation actually realised across the transition zone: the cells the two medians
      // skip over, centred on the candidate. It has to span the whole zone -- measured over too
      // narrow a window it undercounts a real kink and the locality test below rejects it.
      const size_t vlo = i - std::min(i, size_t(prm.gap));
      const size_t vhi = std::min(i + prm.gap + 1, slope.size() - 1);
      for (size_t j = vlo; j < vhi; ++j)
        variation[i] += std::abs(slope[j + 1] - slope[j]);
    }

    for (size_t i = prm.edge_cells + prm.width + prm.gap; i + prm.width + prm.gap < n; ++i) {
      if (!(step[i] > 0.)) continue;
      // A change of grid spacing is a step in d_rho u as far as this is concerned, so a candidate
      // whose window straddles a block boundary says nothing about the solution. SolutionSample
      // carries the local spacing, which makes this a direct test rather than a hard rho cutoff.
      if (std::abs(sample[i + prm.width + prm.gap].cell_width - sample[i - prm.width - prm.gap].cell_width) >
          1e-12 * sample[i].cell_width)
        continue;
      // A kink is a local maximum of the step ...
      const size_t lo = i - prm.width, hi = std::min(i + prm.width + 1, n);
      if (*std::max_element(step.begin() + lo, step.begin() + hi) > step[i]) continue;
      // ... that stands out from the smooth curvature around it ...
      const size_t wlo = i - std::min(i, size_t(prm.window)), whi = std::min(i + prm.window + 1, n);
      neighbourhood.assign(step.begin() + wlo, step.begin() + whi);
      if (!(step[i] > prm.contrast * median_of(neighbourhood))) continue;
      // ... and is actually realised across the transition zone rather than built up by a shoulder.
      if (!(variation[i] > prm.locality * step[i])) continue;
      return i;
    }
    return std::nullopt;
  }
};
