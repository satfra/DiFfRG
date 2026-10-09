#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/interpolation.hh>
#include <DiFfRG/physics/loop_integrals.hh>
#include <DiFfRG/physics/regulators.hh>

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace DiFfRG;

// ---------------------------------------------------------------------------------------------
// Convergence study for the Matsubara (Monien) rule and its T=0 fallback.
//
// predict_size() encodes one answer to "how many nodes does this need":
//     E_max = monien_reach * precision_factor * typical_E,   N = 5 + sqrt(4 E_max / (pi^2 T))
// i.e. the rule is sized to reach monien_reach times the typical energy scale in frequency.
// This study measures the required N directly, per tail behaviour, so that constant and the
// per-model dial /integration/matsubara_precision_factor rest on measurement rather than a guess.
//
// Part 13 is the one that measures the quantity the SUM-vs-INTEGRAL handover trades away; every
// other part measures a rule against itself at larger N.
//
// Not registered with CTest (setup_benchmark): run the binary by hand.
// ---------------------------------------------------------------------------------------------

namespace
{
  constexpr double target_precision = 1e-9;

  // The reach of the Monien rule with N nodes, x_max ~ pi^2 T N^2 / 4, expressed as a multiple of
  // E -- i.e. the reach constant that would produce that N at precision_factor = 1.
  double equivalent_reach(const int N, const double T, const double E)
  {
    const double n = std::max(1., double(N) - 5.);
    return (M_PI * M_PI * T * n * n / 4.) / E;
  }

  struct Family {
    std::string name;
    std::string tail;
    std::function<double(double)> f;                 // summand, as a function of the frequency
    std::function<double(double, double)> reference; // (T, E) -> exact bosonic sum; T=0 -> integral
  };

  // T sum_n 1/(w_n^2 + E^2) = coth(E/2T) / (2E),  and  int dp0/(2pi) 1/(p0^2+E^2) = 1/(2E).
  double ref_pole1(const double T, const double E)
  {
    if (T == 0.) return 1. / (2. * E);
    return 1. / (std::tanh(E / (2. * T)) * 2. * E);
  }

  // T sum_n 1/(w_n^2 + E^2)^2 = -d/dE^2 of the above
  //                           = coth(E/2T)/(4E^3) + csch^2(E/2T)/(8 T E^2),
  // and int dp0/(2pi) 1/(p0^2+E^2)^2 = 1/(4E^3).
  double ref_pole2(const double T, const double E)
  {
    if (T == 0.) return 1. / (4. * powr<3>(E));
    const double x = E / (2. * T);
    const double csch2 = 1. / powr<2>(std::sinh(x));
    return 1. / (std::tanh(x) * 4. * powr<3>(E)) + csch2 / (8. * T * powr<2>(E));
  }

  // The pre-2026-08 vacuum rule, reproduced here so the tangent map can be compared against it
  // directly: 2/3 of the nodes Gauss-Legendre linear on [0, 3E], 1/3 log-spaced on [3E, 1e11 E].
  // Kept local to the study; the library no longer contains it.
  double legacy_vacuum_sum(int m_size, const double typical_E, const std::function<double(double)> &f)
  {
    while (m_size % 3 != 0)
      m_size++;

    DiFfRG::Quadrature<double> quad_up(m_size / 3, QuadratureType::legendre);
    DiFfRG::Quadrature<double> quad_down(m_size / 3 * 2, QuadratureType::legendre);
    const auto up_n = quad_up.nodes<CPU_memory>();
    const auto up_w = quad_up.weights<CPU_memory>();
    const auto dn_n = quad_down.nodes<CPU_memory>();
    const auto dn_w = quad_down.weights<CPU_memory>();

    const long double div = 3 * std::abs(typical_E);
    const long double extent = 1e11 * std::abs(typical_E);
    const long double log_start = std::log(div);
    const long double log_ext = std::log(extent / div);

    double sum = 0.;
    for (int i = 0; i < m_size / 3 * 2; ++i) {
      const double x = dn_n[i] * div;
      const double w = dn_w[i] * div / (2 * M_PI);
      sum += w * (f(x) + f(-x));
    }
    for (int i = 0; i < m_size / 3; ++i) {
      const double x = std::exp(log_start + log_ext * up_n[i]);
      const double w = (up_w[i] * log_ext * x) / (2 * M_PI);
      sum += w * (f(x) + f(-x));
    }
    return sum;
  }

  // Smallest size in `sizes` whose relative error is below `target_precision`, or -1.
  int required_size(const std::vector<int> &sizes, const std::vector<double> &errors)
  {
    for (size_t i = 0; i < sizes.size(); ++i)
      if (errors[i] < target_precision) return sizes[i];
    return -1;
  }

  // -------------------------------------------------------------------------------------------
  // Machinery for Part 13. Kept local because the shipped optimize_x_extent() reads a ConfigTree
  // and caches its answer in a function-local static per (Regulator, dim), so it cannot be asked
  // the same question twice at two tolerances -- which is exactly what Part 13 has to do.
  // -------------------------------------------------------------------------------------------

  /// optimize_x_extent()'s growth loop, with the tolerance as an argument rather than config.
  template <typename Regulator> double x_extent_for(const double tolerance)
  {
    constexpr uint order = 32, quadrature_factor = 8;
    const auto shape = [](const double x) { return Regulator::RBdot(1., x) / (x + Regulator::RB(1., x)); };

    const dealii::QGauss<1> q1(quadrature_factor * 1 * order);
    const dealii::QGauss<1> q2(quadrature_factor * 2 * order);
    const dealii::QGauss<1> q3(quadrature_factor * 10 * order);

    // A HARD-CUTOFF regulator is invisible to the growth loop below: its jump sits INSIDE the
    // Gauss-Legendre rules of I2 and I3, so the convergence test never falls below the ~1/N the
    // jump costs and the extent grows without bound (the shipped optimize_x_extent() has the same
    // blind spot; nothing instantiates it on Litim). Where the shape drops to zero discontinuously
    // the support simply IS the extent.
    {
      double last_live = 0.;
      for (double x = 1e-3; x < 1e4; x *= 1.02)
        if (shape(x) != 0.) last_live = x;
      if (last_live > 0. && shape(last_live) > 1e-3 * shape(0.)) return last_live * 1.02;
    }

    double x_extent = 1., eps1 = 1., eps2 = 1.;
    while (eps1 > tolerance || eps2 > tolerance) {
      const double I1 = LoopIntegrals::integrate<double, 4>(shape, q1, x_extent, 1.);
      const double I2 = LoopIntegrals::integrate<double, 4>(shape, q2, 2. * x_extent, 1.);
      const double I3 = LoopIntegrals::integrate<double, 4>(shape, q3, 10. * x_extent, 1.);
      eps1 = std::abs((I2 - I1) / I1);
      eps2 = std::abs((I2 - I3) / I2);
      if (eps1 > tolerance || eps2 > tolerance) x_extent *= 1.15;
      if (x_extent > 1e6) break; // a regulator with no cutoff at all; caller reports it
    }
    return x_extent;
  }

  /**
   * @brief The production 3+1D summand with the radial direction already integrated out.
   *
   * `A(p0) = int_0^R dl1 l1^2 dtR_B(k^2, q^2) / (q^2 + m2 + R_B(k^2, q^2))^2`, `q^2 = p0^2 + l1^2`,
   * dropping the constant angular prefactor, which cancels from every relative number below.
   *
   * The radial rule is split at the support boundary `l1^2 + p0^2 = R^2`. For a hard-cutoff
   * regulator that boundary is a JUMP inside the radial rule and it moves with `p0`, so the
   * radial error would be a wiggly function of the frequency and would not cancel out of the
   * sum-minus-integral difference this part measures. Production integrates `l1` in one panel and
   * pays that error; here it must be kept out of the aliasing measurement.
   */
  template <typename Regulator> struct AssembledSummand {
    double k, m2, R;
    const dealii::QGauss<1> &gl;

    double panel(const double p02, const double lo, const double hi) const
    {
      if (!(hi > lo)) return 0.;
      const double k2 = powr<2>(k);
      const auto &n = gl.get_points();
      const auto &w = gl.get_weights();
      double acc = 0.;
      for (unsigned i = 0; i < gl.size(); ++i) {
        const double l1 = lo + n[i][0] * (hi - lo);
        const double q2 = p02 + powr<2>(l1);
        acc += w[i] * (hi - lo) * powr<2>(l1) * Regulator::RBdot(k2, q2) / powr<2>(q2 + m2 + Regulator::RB(k2, q2));
      }
      return acc;
    }

    double operator()(const double p0) const
    {
      const double p02 = powr<2>(p0);
      const double edge = std::min(R, std::sqrt(std::max(0., powr<2>(R) - p02)));
      return panel(p02, 0., edge) + panel(p02, edge, R);
    }
  };

  /// `T sum_n A(2 pi n T)` over every mode that carries anything, i.e. the sum itself.
  template <typename F> double bosonic_sum(const F &A, const double T, const double reach)
  {
    const int n_max = int(std::ceil(reach / (2. * M_PI * T)));
    double sum = T * A(0.);
    for (int n = 1; n <= n_max; ++n)
      sum += 2. * T * A(2. * M_PI * T * n);
    return sum;
  }

  /// `(1/2pi) int dp0 A(p0)` over [-reach, reach], converged: two Gauss-Legendre panels, so the
  /// support edge at `R` is a panel boundary rather than something a single rule has to resolve.
  template <typename F> double vacuum_integral(const F &A, const double R, const double reach)
  {
    const dealii::QGauss<1> gl(300);
    const auto &n = gl.get_points();
    const auto &w = gl.get_weights();

    double total = 0.;
    const double edges[3] = {0., R, reach};
    for (int panel = 0; panel < 2; ++panel) {
      const double lo = edges[panel], len = edges[panel + 1] - lo;
      for (unsigned i = 0; i < gl.size(); ++i)
        total += w[i] * len * A(lo + n[i][0] * len);
    }
    return 2. * total / (2. * M_PI);
  }

  /// Least-squares slope of log10(y) against x, over the points where y is above the noise floor.
  double log_slope(const std::vector<double> &x, const std::vector<double> &y)
  {
    double sx = 0., sy = 0., sxx = 0., sxy = 0.;
    int n = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      if (!(y[i] > 1e-14)) continue;
      const double lx = x[i], ly = std::log10(y[i]);
      sx += lx;
      sy += ly;
      sxx += lx * lx;
      sxy += lx * ly;
      ++n;
    }
    if (n < 2) return 0.;
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
  }

  /**
   * @brief One block of Part 13: the sum-vs-integral gap against modes_below, for one regulator
   * at one x_extent_tolerance.
   *
   * The gap IS the aliasing sum of the Poisson formula, i.e. exactly the error the Monien ->
   * vacuum handover commits, and `modes_below = R / (2 pi T)` is the control variable the budget
   * rule selects on. Both fits are reported because they answer different questions: a slope in
   * `modes` that is roughly constant means exponential decay (and the extrapolations of the
   * design's error figures are legitimate), a slope in `log10(modes)` near -1 would mean the
   * flat-top `O(T/R)` law and no extrapolation is allowed.
   */
  template <typename Regulator> void aliasing_table(const char *name, const double tolerance)
  {
    const double k = 1.;
    const double x_extent = x_extent_for<Regulator>(tolerance);
    const double R = std::sqrt(x_extent) * k;
    const double reach = 3. * R; // beyond the support; what the summand still carries out there
    const dealii::QGauss<1> radial(256);

    std::printf("\n-- %-22s x_extent_tolerance %6.0e -> x_extent %8.4f, R = %7.4f\n", name, tolerance, x_extent, R);

    const std::vector<double> ratios = {5., 10., 15., 20., 25., 30., 40., 50., 60., 80., 100., 140., 200., 300.};
    const std::vector<double> masses = {0., -0.9};

    // Truncation at R -- the SECOND, independent error: what cutting the frequency integral at
    // the support radius loses, regardless of node count. Swept over x_extent_tolerance because
    // the shipped examples run that dial at 1e-3/1e-4 rather than its 1e-5 default, and the
    // question is whether that choice, not the aliasing, is what limits a compact-path kernel.
    std::printf("   %-10s %10s %12s %12s\n", "x_ext_tol", "x_extent", "trunc m2=0", "trunc m2=-0.9");
    for (const double tol : {1e-2, 1e-3, 1e-4, 1e-5}) {
      const double xe = x_extent_for<Regulator>(tol);
      const double Rt = std::sqrt(xe) * k;
      double t[2];
      for (int i = 0; i < 2; ++i) {
        const AssembledSummand<Regulator> A{k, masses[i] * powr<2>(k), Rt, radial};
        const double full = vacuum_integral(A, Rt, 3. * Rt);
        const double cut = vacuum_integral(A, Rt, Rt);
        t[i] = std::abs(full - cut) / std::abs(full);
      }
      std::printf("   %10.0e %10.4f %12.2e %12.2e\n", tol, xe, t[0], t[1]);
    }

    // predict_size's sizing law, with the reach as an argument rather than baked in.
    const auto N_from_reach = [](const double E_max, const double T) {
      int n = 5 + int(std::sqrt(4. * E_max / (M_PI * M_PI * T)));
      n = (int)std::ceil(n / 2.) * 2;
      return std::max(8, std::min(128, n));
    };

    // The reference for every rule below is the sum itself.
    const AssembledSummand<Regulator> A9{k, -0.9 * powr<2>(k), R, radial};

    std::printf("%6s %6s %12s %12s | %6s %6s %11s %6s %11s %8s\n", "k/T", "modes", "gap m2=0", "gap m2=-.9", "n_ex",
                "N_cmp", "err_cmp", "N_400", "err_400", "rule");
    std::vector<double> modes_x, gap9;
    for (const double r : ratios) {
      const double T = k / r;
      const int modes = int(std::floor(R / (2. * M_PI * T)));

      double g[2];
      for (int i = 0; i < 2; ++i) {
        const AssembledSummand<Regulator> A{k, masses[i] * powr<2>(k), R, radial};
        const double S = bosonic_sum(A, T, reach);
        const double I = vacuum_integral(A, R, reach);
        g[i] = std::abs(S - I) / std::abs(S);
      }

      // Does a Monien rule sized by the COMPACT reach (E_max = R, multiplier 1) actually RESOLVE
      // the summand, or does it only span it? The sizing law encodes the span; nothing in it says
      // the nodes land where the structure is. Measure both reaches against the exact sum.
      const double S9 = bosonic_sum(A9, T, reach);
      const int N_cmp = N_from_reach(R, T), N_400 = N_from_reach(400. * k, T);
      MatsubaraQuadrature<double> q_cmp, q_400;
      q_cmp.reinit_with_size(N_cmp, T, k);
      q_400.reinit_with_size(N_400, T, k);
      const double e_cmp = std::abs(q_cmp.sum(A9) - S9) / std::abs(S9);
      const double e_400 = std::abs(q_400.sum(A9) - S9) / std::abs(S9);

      // What the sizing law asks for at this point, for orientation.
      const int n = MatsubaraQuadrature<double>::predict_size(T, k);

      modes_x.push_back(double(modes));
      gap9.push_back(g[1]);
      std::printf("%6.4g %6d %12.3e %12.3e | %6d %6d %11.2e %6d %11.2e %8s\n", r, modes, g[0], g[1], modes + 1, N_cmp,
                  e_cmp, N_400, e_400, n > 128 ? "vacuum" : "monien");
    }

    std::vector<double> log_modes;
    for (const double m : modes_x)
      log_modes.push_back(std::log10(std::max(1., m)));
    std::printf("   decay of the m2 = -0.9 gap: %+.3f decades/mode, %+.2f in log10(modes)\n", log_slope(modes_x, gap9),
                log_slope(log_modes, gap9));

    // Reach calibration on the summand production actually integrates, rather than on a bare
    // pole. C = 1 is "the support IS the reach"; the rest are the candidates Part 8 sweeps.
    std::printf("   Monien reach calibration, m2 = -0.9 k^2 (nodes -> rel. error vs the exact sum)\n");
    std::printf("   %6s", "k/T");
    const std::vector<double> Cs = {1., 20., 100., 400., 2000.};
    for (const double C : Cs)
      std::printf(" %14s", (std::string("C=") + std::to_string(int(C))).c_str());
    std::printf("\n");
    for (const double r : ratios) {
      const double T = k / r;
      const double S9 = bosonic_sum(A9, T, reach);
      std::printf("   %6.4g", r);
      for (const double C : Cs) {
        const int N = N_from_reach(C == 1. ? R : C * k, T);
        MatsubaraQuadrature<double> q;
        q.reinit_with_size(N, T, k);
        std::printf(" %5d:%8.1e", N, std::abs(q.sum(A9) - S9) / std::abs(S9));
      }
      std::printf("\n");
    }
  }

  struct PolyExp2Opts {
    static constexpr int order = 2;
  };
  struct PolyExp16Opts {
    static constexpr int order = 16;
  };
} // namespace

TEST_CASE("Matsubara convergence study", "[.study][matsubara][quadrature]")
{
  DiFfRG::Init();

  using Regulator = PolynomialExpRegulator<>;

  // The regulated (4D) family: dtR_B(k^2, w^2 + l1^2) / (w^2 + l1^2 + m2 + RB)^2 with k = E.
  // No closed form, so the reference is the rule's own high-N limit -- legitimate here because
  // the question is the convergence rate, not the value. l1 and m2 are swept because a real
  // kernel integrates over l1 too, and the pole in w sits at w^2 = -(l1^2 + m2 + RB): the
  // hardest configuration is the one with the smallest such energy.
  auto regulated = [](const double E, const double l1, const double m2) {
    return [E, l1, m2](const double w) {
      const double k2 = powr<2>(E);
      const double q2 = powr<2>(w) + powr<2>(l1);
      return Regulator::RBdot(k2, q2) / powr<2>(q2 + m2 + Regulator::RB(k2, q2));
    };
  };
  // l1 in units of k, and m2 in units of k^2.
  const std::vector<double> l1_over_k = {0., 0.5, 1., 2., 4.};
  const std::vector<double> m2_over_k2 = {0., 1., -0.9};

  const std::vector<int> sizes = {4,  6,  8,  10,  12,  16,  20,  24,  28,  32,  40,  48,  56,
                                  64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512};
  const std::vector<double> ratios = {0.1, 0.3, 1., 3., 10., 30., 78., 100., 300., 1000.};

  const double E = 1.0;

  // -------------------------------------------------------------------------------------------
  // Part 1: the Monien branch, closed-form families.
  // -------------------------------------------------------------------------------------------
  for (const auto &fam : std::vector<Family>{
           {"1/(w^2+E^2)", "1/w^2", [E](double w) { return 1. / (powr<2>(w) + powr<2>(E)); }, ref_pole1},
           {"1/(w^2+E^2)^2", "1/w^4", [E](double w) { return 1. / powr<2>(powr<2>(w) + powr<2>(E)); }, ref_pole2}}) {
    std::printf("\n=== Monien branch, summand %s  (tail %s), E = %g ===\n", fam.name.c_str(), fam.tail.c_str(), E);
    std::printf("%10s %10s %10s %12s %14s\n", "E/T", "N_needed", "N_predict", "err@N_pred", "reach needed");

    for (const double ratio : ratios) {
      const double T = E / ratio;

      std::vector<double> errors;
      for (const int N : sizes) {
        MatsubaraQuadrature<double> mq;
        mq.reinit_with_size(N, T, E);
        const double result = mq.sum(fam.f);
        const double ref = fam.reference(T, E);
        errors.push_back(std::abs(result - ref) / std::abs(ref));
      }

      const int n_needed = required_size(sizes, errors);

      // What the shipped heuristic asks for, and what it actually achieves.
      MatsubaraQuadrature<double> shipped;
      shipped.reinit(T, E, 2, 8, 128, 64, 1.);
      const double shipped_err = std::abs(shipped.sum(fam.f) - fam.reference(shipped.get_T(), E)) /
                                 std::abs(fam.reference(shipped.get_T(), E));

      std::printf("%10.3g %10d %10zu %12.3e %14.3g\n", ratio, n_needed, shipped.size(), shipped_err,
                  n_needed > 0 ? equivalent_reach(n_needed, T, E) : -1.);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 2: the regulated (4D) family -- the case where the 2e3 reach is expected to be waste.
  // Reference is the rule's own value at the largest size.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== Monien branch, regulated summand RBdot/(q2+m2+RB)^2 (PolynomialExpRegulator), k = %g ===\n", E);
  std::printf("(N_needed is the worst case over l1/k in {0,0.5,1,2,4} and m2/k^2 in {0,1,-0.9})\n");
  std::printf("%10s %10s %10s %14s %16s\n", "E/T", "N_needed", "N_predict", "reach needed", "worst (l1,m2)");
  for (const double ratio : ratios) {
    const double T = E / ratio;

    int worst_n = -1;
    double worst_l1 = 0., worst_m2 = 0.;
    for (const double l1 : l1_over_k)
      for (const double m2 : m2_over_k2) {
        const auto f = regulated(E, l1 * E, m2 * powr<2>(E));

        MatsubaraQuadrature<double> ref_q;
        ref_q.reinit_with_size(sizes.back(), T, E);
        const double ref = ref_q.sum(f);
        if (std::abs(ref) < 1e-300) continue;

        std::vector<double> errors;
        for (const int N : sizes) {
          MatsubaraQuadrature<double> mq;
          mq.reinit_with_size(N, T, E);
          errors.push_back(std::abs(mq.sum(f) - ref) / std::abs(ref));
        }

        const int n = required_size(sizes, errors);
        if (n < 0) { // never converged: strictly worse than any converged case
          worst_n = -1;
          worst_l1 = l1;
          worst_m2 = m2;
          goto done;
        }
        if (n > worst_n) {
          worst_n = n;
          worst_l1 = l1;
          worst_m2 = m2;
        }
      }
  done:

    MatsubaraQuadrature<double> shipped;
    shipped.reinit(T, E, 2, 8, 128, 64, 1.);

    std::printf("%10.3g %10d %10zu %14.3g %8.3g,%7.3g\n", ratio, worst_n, shipped.size(),
                worst_n > 0 ? equivalent_reach(worst_n, T, E) : -1., worst_l1, worst_m2);
  }

  // -------------------------------------------------------------------------------------------
  // Part 3: the T=0 fallback rule. This is the branch a large-Lambda flow spends most of its
  // time in (E/T > ~78), and its unit test only demands 3e-6.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== T=0 fallback rule (reinit_0), E = %g ===\n", E);
  std::printf("%10s %14s %14s\n", "N", "err 1/w^2", "err 1/w^4");
  for (const int N : sizes) {
    MatsubaraQuadrature<double> mq;
    mq.reinit_with_size(N, 0., E);

    const auto f1 = [E](double w) { return 1. / (powr<2>(w) + powr<2>(E)); };
    const auto f2 = [E](double w) { return 1. / powr<2>(powr<2>(w) + powr<2>(E)); };

    const double e1 = std::abs(mq.sum(f1) - ref_pole1(0., E)) / ref_pole1(0., E);
    const double e2 = std::abs(mq.sum(f2) - ref_pole2(0., E)) / ref_pole2(0., E);
    std::printf("%10zu %14.3e %14.3e\n", mq.size(), e1, e2);
  }

  // A pole away from the mapping scale is what the tangent map has to survive: the map is
  // built around typical_E, and a propagator pole at E' != typical_E is a peak of width
  // ~E'/E in theta. This sweep sets vacuum_quad_size.
  std::printf("\n=== T=0 rule, pole at E' != typical_E (map built at typical_E = %g) ===\n", E);
  std::printf("%10s", "N");
  const std::vector<double> pole_ratios = {1., 0.3, 0.1, 0.03, 0.01, 3., 10.};
  for (const double r : pole_ratios)
    std::printf(" %11s%-4.3g", "E'/E=", r);
  std::printf("\n");
  for (const int N : sizes) {
    MatsubaraQuadrature<double> mq;
    mq.reinit_with_size(N, 0., E);
    std::printf("%10zu", mq.size());
    for (const double r : pole_ratios) {
      const double Ep = r * E;
      const auto f = [Ep](double w) { return 1. / (powr<2>(w) + powr<2>(Ep)); };
      std::printf(" %15.3e", std::abs(mq.sum(f) - ref_pole1(0., Ep)) / ref_pole1(0., Ep));
    }
    std::printf("\n");
  }

  // -------------------------------------------------------------------------------------------
  // Part 4: sizing the T=0 rule on realistic integrands. A production kernel is not a single
  // pole: it is a product of propagators at different energies times a regulator insertion.
  // This is what decides vacuum_quad_size.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== T=0 rule on realistic integrands (typical_E = k = %g) ===\n", E);
  {
    // Two poles a decade apart: int dp0/(2pi) 1/((p0^2+a^2)(p0^2+b^2)) = 1/(2ab(a+b)).
    const auto two_pole = [](const double a, const double b) {
      return [a, b](const double p0) { return 1. / ((powr<2>(p0) + powr<2>(a)) * (powr<2>(p0) + powr<2>(b))); };
    };
    const auto two_pole_ref = [](const double a, const double b) { return 1. / (2. * a * b * (a + b)); };

    // The actual vacuum kernel shape: dtR_B / (q^2 + R_B)^2 at l1 = 0, k = E.
    const auto reg = [E](const double p0) {
      const double k2 = powr<2>(E);
      const double q2 = powr<2>(p0);
      return Regulator::RBdot(k2, q2) / powr<2>(q2 + Regulator::RB(k2, q2));
    };
    MatsubaraQuadrature<double> reg_ref_q;
    reg_ref_q.reinit_with_size(512, 0., E);
    const double reg_ref = reg_ref_q.sum(reg);

    // Reference for the regulated case from the legacy rule too, so neither rule is graded
    // against its own limit.
    const double reg_ref_legacy = legacy_vacuum_sum(4096, E, reg);
    std::printf("(regulated reference: tangent@512 = %.15e, legacy@4096 = %.15e, rel diff %.3e)\n", reg_ref,
                reg_ref_legacy, std::abs(reg_ref - reg_ref_legacy) / std::abs(reg_ref));

    std::printf("%8s %14s %14s %14s %14s | %14s %14s\n", "N", "poles k,10k", "poles k,100k", "poles 0.1k,k",
                "regulated", "LEGACY reg", "LEGACY k,10k");
    for (const int N : {24, 32, 40, 48, 56, 64, 96, 128}) {
      MatsubaraQuadrature<double> mq;
      mq.reinit_with_size(N, 0., E);

      const auto rel = [&](double got, double want) { return std::abs(got - want) / std::abs(want); };
      std::printf("%8zu %14.3e %14.3e %14.3e %14.3e | %14.3e %14.3e\n", mq.size(),
                  rel(mq.sum(two_pole(E, 10. * E)), two_pole_ref(E, 10. * E)),
                  rel(mq.sum(two_pole(E, 100. * E)), two_pole_ref(E, 100. * E)),
                  rel(mq.sum(two_pole(0.1 * E, E)), two_pole_ref(0.1 * E, E)), rel(mq.sum(reg), reg_ref_legacy),
                  rel(legacy_vacuum_sum(N, E, reg), reg_ref_legacy),
                  rel(legacy_vacuum_sum(N, E, two_pole(E, 10. * E)), two_pole_ref(E, 10. * E)));
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 5: the number that actually decides vacuum_quad_size. In production the p0 rule is one
  // axis of Integrator_fT_p2*: the spatial integral over l1 carries the measure l1^2 dl1, so the
  // l1 = 0 slice -- where the regulator's edge sits exactly at p0 = k and the p0 rule is at its
  // worst -- contributes with vanishing weight. Measure the error of the *assembled* integral.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== T=0: error of the assembled 3+1D integral (l1^2 dl1 x p0 rule), k = %g ===\n", E);
  {
    const double x_extent = x_extent_for<Regulator>(1e-5); // the value production would use
    const double l1_max = std::sqrt(x_extent) * E;
    const DiFfRG::Quadrature<double> l1_quad(256, QuadratureType::legendre);
    const auto l1_n = l1_quad.nodes<CPU_memory>();
    const auto l1_w = l1_quad.weights<CPU_memory>();

    auto assembled = [&](const int N, bool legacy) {
      double total = 0.;
      for (size_t j = 0; j < l1_n.size(); ++j) {
        const double l1 = l1_n[j] * l1_max;
        const double meas = l1_w[j] * l1_max * powr<2>(l1);
        const auto f = [&](const double p0) {
          const double k2 = powr<2>(E);
          const double q2 = powr<2>(p0) + powr<2>(l1);
          return Regulator::RBdot(k2, q2) / powr<2>(q2 + Regulator::RB(k2, q2));
        };
        if (legacy)
          total += meas * legacy_vacuum_sum(N, E, f);
        else {
          MatsubaraQuadrature<double> mq;
          mq.reinit_with_size(N, 0., E);
          total += meas * mq.sum(f);
        }
      }
      return total;
    };

    const double ref = assembled(512, false);
    std::printf("(reference: tangent@512 = %.15e, legacy@1536 = %.15e)\n", ref, assembled(1536, true));
    std::printf("%8s %16s %16s\n", "N", "tangent", "legacy");
    for (const int N : {16, 24, 32, 40, 48, 56, 64, 96}) {
      std::printf("%8d %16.3e %16.3e\n", N, std::abs(assembled(N, false) - ref) / std::abs(ref),
                  std::abs(assembled(N, true) - ref) / std::abs(ref));
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 6: node coverage against a stored Matsubara grid.
  //
  // Kernels that read a dressing off a frequency grid (QCD_current: BosonicMatsubaraValues /
  // BosonicCoordinates1DFiniteT with p0_grid_size = 8) are evaluated at the Monien nodes, which
  // are continuous and reach ~pi^2 T N^2 / 4. Two things happen off the end of that grid:
  //   - make_interpolation_stencil clamps a non-periodic axis, so the dressing is frozen at its
  //     boundary value (safe, but it is an approximation nobody chose explicitly);
  //   - BosonicMatsubaraValues::backward returns an *integer* index (round(y/(2 pi T))), so the
  //     stencil weight t is always 0 and the read is nearest-mode, never interpolated.
  // This prints how much of the quadrature actually sees stored data.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== Monien node coverage of a stored Matsubara grid ===\n");
  {
    const double T = 0.105; // QCD_current
    const int grid_modes = 8;
    std::printf("grid: %d bosonic modes (n = 0..%d), i.e. |p0| <= %.3g GeV at T = %g\n", grid_modes, grid_modes - 1,
                (grid_modes - 1) * 2 * M_PI * T, T);
    std::printf("%8s %14s %14s %12s %12s\n", "N", "max node", "max mode n", "nodes in grid", "fraction");
    for (const int N : {16, 32, 64, 96, 128}) {
      MatsubaraQuadrature<double> mq;
      mq.reinit_with_size(N, T, 1.0);
      const auto nodes = mq.nodes<CPU_memory>();

      int inside = 0;
      double max_node = 0.;
      for (size_t i = 0; i < nodes.size(); ++i) {
        max_node = std::max(max_node, (double)nodes[i]);
        if (std::round(nodes[i] / (2 * M_PI * T)) <= grid_modes - 1) inside++;
      }
      std::printf("%8zu %14.4g %14.4g %12d %11.1f%%\n", mq.size(), max_node, max_node / (2 * M_PI * T), inside,
                  100. * inside / (double)mq.size());
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 7: how a dressing on a Matsubara-valued axis actually responds to a continuous
  // frequency. Data is set to f[n] = n, so an interpolating read at p0 = r * 2 pi T must return
  // r, while a nearest-mode read returns round(r), and a clamped read returns 7 beyond the grid.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== reading a BosonicMatsubaraValues dressing at a continuous frequency ===\n");
  {
    const double T = 0.105;
    const int grid_modes = 8;
    BosonicMatsubaraValues<int, float> coords(0, grid_modes, (float)T);
    LinearInterpolator1D<double, BosonicMatsubaraValues<int, float>> dressing(coords);
    std::vector<double> data(grid_modes);
    for (int n = 0; n < grid_modes; ++n)
      data[n] = n; // f[n] = n
    dressing.update(data.data());

    std::printf("%12s %14s %14s %14s\n", "p0/(2 pi T)", "exact f = r", "read back", "verdict");
    for (const double r : {0.0, 0.5, 1.0, 3.7, 6.5, 7.0, 20.0, 500.0}) {
      const double p0 = r * 2 * M_PI * T;
      const double got = dressing((float)p0);
      const char *verdict = (std::abs(got - r) < 1e-6)                                      ? "interpolated"
                            : (std::abs(got - grid_modes + 1) < 1e-6 && r > grid_modes - 1) ? "CLAMPED"
                                                                                            : "nearest-mode";
      std::printf("%12.4g %14.4g %14.4g %14s\n", r, r, got, verdict);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 8: choosing the reach constant C in E_max = C * precision_factor * typical_E.
  // The user-facing dial stays at precision_factor = 1; C is what decides whether that means
  // "sane" or "1000x over-provisioned". For each candidate C this reports the N that
  // predict_size would produce and the error actually achieved at that N.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== candidate reach constants C (E_max = C * typical_E), pf = 1 ===\n");
  {
    // predict_size's formula, with C substituted for the shipped 2e3.
    auto N_for = [](const double C, const double r) {
      int n = 5 + int(std::sqrt(4. * C * r / (M_PI * M_PI)));
      n = (int)std::ceil(n / 2.) * 2;
      return std::max(8, std::min(128, n));
    };

    const std::vector<double> Cs = {2000., 400., 200., 100., 50., 20.};
    const std::vector<double> rs = {0.3, 1., 3., 10., 30., 78.};

    std::printf("%8s", "C");
    for (const double r : rs)
      std::printf("  E/T=%-5.3g N near  far  ", r);
    std::printf("\n");

    for (const double C : Cs) {
      std::printf("%8.4g", C);
      for (const double r : rs) {
        const double T = E / r;
        const int N = N_for(C, r);

        MatsubaraQuadrature<double> mq;
        mq.reinit_with_size(N, T, E);

        // reference for the regulated family: same T, converged size
        const auto freg = regulated(E, 0., 0.);
        MatsubaraQuadrature<double> conv;
        conv.reinit_with_size(512, T, E);
        const double reg_ref = conv.sum(freg);

        // Two poles far apart is the realistic stressor: a real kernel carries a heavy mode as
        // well as the light one, and only the light scale is tracked by typical_E = k.
        // T sum 1/((w^2+a^2)(w^2+b^2)) = [coth(a/2T)/(2a) - coth(b/2T)/(2b)] / (b^2 - a^2).
        const auto two_pole = [](const double a, const double b) {
          return [a, b](const double w) { return 1. / ((powr<2>(w) + powr<2>(a)) * (powr<2>(w) + powr<2>(b))); };
        };
        const auto two_pole_ref = [T](const double a, const double b) {
          return (ref_pole1(T, a) - ref_pole1(T, b)) / (powr<2>(b) - powr<2>(a));
        };

        const auto rel = [](double a, double b) { return std::abs(a - b) / std::abs(b); };
        // "near" = everything whose structure sits within a decade of typical_E, including the
        // 4D-regulated kernel shape. "x100" = a heavy mode two decades above typical_E, which
        // only reaches the frequency integral when the regulator does NOT cut p0 (3D regulator).
        const double near =
            std::max({rel(mq.sum([E](double w) { return 1. / (powr<2>(w) + powr<2>(E)); }), ref_pole1(T, E)),
                      rel(mq.sum([E](double w) { return 1. / powr<2>(powr<2>(w) + powr<2>(E)); }), ref_pole2(T, E)),
                      rel(mq.sum(two_pole(E, 10. * E)), two_pole_ref(E, 10. * E)), rel(mq.sum(freg), reg_ref)});
        const double far = rel(mq.sum(two_pole(E, 100. * E)), two_pole_ref(E, 100. * E));
        std::printf(" %4d %8.1e %8.1e", N, near, far);
      }
      std::printf("\n");
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 8b: the same sweep with typical_E set CORRECTLY.
  //
  // Part 8 sizes every rule on typical_E = k, which is what set_k used to force. Its "far" column
  // -- a heavy mode two decades above k -- is therefore measuring what happens when the sizing
  // scale is wrong, and that is what the reach constant of 400 was calibrated against. Now that
  // typical_E is model input and the reach runs on max(k, typical_E), redo it with the sizing
  // scale equal to the heaviest scale actually in the summand, and ask what reach that needs.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== reach constants C with typical_E = the HEAVIEST scale (pf = 1) ===\n");
  {
    auto N_for = [](const double E_max_over_T) {
      int n = 5 + int(std::sqrt(4. * E_max_over_T / (M_PI * M_PI)));
      n = (int)std::ceil(n / 2.) * 2;
      return std::max(8, std::min(128, n));
    };
    const auto two_pole = [](const double a, const double b) {
      return [a, b](const double w) { return 1. / ((powr<2>(w) + powr<2>(a)) * (powr<2>(w) + powr<2>(b))); };
    };

    const std::vector<double> Cs = {400., 200., 100., 50., 20.};
    const std::vector<double> rs = {0.3, 1., 3., 10., 30., 78.};

    std::printf("%8s %10s", "C", "");
    for (const double r : rs)
      std::printf("  k/T=%-5.3g N   err ", r);
    std::printf("\n");

    for (const double C : Cs) {
      // Two poles two decades apart. The heaviest scale is 100k, so that is what a model would
      // now report through set_typical_E, and the rule is sized on C * 100k rather than C * k.
      std::printf("%8.4g %10s", C, "heavy 100k");
      for (const double r : rs) {
        const double T = E / r;
        const int N = N_for(C * 100. * E / T);
        MatsubaraQuadrature<double> mq;
        mq.reinit_with_size(N, T, 100. * E);
        const double ref = (ref_pole1(T, E) - ref_pole1(T, 100. * E)) / (powr<2>(100. * E) - powr<2>(E));
        std::printf(" %4d %8.1e", N, std::abs(mq.sum(two_pole(E, 100. * E)) - ref) / std::abs(ref));
      }
      std::printf("\n");

      // The 4D-regulated family, whose heaviest scale IS k -- so this row is what C costs on the
      // summand every compact-path flow integrates, with nothing misreported.
      std::printf("%8s %10s", "", "regulated");
      for (const double r : rs) {
        const double T = E / r;
        const int N = N_for(C * E / T);
        MatsubaraQuadrature<double> mq, conv;
        mq.reinit_with_size(N, T, E);
        conv.reinit_with_size(512, T, E);
        const auto freg = regulated(E, 0., -0.9 * powr<2>(E));
        const double ref = conv.sum(freg);
        std::printf(" %4d %8.1e", N, std::abs(mq.sum(freg) - ref) / std::abs(ref));
      }
      std::printf("\n");
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 9: the thermal content of a BARE POLE, 2(coth(E/2T) - 1) ~ 4 exp(-E/T).
  //
  // This is the one summand for which the sum-vs-integral gap really is set by the distance of a
  // singularity from the real p0 axis, and the numbers below are correct for it. It does NOT
  // bound the gap for a 4D-regulated summand, which has no pole at all -- q^2 + R_B >= k^2
  // exactly -- and whose aliasing is edge ringing rather than pole decay. Part 13 measures that
  // case directly; it is 1e-4 where this table reads 1e-13.
  //
  // The switch is therefore no longer decided here. It is decided on cost alone, against
  // max_matsubara_size -- see MatsubaraQuadrature::predict_size.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== thermal content vs E/T (what switching to the T=0 rule discards) ===\n");
  std::printf("%10s %16s %16s\n", "E/T", "exact 1/(w^2+E^2)", "4*exp(-E/T)");
  for (const double r : {5., 10., 15., 20., 25., 30., 35., 40.}) {
    const double T = E / r;
    const double thermal = std::abs(ref_pole1(T, E) - ref_pole1(0., E)) / ref_pole1(0., E);
    std::printf("%10.3g %16.3e %16.3e\n", r, thermal, 4. * std::exp(-r));
  }

  // -------------------------------------------------------------------------------------------
  // Part 10: the 3D-regulated kernel shape -- the actual risk case for the reach constant.
  // Here the regulator argument carries no p0, so as a function of frequency the summand is a
  // pure double pole at E = sqrt(l1^2 + m2 + RB(k^2, l1^2)) with a 1/p0^4 tail and NO cutoff.
  // With m2 ~ k^2 that pole sits on-scale; the danger is the IR, where the physical mass stays
  // finite while k -> 0, so E/typical_E grows without bound.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== 3D-regulated summand: dtRB(k^2, l1^2) / (p0^2 + l1^2 + m2 + RB)^2 ===\n");
  {
    auto reg3d = [](const double k, const double l1, const double m2) {
      return [k, l1, m2](const double p0) {
        const double k2 = powr<2>(k);
        const double l12 = powr<2>(l1);
        return Regulator::RBdot(k2, l12) / powr<2>(powr<2>(p0) + l12 + m2 + Regulator::RB(k2, l12));
      };
    };
    // Exact bosonic sum: the summand is c/(p0^2+E^2)^2 with E^2 = l1^2+m2+RB, c = dtRB.
    auto reg3d_ref = [&](const double T, const double k, const double l1, const double m2) {
      const double k2 = powr<2>(k), l12 = powr<2>(l1);
      const double E2 = l12 + m2 + Regulator::RB(k2, l12);
      return Regulator::RBdot(k2, l12) * ref_pole2(T, std::sqrt(E2));
    };

    // The last two columns are the point of the table: the reach constant only has to cover a
    // heavy mode when the model does NOT report it. `typical_E = max(k, m)` is what the integrator
    // now sizes on if the model sets typical_E, and it removes the problem outright -- which is
    // why the constant could come down from 400 to 100.
    std::printf("%8s %8s | %6s %10s %6s %10s %6s %10s | %6s %10s\n", "m/k", "E/T", "N", "reach 400", "N", "reach 100",
                "N", "reach 20", "N", "100, E=m");
    for (const double m_over_k : {0., 1., 10., 100.})
      for (const double r : {1., 10., 30.}) {
        const double k = E, T = k / r;
        const double m2 = powr<2>(m_over_k * k);
        const auto f = reg3d(k, 0.5 * k, m2);
        const double ref = reg3d_ref(T, k, 0.5 * k, m2);

        auto N_for = [](const double E_max_over_T) {
          int n = 5 + int(std::sqrt(4. * E_max_over_T / (M_PI * M_PI)));
          n = (int)std::ceil(n / 2.) * 2;
          return std::max(8, std::min(128, n));
        };
        auto err_for = [&](const int N) {
          MatsubaraQuadrature<double> q;
          q.reinit_with_size(N, T, k);
          return std::abs(q.sum(f) - ref) / std::abs(ref);
        };

        // Sized on typical_E = k, i.e. the model never reported the mass.
        const int n400 = N_for(400. * r), n100 = N_for(100. * r), n20 = N_for(20. * r);
        // Sized on typical_E = max(k, m), i.e. it did.
        const int n_honest = N_for(100. * std::max(1., m_over_k) * r);

        std::printf("%8.4g %8.4g | %6d %10.2e %6d %10.2e %6d %10.2e | %6d %10.2e\n", m_over_k, r, n400, err_for(n400),
                    n100, err_for(n100), n20, err_for(n20), n_honest, err_for(n_honest));
      }
  }

  // -------------------------------------------------------------------------------------------
  // Part 11: the same 3D-regulated kernel along a REAL flow trajectory.
  // m/k and E/T are not independent: a physical mass stays fixed while k runs, so m/k is large
  // exactly where E/T = k/T is small. Sweeping them independently invents a regime
  // (heavy mode AND large E/T) that a flow never visits. Here T and m are fixed and k runs.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== 3D-regulated kernel along a flow: T = 0.1, m = 1 GeV fixed, k running ===\n");
  {
    const double T = 0.1, m = 1.0;
    auto reg3d = [](const double k, const double l1, const double m2) {
      return [k, l1, m2](const double p0) {
        const double k2 = powr<2>(k), l12 = powr<2>(l1);
        return Regulator::RBdot(k2, l12) / powr<2>(powr<2>(p0) + l12 + m2 + Regulator::RB(k2, l12));
      };
    };
    auto N_for = [](const double C, const double r) {
      int n = 5 + int(std::sqrt(4. * C * r / (M_PI * M_PI)));
      n = (int)std::ceil(n / 2.) * 2;
      return std::max(8, std::min(128, n));
    };

    std::printf("%10s %8s %8s | %6s %11s %6s %11s | %6s %11s\n", "k[GeV]", "k/T", "m/k", "N", "reach 400", "N",
                "reach 100", "N", "100, E=m");
    for (const double k : {10., 3., 1., 0.3, 0.1, 0.03, 0.01, 0.006}) {
      const double r = k / T;
      const double m2 = powr<2>(m);
      const double l1 = 0.5 * k;
      const double k2 = powr<2>(k), l12 = powr<2>(l1);
      const double Epole = std::sqrt(l12 + m2 + Regulator::RB(k2, l12));
      const auto f = reg3d(k, l1, m2);
      const double ref = Regulator::RBdot(k2, l12) * ref_pole2(T, Epole);

      auto err_for = [&](const int N) {
        MatsubaraQuadrature<double> q;
        q.reinit_with_size(N, T, k);
        return std::abs(q.sum(f) - ref) / std::abs(ref);
      };
      // Sized on typical_E = k, at the old reach and the shipped one, and then on the honest
      // typical_E = max(k, m). This is the table that decides whether a reach of 100 is safe: it
      // is the only part that keeps m/k and k/T in the relation a flow actually imposes.
      const int n400 = N_for(400., r), n100 = N_for(100., r);
      const int n_honest = N_for(100. * std::max(1., m / k), r);

      std::printf("%10.4g %8.3g %8.3g | %6d %11.2e %6d %11.2e | %6d %11.2e\n", k, r, m / k, n400, err_for(n400), n100,
                  err_for(n100), n_honest, err_for(n_honest));
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 12: the max_matsubara_size clamp is no longer silent. QuadratureIntegrator_fT never
  // reaches it -- it asks for the vacuum rule as soon as the Monien rule stops fitting the budget,
  // which is the same test -- so this is what a caller that goes to the storage directly gets.
  // Provoke it with a raised precision factor: reach 2000 at E/T = 25 asks for 160 nodes against
  // a ceiling of 128. The warning is once per process, so an earlier part may already have spent
  // it; what this checks is that it appears at most once in the whole run.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== clamp diagnostic (expect one warning on stderr, then silence) ===\n");
  {
    for (const double r : {25., 26., 27.}) {
      MatsubaraQuadrature<double> mq;
      mq.reinit(E / r, E, 2, 8, 128, 64, 10.);
      std::printf("  E/T = %g -> size %zu\n", r, mq.size());
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 13: the gap the Monien -> vacuum handover actually commits.
  //
  // Every other part measures a rule against itself at larger N. This one measures the quantity
  // the handover trades away: `T sum_n A(w_n)` against `(1/2pi) int dp0 A(p0)` on the assembled
  // 3+1D integrand, with the same radial rule on both sides so only the frequency treatment is
  // left. By Poisson summation that difference IS the aliasing sum, so this is the error law the
  // sizing rule has to be built on -- and the control variable is modes_below = R/(2 pi T), not
  // E/T and not x_order.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== Part 13: sum vs integral on the assembled 3+1D integrand, k = 1 ===\n");
  {
    aliasing_table<LitimRegulator<>>("Litim", 1e-5);
    aliasing_table<PolynomialExpRegulator<PolyExp2Opts>>("PolynomialExp<2>", 1e-5);
    aliasing_table<PolynomialExpRegulator<>>("PolynomialExp<8>", 1e-5);
    aliasing_table<PolynomialExpRegulator<PolyExp16Opts>>("PolynomialExp<16>", 1e-5);
    aliasing_table<ExponentialRegulator<>>("Exponential<b=2>", 1e-5);
  }

  SUCCEED("study completed");
}
