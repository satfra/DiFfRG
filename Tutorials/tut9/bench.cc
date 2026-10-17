#include <DiFfRG/DiFfRG.hh>

#include "bench_flows/flows.hh"
#include "flows/flows.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace DiFfRG;

/**
 * Times single evaluations of the flows of model.hh, each against a variant generated with other codegen options:
 *   Mesons (group, hoisted)  vs  ZPi + ZSig (two kernels)  vs  MesonsPlain (group, "Hoist" -> False)
 *   V ("PointArguments" with rhoPhi)  vs  VAuto (the default, without rhoPhi)
 * and checks that every variant gives the same numbers. The inputs are smooth model data at one scale, not a flow.
 */

/**
 * Median wall time in ms of each of @p fs. The GPU clock drifts over a run, so the functions take turns: every
 * round times each of them once.
 */
template <typename... F> std::array<double, sizeof...(F)> time_ms(const F &...fs)
{
  constexpr size_t n = sizeof...(F), rounds = 11;
  auto once = [](const auto &f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    Kokkos::fence();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  for (int w = 0; w < 3; ++w)
    (once(fs), ...);
  std::array<std::vector<double>, n> t;
  for (size_t r = 0; r < rounds; ++r) {
    size_t i = 0;
    ((t[i++].push_back(once(fs))), ...);
  }
  std::array<double, n> median;
  for (size_t i = 0; i < n; ++i) {
    std::sort(t[i].begin(), t[i].end());
    median[i] = t[i][rounds / 2];
  }
  return median;
}

double max_rel_diff(const std::vector<double> &a, const std::vector<double> &b)
{
  double d = 0.;
  for (size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]) / (std::abs(b[i]) + 1e-300));
  return d;
}

int main(int argc, char *argv[])
{
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  const auto config = config_helper.get_config();

  constexpr uint p_grid_size = 24, n_matsubara = 4, n_stack = n_matsubara * p_grid_size;
  using Coordinates = BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>;
  const double T = config.get_double("/physical/T");
  const Coordinates coordinates(0, n_matsubara, T, p_grid_size, config.get_double("/discretization/p_grid_min"),
                                config.get_double("/discretization/p_grid_max"),
                                config.get_double("/discretization/p_grid_bias"));

  // Wave functions with some momentum dependence.
  std::vector<double> zpi(n_stack), zsig(n_stack);
  for (uint i = 0; i < n_stack; ++i) {
    const auto [p0, p] = coordinates.forward(coordinates.from_linear_index(i));
    const double q2 = p0 * p0 + p * p;
    zpi[i] = 1. + 0.2 * q2 / (0.1 + q2);
    zsig[i] = 1. + 0.4 * q2 / (0.2 + q2);
  }
  SplineInterpolator1DStack<double, Coordinates> ZPi(coordinates), ZSig(coordinates);
  ZPi.update(zpi.data());
  ZSig.update(zsig.data());

  const double k = 0.3, hPhi = 6., etaPi = 0.4, rhoPhi = 4e-3, d1V = 0.02, d2V = 5., d3V = 50.;
  QMFlows flows(config);
  QMBenchFlows bench(config);
  flows.set_T(T);
  bench.set_T(T);
  flows.set_k(k);
  bench.set_k(k);
  flows.set_typical_E(std::sqrt(k * k + 0.3));
  bench.set_typical_E(std::sqrt(k * k + 0.3));

  std::cout << "Two-point flows on the " << n_stack << "-point (p0, p) stack, T = " << T << ", k = " << k << "\n";
  const auto arguments = device::tie(k, T, hPhi, etaPi, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
  std::vector<double> pi_group(n_stack), sig_group(n_stack), pi_sep(n_stack), sig_sep(n_stack), pi_plain(n_stack),
      sig_plain(n_stack);
  const auto [t_group, t_sep, t_plain] =
      time_ms([&] { flows.Mesons.map({pi_group.data(), sig_group.data()}, coordinates, arguments); },
              [&] {
                DeferredMaps defer;
                bench.ZPi.map(pi_sep.data(), coordinates, arguments);
                bench.ZSig.map(sig_sep.data(), coordinates, arguments);
              },
              [&] { bench.MesonsPlain.map({pi_plain.data(), sig_plain.data()}, coordinates, arguments); });
  std::cout << "  Mesons, group, hoisted      " << t_group << " ms\n"
            << "  ZPi + ZSig, two kernels     " << t_sep << " ms   (max rel. diff "
            << std::max(max_rel_diff(pi_sep, pi_group), max_rel_diff(sig_sep, sig_group)) << ")\n"
            << "  MesonsPlain, not hoisted    " << t_plain << " ms   (max rel. diff "
            << std::max(max_rel_diff(pi_plain, pi_group), max_rel_diff(sig_plain, sig_group)) << ")\n";

  // The potential at the points of a field grid, as the CG assembler's batch would hand them over.
  const size_t n_points = config.get_int("/bench/n_points", 400);
  std::vector<double> rho(n_points), u(n_points), du(n_points), v(n_points), v_auto(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    rho[i] = 1.4e-2 * (i + 0.5) / n_points;
    u[i] = -0.02 + 8. * rho[i];
    du[i] = 8.;
  }
  std::cout << "Potential at " << n_points << " points (rho, u, u' per point)\n";
  const auto [t_v, t_auto] =
      time_ms([&] { flows.V.map_points(v, k, T, hPhi, etaPi, rho, u, du, d3V, ZPi, ZSig); },
              [&] { bench.VAuto.map_points(v_auto, k, T, hPhi, etaPi, rho, u, du, d3V, ZPi, ZSig); });
  std::cout << "  V, PointArguments with rhoPhi   " << t_v << " ms\n"
            << "  VAuto, default PointArguments   " << t_auto << " ms   (max rel. diff " << max_rel_diff(v_auto, v)
            << ")\n";
  return 0;
}
