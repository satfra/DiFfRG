#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_1ang.hh>
#include <DiFfRG/physics/regulators.hh>

#include <chrono>
#include <cstdio>
#include <vector>

using namespace DiFfRG;

// ---------------------------------------------------------------------------------------------
// What the frequency rule COSTS along a flow, as nodes and as wall time.
//
// Finite-T kernels are fp64-bound, so the node count on the frequency axis is the cost, and the
// selection logic is what sets it. Run this binary before and after a change to the selection and
// diff the tables.
//
// Not registered with CTest (setup_benchmark): run the binary by hand.
// ---------------------------------------------------------------------------------------------

namespace
{
  /// The 4D-regulated bosonic summand, on the Monien/vacuum path (no trait).
  struct TailKernel {
    using Regulator = PolynomialExpRegulator<>;
    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double q, const double, const double q0, const double k,
                                                     const double m2)
    {
      const double k2 = powr<2>(k);
      const double Q2 = powr<2>(q) + powr<2>(q0);
      return Regulator::RBdot(k2, Q2) / powr<2>(Q2 + m2 + Regulator::RB(k2, Q2));
    }
    static KOKKOS_FORCEINLINE_FUNCTION double constant(const double, const double) { return 0.; }
  };

  /// The same summand, declaring the finite extent it actually has: the exact-sum path.
  struct CompactKernel : TailKernel {
    static constexpr bool matsubara_finite_extent = true;
  };

  constexpr double x_extent = 1.5209; // PolynomialExpRegulator<8> at the default tolerance

  template <typename KERNEL> void cost_table(const char *name)
  {
    using Integrator = Integrator_fT_p2_1ang<4, double, KERNEL, GPU_exec>;

    const double T = 0.05;
    const std::array<size_t, 2> grid{{96, 16}};
    constexpr int repeats = 400;

    QuadratureProvider qp;
    Integrator integrator(qp, grid, x_extent, T);
    integrator.set_T(T);

    std::printf("\n-- %s (x_order = %zu, cos_order = %zu, T = %g)\n", name, grid[0], grid[1], T);
    std::printf("%8s %10s %14s %14s %12s\n", "k/T", "p0 nodes", "total nodes", "us / call", "exact sum");

    for (const double r : {5., 10., 20., 30., 40., 50., 60., 80., 93., 100., 140., 200., 400.}) {
      const double k = r * T;
      integrator.set_k(k);

      double sink = 0.;
      // Warm up: the first call at a new size allocates and, on the device path, JITs nothing but
      // does pay a first-touch cost that would otherwise land in the measurement.
      for (int i = 0; i < 20; ++i)
        integrator.get(sink, k, -0.9 * powr<2>(k));

      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < repeats; ++i)
        integrator.get(sink, k, -0.9 * powr<2>(k));
      const auto t1 = std::chrono::steady_clock::now();

      const size_t n_freq = integrator.get_matsubara_size();
      const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / repeats;
      std::printf("%8.4g %10zu %14zu %14.1f %12s\n", r, n_freq, n_freq * grid[0] * grid[1], us,
                  integrator.uses_exact_matsubara_sum() ? "yes" : "no");
    }
  }
} // namespace

TEST_CASE("Matsubara frequency-axis cost", "[.study][matsubara][quadrature]")
{
  DiFfRG::Init();

  cost_table<TailKernel>("no trait -- Monien / vacuum path");
  cost_table<CompactKernel>("matsubara_finite_extent -- exact-sum path");

  SUCCEED("cost table written");
}
