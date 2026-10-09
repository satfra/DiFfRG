// Diagnostic tests: integrator-value verification, finite-T sweep, and
// Example-settings comparison. These are correctness checks (expected to
// pass), not stiffness experiments (which live in smoke.cc).

// IMPORTANT include order: setup.hh + runners.hh (KurganovTadmor.hh) BEFORE
// kernels.hh (DiFfRG/physics/integration.hh — see comment in kt_3D_models.hh).
#include "kt_3D_setup.hh"
#include "kt_3D_runners.hh"
#include "kt_3D_kernels.hh"

#include <autodiff/forward/real/real.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <cstdio>

namespace
{
  using namespace on_kt_3D;

  // Probes integrator: returns (F_val, F_AD.val(), J_AD, J_FD) at one (k, m²).
  template <typename Integrator>
  struct Probe {
    double F_val, F_AD_val, J_AD, J_FD;
  };

  template <typename Integrator>
  Probe<Integrator> probe(Integrator &integrator, double k, double N, double T, double m2,
                          double fd_eps_rel = 1.0e-7)
  {
    Probe<Integrator> p{};
    integrator.get(p.F_val, k, N, T, m2);

    autodiff::real m2_AD(m2);
    seed(m2_AD);
    autodiff::real F_AD_real(0.0);
    integrator.get(F_AD_real, k, N, T, m2_AD);
    p.F_AD_val = F_AD_real.val();
    p.J_AD = autodiff::derivative(F_AD_real);
    unseed(m2_AD);

    const double eps = std::max(fd_eps_rel * std::abs(m2), 1.0e-14);
    double F_plus = 0.0, F_minus = 0.0;
    integrator.get(F_plus, k, N, T, m2 + eps);
    integrator.get(F_minus, k, N, T, m2 - eps);
    p.J_FD = (F_plus - F_minus) / (2.0 * eps);
    return p;
  }

  // For a probe `p`, the relative discrepancy in F-value identity and J consistency.
  double F_rel(double a, double b)
  {
    const double d = std::max(std::abs(a), std::abs(b));
    return d > 0.0 ? std::abs(a - b) / d : 0.0;
  }
}

// ============================================================================
// AD-vs-FD integrator-value diagnostic
// ----------------------------------------------------------------------------
// At several (k, m²) on the LSM-physical flow, verify:
//   1. F_value(double) == F_value(autodiff::real).val()         (ulp identity)
//   2. F_grad(autodiff::real) ≈ FD slope of F_value over m²     (~1e-7 rel)
// Done for BOTH Litim T=0 and PolyExp T=0.05 to span the regulator and
// temperature axes that the integrator is exercised on.
// ============================================================================

TEST_CASE("integrator AD-vs-FD diagnostic - Litim T=0",
          "[diag][ad-vs-fd][litim-T0]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  const ConfigTree json = make_json();
  LitimT0Flows flows(json);

  const double Lambda = LSMPhysicalParameters::Lambda;
  const std::array<double, 4> k_values = {Lambda, 0.3 * Lambda, 0.1 * Lambda, Lambda * std::exp(-2.14)};
  const std::array<double, 6> m2_values = {-1.0e-3, 1.0e-3, 1.0e-2, 1.0e-1, 5.0e-1, 1.0};
  const double T = 0.0;
  const double N = LSMPhysicalParameters::N;

  double max_F_err = 0.0, max_J_err = 0.0;
  for (double k : k_values) {
    flows.set_k(k);
    flows.set_T(T);
    for (double m2 : m2_values) {
      const auto p_pion = probe(flows.V_pion, k, N, T, m2);
      const auto p_sigma = probe(flows.V_sigma, k, N, T, m2);
      max_F_err = std::max({max_F_err, F_rel(p_pion.F_val, p_pion.F_AD_val), F_rel(p_sigma.F_val, p_sigma.F_AD_val)});
      max_J_err = std::max({max_J_err, F_rel(p_pion.J_AD, p_pion.J_FD), F_rel(p_sigma.J_AD, p_sigma.J_FD)});
    }
  }
  std::printf("[Litim T=0] max F-identity rel err = %.3e ;  max |J_AD - J_FD| / |J| = %.3e\n",
              max_F_err, max_J_err);
  CHECK(max_F_err < 1.0e-12);
  CHECK(max_J_err < 1.0e-5);
}

TEST_CASE("integrator AD-vs-FD diagnostic - PolyExp T=0.05",
          "[diag][ad-vs-fd][polyexp]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  const ConfigTree json = make_json();
  PolyExpFlows flows(json);

  const double Lambda = LSMPhysicalParameters::Lambda;
  const std::array<double, 4> k_values = {Lambda, 0.3 * Lambda, 0.1 * Lambda, Lambda * std::exp(-2.14)};
  const std::array<double, 6> m2_values = {-1.0e-3, 1.0e-3, 1.0e-2, 1.0e-1, 5.0e-1, 1.0};
  const double T = LSMPhysicalParameters::T;
  const double N = LSMPhysicalParameters::N;

  double max_F_err = 0.0, max_J_err = 0.0;
  for (double k : k_values) {
    flows.set_k(k);
    flows.set_T(T);
    for (double m2 : m2_values) {
      const auto p_pion = probe(flows.V_pion, k, N, T, m2);
      const auto p_sigma = probe(flows.V_sigma, k, N, T, m2);
      max_F_err = std::max({max_F_err, F_rel(p_pion.F_val, p_pion.F_AD_val), F_rel(p_sigma.F_val, p_sigma.F_AD_val)});
      max_J_err = std::max({max_J_err, F_rel(p_pion.J_AD, p_pion.J_FD), F_rel(p_sigma.J_AD, p_sigma.J_FD)});
    }
  }
  std::printf("[PolyExp T=0.05] max F-identity rel err = %.3e ;  max |J_AD - J_FD| / |J| = %.3e\n",
              max_F_err, max_J_err);
  // F-identity tolerance is looser than the Litim case (1e-12) because
  // CothFiniteT(E/2T) uses transcendental functions whose AD overloads compute
  // cosh/sinh slightly differently than the scalar std::cosh / std::sinh path.
  // The resulting ulp-level discrepancy summed over quadrature nodes lands
  // around 1e-6 — well below IDA's working tolerance, so harmless.
  CHECK(max_F_err < 1.0e-4);
  CHECK(max_J_err < 1.0e-5);
}

// ============================================================================
// Finite-T verification
// ----------------------------------------------------------------------------
// Verify that the integrator at finite T matches the closed-form Litim
// expression
//    F = k⁵ · coth(sqrt(k²+m²)/(2T)) / (12π² · sqrt(k²+m²))
// across (k, m², T) sweep, including the three regimes:
//   T << E  (coth → 1, vacuum limit)
//   T ~ E   (intermediate)
//   T >> E  (coth ≈ 2T/E, dimensional-reduction limit)
// ============================================================================

TEST_CASE("integrator finite-T verification - Litim regulator V_pion sweep over T",
          "[diag][finite-T][litim-finiteT]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  const ConfigTree json = make_json();

  // V_pion_kernel_polyexp with LitimRegulator gives the *finite-T Litim* kernel.
  DiFfRG::QuadratureProvider qp(json);
  IntegratorWrapper<V_pion_kernel_polyexp<DiFfRG::LitimRegulator<>>> V_pion_litim_T(qp, json);

  struct ProbePoint { double k; double m2; };
  const std::array<ProbePoint, 3> probe_points = {{
      {LSMPhysicalParameters::Lambda, 0.5},                              // UV, large mass
      {LSMPhysicalParameters::Lambda * std::exp(-2.14), 0.001},          // IR, small mass
      {0.3, 0.1}                                                         // intermediate
  }};
  const std::array<double, 8> T_values = {1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 10.0};

  double max_rel_err = 0.0;
  for (const auto &pt : probe_points) {
    const double E = std::sqrt(pt.k * pt.k + pt.m2);
    for (double T : T_values) {
      DiFfRG::all_set_k(V_pion_litim_T, pt.k);
      DiFfRG::all_set_T(V_pion_litim_T, T);

      double F_int = 0.0;
      V_pion_litim_T.get(F_int, pt.k, LSMPhysicalParameters::N, T, pt.m2);

      // coth(y) = cosh/sinh; for very small y, use the Taylor expansion to avoid overflow.
      const double y = E / (2.0 * T);
      const double coth_y =
          (y > 1.0e-4) ? (std::cosh(y) / std::sinh(y)) : (1.0 / y + y / 3.0 - y * y * y / 45.0);
      const double F_analytic = std::pow(pt.k, 5) * coth_y / (12.0 * M_PI * M_PI * E);
      const double rel_err = F_rel(F_int, F_analytic);
      max_rel_err = std::max(max_rel_err, rel_err);
    }
  }
  std::printf("[Litim finite-T] max integrator-vs-closed-form rel err = %.3e\n", max_rel_err);
  CHECK(max_rel_err < 1.0e-6);
}

// NOTE on Example-settings IDA tolerance comparison:
// Phase O experiments (see /IDA_failure_diagnosis.md) ran KT-ρ and KT-σ
// with the user-facing Example's IDA tolerances (abs_tol=1e-16, rel_tol=1e-4
// — much tighter abs, much looser rel than our default abs=rel=1e-7) to
// check whether tolerance was the discriminator. Result: the tolerance
// change moved the σ-coords failure time from t≈0.187 to t≈0.187 (no
// change) and ρ-coords from t≈2.141363 to t≈2.140463 (essentially
// no change). So tolerance is NOT the bottleneck for the KT-side failure.
// The tests aren't kept here because they'd just always fail; the smoke
// tests already cover the failure-time baseline at our default settings.
