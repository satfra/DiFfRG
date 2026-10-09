#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/timestepping/linear_solver/UMFPack.hh"
#include "kt_regression_helpers.hh"

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/integration.hh>
#include <DiFfRG/physics/regulators.hh>
#include <DiFfRG/physics/thermodynamics.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 1;
  constexpr double k_ir = 5.0e-2;
  constexpr double sigma_min = 0.0;
  constexpr double sigma_max_over_lambda = 1.5;
  constexpr double delta_sigma = 1.0e-3;
  constexpr double grid_tol = 1.0e-12;

  struct PaperSharedParameters {
    static constexpr double Nc = 3.0;
    static constexpr double Nf = 2.0;
    static constexpr double T = 1.0e-4;
    static constexpr double muq = 0.0;
    static constexpr double hPhi = 3.41;
    static constexpr double cSigma = 0.001695;
  };

  struct PaperVacuumTargets {
    static constexpr double sigma = 0.093;
    static constexpr double mPion = 0.135;
    static constexpr double mSigma = 0.575;
    static constexpr double mQuark = 0.317;
  };

  template <typename Regulator_, int LambdaMilliGeV, int M2PhiMicroGeV2, int LambdaPhiMilli> struct PaperCase {
    using Regulator = Regulator_;
    static constexpr double lambda_uv = static_cast<double>(LambdaMilliGeV) / 1000.0;
    static constexpr double m2Phi = static_cast<double>(M2PhiMicroGeV2) / 100000.0;
    static constexpr double lambdaPhi = static_cast<double>(LambdaPhiMilli) / 1000.0;

    static double final_time() { return std::log(lambda_uv / k_ir); }

    static std::size_t n_cells()
    {
      return static_cast<std::size_t>((sigma_max_over_lambda * lambda_uv - sigma_min) / delta_sigma + 0.5);
    }

    static std::string label() { return "Litim Lambda=" + std::to_string(lambda_uv); }

    static std::string output_prefix()
    {
      std::string prefix = "quark_meson_kt_regression_litim_";
      prefix += std::to_string(LambdaMilliGeV);
      return prefix;
    }
  };

  using LitimLambda06 = PaperCase<LitimRegulator<>, 600, -3006, 19260>;
  using LitimLambda08 = PaperCase<LitimRegulator<>, 800, 19183, 15697>;
  using LitimLambda10 = PaperCase<LitimRegulator<>, 1000, 52987, 11230>;
  using LitimLambda12 = PaperCase<LitimRegulator<>, 1200, 98611, 6561>;

  template <typename Model> class ConstrainExplicitBreakingOrigin
  {
  public:
    template <typename Constraints, typename Context>
    void apply_affine_constraints(Constraints &constraints, const Context &context) const
    {
      const auto candidate = def::internal::select_origin_candidate<"u">(context, context.template support<"u">());
      if (!candidate.has_value()) return;

      constraints.add_line(*candidate);
      constraints.set_inhomogeneity(*candidate, -PaperSharedParameters::cSigma);
    }
  };

  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using NumberType = double;
  using Mesh = RectangularMeshSerial<dim>;
  using Discretization = FV::Discretization<Components, Mesh, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  template <typename Model>
  using ExactJacobianAssembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor>;
  using ImplicitTimeStepper = TimeStepperSUNDIALS_IDA<Discretization>;

  double grid_spacing() { return delta_sigma; }

  template <typename Case> double initial_potential(const double sigma)
  {
    return 0.5 * Case::m2Phi * sigma * sigma + 0.25 * Case::lambdaPhi * std::pow(sigma, 4) -
           PaperSharedParameters::cSigma * sigma;
  }

  void log_bad_bosonic_mass_argument(const char *label, const double x, const double k, const double m2)
  {
    const double regulated_argument = k * k + m2;
    if (std::isfinite(m2) && std::isfinite(regulated_argument) && regulated_argument >= 0.0) return;

    static int pion_logs = 0;
    static int sigma_logs = 0;
    int &log_count = label[0] == 'p' ? pion_logs : sigma_logs;
    if (log_count >= 8) return;
    ++log_count;

    const auto cell = static_cast<std::size_t>(std::max(0.0, std::floor((x - sigma_min) / delta_sigma)));
    std::clog << "[QM DIAG] " << label << " invalid regulated mass argument"
              << " cell=" << cell << " sigma=" << x << " k=" << k << " k2=" << k * k << " m2=" << m2
              << " k2_plus_m2=" << regulated_argument << '\n';
  }

  template <typename REG> class QMPionKernel
  {
  public:
    using Regulator = REG;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double q, const auto k, const auto /*Nc*/, const auto Nf,
                                                   const auto T, const auto /*muq*/, const auto /*hPhi*/,
                                                   const auto m2Pion)
    {
      using namespace DiFfRG;
      using Kokkos::sqrt;

      const auto rb = REG::RB(powr<2>(k), powr<2>(q));
      const auto rbdot = REG::RBdot(powr<2>(k), powr<2>(q));
      const auto energy = sqrt(powr<2>(q) + rb + m2Pion);
      return -0.25 * (-rbdot) * CothFiniteT(energy, T) * (-1. + powr<2>(Nf)) / energy;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto /*k*/, const auto /*Nc*/, const auto /*Nf*/,
                                                     const auto /*T*/, const auto /*muq*/, const auto /*hPhi*/,
                                                     const auto /*m2Pion*/)
    {
      return 0.;
    }
  };

  template <typename REG> class QMSigmaKernel
  {
  public:
    using Regulator = REG;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double q, const auto k, const auto /*Nc*/, const auto /*Nf*/,
                                                   const auto T, const auto /*muq*/, const auto /*hPhi*/,
                                                   const auto m2Sigma)
    {
      using namespace DiFfRG;
      using Kokkos::sqrt;

      const auto rb = REG::RB(powr<2>(k), powr<2>(q));
      const auto rbdot = REG::RBdot(powr<2>(k), powr<2>(q));
      const auto energy = sqrt(powr<2>(q) + rb + m2Sigma);
      return 0.25 * rbdot * CothFiniteT(energy, T) / energy;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto /*k*/, const auto /*Nc*/, const auto /*Nf*/,
                                                     const auto /*T*/, const auto /*muq*/, const auto /*hPhi*/,
                                                     const auto /*m2Sigma*/)
    {
      return 0.;
    }
  };

  template <typename REG> class QMQuarkKernel
  {
  public:
    using Regulator = REG;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double q, const auto k, const auto Nc, const auto Nf,
                                                   const auto T, const auto muq, const auto hPhi, const auto sigma)
    {
      using namespace DiFfRG;
      using Kokkos::sqrt;

      const auto rf = REG::RF(powr<2>(k), powr<2>(q));
      const auto rfdot = REG::RFdot(powr<2>(k), powr<2>(q));
      const auto q_reg = q + rf;
      const auto m2Quark = powr<2>(hPhi * sigma);
      const auto energy = sqrt(powr<2>(q_reg) + m2Quark);
      return -Nc * Nf * q_reg * (-rfdot) * (TanhFiniteT(muq - energy, T) - TanhFiniteT(muq + energy, T)) / energy;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto /*k*/, const auto /*Nc*/, const auto /*Nf*/,
                                                     const auto /*T*/, const auto /*muq*/, const auto /*hPhi*/,
                                                     const auto /*sigma*/)
    {
      return 0.;
    }
  };

  template <typename Kernel> class IntegratorWrapper
  {
  public:
    IntegratorWrapper(QuadratureProvider &quadrature_provider, const ConfigTree &json)
        : integrator(quadrature_provider, json), integrator_ad(quadrature_provider, json),
          integrator_ad2(quadrature_provider, json)
    {
    }

    template <typename... Args> void get(double &dest, Args &&...args) const
    {
      integrator.get(dest, std::forward<Args>(args)...);
    }

    template <typename... Args> void get(autodiff::real &dest, Args &&...args) const
    {
      integrator_ad.get(dest, std::forward<Args>(args)...);
    }

    template <typename... Args> void get(autodiff::Real<2, double> &dest, Args &&...args) const
    {
      integrator_ad2.get(dest, std::forward<Args>(args)...);
    }

    mutable Integrator_p2<3, double, Kernel, TBB_exec> integrator;
    mutable Integrator_p2<3, autodiff::real, Kernel, TBB_exec> integrator_ad;
    mutable Integrator_p2<3, autodiff::Real<2, double>, Kernel, TBB_exec> integrator_ad2;
  };

  template <typename Case> class QuarkMesonFlows
  {
  public:
    using Regulator = typename Case::Regulator;
    using PionIntegrator = IntegratorWrapper<QMPionKernel<Regulator>>;
    using SigmaIntegrator = IntegratorWrapper<QMSigmaKernel<Regulator>>;
    using QuarkIntegrator = IntegratorWrapper<QMQuarkKernel<Regulator>>;

    explicit QuarkMesonFlows(const ConfigTree &json)
        : quadrature_provider(json), pion(quadrature_provider, json), sigma(quadrature_provider, json),
          quark(quadrature_provider, json)
    {
    }

    void set_k(const double k)
    {
      all_set_k(pion, k);
      all_set_k(sigma, k);
      all_set_k(quark, k);
    }

    void set_T(const double T)
    {
      all_set_T(pion, T);
      all_set_T(sigma, T);
      all_set_T(quark, T);
    }

    QuadratureProvider quadrature_provider;
    PionIntegrator pion;
    SigmaIntegrator sigma;
    QuarkIntegrator quark;
  };

  template <typename Case>
  class QuarkMesonKTModel : public def::AbstractModel<QuarkMesonKTModel<Case>, Components>,
                            public def::fRG,
                            public def::OriginShiftedOddLinearExtrapolationBoundaries<QuarkMesonKTModel<Case>>,
                            public ConstrainExplicitBreakingOrigin<QuarkMesonKTModel<Case>>,
                            public def::AD<QuarkMesonKTModel<Case>>
  {
  public:
    explicit QuarkMesonKTModel(const ConfigTree &json) : def::fRG(json), flows(json)
    {
      flows.set_k(this->Lambda);
      flows.set_T(PaperSharedParameters::T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      values[0] = Case::m2Phi * x[0] + Case::lambdaPhi * std::pow(x[0], 3) - PaperSharedParameters::cSigma;
    }

    template <typename NT, std::size_t n_components> std::array<NT, n_components> origin_odd_reflection_values() const
    {
      static_assert(n_components == 1, "QuarkMesonKTModel expects one FE function.");
      return {NT(-PaperSharedParameters::cSigma)};
    }

    void set_time(const double t_)
    {
      this->t = t_;
      k = std::exp(-this->t) * this->Lambda;
      flows.set_k(k);
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "QuarkMesonKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "QuarkMesonKTModel expects one FE function.");

      const auto &fe_functions = get<0>(sol);
      const NT u = fe_functions[0];
      const double sigma_value = x[0];
      const NT sigma_nt = NT(sigma_value);
      const NT m2_pion = (u + NT(PaperSharedParameters::cSigma)) / sigma_nt;
      if constexpr (std::is_same_v<std::decay_t<NT>, double>)
        log_bad_bosonic_mass_argument("pion", sigma_value, k, m2_pion);

      NT pion_loop{};
      flows.pion.get(pion_loop, k, PaperSharedParameters::Nc, PaperSharedParameters::Nf, PaperSharedParameters::T,
                     PaperSharedParameters::muq, PaperSharedParameters::hPhi, m2_pion);

      F_i[0][0] = pion_loop;
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void diffusion_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
                        const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "QuarkMesonKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "QuarkMesonKTModel expects one FE function.");

      const auto &fe_derivatives = get<1>(sol);
      const NT m2_sigma = fe_derivatives[0][0];
      if constexpr (std::is_same_v<std::decay_t<NT>, double>) log_bad_bosonic_mass_argument("sigma", x[0], k, m2_sigma);

      NT sigma_loop{};
      flows.sigma.get(sigma_loop, k, PaperSharedParameters::Nc, PaperSharedParameters::Nf, PaperSharedParameters::T,
                      PaperSharedParameters::muq, PaperSharedParameters::hPhi, m2_sigma);
      // KT residual sums the fluxes: (H + D)·n. The diffusion flux is the physical flux
      // with the same sign as the advection flux (conservation-law / CG convention). The
      // sigma loop is a decreasing function of m²_σ = ∂_σ u, i.e. ∂F_diff/∂(∂_σ u) < 0.
      F_i[0][0] = sigma_loop;
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void source(std::array<NT, n_fe_functions> &s_i, const Point<spatial_dim> &x, const Solutions &) const
    {
      static_assert(spatial_dim == dim, "QuarkMesonKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "QuarkMesonKTModel expects one FE function.");

      const NT sigma_value = NT(x[0]);
      NT source_right{};
      NT source_left{};
      constexpr double source_dx = 1.0e-5;
      flows.quark.get(source_right, k, PaperSharedParameters::Nc, PaperSharedParameters::Nf, PaperSharedParameters::T,
                      PaperSharedParameters::muq, PaperSharedParameters::hPhi, sigma_value + NT(source_dx));
      flows.quark.get(source_left, k, PaperSharedParameters::Nc, PaperSharedParameters::Nf, PaperSharedParameters::T,
                      PaperSharedParameters::muq, PaperSharedParameters::hPhi, sigma_value - NT(source_dx));
      s_i[0] = (source_right - source_left) / NT(2.0 * source_dx);
    }

  private:
    double k = Case::lambda_uv;
    mutable QuarkMesonFlows<Case> flows;
  };

  template <typename Case>
  ConfigTree make_json(const double final_time, const double output_dt = 2.0e-2, const double explicit_dt = 1.0e-8,
                       const double explicit_abs_tol = 1.0e-14, const double explicit_rel_tol = 1.0e-14)
  {
    return json::value(
        {{"physical", {{"Lambda", Case::lambda_uv}, {"cSigma", PaperSharedParameters::cSigma}}},
         {"integration", {{"x_order", 100}, {"x_extent_tolerance", 1.0e-4}, {"jacobian_quadrature_factor", 0.5}}},
         {"discretization",
          {{"fe_order", 0},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 0}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", output_dt},
           {"explicit",
            {{"dt", explicit_dt},
             {"minimal_dt", 1.0e-12},
             {"maximal_dt", 1.0},
             {"abs_tol", explicit_abs_tol},
             {"rel_tol", explicit_rel_tol}}},
           {"implicit",
            {{"dt", 1.0e-9},
             {"minimal_dt", 1.0e-15},
             {"maximal_dt", 1.0e-0},
             {"abs_tol", 1.0e-12},
             {"rel_tol", 1.0e-12},
             {"max_steps", 20000000},
             {"ida_callback_trace", std::is_same_v<Case, LitimLambda12>},
             {"ida_callback_trace_min_t", 3.0},
             {"ida_callback_trace_max_lines", 1200},
             {"ida_callback_trace_successes", true}}}}},
         {"output", {{"verbosity", 1}, {"vtk", false}, {"hdf5", true}}}});
  }

  template <typename Case> Config::ConfigurationMesh<1> make_mesh_config()
  {
    const Config::GridAxis sigma_axis(sigma_min, grid_spacing(), sigma_max_over_lambda * Case::lambda_uv);
    return Config::ConfigurationMesh<1>(0u, std::vector<Config::GridAxis>{sigma_axis});
  }

  template <typename Case, typename AssemblerType, typename TimeStepperType = ImplicitTimeStepper>
  kt_regression::SampledProfile run_flow(const double final_time, const ConfigTree &json,
                                         const std::string &output_prefix = Case::output_prefix())
  {
    kt_regression::ensure_diffrg_initialized();
    QuarkMesonKTModel<Case> model(json);
    Mesh mesh(make_mesh_config<Case>());
    Discretization discretization(mesh, json);
    AssemblerType assembler(discretization, model, json);

    auto data_out_path = OutputPath::temporary(TemporaryRetention::keep, output_prefix, "output");
    std::clog << "[QM DIAG] retaining output directory " << data_out_path.root() << '\n';
    OutputSession<Discretization> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    TimeStepperType time_stepper(json, assembler, data_out, *adaptor);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == Case::n_cells());
    discretization.get_constraints().distribute(state.spatial_data());

    time_stepper.run(state, 0.0, final_time);

    return kt_regression::sample_sorted_profile(state, discretization, grid_tol);
  }

  template <typename Case, typename AssemblerType, typename TimeStepperType = ImplicitTimeStepper>
  kt_regression::SampledProfile run_flow(const double final_time)
  {
    const ConfigTree json = make_json<Case>(final_time);
    return run_flow<Case, AssemblerType, TimeStepperType>(final_time, json);
  }

  std::size_t lower_bracketing_index(const kt_regression::SampledProfile &profile, const double x)
  {
    const auto upper = std::upper_bound(profile.x.begin(), profile.x.end(), x);
    if (upper == profile.x.begin() || upper == profile.x.end()) throw std::runtime_error("x outside sampled profile");
    return static_cast<std::size_t>(std::distance(profile.x.begin(), upper) - 1);
  }

  double interpolate_profile(const kt_regression::SampledProfile &profile, const double x)
  {
    const std::size_t lower = lower_bracketing_index(profile, x);
    const double x0 = profile.x[lower];
    const double x1 = profile.x[lower + 1];
    const double y0 = profile.y[lower];
    const double y1 = profile.y[lower + 1];
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
  }

  double central_profile_slope(const kt_regression::SampledProfile &profile, const double x)
  {
    const auto nearest = std::lower_bound(profile.x.begin(), profile.x.end(), x);
    if (nearest == profile.x.begin() || nearest + 1 == profile.x.end())
      throw std::runtime_error("x outside central-difference stencil");

    const std::size_t center = static_cast<std::size_t>(std::distance(profile.x.begin(), nearest));
    const std::size_t left = center - 1;
    const std::size_t right = center + 1;
    return (profile.y[right] - profile.y[left]) / (profile.x[right] - profile.x[left]);
  }

  double profile_zero_near(const kt_regression::SampledProfile &profile, const double x)
  {
    if (profile.x.size() < 2) throw std::runtime_error("profile too small for zero search");

    double best_zero = std::numeric_limits<double>::quiet_NaN();
    double best_distance = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i + 1 < profile.x.size(); ++i) {
      const double x0 = profile.x[i];
      const double x1 = profile.x[i + 1];
      const double y0 = profile.y[i];
      const double y1 = profile.y[i + 1];

      if (y0 == 0.0) {
        const double distance = std::abs(x0 - x);
        if (distance < best_distance) {
          best_zero = x0;
          best_distance = distance;
        }
      }

      if ((y0 < 0.0 && y1 > 0.0) || (y0 > 0.0 && y1 < 0.0)) {
        const double zero = x0 - y0 * (x1 - x0) / (y1 - y0);
        const double distance = std::abs(zero - x);
        if (distance < best_distance) {
          best_zero = zero;
          best_distance = distance;
        }
      }
    }

    if (!std::isfinite(best_zero)) throw std::runtime_error("profile has no zero crossing");
    return best_zero;
  }

  template <typename Case> void check_case()
  {
    kt_regression::ensure_logger();

    using Model = QuarkMesonKTModel<Case>;
    const auto exact_profile = run_flow<Case, ExactJacobianAssembler<Model>>(Case::final_time());
    REQUIRE(exact_profile.x.size() == Case::n_cells());

    for (const double value : exact_profile.y)
      REQUIRE(std::isfinite(value));

    const double u_at_sigma_vac = interpolate_profile(exact_profile, PaperVacuumTargets::sigma);
    const double pion_mass_at_paper_sigma =
        std::sqrt((u_at_sigma_vac + PaperSharedParameters::cSigma) / PaperVacuumTargets::sigma);
    const double numerical_sigma_vac = profile_zero_near(exact_profile, PaperVacuumTargets::sigma);
    const double u_at_numerical_sigma_vac = interpolate_profile(exact_profile, numerical_sigma_vac);
    const double pion_mass =
        std::sqrt((u_at_numerical_sigma_vac + PaperSharedParameters::cSigma) / numerical_sigma_vac);
    const double sigma_mass = std::sqrt(central_profile_slope(exact_profile, numerical_sigma_vac));
    const double quark_mass = PaperSharedParameters::hPhi * PaperVacuumTargets::sigma;

    CAPTURE(Case::label(), u_at_sigma_vac, pion_mass_at_paper_sigma, numerical_sigma_vac, u_at_numerical_sigma_vac,
            pion_mass, sigma_mass, quark_mass);
    CHECK(u_at_sigma_vac == Catch::Approx(0.0).margin(5.0e-4));
    CHECK(pion_mass == Catch::Approx(PaperVacuumTargets::mPion).margin(2.0e-3));
    CHECK(sigma_mass == Catch::Approx(PaperVacuumTargets::mSigma).margin(2.0e-2));
    CHECK(quark_mass == Catch::Approx(PaperVacuumTargets::mQuark).margin(1.0e-3));
  }

} // namespace

TEST_CASE("KT quark-meson LPA Litim Lambda 0.6 flow reaches final time with exact TVD Jacobian",
          "[FV][KT][QuarkMeson][regression][slow][2601.23005][litim][lambda-0.6]")
{
  check_case<LitimLambda06>();
}

TEST_CASE("KT quark-meson LPA Litim Lambda 0.8 flow reaches final time with exact TVD Jacobian",
          "[FV][KT][QuarkMeson][regression][slow][2601.23005][litim][lambda-0.8]")
{
  check_case<LitimLambda08>();
}

TEST_CASE("KT quark-meson LPA Litim Lambda 1.0 flow reaches final time with exact TVD Jacobian",
          "[FV][KT][QuarkMeson][regression][slow][2601.23005][litim][lambda-1.0]")
{
  check_case<LitimLambda10>();
}

TEST_CASE("KT quark-meson LPA Litim Lambda 1.2 flow reaches final time with exact TVD Jacobian",
          "[FV][KT][QuarkMeson][regression][slow][2601.23005][litim][lambda-1.2]")
{
  check_case<LitimLambda12>();
}
