#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "kt_regression_helpers.hh"

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/thermodynamics.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include <catch2/catch_all.hpp>
#include <deal.II/base/numbers.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 1;
  constexpr double pi = std::numbers::pi_v<double>;
  constexpr double d_gamma = 2.0;
  constexpr double h = 1.0;
  constexpr double Lambda = 1.0e5;
  constexpr double k_ir = 1.0e-4;
  constexpr double final_time = 20.72326583694641;
  constexpr double sigma_max = 6.0;
  constexpr std::size_t n_cells = 1000;
  constexpr double delta_sigma = sigma_max / static_cast<double>(n_cells - 1);
  constexpr double abs_tol = 5.0e-14;
  constexpr double rel_tol = 2.0e-14;
  constexpr double grid_tol = 1.0e-12;
  constexpr double mean_field_n_flavors = std::numeric_limits<double>::quiet_NaN();

  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  template <typename Model> using ConstrainUAtOrigin = def::ConstrainOriginSupportPointToZero<"u", Model>;
  using NumberType = double;
  using Mesh = RectangularMeshSerial<dim>;
  using Discretization = FV::Discretization<Components, Mesh, NumberType>;
  using SampledProfile = kt_regression::SampledProfile;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  template <typename Model> using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor>;
  using ImplicitTimeStepper = TimeStepperSUNDIALS_IDA<Discretization>;

  struct FlowCase {
    std::string relative_path;
    std::string filename;
    bool is_mean_field;
    double temperature;
    double chemical_potential;
    double n_flavors;
    double profile_abs_tol;
    double profile_rel_tol;
  };

  std::string describe_flow_case(const FlowCase &flow_case)
  {
    std::ostringstream description;
    description << std::setprecision(17);
    description << "flow=" << (flow_case.relative_path.empty() ? flow_case.filename : flow_case.relative_path)
                << ", kind=" << (flow_case.is_mean_field ? "MF" : "finite-N") << ", T=" << flow_case.temperature
                << ", mu=" << flow_case.chemical_potential << ", N=";
    if (flow_case.is_mean_field)
      description << "MF";
    else
      description << flow_case.n_flavors;
    return description.str();
  }

  double fermion_derivative_factor(const double energy, const double temperature)
  {
    if (temperature == 0.0) return 0.0;
    const double occupation = nF(energy, temperature);
    return powr<-1>(temperature) * occupation * (1.0 - occupation);
  }

  FlowCase make_flow_case(const std::string_view relative_path, const std::string_view filename,
                          const bool is_mean_field, const double temperature, const double chemical_potential,
                          const double n_flavors)
  {
    constexpr double regression_profile_abs_tol = 1.0e-6;
    constexpr double regression_profile_rel_tol = 1.0e-7;
    return FlowCase{std::string(relative_path),
                    std::string(filename),
                    is_mean_field,
                    temperature,
                    chemical_potential,
                    n_flavors,
                    regression_profile_abs_tol,
                    regression_profile_rel_tol};
  }

  template <typename Derived> struct GrossNeveuFlowDescriptor {
    static FlowCase flow_case()
    {
      return make_flow_case(Derived::relative_path, Derived::filename, Derived::is_mean_field, Derived::temperature,
                            Derived::chemical_potential, Derived::n_flavors);
    }
  };

  struct FlowMF_T00625_Mu06 : GrossNeveuFlowDescriptor<FlowMF_T00625_Mu06> {
    static constexpr auto relative_path = "data/flow_MF_T=0.00625,mu=0.6.json";
    static constexpr auto filename = "flow_MF_T=0.00625,mu=0.6.json";
    static constexpr bool is_mean_field = true;
    static constexpr double temperature = 0.00625;
    static constexpr double chemical_potential = 0.6;
    static constexpr double n_flavors = mean_field_n_flavors;
  };

  struct FlowMF_T00125_Mu06 : GrossNeveuFlowDescriptor<FlowMF_T00125_Mu06> {
    static constexpr auto relative_path = "data/flow_MF_T=0.0125,mu=0.6.json";
    static constexpr auto filename = "flow_MF_T=0.0125,mu=0.6.json";
    static constexpr bool is_mean_field = true;
    static constexpr double temperature = 0.0125;
    static constexpr double chemical_potential = 0.6;
    static constexpr double n_flavors = mean_field_n_flavors;
  };

  struct FlowMF_T01_Mu01 : GrossNeveuFlowDescriptor<FlowMF_T01_Mu01> {
    static constexpr auto relative_path = "data/flow_MF_T=0.1,mu=0.1.json";
    static constexpr auto filename = "flow_MF_T=0.1,mu=0.1.json";
    static constexpr bool is_mean_field = true;
    static constexpr double temperature = 0.1;
    static constexpr double chemical_potential = 0.1;
    static constexpr double n_flavors = mean_field_n_flavors;
  };

  struct FlowN16_T00625_Mu06 : GrossNeveuFlowDescriptor<FlowN16_T00625_Mu06> {
    static constexpr auto relative_path = "data/flow_N=16,T=0.00625,mu=0.6.json";
    static constexpr auto filename = "flow_N=16,T=0.00625,mu=0.6.json";
    static constexpr bool is_mean_field = false;
    static constexpr double temperature = 0.00625;
    static constexpr double chemical_potential = 0.6;
    static constexpr double n_flavors = 16.0;
  };

  struct FlowN2_T00625_Mu06 : GrossNeveuFlowDescriptor<FlowN2_T00625_Mu06> {
    static constexpr auto relative_path = "data/flow_N=2,T=0.00625,mu=0.6.json";
    static constexpr auto filename = "flow_N=2,T=0.00625,mu=0.6.json";
    static constexpr bool is_mean_field = false;
    static constexpr double temperature = 0.00625;
    static constexpr double chemical_potential = 0.6;
    static constexpr double n_flavors = 2.0;
  };

  struct FlowN2_T00125_Mu06 : GrossNeveuFlowDescriptor<FlowN2_T00125_Mu06> {
    static constexpr auto relative_path = "data/flow_N=2,T=0.0125,mu=0.6.json";
    static constexpr auto filename = "flow_N=2,T=0.0125,mu=0.6.json";
    static constexpr bool is_mean_field = false;
    static constexpr double temperature = 0.0125;
    static constexpr double chemical_potential = 0.6;
    static constexpr double n_flavors = 2.0;
  };

  struct FlowN2_T01_Mu01 : GrossNeveuFlowDescriptor<FlowN2_T01_Mu01> {
    static constexpr auto relative_path = "data/flow_N=2,T=0.1,mu=0.1.json";
    static constexpr auto filename = "flow_N=2,T=0.1,mu=0.1.json";
    static constexpr bool is_mean_field = false;
    static constexpr double temperature = 0.1;
    static constexpr double chemical_potential = 0.1;
    static constexpr double n_flavors = 2.0;
  };

  struct GraphData {
    std::string label;
    std::string ylabel;
    std::vector<double> x;
    std::vector<double> y;
  };

  GraphData parse_graph(const json::object &graph)
  {
    GraphData result;
    result.label = std::string(graph.at("label").as_string().c_str());
    result.ylabel = std::string(graph.at("ylabel").as_string().c_str());
    result.x = kt_regression::json_array_to_doubles(graph.at("x").as_array());
    result.y = kt_regression::json_array_to_doubles(graph.at("y").as_array());
    return result;
  }

  double parse_time_label(const std::string &label)
  {
    if (!label.starts_with("t=")) throw std::runtime_error("Unexpected graph label: " + label);
    return std::stod(label.substr(2));
  }

  std::vector<double> reconstruct_potential_from_u(const std::vector<double> &u_values)
  {
    std::vector<double> U_values(u_values.size(), 0.0);
    for (std::size_t i = 0; i < u_values.size(); ++i) {
      double sum = 0.0;
      for (std::size_t m = 0; m <= i; ++m) {
        double denominator = 1.0;
        if (m == 0) denominator += 1.0;
        if (m == i) denominator += 1.0;
        sum += u_values[m] / denominator;
      }
      U_values[i] = delta_sigma * sum;
    }
    return U_values;
  }

  template <typename Derived>
  class GrossNeveuKTSourceOnlyBase : public def::AbstractModel<Derived, Components>,
                                     public def::fRG,
                                     public ConstrainUAtOrigin<Derived>,
                                     public def::OriginOddLinearExtrapolationBoundaries<Derived>,
                                     public def::AD<Derived>
  {
  public:
    GrossNeveuKTSourceOnlyBase(const ConfigTree &json, const FlowCase &flow_case) : def::fRG(json), flow_case(flow_case)
    {
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double sigma = x[0];
      const double inverse_sqrt = 1.0 / std::sqrt(1.0 + powr<-2>(Lambda));
      const double prefactor = d_gamma / (2.0 * pi);
      values[0] = prefactor * sigma * (std::atanh(inverse_sqrt) - inverse_sqrt);
    }

    template <int spatial_dim, typename FluxNumberType, typename Solutions, std::size_t n_fe_functions>
    void diffusion_flux(std::array<Tensor<1, spatial_dim, FluxNumberType>, n_fe_functions> &F_i,
                        [[maybe_unused]] const Point<spatial_dim> &x, [[maybe_unused]] const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "GrossNeveuKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "GrossNeveuKTModel expects a single FE function.");
      F_i[0][0] = FluxNumberType(0.0);
    }

    template <int spatial_dim, typename SourceNumberType, typename Solutions, std::size_t n_fe_functions>
    void source(std::array<SourceNumberType, n_fe_functions> &s_i, const Point<spatial_dim> &x,
                [[maybe_unused]] const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "GrossNeveuKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "GrossNeveuKTModel expects a single FE function.");

      const double sigma = x[0];
      const double E_f = std::sqrt(k2 + powr<2>(h * sigma));
      const double A = (d_gamma / pi) * (k3 / 2.0);

      const double n_plus = nF(E_f + flow_case.chemical_potential, flow_case.temperature);
      const double n_minus = nF(E_f - flow_case.chemical_potential, flow_case.temperature);
      const double g = 1.0 - n_plus - n_minus;
      const double dg_dE = fermion_derivative_factor(E_f + flow_case.chemical_potential, flow_case.temperature) +
                           fermion_derivative_factor(E_f - flow_case.chemical_potential, flow_case.temperature);
      const double S = A * sigma * (dg_dE / powr<2>(E_f) - g / powr<3>(E_f));

      // The KT assembler adds source terms to the residual, so the PDE source enters with a minus sign here.
      s_i[0] = static_cast<SourceNumberType>(-S);
    }

  protected:
    FlowCase flow_case;
  };

  class GrossNeveuKTMeanFieldModel : public GrossNeveuKTSourceOnlyBase<GrossNeveuKTMeanFieldModel>
  {
  public:
    using GrossNeveuKTSourceOnlyBase<GrossNeveuKTMeanFieldModel>::GrossNeveuKTSourceOnlyBase;
  };

  class GrossNeveuKTFluxModel : public GrossNeveuKTSourceOnlyBase<GrossNeveuKTFluxModel>
  {
  public:
    using GrossNeveuKTSourceOnlyBase<GrossNeveuKTFluxModel>::GrossNeveuKTSourceOnlyBase;

    template <int spatial_dim, typename FluxNumberType, typename Solutions, std::size_t n_fe_functions>
    void diffusion_flux(std::array<Tensor<1, spatial_dim, FluxNumberType>, n_fe_functions> &F_i,
                        [[maybe_unused]] const Point<spatial_dim> &x, const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "GrossNeveuKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "GrossNeveuKTModel expects a single FE function.");

      const auto &fe_derivatives = get<1>(sol);
      const FluxNumberType m2_sigma = fe_derivatives[0][0];

      using std::sqrt;

      const FluxNumberType E_b = sqrt(FluxNumberType(k2) + m2_sigma);

      // The physical flux is (k³/2) coth(E_b/2T) / E_b with E_b = sqrt(k² + ∂_σ u). Only its
      // σ-derivative enters the residual, so we may subtract the σ-independent value at
      // ∂_σ u = 0 (E_b = k). Doing that *analytically* is essential: the unsubtracted flux
      // carries an O(k²/N) baseline (≈8·10⁸ at k = Λ = 1e5, N = 2) whose round-off,
      // |F|·ε ≈ 2·10⁻⁷, swamps the O(10⁻⁴) physical flux difference between neighbouring
      // faces. That noise floor is ~10⁶× above the IDA tolerances (1e-10) in the UV and is
      // what forces the solver into tiny steps until k drops to ~1e4. Subtracting the
      // baseline is exact for the residual (a constant flux has zero divergence, and it
      // cancels on the ghost side of every boundary face too) and restores full precision.
      //
      //   1/E_b - 1/k     = -∂_σ u / (k E_b (k + E_b))        [written cancellation-free]
      //   coth(E/2T) / E  = 1/E + (coth(E/2T) - 1) / E,  with coth(E/2T) - 1 = 2 n_B(E)
      //
      // The thermal remainder needs no special care: wherever the baseline is large
      // (k >> T) coth(E/2T) is exactly 1.0 in double precision, so the remainder is
      // exactly zero, and by the time it turns on, k — and with it the k³/2 prefactor —
      // is small enough that its round-off is irrelevant.
      const FluxNumberType vacuum_part = -m2_sigma / (FluxNumberType(k) * E_b * (FluxNumberType(k) + E_b));
      FluxNumberType thermal_part(0.0);
      if (flow_case.temperature > 0.0)
        thermal_part = (cothS(E_b, flow_case.temperature) - 1.0) / E_b -
                       FluxNumberType((cothS(k, flow_case.temperature) - 1.0) / k);

      // KT residual sums the fluxes: (H + D)·n, so the diffusion flux is the physical
      // flux with the same sign as the advection flux (conservation-law / CG convention).
      F_i[0][0] =
          FluxNumberType(1.0 / (pi * flow_case.n_flavors)) * FluxNumberType(k3 / 2.0) * (vacuum_part + thermal_part);
    }
  };

  ConfigTree make_json([[maybe_unused]] const FlowCase &flow_case)
  {
    return json::value(
        {{"physical", {{"Lambda", Lambda}}},
         {"discretization",
          {{"fe_order", 0},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1e-10},
           {"EoM_max_iter", 0}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", 1.0},
           {"explicit",
            {{"dt", 1e-8}, {"minimal_dt", 1e-8}, {"maximal_dt", 1.0}, {"abs_tol", 1e-14}, {"rel_tol", 1e-12}}},
           {"implicit",
            {{"dt", 1e-8},
             {"minimal_dt", 1e-8},
             {"maximal_dt", 1.0},
             {"abs_tol", 1e-10},
             {"rel_tol", 1e-10},
             {"max_steps", 5000000}}}}},
         {"output", {{"verbosity", 1}, {"vtk", false}, {"hdf5", false}}}});
  }
  std::filesystem::path fixture_path(const std::string &filename)
  {
    const auto local_fixture = std::filesystem::path(__FILE__).parent_path() / "data" / filename;
    if (std::filesystem::exists(local_fixture)) return local_fixture;

    throw std::runtime_error("Could not find regression fixture: " + filename);
  }

  std::filesystem::path local_fixture_path(const std::string_view relative_path)
  {
    return std::filesystem::path(__FILE__).parent_path() / std::filesystem::path(relative_path);
  }

  std::filesystem::path fixture_path(const FlowCase &flow_case)
  {
    if (!flow_case.relative_path.empty()) {
      const auto local_fixture = local_fixture_path(flow_case.relative_path);
      if (std::filesystem::exists(local_fixture)) return local_fixture;
    }

    return fixture_path(flow_case.filename);
  }

  std::pair<GraphData, GraphData> load_reference_ir_pair(const FlowCase &flow_case)
  {
    const double expected_final_time = std::log(Lambda / k_ir);

    const ConfigTree fixture_json(fixture_path(flow_case).string());
    json::value fixture_value = static_cast<json::value>(fixture_json);
    const auto &root = fixture_value.as_object();
    const auto &graphs = root.at("graphs").as_array();

    REQUIRE(graphs.size() >= 2);

    // The local fixtures store each time slice as an ordered (U, u) pair even though both
    // graphs are labeled with the same ylabel metadata.
    const GraphData reference_U = parse_graph(graphs[graphs.size() - 2].as_object());
    const GraphData reference_u = parse_graph(graphs[graphs.size() - 1].as_object());

    REQUIRE(reference_U.x.size() == n_cells);
    REQUIRE(reference_U.y.size() == n_cells);
    REQUIRE(reference_u.x.size() == n_cells);
    REQUIRE(reference_u.y.size() == n_cells);
    REQUIRE(reference_U.x == reference_u.x);
    REQUIRE(parse_time_label(reference_U.label) == Catch::Approx(expected_final_time).margin(1e-12));
    REQUIRE(parse_time_label(reference_u.label) == Catch::Approx(expected_final_time).margin(1e-12));

    for (std::size_t i = 0; i < reference_u.x.size(); ++i)
      REQUIRE(reference_u.x[i] == Catch::Approx(static_cast<double>(i) * delta_sigma).margin(grid_tol));

    const auto reconstructed_U = reconstruct_potential_from_u(reference_u.y);
    if (const auto mismatch = kt_regression::compare_profiles("fixture U reconstruction", reference_U.x,
                                                              reconstructed_U, reference_U.y, abs_tol, rel_tol))
      FAIL(*mismatch);

    return {reference_U, reference_u};
  }

  struct SimulationResult {
    std::vector<double> x;
    std::vector<double> u;
    std::vector<double> U;
  };

  Config::ConfigurationMesh<1> make_mesh_config()
  {
    const Config::GridAxis sigma_axis(-0.5 * delta_sigma, delta_sigma, sigma_max + 0.5 * delta_sigma);
    return Config::ConfigurationMesh<1>(0u, std::vector<Config::GridAxis>{sigma_axis});
  }

  template <typename Model, typename Stepper> SimulationResult run_flow_to_ir(const FlowCase &flow_case)
  {
    const ConfigTree json = make_json(flow_case);
    Model model(json, flow_case);
    Mesh mesh(make_mesh_config());
    Discretization discretization(mesh, json);
    Assembler<Model> assembler(discretization, model, json);

    auto data_out_path =
        OutputPath::temporary(TemporaryRetention::remove_on_destruction, "gross_neveu_kt_regression", "output");
    OutputSession<Discretization> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    Stepper time_stepper(json, assembler, data_out, *adaptor);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == n_cells);

    std::vector<double> initial_support_points;
    initial_support_points.reserve(support_points.size());
    for (const auto &point : support_points)
      initial_support_points.push_back(point[0]);
    std::sort(initial_support_points.begin(), initial_support_points.end());

    for (std::size_t i = 0; i < initial_support_points.size(); ++i)
      REQUIRE(initial_support_points[i] == Catch::Approx(static_cast<double>(i) * delta_sigma).margin(grid_tol));

    try {
      time_stepper.run(state, 0.0, final_time);
    } catch (const std::exception &exception) {
      FAIL(std::string("Gross-Neveu KT regression solve failed: ") + exception.what());
    }

    const SampledProfile sampled_u = kt_regression::sample_sorted_profile(state, discretization, grid_tol);
    SimulationResult result;
    result.x = sampled_u.x;
    result.u = sampled_u.y;
    result.U = reconstruct_potential_from_u(result.u);
    return result;
  }

  template <typename FlowParam> SimulationResult run_flow_to_ir(const FlowCase &flow_case)
  {
    if constexpr (FlowParam::is_mean_field)
      return run_flow_to_ir<GrossNeveuKTMeanFieldModel, ImplicitTimeStepper>(flow_case);
    else
      return run_flow_to_ir<GrossNeveuKTFluxModel, ImplicitTimeStepper>(flow_case);
  }

  template <typename FlowParam> struct GrossNeveuFlowFixture {
    static FlowCase flow_case() { return FlowParam::flow_case(); }

    static std::filesystem::path expected_local_fixture_path() { return local_fixture_path(FlowParam::relative_path); }

    void run_regression()
    {
      const FlowCase current_flow = flow_case();
      INFO(describe_flow_case(current_flow));

      REQUIRE(current_flow.filename == std::filesystem::path(current_flow.relative_path).filename().string());
      REQUIRE(fixture_path(current_flow) == expected_local_fixture_path());
      REQUIRE(std::filesystem::exists(expected_local_fixture_path()));

      const auto [reference_U, reference_u] = load_reference_ir_pair(current_flow);
      const auto simulation = run_flow_to_ir<FlowParam>(current_flow);

      for (std::size_t i = 0; i < simulation.x.size(); ++i)
        REQUIRE(simulation.x[i] == Catch::Approx(reference_u.x[i]).margin(grid_tol));

      if (const auto u_mismatch =
              kt_regression::compare_profiles("u(sigma)", simulation.x, simulation.u, reference_u.y,
                                              current_flow.profile_abs_tol, current_flow.profile_rel_tol))
        FAIL(*u_mismatch);

      if (const auto U_mismatch =
              kt_regression::compare_profiles("U(sigma)", simulation.x, simulation.U, reference_U.y,
                                              current_flow.profile_abs_tol, current_flow.profile_rel_tol))
        FAIL(*U_mismatch);
    }
  };

} // namespace

TEMPLATE_TEST_CASE_METHOD(GrossNeveuFlowFixture, "KT Gross-Neveu mean-field regressions match reference flows",
                          "[FV][KT][GrossNeveu][MF][regression][slow]", FlowMF_T00625_Mu06, FlowMF_T00125_Mu06,
                          FlowMF_T01_Mu01)
{
  kt_regression::ensure_logger();

  REQUIRE(TestType::flow_case().is_mean_field);

  this->run_regression();
}

TEMPLATE_TEST_CASE_METHOD(GrossNeveuFlowFixture, "KT Gross-Neveu finite-N regressions match reference flows",
                          "[FV][KT][GrossNeveu][regression][slow]", FlowN16_T00625_Mu06, FlowN2_T00625_Mu06,
                          FlowN2_T00125_Mu06, FlowN2_T01_Mu01)
{
  kt_regression::ensure_logger();

  REQUIRE_FALSE(TestType::flow_case().is_mean_field);

  this->run_regression();
}
