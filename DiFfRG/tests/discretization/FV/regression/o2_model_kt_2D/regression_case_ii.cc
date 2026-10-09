#include "../kt_regression_helpers.hh"
#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/timestepping/linear_solver/ScaledGMRES.hh"
#include "DiFfRG/timestepping/timestepping.hh"

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>
#include <deal.II/base/multithread_info.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 2;
  constexpr std::size_t n_components = 2;
  constexpr double phi_min = -10.0;
  constexpr double phi_max = 10.0;
  constexpr std::size_t default_n_cells_per_direction = 400;
  constexpr std::size_t quarter_domain_n_cells_per_direction = 200;
  constexpr double default_final_time = 60.0;
  constexpr double gamma_reference = 0.316367789;
  constexpr double gamma_tolerance = 5.0e-3;
  constexpr double point_tolerance = 1.0e-10;

  using NumberType = double;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Mesh = RectangularMeshSerial<dim>;
  using Discretization = FV::Discretization<Components, Mesh, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using ImplicitTimeStepper = TimeStepperSUNDIALS_IDA<Discretization>;

  template <template <typename> class BoundaryPolicy>
  class O2_Model_VI_B_II_Base : public def::AbstractModel<O2_Model_VI_B_II_Base<BoundaryPolicy>, Components>,
                                public def::fRG,
                                public def::LLFFlux<O2_Model_VI_B_II_Base<BoundaryPolicy>>,
                                public def::FlowBoundaries<O2_Model_VI_B_II_Base<BoundaryPolicy>>,
                                public BoundaryPolicy<O2_Model_VI_B_II_Base<BoundaryPolicy>>,
                                public def::AD<O2_Model_VI_B_II_Base<BoundaryPolicy>>
  {
  public:
    O2_Model_VI_B_II_Base() : def::fRG(1.0) {}
    explicit O2_Model_VI_B_II_Base(const ConfigTree &json) : def::fRG(json) {}

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      const double phi_1 = pos[0];
      const double phi_2 = pos[1];
      const double r2 = phi_1 * phi_1 + phi_2 * phi_2;

      values[0] = -phi_1 + (1.0 / 6.0) * phi_1 * r2;
      values[1] = -phi_2 + (1.0 / 6.0) * phi_2 * r2;
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
              [[maybe_unused]] const Solution &sol) const
    {
      for (auto &flux_i : F_i)
        flux_i = Tensor<1, dim, NT>();
    }

    template <typename NT, typename Solution>
    void diffusion_flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
                        [[maybe_unused]] const Solution &sol) const
    {
      for (auto &flux_i : F_i)
        flux_i = Tensor<1, dim, NT>();

      const auto &fe_derivatives = get<1>(sol);
      const NT du_dphi1 = fe_derivatives[0][0];
      const NT du_dphi2 = fe_derivatives[0][1];
      const NT dv_dphi1 = fe_derivatives[1][0];
      const NT dv_dphi2 = fe_derivatives[1][1];

      const double r = k;
      const NT denominator = (NT(r) + du_dphi1) * (NT(r) + dv_dphi2) - dv_dphi1 * du_dphi2;
      const NT Q = NT(0.5 * r) * (NT(2.0 * r) + du_dphi1 + dv_dphi2) / denominator;

      F_i[0][0] = Q;
      F_i[1][1] = Q;
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, n_components> &s_i, [[maybe_unused]] const Point<dim> &pos,
                [[maybe_unused]] const Solution &sol) const
    {
      for (auto &source_i : s_i)
        source_i = NT(0.0);
    }
  };

  using O2_Model_VI_B_II = O2_Model_VI_B_II_Base<def::FVDefaultBoundaries>;
  using O2_Model_VI_B_II_OriginCentered = O2_Model_VI_B_II_Base<def::OriginOddLinearExtrapolationBoundaries>;

  template <typename Model> using O2Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor>;

  struct GammaRunParameters {
    std::size_t n_cells_per_direction = default_n_cells_per_direction;
    double final_time = default_final_time;
    bool retain_output = true;
    bool write_output = true;
  };

  struct SquareGrid {
    std::size_t n_cells_per_direction = 0;
    double edge_min = 0.0;
    double edge_max = 0.0;
    double cell_width = 0.0;
    double center_min = 0.0;
  };

  SquareGrid full_domain_grid(const GammaRunParameters &params)
  {
    const double width = (phi_max - phi_min) / static_cast<double>(params.n_cells_per_direction);
    return {params.n_cells_per_direction, phi_min, phi_max, width, phi_min + 0.5 * width};
  }

  SquareGrid origin_centered_quarter_domain_grid()
  {
    const double width = (phi_max - phi_min) / static_cast<double>(default_n_cells_per_direction);
    const double edge_min = -0.5 * width;
    const double edge_max = edge_min + static_cast<double>(quarter_domain_n_cells_per_direction) * width;
    return {quarter_domain_n_cells_per_direction, edge_min, edge_max, width, 0.0};
  }

  std::string grid_axis_expression(const SquareGrid &grid)
  {
    return std::to_string(grid.edge_min) + ":" + std::to_string(grid.cell_width) + ":" + std::to_string(grid.edge_max);
  }

  ConfigTree make_run_json(const GammaRunParameters &params, const SquareGrid &grid)
  {
    const std::string phi_grid = grid_axis_expression(grid);
    return json::value(
        {{"physical", {{"Lambda", 1.0e12}}},
         {"discretization",
          {{"fe_order", 0},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", phi_grid}, {"y_grid", phi_grid}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
         {"timestepping",
          {{"final_time", params.final_time},
           {"output_dt", 1.0},
           {"explicit",
            {{"dt", 1.0e-3}, {"minimal_dt", 1.0e-10}, {"maximal_dt", 1.0}, {"abs_tol", 1.0e-12}, {"rel_tol", 1.0e-10}}},
           {"implicit",
            {{"dt", 1.0e-3},
             {"minimal_dt", 1.0e-14},
             {"maximal_dt", 1.0},
             {"abs_tol", 1.0e-12},
             {"rel_tol", 1.0e-10},
             {"max_steps", 5000000}}}}},
         {"output", {{"verbosity", 3}, {"vtk", params.write_output}, {"hdf5", params.write_output}}}});
  }

  Config::ConfigurationMesh<dim> make_mesh_config(const SquareGrid &grid)
  {
    const std::vector<Config::GridAxis> phi_axis{Config::GridAxis(grid.edge_min, grid.cell_width, grid.edge_max)};
    return Config::ConfigurationMesh<dim>(0u, phi_axis, phi_axis);
  }

  struct SampledGrid {
    std::size_t n_cells_per_direction = 0;
    double cell_width = 0.0;
    std::vector<double> u;
    std::vector<double> v;

    std::size_t index(const std::size_t ix, const std::size_t iy) const { return iy * n_cells_per_direction + ix; }

    double value(const std::vector<double> &component, const std::size_t ix, const std::size_t iy) const
    {
      return component[index(ix, iy)];
    }
  };

  std::size_t coordinate_index(const double x, const SquareGrid &grid)
  {
    const double floating_index = (x - grid.center_min) / grid.cell_width;
    const auto index = static_cast<long>(std::llround(floating_index));
    if (index < 0 || index >= static_cast<long>(grid.n_cells_per_direction))
      throw std::runtime_error("Sampled support point lies outside the expected O(2) Gamma grid.");
    const double expected = grid.center_min + static_cast<double>(index) * grid.cell_width;
    if (std::abs(x - expected) > point_tolerance)
      throw std::runtime_error("Sampled support point does not lie on the expected uniform grid.");
    return static_cast<std::size_t>(index);
  }

  SampledGrid sample_final_grid(const FV::FlowingVariables<Discretization> &state, const Discretization &discretization,
                                const SquareGrid &grid_parameters)
  {
    const auto &dof_handler = discretization.get_dof_handler();
    const auto &fe = discretization.get_fe();
    const auto &solution = state.spatial_data();

    if (dof_handler.get_triangulation().n_active_cells() !=
        grid_parameters.n_cells_per_direction * grid_parameters.n_cells_per_direction)
      throw std::runtime_error("The O(2) Gamma regression does not have the expected number of active FV cells.");

    SampledGrid grid;
    grid.n_cells_per_direction = grid_parameters.n_cells_per_direction;
    grid.cell_width = grid_parameters.cell_width;
    const std::size_t n_grid_points = grid_parameters.n_cells_per_direction * grid_parameters.n_cells_per_direction;
    grid.u.assign(n_grid_points, std::numeric_limits<double>::quiet_NaN());
    grid.v.assign(n_grid_points, std::numeric_limits<double>::quiet_NaN());

    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
      cell->get_dof_indices(dof_indices);
      const std::size_t ix = coordinate_index(cell->center()[0], grid_parameters);
      const std::size_t iy = coordinate_index(cell->center()[1], grid_parameters);
      const std::size_t i = grid.index(ix, iy);

      for (unsigned int local_dof = 0; local_dof < fe.dofs_per_cell; ++local_dof) {
        const auto component = fe.system_to_component_index(local_dof).first;
        if (component == 0) {
          if (std::isfinite(grid.u[i])) throw std::runtime_error("Duplicate u sample in O(2) Gamma grid.");
          grid.u[i] = solution[dof_indices[local_dof]];
        } else if (component == 1) {
          if (std::isfinite(grid.v[i])) throw std::runtime_error("Duplicate v sample in O(2) Gamma grid.");
          grid.v[i] = solution[dof_indices[local_dof]];
        }
      }
    }

    for (std::size_t i = 0; i < n_grid_points; ++i) {
      if (!std::isfinite(grid.u[i]) || !std::isfinite(grid.v[i]))
        throw std::runtime_error("Missing or non-finite sample in O(2) Gamma grid.");
    }

    return grid;
  }

  struct Gamma2 {
    double gamma_11 = 0.0;
    double gamma_22 = 0.0;
    double gamma_12_from_u = 0.0;
    double gamma_12_from_v = 0.0;
    double gamma_12 = 0.0;
  };

  Gamma2 extract_origin_gamma2(const SampledGrid &grid)
  {
    if (grid.n_cells_per_direction % 2 != 0)
      throw std::runtime_error("Origin Gamma extraction expects an even number of cells per direction.");
    const std::size_t lower = grid.n_cells_per_direction / 2 - 1;
    const std::size_t upper = grid.n_cells_per_direction / 2;

    const auto u = [&](const std::size_t ix, const std::size_t iy) { return grid.value(grid.u, ix, iy); };
    const auto v = [&](const std::size_t ix, const std::size_t iy) { return grid.value(grid.v, ix, iy); };

    Gamma2 gamma;
    gamma.gamma_11 = 0.5 * ((u(upper, lower) - u(lower, lower)) / grid.cell_width +
                            (u(upper, upper) - u(lower, upper)) / grid.cell_width);
    gamma.gamma_22 = 0.5 * ((v(lower, upper) - v(lower, lower)) / grid.cell_width +
                            (v(upper, upper) - v(upper, lower)) / grid.cell_width);
    gamma.gamma_12_from_u = 0.5 * ((u(lower, upper) - u(lower, lower)) / grid.cell_width +
                                   (u(upper, upper) - u(upper, lower)) / grid.cell_width);
    gamma.gamma_12_from_v = 0.5 * ((v(upper, lower) - v(lower, lower)) / grid.cell_width +
                                   (v(upper, upper) - v(lower, upper)) / grid.cell_width);
    gamma.gamma_12 = 0.5 * (gamma.gamma_12_from_u + gamma.gamma_12_from_v);
    return gamma;
  }

  Gamma2 extract_origin_centered_gamma2(const SampledGrid &grid)
  {
    if (grid.n_cells_per_direction < 2)
      throw std::runtime_error("Origin-centered Gamma extraction expects at least two cells per direction.");

    const auto u = [&](const std::size_t ix, const std::size_t iy) { return grid.value(grid.u, ix, iy); };
    const auto v = [&](const std::size_t ix, const std::size_t iy) { return grid.value(grid.v, ix, iy); };

    Gamma2 gamma;
    gamma.gamma_11 = (u(1, 0) - u(0, 0)) / grid.cell_width;
    gamma.gamma_22 = (v(0, 1) - v(0, 0)) / grid.cell_width;
    gamma.gamma_12_from_u = (u(0, 1) - u(0, 0)) / grid.cell_width;
    gamma.gamma_12_from_v = (v(1, 0) - v(0, 0)) / grid.cell_width;
    gamma.gamma_12 = 0.5 * (gamma.gamma_12_from_u + gamma.gamma_12_from_v);
    return gamma;
  }

  void log_gamma2(const std::string &label, const Gamma2 &gamma)
  {
    std::clog << std::setprecision(17) << "[O2 DIAG] " << label << " Gamma2: gamma_11=" << gamma.gamma_11
              << ", gamma_22=" << gamma.gamma_22 << ", gamma_12_from_u=" << gamma.gamma_12_from_u
              << ", gamma_12_from_v=" << gamma.gamma_12_from_v << ", gamma_12=" << gamma.gamma_12 << '\n';
  }

  double extracted_gamma2(const Gamma2 &gamma) { return 0.5 * (gamma.gamma_11 + gamma.gamma_22); }

  void log_gamma2_reference_comparison(const std::string &label, const Gamma2 &gamma)
  {
    const double extracted = extracted_gamma2(gamma);
    const double absolute_error = std::abs(extracted - gamma_reference);
    const double relative_error = absolute_error / std::abs(gamma_reference);
    std::clog << std::setprecision(17) << "[O2 DIAG] " << label
              << " Gamma2 reference comparison: extracted=" << extracted << ", absolute_error=" << absolute_error
              << ", relative_error=" << relative_error << '\n';
  }

  Gamma2 run_o2_vi_b_ii_gamma_regression(const GammaRunParameters &params = {})
  {
    kt_regression::ensure_logger();

    const SquareGrid grid = full_domain_grid(params);
    const ConfigTree json = make_run_json(params, grid);
    O2_Model_VI_B_II model(json);
    Mesh mesh(make_mesh_config(grid));
    Discretization discretization(mesh, json);
    O2Assembler<O2_Model_VI_B_II> assembler(discretization, model, json);

    auto data_out_path = OutputPath::temporary(params.retain_output ? TemporaryRetention::keep
                                                                    : TemporaryRetention::remove_on_destruction,
                                               "o2_model_kt_2D_case_ii_regression", "output");
    std::clog << "[O2 DIAG] " << (params.retain_output ? "retaining" : "using temporary") << " output directory "
              << data_out_path.root() << '\n';
    OutputSession<Discretization> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, assembler, data_out, *adaptor);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());

    time_stepper.run(state, 0.0, params.final_time);
    return extract_origin_gamma2(sample_final_grid(state, discretization, grid));
  }

  Gamma2 run_o2_vi_b_ii_origin_centered_quarter_domain_regression()
  {
    kt_regression::ensure_logger();

    const GammaRunParameters params{.n_cells_per_direction = quarter_domain_n_cells_per_direction,
                                    .final_time = default_final_time,
                                    .retain_output = false,
                                    .write_output = false};
    const SquareGrid grid = origin_centered_quarter_domain_grid();
    const ConfigTree json = make_run_json(params, grid);
    O2_Model_VI_B_II_OriginCentered model(json);
    Mesh mesh(make_mesh_config(grid));
    Discretization discretization(mesh, json);
    O2Assembler<O2_Model_VI_B_II_OriginCentered> assembler(discretization, model, json);

    auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction,
                                               "o2_model_kt_2D_case_ii_origin_centered_quarter_domain", "output");
    std::clog << "[O2 DIAG] using temporary output directory " << data_out_path.root() << '\n';
    OutputSession<Discretization> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, assembler, data_out, *adaptor);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());

    time_stepper.run(state, 0.0, params.final_time);
    return extract_origin_centered_gamma2(sample_final_grid(state, discretization, grid));
  }
} // namespace

TEST_CASE("O2 Model VI_B_II exposes phi4 initial condition and diagonal diffusion flux", "[2d][FV][KT][O2][VI_B_II]")
{
  const O2_Model_VI_B_II model{};
  const Point<dim> point(0.25, 0.5);
  const std::array<double, n_components> fe_functions{{1.0, 2.0}};
  const std::array<Tensor<1, dim, double>, n_components> fe_derivatives{
      {Tensor<1, dim, double>({1.0, 1.0}), Tensor<1, dim, double>({1.0, 2.0})}};
  const auto solution = std::tuple{fe_functions, fe_derivatives};

  std::array<double, n_components> initial_values{{1.0, 1.0}};
  model.initial_condition(Point<dim>(1.0, 2.0), initial_values);
  CHECK(initial_values[0] == Catch::Approx(-1.0 / 6.0).margin(1.0e-15));
  CHECK(initial_values[1] == Catch::Approx(-1.0 / 3.0).margin(1.0e-15));

  model.initial_condition(Point<dim>(-3.0, 0.0), initial_values);
  CHECK(initial_values[0] == Catch::Approx(-1.5).margin(1.0e-15));
  CHECK(initial_values[1] == 0.0);

  std::array<Tensor<1, dim, double>, n_components> advection_flux{{Tensor<1, dim, double>(), Tensor<1, dim, double>()}};
  for (auto &flux_i : advection_flux)
    for (uint d = 0; d < dim; ++d)
      flux_i[d] = 1.0;
  model.flux(advection_flux, point, solution);
  for (const auto &flux_i : advection_flux)
    for (uint d = 0; d < dim; ++d)
      CHECK(flux_i[d] == 0.0);

  std::array<Tensor<1, dim, double>, n_components> diffusion_flux{{Tensor<1, dim, double>(), Tensor<1, dim, double>()}};
  for (auto &flux_i : diffusion_flux)
    for (uint d = 0; d < dim; ++d)
      flux_i[d] = 1.0;
  model.diffusion_flux(diffusion_flux, point, solution);
  CHECK(diffusion_flux[0][0] == 0.5);
  CHECK(diffusion_flux[0][1] == 0.0);
  CHECK(diffusion_flux[1][0] == 0.0);
  CHECK(diffusion_flux[1][1] == 0.5);

  std::array<double, n_components> source{{1.0, 1.0}};
  model.source(source, point, solution);
  for (const auto source_i : source)
    CHECK(source_i == 0.0);
}

TEST_CASE("O2 Model VI_B_II run extracts Gamma2 at the origin", "[2d][FV][KT][O2][VI_B_II][slow]")
{
  const Gamma2 gamma = run_o2_vi_b_ii_gamma_regression();
  log_gamma2("400x400", gamma);
  log_gamma2_reference_comparison("400x400", gamma);

  const double extracted_gamma = extracted_gamma2(gamma);
  const double absolute_error = std::abs(extracted_gamma - gamma_reference);
  const double relative_error = absolute_error / std::abs(gamma_reference);

  CAPTURE(gamma.gamma_11, gamma.gamma_22, gamma.gamma_12_from_u, gamma.gamma_12_from_v, gamma.gamma_12, extracted_gamma,
          absolute_error, relative_error);
  REQUIRE(std::isfinite(gamma.gamma_11));
  REQUIRE(std::isfinite(gamma.gamma_22));
  REQUIRE(std::isfinite(gamma.gamma_12_from_u));
  REQUIRE(std::isfinite(gamma.gamma_12_from_v));
  REQUIRE(std::isfinite(gamma.gamma_12));
  REQUIRE(std::isfinite(extracted_gamma));

  CHECK_THAT(gamma.gamma_11, Catch::Matchers::WithinAbs(gamma_reference, gamma_tolerance));
  CHECK_THAT(gamma.gamma_22, Catch::Matchers::WithinAbs(gamma_reference, gamma_tolerance));
  CHECK_THAT(extracted_gamma, Catch::Matchers::WithinAbs(gamma_reference, gamma_tolerance));
  CHECK_THAT(gamma.gamma_11 - gamma.gamma_22, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_u, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_v, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
}

TEST_CASE("O2 Model VI_B_II coarse 40x40 run extracts Gamma2 at the origin", "[2d][FV][KT][O2][VI_B_II][coarse][slow]")
{
  const Gamma2 gamma = run_o2_vi_b_ii_gamma_regression(
      {.n_cells_per_direction = 40, .final_time = default_final_time, .retain_output = false});
  log_gamma2("40x40", gamma);

  CAPTURE(gamma.gamma_11, gamma.gamma_22, gamma.gamma_12_from_u, gamma.gamma_12_from_v, gamma.gamma_12);
  REQUIRE(std::isfinite(gamma.gamma_11));
  REQUIRE(std::isfinite(gamma.gamma_22));
  REQUIRE(std::isfinite(gamma.gamma_12_from_u));
  REQUIRE(std::isfinite(gamma.gamma_12_from_v));
  REQUIRE(std::isfinite(gamma.gamma_12));

  CHECK_THAT(gamma.gamma_11 - gamma.gamma_22, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_u, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_v, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
}

TEST_CASE("O2 Model VI_B_II origin-centered quarter-domain run extracts Gamma2 at the origin",
          "[2d][FV][KT][O2][VI_B_II][quarter-domain][slow]")
{
  const Gamma2 gamma = run_o2_vi_b_ii_origin_centered_quarter_domain_regression();
  log_gamma2("200x200 origin-centered quarter-domain", gamma);
  log_gamma2_reference_comparison("200x200 origin-centered quarter-domain", gamma);

  const double extracted_gamma = extracted_gamma2(gamma);
  const double absolute_error = std::abs(extracted_gamma - gamma_reference);
  const double relative_error = absolute_error / std::abs(gamma_reference);

  CAPTURE(gamma.gamma_11, gamma.gamma_22, gamma.gamma_12_from_u, gamma.gamma_12_from_v, gamma.gamma_12, extracted_gamma,
          absolute_error, relative_error);
  REQUIRE(std::isfinite(gamma.gamma_11));
  REQUIRE(std::isfinite(gamma.gamma_22));
  REQUIRE(std::isfinite(gamma.gamma_12_from_u));
  REQUIRE(std::isfinite(gamma.gamma_12_from_v));
  REQUIRE(std::isfinite(gamma.gamma_12));
  REQUIRE(std::isfinite(extracted_gamma));

  CHECK_THAT(gamma.gamma_11, Catch::Matchers::WithinAbs(gamma_reference, gamma_tolerance));
  CHECK_THAT(gamma.gamma_22, Catch::Matchers::WithinAbs(gamma_reference, gamma_tolerance));
  CHECK_THAT(extracted_gamma, Catch::Matchers::WithinAbs(gamma_reference, gamma_tolerance));
  CHECK_THAT(gamma.gamma_11 - gamma.gamma_22, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_u, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_v, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
}
