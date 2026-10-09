#include "../kt_regression_helpers.hh"
#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/timestepping/linear_solver/ScaledGMRES.hh"
#include "DiFfRG/timestepping/timestepping.hh"

#include <DiFfRG/common/config_tree.hh>
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
#include <optional>
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
  constexpr double hook_tolerance = 1.0e-12;
  constexpr double regression_tolerance = 1.0e-8;
  constexpr double phi_min = -10.0;
  constexpr double phi_max = 10.0;
  constexpr double default_final_time = 60.0;
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

  struct GaussianScenario {
    std::string name;
    double a;
    double b;
    double h_1;
    double h_2;
  };

  std::vector<GaussianScenario> gaussian_scenarios()
  {
    std::vector<GaussianScenario> scenarios;
    for (const double a : {1.0, 2.0})
      for (const double b : {1.0, 2.0})
        for (const double h_1 : {0.0, 0.75})
          for (const double h_2 : {0.0, -0.5})
            scenarios.push_back({"a=" + std::to_string(a) + ", b=" + std::to_string(b) + ", h1=" + std::to_string(h_1) +
                                     ", h2=" + std::to_string(h_2),
                                 a, b, h_1, h_2});
    return scenarios;
  }

  class GaussianModel : public def::AbstractModel<GaussianModel, Components>,
                        public def::fRG,
                        public def::LLFFlux<GaussianModel>,
                        public def::FlowBoundaries<GaussianModel>,
                        public def::FVDefaultBoundaries<GaussianModel>,
                        public def::AD<GaussianModel>
  {
  public:
    explicit GaussianModel(const GaussianScenario &scenario) : def::fRG(1.0), scenario(scenario) {}
    GaussianModel(const ConfigTree &json, const GaussianScenario &scenario) : def::fRG(json), scenario(scenario) {}

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      values[0] = scenario.a * pos[0] - scenario.h_1;
      values[1] = scenario.b * pos[1] - scenario.h_2;
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

    template <typename Vector> std::array<double, dim> EoM(const Point<dim> &, const Vector &values) const
    {
      return {values[0], values[1]};
    }

    template <typename DataOut, typename Solutions>
    void readouts([[maybe_unused]] DataOut &output, const Point<dim> &eom,
                  [[maybe_unused]] const Solutions &solutions) const
    {
      automatic_eom = eom;
    }

    const std::optional<Point<dim>> &last_automatic_eom() const { return automatic_eom; }

  private:
    GaussianScenario scenario;
    mutable std::optional<Point<dim>> automatic_eom;
  };

  using GaussianAssembler = FV::KurganovTadmor::Assembler<Discretization, GaussianModel, Reconstructor>;

  struct RunParameters {
    std::size_t n_cells_per_direction;
    double final_time = default_final_time;
  };

  double cell_width(const RunParameters &params)
  {
    return (phi_max - phi_min) / static_cast<double>(params.n_cells_per_direction);
  }

  double center_min(const RunParameters &params) { return phi_min + 0.5 * cell_width(params); }

  std::string grid_axis_expression(const RunParameters &params)
  {
    return std::to_string(phi_min) + ":" + std::to_string(cell_width(params)) + ":" + std::to_string(phi_max);
  }

  ConfigTree make_run_json(const RunParameters &params)
  {
    const std::string phi_grid = grid_axis_expression(params);
    return json::value(
        {{"physical", {{"Lambda", 1.0e12}}},
         {"discretization",
          {{"fe_order", 0},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 200},
           {"grid", {{"x_grid", phi_grid}, {"y_grid", phi_grid}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
         {"timestepping",
          {{"final_time", params.final_time},
           {"output_dt", params.final_time},
           {"explicit",
            {{"dt", 1.0e-3}, {"minimal_dt", 1.0e-10}, {"maximal_dt", 1.0}, {"abs_tol", 1.0e-12}, {"rel_tol", 1.0e-10}}},
           {"implicit",
            {{"dt", 1.0e-3},
             {"minimal_dt", 1.0e-14},
             {"maximal_dt", 1.0},
             {"abs_tol", 1.0e-12},
             {"rel_tol", 1.0e-10},
             {"max_steps", 5000000}}}}},
         {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", false}}}});
  }

  Config::ConfigurationMesh<dim> make_mesh_config(const RunParameters &params)
  {
    const std::vector<Config::GridAxis> phi_axis{Config::GridAxis(phi_min, cell_width(params), phi_max)};
    return Config::ConfigurationMesh<dim>(0u, phi_axis, phi_axis);
  }

  struct SampledGrid {
    std::size_t n_cells_per_direction = 0;
    double cell_width = 0.0;
    double first_center = 0.0;
    std::vector<double> u;
    std::vector<double> v;

    std::size_t index(const std::size_t ix, const std::size_t iy) const { return iy * n_cells_per_direction + ix; }
    double coordinate(const std::size_t i) const { return first_center + static_cast<double>(i) * cell_width; }
    double value(const std::vector<double> &component, const std::size_t ix, const std::size_t iy) const
    {
      return component[index(ix, iy)];
    }
  };

  std::size_t coordinate_index(const double x, const RunParameters &params)
  {
    const double floating_index = (x - center_min(params)) / cell_width(params);
    const auto index = static_cast<long>(std::llround(floating_index));
    if (index < 0 || index >= static_cast<long>(params.n_cells_per_direction))
      throw std::runtime_error("Sampled support point lies outside the expected Gaussian grid.");
    const double expected = center_min(params) + static_cast<double>(index) * cell_width(params);
    if (std::abs(x - expected) > point_tolerance)
      throw std::runtime_error("Sampled support point does not lie on the expected Gaussian grid.");
    return static_cast<std::size_t>(index);
  }

  SampledGrid sample_final_grid(const FV::FlowingVariables<Discretization> &state, const Discretization &discretization,
                                const RunParameters &params)
  {
    const auto &dof_handler = discretization.get_dof_handler();
    const auto &fe = discretization.get_fe();
    const auto &solution = state.spatial_data();
    const std::size_t expected_cells = params.n_cells_per_direction * params.n_cells_per_direction;
    if (dof_handler.get_triangulation().n_active_cells() != expected_cells)
      throw std::runtime_error("Gaussian regression has an unexpected number of active cells.");

    SampledGrid grid{params.n_cells_per_direction, cell_width(params), center_min(params)};
    grid.u.assign(expected_cells, std::numeric_limits<double>::quiet_NaN());
    grid.v.assign(expected_cells, std::numeric_limits<double>::quiet_NaN());

    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
      cell->get_dof_indices(dof_indices);
      const std::size_t ix = coordinate_index(cell->center()[0], params);
      const std::size_t iy = coordinate_index(cell->center()[1], params);
      const std::size_t i = grid.index(ix, iy);
      for (unsigned int local_dof = 0; local_dof < fe.dofs_per_cell; ++local_dof) {
        const auto component = fe.system_to_component_index(local_dof).first;
        if (component == 0)
          grid.u[i] = solution[dof_indices[local_dof]];
        else if (component == 1)
          grid.v[i] = solution[dof_indices[local_dof]];
      }
    }

    for (std::size_t i = 0; i < expected_cells; ++i)
      if (!std::isfinite(grid.u[i]) || !std::isfinite(grid.v[i]))
        throw std::runtime_error("Gaussian regression grid contains a missing or non-finite sample.");
    return grid;
  }

  struct InterpolatedScalar {
    double value = 0.0;
    double d_phi1 = 0.0;
    double d_phi2 = 0.0;
  };

  struct InterpolatedField {
    double u = 0.0;
    double v = 0.0;
    double du_dphi1 = 0.0;
    double du_dphi2 = 0.0;
    double dv_dphi1 = 0.0;
    double dv_dphi2 = 0.0;
  };

  std::size_t lower_interpolation_index(const SampledGrid &grid, const double x)
  {
    const auto index = static_cast<long>(std::floor((x - grid.first_center) / grid.cell_width));
    return static_cast<std::size_t>(std::clamp(index, 0L, static_cast<long>(grid.n_cells_per_direction) - 2L));
  }

  InterpolatedScalar bilinear_interpolate(const SampledGrid &grid, const std::vector<double> &component,
                                          const double phi_1, const double phi_2)
  {
    const std::size_t ix = lower_interpolation_index(grid, phi_1);
    const std::size_t iy = lower_interpolation_index(grid, phi_2);
    const double tx = (phi_1 - grid.coordinate(ix)) / grid.cell_width;
    const double ty = (phi_2 - grid.coordinate(iy)) / grid.cell_width;
    const double f00 = grid.value(component, ix, iy);
    const double f10 = grid.value(component, ix + 1, iy);
    const double f01 = grid.value(component, ix, iy + 1);
    const double f11 = grid.value(component, ix + 1, iy + 1);
    return {(1.0 - tx) * (1.0 - ty) * f00 + tx * (1.0 - ty) * f10 + (1.0 - tx) * ty * f01 + tx * ty * f11,
            ((1.0 - ty) * (f10 - f00) + ty * (f11 - f01)) / grid.cell_width,
            ((1.0 - tx) * (f01 - f00) + tx * (f11 - f10)) / grid.cell_width};
  }

  InterpolatedField interpolate_field(const SampledGrid &grid, const double phi_1, const double phi_2)
  {
    const auto u = bilinear_interpolate(grid, grid.u, phi_1, phi_2);
    const auto v = bilinear_interpolate(grid, grid.v, phi_1, phi_2);
    return {u.value, v.value, u.d_phi1, u.d_phi2, v.d_phi1, v.d_phi2};
  }

  struct GaussianObservables {
    Point<dim> minimum;
    InterpolatedField field;
    double max_field_error = 0.0;
  };

  GaussianObservables extract_observables(const SampledGrid &grid, const GaussianScenario &scenario,
                                          const Point<dim> &minimum)
  {
    double max_field_error = 0.0;
    for (std::size_t iy = 0; iy < grid.n_cells_per_direction; ++iy)
      for (std::size_t ix = 0; ix < grid.n_cells_per_direction; ++ix) {
        const double phi_1 = grid.coordinate(ix);
        const double phi_2 = grid.coordinate(iy);
        max_field_error =
            std::max(max_field_error, std::abs(grid.value(grid.u, ix, iy) - (scenario.a * phi_1 - scenario.h_1)));
        max_field_error =
            std::max(max_field_error, std::abs(grid.value(grid.v, ix, iy) - (scenario.b * phi_2 - scenario.h_2)));
      }
    return {minimum, interpolate_field(grid, minimum[0], minimum[1]), max_field_error};
  }

  GaussianObservables run_gaussian_regression(const GaussianScenario &scenario, const RunParameters &params)
  {
    kt_regression::ensure_logger();
    const ConfigTree json = make_run_json(params);
    GaussianModel model(json, scenario);
    Mesh mesh(make_mesh_config(params));
    Discretization discretization(mesh, json);
    GaussianAssembler assembler(discretization, model, json);
    auto data_out_path =
        OutputPath::temporary(TemporaryRetention::remove_on_destruction, "zero_dim_2d_kt_gaussian", "output");
    OutputSession<Discretization> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, assembler, data_out, *adaptor);
    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());
    time_stepper.run(state, 0.0, params.final_time);
    if (!model.last_automatic_eom()) throw std::runtime_error("KT readout did not compute the Gaussian model EoM.");
    return extract_observables(sample_final_grid(state, discretization, params), scenario, *model.last_automatic_eom());
  }

  void check_observables(const GaussianScenario &scenario, const RunParameters &params,
                         const GaussianObservables &observables)
  {
    const double expected_phi_1 = scenario.h_1 / scenario.a;
    const double expected_phi_2 = scenario.h_2 / scenario.b;
    const double eom_position_tolerance = cell_width(params);
    const double eom_residual_tolerance = 2.0 * std::max(scenario.a, scenario.b) * cell_width(params);
    CAPTURE(scenario.a, scenario.b, scenario.h_1, scenario.h_2, observables.minimum[0], observables.minimum[1],
            observables.field.u, observables.field.v, observables.field.du_dphi1, observables.field.du_dphi2,
            observables.field.dv_dphi1, observables.field.dv_dphi2, observables.max_field_error);
    REQUIRE(std::isfinite(observables.max_field_error));
    REQUIRE(std::isfinite(observables.minimum[0]));
    REQUIRE(std::isfinite(observables.minimum[1]));
    CHECK_THAT(observables.max_field_error, Catch::Matchers::WithinAbs(0.0, regression_tolerance));
    CHECK_THAT(observables.minimum[0], Catch::Matchers::WithinAbs(expected_phi_1, eom_position_tolerance));
    CHECK_THAT(observables.minimum[1], Catch::Matchers::WithinAbs(expected_phi_2, eom_position_tolerance));
    CHECK_THAT(observables.field.u, Catch::Matchers::WithinAbs(0.0, eom_residual_tolerance));
    CHECK_THAT(observables.field.v, Catch::Matchers::WithinAbs(0.0, eom_residual_tolerance));
    CHECK_THAT(observables.field.du_dphi1, Catch::Matchers::WithinAbs(scenario.a, regression_tolerance));
    CHECK_THAT(observables.field.du_dphi2, Catch::Matchers::WithinAbs(0.0, regression_tolerance));
    CHECK_THAT(observables.field.dv_dphi1, Catch::Matchers::WithinAbs(0.0, regression_tolerance));
    CHECK_THAT(observables.field.dv_dphi2, Catch::Matchers::WithinAbs(scenario.b, regression_tolerance));
    CHECK(observables.field.du_dphi1 * observables.field.dv_dphi2 -
              observables.field.du_dphi2 * observables.field.dv_dphi1 >
          0.0);
  }
} // namespace

TEST_CASE("Zero-dimensional KT Gaussian exposes analytic affine fields and diagonal diffusion flux",
          "[2d][FV][KT][gaussian]")
{
  const std::array<Point<dim>, 3> points{Point<dim>(0.25, 0.5), Point<dim>(-1.25, 0.75), Point<dim>(2.0, -1.5)};
  const std::array<double, n_components> fe_functions{{1.0, 2.0}};
  const std::array<Tensor<1, dim, double>, n_components> fe_derivatives{
      {Tensor<1, dim, double>({1.0, 1.0}), Tensor<1, dim, double>({1.0, 2.0})}};
  const auto solution = std::tuple{fe_functions, fe_derivatives};

  for (const auto &scenario : gaussian_scenarios()) {
    DYNAMIC_SECTION(scenario.name)
    {
      const GaussianModel model(scenario);
      for (const auto &point : points) {
        std::array<double, n_components> initial_values{};
        model.initial_condition(point, initial_values);
        CHECK_THAT(initial_values[0], Catch::Matchers::WithinAbs(scenario.a * point[0] - scenario.h_1, hook_tolerance));
        CHECK_THAT(initial_values[1], Catch::Matchers::WithinAbs(scenario.b * point[1] - scenario.h_2, hook_tolerance));
      }

      std::array<Tensor<1, dim, double>, n_components> advection_flux{};
      for (auto &flux_i : advection_flux)
        for (uint d = 0; d < dim; ++d)
          flux_i[d] = 1.0;
      model.flux(advection_flux, points.front(), solution);
      for (const auto &flux_i : advection_flux)
        for (uint d = 0; d < dim; ++d)
          CHECK(flux_i[d] == 0.0);

      std::array<Tensor<1, dim, double>, n_components> diffusion_flux{};
      for (auto &flux_i : diffusion_flux)
        for (uint d = 0; d < dim; ++d)
          flux_i[d] = 1.0;
      model.diffusion_flux(diffusion_flux, points.front(), solution);
      CHECK_THAT(diffusion_flux[0][0], Catch::Matchers::WithinAbs(0.5, hook_tolerance));
      CHECK(diffusion_flux[0][1] == 0.0);
      CHECK(diffusion_flux[1][0] == 0.0);
      CHECK_THAT(diffusion_flux[1][1], Catch::Matchers::WithinAbs(0.5, hook_tolerance));

      std::array<double, n_components> source{{1.0, 1.0}};
      model.source(source, points.front(), solution);
      CHECK(source[0] == 0.0);
      CHECK(source[1] == 0.0);
    }
  }
}

TEST_CASE("Zero-dimensional KT Gaussian 40x40 matrix preserves analytic observables",
          "[2d][FV][KT][gaussian][coarse][slow]")
{
  for (const auto &scenario : gaussian_scenarios()) {
    DYNAMIC_SECTION(scenario.name)
    {
      const auto observables = run_gaussian_regression(scenario, {.n_cells_per_direction = 40});
      check_observables(scenario, {.n_cells_per_direction = 40}, observables);
    }
  }
}

TEST_CASE("Zero-dimensional KT Gaussian swapped anisotropic 400x400 run preserves analytic observables",
          "[2d][FV][KT][gaussian][slow]")
{
  const GaussianScenario scenario{"a=2, b=1, h1=0.75, h2=-0.5", 2.0, 1.0, 0.75, -0.5};
  const RunParameters params{.n_cells_per_direction = 400};
  const auto observables = run_gaussian_regression(scenario, params);
  check_observables(scenario, params, observables);
}

TEST_CASE("Zero-dimensional KT Gaussian automatically finds the model EoM", "[2d][FV][KT][gaussian][EoM]")
{
  const GaussianScenario scenario{"a=2, b=1, h1=0.75, h2=-0.5", 2.0, 1.0, 0.75, -0.5};
  const RunParameters params{.n_cells_per_direction = 40};
  const auto observables = run_gaussian_regression(scenario, params);

  CHECK_THAT(observables.minimum[0], Catch::Matchers::WithinAbs(scenario.h_1 / scenario.a, cell_width(params)));
  CHECK_THAT(observables.minimum[1], Catch::Matchers::WithinAbs(scenario.h_2 / scenario.b, cell_width(params)));
}
