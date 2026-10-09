#include "kt_regression_helpers.hh"

#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/discretization/mesh/no_adaptivity.hh"
#include "DiFfRG/timestepping/timestepping.hh"

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 2;
  constexpr std::size_t n_components = 2;
  constexpr std::array<unsigned int, 3> grid_sizes{{32, 64, 128}};
  constexpr double domain_min = 0.0;
  constexpr double domain_max = 1.0;
  constexpr double domain_length = domain_max - domain_min;
  constexpr double final_time = 0.1;
  constexpr double lambda = 1.0;
  constexpr double diffusivity = 1.0e-2;
  constexpr double velocity_x = 1.0;
  constexpr double velocity_y = 0.5;
  constexpr double amplitude_u = 1.0;
  constexpr double amplitude_v = 2.0;
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double wave_number = 0.5 * pi;
  constexpr double timestep = 1.0e-3;
  constexpr double finest_l1_tolerance = 5.0e-2;
  constexpr double minimum_l1_order = 0.75;

  using NumberType = double;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Mesh = RectangularMeshSerial<dim>;
  using Discretization = FV::Discretization<Components, Mesh, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using Assembler =
      FV::KurganovTadmor::Assembler<Discretization, class ManufacturedAdvectionDiffusion2DModel, Reconstructor>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Discretization>;

  class ManufacturedAdvectionDiffusion2DModel
      : public def::AbstractModel<ManufacturedAdvectionDiffusion2DModel, Components>,
        public def::Time,
        public def::LLFFlux<ManufacturedAdvectionDiffusion2DModel>,
        public def::FlowBoundaries<ManufacturedAdvectionDiffusion2DModel>,
        public def::OriginOddLinearExtrapolationBoundaries<ManufacturedAdvectionDiffusion2DModel>,
        public def::AD<ManufacturedAdvectionDiffusion2DModel>
  {
  public:
    ManufacturedAdvectionDiffusion2DModel() { set_time(0.0); }

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      const auto exact = exact_solution(pos, 0.0);
      values[0] = exact[0];
      values[1] = exact[1];
    }

    std::array<double, n_components> exact_solution(const Point<dim> &pos, const double time) const
    {
      const double decay = std::exp(-lambda * time);
      const double sin_x = std::sin(wave_number * pos[0]);
      const double cos_x = std::cos(wave_number * pos[0]);
      const double sin_y = std::sin(wave_number * pos[1]);
      const double cos_y = std::cos(wave_number * pos[1]);
      return {amplitude_u * decay * sin_x * cos_y, amplitude_v * decay * cos_x * sin_y};
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
              const Solution &sol) const
    {
      const auto &fe_functions = get<0>(sol);
      for (std::size_t component = 0; component < n_components; ++component) {
        F_i[component][0] = NT(velocity_x) * fe_functions[component];
        F_i[component][1] = NT(velocity_y) * fe_functions[component];
      }
    }

    template <typename NT, typename Solution>
    void diffusion_flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
                        const Solution &sol) const
    {
      const auto &fe_derivatives = get<1>(sol);
      for (std::size_t component = 0; component < n_components; ++component)
        F_i[component] = -diffusivity * fe_derivatives[component];
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, n_components> &s_i, const Point<dim> &pos, [[maybe_unused]] const Solution &sol) const
    {
      const auto manufactured_source = source_terms(pos, get_time());
      for (std::size_t component = 0; component < n_components; ++component)
        s_i[component] = -NT(manufactured_source[component]);
    }

  private:
    static std::array<double, n_components> source_terms(const Point<dim> &pos, const double time)
    {
      const double decay = std::exp(-lambda * time);
      const double sin_x = std::sin(wave_number * pos[0]);
      const double cos_x = std::cos(wave_number * pos[0]);
      const double sin_y = std::sin(wave_number * pos[1]);
      const double cos_y = std::cos(wave_number * pos[1]);
      const double reaction = -lambda + 2.0 * diffusivity * wave_number * wave_number;

      const double source_u = amplitude_u * decay *
                              (reaction * sin_x * cos_y + velocity_x * wave_number * cos_x * cos_y -
                               velocity_y * wave_number * sin_x * sin_y);
      const double source_v = amplitude_v * decay *
                              (reaction * cos_x * sin_y - velocity_x * wave_number * sin_x * sin_y +
                               velocity_y * wave_number * cos_x * cos_y);
      return {source_u, source_v};
    }
  };

  double dx_for_grid(const unsigned int n_cells) { return domain_length / static_cast<double>(n_cells); }

  std::string grid_axis_string(const unsigned int n_cells)
  {
    std::ostringstream stream;
    stream << std::setprecision(17) << domain_min << ':' << dx_for_grid(n_cells) << ':' << domain_max;
    return stream.str();
  }

  ConfigTree make_run_json(const unsigned int n_cells)
  {
    const std::string axis = grid_axis_string(n_cells);
    return json::value({{"physical", {{"Lambda", 1.0}}},
                        {"discretization",
                         {{"fe_order", 0},
                          {"overintegration", 0},
                          {"output_subdivisions", 1},
                          {"EoM_abs_tol", 1.0e-10},
                          {"EoM_max_iter", 0},
                          {"grid", {{"x_grid", axis}, {"y_grid", axis}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
                        {"timestepping",
                         {{"final_time", final_time},
                          {"output_dt", final_time},
                          {"explicit",
                           {{"dt", timestep},
                            {"minimal_dt", 1.0e-8},
                            {"maximal_dt", timestep},
                            {"abs_tol", 1.0e-9},
                            {"rel_tol", 1.0e-9}}},
                          {"implicit",
                           {{"dt", timestep},
                            {"minimal_dt", 1.0e-8},
                            {"maximal_dt", timestep},
                            {"abs_tol", 1.0e-8},
                            {"rel_tol", 1.0e-8},
                            {"max_steps", 1000000}}}}},
                        {"output", {{"verbosity", 1}, {"vtk", false}, {"hdf5", false}}}});
  }

  Config::ConfigurationMesh<dim> make_mesh_config(const unsigned int n_cells)
  {
    const std::vector<Config::GridAxis> axis{Config::GridAxis(domain_min, dx_for_grid(n_cells), domain_max)};
    return Config::ConfigurationMesh<dim>(0u, axis, axis);
  }

  struct ExactSamples {
    VectorType values;
    VectorType measures;
    std::vector<unsigned int> components;
  };

  ExactSamples exact_cell_averages(const Discretization &discretization,
                                   const ManufacturedAdvectionDiffusion2DModel &model, const double time)
  {
    ExactSamples result;
    const auto n_dofs = discretization.get_dof_handler().n_dofs();
    result.values.reinit(n_dofs);
    result.measures.reinit(n_dofs);
    result.components.assign(n_dofs, 0U);

    QGauss<dim> quadrature(16);
    FEValues<dim> fe_values(discretization.get_mapping(), discretization.get_fe(), quadrature,
                            update_quadrature_points | update_JxW_values);
    std::vector<types::global_dof_index> dof_indices(discretization.get_fe().dofs_per_cell);

    for (const auto &cell : discretization.get_dof_handler().active_cell_iterators()) {
      fe_values.reinit(cell);
      cell->get_dof_indices(dof_indices);

      double measure = 0.0;
      std::array<double, n_components> integral{};
      for (unsigned int q = 0; q < quadrature.size(); ++q) {
        const double weight = fe_values.JxW(q);
        const auto exact = model.exact_solution(fe_values.quadrature_point(q), time);
        measure += weight;
        for (std::size_t component = 0; component < n_components; ++component)
          integral[component] += weight * exact[component];
      }

      for (unsigned int local_dof = 0; local_dof < discretization.get_fe().dofs_per_cell; ++local_dof) {
        const unsigned int component = discretization.get_fe().system_to_component_index(local_dof).first;
        const auto dof = dof_indices[local_dof];
        result.values[dof] = integral[component] / measure;
        result.measures[dof] = measure;
        result.components[dof] = component;
      }
    }

    return result;
  }

  struct ErrorMetrics {
    std::array<double, n_components> l1{};
    std::array<double, n_components> l2{};
    std::array<double, n_components> linf{};
    double combined_l1 = 0.0;
    double combined_l2 = 0.0;
    double combined_linf = 0.0;
  };

  ErrorMetrics compute_error(const VectorType &numerical, const ExactSamples &exact)
  {
    ErrorMetrics metrics;
    std::array<double, n_components> l2_squared{};
    for (unsigned int i = 0; i < numerical.size(); ++i) {
      const unsigned int component = exact.components[i];
      const double error = numerical[i] - exact.values[i];
      const double abs_error = std::abs(error);
      metrics.l1[component] += exact.measures[i] * abs_error;
      l2_squared[component] += exact.measures[i] * error * error;
      metrics.linf[component] = std::max(metrics.linf[component], abs_error);
    }

    double combined_l2_squared = 0.0;
    for (std::size_t component = 0; component < n_components; ++component) {
      metrics.l2[component] = std::sqrt(l2_squared[component]);
      metrics.combined_l1 += metrics.l1[component];
      combined_l2_squared += l2_squared[component];
      metrics.combined_linf = std::max(metrics.combined_linf, metrics.linf[component]);
    }
    metrics.combined_l2 = std::sqrt(combined_l2_squared);
    return metrics;
  }

  struct ConvergenceRow {
    unsigned int n_cells = 0;
    double dx = 0.0;
    ErrorMetrics error;
    double observed_l1_order = std::numeric_limits<double>::quiet_NaN();
  };

  ConvergenceRow run_case(const unsigned int n_cells)
  {
    kt_regression::ensure_logger();

    const ConfigTree json = make_run_json(n_cells);
    ManufacturedAdvectionDiffusion2DModel model;
    Mesh mesh(make_mesh_config(n_cells));
    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);

    const std::string output_name =
        "manufactured_advection_diffusion_2D_" + std::to_string(n_cells) + "x" + std::to_string(n_cells);
    auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, output_name, "output");
    OutputSession<Discretization> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    TimeStepper time_stepper(json, assembler, data_out, *adaptor);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());

    time_stepper.run(state, 0.0, final_time);

    ConvergenceRow row;
    row.n_cells = n_cells;
    row.dx = dx_for_grid(n_cells);
    row.error = compute_error(state.spatial_data(), exact_cell_averages(discretization, model, final_time));
    return row;
  }

  double observed_order(const ConvergenceRow &coarse, const ConvergenceRow &fine)
  {
    return std::log(coarse.error.combined_l1 / fine.error.combined_l1) / std::log(coarse.dx / fine.dx);
  }

  std::vector<ConvergenceRow> run_convergence_study()
  {
    std::vector<ConvergenceRow> rows;
    rows.reserve(grid_sizes.size());
    for (const unsigned int n_cells : grid_sizes) {
      rows.push_back(run_case(n_cells));
      if (rows.size() > 1) rows.back().observed_l1_order = observed_order(rows[rows.size() - 2], rows.back());
    }
    return rows;
  }

  void print_table(const std::vector<ConvergenceRow> &rows)
  {
    std::cout << "\n2D manufactured advection-diffusion convergence\n"
              << std::scientific << std::setprecision(6) << std::setw(10) << "grid" << std::setw(14) << "dx"
              << std::setw(14) << "L1" << std::setw(14) << "L2" << std::setw(14) << "Linf" << std::setw(14) << "ord(L1)"
              << std::setw(14) << "L1(u)" << std::setw(14) << "L1(v)" << '\n';
    for (const auto &row : rows) {
      std::cout << std::setw(5) << row.n_cells << "x" << std::left << std::setw(4) << row.n_cells << std::right
                << std::setw(14) << row.dx << std::setw(14) << row.error.combined_l1 << std::setw(14)
                << row.error.combined_l2 << std::setw(14) << row.error.combined_linf;
      if (std::isfinite(row.observed_l1_order))
        std::cout << std::setw(14) << row.observed_l1_order;
      else
        std::cout << std::setw(14) << "-";
      std::cout << std::setw(14) << row.error.l1[0] << std::setw(14) << row.error.l1[1] << '\n';
    }
  }

  void check_convergence_rows(const std::vector<ConvergenceRow> &rows)
  {
    REQUIRE(rows.size() == grid_sizes.size());
    for (const auto &row : rows) {
      REQUIRE(std::isfinite(row.dx));
      REQUIRE(std::isfinite(row.error.combined_l1));
      REQUIRE(std::isfinite(row.error.combined_l2));
      REQUIRE(std::isfinite(row.error.combined_linf));
      for (std::size_t component = 0; component < n_components; ++component) {
        REQUIRE(std::isfinite(row.error.l1[component]));
        REQUIRE(std::isfinite(row.error.l2[component]));
        REQUIRE(std::isfinite(row.error.linf[component]));
      }
    }

    for (std::size_t i = 1; i < rows.size(); ++i) {
      CHECK(rows[i].error.combined_l1 < rows[i - 1].error.combined_l1);
      CHECK(rows[i].error.combined_l2 < rows[i - 1].error.combined_l2);
      CHECK(rows[i].error.combined_linf < rows[i - 1].error.combined_linf);
      CHECK(rows[i].error.l1[0] < rows[i - 1].error.l1[0]);
      CHECK(rows[i].error.l1[1] < rows[i - 1].error.l1[1]);
    }

    CHECK(rows.back().error.combined_l1 < finest_l1_tolerance);
    CHECK(rows.back().observed_l1_order > minimum_l1_order);
  }
} // namespace

TEST_CASE("2D manufactured advection-diffusion converges with asymmetric origin boundaries",
          "[2d][FV][KT][advection][diffusion][manufactured][slow]")
{
  const std::vector<ConvergenceRow> rows = run_convergence_study();
  print_table(rows);
  check_convergence_rows(rows);
}
