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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 2;
  constexpr std::size_t n_components = 1;
  constexpr std::array<unsigned int, 3> regression_grid_sizes{{32, 64, 128}};
  constexpr std::array<unsigned int, 4> benchmark_grid_sizes{{32, 64, 128, 256}};
  constexpr double domain_min = -1.0;
  constexpr double domain_max = 1.0;
  constexpr double domain_length = domain_max - domain_min;
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double angular_velocity = 1.0;
  // Half a revolution rather than a full one, purely for runtime: the error is measured against
  // the ANALYTIC rotated Gaussian at final_time (not against the initial condition), so the
  // reference stays exact and the spatial convergence orders are unchanged. Every assertion in
  // check_convergence_rows is relative or monotonic, so none of them is calibrated to the
  // revolution count. Cost scales with the step count, i.e. linearly in final_time.
  constexpr double final_time = pi;
  constexpr double center_x = 0.0;
  constexpr double center_y = 0.0;
  constexpr double pulse_x0 = 0.5;
  constexpr double pulse_y0 = 0.0;
  constexpr double sigma = 0.15;
  constexpr double cfl_number = 0.80;
  constexpr double bounded_min_tolerance = -5.0e-2;
  constexpr double refined_min_tolerance = -2.0e-4;
  constexpr double overshoot_tolerance = 1.0e-2;
  constexpr double coarse_mass_tolerance = 5.0e-3;
  constexpr double refined_mass_tolerance = 2.0e-4;
  constexpr double minimum_observed_l1_order = 1.0;

  using NumberType = double;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Mesh = RectangularMeshSerial<dim>;
  using Discretization = FV::Discretization<Components, Mesh, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, class RotatingGaussian2DModel, Reconstructor,
                                                  FV::KurganovTadmor::MaxEigenvalueWaveSpeed>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

  template <typename NT> Tensor<1, dim, NT> solid_body_velocity(const Point<dim> &point)
  {
    Tensor<1, dim, NT> velocity;
    velocity[0] = NT(-angular_velocity * (point[1] - center_y));
    velocity[1] = NT(angular_velocity * (point[0] - center_x));
    return velocity;
  }

  double cfl_speed(const Point<dim> &point)
  {
    const auto velocity = solid_body_velocity<double>(point);
    return std::abs(velocity[0]) + std::abs(velocity[1]);
  }

  class RotatingGaussianExactSolution
  {
  public:
    double operator()(const Point<dim> &point, const double time) const
    {
      return initial_value(rotated_back_point(point, time));
    }

    static double initial_value(const Point<dim> &point)
    {
      const double r2 = (point[0] - pulse_x0) * (point[0] - pulse_x0) + (point[1] - pulse_y0) * (point[1] - pulse_y0);
      return std::exp(-r2 / (sigma * sigma));
    }

    static Point<dim> rotated_back_point(const Point<dim> &point, const double time)
    {
      const double angle = angular_velocity * time;
      const double cos_angle = std::cos(angle);
      const double sin_angle = std::sin(angle);
      const double dx_from_center = point[0] - center_x;
      const double dy_from_center = point[1] - center_y;

      Point<dim> rotated;
      rotated[0] = center_x + cos_angle * dx_from_center + sin_angle * dy_from_center;
      rotated[1] = center_y - sin_angle * dx_from_center + cos_angle * dy_from_center;
      return rotated;
    }
  };

  class RotatingGaussian2DModel : public def::AbstractModel<RotatingGaussian2DModel, Components>,
                                  public def::Time,
                                  public def::LLFFlux<RotatingGaussian2DModel>,
                                  public def::FlowBoundaries<RotatingGaussian2DModel>,
                                  public def::FVDefaultBoundaries<RotatingGaussian2DModel>,
                                  public def::AD<RotatingGaussian2DModel>
  {
  public:
    RotatingGaussian2DModel() { set_time(0.0); }

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      values[0] = exact_solution(pos, 0.0);
    }

    double exact_solution(const Point<dim> &pos, const double time) const { return exact(pos, time); }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, const Point<dim> &pos, const Solution &sol) const
    {
      const auto &u = get<0>(sol);
      const auto velocity = solid_body_velocity<NT>(pos);
      F_i[0][0] = velocity[0] * u[0];
      F_i[0][1] = velocity[1] * u[0];
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, n_components> &s_i, [[maybe_unused]] const Point<dim> &pos,
                [[maybe_unused]] const Solution &sol) const
    {
      s_i[0] = NT(0.0);
    }

  private:
    RotatingGaussianExactSolution exact;
  };

  double dx_for_grid(const unsigned int n_cells) { return domain_length / static_cast<double>(n_cells); }

  double max_advection_speed(const unsigned int n_cells)
  {
    const double dx = dx_for_grid(n_cells);
    double max_speed = 0.0;
    for (unsigned int y = 0; y < n_cells; ++y) {
      for (unsigned int x = 0; x < n_cells; ++x) {
        const Point<dim> point(domain_min + (static_cast<double>(x) + 0.5) * dx,
                               domain_min + (static_cast<double>(y) + 0.5) * dx);
        max_speed = std::max(max_speed, cfl_speed(point));
      }
    }
    for (const double x : {domain_min, domain_max})
      for (const double y : {domain_min, domain_max})
        max_speed = std::max(max_speed, cfl_speed(Point<dim>(x, y)));
    return max_speed;
  }

  double cfl_timestep(const unsigned int n_cells)
  {
    const double raw_dt = cfl_number * dx_for_grid(n_cells) / max_advection_speed(n_cells);
    const auto n_steps = static_cast<unsigned int>(std::ceil(final_time / raw_dt));
    return final_time / static_cast<double>(n_steps);
  }

  std::string grid_axis_string(const unsigned int n_cells)
  {
    std::ostringstream stream;
    stream << std::setprecision(17) << domain_min << ':' << dx_for_grid(n_cells) << ':' << domain_max;
    return stream.str();
  }

  ConfigTree make_run_json(const unsigned int n_cells, const double dt)
  {
    const auto axis = grid_axis_string(n_cells);
    return json::value(
        {{"physical", {{"Lambda", 1.0}}},
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
            {{"dt", dt}, {"minimal_dt", dt * 1.0e-8}, {"maximal_dt", dt}, {"abs_tol", 1.0e-7}, {"rel_tol", 1.0e-7}}},
           {"implicit",
            {{"dt", dt}, {"minimal_dt", dt * 1.0e-8}, {"maximal_dt", dt}, {"abs_tol", 1.0e-7}, {"rel_tol", 1.0e-7}}}}},
         {"output", {{"verbosity", 0}, {"vtk", true}, {"hdf5", true}}}});
  }

  Config::ConfigurationMesh<dim> make_mesh_config(const unsigned int n_cells)
  {
    const std::vector<Config::GridAxis> axis{Config::GridAxis(domain_min, dx_for_grid(n_cells), domain_max)};
    return Config::ConfigurationMesh<dim>(0u, axis, axis);
  }

  struct CellAverageSamples {
    VectorType values;
    VectorType measures;
  };

  template <typename ExactEvaluator>
  CellAverageSamples cell_averages(const Discretization &discretization, const ExactEvaluator &exact, const double time)
  {
    CellAverageSamples result;
    result.values.reinit(discretization.get_dof_handler().n_dofs());
    result.measures.reinit(discretization.get_dof_handler().n_dofs());

    QGauss<dim> quadrature(16);
    FEValues<dim> fe_values(discretization.get_mapping(), discretization.get_fe(), quadrature,
                            update_quadrature_points | update_JxW_values);
    std::vector<types::global_dof_index> dof_indices(discretization.get_fe().dofs_per_cell);

    for (const auto &cell : discretization.get_dof_handler().active_cell_iterators()) {
      fe_values.reinit(cell);
      cell->get_dof_indices(dof_indices);

      double measure = 0.0;
      double integral = 0.0;
      for (unsigned int q = 0; q < quadrature.size(); ++q) {
        const double weight = fe_values.JxW(q);
        measure += weight;
        integral += weight * exact(fe_values.quadrature_point(q), time);
      }

      const auto dof = dof_indices[0];
      result.values[dof] = integral / measure;
      result.measures[dof] = measure;
    }

    return result;
  }

  struct Diagnostics {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
    double mass = 0.0;
    double initial_mass = 0.0;
    double mass_error = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double initial_peak = 0.0;
    double peak_loss_relative = 0.0;
  };

  double weighted_mass(const VectorType &values, const VectorType &measures)
  {
    double mass = 0.0;
    for (unsigned int i = 0; i < values.size(); ++i)
      mass += measures[i] * values[i];
    return mass;
  }

  double max_value(const VectorType &values)
  {
    double maximum = -std::numeric_limits<double>::infinity();
    for (unsigned int i = 0; i < values.size(); ++i)
      maximum = std::max(maximum, values[i]);
    return maximum;
  }

  Diagnostics compute_diagnostics(const VectorType &numerical, const CellAverageSamples &exact,
                                  const VectorType &initial_values)
  {
    Diagnostics diagnostics;
    diagnostics.initial_mass = weighted_mass(initial_values, exact.measures);
    diagnostics.initial_peak = max_value(initial_values);

    for (unsigned int i = 0; i < numerical.size(); ++i) {
      const double value = numerical[i];
      const double error = value - exact.values[i];
      const double abs_error = std::abs(error);
      diagnostics.l1 += exact.measures[i] * abs_error;
      diagnostics.l2 += exact.measures[i] * error * error;
      diagnostics.linf = std::max(diagnostics.linf, abs_error);
      diagnostics.mass += exact.measures[i] * value;
      diagnostics.min = std::min(diagnostics.min, value);
      diagnostics.max = std::max(diagnostics.max, value);
    }

    diagnostics.l2 = std::sqrt(diagnostics.l2);
    diagnostics.mass_error = diagnostics.mass - diagnostics.initial_mass;
    diagnostics.peak_loss_relative = (diagnostics.initial_peak - diagnostics.max) / diagnostics.initial_peak;
    return diagnostics;
  }

  struct ConvergenceRow {
    unsigned int n_cells = 0;
    double dx = 0.0;
    double dt = 0.0;
    Diagnostics diagnostics;
    double observed_l1_order = std::numeric_limits<double>::quiet_NaN();
    std::filesystem::path output_dir;
  };

  ConvergenceRow run_case(const unsigned int n_cells, const std::filesystem::path &artifact_root)
  {
    kt_regression::ensure_logger();

    const double dt = cfl_timestep(n_cells);
    const ConfigTree json = make_run_json(n_cells, dt);
    RotatingGaussian2DModel model;
    Mesh mesh(make_mesh_config(n_cells));

    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);

    const std::string output_name = "rotating_gaussian_2D_" + std::to_string(n_cells) + "x" + std::to_string(n_cells);
    const std::filesystem::path output_dir = artifact_root / output_name;
    OutputPath data_out_path(output_dir.string(), output_name, "output");
    OutputSession<Discretization> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    TimeStepper time_stepper(json, assembler, data_out, *adaptor);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());
    const VectorType initial_values = state.spatial_data();

    time_stepper.run(state, 0.0, final_time);

    const RotatingGaussianExactSolution exact;
    const CellAverageSamples exact_final = cell_averages(discretization, exact, final_time);

    ConvergenceRow row;
    row.n_cells = n_cells;
    row.dx = dx_for_grid(n_cells);
    row.dt = dt;
    row.diagnostics = compute_diagnostics(state.spatial_data(), exact_final, initial_values);
    row.output_dir = output_dir;
    return row;
  }

  double observed_order(const ConvergenceRow &coarse, const ConvergenceRow &fine)
  {
    return std::log(coarse.diagnostics.l1 / fine.diagnostics.l1) / std::log(coarse.dx / fine.dx);
  }

  void write_csv(const std::filesystem::path &path, const std::vector<ConvergenceRow> &rows)
  {
    std::ofstream out(path);
    out << "grid,dx,dt,l1,l2,linf,observed_l1_order,mass,mass_error,min,max,initial_peak,peak_loss_relative,"
           "output_dir\n";
    out << std::scientific << std::setprecision(16);
    for (const auto &row : rows) {
      out << row.n_cells << 'x' << row.n_cells << ',' << row.dx << ',' << row.dt << ',' << row.diagnostics.l1 << ','
          << row.diagnostics.l2 << ',' << row.diagnostics.linf << ',';
      if (std::isfinite(row.observed_l1_order))
        out << row.observed_l1_order;
      else
        out << "";
      out << ',' << row.diagnostics.mass << ',' << row.diagnostics.mass_error << ',' << row.diagnostics.min << ','
          << row.diagnostics.max << ',' << row.diagnostics.initial_peak << ',' << row.diagnostics.peak_loss_relative
          << ',' << row.output_dir.string() << '\n';
    }
  }

  void print_table(const std::vector<ConvergenceRow> &rows, const std::filesystem::path &csv_path)
  {
    std::cout << "\n2D rotating Gaussian solid-body rotation convergence\n"
              << "artifacts: " << csv_path.parent_path().string() << '\n'
              << "csv: " << csv_path.string() << '\n'
              << std::scientific << std::setprecision(6) << std::setw(10) << "grid" << std::setw(14) << "dx"
              << std::setw(14) << "dt" << std::setw(14) << "L1" << std::setw(14) << "L2" << std::setw(14) << "Linf"
              << std::setw(14) << "ord(L1)" << std::setw(14) << "mass err" << std::setw(14) << "peak" << '\n';
    for (const auto &row : rows) {
      std::cout << std::setw(5) << row.n_cells << "x" << std::left << std::setw(4) << row.n_cells << std::right
                << std::setw(14) << row.dx << std::setw(14) << row.dt << std::setw(14) << row.diagnostics.l1
                << std::setw(14) << row.diagnostics.l2 << std::setw(14) << row.diagnostics.linf;
      if (std::isfinite(row.observed_l1_order))
        std::cout << std::setw(14) << row.observed_l1_order;
      else
        std::cout << std::setw(14) << "-";
      std::cout << std::setw(14) << row.diagnostics.mass_error << std::setw(14) << row.diagnostics.max << '\n';
    }
  }

  template <std::size_t n_grids>
  std::vector<ConvergenceRow> run_convergence_study(const std::array<unsigned int, n_grids> &grid_set,
                                                    const std::filesystem::path &artifact_root)
  {
    std::error_code ec;
    std::filesystem::remove_all(artifact_root, ec);
    std::filesystem::create_directories(artifact_root);

    std::vector<ConvergenceRow> rows;
    rows.reserve(grid_set.size());
    for (const unsigned int n_cells : grid_set) {
      rows.push_back(run_case(n_cells, artifact_root));
      if (rows.size() > 1) rows.back().observed_l1_order = observed_order(rows[rows.size() - 2], rows.back());
    }
    return rows;
  }

  void require_finite(const ConvergenceRow &row)
  {
    REQUIRE(std::isfinite(row.dx));
    REQUIRE(std::isfinite(row.dt));
    REQUIRE(std::isfinite(row.diagnostics.l1));
    REQUIRE(std::isfinite(row.diagnostics.l2));
    REQUIRE(std::isfinite(row.diagnostics.linf));
    REQUIRE(std::isfinite(row.diagnostics.mass));
    REQUIRE(std::isfinite(row.diagnostics.mass_error));
    REQUIRE(std::isfinite(row.diagnostics.min));
    REQUIRE(std::isfinite(row.diagnostics.max));
    REQUIRE(std::isfinite(row.diagnostics.initial_peak));
    REQUIRE(std::isfinite(row.diagnostics.peak_loss_relative));
  }

  void check_convergence_rows(const std::vector<ConvergenceRow> &rows, const std::filesystem::path &csv_path)
  {
    REQUIRE(std::filesystem::exists(csv_path));
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const auto &row = rows[i];
      require_finite(row);
      CHECK(std::filesystem::exists(row.output_dir / (row.output_dir.filename().string() + ".pvd")));
      CHECK(std::filesystem::exists(row.output_dir / (row.output_dir.filename().string() + ".h5")));
      CHECK(row.diagnostics.min >= bounded_min_tolerance);
      CHECK(row.diagnostics.max <= row.diagnostics.initial_peak * (1.0 + overshoot_tolerance));
      CHECK(std::abs(row.diagnostics.mass_error) < coarse_mass_tolerance);
      if (i > 0) CHECK(std::abs(row.diagnostics.mass_error) < refined_mass_tolerance);
    }

    for (std::size_t i = 1; i < rows.size(); ++i) {
      CHECK(rows[i].diagnostics.l1 < rows[i - 1].diagnostics.l1);
      CHECK(rows[i].diagnostics.l2 < rows[i - 1].diagnostics.l2);
      CHECK(rows[i].diagnostics.linf < rows[i - 1].diagnostics.linf);
      CHECK(rows[i].diagnostics.min > rows[i - 1].diagnostics.min);
      CHECK(rows[i].diagnostics.peak_loss_relative < rows[i - 1].diagnostics.peak_loss_relative);
    }

    CHECK(rows.back().diagnostics.min >= refined_min_tolerance);
    CHECK(rows.back().observed_l1_order > minimum_observed_l1_order);
  }
} // namespace

TEST_CASE("2D rotating Gaussian regression converges after half a solid-body revolution",
          "[2d][FV][KT][advection][slow]")
{
  const std::filesystem::path artifact_root =
      std::filesystem::current_path() / "rotating_gaussian_2D_regression_artifacts";
  const std::vector<ConvergenceRow> rows = run_convergence_study(regression_grid_sizes, artifact_root);
  const std::filesystem::path csv_path = artifact_root / "rotating_gaussian_2D_convergence.csv";
  write_csv(csv_path, rows);
  print_table(rows, csv_path);
  check_convergence_rows(rows, csv_path);
}

TEST_CASE("2D rotating Gaussian benchmark includes 256x256 grid after one revolution",
          "[.][2d][FV][KT][advection][benchmark]")
{
  const std::filesystem::path artifact_root =
      std::filesystem::current_path() / "rotating_gaussian_2D_benchmark_artifacts";
  const std::vector<ConvergenceRow> rows = run_convergence_study(benchmark_grid_sizes, artifact_root);
  const std::filesystem::path csv_path = artifact_root / "rotating_gaussian_2D_convergence.csv";
  write_csv(csv_path, rows);
  print_table(rows, csv_path);
  check_convergence_rows(rows, csv_path);
}
