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
  constexpr std::array<unsigned int, 4> grid_sizes{{32, 64, 128, 256}};
  constexpr double domain_min = -1.0;
  constexpr double domain_max = 1.0;
  constexpr double domain_length = domain_max - domain_min;
  constexpr double final_time = 0.1;
  constexpr double diffusivity = 0.01;
  constexpr double gaussian_alpha = 0.02;
  constexpr double gaussian_x0 = 0.0;
  constexpr double gaussian_y0 = 0.0;
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double diffusion_cfl = 0.1;
  constexpr double timestep_accuracy_cap = 1.0e-3;
  constexpr double boundary_tail_tolerance = 1.0e-5;
  constexpr double positivity_tolerance = -1.0e-8;
  constexpr double coarse_mass_tolerance = 2.0e-3;
  constexpr double finest_mass_tolerance = 5.0e-5;
  constexpr double symmetry_tolerance = 1.0e-8;
  constexpr double finest_width_tolerance = 5.0e-4;
  constexpr double minimum_observed_order = 1.25;

  using NumberType = double;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Mesh = RectangularMeshSerial<dim>;
  using Discretization = FV::Discretization<Components, Mesh, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, class HeatDiffusion2DModel, Reconstructor,
                                                  FV::KurganovTadmor::MaxEigenvalueWaveSpeed>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

  class GaussianHeatKernel2D
  {
  public:
    double operator()(const Point<dim> &point, const double time) const
    {
      const double beta_t = beta(time);
      const double dx = point[0] - gaussian_x0;
      const double dy = point[1] - gaussian_y0;
      const double r2 = dx * dx + dy * dy;
      return peak(time) * std::exp(-r2 / (4.0 * beta_t));
    }

    double beta(const double time) const { return gaussian_alpha + diffusivity * time; }
    double peak(const double time) const { return 1.0 / (4.0 * pi * beta(time)); }
    double axis_variance(const double time) const { return 2.0 * beta(time); }

    double square_domain_mass(const double time) const
    {
      const double argument = domain_max / (2.0 * std::sqrt(beta(time)));
      const double one_axis_mass = std::erf(argument);
      return one_axis_mass * one_axis_mass;
    }

    double boundary_to_peak_ratio(const double time) const
    {
      return std::exp(-domain_max * domain_max / (4.0 * beta(time)));
    }
  };

  class HeatDiffusion2DModel : public def::AbstractModel<HeatDiffusion2DModel, Components>,
                               public def::Time,
                               public def::LLFFlux<HeatDiffusion2DModel>,
                               public def::FlowBoundaries<HeatDiffusion2DModel>,
                               public def::FVDefaultBoundaries<HeatDiffusion2DModel>,
                               public def::AD<HeatDiffusion2DModel>
  {
  public:
    HeatDiffusion2DModel() { set_time(0.0); }

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      values[0] = exact(pos, 0.0);
    }

    double exact_solution(const Point<dim> &pos, const double time) const { return exact(pos, time); }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
              [[maybe_unused]] const Solution &sol) const
    {
      F_i[0] = Tensor<1, dim, NT>();
    }

    template <typename NT, typename Solution>
    void diffusion_flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
                        const Solution &sol) const
    {
      const auto &fe_derivatives = get<1>(sol);
      F_i[0] = -diffusivity * fe_derivatives[0];
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, n_components> &s_i, [[maybe_unused]] const Point<dim> &pos,
                [[maybe_unused]] const Solution &sol) const
    {
      s_i[0] = NT(0.0);
    }

  private:
    GaussianHeatKernel2D exact;
  };

  double dx_for_grid(const unsigned int n_cells) { return domain_length / static_cast<double>(n_cells); }

  double diffusion_timestep(const unsigned int n_cells)
  {
    const double dx = dx_for_grid(n_cells);
    const double raw_dt = std::min(timestep_accuracy_cap, diffusion_cfl * dx * dx / diffusivity);
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
            {{"dt", dt}, {"minimal_dt", dt * 1.0e-8}, {"maximal_dt", dt}, {"abs_tol", 1.0e-9}, {"rel_tol", 1.0e-9}}},
           {"implicit",
            {{"dt", dt},
             {"minimal_dt", dt * 1.0e-8},
             {"maximal_dt", dt},
             {"abs_tol", 1.0e-9},
             {"rel_tol", 1.0e-9},
             {"max_steps", 1000000}}}}},
         {"output", {{"verbosity", 1}, {"vtk", true}, {"hdf5", true}}}});
  }

  Config::ConfigurationMesh<dim> make_mesh_config(const unsigned int n_cells)
  {
    const std::vector<Config::GridAxis> axis{Config::GridAxis(domain_min, dx_for_grid(n_cells), domain_max)};
    return Config::ConfigurationMesh<dim>(0u, axis, axis);
  }

  struct ExactSamples {
    VectorType values;
    VectorType measures;
    double mass = 0.0;
    double moment_x = 0.0;
    double moment_y = 0.0;
  };

  ExactSamples exact_cell_averages(const Discretization &discretization, const GaussianHeatKernel2D &exact,
                                   const double time)
  {
    ExactSamples result;
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
        const auto &point = fe_values.quadrature_point(q);
        const double weight = fe_values.JxW(q);
        const double value = exact(point, time);
        measure += weight;
        integral += weight * value;
        result.mass += weight * value;
        result.moment_x += weight * point[0] * point[0] * value;
        result.moment_y += weight * point[1] * point[1] * value;
      }

      const auto dof = dof_indices[0];
      result.values[dof] = integral / measure;
      result.measures[dof] = measure;
    }

    result.moment_x /= result.mass;
    result.moment_y /= result.mass;
    return result;
  }

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

  std::size_t row_major_cell_index(const Point<dim> &pos, const unsigned int n_cells)
  {
    const double dx = dx_for_grid(n_cells);
    const auto index = [dx, n_cells](const double coordinate) {
      const auto raw = static_cast<int>(std::floor((coordinate - domain_min) / dx));
      return static_cast<unsigned int>(std::clamp(raw, 0, static_cast<int>(n_cells) - 1));
    };
    const unsigned int ix = index(pos[0]);
    const unsigned int iy = index(pos[1]);
    return static_cast<std::size_t>(iy) * n_cells + ix;
  }

  std::vector<double> row_major_cell_values(const Discretization &discretization, const VectorType &values,
                                            const unsigned int n_cells)
  {
    std::vector<double> result(static_cast<std::size_t>(n_cells) * n_cells, std::numeric_limits<double>::quiet_NaN());
    std::vector<unsigned int> counts(result.size(), 0);

    const auto &support_points = discretization.get_support_points();
    if (support_points.size() != values.size())
      throw std::runtime_error("Unexpected support-point/value size mismatch.");

    for (unsigned int dof = 0; dof < values.size(); ++dof) {
      const std::size_t row_major = row_major_cell_index(support_points[dof], n_cells);
      if (counts[row_major] != 0) throw std::runtime_error("Duplicate row-major cell index while sampling FV state.");
      result[row_major] = values[dof];
      ++counts[row_major];
    }

    for (const auto count : counts)
      if (count != 1) throw std::runtime_error("Missing row-major cell index while sampling FV state.");

    return result;
  }

  double symmetry_linf(const std::vector<double> &values, const unsigned int n_cells)
  {
    const auto at = [&](const unsigned int x, const unsigned int y) -> double {
      return values[static_cast<std::size_t>(y) * n_cells + x];
    };

    double maximum = 0.0;
    for (unsigned int y = 0; y < n_cells; ++y) {
      for (unsigned int x = 0; x < n_cells; ++x) {
        const double value = at(x, y);
        maximum = std::max(maximum, std::abs(value - at(n_cells - 1 - x, y)));
        maximum = std::max(maximum, std::abs(value - at(x, n_cells - 1 - y)));
        maximum = std::max(maximum, std::abs(value - at(y, x)));
      }
    }
    return maximum;
  }

  struct Diagnostics {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
    double mass = 0.0;
    double exact_mass = 0.0;
    double mass_error = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double initial_peak = 0.0;
    double moment_x = 0.0;
    double moment_y = 0.0;
    double exact_moment_x = 0.0;
    double exact_moment_y = 0.0;
    double width_error = 0.0;
    double moment_anisotropy = 0.0;
    double symmetry = 0.0;
    double boundary_tail_ratio = 0.0;
    double exact_domain_mass_defect = 0.0;
  };

  Diagnostics compute_diagnostics(const Discretization &discretization, const VectorType &numerical,
                                  const ExactSamples &exact, const VectorType &initial_values,
                                  const unsigned int n_cells)
  {
    Diagnostics diagnostics;
    diagnostics.initial_peak = max_value(initial_values);
    diagnostics.exact_mass = exact.mass;
    diagnostics.exact_moment_x = exact.moment_x;
    diagnostics.exact_moment_y = exact.moment_y;

    const auto &support_points = discretization.get_support_points();
    for (unsigned int i = 0; i < numerical.size(); ++i) {
      const double value = numerical[i];
      const double error = value - exact.values[i];
      const double abs_error = std::abs(error);
      diagnostics.l1 += exact.measures[i] * abs_error;
      diagnostics.l2 += exact.measures[i] * error * error;
      diagnostics.linf = std::max(diagnostics.linf, abs_error);
      diagnostics.mass += exact.measures[i] * value;
      diagnostics.moment_x += exact.measures[i] * support_points[i][0] * support_points[i][0] * value;
      diagnostics.moment_y += exact.measures[i] * support_points[i][1] * support_points[i][1] * value;
      diagnostics.min = std::min(diagnostics.min, value);
      diagnostics.max = std::max(diagnostics.max, value);
    }

    diagnostics.l2 = std::sqrt(diagnostics.l2);
    diagnostics.mass_error = diagnostics.mass - diagnostics.exact_mass;
    diagnostics.moment_x /= diagnostics.mass;
    diagnostics.moment_y /= diagnostics.mass;
    diagnostics.width_error = 0.5 * (std::abs(diagnostics.moment_x - diagnostics.exact_moment_x) +
                                     std::abs(diagnostics.moment_y - diagnostics.exact_moment_y));
    diagnostics.moment_anisotropy = std::abs(diagnostics.moment_x - diagnostics.moment_y);
    diagnostics.symmetry = symmetry_linf(row_major_cell_values(discretization, numerical, n_cells), n_cells);

    GaussianHeatKernel2D exact_solution;
    diagnostics.boundary_tail_ratio = exact_solution.boundary_to_peak_ratio(final_time);
    diagnostics.exact_domain_mass_defect = 1.0 - exact_solution.square_domain_mass(final_time);
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

    const double dt = diffusion_timestep(n_cells);
    const ConfigTree json = make_run_json(n_cells, dt);
    HeatDiffusion2DModel model;
    Mesh mesh(make_mesh_config(n_cells));

    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);

    const std::string output_name = "heat_diffusion_2D_" + std::to_string(n_cells) + "x" + std::to_string(n_cells);
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

    const GaussianHeatKernel2D exact;
    const ExactSamples exact_final = exact_cell_averages(discretization, exact, final_time);

    ConvergenceRow row;
    row.n_cells = n_cells;
    row.dx = dx_for_grid(n_cells);
    row.dt = dt;
    row.diagnostics = compute_diagnostics(discretization, state.spatial_data(), exact_final, initial_values, n_cells);
    row.output_dir = output_dir;
    return row;
  }

  double observed_order(const ConvergenceRow &coarse, const ConvergenceRow &fine)
  {
    return std::log(coarse.diagnostics.l1 / fine.diagnostics.l1) / std::log(coarse.dx / fine.dx);
  }

  std::vector<ConvergenceRow> run_convergence_study(const std::filesystem::path &artifact_root)
  {
    std::error_code ec;
    std::filesystem::remove_all(artifact_root, ec);
    std::filesystem::create_directories(artifact_root);

    std::vector<ConvergenceRow> rows;
    rows.reserve(grid_sizes.size());
    for (const unsigned int n_cells : grid_sizes) {
      rows.push_back(run_case(n_cells, artifact_root));
      if (rows.size() > 1) rows.back().observed_l1_order = observed_order(rows[rows.size() - 2], rows.back());
    }
    return rows;
  }

  void write_csv(const std::filesystem::path &path, const std::vector<ConvergenceRow> &rows)
  {
    std::ofstream out(path);
    out << "grid,dx,dt,l1,l2,linf,observed_l1_order,mass,exact_mass,mass_error,min,max,moment_x,moment_y,"
           "exact_moment_x,exact_moment_y,width_error,moment_anisotropy,symmetry_linf,boundary_tail_ratio,"
           "exact_domain_mass_defect,output_dir\n";
    out << std::scientific << std::setprecision(16);
    for (const auto &row : rows) {
      out << row.n_cells << 'x' << row.n_cells << ',' << row.dx << ',' << row.dt << ',' << row.diagnostics.l1 << ','
          << row.diagnostics.l2 << ',' << row.diagnostics.linf << ',';
      if (std::isfinite(row.observed_l1_order))
        out << row.observed_l1_order;
      else
        out << "";
      out << ',' << row.diagnostics.mass << ',' << row.diagnostics.exact_mass << ',' << row.diagnostics.mass_error
          << ',' << row.diagnostics.min << ',' << row.diagnostics.max << ',' << row.diagnostics.moment_x << ','
          << row.diagnostics.moment_y << ',' << row.diagnostics.exact_moment_x << ',' << row.diagnostics.exact_moment_y
          << ',' << row.diagnostics.width_error << ',' << row.diagnostics.moment_anisotropy << ','
          << row.diagnostics.symmetry << ',' << row.diagnostics.boundary_tail_ratio << ','
          << row.diagnostics.exact_domain_mass_defect << ',' << row.output_dir.string() << '\n';
    }
  }

  void print_table(const std::vector<ConvergenceRow> &rows, const std::filesystem::path &csv_path)
  {
    std::cout << "\n2D heat diffusion Gaussian convergence\n"
              << "artifacts: " << csv_path.parent_path().string() << '\n'
              << "csv: " << csv_path.string() << '\n'
              << std::scientific << std::setprecision(6) << std::setw(10) << "grid" << std::setw(14) << "dx"
              << std::setw(14) << "dt" << std::setw(14) << "L1" << std::setw(14) << "L2" << std::setw(14) << "Linf"
              << std::setw(14) << "ord(L1)" << std::setw(14) << "mass err" << std::setw(14) << "min" << std::setw(14)
              << "max" << '\n';
    for (const auto &row : rows) {
      std::cout << std::setw(5) << row.n_cells << "x" << std::left << std::setw(4) << row.n_cells << std::right
                << std::setw(14) << row.dx << std::setw(14) << row.dt << std::setw(14) << row.diagnostics.l1
                << std::setw(14) << row.diagnostics.l2 << std::setw(14) << row.diagnostics.linf;
      if (std::isfinite(row.observed_l1_order))
        std::cout << std::setw(14) << row.observed_l1_order;
      else
        std::cout << std::setw(14) << "-";
      std::cout << std::setw(14) << row.diagnostics.mass_error << std::setw(14) << row.diagnostics.min << std::setw(14)
                << row.diagnostics.max << '\n';
    }
  }

  void require_finite(const ConvergenceRow &row)
  {
    REQUIRE(std::isfinite(row.dx));
    REQUIRE(std::isfinite(row.dt));
    REQUIRE(std::isfinite(row.diagnostics.l1));
    REQUIRE(std::isfinite(row.diagnostics.l2));
    REQUIRE(std::isfinite(row.diagnostics.linf));
    REQUIRE(std::isfinite(row.diagnostics.mass));
    REQUIRE(std::isfinite(row.diagnostics.exact_mass));
    REQUIRE(std::isfinite(row.diagnostics.mass_error));
    REQUIRE(std::isfinite(row.diagnostics.min));
    REQUIRE(std::isfinite(row.diagnostics.max));
    REQUIRE(std::isfinite(row.diagnostics.moment_x));
    REQUIRE(std::isfinite(row.diagnostics.moment_y));
    REQUIRE(std::isfinite(row.diagnostics.width_error));
    REQUIRE(std::isfinite(row.diagnostics.moment_anisotropy));
    REQUIRE(std::isfinite(row.diagnostics.symmetry));
    REQUIRE(std::isfinite(row.diagnostics.boundary_tail_ratio));
    REQUIRE(std::isfinite(row.diagnostics.exact_domain_mass_defect));
  }

  void check_convergence_rows(const std::vector<ConvergenceRow> &rows, const std::filesystem::path &csv_path)
  {
    REQUIRE(std::filesystem::exists(csv_path));
    REQUIRE(rows.size() == grid_sizes.size());

    for (const auto &row : rows) {
      require_finite(row);
      CHECK(std::filesystem::exists(row.output_dir / (row.output_dir.filename().string() + ".pvd")));
      CHECK(std::filesystem::exists(row.output_dir / (row.output_dir.filename().string() + ".h5")));
      CHECK(row.diagnostics.boundary_tail_ratio < boundary_tail_tolerance);
      CHECK(row.diagnostics.exact_domain_mass_defect < boundary_tail_tolerance);
      CHECK(row.diagnostics.min >= positivity_tolerance);
      CHECK(row.diagnostics.max <= row.diagnostics.initial_peak);
      CHECK(std::abs(row.diagnostics.mass_error) < coarse_mass_tolerance);
      CHECK(row.diagnostics.moment_anisotropy < symmetry_tolerance);
      CHECK(row.diagnostics.symmetry < symmetry_tolerance);
    }

    for (std::size_t i = 1; i < rows.size(); ++i) {
      CHECK(rows[i].diagnostics.l1 < rows[i - 1].diagnostics.l1);
      CHECK(rows[i].diagnostics.l2 < rows[i - 1].diagnostics.l2);
      CHECK(rows[i].diagnostics.linf < rows[i - 1].diagnostics.linf);
      CHECK(std::abs(rows[i].diagnostics.mass_error) < std::abs(rows[i - 1].diagnostics.mass_error));
      CHECK(rows[i].diagnostics.width_error < rows[i - 1].diagnostics.width_error);
    }

    CHECK(rows.back().observed_l1_order > minimum_observed_order);
    CHECK(std::abs(rows.back().diagnostics.mass_error) < finest_mass_tolerance);
    CHECK(rows.back().diagnostics.width_error < finest_width_tolerance);
  }
} // namespace

TEST_CASE("2D heat diffusion Gaussian converges to exact broadening", "[2d][FV][KT][diffusion][slow]")
{
  const std::filesystem::path artifact_root =
      std::filesystem::current_path() / "heat_diffusion_2D_regression_artifacts";
  const std::vector<ConvergenceRow> rows = run_convergence_study(artifact_root);
  const std::filesystem::path csv_path = artifact_root / "heat_diffusion_2D_convergence.csv";
  write_csv(csv_path, rows);
  print_table(rows, csv_path);
  check_convergence_rows(rows, csv_path);
}
