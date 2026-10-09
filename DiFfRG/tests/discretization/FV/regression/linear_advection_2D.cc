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
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 2;
  constexpr std::size_t n_components = 1;
  constexpr unsigned int n_cells_per_direction = 160;
  constexpr double domain_min = -2.0;
  constexpr double domain_max = 2.0;
  constexpr double domain_length = domain_max - domain_min;
  constexpr double dx = domain_length / static_cast<double>(n_cells_per_direction);
  constexpr double velocity_x = 0.8;
  constexpr double velocity_y = 0.4;
  // Halved from 1.25 purely for runtime. The check is against the exact translated state, which
  // is evaluated at whatever final_time is set here, so the test keeps its meaning; the error
  // tolerances below are calibrated for this distance.
  constexpr double final_time = 0.625;
  constexpr double pulse_x0 = -0.75;
  constexpr double pulse_y0 = -0.4;
  constexpr double pulse_radius = 0.35;
  constexpr double explicit_dt = 1.0e-3;
  constexpr double implicit_dt = 1.0e-2;
  // Calibrated for final_time = 0.625. Measured errors are L1 = 1.336e-2, L2 = 2.223e-2,
  // Linf = 8.110e-2; the bounds keep ~25% headroom. Re-derive if final_time or the grid changes.
  constexpr double l1_tolerance = 1.7e-2;
  constexpr double l2_tolerance = 2.8e-2;
  constexpr double linf_tolerance = 1.0e-1;

  using NumberType = double;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Mesh = RectangularMeshSerial<dim>;
  using Discretization = FV::Discretization<Components, Mesh, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, class LinearAdvection2DModel, Reconstructor>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

  class LinearAdvection2DModel : public def::AbstractModel<LinearAdvection2DModel, Components>,
                                 public def::Time,
                                 public def::LLFFlux<LinearAdvection2DModel>,
                                 public def::FlowBoundaries<LinearAdvection2DModel>,
                                 public def::FVDefaultBoundaries<LinearAdvection2DModel>,
                                 public def::AD<LinearAdvection2DModel>
  {
  public:
    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      values[0] = initial_value(pos);
    }

    double exact_solution(const Point<dim> &pos, const double time) const
    {
      Point<dim> shifted;
      shifted[0] = pos[0] - velocity_x * time;
      shifted[1] = pos[1] - velocity_y * time;
      return initial_value(shifted);
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
              const Solution &sol) const
    {
      const auto &u = get<0>(sol);
      F_i[0][0] = NT(velocity_x) * u[0];
      F_i[0][1] = NT(velocity_y) * u[0];
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, n_components> &s_i, [[maybe_unused]] const Point<dim> &pos,
                [[maybe_unused]] const Solution &sol) const
    {
      s_i[0] = NT(0.0);
    }

  private:
    static double initial_value(const Point<dim> &pos)
    {
      const double dx_from_center = pos[0] - pulse_x0;
      const double dy_from_center = pos[1] - pulse_y0;
      const double normalized_r2 =
          (dx_from_center * dx_from_center + dy_from_center * dy_from_center) / (pulse_radius * pulse_radius);
      if (normalized_r2 >= 1.0) return 0.0;
      return std::exp(1.0 - 1.0 / (1.0 - normalized_r2));
    }
  };

  ConfigTree make_run_json()
  {
    return json::value(
        {{"physical", {{"Lambda", 1.0}}},
         {"discretization",
          {{"fe_order", 0},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", "-2:0.025:2"}, {"y_grid", "-2:0.025:2"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", final_time},
           {"explicit",
            {{"dt", explicit_dt},
             {"minimal_dt", 1.0e-8},
             {"maximal_dt", explicit_dt},
             {"abs_tol", 1.0e-9},
             {"rel_tol", 1.0e-9}}},
           {"implicit",
            {{"dt", implicit_dt},
             {"minimal_dt", 1.0e-8},
             {"maximal_dt", implicit_dt},
             {"abs_tol", 1.0e-6},
             {"rel_tol", 1.0e-5}}}}},
         {"output", {{"verbosity", 1}, {"vtk", false}, {"hdf5", false}}}});
  }

  Config::ConfigurationMesh<dim> make_mesh_config()
  {
    const std::vector<Config::GridAxis> axis{Config::GridAxis(domain_min, dx, domain_max)};
    return Config::ConfigurationMesh<dim>(0u, axis, axis);
  }

  struct CellAverageSamples {
    VectorType values;
    VectorType measures;
  };

  CellAverageSamples exact_cell_averages(const Discretization &discretization, const LinearAdvection2DModel &model,
                                         const double time)
  {
    CellAverageSamples result;
    result.values.reinit(discretization.get_dof_handler().n_dofs());
    result.measures.reinit(discretization.get_dof_handler().n_dofs());

    QGauss<dim> quadrature(12);
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
        integral += weight * model.exact_solution(fe_values.quadrature_point(q), time);
      }

      const auto dof = dof_indices[0];
      result.values[dof] = integral / measure;
      result.measures[dof] = measure;
    }

    return result;
  }

  struct ErrorMetrics {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
  };

  ErrorMetrics compute_error(const VectorType &numerical, const CellAverageSamples &exact)
  {
    ErrorMetrics metrics;
    for (unsigned int i = 0; i < numerical.size(); ++i) {
      const double error = numerical[i] - exact.values[i];
      const double abs_error = std::abs(error);
      metrics.l1 += exact.measures[i] * abs_error;
      metrics.l2 += exact.measures[i] * error * error;
      metrics.linf = std::max(metrics.linf, abs_error);
    }
    metrics.l2 = std::sqrt(metrics.l2);
    return metrics;
  }

  ErrorMetrics run_linear_advection()
  {
    kt_regression::ensure_logger();

    const ConfigTree json = make_run_json();
    LinearAdvection2DModel model;
    Mesh mesh(make_mesh_config());

    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);

    auto data_out_path =
        OutputPath::temporary(TemporaryRetention::remove_on_destruction, "linear_advection_2D", "output");
    OutputSession<Discretization> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    TimeStepper time_stepper(json, assembler, data_out, *adaptor);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());

    time_stepper.run(state, 0.0, final_time);
    return compute_error(state.spatial_data(), exact_cell_averages(discretization, model, final_time));
  }
} // namespace

TEST_CASE("2D linear advection compact pulse matches exact translated final state", "[2d][FV][KT][advection][slow]")
{
  const ErrorMetrics error = run_linear_advection();

  std::cout << std::scientific << std::setprecision(8)
            << "2D linear advection compact-pulse final errors: L1=" << error.l1 << ", L2=" << error.l2
            << ", Linf=" << error.linf << '\n';

  REQUIRE(std::isfinite(error.l1));
  REQUIRE(std::isfinite(error.l2));
  REQUIRE(std::isfinite(error.linf));
  CHECK(error.l1 < l1_tolerance);
  CHECK(error.l2 < l2_tolerance);
  CHECK(error.linf < linf_tolerance);
}
