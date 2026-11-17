#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>

#include <boilerplate/eom_accuracy.hh>
#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>

namespace
{
  using namespace dealii;
  using namespace DiFfRG;

  ConfigTree make_json(const int fe_order, const uint refinement = 0)
  {
    return json::value(
        {{"physical", {}},
         {"integration",
          {{"x_quadrature_order", 32},
           {"angle_quadrature_order", 8},
           {"x0_quadrature_order", 16},
           {"x0_summands", 8},
           {"q0_quadrature_order", 16},
           {"q0_summands", 8},
           {"x_extent_tolerance", 1e-3},
           {"x0_extent_tolerance", 1e-3},
           {"q0_extent_tolerance", 1e-3},
           {"jacobian_quadrature_factor", 0.5}}},
         {"discretization",
          {{"fe_order", fe_order},
           {"overintegration", 0},
           {"output_subdivisions", 2},
           {"EoM_abs_tol", 1e-12},
           {"EoM_max_iter", 200},
           {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", refinement}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1e-1},
             {"level", 0},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 5e-2}}}}},
         {"timestepping",
          {{"final_time", 1.},
           {"output_dt", 1e-1},
           {"explicit",
            {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}},
           {"implicit",
            {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}}}},
         {"output", {{"live_plot", false}, {"verbosity", 0}}}});
  }

  void setup_logger()
  {
    try {
      auto log = spdlog::stdout_color_mt("log");
      log->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
  }

  template <int dim>
  Testing::PhysicalParameters parameters_from_affine(const std::array<double, dim> &x0,
                                                     const std::array<double, dim> &x1)
  {
    Testing::PhysicalParameters prm;
    for (uint d = 0; d < dim; ++d) {
      prm.initial_x0[d] = x0[d];
      prm.initial_x1[d] = x1[d];
    }
    return prm;
  }

  template <int dim> Point<dim> affine_minimum(const Testing::PhysicalParameters &prm)
  {
    Point<dim> minimum;
    for (uint d = 0; d < dim; ++d)
      minimum[d] = prm.initial_x0[d] > 0. ? 0. : -prm.initial_x0[d] / prm.initial_x1[d];
    return minimum;
  }

  template <int dim> Testing::PhysicalParameters parameters_with_minimum(const Point<dim> &minimum)
  {
    Testing::PhysicalParameters prm;
    for (uint d = 0; d < dim; ++d) {
      prm.initial_x1[d] = 2.5 + d;
      prm.initial_x0[d] = -prm.initial_x1[d] * minimum[d];
    }
    return prm;
  }

  Testing::PhysicalParameters parameters_with_boundary_minimum()
  {
    Testing::PhysicalParameters prm;
    for (uint d = 0; d < 3; ++d) {
      prm.initial_x0[d] = 1.;
      prm.initial_x1[d] = 1.;
    }
    return prm;
  }

  template <int dim> double distance(const Point<dim> &a, const Point<dim> &b)
  {
    double out = 0.;
    for (uint d = 0; d < dim; ++d)
      out = std::max(out, std::abs(a[d] - b[d]));
    return out;
  }

  template <int dim>
  class ModelStitchedPolynomial
      : public def::AbstractModel<ModelStitchedPolynomial<dim>, typename Testing::compFactory<dim>::value>,
        public def::Time,
        public def::NoNumFlux<ModelStitchedPolynomial<dim>>,
        public def::FlowBoundaries<ModelStitchedPolynomial<dim>>,
        public def::AD<ModelStitchedPolynomial<dim>>
  {
  public:
    explicit ModelStitchedPolynomial(const Point<dim> &roots) : roots(roots) {}

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      for (uint d = 0; d < dim; ++d)
        values[d] = value(d, pos[d]);
    }

    template <typename Vector> std::array<double, dim> EoM(const Point<dim> &, const Vector &values) const
    {
      std::array<double, dim> out{{}};
      for (uint d = 0; d < dim; ++d)
        out[d] = values[d];
      return out;
    }

  private:
    static constexpr std::array<double, 3> left_linear{{1.25, 1.75, 2.25}};
    static constexpr std::array<double, 3> right_linear{{5.0, 4.0, 3.5}};
    static constexpr std::array<double, 3> left_cubic{{2.0, 2.5, 3.0}};
    static constexpr std::array<double, 3> right_cubic{{10.0, 8.0, 6.0}};

    double value(const uint d, const double x) const
    {
      const double s = x - roots[d];
      const double linear = s < 0. ? left_linear[d] : right_linear[d];
      const double cubic = s < 0. ? left_cubic[d] : right_cubic[d];
      return linear * s + cubic * s * s * s;
    }

    const Point<dim> roots;
  };

  template <int dim>
  class ModelAffineEoM : public def::AbstractModel<ModelAffineEoM<dim>, typename Testing::compFactory<dim>::value>,
                         public def::Time,
                         public def::NoNumFlux<ModelAffineEoM<dim>>,
                         public def::FlowBoundaries<ModelAffineEoM<dim>>,
                         public def::AD<ModelAffineEoM<dim>>
  {
  public:
    ModelAffineEoM(const Point<dim> &minimum, const std::array<std::array<double, dim>, dim> &hessian,
                   const std::array<double, dim> &linear = {})
        : minimum(minimum), hessian(hessian), linear(linear)
    {
    }

    Point<dim> reference_point() const { return minimum; }
    Tensor<2, dim> reference_derivative() const
    {
      Tensor<2, dim> derivative;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          derivative[i][j] = hessian[i][j];
      return derivative;
    }

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      for (uint d = 0; d < dim; ++d) {
        values[d] = linear[d];
        for (uint e = 0; e < dim; ++e)
          values[d] += hessian[d][e] * (pos[e] - minimum[e]);
      }
    }

    template <typename Vector> std::array<double, dim> EoM(const Point<dim> &, const Vector &values) const
    {
      std::array<double, dim> out{};
      for (uint d = 0; d < dim; ++d)
        out[d] = values[d];
      return out;
    }

  private:
    const Point<dim> minimum;
    const std::array<std::array<double, dim>, dim> hessian;
    const std::array<double, dim> linear;
  };

  // u = V'(rho) = m2 + lambda rho with a linear source: the EoM V' - c / sqrt(2 rho) is singular at rho = 0.
  class ModelSourceEoM : public def::AbstractModel<ModelSourceEoM, typename Testing::compFactory<1>::value>,
                         public def::Time,
                         public def::NoNumFlux<ModelSourceEoM>,
                         public def::FlowBoundaries<ModelSourceEoM>,
                         public def::AD<ModelSourceEoM>
  {
  public:
    ModelSourceEoM(const double m2, const double lambda, const double c) : m2(m2), lambda(lambda), c(c) {}

    template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
    {
      values[0] = m2 + lambda * pos[0];
    }

    template <typename Vector> std::array<double, 1> EoM(const Point<1> &pos, const Vector &values) const
    {
      return {{double(values[0]) - c / std::sqrt(2. * pos[0] + 1e-300)}};
    }

    // The zero of m2 + lambda rho - c / sqrt(2 rho), by bisection.
    double exact_minimum() const
    {
      double a = 1e-300, b = 1.;
      for (uint i = 0; i < 200; ++i) {
        const double m = 0.5 * (a + b);
        (m2 + lambda * m - c / std::sqrt(2. * m) < 0. ? a : b) = m;
      }
      return 0.5 * (a + b);
    }

  private:
    const double m2, lambda, c;
  };

  // EoM field V'(rho) = (rho - 0.03)(rho - 0.06): a minimum at the boundary rho = 0, a maximum at 0.03 and a
  // higher, metastable minimum at 0.06 (V(0.06) - V(0) = 1.8e-5).
  class ModelTwoMinima : public def::AbstractModel<ModelTwoMinima, typename Testing::compFactory<1>::value>,
                         public def::Time,
                         public def::NoNumFlux<ModelTwoMinima>,
                         public def::FlowBoundaries<ModelTwoMinima>,
                         public def::AD<ModelTwoMinima>
  {
  public:
    template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
    {
      values[0] = (pos[0] - 0.03) * (pos[0] - 0.06);
    }

    template <typename Vector> std::array<double, 1> EoM(const Point<1> &pos, const Vector &) const
    {
      return {{(pos[0] - 0.03) * (pos[0] - 0.06)}};
    }
  };

  class QMDVacuumModel : public def::AbstractModel<QMDVacuumModel, typename Testing::compFactory<2>::value>,
                         public def::Time,
                         public def::NoNumFlux<QMDVacuumModel>,
                         public def::FlowBoundaries<QMDVacuumModel>,
                         public def::AD<QMDVacuumModel>
  {
  public:
    static constexpr double sigma_mass_square = 0.7;
    static constexpr double diquark_mass_square = 0.94;
    static constexpr double diquark_quartic = 0.1;
    static constexpr double explicit_breaking = 0.00175;

    template <typename Vector> void initial_condition(const Point<2> &point, Vector &values) const
    {
      const double sigma = point[0];
      const double delta = point[1];
      values[0] = sigma_mass_square * sigma;
      values[1] = diquark_mass_square * delta + diquark_quartic * delta * delta * delta;
    }

    template <typename Vector> std::array<double, 2> EoM(const Point<2> &, const Vector &values) const
    {
      return {{values[0] - explicit_breaking, values[1]}};
    }
  };

  class CubicGradientModel : public def::AbstractModel<CubicGradientModel, typename Testing::compFactory<1>::value>,
                             public def::Time,
                             public def::NoNumFlux<CubicGradientModel>,
                             public def::FlowBoundaries<CubicGradientModel>,
                             public def::AD<CubicGradientModel>
  {
  public:
    template <typename Vector> void initial_condition(const Point<1> &point, Vector &values) const
    {
      values[0] = point[0] * point[0] * point[0];
    }
  };

  class KinkedRadialPotentialGradientModel
      : public def::AbstractModel<KinkedRadialPotentialGradientModel, typename Testing::compFactory<2>::value>,
        public def::Time,
        public def::NoNumFlux<KinkedRadialPotentialGradientModel>,
        public def::FlowBoundaries<KinkedRadialPotentialGradientModel>,
        public def::AD<KinkedRadialPotentialGradientModel>
  {
  public:
    static constexpr double mass_square = 0.08;
    static constexpr double inner_slope = 0.15;
    static constexpr double outer_slope = 1.6;
    static constexpr double kink_radius = 0.437;
    static constexpr double kink_rho = 0.5 * kink_radius * kink_radius;

    template <typename Vector> void initial_condition(const Point<2> &point, Vector &values) const
    {
      const double radius_square = point[0] * point[0] + point[1] * point[1];
      const double transverse_mass = transverse_mass_square(radius_square);
      values[0] = transverse_mass * point[0];
      values[1] = transverse_mass * point[1];
    }

    static double transverse_mass_square(const double radius_square)
    {
      const double rho = 0.5 * radius_square;
      if (rho <= kink_rho) return mass_square + inner_slope * rho;
      return mass_square + inner_slope * kink_rho + outer_slope * (rho - kink_rho);
    }

    static double radial_mass_square(const double radius_square)
    {
      const double rho = 0.5 * radius_square;
      const double slope = rho <= kink_rho ? inner_slope : outer_slope;
      return transverse_mass_square(radius_square) + 2. * rho * slope;
    }
  };

  struct AccuracyMetrics {
    double rms = 0.;
    double max_abs = 0.;
    double bias = 0.;
    double roughness_rms = 0.;
  };

  AccuracyMetrics accuracy_metrics(const std::vector<double> &values, const std::vector<double> &exact)
  {
    REQUIRE(values.size() == exact.size());
    REQUIRE(values.size() >= 3);

    AccuracyMetrics metrics;
    std::vector<double> error(values.size());
    for (uint i = 0; i < values.size(); ++i) {
      error[i] = values[i] - exact[i];
      metrics.rms += error[i] * error[i];
      metrics.max_abs = std::max(metrics.max_abs, std::abs(error[i]));
      metrics.bias += error[i];
    }
    metrics.rms = std::sqrt(metrics.rms / static_cast<double>(error.size()));
    metrics.bias /= static_cast<double>(error.size());
    for (uint i = 1; i + 1 < error.size(); ++i) {
      const double second_difference = error[i + 1] - 2. * error[i] + error[i - 1];
      metrics.roughness_rms += second_difference * second_difference;
    }
    metrics.roughness_rms = std::sqrt(metrics.roughness_rms / static_cast<double>(error.size() - 2));
    return metrics;
  }

  void print_accuracy(const std::string_view profile, const uint refinement, const std::string_view method,
                      const AccuracyMetrics &metrics)
  {
    std::cout << "mass accuracy | " << profile << " | refinement " << refinement << " | " << method
              << " | rms=" << metrics.rms << " max=" << metrics.max_abs << " bias=" << metrics.bias
              << " roughness=" << metrics.roughness_rms << '\n';
  }

  template <int dim, template <typename...> typename DiscretizationTemplate, typename Model>
  auto locate_with_model(const Model &model, const int fe_order, const uint refinement = 0,
                         const std::optional<Point<dim>> &initial_guess = std::nullopt,

                         const bool report_accuracy = false)
  {
    using NumberType = double;
    using Discretization = DiscretizationTemplate<Model, RectangularMesh<dim>, NumberType>;
    using VectorType = typename Discretization::VectorType;

    setup_logger();

    auto json = make_json(fe_order, refinement);
    RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
    Discretization discretization(mesh, json);

    FE::FlowingVariables initial_condition(discretization);
    initial_condition.interpolate(model);
    const VectorType &src = initial_condition.spatial_data();

    const auto &dof_handler = discretization.get_dof_handler();
    const auto &mapping = discretization.get_mapping();

    auto EoM_cell = dof_handler.begin_active();
    auto EoM_config = Config::EoMConfig(json);
    EoM_config.validate();
    auto result = get_EoM_result(
        EoM_cell, src, dof_handler, mapping, [&](const auto &p, const auto &values) { return model.EoM(p, values); },
        [&](const auto &p, const auto &) { return p; }, EoM_config, initial_guess);
    if constexpr (requires {
                    model.reference_point();
                    model.reference_derivative();
                  }) {
      if (report_accuracy) {
        const auto du = Testing::direct_eom_state_derivative(src, dof_handler, mapping, result.point);
        Testing::print_direct_eom_accuracy(fe_order == 0 ? "FV affine" : "FE affine", result.point,
                                           model.reference_point(), du, model.reference_derivative());
      }
    }
    return result;
  }

  template <int dim, template <typename...> typename DiscretizationTemplate, typename Model>
  Point<dim> locate_minimum_with_model(const Model &model, const int fe_order, const uint refinement = 0,
                                       const std::optional<Point<dim>> &initial_guess = std::nullopt)
  {
    return locate_with_model<dim, DiscretizationTemplate>(model, fe_order, refinement, initial_guess).point;
  }

  template <int dim, template <typename...> typename DiscretizationTemplate>
  Point<dim> locate_minimum(const Testing::PhysicalParameters &prm, const int fe_order, const uint refinement = 0)
  {
    using Model = Testing::ModelConstant<dim, dim>;
    Model model(prm);
    return locate_minimum_with_model<dim, DiscretizationTemplate>(model, fe_order, refinement);
  }
} // namespace

TEST_CASE("CG EoM direct search finds 1D affine minima", "[discretization][EoM][cg][1d]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const int fe_order = GENERATE(1, 2, 3, 4, 5);
  const double slope = GENERATE(1.25, 2., 2.5, 5., 10.);
  const auto prm = parameters_from_affine<dim>(std::array<double, dim>{{-1.}}, std::array<double, dim>{{slope}});
  const auto expected = affine_minimum<dim>(prm);
  const auto EoM = locate_minimum<dim, CG::Discretization>(prm, fe_order);

  REQUIRE(distance(EoM, expected) < 6e-2);
}

TEST_CASE("CG EoM direct search finds 2D affine and boundary minima", "[discretization][EoM][cg][2d]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const int fe_order = GENERATE(1, 2, 3, 4, 5);
  const uint case_index = GENERATE(0u, 1u, 2u, 3u);
  const std::array<std::array<double, 4>, 4> cases{{
      {{-1., 2.5, -1., 5.}},
      {{1., 2., -1., 2.5}},
      {{-1., 5., 1., 4.}},
      {{1., 3., 1., 4.}},
  }};
  const auto &c = cases[case_index];
  const auto prm =
      parameters_from_affine<dim>(std::array<double, dim>{{c[0], c[2]}}, std::array<double, dim>{{c[1], c[3]}});
  const auto expected = affine_minimum<dim>(prm);
  const auto EoM = locate_minimum<dim, CG::Discretization>(prm, fe_order);

  REQUIRE(distance(EoM, expected) < 8e-2);
}

TEST_CASE("CG EoM direct search finds 3D affine minima", "[discretization][EoM][cg][3d]")
{
  DiFfRG::Init();
  constexpr uint dim = 3;
  const int fe_order = GENERATE(1, 2);
  const uint case_index = GENERATE(0u, 1u);
  const std::array<std::array<double, 6>, 2> cases{{
      {{-1., 2.5, -1., 2., -1., 1.6666666666666667}},
      {{-1., 5., -1., 2.5, -1., 1.25}},
  }};
  const auto &c = cases[case_index];
  const auto prm = parameters_from_affine<dim>(std::array<double, dim>{{c[0], c[2], c[4]}},
                                               std::array<double, dim>{{c[1], c[3], c[5]}});
  const auto expected = affine_minimum<dim>(prm);
  const auto EoM = locate_minimum<dim, CG::Discretization>(prm, fe_order);

  REQUIRE(distance(EoM, expected) < 1.1e-1);
}

TEST_CASE("Direct FV vacuum root and vector Hessian accuracy under mesh coarsening",
          "[discretization][EoM][direct][fv][accuracy][coarsening]")
{
  DiFfRG::Init();
  setup_logger();
  constexpr unsigned int dim = 2;
  using Model = QMDVacuumModel;
  using Discretization = FV::Discretization<Model, RectangularMesh<dim>>;
  const double requested_h = GENERATE(.005, .01, .02, .04, .05, .1);
  const unsigned int fit_mode = GENERATE(0u, 1u, 2u);
  auto json = make_json(0);
  const std::string grid = "0:" + std::to_string(requested_h) + ":0.2";
  json.set_string("/discretization/grid/x_grid", grid);
  json.set_string("/discretization/grid/y_grid", grid);
  Model model;
  RectangularMesh<dim> mesh(Config::ConfigurationMesh<dim>(json), RectangularMeshOptions{.origin_cell_centered = true});
  Discretization discretization(mesh, json);
  FE::FlowingVariables state(discretization);
  state.interpolate(model);
  const auto &dofs = discretization.get_dof_handler();
  const auto &mapping = discretization.get_mapping();
  {
    // FV data are always the exact cell averages of the cubic vacuum field.
    std::vector<types::global_dof_index> indices(dofs.get_fe().n_dofs_per_cell());
    for (const auto &cell : dofs.active_cell_iterators()) {
      cell->get_dof_indices(indices);
      const double delta = cell->center()[1];
      const double width = cell->extent_in_direction(1);
      for (unsigned int i = 0; i < indices.size(); ++i)
        if (dofs.get_fe().system_to_component_index(i).first == 1)
          state.spatial_data()[indices[i]] += Model::diquark_quartic * delta * width * width / 4.;
    }
  }
  auto config = Config::EoMConfig(json);
  config.patch_order = fit_mode == 0 ? 0 : 3;
  if (fit_mode >= 2) {
    config.patch_parity = {{-1, 1}, {1, -1}};
    config.patch_direction = {1., 0.};
    config.patch_skip_layers = 0;
  }
  const auto bounds = mesh.get_physical_bounds();
  auto cell = dofs.begin_active();
  const auto result = get_EoM_result<dim>(
      cell, state.spatial_data(), dofs, mapping,
      [&](const auto &point, const auto &values) { return model.EoM(point, values); },
      [](const auto &point, const auto &) { return point; }, config, std::optional<Point<dim>>{}, nullptr, &bounds,
      EoMDataKind::finite_volume);
  const auto du = result.local_fit
                      ? result.local_fit->state_jacobian(result.search_point)
                      : Testing::direct_eom_state_derivative(state.spatial_data(), dofs, mapping, result.point);
  const Point<dim> exact_root(Model::explicit_breaking / Model::sigma_mass_square, 0.);
  Tensor<2, dim> exact_du;
  exact_du[0][0] = Model::sigma_mass_square;
  exact_du[1][1] = Model::diquark_mass_square;
  Testing::print_direct_eom_accuracy("FV vacuum requested_h=" + std::to_string(requested_h) +
                                         " actual_h=" + std::to_string(dofs.begin_active()->extent_in_direction(0)) +
                                         " mode=" + std::to_string(fit_mode) +
                                         " cells=" + std::to_string(mesh.get_triangulation().n_active_cells()),
                                     result.point, exact_root, du, exact_du);
  CAPTURE(requested_h);
  CAPTURE(fit_mode);
  REQUIRE(result.candidates.size() == 1);
  CHECK(result.reconstruction_full_rank);
  CHECK(result.candidates[0].residual < 1e-10);
  CHECK(result.point.distance(exact_root) < .01 * requested_h);
  CHECK(std::abs(du[1][1] - exact_du[1][1]) < .01 * Model::diquark_mass_square);
  REQUIRE(result.hessian.has_value());
  if (fit_mode >= 2) REQUIRE(result.local_fit != nullptr);
  if (fit_mode >= 2) {
    CHECK(result.point.distance(exact_root) < 1e-10);
    CHECK((*result.hessian - exact_du).norm() < 1e-10);
    for (const auto order : result.local_fit->orders)
      CHECK(order == 3);
  }
}

TEST_CASE("CG2 EoM direct search refines an off-grid 1D minimum", "[discretization][EoM][direct][cg2][minimum]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const Point<dim> expected(0.437);
  const auto EoM = locate_minimum<dim, CG::Discretization>(parameters_with_minimum(expected), 2);

  REQUIRE(distance(EoM, expected) < 1e-10);
}

TEST_CASE("CG2 EoM direct search refines multidimensional and constrained minima",
          "[discretization][EoM][direct][cg2][minimum]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;

  SECTION("rotated interior minimum")
  {
    const Point<dim> expected(0.437, 0.583);
    const ModelAffineEoM<dim> model(expected, {{{4., 1.25}, {1.25, 3.}}});
    const auto EoM = locate_minimum_with_model<dim, CG::Discretization>(model, 2);

    REQUIRE(distance(EoM, expected) < 1e-9);
  }

  SECTION("edge minimum away from support points")
  {
    const Point<dim> unconstrained_minimum(0.437, -0.2);
    const Point<dim> expected(0.437, 0.);
    const ModelAffineEoM<dim> model(unconstrained_minimum, {{{4., 0.}, {0., 3.}}});
    const auto EoM = locate_minimum_with_model<dim, CG::Discretization>(model, 2);

    REQUIRE(distance(EoM, expected) < 1e-9);
  }

  SECTION("monotonic corner minimum")
  {
    const Point<dim> expected(0., 0.);
    const auto EoM = locate_minimum<dim, CG::Discretization>(parameters_with_boundary_minimum(), 2);

    REQUIRE(distance(EoM, expected) < 1e-12);
  }
}

TEST_CASE("CG2 EoM direct minimum follows a moving off-grid minimum", "[discretization][EoM][direct][cg2][minimum]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const std::array<double, 4> positions{{0.413, 0.427, 0.441, 0.468}};
  std::optional<Point<dim>> previous_minimum;
  for (const double position : positions) {
    const Point<dim> expected(position);
    using Model = Testing::ModelConstant<dim, dim>;
    const Model model(parameters_with_minimum(expected));
    const auto EoM = locate_minimum_with_model<dim, CG::Discretization>(model, 2, 0, previous_minimum);
    REQUIRE(distance(EoM, expected) < 1e-10);
    REQUIRE(std::isfinite(EoM[0]));
    previous_minimum = EoM;
  }
}

TEST_CASE("CG EoM direct search handles a 1D stitched polynomial minimum", "[discretization][EoM][cg][stitched][1d]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const int fe_order = GENERATE(1, 2, 3, 4);
  const Point<dim> expected(0.4);
  const ModelStitchedPolynomial<dim> model(expected);
  const auto EoM = locate_minimum_with_model<dim, CG::Discretization>(model, fe_order);

  REQUIRE(distance(EoM, expected) < 6e-2);
}

TEST_CASE("CG EoM direct search handles a 2D stitched polynomial minimum", "[discretization][EoM][cg][stitched][2d]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const int fe_order = GENERATE(1, 2, 3);
  const Point<dim> expected(0.4, 0.6);
  const ModelStitchedPolynomial<dim> model(expected);
  const auto EoM = locate_minimum_with_model<dim, CG::Discretization>(model, fe_order);

  REQUIRE(distance(EoM, expected) < 8e-2);
}

TEST_CASE("CG2 EoM direct search supports DG0 sources", "[discretization][EoM][dg][p0]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const Point<dim> expected(0.4);
  const auto EoM = locate_minimum<dim, DG::Discretization>(parameters_with_minimum(expected), 0);

  REQUIRE(distance(EoM, expected) < 1.6e-1);
}

TEST_CASE("CG2 EoM direct search supports higher-order DG sources", "[discretization][EoM][dg]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const int fe_order = GENERATE(1, 2);
  const Point<dim> expected(0.4, 0.6);
  const auto EoM = locate_minimum<dim, DG::Discretization>(parameters_with_minimum(expected), fe_order);

  REQUIRE(distance(EoM, expected) < 1.1e-1);
}

TEST_CASE("EoM direct search can return a boundary minimum", "[discretization][EoM][boundary]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const Point<dim> expected(0., 0.);
  const auto EoM = locate_minimum<dim, CG::Discretization>(parameters_with_boundary_minimum(), 2);

  REQUIRE(distance(EoM, expected) < 1e-12);
}

TEST_CASE("EoM max-iteration zero keeps the origin bypass", "[discretization][EoM][origin]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  using Model = Testing::ModelConstant<dim, dim>;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;

  setup_logger();

  auto json = make_json(1);
  Model model(parameters_with_boundary_minimum());
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  auto EoM_cell = discretization.get_dof_handler().begin_active();
  const auto result = get_EoM_result(
      EoM_cell, initial_condition.spatial_data(), discretization.get_dof_handler(), discretization.get_mapping(),
      [&](const auto &p, const auto &values) { return model.EoM(p, values); },
      [&](const auto &p, const auto &) { return p; }, 1e-12, 0);

  REQUIRE(distance(result.point, Point<dim>(0., 0.)) < 1e-12);
}

TEST_CASE("A source-dominated 1D minimum below the first cell centre is the zero of the EoM",
          "[discretization][EoM][source][fv][cg]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  // Minima at rho ~ 1e-4 .. 1e-2, below the first cell centre (cells of width 0.05): the piecewise-quadratic
  // reconstruction cannot follow c / sqrt(2 rho) there.
  const double c = GENERATE(0.015, 0.05, 0.14);
  const double lambda = GENERATE(0., 2.);
  const ModelSourceEoM model(1., lambda, c);
  const double expected = model.exact_minimum();
  CAPTURE(c, lambda, expected);

  const auto fv = locate_minimum_with_model<dim, FV::Discretization>(model, 0);
  const auto cg = locate_minimum_with_model<dim, CG::Discretization>(model, 1);
  CAPTURE(fv[0], cg[0]);
  CHECK(std::abs(fv[0] - expected) < 1e-12 * std::max(1., expected) + 1e-14);
  CHECK(std::abs(cg[0] - expected) < 1e-12 * std::max(1., expected) + 1e-14);
}

TEST_CASE("The 1D EoM search does not move to a higher, metastable minimum", "[discretization][EoM][fv][cg]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const ModelTwoMinima model;
  const auto fv = locate_minimum_with_model<dim, FV::Discretization>(model, 0);
  const auto cg = locate_minimum_with_model<dim, CG::Discretization>(model, 1);
  CAPTURE(fv[0], cg[0]);
  CHECK(fv[0] < 0.03);
  CHECK(cg[0] < 0.03);
}

TEST_CASE("Direct EoM search solves the vector field with curl without a potential solve",
          "[discretization][EoM][direct]")
{
  DiFfRG::Init();
  const Point<2> expected(0.373, 0.417);
  // The symmetric part is positive definite, but this field is not a gradient.
  const ModelAffineEoM<2> model(expected, {{{4., 3.}, {-2., 3.}}});
  const auto cg = locate_with_model<2, CG::Discretization>(model, 2, 0, std::nullopt, true);
  const auto fv = locate_with_model<2, FV::Discretization>(model, 0, 0, std::nullopt, true);
  CHECK(cg.point.distance(expected) < 1e-9);
  CHECK(fv.point.distance(expected) < 1e-9);
}

TEST_CASE("Direct EoM search checks each physical edge independently", "[discretization][EoM][direct][boundary]")
{
  DiFfRG::Init();
  const bool upper = GENERATE(false, true);
  const Point<2> unconstrained(0.373, upper ? 1.08 : -0.08);
  const Point<2> expected(upper ? 0.389 : 0.357, upper ? 1. : 0.);
  const ModelAffineEoM<2> model(unconstrained, {{{4., .8}, {-.3, 3.}}});
  const auto cg = locate_minimum_with_model<2, CG::Discretization>(model, 2);
  const auto fv = locate_minimum_with_model<2, FV::Discretization>(model, 0);
  CHECK(cg.distance(expected) < 1e-9);
  CHECK(fv.distance(expected) < 1e-9);
}
