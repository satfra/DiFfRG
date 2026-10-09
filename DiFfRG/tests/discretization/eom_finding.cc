#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include <deal.II/grid/grid_generator.h>

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
           {"EoM_smoothing_length", -1.},
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

  using DefaultRawPotentialComponents = typename Testing::compFactory<2>::value;

  class DefaultRawPotentialModel : public def::AbstractModel<DefaultRawPotentialModel, DefaultRawPotentialComponents>
  {
  public:
    template <typename Vector> std::array<double, 2> EoM(const Point<2> &, const Vector &values) const
    {
      return {{values[0] - 1., values[1] - 2.}};
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
  void check_raw_potential_is_scalar_cg2(const Model &model, const int fe_order)
  {
    using Discretization = DiscretizationTemplate<Model, RectangularMesh<dim>>;

    auto json = make_json(fe_order);
    RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
    Discretization discretization(mesh, json);
    FE::FlowingVariables state(discretization);
    state.interpolate(model);

    const auto raw = reconstruct_raw_potential(
        state.spatial_data(), discretization.get_dof_handler(), discretization.get_mapping(),
        [&](const auto &point, const auto &values) { return model.raw_potential_gradient(point, values); },
        Config::EoMConfig(json));

    REQUIRE(raw.finite_element != nullptr);
    CHECK(dynamic_cast<const FE_Q<dim> *>(raw.finite_element.get()) != nullptr);
    CHECK(raw.finite_element->degree == 2);
    CHECK(raw.finite_element->n_components() == 1);
    CHECK(raw.finite_element->conforms(FiniteElementData<dim>::H1));
  }

  template <int dim, template <typename...> typename DiscretizationTemplate, typename Model>
  auto reconstruct_with_potential_with_model(const Model &model, const int fe_order,
                                             const double smoothing_length = -1., const uint refinement = 0,
                                             const std::optional<Point<dim>> &initial_guess = std::nullopt)
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
    EoM_config.smoothing_length = smoothing_length;
    EoM_config.validate();
    return get_EoM_point_with_potential(
        EoM_cell, src, dof_handler, mapping, [&](const auto &p, const auto &values) { return model.EoM(p, values); },
        [&](const auto &p, const auto &) { return p; }, EoM_config, initial_guess);
  }

  template <int dim, template <typename...> typename DiscretizationTemplate, typename Model>
  Point<dim> reconstruct_minimum_with_model(const Model &model, const int fe_order, const double smoothing_length = -1.,
                                            const uint refinement = 0,
                                            const std::optional<Point<dim>> &initial_guess = std::nullopt)
  {
    return reconstruct_with_potential_with_model<dim, DiscretizationTemplate>(model, fe_order, smoothing_length,
                                                                              refinement, initial_guess)
        .point;
  }

  template <int dim, template <typename...> typename DiscretizationTemplate>
  Point<dim> reconstruct_minimum(const Testing::PhysicalParameters &prm, const int fe_order,
                                 const double smoothing_length = -1., const uint refinement = 0)
  {
    using Model = Testing::ModelConstant<dim, dim>;
    Model model(prm);
    return reconstruct_minimum_with_model<dim, DiscretizationTemplate>(model, fe_order, smoothing_length, refinement);
  }
} // namespace

TEST_CASE("CG EoM potential reconstruction finds 1D affine minima", "[discretization][EoM][cg][1d]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const int fe_order = GENERATE(1, 2, 3, 4, 5);
  const double slope = GENERATE(1.25, 2., 2.5, 5., 10.);
  const auto prm = parameters_from_affine<dim>(std::array<double, dim>{{-1.}}, std::array<double, dim>{{slope}});
  const auto expected = affine_minimum<dim>(prm);
  const auto EoM = reconstruct_minimum<dim, CG::Discretization>(prm, fe_order);

  REQUIRE(distance(EoM, expected) < 6e-2);
}

TEST_CASE("CG EoM potential reconstruction finds 2D affine and boundary minima", "[discretization][EoM][cg][2d]")
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
  const auto EoM = reconstruct_minimum<dim, CG::Discretization>(prm, fe_order);

  REQUIRE(distance(EoM, expected) < 8e-2);
}

TEST_CASE("CG EoM potential reconstruction finds 3D affine minima", "[discretization][EoM][cg][3d]")
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
  const auto EoM = reconstruct_minimum<dim, CG::Discretization>(prm, fe_order);

  REQUIRE(distance(EoM, expected) < 1.1e-1);
}

TEST_CASE("Detailed EoM reconstruction owns the scalar potential and its gauge", "[discretization][EoM][potential]")
{
  DiFfRG::Init();
  setup_logger();

  const auto check_result = []<int dim>() {
    const Point<dim> expected = []() {
      if constexpr (dim == 1)
        return Point<dim>(0.4);
      else if constexpr (dim == 2)
        return Point<dim>(0.4, 0.6);
      else
        return Point<dim>(0.4, 0.6, 0.8);
    }();

    using Model = Testing::ModelConstant<dim, dim>;
    using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;

    auto json = make_json(2);
    Model model(parameters_with_minimum(expected));
    RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
    Discretization discretization(mesh, json);
    FE::FlowingVariables initial_condition(discretization);
    initial_condition.interpolate(model);

    const auto &dof_handler = discretization.get_dof_handler();
    auto EoM_cell = dof_handler.begin_active();
    const auto result = get_EoM_point_with_potential(
        EoM_cell, initial_condition.spatial_data(), dof_handler, discretization.get_mapping(),
        [&](const auto &p, const auto &values) { return model.EoM(p, values); },
        [&](const auto &p, const auto &) { return p; }, 1e-12, 100);

    REQUIRE(distance(result.point, expected) < 1.1e-1);
    REQUIRE(result.potential.has_value());
    const auto &potential = result.potential.value();
    REQUIRE(potential.finite_element != nullptr);
    REQUIRE(potential.dof_handler != nullptr);
    REQUIRE(potential.values.size() == potential.dof_handler->n_dofs());
    REQUIRE(potential.finite_element->n_components() == 1);
    REQUIRE(distance(potential.minimum, result.point) < 1e-12);

    const auto support_points = DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), *potential.dof_handler);
    auto gauge_dof = support_points.begin()->first;
    double gauge_distance = support_points.begin()->second.distance(Point<dim>());
    for (const auto &[dof, point] : support_points) {
      const double current_distance = point.distance(Point<dim>());
      if (current_distance < gauge_distance) {
        gauge_distance = current_distance;
        gauge_dof = dof;
      }
    }
    CHECK(potential.values[gauge_dof] == Catch::Approx(0.).margin(1e-12));
  };

  check_result.template operator()<1>();
  check_result.template operator()<2>();
  check_result.template operator()<3>();
}

TEST_CASE("Raw potential evaluation stays independent of the EoM used to select the point",
          "[discretization][EoM][raw-potential]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = ModelAffineEoM<dim>;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;

  setup_logger();
  auto json = make_json(2);
  Model model(Point<dim>(0.2), {{{2.}}});
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);
  FE::FlowingVariables state(discretization);
  state.interpolate(model);

  const auto &dof_handler = discretization.get_dof_handler();
  const auto &mapping = discretization.get_mapping();
  const auto config = Config::EoMConfig(json);
  auto EoM_cell = dof_handler.begin_active();
  const auto EoM_result = get_EoM_point_with_potential(
      EoM_cell, state.spatial_data(), dof_handler, mapping,
      [&](const auto &, const auto &values) { return std::array<double, 1>{{values[0] - 0.6}}; },
      [&](const auto &p, const auto &) { return p; }, config);

  const auto raw_potential = reconstruct_raw_potential(
      state.spatial_data(), dof_handler, mapping,
      [&](const auto &p, const auto &values) { return model.raw_potential_gradient(p, values); }, config);
  const auto raw = evaluate_raw_potential(raw_potential, mapping, EoM_result.point);

  CHECK(EoM_result.point[0] == Catch::Approx(0.5).margin(1e-8));
  CHECK(raw.value == Catch::Approx(0.05).margin(1e-8));
  CHECK(raw.gradient[0] == Catch::Approx(0.6).margin(1e-8));
  CHECK(raw.potential_hessian[0][0] == Catch::Approx(2.).margin(1e-8));
  CHECK(raw.mass_hessian[0][0] == raw.potential_hessian[0][0]);

  auto origin_config = config;
  origin_config.max_iter = 0;
  const auto raw_without_eom_search = reconstruct_raw_potential(
      state.spatial_data(), dof_handler, mapping,
      [&](const auto &p, const auto &values) { return model.raw_potential_gradient(p, values); }, origin_config);
  const auto at_origin = evaluate_raw_potential(raw_without_eom_search, mapping, Point<dim>());
  CHECK(at_origin.value == Catch::Approx(0.).margin(1e-12));
  CHECK(at_origin.gradient[0] == Catch::Approx(-0.4).margin(1e-8));
}

TEST_CASE("Origin-centred FV vacuum keeps the diquark EoM near zero and its raw curvature positive",
          "[discretization][EoM][raw-potential][fv][origin]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  using Model = QMDVacuumModel;
  using Discretization = FV::Discretization<Model, RectangularMesh<dim>>;

  setup_logger();
  auto json = make_json(0);
  json.set_string("/discretization/grid/x_grid", "0:0.005:0.2");
  json.set_string("/discretization/grid/y_grid", "0:0.005:0.2");
  Model model;
  RectangularMesh<dim> mesh(Config::ConfigurationMesh<dim>(json), RectangularMeshOptions{.origin_cell_centered = true});
  Discretization discretization(mesh, json);
  FE::FlowingVariables state(discretization);
  state.interpolate(model);

  const auto &dof_handler = discretization.get_dof_handler();
  const auto &mapping = discretization.get_mapping();
  auto EoM_cell = dof_handler.begin_active();
  const auto config = Config::EoMConfig(json);
  const auto EoM_result = get_EoM_point_with_potential(
      EoM_cell, state.spatial_data(), dof_handler, mapping,
      [&](const auto &point, const auto &values) { return model.EoM(point, values); },
      [](const auto &point, const auto &) { return point; }, config);

  const auto raw_potential = reconstruct_raw_potential(
      state.spatial_data(), dof_handler, mapping,
      [&](const auto &point, const auto &values) { return model.raw_potential_gradient(point, values); }, config);
  const auto raw = evaluate_raw_potential(raw_potential, mapping, EoM_result.point);
  const double h = dof_handler.begin_active()->extent_in_direction(1);
  const double origin_tolerance = 1e-5 * h;
  const double current_mass_square = raw.mass_hessian[1][1];
  const double goldstone_mass_square =
      std::abs(EoM_result.point[1]) <= origin_tolerance ? current_mass_square : raw.gradient[1] / EoM_result.point[1];

  CHECK(std::abs(EoM_result.point[1]) <= origin_tolerance);
  CHECK(EoM_result.point[0] == Catch::Approx(Model::explicit_breaking / Model::sigma_mass_square).margin(h));
  CHECK(std::isfinite(raw.value));
  CHECK(std::isfinite(raw.gradient[1]));
  CHECK(std::isfinite(raw.mass_hessian[1][1]));
  CHECK(current_mass_square == Catch::Approx(Model::diquark_mass_square).epsilon(0.05));
  CHECK(goldstone_mass_square == Catch::Approx(Model::diquark_mass_square).epsilon(0.05));
}

TEST_CASE("Raw-potential reconstruction preserves signed negative curvature",
          "[discretization][EoM][raw-potential][hessian][signed]")
{
  constexpr uint dim = 1;
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 8, 0., 1.);
  FE_Q<dim> source_fe(1);
  DoFHandler<dim> source_dof_handler(triangulation);
  source_dof_handler.distribute_dofs(source_fe);
  Vector<double> source(source_dof_handler.n_dofs());
  MappingQ1<dim> mapping;
  VectorTools::interpolate(source_dof_handler, Functions::ZeroFunction<dim>(), source);

  Config::EoMConfig config;
  config.smoothing_length = 0.;
  const auto raw_potential = reconstruct_raw_potential(
      source, source_dof_handler, mapping,
      [](const Point<dim> &point, const auto &) { return std::array<double, dim>{{-2. * point[0]}}; }, config);
  const auto value = evaluate_raw_potential(raw_potential, mapping, Point<dim>(0.5));

  CHECK(value.potential_hessian[0][0] == Catch::Approx(-2.).margin(1e-10));
  CHECK(value.mass_hessian[0][0] == value.potential_hessian[0][0]);
  CHECK(value.potential_hessian[0][0] < 0.);
}

TEST_CASE("Default raw potential gradient copies solution components instead of the physical EoM",
          "[discretization][EoM][raw-potential][model]")
{
  DefaultRawPotentialModel model;
  Vector<double> values(2);
  values[0] = 3.;
  values[1] = 5.;

  const auto raw_gradient = model.raw_potential_gradient(Point<2>(), values);
  const auto physical_eom = model.EoM(Point<2>(), values);

  CHECK(raw_gradient[0] == 3.);
  CHECK(raw_gradient[1] == 5.);
  CHECK(physical_eom[0] == 2.);
  CHECK(physical_eom[1] == 3.);

  Vector<double> scalar_values(1);
  scalar_values[0] = 7.;
  const auto zero_filled_gradient = model.raw_potential_gradient(Point<2>(), scalar_values);
  CHECK(zero_filled_gradient[0] == 7.);
  CHECK(zero_filled_gradient[1] == 0.);
}

TEST_CASE("EoM and raw potential reconstruction always target scalar CG2",
          "[discretization][EoM][raw-potential][potential][cg2]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim, dim>;
  const Model model(parameters_with_minimum(Point<dim>(0.4)));

  const auto check_cg2_potential = [](const auto &result) {
    REQUIRE(result.potential.has_value());
    const auto &potential = result.potential.value();
    REQUIRE(potential.finite_element != nullptr);
    CHECK(dynamic_cast<const FE_Q<dim> *>(potential.finite_element.get()) != nullptr);
    CHECK(potential.finite_element->degree == 2);
    CHECK(potential.finite_element->n_components() == 1);
    CHECK(potential.finite_element->conforms(FiniteElementData<dim>::H1));
  };

  SECTION("CG source")
  {
    const auto result = reconstruct_with_potential_with_model<dim, CG::Discretization>(model, 5);
    check_cg2_potential(result);
    check_raw_potential_is_scalar_cg2<dim, CG::Discretization>(model, 5);
  }

  SECTION("higher-order DG source")
  {
    const auto result = reconstruct_with_potential_with_model<dim, DG::Discretization>(model, 3);
    check_cg2_potential(result);
    check_raw_potential_is_scalar_cg2<dim, DG::Discretization>(model, 3);
  }

  SECTION("DG0 source")
  {
    const auto result = reconstruct_with_potential_with_model<dim, DG::Discretization>(model, 0);
    check_cg2_potential(result);
    check_raw_potential_is_scalar_cg2<dim, DG::Discretization>(model, 0);
  }

  SECTION("FV source")
  {
    const auto result = reconstruct_with_potential_with_model<dim, FV::Discretization>(model, 0);
    check_cg2_potential(result);
    check_raw_potential_is_scalar_cg2<dim, FV::Discretization>(model, 0);
  }
}

TEST_CASE("CG2 EoM potential reconstruction refines an off-grid 1D minimum",
          "[discretization][EoM][potential][cg2][minimum]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const Point<dim> expected(0.437);
  const auto EoM = reconstruct_minimum<dim, CG::Discretization>(parameters_with_minimum(expected), 2);

  REQUIRE(distance(EoM, expected) < 1e-10);
}

TEST_CASE("CG2 EoM potential reconstruction refines multidimensional and constrained minima",
          "[discretization][EoM][potential][cg2][minimum]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;

  SECTION("rotated interior minimum")
  {
    const Point<dim> expected(0.437, 0.583);
    const ModelAffineEoM<dim> model(expected, {{{4., 1.25}, {1.25, 3.}}});
    const auto EoM = reconstruct_minimum_with_model<dim, CG::Discretization>(model, 2);

    REQUIRE(distance(EoM, expected) < 1e-9);
  }

  SECTION("edge minimum away from support points")
  {
    const Point<dim> unconstrained_minimum(0.437, -0.2);
    const Point<dim> expected(0.437, 0.);
    const ModelAffineEoM<dim> model(unconstrained_minimum, {{{4., 0.}, {0., 3.}}});
    const auto EoM = reconstruct_minimum_with_model<dim, CG::Discretization>(model, 2);

    REQUIRE(distance(EoM, expected) < 1e-9);
  }

  SECTION("monotonic corner minimum")
  {
    const Point<dim> expected(0., 0.);
    const auto EoM = reconstruct_minimum<dim, CG::Discretization>(parameters_with_boundary_minimum(), 2);

    REQUIRE(distance(EoM, expected) < 1e-12);
  }
}

TEST_CASE("CG2 EoM potential minimum follows a moving off-grid minimum",
          "[discretization][EoM][potential][cg2][minimum]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const std::array<double, 4> positions{{0.413, 0.427, 0.441, 0.468}};
  std::optional<Point<dim>> previous_minimum;
  for (const double position : positions) {
    const Point<dim> expected(position);
    using Model = Testing::ModelConstant<dim, dim>;
    const Model model(parameters_with_minimum(expected));
    const auto EoM = reconstruct_minimum_with_model<dim, CG::Discretization>(model, 2, -1., 0, previous_minimum);
    REQUIRE(distance(EoM, expected) < 1e-10);
    REQUIRE(std::isfinite(EoM[0]));
    previous_minimum = EoM;
  }
}

TEST_CASE("CG2 EoM potential refinement handles indefinite and singular Hessians",
          "[discretization][EoM][potential][cg2][minimum]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const std::array<bool, dim> free{{false, false}};
  Tensor<1, dim> gradient;
  gradient[0] = 1.;
  gradient[1] = -1.;
  Tensor<1, dim> direction;

  Tensor<2, dim> indefinite_hessian;
  indefinite_hessian[0][0] = 1.;
  indefinite_hessian[1][1] = -1.;
  const bool indefinite_direction =
      DiFfRG::internal::projected_newton_direction<dim>(indefinite_hessian, gradient, free, direction);
  CHECK_FALSE(indefinite_direction);

  Tensor<2, dim> singular_hessian;
  singular_hessian[0][0] = 1.;
  const bool singular_direction =
      DiFfRG::internal::projected_newton_direction<dim>(singular_hessian, gradient, free, direction);
  CHECK_FALSE(singular_direction);

  Tensor<2, dim> positive_hessian;
  positive_hessian[0][0] = 4.;
  positive_hessian[0][1] = 1.;
  positive_hessian[1][0] = 1.;
  positive_hessian[1][1] = 3.;
  REQUIRE(DiFfRG::internal::projected_newton_direction<dim>(positive_hessian, gradient, free, direction));
  CHECK(direction[0] == Catch::Approx(-4. / 11.).margin(1e-14));
  CHECK(direction[1] == Catch::Approx(5. / 11.).margin(1e-14));

  const std::array<bool, dim> second_coordinate_fixed{{false, true}};
  REQUIRE(DiFfRG::internal::projected_newton_direction<dim>(positive_hessian, gradient, second_coordinate_fixed,
                                                            direction));
  CHECK(direction[0] == Catch::Approx(-0.25).margin(1e-14));
  CHECK(direction[1] == 0.);

  Tensor<2, dim> non_finite_hessian = positive_hessian;
  non_finite_hessian[0][0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_FALSE(DiFfRG::internal::projected_newton_direction<dim>(non_finite_hessian, gradient, free, direction));

  using Model = ModelAffineEoM<dim>;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;
  const Point<dim> saddle(0.437, 0.583);
  const Model model(saddle, {{{1., 0.}, {0., -1.}}});
  auto json = make_json(2);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  const auto &mapping = discretization.get_mapping();
  auto EoM_cell = discretization.get_dof_handler().begin_active();
  const auto result = get_EoM_point_with_potential(
      EoM_cell, initial_condition.spatial_data(), discretization.get_dof_handler(), mapping,
      [&](const auto &p, const auto &values) { return model.EoM(p, values); },
      [](const auto &p, const auto &) { return p; }, 1e-12, 200);
  const auto EoM = result.point;
  for (uint d = 0; d < dim; ++d) {
    CHECK(std::isfinite(EoM[d]));
    CHECK(EoM[d] >= 0.);
    CHECK(EoM[d] <= 1.);
  }

  const auto &potential = result.potential.value();
  const QGauss<dim> quadrature(4);
  FEValues<dim> potential_fe_values(mapping, *potential.finite_element, quadrature, update_values);
  std::vector<double> quadrature_values(quadrature.size());
  double sampled_minimum = std::numeric_limits<double>::max();
  for (const auto &value : potential.values)
    sampled_minimum = std::min(sampled_minimum, (double)value);
  for (const auto &cell : potential.dof_handler->active_cell_iterators()) {
    potential_fe_values.reinit(cell);
    potential_fe_values.get_function_values(potential.values, quadrature_values);
    for (const double value : quadrature_values)
      sampled_minimum = std::min(sampled_minimum, value);
  }

  auto interior_probe = EoM;
  for (uint d = 0; d < dim; ++d)
    interior_probe[d] = std::clamp(interior_probe[d], 1e-10, 1. - 1e-10);
  const auto minimum_cell =
      GridTools::find_active_cell_around_point(mapping, *potential.dof_handler, interior_probe).first;
  const auto minimum_unit_point = mapping.transform_real_to_unit_cell(minimum_cell, EoM);
  std::vector<types::global_dof_index> local_dof_indices(potential.finite_element->n_dofs_per_cell());
  minimum_cell->get_dof_indices(local_dof_indices);
  std::vector<double> local_values(local_dof_indices.size());
  for (uint i = 0; i < local_dof_indices.size(); ++i)
    local_values[i] = potential.values[local_dof_indices[i]];
  const auto refined =
      DiFfRG::internal::evaluate_local_potential(*potential.finite_element, local_values, minimum_unit_point);
  CHECK(std::isfinite(refined.value));
  CHECK(refined.value <= sampled_minimum + 1e-12);
}

TEST_CASE("A previous CG2 minimum warm-starts the constrained Newton refinement",
          "[discretization][EoM][potential][cg2][minimum][warm-start]")
{
  constexpr uint dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  MappingQ1<dim> mapping;
  FE_Q<dim> potential_fe(2);
  DoFHandler<dim> potential_dof_handler(triangulation);
  potential_dof_handler.distribute_dofs(potential_fe);

  constexpr double minimum_x = 0.41;
  constexpr double minimum_y = 0.59;
  Vector<double> potential(potential_dof_handler.n_dofs());
  const auto support_points = DoFTools::map_dofs_to_support_points(mapping, potential_dof_handler);
  for (const auto &[dof, point] : support_points) {
    const double dx = point[0] - minimum_x;
    const double dy = point[1] - minimum_y;
    potential[dof] = dx * dx + dy * dy + 5. * dx * dx * dy * dy;
  }

  const QGauss<dim> quadrature(4);
  const auto cell = potential_dof_handler.begin_active();
  const Config::EoMConfig optimizer_config(1e-14, 1, -1.);
  const auto cold = DiFfRG::internal::refine_cell_minimum<dim, double>(cell, potential_fe, mapping, potential,
                                                                       quadrature, optimizer_config);
  const auto warm = DiFfRG::internal::refine_cell_minimum<dim, double>(
      cell, potential_fe, mapping, potential, quadrature, optimizer_config, Point<dim>(0.412, 0.588));
  const Point<dim> expected(minimum_x, minimum_y);

  CHECK(warm.value < cold.value);
  CHECK(distance(warm.point, expected) < distance(cold.point, expected));
  CHECK(distance(warm.point, expected) < 1e-5);
}

TEST_CASE("CG2 warm starts preserve global minimum changes",
          "[discretization][EoM][potential][cg2][minimum][warm-start]")
{
  constexpr uint dim = 1;
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 4, 0., 1.);
  MappingQ1<dim> mapping;
  FE_Q<dim> potential_fe(2);
  DoFHandler<dim> potential_dof_handler(triangulation);
  potential_dof_handler.distribute_dofs(potential_fe);

  Vector<double> potential(potential_dof_handler.n_dofs());
  const auto support_points = DoFTools::map_dofs_to_support_points(mapping, potential_dof_handler);
  for (const auto &[dof, point] : support_points) {
    const double old_well = -1. + 40. * std::pow(point[0] - 0.125, 2);
    const double new_well = -2. + 40. * std::pow(point[0] - 0.875, 2);
    potential[dof] = std::min(old_well, new_well);
  }

  const QGauss<dim> quadrature(4);
  const Config::EoMConfig optimizer_config(1e-12, 100, -1.);
  const auto minimum =
      DiFfRG::internal::find_potential_minimum(potential_dof_handler, potential_fe, mapping, potential, quadrature,
                                               optimizer_config, std::optional<Point<dim>>(Point<dim>(0.125)));

  CHECK(minimum.point[0] == Catch::Approx(0.875).margin(1e-10));
  CHECK(minimum.value == Catch::Approx(-2.).margin(1e-10));
}

TEST_CASE("CG2 warm starts handle boundaries interfaces adaptation and invalid points",
          "[discretization][EoM][potential][cg2][minimum][warm-start]")
{
  constexpr uint dim = 1;
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 2, 0., 1.);
  triangulation.begin_active()->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();

  MappingQ1<dim> mapping;
  FE_Q<dim> potential_fe(2);
  DoFHandler<dim> potential_dof_handler(triangulation);
  potential_dof_handler.distribute_dofs(potential_fe);
  Vector<double> potential(potential_dof_handler.n_dofs());
  constexpr double expected = 0.73;
  const auto support_points = DoFTools::map_dofs_to_support_points(mapping, potential_dof_handler);
  for (const auto &[dof, point] : support_points)
    potential[dof] = std::pow(point[0] - expected, 2);

  const QGauss<dim> quadrature(4);
  const Config::EoMConfig optimizer_config(1e-12, 100, -1.);
  const auto find = [&](const std::optional<Point<dim>> &initial_guess) {
    return DiFfRG::internal::find_potential_minimum(potential_dof_handler, potential_fe, mapping, potential, quadrature,
                                                    optimizer_config, initial_guess);
  };

  const auto no_seed = find(std::nullopt);
  const auto boundary_seed = find(Point<dim>(0.));
  const auto interface_seed = find(Point<dim>(0.5));
  const auto outside_seed = find(Point<dim>(-1.));
  const auto non_finite_seed = find(Point<dim>(std::numeric_limits<double>::quiet_NaN()));
  for (const auto &minimum : {no_seed, boundary_seed, interface_seed, outside_seed, non_finite_seed}) {
    CHECK(std::isfinite(minimum.value));
    CHECK(minimum.point[0] == Catch::Approx(expected).margin(1e-10));
  }
}

TEST_CASE("CG2 EoM potential refinement checks the sampled winner's active neighbors",
          "[discretization][EoM][potential][cg2][minimum][neighbor]")
{
  constexpr uint dim = 1;
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 2, 0., 1.);
  MappingQ1<dim> mapping;
  FE_Q<dim> potential_fe(2);
  DoFHandler<dim> potential_dof_handler(triangulation);
  potential_dof_handler.distribute_dofs(potential_fe);

  Vector<double> potential(potential_dof_handler.n_dofs());
  const auto support_points = DoFTools::map_dofs_to_support_points(mapping, potential_dof_handler);
  constexpr double neighbor_minimum = 0.7925;
  constexpr double neighbor_curvature = 60.;
  const double shared_value = -1.1 + neighbor_curvature * std::pow(0.5 - neighbor_minimum, 2);
  const double winner_curvature = (shared_value + 1.) / std::pow(0.25, 2);
  for (const auto &[dof, point] : support_points) {
    if (point[0] <= 0.5)
      potential[dof] = -1. + winner_curvature * std::pow(point[0] - 0.25, 2);
    else
      potential[dof] = -1.1 + neighbor_curvature * std::pow(point[0] - neighbor_minimum, 2);
  }

  const QGauss<dim> quadrature(4);
  const Config::EoMConfig optimizer_config(1e-12, 100, -1.);
  const auto minimum = DiFfRG::internal::find_potential_minimum(potential_dof_handler, potential_fe, mapping, potential,
                                                                quadrature, optimizer_config);

  REQUIRE(minimum.point[0] == Catch::Approx(neighbor_minimum).margin(1e-10));
  REQUIRE(minimum.value == Catch::Approx(-1.1).margin(1e-10));
}

TEST_CASE("CG EoM potential reconstruction handles a 1D stitched polynomial minimum",
          "[discretization][EoM][cg][stitched][1d]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const int fe_order = GENERATE(1, 2, 3, 4);
  const Point<dim> expected(0.4);
  const ModelStitchedPolynomial<dim> model(expected);
  const auto EoM = reconstruct_minimum_with_model<dim, CG::Discretization>(model, fe_order);

  REQUIRE(distance(EoM, expected) < 6e-2);
}

TEST_CASE("CG EoM potential reconstruction handles a 2D stitched polynomial minimum",
          "[discretization][EoM][cg][stitched][2d]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const int fe_order = GENERATE(1, 2, 3);
  const Point<dim> expected(0.4, 0.6);
  const ModelStitchedPolynomial<dim> model(expected);
  const auto EoM = reconstruct_minimum_with_model<dim, CG::Discretization>(model, fe_order);

  REQUIRE(distance(EoM, expected) < 8e-2);
}

TEST_CASE("CG2 EoM potential reconstruction supports DG0 sources", "[discretization][EoM][dg][p0]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const Point<dim> expected(0.4);
  const auto EoM = reconstruct_minimum<dim, DG::Discretization>(parameters_with_minimum(expected), 0);

  REQUIRE(distance(EoM, expected) < 1.6e-1);
}

TEST_CASE("EoM smoothing length resolves automatic disabled explicit and invalid policies",
          "[discretization][EoM][potential][smoothing][configuration]")
{
  constexpr uint dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 4, 0., 1.);
  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  CHECK(DiFfRG::internal::minimum_face_normal_cell_width(dof_handler) == Catch::Approx(0.25));
  CHECK(DiFfRG::internal::resolve_potential_smoothing_length(dof_handler, -1.) == Catch::Approx(0.5));
  CHECK(DiFfRG::internal::resolve_potential_smoothing_length(dof_handler, 0.) == 0.);
  CHECK(DiFfRG::internal::resolve_potential_smoothing_length(dof_handler, 0.125) == Catch::Approx(0.125));
  CHECK_THROWS_AS(DiFfRG::internal::resolve_potential_smoothing_length(dof_handler, -0.5), std::invalid_argument);
  CHECK_THROWS_AS(DiFfRG::internal::resolve_potential_smoothing_length(dof_handler, -1.01), std::invalid_argument);
  CHECK_THROWS_AS(
      DiFfRG::internal::resolve_potential_smoothing_length(dof_handler, std::numeric_limits<double>::quiet_NaN()),
      std::invalid_argument);
}

TEST_CASE("EoM configuration provides validated typed defaults", "[discretization][EoM][configuration]")
{
  const Config::EoMConfig defaults;
  CHECK(defaults.abs_tol == Config::EoMConfig::default_abs_tol);
  CHECK(defaults.max_iter == Config::EoMConfig::default_max_iter);
  CHECK(defaults.smoothing_length == Config::EoMConfig::default_smoothing_length);
  CHECK(defaults.bound_tolerance == Config::EoMConfig::default_bound_tolerance);
  CHECK(defaults.armijo_coefficient == Config::EoMConfig::default_armijo_coefficient);
  CHECK(defaults.max_backtracks == Config::EoMConfig::default_max_backtracks);
  CHECK(defaults.raw_potential_order == Config::EoMConfig::default_raw_potential_order);
  CHECK(defaults.raw_potential_recover_mass_hessian == Config::EoMConfig::default_raw_potential_recover_mass_hessian);
  CHECK(defaults.raw_potential_mass_hessian_jump_threshold ==
        Config::EoMConfig::default_raw_potential_mass_hessian_jump_threshold);

  const ConfigTree empty_json = json::value({});
  const Config::EoMConfig from_empty_json(empty_json);
  CHECK(from_empty_json.abs_tol == defaults.abs_tol);
  CHECK(from_empty_json.max_iter == defaults.max_iter);
  CHECK(from_empty_json.smoothing_length == defaults.smoothing_length);
  CHECK(from_empty_json.bound_tolerance == defaults.bound_tolerance);
  CHECK(from_empty_json.armijo_coefficient == defaults.armijo_coefficient);
  CHECK(from_empty_json.max_backtracks == defaults.max_backtracks);
  CHECK(from_empty_json.raw_potential_order == defaults.raw_potential_order);
  CHECK(from_empty_json.raw_potential_recover_mass_hessian == defaults.raw_potential_recover_mass_hessian);
  CHECK(from_empty_json.raw_potential_mass_hessian_jump_threshold ==
        defaults.raw_potential_mass_hessian_jump_threshold);

  const auto parsed_defaults = json::parse(Config::EoMConfig::get_defaults());
  const Config::EoMConfig from_default_json(parsed_defaults);
  CHECK(from_default_json.abs_tol == defaults.abs_tol);
  CHECK(from_default_json.max_iter == defaults.max_iter);
  CHECK(from_default_json.smoothing_length == defaults.smoothing_length);
  CHECK(from_default_json.bound_tolerance == defaults.bound_tolerance);
  CHECK(from_default_json.armijo_coefficient == defaults.armijo_coefficient);
  CHECK(from_default_json.max_backtracks == defaults.max_backtracks);
  CHECK(from_default_json.raw_potential_order == defaults.raw_potential_order);
  CHECK(from_default_json.raw_potential_recover_mass_hessian == defaults.raw_potential_recover_mass_hessian);
  CHECK(from_default_json.raw_potential_mass_hessian_jump_threshold ==
        defaults.raw_potential_mass_hessian_jump_threshold);

  const ConfigTree custom_json = json::value({{"discretization",
                                               {{"EoM_abs_tol", 1e-9},
                                                {"EoM_max_iter", 17},
                                                {"EoM_smoothing_length", 0.25},
                                                {"EoM_bound_tolerance", 1e-10},
                                                {"EoM_armijo_coefficient", 1e-3},
                                                {"EoM_max_backtracks", 12},
                                                {"raw_potential_order", 3},
                                                {"raw_potential_recover_mass_hessian", true},
                                                {"raw_potential_mass_hessian_jump_threshold", 0.6}}}});
  const Config::EoMConfig from_custom_json(custom_json);
  CHECK(from_custom_json.abs_tol == 1e-9);
  CHECK(from_custom_json.max_iter == 17);
  CHECK(from_custom_json.smoothing_length == 0.25);
  CHECK(from_custom_json.bound_tolerance == 1e-10);
  CHECK(from_custom_json.armijo_coefficient == 1e-3);
  CHECK(from_custom_json.max_backtracks == 12);
  CHECK(from_custom_json.raw_potential_order == 3);
  CHECK(from_custom_json.raw_potential_recover_mass_hessian);
  CHECK(from_custom_json.raw_potential_mass_hessian_jump_threshold == 0.6);

  const Config::EoMConfig explicit_config(1e-9, 17, 0.25, 1e-10, 1e-3, 12);
  CHECK(explicit_config.abs_tol == 1e-9);
  CHECK(explicit_config.max_iter == 17);
  CHECK(explicit_config.smoothing_length == 0.25);
  CHECK(explicit_config.bound_tolerance == 1e-10);
  CHECK(explicit_config.armijo_coefficient == 1e-3);
  CHECK(explicit_config.max_backtracks == 12);

  CHECK_THROWS_AS(Config::EoMConfig(-1e-9, 100, -1.), std::invalid_argument);
  CHECK_THROWS_AS(Config::EoMConfig(std::numeric_limits<double>::quiet_NaN(), 100, -1.), std::invalid_argument);
  CHECK_THROWS_AS(Config::EoMConfig(1e-12, 100, -0.5), std::invalid_argument);
  CHECK_THROWS_AS(Config::EoMConfig(1e-12, 100, std::numeric_limits<double>::infinity()), std::invalid_argument);
  CHECK_THROWS_AS(Config::EoMConfig(1e-12, 100, -1., -1e-12), std::invalid_argument);
  CHECK_THROWS_AS(Config::EoMConfig(1e-12, 100, -1., 1e-12, 0.), std::invalid_argument);
  CHECK_THROWS_AS(Config::EoMConfig(1e-12, 100, -1., 1e-12, 1.), std::invalid_argument);
  CHECK_THROWS_AS(Config::EoMConfig(1e-12, 100, -1., 1e-12, 1e-4, 0), std::invalid_argument);

  auto invalid_raw_config = defaults;
  invalid_raw_config.raw_potential_order = 1;
  CHECK_THROWS_AS(invalid_raw_config.validate(), std::invalid_argument);
  invalid_raw_config = defaults;
  invalid_raw_config.raw_potential_mass_hessian_jump_threshold = -0.5;
  CHECK_THROWS_AS(invalid_raw_config.validate(), std::invalid_argument);
}

TEST_CASE("Raw-potential mass-Hessian recovery damps FV face resets without changing the potential Hessian",
          "[discretization][EoM][raw-potential][hessian][fv]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = CubicGradientModel;
  using Discretization = FV::Discretization<Model, RectangularMesh<dim>>;

  setup_logger();
  auto json = make_json(0);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);
  FE::FlowingVariables state(discretization);
  state.interpolate(Model{});

  const auto gradient = [](const auto &, const auto &values) { return std::array<double, 1>{{values[0]}}; };
  Config::EoMConfig default_config;
  default_config.smoothing_length = 0.2;
  const auto default_potential = reconstruct_raw_potential(state.spatial_data(), discretization.get_dof_handler(),
                                                           discretization.get_mapping(), gradient, default_config);

  auto regularized_config = default_config;
  regularized_config.raw_potential_recover_mass_hessian = true;
  regularized_config.raw_potential_mass_hessian_jump_threshold = 0.35;
  regularized_config.validate();
  const auto regularized_potential =
      reconstruct_raw_potential(state.spatial_data(), discretization.get_dof_handler(), discretization.get_mapping(),
                                gradient, regularized_config);

  auto continuous_config = regularized_config;
  continuous_config.raw_potential_mass_hessian_jump_threshold = -1.;
  const auto continuous_potential =
      reconstruct_raw_potential(state.spatial_data(), discretization.get_dof_handler(), discretization.get_mapping(),
                                gradient, continuous_config);

  REQUIRE(default_potential.finite_element->degree == 2);
  REQUIRE(regularized_potential.finite_element->degree == 2);

  constexpr double epsilon = 1e-7;
  double max_default_jump = 0.;
  double max_regularized_jump = 0.;
  double max_continuous_jump = 0.;
  double max_potential_hessian_change = 0.;
  for (uint face = 1; face < 10; ++face) {
    const double x = 0.1 * face;
    const auto default_left =
        evaluate_raw_potential(default_potential, discretization.get_mapping(), Point<dim>(x - epsilon));
    const auto default_right =
        evaluate_raw_potential(default_potential, discretization.get_mapping(), Point<dim>(x + epsilon));
    const auto regularized_left =
        evaluate_raw_potential(regularized_potential, discretization.get_mapping(), Point<dim>(x - epsilon));
    const auto regularized_right =
        evaluate_raw_potential(regularized_potential, discretization.get_mapping(), Point<dim>(x + epsilon));
    const auto continuous_left =
        evaluate_raw_potential(continuous_potential, discretization.get_mapping(), Point<dim>(x - epsilon));
    const auto continuous_right =
        evaluate_raw_potential(continuous_potential, discretization.get_mapping(), Point<dim>(x + epsilon));
    max_default_jump =
        std::max(max_default_jump, std::abs(default_right.mass_hessian[0][0] - default_left.mass_hessian[0][0]));
    max_regularized_jump = std::max(
        max_regularized_jump, std::abs(regularized_right.mass_hessian[0][0] - regularized_left.mass_hessian[0][0]));
    max_continuous_jump = std::max(max_continuous_jump,
                                   std::abs(continuous_right.mass_hessian[0][0] - continuous_left.mass_hessian[0][0]));
    max_potential_hessian_change =
        std::max(max_potential_hessian_change,
                 std::abs(regularized_right.potential_hessian[0][0] - default_right.potential_hessian[0][0]));
  }

  CAPTURE(max_default_jump, max_regularized_jump, max_continuous_jump, max_potential_hessian_change);
  CHECK(max_default_jump > 1e-3);
  CHECK(max_regularized_jump < max_default_jump);
  CHECK(max_continuous_jump < 1e-4);
  CHECK(max_potential_hessian_change < 1e-12);
}

TEST_CASE("Analytical two-field Hessian-jump potential benchmarks masses beyond the non-analytic surface",
          "[discretization][EoM][raw-potential][hessian][fv][accuracy][multifield][nonanalytic]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  constexpr double angle = 0.35;
  using Model = KinkedRadialPotentialGradientModel;
  using Discretization = FV::Discretization<Model, RectangularMesh<dim>>;

  setup_logger();
  const auto gradient = [](const auto &, const auto &values) {
    return std::array<double, dim>{{values[0], values[1]}};
  };
  const auto eigenmasses = [](const Tensor<2, dim> &hessian) {
    const double discriminant = std::sqrt((hessian[0][0] - hessian[1][1]) * (hessian[0][0] - hessian[1][1]) +
                                          4. * hessian[0][1] * hessian[0][1]);
    return std::array<double, 2>{
        {0.5 * (hessian[0][0] + hessian[1][1] - discriminant), 0.5 * (hessian[0][0] + hessian[1][1] + discriminant)}};
  };

  Tensor<1, dim> analytical_normal;
  analytical_normal[0] = std::cos(angle);
  analytical_normal[1] = std::sin(angle);
  Tensor<2, dim> inner_hessian;
  Tensor<2, dim> outer_hessian;
  const double kink_mass = Model::transverse_mass_square(Model::kink_radius * Model::kink_radius);
  for (uint d = 0; d < dim; ++d)
    for (uint e = 0; e < dim; ++e) {
      const double identity = d == e ? kink_mass : 0.;
      const double position_product =
          Model::kink_radius * Model::kink_radius * analytical_normal[d] * analytical_normal[e];
      inner_hessian[d][e] = identity + Model::inner_slope * position_product;
      outer_hessian[d][e] = identity + Model::outer_slope * position_product;
    }
  Tensor<1, dim> unit_scale;
  for (uint d = 0; d < dim; ++d)
    unit_scale[d] = 1.;
  const auto analytical_compatibility =
      DiFfRG::internal::hessian_jump_compatibility(inner_hessian, outer_hessian, unit_scale);
  const double normal_alignment = std::abs(analytical_compatibility.normal * analytical_normal);

  auto full_rank_outer_hessian = outer_hessian;
  full_rank_outer_hessian[0][0] += 0.4;
  full_rank_outer_hessian[1][1] -= 0.4;
  const auto full_rank_compatibility =
      DiFfRG::internal::hessian_jump_compatibility(inner_hessian, full_rank_outer_hessian, unit_scale);

  std::array<AccuracyMetrics, 2> cg2_radial_metrics;
  std::array<AccuracyMetrics, 2> unbiased_radial_metrics;
  std::array<AccuracyMetrics, 2> eom_side_radial_metrics;
  std::array<AccuracyMetrics, 2> gradient_transverse_metrics;
  std::array<AccuracyMetrics, 2> cg2_transverse_metrics;
  std::array<AccuracyMetrics, 2> unbiased_transverse_metrics;
  std::array<AccuracyMetrics, 2> eom_side_transverse_metrics;
  std::array<AccuracyMetrics, 2> near_cg2_radial_metrics;
  std::array<AccuracyMetrics, 2> near_unbiased_radial_metrics;
  std::array<AccuracyMetrics, 2> near_eom_side_radial_metrics;

  constexpr uint near_samples = 46;
  for (uint refinement = 0; refinement <= 1; ++refinement) {
    auto json = make_json(0);
    RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
    mesh.get_triangulation().refine_global(refinement);
    Discretization discretization(mesh, json);
    FE::FlowingVariables state(discretization);
    state.interpolate(Model{});

    Config::EoMConfig cg2_config;
    const auto cg2 = reconstruct_raw_potential(state.spatial_data(), discretization.get_dof_handler(),
                                               discretization.get_mapping(), gradient, cg2_config);
    auto unbiased_config = cg2_config;
    unbiased_config.raw_potential_recover_mass_hessian = true;
    unbiased_config.raw_potential_mass_hessian_jump_threshold = -1.;
    const auto unbiased = reconstruct_raw_potential(state.spatial_data(), discretization.get_dof_handler(),
                                                    discretization.get_mapping(), gradient, unbiased_config);
    auto eom_side_config = unbiased_config;
    eom_side_config.raw_potential_mass_hessian_jump_threshold = 0.35;
    const auto eom_side = reconstruct_raw_potential(state.spatial_data(), discretization.get_dof_handler(),
                                                    discretization.get_mapping(), gradient, eom_side_config);

    std::vector<double> exact_radial;
    std::vector<double> exact_transverse;
    std::vector<double> cg2_radial;
    std::vector<double> unbiased_radial;
    std::vector<double> eom_side_radial;
    std::vector<double> gradient_transverse;
    std::vector<double> cg2_transverse;
    std::vector<double> unbiased_transverse;
    std::vector<double> eom_side_transverse;
    for (uint sample = 0; sample <= 200; ++sample) {
      // This point is the exact EoM of U(phi) - J(phi_EoM) . phi, with
      // J(phi_EoM) chosen parallel to phi_EoM. The source does not alter either mass.
      const double radius = Model::kink_radius + 0.01 + 0.002 * sample;
      const Point<dim> point(radius * std::cos(angle), radius * std::sin(angle));
      const double radius_square = radius * radius;
      exact_radial.push_back(Model::radial_mass_square(radius_square));
      exact_transverse.push_back(Model::transverse_mass_square(radius_square));

      const auto cg2_value = evaluate_raw_potential(cg2, discretization.get_mapping(), point);
      const auto unbiased_value = evaluate_raw_potential(unbiased, discretization.get_mapping(), point);
      const auto eom_side_value = evaluate_raw_potential(eom_side, discretization.get_mapping(), point);
      const auto cg2_masses = eigenmasses(cg2_value.mass_hessian);
      const auto unbiased_masses = eigenmasses(unbiased_value.mass_hessian);
      const auto eom_side_masses = eigenmasses(eom_side_value.mass_hessian);
      cg2_transverse.push_back(cg2_masses[0]);
      cg2_radial.push_back(cg2_masses[1]);
      unbiased_transverse.push_back(unbiased_masses[0]);
      unbiased_radial.push_back(unbiased_masses[1]);
      eom_side_transverse.push_back(eom_side_masses[0]);
      eom_side_radial.push_back(eom_side_masses[1]);
      gradient_transverse.push_back(cg2_value.gradient[0] / point[0]);
    }

    cg2_radial_metrics[refinement] = accuracy_metrics(cg2_radial, exact_radial);
    unbiased_radial_metrics[refinement] = accuracy_metrics(unbiased_radial, exact_radial);
    eom_side_radial_metrics[refinement] = accuracy_metrics(eom_side_radial, exact_radial);
    gradient_transverse_metrics[refinement] = accuracy_metrics(gradient_transverse, exact_transverse);
    cg2_transverse_metrics[refinement] = accuracy_metrics(cg2_transverse, exact_transverse);
    unbiased_transverse_metrics[refinement] = accuracy_metrics(unbiased_transverse, exact_transverse);
    eom_side_transverse_metrics[refinement] = accuracy_metrics(eom_side_transverse, exact_transverse);

    const auto near_end = [&](const std::vector<double> &values) {
      return std::vector<double>(values.begin(), values.begin() + near_samples);
    };
    const auto near_exact_radial = near_end(exact_radial);
    near_cg2_radial_metrics[refinement] = accuracy_metrics(near_end(cg2_radial), near_exact_radial);
    near_unbiased_radial_metrics[refinement] = accuracy_metrics(near_end(unbiased_radial), near_exact_radial);
    near_eom_side_radial_metrics[refinement] = accuracy_metrics(near_end(eom_side_radial), near_exact_radial);

    print_accuracy("2D post-kink radial", refinement, "CG2 eigenmass", cg2_radial_metrics[refinement]);
    print_accuracy("2D post-kink radial", refinement, "continuous unbiased eigenmass",
                   unbiased_radial_metrics[refinement]);
    print_accuracy("2D post-kink radial", refinement, "EoM-side eigenmass", eom_side_radial_metrics[refinement]);
    print_accuracy("2D near-kink radial", refinement, "CG2 eigenmass", near_cg2_radial_metrics[refinement]);
    print_accuracy("2D near-kink radial", refinement, "continuous unbiased eigenmass",
                   near_unbiased_radial_metrics[refinement]);
    print_accuracy("2D near-kink radial", refinement, "EoM-side eigenmass", near_eom_side_radial_metrics[refinement]);
    print_accuracy("2D post-kink transverse", refinement, "gradient/sigma", gradient_transverse_metrics[refinement]);
    print_accuracy("2D post-kink transverse", refinement, "CG2 eigenmass", cg2_transverse_metrics[refinement]);
    print_accuracy("2D post-kink transverse", refinement, "continuous unbiased eigenmass",
                   unbiased_transverse_metrics[refinement]);
    print_accuracy("2D post-kink transverse", refinement, "EoM-side eigenmass",
                   eom_side_transverse_metrics[refinement]);
  }

  const double radial_jump = Model::radial_mass_square(Model::kink_radius * Model::kink_radius + 1e-10) -
                             Model::radial_mass_square(Model::kink_radius * Model::kink_radius - 1e-10);
  CAPTURE(radial_jump, analytical_compatibility.relative_jump, analytical_compatibility.rank_one_confidence,
          normal_alignment, cg2_radial_metrics[0].rms, unbiased_radial_metrics[0].rms, eom_side_radial_metrics[0].rms,
          near_cg2_radial_metrics[0].rms, near_unbiased_radial_metrics[0].rms, near_eom_side_radial_metrics[0].rms,
          gradient_transverse_metrics[1].rms, cg2_transverse_metrics[1].rms, unbiased_transverse_metrics[1].rms,
          eom_side_transverse_metrics[1].rms);
  CHECK(radial_jump > 0.2);
  CHECK(analytical_compatibility.is_rank_one);
  CHECK_FALSE(full_rank_compatibility.is_rank_one);
  CHECK(analytical_compatibility.rank_one_confidence > 1. - 1e-12);
  CHECK(normal_alignment > 1. - 1e-12);
  // The one-sided recovery needs the interface to be resolved: on the coarse mesh the unbiased field can be better.
  // Require convergence and compare methods only on the refined mesh instead of hiding this resolution limit.
  CHECK(cg2_radial_metrics[1].rms < cg2_radial_metrics[0].rms);
  CHECK(unbiased_radial_metrics[1].rms < unbiased_radial_metrics[0].rms);
  CHECK(eom_side_radial_metrics[1].rms < eom_side_radial_metrics[0].rms);
  CHECK(unbiased_radial_metrics[1].rms < cg2_radial_metrics[1].rms);
  CHECK(eom_side_radial_metrics[1].rms < unbiased_radial_metrics[1].rms);
  CHECK(unbiased_radial_metrics[1].roughness_rms < cg2_radial_metrics[1].roughness_rms);
  CHECK(eom_side_radial_metrics[1].roughness_rms < cg2_radial_metrics[1].roughness_rms);
  CHECK(near_eom_side_radial_metrics[1].rms < near_cg2_radial_metrics[1].rms);
  CHECK(near_eom_side_radial_metrics[1].rms < near_unbiased_radial_metrics[1].rms);
  CHECK(gradient_transverse_metrics[1].rms < gradient_transverse_metrics[0].rms);
  CHECK(eom_side_transverse_metrics[1].rms < 1.01 * unbiased_transverse_metrics[1].rms);
}

TEST_CASE("DG0 gradient recovery gives FV sources off-support moving minima with either smoothing policy",
          "[discretization][EoM][potential][smoothing][fv]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  const std::array<double, 4> positions{{0.413, 0.427, 0.441, 0.468}};
  for (const double position : positions) {
    const Point<dim> expected(position);
    const auto model = ModelAffineEoM<dim>(expected, {{{3.}}});
    const auto undamped = reconstruct_minimum_with_model<dim, FV::Discretization>(model, 0, 0.);
    const auto smoothed = reconstruct_minimum_with_model<dim, FV::Discretization>(model, 0, -1.);
    const double closest_support_point = std::round(smoothed[0] / 0.05) * 0.05;

    CAPTURE(position, undamped[0], smoothed[0]);
    CHECK(std::abs(undamped[0] - position) < 1e-9);
    CHECK(std::abs(smoothed[0] - position) < 1e-9);
    CHECK(std::abs(smoothed[0] - closest_support_point) > 1e-6);
    CHECK(std::abs(smoothed[0] - position) < 2e-2);
  }
}

TEST_CASE("Physical gradient-jump smoothing preserves smooth affine CG minima",
          "[discretization][EoM][potential][smoothing][cg]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const Point<dim> expected(0.437, 0.583);
  const ModelAffineEoM<dim> model(expected, {{{4., 1.25}, {1.25, 3.}}});
  const auto undamped = reconstruct_minimum_with_model<dim, CG::Discretization>(model, 2, 0.);
  const auto smoothed = reconstruct_minimum_with_model<dim, CG::Discretization>(model, 2, -1.);

  CHECK(distance(undamped, expected) < 1e-9);
  CHECK(distance(smoothed, expected) < 1e-9);
  CHECK(distance(smoothed, undamped) < 1e-9);
}

TEST_CASE("Physical gradient-jump smoothing gives a rotated FV source an interior minimum",
          "[discretization][EoM][potential][smoothing][fv][2d]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const Point<dim> expected(0.437, 0.583);
  const ModelAffineEoM<dim> model(expected, {{{4., 1.25}, {1.25, 3.}}});
  const auto smoothed = reconstruct_minimum_with_model<dim, FV::Discretization>(model, 0, -1.);

  CAPTURE(smoothed[0], smoothed[1]);
  CHECK(distance(smoothed, expected) < 3e-2);
  for (uint d = 0; d < dim; ++d) {
    const double closest_support_point = std::round(smoothed[d] / 0.05) * 0.05;
    CHECK(std::abs(smoothed[d] - closest_support_point) > 1e-6);
  }
}

TEST_CASE("A fixed physical EoM smoothing length is stable under uniform refinement",
          "[discretization][EoM][potential][smoothing][fv][refinement]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  constexpr double smoothing_length = 0.2;
  const Point<dim> expected(0.437);
  const ModelAffineEoM<dim> model(expected, {{{3.}}});
  const auto coarse =
      reconstruct_minimum_with_model<dim, FV::Discretization>(model, 0, smoothing_length, /*refinement=*/0);
  const auto fine =
      reconstruct_minimum_with_model<dim, FV::Discretization>(model, 0, smoothing_length, /*refinement=*/1);

  CAPTURE(coarse[0], fine[0]);
  CHECK(distance(coarse, expected) < 2e-2);
  CHECK(distance(fine, expected) < 2e-2);
  CHECK(distance(coarse, fine) < 1e-2);
}

TEST_CASE("EoM gradient-jump damping is symmetric positive semidefinite on adaptive interfaces",
          "[discretization][EoM][potential][smoothing][adaptive]")
{
  constexpr uint dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 2, 0., 1.);
  triangulation.begin_active()->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();

  FE_DGQ<dim> solution_fe(0);
  FE_Q<dim> potential_fe(2);
  DoFHandler<dim> solution_dof_handler(triangulation);
  DoFHandler<dim> potential_dof_handler(triangulation);
  solution_dof_handler.distribute_dofs(solution_fe);
  potential_dof_handler.distribute_dofs(potential_fe);
  Vector<double> solution(solution_dof_handler.n_dofs());
  MappingQ1<dim> mapping;
  QGauss<dim> quadrature(4);
  QGauss<dim - 1> face_quadrature(4);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(potential_dof_handler, constraints);
  constraints.close();
  DynamicSparsityPattern dsp(potential_dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(potential_dof_handler, dsp, constraints,
                                       /*keep_constrained_dofs=*/true);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> undamped(sparsity);
  SparseMatrix<double> damped(sparsity);
  Vector<double> undamped_rhs(potential_dof_handler.n_dofs());
  Vector<double> damped_rhs(potential_dof_handler.n_dofs());
  const auto zero_eom = [](const auto &, const auto &) { return std::array<double, dim>{{0., 0.}}; };

  DiFfRG::internal::PotentialAssemblyData<dim, double> assembly(potential_dof_handler, constraints, mapping,
                                                                solution_fe, potential_fe, quadrature, face_quadrature);
  DiFfRG::internal::assemble_potential_system(solution, solution_dof_handler, potential_fe, mapping, zero_eom,
                                              constraints, assembly, undamped, undamped_rhs, 0.,
                                              /*assemble_matrix = */ true);
  DiFfRG::internal::assemble_potential_system(solution, solution_dof_handler, potential_fe, mapping, zero_eom,
                                              constraints, assembly, damped, damped_rhs, 0.25,
                                              /*assemble_matrix = */ true);

  double penalty_norm = 0.;
  double symmetry_error = 0.;
  double quadratic_form = 0.;
  for (uint i = 0; i < potential_dof_handler.n_dofs(); ++i) {
    const double x_i = std::sin(0.37 * (i + 1.));
    for (uint j = 0; j < potential_dof_handler.n_dofs(); ++j) {
      const double penalty_ij = damped.el(i, j) - undamped.el(i, j);
      const double penalty_ji = damped.el(j, i) - undamped.el(j, i);
      const double x_j = std::sin(0.37 * (j + 1.));
      penalty_norm = std::max(penalty_norm, std::abs(penalty_ij));
      symmetry_error = std::max(symmetry_error, std::abs(penalty_ij - penalty_ji));
      quadratic_form += x_i * penalty_ij * x_j;
    }
  }

  CHECK(penalty_norm > 0.);
  CHECK(symmetry_error < 1e-12);
  CHECK(quadratic_form >= -1e-11);
  CHECK(undamped_rhs.l2_norm() == 0.);
  CHECK(damped_rhs.l2_norm() == 0.);
}

TEST_CASE("CG2 EoM potential reconstruction supports higher-order DG sources", "[discretization][EoM][dg]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const int fe_order = GENERATE(1, 2);
  const Point<dim> expected(0.4, 0.6);
  const auto EoM = reconstruct_minimum<dim, DG::Discretization>(parameters_with_minimum(expected), fe_order);

  REQUIRE(distance(EoM, expected) < 1.1e-1);
}

TEST_CASE("EoM potential reconstruction can return a boundary minimum", "[discretization][EoM][boundary]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  const Point<dim> expected(0., 0.);
  const auto EoM = reconstruct_minimum<dim, CG::Discretization>(parameters_with_boundary_minimum(), 2);

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
  const auto result = get_EoM_point_with_potential(
      EoM_cell, initial_condition.spatial_data(), discretization.get_dof_handler(), discretization.get_mapping(),
      [&](const auto &p, const auto &values) { return model.EoM(p, values); },
      [&](const auto &p, const auto &) { return p; }, 1e-12, 0);

  REQUIRE(distance(result.point, Point<dim>(0., 0.)) < 1e-12);
  REQUIRE_FALSE(result.potential.has_value());
}
