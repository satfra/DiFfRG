#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <algorithm>

#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>
#include <deal.II/lac/affine_constraints.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <DiFfRG/common/init.hh>

namespace
{
  struct OriginConstraintModel : DiFfRG::def::ConstrainOriginSupportPointToZero<"u", OriginConstraintModel> {
  };

  auto make_json()
  {
    return DiFfRG::json::value(
        {{"physical", {}},
         {"discretization",
          {{"fe_order", 2},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", "0:1:1"}, {"y_grid", "0:1:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1.0},
             {"level", 0},
             {"refine_percent", 0.0},
             {"coarsen_percent", 0.0}}}}},
         {"output", {{"live_plot", false}, {"verbosity", 0}}}});
  }

  template <typename PointRange> std::vector<double> x_coordinates(const PointRange &points)
  {
    std::vector<double> x;
    x.reserve(points.size());
    for (const auto &point : points)
      x.push_back(point[0]);
    return x;
  }

  template <typename PointRange> std::vector<double> sorted_x_coordinates(const PointRange &points)
  {
    auto x = x_coordinates(points);
    std::sort(x.begin(), x.end());
    return x;
  }

  template <typename Metadata, typename Discretization>
  void require_point_alignment(const Metadata &metadata, const Discretization &discretization)
  {
    for (std::size_t c = 0; c < metadata.component_boundary_dofs.size(); ++c) {
      for (uint i = 0; i < metadata.component_boundary_dofs[c].n_elements(); ++i)
        CHECK(metadata.component_boundary_points[c][i] ==
              discretization.get_support_point(metadata.component_boundary_dofs[c].nth_index_in_set(i)));
      for (uint i = 0; i < metadata.component_support_dofs[c].n_elements(); ++i)
        CHECK(metadata.component_support_points[c][i] ==
              discretization.get_support_point(metadata.component_support_dofs[c].nth_index_in_set(i)));
    }
  }

  void ensure_logger()
  {
    try {
      auto log = spdlog::stdout_color_mt("log");
      log->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
  }

  dealii::IndexSet make_index_set(const std::initializer_list<dealii::types::global_dof_index> indices)
  {
    dealii::IndexSet index_set(128);
    for (const auto index : indices)
      index_set.add_index(index);
    index_set.compress();
    return index_set;
  }

  using TwoDimensionalComponents =
      DiFfRG::ComponentDescriptor<DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u">, DiFfRG::Scalar<"v">>>;
  using ReversedTwoDimensionalComponents =
      DiFfRG::ComponentDescriptor<DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"v">, DiFfRG::Scalar<"u">>>;
  using SingleUTwoDimensionalComponents =
      DiFfRG::ComponentDescriptor<DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u">>>;

  struct ULineConstraintModel : DiFfRG::def::AbstractModel<ULineConstraintModel, TwoDimensionalComponents>,
                                DiFfRG::def::ConstrainOriginSupportPointToZero<"u", ULineConstraintModel> {
    template <DiFfRG::FixedString component_name> struct OriginConstraintCoordinate;
  };

  template <> struct ULineConstraintModel::OriginConstraintCoordinate<"u"> {
    static double signed_coordinate(const dealii::Point<2> &point) { return point[0]; }
  };

  struct VLineConstraintModel : DiFfRG::def::AbstractModel<VLineConstraintModel, TwoDimensionalComponents>,
                                DiFfRG::def::ConstrainOriginSupportPointToZero<"v", VLineConstraintModel> {
    template <DiFfRG::FixedString component_name> struct OriginConstraintCoordinate;
  };

  template <> struct VLineConstraintModel::OriginConstraintCoordinate<"v"> {
    static double signed_coordinate(const dealii::Point<2> &point) { return point[1]; }
  };

  struct ULineTieConstraintModel : DiFfRG::def::AbstractModel<ULineTieConstraintModel, SingleUTwoDimensionalComponents>,
                                   DiFfRG::def::ConstrainOriginSupportPointToZero<"u", ULineTieConstraintModel> {
    template <DiFfRG::FixedString component_name> struct OriginConstraintCoordinate;
  };

  template <> struct ULineTieConstraintModel::OriginConstraintCoordinate<"u"> {
    static double signed_coordinate(const dealii::Point<2> &point) { return point[0]; }
  };

  struct BoundaryLineConstraintModel
      : DiFfRG::def::AbstractModel<BoundaryLineConstraintModel, SingleUTwoDimensionalComponents>,
        DiFfRG::def::ConstrainOriginBoundaryPointToZero<"u", BoundaryLineConstraintModel> {
    template <DiFfRG::FixedString component_name> struct OriginConstraintCoordinate;
  };

  template <> struct BoundaryLineConstraintModel::OriginConstraintCoordinate<"u"> {
    static double signed_coordinate(const dealii::Point<2> &point) { return point[0]; }
  };

  struct ReversedOrderConstraintModel
      : DiFfRG::def::AbstractModel<ReversedOrderConstraintModel, ReversedTwoDimensionalComponents> {
    template <DiFfRG::FixedString component_name> struct OriginConstraintCoordinate;

    template <typename Constraints, typename Context>
    void apply_affine_constraints(Constraints &constraints, const Context &context) const
    {
      DiFfRG::def::ConstrainOriginSupportPointToZero<"u", ReversedOrderConstraintModel>{}.apply_affine_constraints(
          constraints, context);
      DiFfRG::def::ConstrainOriginSupportPointToZero<"v", ReversedOrderConstraintModel>{}.apply_affine_constraints(
          constraints, context);
    }
  };

  template <> struct ReversedOrderConstraintModel::OriginConstraintCoordinate<"u"> {
    static double signed_coordinate(const dealii::Point<2> &point) { return point[0]; }
  };

  template <> struct ReversedOrderConstraintModel::OriginConstraintCoordinate<"v"> {
    static double signed_coordinate(const dealii::Point<2> &point) { return point[1]; }
  };
} // namespace

TEST_CASE("Affine-constraint metadata captures interior support points for scalar CG", "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Discretization = CG::Discretization<Components, RectangularMesh<dim>>;

  auto json = make_json();
  ensure_logger();
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);

  const auto metadata = DiFfRG::internal::build_affine_constraint_metadata<Components, dim>(discretization);

  REQUIRE(metadata.component_boundary_dofs.size() == 1);
  REQUIRE(metadata.component_support_dofs.size() == 1);
  REQUIRE(metadata.component_boundary_dofs[0].n_elements() == 2);
  REQUIRE(metadata.component_support_dofs[0].n_elements() == 3);
  CHECK(sorted_x_coordinates(metadata.component_boundary_points[0]) == std::vector<double>{0.0, 1.0});
  CHECK(sorted_x_coordinates(metadata.component_support_points[0]) == std::vector<double>{0.0, 0.5, 1.0});
  require_point_alignment(metadata, discretization);
}

TEST_CASE("Affine-constraint context exposes named component views", "[discretization][constraints]")
{
  DiFfRG::Init();
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Discretization = CG::Discretization<Components, RectangularMesh<dim>>;

  auto json = make_json();
  ensure_logger();
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);

  const auto metadata = DiFfRG::internal::build_affine_constraint_metadata<Components, dim>(discretization);
  const AffineConstraintContext<Components, dim> context(metadata);

  REQUIRE(metadata.component_boundary_dofs.size() == Components::count_fe_functions());
  REQUIRE(metadata.component_support_dofs.size() == Components::count_fe_functions());
  for (uint c = 0; c < Components::count_fe_functions(); ++c) {
    CHECK(metadata.component_boundary_dofs[c].n_elements() == 2);
    CHECK(metadata.component_support_dofs[c].n_elements() == 3);
    CHECK(sorted_x_coordinates(metadata.component_boundary_points[c]) == std::vector<double>{0.0, 1.0});
    CHECK(sorted_x_coordinates(metadata.component_support_points[c]) == std::vector<double>{0.0, 0.5, 1.0});
  }
  CHECK(context.template boundary<"u">().dofs == metadata.component_boundary_dofs[0]);
  CHECK(context.template boundary<"v">().dofs == metadata.component_boundary_dofs[1]);
  CHECK(context.template support<"u">().points == metadata.component_support_points[0]);
  CHECK(context.template support<"v">().points == metadata.component_support_points[1]);
  require_point_alignment(metadata, discretization);
}

TEST_CASE("ConstrainOriginSupportPointToZero constrains only the selected named component",
          "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  struct NamedComponentConstraintModel : def::AbstractModel<NamedComponentConstraintModel, Components>,
                                         def::ConstrainOriginSupportPointToZero<"v", NamedComponentConstraintModel> {
  };

  std::vector<IndexSet> component_boundary_dofs(2, IndexSet(16));
  std::vector<IndexSet> component_support_dofs(2, IndexSet(16));
  component_boundary_dofs[0].add_index(2);
  component_boundary_dofs[1].add_index(7);
  component_support_dofs[0].add_index(2);
  component_support_dofs[1].add_index(7);
  for (auto &index_set : component_boundary_dofs)
    index_set.compress();
  for (auto &index_set : component_support_dofs)
    index_set.compress();

  const std::vector<std::vector<Point<dim>>> component_boundary_points{{Point<dim>(0.0)}, {Point<dim>(0.0)}};
  const std::vector<std::vector<Point<dim>>> component_support_points{{Point<dim>(0.0)}, {Point<dim>(0.0)}};
  const AffineConstraintContext<Components, dim> context(component_boundary_dofs, component_boundary_points,
                                                         component_support_dofs, component_support_points);

  dealii::AffineConstraints<double> constraints;
  NamedComponentConstraintModel model;
  model.affine_constraints(constraints, context);
  constraints.close();

  CHECK_FALSE(constraints.is_constrained(2));
  CHECK(constraints.is_constrained(7));
  CHECK(constraints.get_inhomogeneity(7) == Catch::Approx(0.0));
}

TEST_CASE("ConstrainOriginSupportPointToZero chooses the nearest support dof when the origin is interior",
          "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;

  std::vector<IndexSet> component_boundary_dofs(1, IndexSet(16));
  std::vector<IndexSet> component_support_dofs(1, IndexSet(16));
  component_boundary_dofs[0].add_index(8);
  component_boundary_dofs[0].compress();
  component_support_dofs[0].add_index(3);
  component_support_dofs[0].add_index(5);
  component_support_dofs[0].add_index(8);
  component_support_dofs[0].compress();

  const std::vector<std::vector<Point<dim>>> component_boundary_points{{Point<dim>(1.5)}};
  const std::vector<std::vector<Point<dim>>> component_support_points{
      {Point<dim>(-0.25), Point<dim>(0.75), Point<dim>(1.5)}};
  const AffineConstraintContext<Components, dim> context(component_boundary_dofs, component_boundary_points,
                                                         component_support_dofs, component_support_points);

  dealii::AffineConstraints<double> constraints;
  OriginConstraintModel model;
  model.apply_affine_constraints(constraints, context);
  constraints.close();

  CHECK(constraints.is_constrained(3));
  CHECK_FALSE(constraints.is_constrained(5));
  CHECK_FALSE(constraints.is_constrained(8));
  CHECK(constraints.get_inhomogeneity(3) == Catch::Approx(0.0));
}

TEST_CASE("ConstrainOriginSupportPointToZero prefers the non-negative side on symmetric support ties",
          "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;

  std::vector<IndexSet> component_boundary_dofs(1, IndexSet(16));
  std::vector<IndexSet> component_support_dofs(1, IndexSet(16));
  component_support_dofs[0].add_index(4);
  component_support_dofs[0].add_index(9);
  component_support_dofs[0].compress();

  const std::vector<std::vector<Point<dim>>> component_boundary_points{{}};
  const std::vector<std::vector<Point<dim>>> component_support_points{{Point<dim>(-0.5), Point<dim>(0.5)}};
  const AffineConstraintContext<Components, dim> context(component_boundary_dofs, component_boundary_points,
                                                         component_support_dofs, component_support_points);

  dealii::AffineConstraints<double> constraints;
  OriginConstraintModel model;
  model.apply_affine_constraints(constraints, context);
  constraints.close();

  CHECK_FALSE(constraints.is_constrained(4));
  CHECK(constraints.is_constrained(9));
  CHECK(constraints.get_inhomogeneity(9) == Catch::Approx(0.0));
}

TEST_CASE("ConstrainOriginBoundaryPointToZero chooses only from boundary dofs", "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  struct BoundaryOriginConstraintModel : def::AbstractModel<BoundaryOriginConstraintModel, Components>,
                                         def::ConstrainOriginBoundaryPointToZero<"u", BoundaryOriginConstraintModel> {
  };

  std::vector<IndexSet> component_boundary_dofs(1, IndexSet(16));
  std::vector<IndexSet> component_support_dofs(1, IndexSet(16));
  component_boundary_dofs[0].add_index(6);
  component_boundary_dofs[0].add_index(12);
  component_boundary_dofs[0].compress();
  component_support_dofs[0].add_index(2);
  component_support_dofs[0].add_index(6);
  component_support_dofs[0].add_index(12);
  component_support_dofs[0].compress();

  const std::vector<std::vector<Point<dim>>> component_boundary_points{{Point<dim>(0.25), Point<dim>(1.25)}};
  const std::vector<std::vector<Point<dim>>> component_support_points{
      {Point<dim>(0.0), Point<dim>(0.25), Point<dim>(1.25)}};
  const AffineConstraintContext<Components, dim> context(component_boundary_dofs, component_boundary_points,
                                                         component_support_dofs, component_support_points);

  dealii::AffineConstraints<double> constraints;
  BoundaryOriginConstraintModel model;
  model.affine_constraints(constraints, context);
  constraints.close();

  CHECK_FALSE(constraints.is_constrained(2));
  CHECK(constraints.is_constrained(6));
  CHECK_FALSE(constraints.is_constrained(12));
  CHECK(constraints.get_inhomogeneity(6) == Catch::Approx(0.0));
}

TEST_CASE("ConstrainOriginSupportPointToZero selects the nearest phi_1 line in 2D", "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 2;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;

  std::vector<IndexSet> component_boundary_dofs(2, IndexSet(128));
  std::vector<IndexSet> component_support_dofs{make_index_set({1, 2, 3, 4, 5, 6, 7}), make_index_set({21, 22, 23})};
  for (auto &index_set : component_boundary_dofs)
    index_set.compress();

  const std::vector<std::vector<Point<dim>>> component_boundary_points(2);
  const std::vector<std::vector<Point<dim>>> component_support_points{
      {Point<dim>(-0.25, -1.0), Point<dim>(-0.25, 0.0), Point<dim>(-0.25, 1.0), Point<dim>(0.25, -1.0),
       Point<dim>(0.25, 0.0), Point<dim>(0.25, 1.0), Point<dim>(0.75, 0.0)},
      {Point<dim>(0.0, -1.0), Point<dim>(0.0, 0.0), Point<dim>(0.0, 1.0)}};
  const AffineConstraintContext<Components, dim> context(component_boundary_dofs, component_boundary_points,
                                                         component_support_dofs, component_support_points);

  dealii::AffineConstraints<double> constraints;
  ULineConstraintModel model;
  model.apply_affine_constraints(constraints, context);
  constraints.close();

  CHECK_FALSE(constraints.is_constrained(1));
  CHECK_FALSE(constraints.is_constrained(2));
  CHECK_FALSE(constraints.is_constrained(3));
  CHECK(constraints.is_constrained(4));
  CHECK(constraints.is_constrained(5));
  CHECK(constraints.is_constrained(6));
  CHECK_FALSE(constraints.is_constrained(7));
  CHECK_FALSE(constraints.is_constrained(21));
  CHECK_FALSE(constraints.is_constrained(22));
  CHECK_FALSE(constraints.is_constrained(23));
  CHECK(constraints.get_inhomogeneity(4) == Catch::Approx(0.0));
  CHECK(constraints.get_inhomogeneity(5) == Catch::Approx(0.0));
  CHECK(constraints.get_inhomogeneity(6) == Catch::Approx(0.0));
}

TEST_CASE("ConstrainOriginSupportPointToZero selects the nearest phi_2 line in 2D", "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 2;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;

  std::vector<IndexSet> component_boundary_dofs(2, IndexSet(128));
  std::vector<IndexSet> component_support_dofs{make_index_set({1, 2, 3}), make_index_set({21, 22, 23, 24, 25, 26, 27})};
  for (auto &index_set : component_boundary_dofs)
    index_set.compress();

  const std::vector<std::vector<Point<dim>>> component_boundary_points(2);
  const std::vector<std::vector<Point<dim>>> component_support_points{
      {Point<dim>(-1.0, 0.0), Point<dim>(0.0, 0.0), Point<dim>(1.0, 0.0)},
      {Point<dim>(-1.0, -0.1), Point<dim>(0.0, -0.1), Point<dim>(1.0, -0.1), Point<dim>(-1.0, 0.1),
       Point<dim>(0.0, 0.1), Point<dim>(1.0, 0.1), Point<dim>(0.0, 0.4)}};
  const AffineConstraintContext<Components, dim> context(component_boundary_dofs, component_boundary_points,
                                                         component_support_dofs, component_support_points);

  dealii::AffineConstraints<double> constraints;
  VLineConstraintModel model;
  model.apply_affine_constraints(constraints, context);
  constraints.close();

  CHECK_FALSE(constraints.is_constrained(1));
  CHECK_FALSE(constraints.is_constrained(2));
  CHECK_FALSE(constraints.is_constrained(3));
  CHECK_FALSE(constraints.is_constrained(21));
  CHECK_FALSE(constraints.is_constrained(22));
  CHECK_FALSE(constraints.is_constrained(23));
  CHECK(constraints.is_constrained(24));
  CHECK(constraints.is_constrained(25));
  CHECK(constraints.is_constrained(26));
  CHECK_FALSE(constraints.is_constrained(27));
  CHECK(constraints.get_inhomogeneity(24) == Catch::Approx(0.0));
  CHECK(constraints.get_inhomogeneity(25) == Catch::Approx(0.0));
  CHECK(constraints.get_inhomogeneity(26) == Catch::Approx(0.0));
}

TEST_CASE("ConstrainOriginSupportPointToZero prefers the non-negative side on 2D support ties",
          "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 2;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;

  std::vector<IndexSet> component_boundary_dofs(1, IndexSet(128));
  component_boundary_dofs[0].compress();
  std::vector<IndexSet> component_support_dofs{make_index_set({3, 4, 5, 6})};

  const std::vector<std::vector<Point<dim>>> component_boundary_points(1);
  const std::vector<std::vector<Point<dim>>> component_support_points{
      {Point<dim>(-0.5, -1.0), Point<dim>(-0.5, 1.0), Point<dim>(0.5, -1.0), Point<dim>(0.5, 1.0)}};
  const AffineConstraintContext<Components, dim> context(component_boundary_dofs, component_boundary_points,
                                                         component_support_dofs, component_support_points);

  dealii::AffineConstraints<double> constraints;
  ULineTieConstraintModel model;
  model.apply_affine_constraints(constraints, context);
  constraints.close();

  CHECK_FALSE(constraints.is_constrained(3));
  CHECK_FALSE(constraints.is_constrained(4));
  CHECK(constraints.is_constrained(5));
  CHECK(constraints.is_constrained(6));
}

TEST_CASE("ConstrainOriginBoundaryPointToZero selects the nearest boundary line in 2D", "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 2;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;

  std::vector<IndexSet> component_boundary_dofs{make_index_set({4, 5, 6, 7})};
  std::vector<IndexSet> component_support_dofs{make_index_set({1, 2, 4, 5, 6, 7})};

  const std::vector<std::vector<Point<dim>>> component_boundary_points{
      {Point<dim>(0.2, -1.0), Point<dim>(0.2, 0.0), Point<dim>(0.2, 1.0), Point<dim>(0.8, 0.0)}};
  const std::vector<std::vector<Point<dim>>> component_support_points{{Point<dim>(0.0, -1.0), Point<dim>(0.0, 1.0),
                                                                       Point<dim>(0.2, -1.0), Point<dim>(0.2, 0.0),
                                                                       Point<dim>(0.2, 1.0), Point<dim>(0.8, 0.0)}};
  const AffineConstraintContext<Components, dim> context(component_boundary_dofs, component_boundary_points,
                                                         component_support_dofs, component_support_points);

  dealii::AffineConstraints<double> constraints;
  BoundaryLineConstraintModel model;
  model.affine_constraints(constraints, context);
  constraints.close();

  CHECK_FALSE(constraints.is_constrained(1));
  CHECK_FALSE(constraints.is_constrained(2));
  CHECK(constraints.is_constrained(4));
  CHECK(constraints.is_constrained(5));
  CHECK(constraints.is_constrained(6));
  CHECK_FALSE(constraints.is_constrained(7));
}

TEST_CASE("ConstrainOriginSupportPointToZero uses explicit 2D coordinate policies instead of component order",
          "[discretization][constraints]")
{
  using namespace DiFfRG;

  constexpr uint dim = 2;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"v">, Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;

  std::vector<IndexSet> component_boundary_dofs(2, IndexSet(128));
  std::vector<IndexSet> component_support_dofs{make_index_set({1, 2, 3, 4}), make_index_set({21, 22, 23, 24})};
  for (auto &index_set : component_boundary_dofs)
    index_set.compress();

  const std::vector<std::vector<Point<dim>>> component_boundary_points(2);
  const std::vector<std::vector<Point<dim>>> component_support_points{
      {Point<dim>(-1.0, -0.2), Point<dim>(1.0, -0.2), Point<dim>(-1.0, 0.2), Point<dim>(1.0, 0.2)},
      {Point<dim>(-0.2, -1.0), Point<dim>(-0.2, 1.0), Point<dim>(0.2, -1.0), Point<dim>(0.2, 1.0)}};
  const AffineConstraintContext<Components, dim> context(component_boundary_dofs, component_boundary_points,
                                                         component_support_dofs, component_support_points);

  dealii::AffineConstraints<double> constraints;
  ReversedOrderConstraintModel model;
  model.apply_affine_constraints(constraints, context);
  constraints.close();

  CHECK_FALSE(constraints.is_constrained(1));
  CHECK_FALSE(constraints.is_constrained(2));
  CHECK(constraints.is_constrained(3));
  CHECK(constraints.is_constrained(4));
  CHECK_FALSE(constraints.is_constrained(21));
  CHECK_FALSE(constraints.is_constrained(22));
  CHECK(constraints.is_constrained(23));
  CHECK(constraints.is_constrained(24));
}
