#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/mesh/h_adaptivity.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <boilerplate/models.hh>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <tuple>

using namespace DiFfRG;
using namespace dealii;

namespace
{
  /// Refines towards large x: the indicator is the position.
  class Model : public Testing::ModelBurgers<1>
  {
  public:
    using Testing::ModelBurgers<1>::ModelBurgers;

    template <int dim, typename NumberType, typename Solution>
    void cell_indicator(NumberType &indicator, const Point<dim> &p, const Solution &) const
    {
      indicator = p[0];
    }
  };

  /// Refines at a front x = front_position that the test moves between adaptations.
  class FrontModel : public Testing::ModelBurgers<1>
  {
  public:
    using Testing::ModelBurgers<1>::ModelBurgers;
    double front_position = 0.2;

    template <int dim, typename NumberType, typename Solution>
    void cell_indicator(NumberType &indicator, const Point<dim> &p, const Solution &) const
    {
      indicator = std::exp(-std::pow((p[0] - front_position) / 0.03, 2));
    }
  };

  ConfigTree make_config(const uint refine, const uint level, const uint coarsen_buffer = 2, const uint coarsen_age = 2,
                         const std::string &marking = "fraction")
  {
    return json::value({{"physical", {}},
                        {"discretization",
                         {{"fe_order", 1},
                          {"overintegration", 0},
                          {"EoM_abs_tol", 1e-10},
                          {"EoM_max_iter", 0},
                          {"grid", {{"x_grid", "0:0.125:1"}, {"refine", refine}}},
                          {"adaptivity",
                           {{"start_adapt_at", 0.},
                            {"adapt_dt", 1e-1},
                            {"level", level},
                            {"refine_percent", 0.3},
                            {"coarsen_percent", 0.3},
                            {"marking", marking},
                            {"tolerance", 1e-3},
                            {"coarsen_buffer", coarsen_buffer},
                            {"coarsen_age", coarsen_age}}}}},
                        {"output", {{"verbosity", 0}}}});
  }

  /// Adapt @p n times and return the smallest and largest active level.
  std::pair<uint, uint> adapt_repeatedly(const uint refine, const uint level, const uint n)
  {
    using Discretization = CG::Discretization<Model, RectangularMeshSerial<1>>;
    const auto config = make_config(refine, level);
    Testing::PhysicalParameters prm;
    prm.initial_x0 = {{0.2, 0., 0.}};
    prm.initial_x1 = {{1., 0., 0.}};
    Model model(prm);
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
    Discretization discretization(mesh, config);
    CG::Assembler<Discretization> assembler(discretization, model, config);
    FE::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    typename Discretization::VectorType u(state.spatial_data());

    HAdaptivity<CG::Assembler<Discretization>> adaptor(assembler, config);
    for (uint i = 0; i < n; ++i)
      adaptor.adapt(u);

    uint lowest = std::numeric_limits<uint>::max(), highest = 0;
    for (const auto &cell : discretization.get_triangulation().active_cell_iterators()) {
      lowest = std::min(lowest, uint(cell->level()));
      highest = std::max(highest, uint(cell->level()));
    }
    return {lowest, highest};
  }
} // namespace

TEST_CASE("HAdaptivity keeps the levels between the initial mesh and adaptivity/level", "[discretization][adaptivity]")
{
  DiFfRG::Init();
  SECTION("Unrefined initial mesh")
  {
    const auto [lowest, highest] = adapt_repeatedly(0, 2, 5);
    CHECK(lowest == 0);
    CHECK(highest == 2);
  }
  SECTION("Initial mesh refined beyond adaptivity/level: no refinement, no coarsening below it")
  {
    const auto [lowest, highest] = adapt_repeatedly(2, 1, 5);
    CHECK(lowest == 2);
    CHECK(highest == 2);
  }
  SECTION("Initial mesh refined below adaptivity/level")
  {
    const auto [lowest, highest] = adapt_repeatedly(1, 3, 5);
    CHECK(lowest == 1);
    CHECK(highest == 3);
  }
}

namespace
{
  /// Move a refinement front across the mesh; returns (cells coarsened right after their creation, any coarsening).
  std::pair<uint, bool> sweep_front(const uint coarsen_buffer, const uint coarsen_age)
  {
    using Discretization = CG::Discretization<FrontModel, RectangularMeshSerial<1>>;
    const auto config = make_config(0, 4, coarsen_buffer, coarsen_age);
    Testing::PhysicalParameters prm;
    prm.initial_x0 = {{0.2, 0., 0.}};
    prm.initial_x1 = {{1., 0., 0.}};
    FrontModel model(prm);
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
    Discretization discretization(mesh, config);
    CG::Assembler<Discretization> assembler(discretization, model, config);
    FE::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    typename Discretization::VectorType u(state.spatial_data());
    HAdaptivity<CG::Assembler<Discretization>> adaptor(assembler, config);

    const auto active = [&]() {
      std::set<CellId> ids;
      for (const auto &cell : discretization.get_triangulation().active_cell_iterators())
        ids.insert(cell->id());
      return ids;
    };
    std::set<CellId> before = active(), created_last;
    uint coarsened_fresh = 0;
    bool coarsened = false;
    for (uint i = 0; i < 12; ++i) {
      model.front_position = 0.2 + 0.03 * i;
      if (!adaptor.adapt(u)) continue;
      const auto after = active();
      // A cell created by the previous adaptation may be refined further, but not coarsened away.
      for (const auto &id : created_last)
        coarsened_fresh +=
            std::none_of(after.begin(), after.end(), [&](const CellId &a) { return a == id || id.is_ancestor_of(a); });
      created_last.clear();
      for (const auto &id : after)
        if (!before.contains(id)) created_last.insert(id);
      // A new cell that is the parent of a previously active one is the result of coarsening.
      for (const auto &id : created_last)
        for (const auto &old : before)
          coarsened |= id.is_parent_of(old);
      before = after;
    }
    return {coarsened_fresh, coarsened};
  }
} // namespace

TEST_CASE("HAdaptivity does not coarsen right behind a moving front", "[discretization][adaptivity]")
{
  DiFfRG::Init();
  const auto [fresh_protected, coarsened_protected] = sweep_front(2, 2);
  CHECK(fresh_protected == 0);
  CHECK(coarsened_protected); // the mesh far behind the front is still coarsened
  const auto [fresh_unprotected, coarsened_unprotected] = sweep_front(0, 0);
  CHECK(fresh_unprotected > 0); // without the protection, the same sweep coarsens fresh cells
}

namespace
{
  /// No refinement indicator of its own: HAdaptivity uses the curvature indicator. u = tanh((x - 0.4) / width).
  class Smooth : public def::AbstractModel<Smooth, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
                 public def::Time,
                 public def::FlowBoundaries<Smooth>,
                 public def::AD<Smooth>
  {
  public:
    double width = 1.;
    template <typename Vector> void initial_condition(const Point<1> &x, Vector &values) const
    {
      values[0] = std::tanh((x[0] - 0.4) / width);
    }
  };
  static_assert(!DiFfRG::internal::defines_refinement_indicator<Smooth>());
  static_assert(DiFfRG::internal::defines_refinement_indicator<Model>());
  static_assert(DiFfRG::internal::defines_refinement_indicator<FrontModel>());

  /// Adapt with tolerance marking and the curvature indicator until the mesh stays; returns (adaptations that
  /// changed the mesh, finest level near x = 0.4, coarsest level far from it).
  std::tuple<uint, uint, uint> adapt_until_fixed(const double width)
  {
    using Discretization = CG::Discretization<Smooth, RectangularMeshSerial<1>>;
    using Asm = CG::Assembler<Discretization>;
    const auto config = make_config(0, 5, 2, 0, "tolerance");
    Smooth model;
    model.width = width;
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
    Discretization discretization(mesh, config);
    Asm assembler(discretization, model, config);
    FE::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    typename Discretization::VectorType u(state.spatial_data());
    HAdaptivity<Asm> adaptor(assembler, config);
    uint changes = 0;
    for (uint i = 0; i < 12; ++i)
      changes += adaptor.adapt(u);
    REQUIRE(!adaptor.adapt(u));
    uint finest = 0, coarsest = std::numeric_limits<uint>::max();
    for (const auto &cell : discretization.get_triangulation().active_cell_iterators()) {
      if (std::abs(cell->center()[0] - 0.4) < 0.05) finest = std::max(finest, uint(cell->level()));
      if (std::abs(cell->center()[0] - 0.4) > 0.3) coarsest = std::min(coarsest, uint(cell->level()));
    }
    return {changes, finest, coarsest};
  }
} // namespace

TEST_CASE("HAdaptivity with tolerance marking refines only where a feature is unresolved",
          "[discretization][adaptivity]")
{
  DiFfRG::Init();
  SECTION("a solution the initial mesh resolves keeps it")
  {
    const auto [changes, finest, coarsest] = adapt_until_fixed(10.);
    CHECK(changes == 0);
    CHECK(finest == 0);
  }
  SECTION("a steep front is refined, the smooth rest is not, and the mesh then stays")
  {
    const auto [changes, finest, coarsest] = adapt_until_fixed(0.02);
    CHECK(changes > 0);
    CHECK(finest >= 3);
    CHECK(coarsest == 0);
  }
}
