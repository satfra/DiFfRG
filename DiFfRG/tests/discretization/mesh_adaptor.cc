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
                            {"refine_threshold", 0.3},
                            {"coarsen_threshold", 0.03},
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

TEST_CASE("HAdaptivity with threshold marking keeps a mesh that resolves the solution", "[discretization][adaptivity]")
{
  DiFfRG::Init();
  using Discretization = CG::Discretization<FrontModel, RectangularMeshSerial<1>>;
  const auto config = make_config(0, 4, 2, 0, "threshold");
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

  const auto levels = [&]() {
    uint at_front = 0, far = std::numeric_limits<uint>::max();
    for (const auto &cell : discretization.get_triangulation().active_cell_iterators()) {
      if (std::abs(cell->center()[0] - model.front_position) < cell->diameter())
        at_front = std::max(at_front, uint(cell->level()));
      if (cell->center()[0] > 0.7) far = std::min(far, uint(cell->level()));
    }
    return std::pair{at_front, far};
  };

  // A fixed front: the mesh is refined at it up to the level, then left alone.
  uint changes = 0;
  for (uint i = 0; i < 10; ++i)
    changes += adaptor.adapt(u);
  CHECK(changes <= 6); // four levels of refinement, plus coarsening what turned out unnecessary
  CHECK(!adaptor.adapt(u));
  CHECK(levels() == std::pair<uint, uint>{4, 0});

  // Moving the front far away coarsens the old refinement, although nothing new needs to be refined.
  model.front_position = 0.9;
  for (uint i = 0; i < 10; ++i)
    adaptor.adapt(u);
  model.front_position = 0.2;
  const auto [at_old_front, _] = levels();
  CHECK(at_old_front == 0);
}
