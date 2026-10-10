#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/mesh/h_adaptivity.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <boilerplate/models.hh>

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

  ConfigTree make_config(const uint refine, const uint level)
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
                            {"coarsen_percent", 0.3}}}}},
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
