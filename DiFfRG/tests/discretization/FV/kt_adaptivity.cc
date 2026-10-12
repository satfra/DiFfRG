#include <catch2/catch_all.hpp>

#include <deal.II/base/quadrature_lib.h>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/mesh/h_adaptivity.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <set>
#include <utility>

using namespace DiFfRG;
using namespace dealii;

// KT on an adapted 1D mesh: neighbouring cells of different refinement levels.

namespace
{
  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>;

  /// Linear advection F = c u; the ghosts continue the cell average constantly; refines where |u'| is large.
  class Advection : public def::AbstractModel<Advection, Components>, public def::Time, public def::AD<Advection>
  {
  public:
    static constexpr double c = 1.5;
    static double exact(const double x) { return 2. + std::sin(3. * x); }

    template <typename Vector> void initial_condition(const Point<1> &x, Vector &values) const
    {
      values[0] = exact(x[0]);
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, 1, NT>, 1> &F, const Point<1> &, const Solution &sol) const
    {
      F[0][0] = c * get<"fe_functions">(sol)[0];
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, 1> &S, const Point<1> &, const Solution &) const
    {
      S[0] = 0.;
    }

    template <int dim, typename NT, typename Solution>
    void cell_indicator(NT &indicator, const Point<dim> &, const Solution &sol) const
    {
      indicator = std::abs(get<"fe_derivatives">(sol)[0][0]);
    }

    template <int mdim, typename NT, size_t n>
    bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n> &u, def::BoundaryStencilPoints<mdim> &x,
                                const Point<mdim> &x_face) const
    {
      using namespace def::BoundaryStencilIndex;
      const Tensor<1, mdim> to_ghost = 2. * (x_face - x[physical_cell]);
      const bool lower = to_ghost[0] < 0.;
      const auto [inner, outer] = lower ? std::pair{lower_inner, lower_outer} : std::pair{upper_inner, upper_outer};
      x[inner] = x[physical_cell] + to_ghost;
      x[outer] = x[physical_cell] + 2. * to_ghost;
      u[inner][0] = u[outer][0] = u[physical_cell][0];
      return true;
    }
  };

  using Discretization = FV::Discretization<Advection, RectangularMeshSerial<1>, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Advection>;
  using VectorType = typename Discretization::VectorType;

  ConfigTree make_config()
  {
    try {
      spdlog::stdout_color_mt("log")->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
    return json::value({{"physical", {{"Lambda", 1.}}},
                        {"discretization",
                         {{"fe_order", 0},
                          {"overintegration", 0},
                          {"EoM_abs_tol", 1e-10},
                          {"EoM_max_iter", 0},
                          {"grid", {{"x_grid", "0:0.05:1"}, {"refine", 0}}},
                          {"adaptivity",
                           {{"start_adapt_at", 0.},
                            {"adapt_dt", 0.1},
                            {"level", 3},
                            {"marking", "fraction"},
                            {"refine_percent", 0.3},
                            {"coarsen_percent", 0.},
                            {"coarsen_buffer", 0},
                            {"coarsen_age", 0}}}}},
                        {"output", {{"verbosity", 0}}}});
  }

  struct Setup {
    ConfigTree config = make_config();
    Advection model;
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
    Discretization discretization{mesh, config};
    Assembler assembler{discretization, model, config};
    VectorType u;

    /// Refine the cells in [0.3, 0.55] once and those in [0.4, 0.45] twice: level jumps in both directions.
    void refine_locally()
    {
      auto &tria = discretization.get_triangulation();
      for (const auto [lower, upper] : {std::pair{0.3, 0.55}, std::pair{0.4, 0.45}}) {
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->center()[0] > lower && cell->center()[0] < upper) cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }
      discretization.reinit();
      assembler.reinit();
    }

    void interpolate()
    {
      FV::FlowingVariables<Discretization> state(discretization);
      state.interpolate(model);
      u = state.spatial_data();
    }

    VectorType residual(const VectorType &v) const
    {
      VectorType r(v.size()), v_dot(v.size());
      const_cast<Assembler &>(assembler).residual(r, v, 1., v_dot, 0.);
      return r;
    }
  };
} // namespace

static_assert(Assembler::supports_mesh_adaptation);
static_assert(!FV::KurganovTadmor::Assembler<FV::Discretization<ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>,
                                                                RectangularMeshSerial<2>, double>,
                                             Advection>::supports_mesh_adaptation,
              "2D KT must refuse HAdaptivity at compile time");

TEST_CASE("KT on a 1D mesh with level jumps", "[FV][KT][adaptivity]")
{
  DiFfRG::Init();
  Setup s;
  s.refine_locally();
  std::set<uint> levels;
  for (const auto &cell : s.discretization.get_triangulation().active_cell_iterators())
    levels.insert(cell->level());
  REQUIRE(levels.size() == 3);
  s.interpolate();
  const auto r = s.residual(s.u);

  SECTION("Conservative: the residual sums to the net boundary flux")
  {
    double total = 0.;
    for (const auto v : r)
      total += v;
    // Constant ghosts: the boundary fluxes are c times the first and last cell averages.
    double u_first = 0., u_last = 0., x_first = 2., x_last = -1.;
    for (const auto &cell : s.discretization.get_dof_handler().active_cell_iterators()) {
      std::vector<types::global_dof_index> dof(1);
      cell->get_dof_indices(dof);
      if (cell->center()[0] < x_first) x_first = cell->center()[0], u_first = s.u[dof[0]];
      if (cell->center()[0] > x_last) x_last = cell->center()[0], u_last = s.u[dof[0]];
    }
    CHECK(total == Catch::Approx(Advection::c * (u_last - u_first)).margin(1e-12));
  }

  SECTION("Consistent: refining locally does not make residual / |K| a worse approximation of c u'")
  {
    // Max error over the cells away from the boundary, where the constant ghosts make the reconstruction first order.
    const auto worst_error = [](const Setup &setup, const VectorType &residual) {
      double worst = 0.;
      for (const auto &cell : setup.discretization.get_dof_handler().active_cell_iterators()) {
        if (cell->center()[0] < 0.1 || cell->center()[0] > 0.9) continue;
        std::vector<types::global_dof_index> dof(1);
        cell->get_dof_indices(dof);
        const double expected = Advection::c * 3. * std::cos(3. * cell->center()[0]);
        worst = std::max(worst, std::abs(residual[dof[0]] / cell->measure() - expected));
      }
      return worst;
    };
    Setup uniform;
    uniform.interpolate();
    const double worst_uniform = worst_error(uniform, uniform.residual(uniform.u));
    const double worst_adapted = worst_error(s, r);
    CAPTURE(worst_uniform, worst_adapted);
    CHECK(worst_adapted <= worst_uniform);
  }

  SECTION("Jacobian matches finite differences")
  {
    SparseMatrix<double> J(s.assembler.get_sparsity_pattern_jacobian());
    VectorType u_dot(s.u.size());
    s.assembler.jacobian(J, s.u, 1., u_dot, 0., 0.);
    double worst = 0., scale = 0.;
    for (size_t j = 0; j < s.u.size(); ++j) {
      auto up = s.u, down = s.u;
      up[j] += 1e-6;
      down[j] -= 1e-6;
      auto column = s.residual(up);
      column -= s.residual(down);
      column /= 2e-6;
      for (size_t i = 0; i < s.u.size(); ++i) {
        worst = std::max(worst, std::abs(column[i] - J.el(i, j)));
        scale = std::max(scale, std::abs(column[i]));
      }
    }
    REQUIRE(scale > 0.);
    CHECK(worst <= 1e-6 * scale);
  }
}

TEST_CASE("HAdaptivity refines a 1D KT mesh where the indicator is large", "[FV][KT][adaptivity]")
{
  DiFfRG::Init();
  Setup s;
  s.interpolate();
  HAdaptivity<Assembler> adaptor(s.assembler, s.config);
  const auto n_before = s.discretization.get_triangulation().n_active_cells();
  REQUIRE(adaptor.adapt(s.u));
  REQUIRE(s.discretization.get_triangulation().n_active_cells() > n_before);
  REQUIRE(s.u.size() == s.discretization.get_dof_handler().n_dofs());

  // |u'| = 3 |cos 3x| is largest near x = 0 and x = pi/3 ~ 1.05: refined cells sit there, not around x = pi/6.
  for (const auto &cell : s.discretization.get_triangulation().active_cell_iterators())
    if (cell->level() > 0) CHECK(std::abs(cell->center()[0] - M_PI / 6.) > 0.15);

  // Transfer of cell averages: a refined cell's children inherit its average.
  double total_before = 0., total_after = 0.;
  {
    Setup fresh;
    fresh.interpolate();
    for (const auto &cell : fresh.discretization.get_dof_handler().active_cell_iterators()) {
      std::vector<types::global_dof_index> dof(1);
      cell->get_dof_indices(dof);
      total_before += fresh.u[dof[0]] * cell->measure();
    }
  }
  for (const auto &cell : s.discretization.get_dof_handler().active_cell_iterators()) {
    std::vector<types::global_dof_index> dof(1);
    cell->get_dof_indices(dof);
    total_after += s.u[dof[0]] * cell->measure();
  }
  CHECK(total_after == Catch::Approx(total_before).epsilon(1e-12));

  // And the assembler works on the adapted mesh.
  const auto r = s.residual(s.u);
  for (const auto v : r)
    REQUIRE(std::isfinite(v));
}

TEST_CASE("HAdaptivity prolongates 1D KT cell averages with their slope", "[FV][KT][adaptivity]")
{
  DiFfRG::Init();
  // The exact average over [a, b] of q(x) = c0 + c1 x + c2 x^2.
  const double c0 = 0.3, c1 = 2.;
  const auto average = [&](const double c2, const double a, const double b) {
    return c0 + c1 * (a + b) / 2. + c2 * (a * a + a * b + b * b) / 3.;
  };
  const auto check = [&](const double c2, const bool interior_only) {
    Setup s;
    s.interpolate();
    std::vector<types::global_dof_index> dof(1);
    for (const auto &cell : s.discretization.get_dof_handler().active_cell_iterators()) {
      cell->get_dof_indices(dof);
      s.u[dof[0]] = average(c2, cell->vertex(0)[0], cell->vertex(1)[0]);
    }
    HAdaptivity<Assembler> adaptor(s.assembler, s.config);
    REQUIRE(adaptor.adapt(s.u));
    uint refined = 0;
    for (const auto &cell : s.discretization.get_dof_handler().active_cell_iterators()) {
      if (cell->level() == 0) continue;
      const auto parent = cell->parent();
      if (interior_only && (parent->at_boundary(0) || parent->at_boundary(1))) continue;
      cell->get_dof_indices(dof);
      ++refined;
      CHECK(s.u[dof[0]] == Catch::Approx(average(c2, cell->vertex(0)[0], cell->vertex(1)[0])).epsilon(1e-12));
    }
    CHECK(refined > 0);
  };
  // Linear data: exact everywhere, also next to the boundary (one-sided slope there).
  SECTION("linear") { check(0., false); }
  // Monotone quadratic data: exact child averages away from the boundary.
  SECTION("quadratic") { check(-0.5, true); }
}

TEST_CASE("HAdaptivity prolongates 1D KT cell averages without new extrema", "[FV][KT][adaptivity]")
{
  DiFfRG::Init();
  Setup s;
  s.interpolate();
  // A step: the slope of the cells next to it must not push their children beyond the values on either side.
  std::vector<types::global_dof_index> dof(1);
  for (const auto &cell : s.discretization.get_dof_handler().active_cell_iterators()) {
    cell->get_dof_indices(dof);
    s.u[dof[0]] = cell->center()[0] < 0.5 ? 1. : 2.;
  }
  HAdaptivity<Assembler> adaptor(s.assembler, s.config);
  REQUIRE(adaptor.adapt(s.u));
  CHECK(s.u.linfty_norm() <= 2. + 1e-12);
  for (std::size_t i = 0; i < s.u.size(); ++i)
    CHECK(s.u[i] >= 1. - 1e-12);
}

namespace
{
  /// Cell averages of @p f over the cells of @p s, by Gauss quadrature.
  template <typename F> VectorType cell_averages(Setup &s, const F &f)
  {
    VectorType v(s.discretization.get_dof_handler().n_dofs());
    std::vector<types::global_dof_index> dof(1);
    const QGauss<1> quadrature(8);
    for (const auto &cell : s.discretization.get_dof_handler().active_cell_iterators()) {
      cell->get_dof_indices(dof);
      const double a = cell->vertex(0)[0], b = cell->vertex(1)[0];
      double sum = 0.;
      for (uint q = 0; q < quadrature.size(); ++q)
        sum += quadrature.weight(q) * f(a + (b - a) * quadrature.point(q)[0]);
      v[dof[0]] = sum;
    }
    return v;
  }
} // namespace

TEST_CASE("KT prolongation slopes keep smooth extrema and limit kinks and steps", "[FV][KT][adaptivity]")
{
  DiFfRG::Init();
  Setup s;
  const double x0 = 0.512; // inside the cell [0.5, 0.55], off its centre
  std::vector<types::global_dof_index> dof(1);
  // The children's averages ubar -+ slope * width / 4 of every cell, with the parent's average.
  const auto children = [&](const VectorType &u, const VectorType &slope_source) {
    const auto slopes = s.assembler.prolongation_slopes(std::vector<VectorType>{u, slope_source});
    std::vector<std::tuple<double, double, double, double, double>> result; // a, b, left, right, parent
    for (const auto &cell : s.discretization.get_dof_handler().active_cell_iterators()) {
      cell->get_dof_indices(dof);
      const double a = cell->vertex(0)[0], b = cell->vertex(1)[0];
      const double shift = slopes[1][cell->active_cell_index()][0][0] * (b - a) / 4.;
      result.emplace_back(a, b, slope_source[dof[0]] - shift, slope_source[dof[0]] + shift, slope_source[dof[0]]);
    }
    return result;
  };

  SECTION("a smooth minimum is prolongated exactly")
  {
    const auto f = [&](const double x) { return 0.3 + std::pow(x - x0, 2); };
    const auto u = cell_averages(s, f);
    for (const auto &[a, b, left, right, parent] : children(u, u)) {
      if (a < 1e-12 || b > 1. - 1e-12) continue; // one-sided slope at the boundary
      const auto half = [&](const double lo, const double hi) {
        // exact average of 0.3 + (x - x0)^2 over [lo, hi]
        return 0.3 + (std::pow(hi - x0, 3) - std::pow(lo - x0, 3)) / (3. * (hi - lo));
      };
      CHECK(left == Catch::Approx(half(a, (a + b) / 2.)).epsilon(1e-12));
      CHECK(right == Catch::Approx(half((a + b) / 2., b)).epsilon(1e-12));
    }
  }

  SECTION("a kink's minimum dips no further than the profile")
  {
    // |x - x0| curves the same way on both sides of the kink cell, so its children may dip below the averages,
    // by the smallest of the three curvatures -- toward the exact child averages, never beyond them.
    const auto f = [&](const double x) { return std::abs(x - x0); };
    const auto u = cell_averages(s, f);
    for (const auto &[a, b, left, right, parent] : children(u, u)) {
      CHECK(left >= 0.);
      CHECK(right >= 0.);
      if (a < x0 && x0 < b) {
        const double m = (a + b) / 2.;
        const double exact_left = (std::pow(x0 - a, 2) + std::pow(m - x0, 2)) / (2. * (m - a));
        CHECK(left < parent);
        CHECK(left > exact_left);
      }
    }
  }

  SECTION("every vector is prolongated with the limiting of the first")
  {
    // u is a step at x = 0.5: limited to zero slope in the two cells next to it, unlimited elsewhere (zero slope).
    const auto u = cell_averages(s, [](const double x) { return x < 0.5 ? 1. : 2.; });
    const auto v = cell_averages(s, [](const double x) { return 3. * x; });
    for (const auto &[a, b, left, right, parent] : children(u, v)) {
      const bool next_to_step = std::abs(a - 0.5) < 1e-12 || std::abs(b - 0.5) < 1e-12;
      if (next_to_step) {
        CHECK(left == Catch::Approx(parent).epsilon(1e-14));
        CHECK(right == Catch::Approx(parent).epsilon(1e-14));
      } else if (a > 1e-12 && b < 1. - 1e-12) {
        CHECK(left == Catch::Approx(3. * (3. * a + b) / 4.).epsilon(1e-12));
        CHECK(right == Catch::Approx(3. * (a + 3. * b) / 4.).epsilon(1e-12));
      }
    }
  }
}
