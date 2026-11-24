#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/backend/dealii.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/FV/mesh.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/first_order_reconstructor.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>
#include <boilerplate/kt_models.hh>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <type_traits>

using namespace DiFfRG;
using namespace dealii;

namespace
{
  /**
   * Two components with a nonlinear, gradient-dependent advection flux, a diffusion flux that also reads the
   * third derivatives, and a gradient-dependent source. With `batched`, the model evaluates all three itself in
   * evaluate_batch from the batch columns; otherwise it uses the default, which calls flux, diffusion_flux and
   * source per point.
   *
   * With Diffusion::zero the diffusion flux vanishes; Diffusion::absent also declares that through
   * `use_diffusion_flux = false`, so the assembler skips it.
   */
  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>>;

  enum class Diffusion { on, zero, absent };

  template <uint dim, bool batched, Diffusion diffusion_mode = Diffusion::on>
  class Model : public def::AbstractModel<Model<dim, batched, diffusion_mode>, Components>,
                public def::Time,
                public def::LLFFlux<Model<dim, batched, diffusion_mode>>,
                public def::FlowBoundaries<Model<dim, batched, diffusion_mode>>,
                public def::AD<Model<dim, batched, diffusion_mode>>
  {
  public:
    static constexpr bool use_diffusion_flux = diffusion_mode != Diffusion::absent;

    static std::array<double, 2> exact(const Point<dim> &x)
    {
      const double y = dim > 1 ? x[dim - 1] : 0.;
      return {{1. + 0.3 * std::sin(2. * x[0]) + 0.4 * x[0] * x[0] * x[0] + 0.2 * y * y * x[0],
               0.5 + 0.4 * std::cos(3. * x[0]) - 0.3 * y + 0.25 * y * y * y}};
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const auto u = exact(x);
      values[0] = u[0];
      values[1] = u[1];
    }

    // The physics, shared by the per-point and the batched path.
    template <typename NT>
    static void advection(std::array<Tensor<1, dim, NT>, 2> &F, const Point<dim> &x, const std::array<NT, 2> &u,
                          const std::array<Tensor<1, dim, NT>, 2> &du)
    {
      for (uint d = 0; d < dim; ++d) {
        F[0][d] = 0.5 * u[0] * u[0] + 0.2 * u[0] * u[1] - 0.1 * u[0] * du[0][d] + 0.05 * x[d];
        F[1][d] = 0.3 * u[1] * u[1] * u[0] - 0.05 * u[1] * du[1][d] * (d + 1.);
      }
    }
    template <typename NT, typename Third>
    static void diffusion(std::array<Tensor<1, dim, NT>, 2> &D, const std::array<NT, 2> &u,
                          const std::array<Tensor<1, dim, NT>, 2> &du, const Third &third)
    {
      const double scale = diffusion_mode == Diffusion::on ? 1. : 0.;
      for (uint d = 0; d < dim; ++d) {
        D[0][d] = scale * (-(0.3 + 0.1 * u[1] * u[1]) * du[0][d] + 0.01 * third(0, d));
        D[1][d] = scale * (-0.2 * (1. + u[0]) * du[1][d] + 0.02 * u[0] * du[0][d] + 0.01 * third(1, d));
      }
    }
    template <typename NT>
    static void source_term(std::array<NT, 2> &S, const Point<dim> &x, const std::array<NT, 2> &u,
                            const std::array<Tensor<1, dim, NT>, 2> &du)
    {
      using std::sin;
      S[0] = sin(u[0]) * u[1] + 0.1 * du[0][0] * du[1][dim - 1] + x[0];
      S[1] = u[0] * u[0] * u[1] - 0.2 * du[1][0] * u[0];
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, 2> &F, const Point<dim> &x, const Solution &sol) const
    {
      advection(F, x, as_array<NT>(get<"fe_functions">(sol)), as_gradients<NT>(get<"fe_derivatives">(sol)));
    }
    template <typename NT, typename Solution>
    void diffusion_flux(std::array<Tensor<1, dim, NT>, 2> &D, const Point<dim> &, const Solution &sol) const
    {
      const auto &t = get<"fe_third_derivatives">(sol);
      diffusion(D, as_array<NT>(get<"fe_functions">(sol)), as_gradients<NT>(get<"fe_derivatives">(sol)),
                [&](uint c, uint d) { return NT(t[c][d][d][d]); });
    }
    template <typename NT, typename Solution>
    void source(std::array<NT, 2> &S, const Point<dim> &x, const Solution &sol) const
    {
      source_term(S, x, as_array<NT>(get<"fe_functions">(sol)), as_gradients<NT>(get<"fe_derivatives">(sol)));
    }

    template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
    {
      if constexpr (!batched) {
        def::AbstractModel<Model, Components>::evaluate_batch(out, batch);
        return;
      }
      using NT = typename Batch::number_type;
      for (size_t i = 0; i < batch.size(); ++i) {
        std::array<NT, 2> u{{batch.values(0)[i], batch.values(1)[i]}};
        std::array<Tensor<1, dim, NT>, 2> du;
        Point<dim> x;
        for (uint d = 0; d < dim; ++d) {
          x[d] = batch.coordinates(d)[i];
          for (uint c = 0; c < 2; ++c)
            du[c][d] = batch.derivatives(c, d)[i];
        }
        if (out.requested(Term::flux)) {
          std::array<Tensor<1, dim, NT>, 2> F;
          advection(F, x, u, du);
          out.store_flux(i, F);
        }
        if constexpr (Batch::with_third)
          if (out.requested(Term::diffusion_flux)) {
            std::array<Tensor<1, dim, NT>, 2> D;
            diffusion(D, u, du, [&](uint c, uint d) { return batch.third_derivatives(c, d, d, d)[i]; });
            out.store_diffusion_flux(i, D);
          }
        if (out.requested(Term::source)) {
          std::array<NT, 2> S;
          source_term(S, x, u, du);
          out.store_source(i, S);
        }
      }
    }

    template <int mdim, typename NT, size_t n_components>
    bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                                def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
    {
      if constexpr (mdim == 1) {
        Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face, exact);
        return true;
      }
      for (size_t i = 0; i < x_stencil.size(); ++i) {
        const auto u = exact(x_stencil[i]);
        for (size_t c = 0; c < n_components; ++c)
          u_stencil[i][c] = NT(u[c]);
      }
      return true;
    }

  private:
    template <typename NT, typename V> static std::array<NT, 2> as_array(const V &v) { return {{NT(v[0]), NT(v[1])}}; }
    template <typename NT, typename V> static std::array<Tensor<1, dim, NT>, 2> as_gradients(const V &v)
    {
      std::array<Tensor<1, dim, NT>, 2> r;
      for (uint c = 0; c < 2; ++c)
        for (uint d = 0; d < dim; ++d)
          r[c][d] = NT(v[c][d]);
      return r;
    }
  };

  ConfigTree make_config(const uint dim, const std::string &grid, const uint max_stacked_points)
  {
    try {
      spdlog::stdout_color_mt("log")->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
    const std::string y_grid = dim > 1 ? grid : "0:1:1";
    json::value batched = json::object();
    if (max_stacked_points > 0) batched = {{"max_stacked_points", max_stacked_points}};
    return json::value({{"physical", {{"Lambda", 1.}}},
                        {"discretization",
                         {{"fe_order", 0},
                          {"overintegration", 0},
                          {"EoM_abs_tol", 1e-10},
                          {"EoM_max_iter", 0},
                          {"batched", batched},
                          {"grid", {{"x_grid", grid}, {"y_grid", y_grid}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
                        {"output", {{"verbosity", 0}}}});
  }

  /// One KT assembler on its own mesh, with a nonsmooth state (so the limiter is active somewhere).
  template <uint dim, bool batched, Diffusion diffusion_mode = Diffusion::on> struct Setup {
    using M = Model<dim, batched, diffusion_mode>;
    // Serial on purpose: this tests serial linear algebra, and a plain RectangularMesh is partitioned in an MPI build.
    using Discretization = FV::Discretization<M, RectangularMeshSerial<dim>, double>;
    using Assembler = FV::KurganovTadmor::Assembler<Discretization, M>;
    using VectorType = typename Discretization::VectorType;

    ConfigTree config;
    M model;
    RectangularMeshSerial<dim> mesh;
    Discretization discretization;
    Assembler assembler;
    VectorType u, u_dot;
    SparseMatrix<double> J;

    Setup(const std::string &grid, const uint max_stacked_points = 0)
        : config(make_config(dim, grid, max_stacked_points)), mesh(Config::ConfigurationMesh<dim>(config)),
          discretization(mesh, config), assembler(discretization, model, config)
    {
      FV::FlowingVariables<Discretization> state(discretization);
      state.interpolate(model);
      u = state.spatial_data();
      for (size_t i = 0; i < u.size(); ++i)
        u[i] += 0.05 * std::sin(1.7 * double(i)) + (i % 7 == 3 ? 0.1 : 0.);
      u_dot.reinit(u.size());
      J.reinit(assembler.get_sparsity_pattern_jacobian());
    }

    VectorType residual()
    {
      VectorType r(u.size());
      assembler.residual(r, u, 1., u_dot, 0.);
      return r;
    }
    const SparseMatrix<double> &jacobian()
    {
      J = 0.;
      assembler.jacobian(J, u, 1., u_dot, 0., 0.);
      return J;
    }
  };

  double max_abs(const Vector<double> &v) { return v.linfty_norm(); }

  /// The largest entry deviation of b from a, and the largest entry of a.
  std::pair<double, double> matrix_deviation(const SparseMatrix<double> &a, const SparseMatrix<double> &b)
  {
    double worst = 0., scale = 0.;
    for (auto it = a.begin(); it != a.end(); ++it) {
      worst = std::max(worst, std::abs(it->value() - b.el(it->row(), it->column())));
      scale = std::max(scale, std::abs(it->value()));
    }
    return {worst, scale};
  }

  template <typename A, typename B> void require_same_assembly(A &a, B &b, const double tol)
  {
    const auto r_a = a.residual(), r_b = b.residual();
    auto deviation = r_a;
    deviation -= r_b;
    REQUIRE(max_abs(r_b) > 0.);
    REQUIRE(max_abs(deviation) <= tol * max_abs(r_b));
    const auto [worst, scale] = matrix_deviation(b.jacobian(), a.jacobian());
    REQUIRE(scale > 0.);
    REQUIRE(worst <= tol * scale);
  }
} // namespace

TEST_CASE("A model's own evaluate_batch matches the per-point default under KT", "[FV][KT][batch]")
{
  DiFfRG::Init();
  SECTION("1D")
  {
    Setup<1, true> batched("0:0.05:1");
    Setup<1, false> per_point("0:0.05:1");
    require_same_assembly(batched, per_point, 1e-12);
  }
  SECTION("2D")
  {
    Setup<2, true> batched("0:0.125:1");
    Setup<2, false> per_point("0:0.125:1");
    require_same_assembly(batched, per_point, 1e-12);
  }
}

// Grouping and thread count must not change the result beyond rounding. Not bitwise: under -ffast-math the
// compiler handles a loop's vector body and its remainder differently (e.g. vectorised libm calls), and both
// knobs move points between the two, which shows up as a few ulps on some compilers and -march targets.
constexpr double rounding_tol = 1e-14;

// A bound far below one stack splits every AD evaluation into many groups, each a separate call.
TEST_CASE("The KT stacking bound does not change the assembly", "[FV][KT][batch]")
{
  DiFfRG::Init();
  Setup<2, false> grouped("0:0.125:1", 7);
  Setup<2, false> whole("0:0.125:1");
  require_same_assembly(grouped, whole, rounding_tol);
}

TEST_CASE("FV backend rebuilds geometry and real or ghost face connections", "[FV][geometry]")
{
  DiFfRG::Init();
  const auto config = make_config(2, "0:0.25:1", 0);
  RectangularMeshSerial<2> mesh{Config::ConfigurationMesh<2>(config)};
  using Discretization = FV::Discretization<Components, RectangularMeshSerial<2>, double>;
  Discretization discretization(mesh, config);
  FV::DealIIBackend<Discretization> backend(discretization);
  FV::KurganovTadmor::Mesh<2, double, 2> topology;

  static_assert(!std::is_copy_constructible_v<FV::KurganovTadmor::Mesh<2, double, 2>>);
  static_assert(!std::is_move_constructible_v<FV::KurganovTadmor::Mesh<2, double, 2>>);
  const auto check_geometry = [&](const unsigned int n, const double dx, const double dy) {
    backend.populate(topology, QGauss<2>(1));
    REQUIRE(topology.cells().size() == n * n);
    REQUIRE(topology.ghost_cells().size() == 4 * n);
    REQUIRE(topology.faces().size() == 2 * n * (n + 1));
    std::set<const FV::KurganovTadmor::Mesh<2, double, 2>::Cell *> real_cells;
    std::set<const FV::KurganovTadmor::Mesh<2, double, 2>::Cell *> all_cells;
    for (const auto &cell : topology.cells()) {
      REQUIRE_FALSE(cell.is_boundary_ghost);
      real_cells.insert(&cell);
      all_cells.insert(&cell);
    }
    for (const auto &ghost : topology.ghost_cells()) {
      REQUIRE(ghost.is_boundary_ghost);
      all_cells.insert(&ghost);
    }
    std::map<const FV::KurganovTadmor::Mesh<2, double, 2>::Face *, unsigned int> incidences;
    for (const auto &face : topology.faces())
      incidences.emplace(&face, 0);
    std::vector<bool> occupied_cache_slots(topology.cells().size() + topology.ghost_cells().size(), false);
    for (const auto storage : {topology.cells(), topology.ghost_cells()}) {
      for (const auto &cell : storage) {
        REQUIRE(cell.index < occupied_cache_slots.size());
        REQUIRE_FALSE(occupied_cache_slots[cell.index]);
        occupied_cache_slots[cell.index] = true;
        REQUIRE((cell.index >= topology.cells().size()) == cell.is_boundary_ghost);
        REQUIRE(cell.faces.size() == (cell.is_boundary_ghost ? 1 : 4));
        REQUIRE(cell.neighbors.size() == cell.faces.size());
        for (unsigned int f = 0; f < cell.faces.size(); ++f) {
          const auto &face = cell.faces[f].get();
          REQUIRE(incidences.contains(&face));
          ++incidences.at(&face);
          const bool is_minus = &cell == &face.minus;
          REQUIRE((is_minus || &cell == &face.plus));
          const auto &neighbor = cell.neighbors[f].get();
          REQUIRE(all_cells.contains(&neighbor));
          REQUIRE(&neighbor == &(is_minus ? face.plus : face.minus));
          REQUIRE(std::any_of(neighbor.neighbors.begin(), neighbor.neighbors.end(),
                              [&](const auto &entry) { return &entry.get() == &cell; }));
          const auto outward_normal = is_minus ? face.normal : -face.normal;
          const auto offset = face.center - cell.x;
          REQUIRE((outward_normal - offset / offset.norm()).norm() < 1e-14);
          if (cell.is_boundary_ghost) continue;
          const unsigned int axis = f / 2;
          const double sign = f % 2 == 0 ? -1. : 1.;
          for (unsigned int d = 0; d < 2; ++d)
            REQUIRE(std::abs(outward_normal[d] - (d == axis ? sign : 0.)) < 1e-14);
          auto expected_center = cell.x;
          expected_center[axis] += sign * 0.5 * (axis == 0 ? dx : dy);
          REQUIRE(face.center.distance(expected_center) < 1e-14);
          REQUIRE(std::abs(face.jxw - (axis == 0 ? dy : dx)) < 1e-14);
        }
      }
    }

    unsigned int n_ghost_faces = 0;
    for (const auto &face : topology.faces()) {
      REQUIRE(incidences.at(&face) == 2);
      REQUIRE(real_cells.contains(&face.minus));
      REQUIRE(all_cells.contains(&face.plus));
      REQUIRE(face.minus_face_index < face.minus.faces.size());
      REQUIRE(face.plus_face_index < face.plus.faces.size());
      REQUIRE(&face.minus.faces[face.minus_face_index].get() == &face);
      REQUIRE(&face.plus.faces[face.plus_face_index].get() == &face);
      const auto references_face = [&](const auto &entry) { return &entry.get() == &face; };
      REQUIRE(std::count_if(face.minus.faces.begin(), face.minus.faces.end(), references_face) == 1);
      REQUIRE(std::count_if(face.plus.faces.begin(), face.plus.faces.end(), references_face) == 1);
      REQUIRE_FALSE(face.minus.is_boundary_ghost);
      // A single normal points across the interface from minus to plus, even at a lower boundary.
      const auto width = std::abs(face.normal[0]) * dx + std::abs(face.normal[1]) * dy;
      const auto expected_plus = face.minus.x + width * face.normal;
      REQUIRE(face.plus.x.distance(expected_plus) < 1e-14);
      if (face.plus.is_boundary_ghost) {
        ++n_ghost_faces;
        REQUIRE_FALSE(real_cells.contains(&face.plus));
        const auto midpoint = face.minus.x + 0.5 * (face.plus.x - face.minus.x);
        REQUIRE(midpoint.distance(face.center) < 1e-14);
      } else {
        REQUIRE(real_cells.contains(&face.plus));
      }
    }
    REQUIRE(n_ghost_faces == 4 * n);
  };

  check_geometry(4, 0.25, 0.25);
  GridTools::transform([](const Point<2> &p) { return Point<2>(2. * p[0], 3. * p[1]); },
                       discretization.get_triangulation());
  check_geometry(4, 0.5, 0.75);
  discretization.get_triangulation().refine_global(1);
  discretization.reinit();
  check_geometry(8, 0.25, 0.375);
}

TEST_CASE("FV backend imports cell DoFs boundary IDs and quadrature", "[FV][backend][quadrature]")
{
  DiFfRG::Init();
  const auto config = make_config(2, "0:0.25:1", 0);
  RectangularMeshSerial<2> mesh{Config::ConfigurationMesh<2>(config)};
  using Discretization = FV::Discretization<Components, RectangularMeshSerial<2>, double>;
  Discretization discretization(mesh, config);
  GridTools::transform([](const Point<2> &p) { return Point<2>(2. * p[0], 3. * p[1]); },
                       discretization.get_triangulation());
  for (const auto &cell : discretization.get_dof_handler().active_cell_iterators())
    for (const auto f : cell->face_indices())
      if (cell->at_boundary(f)) cell->face(f)->set_boundary_id(10 + f);
  FV::DealIIBackend<Discretization> backend(discretization);

  const auto check_import = [&](const unsigned int n_cells, const double dx, const double dy,
                                const unsigned int order) {
    FV::KurganovTadmor::Mesh<2, double, 2> imported;
    backend.populate(imported, QGauss<2>(order));
    const auto data = imported.cells();
    REQUIRE(data.size() == n_cells);
    std::vector<types::global_dof_index> expected_dofs(2);
    for (const auto &cell : discretization.get_dof_handler().active_cell_iterators()) {
      const auto &entry = data[cell->active_cell_index()];
      cell->get_dof_indices(expected_dofs);
      for (unsigned int c = 0; c < expected_dofs.size(); ++c)
        REQUIRE(entry.dof_indices[c] == expected_dofs[c]);
      for (const auto f : cell->face_indices()) {
        const auto expected = cell->at_boundary(f) ? cell->face(f)->boundary_id() : numbers::invalid_boundary_id;
        REQUIRE(entry.boundary_id(f) == expected);
      }
      REQUIRE(entry.quadrature_points.size() == order * order);
      REQUIRE(entry.jxw.size() == entry.quadrature_points.size());
      double area = 0., first_x = 0., first_y = 0., second_x = 0., second_y = 0.;
      for (unsigned int q = 0; q < entry.jxw.size(); ++q) {
        const auto offset = entry.quadrature_points[q] - cell->center();
        const auto weight = entry.jxw[q];
        REQUIRE(weight > 0.);
        REQUIRE(std::abs(offset[0]) <= 0.5 * dx + 1e-14);
        REQUIRE(std::abs(offset[1]) <= 0.5 * dy + 1e-14);
        area += weight;
        first_x += weight * offset[0];
        first_y += weight * offset[1];
        second_x += weight * offset[0] * offset[0];
        second_y += weight * offset[1] * offset[1];
      }
      REQUIRE(std::abs(area - dx * dy) < 1e-14);
      REQUIRE(std::abs(first_x) < 1e-14);
      REQUIRE(std::abs(first_y) < 1e-14);
      if (order > 1) {
        REQUIRE(std::abs(second_x - dx * dy * dx * dx / 12.) < 1e-14);
        REQUIRE(std::abs(second_y - dx * dy * dy * dy / 12.) < 1e-14);
      }
    }
  };
  check_import(16, 0.5, 0.75, 1);
  check_import(16, 0.5, 0.75, 3);
  discretization.get_triangulation().refine_global(1);
  discretization.reinit();
  check_import(64, 0.25, 0.375, 3);
}

TEST_CASE("FV assembly context exposes the actual mesh entities", "[FV][KT][hook][geometry]")
{
  DiFfRG::Init();
  const auto config = make_config(2, "0:0.25:1", 0);
  RectangularMeshSerial<2> grid{Config::ConfigurationMesh<2>(config)};
  using Discretization = FV::Discretization<Components, RectangularMeshSerial<2>, double>;
  Discretization discretization(grid, config);
  FV::DealIIBackend<Discretization> backend(discretization);
  using Mesh = FV::KurganovTadmor::Mesh<2, double, 2>;
  static_assert(!std::is_copy_constructible_v<Mesh::Cell>);
  static_assert(!std::is_copy_constructible_v<Mesh::Face>);
  static_assert(std::is_move_constructible_v<Mesh::Cell>);
  static_assert(std::is_move_constructible_v<Mesh::Face>);
  Mesh mesh;
  backend.populate(mesh, QGauss<2>(1));
  const FV::KurganovTadmor::AssemblyContextView<Mesh> context(mesh);
  REQUIRE(context.cells().size() == 16);
  REQUIRE(context.faces().size() == 40);
  for (unsigned int i = 0; i < mesh.cells().size(); ++i) {
    mesh.cells()[i].state.u[0] = 0.5 + i;
    REQUIRE(&context.cells()[i] == &mesh.cells()[i]);
    REQUIRE(&context.cells()[i].values() == &mesh.cells()[i].state.u);
    REQUIRE(context.cells()[i].values()[0] == 0.5 + i);
  }
  for (unsigned int i = 0; i < mesh.faces().size(); ++i) {
    const auto &face = context.faces()[i];
    REQUIRE(&face == &mesh.faces()[i]);
    REQUIRE(&face == &face.minus.faces[face.minus_face_index].get());
    REQUIRE(&face == &face.plus.faces[face.plus_face_index].get());
    REQUIRE(&face.minus == &mesh.cells()[face.minus.index]);
    REQUIRE(face.at_boundary() == face.plus.is_boundary_ghost);
  }
  mesh.faces()[0].state.reconstruction.u_minus[0] = 42.;
  REQUIRE(context.faces()[0].state.reconstruction.u_minus[0] == 42.);
  discretization.get_triangulation().refine_global(1);
  discretization.reinit();
  backend.populate(mesh, QGauss<2>(1));
  const FV::KurganovTadmor::AssemblyContextView<Mesh> rebuilt(mesh);
  REQUIRE(rebuilt.faces().size() == 144);
  REQUIRE(rebuilt.cells().size() == 64);
  for (const auto &face : rebuilt.faces()) {
    REQUIRE_FALSE(face.state.reconstruction_valid);
    REQUIRE(&face == &mesh.face(face.minus.index, face.minus_face_index));
    REQUIRE(&face == &face.plus.faces[face.plus_face_index].get());
  }
}

TEST_CASE("FV reconstructors read live cell and neighbor data directly from mesh entities",
          "[FV][KT][reconstruction][mesh]")
{
  DiFfRG::Init();
  const auto config = make_config(2, "0:0.25:1", 0);
  RectangularMeshSerial<2> mesh{Config::ConfigurationMesh<2>(config)};
  using Discretization = FV::Discretization<Components, RectangularMeshSerial<2>, double>;
  Discretization discretization(mesh, config);
  GridTools::transform([](const Point<2> &x) { return Point<2>(2. * x[0], 3. * x[1]); },
                       discretization.get_triangulation());
  FV::DealIIBackend<Discretization> backend(discretization);
  FV::KurganovTadmor::Mesh<2, double, 2> topology;
  backend.populate(topology, QGauss<2>(1));
  for (const auto &cell : topology.cells())
    topology.cells()[cell.index].state.u = {1. + 2. * cell.x[0] - 3. * cell.x[1], -2. + 4. * cell.x[0] + cell.x[1]};

  for (const auto &cell : topology.cells()) {
    if (std::any_of(cell.neighbors.begin(), cell.neighbors.end(),
                    [](const auto &n) { return n.get().is_boundary_ghost; }))
      continue;
    const auto &view = cell;
    REQUIRE(&view.center() == &cell.x);
    REQUIRE(&view.values() == &topology.cells()[cell.index].state.u);
    REQUIRE(&view.dofs() == &cell.dof_indices);
    for (unsigned int f = 0; f < cell.faces.size(); ++f) {
      const auto &neighbor = cell.neighbors[f].get();
      REQUIRE(&view.neighbor_center(f) == &neighbor.x);
      REQUIRE(&view.neighbor_values(f) == &neighbor.state.u);
      REQUIRE(&view.face_center(f) == &cell.faces[f].get().center);
    }
    using TVD = def::TVDReconstructor<2, def::MinModLimiter, double>;
    using Diffusion = def::CorrectedWeightedLeastSquaresDiffusionReconstructor<2, double>;
    const auto gradient = TVD::compute_gradient<2>(view);
    const auto diffusion = Diffusion::compute_cell_gradient<2>(view);
    const std::array<std::array<double, 2>, 2> expected{{{2., -3.}, {4., 1.}}};
    for (unsigned int c = 0; c < 2; ++c)
      for (unsigned int d = 0; d < 2; ++d) {
        REQUIRE(std::abs(gradient[c][d] - expected[c][d]) < 1e-12);
        REQUIRE(std::abs(diffusion[c][d] - expected[c][d]) < 1e-12);
      }

    // A reference to the cell must observe a changed neighbor without being rebuilt.
    const auto snapshot = FV::materialize_stencil(view);
    topology.cells()[cell.neighbors[0].get().index].state.u[0] += 2.;
    REQUIRE(std::abs(TVD::compute_gradient<2>(view)[0][0]) < 1e-12);
    REQUIRE(std::abs(TVD::compute_gradient<2>(snapshot)[0][0] - 2.) < 1e-12);
    return;
  }
  FAIL("Expected an interior cell for the cell reconstruction test.");
}

// Thousands of points, so that every phase really runs on many threads. The result must not depend on the
// thread count beyond rounding.
TEST_CASE("KT assembly does not depend on the thread count", "[FV][KT][batch]")
{
  DiFfRG::Init();
  Setup<2, false> s("0:0.015625:1");
  const auto r_all = s.residual();
  SparseMatrix<double> J_all(s.assembler.get_sparsity_pattern_jacobian());
  J_all.copy_from(s.jacobian());
  std::optional<tbb::global_control> serial;
  serial.emplace(tbb::global_control::max_allowed_parallelism, 1);
  const auto r_one = s.residual();
  const auto &J_one = s.jacobian();
  serial.reset();

  auto deviation = r_all;
  deviation -= r_one;
  REQUIRE(max_abs(r_all) > 0.);
  REQUIRE(max_abs(deviation) <= rounding_tol * max_abs(r_all));
  const auto [worst, scale] = matrix_deviation(J_all, J_one);
  REQUIRE(scale > 0.);
  REQUIRE(worst <= rounding_tol * scale);
}

// A model without diffusion flux that says so (use_diffusion_flux = false) skips the diffusion reconstruction and
// jacobian, but must assemble what the same model with a zero diffusion flux assembles.
TEST_CASE("use_diffusion_flux = false matches a zero diffusion flux under KT", "[FV][KT][batch]")
{
  DiFfRG::Init();
  SECTION("1D")
  {
    Setup<1, false, Diffusion::absent> absent("0:0.05:1");
    Setup<1, false, Diffusion::zero> zero("0:0.05:1");
    require_same_assembly(absent, zero, rounding_tol);
  }
  SECTION("2D")
  {
    Setup<2, true, Diffusion::absent> absent("0:0.125:1");
    Setup<2, true, Diffusion::zero> zero("0:0.125:1");
    require_same_assembly(absent, zero, rounding_tol);
  }
}

TEST_CASE("KT refreshes one mesh with independent residual and Jacobian reconstructors", "[FV][KT][mesh]")
{
  DiFfRG::Init();
  Setup<2, false> setup("0:0.25:1");
  using S = Setup<2, false>;
  using TVD = def::TVDReconstructor<2, def::MinModLimiter, double>;
  using FirstOrder = def::FirstOrderReconstructor<2, double>;
  using Assembler = FV::KurganovTadmor::Assembler<S::Discretization, S::M, TVD,
                                                  FV::KurganovTadmor::MaxEigenvalueWaveSpeed, FirstOrder>;
  Assembler assembler(setup.discretization, setup.model, setup.config);
  Vector<double> residual(setup.u.size()), repeated(setup.u.size());
  const auto &mesh = assembler.mesh();
  const auto *cell_address = &mesh.cells()[0];
  const auto *face_address = &mesh.faces()[0];
  assembler.residual(residual, setup.u, 1., setup.u_dot, 0.);
  bool saw_limited_gradient = false;
  for (const auto &face : mesh.faces())
    for (const auto &gradient : face.state.reconstruction.face_grad_minus)
      saw_limited_gradient |= gradient.norm() > 1e-10;
  REQUIRE(saw_limited_gradient);
  SparseMatrix<double> jacobian(assembler.get_sparsity_pattern_jacobian());
  auto changed = setup.u;
  for (unsigned int i = 0; i < changed.size(); ++i)
    changed[i] += 0.2;
  assembler.jacobian(jacobian, changed, 1., setup.u_dot, 0., 0.);
  for (const auto &cell : mesh.cells())
    for (unsigned int c = 0; c < 2; ++c)
      REQUIRE(cell.values()[c] == changed[cell.dofs()[c]]);
  for (const auto &face : mesh.faces()) {
    REQUIRE(face.state.reconstruction_valid);
    for (const auto &gradient : face.state.reconstruction.face_grad_minus)
      REQUIRE(gradient.norm() == 0.);
    for (const auto &gradient : face.state.reconstruction.face_grad_plus)
      REQUIRE(gradient.norm() == 0.);
    REQUIRE(face.state.jacobian.n() == face.state.dependencies.from_dofs.size());
    REQUIRE(&face == &face.minus.faces[face.minus_face_index].get());
    REQUIRE(&face == &face.plus.faces[face.plus_face_index].get());
  }
  assembler.residual(repeated, setup.u, 1., setup.u_dot, 0.);
  repeated -= residual;
  REQUIRE(repeated.linfty_norm() <= rounding_tol * (1. + residual.linfty_norm()));
  REQUIRE(&mesh.cells()[0] == cell_address);
  REQUIRE(&mesh.faces()[0] == face_address);
  for (const auto &ghost : mesh.ghost_cells())
    for (const auto dof : ghost.dofs())
      REQUIRE(dof == numbers::invalid_dof_index);
}
