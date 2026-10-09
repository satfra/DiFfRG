#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>
#include <boilerplate/kt_models.hh>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tbb/global_control.h>

#include <cmath>
#include <optional>
#include <string>

using namespace DiFfRG;
using namespace dealii;

namespace
{
  /**
   * Two components with a nonlinear, gradient-dependent advection flux, a diffusion flux that also reads the
   * third derivatives, and a gradient-dependent source. With `batched`, the model evaluates all three itself in
   * evaluate_batch from the batch columns; otherwise it uses the default, which calls flux, diffusion_flux and
   * source per point.
   */
  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>>;

  template <uint dim, bool batched>
  class Model : public def::AbstractModel<Model<dim, batched>, Components>,
                public def::Time,
                public def::LLFFlux<Model<dim, batched>>,
                public def::FlowBoundaries<Model<dim, batched>>,
                public def::AD<Model<dim, batched>>
  {
  public:
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
      for (uint d = 0; d < dim; ++d) {
        D[0][d] = -(0.3 + 0.1 * u[1] * u[1]) * du[0][d] + 0.01 * third(0, d);
        D[1][d] = -0.2 * (1. + u[0]) * du[1][d] + 0.02 * u[0] * du[0][d] + 0.01 * third(1, d);
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
  template <uint dim, bool batched> struct Setup {
    using M = Model<dim, batched>;
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
