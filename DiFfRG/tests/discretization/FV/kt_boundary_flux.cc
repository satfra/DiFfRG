#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/first_order_reconstructor.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cmath>

using namespace DiFfRG;
using namespace dealii;

// The KT flux through a face must be upwind for linear advection whichever way the face normal points. The lower
// boundary face is the case where it points in the negative direction: there the ghost is the upwind state for
// c > 0, and the physical cell for c < 0.

namespace
{
  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>;

  /// Linear advection F = c u along axis `axis`; below that axis the ghost cells hold the value `ghost`.
  template <uint dim> class Advection : public def::AbstractModel<Advection<dim>, Components>,
                                        public def::Time,
                                        public def::AD<Advection<dim>>
  {
  public:
    double c = 1., ghost = 0.;
    uint axis = 0;

    template <typename Vector> void initial_condition(const Point<dim> &, Vector &values) const { values[0] = 1.; }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> &, const Solution &sol) const
    {
      F[0] = Tensor<1, dim, NT>();
      F[0][axis] = c * get<"fe_functions">(sol)[0];
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, 1> &S, const Point<dim> &, const Solution &) const
    {
      S[0] = 0.;
    }

    template <int mdim, typename NT, size_t n>
    bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n> &u, def::BoundaryStencilPoints<mdim> &x,
                                const Point<mdim> &x_face) const
    {
      using namespace def::BoundaryStencilIndex;
      // Two ghosts beyond the face, along its normal. Constant extrapolation, except below `axis`.
      const Tensor<1, mdim> to_ghost = 2. * (x_face - x[physical_cell]);
      const bool lower = to_ghost[0] + (mdim > 1 ? to_ghost[mdim - 1] : 0.) < 0.;
      const auto [inner, outer] = lower ? std::pair{lower_inner, lower_outer} : std::pair{upper_inner, upper_outer};
      x[inner] = x[physical_cell] + to_ghost;
      x[outer] = x[physical_cell] + 2. * to_ghost;
      u[inner][0] = u[outer][0] = to_ghost[axis] < 0. ? NT(ghost) : u[physical_cell][0];
      return true;
    }
  };

  ConfigTree make_config(const uint dim)
  {
    try {
      spdlog::stdout_color_mt("log")->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
    const std::string grid = "0:0.1:1";
    return json::value({{"physical", {{"Lambda", 1.}}},
                        {"discretization",
                         {{"fe_order", 0},
                          {"overintegration", 0},
                          {"EoM_abs_tol", 1e-10},
                          {"EoM_max_iter", 0},
                          {"grid",
                           {{"x_grid", grid}, {"y_grid", dim > 1 ? grid : "0:1:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
                        {"output", {{"verbosity", 0}}}});
  }

  template <uint dim> struct Setup {
    using M = Advection<dim>;
    using Discretization = FV::Discretization<M, RectangularMeshSerial<dim>, double>;
    using Assembler = FV::KurganovTadmor::Assembler<Discretization, M, def::FirstOrderReconstructor<dim, double>>;
    using VectorType = typename Discretization::VectorType;

    ConfigTree config;
    M model;
    RectangularMeshSerial<dim> mesh;
    Discretization discretization;
    std::unique_ptr<Assembler> assembler;
    VectorType u, u_dot;

    Setup(const double c, const double ghost, const uint axis)
        : config(make_config(dim)), mesh(Config::ConfigurationMesh<dim>(config)), discretization(mesh, config)
    {
      model.c = c;
      model.ghost = ghost;
      model.axis = axis;
      assembler = std::make_unique<Assembler>(discretization, model, config);
      u.reinit(discretization.get_dof_handler().n_dofs());
      u = 1.;
      u_dot.reinit(u.size());
    }

    VectorType residual(const VectorType &v)
    {
      VectorType r(v.size());
      assembler->residual(r, v, 1., u_dot, 0.);
      return r;
    }
  };

  /// For u = 1 inside, the residual sums to the net outflow through the boundary: c at the upper side, minus the
  /// upwind flux at the lower side (the lower and upper boundary both have area 1).
  template <uint dim> void check_lower_boundary(const uint axis)
  {
    for (const double c : {2., -2.}) {
      const double g = 3.;
      Setup<dim> s(c, g, axis);
      const auto r = s.residual(s.u);
      double total = 0.;
      for (const auto v : r)
        total += v;
      // c > 0: inflow, the ghost is upwind. c < 0: outflow, the physical cell (u = 1) is upwind.
      const double inflow = c > 0 ? c * g : c * 1.;
      CHECK(total == Catch::Approx(c * 1. - inflow).margin(1e-12));
    }
  }

  /// The assembled jacobian against central finite differences of the residual, on a non-uniform state.
  template <uint dim> void check_jacobian(const uint axis)
  {
    Setup<dim> s(2., 3., axis);
    for (size_t i = 0; i < s.u.size(); ++i)
      s.u[i] = 1. + 0.3 * std::sin(1.3 * double(i));
    SparseMatrix<double> J(s.assembler->get_sparsity_pattern_jacobian());
    s.assembler->jacobian(J, s.u, 1., s.u_dot, 0., 0.);
    const double eps = 1e-6;
    double worst = 0., scale = 0.;
    for (size_t j = 0; j < s.u.size(); ++j) {
      auto up = s.u, down = s.u;
      up[j] += eps;
      down[j] -= eps;
      auto column = s.residual(up);
      column -= s.residual(down);
      column /= 2. * eps;
      for (size_t i = 0; i < s.u.size(); ++i) {
        worst = std::max(worst, std::abs(column[i] - J.el(i, j)));
        scale = std::max(scale, std::abs(column[i]));
      }
    }
    REQUIRE(scale > 0.);
    CHECK(worst <= 1e-7 * scale);
  }
} // namespace

TEST_CASE("KT is upwind at a lower boundary face", "[FV][KT]")
{
  DiFfRG::Init();
  SECTION("1D") { check_lower_boundary<1>(0); }
  SECTION("2D, x") { check_lower_boundary<2>(0); }
  SECTION("2D, y") { check_lower_boundary<2>(1); }
}

TEST_CASE("KT jacobian matches the residual at a lower boundary face", "[FV][KT]")
{
  DiFfRG::Init();
  SECTION("1D") { check_jacobian<1>(0); }
  SECTION("2D, y") { check_jacobian<2>(1); }
}
