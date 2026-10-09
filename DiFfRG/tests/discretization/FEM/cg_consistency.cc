#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/model/model.hh>

#include <tbb/global_control.h>

#include <cmath>
#include <optional>

using namespace dealii;
using namespace DiFfRG;

namespace
{
  template <bool extractors>
  using RichComponents = std::conditional_t<extractors,
                                            ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>,
                                                                VariableDescriptor<>, ExtractorDescriptor<Scalar<"e">>>,
                                            ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>>>;

  /**
   * Two components; flux and source depend nonlinearly on values, derivatives, (optionally) hessians,
   * the position and (optionally) an extractor. With `batched`, the model evaluates them itself in
   * evaluate_batch from the batch columns; otherwise it uses the default, which calls flux/source.
   */
  template <uint dim, bool batched, bool hessians, bool extractors>
  class ModelRich
      : public def::AbstractModel<ModelRich<dim, batched, hessians, extractors>, RichComponents<extractors>>,
        public def::Time,
        public def::NoNumFlux<ModelRich<dim, batched, hessians, extractors>>,
        public def::FlowBoundaries<ModelRich<dim, batched, hessians, extractors>>,
        public def::AD<ModelRich<dim, batched, hessians, extractors>>
  {
  public:
    static constexpr bool batch_reads_hessians = hessians;

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      double y = dim > 1 ? x[dim - 1] : 0.;
      values[0] = 1. + 0.3 * std::sin(2. * x[0]) + 0.2 * x[0] * x[0] + 0.4 * y * y * x[0];
      values[1] = 0.5 + 0.4 * std::cos(3. * x[0]) - 0.3 * y;
    }

    template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector &) const
    {
      return {{x[0] - 0.37}};
    }

    template <typename NT, typename Solution>
    void extract(std::array<NT, 1> &e, const Point<dim> &, const Solution &sol) const
    {
      e[0] = get<"fe_functions">(sol)[0] * get<"fe_functions">(sol)[1];
    }

    // The physics, shared by the per-point and the batched path.
    template <typename NT, typename E>
    static void evaluate(std::array<Tensor<1, dim, NT>, 2> &F, std::array<NT, 2> &S, const Point<dim> &x,
                         const std::array<NT, 2> &u, const std::array<Tensor<1, dim, NT>, 2> &du,
                         const std::array<Tensor<2, dim, NT>, 2> &ddu, const E &e)
    {
      using std::exp;
      using std::sin;
      NT lap = 0.;
      if constexpr (hessians)
        for (uint d = 0; d < dim; ++d)
          lap += ddu[0][d][d] + 0.3 * ddu[1][d][(d + 1) % dim];
      NT ext = 0.;
      if constexpr (extractors) ext = e[0];
      for (uint d = 0; d < dim; ++d) {
        F[0][d] = u[0] * u[1] + 0.3 * sin(x[0] + d) * du[0][d] + 0.1 * ext * u[0] * u[0] + 0.05 * u[1] * lap;
        F[1][d] = u[1] * u[1] * du[1][d] + 0.2 * u[0] * du[0][(d + 1) % dim];
      }
      S[0] = u[0] * u[0] * u[0] - ext * u[1] + 0.1 * lap * u[0];
      S[1] = exp(0.1 * u[0]) * u[1] + du[0][0] * du[1][0];
    }

    // With batch_reads_hessians = false the per-point tuple has no hessians: the physics then sees zeros.
    template <typename NT, typename Solution> static auto hessians_of(const Solution &sol)
    {
      if constexpr (tuple_has<"fe_hessians", Solution>)
        return as_tensors<NT, 2>(get<"fe_hessians">(sol));
      else
        return std::array<Tensor<2, dim, NT>, 2>{};
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, 2> &F, const Point<dim> &x, const Solution &sol) const
    {
      std::array<NT, 2> S;
      evaluate(F, S, x, as_array<NT>(get<"fe_functions">(sol)), as_tensors<NT, 1>(get<"fe_derivatives">(sol)),
               hessians_of<NT>(sol), get<"extractors">(sol));
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, 2> &S, const Point<dim> &x, const Solution &sol) const
    {
      std::array<Tensor<1, dim, NT>, 2> F;
      evaluate(F, S, x, as_array<NT>(get<"fe_functions">(sol)), as_tensors<NT, 1>(get<"fe_derivatives">(sol)),
               hessians_of<NT>(sol), get<"extractors">(sol));
    }

    template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
    {
      if constexpr (!batched) {
        def::AbstractModel<ModelRich, RichComponents<extractors>>::evaluate_batch(out, batch);
        return;
      }
      using NT = typename Batch::number_type;
      const auto u = batch.values(0), v = batch.values(1);
      for (size_t i = 0; i < batch.size(); ++i) {
        std::array<NT, 2> ui{{u[i], v[i]}};
        std::array<Tensor<1, dim, NT>, 2> dui;
        std::array<Tensor<2, dim, NT>, 2> ddui;
        Point<dim> x;
        for (uint d1 = 0; d1 < dim; ++d1) {
          x[d1] = batch.coordinates(d1)[i];
          for (uint c = 0; c < 2; ++c) {
            dui[c][d1] = batch.derivatives(c, d1)[i];
            if constexpr (hessians)
              for (uint d2 = 0; d2 < dim; ++d2)
                ddui[c][d1][d2] = batch.hessians(c, d1, d2)[i];
          }
        }
        std::array<Tensor<1, dim, NT>, 2> F;
        std::array<NT, 2> S;
        evaluate(F, S, x, ui, dui, ddui, batch.extractors());
        for (uint c = 0; c < 2; ++c) {
          for (uint d = 0; d < dim; ++d)
            out.flux(c, d)[i] = F[c][d];
          if (out.requested(Term::source)) out.source(c)[i] = S[c];
        }
      }
    }

  private:
    template <typename NT, typename V> static std::array<NT, 2> as_array(const V &v) { return {{NT(v[0]), NT(v[1])}}; }
    template <typename NT, int rank, typename V> static std::array<Tensor<rank, dim, NT>, 2> as_tensors(const V &v)
    {
      std::array<Tensor<rank, dim, NT>, 2> t;
      for (uint c = 0; c < 2; ++c)
        t[c] = v[c];
      return t;
    }
  };

  ConfigTree make_config(const uint dim, const int fe_order, const std::string &grid)
  {
    return ConfigTree(json::value({{"physical", {}},
                                   {"discretization",
                                    {{"fe_order", fe_order},
                                     {"overintegration", 0},
                                     {"output_subdivisions", 1},
                                     {"EoM_abs_tol", 1e-10},
                                     {"EoM_max_iter", 100},
                                     {"grid", {{"x_grid", grid}, {"y_grid", grid}, {"z_grid", grid}, {"refine", 0}}},
                                     {"adaptivity",
                                      {{"start_adapt_at", 0.},
                                       {"adapt_dt", 1e-1},
                                       {"level", 0},
                                       {"refine_percent", 1e-1},
                                       {"coarsen_percent", 5e-2}}}}},
                                   {"output", {{"live_plot", false}, {"verbosity", 0}}}}));
  }

  constexpr double weight = 1.3, weight_mass = 0.7;

  /// Everything a test needs: the assembler on a nontrivial state.
  template <uint dim, bool batched, bool hessians, bool extractors> struct Setup {
    using Model = ModelRich<dim, batched, hessians, extractors>;
    // Serial linear algebra (plain-assembly comparisons, serial sparsity pattern): a plain
    // RectangularMesh is partitioned in an MPI build, which would select PETSc types.
    using Discretization = CG::Discretization<Model, RectangularMeshSerial<dim>>;
    using VectorType = typename Discretization::VectorType;
    using SparseMatrixType = typename Discretization::SparseMatrixType;

    Setup(const int fe_order, const std::string &grid)
        : config(make_config(dim, fe_order, grid)), mesh(Config::ConfigurationMesh<dim>(config)),
          discretization(mesh, config), assembler(discretization, model, config), state(discretization)
    {
      state.interpolate(model);
      u = state.spatial_data();
      u_dot = u;
      for (uint i = 0; i < u_dot.size(); ++i)
        u_dot[i] = std::sin(1. + i);
    }

    VectorType residual(const VectorType &at)
    {
      VectorType r(at);
      r = 0;
      assembler.residual(r, at, weight, u_dot, weight_mass);
      return r;
    }

    /// d(residual)/du: the mass term enters through beta = weight_mass, u_dot is held fixed.
    SparseMatrixType &jacobian()
    {
      J.reinit(assembler.get_sparsity_pattern_jacobian());
      assembler.jacobian(J, u, weight, u_dot, 0., weight_mass);
      return J;
    }

    ConfigTree config;
    Model model;
    RectangularMeshSerial<dim> mesh;
    Discretization discretization;
    CG::Assembler<Discretization> assembler;
    FE::FlowingVariables<Discretization> state;
    VectorType u, u_dot;
    SparseMatrixType J;
  };

  /// The residual assembled the plain way, point by point and serially, as an independent reference.
  template <typename S> typename S::VectorType reference_residual(S &s)
  {
    constexpr uint dim = S::Discretization::dim;
    const auto &fe = s.discretization.get_fe();
    const QGauss<dim> quadrature(fe.degree + 1);
    const QGauss<dim - 1> quadrature_face(fe.degree + 1);
    const UpdateFlags flags = update_values | update_gradients | update_hessians | update_quadrature_points |
                              update_JxW_values | update_normal_vectors;
    FEValues<dim> fe_v(s.discretization.get_mapping(), fe, quadrature, flags);
    FEFaceValues<dim> fe_fv(s.discretization.get_mapping(), fe, quadrature_face, flags);

    typename S::VectorType r(s.u);
    r = 0;
    std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
    std::vector<Vector<double>> u(quadrature.size(), Vector<double>(2)), u_dot(u);
    std::vector<std::vector<Tensor<1, dim>>> du(quadrature.size(), std::vector<Tensor<1, dim>>(2));
    std::vector<std::vector<Tensor<2, dim>>> ddu(quadrature.size(), std::vector<Tensor<2, dim>>(2));
    const std::array<double, 0> no_extractors{};
    const double cell_width = 0.;
    const auto at = [&](const uint q) {
      return batch_tie<true, true>(u[q], du[q], ddu[q], no_extractors, s.u, cell_width);
    };

    for (const auto &cell : s.discretization.get_dof_handler().active_cell_iterators()) {
      cell->get_dof_indices(dofs);
      fe_v.reinit(cell);
      fe_v.get_function_values(s.u, u);
      fe_v.get_function_values(s.u_dot, u_dot);
      fe_v.get_function_gradients(s.u, du);
      fe_v.get_function_hessians(s.u, ddu);
      for (const auto q : fe_v.quadrature_point_indices()) {
        std::array<Tensor<1, dim>, 2> F{};
        std::array<double, 2> src{}, m{};
        s.model.flux(F, fe_v.quadrature_point(q), at(q));
        s.model.source(src, fe_v.quadrature_point(q), at(q));
        s.model.mass(m, fe_v.quadrature_point(q), u[q], u_dot[q]);
        for (uint i = 0; i < dofs.size(); ++i) {
          const uint c = fe.system_to_component_index(i).first;
          r[dofs[i]] +=
              fe_v.JxW(q) *
              (weight * (-fe_v.shape_grad_component(i, q, c) * F[c] + fe_v.shape_value_component(i, q, c) * src[c]) +
               weight_mass * fe_v.shape_value_component(i, q, c) * m[c]);
        }
      }
      for (const auto f : cell->face_indices()) {
        if (!cell->at_boundary(f)) continue;
        fe_fv.reinit(cell, f);
        fe_fv.get_function_values(s.u, u);
        fe_fv.get_function_gradients(s.u, du);
        fe_fv.get_function_hessians(s.u, ddu);
        for (const auto q : fe_fv.quadrature_point_indices()) {
          std::array<Tensor<1, dim>, 2> F{};
          s.model.boundary_numflux(F, fe_fv.normal_vector(q), fe_fv.quadrature_point(q), at(q));
          for (uint i = 0; i < dofs.size(); ++i) {
            const uint c = fe.system_to_component_index(i).first;
            r[dofs[i]] +=
                weight * fe_fv.JxW(q) * fe_fv.shape_value_component(i, q, c) * (F[c] * fe_fv.normal_vector(q));
          }
        }
      }
    }
    return r;
  }

  template <typename V> double max_abs(const V &v)
  {
    double m = 0.;
    for (uint i = 0; i < v.size(); ++i)
      m = std::max(m, std::abs(v[i]));
    return m;
  }

  /// max |a - b| over the entries of both (they share a sparsity pattern), and max |a|.
  template <typename M> std::pair<double, double> matrix_deviation(const M &a, const M &b)
  {
    double worst = 0., scale = 0.;
    for (const auto &entry : a) {
      scale = std::max(scale, std::abs(entry.value()));
      worst = std::max(worst, std::abs(entry.value() - b.el(entry.row(), entry.column())));
    }
    return {worst, scale};
  }

  /// Central finite differences of the residual, column by column, against the assembled jacobian.
  template <typename S> void require_jacobian_matches_finite_differences(S &s)
  {
    const auto &J = s.jacobian();
    double worst = 0., scale = 0.;
    for (uint j = 0; j < s.u.size(); ++j) {
      const double h = 1e-6 * std::max(1., std::abs(s.u[j]));
      auto plus = s.u, minus = s.u;
      plus[j] += h;
      minus[j] -= h;
      const auto r_plus = s.residual(plus), r_minus = s.residual(minus);
      for (uint i = 0; i < s.u.size(); ++i) {
        const double fd = (r_plus[i] - r_minus[i]) / (2 * h);
        scale = std::max(scale, std::abs(fd));
        worst = std::max(worst, std::abs(fd - J.el(i, j)));
      }
    }
    REQUIRE(scale > 0.);
    INFO("jacobian vs finite differences: worst " << worst << ", scale " << scale);
    REQUIRE(worst <= 1e-6 * scale);
  }
} // namespace

TEST_CASE("CG residual matches a plain per-point assembly", "[discretization][cg]")
{
  DiFfRG::Init();
  const int fe_order = GENERATE(1, 2, 3);
  const auto check = [&](auto &s) {
    const auto r = s.residual(s.u);
    auto deviation = r;
    deviation -= reference_residual(s);
    REQUIRE(max_abs(r) > 0.);
    INFO("worst deviation " << max_abs(deviation) << ", scale " << max_abs(r));
    REQUIRE(max_abs(deviation) <= 1e-12 * max_abs(r));
  };
  SECTION("1D, hessians")
  {
    Setup<1, false, true, false> s(fe_order, "0:0.1:1");
    check(s);
  }
  SECTION("2D, hessians")
  {
    Setup<2, false, true, false> s(fe_order, "0:0.25:1");
    check(s);
  }
}

TEST_CASE("CG jacobian matches finite differences of the residual", "[discretization][cg]")
{
  DiFfRG::Init();
  const int fe_order = GENERATE(1, 2, 3);
  SECTION("1D, batched model, hessians, extractor")
  {
    Setup<1, true, true, true> s(fe_order, "0:0.1:1");
    require_jacobian_matches_finite_differences(s);
  }
  SECTION("1D, per-point model, no hessians")
  {
    Setup<1, false, false, false> s(fe_order, "0:0.1:1");
    require_jacobian_matches_finite_differences(s);
  }
  SECTION("2D, batched model, hessians")
  {
    Setup<2, true, true, false> s(fe_order, "0:0.25:1");
    require_jacobian_matches_finite_differences(s);
  }
}

TEST_CASE("A model's own evaluate_batch matches the per-point default", "[discretization][cg]")
{
  DiFfRG::Init();
  const auto check = [](auto &batched, auto &per_point) {
    const auto r_b = batched.residual(batched.u), r_p = per_point.residual(per_point.u);
    auto deviation = r_b;
    deviation -= r_p;
    REQUIRE(max_abs(deviation) <= 1e-12 * max_abs(r_p));
    const auto [worst, scale] = matrix_deviation(per_point.jacobian(), batched.jacobian());
    REQUIRE(worst <= 1e-12 * scale);
  };
  SECTION("1D with extractor")
  {
    Setup<1, true, true, true> b(2, "0:0.1:1");
    Setup<1, false, true, true> p(2, "0:0.1:1");
    check(b, p);
  }
  SECTION("2D")
  {
    Setup<2, true, true, false> b(2, "0:0.25:1");
    Setup<2, false, true, false> p(2, "0:0.25:1");
    check(b, p);
  }
}

// Thousands of cells, so that the cells of one color really are assembled on many threads at once. The
// result must not depend on the thread count at all; a coloring that lets two concurrently written
// cells share a row breaks that.
TEST_CASE("CG assembly does not depend on the thread count", "[discretization][cg]")
{
  DiFfRG::Init();
  Setup<1, true, false, false> s(2, "0:0.00025:1");
  const auto r_all = s.residual(s.u);
  typename decltype(s)::SparseMatrixType J_all(s.assembler.get_sparsity_pattern_jacobian());
  J_all.copy_from(s.jacobian());
  std::optional<tbb::global_control> serial;
  serial.emplace(tbb::global_control::max_allowed_parallelism, 1);
  const auto r_one = s.residual(s.u);
  const auto &J_one = s.jacobian();
  serial.reset();

  auto deviation = r_all;
  deviation -= r_one;
  REQUIRE(max_abs(r_all) > 0.);
  REQUIRE(max_abs(deviation) == 0.);
  const auto [worst, scale] = matrix_deviation(J_all, J_one);
  REQUIRE(scale > 0.);
  REQUIRE(worst == 0.);
}
