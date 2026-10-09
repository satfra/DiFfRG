#pragma once

// Shared by dg_consistency.cc and dg_batch_equivalence.cc: a nonlinear two-component model, its setup on the
// DG assembler, and a plain mesh_loop assembly of its residual as an independent reference.

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>
#include <deal.II/meshworker/mesh_loop.h>

#include <cmath>

namespace dg_test
{
  using namespace dealii;
  using namespace DiFfRG;

  template <bool extractors>
  using RichComponents = std::conditional_t<extractors,
                                            ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>,
                                                                VariableDescriptor<>, ExtractorDescriptor<Scalar<"e">>>,
                                            ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>>>;

  /// The numerical flux of the test model: LLF (batched, or forced per point), or a nonlinear custom one.
  enum class NumFlux { llf, llf_per_point, custom };

  /**
   * Two components; flux and source depend nonlinearly on values, (optionally) derivatives and hessians,
   * the position and (optionally) an extractor. With `batched`, the model evaluates them itself in
   * evaluate_batch from the batch columns; otherwise it uses the default, which calls flux/source.
   */
  template <uint dim, bool batched, bool hessians, bool extractors, NumFlux numflux_kind, bool derivatives>
  class Model : public def::AbstractModel<Model<dim, batched, hessians, extractors, numflux_kind, derivatives>,
                                          RichComponents<extractors>>,
                public def::Time,
                public def::LLFFlux<Model<dim, batched, hessians, extractors, numflux_kind, derivatives>>,
                public def::FlowBoundaries<Model<dim, batched, hessians, extractors, numflux_kind, derivatives>>,
                public def::AD<Model<dim, batched, hessians, extractors, numflux_kind, derivatives>>
  {
    using LLF = def::LLFFlux<Model>;

  public:
    static constexpr bool batch_reads_derivatives = derivatives;
    static constexpr bool batch_reads_hessians = hessians;

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      double y = dim > 1 ? x[dim - 1] : 0.;
      values[0] = 1. + 0.3 * std::sin(2. * x[0]) + 0.2 * x[0] * x[0] + 0.4 * y * y * x[0];
      // Nonlinear in y too: a component linear along a hanging face has equal traces there, which puts LLF's
      // max() of the two wave speeds exactly on its kink, where the jacobian is one-sided.
      values[1] = 0.5 + 0.4 * std::cos(3. * x[0]) - 0.3 * y + 0.25 * y * y * x[0];
    }

    template <int d, typename Vector> std::array<double, d> EoM(const Point<d> &x, const Vector &) const
    {
      std::array<double, d> r;
      for (int i = 0; i < d; ++i)
        r[i] = x[i] - 0.37;
      return r;
    }

    template <typename NT, typename Solution>
    void extract(std::array<NT, RichComponents<extractors>::count_extractors()> &e, const Point<dim> &,
                 const Solution &sol) const
    {
      if constexpr (extractors) e[0] = get<"fe_functions">(sol)[0] * get<"fe_functions">(sol)[1];
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
        F[0][d] = u[0] * u[1] + 0.3 * sin(x[0] + d) * du[0][d] + 0.1 * ext * u[0] * u[0] + 0.05 * u[1] * lap +
                  0.2 * (d + 1) * u[0];
        F[1][d] = u[1] * u[1] * (1. + du[1][d]) + 0.2 * u[0] * du[0][(d + 1) % dim] - 0.1 * (d + 1) * u[1];
      }
      S[0] = u[0] * u[0] * u[0] - ext * u[1] + 0.1 * lap * u[0];
      S[1] = exp(0.1 * u[0]) * u[1] + du[0][0] * du[1][0];
    }

    // DG hands the per-point callbacks no derivatives and hessians: the physics then sees zeros.
    template <typename NT, typename Solution> static auto derivatives_of(const Solution &sol)
    {
      if constexpr (tuple_has<"fe_derivatives", Solution>)
        return as_tensors<NT, 1>(get<"fe_derivatives">(sol));
      else
        return std::array<Tensor<1, dim, NT>, 2>{};
    }
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
      evaluate(F, S, x, as_array<NT>(get<"fe_functions">(sol)), derivatives_of<NT>(sol), hessians_of<NT>(sol),
               get<"extractors">(sol));
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, 2> &S, const Point<dim> &x, const Solution &sol) const
    {
      std::array<Tensor<1, dim, NT>, 2> F;
      evaluate(F, S, x, as_array<NT>(get<"fe_functions">(sol)), derivatives_of<NT>(sol), hessians_of<NT>(sol),
               get<"extractors">(sol));
    }

    /// LLF, or a central flux with a nonlinear, trace-coupling dissipation.
    template <int d, typename NT, typename Solutions_s, typename Solutions_n>
    void numflux(std::array<Tensor<1, d, NT>, 2> &NF, const Tensor<1, d> &normal, const Point<d> &x,
                 const Solutions_s &sol_s, const Solutions_n &sol_n) const
    {
      if constexpr (numflux_kind != NumFlux::custom)
        LLF::numflux(NF, normal, x, sol_s, sol_n);
      else {
        std::array<Tensor<1, d, NT>, 2> F_s, F_n;
        flux(F_s, x, sol_s);
        flux(F_n, x, sol_n);
        const auto &u_s = get<"fe_functions">(sol_s);
        const auto &u_n = get<"fe_functions">(sol_n);
        for (uint c = 0; c < 2; ++c)
          for (int i = 0; i < d; ++i)
            NF[c][i] = 0.5 * (F_s[c][i] + F_n[c][i]) - 0.25 * (u_n[c] - u_s[c]) * (1. + u_s[c] * u_n[c]) * normal[i];
      }
    }

    /// LLF's batched numflux; the other kinds have none, so the assembler calls numflux per point.
    template <typename Out, typename Normals, typename Batch>
    void numflux_batch(Out &out, const Normals &normals, const Batch &s, const Batch &n) const
      requires(numflux_kind == NumFlux::llf)
    {
      LLF::numflux_batch(out, normals, s, n);
    }

    template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
    {
      if constexpr (!batched) {
        def::AbstractModel<Model, RichComponents<extractors>>::evaluate_batch(out, batch);
        return;
      }
      using NT = typename Batch::number_type;
      for (size_t i = 0; i < batch.size(); ++i) {
        std::array<NT, 2> ui{{batch.values(0)[i], batch.values(1)[i]}};
        std::array<Tensor<1, dim, NT>, 2> dui{};
        std::array<Tensor<2, dim, NT>, 2> ddui{};
        Point<dim> x;
        for (uint d1 = 0; d1 < dim; ++d1) {
          x[d1] = batch.coordinates(d1)[i];
          for (uint c = 0; c < 2; ++c) {
            if constexpr (Batch::has_derivatives) dui[c][d1] = batch.derivatives(c, d1)[i];
            if constexpr (hessians && Batch::has_hessians)
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

  inline ConfigTree make_config(const int fe_order, const std::string &grid)
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

  /// Refine the cells in the lower left quarter once, so that there are hanging faces.
  template <typename Mesh> Mesh &refine_corner(Mesh &mesh, const bool refine)
  {
    if (!refine) return mesh;
    auto &tria = mesh.get_triangulation();
    for (const auto &cell : tria.active_cell_iterators()) {
      bool corner = true;
      for (uint d = 0; d < Mesh::dim; ++d)
        corner &= cell->center()[d] < 0.5;
      if (corner) cell->set_refine_flag();
    }
    tria.execute_coarsening_and_refinement();
    return mesh;
  }

  /// Everything a test needs: the DG assembler on a nontrivial state, with a model that reads the derivatives (and,
  /// with `hessians`, the hessians) or the values only.
  template <uint dim, bool derivatives, bool batched, bool hessians, bool extractors, NumFlux numflux = NumFlux::llf>
  struct Setup {
    using Model = dg_test::Model<dim, batched, hessians, extractors, numflux, derivatives>;
    // Serial linear algebra (plain-assembly comparisons, serial sparsity pattern): a plain
    // RectangularMesh is partitioned in an MPI build, which would select PETSc types.
    using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>>;
    using Assembler = DG::Assembler<Discretization>;
    using VectorType = typename Discretization::VectorType;
    using SparseMatrixType = typename Discretization::SparseMatrixType;
    static constexpr bool with_derivatives = derivatives;
    static constexpr bool with_hessians = derivatives && hessians;

    Setup(const int fe_order, const std::string &grid, const bool refine = false)
        : config(make_config(fe_order, grid)), mesh(Config::ConfigurationMesh<dim>(config)),
          discretization(refine_corner(mesh, refine), config), assembler(discretization, model, config),
          state(discretization)
    {
      state.interpolate(model);
      u = state.spatial_data();
      // The interpolant is continuous across conforming faces; a jump at every face makes the numerical flux's
      // dissipation (and its jacobian) part of the test.
      for (uint i = 0; i < u.size(); ++i)
        u[i] += 0.02 * std::sin(1.7 * i);
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
    Assembler assembler;
    FE::FlowingVariables<Discretization> state;
    VectorType u, u_dot;
    SparseMatrixType J;
  };

  /**
   * The residual assembled the plain way: deal.II's mesh_loop, serially, point by point, with the model's
   * per-point flux, source, boundary_numflux and numflux, as an independent reference. Derivatives and
   * hessians are zero for DG, as the DG assembler hands them to the model.
   */
  template <typename S> typename S::VectorType reference_residual(S &s)
  {
    constexpr uint dim = S::Discretization::dim;
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    const auto &fe = s.discretization.get_fe();
    const QGauss<dim> quadrature(fe.degree + 1);
    const QGauss<dim - 1> quadrature_face(fe.degree + 1);
    const UpdateFlags flags = update_values | update_gradients | update_hessians | update_quadrature_points |
                              update_JxW_values | update_normal_vectors;
    struct Scratch {
      FEValues<dim> fe_v;
      FEFaceValues<dim> fe_fv;
      FEInterfaceValues<dim> fe_iv;
      Scratch(const Mapping<dim> &m, const FiniteElement<dim> &fe, const Quadrature<dim> &q,
              const Quadrature<dim - 1> &qf, const UpdateFlags flags)
          : fe_v(m, fe, q, flags), fe_fv(m, fe, qf, flags), fe_iv(m, fe, qf, flags)
      {
      }
      Scratch(const Scratch &o)
          : Scratch(o.fe_v.get_mapping(), o.fe_v.get_fe(), o.fe_v.get_quadrature(), o.fe_fv.get_quadrature(),
                    o.fe_v.get_update_flags())
      {
      }
    };
    struct Copy {
      std::vector<std::pair<std::vector<types::global_dof_index>, Vector<double>>> parts;
    };

    typename S::VectorType r(s.u);
    r = 0;
    const std::array<double, S::Model::Components::count_extractors()> no_extractors{};
    const double cell_width = 0.;
    // The point data the assembler hands the model: zero derivatives / hessians unless it gathers them.
    struct PointData {
      Vector<double> u{2};
      std::vector<Tensor<1, dim>> du = std::vector<Tensor<1, dim>>(2);
      std::vector<Tensor<2, dim>> ddu = std::vector<Tensor<2, dim>>(2);
    };
    const auto load = [&](const auto &fe_values) {
      const uint n = fe_values.n_quadrature_points;
      std::vector<PointData> p(n);
      std::vector<Vector<double>> u(n, Vector<double>(2));
      std::vector<std::vector<Tensor<1, dim>>> du(n, std::vector<Tensor<1, dim>>(2));
      std::vector<std::vector<Tensor<2, dim>>> ddu(n, std::vector<Tensor<2, dim>>(2));
      fe_values.get_function_values(s.u, u);
      fe_values.get_function_gradients(s.u, du);
      fe_values.get_function_hessians(s.u, ddu);
      for (uint q = 0; q < n; ++q) {
        p[q].u = u[q];
        if (S::with_derivatives) p[q].du = du[q];
        if (S::with_hessians) p[q].ddu = ddu[q];
      }
      return p;
    };
    // The full tuple, with zeros for what the assembler does not gather.
    const auto at = [&](PointData &p) {
      return batch_tie<true, true>(p.u, p.du, p.ddu, no_extractors, s.u, cell_width);
    };

    const auto cell_worker = [&](const Iterator &cell, Scratch &sc, Copy &copy) {
      copy.parts.clear();
      auto &fe_v = sc.fe_v;
      fe_v.reinit(cell);
      auto p = load(fe_v);
      std::vector<Vector<double>> u_dot(fe_v.n_quadrature_points, Vector<double>(2));
      fe_v.get_function_values(s.u_dot, u_dot);
      auto &[dofs, local] = copy.parts.emplace_back();
      dofs.resize(fe.n_dofs_per_cell());
      cell->get_dof_indices(dofs);
      local.reinit(dofs.size());
      for (const auto q : fe_v.quadrature_point_indices()) {
        std::array<Tensor<1, dim>, 2> F{};
        std::array<double, 2> src{}, m{};
        s.model.flux(F, fe_v.quadrature_point(q), at(p[q]));
        s.model.source(src, fe_v.quadrature_point(q), at(p[q]));
        s.model.mass(m, fe_v.quadrature_point(q), p[q].u, u_dot[q]);
        for (uint i = 0; i < dofs.size(); ++i) {
          const uint c = fe.system_to_component_index(i).first;
          local[i] +=
              fe_v.JxW(q) *
              (weight * (-fe_v.shape_grad_component(i, q, c) * F[c] + fe_v.shape_value_component(i, q, c) * src[c]) +
               weight_mass * fe_v.shape_value_component(i, q, c) * m[c]);
        }
      }
    };
    const auto boundary_worker = [&](const Iterator &cell, const uint f, Scratch &sc, Copy &copy) {
      auto &fe_fv = sc.fe_fv;
      fe_fv.reinit(cell, f);
      auto p = load(fe_fv);
      auto &[dofs, local] = copy.parts.emplace_back();
      dofs.resize(fe.n_dofs_per_cell());
      cell->get_dof_indices(dofs);
      local.reinit(dofs.size());
      for (const auto q : fe_fv.quadrature_point_indices()) {
        std::array<Tensor<1, dim>, 2> F{};
        s.model.boundary_numflux(F, fe_fv.normal_vector(q), fe_fv.quadrature_point(q), at(p[q]));
        for (uint i = 0; i < dofs.size(); ++i) {
          const uint c = fe.system_to_component_index(i).first;
          local[i] += weight * fe_fv.JxW(q) * fe_fv.shape_value_component(i, q, c) * (F[c] * fe_fv.normal_vector(q));
        }
      }
    };
    const auto face_worker = [&](const Iterator &cell, const uint f, const uint sf, const Iterator &ncell,
                                 const uint nf, const uint nsf, Scratch &sc, Copy &copy) {
      auto &fe_iv = sc.fe_iv;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
      auto p_s = load(fe_iv.get_fe_face_values(0));
      auto p_n = load(fe_iv.get_fe_face_values(1));
      auto &[dofs, local] = copy.parts.emplace_back();
      dofs = fe_iv.get_interface_dof_indices();
      local.reinit(dofs.size());
      for (const auto q : fe_iv.quadrature_point_indices()) {
        std::array<Tensor<1, dim>, 2> NF{};
        s.model.numflux(NF, fe_iv.normal_vector(q), fe_iv.quadrature_point(q), at(p_s[q]), at(p_n[q]));
        for (uint i = 0; i < dofs.size(); ++i) {
          const auto &cd = fe_iv.interface_dof_to_dof_indices(i);
          const uint c = fe.system_to_component_index(cd[0] != numbers::invalid_unsigned_int ? cd[0] : cd[1]).first;
          local[i] += weight * fe_iv.JxW(q) * fe_iv.jump_in_shape_values(i, q, c) * (NF[c] * fe_iv.normal_vector(q));
        }
      }
    };
    const auto copier = [&](const Copy &copy) {
      for (const auto &[dofs, local] : copy.parts)
        for (uint i = 0; i < dofs.size(); ++i)
          r[dofs[i]] += local[i];
    };
    Scratch scratch(s.discretization.get_mapping(), fe, quadrature, quadrature_face, flags);
    Copy copy;
    MeshWorker::mesh_loop(s.discretization.get_dof_handler().begin_active(), s.discretization.get_dof_handler().end(),
                          cell_worker, copier, scratch, copy,
                          MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker, face_worker, 1, 1);
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

  template <typename S> void require_residual_matches_reference(S &s)
  {
    const auto r = s.residual(s.u);
    auto deviation = r;
    deviation -= reference_residual(s);
    REQUIRE(max_abs(r) > 0.);
    INFO("worst deviation " << max_abs(deviation) << ", scale " << max_abs(r));
    REQUIRE(max_abs(deviation) <= 1e-12 * max_abs(r));
  }
} // namespace dg_test
