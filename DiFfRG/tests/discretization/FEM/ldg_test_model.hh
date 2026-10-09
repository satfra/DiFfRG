#pragma once

// Shared by ldg_consistency.cc and ldg_batch_equivalence.cc: nonlinear LDG models with one and two levels, their
// setup on the LDG assembler, and a plain mesh_loop assembly of the residual as an independent reference.

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cmath>

namespace ldg_test
{
  using namespace dealii;
  using namespace DiFfRG;

  template <bool extractors>
  using Components2 =
      ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>, VariableDescriptor<>,
                          std::conditional_t<extractors, ExtractorDescriptor<Scalar<"e">>, ExtractorDescriptor<>>,
                          FEFunctionDescriptor<Scalar<"lu">, Scalar<"lv">>>;
  using Components3 =
      ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>, VariableDescriptor<>, ExtractorDescriptor<>,
                          FEFunctionDescriptor<Scalar<"lu">, Scalar<"lv">>, FEFunctionDescriptor<Scalar<"llu">>>;

  using Level1Fluxes = def::UpDownFlux<def::FlowDirections<0, 0>, def::UpDown<def::from_right, def::from_left>>;
  using Level2Fluxes = def::UpDownFlux<def::FlowDirections<0>, def::UpDown<def::from_left>>;

  /// The pieces both models share. Main level: u, v; level 1: (lu, lv) = (u', v'); level 2 (3 levels only): llu.
  template <typename Derived, typename Components, bool extractors, bool batched> struct ModelBase {
    static constexpr uint dim = 1;

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      values[0] = 1. + 0.3 * std::sin(2. * x[0]) + 0.2 * x[0] * x[0];
      values[1] = 0.5 + 0.4 * std::cos(3. * x[0]);
    }

    template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector &) const
    {
      return {{x[0] - 0.37}};
    }

    template <typename NT, typename Solution>
    void extract(std::array<NT, Components::count_extractors()> &e, const Point<dim> &, const Solution &sol) const
    {
      if constexpr (extractors) e[0] = get<"fe_functions">(sol)[0] * get<"fe_functions">(sol)[1];
    }

    // The main-level physics, shared by the per-point and the batched path; l1, l2 are the levels (l2 = 0 for
    // the one-level model).
    template <typename NT, typename E>
    static void evaluate(std::array<Tensor<1, dim, NT>, 2> &F, std::array<NT, 2> &S, const Point<dim> &x,
                         const std::array<NT, 2> &u, const std::array<NT, 2> &l1, const NT &l2, const E &e)
    {
      using std::exp;
      F[0][0] = u[0] * u[1] + 0.3 * l1[0] * u[0] + 0.1 * e * u[0] * u[0] + 0.05 * x[0] + 0.2 * l2 * u[1];
      F[1][0] = 0.5 * u[1] * u[1] + 0.2 * l1[1] * u[0];
      S[0] = 0.1 * u[0] * u[0] * u[0] - e * u[1] + l1[0] * l1[1] + 0.1 * l2;
      S[1] = exp(0.1 * u[0]) * u[1];
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, 2> &F, const Point<dim> &x, const Solution &sol) const
    {
      std::array<NT, 2> S;
      evaluate_at(F, S, x, sol);
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, 2> &S, const Point<dim> &x, const Solution &sol) const
    {
      std::array<Tensor<1, dim, NT>, 2> F;
      evaluate_at(F, S, x, sol);
    }

    template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
    {
      if constexpr (!batched) {
        static_cast<const Derived &>(*this).default_evaluate_batch(out, batch);
        return;
      }
      using NT = typename Batch::number_type;
      for (size_t i = 0; i < batch.size(); ++i) {
        const std::array<NT, 2> u{{batch.values(0)[i], batch.values(1)[i]}};
        const std::array<NT, 2> l1{{batch.ldg_values(1, 0)[i], batch.ldg_values(1, 1)[i]}};
        NT l2 = 0.;
        if constexpr (Components::count_fe_subsystems() > 2) l2 = batch.ldg_values(2, 0)[i];
        NT e = 0.;
        if constexpr (extractors) e = batch.extractors()[0];
        std::array<Tensor<1, dim, NT>, 2> F;
        std::array<NT, 2> S;
        evaluate(F, S, batch.x(i), u, l1, l2, e);
        for (uint c = 0; c < 2; ++c) {
          out.flux(c, 0)[i] = F[c][0];
          if (out.requested(Term::source)) out.source(c)[i] = S[c];
        }
      }
    }

  private:
    template <typename NT, typename Solution>
    void evaluate_at(std::array<Tensor<1, dim, NT>, 2> &F, std::array<NT, 2> &S, const Point<dim> &x,
                     const Solution &sol) const
    {
      const auto &u = get<"fe_functions">(sol);
      const auto &l1 = get<"LDG1">(sol);
      NT l2 = 0.;
      if constexpr (Components::count_fe_subsystems() > 2) l2 = get<"LDG2">(sol)[0];
      NT e = 0.;
      if constexpr (extractors) e = get<"extractors">(sol)[0];
      evaluate(F, S, x, {{NT(u[0]), NT(u[1])}}, {{NT(l1[0]), NT(l1[1])}}, l2, e);
    }
  };

  /**
   * Two FE functions u, v and one LDG level (lu, lv) = (u', v'). The main flux and source depend nonlinearly on
   * both levels and (optionally) on an extractor, which also reaches the LLF numerical flux. With `constant`, the
   * model declares the level jacobian constant, so the assembler builds the level from the solution by a
   * precomputed matrix instead of assembling it. With `batched`, it evaluates the main level in evaluate_batch.
   */
  template <bool constant, bool extractors, bool batched = false>
  class Model2 : public def::AbstractModel<Model2<constant, extractors, batched>, Components2<extractors>>,
                 public ModelBase<Model2<constant, extractors, batched>, Components2<extractors>, extractors, batched>,
                 public def::Time,
                 public def::LLFFlux<Model2<constant, extractors, batched>>,
                 public def::LDGUpDownFluxes<Model2<constant, extractors, batched>, Level1Fluxes>,
                 public def::FlowBoundaries<Model2<constant, extractors, batched>>,
                 public def::AD<Model2<constant, extractors, batched>>
  {
    using Abstract = def::AbstractModel<Model2, Components2<extractors>>;
    using Shared = ModelBase<Model2, Components2<extractors>, extractors, batched>;

  public:
    using Components = Components2<extractors>;
    using Shared::dim;
    using Shared::EoM;
    using Shared::evaluate_batch;
    using Shared::extract;
    using Shared::flux;
    using Shared::initial_condition;
    using Shared::source;

    Model2()
    {
      this->components().add_dependency(1, 0, 0, 0);
      this->components().add_dependency(1, 1, 0, 1);
      if constexpr (constant) this->components().set_jacobian_constant(1, 0);
    }

    template <typename Out, typename Batch> void default_evaluate_batch(Out &out, const Batch &batch) const
    {
      Abstract::evaluate_batch(out, batch);
    }

    template <uint submodel, typename NT, typename Vector>
    void ldg_flux(std::array<Tensor<1, dim, NT>, 2> &F, const Point<dim> &, const Vector &u) const
    {
      F[0][0] = u[0];
      F[1][0] = u[1];
    }
  };

  /**
   * Model2's physics plus a second level llu = (lu^2 / 2 + 0.3 lu lv)' + 0.1 lv, which is nonlinear in level 1 and
   * depends on both of its components; the main level reads it. Level 1 is constant or assembled, level 2 is
   * always assembled.
   */
  template <bool constant, bool batched = false>
  class Model3 : public def::AbstractModel<Model3<constant, batched>, Components3>,
                 public ModelBase<Model3<constant, batched>, Components3, false, batched>,
                 public def::Time,
                 public def::LLFFlux<Model3<constant, batched>>,
                 public def::LDGUpDownFluxes<Model3<constant, batched>, Level1Fluxes, Level2Fluxes>,
                 public def::FlowBoundaries<Model3<constant, batched>>,
                 public def::AD<Model3<constant, batched>>
  {
    using Abstract = def::AbstractModel<Model3, Components3>;
    using Shared = ModelBase<Model3, Components3, false, batched>;

  public:
    using Components = Components3;
    using Shared::dim;
    using Shared::EoM;
    using Shared::evaluate_batch;
    using Shared::extract;
    using Shared::flux;
    using Shared::initial_condition;
    using Shared::source;

    Model3()
    {
      this->components().add_dependency(1, 0, 0, 0);
      this->components().add_dependency(1, 1, 0, 1);
      this->components().add_dependency(2, 0, 1, 0);
      this->components().add_dependency(2, 0, 1, 1);
      if constexpr (constant) this->components().set_jacobian_constant(1, 0);
    }

    template <typename Out, typename Batch> void default_evaluate_batch(Out &out, const Batch &batch) const
    {
      Abstract::evaluate_batch(out, batch);
    }

    template <uint submodel, typename NT, typename Vector>
    void ldg_flux(std::array<Tensor<1, dim, NT>, Components3::count_fe_functions(submodel)> &F, const Point<dim> &,
                  const Vector &u) const
    {
      if constexpr (submodel == 1) {
        F[0][0] = u[0];
        F[1][0] = u[1];
      } else
        F[0][0] = 0.5 * u[0] * u[0] + 0.3 * u[0] * u[1];
    }

    template <uint submodel, typename NT, typename Vector>
    void ldg_source(std::array<NT, Components3::count_fe_functions(submodel)> &s, const Point<dim> &,
                    const Vector &u) const
    {
      if constexpr (submodel == 2) s[0] = 0.1 * u[1];
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
                                     {"grid", {{"x_grid", grid}, {"refine", 0}}},
                                     {"adaptivity",
                                      {{"start_adapt_at", 0.},
                                       {"adapt_dt", 1e-1},
                                       {"level", 0},
                                       {"refine_percent", 1e-1},
                                       {"coarsen_percent", 5e-2}}}}},
                                   {"output", {{"live_plot", false}, {"verbosity", 0}}}}));
  }

  inline void ensure_logger()
  {
    try {
      spdlog::stdout_color_mt("log");
    } catch (const spdlog::spdlog_ex &) {
    }
  }

  constexpr double weight = 1.3, weight_mass = 0.7;

  template <typename Model_> struct Setup {
    using Model = Model_;
    using Discretization = LDG::Discretization<Model, RectangularMeshSerial<1>>;
    using VectorType = typename Discretization::VectorType;
    using SparseMatrixType = typename Discretization::SparseMatrixType;

    Setup(const int fe_order, const std::string &grid = "0:0.1:1")
        : config(make_config(fe_order, grid)), mesh((ensure_logger(), Config::ConfigurationMesh<1>(config))),
          discretization(mesh, config), assembler(discretization, model, config), state(discretization)
    {
      state.interpolate(model);
      u = state.spatial_data();
      // A jump at every face, so that the numerical fluxes' dissipation is part of the test.
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
    RectangularMeshSerial<1> mesh;
    Discretization discretization;
    LDG::Assembler<Discretization> assembler;
    FE::FlowingVariables<Discretization> state;
    VectorType u, u_dot;
    SparseMatrixType J;
  };

  namespace reference
  {
    /// The point tuple of the main level, with the names of the LDG assembler (no extractors).
    template <typename L0, typename L1>
    auto tie(L0 &l0, L1 &l1, const std::array<double, 0> &e, const Vector<double> &v, const double &w)
    {
      return named_tuple<std::tuple<L0 &, L1 &, const std::array<double, 0> &, const Vector<double> &, const double &>,
                         StringSet<"fe_functions", "LDG1", "extractors", "variables", "cell_width">>(
          std::tie(l0, l1, e, v, w));
    }
    template <typename L0, typename L1, typename L2>
    auto tie(L0 &l0, L1 &l1, L2 &l2, const std::array<double, 0> &e, const Vector<double> &v, const double &w)
    {
      return named_tuple<
          std::tuple<L0 &, L1 &, L2 &, const std::array<double, 0> &, const Vector<double> &, const double &>,
          StringSet<"fe_functions", "LDG1", "LDG2", "extractors", "variables", "cell_width">>(
          std::tie(l0, l1, l2, e, v, w));
    }

    /**
     * Assemble sum_cells B(phi) + sum_boundary_faces F(phi) + sum_interior_faces I(phi) over the test functions phi
     * of @p dofh, serially with deal.II's mesh_loop, into @p rhs. The workers get the cell (or face) FEValues of
     * every level as an array, indexed like `dofhs`.
     */
    template <int dim, typename Cell, typename Boundary, typename Face>
    void loop(const std::vector<const DoFHandler<dim> *> &dofhs, const uint test_level, const Mapping<dim> &mapping,
              const uint n_q, Vector<double> &rhs, const Cell &cell_term, const Boundary &boundary_term,
              const Face &face_term)
    {
      using Iterator = typename DoFHandler<dim>::active_cell_iterator;
      const UpdateFlags flags =
          update_values | update_gradients | update_quadrature_points | update_JxW_values | update_normal_vectors;
      const QGauss<dim> q(n_q);
      const QGauss<dim - 1> q_face(n_q);
      struct Scratch {
        Scratch(const std::vector<const DoFHandler<dim> *> &dofhs, const Mapping<dim> &mapping,
                const Quadrature<dim> &q, const Quadrature<dim - 1> &q_face, const UpdateFlags flags)
            : dofhs(dofhs), mapping(mapping), q(q), q_face(q_face), flags(flags)
        {
          for (const auto *d : dofhs) {
            fe_v.push_back(std::make_unique<FEValues<dim>>(mapping, d->get_fe(), q, flags));
            fe_fv.push_back(std::make_unique<FEFaceValues<dim>>(mapping, d->get_fe(), q_face, flags));
            fe_iv.push_back(std::make_unique<FEInterfaceValues<dim>>(mapping, d->get_fe(), q_face, flags));
          }
        }
        Scratch(const Scratch &o) : Scratch(o.dofhs, o.mapping, o.q, o.q_face, o.flags) {}
        const std::vector<const DoFHandler<dim> *> &dofhs;
        const Mapping<dim> &mapping;
        const Quadrature<dim> &q;
        const Quadrature<dim - 1> &q_face;
        const UpdateFlags flags;
        std::vector<std::unique_ptr<FEValues<dim>>> fe_v;
        std::vector<std::unique_ptr<FEFaceValues<dim>>> fe_fv;
        std::vector<std::unique_ptr<FEInterfaceValues<dim>>> fe_iv;
      };
      struct Copy {
        std::vector<std::pair<std::vector<types::global_dof_index>, Vector<double>>> parts;
      };
      Scratch scratch(dofhs, mapping, q, q_face, flags);
      const auto on = [&](const uint l, const Iterator &c) {
        return Iterator(&dofhs[l]->get_triangulation(), c->level(), c->index(), dofhs[l]);
      };
      const auto &test_fe = dofhs[test_level]->get_fe();
      Copy copy;
      const auto cell_worker = [&](const Iterator &cell, Scratch &s, Copy &c) {
        c.parts.clear();
        for (uint l = 0; l < dofhs.size(); ++l)
          s.fe_v[l]->reinit(on(l, cell));
        auto &[dofs, local] = c.parts.emplace_back();
        dofs.resize(test_fe.n_dofs_per_cell());
        on(test_level, cell)->get_dof_indices(dofs);
        local.reinit(dofs.size());
        cell_term(s.fe_v, local);
      };
      const auto boundary_worker = [&](const Iterator &cell, const uint f, Scratch &s, Copy &c) {
        for (uint l = 0; l < dofhs.size(); ++l)
          s.fe_fv[l]->reinit(on(l, cell), f);
        auto &[dofs, local] = c.parts.emplace_back();
        dofs.resize(test_fe.n_dofs_per_cell());
        on(test_level, cell)->get_dof_indices(dofs);
        local.reinit(dofs.size());
        boundary_term(s.fe_fv, local);
      };
      const auto face_worker = [&](const Iterator &cell, const uint f, const uint sf, const Iterator &ncell,
                                   const uint nf, const uint nsf, Scratch &s, Copy &c) {
        for (uint l = 0; l < dofhs.size(); ++l)
          s.fe_iv[l]->reinit(on(l, cell), f, sf, on(l, ncell), nf, nsf);
        auto &[dofs, local] = c.parts.emplace_back();
        dofs = s.fe_iv[test_level]->get_interface_dof_indices();
        local.reinit(dofs.size());
        face_term(s.fe_iv, local);
      };
      const auto copier = [&](const Copy &c) {
        for (const auto &[dofs, local] : c.parts)
          for (uint i = 0; i < dofs.size(); ++i)
            rhs[dofs[i]] += local[i];
      };
      MeshWorker::mesh_loop(dofhs[0]->begin_active(), dofhs[0]->end(), cell_worker, copier, scratch, copy,
                            MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                                MeshWorker::assemble_own_interior_faces_once,
                            boundary_worker, face_worker, 1, 1);
    }

    /// Values of every component of @p v at the points of @p fe_v.
    template <typename FEV> std::vector<Vector<double>> values(const FEV &fe_v, const Vector<double> &v)
    {
      std::vector<Vector<double>> out(fe_v.n_quadrature_points, Vector<double>(fe_v.get_fe().n_components()));
      fe_v.get_function_values(v, out);
      return out;
    }

    template <size_t n> std::array<double, n> as_array(const Vector<double> &v)
    {
      std::array<double, n> a;
      for (size_t i = 0; i < n; ++i)
        a[i] = v[i];
      return a;
    }

    /// Level `to` from level `to - 1`: M l_to = -(grad phi, F) + (phi, s) + boundary and face numerical fluxes.
    template <uint to, typename S> Vector<double> level(S &s, const Vector<double> &from_vector)
    {
      constexpr uint dim = 1;
      using C = typename S::Model::Components;
      constexpr size_t n_from = C::count_fe_functions(to - 1), n_to = C::count_fe_functions(to);
      const auto &dofh_from = s.discretization.get_dof_handler(to - 1);
      const auto &dofh_to = s.discretization.get_dof_handler(to);
      const auto &fe_to = dofh_to.get_fe();
      const uint n_q = s.discretization.get_fe(0).degree + 1;
      const std::vector<const DoFHandler<dim> *> dofhs{&dofh_from, &dofh_to};
      const auto comp = [&](const uint i) { return fe_to.system_to_component_index(i).first; };

      Vector<double> rhs(dofh_to.n_dofs());
      loop<dim>(
          dofhs, 1, s.discretization.get_mapping(), n_q, rhs,
          [&](auto &fe_v, Vector<double> &local) {
            const auto u = values(*fe_v[0], from_vector);
            for (const auto q : fe_v[1]->quadrature_point_indices()) {
              std::array<Tensor<1, dim>, n_to> F{};
              std::array<double, n_to> src{};
              const auto uq = as_array<n_from>(u[q]);
              s.model.template ldg_flux<to>(F, fe_v[1]->quadrature_point(q), uq);
              s.model.template ldg_source<to>(src, fe_v[1]->quadrature_point(q), uq);
              for (uint i = 0; i < local.size(); ++i)
                local[i] += fe_v[1]->JxW(q) * (-fe_v[1]->shape_grad_component(i, q, comp(i)) * F[comp(i)] +
                                               fe_v[1]->shape_value_component(i, q, comp(i)) * src[comp(i)]);
            }
          },
          [&](auto &fe_fv, Vector<double> &local) {
            const auto u = values(*fe_fv[0], from_vector);
            for (const auto q : fe_fv[1]->quadrature_point_indices()) {
              std::array<Tensor<1, dim>, n_to> F{};
              s.model.template ldg_boundary_numflux<to>(F, fe_fv[1]->normal_vector(q), fe_fv[1]->quadrature_point(q),
                                                        as_array<n_from>(u[q]));
              for (uint i = 0; i < local.size(); ++i)
                local[i] += fe_fv[1]->JxW(q) * fe_fv[1]->shape_value_component(i, q, comp(i)) *
                            (F[comp(i)] * fe_fv[1]->normal_vector(q));
            }
          },
          [&](auto &fe_iv, Vector<double> &local) {
            const auto u_s = values(fe_iv[0]->get_fe_face_values(0), from_vector);
            const auto u_n = values(fe_iv[0]->get_fe_face_values(1), from_vector);
            for (const auto q : fe_iv[1]->quadrature_point_indices()) {
              std::array<Tensor<1, dim>, n_to> NF{};
              s.model.template ldg_numflux<to>(NF, fe_iv[1]->normal_vector(q), fe_iv[1]->quadrature_point(q),
                                               as_array<n_from>(u_s[q]), as_array<n_from>(u_n[q]));
              for (uint i = 0; i < local.size(); ++i) {
                const auto &cd = fe_iv[1]->interface_dof_to_dof_indices(i);
                const uint c =
                    fe_to.system_to_component_index(cd[0] != numbers::invalid_unsigned_int ? cd[0] : cd[1]).first;
                local[i] +=
                    fe_iv[1]->JxW(q) * fe_iv[1]->jump_in_shape_values(i, q, c) * (NF[c] * fe_iv[1]->normal_vector(q));
              }
            }
          });

      // The level's own mass matrix, assembled independently of the assembler's.
      DynamicSparsityPattern dsp(dofh_to.n_dofs());
      DoFTools::make_sparsity_pattern(dofh_to, dsp);
      SparsityPattern sp;
      sp.copy_from(dsp);
      SparseMatrix<double> mass(sp);
      MatrixCreator::create_mass_matrix(s.discretization.get_mapping(), dofh_to, QGauss<dim>(n_q), mass);
      SparseDirectUMFPACK solver;
      solver.initialize(mass);
      solver.solve(rhs);
      return rhs;
    }
  } // namespace reference

  /**
   * The residual assembled the plain way: each level from the previous one with deal.II's mesh_loop and a direct
   * solve of its mass matrix, then the main level with the model's per-point flux, source, boundary_numflux and
   * numflux. No extractors.
   */
  template <typename S> Vector<double> reference_residual(S &s)
  {
    constexpr uint dim = 1;
    using C = typename S::Model::Components;
    constexpr uint n_levels = C::count_fe_subsystems();
    static_assert(C::count_extractors() == 0, "The reference residual does not evaluate extractors.");

    std::vector<Vector<double>> levels{s.u};
    levels.push_back(reference::level<1>(s, levels[0]));
    if constexpr (n_levels > 2) levels.push_back(reference::level<2>(s, levels[1]));

    std::vector<const DoFHandler<dim> *> dofhs;
    for (uint l = 0; l < n_levels; ++l)
      dofhs.push_back(&s.discretization.get_dof_handler(l));
    const auto &fe = dofhs[0]->get_fe();
    const auto comp = [&](const uint i) { return fe.system_to_component_index(i).first; };
    const std::array<double, 0> no_extractors{};
    const Vector<double> variables;
    const double width = 0.;

    // The point tuple at point q of every level's values.
    const auto at = [&](auto &vals, const uint q, auto &&body) {
      auto l0 = reference::as_array<2>(vals[0][q]);
      auto l1 = reference::as_array<2>(vals[1][q]);
      if constexpr (n_levels > 2) {
        auto l2 = reference::as_array<1>(vals[2][q]);
        body(reference::tie(l0, l1, l2, no_extractors, variables, width), l0);
      } else
        body(reference::tie(l0, l1, no_extractors, variables, width), l0);
    };
    const auto all_values = [&](const auto &fe_values_of) {
      std::vector<std::vector<Vector<double>>> vals;
      for (uint l = 0; l < n_levels; ++l)
        vals.push_back(reference::values(fe_values_of(l), levels[l]));
      return vals;
    };

    Vector<double> r(s.u.size());
    reference::loop<dim>(
        dofhs, 0, s.discretization.get_mapping(), fe.degree + 1, r,
        [&](auto &fe_v, Vector<double> &local) {
          auto vals = all_values([&](const uint l) -> auto & { return *fe_v[l]; });
          const auto u_dot = reference::values(*fe_v[0], s.u_dot);
          for (const auto q : fe_v[0]->quadrature_point_indices())
            at(vals, q, [&](const auto &sol, const auto &u) {
              std::array<Tensor<1, dim>, 2> F{};
              std::array<double, 2> src{}, m{};
              s.model.flux(F, fe_v[0]->quadrature_point(q), sol);
              s.model.source(src, fe_v[0]->quadrature_point(q), sol);
              s.model.mass(m, fe_v[0]->quadrature_point(q), u, reference::as_array<2>(u_dot[q]));
              for (uint i = 0; i < local.size(); ++i)
                local[i] +=
                    fe_v[0]->JxW(q) * (weight * (-fe_v[0]->shape_grad_component(i, q, comp(i)) * F[comp(i)] +
                                                 fe_v[0]->shape_value_component(i, q, comp(i)) * src[comp(i)]) +
                                       weight_mass * fe_v[0]->shape_value_component(i, q, comp(i)) * m[comp(i)]);
            });
        },
        [&](auto &fe_fv, Vector<double> &local) {
          auto vals = all_values([&](const uint l) -> auto & { return *fe_fv[l]; });
          for (const auto q : fe_fv[0]->quadrature_point_indices())
            at(vals, q, [&](const auto &sol, const auto &) {
              std::array<Tensor<1, dim>, 2> F{};
              s.model.boundary_numflux(F, fe_fv[0]->normal_vector(q), fe_fv[0]->quadrature_point(q), sol);
              for (uint i = 0; i < local.size(); ++i)
                local[i] += weight * fe_fv[0]->JxW(q) * fe_fv[0]->shape_value_component(i, q, comp(i)) *
                            (F[comp(i)] * fe_fv[0]->normal_vector(q));
            });
        },
        [&](auto &fe_iv, Vector<double> &local) {
          auto vals_s = all_values([&](const uint l) -> auto & { return fe_iv[l]->get_fe_face_values(0); });
          auto vals_n = all_values([&](const uint l) -> auto & { return fe_iv[l]->get_fe_face_values(1); });
          for (const auto q : fe_iv[0]->quadrature_point_indices())
            at(vals_s, q, [&](const auto &sol_s, const auto &) {
              at(vals_n, q, [&](const auto &sol_n, const auto &) {
                std::array<Tensor<1, dim>, 2> NF{};
                s.model.numflux(NF, fe_iv[0]->normal_vector(q), fe_iv[0]->quadrature_point(q), sol_s, sol_n);
                for (uint i = 0; i < local.size(); ++i) {
                  const auto &cd = fe_iv[0]->interface_dof_to_dof_indices(i);
                  const uint c =
                      fe.system_to_component_index(cd[0] != numbers::invalid_unsigned_int ? cd[0] : cd[1]).first;
                  local[i] += weight * fe_iv[0]->JxW(q) * fe_iv[0]->jump_in_shape_values(i, q, c) *
                              (NF[c] * fe_iv[0]->normal_vector(q));
                }
              });
            });
        });
    return r;
  }

  template <typename V> double max_abs(const V &v)
  {
    double m = 0.;
    for (uint i = 0; i < v.size(); ++i)
      m = std::max(m, std::abs(v[i]));
    return m;
  }

  /// max |a - b| over all entries of a (b shares a's pattern or is larger), and max |a|.
  template <typename M> std::pair<double, double> matrix_deviation(const M &a, const M &b)
  {
    double worst = 0., scale = 0.;
    for (uint i = 0; i < a.m(); ++i)
      for (uint j = 0; j < a.n(); ++j) {
        scale = std::max(scale, std::abs(a.el(i, j)));
        worst = std::max(worst, std::abs(a.el(i, j) - b.el(i, j)));
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
    REQUIRE(max_abs(deviation) <= 1e-11 * max_abs(r));
  }
} // namespace ldg_test
