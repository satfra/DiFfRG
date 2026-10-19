#pragma once

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/discretization/FEM/assembler/common.hh>
#include <DiFfRG/discretization/common/batched_scatter.hh>
#include <DiFfRG/discretization/common/cell_geometry.hh>
#include <DiFfRG/discretization/common/phase_times.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/model/batch.hh>

// external libraries
#include <tbb/enumerable_thread_specific.h>

// standard library
#include <memory>

namespace DiFfRG
{
  namespace CG
  {
    using namespace dealii;
    using std::array;

    namespace internal
    {
      template <typename... T> auto i_tie(T &&...t)
      {
        return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives", "fe_hessians">>(
            std::tie(t...));
      }
    } // namespace internal

    // clang-format off
    /**
     * @brief Assembles the continuous Galerkin discretization (CG::Discretization) of
     * \f$m(u, \partial_t u) + \nabla\cdot F + s = 0\f$.
     *
     * ### Residual
     * With \f$\phi_i\f$ the shape function of dof \f$i\f$, the residual of dof \f$i\f$ is the weak form of the
     * equation, the flux integrated by parts:
     * \f[
     *   R_i = \sum_K \int_K \Big[ w_m\, \phi_i\, m(u, \partial_t u) - w\, \nabla\phi_i \cdot F + w\, \phi_i\, s \Big]
     *       + w \sum_{f \subset \partial\Omega} \int_f \phi_i\, \hat F_b \cdot n\,.
     * \f]
     * \f$K\f$ runs over the cells and \f$f\f$ over the faces on the domain boundary (periodic faces are not among
     * them), \f$n\f$ is the outward unit normal, \f$w\f$ and \f$w_m\f$ are the residual and mass weights the
     * timestepper passes. \f$F\f$ and \f$s\f$ are evaluated at the Gauss points, as functions of
     * \f$x, u, \nabla u, \nabla^2 u\f$, the extractors and the variables. \f$\hat F_b\f$ is Model::boundary_numflux;
     * def::FlowBoundaries takes the model's own flux there. There is no flux term on interior faces, since \f$u\f$ is
     * continuous.
     *
     * ### Constraints
     * Hanging-node constraints and the model's affine constraints (Model::affine_constraints, e.g. Dirichlet values)
     * are applied to \f$u\f$, and their homogeneous part to \f$\partial_t u\f$, before every evaluation. Constrained
     * dofs have an empty residual row and are algebraic for IDA (get_differential_indices), so the values a
     * timestepper stores for them never matter.
     *
     * ### Model hooks
     * mass, flux and source, or evaluate_batch (all quadrature points in one call; the default calls flux and source
     * per point, a model can override it to evaluate its momentum integrals in one map() call, on the CPU or GPU);
     * boundary_numflux; cell_indicator for mesh adaptation; the extractor and readout hooks. A model whose flux and
     * source read no derivatives declares `static constexpr bool batch_reads_derivatives = false`, one that reads
     * gradients but no hessians `batch_reads_hessians = false`; the assembler then neither gathers nor differentiates
     * them.
     *
     * ### Parameters
     * | Key                                          | Type | Default              | Meaning |
     * |----------------------------------------------|------|----------------------|---------|
     * | `/discretization/overintegration`            | uint | 0                    | quadrature QGauss(\f$p + 1 +\f$ overintegration) on cells and faces |
     * | `/discretization/batched/max_stacked_points` | uint | 256 MiB / point size | points per automatic-differentiation evaluation of the jacobian |
     *
     * ### Implementation
     * Residual and jacobian are assembled in three phases:
     *  1. gather: the FE solution at every quadrature point of the locally owned cells and boundary faces, into a
     *     PointBatch;
     *  2. evaluate: the model's evaluate_batch and boundary numflux over the whole batch, or their jacobians (see
     *     model/batch.hh);
     *  3. scatter: the contraction with the shape functions, in a parallel loop over cell colors (see
     *     DiFfRG::internal::ColoredCells).
     *
     * Each MPI rank batches its own cells and phase 2 communicates nothing. The results do not depend on the thread
     * count: contributions to a global entry are summed in color order.
     *
     * @tparam Discretization_ A CG::Discretization.
     * @tparam Model_ The model; by default that of the discretization.
     */
    // clang-format on
    template <typename Discretization_,
              typename Model_ = typename DiFfRG::internal::assembler_model_of<Discretization_>::type>
    class Assembler : public FEMAssembler<Discretization_, Model_>
    {
      using Base = FEMAssembler<Discretization_, Model_>;

    public:
      using Discretization = Discretization_;
      using Model = Model_;
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;
      using VariablesVector = DiFfRG::internal::VariablesVector<VectorType>;
      using SparseMatrixType = typename Discretization::SparseMatrixType;
      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;

      static constexpr size_t n_fe = Components::count_fe_functions();
      static constexpr size_t n_extr = Components::count_extractors();
      using Extractors = std::array<NumberType, n_extr>;
      /// Which inputs the model reads (batch_reads_derivatives / batch_reads_hessians), and hence which the batches
      /// hold.
      static constexpr bool reads_derivatives = batch_reads_derivatives<Model>();
      static constexpr bool reads_hessians = reads_derivatives && batch_reads_hessians<Model>();
      using Batch = PointBatch<dim, NumberType, n_fe, Extractors, VariablesVector, reads_derivatives, reads_hessians>;

      using PhaseTimes = AssemblyPhaseTimes;

      /// Set up quadrature, constraints, mass matrix and jacobian sparsity; @p discretization and @p model must
      /// outlive the assembler.
      Assembler(Discretization &discretization, Model &model, const ConfigTree &config)
          : Base(discretization, model, config, "CG"),
            quadrature(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
            quadrature_face(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
            max_stacked_points(Base::read_max_stacked_points(config, stacked_point_bytes, Base::fem_stacked_budget))
      {
        static_assert(Components::count_fe_subsystems() == 1, "A CG model cannot have multiple submodels!");
        reinit();
      }

      /// Rebuild constraints, mass matrix, sparsity and cell lists after the mesh changed.
      virtual void reinit() override
      {
        Timer timer;
        this->reinit_common();
        this->build_mass_matrix(quadrature);
        rebuild_jacobian_sparsity();
        setup_cells();
        timings_reinit.push_back(timer.wall_time());
      }

      /// Rebuild the jacobian sparsity: the FE couplings, plus a full column per dof the extractors read.
      virtual void rebuild_jacobian_sparsity() override
      {
        DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                        /*keep_constrained_dofs = */ true);
        for (const auto &row : discretization.get_locally_relevant_dofs())
          for (const auto &col : extractor_dof_indices)
            dsp.add(row, col);
        DiFfRG::internal::finalize_la_sparsity<SparseMatrixType>(
            dsp, sparsity_pattern_jacobian, discretization.get_locally_owned_dofs(),
            discretization.get_locally_relevant_dofs(), discretization.get_communicator());
      }

      /**
       * @brief Add to @p indicator, per active cell \f$K\f$, \f$\int_K \eta\f$ with \f$\eta\f$ = Model::cell_indicator.
       * CG has no face contribution.
       */
      void refinement_indicator(Vector<double> &indicator, const VectorType &solution)
      {
        const auto state = this->constrained(solution);
        const VectorType &solution_global = state.u();
        tbb::enumerable_thread_specific<CellScratch> scratch([this]() {
          return CellScratch(mapping, fe, quadrature, quadrature_face, gather_flags() | update_hessians);
        });
        // map() is collective and each rank visits only its own cells; see NoMapsHere.
        const NoMapsHere no_maps_during_assembly;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const tbb::blocked_range<size_t> &r) {
          auto &s = scratch.local();
          for (size_t k = r.begin(); k != r.end(); ++k) {
            auto &fe_v = s.fe_values;
            fe_v.reinit(cells[k]);
            fe_v.get_function_values(solution_global, s.values);
            fe_v.get_function_gradients(solution_global, s.gradients);
            fe_v.get_function_hessians(solution_global, s.hessians);
            double value = 0., local = 0.;
            for (const auto &q : fe_v.quadrature_point_indices()) {
              model.cell_indicator(local, fe_v.quadrature_point(q),
                                   internal::i_tie(s.values[q], s.gradients[q], s.hessians[q]));
              value += fe_v.JxW(q) * local;
            }
            indicator[cells[k]->active_cell_index()] += value;
          }
        });
      }

      /// \f$w_m \int \phi_i\, m(u, \partial_t u)\f$, see AbstractAssembler::mass.
      virtual void mass(VectorType &mass, const VectorType &solution, const VectorType &solution_dot,
                        NumberType weight) override
      {
        const auto state = this->constrained(solution, solution_dot);
        const VectorType &solution_global = state.u(), &solution_global_dot = state.u_dot();
        scatter_residual(mass, [&](const size_t, CellScratch &s, Vector<NumberType> &r) {
          const auto &fe_v = s.fe_values;
          fe_v.get_function_values(solution_global, s.values);
          fe_v.get_function_values(solution_global_dot, s.values_dot);
          array<NumberType, n_fe> m{};
          for (const auto &q : fe_v.quadrature_point_indices()) {
            model.mass(m, fe_v.quadrature_point(q), s.values[q], s.values_dot[q]);
            for (uint i = 0; i < r.size(); ++i)
              r(i) += weight * fe_v.JxW(q) * fe_v.shape_value_component(i, q, s.comp[i]) * m[s.comp[i]];
          }
        });
      }

      /// The residual \f$R_i\f$ of the class documentation, added to @p residual; see AbstractAssembler::residual.
      virtual void residual(VectorType &residual, const VectorType &solution, NumberType weight,
                            const VectorType &solution_dot, NumberType weight_mass,
                            const VectorType &variables = VectorType()) override
      {
        const DiFfRG::internal::LocalVariables<VectorType> local_variables(variables);
        const auto state = this->constrained(solution, solution_dot);
        const VectorType &solution_global = state.u(), &solution_global_dot = state.u_dot();
        DiFfRG::internal::PhaseTimer phase(residual_times);
        Extractors extracted_data{{}};
        if constexpr (n_extr > 0)
          this->extract(extracted_data, solution_global, local_variables.get(), true, false, true);
        phase.lap(&PhaseTimes::extract);
        gather(solution_global, extracted_data, local_variables.get());
        phase.lap(&PhaseTimes::gather);
        {
          const NoMapsHere no_maps_during_assembly; // map() is collective, the points are rank-local
          cell_result.reinit(cell_batch.size(), Term::flux | Term::source);
          model.evaluate_batch(cell_result, cell_batch);
          face_result.reinit(face_batch.size(), Term::flux);
          DiFfRG::internal::evaluate_boundary_numflux(model, face_result, face_normals, face_batch);
        }
        phase.lap(&PhaseTimes::evaluate);
        scatter_residual(residual, [&](const size_t k, CellScratch &s, Vector<NumberType> &r) {
          const auto &fe_v = s.fe_values;
          fe_v.get_function_values(solution_global, s.values);
          fe_v.get_function_values(solution_global_dot, s.values_dot);
          array<Tensor<1, dim, NumberType>, n_fe> flux;
          array<NumberType, n_fe> source, mass{};
          for (const auto &q : fe_v.quadrature_point_indices()) {
            const size_t p = k * n_q + q;
            for (size_t c = 0; c < n_fe; ++c) {
              for (uint d = 0; d < dim; ++d)
                flux[c][d] = cell_result.flux(c, d)[p];
              source[c] = cell_result.source(c)[p];
            }
            model.mass(mass, fe_v.quadrature_point(q), s.values[q], s.values_dot[q]);
            for (uint i = 0; i < r.size(); ++i) {
              const auto c = s.comp[i];
              r(i) += fe_v.JxW(q) * (weight * (-scalar_product(fe_v.shape_grad_component(i, q, c), flux[c]) +
                                               fe_v.shape_value_component(i, q, c) * source[c]) +
                                     weight_mass * fe_v.shape_value_component(i, q, c) * mass[c]);
            }
          }
          for (size_t f = face_begin[k]; f < face_begin[k + 1]; ++f) {
            auto &fe_fv = s.fe_face_values;
            fe_fv.reinit(cells[k], boundary_faces[f].second);
            for (const auto &q : fe_fv.quadrature_point_indices())
              for (uint i = 0; i < r.size(); ++i) {
                const auto c = s.comp[i];
                Tensor<1, dim, NumberType> numflux;
                for (uint d = 0; d < dim; ++d)
                  numflux[d] = face_result.flux(c, d)[f * n_q_face + q];
                r(i) += weight * fe_fv.JxW(q) * fe_fv.shape_value_component(i, q, c) *
                        scalar_product(numflux, fe_fv.normal_vector(q));
              }
          }
        });
        phase.lap(&PhaseTimes::scatter);
        timings_residual.push_back(phase.finish());
      }

      /// \f$\alpha\, \partial m/\partial(\partial_t u) + \beta\, \partial m / \partial u\f$, contracted with the shape
      /// functions; see AbstractAssembler::jacobian_mass.
      virtual void jacobian_mass(SparseMatrixType &jacobian, const VectorType &solution, const VectorType &solution_dot,
                                 NumberType alpha, NumberType beta) override
      {
        const auto state = this->constrained(solution, solution_dot);
        const VectorType &solution_global = state.u(), &solution_global_dot = state.u_dot();
        Timer timer;
        scatter_jacobian(jacobian, [&](const size_t, CellScratch &s, LocalData &out) {
          add_mass_jacobian(s, out.jacobian, solution_global, solution_global_dot, alpha, beta);
        });
        timings_jacobian.push_back(timer.wall_time());
      }

      /**
       * @brief \f$\partial R / \partial u\f$: \f$w\f$ times the derivative of the flux, source and boundary terms
       * (including their dependence on the extractors), plus the mass terms as in jacobian_mass. See
       * AbstractAssembler::jacobian.
       */
      virtual void jacobian(SparseMatrixType &jacobian, const VectorType &solution, NumberType weight,
                            const VectorType &solution_dot, NumberType alpha, NumberType beta,
                            const VectorType &variables = VectorType()) override
      {
        const DiFfRG::internal::LocalVariables<VectorType> local_variables(variables);
        const auto state = this->constrained(solution, solution_dot);
        const VectorType &solution_global = state.u(), &solution_global_dot = state.u_dot();
        DiFfRG::internal::PhaseTimer phase(jacobian_times);
        Extractors extracted_data{{}};
        if constexpr (n_extr > 0) {
          if (this->extract_with_jacobian(extracted_data, solution_global, local_variables.get()))
            this->reinit_matrix(jacobian);
        }
        phase.lap(&PhaseTimes::extract);
        gather(solution_global, extracted_data, local_variables.get());
        phase.lap(&PhaseTimes::gather);
        {
          const NoMapsHere no_maps_during_assembly; // map() is collective, the points are rank-local
          DiFfRG::internal::evaluate_flux_source_jacobian(model, cell_jacobians, cell_batch, max_stacked_points,
                                                          cell_workspace);
          DiFfRG::internal::evaluate_boundary_numflux_jacobian(model, face_jacobians, face_normals, face_batch,
                                                               max_stacked_points, face_workspace);
        }
        phase.lap(&PhaseTimes::evaluate);
        scatter_jacobian(jacobian, [&](const size_t k, CellScratch &s, LocalData &out) {
          add_mass_jacobian(s, out.jacobian, solution_global, solution_global_dot, alpha, beta);
          const auto &fe_v = s.fe_values;
          for (const auto &q : fe_v.quadrature_point_indices()) {
            s.cache_shapes(fe_v, q);
            add_flux_source_jacobian(s, out, cell_jacobians[k * n_q + q], weight * fe_v.JxW(q));
          }
          for (size_t f = face_begin[k]; f < face_begin[k + 1]; ++f) {
            auto &fe_fv = s.fe_face_values;
            fe_fv.reinit(cells[k], boundary_faces[f].second);
            for (const auto &q : fe_fv.quadrature_point_indices()) {
              s.cache_shapes(fe_fv, q);
              add_boundary_jacobian(s, out, face_jacobians[f * n_q_face + q], fe_fv.normal_vector(q),
                                    weight * fe_fv.JxW(q));
            }
          }
        });
        phase.lap(&PhaseTimes::scatter);
        timings_jacobian.push_back(phase.finish());
      }

    protected:
      using Base::discretization;
      using Base::dof_handler;
      using Base::extractor_dof_indices;
      using Base::fe;
      using Base::jacobian_times;
      using Base::mapping;
      using Base::model;
      using Base::residual_times;
      using Base::sparsity_pattern_jacobian;
      using Base::timings_jacobian;
      using Base::timings_reinit;
      using Base::timings_residual;

      QGauss<dim> quadrature;
      QGauss<dim - 1> quadrature_face;

    private:
      /// AD inputs and outputs of one point of a seed-stacked evaluation; see Base::fem_stacked_budget.
      static constexpr size_t stacked_point_bytes =
          sizeof(autodiff::real) * n_fe * (2 + dim + (reads_derivatives ? dim : 0) + (reads_hessians ? dim * dim : 0));

      static UpdateFlags gather_flags()
      {
        return update_values | update_gradients | update_quadrature_points | update_JxW_values |
               (reads_hessians ? update_hessians : update_default);
      }

      /// One cell's contribution to the global system, before insertion.
      struct LocalData {
        std::vector<types::global_dof_index> dofs;
        /// Whether a dof of the cell is constrained, i.e. insertion has to go through the constraints.
        bool constrained = false;
        Vector<NumberType> residual;
        FullMatrix<NumberType> jacobian;
        FullMatrix<NumberType> extractor_jacobian;
        FullMatrix<NumberType> extractor_dependence;
      };

      /// Per-thread FE data, for the gather and the scatter.
      struct CellScratch {
        CellScratch(const Mapping<dim> &mapping, const FiniteElement<dim> &fe,
                    const dealii::Quadrature<dim> &quadrature, const dealii::Quadrature<dim - 1> &quadrature_face,
                    const UpdateFlags flags)
            : fe_values(mapping, fe, quadrature, flags),
              fe_face_values(mapping, fe, quadrature_face, flags | update_normal_vectors),
              values(quadrature.size(), Vector<NumberType>(fe.n_components())), values_dot(values),
              gradients(quadrature.size(), std::vector<Tensor<1, dim, NumberType>>(fe.n_components())),
              hessians(quadrature.size(), std::vector<Tensor<2, dim, NumberType>>(fe.n_components())),
              face_values(quadrature_face.size(), Vector<NumberType>(fe.n_components())),
              face_gradients(quadrature_face.size(), std::vector<Tensor<1, dim, NumberType>>(fe.n_components())),
              face_hessians(quadrature_face.size(), std::vector<Tensor<2, dim, NumberType>>(fe.n_components())),
              comp(fe.n_dofs_per_cell()), sv(fe.n_dofs_per_cell()), sg(fe.n_dofs_per_cell()), sh(fe.n_dofs_per_cell())
        {
          for (uint i = 0; i < comp.size(); ++i)
            comp[i] = fe.system_to_component_index(i).first;
        }

        /// Shape values, gradients and (if read) hessians of every dof at point q.
        template <typename FEV> void cache_shapes(const FEV &fe_v, const uint q)
        {
          for (uint i = 0; i < comp.size(); ++i) {
            sv[i] = fe_v.shape_value_component(i, q, comp[i]);
            sg[i] = fe_v.shape_grad_component(i, q, comp[i]);
            if constexpr (reads_hessians) sh[i] = fe_v.shape_hessian_component(i, q, comp[i]);
          }
        }

        FEValues<dim> fe_values;
        FEFaceValues<dim> fe_face_values;
        std::vector<Vector<NumberType>> values, values_dot;
        std::vector<std::vector<Tensor<1, dim, NumberType>>> gradients;
        std::vector<std::vector<Tensor<2, dim, NumberType>>> hessians;
        // Sized for the face quadrature: deal.II's get_function_* fill as many points as the output holds.
        std::vector<Vector<NumberType>> face_values;
        std::vector<std::vector<Tensor<1, dim, NumberType>>> face_gradients;
        std::vector<std::vector<Tensor<2, dim, NumberType>>> face_hessians;
        std::vector<uint> comp;
        std::vector<double> sv;
        std::vector<Tensor<1, dim>> sg;
        std::vector<Tensor<2, dim>> sh;
        LocalData local;
      };

      /// Number the owned cells and their boundary faces, and color the cells for the scatter.
      void setup_cells()
      {
        n_q = quadrature.size();
        n_q_face = quadrature_face.size();
        cells.reinit(dof_handler, discretization.get_constraints());
        boundary_faces.clear();
        face_begin.assign(1, 0);
        for (const auto &cell : cells.all()) {
          for (const uint f : cell->face_indices())
            if (cell->at_boundary(f) && !cell->has_periodic_neighbor(f)) boundary_faces.emplace_back(cell, f);
          face_begin.push_back(boundary_faces.size());
        }
        scratch = std::make_unique<tbb::enumerable_thread_specific<CellScratch>>(
            [this]() { return CellScratch(mapping, fe, quadrature, quadrature_face, gather_flags()); });
      }

      /// Phase 1: the solution at every quadrature point of the owned cells and boundary faces.
      void gather(const VectorType &solution_global, const Extractors &extracted_data, const VariablesVector &variables)
      {
        cell_batch.reinit(cells.size() * n_q);
        cell_batch.set_shared(extracted_data, variables);
        face_batch.reinit(boundary_faces.size() * n_q_face);
        face_batch.set_shared(extracted_data, variables);
        face_normals.resize(face_batch.size());

        const auto store = [&](Batch &batch, const size_t first, const auto &fe_v, auto &values, auto &gradients,
                               auto &hessians, const double width) {
          fe_v.get_function_values(solution_global, values);
          if constexpr (reads_derivatives) fe_v.get_function_gradients(solution_global, gradients);
          if constexpr (reads_hessians) fe_v.get_function_hessians(solution_global, hessians);
          for (const auto &q : fe_v.quadrature_point_indices()) {
            const size_t i = first + q;
            for (size_t c = 0; c < n_fe; ++c) {
              batch.value(c, i) = values[q][c];
              for (uint d1 = 0; d1 < dim; ++d1) {
                if constexpr (reads_derivatives) batch.derivative(c, d1, i) = gradients[q][c][d1];
                if constexpr (reads_hessians)
                  for (uint d2 = 0; d2 < dim; ++d2)
                    batch.hessian(c, d1, d2, i) = hessians[q][c][d1][d2];
              }
            }
            for (uint d = 0; d < dim; ++d)
              batch.coordinate(d, i) = fe_v.quadrature_point(q)[d];
            batch.width(i) = width;
          }
        };

        tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const tbb::blocked_range<size_t> &r) {
          auto &s = scratch->local();
          for (size_t k = r.begin(); k != r.end(); ++k) {
            s.fe_values.reinit(cells[k]);
            store(cell_batch, k * n_q, s.fe_values, s.values, s.gradients, s.hessians,
                  DiFfRG::internal::cell_width(cells[k]));
          }
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, boundary_faces.size()),
                          [&](const tbb::blocked_range<size_t> &r) {
                            auto &s = scratch->local();
                            for (size_t f = r.begin(); f != r.end(); ++f) {
                              const auto &[cell, face_no] = boundary_faces[f];
                              s.fe_face_values.reinit(cell, face_no);
                              store(face_batch, f * n_q_face, s.fe_face_values, s.face_values, s.face_gradients,
                                    s.face_hessians, DiFfRG::internal::cell_width(cell));
                              for (uint q = 0; q < n_q_face; ++q)
                                face_normals[f * n_q_face + q] = s.fe_face_values.normal_vector(q);
                            }
                          });
      }

      /// Phase 3: assemble(k, scratch, local) every owned cell, after reinit of its FEValues and dof indices,
      /// and insert(local) the result; see DiFfRG::internal::ColoredCells::scatter.
      template <typename Global, typename Assemble, typename Insert>
      void scatter(Global &global, const Assemble &assemble, const Insert &insert)
      {
        cells.scatter(
            global, *scratch, &CellScratch::local, local_buffer,
            [&](const size_t k, CellScratch &s, LocalData &local) {
              s.fe_values.reinit(cells[k]);
              local.dofs.resize(fe.n_dofs_per_cell());
              cells[k]->get_dof_indices(local.dofs);
              local.constrained = cells.is_constrained(k);
              assemble(k, s, local);
            },
            insert);
      }

      template <typename Assemble> void scatter_residual(VectorType &residual, const Assemble &assemble)
      {
        const auto &constraints = discretization.get_constraints();
        scatter(
            residual,
            [&](const size_t k, CellScratch &s, LocalData &local) {
              local.residual.reinit(local.dofs.size());
              assemble(k, s, local.residual);
            },
            [&](LocalData &local) {
              if (local.constrained)
                constraints.distribute_local_to_global(local.residual, local.dofs, residual);
              else
                residual.add(local.dofs, local.residual);
            });
      }

      template <typename Assemble> void scatter_jacobian(SparseMatrixType &jacobian, const Assemble &assemble)
      {
        const auto &constraints = discretization.get_constraints();
        scatter(
            jacobian,
            [&](const size_t k, CellScratch &s, LocalData &local) {
              local.jacobian.reinit(local.dofs.size(), local.dofs.size());
              if constexpr (n_extr > 0) local.extractor_jacobian.reinit(local.dofs.size(), n_extr);
              assemble(k, s, local);
            },
            [&](LocalData &local) {
              if (local.constrained)
                constraints.distribute_local_to_global(local.jacobian, local.dofs, jacobian);
              else
                jacobian.add(local.dofs, local.jacobian, false);
              if constexpr (n_extr > 0) {
                local.extractor_dependence.reinit(local.dofs.size(), extractor_dof_indices.size());
                local.extractor_jacobian.mmult(local.extractor_dependence, this->extractor_jacobian);
                constraints.distribute_local_to_global(local.extractor_dependence, local.dofs, extractor_dof_indices,
                                                       jacobian);
              }
            });
      }

      /// J += alpha d(mass)/d(u_dot) + beta d(mass)/du, contracted with the shape values.
      void add_mass_jacobian(CellScratch &s, FullMatrix<NumberType> &J, const VectorType &u, const VectorType &u_dot,
                             const NumberType alpha, const NumberType beta) const
      {
        const auto &fe_v = s.fe_values;
        fe_v.get_function_values(u, s.values);
        fe_v.get_function_values(u_dot, s.values_dot);
        SimpleMatrix<NumberType, n_fe> j_mass, j_mass_dot;
        for (const auto &q : fe_v.quadrature_point_indices()) {
          model.template jacobian_mass<0>(j_mass, fe_v.quadrature_point(q), s.values[q], s.values_dot[q]);
          model.template jacobian_mass<1>(j_mass_dot, fe_v.quadrature_point(q), s.values[q], s.values_dot[q]);
          for (uint i = 0; i < J.m(); ++i)
            for (uint j = 0; j < J.n(); ++j)
              J(i, j) += fe_v.JxW(q) * fe_v.shape_value_component(i, q, s.comp[i]) *
                         fe_v.shape_value_component(j, q, s.comp[j]) *
                         (alpha * j_mass_dot(s.comp[i], s.comp[j]) + beta * j_mass(s.comp[i], s.comp[j]));
        }
      }

      /// The flux/source jacobian blocks Jq at one cell point, contracted with the cached shapes, times w.
      template <typename PJ>
      void add_flux_source_jacobian(const CellScratch &s, LocalData &out, const PJ &Jq, const NumberType w) const
      {
        const auto &[comp, sv, sg, sh] = std::tie(s.comp, s.sv, s.sg, s.sh);
        for (uint i = 0; i < comp.size(); ++i) {
          const auto ci = comp[i];
          for (uint j = 0; j < comp.size(); ++j) {
            const auto cj = comp[j];
            NumberType c = sv[j] * (-scalar_product(sg[i], Jq.j_flux(ci, cj)) + sv[i] * Jq.j_source(ci, cj)) +
                           scalar_product(sg[j], -scalar_product(sg[i], Jq.j_grad_flux(ci, cj)) +
                                                     sv[i] * Jq.j_grad_source(ci, cj));
            if constexpr (reads_hessians)
              c += scalar_product(sh[j],
                                  -scalar_product(sg[i], Jq.j_hess_flux(ci, cj)) + sv[i] * Jq.j_hess_source(ci, cj));
            out.jacobian(i, j) += w * c;
          }
          for (uint e = 0; e < n_extr; ++e)
            out.extractor_jacobian(i, e) +=
                w * (-scalar_product(sg[i], Jq.j_extr_flux(ci, e)) + sv[i] * Jq.j_extr_source(ci, e));
        }
      }

      /// The boundary numflux jacobian blocks Jq at one face point, contracted with the normal and the shapes.
      template <typename PJ>
      void add_boundary_jacobian(const CellScratch &s, LocalData &out, const PJ &Jq, const Tensor<1, dim> &normal,
                                 const NumberType w) const
      {
        const auto &[comp, sv, sg, sh] = std::tie(s.comp, s.sv, s.sg, s.sh);
        for (uint i = 0; i < comp.size(); ++i) {
          const auto ci = comp[i];
          for (uint j = 0; j < comp.size(); ++j) {
            const auto cj = comp[j];
            NumberType c = sv[j] * scalar_product(Jq.j_flux(ci, cj), normal) +
                           scalar_product(sg[j], scalar_product(Jq.j_grad_flux(ci, cj), normal));
            if constexpr (reads_hessians) c += scalar_product(sh[j], scalar_product(Jq.j_hess_flux(ci, cj), normal));
            out.jacobian(i, j) += w * sv[i] * c;
          }
          for (uint e = 0; e < n_extr; ++e)
            out.extractor_jacobian(i, e) += w * sv[i] * scalar_product(Jq.j_extr_flux(ci, e), normal);
        }
      }

      const size_t max_stacked_points;

      uint n_q = 0, n_q_face = 0;
      DiFfRG::internal::ColoredCells<dim> cells;
      /// Boundary faces in cell order; those of cell k are [face_begin[k], face_begin[k + 1]).
      std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, uint>> boundary_faces;
      std::vector<size_t> face_begin;
      std::unique_ptr<tbb::enumerable_thread_specific<CellScratch>> scratch;
      std::vector<LocalData> local_buffer;

      Batch cell_batch, face_batch;
      std::vector<Tensor<1, dim>> face_normals;
      BatchOutput<dim, NumberType, n_fe> cell_result, face_result;
      std::vector<DiFfRG::internal::PointJacobian<dim, n_fe, n_fe, n_extr>> cell_jacobians, face_jacobians;
      DiFfRG::internal::SeedStackWorkspace<Batch, n_fe> cell_workspace, face_workspace;
    };
  } // namespace CG
} // namespace DiFfRG
