#pragma once

// standard library
#include <functional>
#include <memory>
#include <sstream>

// external libraries
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/assembler_core.hh>
#include <DiFfRG/discretization/common/batched_scatter.hh>
#include <DiFfRG/discretization/common/cell_geometry.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/common/phase_times.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/model/abs_tolerances.hh>
#include <DiFfRG/model/batch.hh>

namespace DiFfRG
{
  namespace LDG
  {
    using namespace dealii;
    using std::array, std::vector, std::unique_ptr;

    /**
     * @brief The LDG assembler: the FE functions (level 0) plus up to three LDG levels, level k built from
     * level k - 1 by the model's ldg_flux / ldg_source and a mass-matrix solve.
     *
     * Each level, and the main level, is assembled in three phases, as in CG::Assembler:
     *  1. gather: the values of the previous level (main level: of all levels) at every quadrature point of the
     *     cells, boundary faces and interior faces (both traces), into a batch;
     *  2. evaluate: ldg_evaluate_batch<k> (main level: evaluate_batch) and the numerical fluxes over the whole
     *     batch, or their seed-stacked AD jacobians, see model/batch.hh; each interior face once;
     *  3. scatter: every cell contracts its cell term and its share of each of its faces into its own rows
     *     (internal::ColoredCells).
     *
     * The jacobian is J = weight (J_uu + sum_k J_ug[k] J_gu[k]) + mass terms, with J_gu[k] = d(level k)/du built from
     * the level jacobians by the chain rule and cached while the model declares them constant.
     *
     * Config: /discretization/batched/max_stacked_points bounds the size of one AD evaluation of the
     * jacobian (default: 256 MB of AD inputs and outputs), see DiFfRG::internal::seed_stacked_jacobian.
     */
    template <typename Discretization_,
              typename Model_ = typename DiFfRG::internal::assembler_model_of<Discretization_>::type>
    class Assembler
        : public DiFfRG::internal::AssemblerCore<Assembler<Discretization_, Model_>, Discretization_, Model_>
    {
      using Base = DiFfRG::internal::AssemblerCore<Assembler<Discretization_, Model_>, Discretization_, Model_>;
      friend Base;

    public:
      using Discretization = Discretization_;
      using Model = Model_;
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;
      using SparseMatrixType = typename Discretization::SparseMatrixType;

      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;
      static constexpr uint stencil = Components::count_fe_subsystems();
      static constexpr uint n_levels = stencil;
      static constexpr size_t n_fe = Components::count_fe_functions(0);
      static constexpr size_t n_extr = Components::count_extractors();
      using Extractors = std::array<NumberType, n_extr>;
      /// The batch of the main level: the values of all levels.
      using MainBatch = LDGPointBatch<dim, NumberType, Components, Extractors, VectorType>;
      static constexpr size_t n_all = MainBatch::n_fe_functions;
      /// The batch level `from` hands to the construction of level from + 1: its values, nothing shared.
      template <uint from>
      using LevelBatch = PointBatch<dim, NumberType, Components::count_fe_functions(from), std::array<NumberType, 0>,
                                    VectorType, false, false>;

      using PhaseTimes = AssemblyPhaseTimes;

    private:
      template <typename... T> auto fe_more_conv(std::tuple<T &...> &t) const
      {
        if constexpr (stencil == 2)
          return named_tuple<std::tuple<T &...>,
                             StringSet<"fe_functions", "LDG1", "fe_derivatives", "fe_hessians", "extractors",
                                       "variables", "potential", "potential_gradient", "potential_hessian">>(t);
        else if constexpr (stencil == 3)
          return named_tuple<std::tuple<T &...>,
                             StringSet<"fe_functions", "LDG1", "LDG2", "fe_derivatives", "fe_hessians", "extractors",
                                       "variables", "potential", "potential_gradient", "potential_hessian">>(t);
        else if constexpr (stencil == 4)
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "LDG1", "LDG2", "LDG3", "fe_derivatives",
                                                           "fe_hessians", "extractors", "variables", "potential",
                                                           "potential_gradient", "potential_hessian">>(t);
        else
          throw std::runtime_error("Only <= 3 LDG subsystems are supported.");
      }

      template <typename... T> auto ref_conv(std::tuple<T &...> &t) const
      {
        if constexpr (stencil == 2)
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "LDG1">>(t);
        else if constexpr (stencil == 3)
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "LDG1", "LDG2">>(t);
        else if constexpr (stencil == 4)
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "LDG1", "LDG2", "LDG3">>(t);
        else
          throw std::runtime_error("Only <= 3 LDG subsystems are supported.");
      }

    public:
      Assembler(Discretization &discretization, Model &model, const ConfigTree &config)
          : Base(discretization, model, config, "LDG"),
            quadrature(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
            quadrature_face(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
            dof_handler_list(discretization.get_dof_handler_list()),
            max_stacked_points(Base::read_max_stacked_points(config, stacked_point_bytes, Base::fem_stacked_budget))
      {
        static_assert(Components::count_fe_subsystems() > 1, "LDG must have a submodel with index 1.");
        reinit();
      }

      virtual void reinit_vector(VectorType &vec) const override { vec.reinit(dof_handler.n_dofs()); }
      // LDG stays on the serial policy (see LDG::Discretization for why), so this is the plain
      // pattern-based reinit rather than a policy call.
      virtual void reinit_matrix(SparseMatrixType &matrix) const override { matrix.reinit(sparsity_pattern_jacobian); }
      /**
       * @brief Per-dof absolute tolerances from Model::abs_tolerances, evaluated at each dof's support point on
       * the FE solution (values, gradients, hessians of the primary FE functions; not the LDG auxiliaries).
       * @see AbstractAssembler::local_abs_tolerances. Serial vectors only.
       */
      virtual bool local_abs_tolerances(VectorType &atol, const VectorType &solution, double abs_tol,
                                        double rel_tol) const override
      {
        constexpr uint n = Components::count_fe_functions(0);
        using Values = std::array<NumberType, n>;
        using Gradients = std::array<Tensor<1, dim, NumberType>, n>;
        using Hessians = std::array<Tensor<2, dim, NumberType>, n>;
        using Solution = named_tuple<std::tuple<Values &, Gradients &, Hessians &>,
                                     StringSet<"fe_functions", "fe_derivatives", "fe_hessians">>;
        if constexpr (!def::HasAbsTolerances<Model, dim, Solution, n> ||
                      !std::is_same_v<VectorType, dealii::Vector<NumberType>>)
          return false;
        else {
          atol.reinit(dof_handler.n_dofs());
          const dealii::Quadrature<dim> support(fe.get_unit_support_points());
          FEValues<dim> fe_v(mapping, fe, support,
                             update_values | update_gradients | update_hessians | update_quadrature_points);
          std::vector<Vector<NumberType>> vals(support.size(), Vector<NumberType>(n));
          std::vector<std::vector<Tensor<1, dim, NumberType>>> grads(support.size(),
                                                                     std::vector<Tensor<1, dim, NumberType>>(n));
          std::vector<std::vector<Tensor<2, dim, NumberType>>> hess(support.size(),
                                                                    std::vector<Tensor<2, dim, NumberType>>(n));
          std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
          Values v{};
          Gradients g{};
          Hessians h{};
          std::array<double, n> point_atol{};
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (!cell->is_locally_owned()) continue;
            fe_v.reinit(cell);
            fe_v.get_function_values(solution, vals);
            fe_v.get_function_gradients(solution, grads);
            fe_v.get_function_hessians(solution, hess);
            cell->get_dof_indices(dofs);
            // the quadrature is the list of dof support points, so point q belongs to dof q
            for (uint q = 0; q < support.size(); ++q) {
              for (uint c = 0; c < n; ++c) {
                v[c] = vals[q][c];
                g[c] = grads[q][c];
                h[c] = hess[q][c];
              }
              point_atol.fill(abs_tol);
              model.abs_tolerances(point_atol, fe_v.quadrature_point(q), Solution(std::tie(v, g, h)), abs_tol, rel_tol);
              atol[dofs[q]] = point_atol[fe.system_to_component_index(q).first];
            }
          }
          return true;
        }
      }

      using Base::attach_data_output;
      /// The FE functions, the readouts and every LDG level.
      virtual void attach_data_output(OutputFrame<dim, VectorType> &data_out, const VectorType &solution,
                                      const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                      const VectorType &residual = VectorType()) override
      {
        // The readouts in Base::attach_data_output build the levels attached below.
        Base::attach_data_output(data_out, solution, variables, dt_solution, residual);
        auto fe_out = data_out.fields();
        for (uint k = 1; k < Components::count_fe_subsystems(); ++k) {
          sol_vector_vec_tmp[k] = sol_vector[k];
          fe_out.attach(*dof_handler_list[k], sol_vector_vec_tmp[k], "LDG" + std::to_string(k));
        }
      }

      virtual void reinit() override
      {
        const auto init_mass = [&](uint i) {
          // build the sparsity of the mass matrix and mass matrix of all ldg levels
          auto dofs_per_component = DoFTools::count_dofs_per_fe_component(*(dof_handler_list[i]));

          if (i == 0) {
            auto n_fe = dofs_per_component.size();
            for (uint j = 1; j < n_fe; ++j)
              if (dofs_per_component[j] != dofs_per_component[0])
                throw std::runtime_error("For LDG the FE basis of all systems must be equal!");

            BlockDynamicSparsityPattern dsp(n_fe, n_fe);
            for (uint i = 0; i < n_fe; ++i)
              for (uint j = 0; j < n_fe; ++j)
                dsp.block(i, j).reinit(dofs_per_component[0], dofs_per_component[0]);
            dsp.collect_sizes();
            DoFTools::make_sparsity_pattern(*(dof_handler_list[0]), dsp, discretization.get_constraints(0), true);
            sparsity_pattern_mass.copy_from(dsp);

            component_mass_matrix_inverse.reinit(sparsity_pattern_mass.block(0, 0));
            mass_matrix.reinit(sparsity_pattern_mass);

            MatrixCreator::create_mass_matrix(*(dof_handler_list[0]), quadrature, mass_matrix,
                                              (Function<dim, NumberType> *)nullptr, discretization.get_constraints(0));
            build_inverse(mass_matrix.block(0, 0), component_mass_matrix_inverse);
            sol_block.reinit(dofs_per_component);
          } else {
            sol_vector[i].reinit(dofs_per_component);
            sol_vector_tmp[i].reinit(dofs_per_component);
            ldg_matrix_built[i] = false;
          }
        };

        const auto init_jacobian = [&](uint i) {
          // build the jacobian and subjacobians
          if (i == 0) {
            build_ldg_sparsity(sparsity_pattern_jacobian, *(dof_handler_list[0]), *(dof_handler_list[0]), stencil,
                               true);
            for (uint k = 1; k < Components::count_fe_subsystems(); ++k)
              jacobian_tmp[k].reinit(sparsity_pattern_jacobian);
          } else {
            build_ldg_sparsity(sparsity_pattern_ug[i], *(dof_handler_list[0]), *(dof_handler_list[i]), 1);
            j_ug[i].reinit(sparsity_pattern_ug[i]);
          }
        };

        auto init_ldg = [&](uint i) {
          // build the subjacobian sparsity patterns of all matrices that contribute to the jacobian = uu + ug*gu
          build_ldg_sparsity(sparsity_pattern_gu[i], *(dof_handler_list[i]), *(dof_handler_list[0]), i);
          j_gu[i].reinit(sparsity_pattern_gu[i]);

          // these are the "in-between" dependencies of the ldg levels
          build_ldg_sparsity(sparsity_pattern_wg[i], *(dof_handler_list[i]), *(dof_handler_list[i - 1]), 1);
          j_wg[i].reinit(sparsity_pattern_wg[i]);
          j_wg_tmp[i].reinit(sparsity_pattern_wg[i]);
        };

        Timer timer;

        this->reinit_common();

        vector<std::thread> init_threads;
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i)
          init_threads.emplace_back(std::thread(init_mass, i));
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i)
          init_threads.emplace_back(std::thread(init_jacobian, i));
        for (uint i = 1; i < Components::count_fe_subsystems(); ++i)
          init_threads.emplace_back(std::thread(init_ldg, i));
        for (auto &t : init_threads)
          t.join();

        setup_cells();
        timings_reinit.push_back(timer.wall_time());
      }

      void rebuild_jacobian_sparsity()
      {
        build_ldg_sparsity(sparsity_pattern_jacobian, *(dof_handler_list[0]), *(dof_handler_list[0]), stencil, true);
        for (uint k = 1; k < Components::count_fe_subsystems(); ++k)
          jacobian_tmp[k].reinit(sparsity_pattern_jacobian);
      }

      /**
       * @brief The model's cell_indicator integrated over each cell, plus its face_indicator integrated over each
       * interior face, added to the two cells of the face; every level's values at the points.
       *
       * Each interior face is evaluated once; then every owned cell adds its own integral and its shares of its faces
       * into its own entry.
       */
      void refinement_indicator(Vector<double> &indicator, const VectorType &solution_global)
      {
        constexpr uint n_sub = Components::count_fe_subsystems();
        struct IndicatorScratch {
          IndicatorScratch(const Mapping<dim> &mapping, const vector<const DoFHandler<dim> *> &dofh,
                           const dealii::Quadrature<dim> &q, const dealii::Quadrature<dim - 1> &q_face)
          {
            const UpdateFlags flags = update_values | update_quadrature_points | update_JxW_values;
            for (uint i = 0; i < n_sub; ++i) {
              const auto &fe = dofh[i]->get_fe();
              fe_values[i] = std::make_unique<FEValues<dim>>(mapping, fe, q, flags);
              fe_interface_values[i] =
                  std::make_unique<FEInterfaceValues<dim>>(mapping, fe, q_face, flags | update_normal_vectors);
              solution[i].resize(q.size(), Vector<NumberType>(fe.n_components()));
              for (auto &side : solution_interface)
                side[i].resize(q_face.size(), Vector<NumberType>(fe.n_components()));
            }
          }
          array<unique_ptr<FEValues<dim>>, n_sub> fe_values;
          array<unique_ptr<FEInterfaceValues<dim>>, n_sub> fe_interface_values;
          array<vector<Vector<NumberType>>, n_sub> solution;
          array<array<vector<Vector<NumberType>>, n_sub>, 2> solution_interface;
        };
        rebuild_ldg_vectors(solution_global);
        // Level 0 is the solution, the levels above it the LDG vectors (of another type).
        const auto read = [&](const auto &fe_v, const uint i, vector<Vector<NumberType>> &values) {
          if (i == 0)
            fe_v.get_function_values(solution_global, values);
          else
            fe_v.get_function_values(sol_vector[i], values);
        };
        tbb::enumerable_thread_specific<std::unique_ptr<IndicatorScratch>> indicator_scratch(
            [&] { return std::make_unique<IndicatorScratch>(mapping, dof_handler_list, quadrature, quadrature_face); });
        // map() is collective and each rank visits only its own cells; see NoMapsHere.
        const NoMapsHere no_maps_during_assembly;

        std::vector<std::array<double, 2>> face_values(topology.faces.size());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, topology.faces.size()), [&](const auto &r) {
          auto &s = *indicator_scratch.local();
          for (size_t f = r.begin(); f != r.end(); ++f) {
            const auto &face = topology.faces[f];
            for (uint i = 0; i < n_sub; ++i) {
              const auto &fe_iv = topology.reinit(*s.fe_interface_values[i], f, *dof_handler_list[i]);
              read(fe_iv.get_fe_face_values(0), i, s.solution_interface[0][i]);
              read(fe_iv.get_fe_face_values(1), i, s.solution_interface[1][i]);
            }
            const auto &fe_iv = *s.fe_interface_values[0];
            array<double, 2> local{};
            face_values[f] = {0., 0.};
            for (const auto &q : fe_iv.quadrature_point_indices()) {
              auto sol_q_s = local_sol_q(s.solution_interface[0], q);
              auto sol_q_n = local_sol_q(s.solution_interface[1], q);
              model.face_indicator(local, fe_iv.normal_vector(q), fe_iv.quadrature_point(q), ref_conv(sol_q_s),
                                   ref_conv(sol_q_n));
              face_values[f][0] += fe_iv.JxW(q) * local[0] * (1. + face.cell[0]->at_boundary());
              face_values[f][1] += fe_iv.JxW(q) * local[1] * (1. + face.cell[1]->at_boundary());
            }
          }
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cells[0].size()), [&](const auto &r) {
          auto &s = *indicator_scratch.local();
          for (size_t k = r.begin(); k != r.end(); ++k) {
            const auto &cell = cells[0][k];
            for (uint i = 0; i < n_sub; ++i) {
              s.fe_values[i]->reinit(DiFfRG::internal::on(*dof_handler_list[i], cell));
              read(*s.fe_values[i], i, s.solution[i]);
            }
            const auto &fe_v = *s.fe_values[0];
            double value = 0., local = 0.;
            for (const auto &q : fe_v.quadrature_point_indices()) {
              auto sol_q = local_sol_q(s.solution, q);
              model.cell_indicator(local, fe_v.quadrature_point(q), ref_conv(sol_q));
              value += fe_v.JxW(q) * local;
            }
            for (size_t i = topology.face_ref_begin[k]; i < topology.face_ref_begin[k + 1]; ++i)
              value += face_values[topology.face_refs[i].face][topology.face_refs[i].side];
            indicator[cell->active_cell_index()] += value;
          }
        });
      }

      /**
       * @brief Construct the mass
       *
       * @param residual The result is stored here.
       * @param solution_global The current global solution.
       * @param weight A factor to multiply the whole residual with.
       */
      virtual void mass(VectorType &residual, const VectorType &solution_global, const VectorType &solution_global_dot,
                        NumberType weight) override
      {
        scatter_residual(residual, [&](const size_t, Scratch &s, Vector<NumberType> &r) {
          const auto &fe_v = *s.fe_v[0];
          fe_v.get_function_values(solution_global, s.values[0]);
          fe_v.get_function_values(solution_global_dot, s.values_dot);
          array<NumberType, n_fe> m{};
          for (const auto &q : fe_v.quadrature_point_indices()) {
            model.mass(m, fe_v.quadrature_point(q), s.values[0][q], s.values_dot[q]);
            for (uint i = 0; i < r.size(); ++i)
              r(i) += weight * fe_v.JxW(q) * fe_v.shape_value_component(i, q, s.comp[0][i]) * m[s.comp[0][i]];
          }
        });
      }

      /**
       * @brief Construct the system residual, i.e. Res = grad(flux) - source
       *
       * @param residual The result is stored here.
       * @param solution_global The current global solution.
       * @param weight A factor to multiply the whole residual with.
       */
      virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                            const VectorType &solution_global_dot, NumberType weight_mass,
                            const VectorType &variables = VectorType()) override
      {
        DiFfRG::internal::PhaseTimer phase(residual_times);
        // Find the EoM and extract whatever data is needed for the model; extract() builds the LDG levels.
        Extractors extracted_data{{}};
        if constexpr (n_extr > 0)
          this->extract(extracted_data, solution_global, variables, true, false, true);
        else
          rebuild_ldg_vectors(solution_global);
        phase.lap(&PhaseTimes::extract);
        gather_main(solution_global, extracted_data, variables);
        phase.lap(&PhaseTimes::gather);
        {
          const NoMapsHere no_maps_during_assembly; // map() is collective, the points are rank-local
          cell_result.reinit(cell_batch.size(), Term::flux | Term::source);
          model.evaluate_batch(cell_result, cell_batch);
          boundary_result.reinit(boundary_batch.size(), Term::flux);
          evaluate_boundary_numflux(model, boundary_result, boundary_normals, boundary_batch);
          face_result.reinit(face_batch[0].size(), Term::flux);
          evaluate_numflux(model, face_result, face_normals, face_batch[0], face_batch[1]);
        }
        phase.lap(&PhaseTimes::evaluate);
        scatter_residual(residual, [&](const size_t k, Scratch &s, Vector<NumberType> &r) {
          const auto &fe_v = *s.fe_v[0];
          const auto &comp = s.comp[0];
          fe_v.get_function_values(solution_global, s.values[0]);
          fe_v.get_function_values(solution_global_dot, s.values_dot);
          array<NumberType, n_fe> mass{};
          for (const auto &q : fe_v.quadrature_point_indices()) {
            const size_t p = k * n_q + q;
            model.mass(mass, fe_v.quadrature_point(q), s.values[0][q], s.values_dot[q]);
            for (uint i = 0; i < r.size(); ++i) {
              const auto c = comp[i];
              Tensor<1, dim, NumberType> flux;
              for (uint d = 0; d < dim; ++d)
                flux[d] = cell_result.flux(c, d)[p];
              r(i) += fe_v.JxW(q) * (weight * (-scalar_product(fe_v.shape_grad_component(i, q, c), flux) +
                                               fe_v.shape_value_component(i, q, c) * cell_result.source(c)[p]) +
                                     weight_mass * fe_v.shape_value_component(i, q, c) * mass[c]);
            }
          }
          add_face_terms(s, k, 0, r, boundary_result, face_result, weight);
        });
        phase.lap(&PhaseTimes::scatter);
        timings_residual.push_back(phase.finish());
      }

      virtual void jacobian_mass(BlockSparseMatrix<NumberType> &jacobian, const VectorType &solution_global,
                                 const VectorType &solution_global_dot, NumberType alpha = 1.,
                                 NumberType beta = 1.) override
      {
        Timer timer;
        scatter_jacobian(jacobian, false, [&](const size_t, Scratch &s, LocalData &out) {
          add_mass_jacobian(s, out.blocks[0][0], solution_global, solution_global_dot, alpha, beta);
        });
        timings_jacobian.push_back(timer.wall_time());
      }

      /**
       * @brief Construct the system jacobian, i.e. dRes/du
       *
       * @param jacobian The result is stored here.
       * @param solution_global The current global solution.
       * @param weight A factor to multiply the whole jacobian with.
       */
      virtual void jacobian(BlockSparseMatrix<NumberType> &jacobian, const VectorType &solution_global,
                            NumberType weight, const VectorType &solution_global_dot, NumberType alpha, NumberType beta,
                            const VectorType &variables = VectorType()) override
      {
        DiFfRG::internal::PhaseTimer phase(jacobian_times);
        // Find the EoM and extract whatever data is needed for the model; extract() builds the LDG levels.
        Extractors extracted_data{{}};
        if constexpr (n_extr > 0) {
          if (this->extract_with_jacobian(extracted_data, solution_global, variables)) this->reinit_matrix(jacobian);
        } else
          rebuild_ldg_vectors(solution_global);
        // The levels are current now, so the jacobians built from them are, too.
        constexpr_for<1, n_levels, 1>([&](auto k) {
          if (!ldg_matrix_built[k] || !model.get_components().jacobians_constant(k, k - 1))
            rebuild_ldg_jacobian<k>(solution_global);
        });
        phase.lap(&PhaseTimes::extract);
        gather_main(solution_global, extracted_data, variables);
        phase.lap(&PhaseTimes::gather);
        {
          const NoMapsHere no_maps_during_assembly; // map() is collective, the points are rank-local
          evaluate_flux_source_jacobian(model, cell_jacobians, cell_batch, max_stacked_points, cell_workspace);
          evaluate_boundary_numflux_jacobian(model, boundary_jacobians, boundary_normals, boundary_batch,
                                             max_stacked_points, boundary_workspace);
          evaluate_numflux_jacobian(model, face_jacobians, face_normals, face_batch[0], face_batch[1],
                                    max_stacked_points, face_workspace);
        }
        require_finite(cell_jacobians);
        require_finite(boundary_jacobians);
        require_finite(face_jacobians);
        phase.lap(&PhaseTimes::evaluate);
        for (uint k = 1; k < n_levels; ++k)
          j_ug[k] = 0;
        scatter_jacobian(jacobian, true, [&](const size_t k, Scratch &s, LocalData &out) {
          add_mass_jacobian(s, out.blocks[0][0], solution_global, solution_global_dot, alpha, beta);
          const auto &fe_v = *s.fe_v[0];
          for (const auto &q : fe_v.quadrature_point_indices()) {
            s.cache(0, [&](const uint l) -> const FEValuesBase<dim> & { return *s.fe_v[l]; }, q);
            add_cell_jacobian(s, out, cell_jacobians[k * n_q + q], weight * fe_v.JxW(q));
          }
          for (size_t f = topology.boundary_begin[k]; f < topology.boundary_begin[k + 1]; ++f) {
            const auto &fe_fv = reinit_boundary(s, k, f);
            for (const auto &q : fe_fv.quadrature_point_indices()) {
              s.cache(0, [&](const uint l) -> const FEValuesBase<dim> & { return *s.fe_fv[l]; }, q);
              add_face_jacobian(s, out, 0, 0, boundary_jacobians[f * n_q_face + q], fe_fv.normal_vector(q),
                                weight * fe_fv.JxW(q), true);
            }
          }
          for (size_t ref = topology.face_ref_begin[k]; ref < topology.face_ref_begin[k + 1]; ++ref) {
            const auto [f, side] = topology.face_refs[ref];
            const uint slot = 1 + ref - topology.face_ref_begin[k];
            const auto &fe_iv0 = reinit_interface(s, f);
            for (uint l = 0; l < n_levels; ++l) {
              out.blocks[l][slot].reinit(out.dofs.size(), dof_handler_list[l]->get_fe().n_dofs_per_cell());
              out.columns[l][slot].resize(dof_handler_list[l]->get_fe().n_dofs_per_cell());
              DiFfRG::internal::on(*dof_handler_list[l], topology.faces[f].cell[1 - side])
                  ->get_dof_indices(out.columns[l][slot]);
            }
            const double sign = side == 0 ? 1. : -1.;
            for (const auto &q : fe_iv0.quadrature_point_indices()) {
              for (uint t = 0; t < 2; ++t)
                s.cache(
                    t, [&](const uint l) -> const FEValuesBase<dim> & { return s.fe_iv[l]->get_fe_face_values(t); }, q);
              const auto &Jq = face_jacobians[f * n_q_face + q];
              const NumberType w = sign * weight * fe_iv0.JxW(q);
              add_face_jacobian(s, out, side, side, Jq[side], fe_iv0.normal_vector(q), w, false, 0);
              add_face_jacobian(s, out, side, 1 - side, Jq[1 - side], fe_iv0.normal_vector(q), w, false, slot);
              add_face_extractor_jacobian(s, out, side, Jq[0], fe_iv0.normal_vector(q), w);
            }
          }
        });

        // The chain rule through the levels: d(main)/d(level k) * d(level k)/du.
        for (uint k = 1; k < n_levels; ++k) {
          jacobian_tmp[k] = 0;
          tbb::parallel_for(tbb::blocked_range<uint>(0, n_fe), [&](const tbb::blocked_range<uint> &r) {
            for (uint q = r.begin(); q < r.end(); ++q)
              for (const auto &c : model.get_components().ldg_couplings(k, 0))
                j_ug[k].block(q, c[0]).mmult(jacobian_tmp[k].block(q, c[1]), j_gu[k].block(c[0], c[1]),
                                             Vector<NumberType>(), false);
          });
        }
        tbb::parallel_for(tbb::blocked_range<uint>(0, n_fe * n_fe), [&](const tbb::blocked_range<uint> &r) {
          for (uint b = r.begin(); b < r.end(); ++b)
            for (uint k = 1; k < n_levels; ++k)
              jacobian.block(b / n_fe, b % n_fe).add(NumberType(1.), jacobian_tmp[k].block(b / n_fe, b % n_fe));
        });
        phase.lap(&PhaseTimes::scatter);
        timings_jacobian.push_back(phase.finish());
      }

    protected:
      using Base::discretization;
      using Base::dof_handler;
      using Base::fe;
      using Base::mapping;
      using Base::model;

      QGauss<dim> quadrature;
      QGauss<dim - 1> quadrature_face;

      std::vector<const DoFHandler<dim> *> dof_handler_list;

      mutable array<BlockVector<NumberType>, Components::count_fe_subsystems()> sol_vector;
      mutable array<BlockVector<NumberType>, Components::count_fe_subsystems()> sol_vector_tmp;
      mutable array<Vector<NumberType>, Components::count_fe_subsystems()> sol_vector_vec_tmp;
      mutable BlockVector<NumberType> sol_block; // FE solution in component blocks, for j_gu[k].vmult

      array<BlockSparsityPattern, Components::count_fe_subsystems()> sparsity_pattern_ug;
      array<BlockSparsityPattern, Components::count_fe_subsystems()> sparsity_pattern_gu;
      array<BlockSparsityPattern, Components::count_fe_subsystems()> sparsity_pattern_wg;

      array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> jacobian_tmp;

      SparseMatrix<NumberType> component_mass_matrix_inverse;
      array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_ug;
      mutable array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_gu;
      mutable array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_wg;
      mutable array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_wg_tmp;

      /// Whether j_gu[k] has been built; it is rebuilt on every jacobian unless the model declares level k's
      /// jacobian constant.
      mutable array<bool, Components::count_fe_subsystems()> ldg_matrix_built{};

      using Base::extractor_dof_indices;
      using Base::jacobian_times;
      using Base::mass_matrix;
      using Base::residual_times;
      using Base::sparsity_pattern_jacobian;
      using Base::sparsity_pattern_mass;
      using Base::timings_jacobian;
      using Base::timings_reinit;
      using Base::timings_residual;

    private:
      /// AD inputs and outputs of one point of a seed-stacked evaluation; see Base::fem_stacked_budget.
      static constexpr size_t stacked_point_bytes = sizeof(autodiff::real) * (n_all + n_fe * (1 + dim) + dim);

      template <typename Js> static void require_finite(const Js &J)
      {
        const bool finite = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, J.size()), true,
            [&](const tbb::blocked_range<size_t> &r, bool ok) {
              for (size_t i = r.begin(); ok && i != r.end(); ++i) {
                if constexpr (requires { J[i].is_finite(); })
                  ok = J[i].is_finite();
                else
                  ok = J[i][0].is_finite() && J[i][1].is_finite();
              }
              return ok;
            },
            std::logical_and<bool>());
        if (!finite) throw std::runtime_error("Infinity encountered in jacobian construction");
      }

      /// One cell's contribution to its own rows (of level 0, or of the level being built), before insertion.
      struct LocalData {
        std::vector<types::global_dof_index> dofs;
        /// Whether a dof of the cell is constrained, i.e. insertion has to go through the constraints.
        bool constrained = false;
        Vector<NumberType> residual;
        /// blocks[l][0]: columns of level l of the cell itself, blocks[l][1 + f]: of the neighbor across its
        /// f-th interior face; columns[l][...] the matching dof indices.
        array<std::vector<FullMatrix<NumberType>>, n_levels> blocks;
        array<std::vector<std::vector<types::global_dof_index>>, n_levels> columns;
        FullMatrix<NumberType> extractor_jacobian;
        FullMatrix<NumberType> extractor_dependence;
      };

      /// Per-thread FE data of every level, for the gathers and the scatters.
      struct Scratch {
        Scratch(const Mapping<dim> &mapping, const std::vector<const DoFHandler<dim> *> &dofhs,
                const dealii::Quadrature<dim> &quadrature, const dealii::Quadrature<dim - 1> &quadrature_face)
        {
          const UpdateFlags flags = update_values | update_gradients | update_quadrature_points | update_JxW_values;
          for (uint l = 0; l < n_levels; ++l) {
            const auto &fe = dofhs[l]->get_fe();
            fe_v[l] = std::make_unique<FEValues<dim>>(mapping, fe, quadrature, flags);
            fe_fv[l] = std::make_unique<FEFaceValues<dim>>(mapping, fe, quadrature_face, flags | update_normal_vectors);
            fe_iv[l] =
                std::make_unique<FEInterfaceValues<dim>>(mapping, fe, quadrature_face, flags | update_normal_vectors);
            values[l].resize(quadrature.size(), Vector<NumberType>(fe.n_components()));
            face_values[l].resize(quadrature_face.size(), Vector<NumberType>(fe.n_components()));
            comp[l].resize(fe.n_dofs_per_cell());
            for (uint i = 0; i < comp[l].size(); ++i)
              comp[l][i] = fe.system_to_component_index(i).first;
            for (uint t = 0; t < 2; ++t)
              shape_values[t][l].resize(fe.n_dofs_per_cell());
          }
          values_dot.resize(quadrature.size(), Vector<NumberType>(n_fe));
          test_gradients.resize(dofhs[0]->get_fe().n_dofs_per_cell());
        }

        /**
         * @brief The shape values of every level at point q into shape_values[t], of the FEValues fe_values(l);
         * for t = 0 also the test functions' gradients (level 0).
         */
        template <typename FEVOf> void cache(const uint t, const FEVOf &fe_values, const uint q)
        {
          for (uint l = 0; l < n_levels; ++l) {
            const auto &fev = fe_values(l);
            for (uint j = 0; j < comp[l].size(); ++j)
              shape_values[t][l][j] = fev.shape_value_component(j, q, comp[l][j]);
          }
          if (t == 0)
            for (uint i = 0; i < comp[0].size(); ++i)
              test_gradients[i] = fe_values(0).shape_grad_component(i, q, comp[0][i]);
        }

        array<std::unique_ptr<FEValues<dim>>, n_levels> fe_v;
        array<std::unique_ptr<FEFaceValues<dim>>, n_levels> fe_fv;
        array<std::unique_ptr<FEInterfaceValues<dim>>, n_levels> fe_iv;
        array<std::vector<uint>, n_levels> comp;
        array<std::vector<Vector<NumberType>>, n_levels> values, face_values;
        std::vector<Vector<NumberType>> values_dot;
        /// shape_values[t][l][j]: shape function j of level l at the current point, on trace t.
        array<array<std::vector<double>, n_levels>, 2> shape_values;
        std::vector<Tensor<1, dim>> test_gradients;
        LocalData local, level_local;
      };

      /// Number the owned cells of every level, their faces, and color the cells for the scatters.
      void setup_cells()
      {
        n_q = quadrature.size();
        n_q_face = quadrature_face.size();
        for (uint l = 0; l < n_levels; ++l)
          cells[l].reinit(*dof_handler_list[l], discretization.get_constraints(l));
        topology.reinit(cells[0]);
        scratch = std::make_unique<tbb::enumerable_thread_specific<Scratch>>(
            [this]() { return Scratch(mapping, dof_handler_list, quadrature, quadrature_face); });
      }

      const FEFaceValues<dim> &reinit_boundary(Scratch &s, const size_t k, const size_t f) const
      {
        for (uint l = 0; l < n_levels; ++l)
          s.fe_fv[l]->reinit(cells[l][k], topology.boundary_faces[f].second);
        return *s.fe_fv[0];
      }

      const FEInterfaceValues<dim> &reinit_interface(Scratch &s, const size_t f) const
      {
        for (uint l = 0; l < n_levels; ++l)
          topology.reinit(*s.fe_iv[l], f, *dof_handler_list[l]);
        return *s.fe_iv[0];
      }

      /// The level-l vector: the solution for l = 0, else the LDG level built from it.
      template <uint l> const auto &level_vector(const VectorType &solution) const
      {
        if constexpr (l == 0)
          return solution;
        else
          return sol_vector[l];
      }

      /**
       * @brief Fill batches at the points of the cells, boundary faces and interior faces: store(batch, first,
       * fe_values, fe_values_of_level, values_buffer_of_level, width) is called per cell / face (side) with the
       * FEValues of every level.
       */
      template <typename Batch, typename Store>
      void gather(Batch &cells_b, Batch &boundary_b, std::array<Batch, 2> &faces_b, const Store &store)
      {
        cells_b.reinit(cells[0].size() * n_q);
        boundary_b.reinit(topology.boundary_faces.size() * n_q_face);
        for (auto &b : faces_b)
          b.reinit(topology.faces.size() * n_q_face);
        boundary_normals.resize(boundary_b.size());
        face_normals.resize(faces_b[0].size());

        const auto geometry = [](Batch &batch, const size_t first, const auto &fev, const double width) {
          for (const auto &q : fev.quadrature_point_indices()) {
            for (uint d = 0; d < dim; ++d)
              batch.coordinate(d, first + q) = fev.quadrature_point(q)[d];
            batch.width(first + q) = width;
          }
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cells[0].size()), [&](const tbb::blocked_range<size_t> &r) {
          auto &s = scratch->local();
          for (size_t k = r.begin(); k != r.end(); ++k) {
            for (uint l = 0; l < n_levels; ++l)
              s.fe_v[l]->reinit(cells[l][k]);
            geometry(cells_b, k * n_q, *s.fe_v[0], DiFfRG::internal::cell_width(cells[0][k]));
            store(
                cells_b, k * n_q, [&](const uint l) -> const FEValuesBase<dim> & { return *s.fe_v[l]; },
                [&](const uint l) -> auto & { return s.values[l]; });
          }
        });
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, topology.boundary_faces.size()), [&](const tbb::blocked_range<size_t> &r) {
              auto &s = scratch->local();
              for (size_t f = r.begin(); f != r.end(); ++f) {
                const auto &cell = topology.boundary_faces[f].first;
                for (uint l = 0; l < n_levels; ++l)
                  s.fe_fv[l]->reinit(DiFfRG::internal::on(*dof_handler_list[l], cell),
                                     topology.boundary_faces[f].second);
                geometry(boundary_b, f * n_q_face, *s.fe_fv[0], DiFfRG::internal::cell_width(cell));
                store(
                    boundary_b, f * n_q_face, [&](const uint l) -> const FEValuesBase<dim> & { return *s.fe_fv[l]; },
                    [&](const uint l) -> auto & { return s.face_values[l]; });
                for (uint q = 0; q < n_q_face; ++q)
                  boundary_normals[f * n_q_face + q] = s.fe_fv[0]->normal_vector(q);
              }
            });
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, topology.faces.size()), [&](const tbb::blocked_range<size_t> &r) {
              auto &s = scratch->local();
              for (size_t f = r.begin(); f != r.end(); ++f) {
                const auto &fe_iv0 = reinit_interface(s, f);
                for (uint side = 0; side < 2; ++side) {
                  geometry(faces_b[side], f * n_q_face, fe_iv0.get_fe_face_values(side),
                           DiFfRG::internal::cell_width(topology.faces[f].cell[side]));
                  store(
                      faces_b[side], f * n_q_face,
                      [&](const uint l) -> const FEValuesBase<dim> & { return s.fe_iv[l]->get_fe_face_values(side); },
                      [&](const uint l) -> auto & { return s.face_values[l]; });
                }
                for (uint q = 0; q < n_q_face; ++q)
                  face_normals[f * n_q_face + q] = fe_iv0.normal_vector(q);
              }
            });
      }

      /// Phase 1 of the main level: the values of all levels at every point.
      void gather_main(const VectorType &solution_global, const Extractors &extracted_data, const VectorType &variables)
      {
        gather(cell_batch, boundary_batch, face_batch,
               [&](MainBatch &batch, const size_t first, const auto &fe_values, const auto &buffer) {
                 constexpr_for<0, n_levels, 1>([&](auto l) {
                   auto &vals = buffer(l);
                   fe_values(l).get_function_values(this->template level_vector<l>(solution_global), vals);
                   for (uint q = 0; q < vals.size(); ++q)
                     for (uint c = 0; c < Components::count_fe_functions(l); ++c)
                       batch.ldg_value(l, c, first + q) = vals[q][c];
                 });
               });
        for (auto *b : {&cell_batch, &boundary_batch, &face_batch[0], &face_batch[1]})
          b->set_shared(extracted_data, variables);
      }

      /// Phase 1 of level `from + 1`: the values of level `from` at every point.
      template <uint from, typename Source>
      void gather_level(LevelBatch<from> &cells_b, LevelBatch<from> &boundary_b,
                        std::array<LevelBatch<from>, 2> &faces_b, const Source &source)
      {
        gather(cells_b, boundary_b, faces_b,
               [&](LevelBatch<from> &batch, const size_t first, const auto &fe_values, const auto &buffer) {
                 auto &vals = buffer(from);
                 fe_values(from).get_function_values(source, vals);
                 for (uint q = 0; q < vals.size(); ++q)
                   for (uint c = 0; c < Components::count_fe_functions(from); ++c)
                     batch.value(c, first + q) = vals[q][c];
               });
        for (auto *b : {&cells_b, &boundary_b, &faces_b[0], &faces_b[1]})
          b->set_shared(no_extractors, no_variables);
      }

      /**
       * @brief Phase 3: assemble(k, scratch, local) every owned cell of level `level` (the level of the rows), after
       * reinit of its FEValues on every level and of its dof indices, and insert(local) the result; see
       * internal::ColoredCells::scatter.
       */
      template <typename Global, typename Assemble, typename Insert>
      void scatter(const uint level, LocalData Scratch::*local, Global &global, const Assemble &assemble,
                   const Insert &insert)
      {
        cells[level].scatter(
            global, *scratch, local, local_buffer,
            [&](const size_t k, Scratch &s, LocalData &data) {
              for (uint l = 0; l < n_levels; ++l)
                s.fe_v[l]->reinit(cells[l][k]);
              data.dofs.resize(dof_handler_list[level]->get_fe().n_dofs_per_cell());
              cells[level][k]->get_dof_indices(data.dofs);
              data.constrained = cells[level].is_constrained(k);
              assemble(k, s, data);
            },
            insert);
      }

      template <typename Assemble> void scatter_residual(VectorType &residual, const Assemble &assemble)
      {
        const auto &constraints = discretization.get_constraints(0);
        scatter(
            0, &Scratch::local, residual,
            [&](const size_t k, Scratch &s, LocalData &local) {
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

      /**
       * @brief The rows of level 0 of the jacobian: columns of level 0 into @p jacobian, through the constraints;
       * with @p with_levels, columns of level l >= 1 into j_ug[l] and the extractor dependence into @p jacobian.
       */
      template <typename Assemble>
      void scatter_jacobian(BlockSparseMatrix<NumberType> &jacobian, const bool with_levels, const Assemble &assemble)
      {
        const auto &constraints = discretization.get_constraints(0);
        const bool any_constraints = constraints.n_constraints() > 0;
        scatter(
            0, &Scratch::local, jacobian,
            [&](const size_t k, Scratch &s, LocalData &local) {
              const size_t n_blocks = 1 + topology.n_interior_faces(k);
              for (uint l = 0; l < (with_levels ? n_levels : 1); ++l) {
                local.blocks[l].resize(n_blocks);
                local.columns[l].resize(n_blocks);
                local.blocks[l][0].reinit(local.dofs.size(), dof_handler_list[l]->get_fe().n_dofs_per_cell());
                local.columns[l][0].resize(dof_handler_list[l]->get_fe().n_dofs_per_cell());
                cells[l][k]->get_dof_indices(local.columns[l][0]);
                for (size_t b = 1; b < n_blocks; ++b)
                  local.blocks[l][b].reinit(0, 0);
              }
              if constexpr (n_extr > 0) local.extractor_jacobian.reinit(local.dofs.size(), n_extr);
              assemble(k, s, local);
            },
            [&](LocalData &local) {
              if (local.constrained)
                constraints.distribute_local_to_global(local.blocks[0][0], local.dofs, jacobian);
              else
                DiFfRG::internal::add_to_blocks(jacobian, local.dofs, local.dofs, local.blocks[0][0]);
              for (size_t b = 1; b < local.blocks[0].size(); ++b) {
                if (local.blocks[0][b].m() == 0) continue;
                if (any_constraints)
                  constraints.distribute_local_to_global(local.blocks[0][b], local.dofs, local.columns[0][b], jacobian);
                else
                  DiFfRG::internal::add_to_blocks(jacobian, local.dofs, local.columns[0][b], local.blocks[0][b]);
              }
              if (!with_levels) return;
              for (uint l = 1; l < n_levels; ++l)
                for (size_t b = 0; b < local.blocks[l].size(); ++b)
                  if (local.blocks[l][b].m() > 0)
                    DiFfRG::internal::add_to_blocks(j_ug[l], local.dofs, local.columns[l][b], local.blocks[l][b]);
              if constexpr (n_extr > 0) {
                local.extractor_dependence.reinit(local.dofs.size(), extractor_dof_indices.size());
                local.extractor_jacobian.mmult(local.extractor_dependence, this->extractor_jacobian);
                constraints.distribute_local_to_global(local.extractor_dependence, local.dofs, extractor_dof_indices,
                                                       jacobian);
              }
            });
      }

      /**
       * @brief r += w phi_i (F . n) over the boundary faces of cell k and +- w phi_i (NF . n) over its interior
       * faces, for the test functions phi of level @p level; F, NF from @p boundary and @p faces.
       */
      template <typename Out>
      void add_face_terms(Scratch &s, const size_t k, const uint level, Vector<NumberType> &r, const Out &boundary,
                          const Out &faces, const NumberType w) const
      {
        const auto &comp = s.comp[level];
        const auto normal_component = [](const Out &result, const size_t c, const size_t p, const Tensor<1, dim> &n) {
          NumberType value = 0.;
          for (uint d = 0; d < dim; ++d)
            value += result.flux(c, d)[p] * n[d];
          return value;
        };
        for (size_t f = topology.boundary_begin[k]; f < topology.boundary_begin[k + 1]; ++f) {
          reinit_boundary(s, k, f);
          const auto &fe_fv = *s.fe_fv[level];
          for (const auto &q : fe_fv.quadrature_point_indices())
            for (uint i = 0; i < r.size(); ++i)
              r(i) += w * fe_fv.JxW(q) * fe_fv.shape_value_component(i, q, comp[i]) *
                      normal_component(boundary, comp[i], f * n_q_face + q, fe_fv.normal_vector(q));
        }
        // [[phi_i]] * numflux * n: the trace of phi_i on its own side, with a minus sign on side 1.
        for (size_t ref = topology.face_ref_begin[k]; ref < topology.face_ref_begin[k + 1]; ++ref) {
          const auto [f, side] = topology.face_refs[ref];
          reinit_interface(s, f);
          const auto &fe_iv = *s.fe_iv[level];
          const auto &fe_fv = fe_iv.get_fe_face_values(side);
          const double sign = side == 0 ? 1. : -1.;
          for (const auto &q : fe_iv.quadrature_point_indices())
            for (uint i = 0; i < r.size(); ++i)
              r(i) += sign * w * fe_iv.JxW(q) * fe_fv.shape_value_component(i, q, comp[i]) *
                      normal_component(faces, comp[i], f * n_q_face + q, fe_iv.normal_vector(q));
        }
      }

      /// J += alpha d(mass)/d(u_dot) + beta d(mass)/du, contracted with the shape values.
      void add_mass_jacobian(Scratch &s, FullMatrix<NumberType> &J, const VectorType &u, const VectorType &u_dot,
                             const NumberType alpha, const NumberType beta) const
      {
        const auto &fe_v = *s.fe_v[0];
        const auto &comp = s.comp[0];
        fe_v.get_function_values(u, s.values[0]);
        fe_v.get_function_values(u_dot, s.values_dot);
        SimpleMatrix<NumberType, n_fe> j_mass, j_mass_dot;
        for (const auto &q : fe_v.quadrature_point_indices()) {
          model.template jacobian_mass<0>(j_mass, fe_v.quadrature_point(q), s.values[0][q], s.values_dot[q]);
          model.template jacobian_mass<1>(j_mass_dot, fe_v.quadrature_point(q), s.values[0][q], s.values_dot[q]);
          for (uint i = 0; i < J.m(); ++i)
            for (uint j = 0; j < J.n(); ++j)
              J(i, j) += fe_v.JxW(q) * fe_v.shape_value_component(i, q, comp[i]) *
                         fe_v.shape_value_component(j, q, comp[j]) *
                         (alpha * j_mass_dot(comp[i], comp[j]) + beta * j_mass(comp[i], comp[j]));
        }
      }

      /// The main-level jacobian blocks Jq at one cell point, contracted with the cached shapes, times w.
      template <typename PJ>
      void add_cell_jacobian(const Scratch &s, LocalData &out, const PJ &Jq, const NumberType w) const
      {
        const auto &comp0 = s.comp[0];
        const auto &sv = s.shape_values[0];
        for (uint l = 0; l < n_levels; ++l) {
          const auto &comp = s.comp[l];
          auto &M = out.blocks[l][0];
          const size_t offset = MainBatch::level_offset(l);
          for (uint i = 0; i < comp0.size(); ++i) {
            const auto ci = comp0[i];
            for (uint j = 0; j < comp.size(); ++j) {
              const auto cj = offset + comp[j];
              M(i, j) += w * sv[l][j] *
                         (-scalar_product(s.test_gradients[i], Jq.j_flux(ci, cj)) + sv[0][i] * Jq.j_source(ci, cj));
            }
          }
        }
        for (uint i = 0; i < comp0.size(); ++i)
          for (uint e = 0; e < n_extr; ++e)
            out.extractor_jacobian(i, e) += w * (-scalar_product(s.test_gradients[i], Jq.j_extr_flux(comp0[i], e)) +
                                                 sv[0][i] * Jq.j_extr_source(comp0[i], e));
      }

      /**
       * @brief w phi_i d(numflux.n)/d(trial_j) at one face point into column block @p block of every level, for test
       * functions of trace `row_side` and trial functions of trace `column_side`, whose jacobian blocks are Jq. With
       * @p with_extractors, also the extractor blocks of Jq (boundary faces; interior faces add their total extractor
       * dependence once, see add_face_extractor_jacobian).
       */
      template <typename PJ>
      void add_face_jacobian(const Scratch &s, LocalData &out, const uint row_side, const uint column_side,
                             const PJ &Jq, const Tensor<1, dim> &normal, const NumberType w, const bool with_extractors,
                             const size_t block = 0) const
      {
        const auto &comp0 = s.comp[0];
        const auto &rows = s.shape_values[row_side][0];
        for (uint l = 0; l < n_levels; ++l) {
          const auto &comp = s.comp[l];
          const auto &cols = s.shape_values[column_side][l];
          auto &M = out.blocks[l][block];
          const size_t offset = MainBatch::level_offset(l);
          for (uint i = 0; i < comp0.size(); ++i)
            for (uint j = 0; j < comp.size(); ++j)
              M(i, j) += w * rows[i] * cols[j] * scalar_product(Jq.j_flux(comp0[i], offset + comp[j]), normal);
        }
        if (with_extractors)
          for (uint i = 0; i < comp0.size(); ++i)
            for (uint e = 0; e < n_extr; ++e)
              out.extractor_jacobian(i, e) += w * rows[i] * scalar_product(Jq.j_extr_flux(comp0[i], e), normal);
      }

      /// The extractor dependence of an interior face's numerical flux, for the test functions of trace `side`.
      template <typename PJ>
      void add_face_extractor_jacobian(const Scratch &s, LocalData &out, const uint side, const PJ &Jq,
                                       const Tensor<1, dim> &normal, const NumberType w) const
      {
        for (uint i = 0; i < s.comp[0].size(); ++i)
          for (uint e = 0; e < n_extr; ++e)
            out.extractor_jacobian(i, e) +=
                w * s.shape_values[side][0][i] * scalar_product(Jq.j_extr_flux(s.comp[0][i], e), normal);
      }

      void rebuild_ldg_vectors(const VectorType &sol) const
      {
        constexpr_for<1, Components::count_fe_subsystems(), 1>([&](auto k) {
          if (!model.get_components().jacobians_constant(k, k - 1)) {
            if constexpr (k == 1)
              build_ldg_vector<k - 1, k>(sol, sol_vector[k], sol_vector_tmp[k]);
            else
              build_ldg_vector<k - 1, k>(sol_vector[k - 1], sol_vector[k], sol_vector_tmp[k]);
          } else {
            if (!ldg_matrix_built[k]) rebuild_ldg_jacobian<k>(sol);

            // sol is a plain Vector; BlockSparseMatrix::vmult(BlockVector, Vector) only uses column block 0,
            // i.e. FE component 0. Copy into the (component-wise) block structure so all couplings act.
            sol_block = sol;
            j_gu[k].vmult(sol_vector[k], sol_block);
          }
        });
      }

      /// j_gu[k] = d(level k)/du at the current levels: level k's own jacobian, chained through j_gu[k - 1].
      template <int k> void rebuild_ldg_jacobian(const VectorType &sol) const
      {
        static_assert(k > 0);
        ldg_matrix_built[k] = true;
        if constexpr (k == 1)
          build_ldg_jacobian<0, 1>(sol, j_gu[1], j_wg_tmp[1]);
        else {
          if (!ldg_matrix_built[k - 1]) rebuild_ldg_jacobian<k - 1>(sol);
          build_ldg_jacobian<k - 1, k>(sol_vector[k - 1], j_wg[k], j_wg_tmp[k]);
          // mmult adds into its target, which therefore has to start from zero.
          j_gu[k] = 0;
          for (const auto &c : model.get_components().ldg_couplings(k, 0))
            for (const auto &b : model.get_components().ldg_couplings(k, k - 1))
              if (b[0] == c[0])
                j_wg[k]
                    .block(b[0], b[1])
                    .mmult(j_gu[k].block(c[0], c[1]), j_gu[k - 1].block(b[1], c[1]), Vector<NumberType>(), false);
        }
      }

      /**
       * @brief Build the LDG vector at level 'to' from level 'from' = to - 1: assemble M l_to = -(grad phi, F) +
       * (phi, s) + the boundary and interior numerical fluxes, then solve with the mass matrix.
       */
      template <int from, int to, typename SourceVector, typename VectorTypeldg>
      void build_ldg_vector(const SourceVector &source, VectorTypeldg &ldg_vector, VectorTypeldg &ldg_vector_tmp) const
      {
        static_assert(to - from == 1, "can only build LDG from last level!");
        constexpr size_t n_to = Components::count_fe_functions(to);
        auto &self = const_cast<Assembler &>(*this);
        LevelBatch<from> cells_b, boundary_b;
        std::array<LevelBatch<from>, 2> faces_b;
        self.template gather_level<from>(cells_b, boundary_b, faces_b, source);

        const NoMapsHere no_maps_during_assembly; // map() is collective, the points are rank-local
        BatchOutput<dim, NumberType, n_to> cell_out, boundary_out, face_out;
        cell_out.reinit(cells_b.size(), Term::flux | Term::source);
        evaluate_ldg_level<to>(model, cell_out, cells_b);
        boundary_out.reinit(boundary_b.size(), Term::flux);
        evaluate_ldg_boundary_numflux<to>(model, boundary_out, boundary_normals, boundary_b);
        face_out.reinit(faces_b[0].size(), Term::flux);
        evaluate_ldg_numflux<to>(model, face_out, face_normals, faces_b[0], faces_b[1]);

        const auto &constraints = discretization.get_constraints(to);
        ldg_vector_tmp = 0;
        self.scatter(
            to, &Scratch::level_local, ldg_vector_tmp,
            [&](const size_t k, Scratch &s, LocalData &local) {
              auto &r = local.residual;
              r.reinit(local.dofs.size());
              const auto &fe_v = *s.fe_v[to];
              const auto &comp = s.comp[to];
              for (const auto &q : fe_v.quadrature_point_indices()) {
                const size_t p = k * n_q + q;
                for (uint i = 0; i < r.size(); ++i) {
                  const auto c = comp[i];
                  Tensor<1, dim, NumberType> flux;
                  for (uint d = 0; d < dim; ++d)
                    flux[d] = cell_out.flux(c, d)[p];
                  r(i) += fe_v.JxW(q) * (-scalar_product(fe_v.shape_grad_component(i, q, c), flux) +
                                         fe_v.shape_value_component(i, q, c) * cell_out.source(c)[p]);
                }
              }
              add_face_terms(s, k, to, r, boundary_out, face_out, 1.);
            },
            [&](LocalData &local) {
              if (local.constrained)
                constraints.distribute_local_to_global(local.residual, local.dofs, ldg_vector_tmp);
              else
                ldg_vector_tmp.add(local.dofs, local.residual);
            });

        for (uint i = 0; i < n_to; ++i)
          component_mass_matrix_inverse.vmult(ldg_vector.block(i), ldg_vector_tmp.block(i));
      }

      /**
       * @brief Build the LDG jacobian d(level to)/d(level from), from = to - 1: assemble the derivative of the
       * right-hand side of build_ldg_vector, then multiply with the inverse mass matrix.
       */
      template <int from, int to, typename SourceVector>
      void build_ldg_jacobian(const SourceVector &source, BlockSparseMatrix<NumberType> &ldg_jacobian,
                              BlockSparseMatrix<NumberType> &ldg_jacobian_tmp) const
      {
        static_assert(to - from == 1, "can only build LDG from last level!");
        constexpr size_t n_to = Components::count_fe_functions(to), n_from = Components::count_fe_functions(from);
        auto &self = const_cast<Assembler &>(*this);
        LevelBatch<from> cells_b, boundary_b;
        std::array<LevelBatch<from>, 2> faces_b;
        self.template gather_level<from>(cells_b, boundary_b, faces_b, source);

        std::vector<PointJacobian<dim, n_to, n_from, 0>> J_cells, J_boundary;
        std::vector<FaceJacobian<dim, n_to, n_from, 0>> J_faces;
        SeedStackWorkspace<LevelBatch<from>, n_to> workspace;
        SeedStackWorkspace<LevelBatch<from>, n_to, 2> face_workspace;
        const NoMapsHere no_maps_during_assembly; // map() is collective, the points are rank-local
        evaluate_ldg_level_jacobian<to>(model, J_cells, cells_b, max_stacked_points, workspace);
        evaluate_ldg_boundary_numflux_jacobian<to>(model, J_boundary, boundary_normals, boundary_b, max_stacked_points,
                                                   workspace);
        evaluate_ldg_numflux_jacobian<to>(model, J_faces, face_normals, faces_b[0], faces_b[1], max_stacked_points,
                                          face_workspace);
        require_finite(J_cells);
        require_finite(J_boundary);
        require_finite(J_faces);

        ldg_jacobian_tmp = 0;
        ldg_jacobian = 0;
        self.scatter(
            to, &Scratch::level_local, ldg_jacobian_tmp,
            [&](const size_t k, Scratch &s, LocalData &local) {
              const size_t n_blocks = 1 + topology.n_interior_faces(k);
              const uint n_cols = dof_handler_list[from]->get_fe().n_dofs_per_cell();
              auto &blocks = local.blocks[from];
              auto &columns = local.columns[from];
              blocks.resize(n_blocks);
              columns.resize(n_blocks);
              for (auto &b : blocks)
                b.reinit(local.dofs.size(), n_cols);
              columns[0].resize(n_cols);
              cells[from][k]->get_dof_indices(columns[0]);

              const auto &comp_to = s.comp[to];
              const auto &comp_from = s.comp[from];
              // M(i, j) += w test_i (column block of the trial functions) for the jacobian blocks J of one point.
              const auto add = [&](FullMatrix<NumberType> &M, const auto &test, const auto &trial, const auto &J,
                                   const NumberType w, const auto &grad_test, const Tensor<1, dim> *normal) {
                for (uint i = 0; i < comp_to.size(); ++i)
                  for (uint j = 0; j < comp_from.size(); ++j) {
                    const auto &jF = J.j_flux(comp_to[i], comp_from[j]);
                    const NumberType value =
                        normal ? test[i] * scalar_product(jF, *normal)
                               : -scalar_product(grad_test[i], jF) + test[i] * J.j_source(comp_to[i], comp_from[j]);
                    M(i, j) += w * trial[j] * value;
                  }
              };
              std::vector<double> test(comp_to.size()), trial(comp_from.size()), trial_n(comp_from.size());
              std::vector<Tensor<1, dim>> grad_test(comp_to.size());
              const auto cache = [&](std::vector<double> &v, const auto &fev, const auto &comp, const uint q) {
                for (uint i = 0; i < comp.size(); ++i)
                  v[i] = fev.shape_value_component(i, q, comp[i]);
              };

              const auto &fe_v_to = *s.fe_v[to];
              for (const auto &q : fe_v_to.quadrature_point_indices()) {
                cache(test, fe_v_to, comp_to, q);
                cache(trial, *s.fe_v[from], comp_from, q);
                for (uint i = 0; i < comp_to.size(); ++i)
                  grad_test[i] = fe_v_to.shape_grad_component(i, q, comp_to[i]);
                add(blocks[0], test, trial, J_cells[k * n_q + q], fe_v_to.JxW(q), grad_test, nullptr);
              }
              for (size_t f = topology.boundary_begin[k]; f < topology.boundary_begin[k + 1]; ++f) {
                reinit_boundary(s, k, f);
                const auto &fe_fv = *s.fe_fv[to];
                for (const auto &q : fe_fv.quadrature_point_indices()) {
                  cache(test, fe_fv, comp_to, q);
                  cache(trial, *s.fe_fv[from], comp_from, q);
                  const Tensor<1, dim> normal = fe_fv.normal_vector(q);
                  add(blocks[0], test, trial, J_boundary[f * n_q_face + q], fe_fv.JxW(q), grad_test, &normal);
                }
              }
              for (size_t ref = topology.face_ref_begin[k]; ref < topology.face_ref_begin[k + 1]; ++ref) {
                const auto [f, side] = topology.face_refs[ref];
                const size_t slot = 1 + ref - topology.face_ref_begin[k];
                columns[slot].resize(n_cols);
                DiFfRG::internal::on(*dof_handler_list[from], topology.faces[f].cell[1 - side])
                    ->get_dof_indices(columns[slot]);
                reinit_interface(s, f);
                const auto &fe_iv = *s.fe_iv[to];
                const double sign = side == 0 ? 1. : -1.;
                for (const auto &q : fe_iv.quadrature_point_indices()) {
                  cache(test, fe_iv.get_fe_face_values(side), comp_to, q);
                  cache(trial, s.fe_iv[from]->get_fe_face_values(side), comp_from, q);
                  cache(trial_n, s.fe_iv[from]->get_fe_face_values(1 - side), comp_from, q);
                  const auto &Jq = J_faces[f * n_q_face + q];
                  const Tensor<1, dim> normal = fe_iv.normal_vector(q);
                  add(blocks[0], test, trial, Jq[side], sign * fe_iv.JxW(q), grad_test, &normal);
                  add(blocks[slot], test, trial_n, Jq[1 - side], sign * fe_iv.JxW(q), grad_test, &normal);
                }
              }
            },
            [&](LocalData &local) {
              for (size_t b = 0; b < local.blocks[from].size(); ++b)
                DiFfRG::internal::add_to_blocks(ldg_jacobian_tmp, local.dofs, local.columns[from][b],
                                                local.blocks[from][b]);
            });

        for (const auto &c : model.get_components().ldg_couplings(to, from))
          component_mass_matrix_inverse.mmult(ldg_jacobian.block(c[0], c[1]), ldg_jacobian_tmp.block(c[0], c[1]),
                                              Vector<NumberType>(), false);
      }

      /**
       * @brief Create a sparsity pattern for matrices between the DoFs of two DoFHandlers, with given stencil size.
       *
       * @param sparsity_pattern The pattern to store the result in.
       * @param dofh_to DoFHandler giving the row DoFs.
       * @param dofh_from DoFHandler giving the column DoFs.
       * @param stencil Stencil size of the resulting pattern.
       */
      void build_ldg_sparsity(BlockSparsityPattern &sparsity_pattern, const DoFHandler<dim> &to_dofh,
                              const DoFHandler<dim> &from_dofh, const int stencil = 1,
                              bool add_extractor_dofs = false) const
      {
        const auto &triangulation = discretization.get_triangulation();
        auto to_dofs_per_component = DoFTools::count_dofs_per_fe_component(to_dofh);
        auto from_dofs_per_component = DoFTools::count_dofs_per_fe_component(from_dofh);
        auto to_n_fe = to_dofs_per_component.size();
        auto from_n_fe = from_dofs_per_component.size();
        for (uint j = 1; j < from_dofs_per_component.size(); ++j)
          if (from_dofs_per_component[j] != from_dofs_per_component[0])
            throw std::runtime_error("For LDG the FE basis of all systems must be equal!");
        for (uint j = 1; j < to_dofs_per_component.size(); ++j)
          if (to_dofs_per_component[j] != to_dofs_per_component[0])
            throw std::runtime_error("For LDG the FE basis of all systems must be equal!");

        BlockDynamicSparsityPattern dsp(to_n_fe, from_n_fe);
        for (uint i = 0; i < to_n_fe; ++i)
          for (uint j = 0; j < from_n_fe; ++j)
            dsp.block(i, j).reinit(to_dofs_per_component[i], from_dofs_per_component[j]);
        dsp.collect_sizes();

        const auto to_dofs_per_cell = to_dofh.get_fe().dofs_per_cell;
        const auto from_dofs_per_cell = from_dofh.get_fe().dofs_per_cell;

        for (const auto &t_cell : triangulation.active_cell_iterators()) {
          std::vector<types::global_dof_index> to_dofs(to_dofs_per_cell);
          std::vector<types::global_dof_index> from_dofs(from_dofs_per_cell);
          const auto to_cell = typename DoFHandler<dim>::active_cell_iterator(
              &to_dofh.get_triangulation(), t_cell->level(), t_cell->index(), &to_dofh);
          const auto from_cell = typename DoFHandler<dim>::active_cell_iterator(
              &from_dofh.get_triangulation(), t_cell->level(), t_cell->index(), &from_dofh);
          to_cell->get_dof_indices(to_dofs);
          from_cell->get_dof_indices(from_dofs);

          std::function<void(decltype(from_cell) &, const int)> add_all_neighbor_dofs = [&](const auto &from_cell,
                                                                                            const int stencil_level) {
            for (const auto face_no : from_cell->face_indices()) {
              const auto face = from_cell->face(face_no);
              if (!face->at_boundary()) {
                auto neighbor_cell = from_cell->neighbor(face_no);

                if (dim == 1)
                  while (neighbor_cell->has_children())
                    neighbor_cell = neighbor_cell->child(face_no == 0 ? 1 : 0);

                // add all children
                else if (neighbor_cell->has_children()) {
                  throw std::runtime_error("not yet implemented lol");
                }

                if (!neighbor_cell->is_active()) continue;

                std::vector<types::global_dof_index> tmp(from_dofs_per_cell);
                neighbor_cell->get_dof_indices(tmp);

                from_dofs.insert(std::end(from_dofs), std::begin(tmp), std::end(tmp));

                if (stencil_level < stencil) add_all_neighbor_dofs(neighbor_cell, stencil_level + 1);
              }
            }
          };

          add_all_neighbor_dofs(from_cell, 1);

          for (const auto i : to_dofs)
            for (const auto j : from_dofs)
              dsp.add(i, j);
        }

        if (add_extractor_dofs)
          for (uint row = 0; row < dsp.n_rows(); ++row)
            for (const auto &col : extractor_dof_indices)
              dsp.add(row, col);

        sparsity_pattern.copy_from(dsp);
      }

      /**
       * @brief Build the inverse of matrix in and save the result to out.
       *
       * @param in The matrix to invert. Must have a valid sparsity pattern.
       * @param out The matrix to store the result. Must have a valid sparsity pattern.
       */
      void build_inverse(const SparseMatrix<NumberType> &in, SparseMatrix<NumberType> &out) const
      {
        GrowingVectorMemory<Vector<NumberType>> mem;
        SparseDirectUMFPACK inverse;
        inverse.initialize(in);
        // out is m x n
        // we go row-wise, i.e. we keep one n fixed and insert a row
        tbb::parallel_for(tbb::blocked_range<int>(0, out.n()), [&](tbb::blocked_range<int> r) {
          typename VectorMemory<Vector<NumberType>>::Pointer tmp(mem);
          tmp->reinit(out.m());
          for (int n = r.begin(); n < r.end(); ++n) {
            *tmp = 0;
            (*tmp)[n] = 1.;
            inverse.solve(*tmp);
            for (auto it = out.begin(n); it != out.end(n); ++it)
              it->value() = (*tmp)[it->column()];
          }
        });
      }

      const size_t max_stacked_points;
      uint n_q = 0, n_q_face = 0;
      /// The owned cells, numbered alike on every level; colored by the constraints of their level.
      array<DiFfRG::internal::ColoredCells<dim>, n_levels> cells;
      DiFfRG::internal::FaceTopology<dim> topology;
      std::unique_ptr<tbb::enumerable_thread_specific<Scratch>> scratch;
      std::vector<LocalData> local_buffer;
      const std::array<NumberType, 0> no_extractors{};
      const VectorType no_variables;

      MainBatch cell_batch, boundary_batch;
      std::array<MainBatch, 2> face_batch;
      std::vector<Tensor<1, dim>> boundary_normals, face_normals;
      BatchOutput<dim, NumberType, n_fe> cell_result, boundary_result, face_result;
      std::vector<PointJacobian<dim, n_fe, n_all, n_extr>> cell_jacobians, boundary_jacobians;
      std::vector<FaceJacobian<dim, n_fe, n_all, n_extr>> face_jacobians;
      SeedStackWorkspace<MainBatch, n_fe> cell_workspace, boundary_workspace;
      SeedStackWorkspace<MainBatch, n_fe, 2> face_workspace;

      // The point evaluation of the extractors and readouts; see DiFfRG::internal::AssemblerCore.
      /// The extractors see the levels in the tuple slots of the derivatives.
      static constexpr bool extractors_see_derivatives = false;

      void prepare_point_evaluation(const VectorType &solution) const { rebuild_ldg_vectors(solution); }

      /// The LDG solution across all subsystems, its gradients and hessians, and the raw potential,
      /// all at one point. Built by evaluate_at().
      template <typename PotentialEvaluation = RawPotentialEvaluation<dim, NumberType>> struct PointEvaluation {
        std::vector<Vector<NumberType>> solutions;
        std::vector<std::vector<Tensor<1, dim, NumberType>>> gradients{
            std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
        std::vector<std::vector<Tensor<2, dim, NumberType>>> hessians{
            std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
        PotentialEvaluation potential;
        /// Subsystem 0's FEValues at the point, kept for the shape values the extractor jacobian needs.
        std::shared_ptr<FEValues<dim>> fe_values;
      };

      /**
       * @brief Evaluate every LDG subsystem at @p x, which lies in @p x_cell.
       *
       * Each subsystem carries its own DoFHandler over the same triangulation, so the cell has to be
       * re-addressed per subsystem before its FEValues can be reinit'ed. Only subsystem 0 (the
       * solution proper) has gradients and hessians read off it; the LDG levels are values only.
       */
      template <typename RawPotential>
      auto evaluate_at(const Point<dim> &x, const typename DoFHandler<dim>::cell_iterator &x_cell,
                       const VectorType &solution_global, const RawPotential &raw_potential) const
      {
        using t_Iterator = typename Triangulation<dim>::active_cell_iterator;
        const auto x_unit = mapping.transform_real_to_unit_cell(x_cell, x);

        std::vector<std::shared_ptr<FEValues<dim>>> fe_v;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
          fe_v.emplace_back(std::make_shared<FEValues<dim>>(
              mapping, discretization.get_fe(k), x_unit,
              update_values | update_gradients | update_quadrature_points | update_JxW_values | update_hessians));
          auto cell = dof_handler_list[k]->begin_active();
          cell->copy_from(*t_Iterator(x_cell));
          fe_v[k]->reinit(cell);
        }

        PointEvaluation<decltype(evaluate_raw_potential(raw_potential, mapping, x))> evaluation;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
          std::vector<Vector<NumberType>> values{Vector<NumberType>(Components::count_fe_functions(k))};
          if (k == 0)
            fe_v[0]->get_function_values(solution_global, values);
          else
            fe_v[k]->get_function_values(sol_vector[k], values);
          evaluation.solutions.push_back(values[0]);
        }
        fe_v[0]->get_function_gradients(solution_global, evaluation.gradients);
        fe_v[0]->get_function_hessians(solution_global, evaluation.hessians);
        evaluation.potential = evaluate_raw_potential(raw_potential, mapping, x);
        evaluation.fe_values = fe_v[0];
        return evaluation;
      }

      template <typename Evaluation, typename Extractors_>
      auto solution_tie(const Evaluation &e, const Extractors_ &extractors, const VectorType &variables) const
      {
        auto t = std::tuple_cat(vector_to_tuple<Components::count_fe_subsystems()>(e.solutions),
                                std::tie(e.gradients[0], e.hessians[0], extractors, variables, e.potential.value,
                                         e.potential.gradient, e.potential.mass_hessian));
        return fe_more_conv(t);
      }
    };
  } // namespace LDG
} // namespace DiFfRG
