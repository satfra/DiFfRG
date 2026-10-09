#pragma once

// standard library
#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// external libraries
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/matrix_tools.h>

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/discretization/common/phase_times.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/discretization/data/output_session.hh>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief What every assembler (CG, DG, LDG, KT) does the same way: the EoM search, the extractors and their
     * jacobian, the readouts, the variables' residual, the constraints, the mass matrix, the linear-algebra setup
     * and the timings.
     *
     * The assemblers differ in how they evaluate the solution at a single point, which they provide (CRTP):
     *  - `evaluate_at(x, cell, solution, raw_potential)`: the solution and the raw potential at @p x, which lies in
     *    @p cell; an object with a `potential` member (a RawPotentialEvaluation) and, for extract_with_jacobian(),
     *    `fe_values` (an FEValues at the point);
     *  - `solution_tie(evaluation, extractors, variables)`: the named tuple the model's extract() and readouts see.
     *
     * They may shadow
     *  - `prepare_point_evaluation(solution)`, called before any evaluate_at() of extract() and readouts() (LDG
     *    builds its levels there);
     *  - `extractor_sample(solution)`, the SolutionSample handed to a model's extractor_point;
     *  - `extractors_see_derivatives`: whether extract_with_jacobian() also differentiates by the derivatives and
     *    hessians, i.e. whether the tuple's slots 1 and 2 are those (not for LDG, where they are levels).
     *
     * Derived is the class that provides these, which for CG and DG is FEMAssembler, not the assembler itself.
     */
    template <typename Derived, typename Discretization_, typename Model_>
    class AssemblerCore : public AbstractAssembler<typename Discretization_::VectorType,
                                                   typename Discretization_::SparseMatrixType, Discretization_::dim>
    {
      using AbstractBase = AbstractAssembler<typename Discretization_::VectorType,
                                             typename Discretization_::SparseMatrixType, Discretization_::dim>;

    public:
      using Discretization = Discretization_;
      using Model = Model_;
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;
      using SparseMatrixType = typename Discretization::SparseMatrixType;
      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;
      using Extractors = std::array<NumberType, Components::count_extractors()>;
      using CellIterator = typename dealii::DoFHandler<dim>::cell_iterator;
      using PhaseTimes = AssemblyPhaseTimes;

      /// @param component the name of the assembler in its summary().
      AssemblerCore(Discretization &discretization, Model &model, const ConfigTree &config,
                    const std::string_view component)
          : discretization(discretization), model(model), fe(discretization.get_fe()),
            dof_handler(discretization.get_dof_handler()), mapping(discretization.get_mapping()), EoM_cell(no_cell()),
            old_extractor_cell(no_cell()), EoM_config(resolve_eom_config(dof_handler, Config::EoMConfig(config))),
            component(component)
      {
      }

      const auto &get_discretization() const { return discretization; }
      auto &get_discretization() { return discretization; }

      virtual void set_time(double t) override { model.set_time(t); }

      virtual MPI_Comm get_communicator() const override { return discretization.get_communicator(); }

      virtual void reinit_vector(VectorType &vec) const override
      {
        reinit_la_vector(vec, discretization.get_locally_owned_dofs(), discretization.get_communicator());
      }

      virtual void reinit_matrix(SparseMatrixType &matrix) const override
      {
        reinit_la_matrix(matrix, sparsity_pattern_jacobian, discretization.get_locally_owned_dofs(),
                         discretization.get_communicator());
      }

      virtual void reinit_solution_view(SolutionView<VectorType> &view) const override
      {
        view.reinit(discretization.get_locally_owned_dofs(), discretization.get_locally_relevant_dofs(),
                    discretization.get_communicator());
      }

      virtual const get_type::SparsityPattern<SparseMatrixType> &get_sparsity_pattern_jacobian() const override
      {
        return sparsity_pattern_jacobian;
      }
      virtual const SparseMatrixType &get_mass_matrix() const override { return mass_matrix; }

      virtual IndexSet get_differential_indices() const override
      {
        ComponentMask component_mask(model.template differential_components<dim>());
        // Restricted to owned rows under the distributed policy: deal.II writes every index of this set into a
        // distributed vector and compresses with VectorOperation::insert, so an unrestricted set has every rank
        // inserting into every entry.
        return restrict_to_owned<VectorType>(DoFTools::extract_dofs(dof_handler, component_mask),
                                             discretization.get_locally_owned_dofs());
      }

      virtual SnapshotSpatialState capture_snapshot_state(const VectorType &spatial_replica) const override
      {
        return capture_cellwise_state(dof_handler, spatial_replica);
      }

      virtual void restore_snapshot_state(const SnapshotSpatialState &state, VectorType &spatial) override
      {
        restore_spatial_state(state, discretization, *this, spatial);
      }

      virtual void save_model_state(ModelState &state) const override
      {
        DiFfRG::internal::save_model_state(model, state);
      }

      virtual bool load_model_state(const ModelState &state) override
      {
        return DiFfRG::internal::load_model_state(model, state);
      }

      using AbstractBase::attach_data_output;
      /// The FE functions, their time derivatives and residuals (when given), and the model's readouts.
      virtual void attach_data_output(OutputFrame<dim, VectorType> &data_out, const VectorType &solution,
                                      const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                      const VectorType &residual = VectorType()) override
      {
        const auto names = Components::FEFunction_Descriptor::get_names_vector();
        std::vector<std::string> names_dot, names_residual;
        for (const auto &name : names) {
          names_dot.push_back(name + "_dot");
          names_residual.push_back(name + "_residual");
        }
        auto fe_out = data_out.fields();
        fe_out.attach(dof_handler, solution, names);
        if (dt_solution.size() > 0) fe_out.attach(dof_handler, dt_solution, names_dot);
        if (residual.size() > 0) fe_out.attach(dof_handler, residual, names_residual);

        readouts(data_out, solution, variables);
      }

      /// The model's dt_variables, with the extractors at the EoM.
      virtual void residual_variables(VectorType &residual, const VectorType &variables,
                                      const VectorType &spatial_solution) override
      {
        Timer timer;
        Extractors extracted{};
        if constexpr (Components::count_extractors() > 0)
          extract(extracted, spatial_solution, variables, true, false, false);
        model.dt_variables(residual, v_tie(variables, std::as_const(extracted)));
        Kokkos::fence();
        timings_variable_residual.push_back(timer.wall_time());
      }

      virtual void jacobian_variables(FullMatrix<NumberType> &jacobian, const VectorType &variables,
                                      const VectorType &spatial_solution) override
      {
        Timer timer;
        Extractors extracted{};
        if constexpr (Components::count_extractors() > 0)
          extract(extracted, spatial_solution, variables, true, false, false);
        model.template jacobian_variables<0>(jacobian, v_tie(variables, std::as_const(extracted)));
        Kokkos::fence();
        timings_variable_jacobian.push_back(timer.wall_time());
      }

      const PhaseTimes &residual_phase_times() const { return residual_times; }
      const PhaseTimes &jacobian_phase_times() const { return jacobian_times; }
      void reset_phase_times() { residual_times = jacobian_times = PhaseTimes{}; }

      SummaryEvent summary() const override
      {
        SummaryEvent result{.component = component};
        result.timing("reinit", average(timings_reinit) * 1000, timings_reinit.size())
            .timing("residual", average(timings_residual) * 1000, timings_residual.size())
            .timing("jac", average(timings_jacobian) * 1000, timings_jacobian.size());
        if (jacobian_times.calls > 0)
          result.timing("jac eval", jacobian_times.evaluate / jacobian_times.calls * 1000, jacobian_times.calls);
        return result;
      }

      double average_time_reinit() const { return average(timings_reinit); }
      uint num_reinits() const { return timings_reinit.size(); }
      double average_time_residual_assembly() const { return average(timings_residual); }
      uint num_residuals() const { return timings_residual.size(); }
      double average_time_jacobian_assembly() const { return average(timings_jacobian); }
      uint num_jacobians() const { return timings_jacobian.size(); }
      double average_time_variable_residual_assembly() const { return average(timings_variable_residual); }
      uint num_variable_residuals() const { return timings_variable_residual.size(); }
      double average_time_variable_jacobian_assembly() const { return average(timings_variable_jacobian); }
      uint num_variable_jacobians() const { return timings_variable_jacobian.size(); }

      /**
       * @brief The model's extractors at its extractor point (the EoM unless it defines extractor_point).
       *
       * @param search_EoM re-locate the EoM instead of reusing the stored one.
       * @param set_EoM store the located EoM.
       * @param postprocess apply the model's EoM_postprocess to the located point.
       */
      void extract(Extractors &data, const VectorType &solution, const VectorType &variables, bool search_EoM,
                   bool set_EoM, bool postprocess) const
      {
        auto EoM = this->EoM;
        auto cell = EoM_cell;
        if (search_EoM || cell == no_cell())
          EoM =
              locate_EoM(
                  cell, solution, [&](const auto &p, const auto &values) { return model.EoM(p, values); },
                  [&](const auto &p, const auto &values) { return postprocess ? model.EoM_postprocess(p, values) : p; })
                  .point;
        if (set_EoM) {
          this->EoM = EoM;
          EoM_cell = cell;
        }
        derived().prepare_point_evaluation(solution);
        const auto [x, x_cell] = resolve_extractor_point(EoM, cell, solution);
        const auto e = derived().evaluate_at(x, x_cell, solution, extractor_raw_potential(solution));
        model.extract(data, x, derived().solution_tie(e, nothing, variables));
      }

    protected:
      /// Placeholder for the "extractors" slot of solution_tie() while the extractors themselves are computed.
      static constexpr int nothing = 0;

      template <typename... T> static constexpr auto v_tie(T &&...t)
      {
        return named_tuple<std::tuple<T &...>, StringSet<"variables", "extractors">>(std::tie(t...));
      }

      /// The solution tuple of the extractors and readouts, from the values, derivatives and hessians at a point.
      template <typename... T> static constexpr auto e_tie(T &&...t)
      {
        return named_tuple<std::tuple<T &...>,
                           StringSet<"fe_functions", "fe_derivatives", "fe_hessians", "extractors", "variables",
                                     "potential", "potential_gradient", "potential_hessian">>(std::tie(t...));
      }

      /// What a derived assembler may shadow; see the class documentation.
      void prepare_point_evaluation(const VectorType &) const {}
      auto extractor_sample(const VectorType &solution) const
      {
        return make_solution_sample(solution, dof_handler, mapping);
      }
      static constexpr bool extractors_see_derivatives = true;

      /**
       * @brief The part of reinit() every assembler shares: the hanging-node and the model's affine constraints of
       * the current mesh, and forgetting the EoM and extractor cells, which may belong to a previous one.
       */
      void reinit_common()
      {
        EoM_cell = old_extractor_cell = no_cell();
        extractor_dof_indices.clear();
        const auto metadata = build_affine_constraint_metadata<Components, dim>(discretization);
        const AffineConstraintContext<Components, dim> context(metadata);
        auto &constraints = discretization.get_constraints();
        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        apply_model_affine_constraints(model, constraints, context);
        constraints.close();
      }

      /// The mass matrix and its sparsity.
      void build_mass_matrix(const dealii::Quadrature<dim> &quadrature)
      {
        DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                        /*keep_constrained_dofs = */ true);
        finalize_la_sparsity<SparseMatrixType>(dsp, sparsity_pattern_mass, discretization.get_locally_owned_dofs(),
                                               discretization.get_locally_relevant_dofs(),
                                               discretization.get_communicator());
        reinit_la_matrix(mass_matrix, sparsity_pattern_mass, discretization.get_locally_owned_dofs(),
                         discretization.get_communicator());
        MatrixCreator::create_mass_matrix(dof_handler, quadrature, mass_matrix,
                                          static_cast<Function<dim, NumberType> *>(nullptr),
                                          discretization.get_constraints());
      }

      /// The default budget of one seed-stacked AD evaluation of the FEM assemblers (CG, DG, LDG). They stack whole
      /// seed directions, never part of the points, so unlike for KT a smaller budget does not keep the batch in
      /// cache; fewer, larger evaluations are faster.
      static constexpr size_t fem_stacked_budget = size_t(256) << 20;

      /**
       * @brief /discretization/batched/max_stacked_points, by default as many points as fit into @p budget bytes
       * at @p bytes_per_point.
       */
      static size_t read_max_stacked_points(const ConfigTree &config, const size_t bytes_per_point, const size_t budget)
      {
        return config.get_uint("/discretization/batched/max_stacked_points",
                               std::max<size_t>(1, budget / bytes_per_point));
      }

      /**
       * @brief Locate the EoM, starting from @p cell, and move @p cell to the cell holding it.
       *
       * Also refreshes the starting guess of the next search.
       */
      template <typename EoMFun, typename PostprocessFun>
      auto locate_EoM(CellIterator &cell, const VectorType &solution, const EoMFun &EoM_fun,
                      const PostprocessFun &postprocess) const
      {
        auto result = get_EoM_point_with_potential(cell, solution, dof_handler, mapping, EoM_fun, postprocess,
                                                   EoM_config, EoM_minimum_guess, &potential_cache);
        if (result.potential) EoM_minimum_guess = result.potential->minimum;
        return result;
      }

      /**
       * @brief The raw potential for the extractors, or an inert placeholder if the model does not read it.
       *
       * Reconstructing it is a direct solve over the whole mesh, and extract() runs on every residual and every
       * jacobian -- so a model that never touches the potential slots should say so and skip it.
       */
      auto extractor_raw_potential(const VectorType &solution) const
      {
        if constexpr (Model::extract_uses_potential)
          return reconstruct_potential(solution);
        else
          return UnusedPotential{};
      }

      /// The raw potential of the readouts; see reconstruct_raw_potential.
      auto reconstruct_potential(const VectorType &solution) const
      {
        return reconstruct_raw_potential(
            solution, dof_handler, mapping,
            [&](const auto &p, const auto &values) { return model.raw_potential_gradient(p, values); }, EoM_config,
            &potential_cache);
      }

      /**
       * @brief Where the model wants its extractors evaluated, and the cell holding that point.
       *
       * The EoM itself when the model does not define `extractor_point` -- and then this costs nothing, because
       * building the SolutionSample is inside the `if constexpr`.
       */
      std::pair<Point<dim>, CellIterator> resolve_extractor_point(const Point<dim> &EoM_point,
                                                                  const CellIterator &EoM_point_cell,
                                                                  [[maybe_unused]] const VectorType &solution) const
      {
        if constexpr (HasExtractorPoint<Model, dim, NumberType>) {
          const auto point =
              model.template extractor_point<dim, NumberType>(EoM_point, derived().extractor_sample(solution));
          if (point == EoM_point) return {EoM_point, EoM_point_cell};
          return {point, GridTools::find_active_cell_around_point(dof_handler, point)};
        } else
          return {EoM_point, EoM_point_cell};
      }

      /// One readout per entry of the model's readouts_multiple, each at its own EoM.
      void readouts(OutputFrame<dim, VectorType> &data_out, const VectorType &solution,
                    const VectorType &variables) const
      {
        derived().prepare_point_evaluation(solution);
        auto potential = reconstruct_potential(solution);
        // GCC 16 fails to find members inside `if constexpr` in this variadic lambda; keep `this->`.
        auto helper = [&](auto &&...args) {
          if constexpr (sizeof...(args) == 3) {
            auto &&[id, EoMfun, outputter] = std::forward_as_tuple(std::forward<decltype(args)>(args)...);
            data_out.register_readout(id);
            auto cell = this->EoM_cell;
            auto EoM_result = this->locate_EoM(cell, solution, EoMfun, [](const auto &p, const auto &) { return p; });
            const auto EoM = EoM_result.point;
            const auto readout_solution = this->derived().evaluate_at(EoM, cell, solution, potential);

            // The readout is always at this readout's EoM. The extractors may not be: a model that defines
            // extractor_point reads them elsewhere, and dt_variables must see the same values here as it does
            // during assembly.
            Extractors extracted{};
            if constexpr (Components::count_extractors() > 0) {
              const auto [x, x_cell] = this->resolve_extractor_point(EoM, cell, solution);
              const auto e = this->derived().evaluate_at(x, x_cell, solution, potential);
              this->model.extract(extracted, x, this->derived().solution_tie(e, this->nothing, variables));
            }
            outputter(data_out, EoM,
                      this->derived().solution_tie(readout_solution, std::as_const(extracted), variables));
            data_out.attach_eom_potential(std::move(EoM_result));
          } else {
            validate_readout_helper_arity<decltype(args)...>();
          }
        };
        model.readouts_multiple(helper, data_out);
        data_out.attach_raw_potential(std::move(potential));
      }

      /**
       * @brief extract() at a freshly located (and stored) EoM, plus the derivative of the extractors with respect
       * to the dofs of the cell they are evaluated in, into extractor_jacobian.
       *
       * @return whether that cell changed, in which case extractor_dof_indices is updated and the jacobian sparsity
       * rebuilt, so the caller has to reinit its jacobian. Agreed across ranks, because the rebuild is collective.
       */
      bool extract_with_jacobian(Extractors &data, const VectorType &solution, const VectorType &variables)
      {
        constexpr uint n_extr = Components::count_extractors();
        constexpr uint n_fe = Components::count_fe_functions();
        EoM = locate_EoM(
                  EoM_cell, solution, [&](const auto &p, const auto &values) { return model.EoM(p, values); },
                  [&](const auto &p, const auto &values) { return model.EoM_postprocess(p, values); })
                  .point;
        derived().prepare_point_evaluation(solution);

        // The extractor jacobian couples to the dofs of the cell the extractors are actually evaluated in, which is
        // the extractor point's cell, not the EoM's. any_of: if any rank needs the collective sparsity rebuild,
        // every rank must enter it.
        const auto [x, cell] = resolve_extractor_point(EoM, EoM_cell, solution);
        const bool new_cell = MPI::any_of(discretization.get_communicator(), old_extractor_cell != cell);
        old_extractor_cell = cell;

        const auto e = derived().evaluate_at(x, cell, solution, extractor_raw_potential(solution));
        const auto solution_tuple = derived().solution_tie(e, nothing, variables);
        model.extract(data, x, solution_tuple);

        const auto &fe_v = *e.fe_values;
        const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();
        if (new_cell) {
          extractor_dof_indices.resize(n_dofs);
          cell->get_dof_indices(extractor_dof_indices);
          derived().rebuild_jacobian_sparsity();
        }

        FullMatrix<NumberType> j_u(n_extr, n_fe), j_du, j_ddu;
        model.template jacobian_extractors<0>(j_u, x, solution_tuple);
        if constexpr (Derived::extractors_see_derivatives) {
          j_du.reinit(n_extr, n_fe * dim);
          j_ddu.reinit(n_extr, n_fe * dim * dim);
          model.template jacobian_extractors<1>(j_du, x, solution_tuple);
          model.template jacobian_extractors<2>(j_ddu, x, solution_tuple);
        }

        extractor_jacobian.reinit(n_extr, n_dofs);
        for (uint i = 0; i < n_dofs; ++i) {
          const auto c = fe_v.get_fe().system_to_component_index(i).first;
          for (uint k = 0; k < n_extr; ++k) {
            extractor_jacobian(k, i) = j_u(k, c) * fe_v.shape_value_component(i, 0, c);
            if constexpr (Derived::extractors_see_derivatives)
              for (uint d1 = 0; d1 < dim; ++d1) {
                extractor_jacobian(k, i) += j_du(k, c * dim + d1) * fe_v.shape_grad_component(i, 0, c)[d1];
                for (uint d2 = 0; d2 < dim; ++d2)
                  extractor_jacobian(k, i) +=
                      j_ddu(k, c * dim * dim + d1 * dim + d2) * fe_v.shape_hessian_component(i, 0, c)[d1][d2];
              }
          }
        }
        return new_cell;
      }

      static double average(const std::vector<double> &t)
      {
        return t.empty() ? 0. : std::accumulate(t.begin(), t.end(), 0.) / t.size();
      }

      Discretization &discretization;
      Model &model;
      const FiniteElement<dim> &fe;
      const DoFHandler<dim> &dof_handler;
      const Mapping<dim> &mapping;

      mutable Point<dim> EoM;
      mutable CellIterator EoM_cell;
      /// The cell the extractors were last differentiated in; see extract_with_jacobian().
      CellIterator old_extractor_cell;
      const Config::EoMConfig EoM_config;
      mutable std::optional<Point<dim>> EoM_minimum_guess;
      /// Mesh-dependent half of the potential reconstructions, built once and reused; see PotentialSystemCache.
      mutable PotentialSystemCache<dim, NumberType> potential_cache;
      /// d(extractors)/d(dofs of extractor_dof_indices), from extract_with_jacobian().
      FullMatrix<NumberType> extractor_jacobian;
      std::vector<types::global_dof_index> extractor_dof_indices;

      get_type::SparsityPattern<SparseMatrixType> sparsity_pattern_mass;
      get_type::SparsityPattern<SparseMatrixType> sparsity_pattern_jacobian;
      SparseMatrixType mass_matrix;

      std::vector<double> timings_reinit, timings_residual, timings_jacobian;
      std::vector<double> timings_variable_residual, timings_variable_jacobian;
      PhaseTimes residual_times, jacobian_times;

    private:
      const Derived &derived() const { return static_cast<const Derived &>(*this); }
      Derived &derived() { return static_cast<Derived &>(*this); }
      CellIterator no_cell() const { return *(dof_handler.active_cell_iterators().end()); }

      const std::string_view component;
    };
  } // namespace internal
} // namespace DiFfRG
