#pragma once

// standard library
#include <sstream>

// external libraries
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/physics/integration/map_completion.hh>

namespace DiFfRG
{
  namespace Variables
  {
    using namespace dealii;

    template <typename... T> auto fe_tie(T &&...t)
    {
      return named_tuple<std::tuple<T &...>, StringSet<"variables">>(std::tie(t...));
    }

    /**
     * @brief The basic assembler that can be used for any standard CG scheme with flux and source.
     *
     * @tparam Model The model class which contains the physical equations.
     */
    template <typename Model_> class Assembler : public AbstractAssembler<Vector<double>, SparseMatrix<double>, 0>
    {
      constexpr static int nothing = 0;

    public:
      using Model = Model_;
      using NumberType = double;
      using VectorType = Vector<double>;
      using SparseMatrixType = SparseMatrix<double>;

      using Components = typename Model_::Components;
      static constexpr uint dim = 0;

      /// This assembler has no FE space at all (dim == 0), so it needs no report port; the config is
      /// unused.
      Assembler(Model &model, const ConfigTree & /*config*/) : model(model)
      {
        static_assert(Components::count_fe_functions() == 0, "The pure variable assembler cannot handle FE functions!");
        reinit();
      }

      virtual void reinit_vector(VectorType &vec) const override { vec.reinit(0); }
      // Serial by construction: this assembler has no FE space at all (dim == 0) and hard-codes
      // dealii::Vector/SparseMatrix in its base-class list, so there is no distributed policy here.
      virtual void reinit_matrix(SparseMatrixType &matrix) const override
      {
        matrix.reinit(get_sparsity_pattern_jacobian());
      }
      virtual MPI_Comm get_communicator() const override { return MPI_COMM_SELF; }
      // dim == 0: no FE space, and the vector type is serial by construction, so the view is a
      // passthrough and the layout arguments are ignored.
      virtual void reinit_solution_view(SolutionView<VectorType> &view) const override
      {
        view.reinit(dealii::IndexSet(), dealii::IndexSet(), MPI_COMM_SELF);
      }

      virtual IndexSet get_differential_indices() const override { return IndexSet(); }

      virtual void attach_data_output(OutputFrame<dim, VectorType> &data_out, const VectorType &solution,
                                      const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                      const VectorType &residual = VectorType()) override
      {
        (void)dt_solution;
        (void)residual;
        readouts(data_out, solution, variables);
      }

      virtual void reinit() override {}

      virtual void set_time(double t) override { model.set_time(t); }

      /// There is no spatial part: a snapshot of a variables-only flow holds just the variables.
      virtual SnapshotSpatialState capture_snapshot_state(const Vector<double> & /*spatial_replica*/) const override
      {
        return {};
      }

      virtual void restore_snapshot_state(const SnapshotSpatialState &state, Vector<double> & /*spatial*/) override
      {
        if (!state.empty())
          throw std::runtime_error("Snapshot restore: the snapshot holds a spatial discretization, but this is a "
                                   "variables-only (dim = 0) flow.");
      }

      virtual void save_model_state(ModelState &state) const override
      {
        DiFfRG::internal::save_model_state(model, state);
      }

      virtual bool load_model_state(const ModelState &state) override
      {
        return DiFfRG::internal::load_model_state(model, state);
      }

      virtual const SparsityPattern &get_sparsity_pattern_jacobian() const override
      {
        return sparsity_pattern_jacobian;
      }
      virtual const SparseMatrix<NumberType> &get_mass_matrix() const override { return mass_matrix; }

      virtual void residual_variables(VectorType &residual, const VectorType &variables, const VectorType &) override
      {
        Timer timer;
        model.dt_variables(residual, fe_tie(variables));
        // Fences AND lands any map() results still sitting in pinned staging -- see MapCompletion.
        flush_maps();
        timings_residual.push_back(timer.wall_time());
      };

      virtual void jacobian_variables(FullMatrix<NumberType> &jacobian, const VectorType &variables,
                                      const VectorType &) override
      {
        Timer timer;
        model.template jacobian_variables<0>(jacobian, fe_tie(variables));
        flush_maps();
        timings_jacobian.push_back(timer.wall_time());
      };

      void readouts(OutputFrame<dim, VectorType> &data_out, const VectorType &, const VectorType &variables) const
      {
        auto helper = [&](auto &&...args) {
          if constexpr (sizeof...(args) == 3) {
            auto &&[id, EoMfun, outputter] = std::forward_as_tuple(std::forward<decltype(args)>(args)...);
            data_out.register_readout(id);
            (void)EoMfun;
            outputter(data_out, Point<0>(), fe_tie(variables));
          } else {
            DiFfRG::internal::validate_readout_helper_arity<decltype(args)...>();
          }
        };
        model.readouts_multiple(helper, data_out);
      }

      virtual void mass(VectorType &, const VectorType &, const VectorType &, NumberType) override {}

      virtual void residual(VectorType &, const VectorType &, NumberType, const VectorType &, NumberType,
                            const VectorType &variables = VectorType()) override
      {
        (void)variables;
      }

      virtual void jacobian_mass(SparseMatrix<NumberType> &, const VectorType &, const VectorType &, NumberType,
                                 NumberType) override
      {
      }

      virtual void jacobian(SparseMatrix<NumberType> &, const VectorType &, NumberType, const VectorType &, NumberType,
                            NumberType, const VectorType &variables = VectorType()) override
      {
        (void)variables;
      }
      SummaryEvent summary() const override
      {
        SummaryEvent result{.component = "variables"};
        result.timing("residual", average_time_residual_assembly() * 1000, num_residuals())
            .timing("jac", average_time_jacobian_assembly() * 1000, num_jacobians());
        return result;
      }

      double average_time_residual_assembly() const
      {
        double t = 0.;
        double n = timings_residual.size();
        for (const auto &t_ : timings_residual)
          t += t_ / n;
        return t;
      }
      uint num_residuals() const { return timings_residual.size(); }

      double average_time_jacobian_assembly() const
      {
        double t = 0.;
        double n = timings_jacobian.size();
        for (const auto &t_ : timings_jacobian)
          t += t_ / n;
        return t;
      }
      uint num_jacobians() const { return timings_jacobian.size(); }

    private:
      Model &model;

      SparsityPattern sparsity_pattern_mass;
      SparsityPattern sparsity_pattern_jacobian;
      SparseMatrix<NumberType> mass_matrix;

      std::vector<double> timings_residual;
      std::vector<double> timings_jacobian;
    };
  } // namespace Variables
} // namespace DiFfRG
