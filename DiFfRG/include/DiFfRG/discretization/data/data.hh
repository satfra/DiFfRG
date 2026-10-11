#pragma once

// standard library
#include <stdexcept>

// external libraries
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace DiFfRG
{
  using namespace dealii;

  namespace internal
  {
    /// A deal.II Function whose vector_value calls a lambda.
    template <uint dim, typename NumberType> class FunctionFromLambda : public dealii::Function<dim, NumberType>
    {
      using FUN = std::function<void(const Point<dim> &, Vector<NumberType> &)>;

    public:
      FunctionFromLambda(FUN &&fun_, uint components) : Function<dim, NumberType>(components), fun(std::move(fun_)) {}
      virtual void vector_value(const Point<dim> &p, Vector<NumberType> &values) const override { fun(p, values); }

    private:
      FUN fun;
    };
  } // namespace internal

  /**
   * @brief State of a flow without field-space discretization: only variables (block 1; block 0 is empty).
   *
   * Usage: construct, then interpolate(model) to set the initial condition.
   *
   * @tparam NT Scalar type.
   */
  template <typename NT = double> class FlowingVariables : public AbstractFlowingVariables<NT>
  {
  public:
    using NumberType = NT;

    /// An empty state; call interpolate().
    FlowingVariables() {}

    /**
     * @brief Size the state for @p model and fill the variables with Model::initial_condition_variables.
     */
    template <typename Model> void interpolate(const Model &model)
    {
      std::vector<uint> block_structure{0};
      block_structure.push_back(Model::Components::count_variables());
      m_data = (block_structure);
      if (m_data.n_blocks() > 1) model.initial_condition_variables(m_data.block(1));
    }

    /// The full state: spatial dofs (block 0) and variables (block 1).
    virtual BlockVector<NumberType> &data() override { return m_data; }
    virtual const BlockVector<NumberType> &data() const override { return m_data; }

    /// The spatial dofs (block 0).
    virtual Vector<NumberType> &spatial_data() override { return m_data.block(0); }
    virtual const Vector<NumberType> &spatial_data() const override { return m_data.block(0); }

    /// The variables (block 1).
    virtual Vector<NumberType> &variable_data() override { return m_data.block(1); }
    virtual const Vector<NumberType> &variable_data() const override { return m_data.block(1); }

  private:
    BlockVector<NumberType> m_data;
  };

  namespace FE
  {
    /**
     * @brief State of a flow on a finite-element discretization (CG, DG, LDG): FE dofs (block 0) and variables
     * (block 1, present only if the model has variables).
     *
     * Usage: `FE::FlowingVariables initial_condition(discretization); initial_condition.interpolate(model);`.
     *
     * @tparam Discretization The discretization, e.g. CG::Discretization<Model, Mesh>.
     */
    template <typename Discretization>
    class FlowingVariables
        : public AbstractFlowingVariables<typename Discretization::NumberType, typename Discretization::VectorType>
    {
    public:
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;
      using BlockVectorType = get_type::BlockVectorType<VectorType>;
      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;

      /// Bind to @p discretization (kept by reference); call interpolate() to fill.
      FlowingVariables(const Discretization &discretization)
          : discretization(discretization), dof_handler(discretization.get_dof_handler())
      {
      }

      /**
       * @brief Size the state for the current mesh and set the initial condition.
       *
       * The FE functions are interpolated from `model.initial_condition(x, values)` at the support points; the
       * variables are set by `model.initial_condition_variables(v)`.
       */
      template <typename Model> void interpolate(const Model &model)
      {
        auto block_structure = discretization.get_block_structure();
        internal::reinit_la_block_vector(m_data, block_structure, discretization.get_locally_owned_dofs(),
                                         discretization.get_communicator());

        if constexpr (Model::Components::count_fe_functions() > 0) {
          auto interpolating_function = [&model](const auto &p, auto &values) { model.initial_condition(p, values); };
          internal::FunctionFromLambda<dim, NumberType> initial_condition_function(std::move(interpolating_function),
                                                                                   dof_handler.get_fe().n_components());
          VectorTools::interpolate(dof_handler, initial_condition_function, m_data.block(0));
        }
        if (m_data.n_blocks() > 1) {
          // The variables block is owned by rank 0; compute on every rank and keep the owned entries.
          VectorType scratch;
          internal::reinit_local_variables_vector(scratch, block_structure[1]);
          internal::compute_variables_into(m_data.block(1), scratch,
                                           [&](VectorType &out) { model.initial_condition_variables(out); });
        }
      }

      /// The full state: spatial dofs (block 0) and variables (block 1).
      virtual BlockVectorType &data() override { return m_data; }
      virtual const BlockVectorType &data() const override { return m_data; }

      /// The spatial dofs (block 0).
      virtual VectorType &spatial_data() override { return m_data.block(0); }
      virtual const VectorType &spatial_data() const override { return m_data.block(0); }

      /// The variables (block 1); throws std::logic_error if the model has no variables.
      virtual VectorType &variable_data() override { return m_data.block(variables_block()); }
      virtual const VectorType &variable_data() const override { return m_data.block(variables_block()); }

    private:
      /// Index of the variables block; throws if the model has no variables.
      unsigned int variables_block() const
      {
        if (m_data.n_blocks() < 2)
          throw std::logic_error("FlowingVariables::variable_data: the model has no variables.");
        return 1;
      }

      const Discretization &discretization;
      const DoFHandler<dim> &dof_handler;
      BlockVectorType m_data;
    };
  } // namespace FE

  namespace FV
  {
    /**
     * @brief State of a flow on a finite-volume discretization: cell averages (block 0) and variables (block 1, present
     * only if the model has variables).
     *
     * Unlike FE::FlowingVariables, the initial condition is averaged over each cell, not interpolated.
     *
     * @tparam Discretization The FV discretization.
     */
    template <typename Discretization>
    class FlowingVariables
        : public AbstractFlowingVariables<typename Discretization::NumberType, typename Discretization::VectorType>
    {
    public:
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;
      using BlockVectorType = get_type::BlockVectorType<VectorType>;
      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;

      /// Bind to @p discretization (kept by reference); call interpolate() to fill.
      FlowingVariables(const Discretization &discretization)
          : discretization(discretization), dof_handler(discretization.get_dof_handler())
      {
      }

      /**
       * @brief Size the state for the current mesh and set the initial condition: each cell value is the average of
       * `model.initial_condition(x, values)` over the cell (2-point Gauss rule per direction); the variables are set by
       * `model.initial_condition_variables(v)`.
       */
      template <typename Model> void interpolate(const Model &model)
      {
        auto block_structure = discretization.get_block_structure();
        internal::reinit_la_block_vector(m_data, block_structure, discretization.get_locally_owned_dofs(),
                                         discretization.get_communicator());

        if constexpr (Model::Components::count_fe_functions() > 0) {
          const auto &fe = discretization.get_fe();
          const unsigned int n_components = fe.n_components();

          QGauss<dim> quadrature(2);
          FEValues<dim> fe_values(discretization.get_mapping(), fe, quadrature,
                                  update_quadrature_points | update_JxW_values);

          std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
          Vector<NumberType> values(n_components);
          std::vector<NumberType> averaged_values(n_components);

          for (const auto &cell : dof_handler.active_cell_iterators()) {
            // Each rank writes the cells it owns.
            if (!cell->is_locally_owned()) continue;
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            std::fill(averaged_values.begin(), averaged_values.end(), NumberType(0.0));

            NumberType cell_measure = 0.0;
            for (unsigned int q = 0; q < quadrature.size(); ++q) {
              values = 0.0;
              model.initial_condition(fe_values.quadrature_point(q), values);

              const NumberType weight = fe_values.JxW(q);
              cell_measure += weight;
              for (unsigned int c = 0; c < n_components; ++c)
                averaged_values[c] += weight * values[c];
            }

            AssertThrow(cell_measure > NumberType(0.0),
                        ExcMessage("Cannot initialize FV cell average on a cell with non-positive measure."));
            for (auto &value : averaged_values)
              value /= cell_measure;

            for (unsigned int local_dof = 0; local_dof < fe.dofs_per_cell; ++local_dof) {
              const auto component = fe.system_to_component_index(local_dof).first;
              m_data.block(0)[dof_indices[local_dof]] = averaged_values[component];
            }
          }
          // Each owned cell is written once by its owner.
          m_data.block(0).compress(dealii::VectorOperation::insert);
        }
        if (m_data.n_blocks() > 1) {
          // The variables block is owned by rank 0; compute on every rank and keep the owned entries.
          VectorType scratch;
          internal::reinit_local_variables_vector(scratch, block_structure[1]);
          internal::compute_variables_into(m_data.block(1), scratch,
                                           [&](VectorType &out) { model.initial_condition_variables(out); });
        }
      }

      /// The full state: spatial dofs (block 0) and variables (block 1).
      virtual BlockVectorType &data() override { return m_data; }
      virtual const BlockVectorType &data() const override { return m_data; }

      /// The spatial dofs (block 0).
      virtual VectorType &spatial_data() override { return m_data.block(0); }
      virtual const VectorType &spatial_data() const override { return m_data.block(0); }

      /// The variables (block 1); throws std::logic_error if the model has no variables.
      virtual VectorType &variable_data() override { return m_data.block(variables_block()); }
      virtual const VectorType &variable_data() const override { return m_data.block(variables_block()); }

    private:
      /// Index of the variables block; throws if the model has no variables.
      unsigned int variables_block() const
      {
        if (m_data.n_blocks() < 2)
          throw std::logic_error("FlowingVariables::variable_data: the model has no variables.");
        return 1;
      }

      const Discretization &discretization;
      const DoFHandler<dim> &dof_handler;
      BlockVectorType m_data;
    };
  } // namespace FV

} // namespace DiFfRG
