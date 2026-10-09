#pragma once

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
   * @brief A class to set up initial data for whatever discretization we have chosen.
   *        Also used to switch/manage memory, vectors, matrices over interfaces between spatial discretization and
   * separate variables.
   *
   * @tparam Discretization Spatial Discretization used in the system
   */
  template <typename NT = double> class FlowingVariables : public AbstractFlowingVariables<NT>
  {
  public:
    using NumberType = NT;

    /**
     * @brief Construct a new Flowing Variables object
     */
    FlowingVariables() {}

    /**
     * @brief Interpolates the initial condition from a numerical model.
     *
     * @param model The model to interpolate from. Must provide a method initial_condition(const Point<dim> &,
     * Vector<NumberType> &)
     */
    template <typename Model> void interpolate(const Model &model)
    {
      std::vector<uint> block_structure{0};
      block_structure.push_back(Model::Components::count_variables());
      m_data = (block_structure);
      if (m_data.n_blocks() > 1) model.initial_condition_variables(m_data.block(1));
    }

    /**
     * @brief Obtain the data vector holding both spatial (block 0) and variable (block 1) data.
     *
     * @return BlockVector<NumberType>& The data vector.
     */
    virtual BlockVector<NumberType> &data() override { return m_data; }
    virtual const BlockVector<NumberType> &data() const override { return m_data; }

    /**
     * @brief Obtain the spatial data vector.
     *
     * @return Vector<NumberType>& The spatial data vector.
     */
    virtual Vector<NumberType> &spatial_data() override { return m_data.block(0); }
    virtual const Vector<NumberType> &spatial_data() const override { return m_data.block(0); }

    /**
     * @brief Obtain the variable data vector.
     *
     * @return Vector<NumberType>& The variable data vector.
     */
    virtual Vector<NumberType> &variable_data() override { return m_data.block(1); }
    virtual const Vector<NumberType> &variable_data() const override { return m_data.block(1); }

  private:
    BlockVector<NumberType> m_data;
  };

  namespace FE
  {
    /**
     * @brief A class to set up initial data for whatever discretization we have chosen.
     *        Also used to switch/manage memory, vectors, matrices over interfaces between spatial discretization and
     * separate variables.
     *
     * @tparam Discretization Spatial Discretization used in the system
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

      /**
       * @brief Construct a new Flowing Variables object
       *
       * @param discretization The spatial discretization to use
       */
      FlowingVariables(const Discretization &discretization)
          : discretization(discretization), dof_handler(discretization.get_dof_handler())
      {
      }

      /**
       * @brief Interpolates the initial condition from a numerical model.
       *
       * @param model The model to interpolate from. Must provide a method initial_condition(const Point<dim> &,
       * Vector<NumberType> &)
       */
      template <typename Model> void interpolate(const Model &model)
      {
        auto block_structure = discretization.get_block_structure();
        reinit_la_block_vector(m_data, block_structure, discretization.get_locally_owned_dofs(),
                               discretization.get_communicator());

        if constexpr (Model::Components::count_fe_functions() > 0) {
          auto interpolating_function = [&model](const auto &p, auto &values) { model.initial_condition(p, values); };
          internal::FunctionFromLambda<dim, NumberType> initial_condition_function(std::move(interpolating_function),
                                                                                   dof_handler.get_fe().n_components());
          VectorTools::interpolate(dof_handler, initial_condition_function, m_data.block(0));
        }
        if (m_data.n_blocks() > 1) {
          // Under distribution rank 0 owns the variables block outright; the model writes every entry on every rank,
          // which elsewhere would stage off-process inserts that the next compress() of the block flushes over the
          // owner's values. compute_variables_into lands only what each rank owns.
          VectorType scratch;
          reinit_local_variables_vector(scratch, block_structure[1]);
          compute_variables_into(m_data.block(1), scratch,
                                 [&](VectorType &out) { model.initial_condition_variables(out); });
        }
      }

      /**
       * @brief Obtain the data vector holding both spatial (block 0) and variable (block 1) data.
       *
       * @return BlockVectorType& The data vector.
       */
      virtual BlockVectorType &data() override { return m_data; }
      virtual const BlockVectorType &data() const override { return m_data; }

      /**
       * @brief Obtain the spatial data vector.
       *
       * @return VectorType& The spatial data vector.
       */
      virtual VectorType &spatial_data() override { return m_data.block(0); }
      virtual const VectorType &spatial_data() const override { return m_data.block(0); }

      /**
       * @brief Obtain the variable data vector.
       *
       * @return VectorType& The variable data vector.
       */
      virtual VectorType &variable_data() override { return m_data.block(1); }
      virtual const VectorType &variable_data() const override { return m_data.block(1); }

    private:
      const Discretization &discretization;
      const DoFHandler<dim> &dof_handler;
      BlockVectorType m_data;
    };
  } // namespace FE

  namespace FV
  {
    /**
     * @brief A class to set up cell-averaged initial data for finite-volume systems.
     *
     * FV unknowns represent cell averages. Unlike FE::FlowingVariables, this class
     * integrates the model's pointwise initial condition over each active cell.
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

      /**
       * @brief Construct a new Flowing Variables object
       *
       * @param discretization The spatial discretization to use
       */
      FlowingVariables(const Discretization &discretization)
          : discretization(discretization), dof_handler(discretization.get_dof_handler())
      {
      }

      /**
       * @brief Initializes FV data with cell averages of the model initial condition.
       *
       * @param model The model to initialize from. Must provide
       * initial_condition(const Point<dim> &, Vector<NumberType> &).
       */
      template <typename Model> void interpolate(const Model &model)
      {
        auto block_structure = discretization.get_block_structure();
        reinit_la_block_vector(m_data, block_structure, discretization.get_locally_owned_dofs(),
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
            // Always true on a serial mesh, so this costs the serial path nothing. On a
            // partitioned one it stops every rank writing every cell average: those writes go
            // through a single-valued insert, so without the filter the result is only correct by
            // accident, and the compress() below would have nothing well-defined to do.
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
          // insert, not add: each owned cell's average is written exactly once by its owner. A
          // no-op for the serial vector types.
          m_data.block(0).compress(dealii::VectorOperation::insert);
        }
        if (m_data.n_blocks() > 1) {
          // Under distribution rank 0 owns the variables block outright; the model writes every entry on every rank,
          // which elsewhere would stage off-process inserts that the next compress() of the block flushes over the
          // owner's values. compute_variables_into lands only what each rank owns.
          VectorType scratch;
          reinit_local_variables_vector(scratch, block_structure[1]);
          compute_variables_into(m_data.block(1), scratch,
                                 [&](VectorType &out) { model.initial_condition_variables(out); });
        }
      }

      /**
       * @brief Obtain the data vector holding both spatial (block 0) and variable (block 1) data.
       *
       * @return BlockVectorType& The data vector.
       */
      virtual BlockVectorType &data() override { return m_data; }
      virtual const BlockVectorType &data() const override { return m_data; }

      /**
       * @brief Obtain the spatial data vector.
       *
       * @return VectorType& The spatial data vector.
       */
      virtual VectorType &spatial_data() override { return m_data.block(0); }
      virtual const VectorType &spatial_data() const override { return m_data.block(0); }

      /**
       * @brief Obtain the variable data vector.
       *
       * @return VectorType& The variable data vector.
       */
      virtual VectorType &variable_data() override { return m_data.block(1); }
      virtual const VectorType &variable_data() const override { return m_data.block(1); }

    private:
      const Discretization &discretization;
      const DoFHandler<dim> &dof_handler;
      BlockVectorType m_data;
    };
  } // namespace FV

} // namespace DiFfRG
