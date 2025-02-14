#pragma once

// external libraries
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <spdlog/spdlog.h>
#include <tbb/tbb.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/EoM.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/data_output.hh>

namespace DiFfRG
{
  using namespace dealii;
  using std::array;

  /**
   * @brief The basic assembler that can be used for any standard CG scheme with flux and source.
   *
   * @tparam Model The model class which contains the physical equations.
   */
  template <typename Discretization_, typename Model_>
  class FEMAssembler : public AbstractAssembler<typename Discretization_::VectorType,
                                                typename Discretization_::SparseMatrixType, Discretization_::dim>
  {
  protected:
    constexpr static int nothing = 0;

    template <typename... T> static constexpr auto v_tie(T &&...t)
    {
      return named_tuple<std::tuple<T &...>, "variables", "extractors">(std::tie(t...));
    }

    template <typename... T> static constexpr auto e_tie(T &&...t)
    {
      return named_tuple<std::tuple<T &...>, "fe_functions", "fe_derivatives", "fe_hessians", "extractors",
                         "variables">(std::tie(t...));
    }

  public:
    using Discretization = Discretization_;
    using Model = Model_;
    using NumberType = typename Discretization::NumberType;
    using VectorType = typename Discretization::VectorType;

    using Components = typename Discretization::Components;
    static constexpr uint dim = Discretization::dim;

    FEMAssembler(Discretization &discretization, Model &model, const JSONValue &json)
        : discretization(discretization), model(model), fe(discretization.get_fe()),
          dof_handler(discretization.get_dof_handler()), mapping(discretization.get_mapping()),
          threads(json.get_uint("/discretization/threads")), batch_size(json.get_uint("/discretization/batch_size")),
          EoM_cell(*(dof_handler.active_cell_iterators().end())),
          old_EoM_cell(*(dof_handler.active_cell_iterators().end())),
          EoM_abs_tol(json.get_double("/discretization/EoM_abs_tol")),
          EoM_max_iter(json.get_uint("/discretization/EoM_max_iter"))
    {
      if (this->threads == 0) this->threads = dealii::MultithreadInfo::n_threads() / 2;
      spdlog::get("log")->info("FEM: Using {} threads for assembly.", threads);
    }

    virtual IndexSet get_differential_indices() const override
    {
      ComponentMask component_mask(model.template differential_components<dim>());
      return DoFTools::extract_dofs(dof_handler, component_mask);
    }

    virtual void attach_data_output(DataOutput<dim, VectorType> &data_out, const VectorType &solution,
                                    const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                    const VectorType &residual = VectorType()) override
    {
      const auto fe_function_names = Components::FEFunction_Descriptor::get_names_vector();
      std::vector<std::string> fe_function_names_residual;
      for (const auto &name : fe_function_names)
        fe_function_names_residual.push_back(name + "_residual");
      std::vector<std::string> fe_function_names_dot;
      for (const auto &name : fe_function_names)
        fe_function_names_dot.push_back(name + "_dot");

      auto &fe_out = data_out.fe_output();
      fe_out.attach(dof_handler, solution, fe_function_names);
      if (dt_solution.size() > 0) fe_out.attach(dof_handler, dt_solution, fe_function_names_dot);
      if (residual.size() > 0) fe_out.attach(dof_handler, residual, fe_function_names_residual);

      readouts(data_out, solution, variables);
    }

    const auto &get_discretization() const { return discretization; }
    auto &get_discretization() { return discretization; }

    virtual void reinit() override
    {
      // Boundary dofs
      std::vector<IndexSet> component_boundary_dofs(Components::count_fe_functions());
      for (uint c = 0; c < Components::count_fe_functions(); ++c) {
        ComponentMask component_mask(Components::count_fe_functions(), false);
        component_mask.set(c, true);
        component_boundary_dofs[c] = DoFTools::extract_boundary_dofs(dof_handler, component_mask);
      }
      std::vector<std::vector<Point<dim>>> component_boundary_points(Components::count_fe_functions());
      for (uint c = 0; c < Components::count_fe_functions(); ++c) {
        component_boundary_points[c].resize(component_boundary_dofs[c].n_elements());
        for (uint i = 0; i < component_boundary_dofs[c].n_elements(); ++i)
          component_boundary_points[c][i] =
              discretization.get_support_point(component_boundary_dofs[c].nth_index_in_set(i));
      }

      auto &constraints = discretization.get_constraints();
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      model.affine_constraints(constraints, component_boundary_dofs, component_boundary_points);
      constraints.close();
    }

    virtual void rebuild_jacobian_sparsity() = 0;

    virtual void set_time(double t) override { model.set_time(t); }

    virtual void refinement_indicator(Vector<double> & /*indicator*/, const VectorType & /*solution*/) = 0;

    virtual void residual_variables(VectorType &residual, const VectorType &variables,
                                    const VectorType &spatial_solution) override
    {
      Timer timer;
      std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
      if constexpr (Components::count_extractors() > 0)
        extract(__extracted_data, spatial_solution, variables, true, false, false);
      const auto &extracted_data = __extracted_data;
      model.dt_variables(residual, v_tie(variables, extracted_data));
      timings_variable_residual.push_back(timer.wall_time());
    };
    virtual void jacobian_variables(FullMatrix<NumberType> &jacobian, const VectorType &variables,
                                    const VectorType &spatial_solution) override
    {
      Timer timer;
      std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
      if constexpr (Components::count_extractors() > 0)
        extract(__extracted_data, spatial_solution, variables, true, false, false);
      const auto &extracted_data = __extracted_data;
      model.template jacobian_variables<0>(jacobian, v_tie(variables, extracted_data));
      timings_variable_jacobian.push_back(timer.wall_time());
    };

    void readouts(DataOutput<dim, VectorType> &data_out, const VectorType &solution_global,
                  const VectorType &variables) const
    {
      auto helper = [&](auto EoMfun, auto outputter) {
        auto EoM_cell = this->EoM_cell;
        auto EoM = get_EoM_point(
            EoM_cell, solution_global, dof_handler, mapping, EoMfun, [&](const auto &p, const auto &) { return p; },
            EoM_abs_tol, EoM_max_iter);
        auto EoM_unit = mapping.transform_real_to_unit_cell(EoM_cell, EoM);

        Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
        std::vector<Tensor<1, dim, typename VectorType::value_type>> gradients(dof_handler.get_fe().n_components());

        FEValues<dim> fe_v(mapping, fe, EoM_unit,
                           update_values | update_gradients | update_quadrature_points | update_JxW_values |
                               update_hessians);
        fe_v.reinit(EoM_cell);

        std::vector<Vector<NumberType>> solution{Vector<NumberType>(Components::count_fe_functions())};
        std::vector<std::vector<Tensor<1, dim, NumberType>>> solution_grad{
            std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
        std::vector<std::vector<Tensor<2, dim, NumberType>>> solution_hess{
            std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
        fe_v.get_function_values(solution_global, solution);
        fe_v.get_function_gradients(solution_global, solution_grad);
        fe_v.get_function_hessians(solution_global, solution_hess);

        std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
        if constexpr (Components::count_extractors() > 0)
          extract(__extracted_data, solution_global, variables, true, false, false);
        const auto &extracted_data = __extracted_data;

        outputter(data_out, EoM, e_tie(solution[0], solution_grad[0], solution_hess[0], extracted_data, variables));
      };
      model.template readouts_multiple(helper, data_out);
    }

    void extract(std::array<NumberType, Components::count_extractors()> &data, const VectorType &solution_global,
                 const VectorType &variables, bool search_EoM, bool set_EoM, bool postprocess) const
    {
      auto EoM = this->EoM;
      auto EoM_cell = this->EoM_cell;
      if (search_EoM || EoM_cell == *(dof_handler.active_cell_iterators().end()))
        EoM = get_EoM_point(
            EoM_cell, solution_global, dof_handler, mapping,
            [&](const auto &p, const auto &values) { return model.EoM(p, values); },
            [&](const auto &p, const auto &values) { return postprocess ? model.EoM_postprocess(p, values) : p; },
            EoM_abs_tol, EoM_max_iter);
      if (set_EoM) {
        this->EoM = EoM;
        this->EoM_cell = EoM_cell;
      }
      auto EoM_unit = mapping.transform_real_to_unit_cell(EoM_cell, EoM);

      Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
      std::vector<Tensor<1, dim, typename VectorType::value_type>> gradients(dof_handler.get_fe().n_components());

      FEValues<dim> fe_v(mapping, fe, EoM_unit,
                         update_values | update_gradients | update_quadrature_points | update_JxW_values |
                             update_hessians);
      fe_v.reinit(EoM_cell);

      std::vector<Vector<NumberType>> solution{Vector<NumberType>(Components::count_fe_functions())};
      std::vector<std::vector<Tensor<1, dim, NumberType>>> solution_grad{
          std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
      std::vector<std::vector<Tensor<2, dim, NumberType>>> solution_hess{
          std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
      fe_v.get_function_values(solution_global, solution);
      fe_v.get_function_gradients(solution_global, solution_grad);
      fe_v.get_function_hessians(solution_global, solution_hess);

      model.template extract(data, EoM, e_tie(solution[0], solution_grad[0], solution_hess[0], nothing, variables));
    }

    bool jacobian_extractors(FullMatrix<NumberType> &extractor_jacobian, const VectorType &solution_global,
                             const VectorType &variables)
    {
      if (extractor_jacobian_u.m() != Components::count_extractors() ||
          extractor_jacobian_u.n() != Components::count_fe_functions())
        extractor_jacobian_u = FullMatrix<NumberType>(Components::count_extractors(), Components::count_fe_functions());
      if (extractor_jacobian_du.m() != Components::count_extractors() ||
          extractor_jacobian_du.n() != Components::count_fe_functions() * dim)
        extractor_jacobian_du =
            FullMatrix<NumberType>(Components::count_extractors(), Components::count_fe_functions() * dim);
      if (extractor_jacobian_ddu.m() != Components::count_extractors() ||
          extractor_jacobian_ddu.n() != Components::count_fe_functions() * dim * dim)
        extractor_jacobian_ddu =
            FullMatrix<NumberType>(Components::count_extractors(), Components::count_fe_functions() * dim * dim);

      EoM = get_EoM_point(
          EoM_cell, solution_global, dof_handler, mapping,
          [&](const auto &p, const auto &values) { return model.EoM(p, values); },
          [&](const auto &p, const auto &values) { return model.EoM_postprocess(p, values); }, EoM_abs_tol,
          EoM_max_iter);
      auto EoM_unit = mapping.transform_real_to_unit_cell(EoM_cell, EoM);
      bool new_cell = (old_EoM_cell != EoM_cell);
      old_EoM_cell = EoM_cell;

      Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
      std::vector<Tensor<1, dim, typename VectorType::value_type>> gradients(dof_handler.get_fe().n_components());

      FEValues<dim> fe_v(mapping, fe, EoM_unit,
                         update_values | update_gradients | update_quadrature_points | update_JxW_values |
                             update_hessians);
      fe_v.reinit(EoM_cell);

      const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();
      if (new_cell) {
        // spdlog::get("log")->info("FEM: Rebuilding the jacobian sparsity pattern");
        extractor_dof_indices.resize(n_dofs);
        EoM_cell->get_dof_indices(extractor_dof_indices);
        rebuild_jacobian_sparsity();
      }

      std::vector<Vector<NumberType>> solution{Vector<NumberType>(Components::count_fe_functions())};
      std::vector<std::vector<Tensor<1, dim, NumberType>>> solution_grad{
          std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
      std::vector<std::vector<Tensor<2, dim, NumberType>>> solution_hess{
          std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
      fe_v.get_function_values(solution_global, solution);
      fe_v.get_function_gradients(solution_global, solution_grad);
      fe_v.get_function_hessians(solution_global, solution_hess);

      extractor_jacobian_u = 0;
      extractor_jacobian_du = 0;
      extractor_jacobian_ddu = 0;
      model.template jacobian_extractors<0>(extractor_jacobian_u, EoM,
                                            e_tie(solution[0], solution_grad[0], solution_hess[0], nothing, variables));
      model.template jacobian_extractors<1>(extractor_jacobian_du, EoM,
                                            e_tie(solution[0], solution_grad[0], solution_hess[0], nothing, variables));
      model.template jacobian_extractors<2>(extractor_jacobian_ddu, EoM,
                                            e_tie(solution[0], solution_grad[0], solution_hess[0], nothing, variables));

      if (extractor_jacobian.m() != Components::count_extractors() || extractor_jacobian.n() != n_dofs)
        extractor_jacobian = FullMatrix<NumberType>(Components::count_extractors(), n_dofs);

      for (uint e = 0; e < Components::count_extractors(); ++e)
        for (uint i = 0; i < n_dofs; ++i) {
          const auto component_i = fe_v.get_fe().system_to_component_index(i).first;
          extractor_jacobian(e, i) =
              extractor_jacobian_u(e, component_i) * fe_v.shape_value_component(i, 0, component_i);
          for (uint d1 = 0; d1 < dim; ++d1) {
            extractor_jacobian(e, i) +=
                extractor_jacobian_du(e, component_i * dim + d1) * fe_v.shape_grad_component(i, 0, component_i)[d1];
            for (uint d2 = 0; d2 < dim; ++d2)
              extractor_jacobian(e, i) += extractor_jacobian_ddu(e, component_i * dim * dim + d1 * dim + d2) *
                                          fe_v.shape_hessian_component(i, 0, component_i)[d1][d2];
          }
        }

      return new_cell;
    }

    double average_time_variable_residual_assembly()
    {
      double t = 0.;
      double n = timings_variable_residual.size();
      for (const auto &t_ : timings_variable_residual)
        t += t_ / n;
      return t;
    }
    uint num_variable_residuals() const { return timings_variable_residual.size(); }

    double average_time_variable_jacobian_assembly()
    {
      double t = 0.;
      double n = timings_variable_jacobian.size();
      for (const auto &t_ : timings_variable_jacobian)
        t += t_ / n;
      return t;
    }
    uint num_variable_jacobians() const { return timings_variable_jacobian.size(); }

  protected:
    Discretization &discretization;
    Model &model;
    const FiniteElement<dim> &fe;
    const DoFHandler<dim> &dof_handler;
    const Mapping<dim> &mapping;

    uint threads;
    uint batch_size;

    mutable typename DoFHandler<dim>::cell_iterator EoM_cell;
    typename DoFHandler<dim>::cell_iterator old_EoM_cell;
    const double EoM_abs_tol;
    const uint EoM_max_iter;
    mutable Point<dim> EoM;
    FullMatrix<NumberType> extractor_jacobian;
    FullMatrix<NumberType> extractor_jacobian_u;
    FullMatrix<NumberType> extractor_jacobian_du;
    FullMatrix<NumberType> extractor_jacobian_ddu;
    std::vector<types::global_dof_index> extractor_dof_indices;

    std::vector<double> timings_variable_residual;
    std::vector<double> timings_variable_jacobian;
  };
} // namespace DiFfRG
