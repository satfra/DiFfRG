#pragma once

// standard library
#include <memory>
#include <sstream>

// external libraries
#include <deal.II/base/multithread_info.h>
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
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/EoM.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/data_output.hh>

namespace DiFfRG
{
  namespace LDG
  {
    using namespace dealii;
    using std::array, std::vector, std::unique_ptr;

    template <typename Discretization_, typename Model_>
    class LDGAssemblerBase : public AbstractAssembler<typename Discretization_::VectorType,
                                                      typename Discretization_::SparseMatrixType, Discretization_::dim>
    {
    public:
      using Discretization = Discretization_;
      using Model = Model_;
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;

      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;

      LDGAssemblerBase(Discretization &discretization, Model &model, const JSONValue &json)
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

    namespace internal
    {
      /**
       * @brief Class to hold data for each assembly thread, i.e. FEValues for cells, interfaces, as well as
       * pre-allocated data structures for the solutions
       */
      template <typename Discretization> struct ScratchData {
        static constexpr uint dim = Discretization::dim;
        using NumberType = typename Discretization::NumberType;
        static constexpr uint n_fe_subsystems = Discretization::Components::count_fe_subsystems();
        using Iterator = typename DoFHandler<dim>::active_cell_iterator;
        using t_Iterator = typename Triangulation<dim>::active_cell_iterator;

        ScratchData(const Mapping<dim> &mapping, const vector<const DoFHandler<dim> *> &dofh,
                    const Quadrature<dim> &quadrature, const Quadrature<dim - 1> &quadrature_face,
                    const UpdateFlags update_flags = update_values | update_gradients | update_quadrature_points |
                                                     update_JxW_values,
                    const UpdateFlags interface_update_flags = update_values | update_gradients |
                                                               update_quadrature_points | update_JxW_values |
                                                               update_normal_vectors)
        {
          AssertThrow(dofh.size() >= n_fe_subsystems,
                      StandardExceptions::ExcDimensionMismatch(dofh.size(), n_fe_subsystems));

          for (uint i = 0; i < n_fe_subsystems; ++i) {
            const auto &fe = dofh[i]->get_fe();
            fe_values[i] = unique_ptr<FEValues<dim>>(new FEValues<dim>(mapping, fe, quadrature, update_flags));
            fe_interface_values[i] = unique_ptr<FEInterfaceValues<dim>>(
                new FEInterfaceValues<dim>(mapping, fe, quadrature_face, interface_update_flags));
            fe_boundary_values[i] = unique_ptr<FEFaceValues<dim>>(
                new FEFaceValues<dim>(mapping, fe, quadrature_face, interface_update_flags));

            n_components[i] = fe.n_components();
            solution[i].resize(quadrature.size(), Vector<NumberType>(n_components[i]));
            solution_interface[0][i].resize(quadrature_face.size(), Vector<NumberType>(n_components[i]));
            solution_interface[1][i].resize(quadrature_face.size(), Vector<NumberType>(n_components[i]));

            cell[i] = dofh[i]->begin_active();
            ncell[i] = dofh[i]->begin_active();
          }
          solution_dot.resize(quadrature.size(), Vector<NumberType>(n_components[0]));
        }

        ScratchData(const ScratchData<Discretization> &scratch_data)
        {
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            const auto &old_fe = scratch_data.fe_values[i];
            const auto &old_fe_i = scratch_data.fe_interface_values[i];
            const auto &old_fe_b = scratch_data.fe_boundary_values[i];

            fe_values[i] = unique_ptr<FEValues<dim>>(new FEValues<dim>(
                old_fe->get_mapping(), old_fe->get_fe(), old_fe->get_quadrature(), old_fe->get_update_flags()));
            fe_interface_values[i] = unique_ptr<FEInterfaceValues<dim>>(new FEInterfaceValues<dim>(
                old_fe_i->get_mapping(), old_fe_i->get_fe(), old_fe_i->get_quadrature(), old_fe_i->get_update_flags()));
            fe_boundary_values[i] = unique_ptr<FEFaceValues<dim>>(new FEFaceValues<dim>(
                old_fe_b->get_mapping(), old_fe_b->get_fe(), old_fe_b->get_quadrature(), old_fe_b->get_update_flags()));

            n_components[i] = scratch_data.n_components[i];
            solution[i].resize(scratch_data.solution[i].size(), Vector<NumberType>(n_components[i]));
            solution_interface[0][i].resize(scratch_data.solution_interface[0][i].size(),
                                            Vector<NumberType>(n_components[i]));
            solution_interface[1][i].resize(scratch_data.solution_interface[1][i].size(),
                                            Vector<NumberType>(n_components[i]));

            cell[i] = scratch_data.cell[i];
            ncell[i] = scratch_data.ncell[i];
          }
          solution_dot.resize(scratch_data.solution_dot.size(), Vector<NumberType>(n_components[0]));
        }

        const auto &new_fe_values(const t_Iterator &t_cell)
        {
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            cell[i]->copy_from(*t_cell);
            fe_values[i]->reinit(cell[i]);
          }
          return fe_values;
        }
        const auto &new_fe_interface_values(const t_Iterator &t_cell, uint f, uint sf, const t_Iterator &t_ncell,
                                            uint nf, unsigned int nsf)
        {
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            cell[i]->copy_from(*t_cell);
            ncell[i]->copy_from(*t_ncell);
            fe_interface_values[i]->reinit(cell[i], f, sf, ncell[i], nf, nsf);
          }
          return fe_interface_values;
        }
        const auto &new_fe_boundary_values(const t_Iterator &t_cell, uint face_no)
        {
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            cell[i]->copy_from(*t_cell);
            fe_boundary_values[i]->reinit(cell[i], face_no);
          }
          return fe_boundary_values;
        }

        array<uint, n_fe_subsystems> n_components;
        array<Iterator, n_fe_subsystems> cell;
        array<Iterator, n_fe_subsystems> ncell;

        array<unique_ptr<FEValues<dim>>, n_fe_subsystems> fe_values;
        array<unique_ptr<FEInterfaceValues<dim>>, n_fe_subsystems> fe_interface_values;
        array<unique_ptr<FEFaceValues<dim>>, n_fe_subsystems> fe_boundary_values;

        array<vector<Vector<NumberType>>, n_fe_subsystems> solution;
        vector<Vector<NumberType>> solution_dot;
        array<array<vector<Vector<NumberType>>, n_fe_subsystems>, 2> solution_interface;
      };

      template <typename NumberType> struct CopyData_R {
        struct CopyDataFace_R {
          Vector<NumberType> cell_residual;
          std::vector<types::global_dof_index> joint_dof_indices;
        };

        Vector<NumberType> cell_residual;
        Vector<NumberType> cell_mass;
        std::vector<types::global_dof_index> local_dof_indices;
        std::vector<CopyDataFace_R> face_data;

        template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell)
        {
          cell_residual.reinit(dofs_per_cell);
          cell_mass.reinit(dofs_per_cell);
          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);
        }

        template <int dim> CopyDataFace_R &new_face_data(const FEInterfaceValues<dim> &fe_iv)
        {
          face_data.emplace_back();
          auto &copy_data_face = face_data.back();
          copy_data_face.cell_residual.reinit(fe_iv.n_current_interface_dofs());
          copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();
          return copy_data_face;
        }
      };

      template <typename NumberType> struct CopyData_J {
        struct CopyDataFace_J {
          FullMatrix<NumberType> cell_jacobian;
          std::vector<types::global_dof_index> joint_dof_indices_from;
          std::vector<types::global_dof_index> joint_dof_indices_to;
        };

        FullMatrix<NumberType> cell_jacobian;
        std::vector<types::global_dof_index> local_dof_indices_from;
        std::vector<types::global_dof_index> local_dof_indices_to;
        std::vector<CopyDataFace_J> face_data;

        template <class Iterator>
        void reinit(const Iterator &cell_from, const Iterator &cell_to, uint dofs_per_cell_from, uint dofs_per_cell_to)
        {
          cell_jacobian.reinit(dofs_per_cell_to, dofs_per_cell_from);
          local_dof_indices_from.resize(dofs_per_cell_from);
          local_dof_indices_to.resize(dofs_per_cell_to);
          cell_from->get_dof_indices(local_dof_indices_from);
          cell_to->get_dof_indices(local_dof_indices_to);
        }

        template <int dim>
        CopyDataFace_J &new_face_data(const FEInterfaceValues<dim> &fe_iv_from, const FEInterfaceValues<dim> &fe_iv_to)
        {
          auto &copy_data_face = face_data.emplace_back();
          copy_data_face.cell_jacobian.reinit(fe_iv_to.n_current_interface_dofs(),
                                              fe_iv_from.n_current_interface_dofs());
          copy_data_face.joint_dof_indices_from = fe_iv_from.get_interface_dof_indices();
          copy_data_face.joint_dof_indices_to = fe_iv_to.get_interface_dof_indices();
          return copy_data_face;
        }
      };

      template <typename NumberType, uint n_fe_subsystems> struct CopyData_J_full {
        struct CopyDataFace_J {
          array<FullMatrix<NumberType>, n_fe_subsystems> cell_jacobian;
          FullMatrix<NumberType> extractor_cell_jacobian;
          array<vector<types::global_dof_index>, n_fe_subsystems> joint_dof_indices;
        };

        array<FullMatrix<NumberType>, n_fe_subsystems> cell_jacobian;
        FullMatrix<NumberType> cell_mass_jacobian;
        FullMatrix<NumberType> extractor_cell_jacobian;
        array<vector<types::global_dof_index>, n_fe_subsystems> local_dof_indices;
        vector<CopyDataFace_J> face_data;

        uint dofs_per_cell;

        template <class Iterator> void reinit(const array<Iterator, n_fe_subsystems> &cell, const uint n_extractors)
        {
          const uint n_dofs = cell[0]->get_fe().n_dofs_per_cell();
          dofs_per_cell = n_dofs;
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            const uint from_n_dofs = cell[i]->get_fe().n_dofs_per_cell();
            if (i == 0) cell_mass_jacobian.reinit(n_dofs, from_n_dofs);
            cell_jacobian[i].reinit(n_dofs, from_n_dofs);
            local_dof_indices[i].resize(from_n_dofs);
            cell[i]->get_dof_indices(local_dof_indices[i]);
          }
          if (n_extractors > 0) extractor_cell_jacobian.reinit(dofs_per_cell, n_extractors);
        }

        template <int dim>
        CopyDataFace_J &new_face_data(const array<unique_ptr<FEInterfaceValues<dim>>, n_fe_subsystems> &fe_iv,
                                      const uint n_extractors)
        {
          auto &copy_data_face = face_data.emplace_back();
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            copy_data_face.cell_jacobian[i].reinit(fe_iv[0]->n_current_interface_dofs(),
                                                   fe_iv[i]->n_current_interface_dofs());
            copy_data_face.joint_dof_indices[i] = fe_iv[i]->get_interface_dof_indices();
          }
          if (n_extractors > 0)
            copy_data_face.extractor_cell_jacobian.reinit(fe_iv[0]->n_current_interface_dofs(), n_extractors);
          return copy_data_face;
        }
      };

      template <typename NumberType> struct CopyData_I {
        struct CopyFaceData_I {
          std::array<uint, 2> cell_indices;
          std::array<double, 2> values;
        };
        std::vector<CopyFaceData_I> face_data;
        double value;
        uint cell_index;
      };
    } // namespace internal

    /**
     * @brief The LDG assembler that can be used for any LDG scheme, with as many levels as one wants.
     *
     * @tparam Discretization Discretization on which to assemble
     * @tparam Model The model class which contains the physical equations.
     */
    template <typename Discretization_, typename Model_>
    class Assembler : public LDGAssemblerBase<Discretization_, Model_>
    {
      using Base = LDGAssemblerBase<Discretization_, Model_>;

    public:
      using Discretization = Discretization_;
      using Model = Model_;
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;

      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;
      static constexpr uint stencil = Components::count_fe_subsystems();

    private:
      template <typename... T> auto fe_conv(std::tuple<T &...> &t) const
      {
        if constexpr (stencil == 2)
          return named_tuple<std::tuple<T &...>, "fe_functions", "LDG1", "extractors", "variables">(t);
        else if constexpr (stencil == 3)
          return named_tuple<std::tuple<T &...>, "fe_functions", "LDG1", "LDG2", "extractors", "variables">(t);
        else if constexpr (stencil == 4)
          return named_tuple<std::tuple<T &...>, "fe_functions", "LDG1", "LDG2", "LDG3", "extractors", "variables">(t);
        else
          throw std::runtime_error("Only <= 3 LDG subsystems are supported.");
      }

      template <typename... T> auto fe_more_conv(std::tuple<T &...> &t) const
      {
        if constexpr (stencil == 2)
          return named_tuple<std::tuple<T &...>, "fe_functions", "LDG1", "fe_derivatives", "fe_hessians", "extractors",
                             "variables">(t);
        else if constexpr (stencil == 3)
          return named_tuple<std::tuple<T &...>, "fe_functions", "LDG1", "LDG2", "fe_derivatives", "fe_hessians",
                             "extractors", "variables">(t);
        else if constexpr (stencil == 4)
          return named_tuple<std::tuple<T &...>, "fe_functions", "LDG1", "LDG2", "LDG3", "fe_derivatives",
                             "fe_hessians", "extractors", "variables">(t);
        else
          throw std::runtime_error("Only <= 3 LDG subsystems are supported.");
      }

      template <typename... T> auto ref_conv(std::tuple<T &...> &t) const
      {
        if constexpr (stencil == 2)
          return named_tuple<std::tuple<T &...>, "fe_functions", "LDG1">(t);
        else if constexpr (stencil == 3)
          return named_tuple<std::tuple<T &...>, "fe_functions", "LDG1", "LDG2">(t);
        else if constexpr (stencil == 4)
          return named_tuple<std::tuple<T &...>, "fe_functions", "LDG1", "LDG2", "LDG3">(t);
        else
          throw std::runtime_error("Only <= 3 LDG subsystems are supported.");
      }

    public:
      Assembler(Discretization &discretization, Model &model, const JSONValue &json)
          : Base(discretization, model, json),
            quadrature(fe.degree + 1 + json.get_uint("/discretization/overintegration")),
            quadrature_face(fe.degree + 1 + json.get_uint("/discretization/overintegration")),
            dof_handler_list(discretization.get_dof_handler_list())
      {
        static_assert(Components::count_fe_subsystems() > 1, "LDG must have a submodel with index 1.");
        reinit();
      }

      virtual void reinit_vector(VectorType &vec) const override { vec.reinit(dof_handler.n_dofs()); }

      /**
       * @brief Attach all intermediate (ldg) vectors to the data output
       *
       * @param data_out A DataOutput to which we attach data
       * @param sol The current global solution
       */
      virtual void attach_data_output(DataOutput<dim, VectorType> &data_out, const VectorType &solution,
                                      const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                      const VectorType &residual = VectorType()) override
      {
        rebuild_ldg_vectors(solution);
        readouts(data_out, solution, variables);

        const auto fe_function_names = Components::FEFunction_Descriptor::get_names_vector();
        std::vector<std::string> fe_function_names_residual;
        for (const auto &name : fe_function_names)
          fe_function_names_residual.push_back(name + "_residual");
        std::vector<std::string> fe_function_names_dot;
        for (const auto &name : fe_function_names)
          fe_function_names_dot.push_back(name + "_dot");

        auto &fe_out = data_out.fe_output();
        fe_out.attach(*dof_handler_list[0], solution, fe_function_names);
        if (dt_solution.size() > 0) fe_out.attach(dof_handler, dt_solution, fe_function_names_dot);
        if (residual.size() > 0) fe_out.attach(dof_handler, residual, fe_function_names_residual);
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

            // i do not understand why this is needed.
            component_mass_matrix_inverse.reinit(sparsity_pattern_mass.block(0, 0));
            mass_matrix.reinit(sparsity_pattern_mass);

            MatrixCreator::create_mass_matrix(*(dof_handler_list[0]), quadrature, mass_matrix,
                                              (const Function<dim, NumberType> *const)nullptr,
                                              discretization.get_constraints(0));
            build_inverse(mass_matrix.block(0, 0), component_mass_matrix_inverse);
          } else {
            sol_vector[i].reinit(dofs_per_component);
            sol_vector_tmp[i].reinit(dofs_per_component);
            ldg_matrix_built[i] = false;
            jacobian_tmp_built[i] = false;
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

        Base::reinit();

        vector<std::thread> threads;
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i)
          threads.emplace_back(std::thread(init_mass, i));
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i)
          threads.emplace_back(std::thread(init_jacobian, i));
        for (uint i = 1; i < Components::count_fe_subsystems(); ++i)
          threads.emplace_back(std::thread(init_ldg, i));
        for (auto &t : threads)
          t.join();

        timings_reinit.push_back(timer.wall_time());
      }

      virtual void rebuild_jacobian_sparsity() override
      {
        build_ldg_sparsity(sparsity_pattern_jacobian, *(dof_handler_list[0]), *(dof_handler_list[0]), stencil, true);
        for (uint k = 1; k < Components::count_fe_subsystems(); ++k)
          jacobian_tmp[k].reinit(sparsity_pattern_jacobian);
      }

      virtual void refinement_indicator(Vector<double> &indicator, const VectorType &solution_global) override
      {
        using Iterator = typename Triangulation<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_I<NumberType>;

        const auto cell_worker = [&](const Iterator &t_cell, Scratch &scratch_data, CopyData &copy_data) {
          const auto &fe_v = scratch_data.new_fe_values(t_cell);
          copy_data.cell_index = t_cell->active_cell_index();
          copy_data.value = 0;

          const auto &JxW = fe_v[0]->get_JxW_values();
          const auto &q_points = fe_v[0]->get_quadrature_points();
          const auto &q_indices = fe_v[0]->quadrature_point_indices();

          auto &solution = scratch_data.solution;
          fe_v[0]->get_function_values(solution_global, solution[0]);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i)
            fe_v[i]->get_function_values(sol_vector[i], solution[i]);

          double local_indicator = 0.;
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            auto sol_q = local_sol_q(solution, q_index);
            model.cell_indicator(local_indicator, x_q, ref_conv(sol_q));

            copy_data.value += JxW[q_index] * local_indicator;
          }
        };
        const auto face_worker = [&](const Iterator &t_cell, const uint &f, const uint &sf, const Iterator &t_ncell,
                                     const uint &nf, const unsigned int &nsf, Scratch &scratch_data,
                                     CopyData &copy_data) {
          const auto &fe_iv = scratch_data.new_fe_interface_values(t_cell, f, sf, t_ncell, nf, nsf);

          auto &copy_data_face = copy_data.face_data.emplace_back();
          copy_data_face.cell_indices[0] = t_cell->active_cell_index();
          copy_data_face.cell_indices[1] = t_ncell->active_cell_index();
          copy_data_face.values[0] = 0;
          copy_data_face.values[1] = 0;

          const auto &JxW = fe_iv[0]->get_JxW_values();
          const auto &q_points = fe_iv[0]->get_quadrature_points();
          const auto &q_indices = fe_iv[0]->quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_iv[0]->get_normal_vectors();
          array<double, 2> local_indicator{};

          auto &solution = scratch_data.solution_interface;
          fe_iv[0]->get_fe_face_values(0).get_function_values(solution_global, solution[0][0]);
          fe_iv[0]->get_fe_face_values(1).get_function_values(solution_global, solution[1][0]);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i) {
            fe_iv[i]->get_fe_face_values(0).get_function_values(sol_vector[i], solution[0][i]);
            fe_iv[i]->get_fe_face_values(1).get_function_values(sol_vector[i], solution[1][i]);
          }

          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            auto sol_q_s = local_sol_q(solution[0], q_index);
            auto sol_q_n = local_sol_q(solution[1], q_index);
            model.face_indicator(local_indicator, normals[q_index], x_q, ref_conv(sol_q_s), ref_conv(sol_q_n));

            copy_data_face.values[0] += JxW[q_index] * local_indicator[0] * (1. + t_cell->at_boundary());
            copy_data_face.values[1] += JxW[q_index] * local_indicator[1] * (1. + t_ncell->at_boundary());
          }
        };
        const auto copier = [&](const CopyData &c) {
          for (auto &cdf : c.face_data)
            for (uint j = 0; j < 2; ++j)
              indicator[cdf.cell_indices[j]] += cdf.values[j];
          indicator[c.cell_index] += c.value;
        };

        const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
        Scratch scratch_data(mapping, dof_handler_list, quadrature, quadrature_face, update_flags);
        CopyData copy_data;
        MeshWorker::AssembleFlags assemble_flags =
            MeshWorker::assemble_own_cells | MeshWorker::assemble_own_interior_faces_once;

        rebuild_ldg_vectors(solution_global);
        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, assemble_flags, nullptr, face_worker, threads, batch_size);
      }

      virtual const BlockSparsityPattern &get_sparsity_pattern_jacobian() const override
      {
        return sparsity_pattern_jacobian;
      }
      virtual const BlockSparseMatrix<NumberType> &get_mass_matrix() const override { return mass_matrix; }

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
        using Iterator = typename Triangulation<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_R<NumberType>;
        const auto &constraints = discretization.get_constraints();

        const auto cell_worker = [&](const Iterator &t_cell, Scratch &scratch_data, CopyData &copy_data) {
          const auto &fe_v = scratch_data.new_fe_values(t_cell);
          const uint n_dofs = fe_v[0]->get_fe().n_dofs_per_cell();
          copy_data.reinit(scratch_data.cell[0], n_dofs);

          const auto &JxW = fe_v[0]->get_JxW_values();
          const auto &q_points = fe_v[0]->get_quadrature_points();
          const auto &q_indices = fe_v[0]->quadrature_point_indices();

          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;
          fe_v[0]->get_function_values(solution_global, solution[0]);
          fe_v[0]->get_function_values(solution_global_dot, solution_dot);

          array<NumberType, Components::count_fe_functions(0)> mass{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.mass(mass, x_q, solution[0][q_index], solution_dot[q_index]);

            for (uint i = 0; i < n_dofs; ++i) {
              const auto component_i = fe_v[0]->get_fe().system_to_component_index(i).first;
              copy_data.cell_residual(i) += weight * JxW[q_index] * // dx
                                            fe_v[0]->shape_value_component(i, q_index, component_i) *
                                            mass[component_i]; // phi_i(x_q) * mass(x_q, u_q)
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, residual);
        };

        const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
        const MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_own_cells;
        Scratch scratch_data(mapping, dof_handler_list, quadrature, quadrature_face, update_flags);
        CopyData copy_data;

        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, assemble_flags, nullptr, nullptr, threads, batch_size);
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
        using Iterator = typename Triangulation<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_R<NumberType>;
        const auto &constraints = discretization.get_constraints();

        // Find the EoM and extract whatever data is needed for the model.
        std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
        if constexpr (Components::count_extractors() > 0)
          this->extract(__extracted_data, solution_global, variables, true, false, true);
        const auto &extracted_data = __extracted_data;

        const auto cell_worker = [&](const Iterator &t_cell, Scratch &scratch_data, CopyData &copy_data) {
          const auto &fe_v = scratch_data.new_fe_values(t_cell);
          const uint n_dofs = fe_v[0]->get_fe().n_dofs_per_cell();
          copy_data.reinit(scratch_data.cell[0], n_dofs);

          const auto &JxW = fe_v[0]->get_JxW_values();
          const auto &q_points = fe_v[0]->get_quadrature_points();
          const auto &q_indices = fe_v[0]->quadrature_point_indices();

          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;
          fe_v[0]->get_function_values(solution_global, solution[0]);
          fe_v[0]->get_function_values(solution_global_dot, solution_dot);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i)
            fe_v[i]->get_function_values(sol_vector[i], solution[i]);

          array<NumberType, Components::count_fe_functions(0)> mass{};
          array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> flux{};
          array<NumberType, Components::count_fe_functions(0)> source{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            auto sol_q = std::tuple_cat(local_sol_q(solution, q_index), std::tie(extracted_data, variables));
            model.mass(mass, x_q, solution[0][q_index], solution_dot[q_index]);
            model.flux(flux, x_q, fe_conv(sol_q));
            model.source(source, x_q, fe_conv(sol_q));

            for (uint i = 0; i < n_dofs; ++i) {
              const auto component_i = fe_v[0]->get_fe().system_to_component_index(i).first;
              copy_data.cell_residual(i) += weight * JxW[q_index] * // dx
                                            (-scalar_product(fe_v[0]->shape_grad_component(i, q_index, component_i),
                                                             flux[component_i]) // -dphi_i(x_q) * flux(x_q, u_q)
                                             + fe_v[0]->shape_value_component(i, q_index, component_i) *
                                                   (source[component_i])); // -phi_i(x_q) * source(x_q, u_q)
              copy_data.cell_mass(i) += weight_mass * JxW[q_index] *       // dx
                                        fe_v[0]->shape_value_component(i, q_index, component_i) *
                                        mass[component_i]; // phi_i(x_q) * mass(x_q, u_q)
            }
          }
        };
        const auto boundary_worker = [&](const Iterator &t_cell, const uint &face_no, Scratch &scratch_data,
                                         CopyData &copy_data) {
          const auto &fe_fv = scratch_data.new_fe_boundary_values(t_cell, face_no);
          const uint n_dofs = fe_fv[0]->get_fe().n_dofs_per_cell();

          const auto &JxW = fe_fv[0]->get_JxW_values();
          const auto &q_points = fe_fv[0]->get_quadrature_points();
          const auto &q_indices = fe_fv[0]->quadrature_point_indices();
          const auto &normals = fe_fv[0]->get_normal_vectors();

          auto &solution = scratch_data.solution;
          fe_fv[0]->get_function_values(solution_global, solution[0]);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i)
            fe_fv[i]->get_function_values(sol_vector[i], solution[i]);

          array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> numflux{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            auto sol_q = std::tuple_cat(local_sol_q(solution, q_index), std::tie(extracted_data, variables));
            model.boundary_numflux(numflux, normals[q_index], x_q, fe_conv(sol_q));

            for (uint i = 0; i < n_dofs; ++i) {
              const auto component_i = fe_fv[0]->get_fe().system_to_component_index(i).first;
              copy_data.cell_residual(i) +=
                  weight * JxW[q_index] * // weight * dx
                  (fe_fv[0]->shape_value_component(i, q_index, component_i) *
                   scalar_product(numflux[component_i], normals[q_index])); // phi_i(x_q) * numflux(x_q, u_q) * n(x_q)
            }
          }
        };
        const auto face_worker = [&](const Iterator &t_cell, const uint &f, const uint &sf, const Iterator &t_ncell,
                                     const uint &nf, const unsigned int &nsf, Scratch &scratch_data,
                                     CopyData &copy_data) {
          const auto &fe_iv = scratch_data.new_fe_interface_values(t_cell, f, sf, t_ncell, nf, nsf);
          const uint n_dofs = fe_iv[0]->n_current_interface_dofs();
          auto &copy_data_face = copy_data.new_face_data(*(fe_iv[0]));

          const auto &JxW = fe_iv[0]->get_JxW_values();
          const auto &q_points = fe_iv[0]->get_quadrature_points();
          const auto &q_indices = fe_iv[0]->quadrature_point_indices();
          const auto &normals = fe_iv[0]->get_normal_vectors();

          auto &solution = scratch_data.solution_interface;
          fe_iv[0]->get_fe_face_values(0).get_function_values(solution_global, solution[0][0]);
          fe_iv[0]->get_fe_face_values(1).get_function_values(solution_global, solution[1][0]);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i) {
            fe_iv[i]->get_fe_face_values(0).get_function_values(sol_vector[i], solution[0][i]);
            fe_iv[i]->get_fe_face_values(1).get_function_values(sol_vector[i], solution[1][i]);
          }

          array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> numflux{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            auto sol_q_s = std::tuple_cat(local_sol_q(solution[0], q_index), std::tie(extracted_data, variables));
            auto sol_q_n = std::tuple_cat(local_sol_q(solution[1], q_index), std::tie(extracted_data, variables));
            model.numflux(numflux, normals[q_index], x_q, fe_conv(sol_q_s), fe_conv(sol_q_n));

            for (uint i = 0; i < n_dofs; ++i) {
              const auto &cd_i = fe_iv[0]->interface_dof_to_dof_indices(i);
              const auto component_i = cd_i[0] == numbers::invalid_unsigned_int
                                           ? fe_iv[0]->get_fe().system_to_component_index(cd_i[1]).first
                                           : fe_iv[0]->get_fe().system_to_component_index(cd_i[0]).first;
              copy_data_face.cell_residual(i) +=
                  weight * JxW[q_index] * // weight * dx
                  (fe_iv[0]->jump_in_shape_values(i, q_index, component_i) *
                   scalar_product(numflux[component_i],
                                  normals[q_index])); // [[phi_i(x_q)]] * numflux(x_q, u_q) * n(x_q)
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, residual);
          constraints.distribute_local_to_global(c.cell_mass, c.local_dof_indices, residual);
          for (auto &cdf : c.face_data)
            constraints.distribute_local_to_global(cdf.cell_residual, cdf.joint_dof_indices, residual);
        };

        const UpdateFlags update_flags =
            update_values | update_gradients | update_quadrature_points | update_JxW_values;
        const MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_own_cells |
                                                         MeshWorker::assemble_boundary_faces |
                                                         MeshWorker::assemble_own_interior_faces_once;
        Scratch scratch_data(mapping, dof_handler_list, quadrature, quadrature_face, update_flags);
        CopyData copy_data;

        Timer timer;

        rebuild_ldg_vectors(solution_global);
        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, assemble_flags, boundary_worker, face_worker, threads, batch_size);

        timings_residual.push_back(timer.wall_time());
      }

      virtual void jacobian_mass(BlockSparseMatrix<NumberType> &jacobian, const VectorType &solution_global,
                                 const VectorType &solution_global_dot, NumberType alpha = 1.,
                                 NumberType beta = 1.) override
      {
        using Iterator = typename Triangulation<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_J_full<NumberType, Components::count_fe_subsystems()>;
        const auto &constraints = discretization.get_constraints();

        const auto cell_worker = [&](const Iterator &t_cell, Scratch &scratch_data, CopyData &copy_data) {
          const auto &fe_v = scratch_data.new_fe_values(t_cell);
          const uint to_n_dofs = fe_v[0]->get_fe().n_dofs_per_cell();
          copy_data.reinit(scratch_data.cell, Components::count_extractors());

          const auto &JxW = fe_v[0]->get_JxW_values();
          const auto &q_points = fe_v[0]->get_quadrature_points();
          const auto &q_indices = fe_v[0]->quadrature_point_indices();

          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;

          fe_v[0]->get_function_values(solution_global, solution[0]);
          fe_v[0]->get_function_values(solution_global_dot, solution_dot);

          SimpleMatrix<NumberType, Components::count_fe_functions(0)> j_mass;
          SimpleMatrix<NumberType, Components::count_fe_functions(0)> j_mass_dot;

          const uint from_n_dofs = fe_v[0]->get_fe().n_dofs_per_cell();
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];

            model.template jacobian_mass<0>(j_mass, x_q, solution[0][q_index], solution_dot[q_index]);
            model.template jacobian_mass<1>(j_mass_dot, x_q, solution[0][q_index], solution_dot[q_index]);

            for (uint i = 0; i < to_n_dofs; ++i) {
              const auto component_i = fe_v[0]->get_fe().system_to_component_index(i).first;
              for (uint j = 0; j < from_n_dofs; ++j) {
                const auto component_j = fe_v[0]->get_fe().system_to_component_index(j).first;
                copy_data.cell_jacobian[0](i, j) +=
                    JxW[q_index] * fe_v[0]->shape_value_component(j, q_index, component_j) * // weight * dx * phi_j(x_q)
                    fe_v[0]->shape_value_component(i, q_index, component_i) *
                    (alpha * j_mass_dot(component_i, component_j) +
                     beta * j_mass(component_i, component_j)); // -phi_i(x_q) * jsource(x_q, u_q)
              }
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_jacobian[0], c.local_dof_indices[0], jacobian);
        };

        const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
        const MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_own_cells;
        Scratch scratch_data(mapping, dof_handler_list, quadrature, quadrature_face, update_flags);
        CopyData copy_data;

        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, assemble_flags, nullptr, nullptr, threads, batch_size);
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
        if (is_close(weight, 0.))
          throw std::runtime_error("Please call jacobian_mass instead of jacobian for weight == 0).");
        using Iterator = typename Triangulation<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_J_full<NumberType, Components::count_fe_subsystems()>;
        const auto &constraints = discretization.get_constraints();

        // Find the EoM and extract whatever data is needed for the model.
        std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
        if constexpr (Components::count_extractors() > 0) {
          this->extract(__extracted_data, solution_global, variables, true, true, true);
          if (this->jacobian_extractors(this->extractor_jacobian, solution_global, variables))
            jacobian.reinit(sparsity_pattern_jacobian);
        }
        const auto &extracted_data = __extracted_data;

        bool exception = false;

        const auto cell_worker = [&](const Iterator &t_cell, Scratch &scratch_data, CopyData &copy_data) {
          const auto &fe_v = scratch_data.new_fe_values(t_cell);
          const uint to_n_dofs = fe_v[0]->get_fe().n_dofs_per_cell();
          copy_data.reinit(scratch_data.cell, Components::count_extractors());

          const auto &JxW = fe_v[0]->get_JxW_values();
          const auto &q_points = fe_v[0]->get_quadrature_points();
          const auto &q_indices = fe_v[0]->quadrature_point_indices();

          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;

          fe_v[0]->get_function_values(solution_global, solution[0]);
          fe_v[0]->get_function_values(solution_global_dot, solution_dot);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i)
            fe_v[i]->get_function_values(sol_vector[i], solution[i]);

          SimpleMatrix<NumberType, Components::count_fe_functions(0)> j_mass;
          SimpleMatrix<NumberType, Components::count_fe_functions(0)> j_mass_dot;
          auto j_flux = jacobian_tuple<Tensor<1, dim, NumberType>, Model>();
          auto j_source = jacobian_tuple<NumberType, Model>();
          SimpleMatrix<Tensor<1, dim, NumberType>, Components::count_fe_functions(), Components::count_extractors()>
              j_extr_flux;
          SimpleMatrix<NumberType, Components::count_fe_functions(), Components::count_extractors()> j_extr_source;
          constexpr_for<0, Components::count_fe_subsystems(), 1>([&](auto k) {
            if (jacobian_tmp_built[k] && model.get_components().jacobians_constant(0, k)) return;

            const uint from_n_dofs = fe_v[k]->get_fe().n_dofs_per_cell();
            for (const auto &q_index : q_indices) {
              const auto &x_q = q_points[q_index];
              auto sol_q = std::tuple_cat(local_sol_q(solution, q_index), std::tie(extracted_data, variables));

              if constexpr (k == 0) {
                this->model.template jacobian_mass<0>(j_mass, x_q, solution[0][q_index], solution_dot[q_index]);
                this->model.template jacobian_mass<1>(j_mass_dot, x_q, solution[0][q_index], solution_dot[q_index]);
                if constexpr (Components::count_extractors() > 0) {
                  this->model.template jacobian_flux_extr<stencil>(j_extr_flux, x_q, fe_conv(sol_q));
                  this->model.template jacobian_source_extr<stencil>(j_extr_source, x_q, fe_conv(sol_q));
                }
              }
              model.template jacobian_flux<k, 0>(std::get<k>(j_flux), x_q, fe_conv(sol_q));
              model.template jacobian_source<k, 0>(std::get<k>(j_source), x_q, fe_conv(sol_q));

              if (!std::get<k>(j_flux).is_finite() || !std::get<k>(j_source).is_finite()) exception = true;

              for (uint i = 0; i < to_n_dofs; ++i) {
                const auto component_i = fe_v[0]->get_fe().system_to_component_index(i).first;
                for (uint j = 0; j < from_n_dofs; ++j) {
                  const auto component_j = fe_v[k]->get_fe().system_to_component_index(j).first;

                  copy_data.cell_jacobian[k](i, j) +=
                      JxW[q_index] *
                      fe_v[k]->shape_value_component(j, q_index, component_j) * // weight * dx * phi_j(x_q)
                      (-scalar_product(fe_v[0]->shape_grad_component(i, q_index, component_i),
                                       std::get<k>(j_flux)(component_i, component_j)) // -dphi_i(x_q) * jflux(x_q, u_q)
                       + fe_v[0]->shape_value_component(i, q_index, component_i) *
                             std::get<k>(j_source)(component_i, component_j)); // -phi_i(x_q) * jsource(x_q, u_q)
                  if constexpr (k == 0) {
                    copy_data.cell_mass_jacobian(i, j) +=
                        JxW[q_index] *
                        fe_v[0]->shape_value_component(j, q_index, component_j) * // weight * dx * phi_j(x_q)
                        fe_v[0]->shape_value_component(i, q_index, component_i) *
                        (alpha / weight * j_mass_dot(component_i, component_j) +
                         beta / weight * j_mass(component_i, component_j)); // -phi_i(x_q) * jsource(x_q, u_q)
                  }
                }

                // extractor contribution
                if constexpr (k == 0)
                  if constexpr (Components::count_extractors() > 0)
                    for (uint e = 0; e < Components::count_extractors(); ++e)
                      copy_data.extractor_cell_jacobian(i, e) +=
                          weight * JxW[q_index] * // dx * phi_j * (
                          (-scalar_product(fe_v[0]->shape_grad_component(i, q_index, component_i),
                                           j_extr_flux(component_i, e)) // -dphi_i * jflux
                           + fe_v[0]->shape_value_component(i, q_index, component_i) *
                                 j_extr_source(component_i, e)); // -phi_i * jsource)
              }
            }
          });
        };
        const auto boundary_worker = [&](const Iterator &t_cell, const uint &face_no, Scratch &scratch_data,
                                         CopyData &copy_data) {
          const auto &fe_fv = scratch_data.new_fe_boundary_values(t_cell, face_no);
          const uint to_n_dofs = fe_fv[0]->get_fe().n_dofs_per_cell();

          const auto &JxW = fe_fv[0]->get_JxW_values();
          const auto &q_points = fe_fv[0]->get_quadrature_points();
          const auto &q_indices = fe_fv[0]->quadrature_point_indices();
          const auto &normals = fe_fv[0]->get_normal_vectors();
          auto &solution = scratch_data.solution;

          fe_fv[0]->get_function_values(solution_global, solution[0]);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i)
            fe_fv[i]->get_function_values(sol_vector[i], solution[i]);

          auto j_boundary_numflux = jacobian_tuple<Tensor<1, dim>, Model>();
          constexpr_for<0, Components::count_fe_subsystems(), 1>([&](auto k) {
            if (jacobian_tmp_built[k] && model.get_components().jacobians_constant(0, k)) return;
            for (const auto &q_index : q_indices) {
              const auto &x_q = q_points[q_index];
              auto sol_q = std::tuple_cat(local_sol_q(solution, q_index), std::tie(extracted_data, variables));

              const uint from_n_dofs = fe_fv[k]->get_fe().n_dofs_per_cell();

              model.template jacobian_boundary_numflux<k, 0>(std::get<k>(j_boundary_numflux), normals[q_index], x_q,
                                                             fe_conv(sol_q));

              if (!std::get<k>(j_boundary_numflux).is_finite()) exception = true;

              for (uint i = 0; i < to_n_dofs; ++i) {
                const auto component_i = fe_fv[0]->get_fe().system_to_component_index(i).first;
                for (uint j = 0; j < from_n_dofs; ++j) {
                  const auto component_j = fe_fv[k]->get_fe().system_to_component_index(j).first;

                  copy_data.cell_jacobian[k](i, j) +=
                      JxW[q_index] *
                      fe_fv[k]->shape_value_component(j, q_index, component_j) * // weight * dx * phi_j(x_q)
                      (fe_fv[0]->shape_value_component(i, q_index, component_i) *
                       scalar_product(std::get<k>(j_boundary_numflux)(component_i, component_j),
                                      normals[q_index])); // phi_i(x_q) * j_numflux(x_q, u_q) * n(x_q)
                }
              }
            }
          });
        };
        const auto face_worker = [&](const Iterator &t_cell, const uint &f, const uint &sf, const Iterator &t_ncell,
                                     const uint &nf, const unsigned int &nsf, Scratch &scratch_data,
                                     CopyData &copy_data) {
          const auto &fe_iv = scratch_data.new_fe_interface_values(t_cell, f, sf, t_ncell, nf, nsf);
          const uint to_n_dofs = fe_iv[0]->n_current_interface_dofs();
          auto &copy_data_face = copy_data.new_face_data(fe_iv, Components::count_extractors());

          const auto &JxW = fe_iv[0]->get_JxW_values();
          const auto &q_points = fe_iv[0]->get_quadrature_points();
          const auto &q_indices = fe_iv[0]->quadrature_point_indices();
          const auto &normals = fe_iv[0]->get_normal_vectors();

          auto &solution = scratch_data.solution_interface;
          fe_iv[0]->get_fe_face_values(0).get_function_values(solution_global, solution[0][0]);
          fe_iv[0]->get_fe_face_values(1).get_function_values(solution_global, solution[1][0]);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i) {
            fe_iv[i]->get_fe_face_values(0).get_function_values(sol_vector[i], solution[0][i]);
            fe_iv[i]->get_fe_face_values(1).get_function_values(sol_vector[i], solution[1][i]);
          }

          auto j_numflux = jacobian_2_tuple<Tensor<1, dim>, Model>();
          constexpr_for<0, Components::count_fe_subsystems(), 1>([&](auto k) {
            if (jacobian_tmp_built[k] && model.get_components().jacobians_constant(0, k)) return;
            for (const auto &q_index : q_indices) {
              const auto &x_q = q_points[q_index];
              auto sol_q_s = std::tuple_cat(local_sol_q(solution[0], q_index), std::tie(extracted_data, variables));
              auto sol_q_n = std::tuple_cat(local_sol_q(solution[1], q_index), std::tie(extracted_data, variables));

              const uint from_n_dofs = fe_iv[k]->n_current_interface_dofs();

              model.template jacobian_numflux<k, 0>(std::get<k>(j_numflux), normals[q_index], x_q, fe_conv(sol_q_s),
                                                    fe_conv(sol_q_n));

              if (!std::get<k>(j_numflux)[0].is_finite() || !std::get<k>(j_numflux)[1].is_finite()) exception = true;

              for (uint i = 0; i < to_n_dofs; ++i) {
                const auto &cd_i = fe_iv[0]->interface_dof_to_dof_indices(i);
                const uint face_no_i = cd_i[0] == numbers::invalid_unsigned_int ? 1 : 0;
                const auto &component_i = fe_iv[0]->get_fe().system_to_component_index(cd_i[face_no_i]).first;
                for (uint j = 0; j < from_n_dofs; ++j) {
                  const auto &cd_j = fe_iv[k]->interface_dof_to_dof_indices(j);
                  const uint face_no_j = cd_j[0] == numbers::invalid_unsigned_int ? 1 : 0;
                  const auto &component_j = fe_iv[k]->get_fe().system_to_component_index(cd_j[face_no_j]).first;

                  copy_data_face.cell_jacobian[k](i, j) +=
                      JxW[q_index] *
                      fe_iv[k]->get_fe_face_values(face_no_j).shape_value_component(
                          cd_j[face_no_j], q_index, component_j) * // weight * dx * phi_j(x_q)
                      (fe_iv[0]->jump_in_shape_values(i, q_index, component_i) *
                       scalar_product(std::get<k>(j_numflux)[face_no_j](component_i, component_j),
                                      normals[q_index])); // [[phi_i(x_q)]] * j_numflux(x_q, u_q)
                }
              }
            }
          });
        };
        const auto copier = [&](const CopyData &c) {
          try {
            constraints.distribute_local_to_global(c.cell_jacobian[0], c.local_dof_indices[0], jacobian);
            constraints.distribute_local_to_global(c.cell_mass_jacobian, c.local_dof_indices[0], jacobian);
            for (auto &cdf : c.face_data) {
              constraints.distribute_local_to_global(cdf.cell_jacobian[0], cdf.joint_dof_indices[0], jacobian);
              if constexpr (Components::count_extractors() > 0) {
                FullMatrix<NumberType> extractor_dependence(cdf.joint_dof_indices[0].size(),
                                                            extractor_dof_indices.size());
                cdf.extractor_cell_jacobian.mmult(extractor_dependence, this->extractor_jacobian);
                constraints.distribute_local_to_global(extractor_dependence, cdf.joint_dof_indices[0],
                                                       extractor_dof_indices, jacobian);
              }
            }
            if constexpr (Components::count_extractors() > 0) {
              FullMatrix<NumberType> extractor_dependence(c.local_dof_indices[0].size(), extractor_dof_indices.size());
              c.extractor_cell_jacobian.mmult(extractor_dependence, this->extractor_jacobian);
              constraints.distribute_local_to_global(extractor_dependence, c.local_dof_indices[0],
                                                     extractor_dof_indices, jacobian);
            }

            // LDG things

            for (uint i = 1; i < Components::count_fe_subsystems(); ++i) {
              if (jacobian_tmp_built[i] && model.get_components().jacobians_constant(0, i)) continue;
              j_ug[i].add(c.local_dof_indices[0], c.local_dof_indices[i], c.cell_jacobian[i]);
            }
            for (auto &cdf : c.face_data) {
              for (uint i = 1; i < Components::count_fe_subsystems(); ++i) {
                if (jacobian_tmp_built[i] && model.get_components().jacobians_constant(0, i)) continue;
                j_ug[i].add(cdf.joint_dof_indices[0], cdf.joint_dof_indices[i], cdf.cell_jacobian[i]);
              }
            }
          } catch (...) {
            exception = true;
          }
        };

        const UpdateFlags update_flags =
            update_values | update_gradients | update_quadrature_points | update_JxW_values;
        const MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_own_cells |
                                                         MeshWorker::assemble_boundary_faces |
                                                         MeshWorker::assemble_own_interior_faces_once;
        Scratch scratch_data(mapping, dof_handler_list, quadrature, quadrature_face, update_flags);
        CopyData copy_data;

        Timer timer;

        constexpr_for<1, Components::count_fe_subsystems(), 1>([&](auto k) {
          if (!ldg_matrix_built[k] || !model.get_components().jacobians_constant(k, k - 1))
            rebuild_ldg_jacobian<k>(solution_global);
          if (!jacobian_tmp_built[k] || !model.get_components().jacobians_constant(0, k)) j_ug[k] = 0;
        });
        rebuild_ldg_vectors(solution_global);

        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, assemble_flags, boundary_worker, face_worker, threads, batch_size);

        if (exception) throw std::runtime_error("Infinity encountered in jacobian construction");

        tbb::parallel_for(
            tbb::blocked_range<uint>(1, Components::count_fe_subsystems()), [&](tbb::blocked_range<uint> rk) {
              for (uint k = rk.begin(); k < rk.end(); ++k) {
                if (!jacobian_tmp_built[k] || !model.get_components().jacobians_constant(0, k)) {
                  jacobian_tmp[k] = 0;
                  tbb::parallel_for(
                      tbb::blocked_range<uint>(0, Components::count_fe_functions(0)), [&](tbb::blocked_range<uint> r) {
                        for (uint q = r.begin(); q < r.end(); ++q)
                          for (const auto &c : model.get_components().ldg_couplings(k, 0))
                            j_ug[k].block(q, c[0]).mmult(jacobian_tmp[k].block(q, c[1]), j_gu[k].block(c[0], c[1]),
                                                         Vector<NumberType>(), false);
                      });
                  jacobian_tmp_built[k] = true;
                }
              }
            });

        tbb::parallel_for(
            tbb::blocked_range<uint>(0, Components::count_fe_functions(0)), [&](tbb::blocked_range<uint> rc1) {
              tbb::parallel_for(tbb::blocked_range<uint>(0, Components::count_fe_functions(0)),
                                [&](tbb::blocked_range<uint> rc2) {
                                  for (uint c1 = rc1.begin(); c1 < rc1.end(); ++c1)
                                    for (uint c2 = rc2.begin(); c2 < rc2.end(); ++c2) {
                                      for (uint k = 1; k < Components::count_fe_subsystems(); ++k)
                                        jacobian.block(c1, c2).add(NumberType(1.), jacobian_tmp[k].block(c1, c2));
                                      jacobian.block(c1, c2) *= weight;
                                    }
                                });
            });

        timings_jacobian.push_back(timer.wall_time());
      }

      void log(const std::string logger)
      {
        std::stringstream ss;
        ss << "LDG Assembler: " << std::endl;
        ss << "        Reinit: " << average_time_reinit() * 1000 << "ms (" << num_reinits() << ")" << std::endl;
        ss << "        Residual: " << average_time_residual_assembly() * 1000 << "ms (" << num_residuals() << ")"
           << std::endl;
        ss << "        Jacobian: " << average_time_jacobian_assembly() * 1000 << "ms (" << num_jacobians() << ")"
           << std::endl;
        spdlog::get(logger)->info(ss.str());
      }

      double average_time_reinit() const
      {
        double t = 0.;
        double n = timings_reinit.size();
        for (const auto &t_ : timings_reinit)
          t += t_ / n;
        return t;
      }
      uint num_reinits() const { return timings_reinit.size(); }

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

    protected:
      using Base::discretization;
      using Base::dof_handler;
      using Base::fe;
      using Base::mapping;
      using Base::model;

      QGauss<dim> quadrature;
      QGauss<dim - 1> quadrature_face;
      using Base::batch_size;
      using Base::threads;

      std::vector<const DoFHandler<dim> *> dof_handler_list;

      mutable array<BlockVector<NumberType>, Components::count_fe_subsystems()> sol_vector;
      mutable array<BlockVector<NumberType>, Components::count_fe_subsystems()> sol_vector_tmp;
      mutable array<Vector<NumberType>, Components::count_fe_subsystems()> sol_vector_vec_tmp;

      BlockSparsityPattern sparsity_pattern_jacobian;
      BlockSparsityPattern sparsity_pattern_mass;
      array<BlockSparsityPattern, Components::count_fe_subsystems()> sparsity_pattern_ug;
      array<BlockSparsityPattern, Components::count_fe_subsystems()> sparsity_pattern_gu;
      array<BlockSparsityPattern, Components::count_fe_subsystems()> sparsity_pattern_wg;

      array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> jacobian_tmp;

      BlockSparseMatrix<NumberType> mass_matrix;
      SparseMatrix<NumberType> component_mass_matrix_inverse;
      array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_ug;
      mutable array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_gu;
      mutable array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_wg;
      mutable array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_wg_tmp;

      std::vector<double> timings_reinit;
      std::vector<double> timings_residual;
      std::vector<double> timings_jacobian;

      mutable array<bool, Components::count_fe_subsystems()> ldg_matrix_built;
      mutable array<bool, Components::count_fe_subsystems()> jacobian_tmp_built;

      using Base::EoM_abs_tol;
      using Base::EoM_max_iter;
      using Base::extractor_dof_indices;

      void rebuild_ldg_vectors(const VectorType &sol) const
      {
        constexpr_for<1, Components::count_fe_subsystems(), 1>([&](auto k) {
          if (!model.get_components().jacobians_constant(k, k - 1)) {
            if (k == 1)
              build_ldg_vector<k - 1, k>(sol, sol_vector[k], sol_vector_tmp[k]);
            else
              build_ldg_vector<k - 1, k>(sol_vector[k - 1], sol_vector[k], sol_vector_tmp[k]);
          } else {
            if (!ldg_matrix_built[k]) rebuild_ldg_jacobian<k>(sol);

            j_gu[k].vmult(sol_vector[k], sol);
          }
        });
      }

      template <int k> void rebuild_ldg_jacobian(const VectorType &sol) const
      {
        static_assert(k > 0);
        ldg_matrix_built[k] = true;
        if constexpr (k == 1)
          build_ldg_jacobian<0, 1>(sol, j_gu[1], j_wg_tmp[1]);
        else {
          if (!ldg_matrix_built[k - 1]) rebuild_ldg_jacobian<k - 1>(sol);
          build_ldg_jacobian<k - 1, k>(sol_vector[k - 1], j_wg[k], j_wg_tmp[k]);
          for (const auto &c : model.get_components().ldg_couplings(k, 0))
            for (const auto &b : model.get_components().ldg_couplings(k, k - 1))
              if (b[0] == c[0])
                j_wg[k]
                    .block(b[0], b[1])
                    .mmult(j_gu[k].block(c[0], c[1]), j_gu[k - 1].block(b[1], c[1]), Vector<NumberType>(), false);
        }
      }

      /**
       * @brief Build the LDG vector at level 'to', which takes information from level 'from'
       *
       * @tparam from Level to take information from.
       * @tparam to Level to write information to.
       * @param solution_global The solution at level 'from'
       * @param ldg_vector Where to store the result.
       * @param ldg_vector_tmp A temporary of the same size as ldg_vector.
       */
      template <int from, int to, typename VectorType, typename VectorTypeldg>
      void build_ldg_vector(const VectorType &solution_global, VectorTypeldg &ldg_vector,
                            VectorTypeldg &ldg_vector_tmp) const
      {
        static_assert(to - from == 1, "can only build LDG from last level!");
        using Iterator = typename Triangulation<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_R<NumberType>;
        const auto &constraints = discretization.get_constraints(to);

        const auto cell_worker = [&](const Iterator &t_cell, Scratch &scratch_data, CopyData &copy_data) {
          const auto &fe_v = scratch_data.new_fe_values(t_cell);
          const uint to_n_dofs = fe_v[to]->get_fe().n_dofs_per_cell();
          copy_data.reinit(scratch_data.cell[to], to_n_dofs);

          const auto &JxW = fe_v[to]->get_JxW_values();
          const auto &q_points = fe_v[to]->get_quadrature_points();
          const auto &q_indices = fe_v[to]->quadrature_point_indices();
          auto &solution = scratch_data.solution;

          fe_v[from]->get_function_values(solution_global, solution[from]);

          array<Tensor<1, dim, NumberType>, Components::count_fe_functions(to)> flux{};
          array<NumberType, Components::count_fe_functions(to)> source{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template ldg_flux<to>(flux, x_q, solution[from][q_index]);
            model.template ldg_source<to>(source, x_q, solution[from][q_index]);

            for (uint i = 0; i < to_n_dofs; ++i) {
              const auto component_i = fe_v[to]->get_fe().system_to_component_index(i).first;
              copy_data.cell_residual(i) += JxW[q_index] * // dx
                                            (-scalar_product(fe_v[to]->shape_grad_component(i, q_index, component_i),
                                                             flux[component_i]) // -dphi_i(x_q) * flux(x_q, u_q)
                                             + fe_v[to]->shape_value_component(i, q_index, component_i) *
                                                   source[component_i]); // -phi_i(x_q) * source(x_q, u_q)
            }
          }
        };
        const auto boundary_worker = [&](const Iterator &t_cell, const uint &face_no, Scratch &scratch_data,
                                         CopyData &copy_data) {
          const auto &fe_fv = scratch_data.new_fe_boundary_values(t_cell, face_no);
          const uint to_n_dofs = fe_fv[to]->get_fe().n_dofs_per_cell();

          const auto &JxW = fe_fv[to]->get_JxW_values();
          const auto &q_points = fe_fv[to]->get_quadrature_points();
          const auto &q_indices = fe_fv[to]->quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_fv[from]->get_normal_vectors();
          auto &solution = scratch_data.solution;

          fe_fv[from]->get_function_values(solution_global, solution[from]);

          array<Tensor<1, dim, NumberType>, Components::count_fe_functions(to)> numflux{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template ldg_boundary_numflux<to>(numflux, normals[q_index], x_q, solution[from][q_index]);

            for (uint i = 0; i < to_n_dofs; ++i) {
              const auto component_i = fe_fv[to]->get_fe().system_to_component_index(i).first;
              copy_data.cell_residual(i) +=
                  JxW[q_index] * // weight * dx
                  (fe_fv[to]->shape_value_component(i, q_index, component_i) *
                   scalar_product(numflux[component_i], normals[q_index])); // phi_i(x_q) * numflux(x_q, u_q) * n(x_q)
            }
          }
        };
        const auto face_worker = [&](const Iterator &t_cell, const uint &f, const uint &sf, const Iterator &t_ncell,
                                     const uint &nf, const unsigned int &nsf, Scratch &scratch_data,
                                     CopyData &copy_data) {
          const auto &fe_iv = scratch_data.new_fe_interface_values(t_cell, f, sf, t_ncell, nf, nsf);
          const uint to_n_dofs = fe_iv[to]->n_current_interface_dofs();
          auto &copy_data_face = copy_data.new_face_data(*(fe_iv[to]));

          const auto &JxW = fe_iv[to]->get_JxW_values();
          const auto &q_points = fe_iv[to]->get_quadrature_points();
          const auto &q_indices = fe_iv[to]->quadrature_point_indices();
          // normals are facing outwards!
          const std::vector<Tensor<1, dim>> &normals = fe_iv[to]->get_normal_vectors();
          auto &solution = scratch_data.solution_interface;

          fe_iv[from]->get_fe_face_values(0).get_function_values(solution_global, solution[0][from]);
          fe_iv[from]->get_fe_face_values(1).get_function_values(solution_global, solution[1][from]);

          array<Tensor<1, dim, NumberType>, Components::count_fe_functions(to)> numflux{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template ldg_numflux<to>(numflux, normals[q_index], x_q, solution[0][from][q_index],
                                           solution[1][from][q_index]);

            for (uint i = 0; i < to_n_dofs; ++i) {
              const auto &cd_i = fe_iv[to]->interface_dof_to_dof_indices(i);
              const auto component_i = cd_i[0] == numbers::invalid_unsigned_int
                                           ? fe_iv[to]->get_fe().system_to_component_index(cd_i[1]).first
                                           : fe_iv[to]->get_fe().system_to_component_index(cd_i[0]).first;
              copy_data_face.cell_residual(i) +=
                  JxW[q_index] * // weight * dx
                  (fe_iv[to]->jump_in_shape_values(i, q_index, component_i) *
                   scalar_product(numflux[component_i],
                                  normals[q_index])); // [[phi_i(x_q)]] * numflux(x_q, u_q) * n(x_q)
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, ldg_vector_tmp);
          for (auto &cdf : c.face_data)
            constraints.distribute_local_to_global(cdf.cell_residual, cdf.joint_dof_indices, ldg_vector_tmp);
        };

        const MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_own_cells |
                                                         MeshWorker::assemble_boundary_faces |
                                                         MeshWorker::assemble_own_interior_faces_once;
        Scratch scratch_data(mapping, dof_handler_list, quadrature, quadrature_face);
        CopyData copy_data;

        ldg_vector_tmp = 0;
        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, assemble_flags, boundary_worker, face_worker, threads, batch_size);

        for (uint i = 0; i < Components::count_fe_functions(to); ++i)
          component_mass_matrix_inverse.vmult(ldg_vector.block(i), ldg_vector_tmp.block(i));
      }

      /**
       * @brief Build the LDG jacobian at level 'to', which takes information from level 'from'
       *
       * @tparam from Level to take information from.
       * @tparam to Level to write information to.
       * @param solution_global The solution at level 'from'
       * @param ldg_jacobian Where to store the result.
       * @param ldg_jacobian_tmp A temporary of the same size as ldg_jacobian.
       */
      template <int from, int to, typename VectorType>
      void build_ldg_jacobian(const VectorType &solution_global, BlockSparseMatrix<NumberType> &ldg_jacobian,
                              BlockSparseMatrix<NumberType> &ldg_jacobian_tmp) const
      {
        static_assert(to - from == 1, "can only build LDG from last level!");
        using Iterator = typename Triangulation<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_J<NumberType>;

        const auto cell_worker = [&](const Iterator &t_cell, Scratch &scratch_data, CopyData &copy_data) {
          const auto &fe_v = scratch_data.new_fe_values(t_cell);
          const uint to_n_dofs = fe_v[to]->get_fe().n_dofs_per_cell();
          const uint from_n_dofs = fe_v[from]->get_fe().n_dofs_per_cell();
          copy_data.reinit(scratch_data.cell[from], scratch_data.cell[to], from_n_dofs, to_n_dofs);

          const auto &JxW = fe_v[to]->get_JxW_values();
          const auto &q_points = fe_v[to]->get_quadrature_points();
          const auto &q_indices = fe_v[to]->quadrature_point_indices();
          auto &solution = scratch_data.solution;

          fe_v[from]->get_function_values(solution_global, solution[from]);

          SimpleMatrix<Tensor<1, dim>, Components::count_fe_functions(to), Components::count_fe_functions(from)> j_flux;
          SimpleMatrix<NumberType, Components::count_fe_functions(to), Components::count_fe_functions(from)> j_source;
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template jacobian_flux<from, to>(j_flux, x_q, solution[from][q_index]);
            model.template jacobian_source<from, to>(j_source, x_q, solution[from][q_index]);

            for (uint i = 0; i < to_n_dofs; ++i) {
              const auto component_i = fe_v[to]->get_fe().system_to_component_index(i).first;
              for (uint j = 0; j < from_n_dofs; ++j) {
                const auto component_j = fe_v[from]->get_fe().system_to_component_index(j).first;
                copy_data.cell_jacobian(i, j) +=
                    JxW[q_index] *
                    fe_v[from]->shape_value_component(j, q_index, component_j) * // weight * dx * phi_j(x_q)
                    (-scalar_product(fe_v[to]->shape_grad_component(i, q_index, component_i),
                                     j_flux(component_i, component_j)) // -dphi_i(x_q) * jflux(x_q, u_q)
                     + fe_v[to]->shape_value_component(i, q_index, component_i) *
                           j_source(component_i, component_j)); // -phi_i(x_q) * jsource(x_q, u_q)
              }
            }
          }
        };
        const auto boundary_worker = [&](const Iterator &t_cell, const uint &face_no, Scratch &scratch_data,
                                         CopyData &copy_data) {
          const auto &fe_fv = scratch_data.new_fe_boundary_values(t_cell, face_no);
          const uint to_n_dofs = fe_fv[to]->get_fe().n_dofs_per_cell();
          const uint from_n_dofs = fe_fv[from]->get_fe().n_dofs_per_cell();

          const auto &JxW = fe_fv[to]->get_JxW_values();
          const auto &q_points = fe_fv[to]->get_quadrature_points();
          const auto &q_indices = fe_fv[to]->quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_fv[to]->get_normal_vectors();
          auto &solution = scratch_data.solution;

          fe_fv[from]->get_function_values(solution_global, solution[from]);

          SimpleMatrix<Tensor<1, dim>, Components::count_fe_functions(to), Components::count_fe_functions(from)>
              j_boundary_numflux;
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template jacobian_boundary_numflux<from, to>(j_boundary_numflux, normals[q_index], x_q,
                                                               solution[from][q_index]);

            for (uint i = 0; i < to_n_dofs; ++i) {
              const auto component_i = fe_fv[to]->get_fe().system_to_component_index(i).first;
              for (uint j = 0; j < from_n_dofs; ++j) {
                const auto component_j = fe_fv[from]->get_fe().system_to_component_index(j).first;
                copy_data.cell_jacobian(i, j) +=
                    JxW[q_index] *
                    fe_fv[from]->shape_value_component(j, q_index, component_j) * // weight * dx * phi_j(x_q)
                    (fe_fv[to]->shape_value_component(i, q_index, component_i) *
                     scalar_product(j_boundary_numflux(component_i, component_j),
                                    normals[q_index])); // phi_i(x_q) * j_numflux(x_q, u_q) * n(x_q)
              }
            }
          }
        };
        const auto face_worker = [&](const Iterator &t_cell, const uint &f, const uint &sf, const Iterator &t_ncell,
                                     const uint &nf, const unsigned int &nsf, Scratch &scratch_data,
                                     CopyData &copy_data) {
          const auto &fe_iv = scratch_data.new_fe_interface_values(t_cell, f, sf, t_ncell, nf, nsf);
          const uint to_n_dofs = fe_iv[to]->n_current_interface_dofs();
          const uint from_n_dofs = fe_iv[from]->n_current_interface_dofs();
          auto &copy_data_face = copy_data.new_face_data(*(fe_iv[from]), *(fe_iv[to]));

          const auto &JxW = fe_iv[to]->get_JxW_values();
          const auto &q_points = fe_iv[to]->get_quadrature_points();
          const auto &q_indices = fe_iv[to]->quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_iv[to]->get_normal_vectors();
          auto &solution = scratch_data.solution_interface;

          fe_iv[from]->get_fe_face_values(0).get_function_values(solution_global, solution[0][from]);
          fe_iv[from]->get_fe_face_values(1).get_function_values(solution_global, solution[1][from]);

          array<SimpleMatrix<Tensor<1, dim>, Components::count_fe_functions(to), Components::count_fe_functions(from)>,
                2>
              j_numflux;
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template jacobian_numflux<from, to>(j_numflux, normals[q_index], x_q, solution[0][from][q_index],
                                                      solution[1][from][q_index]);

            for (uint i = 0; i < to_n_dofs; ++i) {
              const auto &cd_i = fe_iv[to]->interface_dof_to_dof_indices(i);
              const uint face_no_i = cd_i[0] == numbers::invalid_unsigned_int ? 1 : 0;
              const auto &component_i = fe_iv[to]->get_fe().system_to_component_index(cd_i[face_no_i]).first;
              for (uint j = 0; j < from_n_dofs; ++j) {
                const auto &cd_j = fe_iv[from]->interface_dof_to_dof_indices(j);
                const uint face_no_j = cd_j[0] == numbers::invalid_unsigned_int ? 1 : 0;
                const auto &component_j = fe_iv[from]->get_fe().system_to_component_index(cd_j[face_no_j]).first;

                copy_data_face.cell_jacobian(i, j) +=
                    JxW[q_index] *
                    fe_iv[from]->get_fe_face_values(face_no_j).shape_value_component(
                        cd_j[face_no_j], q_index, component_j) * // weight * dx * phi_j(x_q)
                    (fe_iv[to]->jump_in_shape_values(i, q_index, component_i) *
                     scalar_product(j_numflux[face_no_j](component_i, component_j),
                                    normals[q_index])); // [[phi_i(x_q)]] * j_numflux(x_q, u_q)
              }
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          ldg_jacobian_tmp.add(c.local_dof_indices_to, c.local_dof_indices_from, c.cell_jacobian);
          for (auto &cdf : c.face_data)
            ldg_jacobian_tmp.add(cdf.joint_dof_indices_to, cdf.joint_dof_indices_from, cdf.cell_jacobian);
        };

        const MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_own_cells |
                                                         MeshWorker::assemble_boundary_faces |
                                                         MeshWorker::assemble_own_interior_faces_once;
        Scratch scratch_data(mapping, dof_handler_list, quadrature, quadrature_face);
        CopyData copy_data;

        ldg_jacobian_tmp = 0;
        ldg_jacobian = 0;
        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, assemble_flags, boundary_worker, face_worker, threads, batch_size);
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
          const auto to_cell = t_cell->as_dof_handler_iterator(to_dofh);
          const auto from_cell = t_cell->as_dof_handler_iterator(from_dofh);
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
            dsp.add_row_entries(row, extractor_dof_indices);

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

    protected:
      constexpr static int nothing = 0;
      using Base::EoM;
      using Base::EoM_cell;
      using Base::extractor_jacobian_u;
      using Base::old_EoM_cell;

      void readouts(DataOutput<dim, VectorType> &data_out, const VectorType &solution_global,
                    const VectorType &variables) const
      {
        auto helper = [&](auto EoMfun, auto outputter) {
          auto EoM_cell = this->EoM_cell;
          auto EoM = get_EoM_point(
              EoM_cell, solution_global, dof_handler, mapping, EoMfun, [&](const auto &p, const auto &) { return p; },
              EoM_abs_tol, EoM_max_iter);
          auto EoM_unit = mapping.transform_real_to_unit_cell(EoM_cell, EoM);

          using t_Iterator = typename Triangulation<dim>::active_cell_iterator;

          std::vector<std::shared_ptr<FEValues<dim>>> fe_v;
          for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
            fe_v.emplace_back(std::make_shared<FEValues<dim>>(
                mapping, discretization.get_fe(k), EoM_unit,
                update_values | update_gradients | update_quadrature_points | update_JxW_values | update_hessians));

            auto cell = dof_handler_list[k]->begin_active();
            cell->copy_from(*t_Iterator(EoM_cell));
            fe_v[k]->reinit(cell);
          }

          std::vector<std::vector<Vector<NumberType>>> solutions;
          for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
            solutions.push_back({Vector<NumberType>(Components::count_fe_functions(k))});
            if (k == 0)
              fe_v[0]->get_function_values(solution_global, solutions[k]);
            else
              fe_v[k]->get_function_values(sol_vector[k], solutions[k]);
          }
          std::vector<Vector<NumberType>> solutions_vector;
          for (uint k = 0; k < Components::count_fe_subsystems(); ++k)
            solutions_vector.push_back(solutions[k][0]);

          std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
          if constexpr (Components::count_extractors() > 0)
            extract(__extracted_data, solution_global, variables, true, false, false);
          const auto &extracted_data = __extracted_data;

          std::vector<std::vector<Tensor<1, dim, NumberType>>> solution_grad{
              std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
          std::vector<std::vector<Tensor<2, dim, NumberType>>> solution_hess{
              std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
          fe_v[0]->get_function_gradients(solution_global, solution_grad);
          fe_v[0]->get_function_hessians(solution_global, solution_hess);

          auto solution_tuple = std::tuple_cat(vector_to_tuple<Components::count_fe_subsystems()>(solutions_vector),
                                               std::tie(solution_grad[0], solution_hess[0], extracted_data, variables));

          outputter(data_out, EoM, fe_more_conv(solution_tuple));
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

        rebuild_ldg_vectors(solution_global);

        using t_Iterator = typename Triangulation<dim>::active_cell_iterator;

        std::vector<std::shared_ptr<FEValues<dim>>> fe_v;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
          fe_v.emplace_back(std::make_shared<FEValues<dim>>(
              mapping, discretization.get_fe(k), EoM_unit,
              update_values | update_gradients | update_quadrature_points | update_JxW_values | update_hessians));

          auto cell = dof_handler_list[k]->begin_active();
          cell->copy_from(*t_Iterator(EoM_cell));
          fe_v[k]->reinit(cell);
        }

        std::vector<std::vector<Vector<NumberType>>> solutions;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
          solutions.push_back({Vector<NumberType>(Components::count_fe_functions(k))});
          if (k == 0)
            fe_v[0]->get_function_values(solution_global, solutions[k]);
          else
            fe_v[k]->get_function_values(sol_vector[k], solutions[k]);
        }
        std::vector<Vector<NumberType>> solutions_vector;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k)
          solutions_vector.push_back(solutions[k][0]);

        std::vector<std::vector<Tensor<1, dim, NumberType>>> solution_grad{
            std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
        std::vector<std::vector<Tensor<2, dim, NumberType>>> solution_hess{
            std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
        fe_v[0]->get_function_gradients(solution_global, solution_grad);
        fe_v[0]->get_function_hessians(solution_global, solution_hess);

        auto solution_tuple = std::tuple_cat(vector_to_tuple<Components::count_fe_subsystems()>(solutions_vector),
                                             std::tie(solution_grad[0], solution_hess[0], this->nothing, variables));

        model.template extract(data, EoM, fe_more_conv(solution_tuple));
      }

      bool jacobian_extractors(FullMatrix<NumberType> &extractor_jacobian, const VectorType &solution_global,
                               const VectorType &variables)
      {
        if (extractor_jacobian_u.m() != Components::count_extractors() ||
            extractor_jacobian_u.n() != Components::count_fe_functions())
          extractor_jacobian_u =
              FullMatrix<NumberType>(Components::count_extractors(), Components::count_fe_functions());

        EoM = get_EoM_point(
            EoM_cell, solution_global, dof_handler, mapping,
            [&](const auto &p, const auto &values) { return model.EoM(p, values); },
            [&](const auto &p, const auto &values) { return model.EoM_postprocess(p, values); }, EoM_abs_tol,
            EoM_max_iter);
        auto EoM_unit = mapping.transform_real_to_unit_cell(EoM_cell, EoM);
        bool new_cell = (old_EoM_cell != EoM_cell);
        old_EoM_cell = EoM_cell;

        using t_Iterator = typename Triangulation<dim>::active_cell_iterator;

        std::vector<std::shared_ptr<FEValues<dim>>> fe_v;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
          fe_v.emplace_back(std::make_shared<FEValues<dim>>(
              mapping, discretization.get_fe(k), EoM_unit,
              update_values | update_gradients | update_quadrature_points | update_JxW_values | update_hessians));

          auto cell = dof_handler_list[k]->begin_active();
          cell->copy_from(*t_Iterator(EoM_cell));
          fe_v[k]->reinit(cell);
        }

        const uint n_dofs = fe_v[0]->get_fe().n_dofs_per_cell();
        if (new_cell) {
          // spdlog::get("log")->info("FEM: Rebuilding the jacobian sparsity pattern");
          extractor_dof_indices.resize(n_dofs);
          EoM_cell->get_dof_indices(extractor_dof_indices);
          rebuild_jacobian_sparsity();
        }

        std::vector<std::vector<Vector<NumberType>>> solutions;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
          solutions.push_back({Vector<NumberType>(Components::count_fe_functions(k))});
          if (k == 0)
            fe_v[0]->get_function_values(solution_global, solutions[k]);
          else
            fe_v[k]->get_function_values(sol_vector[k], solutions[k]);
        }
        std::vector<Vector<NumberType>> solutions_vector;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k)
          solutions_vector.push_back(solutions[k][0]);

        std::vector<std::vector<Tensor<1, dim, NumberType>>> solution_grad{
            std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
        std::vector<std::vector<Tensor<2, dim, NumberType>>> solution_hess{
            std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
        fe_v[0]->get_function_gradients(solution_global, solution_grad);
        fe_v[0]->get_function_hessians(solution_global, solution_hess);

        auto solution_tuple = std::tuple_cat(vector_to_tuple<Components::count_fe_subsystems()>(solutions_vector),
                                             std::tie(solution_grad[0], solution_hess[0], this->nothing, variables));

        extractor_jacobian_u = 0;
        model.template jacobian_extractors<0>(extractor_jacobian_u, EoM, fe_more_conv(solution_tuple));

        if (extractor_jacobian.m() != Components::count_extractors() || extractor_jacobian.n() != n_dofs)
          extractor_jacobian = FullMatrix<NumberType>(Components::count_extractors(), n_dofs);

        for (uint e = 0; e < Components::count_extractors(); ++e)
          for (uint i = 0; i < n_dofs; ++i) {
            const auto component_i = fe_v[0]->get_fe().system_to_component_index(i).first;
            extractor_jacobian(e, i) =
                extractor_jacobian_u(e, component_i) * fe_v[0]->shape_value_component(i, 0, component_i);
          }

        return new_cell;
      }

      using Base::timings_variable_jacobian;
      using Base::timings_variable_residual;
      template <typename... T> static constexpr auto v_tie(T &&...t)
      {
        return named_tuple<std::tuple<T &...>, "variables", "extractors">(std::tie(t...));
      }

      template <typename... T> static constexpr auto e_tie(T &&...t)
      {
        return named_tuple<std::tuple<T &...>, "fe_functions", "fe_derivatives", "fe_hessians", "extractors",
                           "variables">(std::tie(t...));
      }

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
      }

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
      }
    };
  } // namespace LDG
} // namespace DiFfRG
