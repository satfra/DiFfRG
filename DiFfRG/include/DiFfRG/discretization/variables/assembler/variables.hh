#pragma once

// standard library
#include <sstream>

// external libraries
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/EoM.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/data_output.hh>

namespace DiFfRG
{
  namespace Variables
  {
    using namespace dealii;
    using std::array;

    template <typename... T> auto fe_tie(T &&...t)
    {
      return named_tuple<std::tuple<T &...>, "variables">(std::tie(t...));
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

      using Components = typename Model_::Components;
      static constexpr uint dim = 0;

      Assembler(Model &model, const JSONValue &json) : model(model), threads(json.get_uint("/discretization/threads"))
      {
        if (threads == 0) threads = dealii::MultithreadInfo::n_threads() / 2;
        static_assert(Components::count_fe_functions() == 0, "The pure variable assembler cannot handle FE functions!");
        reinit();
      }

      virtual void reinit_vector(VectorType &vec) const override { vec.reinit(0); }

      virtual IndexSet get_differential_indices() const override { return IndexSet(); }

      virtual void attach_data_output(DataOutput<dim, VectorType> &data_out, const VectorType &solution,
                                      const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                      const VectorType &residual = VectorType())
      {
        (void)dt_solution;
        (void)residual;
        readouts(data_out, solution, variables);
      }

      virtual void reinit() override {}

      virtual void set_time(double t) override { model.set_time(t); }

      virtual const SparsityPattern &get_sparsity_pattern_jacobian() const override
      {
        return sparsity_pattern_jacobian;
      }
      virtual const SparseMatrix<NumberType> &get_mass_matrix() const override { return mass_matrix; }

      virtual void residual_variables(VectorType &residual, const VectorType &variables, const VectorType &) override
      {
        Timer timer;
        model.dt_variables(residual, fe_tie(variables));
        timings_residual.push_back(timer.wall_time());
      };

      virtual void jacobian_variables(FullMatrix<NumberType> &jacobian, const VectorType &variables,
                                      const VectorType &) override
      {
        Timer timer;
        model.template jacobian_variables<0>(jacobian, fe_tie(variables));
        timings_jacobian.push_back(timer.wall_time());
      };

      void readouts(DataOutput<dim, VectorType> &data_out, const VectorType &, const VectorType &variables) const
      {
        auto helper = [&](auto EoMfun, auto outputter) {
          (void)EoMfun;
          outputter(data_out, Point<0>(), fe_tie(variables));
        };
        model.template readouts_multiple(helper, data_out);
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

      void log(const std::string logger) const
      {
        std::stringstream ss;
        ss << "Variable Assembler: " << std::endl;
        ss << "        Residual: " << average_time_residual_assembly() * 1000 << "ms (" << num_residuals() << ")"
           << std::endl;
        ss << "        Jacobian: " << average_time_jacobian_assembly() * 1000 << "ms (" << num_jacobians() << ")"
           << std::endl;
        spdlog::get(logger)->info(ss.str());
      }

      double average_time_residual_assembly()
      {
        double t = 0.;
        double n = timings_residual.size();
        for (const auto &t_ : timings_residual)
          t += t_ / n;
        return t;
      }
      uint num_residuals() const { return timings_residual.size(); }

      double average_time_jacobian_assembly()
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

      uint threads;

      SparsityPattern sparsity_pattern_mass;
      SparsityPattern sparsity_pattern_jacobian;
      SparseMatrix<NumberType> mass_matrix;

      std::vector<double> timings_residual;
      std::vector<double> timings_jacobian;
    };
  } // namespace Variables
} // namespace DiFfRG
