#pragma once

// DiFfRG
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/data/data_output.hh>

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief This is the general assembler interface for any kind of discretization.
   * An assembler is responsible for calculating residuals and their jacobians for any given discretization, including
   * both the spatial part and any further variables. Any assembler for a specific spatial discretization must fully
   * implement this interface.
   *
   * In general, we have either a system of differential and algebraic equations which can be written as
   * \f[\large
   * m_i(u, \partial_t u) + \partial_x F_i(u, \dots) + s_i(u, \dots) = 0\,,i\in \{0, \dots, N_f-1\}\,,
   * \f]
   * or a system of differential equations which can be written as
   * \f[\large
   * m_{ij}\partial_t u_j + \partial_x F_i(u, \dots) + s_i(u, \dots) = 0\,,i\in \{0, \dots, N_f-1\}\,.
   * \f]
   * Here, \f$u_i\f$ is the solution vector, \f$\partial_t u_i\f$ is the time derivative of the solution vector,
   * \f$m_i\f$ is the mass function and \f$m_{ij}\f$ the mass matrix, \f$F_i\f$ is called the flux, and \f$s_i\f$ is the
   * source term. We will henceforth call the first equation a **DAE** (differential algebraic equation) and the second
   * equation an **ODE** (ordinary differential equation).
   *
   * For more information, please also refer to the [guide](#HowTo) and the documentation of DiFfRG::def::AbstractModel.
   *
   * The second equation is used in all residual and jacobian functions which do not take the time derivative as an
   * explicit argument, while the first equation is used in all other methods.
   *
   * @tparam VectorType The vector type used in the spatial discretization
   * @tparam dim The dimension of the spatial discretization
   */
  template <typename VectorType, typename SparseMatrixType, uint dim> class AbstractAssembler
  {
  public:
    using NumberType = typename get_type::NumberType<VectorType>;

    /**
     * @brief Attach any data output to the DataOutput object provided. This can be used to extract additional data from
     * the solution and write it to the output file. This includes both derivatives and other spatial functions, as well
     * as single values that can be appended to the .csv file.
     *
     * @param data_out The DataOutput object to attach the data to
     * @param solution The spatial solution vector
     * @param variables The additional variables vector
     */
    virtual void attach_data_output(DataOutput<dim, VectorType> &data_out, const VectorType &solution,
                                    const VectorType &variables = VectorType(),
                                    const VectorType &dt_solution = VectorType(),
                                    const VectorType &residual = VectorType()) = 0;

    /**
     * @brief Reinitialize the assembler. This is necessary if the mesh has changed, e.g. after a mesh refinement.
     *
     */
    virtual void reinit() = 0;

    /**
     * @brief Set the current time. The assembler should usually just forward this to the numerical model.
     *
     * @param t The current time
     */
    virtual void set_time(double t) = 0;

    /**
     * @brief Obtain the sparsity pattern of the jacobian matrix.
     *
     * @return const SparsityPattern<VectorType>& The sparsity pattern of the jacobian matrix
     */
    virtual const get_type::SparsityPattern<SparseMatrixType> &get_sparsity_pattern_jacobian() const = 0;

    /**
     * @brief Obtain the dofs which contain time derivatives.
     *
     * @return IndexSet The indices of the dofs which contain time derivatives
     */
    virtual IndexSet get_differential_indices() const = 0;

    /**
     * @brief Reinitialize an arbitrary vector so that it has the correct size and structure.
     *
     * @param vector The vector to be reinitialized
     */
    virtual void reinit_vector(VectorType &vector) const = 0;

    /**
     * @brief Obtain the mass matrix.
     *
     * @return const SparseMatrixType& The mass matrix
     */
    virtual const SparseMatrixType &get_mass_matrix() const = 0;

    /**
     * @name Residual and jacobian functions
     */
    //@{

    /**
     * @brief When coupling the spatial discretization to additional variables, this function should calculate the
     * residual for the additional variables.
     *
     * @param residual The residual vector of the additional variables
     * @param variables The current additional variables vector
     * @param spatial_solution The spatial solution vector, which is needed for the calculation of the residual
     */
    virtual void residual_variables([[maybe_unused]] VectorType &residual, [[maybe_unused]] const VectorType &variables,
                                    [[maybe_unused]] const VectorType &spatial_solution)
    {
      throw std::runtime_error("Not implemented!");
    };

    /**
     * @brief When coupling the spatial discretization to additional variables, this function should calculate the
     * jacobian for the additional variables.
     *
     * @param jacobian The jacobian matrix of the additional variables
     * @param variables The current additional variables vector
     * @param spatial_solution The spatial solution vector, which is needed for the calculation of the jacobian
     */
    virtual void jacobian_variables([[maybe_unused]] FullMatrix<NumberType> &jacobian,
                                    [[maybe_unused]] const VectorType &variables,
                                    [[maybe_unused]] const VectorType &spatial_solution)
    {
      throw std::runtime_error("Not implemented!");
    };

    /**
     * @brief Calculates the mass \f$m_i(u)\f$ for an **ODE**.
     *
     * This function calculates the mass as \f$ w_m \, \int_\Omega\, m_ij \partial_t u_j \phi_i\f$.
     *
     * @param mass The mass vector to be filled
     * @param solution_global The spatial solution vector
     * @param weight The weight for the mass \f$ w_m \f$
     */
    void mass(VectorType &mass, const VectorType &solution_global, NumberType weight)
    {
      this->mass(mass, solution_global, solution_global, weight);
    }

    /**
     * @brief Calculates the mass \f$m(u, \partial_t u)\f$ for a **DAE**.
     *
     * This function calculates the mass as \f$ w_m \, \int_\Omega\, m_i(u_j, \partial_t u_j) \phi_i\f$.
     *
     * @param mass The mass vector to be filled
     * @param solution_global The spatial solution vector
     * @param solution_global_dot The spatial solution vector time derivative
     * @param weight The weight for the mass \f$ w_m \f$
     */
    virtual void mass(VectorType &mass, const VectorType &solution_global, const VectorType &solution_global_dot,
                      NumberType weight) = 0;

    /**
     * @brief Calculates the residual for an **ODE**.
     *
     * This function calculates the sum
     * \f[\large
     *    \int_\Omega\, \bigg(w_m m_ij \partial_t u_j \phi_i - w_r F_i(u_j, \dots) \partial_x \phi_i + w_r s_i(u_j,
     * \dots) \phi_i \bigg)
     *    + \int_{\partial\Omega}\, w_r \widehat{F}_i(u_j, \dots) \phi_i
     * \f].
     *
     * Here, \f$w_r\f$ is the residual weight, \f$w_m\,\,\f$ is the weight for the mass, and \f$\widehat{F}_i\f$
     * are any boundary terms picked up by the partial integration of the flux.
     *
     * @param residual The residual vector to be filled
     * @param solution_global The spatial solution vector
     * @param weight The weight for the residual \f$ w_r \f$
     * @param weight_mass The weight for the mass \f$ w_m \f$
     */
    void residual(VectorType &residual, const VectorType &solution_global, NumberType weight, NumberType weight_mass,
                  const VectorType &variables = VectorType())
    {
      this->residual(residual, solution_global, weight, solution_global, weight_mass, variables);
    }

    /**
     * @brief Calculates the residual for a **DAE**.
     *
     * This function calculates the sum
     * \f[\large
     *    \int_\Omega\, \bigg(w_m m_i(u_j, \partial_t u_j) \phi_i - w_r F_i(u_j, \dots) \partial_x \phi_i + w_r s_i(u_j,
     * \dots) \phi_i \bigg)
     *    + \int_{\partial\Omega}\, w_r \widehat{F}_i(u_j, \dots) \phi_i
     * \f].
     *
     * Here, \f$w_r\f$ is the residual weight, \f$w_m\,\,\f$ is the weight for the mass, and \f$\widehat{F}_i\f$
     * are the boundary terms picked up by the partial integration of the  flux.
     *
     * @param residual The residual vector to be filled
     * @param solution_global The spatial solution vector
     * @param weight The weight for the residual \f$ w_r \f$
     * @param solution_global_dot The spatial solution vector time derivative
     * @param weight_mass The weight for the mass \f$ w_m \f$
     */
    virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                          const VectorType &solution_global_dot, NumberType weight_mass,
                          const VectorType &variables = VectorType()) = 0;

    /**
     * @brief Calculates the jacobian of the mass function for an **ODE**.
     *
     * This function calculates the jacobian of the mass, i.e.
     * \f[\large
     * w_m\, \int_\Omega\,  m_{ij} \phi_i
     * \f].
     *
     * @param jacobian The jacobian matrix to be filled. This function adds to the matrix.
     * @param solution_global The spatial solution vector
     * @param mass_weight The weight for the mass \f$ w_m \f$
     */
    void jacobian_mass(SparseMatrixType &jacobian, const VectorType &solution_global, NumberType mass_weight = 1.)
    {
      this->jacobian_mass(jacobian, solution_global, solution_global, mass_weight, 0.);
    }

    /**
     * @brief Calculates the jacobian of the mass function for a **DAE**.
     *
     * This function calculates the jacobian of the mass, i.e.
     * \f[\large
     * \alpha\, \int_\Omega\, \frac{\partial m_i(u_k, \partial_t u_k)}{\partial (\partial_t u_j)} \phi_i
     * + \beta\, \int_\Omega\, \frac{\partial m_i(u_k, \partial_t u_k)}{\partial u_j} \phi_i
     * \f].
     *
     * @param jacobian The jacobian matrix to be filled. This function adds to the matrix.
     * @param solution_global The spatial solution vector
     * @param solution_global_dot The spatial solution vector time derivative
     * @param alpha The weight \f$ \alpha \f$ for the derivative with respect to \f$ \partial_t \f$
     * @param beta The weight \f$ \beta \f$ for the derivative with respect to \f$ u \f$
     */
    virtual void jacobian_mass(SparseMatrixType &jacobian, const VectorType &solution_global,
                               const VectorType &solution_global_dot, NumberType alpha = 1., NumberType beta = 1.) = 0;

    /**
     * @brief Calculates the jacobian of the residual function for an **ODE**.
     *
     *
     *
     * @param jacobian
     * @param solution_global
     * @param weight
     * @param mass_weight
     */
    void jacobian(SparseMatrixType &jacobian, const VectorType &solution_global, NumberType weight,
                  NumberType mass_weight, const VectorType &variables = VectorType())
    {
      this->jacobian(jacobian, solution_global, weight, solution_global, mass_weight, mass_weight, variables);
    }

    /**
     * @brief Calculates the jacobian of the residual function for a **DAE**.
     *
     * @param jacobian
     * @param solution_global
     * @param weight
     * @param solution_global_dot
     * @param alpha
     * @param beta
     */
    virtual void jacobian(SparseMatrixType &jacobian, const VectorType &solution_global, NumberType weight,
                          const VectorType &solution_global_dot, NumberType alpha, NumberType beta,
                          const VectorType &variables = VectorType()) = 0;
    //@}
  };
} // namespace DiFfRG