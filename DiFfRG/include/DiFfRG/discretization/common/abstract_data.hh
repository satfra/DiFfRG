#pragma once

// external libraries
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief A class to set up initial data for whatever discretization we have chosen.
   *        Also used to switch/manage memory, vectors, matrices over interfaces between spatial discretization and
   * separate variables.
   *
   * @tparam Discretization Spatial Discretization used in the system
   */
  template <typename NumberType, typename VectorType_ = dealii::Vector<NumberType>> class AbstractFlowingVariables
  {
  public:
    /**
     * @brief The vector type holding the state.
     *
     * Appended with a default so every existing `AbstractFlowingVariables<double>` keeps its
     * meaning. A distributed run substitutes a PETSc MPI vector here; the matching block type
     * is derived from it rather than hard-wired, because dealii::BlockVector cannot hold a
     * distributed block.
     */
    using VectorType = VectorType_;
    using BlockVectorType = get_type::BlockVectorType<VectorType>;

    /**
     * @brief Obtain the data vector holding both spatial (block 0) and variable (block 1) data.
     *
     * @return BlockVectorType& The data vector.
     */
    virtual BlockVectorType &data() = 0;
    virtual const BlockVectorType &data() const = 0;

    /**
     * @brief Obtain the spatial data vector.
     *
     * @return VectorType& The spatial data vector.
     */
    virtual VectorType &spatial_data() = 0;
    virtual const VectorType &spatial_data() const = 0;

    /**
     * @brief Obtain the variable data vector.
     *
     * @return VectorType& The variable data vector.
     */
    virtual VectorType &variable_data() = 0;
    virtual const VectorType &variable_data() const = 0;
  };

} // namespace DiFfRG