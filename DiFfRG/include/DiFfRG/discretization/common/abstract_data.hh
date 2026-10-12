#pragma once

// external libraries
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief Interface of the state of a flow: the spatial dofs and the additional variables in one block vector.
   *
   * Block 0 holds the FE/FV dofs, block 1 the variables. See FlowingVariables for the implementation.
   *
   * @tparam NumberType Scalar type of the state.
   * @tparam VectorType_ Vector type of one block; a distributed run uses a parallel (PETSc) vector.
   */
  template <typename NumberType, typename VectorType_ = dealii::Vector<NumberType>> class AbstractFlowingVariables
  {
  public:
    /// Vector type of one block.
    using VectorType = VectorType_;
    /// Block vector type matching VectorType.
    using BlockVectorType = get_type::BlockVectorType<VectorType>;

    /// The full state: spatial dofs (block 0) and variables (block 1).
    virtual BlockVectorType &data() = 0;
    virtual const BlockVectorType &data() const = 0;

    /// The spatial dofs (block 0).
    virtual VectorType &spatial_data() = 0;
    virtual const VectorType &spatial_data() const = 0;

    /// The variables (block 1).
    virtual VectorType &variable_data() = 0;
    virtual const VectorType &variable_data() const = 0;
  };

} // namespace DiFfRG