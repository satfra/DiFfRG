#pragma once

#include <Eigen/Dense>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_vector.h>
#endif

namespace DiFfRG
{
  /**
   * @brief Converts a dealii vector to an Eigen vector
   *
   * @param dealii a dealii vector
   * @param eigen an Eigen vector
   */
  void dealii_to_eigen(const dealii::Vector<double> &dealii, Eigen::VectorXd &eigen);

  /**
   * @brief Converts a dealii block vector to an Eigen vector
   *
   * @param dealii a dealii block vector
   * @param eigen an Eigen vector
   */
  void dealii_to_eigen(const dealii::BlockVector<double> &dealii, Eigen::VectorXd &eigen);

  /**
   * @brief Converts an Eigen vector to a dealii vector
   *
   * @param eigen an Eigen vector
   * @param dealii a dealii vector
   */
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::Vector<double> &dealii);

  /**
   * @brief Converts an Eigen vector to a dealii block vector
   *
   * @param eigen an Eigen vector
   * @param dealii a dealii block vector
   */
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::BlockVector<double> &dealii);

#ifdef DEAL_II_WITH_PETSC
  /**
   * @brief Converts a process-local PETSc vector to an Eigen vector
   *
   * Only for vectors that hold every entry on this rank (MPI_COMM_SELF, a complete index set), such as
   * the replicated extra variables of the hybrid steppers; throws for a partitioned vector.
   */
  void dealii_to_eigen(const dealii::PETScWrappers::MPI::Vector &dealii, Eigen::VectorXd &eigen);

  /**
   * @brief Converts an Eigen vector to a process-local PETSc vector
   *
   * Same restriction as the reverse direction. An unsized vector is created on MPI_COMM_SELF.
   */
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::PETScWrappers::MPI::Vector &dealii);
#endif
} // namespace DiFfRG