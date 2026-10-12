#pragma once

#include <Eigen/Dense>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_vector.h>
#endif

/**
 * @file eigen.hh
 * @brief Copy vectors between deal.II and Eigen (`Eigen::VectorXd`), entry by entry.
 */

namespace DiFfRG
{
  /// Copy `dealii` into `eigen`, resizing `eigen` if needed.
  void dealii_to_eigen(const dealii::Vector<double> &dealii, Eigen::VectorXd &eigen);

  /// Copy the concatenated blocks of `dealii` into `eigen`, resizing `eigen` if needed.
  void dealii_to_eigen(const dealii::BlockVector<double> &dealii, Eigen::VectorXd &eigen);

  /// Copy `eigen` into `dealii`, resizing `dealii` if needed.
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::Vector<double> &dealii);

  /// Copy `eigen` into the blocks of `dealii`; throws if the total sizes differ (blocks are not resized).
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::BlockVector<double> &dealii);

#ifdef DEAL_II_WITH_PETSC
  /**
   * @brief Copy a process-local PETSc vector into `eigen`.
   *
   * Only for vectors living on a single rank (a one-process communicator such as MPI_COMM_SELF), e.g. the
   * replicated variables of the hybrid steppers; throws for a vector distributed over several ranks.
   */
  void dealii_to_eigen(const dealii::PETScWrappers::MPI::Vector &dealii, Eigen::VectorXd &eigen);

  /**
   * @brief Copy `eigen` into a process-local PETSc vector.
   *
   * If the sizes differ, `dealii` is re-created on MPI_COMM_SELF with the size of `eigen`, replacing its previous
   * layout. Throws if `dealii` already has the right size but is distributed over several ranks.
   */
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::PETScWrappers::MPI::Vector &dealii);
#endif
} // namespace DiFfRG