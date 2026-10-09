// DiFfRG
#include <DiFfRG/common/eigen.hh>

// external libraries
#include <deal.II/base/mpi.h>

// standard library
#include <stdexcept>
#include <string>

namespace DiFfRG
{
  void dealii_to_eigen(const dealii::Vector<double> &dealii, Eigen::VectorXd &eigen)
  {
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size())) eigen.resize(dealii.size());
    for (uint i = 0; i < dealii.size(); ++i)
      eigen(i) = dealii(i);
  }

  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::Vector<double> &dealii)
  {
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size())) dealii.reinit(eigen.size());
    for (uint i = 0; i < eigen.size(); ++i)
      dealii(i) = eigen(i);
  }

  void dealii_to_eigen(const dealii::BlockVector<double> &dealii, Eigen::VectorXd &eigen)
  {
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size())) eigen.resize(dealii.size());
    for (uint i = 0; i < dealii.size(); ++i)
      eigen(i) = dealii(i);
  }

  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::BlockVector<double> &dealii)
  {
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size()))
      throw std::runtime_error("eigen_to_dealii: dealii and eigen vectors have different sizes!");
    for (uint i = 0; i < eigen.size(); ++i)
      dealii(i) = eigen(i);
  }

#ifdef DEAL_II_WITH_PETSC
  namespace
  {
    void require_process_local(const dealii::PETScWrappers::MPI::Vector &vector, const char *caller)
    {
      // Decided on the communicator size, which every rank agrees on: an ownership test would pass on the owner
      // of a rank-0-owned vector and throw elsewhere, sending the owner alone into the next collective.
      if (dealii::Utilities::MPI::n_mpi_processes(vector.get_mpi_communicator()) != 1)
        throw std::runtime_error(std::string(caller) +
                                 ": the PETSc vector is distributed; only process-local vectors convert to Eigen.");
    }
  } // namespace

  void dealii_to_eigen(const dealii::PETScWrappers::MPI::Vector &dealii, Eigen::VectorXd &eigen)
  {
    require_process_local(dealii, "dealii_to_eigen");
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size())) eigen.resize(dealii.size());
    for (uint i = 0; i < dealii.size(); ++i)
      eigen(i) = dealii(i);
  }

  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::PETScWrappers::MPI::Vector &dealii)
  {
    if (static_cast<size_t>(dealii.size()) != static_cast<size_t>(eigen.size()))
      dealii.reinit(dealii::complete_index_set(eigen.size()), MPI_COMM_SELF);
    require_process_local(dealii, "eigen_to_dealii");
    for (uint i = 0; i < eigen.size(); ++i)
      dealii(i) = eigen(i);
    // Element writes through VectorBase::operator() are staged, not stored.
    dealii.compress(dealii::VectorOperation::insert);
  }
#endif
} // namespace DiFfRG
