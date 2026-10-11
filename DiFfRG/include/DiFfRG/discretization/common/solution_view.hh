#pragma once

// external libraries
#include <deal.II/base/index_set.h>
#include <deal.II/lac/vector.h>
#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_vector.h>
#endif

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/mpi.hh>

// std
#include <stdexcept>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Read-only copy of the solution in which every global dof is readable on every MPI rank.
   *
   * Output, the EoM search and the FV reconstruction read the solution at arbitrary global dof indices, which a
   * distributed vector cannot answer. Usage: reinit() once per mesh (AbstractAssembler::reinit_solution_view does
   * this), refresh() with the current solution, then read through get() or operator[].
   *
   * For serial vectors the view only stores a pointer to the source, so it sees later changes of it; for distributed
   * vectors (the PETSc specialization) it is a copy taken at refresh(). Refresh right before reading to get the same
   * behaviour from both.
   *
   * @tparam VectorType Vector type of the solution.
   */
  template <typename VectorType> class SolutionView
  {
  public:
    using NumberType = get_type::NumberType<VectorType>;
    using size_type = dealii::types::global_dof_index;

    /// An empty view; get() throws until refresh().
    SolutionView() = default;

    /// Set up the layout. Nothing to do for serial vectors.
    void reinit(const dealii::IndexSet &, const dealii::IndexSet &, MPI_Comm) {}

    /// Point the view at @p owned.
    void refresh(const VectorType &owned) { source = &owned; }

    /// The full solution vector; throws before the first refresh().
    const VectorType &get() const
    {
      if (source == nullptr)
        throw std::runtime_error("SolutionView::get() before refresh(): the view holds no solution.");
      return *source;
    }
    /// Same as get().
    operator const VectorType &() const { return get(); }

    /// The value of global dof @p i.
    NumberType operator[](const size_type i) const { return get()[i]; }
    /// The number of global dofs.
    size_type size() const { return get().size(); }

  private:
    const VectorType *source = nullptr;
  };

#ifdef DEAL_II_WITH_PETSC
  /**
   * @brief Distributed vectors: every rank holds a local copy (on MPI_COMM_SELF) of all entries, gathered from the
   * owning ranks at refresh().
   *
   * The copy is a non-ghosted vector, because reading a ghosted PETSc vector is not thread-safe and the assemblers read
   * the solution from parallel cell loops.
   */
  template <> class SolutionView<dealii::PETScWrappers::MPI::Vector>
  {
    static_assert(std::is_same_v<PetscScalar, double>,
                  "SolutionView gathers with MPI_DOUBLE; PETSc must be built --with-scalar-type=real "
                  "and --with-precision=double, which is what the bundled recipe pins.");

  public:
    using VectorType = dealii::PETScWrappers::MPI::Vector;
    using NumberType = VectorType::value_type;
    using size_type = dealii::types::global_dof_index;

    /// An empty view; get() throws until reinit().
    SolutionView() = default;

    /**
     * @brief Set up the gather from the ownership partition. Collective.
     *
     * Throws if the numbers of owned dofs of all ranks do not add up to the total. The owned sets are assumed to be
     * contiguous and ordered by rank (as PETSc requires); this is not checked.
     *
     * @param locally_owned The dofs owned by this rank.
     * @param ghost Unused; the copy always holds all dofs.
     * @param comm The communicator of the distributed vector.
     */
    void reinit(const dealii::IndexSet &locally_owned, const dealii::IndexSet &ghost, MPI_Comm comm)
    {
      (void)ghost;
      communicator = comm;
      n_global = locally_owned.size();
      const int n_ranks = static_cast<int>(MPI::size(comm));

      counts.assign(n_ranks, 0);
      displacements.assign(n_ranks, 0);
      local_count = static_cast<int>(locally_owned.n_elements());

      const int ierr = MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
      if (ierr != MPI_SUCCESS) throw std::runtime_error("SolutionView::reinit(): MPI_Allgather failed.");

      std::size_t total = 0;
      for (int r = 0; r < n_ranks; ++r) {
        displacements[r] = static_cast<int>(total);
        total += static_cast<std::size_t>(counts[r]);
      }
      // A PETSc MPI vector has contiguous owned blocks in rank order, which is the Allgatherv layout. Only the
      // total count is checked here.
      if (total != n_global)
        throw std::runtime_error("SolutionView::reinit(): the locally owned sets do not partition "
                                 "the index range exactly once; a gathered replica would be wrong.");

      if (n_global > 0) replica.reinit(dealii::complete_index_set(n_global), MPI_COMM_SELF);
      initialized = true;
      is_empty = (n_global == 0);
    }

    /**
     * @brief Copy @p owned into the local copy on every rank (exact, no arithmetic). Collective.
     *
     * An empty @p owned gives an empty view. Throws if @p owned does not match the layout given to reinit().
     */
    void refresh(const VectorType &owned)
    {
      if (!initialized)
        throw std::runtime_error("SolutionView::refresh() before reinit(): the gather plan is not known yet.");

      // An unsized source is a legitimate "nothing to show" -- the residual is optional in the
      // output path. Hold the empty vector separately instead of gathering it.
      is_empty = (owned.size() == 0) || (n_global == 0);
      if (is_empty) return;

      if (owned.size() != n_global || static_cast<int>(owned.locally_owned_size()) != local_count)
        throw std::runtime_error("SolutionView::refresh(): the vector does not match the layout "
                                 "reinit() was given. Rebuild the view after a mesh change.");

      const PetscScalar *src = nullptr;
      PetscScalar *dst = nullptr;
      Vec src_vec = static_cast<Vec>(owned);
      Vec dst_vec = static_cast<Vec>(replica);

      if (VecGetArrayRead(src_vec, &src) != 0) throw std::runtime_error("SolutionView::refresh(): VecGetArrayRead.");
      if (VecGetArray(dst_vec, &dst) != 0) throw std::runtime_error("SolutionView::refresh(): VecGetArray.");

      const int ierr = MPI_Allgatherv(src, local_count, MPI_DOUBLE, dst, counts.data(), displacements.data(),
                                      MPI_DOUBLE, communicator);

      VecRestoreArray(dst_vec, &dst);
      VecRestoreArrayRead(src_vec, &src);

      if (ierr != MPI_SUCCESS) throw std::runtime_error("SolutionView::refresh(): MPI_Allgatherv failed.");
    }

    /// The full solution vector; throws before reinit().
    const VectorType &get() const
    {
      if (!initialized) throw std::runtime_error("SolutionView::get() before reinit(): the view holds no solution.");
      return is_empty ? empty : replica;
    }
    /// Same as get().
    operator const VectorType &() const { return get(); }

    /// The value of global dof @p i.
    NumberType operator[](const size_type i) const { return get()(i); }
    /// The number of global dofs (0 if the view is empty).
    size_type size() const { return get().size(); }

  private:
    VectorType replica;
    VectorType empty;
    MPI_Comm communicator = MPI_COMM_SELF;
    std::vector<int> counts;
    std::vector<int> displacements;
    int local_count = 0;
    size_type n_global = 0;
    bool initialized = false;
    bool is_empty = true;
  };
#endif
} // namespace DiFfRG
