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
   * @brief A read-only, fully-replicated view of the solution.
   *
   * A large amount of code under discretization/ reads the solution at *arbitrary global dof
   * indices*: the EoM search (common/eom.hh), the KT reconstruction caches
   * (FV/assembler/reconstruction_cache.hh), the output path (data/fe_output.cc) and the support
   * point lookups. None of that is expressible against a vector that only holds this rank's rows.
   *
   * Rather than rewrite all of it, this class maintains one replica holding *every* index,
   * refreshed once per residual/Jacobian evaluation. Those call sites then read the replica and
   * need no logic change at all. At replicated-mesh sizes the replica is exactly today's memory
   * footprint.
   *
   * The serial specialization is a pure passthrough -- it stores a pointer, copies nothing, and
   * compiles to the same code as passing the vector directly -- so a serial build pays nothing for
   * the abstraction.
   *
   * @warning The two policies differ in one respect that matters if you hold a view across a
   * write: the serial view aliases the source vector, so it observes later mutations, while the
   * distributed view is a snapshot taken at refresh(). Always refresh() immediately before reading
   * and never hold a view across an assembly step, and the two behave identically.
   *
   * Narrowing the replica from "everything" to the locally relevant dofs is the defining step of
   * the fully-distributed rung. It is deliberately confined to this one class so that step has a
   * single place to happen, and so the call sites that will need real work are exactly the ones
   * that name this type.
   */
  template <typename VectorType> class SolutionView
  {
  public:
    using NumberType = get_type::NumberType<VectorType>;
    using size_type = dealii::types::global_dof_index;

    SolutionView() = default;

    /**
     * @brief Establish the layout. No-op for the serial policy.
     */
    void reinit(const dealii::IndexSet &, const dealii::IndexSet &, MPI_Comm) {}

    /**
     * @brief Make the view reflect @p owned.
     */
    void refresh(const VectorType &owned) { source = &owned; }

    const VectorType &get() const
    {
      if (source == nullptr)
        throw std::runtime_error("SolutionView::get() before refresh(): the view holds no solution.");
      return *source;
    }
    operator const VectorType &() const { return get(); }

    NumberType operator[](const size_type i) const { return get()[i]; }
    size_type size() const { return get().size(); }

  private:
    const VectorType *source = nullptr;
  };

#ifdef DEAL_II_WITH_PETSC
  /**
   * @brief Distributed policy: a process-local replica holding every index.
   *
   * The replica lives on MPI_COMM_SELF and owns all n_dofs entries, refreshed by one MPI_Allgatherv
   * of the owning ranks' contiguous blocks.
   *
   * @warning It is tempting -- and it was the first implementation -- to build the replica as a
   * PETSc *ghosted* vector on the global communicator with complete_index_set() as its ghost set,
   * and let deal.II's step-40 assignment idiom do the communication. That is correct in a single
   * thread and crashes under assembly. deal.II's read path for a ghosted PETSc vector
   * (VectorBase::extract_subvector_to) calls VecGhostGetLocalForm / VecGhostRestoreLocalForm around
   * every read, and those two do PetscObjectReference / PetscObjectDereference on the local form
   * plus a PetscObjectStateSet on both vectors -- a non-atomic refcount and a shared write. Every
   * assembler reads the solution from inside its parallel cell loops, i.e. from N TBB
   * worker threads at once, so the refcount races; a lost increment takes it to zero, PETSc frees
   * the local form, and the next thread segfaults in VecGetSize_Seq on freed memory (the
   * accompanying "free(): invalid pointer" is the double free). PETSc is not thread safe unless
   * built --with-threadsafety.
   *
   * The non-ghosted read path does none of that -- VecGetOwnershipRange plus VecGetArrayRead and
   * pointer arithmetic, no refcount, no state write -- so a replica that owns everything locally is
   * both the simplest expression of "every rank holds every entry" and the only one that survives
   * threaded assembly.
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

    SolutionView() = default;

    /**
     * @brief Establish the layout and capture the gather plan.
     *
     * @p ghost is accepted for interface symmetry with the serial policy and with the
     * fully-distributed rung; at this rung the replica always covers everything, so only the
     * partition described by @p locally_owned is used.
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
      // A PETSc MPI vector requires an ascending, one-to-one partition anyway (deal.II asserts it,
      // and the assert is compiled out in Release), so the owned blocks are contiguous and in rank
      // order -- which is exactly the Allgatherv layout. If that ever stops holding, the gather
      // would silently permute the solution, so check it rather than trust it.
      if (total != n_global)
        throw std::runtime_error("SolutionView::reinit(): the locally owned sets do not partition "
                                 "the index range exactly once; a gathered replica would be wrong.");

      if (n_global > 0) replica.reinit(dealii::complete_index_set(n_global), MPI_COMM_SELF);
      initialized = true;
      is_empty = (n_global == 0);
    }

    /**
     * @brief Communicate @p owned into the replica.
     *
     * Exact: the gather moves each rank's owned block verbatim, with no arithmetic anywhere, so the
     * replica is bit-identical to the distributed vector on every rank.
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

    const VectorType &get() const
    {
      if (!initialized) throw std::runtime_error("SolutionView::get() before reinit(): the view holds no solution.");
      return is_empty ? empty : replica;
    }
    operator const VectorType &() const { return get(); }

    NumberType operator[](const size_type i) const { return get()(i); }
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
