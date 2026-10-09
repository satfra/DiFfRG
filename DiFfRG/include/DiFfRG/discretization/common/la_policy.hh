#pragma once

// external libraries
#include <deal.II/base/index_set.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/discretization/common/solution_view.hh>

// std
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Size a vector to the rank's share of the rows.
   *
   * Serial: the plain size-only reinit, exactly as before. Distributed: PETSc needs the owned
   * IndexSet and a communicator -- there is no size-only reinit that would give a correct
   * partitioning, which is why this indirection exists rather than a plain `vec.reinit(n)`.
   */
  template <typename VectorType>
  void reinit_la_vector(VectorType &vec, const dealii::IndexSet &locally_owned, MPI_Comm comm)
  {
    if constexpr (is_distributed_la<VectorType>) {
      vec.reinit(locally_owned, comm);
    } else {
      (void)comm;
      vec.reinit(locally_owned.size());
    }
  }

  /**
   * @brief Turn a freshly built DynamicSparsityPattern into the pattern type the matrix wants.
   *
   * Distributed: the rows a rank builds are not the rows it owns -- assembling a partition-boundary
   * cell adds entries to rows owned by a neighbour. distribute_sparsity_pattern
   * ships those to their owner. Skipping it does not fail loudly; it produces a matrix that is
   * missing exactly the couplings across partition boundaries, and PETSc then drops the
   * corresponding matrix entries at assembly time.
   *
   * The distributed branch copies rather than moves because DynamicSparsityPattern declares a copy
   * constructor and so has no implicit move. This runs once per reinit(), not per timestep.
   */
  template <typename SparseMatrixType>
  void finalize_la_sparsity(dealii::DynamicSparsityPattern &dsp, get_type::SparsityPattern<SparseMatrixType> &pattern,
                            const dealii::IndexSet &locally_owned, const dealii::IndexSet &locally_relevant,
                            MPI_Comm comm)
  {
    if constexpr (is_distributed_la<SparseMatrixType>) {
      // Guarded, not merely discarded: SparsityTools::distribute_sparsity_pattern only exists in
      // an MPI-enabled deal.II, and the call's arguments are non-dependent, so an `if constexpr`
      // alone would still fail name lookup in a serial build.
#ifdef DEAL_II_WITH_MPI
      dealii::SparsityTools::distribute_sparsity_pattern(dsp, locally_owned, comm, locally_relevant);

      // Emphatically NOT `pattern = dsp`. DynamicSparsityPattern::operator= is documented to work
      // "only for empty objects" and enforces that with an Assert -- which is compiled out in
      // Release. So the obvious assignment silently leaves the pattern 0x0, PETSc then preallocates
      // from an uninitialised row-start array, and the failure surfaces as
      // "nnz cannot be greater than row length: value 545352472" from deep inside MatSeqAIJ.
      // Copy the entries explicitly instead.
      const auto &rows = dsp.row_index_set();
      pattern.reinit(dsp.n_rows(), dsp.n_cols(), rows);
      const auto copy_row = [&](const dealii::types::global_dof_index row) {
        for (auto it = dsp.begin(row); it != dsp.end(row); ++it)
          pattern.add(row, it->column());
      };
      if (rows.n_elements() > 0)
        for (const auto row : rows)
          copy_row(row);
      else
        for (dealii::types::global_dof_index row = 0; row < dsp.n_rows(); ++row)
          copy_row(row);
#endif
    } else {
      (void)locally_owned;
      (void)locally_relevant;
      (void)comm;
      pattern.copy_from(dsp);
    }
  }

  /**
   * @brief Size a matrix from a finalized sparsity pattern.
   */
  template <typename SparseMatrixType>
  void reinit_la_matrix(SparseMatrixType &matrix, const get_type::SparsityPattern<SparseMatrixType> &pattern,
                        const dealii::IndexSet &locally_owned, MPI_Comm comm)
  {
    if constexpr (is_distributed_la<SparseMatrixType>) {
      matrix.reinit(locally_owned, pattern, comm);
    } else {
      (void)locally_owned;
      (void)comm;
      matrix.reinit(pattern);
    }
  }

  /**
   * @brief Restrict a global index set to what this rank may write.
   *
   * deal.II builds IDA's differential/algebraic mask by writing every index of the returned set
   * into a distributed vector and then compress(VectorOperation::insert)-ing it
   * (source/sundials/ida.cc). An unrestricted set means every rank writes every index, which is an
   * insert race on the same entries -- benign only as long as the values agree, and not something
   * to rely on. Intersecting here makes each entry written by exactly its owner.
   */
  template <typename VectorType>
  dealii::IndexSet restrict_to_owned(const dealii::IndexSet &global_set, const dealii::IndexSet &locally_owned)
  {
    if constexpr (is_distributed_la<VectorType>) {
      return global_set & locally_owned;
    } else {
      (void)locally_owned;
      return global_set;
    }
  }

  /**
   * @brief The ownership set for the extra-variables block: everything on rank 0, nothing elsewhere.
   *
   * Factored out so reinit_la_block_vector and reinit_la_variables_vector cannot drift apart. Two
   * places building disagreeing layouts for the same block is a hang, not a wrong number: the
   * blocks would have different global sizes and the first collective on them would not match up.
   */
  inline dealii::IndexSet variables_owner_set(const dealii::types::global_dof_index n_vars, MPI_Comm comm)
  {
    constexpr unsigned int variable_owner = 0;
    dealii::IndexSet variables(n_vars);
    if (MPI::rank(comm) == variable_owner) variables.add_range(0, n_vars);
    variables.compress();
    return variables;
  }

  /**
   * @brief Establish the layout of a fully-replicated view of the extra-variables block.
   *
   * Separate from the dof-shaped view because the variables block has its own ownership: rank 0
   * holds all of it (variables_owner_set), so a view built from the dof partition would not fit.
   */
  template <typename VectorType>
  void reinit_variables_view(SolutionView<VectorType> &view, const dealii::types::global_dof_index n_vars,
                             MPI_Comm comm)
  {
    view.reinit(variables_owner_set(n_vars, comm), dealii::complete_index_set(n_vars), comm);
  }

  /**
   * @brief Size a standalone vector holding only the extra variables.
   */
  template <typename VectorType>
  void reinit_la_variables_vector(VectorType &vec, const dealii::types::global_dof_index n_vars, MPI_Comm comm)
  {
    if constexpr (is_distributed_la<VectorType>) {
      vec.reinit(variables_owner_set(n_vars, comm), comm);
    } else {
      (void)comm;
      vec.reinit(n_vars);
    }
  }

  /**
   * @brief Size a block vector: block 0 is the FE dofs, block 1 (if present) the extra variables.
   *
   * The two blocks get different ownership policies, and the reason is not symmetric:
   *
   *  * Block 0 is partitioned by the dof distribution, like any FE vector.
   *  * Block 1 holds the model's extra variables. Its right hand side comes from map(), which is
   *    already computed identically on every rank, and essentially every consumer needs all of it.
   *    PETSc block vectors have no replicated block, so it is owned outright by rank 0 and read
   *    elsewhere through a SolutionView. Giving every rank the whole block instead would make each
   *    entry owned n_ranks times, and IDA would then see a state vector n_ranks times too long.
   *
   * @param variable_owner the rank that owns block 1. Fixed at 0 rather than exposed, so the
   *        layout cannot silently differ between two places that both build one of these.
   */
  template <typename BlockVectorType>
  void reinit_la_block_vector(BlockVectorType &vec, const std::vector<uint> &block_structure,
                              const dealii::IndexSet &locally_owned, MPI_Comm comm)
  {
    if constexpr (is_distributed_la<BlockVectorType>) {
      std::vector<dealii::IndexSet> owned{locally_owned};
      if (block_structure.size() > 1) owned.push_back(variables_owner_set(block_structure[1], comm));
      vec.reinit(owned, comm);
    } else {
      (void)locally_owned;
      (void)comm;
      vec.reinit(block_structure);
    }
  }

  /**
   * @brief Run a model's variables computation and land the result in the distributed block.
   *
   * The models write their variables residual entry by entry (`residual[i] = ...`). Under
   * distribution that block is owned outright by rank 0, so on every other rank the destination has
   * no local entries at all and the writes would go into PETSc's off-process stash -- n_ranks ranks
   * each inserting the same index, with no compress() anywhere to resolve it.
   *
   * The computation is redundant and identical on every rank anyway (its inputs are the replicated
   * variables view and the replicated solution), so compute into a process-local scratch vector and
   * copy back only what this rank owns.
   *
   * The serial branch calls @p compute directly on the destination, so the serial path keeps
   * exactly today's instruction sequence and stays bit-exact.
   *
   * @param scratch a process-local vector of full variables size, from reinit_local_variables_vector.
   */
  template <typename VectorType, typename Fn>
  void compute_variables_into(VectorType &dst, VectorType &scratch, Fn &&compute)
  {
    if constexpr (!is_distributed_la<VectorType>) {
      (void)scratch;
      compute(dst);
    } else {
      scratch = 0;
      compute(scratch);
      // The model wrote through VectorBase::operator(), which stages rather than stores.
      scratch.compress(dealii::VectorOperation::insert);
      for (const auto i : dst.locally_owned_elements())
        dst(i) = scratch(i);
      dst.compress(dealii::VectorOperation::insert);
    }
  }

  /**
   * @brief Size a process-local vector holding every extra variable on every rank.
   */
  template <typename VectorType>
  void reinit_local_variables_vector(VectorType &vec, const dealii::types::global_dof_index n_vars)
  {
    if constexpr (is_distributed_la<VectorType>) {
      if (n_vars > 0) vec.reinit(dealii::complete_index_set(n_vars), MPI_COMM_SELF);
    } else {
      vec.reinit(n_vars);
    }
  }

  /**
   * @brief Apply a dense matrix to the extra-variables block.
   *
   * The variables block is small, dense, and owned outright by rank 0 (variables_owner_set), so on
   * every other rank `src` is locally empty and dealii::FullMatrix::vmult -- which indexes elements
   * directly -- cannot be applied to it at all.
   *
   * Broadcast from the owner, apply the dense operator redundantly on every rank, then write back
   * only what each rank owns. Broadcasting rather than sum-reducing partial contributions is
   * deliberate: it is exact. A sum over zero-filled buffers would turn a -0.0 into +0.0 and is not
   * bit-safe, which is the same reason MPI::allgatherv_bytes exists instead of an Allreduce.
   *
   * Redundant application is the right trade here: the matrix is n_variables squared with
   * n_variables tiny, so replicating the work costs far less than a distributed solve, and it makes
   * every rank agree bit-for-bit without a second collective.
   */
  template <typename VectorType, typename NumberType>
  void dense_vmult_variables(const dealii::FullMatrix<NumberType> &matrix, VectorType &dst, const VectorType &src,
                             MPI_Comm comm)
  {
    if constexpr (!is_distributed_la<VectorType>) {
      (void)comm;
      matrix.vmult(dst, src);
    } else {
      const auto n = src.size();
      dealii::Vector<NumberType> local_src(n), local_dst(n);

      const auto owned = src.locally_owned_elements();
      for (const auto i : owned)
        local_src[i] = src(i);
      MPI::bcast(comm, local_src.data(), static_cast<size_t>(n) * sizeof(NumberType), /*root=*/0);

      matrix.vmult(local_dst, local_src);

      for (const auto i : dst.locally_owned_elements())
        dst(i) = local_dst[i];
      dst.compress(dealii::VectorOperation::insert);
    }
  }
} // namespace DiFfRG
