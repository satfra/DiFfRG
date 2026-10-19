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
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace DiFfRG
{
  namespace internal
  {
    /// Size a vector to the dofs, with this rank's share of the rows if distributed.
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

    /// Copy a DynamicSparsityPattern into the matrix's pattern type; distributed: first send rows to their owners.
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

        // Not `pattern = dsp`: that operator only works for empty objects (unchecked in Release). Copy entries.
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

    /// Size a matrix from a finalized sparsity pattern.
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

    /// Distributed: the part of @p global_set owned by this rank; serial: @p global_set.
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

    /// Ownership of the variables block: all of it on rank 0.
    inline dealii::IndexSet variables_owner_set(const dealii::types::global_dof_index n_vars, MPI_Comm comm)
    {
      constexpr unsigned int variable_owner = 0;
      dealii::IndexSet variables(n_vars);
      if (MPI::rank(comm) == variable_owner) variables.add_range(0, n_vars);
      variables.compress();
      return variables;
    }

    /// Set up a SolutionView of the variables block (owned by rank 0).
    template <typename VectorType>
    void reinit_variables_view(SolutionView<VectorType> &view, const dealii::types::global_dof_index n_vars,
                               MPI_Comm comm)
    {
      view.reinit(variables_owner_set(n_vars, comm), dealii::complete_index_set(n_vars), comm);
    }

    /// Size a vector holding only the variables.
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

    /// Size a block vector: block 0 the dofs (partitioned), block 1 (if present) the variables (owned by rank 0).
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

    /// The vector type through which the model reads and writes its variables, in serial and distributed runs alike.
    template <typename VectorType> using VariablesVector = dealii::Vector<get_type::NumberType<VectorType>>;

    /// Copy the entries of @p src that @p dst owns into @p dst. @p src must hold all entries on this rank.
    template <typename Dst, typename Src> void assign_owned(Dst &dst, const Src &src)
    {
      if constexpr (!is_distributed_la<Dst>) {
        dst = src;
      } else {
        for (const auto i : dst.locally_owned_elements())
          dst(i) = src(i);
        dst.compress(dealii::VectorOperation::insert);
      }
    }

    /**
     * @brief The variables as the model sees them: a VariablesVector. Refers to @p variables if it already is one,
     * otherwise holds a copy.
     *
     * A distributed @p variables must be a replica, i.e. hold all entries on this rank (see SolutionView).
     */
    template <typename VectorType> class LocalVariables
    {
    public:
      using Vector = VariablesVector<VectorType>;

      explicit LocalVariables(const VectorType &variables)
      {
        if constexpr (std::is_same_v<VectorType, Vector>) {
          ptr = &variables;
        } else {
          if constexpr (is_distributed_la<VectorType>)
            if (variables.locally_owned_size() != variables.size())
              throw std::logic_error("LocalVariables: the variables vector is not replicated on this rank.");
          copy.reinit(variables.size());
          for (dealii::types::global_dof_index i = 0; i < variables.size(); ++i)
            copy[i] = variables(i);
          ptr = &copy;
        }
      }
      LocalVariables(const LocalVariables &) = delete;
      LocalVariables &operator=(const LocalVariables &) = delete;

      const Vector &get() const { return *ptr; }

    private:
      Vector copy;
      const Vector *ptr = nullptr;
    };

    /**
     * @brief Let @p compute write all variables into a VariablesVector, then store the entries @p dst owns. Every rank
     * computes; serially, and if @p dst already is a VariablesVector, @p compute writes into @p dst directly.
     */
    template <typename VectorType, typename Fn> void compute_variables_into(VectorType &dst, Fn &&compute)
    {
      if constexpr (std::is_same_v<VectorType, VariablesVector<VectorType>>) {
        compute(dst);
      } else {
        VariablesVector<VectorType> local(dst.size());
        compute(local);
        assign_owned(dst, local);
      }
    }

    /// dst = matrix * src on the variables block; distributed: broadcast src from rank 0 and apply on every rank.
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
  } // namespace internal
} // namespace DiFfRG
