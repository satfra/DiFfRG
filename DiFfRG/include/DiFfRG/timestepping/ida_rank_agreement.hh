#pragma once

// DiFfRG
#include <DiFfRG/common/mpi.hh>

namespace DiFfRG
{
  namespace internal
  {
    constexpr int recoverable_ida_callback_failure = 1;

    /**
     * @brief Agree across ranks on whether an IDA callback failed.
     *
     * IDA reacts to a recoverable failure by cutting the step and retrying. That decision must be
     * unanimous: if one rank reports failure and another success, they take different step
     * sequences, and the next collective -- an assembly compress(), a norm, this very agreement --
     * is entered by different numbers of ranks. The symptom is a hang, not a wrong number, and it
     * appears only for the input that first made the ranks disagree.
     *
     * Disagreement is easy to produce. The vector/matrix finiteness probes happen to be safe by
     * themselves, because l1_norm() and frobenius_norm() are collective for PETSc and already
     * return a global answer. The exception handlers are not: a model that throws on one cell makes
     * exactly the rank owning that cell return failure.
     *
     * Every exit path of every callback routes through here exactly once, success and failure
     * alike. That keeps the collectives matched only for failures raised after a callback's last
     * collective (or before its first): a rank that throws in the middle of an assembly leaves the
     * others waiting in its compress(). A serial discretization reports MPI_COMM_SELF, so this costs
     * nothing there.
     */
    inline int agreed_ida_result(MPI_Comm comm, const bool failed)
    {
      return DiFfRG::MPI::any_of(comm, failed) ? recoverable_ida_callback_failure : 0;
    }
  } // namespace internal
} // namespace DiFfRG
