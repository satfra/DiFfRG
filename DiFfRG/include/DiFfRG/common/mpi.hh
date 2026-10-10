#pragma once

#include <DiFfRG/common/utils.hh>
#include <deal.II/base/mpi.h>

// std
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// deal.II is the single source of truth for whether MPI is available: it is the library that
// either includes <mpi.h> (giving us the real types at global scope) or declares dummy types in
// namespace dealii. DiFfRG's own HAVE_MPI must agree with it, otherwise mpi.cc would compile the
// serial fallbacks against real MPI types (or the reverse) and the two halves of the program would
// disagree about what a communicator is.
#if defined(HAVE_MPI) && !defined(DEAL_II_WITH_MPI)
#error                                                                                                                 \
    "DiFfRG was configured with MPI=ON but the deal.II it links against was built without DEAL_II_WITH_MPI. Rebuild deal.II with MPI, or configure DiFfRG with -DMPI=OFF."
#endif

#ifdef DEAL_II_WITH_MPI
/// Defined (as 1) when DiFfRG is built with MPI.
#define DIFFRG_WITH_MPI 1
#endif

/**
 * @file mpi.hh
 * @brief Thin MPI helpers that also work in builds without MPI.
 *
 * Every function can be called in any build and at any time. Without MPI, or before DiFfRG::Init has started
 * MPI, they behave as if there were a single rank: rank() is 0, size() is 1, reductions return their input
 * and broadcasts do nothing. Note that this also means that "print on rank 0 only" checks done before Init
 * print on every rank.
 *
 * Communicators are plain `MPI_Comm`, e.g. `MPI_COMM_WORLD`. In a build without MPI these names are provided
 * in namespace DiFfRG by deal.II's stand-ins.
 *
 * @code
 * if (DiFfRG::MPI::rank(MPI_COMM_WORLD) == 0) std::cout << "hello from rank 0\n";
 * const double total = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, local_value);
 * @endcode
 */

namespace DiFfRG
{
#ifndef DEAL_II_WITH_MPI
  // Without MPI, deal.II (>= 9.7) declares the MPI types and constants in
  // namespace dealii rather than at global scope. Pull them into namespace
  // DiFfRG so that the API below resolves identically with and without MPI.
  using dealii::MPI_Comm;
  using dealii::MPI_Datatype;
  using dealii::MPI_Op;
  using dealii::MPI_Request;

  using dealii::MPI_COMM_NULL;
  using dealii::MPI_COMM_SELF;
  using dealii::MPI_COMM_WORLD;
  using dealii::MPI_LOR;
  using dealii::MPI_MAX;
  using dealii::MPI_MIN;
  using dealii::MPI_REQUEST_NULL;
  using dealii::MPI_SUM;
#endif

  namespace MPI
  {
    /**
     * @brief Whether this build has MPI and MPI is running (initialized, not yet finalized).
     *
     * When false, all functions below act as on a single rank.
     */
    bool available();

    /// This process' rank in `comm`; 0 without MPI.
    uint rank(MPI_Comm comm);
    /// The number of ranks in `comm`; 1 without MPI.
    uint size(MPI_Comm comm);
    /// Wait for all running Kokkos work (Kokkos::fence(), also without MPI), then for all ranks of `comm`.
    void barrier(MPI_Comm comm);

    /** @name Reductions
     *
     * The array forms reduce `size` entries element-wise in place: afterwards every rank holds the sum, maximum
     * or minimum over all ranks. The scalar forms return the reduced value on every rank.
     * @{
     */
    /// Element-wise sum over all ranks, in place.
    void sum_reduce(MPI_Comm comm, int *data, int size);
    /// Element-wise maximum over all ranks, in place.
    void max_reduce(MPI_Comm comm, int *data, int size);
    /// Element-wise minimum over all ranks, in place.
    void min_reduce(MPI_Comm comm, int *data, int size);

    /// Element-wise sum over all ranks, in place.
    void sum_reduce(MPI_Comm comm, double *data, int size);
    /// Element-wise maximum over all ranks, in place.
    void max_reduce(MPI_Comm comm, double *data, int size);
    /// Element-wise minimum over all ranks, in place.
    void min_reduce(MPI_Comm comm, double *data, int size);

    /// Sum over all ranks.
    int sum_reduce(MPI_Comm comm, int value);
    /// Maximum over all ranks.
    int max_reduce(MPI_Comm comm, int value);
    /// Minimum over all ranks.
    int min_reduce(MPI_Comm comm, int value);

    /// Sum over all ranks.
    double sum_reduce(MPI_Comm comm, double value);
    /// Maximum over all ranks.
    double max_reduce(MPI_Comm comm, double value);
    /// Minimum over all ranks.
    double min_reduce(MPI_Comm comm, double value);
    /** @} */

    /**
     * @brief Whether `value` is true on at least one rank.
     *
     * Use it to turn a decision made on one rank (stop, retry) into one that all ranks take: ranks that take
     * different branches around a collective operation wait for each other forever.
     */
    bool any_of(MPI_Comm comm, bool value);
    /// Whether `value` is true on all ranks.
    bool all_of(MPI_Comm comm, bool value);

    /**
     * @brief Whether every rank passed the same @p token. One collective.
     *
     * For checking agreement on something whose disagreement would otherwise show up as a hang
     * (mismatched collective arguments) rather than an error.
     */
    bool agree(MPI_Comm comm, uint64_t token);

    /// Copy `bytes` bytes at `data` from rank `root` to all ranks. At most 2 GiB.
    void bcast(MPI_Comm comm, void *data, size_t bytes, uint root = 0);

    /**
     * @brief Concatenating all-gather of byte blocks of differing size.
     *
     * @p recv_counts and @p displs are in bytes and must be identical on every rank; @p recv must
     * hold sum(recv_counts) bytes. Used by MapScheduler to exchange map() slices. Deliberately not
     * an Allreduce(SUM) over zero-filled buffers: that moves N times more data and is not bit-safe,
     * since (-0.0) + 0.0 == +0.0 destroys a signed zero.
     *
     * Without MPI this does nothing; in particular `send` is not copied into `recv`.
     */
    void allgatherv_bytes(MPI_Comm comm, const void *send, size_t send_bytes, void *recv,
                          const std::vector<int> &recv_counts, const std::vector<int> &displs);

    /**
     * @brief A communicator over the ranks of `comm` on this node.
     *
     * `MPI_COMM_SELF` without MPI. Free it with free_comm().
     */
    MPI_Comm split_shared(MPI_Comm comm);
    /// Free a communicator from split_shared().
    void free_comm(MPI_Comm &comm);

    /**
     * @brief Tear the whole job down with a message on stderr.
     *
     * Used where a `throw` would deadlock: an exception unwinds one rank out of a collective the
     * others are still waiting in, so the failure would present as a hang instead of an error.
     */
    [[noreturn]] void abort(MPI_Comm comm, const std::string &message, int code = 1);
  } // namespace MPI
} // namespace DiFfRG
