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
#define DIFFRG_WITH_MPI 1
#endif

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
     * @brief Whether this build has MPI *and* MPI_Init has run. Everything below degrades to a
     * single-rank no-op when this is false, so callers never have to branch on the build type.
     */
    bool available();

    uint rank(MPI_Comm comm);
    uint size(MPI_Comm comm);
    void barrier(MPI_Comm comm);

    // Element-wise in-place array reductions: on return every rank holds the reduced array.
    void sum_reduce(MPI_Comm comm, int *data, int size);
    void max_reduce(MPI_Comm comm, int *data, int size);
    void min_reduce(MPI_Comm comm, int *data, int size);

    void sum_reduce(MPI_Comm comm, double *data, int size);
    void max_reduce(MPI_Comm comm, double *data, int size);
    void min_reduce(MPI_Comm comm, double *data, int size);

    int sum_reduce(MPI_Comm comm, int value);
    int max_reduce(MPI_Comm comm, int value);
    int min_reduce(MPI_Comm comm, int value);

    double sum_reduce(MPI_Comm comm, double value);
    double max_reduce(MPI_Comm comm, double value);
    double min_reduce(MPI_Comm comm, double value);

    /**
     * @brief Collective agreement on a predicate. Used to turn a rank-local decision (an abort, a
     * convergence flag) into one that every rank makes identically -- ranks that disagree about
     * control flow enter different numbers of collectives and hang.
     */
    bool any_of(MPI_Comm comm, bool value);
    bool all_of(MPI_Comm comm, bool value);

    /**
     * @brief Whether every rank passed the same @p token. One collective.
     *
     * For checking agreement on something whose disagreement would otherwise show up as a hang
     * (mismatched collective arguments) rather than an error.
     */
    bool agree(MPI_Comm comm, uint64_t token);

    /// Broadcast raw bytes from @p root to all ranks.
    void bcast(MPI_Comm comm, void *data, size_t bytes, uint root = 0);

    /**
     * @brief Concatenating all-gather of byte blocks of differing size.
     *
     * @p recv_counts and @p displs are in bytes and must be identical on every rank; @p recv must
     * hold sum(recv_counts) bytes. Used by MapScheduler to exchange map() slices. Deliberately not
     * an Allreduce(SUM) over zero-filled buffers: that moves N times more data and is not bit-safe,
     * since (-0.0) + 0.0 == +0.0 destroys a signed zero.
     */
    void allgatherv_bytes(MPI_Comm comm, const void *send, size_t send_bytes, void *recv,
                          const std::vector<int> &recv_counts, const std::vector<int> &displs);

    /**
     * @brief A communicator over the ranks sharing this node's memory, for GPU affinity.
     *
     * Returns MPI_COMM_SELF-equivalent behaviour (size 1) without MPI. Free it with free_comm().
     */
    MPI_Comm split_shared(MPI_Comm comm);
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
