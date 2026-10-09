#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>

// std
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>

namespace DiFfRG
{
  namespace MPI
  {
#ifdef DIFFRG_WITH_MPI

    bool available()
    {
      int flag = 0;
      MPI_Initialized(&flag);
      if (flag == 0) return false;
      MPI_Finalized(&flag);
      return flag == 0;
    }

    uint rank(MPI_Comm comm)
    {
      if (!available()) return 0;
      int rank;
      MPI_Comm_rank(comm, &rank);
      return rank;
    }
    uint size(MPI_Comm comm)
    {
      if (!available()) return 1;
      int size;
      MPI_Comm_size(comm, &size);
      return size;
    }
    void barrier(MPI_Comm comm)
    {
      Kokkos::fence();
      if (available()) MPI_Barrier(comm);
    }

    namespace
    {
      template <typename T> void array_reduce(MPI_Comm comm, T *data, int size, MPI_Datatype type, MPI_Op op)
      {
        if (size <= 0 || !available()) return;
        MPI_Allreduce(MPI_IN_PLACE, data, size, type, op, comm);
      }
    } // namespace

    void sum_reduce(MPI_Comm comm, int *data, int size) { array_reduce(comm, data, size, MPI_INT, MPI_SUM); }
    void max_reduce(MPI_Comm comm, int *data, int size) { array_reduce(comm, data, size, MPI_INT, MPI_MAX); }
    void min_reduce(MPI_Comm comm, int *data, int size) { array_reduce(comm, data, size, MPI_INT, MPI_MIN); }

    void sum_reduce(MPI_Comm comm, double *data, int size) { array_reduce(comm, data, size, MPI_DOUBLE, MPI_SUM); }
    void max_reduce(MPI_Comm comm, double *data, int size) { array_reduce(comm, data, size, MPI_DOUBLE, MPI_MAX); }
    void min_reduce(MPI_Comm comm, double *data, int size) { array_reduce(comm, data, size, MPI_DOUBLE, MPI_MIN); }

    namespace
    {
      template <typename T> T scalar_reduce(MPI_Comm comm, T value, MPI_Datatype type, MPI_Op op)
      {
        if (!available()) return value;
        T result;
        MPI_Allreduce(&value, &result, 1, type, op, comm);
        return result;
      }
    } // namespace

    int sum_reduce(MPI_Comm comm, int value) { return scalar_reduce(comm, value, MPI_INT, MPI_SUM); }
    int max_reduce(MPI_Comm comm, int value) { return scalar_reduce(comm, value, MPI_INT, MPI_MAX); }
    int min_reduce(MPI_Comm comm, int value) { return scalar_reduce(comm, value, MPI_INT, MPI_MIN); }

    double sum_reduce(MPI_Comm comm, double value) { return scalar_reduce(comm, value, MPI_DOUBLE, MPI_SUM); }
    double max_reduce(MPI_Comm comm, double value) { return scalar_reduce(comm, value, MPI_DOUBLE, MPI_MAX); }
    double min_reduce(MPI_Comm comm, double value) { return scalar_reduce(comm, value, MPI_DOUBLE, MPI_MIN); }

    bool any_of(MPI_Comm comm, bool value)
    {
      if (!available()) return value;
      int v = value ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MAX, comm);
      return v != 0;
    }
    bool all_of(MPI_Comm comm, bool value)
    {
      if (!available()) return value;
      int v = value ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MIN, comm);
      return v != 0;
    }

    bool agree(MPI_Comm comm, uint64_t token)
    {
      if (!available()) return true;
      // The token and its complement under MPI_MAX. If any rank differs, at most one of the two
      // can survive unchanged, so one collective suffices where MIN *and* MAX would need two.
      int64_t check[2] = {static_cast<int64_t>(token), static_cast<int64_t>(~token)};
      MPI_Allreduce(MPI_IN_PLACE, check, 2, MPI_INT64_T, MPI_MAX, comm);
      return check[0] == static_cast<int64_t>(token) && check[1] == static_cast<int64_t>(~token);
    }

    void bcast(MPI_Comm comm, void *data, size_t bytes, uint root)
    {
      if (!available() || bytes == 0) return;
      MPI_Bcast(data, static_cast<int>(bytes), MPI_BYTE, static_cast<int>(root), comm);
    }

    void allgatherv_bytes(MPI_Comm comm, const void *send, size_t send_bytes, void *recv,
                          const std::vector<int> &recv_counts, const std::vector<int> &displs)
    {
      if (!available()) return;
      // A rank can legitimately own no slice at all. Its count is 0, but some MPI implementations
      // still dereference-check the buffer, so never hand them a null pointer.
      static const char no_contribution = 0;
      MPI_Allgatherv(send_bytes == 0 ? static_cast<const void *>(&no_contribution) : send,
                     static_cast<int>(send_bytes), MPI_BYTE, recv, recv_counts.data(), displs.data(), MPI_BYTE, comm);
    }

    MPI_Comm split_shared(MPI_Comm comm)
    {
      if (!available()) return MPI_COMM_SELF;
      MPI_Comm node_comm;
      MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
      return node_comm;
    }

    void free_comm(MPI_Comm &comm)
    {
      if (!available()) return;
      if (comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF) MPI_Comm_free(&comm);
      comm = MPI_COMM_NULL;
    }

    void abort(MPI_Comm comm, const std::string &message, int code)
    {
      std::cerr << "\nDiFfRG MPI abort (rank " << rank(comm) << "): " << message << std::endl;
      if (available())
        MPI_Abort(comm, code);
      else
        std::abort();
      std::abort(); // unreachable; keeps [[noreturn]] honest for compilers that do not know MPI_Abort
    }

#else

    bool available() { return false; }

    uint rank(MPI_Comm) { return 0; }
    uint size(MPI_Comm) { return 1; }
    void barrier(MPI_Comm) { Kokkos::fence(); }

    // A one-rank reduction is the identity. The old implementation folded the array into a scalar
    // here, which is a different operation entirely and disagreed with the MPI branch.
    void sum_reduce(MPI_Comm, int *, int) {}
    void max_reduce(MPI_Comm, int *, int) {}
    void min_reduce(MPI_Comm, int *, int) {}

    void sum_reduce(MPI_Comm, double *, int) {}
    void max_reduce(MPI_Comm, double *, int) {}
    void min_reduce(MPI_Comm, double *, int) {}

    int sum_reduce(MPI_Comm, int value) { return value; }
    int max_reduce(MPI_Comm, int value) { return value; }
    int min_reduce(MPI_Comm, int value) { return value; }

    double sum_reduce(MPI_Comm, double value) { return value; }
    double max_reduce(MPI_Comm, double value) { return value; }
    double min_reduce(MPI_Comm, double value) { return value; }

    bool any_of(MPI_Comm, bool value) { return value; }
    bool all_of(MPI_Comm, bool value) { return value; }

    bool agree(MPI_Comm, uint64_t) { return true; }

    void bcast(MPI_Comm, void *, size_t, uint) {}

    void allgatherv_bytes(MPI_Comm, const void *, size_t, void *, const std::vector<int> &, const std::vector<int> &) {}

    MPI_Comm split_shared(MPI_Comm) { return MPI_COMM_SELF; }
    void free_comm(MPI_Comm &comm) { comm = MPI_COMM_NULL; }

    void abort(MPI_Comm, const std::string &message, int)
    {
      std::cerr << "\nDiFfRG abort: " << message << std::endl;
      std::abort();
    }

#endif
  } // namespace MPI
} // namespace DiFfRG
