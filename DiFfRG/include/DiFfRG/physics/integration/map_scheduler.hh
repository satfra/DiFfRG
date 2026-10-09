#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>

// std
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Kernel evaluations needed to saturate one rank's compute resource.
     *
     * Both are derived from `Kokkos::…::concurrency()` and cached on first use, so the split width
     * adapts to the hardware instead of being a constant tuned on one machine. That matters: the
     * measured fill threshold on the development GPU (5e4) is within 10% of its resident-thread
     * capacity (55 296 = 36 SMs x 1536), and an A100 has four times that. A hardcoded constant would
     * over-split by 4x on the very cluster this feature exists for.
     *
     * The two differ in more than magnitude. On the device `concurrency()` counts *resident threads*
     * and each thread evaluates the kernel once, so the fill threshold is the concurrency itself. On
     * the host it counts *workers*, each of which loops over many evaluations, so the threshold is
     * workers x a grain large enough that one parallel_for spawn (order 10 us) disappears into the
     * work it dispatches.
     */
    double device_fill_threshold();
    double host_fill_threshold();

    /// Evaluations per worker the host backend needs to amortize a parallel_for spawn. A judgement,
    /// not a measurement -- see the note in documentation/multi_gpu.md.
    inline constexpr double host_grain = 1024.;

    /**
     * @brief Evaluations below which a slice cannot pay for the launch that computes it.
     *
     * This is the threshold the *fill* thresholds above deliberately are not: for a map inside a
     * batch, splitting past the fill threshold buys nothing, because the ranks it would leave out
     * are about to be filled by the other maps in the batch. For a map that is flushed on return
     * there are no other maps, so those ranks are provably idle and the only reason left not to
     * split is that a slice must still cover its own launch (order 10 us). See
     * MapScheduler::schedule().
     */
    inline constexpr double launch_threshold = 1024.;
  } // namespace internal

  /**
   * @brief The physical resource a map() competes for.
   *
   * A rank owns *both*: one GPU and a slice of the node's cores. Work placed on one does not make
   * the other busy, and in a DeferredMaps scope the two genuinely run at the same time -- the device
   * path launches asynchronously and returns, so a host map issued after it executes while the
   * device is still working. Ownership is therefore balanced against a separate budget per resource;
   * see MapScheduler::least_loaded.
   */
  enum class MapResource : int { device = 0, host = 1 };
  inline constexpr int n_map_resources = 2;

  inline const char *to_string(const MapResource r) { return r == MapResource::host ? "host" : "device"; }

  /// Which resource a map() runs on and how many evaluations saturate one rank's share of it.
  struct MapTarget {
    MapResource resource;
    double fill_threshold;
  };

  /**
   * @brief The scheduling target of an execution space, selected at compile time.
   *
   * `memory_space == CPU_memory` is the same test `map_dist()` uses to pick its staging path, so a
   * CUDA-less build -- where `GPU_exec` *is* the host space -- automatically resolves to the host
   * resource and threshold with no configuration at all, collapsing back to a single budget.
   */
  template <typename ExecutionSpace> inline MapTarget map_target()
  {
    if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, CPU_memory>)
      return MapTarget{MapResource::host, internal::host_fill_threshold()};
    else
      return MapTarget{MapResource::device, internal::device_fill_threshold()};
  }

  /// The fill threshold alone, for callers that do not need the resource class.
  template <typename ExecutionSpace> inline double map_fill_threshold()
  {
    return map_target<ExecutionSpace>().fill_threshold;
  }

  /**
   * @brief This rank's window into the external grid of one QuadratureIntegrator::map() call.
   */
  struct MapSlice {
    /// First external grid point this rank computes.
    size_t offset = 0;
    /// Number of grid points; 0 means this rank does not participate in this map().
    size_t count = 0;

    bool owns_all(const size_t grid_size) const { return offset == 0 && count == grid_size; }
  };

  /**
   * @brief Decides, without any user input, which rank computes which part of each map().
   *
   * ## What is split
   *
   * Only the **external** coordinate grid of a `map()` -- never a quadrature reduction. Every
   * individual integral is therefore still summed entirely on one rank, in the original order, so
   * the result is **bitwise identical to a serial run at any rank count**. That is not a nicety:
   * the flow kernels feed a stiff DAE (see common/tbb.hh), where a last-bit change in a residual
   * changes the accepted step sequence and hence the whole trajectory. It also makes rebalancing
   * numerically free -- moving a partition boundary changes *which* rank computes a point, never
   * *what* it computes.
   *
   * Splitting applies to grids of any dimension. `SubCoordinates` is a window into the *linear*
   * index range, matching `part()` above, so the split needs no per-axis structure; it used to
   * derive a per-axis box, which is the same set of points only for `dim == 1`, and multi-dimensional
   * grids were therefore assigned whole to a single rank. That mattered: the multi-angle vertex
   * flows are both the most expensive and the ones with multi-dimensional external grids.
   *
   * ## How the split is chosen
   *
   * Per call, with `G` external grid points and a quadrature volume `Q = prod(grid_size)`:
   *
   *     S = G * Q                                   (kernel evaluations in this map)
   *     r = clamp(S / quantum, 1, min(n_ranks, G))  (how many ranks to spread it over)
   *     owners = the r least-loaded ranks
   *
   * Inside a batch, `quantum` is the **fill threshold**, not a launch-overhead threshold: below
   * roughly 5e4 evaluations these register-heavy kernels do not occupy a modern device anyway, so
   * splitting further buys nothing and costs a launch plus a gather. Deriving it from launch cost
   * (~1e3 evaluations) would oversplit by two orders of magnitude.
   *
   * The practical consequence is the one that matters for load balance: a **cheap flow is assigned
   * whole** to the least-loaded rank rather than chopped into slivers, so a block of many small
   * flows spreads across ranks instead of every rank doing a sliver of every flow.
   *
   * That reasoning is an argument about *opportunity cost*, and it holds only while there are other
   * maps to take the ranks this one leaves out. Outside a `DeferredMaps` scope there are none:
   * `map()` flushes on return, so the map is the whole batch and every rank without a slice sits in
   * the `Allgatherv` until the owners finish. There the fill threshold gives away ranks for nothing,
   * and the threshold that applies is instead `internal::launch_threshold` -- a slice must still
   * cover the launch that computes it, and that is the only remaining constraint. `set_batched()`
   * is how the scheduler learns which case it is in. An explicit user override wins over both.
   *
   * ## Mixed device and host models
   *
   * A rank holds two independent resources -- its GPU and its share of the node's cores -- and both
   * `r` and the ownership choice are made against the one the calling integrator actually uses.
   * `r` follows from that space's fill threshold (a device is saturated by its resident threads, a
   * host by workers x a grain), and the slices are charged to a **separate per-resource budget**, so
   * a rank that has just taken a large TBB flow is still the natural home for the next GPU flow.
   *
   * That is what makes a model mixing `GPU_exec` and `TBB_exec` integrators use both at once rather
   * than merely correctly: within a `DeferredMaps` scope the device path launches asynchronously and
   * returns, so a host map issued after it runs concurrently with it, and each resource is levelled
   * across ranks on its own. What the budgets deliberately do *not* do is convert between the two --
   * there is no exchange rate at which a device evaluation equals a host one, and assuming one is
   * exactly the mistake a single shared budget makes.
   *
   * Two limits remain, and both are outside this class. Within one resource the weight per
   * evaluation is 1, so flows with equal `S` but different per-evaluation cost still mis-weigh
   * (calibration, plan stage 4). And the overlap only exists inside a deferral scope: a host map
   * issued with deferral off flushes -- and therefore synchronises every rank -- before the device
   * work behind it can catch up.
   *
   * ## Determinism
   *
   * Every rank runs the same program and issues the same sequence of `map()` calls, so a schedule
   * that is a pure function of that sequence is identical on all ranks without any communication.
   * The running per-rank load is part of that state and is reset by `complete()`. A plan checksum
   * is verified inside `complete()`; a mismatch aborts the job rather than hanging it.
   */
  /**
   * @brief A scope in which map() must not be called.
   *
   * map() is a collective: every rank must issue the same sequence of them. Distributed FE assembly
   * breaks that guarantee, because each rank visits only its own cells -- so a map() from a model's
   * flux, source or evaluate_batch is issued a different number of times on different ranks, and the
   * run hangs in the next Allgatherv with no indication of which model did it.
   *
   * The assemblers open one of these around every model evaluation and every loop over their cells,
   * and a map() inside aborts the job with a message naming the problem instead of hanging. (A
   * model's batched evaluation uses map_points(), which is rank-local.)
   *
   * Deliberately a process-wide counter rather than thread_local: assembly calls the model on TBB
   * worker threads, and a thread_local flag set on the main thread would be invisible to exactly the
   * threads that need checking.
   */
  class NoMapsHere
  {
  public:
    NoMapsHere();
    ~NoMapsHere();
    NoMapsHere(const NoMapsHere &) = delete;
    NoMapsHere &operator=(const NoMapsHere &) = delete;

    /// Whether any NoMapsHere scope is currently open.
    static bool active();
  };

  class MapScheduler
  {
  public:
    static MapScheduler &instance();

    /// Whether there is more than one rank to schedule over. False in a serial run, which makes
    /// every method below a no-op and the whole feature zero-overhead.
    bool active() const { return m_n_ranks > 1; }

    uint rank() const { return m_rank; }
    uint n_ranks() const { return m_n_ranks; }
    MPI_Comm comm() const { return m_comm; }

    /**
     * @brief Register one map() call and return this rank's slice of it.
     *
     * Called by every rank for every map(), **including ranks that end up computing nothing** --
     * the entry with `count == 0` is what keeps the collective in `complete()` matched. Deciding
     * from rank-local state instead is the one way to deadlock this design.
     *
     * @param integrator_id stable, construction-order id of the calling integrator
     * @param dest base of the caller's destination array (unoffset)
     * @param elem_size sizeof one result element
     * @param grid_size number of external grid points, G
     * @param quadrature_volume points per integral, Q
     * @param splittable whether this coordinate system may be windowed (see class docs). True for
     *        every coordinate system today; kept as a parameter so a future one that cannot be
     *        windowed has somewhere to say so.
     * @param target the resource the calling integrator's execution space runs on, and the number of
     *        evaluations that saturate one rank's share of it, from map_target<ExecutionSpace>().
     *        The threshold sets the split width; the resource selects which budget the resulting
     *        slices are charged to. An explicit user override (DIFFRG_MAP_QUANTUM or
     *        /integration/map_quantum) replaces the threshold, but never the resource.
     */
    MapSlice schedule(size_t integrator_id, void *dest, size_t elem_size, size_t grid_size, size_t quadrature_volume,
                      bool splittable, MapTarget target);

    /// Whether the open plan already contains a map() from this integrator. Used in place of a
    /// rank-local "is a result still staged" test, which ranks would answer differently.
    bool plan_contains(size_t integrator_id) const;

    bool has_open_plan() const { return !m_plan.empty(); }

    /**
     * @brief Exchange every slice registered since the last completion.
     *
     * One `MPI_Allgatherv` for the whole batch. Called by MapCompletion::flush() *after* the device
     * results have landed in host memory.
     */
    void complete();

    /// Discard the open plan without communicating and poison the scheduler, so that the next
    /// collective aborts with a diagnosis instead of hanging. See DeferredMaps' destructor.
    void poison(const char *reason);

    /**
     * @brief Tell the scheduler whether the caller has a deferral scope open.
     *
     * Only the split width depends on it, and only through which threshold schedule() measures
     * against -- see internal::launch_threshold. Pushed in by MapCompletion::set_deferral() rather
     * than read back out of MapCompletion, so the dependency between the two stays one-way.
     *
     * Safe for the plan hash: DeferredMaps is constructed by replicated model code, so this flag
     * holds the same value on every rank at every schedule() call and the plan remains a pure
     * function of the call sequence.
     */
    void set_batched(const bool batched) { m_batched = batched; }

    /// Override the per-execution-space fill thresholds with one explicit value. 0 restores the
    /// automatic, hardware-derived defaults.
    void set_quantum(double quantum);
    /// The explicit override, or 0 when the automatic defaults are in force.
    double quantum() const { return m_quantum_override; }
    void set_verbose(bool verbose) { m_verbose = verbose; }

  protected:
    MapScheduler();

    /**
     * @brief Bind to an explicit rank and rank count instead of MPI_COMM_WORLD.
     *
     * For tests that simulate several ranks inside one process: the schedule, the byte accounting
     * and the packing are all pure functions of the call sequence, so they can be exercised without
     * an MPI build. Only the single Allgatherv in between cannot.
     */
    MapScheduler(uint rank, uint n_ranks);

    /// Verify agreement, compute the byte accounting and pack this rank's slices. False if there is
    /// nothing to exchange, in which case the batch is already closed.
    bool prepare_batch();
    /// Land every other rank's slices out of the receive buffer, then close the batch.
    void finish_batch();

    // Grow-only scratch, so a flush does not allocate. Protected rather than private so the
    // multi-rank simulation in the tests can play the part of the network.
    std::vector<char> m_send;
    std::vector<char> m_recv;
    std::vector<int> m_counts;
    std::vector<int> m_displs;

  private:
    struct Entry {
      size_t integrator_id;
      void *dest;
      size_t elem_size;
      size_t grid_size;
      MapResource resource;
      /// The r owning ranks, ascending. Slice j of the grid belongs to owners[j].
      std::vector<uint> owners;
    };

    /// Canonical partition: slice j of r covers [part(G, r, j), part(G, r, j + 1)).
    static size_t part(size_t grid_size, size_t r, size_t j) { return (j * grid_size) / r; }

    std::vector<uint> least_loaded(size_t r, MapResource resource) const;
    /// Zero every resource budget, sized to the rank count.
    void reset_load();
    uint64_t plan_hash() const;
    void log_plan() const;

    MPI_Comm m_comm;
    uint m_rank = 0;
    uint m_n_ranks = 1;

    /// 0 == no override, use the hardware-derived per-space thresholds.
    double m_quantum_override = 0.;
    bool m_quantum_pinned = false;
    /// Whether a DeferredMaps scope is open; see set_batched().
    bool m_batched = false;
    /// Completed batches, used to space out the plan-agreement collective. Only ever incremented
    /// on the path where every rank agrees there is a batch, so it cannot drift between ranks.
    size_t m_batch_index = 0;
    /// Verify the plan hash on batch 1..verify_head and then every m_verify_every-th batch.
    /// 0 == verify every batch (DIFFRG_MAP_VERIFY_EVERY).
    size_t m_verify_every = 64;
    /// Batches at the start of a run that are always verified, whatever m_verify_every says.
    static constexpr size_t verify_head = 8;
    bool m_verbose = false;
    bool m_logged = false;
    const char *m_poisoned = nullptr;

    std::vector<Entry> m_plan;
    /// Running kernel-evaluation count per rank within the open plan, one budget per resource;
    /// reset by complete(). Never summed across resources -- see the class docs.
    std::array<std::vector<double>, n_map_resources> m_load;
    /// Simulated instances (see the rank/n_ranks constructor) never communicate.
    bool m_simulated = false;
  };
} // namespace DiFfRG
